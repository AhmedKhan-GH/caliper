// Engine tests for TwinScope v2's thermal model + learner
// (applets/twin_scope/twin_model.h), design §10 rows:
//   * energy decay to ambient under zero sources (M-weighted distance from
//     ambient non-increasing);
//   * batched B=50 step == 50 independent single-variant steps (allclose);
//   * duty-cycle determinism (same seed ⇒ identical factors over 100 samples,
//     different seed ⇒ differs);
//   * learner loss decreases on a fixed run, predictions finite / in-bounds.
// Small meshes only (subdivided tetra) so the suite stays fast; pure torch,
// CPU, its own binary (label "torch"). REQUIRE only — <torch/torch.h> defines
// a bare CHECK macro that shadows doctest's.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "twin_model.h"   // applets/twin_scope — pulls in twin_surface.h

#include <torch/torch.h>

using namespace twinscope;

// Regular tetrahedron — a small closed 2-manifold (V=4,F=4,E=6); subdivide for
// a mesh with genuine interior structure but only tens of vertices.
static SurfaceMesh make_tetrahedron() {
    auto positions = torch::tensor({{1.f, 1.f, 1.f},
                                    {1.f, -1.f, -1.f},
                                    {-1.f, 1.f, -1.f},
                                    {-1.f, -1.f, 1.f}},
                                   torch::kFloat32);
    auto indices = torch::tensor({0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2},
                                 torch::kLong);
    auto uvs = torch::tensor({{0.1f, 0.1f}, {0.9f, 0.1f},
                              {0.1f, 0.9f}, {0.9f, 0.9f}},
                             torch::kFloat32);
    return SurfaceMesh{positions, indices, uvs};
}

// --------------------------------------------------------------------------
// Energy decay under zero sources.
// --------------------------------------------------------------------------
TEST_CASE("thermal step: M-weighted energy is non-increasing with sources off") {
    auto mesh = subdivide_midpoint(make_tetrahedron(), 2);
    ThermalConfig cfg;
    cfg.source_gain = 0.f;  // sources off — pure diffusion + convective loss
    auto sim = make_thermal_sim(mesh, /*batch=*/4, torch::kCPU, /*seed=*/3, cfg);

    // Random departure from ambient, strictly finite.
    torch::manual_seed(11);
    sim.T = cfg.ambient + 10.f * torch::randn({sim.B, sim.V});

    auto M = sim.M;  // (V,)
    auto energy = [&](const torch::Tensor& T) {
        auto u = T - cfg.ambient;                       // (B,V)
        return (M * u * u).sum().item<double>();        // M-weighted, all variants
    };

    double prev = energy(sim.T);
    for (int i = 0; i < 100; ++i) {
        sim.step(/*t=*/static_cast<double>(i) * sim.dt);
        const double e = energy(sim.T);
        REQUIRE(std::isfinite(e));
        REQUIRE(e <= prev + 1e-6);
        prev = e;
    }
    // And it actually relaxes toward ambient (decayed well below the start).
    REQUIRE(prev < energy(cfg.ambient + 10.f * torch::ones({sim.B, sim.V})));
}

// --------------------------------------------------------------------------
// Batched step == independent single-variant steps.
// --------------------------------------------------------------------------
TEST_CASE("thermal step: batched B=50 == 50 independent single-variant steps") {
    auto mesh = subdivide_midpoint(make_tetrahedron(), 2);
    const int64_t B = 50;
    auto big = make_thermal_sim(mesh, B, torch::kCPU, /*seed=*/5);

    // Give every variant a distinct starting field so a cross-variant leak
    // would show up.
    torch::manual_seed(21);
    big.T = kAmbient + 5.f * torch::randn({B, big.V});

    const double t = 1.7;
    ThermalSim batched = big;          // shares operator handles
    batched.step(t);                   // one batched step over all 50

    for (int64_t i = 0; i < B; ++i) {
        ThermalSim one = big;          // copy handles, then narrow to variant i
        one.B = 1;
        one.T = big.T[i].unsqueeze(0).clone();
        one.intensities = big.intensities[i].unsqueeze(0).clone();
        one.step(t);
        REQUIRE(torch::allclose(one.T[0], batched.T[i], 1e-5, 1e-6));
    }
}

// --------------------------------------------------------------------------
// Duty-cycle determinism.
// --------------------------------------------------------------------------
TEST_CASE("duty schedule: deterministic per seed, differs across seeds") {
    auto mesh = subdivide_midpoint(make_tetrahedron(), 1);
    auto a = make_thermal_sim(mesh, /*batch=*/2, torch::kCPU, /*seed=*/42);
    auto b = make_thermal_sim(mesh, /*batch=*/2, torch::kCPU, /*seed=*/42);
    auto c = make_thermal_sim(mesh, /*batch=*/2, torch::kCPU, /*seed=*/1000);

    double ab_max = 0.0, ac_max = 0.0;
    for (int i = 0; i < 100; ++i) {
        const double t = 20.0 * i / 99.0;
        auto da = a.duty(t), db = b.duty(t), dc = c.duty(t);
        ab_max = std::max(ab_max, (da - db).abs().max().item<double>());
        ac_max = std::max(ac_max, (da - dc).abs().max().item<double>());
    }
    REQUIRE(ab_max == 0.0);   // same seed ⇒ identical schedule at every sample
    REQUIRE(ac_max > 0.0);    // different seed ⇒ schedule differs somewhere
}

// --------------------------------------------------------------------------
// MPS device parity: the sim constructs and steps on MPS, matching the CPU
// reference. Guards the Apple path that libtorch's missing SparseMPS support
// would otherwise crash at construction (sparse L .to(kMPS) throws
// NotImplementedError in torch 2.5.1) — the sim must carry the Laplacian in
// an MPS-legal form and produce the same physics.
// --------------------------------------------------------------------------
TEST_CASE("thermal step: MPS sim constructs and matches the CPU reference") {
    if (!torch::mps::is_available()) { MESSAGE("no MPS device - skipping"); return; }
    auto mesh = subdivide_midpoint(make_tetrahedron(), 2);
    const int64_t B = 4;
    auto cpu = make_thermal_sim(mesh, B, torch::kCPU, /*seed=*/13);
    auto mps = make_thermal_sim(mesh, B, torch::kMPS, /*seed=*/13);

    // Same departure from ambient on both devices.
    torch::manual_seed(31);
    auto T0 = kAmbient + 5.f * torch::randn({B, cpu.V});
    cpu.T = T0.clone();
    mps.T = T0.to(torch::kMPS);

    for (int i = 0; i < 50; ++i) {
        const double t = static_cast<double>(i) * cpu.dt;
        cpu.step(t);
        mps.step(t);
    }
    auto back = mps.T.to(torch::kCPU);
    REQUIRE(torch::isfinite(back).all().item<bool>());
    REQUIRE(torch::allclose(back, cpu.T, 1e-4, 1e-4));
}

// --------------------------------------------------------------------------
// The chasing learner.
// --------------------------------------------------------------------------
TEST_CASE("learner: loss decreases on a fixed run, predictions finite/in-bounds") {
    auto mesh = subdivide_midpoint(make_tetrahedron(), 2);
    const int64_t B = 8;
    auto sim = make_thermal_sim(mesh, B, torch::kCPU, /*seed=*/9);

    // A smooth, learnable synthetic field: a position bump modulated per
    // variant. (This test exercises the LEARNER; the physics has its own rows.)
    auto r2 = sim.positions.pow(2).sum(1);                       // (V,)
    auto field = 20.f * torch::exp(-0.6f * r2);                  // (V,)
    auto vscale = 1.f + 0.3f * torch::arange(B, torch::kFloat32) / B;  // (B,)
    sim.T = kAmbient + field.unsqueeze(0) * vscale.unsqueeze(1); // (B,V)

    ThermalLearner learner(sim, /*seed=*/7, /*lr=*/3e-3f);

    std::vector<float> losses;
    for (int i = 0; i < 200; ++i)
        losses.push_back(learner.train_step(sim, /*sample_count=*/1024));

    auto avg = [&](int lo, int hi) {
        float s = 0.f;
        for (int i = lo; i < hi; ++i) s += losses[i];
        return s / (hi - lo);
    };
    const float first10 = avg(0, 10);
    const float last10 = avg(190, 200);
    for (float l : losses) REQUIRE(std::isfinite(l));
    REQUIRE(last10 < first10);

    // Predictions at arbitrary query points are finite and within bounds.
    auto pts = 1.5f * torch::randn({32, 3});
    auto pred = learner.predict(pts, sim.intensities[0]);
    REQUIRE(pred.numel() == 32);
    REQUIRE(torch::isfinite(pred).all().item<bool>());
    REQUIRE(pred.min().item<float>() >= kAmbient - 1e-3f);
    REQUIRE(pred.max().item<float>() <= kAmbient + kTemperatureSpan + 1e-3f);
}
