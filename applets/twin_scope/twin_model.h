#pragma once

// TwinScope v2 thermal model — the surface-aware heat twin and its chasing
// learner (design doc §3 physics, §4 learner, §7 the ONE source-site table).
// Header-only, pure batched torch on top of T6's surface operators
// (twin_surface.h). No host/renderer/UI code: this is the worker-side compute
// the applet (T8) drives and the engine tests exercise in isolation.
//
// v2 replaces v1's UV-space finite-difference diffusion with a genuine surface
// operator: heat flows on the 3-D mesh via the cotangent Laplacian L with
// Voronoi-third masses M, sources deposit as 3-D Gaussian bumps around K = 4
// fixed sites, and the convective loss is area-weighted so the geometry is
// thermally load-bearing (design §3). One batched explicit step services all
// B boundary-condition variants; nothing loops over B or V. Works on CPU and
// CUDA (device is the caller's business — every tensor is built on `device`).
//
// Conventions pinned for the applet (T8):
//   * State T is (B, V_sim) f32, ambient-initialised. Variant 0 is the hero.
//   * The source-site table (positions + base intensities) is defined EXACTLY
//     ONCE here (design §7 — the donor duplicated it across two files).
//   * duty(t) is the seeded on/off schedule (K,); the hero variant may replace
//     a source's schedule with a held user override (design §3/§7).

#include "twin_surface.h"

#include <torch/torch.h>

#include <ATen/ATen.h>

#include <cstdint>
#include <memory>

namespace twinscope {

// ---------------------------------------------------------------------------
// The single source-site table (design §7). K = 4 sites as 3-D points on the
// committed housing surface (asset bounds: x∈[-1.875,1.875], y∈[0,1.2] up,
// z∈[-1,1]). Positions are plausible heat entry points on the housing:
//   0  bolt boss, base corner (−x,−z)      (-1.65, 0.06, -0.75)
//   1  bolt boss, base corner (+x,+z)      ( 1.65, 0.06,  0.75)
//   2  die/core centre, low on the base    ( 0.00, 0.12,  0.00)
//   3  fin-array root, mid height          ( 0.00, 0.50,  0.00)
// Base intensities: the core runs hottest, the bolt bosses conduct board heat,
// the fin root is a weaker secondary. Units are model-relative (calibrated by
// source_gain, not physical W); the twin claim is the dataflow, not FEA (§3).
// ---------------------------------------------------------------------------
constexpr int kSourceCount = 4;
constexpr float kAmbient = 22.0f;        // °C ambient / initial state
constexpr float kTemperatureSpan = 100.0f;  // learner output range: [amb, amb+span]

inline torch::Tensor source_sites(torch::Device device) {
    return torch::tensor({{-1.65f, 0.06f, -0.75f},
                          { 1.65f, 0.06f,  0.75f},
                          { 0.00f, 0.12f,  0.00f},
                          { 0.00f, 0.50f,  0.00f}},
                         torch::TensorOptions(device).dtype(torch::kFloat32));
}

inline torch::Tensor source_base_intensity(torch::Device device) {
    return torch::tensor({0.85f, 0.85f, 1.00f, 0.60f},
                         torch::TensorOptions(device).dtype(torch::kFloat32));
}

struct ThermalConfig {
    float kappa = 1.0f;          // surface diffusivity (scales κ·L·T)
    float cooling = 0.5f;        // convective loss rate h (area-weighted)
    float source_gain = 50.0f;   // deposition scale for the Gaussian injection
    float ambient = kAmbient;    // T_amb
    float span = kTemperatureSpan;
    // Gaussian injection radius σ in WORLD units. 0.15 ≪ the ~3.75-wide housing
    // and the ~0.1-spaced fin pitch — a source heats a compact patch, not the
    // whole face (design §3 "radius ≪ feature size").
    float sigma = 0.15f;
    float variant_spread = 0.40f;  // ± fractional spread of per-variant intensity
    // Duty schedule: each source's on/off period is seeded in [period_lo,hi]
    // seconds with a seeded phase offset; 50% duty. Mutually offset so the
    // field never settles and the net chases forever (design §3 drama).
    float period_lo = 3.0f;
    float period_hi = 7.0f;
    float duty_ratio = 0.5f;
};

// ---------------------------------------------------------------------------
// ThermalSim — device tensors for the batched surface heat step.
//
// Step (design §3):  T ← T + dt·M⁻¹·(−κ·L·T + inject(t) − h·A·(T − T_amb))
//   inject(t) = source_gain · (duty(t) ⊙ intensities) @ gauss_weights
//   A = M (vertex masses): the loss is area-weighted, so with the lumped mass
//   the per-vertex rate is uniform Newton cooling h — the fin geometry sheds
//   heat through the LONG surface diffusion path to the tips, not a per-vertex
//   coefficient (honest surface twin, not FEA).
//
// dt is the explicit-scheme stable step for the FULL operator κL + hM. T6's
// stable_dt bounds only the κL part (0.9·2/max_i 2κL_ii/M_i). Adding the hM
// term adds h to every Gershgorin row of M⁻¹(κL+hM), so the safe step is
//   dt = dt_κ / (1 + h·dt_κ/1.8)
// (1.8 = the 0.9 safety × the factor-2 Gershgorin bound baked into stable_dt).
// This keeps the M-weighted energy non-increasing under zero sources.
// ---------------------------------------------------------------------------
struct ThermalSim {
    torch::Device device = torch::kCPU;
    int64_t B = 0, K = kSourceCount, V = 0;
    double dt = 0.0;
    ThermalConfig cfg;

    torch::Tensor L;             // sparse (V,V) f32 — cotan Laplacian
    torch::Tensor M;             // (V,) f32 — Voronoi-third masses (== A)
    torch::Tensor Minv;          // (V,) f32 — 1/M
    torch::Tensor positions;     // (V,3) f32
    torch::Tensor gauss_weights; // (K,V) f32 — 3-D Gaussian injection weights

    torch::Tensor T;             // (B,V) f32 — temperature state
    torch::Tensor intensities;   // (B,K) f32 — per-variant source strengths
    torch::Tensor active;        // (B,K) f32 — duty⊙intensity used last step

    torch::Tensor periods;       // (K,) f32 — duty period per source (seconds)
    torch::Tensor phases;        // (K,) f32 — duty phase offset per source
    torch::Tensor override_flag; // (K,) bool — hero source under user control
    torch::Tensor override_value;// (K,) f32 — held duty factor when overridden

    // duty(t) — the seeded schedule's on/off factor per source, (K,) f32 in
    // {0,1}. Deterministic in (periods, phases): identical seeds ⇒ identical
    // factors at every t; different seeds ⇒ different periods/phases ⇒ differ.
    torch::Tensor duty(double t) const {
        auto cyc = (static_cast<float>(t) + phases) / periods;
        auto frac = cyc - torch::floor(cyc);
        return (frac < cfg.duty_ratio).to(torch::kFloat32);
    }

    // Per-variant active source factor (B,K): all variants follow the schedule;
    // the hero (variant 0) replaces overridden sources with their held value.
    torch::Tensor active_intensities(double t) const {
        auto d = duty(t);                                   // (K,)
        auto mat = d.unsqueeze(0).expand({B, K}).clone();   // (B,K)
        auto hero = torch::where(override_flag, override_value, d);  // (K,)
        mat.index_put_({0}, hero);
        return mat * intensities;                           // (B,K)
    }

    // One batched explicit surface step at wall time `t` (seconds). Pure torch;
    // no loop over B or V. Caches `active` for the learner's current-duty input.
    void step(double t) {
        auto act = active_intensities(t);                   // (B,K)
        active = act;
        auto Tt = T.t().contiguous();                       // (V,B)
        auto LT = torch::mm(L, Tt).t();                     // (B,V) = (κL·T layout)
        auto inject = cfg.source_gain * torch::mm(act, gauss_weights);  // (B,V)
        auto loss = cfg.cooling * M * (T - cfg.ambient);    // (B,V), A=M broadcast
        auto rhs = -cfg.kappa * LT + inject - loss;         // (B,V)
        T = T + static_cast<float>(dt) * (Minv * rhs);      // Minv (V,) broadcast
    }
};

// Build a sim on `device` from a (already subdivided) sim mesh. Variant 0 is
// the hero (exact base intensities); variants 1.. get a deterministic seeded
// spread. Periods/phases are seeded from the same generator (so the whole
// schedule is reproducible per seed). State starts at ambient.
inline ThermalSim make_thermal_sim(const SurfaceMesh& mesh, int64_t batch,
                                   torch::Device device, uint64_t seed = 1,
                                   const ThermalConfig& cfg = {}) {
    TORCH_CHECK(batch >= 1, "batch must be >= 1");
    ThermalSim s;
    s.device = device;
    s.cfg = cfg;
    s.B = batch;
    s.K = kSourceCount;

    // Operators (built on CPU by T6, then moved to device).
    auto Lcpu = cotan_laplacian(mesh);
    auto Mcpu = vertex_masses(mesh);
    const double dt_k = stable_dt(Lcpu, Mcpu, cfg.kappa);
    s.dt = dt_k / (1.0 + static_cast<double>(cfg.cooling) * dt_k / 1.8);

    s.L = Lcpu.to(device);
    s.M = Mcpu.to(device);
    s.Minv = (1.0f / Mcpu).to(device);
    s.positions = mesh.positions.to(device, torch::kFloat32).contiguous();
    s.V = s.positions.size(0);

    // Gaussian injection weights (K,V) from 3-D Euclidean distance to each site.
    auto sites = source_sites(device);                       // (K,3)
    auto diff = sites.unsqueeze(1) - s.positions.unsqueeze(0);  // (K,V,3)
    auto d2 = diff.pow(2).sum(-1);                           // (K,V)
    const float inv2s2 = 1.0f / (2.0f * cfg.sigma * cfg.sigma);
    s.gauss_weights = torch::exp(-d2 * inv2s2).contiguous(); // (K,V)

    // Per-variant intensities: hero exact, others seeded ± spread.
    torch::Generator gen = at::detail::createCPUGenerator(seed);
    auto cpu_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto base = source_base_intensity(torch::kCPU);          // (K,)
    auto noise = torch::rand({batch, s.K}, gen, cpu_f);      // [0,1)
    auto scale = 1.0f + cfg.variant_spread * (noise - 0.5f) * 2.0f;  // [1∓spread]
    auto inten = base.unsqueeze(0) * scale;                  // (B,K)
    inten.index_put_({0}, base);                             // hero exact
    s.intensities = inten.to(device).contiguous();

    // Seeded duty schedule (same generator ⇒ reproducible per seed).
    auto pr = cfg.period_lo +
              (cfg.period_hi - cfg.period_lo) * torch::rand({s.K}, gen, cpu_f);
    auto ph = pr * torch::rand({s.K}, gen, cpu_f);
    s.periods = pr.to(device);
    s.phases = ph.to(device);
    s.override_flag = torch::zeros({s.K}, torch::TensorOptions(device).dtype(torch::kBool));
    s.override_value = torch::zeros({s.K}, torch::TensorOptions(device).dtype(torch::kFloat32));

    // State + a valid pre-step `active` (duty at t=0), so the learner can train
    // before the first step is taken.
    s.T = torch::full({batch, s.V}, cfg.ambient,
                      torch::TensorOptions(device).dtype(torch::kFloat32));
    s.active = s.active_intensities(0.0);
    return s;
}

// ---------------------------------------------------------------------------
// ThermalLearner — the chasing MLP (design §4). f_θ(x,y,z, s_1..s_K) → T, a
// 3+K → 64 → 64 → 1 net trained every step on random (variant, vertex) samples
// across ALL variants. Adam. Sampling randomness flows through a LOCAL
// torch::Generator (NOT global torch::manual_seed): the donor perturbed the
// process-global RNG on every reset (review Minor); a local generator keeps
// the sampler reproducible without touching net init or any other torch RNG
// consumer (design §4 — "seed via a local Generator; accept deterministic-
// enough init").
//
// Positions are normalised to ~[-1,1] by the mesh bbox (seam-free 3-D input);
// the sigmoid output is denormalised to [ambient, ambient+span], so every
// prediction is finite and in-bounds by construction.
// ---------------------------------------------------------------------------
struct ThermalNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    explicit ThermalNetImpl(int64_t sources) {
        fc1 = register_module("fc1", torch::nn::Linear(3 + sources, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 1));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::silu(fc1->forward(x));
        x = torch::silu(fc2->forward(x));
        return torch::sigmoid(fc3->forward(x)).squeeze(-1);  // (N,) in [0,1]
    }
};
TORCH_MODULE(ThermalNet);

struct ThermalLearner {
    ThermalNet net{nullptr};
    std::unique_ptr<torch::optim::Adam> optimizer;
    torch::Generator gen;
    torch::Device device;
    int64_t K;
    float ambient, span;
    torch::Tensor center, half;  // (3,) position normalisation, on device

    ThermalLearner(const ThermalSim& sim, uint64_t seed = 7,
                   float learning_rate = 2e-3f)
        : gen(at::detail::createCPUGenerator(seed)),
          device(sim.device),
          K(sim.K),
          ambient(sim.cfg.ambient),
          span(sim.cfg.span) {
        net = ThermalNet(sim.K);
        net->to(device);
        optimizer = std::make_unique<torch::optim::Adam>(
            net->parameters(), torch::optim::AdamOptions(learning_rate));
        auto lo = std::get<0>(sim.positions.min(0));
        auto hi = std::get<0>(sim.positions.max(0));
        center = (0.5f * (lo + hi)).detach();
        half = (0.5f * (hi - lo)).clamp_min(1e-4f).detach();
    }

    torch::Tensor normalize_pos(const torch::Tensor& p) const {
        return (p - center) / half;  // (N,3) ~ [-1,1]
    }

    // One batched fwd/bwd on `sample_count` random (variant, vertex) pairs from
    // ALL variants. Inputs are (x,y,z, current-duty-scaled source strengths of
    // that variant); target is the sim state at that vertex. Returns the loss.
    float train_step(const ThermalSim& sim, int64_t sample_count = 4096) {
        auto cpu_l = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
        auto which = torch::randint(sim.B, {sample_count}, gen, cpu_l).to(device);
        auto vtx = torch::randint(sim.V, {sample_count}, gen, cpu_l).to(device);

        auto pos = sim.positions.index_select(0, vtx);        // (n,3)
        auto src = sim.active.index_select(0, which);         // (n,K)
        auto input = torch::cat({normalize_pos(pos), src}, 1);// (n,3+K)

        auto flatT = sim.T.reshape({sim.B * sim.V});
        auto target_real = flatT.index_select(0, which * sim.V + vtx);  // (n,)
        auto target = ((target_real - ambient) / span).clamp(0.f, 1.f);

        auto pred = net->forward(input);                      // (n,) in [0,1]
        auto loss = torch::mse_loss(pred, target);
        optimizer->zero_grad();
        loss.backward();
        optimizer->step();
        return loss.item<float>();
    }

    // Evaluate f_θ at arbitrary 3-D points with source strengths `s` (K,).
    // Returns (N,) temperatures in [ambient, ambient+span]. No grad.
    torch::Tensor predict(const torch::Tensor& points, const torch::Tensor& s) {
        torch::NoGradGuard ng;
        auto p = points.to(device, torch::kFloat32);
        const int64_t N = p.size(0);
        auto src = s.to(device, torch::kFloat32).reshape({1, K}).expand({N, K});
        auto input = torch::cat({normalize_pos(p), src}, 1);
        return ambient + span * net->forward(input);         // (N,)
    }
};

}  // namespace twinscope
