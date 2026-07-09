// Physics tests for FieldScope's PIC core (applets/field_scope/em_pic.h): the
// FFT Poisson solve, charge-conserving deposition, the Boris pusher (gyration +
// E×B drift), and momentum conservation of a neutral two-species plasma. Pure
// torch, CPU, its own binary (label "torch"). REQUIRE only — <torch/torch.h>
// defines a bare CHECK macro that shadows doctest's.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "em_pic.h"   // applets/field_scope — on the target's include path

#include <torch/torch.h>
#include <cmath>

using namespace fieldscope;

TEST_CASE("Poisson solve reproduces an analytic sinusoidal mode") {
    const int64_t G = 32;
    const double L = 2.0 * M_PI, dx = L / G;
    const double m = 2.0, k = 2.0 * M_PI * m / L;          // integer grid mode

    auto xs = torch::arange(G, torch::kFloat) * dx;        // (G,)
    auto src = torch::cos(k * xs).reshape({G, 1, 1}).expand({G, G, G}).contiguous();

    auto E = poisson_E(src, L);                            // (G,G,G,3)
    // div^2 phi = -source, phi = cos(kx)/k^2, E_x = -dphi/dx = sin(kx)/k.
    auto Ex_an = (torch::sin(k * xs) / k).reshape({G, 1, 1}).expand({G, G, G});
    REQUIRE(torch::allclose(E.select(3, 0), Ex_an, /*rtol=*/1e-3, /*atol=*/1e-4));
    REQUIRE(E.select(3, 1).abs().max().item<float>() < 1e-4f);
    REQUIRE(E.select(3, 2).abs().max().item<float>() < 1e-4f);
}

TEST_CASE("free-space Poisson gives a 1/r point-charge potential") {
    const int64_t G = 32;
    const double L = 10.0, dx = L / G;
    auto rho = torch::zeros({G, G, G});
    const int c = G / 2;
    rho.index_put_({c, c, c}, 1.0f);                       // unit charge at centre
    auto phi = poisson_phi_free(rho, L);
    for (int d : {4, 6, 8}) {                              // phi(r) ~ 1/(4 pi r)
        const double r = d * dx;
        const float got  = phi.index({c + d, c, c}).item<float>();
        const float want = 1.0f / (4.0f * (float)M_PI * (float)r);
        REQUIRE(got == doctest::Approx(want).epsilon(0.1));
    }
}

TEST_CASE("CIC deposition conserves total charge") {
    torch::manual_seed(0);
    const int64_t N = 5000, G = 32;
    const double L = 2.0 * M_PI;
    auto pos = torch::rand({N, 3}) * L;
    auto rho = deposit_cic(pos, G, L);                     // default charge +1
    REQUIRE(rho.sum().item<float>() == doctest::Approx((float)N).epsilon(1e-4));
    // Signed: a neutral set deposits ~zero net charge.
    auto charge = torch::ones({N});
    charge.index_put_({torch::indexing::Slice(0, N, 2)}, -1.f);
    auto rho_s = deposit_cic(pos, G, L, charge);
    REQUIRE(std::abs(rho_s.sum().item<float>()) < 1e-2f);
}

TEST_CASE("Boris pusher gyrates at the right frequency and conserves energy") {
    const double dt = 0.005;                               // omega = qB/m = 1
    const int steps = (int)std::lround(2.0 * M_PI / dt);   // one full period

    auto vel    = torch::tensor({{1.f, 0.f, 0.f}});
    auto v0     = vel.clone();
    auto p      = torch::zeros({1, 3});
    auto pout   = torch::zeros({1, 3});
    auto accel  = torch::zeros({1, 3});                    // E = 0
    auto charge = torch::ones({1, 1});
    auto B      = torch::tensor({0.f, 0.f, 1.f});

    const float ke0 = 0.5f * (vel * vel).sum().item<float>();
    for (int i = 0; i < steps; ++i) {
        boris_push(p, pout, vel, accel, charge, B, dt);
        p = pout.clone();
    }
    const float ke1 = 0.5f * (vel * vel).sum().item<float>();
    REQUIRE(ke1 == doctest::Approx(ke0).epsilon(1e-4));    // |v| conserved
    REQUIRE(torch::allclose(vel, v0, /*rtol=*/0, /*atol=*/2e-2));  // returns after T
}

TEST_CASE("Boris pusher reproduces the E×B drift") {
    const double dt = 0.005, Ex = 0.5, Bz = 1.0;
    const int steps = (int)std::lround(10.0 * 2.0 * M_PI / dt);

    auto vel    = torch::zeros({1, 3});
    auto p      = torch::zeros({1, 3});
    auto pout   = torch::zeros({1, 3});
    auto accel  = torch::tensor({{(float)Ex, 0.f, 0.f}});  // charge*E, charge=+1
    auto charge = torch::ones({1, 1});
    auto B      = torch::tensor({0.f, 0.f, (float)Bz});

    auto vsum = torch::zeros({1, 3});
    for (int i = 0; i < steps; ++i) {
        boris_push(p, pout, vel, accel, charge, B, dt);
        p = pout.clone();
        vsum += vel;
    }
    auto vmean = vsum / (float)steps;
    auto vd = torch::tensor({{0.f, -(float)(Ex / Bz), 0.f}});   // E×B/|B|^2
    REQUIRE(torch::allclose(vmean, vd, /*rtol=*/0, /*atol=*/3e-2));
}

TEST_CASE("neutral two-species plasma conserves total momentum") {
    torch::manual_seed(1);
    const int64_t N = 8000, G = 16;
    const double L = 2.0 * M_PI, dt = 0.03, coupling = 1.0;

    auto pos = torch::rand({N, 3}) * L;
    auto vel = torch::randn({N, 3}) * 0.2f;
    auto charge = torch::ones({N, 1});
    charge.index_put_({torch::indexing::Slice(0, N, 2)}, -1.f);   // neutral
    auto B    = torch::zeros({3});
    auto pout = torch::zeros_like(pos);

    auto mom0 = vel.sum(0);
    for (int i = 0; i < 40; ++i) {
        auto rho   = deposit_cic(pos, G, L, charge.squeeze(1)) * (float)coupling;
        auto E     = poisson_E(rho, L);
        auto accel = charge * gather_cic(E, pos, L);            // charge*E
        boris_push(pos, pout, vel, accel, charge, B, dt);
        pos = pout.remainder(L);   // keep in the periodic solver domain
    }
    auto mom1 = vel.sum(0);
    // Newton's third law + spectral +k/-k cancellation -> momentum conserved.
    REQUIRE((mom1 - mom0).abs().max().item<float>() < 1e-2f);
}
