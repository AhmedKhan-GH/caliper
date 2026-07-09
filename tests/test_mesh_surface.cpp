// Unit tests for MeshScope's learning core (applets/mesh_scope/mesh_model.h):
// that the tanh MLP actually learns the target surface, that the per-vertex
// error map is honest, that the finite-difference normals match the analytic
// normals of a known heightfield, that reset(seed) is reproducible, and — for
// the paint-the-target upgrade — that TargetGrid's bilinear sampler is exact,
// its brush is local/bounded/signed, and training chases an edited target.
//
// Pure torch (no Caliper host), CPU-only, in its own binary (caliper_mesh_tests)
// to keep the fast unit suite free of the torch link cost — the same discipline
// as caliper_sculpt_tests / caliper_em_pic_tests.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "mesh_model.h"   // applets/mesh_scope — on the target's include path

#include <torch/torch.h>

#include <cmath>

using namespace meshscope;

TEST_CASE("training reduces the grid-MSE substantially over 400 steps") {
    const auto dev = torch::kCPU;
    MeshModel model(dev);
    model.reset(7);

    const float mse0 = model.grid_mse();          // fresh net, ~variance of z*
    for (int i = 0; i < 400; ++i) model.train_step();
    const float mse_final = model.grid_mse();

    // Generous margin: a smoke test of the loop's sign, not a benchmark.
    REQUIRE(mse_final < 0.25f * mse0);
}

TEST_CASE("the per-vertex error map is honest") {
    const auto dev = torch::kCPU;
    MeshModel model(dev);
    model.reset(3);
    for (int i = 0; i < 50; ++i) model.train_step();

    auto err = model.err_sq();
    REQUIRE(err.dim() == 1);
    REQUIRE(err.size(0) == (int64_t)kGrid * kGrid);
    REQUIRE(torch::isfinite(err).all().item<bool>());
    REQUIRE((err >= 0.f).all().item<bool>());
    // mean(err^2) is exactly the grid-MSE the status line reports.
    REQUIRE(err.mean().item<float>() ==
            doctest::Approx(model.grid_mse()).epsilon(1e-5));
}

TEST_CASE("finite-difference normals match the analytic normals of a known "
          "heightfield away from borders") {
    const auto dev = torch::kCPU;
    auto xy = grid_xy(dev);
    auto x = xy.select(1, 0), y = xy.select(1, 1);

    // A smooth quadratic heightfield: central differences are exact for it, so
    // the interior comparison is machine-tight. h(x,y) = 0.3xy + 0.15x^2 - 0.1y^2.
    auto z = 0.30f * x * y + 0.15f * x.pow(2) - 0.10f * y.pow(2);
    auto n = finite_diff_normals(z);

    REQUIRE(n.dim() == 2);
    REQUIRE(n.size(0) == (int64_t)kGrid * kGrid);
    REQUIRE(n.size(1) == 3);
    REQUIRE(torch::isfinite(n).all().item<bool>());

    // Unit length everywhere (including one-sided borders).
    auto len = n.norm(2, {1});
    REQUIRE((len - 1.f).abs().max().item<float>() < 1e-3f);

    // Analytic up-normal: (-h_x, 1, -h_y)/|.|, h_x = 0.3y + 0.3x, h_y = 0.3x - 0.2y.
    auto hx = 0.30f * y + 0.30f * x;
    auto hy = 0.30f * x - 0.20f * y;
    auto inv = torch::rsqrt(hx * hx + hy * hy + 1.0f);
    auto analytic = torch::stack({-hx * inv, inv, -hy * inv}, 1);   // (N,3)

    // Compare interior only (borders are one-sided, first-order): mask the outer
    // ring of the kGrid x kGrid grid.
    auto interior = torch::zeros({kGrid, kGrid}, torch::kBool);
    interior.narrow(0, 1, kGrid - 2).narrow(1, 1, kGrid - 2).fill_(true);
    auto mask = interior.reshape({-1});
    auto diff = (n - analytic).norm(2, {1}).index({mask});
    REQUIRE(diff.max().item<float>() < 1e-3f);
}

TEST_CASE("TargetGrid::sample is bilinear-exact at nodes and midpoints of a "
          "hand-set 2x2 cell") {
    const auto dev = torch::kCPU;
    TargetGrid tg(dev);

    // Zero the grid, then hand-set one 2x2 cell (rows 10-11, cols 10-11) to
    // known corner values: nodes read back exactly, edge midpoints are the
    // pairwise means, the cell center is the mean of all four — computable by
    // hand from the bilinear formula.
    const int r0 = 10, c0 = 10;
    const float a = 0.20f, b = -0.40f, c = 0.60f, d = 0.10f;
    tg.grid = torch::zeros({(int64_t)kGrid * kGrid});
    tg.grid[r0 * kGrid + c0]           = a;
    tg.grid[r0 * kGrid + c0 + 1]       = b;
    tg.grid[(r0 + 1) * kGrid + c0]     = c;
    tg.grid[(r0 + 1) * kGrid + c0 + 1] = d;

    const float h = 2.0f * kDomain / (float)(kGrid - 1);
    const float x0 = -kDomain + c0 * h, y0 = -kDomain + r0 * h;

    auto sample1 = [&](float x, float y) {
        return tg.sample(torch::tensor({{x, y}})).item<float>();
    };

    // Nodes exact.
    REQUIRE(sample1(x0, y0)         == doctest::Approx(a).epsilon(1e-5));
    REQUIRE(sample1(x0 + h, y0)     == doctest::Approx(b).epsilon(1e-5));
    REQUIRE(sample1(x0, y0 + h)     == doctest::Approx(c).epsilon(1e-5));
    REQUIRE(sample1(x0 + h, y0 + h) == doctest::Approx(d).epsilon(1e-5));

    // Midpoints: edges = pairwise means, center = mean of all four.
    REQUIRE(sample1(x0 + 0.5f * h, y0) ==
            doctest::Approx(0.5f * (a + b)).epsilon(1e-4));
    REQUIRE(sample1(x0, y0 + 0.5f * h) ==
            doctest::Approx(0.5f * (a + c)).epsilon(1e-4));
    REQUIRE(sample1(x0 + 0.5f * h, y0 + 0.5f * h) ==
            doctest::Approx(0.25f * (a + b + c + d)).epsilon(1e-4));

    // Domain clamping: far-outside coords read the clamped border, finite.
    REQUIRE(std::isfinite(sample1(kDomain + 5.f, -kDomain - 5.f)));
}

TEST_CASE("TargetGrid::brush is local, bounded, and signed") {
    const auto dev = torch::kCPU;
    const float radius = 0.3f;

    TargetGrid tg(dev);
    tg.grid = torch::zeros({(int64_t)kGrid * kGrid});

    // Center the stroke on a lattice node so the exact peak is a grid value.
    const float h = 2.0f * kDomain / (float)(kGrid - 1);
    const float cx = -kDomain + 30 * h, cy = -kDomain + 40 * h;
    tg.brush(cx, cy, radius, 0.5f);

    // Peak == amp at the center node (exp(0) = 1).
    REQUIRE(tg.grid[40 * kGrid + 30].item<float>() ==
            doctest::Approx(0.5f).epsilon(1e-5));

    // Local: nodes beyond 4*radius from the center are numerically unchanged
    // (exp(-32) with sigma = radius/2).
    auto dx = tg.nodes.select(1, 0) - cx;
    auto dy = tg.nodes.select(1, 1) - cy;
    auto far = (dx * dx + dy * dy) > (4.f * radius) * (4.f * radius);
    REQUIRE(tg.grid.index({far}).abs().max().item<float>() < 1e-6f);

    // Signed: negative amp lowers.
    tg.brush(cx, cy, radius, -0.8f);
    REQUIRE(tg.grid[40 * kGrid + 30].item<float>() ==
            doctest::Approx(-0.3f).epsilon(1e-4));

    // Bounded: repeated strokes clamp to [-1, 1], both signs.
    for (int i = 0; i < 10; ++i) tg.brush(cx, cy, radius, 1.0f);
    REQUIRE(tg.grid.max().item<float>() <= 1.0f + 1e-6f);
    REQUIRE(tg.grid[40 * kGrid + 30].item<float>() ==
            doctest::Approx(1.0f).epsilon(1e-5));
    for (int i = 0; i < 30; ++i) tg.brush(cx, cy, radius, -1.0f);
    REQUIRE(tg.grid.min().item<float>() >= -1.0f - 1e-6f);
}

TEST_CASE("the chase: one brush stroke raises the grid-MSE strictly and "
          "further training re-converges on the edited target") {
    const auto dev = torch::kCPU;
    MeshModel model(dev);
    model.reset(11);

    for (int i = 0; i < 800; ++i) model.train_step();
    const float mse_converged = model.grid_mse();

    // One fat stroke the converged net has never seen.
    model.target.brush(0.0f, 0.0f, 0.5f, 0.8f);
    const float mse_stroked = model.grid_mse();
    REQUIRE(mse_stroked > mse_converged);   // strictly worse: the target moved

    for (int i = 0; i < 1200; ++i) model.train_step();
    const float mse_chased = model.grid_mse();
    REQUIRE(mse_chased < 0.25f * mse_stroked);   // it learns the EDITED target
}

TEST_CASE("reset(seed) is deterministic: identical step-10 loss across runs") {
    const auto dev = torch::kCPU;

    auto run = [dev]() {
        MeshModel model(dev);
        model.reset(123);
        float last = 0.f;
        for (int i = 0; i < 10; ++i) last = model.train_step();
        return last;
    };

    const float a = run();
    const float b = run();
    REQUIRE(a == b);
}
