// Unit tests for MeshScope's learning core (applets/mesh_scope/mesh_model.h):
// that the tanh MLP actually learns the target surface, that the per-vertex
// error map is honest, that the finite-difference normals match the analytic
// normals of a known heightfield, and that reset(seed) is reproducible.
//
// Pure torch (no Caliper host), CPU-only, in its own binary (caliper_mesh_tests)
// to keep the fast unit suite free of the torch link cost — the same discipline
// as caliper_sculpt_tests / caliper_em_pic_tests.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "mesh_model.h"   // applets/mesh_scope — on the target's include path

#include <torch/torch.h>

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
