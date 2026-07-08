// Unit tests for SculptScope's learning core (applets/sculpt_scope/sculpt_model.h):
// the energy-distance loss, the analytic target samplers, that the net actually
// learns, and that the addmm_out display fusion equals the net's true forward.
//
// Pure torch (no Caliper host), CPU-only assertions so it runs everywhere. Lives
// in its own binary (caliper_sculpt_tests, label "torch") to keep the fast unit
// suite free of the torch link cost — same discipline as caliper_torch_tests.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "sculpt_model.h"   // applets/sculpt_scope — on the target's include path

#include <torch/torch.h>

using namespace sculptscope;

TEST_CASE("energy distance: zero to self, positive and symmetric across shift") {
    torch::manual_seed(0);
    auto X = torch::randn({128, 3});
    auto Y = X + 0.5f;                                   // a rigid shift

    // E(X,X) == 0 exactly: the self-term diagonals cancel the cross term.
    REQUIRE(energy_distance(X, X).item<float>() == doctest::Approx(0.f).epsilon(1e-4));
    // Distinct distributions -> strictly positive.
    REQUIRE(energy_distance(X, Y).item<float>() > 1e-3f);
    // Symmetric in its arguments.
    REQUIRE(energy_distance(X, Y).item<float>() ==
          doctest::Approx(energy_distance(Y, X).item<float>()).epsilon(1e-5));
}

TEST_CASE("energy distance: gradient is finite and nonzero") {
    torch::manual_seed(1);
    auto X = torch::randn({64, 3}, torch::requires_grad());
    auto Y = torch::randn({64, 3}) + 1.0f;
    auto loss = energy_distance(X, Y);
    loss.backward();
    REQUIRE(X.grad().defined());
    REQUIRE(torch::isfinite(X.grad()).all().item<bool>());
    REQUIRE(X.grad().abs().sum().item<float>() > 0.f);
}

TEST_CASE("target samplers land on their analytic manifold") {
    torch::manual_seed(2);
    const auto dev = torch::kCPU;

    auto s = sample_target(Shape::kSphere, 4096, dev);
    auto radius = s.norm(2, {1});
    REQUIRE(radius.mean().item<float>() == doctest::Approx(1.f).epsilon(1e-4));
    REQUIRE((radius - 1.f).abs().max().item<float>() < 1e-3f);

    auto t = sample_target(Shape::kTorus, 4096, dev);
    auto x = t.select(1, 0), y = t.select(1, 1), z = t.select(1, 2);
    // (sqrt(x^2+y^2) - R)^2 + z^2 == r^2 on the torus surface.
    auto rho = torch::sqrt(x * x + y * y);
    auto constraint = (rho - kTorusR).pow(2) + z * z;    // should equal r^2
    REQUIRE(constraint.mean().item<float>() ==
          doctest::Approx(kTorusr * kTorusr).epsilon(1e-3));
}

TEST_CASE("training reduces the steering loss over a short window") {
    torch::manual_seed(3);
    const auto dev = torch::kCPU;
    SculptNet net;
    net->to(dev);
    torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(2e-3));

    const int64_t N = 4000, B = 256;
    auto z = torch::randn({N, kLatentDim}, torch::TensorOptions(dev));

    auto eval_loss = [&] {
        torch::NoGradGuard ng;
        auto gen = net->forward(z.narrow(0, 0, B));
        auto tgt = sample_target(Shape::kSphere, B, dev);
        return energy_distance(gen, tgt).item<float>();
    };
    const float before = eval_loss();

    for (int i = 0; i < 120; ++i) {
        auto idx = torch::randint(0, N, {B}, torch::TensorOptions(dev).dtype(torch::kLong));
        auto gen = net->forward(z.index_select(0, idx));
        auto tgt = sample_target(Shape::kSphere, B, dev);
        auto loss = energy_distance(gen, tgt);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
    const float after = eval_loss();
    REQUIRE(after < before);      // the net genuinely learned to approach the shell
}

TEST_CASE("addmm_out fusion equals the true forward, bit-for-bit tolerance") {
    torch::manual_seed(4);
    SculptNet net;
    net->eval();
    torch::NoGradGuard ng;
    auto z = torch::randn({512, kLatentDim});

    auto reference = net->forward(z);                    // fc_out(hidden(z))
    // The display path: final layer fused into a preallocated (N,3) slot.
    auto slot = torch::empty({512, 3});
    torch::addmm_out(slot, net->fc_out->bias, net->hidden(z), net->fc_out->weight.t());

    REQUIRE(torch::allclose(slot, reference, /*rtol=*/1e-5, /*atol=*/1e-6));
}
