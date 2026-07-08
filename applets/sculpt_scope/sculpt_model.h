#pragma once
// ============================================================================
// SculptScope model — a generator MLP g_θ: R^k -> R^3, plus the two pieces of
// pure-torch logic the applet's worker uses and the unit tests pin: the
// energy-distance distribution loss and the analytic target-shape samplers.
//
// Everything here is host-free (only <torch/torch.h>), so tests/test_sculpt.cpp
// exercises the learning math without a Caliper host. The facade at the bottom
// mirrors field_scope/embed_scope: heavy state lives in sculpt_scope.cpp.
//
// The fusion op lives in the applet, not here, but the shapes are fixed by this
// module: fc_out->weight is (3,128), fc_out->bias is (3,), so the display path
//   torch::addmm_out(slot, fc_out->bias, hidden(z), fc_out->weight.t())
// writes the net's true (N,3) output straight into the pool-born render slot.
// ============================================================================
#include <torch/torch.h>

namespace sculptscope {

constexpr int kLatentDim = 3;    // g_θ input width
constexpr int kHidden    = 128;  // hidden width (fc_out reads this)

// ---- generator network -----------------------------------------------------
// Linear(k,128)->SiLU ->Linear(128,128)->SiLU ->Linear(128,128)->SiLU
//  ->Linear(128,3). hidden() stops at the last SiLU so the applet can fuse the
// final layer into the render slot via addmm_out.
struct SculptNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc_out{nullptr};

    SculptNetImpl() {
        fc1    = register_module("fc1",    torch::nn::Linear(kLatentDim, kHidden));
        fc2    = register_module("fc2",    torch::nn::Linear(kHidden, kHidden));
        fc3    = register_module("fc3",    torch::nn::Linear(kHidden, kHidden));
        fc_out = register_module("fc_out", torch::nn::Linear(kHidden, 3));
    }

    // (M,k) -> (M,128): everything up to and including the last SiLU.
    torch::Tensor hidden(torch::Tensor z) {
        z = torch::silu(fc1->forward(z));
        z = torch::silu(fc2->forward(z));
        z = torch::silu(fc3->forward(z));
        return z;
    }

    // (M,k) -> (M,3): the generated points.
    torch::Tensor forward(torch::Tensor z) { return fc_out->forward(hidden(z)); }
};
TORCH_MODULE(SculptNet);

// ---- target shapes ---------------------------------------------------------
enum class Shape { kSphere = 0, kTorus = 1, kHelix = 2, kTwoLobes = 3 };

// Torus radii — shared with the sampler test's analytic constraint.
constexpr float kTorusR = 0.9f;   // center-of-tube radius
constexpr float kTorusr = 0.35f;  // tube radius

// Draw M points from the target manifold on `device`. Analytic and fresh every
// call (unlimited target support; the energy loss needs no correspondence).
inline torch::Tensor sample_target(Shape shape, int64_t M, torch::Device device) {
    auto opt = torch::TensorOptions(device).dtype(torch::kFloat32);
    const float pi = 3.14159265358979323846f;
    switch (shape) {
        case Shape::kSphere: {
            // Gaussian directions normalized -> uniform-ish on the unit shell.
            auto v = torch::randn({M, 3}, opt);
            return v / v.norm(2, {1}, /*keepdim=*/true).clamp_min(1e-6f);
        }
        case Shape::kTorus: {
            auto th = torch::rand({M}, opt) * (2 * pi);
            auto ph = torch::rand({M}, opt) * (2 * pi);
            auto x = (kTorusR + kTorusr * torch::cos(ph)) * torch::cos(th);
            auto y = (kTorusR + kTorusr * torch::cos(ph)) * torch::sin(th);
            auto z = kTorusr * torch::sin(ph);
            return torch::stack({x, y, z}, 1);
        }
        case Shape::kHelix: {
            // A tube around a 2-turn helix; z climbs from -1 to 1.
            auto t  = torch::rand({M}, opt) * (4 * pi);
            auto cx = 0.8f * torch::cos(t), cy = 0.8f * torch::sin(t);
            auto cz = t / (2 * pi) - 1.f;
            auto jitter = torch::randn({M, 3}, opt) * 0.06f;
            return torch::stack({cx, cy, cz}, 1) + jitter;
        }
        case Shape::kTwoLobes: default: {
            // Two Gaussian blobs at (±0.7,0,0) — the "embed_scope look", no data.
            auto blob = torch::randn({M, 3}, opt) * 0.25f;
            auto sign = (torch::rand({M, 1}, opt) < 0.5f).to(torch::kFloat32) * 2.f - 1.f;
            auto shift = torch::zeros({M, 3}, opt);
            shift.select(1, 0).copy_(sign.squeeze(1) * 0.7f);
            return blob + shift;
        }
    }
}

// ---- energy distance -------------------------------------------------------
// E(X,Y) = 2·mean‖xᵢ−yⱼ‖ − mean‖xᵢ−xᵢ′‖ − mean‖yⱼ−yⱼ′‖  (Székely's energy
// distance): a correspondence-free, differentiable distribution metric. Zero
// iff the empirical distributions coincide; the self-term diagonals (‖x−x‖=0)
// cancel between the cross and self means when X==Y, so E(X,X)==0 exactly.
inline torch::Tensor energy_distance(const torch::Tensor& X, const torch::Tensor& Y) {
    auto dxy = torch::cdist(X, Y);
    auto dxx = torch::cdist(X, X);
    auto dyy = torch::cdist(Y, Y);
    return 2.f * dxy.mean() - dxx.mean() - dyy.mean();
}

}  // namespace sculptscope
