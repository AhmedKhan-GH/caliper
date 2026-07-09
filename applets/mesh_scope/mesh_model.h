#pragma once
// ============================================================================
// MeshScope model — the pure-torch learning core of the geometry.v1_1 surface
// exemplar: the fixed target surface z*(x,y), the 2->64->64->1 tanh MLP that
// learns it, one Adam/MSE training step on a 512-point continuous-coord
// minibatch, the full-grid prediction, per-vertex squared error, and the
// central finite-difference surface normals.
//
// Host-free (only <torch/torch.h>), so tests/test_mesh_surface.cpp exercises
// the math on the CPU with no Caliper host — the sibling pattern of
// sculpt_scope/sculpt_model.h. The applet file (mesh_scope.cpp) owns the UI,
// the services, and the triple-buffered pool plumbing; everything a unit test
// can pin lives here.
//
// Layout convention (shared with mesh_scope.cpp's static index buffers): the
// kGrid x kGrid vertex i = row*kGrid + col, with row indexing y and col
// indexing x, so x varies fastest. The surface height is written into the
// world Y (up) axis: a vertex is (x, z, y); normals come back (x, up, z).
// ============================================================================
#include <torch/torch.h>

#include <cstdint>
#include <memory>

namespace meshscope {

constexpr int   kGrid   = 72;     // vertices per side (kGrid^2 total)
constexpr float kDomain = 1.6f;   // surface spans [-kDomain, kDomain]^2
constexpr int   kBatch  = 512;    // training minibatch (continuous coords)
constexpr int   kHidden = 64;     // MLP hidden width

// ---- target surface --------------------------------------------------------
// z*(x,y): two Gaussian lobes (one up, one down) plus a low ripple, amplitude
// ~±0.4 over the domain. Visually distinct features so the net's capture order
// is watchable. Pure analytic, vectorized: coords (M,2) -> heights (M,). Now the
// *preset* for the mutable TargetGrid below (paint-the-target upgrade).
inline torch::Tensor target_z(const torch::Tensor& xy) {
    auto x = xy.select(1, 0), y = xy.select(1, 1);
    auto bump  =  0.38f * torch::exp(-3.0f * ((x - 0.55f).pow(2) + (y - 0.55f).pow(2)));
    auto pit   = -0.34f * torch::exp(-3.2f * ((x + 0.60f).pow(2) + (y + 0.50f).pow(2)));
    auto ripple = 0.06f * torch::sin(3.0f * x) * torch::cos(3.0f * y);
    return bump + pit + ripple;
}

// ---- the learner -----------------------------------------------------------
// Linear(2,64)->tanh ->Linear(64,64)->tanh ->Linear(64,1). forward returns the
// (M,) height prediction (last dim squeezed away).
struct MeshNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    MeshNetImpl() {
        fc1 = register_module("fc1", torch::nn::Linear(2, kHidden));
        fc2 = register_module("fc2", torch::nn::Linear(kHidden, kHidden));
        fc3 = register_module("fc3", torch::nn::Linear(kHidden, 1));
    }

    torch::Tensor forward(torch::Tensor c) {
        c = torch::tanh(fc1->forward(c));
        c = torch::tanh(fc2->forward(c));
        return fc3->forward(c).squeeze(-1);   // (M,)
    }
};
TORCH_MODULE(MeshNet);

// ---- the regular grid ------------------------------------------------------
// (N,2) coords over [-kDomain,kDomain]^2 in the i = row*kGrid + col layout
// (x fastest), matching mesh_scope.cpp's triangle/line index buffers.
inline torch::Tensor grid_xy(torch::Device device) {
    auto opt = torch::TensorOptions(device).dtype(torch::kFloat32);
    auto lin = torch::linspace(-kDomain, kDomain, kGrid, opt);   // (kGrid,)
    auto gx = lin.view({1, kGrid}).expand({kGrid, kGrid}).reshape({-1});  // col -> x
    auto gy = lin.view({kGrid, 1}).expand({kGrid, kGrid}).reshape({-1});  // row -> y
    return torch::stack({gx, gy}, 1);   // (N,2)
}

// ---- the mutable target grid -----------------------------------------------
// The answer key you can paint. `grid` (N,) holds the target height at each
// lattice node (i = row*kGrid + col, x fastest — the shared layout). Init from
// the analytic preset above; `brush` raises/lowers a Gaussian bump; `sample`
// bilinearly reads a continuous coord for the training minibatch loss. Owned by
// the worker thread — the frame thread only ever enqueues brush requests.
struct TargetGrid {
    torch::Tensor grid;    // (N,) mutable target heights on the lattice
    torch::Tensor nodes;   // (N,2) constant node coords (for brush distances)
    torch::Device device;

    explicit TargetGrid(torch::Device dev) : device(dev) {
        nodes = grid_xy(device);
        reset_preset();
    }

    // Restore the two-lobe + ripple analytic preset (the cold-open demo).
    void reset_preset() {
        torch::NoGradGuard ng;
        grid = target_z(nodes).contiguous();
    }

    // Bilinear read of `grid` at continuous domain coords (B,2) -> heights (B,).
    // Coords are clamped to the domain (a brush can't push a sample outside).
    torch::Tensor sample(const torch::Tensor& xy) const {
        torch::NoGradGuard ng;
        const float h = 2.0f * kDomain / static_cast<float>(kGrid - 1);
        auto x = xy.select(1, 0).clamp(-kDomain, kDomain);
        auto y = xy.select(1, 1).clamp(-kDomain, kDomain);
        auto fx = (x + kDomain) / h;                  // [0, kGrid-1]
        auto fy = (y + kDomain) / h;
        auto x0 = fx.floor().clamp(0.0, kGrid - 2);   // clamp so x0+1 is valid
        auto y0 = fy.floor().clamp(0.0, kGrid - 2);
        auto tx = fx - x0;
        auto ty = fy - y0;
        auto c0 = x0.to(torch::kLong), c1 = c0 + 1;
        auto r0 = y0.to(torch::kLong), r1 = r0 + 1;
        auto v00 = grid.index_select(0, r0 * kGrid + c0);
        auto v01 = grid.index_select(0, r0 * kGrid + c1);
        auto v10 = grid.index_select(0, r1 * kGrid + c0);
        auto v11 = grid.index_select(0, r1 * kGrid + c1);
        auto v0 = v00 * (1.0f - tx) + v01 * tx;       // interp along x, row r0
        auto v1 = v10 * (1.0f - tx) + v11 * tx;       //                  row r1
        return v0 * (1.0f - ty) + v1 * ty;            // interp along y
    }

    // Add a Gaussian bump centered at domain (cx,cy). sigma = radius/2 (the
    // brush "radius" is ~2 sigma, its visible edge); result clamped to [-1,1] so
    // the paint can't push the camera framing out of range.
    void brush(float cx, float cy, float radius, float amp) {
        torch::NoGradGuard ng;
        const float sigma = radius * 0.5f;
        auto dx = nodes.select(1, 0) - cx;
        auto dy = nodes.select(1, 1) - cy;
        auto bump = amp * torch::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
        grid = (grid + bump).clamp(-1.0f, 1.0f);
    }
};

// ---- central finite-difference normals -------------------------------------
// z_flat: (N,) heights on the regular grid (layout above). Central differences
// on the interior, one-sided at the four borders, then normalize
// (-dz/dx, 1, -dz/dy). Returns (N,3) in world (x, up, z). No autograd.
inline torch::Tensor finite_diff_normals(const torch::Tensor& z_flat) {
    const float h = 2.0f * kDomain / static_cast<float>(kGrid - 1);
    auto Z = z_flat.view({kGrid, kGrid});   // [row = y, col = x]
    auto dzdx = torch::empty_like(Z);
    auto dzdy = torch::empty_like(Z);

    // d/dx along columns (dim 1).
    dzdx.narrow(1, 1, kGrid - 2)
        .copy_((Z.narrow(1, 2, kGrid - 2) - Z.narrow(1, 0, kGrid - 2)) / (2.0f * h));
    dzdx.narrow(1, 0, 1).copy_((Z.narrow(1, 1, 1) - Z.narrow(1, 0, 1)) / h);
    dzdx.narrow(1, kGrid - 1, 1)
        .copy_((Z.narrow(1, kGrid - 1, 1) - Z.narrow(1, kGrid - 2, 1)) / h);

    // d/dy along rows (dim 0).
    dzdy.narrow(0, 1, kGrid - 2)
        .copy_((Z.narrow(0, 2, kGrid - 2) - Z.narrow(0, 0, kGrid - 2)) / (2.0f * h));
    dzdy.narrow(0, 0, 1).copy_((Z.narrow(0, 1, 1) - Z.narrow(0, 0, 1)) / h);
    dzdy.narrow(0, kGrid - 1, 1)
        .copy_((Z.narrow(0, kGrid - 1, 1) - Z.narrow(0, kGrid - 2, 1)) / h);

    auto gx = dzdx.reshape({-1});
    auto gy = dzdy.reshape({-1});
    auto inv = torch::rsqrt(gx * gx + gy * gy + 1.0f);
    return torch::stack({-gx * inv, inv, -gy * inv}, 1);   // (N,3)
}

// ---- the model wrapper -----------------------------------------------------
// Owns the net, the Adam optimizer, the constant grid + its target heights, and
// the last minibatch (for the training-sample point overlay). Both the worker
// thread (mesh_scope.cpp) and the unit tests drive it.
struct MeshModel {
    MeshNet net{nullptr};
    std::unique_ptr<torch::optim::Adam> opt;
    torch::Device device;
    float lr;

    torch::Tensor grid;       // (N,2) constant coords
    TargetGrid    target;     // mutable target heights (paint-the-target)
    torch::Tensor batch_xy;   // (B,2) last training minibatch coords
    torch::Tensor batch_pred; // (B,)  net height at that minibatch (current θ)

    explicit MeshModel(torch::Device dev, float lr_ = 3e-3f)
        : device(dev), lr(lr_), target(dev) {
        grid = grid_xy(device);
        reset(0);
    }

    // Deterministic re-init of weights + optimizer (guards the reproducible-demo
    // property). Re-seeds the global generator so the init and the subsequent
    // minibatch stream are identical across runs from the same seed.
    void reset(uint64_t seed) {
        torch::manual_seed(seed);
        net = MeshNet();
        net->to(device);
        opt = std::make_unique<torch::optim::Adam>(
            net->parameters(), torch::optim::AdamOptions(lr));
        auto opt_f = torch::TensorOptions(device).dtype(torch::kFloat32);
        batch_xy   = torch::zeros({kBatch, 2}, opt_f);
        batch_pred = torch::zeros({kBatch}, opt_f);
    }

    void set_lr(float lr_) {
        lr = lr_;
        for (auto& g : opt->param_groups())
            static_cast<torch::optim::AdamOptions&>(g.options()).lr(lr_);
    }

    // One optimizer step on a fresh 512-point uniform minibatch; returns the MSE
    // loss value. Records the minibatch and the current-θ heights at it.
    float train_step() {
        auto opt_f = torch::TensorOptions(device).dtype(torch::kFloat32);
        auto batch = torch::rand({kBatch, 2}, opt_f) * (2.0f * kDomain) - kDomain;
        auto tgt   = target.sample(batch);
        auto pred  = net->forward(batch);
        auto loss  = torch::mse_loss(pred, tgt);
        opt->zero_grad();
        loss.backward();
        opt->step();
        {
            torch::NoGradGuard ng;
            batch_xy = batch.detach();
            batch_pred = net->forward(batch).detach();   // heights at post-step θ
        }
        return loss.item<float>();
    }

    // Full-grid prediction (N,), no autograd.
    torch::Tensor grid_pred() {
        torch::NoGradGuard ng;
        return net->forward(grid);
    }

    // Per-vertex squared error (N,) against the grid target, no autograd.
    torch::Tensor err_sq() {
        torch::NoGradGuard ng;
        auto p = net->forward(grid);
        return (p - target.grid).pow(2);
    }

    // Mean squared error over the whole grid (the "grid-MSE" the status reports).
    float grid_mse() { return err_sq().mean().item<float>(); }
};

}  // namespace meshscope
