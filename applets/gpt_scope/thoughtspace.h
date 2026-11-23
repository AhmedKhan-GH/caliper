// ============================================================================
// ThoughtSpace — the seeded projection basis for GPTScope's residual-stream
// constellation (see docs/applets/in-tree-applets.md).
//
// This header is PURE COMPUTE: torch tensors in, torch tensors out. No ImGui,
// no host headers, no model, no applet state.
// ============================================================================
#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <tuple>

// Basis orthonormalization backend. QR (torch::linalg::qr) is verified present
// in this libtorch (torch/csrc/api/include/torch/linalg.h). Define
// GPTSCOPE_TS_BASIS_QR=0 before including to force the dependency-free
// Gram-Schmidt fallback instead (identical result for a full-rank randn(C,3)).
#ifndef GPTSCOPE_TS_BASIS_QR
#define GPTSCOPE_TS_BASIS_QR 1
#endif

namespace gptscope { namespace ts {

namespace detail {

#if GPTSCOPE_TS_BASIS_QR
// Reduced-QR orthonormal columns of `m (C,3)` -> Q (C,3) for C >= 3.
inline torch::Tensor orthonormal3(const torch::Tensor& m) {
  auto qr = torch::linalg::qr(m, "reduced");   // Q:(C,min(C,3)) R:(min(C,3),3)
  torch::Tensor Q = std::get<0>(qr);
  if (Q.size(1) > 3) Q = Q.narrow(1, 0, 3);    // defensive; reduced already 3
  return Q.contiguous();
}
#else
// Dependency-free Gram-Schmidt on the 3 columns of `m (C,3)`. Identical to the
// reduced-QR Q (up to per-column sign) for a full-rank randn input.
inline torch::Tensor orthonormal3(const torch::Tensor& m) {
  torch::NoGradGuard ng;
  auto c0 = m.select(1, 0);
  auto c1 = m.select(1, 1);
  auto c2 = m.select(1, 2);
  auto u0 = c0 / (c0.norm() + 1e-8);
  auto v1 = c1 - (c1 * u0).sum() * u0;
  auto u1 = v1 / (v1.norm() + 1e-8);
  auto v2 = c2 - (c2 * u0).sum() * u0 - (c2 * u1).sum() * u1;
  auto u2 = v2 / (v2.norm() + 1e-8);
  return torch::stack({u0, u1, u2}, /*dim=*/1).contiguous();  // (C,3)
}
#endif

}  // namespace detail

// ---------------------------------------------------------------------------
// Basis: seeded (C,3) orthonormal projection, stable across frames so all
// on-screen motion is MODEL change, never basis change (§4.2).
// ---------------------------------------------------------------------------
inline torch::Tensor make_basis(int64_t C, uint64_t seed, torch::Device device) {
  torch::NoGradGuard ng;
  // Seeded CPU generator -> deterministic randn -> orthonormalize -> to device.
  torch::Generator g = at::detail::createCPUGenerator(seed);
  auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor m = torch::randn({C, 3}, g, opt);      // (C,3)
  torch::Tensor Q = detail::orthonormal3(m);           // (C,3) orthonormal cols
  return Q.to(device).contiguous();
}

}}  // namespace gptscope::ts
