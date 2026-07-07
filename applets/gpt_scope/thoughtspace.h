// ============================================================================
// ThoughtSpace — pure-compute geometry + attribute layout for GPTScope's
// residual-stream constellation (see
//   docs/superpowers/specs/2026-07-07-gptscope-thoughtspace-design.md §4).
//
// This header is PURE COMPUTE: torch tensors in, torch tensors written in
// place. No ImGui, no host headers, no model, no applet state. A separate
// integrator calls these from the worker to fill the caller's pool buffers
// (`pos_out (N_max,3) f32`, `attr_out (N_max,) f32`) which a renderer then
// draws zero-copy.
//
// The buffer is ONE contiguous layout, documented in §4.3 and implemented
// verbatim here:
//   [ probe stations (D,S,T) ] [ probe trails (D-1,K,S,T) ]
//   [ gen   stations (D,T)   ] [ gen   trails (D-1,K,T)   ]
// all flattened row-major (leftmost dim outermost). Positions and attributes
// use the SAME station/trail ordering and the SAME lerp fractions
// `(k+1)/(K+1)` for k in [0,K), so a trail point's colour matches its place.
//
// Only pos_out / attr_out are the caller's pool buffers we write into; every
// temporary here lands in torch's DEFAULT allocator (expected and fine).
// ============================================================================
#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <tuple>
#include <vector>

// Basis orthonormalization backend. QR (torch::linalg::qr) is verified present
// in this libtorch (torch/csrc/api/include/torch/linalg.h). Define
// GPTSCOPE_TS_BASIS_QR=0 before including to force the dependency-free
// Gram-Schmidt fallback instead (identical result for a full-rank randn(C,3)).
#ifndef GPTSCOPE_TS_BASIS_QR
#define GPTSCOPE_TS_BASIS_QR 1
#endif

namespace gptscope { namespace ts {

// ---------------------------------------------------------------------------
// Dimensions. All counts are exact row spans into pos_out/attr_out.
//   S sequences, T tokens/seq, D depth stations, K trail points per segment,
//   C residual width. Row-major ordering is the contract (§4.3).
// ---------------------------------------------------------------------------
struct Dims {
  int64_t S, T, D, K, C;
  int64_t stations()     const { return D * S * T; }                 // probe stations
  int64_t trails()       const { return (D - 1) * K * S * T; }        // probe trails
  int64_t n_probe()      const { return stations() + trails(); }
  int64_t gen_stations() const { return D * T; }                     // one gen thread
  int64_t gen_trails()   const { return (D - 1) * K * T; }
  int64_t n_gen()        const { return gen_stations() + gen_trails(); }
  int64_t n_max()        const { return n_probe() + n_gen(); }
};

// ---------------------------------------------------------------------------
// internal helpers (anonymous-ish; inline so header-only, no ODR issues)
// ---------------------------------------------------------------------------
namespace detail {

// Shared lerp fractions (k+1)/(K+1) for k in [0,K), shape (K,) f32 on `device`.
// The SINGLE source of truth for both position and attribute interpolation, so
// a trail point's geometry and colour always correspond.
inline torch::Tensor lerp_fracs(int64_t K, torch::Device device) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  // arange(1..K) / (K+1)  ->  [1/(K+1), ..., K/(K+1)]
  return torch::arange(1, K + 1, opt) / static_cast<double>(K + 1);
}

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

// Project a residual block `resid (D, *, C)` to `(D, *, 3)`.
//   - If !raw_norms: divide each depth station d by its per-depth MEAN L2 norm
//     (mean over the spatial dims of the per-token ||.||_2), detached, +1e-8.
//   - coord = normed @ basis   (basis (C,3)).
// Returns the UNSCALED projected coordinates.
inline torch::Tensor project(const torch::Tensor& resid,
                             const torch::Tensor& basis, bool raw_norms) {
  torch::NoGradGuard ng;
  const int64_t nd = resid.dim();               // >= 3 (probe: 4, gen: 3)
  torch::Tensor normed = resid;
  if (!raw_norms) {
    // per-token L2 norm -> (D, *spatial*)
    auto norms = resid.norm(/*p=*/2, /*dim=*/{-1}, /*keepdim=*/false);
    // reduce every spatial dim (everything after depth) to a per-depth scalar.
    std::vector<int64_t> reduce_dims;
    for (int64_t i = 1; i < norms.dim(); ++i) reduce_dims.push_back(i);
    auto mean_d = norms.mean(reduce_dims).detach() + 1e-8;   // (D,)
    // broadcast shape (D,1,...,1) matching resid rank.
    std::vector<int64_t> vshape(static_cast<size_t>(nd), 1);
    vshape[0] = mean_d.size(0);
    normed = resid / mean_d.view(vshape);
  }
  // f32 cast on both operands keeps matmul valid regardless of resid dtype.
  return torch::matmul(normed.to(torch::kFloat32), basis.to(torch::kFloat32));
}

// Expand adjacent-station endpoints `proj (D, *, 3)` into trail points
// `(D-1, K, *, 3)` using the shared lerp fractions, then flatten to (trails,3).
// Row-major (seg, k, *spatial*) — matches attr trail ordering exactly.
inline torch::Tensor trail_positions(const torch::Tensor& proj, int64_t K,
                                     const torch::Tensor& fracs) {
  const int64_t D = proj.size(0);
  auto a = proj.narrow(0, 0, D - 1);            // (D-1, *, 3)  segment starts
  auto b = proj.narrow(0, 1, D - 1);            // (D-1, *, 3)  segment ends
  auto diff = (b - a).unsqueeze(1);             // (D-1, 1, *, 3)
  auto a1   = a.unsqueeze(1);                    // (D-1, 1, *, 3)
  // frac broadcast shape: (1, K, 1, ..., 1) matching a1's rank.
  std::vector<int64_t> fshape(static_cast<size_t>(a1.dim()), 1);
  fshape[1] = K;
  auto tr = a1 + fracs.view(fshape) * diff;     // (D-1, K, *, 3)
  return tr.reshape({-1, 3});                    // (trails, 3)
}

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

// ---------------------------------------------------------------------------
// Probe positions -> pos_out rows [0, n_probe).
// resid is (D,S,T,C); pos_out is (n_max,3) f32 on resid's device.
// Stations first (D*S*T rows, row-major d,s,t), then trails ((D-1)*K*S*T rows,
// row-major seg,k,s,t) with trail = lerp(station[seg], station[seg+1],
// (k+1)/(K+1)). Everything scaled by `scale`. Returns the max |coord| BEFORE
// scaling, so the caller can keep a smoothed fit scale across frames.
// ---------------------------------------------------------------------------
inline float write_probe_positions(torch::Tensor& pos_out,
                                   const torch::Tensor& resid,
                                   const torch::Tensor& basis, const Dims& dm,
                                   bool raw_norms, float scale) {
  torch::NoGradGuard ng;
  // proj: (D,S,T,3) unscaled projected station coords.
  auto proj = detail::project(resid, basis, raw_norms);
  const float max_abs = proj.abs().max().item<float>();   // report pre-scale

  auto proj_s = proj * scale;                              // (D,S,T,3) scaled
  auto fracs  = detail::lerp_fracs(dm.K, proj_s.device()); // shared fractions

  // Stations block: flatten (D,S,T,3) -> (D*S*T,3), row-major d,s,t.
  pos_out.narrow(0, 0, dm.stations())
         .copy_(proj_s.reshape({dm.stations(), 3}));

  // Trails block: lerp adjacent scaled stations -> (D-1,K,S,T,3) -> flatten.
  pos_out.narrow(0, dm.stations(), dm.trails())
         .copy_(detail::trail_positions(proj_s, dm.K, fracs));

  return max_abs;
}

// ---------------------------------------------------------------------------
// Generated-sequence positions -> pos_out rows [n_probe, n_max).
// resid_gen is (D,T,C). Same normalization convention and the caller-supplied
// `scale` (NOT recomputed). Stations (d,t) then trails (seg,k,t).
// ---------------------------------------------------------------------------
inline void write_gen_positions(torch::Tensor& pos_out,
                               const torch::Tensor& resid_gen,
                               const torch::Tensor& basis, const Dims& dm,
                               bool raw_norms, float scale) {
  torch::NoGradGuard ng;
  auto proj_s = detail::project(resid_gen, basis, raw_norms) * scale;  // (D,T,3)
  auto fracs  = detail::lerp_fracs(dm.K, proj_s.device());

  const int64_t base = dm.n_probe();
  pos_out.narrow(0, base, dm.gen_stations())
         .copy_(proj_s.reshape({dm.gen_stations(), 3}));               // (D*T,3)
  pos_out.narrow(0, base + dm.gen_stations(), dm.gen_trails())
         .copy_(detail::trail_positions(proj_s, dm.K, fracs));         // ((D-1)*K*T,3)
}

// ---------------------------------------------------------------------------
// Attr: depth mode. Rows [0, n_probe). Station value = d/(D-1); trails lerp
// between adjacent depths with the SAME fractions as positions.
// ---------------------------------------------------------------------------
inline void write_attr_depth(torch::Tensor& attr_out, const Dims& dm) {
  torch::NoGradGuard ng;
  const int64_t D = dm.D, K = dm.K, S = dm.S, T = dm.T;
  auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(attr_out.device());
  const double denom = D > 1 ? static_cast<double>(D - 1) : 1.0;
  auto depth_vals = torch::arange(D, opt) / denom;          // (D,)  in [0,1]

  // Stations: broadcast each depth's scalar over all (s,t).
  attr_out.narrow(0, 0, dm.stations())
          .copy_(depth_vals.view({D, 1, 1}).expand({D, S, T}).reshape({dm.stations()}));

  // Trails: lerp between adjacent depth scalars (independent of s,t), then
  // broadcast over (s,t). Shape (D-1,K) -> (D-1,K,S,T).
  auto a = depth_vals.narrow(0, 0, D - 1);                  // (D-1,)
  auto b = depth_vals.narrow(0, 1, D - 1);                  // (D-1,)
  auto fracs = detail::lerp_fracs(K, attr_out.device());    // (K,)
  auto tr = a.view({D - 1, 1}) + fracs.view({1, K}) * (b - a).view({D - 1, 1}); // (D-1,K)
  attr_out.narrow(0, dm.stations(), dm.trails())
          .copy_(tr.view({D - 1, K, 1, 1}).expand({D - 1, K, S, T}).reshape({dm.trails()}));
}

// ---------------------------------------------------------------------------
// Attr: per-token (loss) mode. per_token is (S,T) f32. Broadcast each token's
// scalar to its ENTIRE thread — all D stations plus all K*(D-1) trail points.
// Rows [0, n_probe).
// ---------------------------------------------------------------------------
inline void write_attr_per_token(torch::Tensor& attr_out,
                                const torch::Tensor& per_token, const Dims& dm) {
  torch::NoGradGuard ng;
  const int64_t D = dm.D, K = dm.K, S = dm.S, T = dm.T;
  auto pt = per_token.to(torch::kFloat32);                  // (S,T)

  // Stations: same token value at every depth.
  attr_out.narrow(0, 0, dm.stations())
          .copy_(pt.view({1, S, T}).expand({D, S, T}).reshape({dm.stations()}));

  // Trails: same token value at every (seg,k).
  attr_out.narrow(0, dm.stations(), dm.trails())
          .copy_(pt.view({1, 1, S, T}).expand({D - 1, K, S, T}).reshape({dm.trails()}));
}

// ---------------------------------------------------------------------------
// Attr: per-station (confidence) mode. per_station is (D,S,T) f32. Stations
// take their own value; trails lerp between adjacent stations with the SAME
// fractions as positions. Rows [0, n_probe).
// ---------------------------------------------------------------------------
inline void write_attr_per_station(torch::Tensor& attr_out,
                                   const torch::Tensor& per_station,
                                   const Dims& dm) {
  torch::NoGradGuard ng;
  const int64_t D = dm.D, K = dm.K;
  auto ps = per_station.to(torch::kFloat32);                // (D,S,T)

  // Stations: flatten (D,S,T) row-major.
  attr_out.narrow(0, 0, dm.stations())
          .copy_(ps.reshape({dm.stations()}));

  // Trails: lerp adjacent station values -> (D-1,K,S,T) -> flatten.
  auto a = ps.narrow(0, 0, D - 1);                          // (D-1,S,T)
  auto b = ps.narrow(0, 1, D - 1);                          // (D-1,S,T)
  auto fracs = detail::lerp_fracs(K, attr_out.device());    // (K,)
  auto tr = a.unsqueeze(1) + fracs.view({1, K, 1, 1}) * (b - a).unsqueeze(1); // (D-1,K,S,T)
  attr_out.narrow(0, dm.stations(), dm.trails())
          .copy_(tr.reshape({dm.trails()}));
}

// ---------------------------------------------------------------------------
// Pin the generation region attr_out[n_probe, n_max) to `value` (white-hot).
// ---------------------------------------------------------------------------
inline void write_gen_attr(torch::Tensor& attr_out, const Dims& dm, float value) {
  torch::NoGradGuard ng;
  attr_out.narrow(0, dm.n_probe(), dm.n_gen()).fill_(value);
}

}}  // namespace gptscope::ts
