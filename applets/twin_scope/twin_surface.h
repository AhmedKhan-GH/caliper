#pragma once

// TwinScope v2 surface engine — the pure-torch/CPU precompute behind the
// surface-aware thermal twin (design doc §2 sim mesh, §3 operator/step/dt,
// §5 bake). Header-only, depends only on <torch/torch.h> + the C++ standard
// library so it can be unit-tested in isolation and consumed by the thermal
// model (T7) and applet (T8) without pulling in host/renderer code.
//
// Everything here builds on CPU in f32 storage; angle-sensitive geometry is
// accumulated in double and cast down. Nothing here requires a GPU — device
// moves (state.to(device), L.to_sparse_csr()) are the caller's business.
//
// Conventions pinned for downstream tasks (do not change without updating
// T7/T8):
//   * SurfaceMesh::indices is a FLAT (F*3,) int64 tensor: entries [3f, 3f+1,
//     3f+2] are the three corner vertex ids of triangle f.
//   * cotan_laplacian returns a COO sparse (V,V) f32 tensor, coalesced. It is
//     the standard positive-semidefinite operator L = D - W (rows sum to zero,
//     off-diagonals <= 0). T7 diffuses with T <- T + dt*Minv*(-kappa*L*T) and
//     may re-materialise it as sparse_csr on device for L @ T^T products.

#include <torch/torch.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

namespace twinscope {

// Sim/render mesh in torch form. positions (V,3) f32, uvs (V,2) f32, indices
// flat (F*3,) int64. Kept deliberately minimal — normals are re-derived where
// needed, they are not load-bearing for the surface operators.
struct SurfaceMesh {
    torch::Tensor positions;  // (V,3) f32
    torch::Tensor indices;    // (F*3,) int64 — 3 corner ids per triangle
    torch::Tensor uvs;        // (V,2) f32
};

// ---------------------------------------------------------------------------
// Midpoint (1-to-4) subdivision.
//
// Each level splits every triangle (a,b,c) into four by inserting the three
// edge midpoints (mab, mbc, mca):
//     (a, mab, mca) (b, mbc, mab) (c, mca, mbc) (mab, mbc, mca)
// Midpoints are deduplicated through an edge map keyed on the unordered
// endpoint id pair, so a shared interior edge yields ONE midpoint and the
// mesh stays a manifold (V' = V + E, F' = 4F). The original vertices keep
// their ids as the first V entries of the output — the design's per-vertex
// fallback reads state[:, :V_render] straight off the subdivided state.
//
// Midpoint positions and UVs are the arithmetic mean of their two endpoints.
// The dedup key is the VERTEX id pair, and the OBJ loader splits UV seams into
// distinct vertex ids (one wedge per chart), so a midpoint that lies on a seam
// is created independently for each wedge and simply inherits that wedge's UV
// average — seams never bleed a midpoint UV across charts.
// ---------------------------------------------------------------------------
namespace detail {

inline SurfaceMesh subdivide_once(const SurfaceMesh& in) {
    auto pos = in.positions.to(torch::kCPU, torch::kFloat32).contiguous();
    auto uv = in.uvs.to(torch::kCPU, torch::kFloat32).contiguous();
    auto idx = in.indices.to(torch::kCPU, torch::kLong).contiguous();
    const int64_t V = pos.size(0);
    const int64_t F = idx.size(0) / 3;

    auto pa = pos.accessor<float, 2>();
    auto ua = uv.accessor<float, 2>();
    auto ia = idx.accessor<int64_t, 1>();

    std::vector<std::array<float, 3>> out_pos;
    std::vector<std::array<float, 2>> out_uv;
    out_pos.reserve(static_cast<size_t>(V) * 2);
    out_uv.reserve(static_cast<size_t>(V) * 2);
    for (int64_t v = 0; v < V; ++v) {
        out_pos.push_back({pa[v][0], pa[v][1], pa[v][2]});
        out_uv.push_back({ua[v][0], ua[v][1]});
    }

    std::unordered_map<int64_t, int64_t> edge_mid;
    auto midpoint = [&](int64_t a, int64_t b) -> int64_t {
        const int64_t lo = a < b ? a : b;
        const int64_t hi = a < b ? b : a;
        const int64_t key = lo * V + hi;  // a,b are original ids in [0,V)
        auto it = edge_mid.find(key);
        if (it != edge_mid.end()) return it->second;
        const int64_t m = static_cast<int64_t>(out_pos.size());
        out_pos.push_back({0.5f * (pa[a][0] + pa[b][0]),
                           0.5f * (pa[a][1] + pa[b][1]),
                           0.5f * (pa[a][2] + pa[b][2])});
        out_uv.push_back({0.5f * (ua[a][0] + ua[b][0]),
                          0.5f * (ua[a][1] + ua[b][1])});
        edge_mid.emplace(key, m);
        return m;
    };

    std::vector<int64_t> out_idx;
    out_idx.reserve(static_cast<size_t>(F) * 12);
    for (int64_t f = 0; f < F; ++f) {
        const int64_t a = ia[3 * f], b = ia[3 * f + 1], c = ia[3 * f + 2];
        const int64_t mab = midpoint(a, b);
        const int64_t mbc = midpoint(b, c);
        const int64_t mca = midpoint(c, a);
        const int64_t tri[12] = {a, mab, mca, b, mbc, mab,
                                 c, mca, mbc, mab, mbc, mca};
        for (int64_t t : tri) out_idx.push_back(t);
    }

    const int64_t Vp = static_cast<int64_t>(out_pos.size());
    SurfaceMesh out;
    out.positions = torch::empty({Vp, 3}, torch::kFloat32);
    out.uvs = torch::empty({Vp, 2}, torch::kFloat32);
    auto opa = out.positions.accessor<float, 2>();
    auto oua = out.uvs.accessor<float, 2>();
    for (int64_t v = 0; v < Vp; ++v) {
        opa[v][0] = out_pos[v][0];
        opa[v][1] = out_pos[v][1];
        opa[v][2] = out_pos[v][2];
        oua[v][0] = out_uv[v][0];
        oua[v][1] = out_uv[v][1];
    }
    out.indices = torch::from_blob(
                      out_idx.data(),
                      {static_cast<int64_t>(out_idx.size())}, torch::kLong)
                      .clone();
    return out;
}

}  // namespace detail

inline SurfaceMesh subdivide_midpoint(const SurfaceMesh& mesh, int levels) {
    TORCH_CHECK(levels >= 0, "subdivide levels must be >= 0");
    SurfaceMesh out = mesh;
    // Normalise storage even at level 0 so callers get the pinned dtypes.
    out.positions = out.positions.to(torch::kCPU, torch::kFloat32).contiguous();
    out.uvs = out.uvs.to(torch::kCPU, torch::kFloat32).contiguous();
    out.indices = out.indices.to(torch::kCPU, torch::kLong).contiguous();
    for (int l = 0; l < levels; ++l) out = detail::subdivide_once(out);
    return out;
}

// ---------------------------------------------------------------------------
// Cotangent Laplacian — sparse COO (V,V) f32, the standard PSD operator.
//
// Assembly: for triangle (i,j,k) the cotangent of the angle at each corner is
// added (halved) to the weight of the OPPOSITE edge:
//     cot(angle at i) -> w(j,k),  cot(at j) -> w(i,k),  cot(at k) -> w(i,j).
// An interior edge collects a contribution from each of its two triangles, so
// w(i,j) = (cot alpha + cot beta)/2 with alpha,beta the two opposite angles.
// The operator is L = D - W:  off-diagonal L[i,j] = -w(i,j) (<= 0), diagonal
// L[i,i] = sum_j w(i,j). Every row sums to zero (L @ 1 = 0) and, with the
// clamp below, L is symmetric positive-semidefinite.
//
// Cotangents are clamped to [0, 1e6]: obtuse triangles give a NEGATIVE cot,
// which would make some off-diagonal weights positive and could break the PSD
// guarantee on an imperfect (badly-shaped) mesh. Clamping the negatives to 0
// keeps every weight >= 0, hence L stays a genuine graph Laplacian (D - W with
// W >= 0 => PSD). The 1e6 ceiling tames near-degenerate slivers. Angles are
// accumulated in double, then cast to f32.
// ---------------------------------------------------------------------------
inline torch::Tensor cotan_laplacian(const SurfaceMesh& mesh) {
    auto pos = mesh.positions.to(torch::kCPU, torch::kFloat64).contiguous();
    auto idx = mesh.indices.to(torch::kCPU, torch::kLong).contiguous();
    const int64_t V = pos.size(0);
    const int64_t F = idx.size(0) / 3;
    auto pa = pos.accessor<double, 2>();
    auto ia = idx.accessor<int64_t, 1>();

    std::unordered_map<int64_t, double> edge_w;  // key lo*V+hi -> weight
    std::vector<double> diag(static_cast<size_t>(V), 0.0);

    auto add_edge = [&](int64_t a, int64_t b, double w) {
        const int64_t lo = a < b ? a : b;
        const int64_t hi = a < b ? b : a;
        edge_w[lo * V + hi] += w;
        diag[static_cast<size_t>(a)] += w;
        diag[static_cast<size_t>(b)] += w;
    };

    auto cot_at = [&](int64_t apex, int64_t p, int64_t q) -> double {
        // Cotangent of the angle at `apex` in triangle (apex,p,q).
        const double ux = pa[p][0] - pa[apex][0];
        const double uy = pa[p][1] - pa[apex][1];
        const double uz = pa[p][2] - pa[apex][2];
        const double vx = pa[q][0] - pa[apex][0];
        const double vy = pa[q][1] - pa[apex][1];
        const double vz = pa[q][2] - pa[apex][2];
        const double dot = ux * vx + uy * vy + uz * vz;
        const double cx = uy * vz - uz * vy;
        const double cy = uz * vx - ux * vz;
        const double cz = ux * vy - uy * vx;
        const double cross = std::sqrt(cx * cx + cy * cy + cz * cz);
        if (cross < 1e-12) return 0.0;  // degenerate triangle
        double cot = dot / cross;
        if (cot < 0.0) cot = 0.0;
        if (cot > 1e6) cot = 1e6;
        return cot;
    };

    for (int64_t f = 0; f < F; ++f) {
        const int64_t i = ia[3 * f], j = ia[3 * f + 1], k = ia[3 * f + 2];
        add_edge(j, k, 0.5 * cot_at(i, j, k));  // angle at i opposite edge (j,k)
        add_edge(i, k, 0.5 * cot_at(j, i, k));  // angle at j opposite edge (i,k)
        add_edge(i, j, 0.5 * cot_at(k, i, j));  // angle at k opposite edge (i,j)
    }

    const int64_t nnz = static_cast<int64_t>(edge_w.size()) * 2 + V;
    auto rows = torch::empty({nnz}, torch::kLong);
    auto cols = torch::empty({nnz}, torch::kLong);
    auto vals = torch::empty({nnz}, torch::kFloat32);
    auto ra = rows.accessor<int64_t, 1>();
    auto ca = cols.accessor<int64_t, 1>();
    auto va = vals.accessor<float, 1>();

    int64_t n = 0;
    for (const auto& kv : edge_w) {
        const int64_t lo = kv.first / V;
        const int64_t hi = kv.first % V;
        const float w = static_cast<float>(kv.second);
        ra[n] = lo; ca[n] = hi; va[n] = -w; ++n;
        ra[n] = hi; ca[n] = lo; va[n] = -w; ++n;
    }
    for (int64_t v = 0; v < V; ++v) {
        ra[n] = v; ca[n] = v; va[n] = static_cast<float>(diag[static_cast<size_t>(v)]);
        ++n;
    }

    auto indices = torch::stack({rows, cols}, 0);
    return torch::sparse_coo_tensor(indices, vals, {V, V}).coalesce();
}

// ---------------------------------------------------------------------------
// DeviceSparse — a device-legal carrier for the sparse operators (L, S).
// libtorch 2.5.1 has no SparseMPS kernels: a sparse .to(kMPS) throws
// NotImplementedError. On MPS the COO coefficients ride as three dense
// tensors and mm() spells the same product as gather + index_add_; every
// other device keeps the true sparse tensor and defers to torch::mm.
// ---------------------------------------------------------------------------
struct DeviceSparse {
    torch::Tensor S;             // the sparse tensor on device (non-MPS path)
    torch::Tensor row, col, val; // (nnz,) i64/i64/f32 COO components (MPS path)
    int64_t rows = 0;

    static DeviceSparse to_device(const torch::Tensor& coo_cpu,
                                  torch::Device device) {
        DeviceSparse d;
        d.rows = coo_cpu.size(0);
        if (device.is_mps()) {
            auto c = coo_cpu.coalesce();
            d.row = c.indices()[0].to(device);
            d.col = c.indices()[1].to(device);
            d.val = c.values().to(device);
        } else {
            d.S = coo_cpu.to(device);
        }
        return d;
    }

    // (rows, V) · (V, C) -> (rows, C), matching torch::mm(S, dense).
    torch::Tensor mm(const torch::Tensor& dense) const {
        if (val.defined()) {
            auto out = torch::zeros({rows, dense.size(1)}, dense.options());
            return out.index_add_(0, row,
                                  val.unsqueeze(1) * dense.index_select(0, col));
        }
        return torch::mm(S, dense);
    }
};

// ---------------------------------------------------------------------------
// Voronoi-third vertex masses — dense (V,) f32. Each triangle contributes a
// third of its area to each of its three vertices (the barycentric / Voronoi
// lumped-mass approximation): M[v] = (1/3) * sum of incident triangle areas.
// ---------------------------------------------------------------------------
inline torch::Tensor vertex_masses(const SurfaceMesh& mesh) {
    auto pos = mesh.positions.to(torch::kCPU, torch::kFloat64).contiguous();
    auto idx = mesh.indices.to(torch::kCPU, torch::kLong).contiguous();
    const int64_t V = pos.size(0);
    const int64_t F = idx.size(0) / 3;
    auto pa = pos.accessor<double, 2>();
    auto ia = idx.accessor<int64_t, 1>();

    std::vector<double> mass(static_cast<size_t>(V), 0.0);
    for (int64_t f = 0; f < F; ++f) {
        const int64_t i = ia[3 * f], j = ia[3 * f + 1], k = ia[3 * f + 2];
        const double ux = pa[j][0] - pa[i][0];
        const double uy = pa[j][1] - pa[i][1];
        const double uz = pa[j][2] - pa[i][2];
        const double vx = pa[k][0] - pa[i][0];
        const double vy = pa[k][1] - pa[i][1];
        const double vz = pa[k][2] - pa[i][2];
        const double cx = uy * vz - uz * vy;
        const double cy = uz * vx - ux * vz;
        const double cz = ux * vy - uy * vx;
        const double area = 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
        const double third = area / 3.0;
        mass[static_cast<size_t>(i)] += third;
        mass[static_cast<size_t>(j)] += third;
        mass[static_cast<size_t>(k)] += third;
    }

    auto out = torch::empty({V}, torch::kFloat32);
    auto oa = out.accessor<float, 1>();
    for (int64_t v = 0; v < V; ++v) oa[v] = static_cast<float>(mass[static_cast<size_t>(v)]);
    return out;
}

// ---------------------------------------------------------------------------
// Explicit-scheme stable time step.
//
// The explicit update is T_{n+1} = (I - dt*kappa*Minv*L) T_n. Stability needs
// the spectral radius of dt*kappa*Minv*L to be <= 2. By Gershgorin applied to
// Minv*L, every eigenvalue is bounded by
//     max_i (1/M_i) * (L_ii + sum_{j!=i} |L_ij|).
// L has zero row sums with non-positive off-diagonals, so sum_{j!=i}|L_ij| =
// L_ii, giving the bound max_i 2*L_ii/M_i. Hence
//     dt <= 2 / (kappa * max_i 2*L_ii/M_i) = min_i M_i / (kappa * L_ii),
// and we take 0.9x that for safety. Vertices with L_ii <= 0 (isolated, no
// incident area) impose no bound and are skipped. Diagonal is read straight
// off the sparse operator — no dense (V,V) materialisation.
// ---------------------------------------------------------------------------
inline double stable_dt(const torch::Tensor& L, const torch::Tensor& M,
                        double kappa) {
    TORCH_CHECK(kappa > 0.0, "kappa must be positive");
    auto Lc = L.coalesce();
    auto ind = Lc.indices();  // (2, nnz)
    auto val = Lc.values().to(torch::kFloat64);
    const int64_t V = L.size(0);

    auto diag_mask = ind[0].eq(ind[1]);
    auto diag_rows = ind[0].masked_select(diag_mask);
    auto diag_vals = val.masked_select(diag_mask);
    auto Ldiag = torch::zeros({V}, torch::kFloat64);
    Ldiag.index_put_({diag_rows}, diag_vals);

    auto Md = M.to(torch::kCPU, torch::kFloat64).contiguous();
    auto ratio = Md / (kappa * Ldiag);
    auto inf = torch::full({V}, std::numeric_limits<double>::infinity(),
                           torch::kFloat64);
    ratio = torch::where(Ldiag > 0.0, ratio, inf);
    const double min_ratio = ratio.min().item<double>();
    TORCH_CHECK(std::isfinite(min_ratio), "no vertex imposes a dt bound");
    return 0.9 * min_ratio;
}

// ---------------------------------------------------------------------------
// UV bake — state -> texture at (H,W).
//
// bake_matrix rasterises every triangle in UV space with an edge-function
// scan at texel centres ((x+0.5)/W, (y+0.5)/H). A texel whose centre lies
// inside a triangle receives that triangle's barycentric weights, written into
// row (y*W + x) of the sparse S (H*W, V) at the triangle's three vertex
// columns. Because l0+l1+l2 == 1 by construction, every inside texel row sums
// to exactly 1 (partition of unity). Charts are gutter-separated so a centre
// falls in at most one triangle in practice; where two triangles do claim the
// same centre (a shared edge), LAST WRITER WINS — harmless, both give a valid
// convex combination summing to 1.
//
// gutter_src is a nearest-inside map: a multi-source BFS (4-connectivity) from
// all inside texels labels every outside texel with the inside texel reached
// first (nearest by grid distance); inside texels map to themselves. Publish
// does index_select(tex_inside, gutter_src) so clamp/bilinear sampling in the
// gutter never reads uninitialised texels.
// ---------------------------------------------------------------------------
struct BakeResult {
    torch::Tensor S;           // sparse (H*W, V) f32
    torch::Tensor gutter_src;  // (H*W,) int64 — nearest inside texel per texel
    torch::Tensor inside_mask; // (H*W,) bool
};

inline BakeResult bake_matrix(const SurfaceMesh& mesh, int H, int W) {
    TORCH_CHECK(H > 0 && W > 0, "bake resolution must be positive");
    auto uv = mesh.uvs.to(torch::kCPU, torch::kFloat64).contiguous();
    auto idx = mesh.indices.to(torch::kCPU, torch::kLong).contiguous();
    const int64_t V = uv.size(0);
    const int64_t F = idx.size(0) / 3;
    const int64_t T = static_cast<int64_t>(H) * static_cast<int64_t>(W);
    auto ua = uv.accessor<double, 2>();
    auto ia = idx.accessor<int64_t, 1>();

    // Per-texel winning triangle's columns/weights (last writer wins).
    std::vector<std::array<int64_t, 3>> cols(static_cast<size_t>(T));
    std::vector<std::array<float, 3>> wts(static_cast<size_t>(T));
    std::vector<char> inside(static_cast<size_t>(T), 0);

    for (int64_t f = 0; f < F; ++f) {
        const int64_t i0 = ia[3 * f], i1 = ia[3 * f + 1], i2 = ia[3 * f + 2];
        const double x0 = ua[i0][0], y0 = ua[i0][1];
        const double x1 = ua[i1][0], y1 = ua[i1][1];
        const double x2 = ua[i2][0], y2 = ua[i2][1];
        const double denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
        if (std::abs(denom) < 1e-18) continue;  // zero UV-area triangle
        const double inv = 1.0 / denom;

        // Texel-index bounding box (centre model): centre (px+0.5)/W in [uvmin].
        double umin = std::min({x0, x1, x2});
        double umax = std::max({x0, x1, x2});
        double vmin = std::min({y0, y1, y2});
        double vmax = std::max({y0, y1, y2});
        int64_t xlo = static_cast<int64_t>(std::floor(umin * W - 0.5));
        int64_t xhi = static_cast<int64_t>(std::ceil(umax * W - 0.5));
        int64_t ylo = static_cast<int64_t>(std::floor(vmin * H - 0.5));
        int64_t yhi = static_cast<int64_t>(std::ceil(vmax * H - 0.5));
        if (xlo < 0) xlo = 0;
        if (ylo < 0) ylo = 0;
        if (xhi > W - 1) xhi = W - 1;
        if (yhi > H - 1) yhi = H - 1;

        for (int64_t py = ylo; py <= yhi; ++py) {
            const double vy = (static_cast<double>(py) + 0.5) / H;
            for (int64_t px = xlo; px <= xhi; ++px) {
                const double ux = (static_cast<double>(px) + 0.5) / W;
                const double l0 =
                    ((y1 - y2) * (ux - x2) + (x2 - x1) * (vy - y2)) * inv;
                const double l1 =
                    ((y2 - y0) * (ux - x2) + (x0 - x2) * (vy - y2)) * inv;
                const double l2 = 1.0 - l0 - l1;
                const double eps = 1e-9;
                if (l0 < -eps || l1 < -eps || l2 < -eps) continue;
                const int64_t t = py * W + px;
                cols[static_cast<size_t>(t)] = {i0, i1, i2};
                wts[static_cast<size_t>(t)] = {static_cast<float>(l0),
                                               static_cast<float>(l1),
                                               static_cast<float>(l2)};
                inside[static_cast<size_t>(t)] = 1;
            }
        }
    }

    // Build sparse S from inside texels (3 nnz per row).
    int64_t inside_count = 0;
    for (char c : inside) inside_count += c ? 1 : 0;
    const int64_t nnz = inside_count * 3;
    auto rows = torch::empty({nnz}, torch::kLong);
    auto scols = torch::empty({nnz}, torch::kLong);
    auto svals = torch::empty({nnz}, torch::kFloat32);
    auto ra = rows.accessor<int64_t, 1>();
    auto sca = scols.accessor<int64_t, 1>();
    auto sva = svals.accessor<float, 1>();
    int64_t n = 0;
    for (int64_t t = 0; t < T; ++t) {
        if (!inside[static_cast<size_t>(t)]) continue;
        for (int e = 0; e < 3; ++e) {
            ra[n] = t;
            sca[n] = cols[static_cast<size_t>(t)][e];
            sva[n] = wts[static_cast<size_t>(t)][e];
            ++n;
        }
    }
    auto sidx = torch::stack({rows, scols}, 0);
    auto S = torch::sparse_coo_tensor(sidx, svals, {T, V}).coalesce();

    // Nearest-inside multi-source BFS for the gutter map.
    auto gutter = torch::empty({T}, torch::kLong);
    auto ga = gutter.accessor<int64_t, 1>();
    std::vector<char> visited(static_cast<size_t>(T), 0);
    std::queue<int64_t> q;
    for (int64_t t = 0; t < T; ++t) {
        if (inside[static_cast<size_t>(t)]) {
            ga[t] = t;
            visited[static_cast<size_t>(t)] = 1;
            q.push(t);
        } else {
            ga[t] = t;  // default self; overwritten if reached
        }
    }
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};
    while (!q.empty()) {
        const int64_t t = q.front();
        q.pop();
        const int64_t cy = t / W, cx = t % W;
        const int64_t src = ga[t];
        for (int d = 0; d < 4; ++d) {
            const int64_t nx = cx + dx[d], ny = cy + dy[d];
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            const int64_t nt = ny * W + nx;
            if (visited[static_cast<size_t>(nt)]) continue;
            visited[static_cast<size_t>(nt)] = 1;
            ga[nt] = src;
            q.push(nt);
        }
    }

    auto mask = torch::empty({T}, torch::kBool);
    auto ma = mask.accessor<bool, 1>();
    for (int64_t t = 0; t < T; ++t) ma[t] = inside[static_cast<size_t>(t)] != 0;

    return BakeResult{S, gutter, mask};
}

}  // namespace twinscope
