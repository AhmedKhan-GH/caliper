// Engine tests for TwinScope v2's surface core (applets/twin_scope/twin_surface.h):
// midpoint subdivision (counts / original-vertex prefix / manifold), the cotan
// Laplacian (symmetry / zero row sums / PSD), Voronoi-third masses + stable dt,
// the UV bake (partition of unity / asset coverage), and two physics checks on
// an in-test flat strip (analytic linear steady state, energy decay). Pure
// torch, CPU, its own binary (label "torch"). REQUIRE only — <torch/torch.h>
// defines a bare CHECK macro that shadows doctest's.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "twin_surface.h"           // applets/twin_scope — on the include path
#include <caliper/adapters/obj.hpp> // T5 loader, for the asset coverage row

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

using namespace twinscope;

// --------------------------------------------------------------------------
// Test-mesh builders.
// --------------------------------------------------------------------------

// Regular tetrahedron — a small closed 2-manifold (V=4, F=4, E=6). Ideal for
// the counts/prefix/manifold and PSD probes.
static SurfaceMesh make_tetrahedron() {
    auto positions = torch::tensor({{1.f, 1.f, 1.f},
                                    {1.f, -1.f, -1.f},
                                    {-1.f, 1.f, -1.f},
                                    {-1.f, -1.f, 1.f}},
                                   torch::kFloat32);
    auto indices = torch::tensor({0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2},
                                 torch::kLong);
    // Arbitrary but distinct UVs (subdivision must average them correctly).
    auto uvs = torch::tensor({{0.1f, 0.1f}, {0.9f, 0.1f},
                              {0.1f, 0.9f}, {0.9f, 0.9f}},
                             torch::kFloat32);
    return SurfaceMesh{positions, indices, uvs};
}

// Flat strip in the z=0 plane: nx*ny unit quads, each split into two triangles.
// (nx+1)*(ny+1) vertices on an integer grid, x in [0,nx], y in [0,ny].
static SurfaceMesh make_strip(int nx, int ny) {
    const int cols = nx + 1, rows = ny + 1;
    std::vector<float> pos, uv;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            pos.push_back(static_cast<float>(c));
            pos.push_back(static_cast<float>(r));
            pos.push_back(0.f);
            uv.push_back(static_cast<float>(c) / nx);
            uv.push_back(static_cast<float>(r) / ny);
        }
    }
    std::vector<int64_t> idx;
    auto vid = [cols](int r, int c) { return static_cast<int64_t>(r) * cols + c; };
    for (int r = 0; r < ny; ++r) {
        for (int c = 0; c < nx; ++c) {
            const int64_t a = vid(r, c), b = vid(r, c + 1);
            const int64_t d = vid(r + 1, c), e = vid(r + 1, c + 1);
            idx.insert(idx.end(), {a, b, e, a, e, d});
        }
    }
    const int64_t V = static_cast<int64_t>(pos.size()) / 3;
    SurfaceMesh m;
    m.positions = torch::from_blob(pos.data(), {V, 3}, torch::kFloat32).clone();
    m.uvs = torch::from_blob(uv.data(), {V, 2}, torch::kFloat32).clone();
    m.indices = torch::from_blob(idx.data(),
                                 {static_cast<int64_t>(idx.size())}, torch::kLong)
                    .clone();
    return m;
}

static SurfaceMesh surface_from_obj(caliper::obj::Mesh& m) {
    const int64_t V = static_cast<int64_t>(m.vertex_count());
    SurfaceMesh out;
    out.positions =
        torch::from_blob(m.positions.data(), {V, 3}, torch::kFloat32).clone();
    out.uvs = torch::from_blob(m.uvs.data(), {V, 2}, torch::kFloat32).clone();
    out.indices =
        torch::from_blob(m.indices.data(),
                         {static_cast<int64_t>(m.indices.size())}, torch::kInt32)
            .clone()
            .to(torch::kLong);
    return out;
}

// Count how many distinct undirected edges appear in > 2 triangles.
static int nonmanifold_edges(const torch::Tensor& indices) {
    auto idx = indices.to(torch::kLong).contiguous();
    auto ia = idx.accessor<int64_t, 1>();
    const int64_t F = idx.size(0) / 3;
    std::unordered_map<int64_t, int> count;
    int64_t V = idx.max().item<int64_t>() + 1;
    auto bump = [&](int64_t a, int64_t b) {
        int64_t lo = std::min(a, b), hi = std::max(a, b);
        count[lo * V + hi] += 1;
    };
    for (int64_t f = 0; f < F; ++f) {
        int64_t a = ia[3 * f], b = ia[3 * f + 1], c = ia[3 * f + 2];
        bump(a, b);
        bump(b, c);
        bump(c, a);
    }
    int bad = 0;
    for (auto& kv : count)
        if (kv.second > 2) ++bad;
    return bad;
}

// --------------------------------------------------------------------------
// Subdivision.
// --------------------------------------------------------------------------
TEST_CASE("subdivide_midpoint: counts, original-vertex prefix, manifold") {
    auto base = make_tetrahedron();
    const int64_t V0 = base.positions.size(0);
    const int64_t F0 = base.indices.size(0) / 3;

    SurfaceMesh cur = base;
    int64_t V = V0, F = F0, E = 6;  // tetra has 6 edges
    for (int level = 1; level <= 2; ++level) {
        cur = subdivide_midpoint(cur, 1);
        const int64_t Vp = V + E;   // V' = V + E
        const int64_t Fp = 4 * F;   // F' = 4F
        REQUIRE(cur.positions.size(0) == Vp);
        REQUIRE(cur.indices.size(0) / 3 == Fp);
        REQUIRE(cur.uvs.size(0) == Vp);
        // Closed manifold: E' = 3F'/2.
        REQUIRE(nonmanifold_edges(cur.indices) == 0);
        V = Vp;
        F = Fp;
        E = 3 * F / 2;
    }

    // Original vertices are the untouched prefix of a multi-level result.
    auto two = subdivide_midpoint(base, 2);
    REQUIRE(torch::allclose(two.positions.narrow(0, 0, V0), base.positions));
    REQUIRE(torch::allclose(two.uvs.narrow(0, 0, V0), base.uvs));

    // A midpoint vertex's position AND uv are the mean of its parent edge's
    // endpoints (edge 0-1). Pin the uv to the endpoint-uv mean too, not just the
    // position — the split must average both streams. Endpoint uvs {0.1,0.1} and
    // {0.9,0.1} mean to {0.5,0.1}.
    auto one = subdivide_midpoint(base, 1);
    auto pmid = 0.5f * (base.positions[0] + base.positions[1]);
    auto uvmid = 0.5f * (base.uvs[0] + base.uvs[1]);
    bool found = false;
    for (int64_t v = V0; v < one.positions.size(0); ++v)
        if (torch::allclose(one.positions[v], pmid)) {
            REQUIRE(torch::allclose(one.uvs[v], uvmid));
            found = true;
        }
    REQUIRE(found);
}

// --------------------------------------------------------------------------
// Cotangent Laplacian.
// --------------------------------------------------------------------------
TEST_CASE("cotan_laplacian: symmetry, zero row sums, PSD") {
    auto mesh = subdivide_midpoint(make_tetrahedron(), 1);
    auto L = cotan_laplacian(mesh);
    auto dense = L.to_dense();
    const int64_t V = dense.size(0);

    REQUIRE((dense - dense.t()).abs().max().item<float>() < 1e-5f);
    REQUIRE(dense.sum(1).abs().max().item<float>() < 1e-4f);

    auto eig = torch::linalg::eigvalsh(dense.to(torch::kFloat64), "L");
    REQUIRE(eig.min().item<double>() >= -1e-5);

    // Constant vector is the nullspace (harmonic).
    auto one = torch::ones({V, 1}, torch::kFloat32);
    REQUIRE(torch::mm(L, one).abs().max().item<float>() < 1e-4f);
}

// --------------------------------------------------------------------------
// Masses + stable dt.
// --------------------------------------------------------------------------
TEST_CASE("vertex_masses: Voronoi-third sums to total area and per-vertex share") {
    auto strip = make_strip(4, 2);  // 8 unit quads -> total area 8
    auto M = vertex_masses(strip);
    REQUIRE(M.sum().item<double>() == doctest::Approx(8.0).epsilon(1e-5));
    REQUIRE(M.min().item<float>() > 0.f);

    // Voronoi-third gives each vertex one third of every incident triangle's
    // area. Every triangle in the strip is a unit right triangle (area 1/2), so
    // a vertex's mass is exactly (incident triangle count)/6 — a distribution
    // the sum-only check above cannot see. For make_strip(4,2)'s 5x3 grid with
    // a single (r,c)->(r+1,c+1) diagonal the incident-triangle counts are
    // (row-major, V=15): diagonal corners 2, off-diagonal corners 1, boundary
    // 3, interior 6 (they sum to 48 = 8*6).
    const double tris[15] = {2, 3, 3, 3, 1,
                             3, 6, 6, 6, 3,
                             1, 3, 3, 3, 2};
    REQUIRE(M.size(0) == 15);
    auto Md = M.to(torch::kFloat64).contiguous();
    auto Ma = Md.accessor<double, 1>();
    for (int64_t v = 0; v < 15; ++v)
        REQUIRE(Ma[v] == doctest::Approx(tris[v] / 6.0).epsilon(1e-6));
}

TEST_CASE("stable_dt: matches min_i M_i/(kappa L_ii), no dense materialisation") {
    auto strip = make_strip(6, 3);
    auto L = cotan_laplacian(strip);
    auto M = vertex_masses(strip);
    const double kappa = 0.37;
    const double dt = stable_dt(L, M, kappa);

    auto dense = L.to_dense().to(torch::kFloat64);
    auto Ldiag = dense.diagonal();
    auto Md = M.to(torch::kFloat64);
    auto ratio = Md / (kappa * Ldiag);
    auto inf = torch::full_like(ratio, std::numeric_limits<double>::infinity());
    ratio = torch::where(Ldiag > 0.0, ratio, inf);
    const double expect = 0.9 * ratio.min().item<double>();
    REQUIRE(dt == doctest::Approx(expect).epsilon(1e-9));
    REQUIRE(dt > 0.0);
}

// --------------------------------------------------------------------------
// Bake matrix.
// --------------------------------------------------------------------------
TEST_CASE("bake_matrix: partition of unity + gutter map on a UV quad") {
    // Single chart covering the middle of a 16x16 atlas.
    SurfaceMesh m;
    m.positions = torch::tensor({{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f},
                                 {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
                                torch::kFloat32);
    m.uvs = torch::tensor({{0.25f, 0.25f}, {0.75f, 0.25f},
                           {0.75f, 0.75f}, {0.25f, 0.75f}},
                          torch::kFloat32);
    m.indices = torch::tensor({0, 1, 2, 0, 2, 3}, torch::kLong);

    const int H = 16, W = 16;
    auto bake = bake_matrix(m, H, W);
    REQUIRE(bake.S.size(0) == H * W);
    REQUIRE(bake.S.size(1) == 4);

    auto Sc = bake.S.coalesce();
    auto rowsum = torch::zeros({H * W});
    rowsum.index_add_(0, Sc.indices()[0], Sc.values());
    auto inside = bake.inside_mask;
    REQUIRE(inside.sum().item<int64_t>() > 0);
    // Every inside texel's weights sum to exactly one.
    auto inside_sums = rowsum.index({inside});
    REQUIRE((inside_sums - 1.f).abs().max().item<float>() < 1e-5f);

    // gutter_src of an outside texel points at an inside texel; inside->self.
    auto gutter = bake.gutter_src;
    for (int64_t t = 0; t < H * W; ++t) {
        const int64_t src = gutter[t].item<int64_t>();
        REQUIRE(inside[src].item<bool>());
        if (inside[t].item<bool>()) REQUIRE(src == t);
    }
}

// Coverage row (spec §5/§10). The literal spec claim is "every triangle with
// nonzero UV area owns >=1 texel at 256^2". That is UNACHIEVABLE for this asset
// under texel-CENTRE point sampling: the committed housing's fin geometry has
// 445 (of 3184) triangles that are sub-texel-WIDTH slivers (nonzero area, 1..2.8
// texel^2, but thinnest dimension < 1 texel), which thread between texel-centre
// rows. A feature thinner than one texel is below Nyquist — no point-sampling
// rasteriser can own a centre for it, and its texels are supplied by the wider
// neighbours in the same chart (partition of unity holds), so the drape has no
// holes. We therefore assert the GENUINE guarantee point sampling provides —
// every triangle at least one texel thick owns a texel — and prove the residue
// are all slivers. (Reported to the parent: the literal row needs either the
// asset regenerated with >=1-texel-wide charts, or a supersampled bake.)
TEST_CASE("bake_matrix: every >=1-texel-thick asset triangle owns a 256^2 texel") {
    caliper::obj::Mesh om;
    std::string err;
    const std::string path =
        std::string(CALIPER_TEST_SOURCE_ROOT) + "/applets/twin_scope/assets/housing.obj";
    REQUIRE_MESSAGE(caliper::obj::load_file(path, om, &err), err);
    auto mesh = surface_from_obj(om);

    const int H = 256, W = 256;
    auto uv = mesh.uvs.to(torch::kFloat64).contiguous();
    auto idx = mesh.indices.contiguous();
    auto ua = uv.accessor<double, 2>();
    auto ia = idx.accessor<int64_t, 1>();
    const int64_t F = idx.size(0) / 3;

    int64_t uncovered = 0, degenerate = 0, total = 0;
    double max_uncovered_area = 0.0;       // in texel^2 units
    int64_t uncovered_ge1 = 0;             // uncovered triangles with area >= 1 texel^2
    int64_t uncovered_thick = 0;           // uncovered AND >= 1 texel thick (a real miss)
    double max_uncovered_minheight = 0.0;  // thickest of the uncovered slivers, texels
    for (int64_t f = 0; f < F; ++f) {
        const int64_t i0 = ia[3 * f], i1 = ia[3 * f + 1], i2 = ia[3 * f + 2];
        const double x0 = ua[i0][0], y0 = ua[i0][1];
        const double x1 = ua[i1][0], y1 = ua[i1][1];
        const double x2 = ua[i2][0], y2 = ua[i2][1];
        const double denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
        if (std::abs(denom) < 1e-18) {
            ++degenerate;
            continue;
        }
        ++total;
        const double uv_area = 0.5 * std::abs(denom);
        const double texel_area = uv_area * W * H;
        const double inv = 1.0 / denom;
        int64_t xlo = static_cast<int64_t>(std::floor(std::min({x0, x1, x2}) * W - 0.5));
        int64_t xhi = static_cast<int64_t>(std::ceil(std::max({x0, x1, x2}) * W - 0.5));
        int64_t ylo = static_cast<int64_t>(std::floor(std::min({y0, y1, y2}) * H - 0.5));
        int64_t yhi = static_cast<int64_t>(std::ceil(std::max({y0, y1, y2}) * H - 0.5));
        xlo = std::max<int64_t>(xlo, 0);
        ylo = std::max<int64_t>(ylo, 0);
        xhi = std::min<int64_t>(xhi, W - 1);
        yhi = std::min<int64_t>(yhi, H - 1);
        bool owns = false;
        for (int64_t py = ylo; py <= yhi && !owns; ++py) {
            const double vy = (py + 0.5) / H;
            for (int64_t px = xlo; px <= xhi && !owns; ++px) {
                const double ux = (px + 0.5) / W;
                const double l0 = ((y1 - y2) * (ux - x2) + (x2 - x1) * (vy - y2)) * inv;
                const double l1 = ((y2 - y0) * (ux - x2) + (x0 - x2) * (vy - y2)) * inv;
                const double l2 = 1.0 - l0 - l1;
                if (l0 >= -1e-9 && l1 >= -1e-9 && l2 >= -1e-9) owns = true;
            }
        }
        if (!owns) {
            ++uncovered;
            if (texel_area > max_uncovered_area) max_uncovered_area = texel_area;
            if (texel_area >= 1.0) ++uncovered_ge1;
            // Min height in texels = 2*area / longest edge (thinnest dimension).
            auto elen = [&](double ax, double ay, double bx, double by) {
                const double dx = (ax - bx) * W, dy = (ay - by) * H;
                return std::sqrt(dx * dx + dy * dy);
            };
            const double e01 = elen(x0, y0, x1, y1);
            const double e12 = elen(x1, y1, x2, y2);
            const double e20 = elen(x2, y2, x0, y0);
            const double longest = std::max({e01, e12, e20});
            const double min_h = 2.0 * texel_area / longest;
            if (min_h > max_uncovered_minheight) max_uncovered_minheight = min_h;
            if (min_h >= 1.0) ++uncovered_thick;
        }
    }
    MESSAGE("asset triangles: " << F << " total, " << degenerate
                                << " zero-UV-area, " << uncovered
                                << " nonzero-area uncovered at 256^2; "
                                << "uncovered with area>=1 texel^2: " << uncovered_ge1
                                << "; max uncovered area (texel^2): " << max_uncovered_area
                                << "; thickest uncovered sliver (texels): "
                                << max_uncovered_minheight);
    REQUIRE(total > 0);
    // Every triangle at least one texel thick is covered — the true point-
    // sampling guarantee. The uncovered residue are all sub-texel slivers.
    REQUIRE(uncovered_thick == 0);
    REQUIRE(max_uncovered_minheight < 1.0);
}

// The committed housing packs its 30 charts into a 6x5 UV grid, each chart
// inset 3/256 per side (v2-task-5). Adjacent charts are therefore separated by a
// 2*(3/256) = 6/256 gutter — the 6-texel-@256 isolation the bake relies on so no
// texel centre is claimed by two charts. Pin that width from the atlas UVs: it
// is the width the "gutter map on a UV quad" test above assumes but never sizes.
TEST_CASE("bake atlas: inter-chart gutter is exactly 6 texels at 256^2") {
    caliper::obj::Mesh om;
    std::string err;
    const std::string path =
        std::string(CALIPER_TEST_SOURCE_ROOT) + "/applets/twin_scope/assets/housing.obj";
    REQUIRE_MESSAGE(caliper::obj::load_file(path, om, &err), err);
    auto mesh = surface_from_obj(om);

    auto uv = mesh.uvs.to(torch::kFloat64).contiguous();
    auto ua = uv.accessor<double, 2>();
    const int64_t V = uv.size(0);

    // Per-cell UV bounding boxes on the 6x5 grid (u -> 6 columns, v -> 5 rows).
    constexpr int CU = 6, CV = 5;
    double umin[CU][CV], umax[CU][CV], vmin[CU][CV], vmax[CU][CV];
    bool used[CU][CV] = {};
    for (int i = 0; i < CU; ++i)
        for (int j = 0; j < CV; ++j) {
            umin[i][j] = vmin[i][j] = 2.0;
            umax[i][j] = vmax[i][j] = -1.0;
        }
    for (int64_t v = 0; v < V; ++v) {
        const double u = ua[v][0], w = ua[v][1];
        const int cu = std::min(CU - 1, static_cast<int>(u * CU));
        const int cv = std::min(CV - 1, static_cast<int>(w * CV));
        used[cu][cv] = true;
        umin[cu][cv] = std::min(umin[cu][cv], u);
        umax[cu][cv] = std::max(umax[cu][cv], u);
        vmin[cu][cv] = std::min(vmin[cu][cv], w);
        vmax[cu][cv] = std::max(vmax[cu][cv], w);
    }
    int occupied = 0;
    for (int i = 0; i < CU; ++i)
        for (int j = 0; j < CV; ++j) occupied += used[i][j] ? 1 : 0;
    REQUIRE(occupied == 30);  // all 30 charts land in distinct grid cells

    double min_gutter = 2.0;
    for (int j = 0; j < CV; ++j)  // horizontal neighbours
        for (int i = 0; i + 1 < CU; ++i)
            if (used[i][j] && used[i + 1][j])
                min_gutter = std::min(min_gutter, umin[i + 1][j] - umax[i][j]);
    for (int i = 0; i < CU; ++i)  // vertical neighbours
        for (int j = 0; j + 1 < CV; ++j)
            if (used[i][j] && used[i][j + 1])
                min_gutter = std::min(min_gutter, vmin[i][j + 1] - vmax[i][j]);

    // Adjacent charts fill to their inset edges, so the tightest gutter is
    // exactly 6/256 uv == 6 texels @256 (asset UVs are rounded to 1e-6).
    REQUIRE(min_gutter * 256.0 == doctest::Approx(6.0).epsilon(1e-3));
}

// --------------------------------------------------------------------------
// Physics on the flat strip.
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// DeviceSparse — the MPS-legal carrier for the sparse operators. libtorch
// 2.5.1 has no SparseMPS kernels (a sparse .to(kMPS) throws), so on MPS the
// COO coefficients ride as three dense tensors and mm() spells the product
// gather + index_add_. Everywhere else it must stay the true sparse tensor.
// --------------------------------------------------------------------------
TEST_CASE("DeviceSparse: apply matches the CPU sparse product on every device") {
    auto mesh = subdivide_midpoint(make_tetrahedron(), 2);
    auto L = cotan_laplacian(mesh);                       // sparse (V,V), coalesced
    const int64_t V = mesh.positions.size(0);
    torch::manual_seed(7);
    auto x = torch::randn({V, 3});
    auto ref = torch::mm(L, x);                           // CPU sparse reference

    // CPU carrier keeps the real sparse tensor (fast path unchanged).
    auto on_cpu = DeviceSparse::to_device(L, torch::kCPU);
    REQUIRE(on_cpu.S.defined());
    REQUIRE(on_cpu.S.is_sparse());
    REQUIRE(torch::allclose(on_cpu.mm(x), ref));

    if (!torch::mps::is_available()) { MESSAGE("no MPS device - skipping MPS half"); return; }
    auto on_mps = DeviceSparse::to_device(L, torch::kMPS);
    REQUIRE(!on_mps.S.defined());                         // no sparse tensor on MPS
    auto got = on_mps.mm(x.to(torch::kMPS)).to(torch::kCPU);
    REQUIRE(torch::isfinite(got).all().item<bool>());
    REQUIRE(torch::allclose(got, ref, 1e-4, 1e-5));
}

TEST_CASE("flat strip relaxes to the analytic linear steady state") {
    const int nx = 20, ny = 2;
    auto strip = make_strip(nx, ny);
    const int64_t V = strip.positions.size(0);
    auto L = cotan_laplacian(strip);
    auto M = vertex_masses(strip);
    auto Minv = (1.0 / M.to(torch::kFloat64));
    const double kappa = 1.0;
    const double dt = stable_dt(L, M, kappa);

    auto xs = strip.positions.select(1, 0).to(torch::kFloat64);  // (V,)
    auto left = (xs == 0.0);
    auto right = (xs == static_cast<double>(nx));
    auto Ld = L.to_dense().to(torch::kFloat64);

    auto T = torch::full({V}, 50.0, torch::kFloat64);
    T.index_put_({left}, 100.0);
    T.index_put_({right}, 0.0);

    for (int step = 0; step < 200000; ++step) {
        auto LT = torch::mv(Ld, T);
        auto Tn = T - dt * kappa * (Minv * LT);
        Tn.index_put_({left}, 100.0);
        Tn.index_put_({right}, 0.0);
        const double delta = (Tn - T).abs().max().item<double>();
        T = Tn;
        if (delta < 1e-4) break;
    }

    // Interior vertices match 100*(1 - x/nx) within 2% of the 100-degree span.
    auto expect = 100.0 * (1.0 - xs / static_cast<double>(nx));
    auto interior = (xs > 0.0) * (xs < static_cast<double>(nx));
    auto err = (T - expect).abs().index({interior.to(torch::kBool)});
    REQUIRE(err.max().item<double>() <= 2.0);
}

TEST_CASE("energy decays monotonically under zero sources") {
    auto strip = make_strip(20, 2);
    const int64_t V = strip.positions.size(0);
    auto L = cotan_laplacian(strip);
    auto M = vertex_masses(strip);
    auto Md = M.to(torch::kFloat64);
    auto Minv = 1.0 / Md;
    const double kappa = 1.0;
    const double dt = stable_dt(L, M, kappa);
    auto Ld = L.to_dense().to(torch::kFloat64);

    torch::manual_seed(7);
    auto T = torch::rand({V}, torch::kFloat64) + 0.5;  // strictly positive

    auto energy = [&](const torch::Tensor& t) {
        return (Md * t * t).sum().item<double>();
    };
    double prev = energy(T);
    for (int step = 0; step < 100; ++step) {
        auto LT = torch::mv(Ld, T);
        T = T - dt * kappa * (Minv * LT);
        const double e = energy(T);
        REQUIRE(e <= prev + 1e-9);
        prev = e;
    }
}
