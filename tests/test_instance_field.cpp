// Pure-math unit tests for InstanceScope's pose/tint/mesh reference (torch-free,
// rides the fast caliper_tests suite). Checks the three load-bearing properties:
// the grid is a centered near-square lattice, every pose is RIGID (RᵀR = I, so
// the §5.1 G14 gate never refuses), the tint is analytic in [-1,1], and the
// procedural gem has unit outward normals.
#include <doctest/doctest.h>
#include "instance_field.h"

#include <array>
#include <cmath>

using namespace instancescope;

namespace {

// RᵀR for the upper-3x3 of a column-major 4x4; returns max |RᵀR - I| entry.
float ortho_defect(const std::array<float, 16>& m) {
    // columns of R (basis vectors)
    const float col[3][3] = {
        {m[0], m[1], m[2]},   // R e_x
        {m[4], m[5], m[6]},   // R e_y
        {m[8], m[9], m[10]},  // R e_z
    };
    float worst = 0.0f;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            const float dot = col[a][0] * col[b][0] + col[a][1] * col[b][1] +
                              col[a][2] * col[b][2];
            const float target = (a == b) ? 1.0f : 0.0f;
            worst = std::max(worst, std::fabs(dot - target));
        }
    return worst;
}

float det3(const std::array<float, 16>& m) {
    // column-major upper-3x3 determinant
    const float a = m[0], b = m[4], c = m[8];
    const float d = m[1], e = m[5], f = m[9];
    const float g = m[2], h = m[6], i = m[10];
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

}  // namespace

TEST_CASE("grid_dims is a near-square lattice that holds n") {
    for (int n : {1, 2, 5, 999, 1000, 1001, 5000}) {
        const GridDims d = grid_dims(n);
        CHECK(d.cols >= 1);
        CHECK(d.rows >= 1);
        CHECK(d.cols * d.rows >= n);              // fits all instances
        CHECK((d.cols - 1) * d.rows < n);         // no wholly-empty last column set
        // near-square: cols within 1 of ceil(sqrt(n))
        const int s = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
        CHECK(std::abs(d.cols - s) <= 1);
    }
}

TEST_CASE("grid is centered on the origin with the requested spacing") {
    const int n = 1000;
    const float spacing = 2.2f;
    const GridDims d = grid_dims(n);
    // neighbor in +col is exactly `spacing` away in x
    const Vec3 a = grid_center(0, n, spacing);
    const Vec3 b = grid_center(1, n, spacing);
    CHECK(b.x - a.x == doctest::Approx(spacing));
    // neighbor in +row is `spacing` away in z
    const Vec3 c = grid_center(d.cols, n, spacing);
    CHECK(c.z - a.z == doctest::Approx(spacing));
    // centroid of the full grid sits on the origin (x,z)
    double sx = 0, sz = 0;
    for (int i = 0; i < d.cols * d.rows; ++i) {
        const Vec3 p = grid_center(i, n, spacing);
        sx += p.x; sz += p.z;
    }
    const int total = d.cols * d.rows;
    CHECK(sx / total == doctest::Approx(0.0).epsilon(0.001));
    CHECK(sz / total == doctest::Approx(0.0).epsilon(0.001));
}

TEST_CASE("pose_matrix is rigid: RᵀR = I and det R = +1 (G14 accepts)") {
    WaveParams w;
    const int n = 1000;
    const float spacing = 2.2f;
    for (int i : {0, 1, 37, 500, 999}) {
        for (float t : {0.0f, 0.13f, 1.7f, 42.0f}) {
            const auto m = pose_matrix(i, n, spacing, t, w);
            CHECK(ortho_defect(m) < 1e-5f);
            CHECK(det3(m) == doctest::Approx(1.0f).epsilon(1e-4));
            // bottom row is (0,0,0,1) — an affine transform
            CHECK(m[3] == 0.0f);
            CHECK(m[7] == 0.0f);
            CHECK(m[11] == 0.0f);
            CHECK(m[15] == 1.0f);
        }
    }
}

TEST_CASE("pose translation = grid center + analytic bob") {
    WaveParams w;
    const int n = 1000;
    const float spacing = 2.2f;
    const int i = 123;
    const float t = 0.9f;
    const auto m = pose_matrix(i, n, spacing, t, w);
    const Vec3 c = grid_center(i, n, spacing);
    const float ph = wave_phase(c, t, w);
    CHECK(m[12] == doctest::Approx(c.x));
    CHECK(m[13] == doctest::Approx(w.amp * std::sin(ph)));
    CHECK(m[14] == doctest::Approx(c.z));
}

TEST_CASE("tint_signal is analytic and stays inside the fixed [-1,1] window") {
    WaveParams w;
    const int n = 2000;
    const float spacing = 2.2f;
    for (int i = 0; i < n; i += 17) {
        for (float t : {0.0f, 0.5f, 3.3f, 100.0f}) {
            const float v = tint_signal(i, n, spacing, t, w);
            CHECK(v >= -1.0f);
            CHECK(v <= 1.0f);
        }
    }
}

TEST_CASE("gem_mesh: sequential index, unit outward normals, flat facets") {
    const Mesh g = gem_mesh();
    CHECK(g.vertex_count() == 36);            // 2*6 facets * 3 verts
    CHECK(g.index_count() == 36);
    for (int i = 0; i < g.index_count(); ++i) CHECK(g.index[i] == i);
    const int faces = g.vertex_count() / 3;
    for (int f = 0; f < faces; ++f) {
        const int v = f * 3;
        const float nx = g.normal[v * 3 + 0];
        const float ny = g.normal[v * 3 + 1];
        const float nz = g.normal[v * 3 + 2];
        CHECK(std::sqrt(nx * nx + ny * ny + nz * nz) == doctest::Approx(1.0f));
        // all three verts of a flat facet share the normal
        for (int k = 1; k < 3; ++k) {
            CHECK(g.normal[(v + k) * 3 + 0] == doctest::Approx(nx));
            CHECK(g.normal[(v + k) * 3 + 1] == doctest::Approx(ny));
            CHECK(g.normal[(v + k) * 3 + 2] == doctest::Approx(nz));
        }
        // outward: normal aligns with the facet centroid direction
        float cx = 0, cy = 0, cz = 0;
        for (int k = 0; k < 3; ++k) {
            cx += g.pos[(v + k) * 3 + 0];
            cy += g.pos[(v + k) * 3 + 1];
            cz += g.pos[(v + k) * 3 + 2];
        }
        CHECK(nx * cx + ny * cy + nz * cz > 0.0f);
    }
}
