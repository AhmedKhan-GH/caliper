#pragma once
// ============================================================================
// instance_field.h — the pure (torch-free, std::-only) math behind InstanceScope.
//
// The worker translates these formulas to vectorized device tensors; keeping the
// reference here as scalar std:: functions lets the fast unit suite (caliper_tests)
// check the load-bearing properties WITHOUT linking torch:
//   * grid_dims / grid_center — the N-object lattice, centered on the origin;
//   * pose_matrix — a RIGID column-major model transform (rotation + translation
//     ONLY, so the LAMBERT §5.1 rigidity gate G14 never refuses it);
//   * tint_signal — an ANALYTIC signal in [-1,1] (sin), so a FIXED [-1,1] MAGMA
//     window is correct and can never saturate (the TwinScope lesson);
//   * gem_mesh — the small faceted procedural gem (flat-shaded, outward normals).
// ============================================================================
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace instancescope {

inline constexpr float kPi = 3.14159265358979323846f;

// ---- grid layout -----------------------------------------------------------
struct GridDims { int cols, rows; };

// A near-square grid that holds n instances (i = row*cols + col).
inline GridDims grid_dims(int n) {
    if (n < 1) n = 1;
    int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
    if (cols < 1) cols = 1;
    int rows = (n + cols - 1) / cols;
    return {cols, rows};
}

struct Vec3 { float x, y, z; };

// The (x, z) center of instance i on the origin-centered grid; y stays 0 (the
// bob is added by pose_matrix). Uniform `spacing` between grid neighbors.
inline Vec3 grid_center(int i, int n, float spacing) {
    const GridDims d = grid_dims(n);
    const int c = i % d.cols;
    const int r = i / d.cols;
    const float x = (static_cast<float>(c) - static_cast<float>(d.cols - 1) * 0.5f) * spacing;
    const float z = (static_cast<float>(r) - static_cast<float>(d.rows - 1) * 0.5f) * spacing;
    return {x, 0.0f, z};
}

// ---- traveling wave --------------------------------------------------------
struct WaveParams {
    float k     = 0.42f;   // spatial frequency along the x+z diagonal
    float omega = 1.7f;    // temporal frequency (rad/s)
    float amp   = 0.75f;   // bob amplitude (world units)
    float tilt  = 0.55f;   // constant facet tilt (rad) so faces catch the light
};

inline float wave_phase(Vec3 c, float t, const WaveParams& w) {
    return w.k * (c.x + c.z) - w.omega * t;
}

// tint = sin(phase) in [-1,1]. Analytic ⇒ a fixed [-1,1] colormap window is
// exactly right and can never saturate.
inline float tint_signal(int i, int n, float spacing, float t, const WaveParams& w) {
    return std::sin(wave_phase(grid_center(i, n, spacing), t, w));
}

// A RIGID column-major 4x4 model matrix for instance i at time t:
//   R = Ry(theta) * Rx(tilt),  theta = phase (spin),  T = center + bob*ŷ.
// A product of two rotations is a rotation, so R is orthonormal (RᵀR = I) with
// det = +1 — the G14 rigidity gate accepts it; shear/scale would be refused.
inline std::array<float, 16> pose_matrix(int i, int n, float spacing, float t,
                                         const WaveParams& w) {
    const Vec3 c = grid_center(i, n, spacing);
    const float ph  = wave_phase(c, t, w);
    const float bob = w.amp * std::sin(ph);
    const float cy = std::cos(ph),     sy = std::sin(ph);      // spin (about Y)
    const float ca = std::cos(w.tilt), sa = std::sin(w.tilt);  // fixed tilt (about X)
    std::array<float, 16> m{};
    // column 0
    m[0]  = cy;      m[1]  = 0.0f;    m[2]  = -sy;     m[3]  = 0.0f;
    // column 1
    m[4]  = sy * sa; m[5]  = ca;      m[6]  = cy * sa; m[7]  = 0.0f;
    // column 2
    m[8]  = sy * ca; m[9]  = -sa;     m[10] = cy * ca; m[11] = 0.0f;
    // column 3 (translation)
    m[12] = c.x;     m[13] = bob;     m[14] = c.z;     m[15] = 1.0f;
    return m;
}

// ---- procedural faceted gem ------------------------------------------------
struct Mesh {
    std::vector<float>   pos;     // (V*3) f32, flat-shaded (per-face duplicated)
    std::vector<float>   normal;  // (V*3) f32, one outward normal per face
    std::vector<int32_t> index;   // (V,) sequential — each facet its own verts
    int vertex_count() const { return static_cast<int>(pos.size() / 3); }
    int index_count()  const { return static_cast<int>(index.size()); }
};

// A small bipyramid gem: an M-gon girdle ring (radius r, y=0) capped by a top
// and a bottom apex, flat-shaded. Each of the 2*M triangular facets carries its
// own three vertices and a single geometric normal oriented OUTWARD (aligned
// with the facet centroid — robust regardless of winding, and the Metal geom
// pipeline does not cull). ~36 verts at the default sides=6 ("tens of vertices").
inline Mesh gem_mesh(int sides = 6, float r = 0.7f, float top = 1.05f,
                     float bot = 0.75f) {
    if (sides < 3) sides = 3;
    Mesh mesh;
    auto ring = [&](int j) -> Vec3 {
        const float a = 2.0f * kPi * static_cast<float>(j % sides) /
                        static_cast<float>(sides);
        return {r * std::cos(a), 0.0f, r * std::sin(a)};
    };
    const Vec3 apex_top{0.0f, top, 0.0f};
    const Vec3 apex_bot{0.0f, -bot, 0.0f};
    auto add_tri = [&](Vec3 A, Vec3 B, Vec3 C) {
        const Vec3 e1{B.x - A.x, B.y - A.y, B.z - A.z};
        const Vec3 e2{C.x - A.x, C.y - A.y, C.z - A.z};
        Vec3 nrm{e1.y * e2.z - e1.z * e2.y,
                 e1.z * e2.x - e1.x * e2.z,
                 e1.x * e2.y - e1.y * e2.x};
        const float len = std::sqrt(nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z);
        if (len > 0.0f) { nrm.x /= len; nrm.y /= len; nrm.z /= len; }
        const Vec3 cen{(A.x + B.x + C.x) / 3.0f,
                       (A.y + B.y + C.y) / 3.0f,
                       (A.z + B.z + C.z) / 3.0f};
        if (nrm.x * cen.x + nrm.y * cen.y + nrm.z * cen.z < 0.0f) {
            nrm.x = -nrm.x; nrm.y = -nrm.y; nrm.z = -nrm.z;
        }
        for (const Vec3& v : {A, B, C}) {
            mesh.pos.push_back(v.x); mesh.pos.push_back(v.y); mesh.pos.push_back(v.z);
            mesh.normal.push_back(nrm.x); mesh.normal.push_back(nrm.y); mesh.normal.push_back(nrm.z);
            mesh.index.push_back(static_cast<int32_t>(mesh.index.size()));
        }
    };
    for (int j = 0; j < sides; ++j) add_tri(apex_top, ring(j), ring(j + 1));
    for (int j = 0; j < sides; ++j) add_tri(apex_bot, ring(j + 1), ring(j));
    return mesh;
}

}  // namespace instancescope
