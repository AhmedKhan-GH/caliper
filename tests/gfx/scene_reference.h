// scene_reference.h: the CPU reference of the graphics suite's surface scene (tests/scene/
// surface_row_fixture.h), shared by the scene core's surface row (gfx_scene.cpp) and the scene
// view's rows (gfx_scene_view.cpp), which draw the same scene, the second from wire protocol 3.2
// messages. Compiled as Objective-C++ with ARC on Apple, like the rows that include it.
#pragma once

#include <doctest/doctest.h>

#include "../scene/surface_row_fixture.h"
#include "gfx/gfx.h"
#include "gfx/shaders/generated/uniforms.h"
#include "scene/compile.h"
#include "scene/scene.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace scene_reference {

using namespace caliper_gfx;
using namespace caliper_scene;
namespace sh = caliper_gfx::shaders;

// ---- The CPU reference of the surface row ----------------------------------------------------------
//
// The compiled draws rasterised as the generated programs place them: each draw's parameter block
// is read from the draw list's arena, the vertex stage's arithmetic is repeated in f32 (vs_mesh,
// vs_lines with segment_corner, vs_points), triangles cover the pixel centres inside them (shared
// edges by the top-left rule), depth is interpolated linearly in screen space with a less-or-equal
// test, other attributes perspective-correctly, and fs_mesh_flat's light comes from the face's
// normal through the draw's normal matrix.

struct DrawInfo {
    std::string path;
    Primitive primitive = Primitive::Points;
    sh::PrimParams p{};
    DrawRecord record;
};

inline std::vector<DrawInfo> draws_of(const DrawList& list, const DrawMap& map, const SceneTree& t) {
    std::vector<DrawInfo> out;
    for (const DrawRecord& r : list.records()) {
        DrawInfo d;
        d.record = r;
        std::memcpy(&d.p, list.arena().data + r.dynamic_offsets[1], sizeof d.p);
        const DrawMapEntry* e = map.find(r.pick_id);
        REQUIRE(e != nullptr);
        d.path = t.path(e->entity);
        d.primitive = e->primitive;
        out.push_back(d);
    }
    return out;
}

struct F4 { float x, y, z, w; };
inline F4 mul(const float m[16], float x, float y, float z) {
    return {m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13],
            m[2] * x + m[6] * y + m[10] * z + m[14], m[3] * x + m[7] * y + m[11] * z + m[15]};
}
inline F4 mix(F4 a, F4 b, float t) {
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t};
}
inline constexpr float kQuad[6][2] = {{0, -1}, {1, -1}, {0, 1}, {0, 1}, {1, -1}, {1, 1}};   // quad_corner

inline bool clip_near(F4& a, F4& b) {
    if (a.z < 0 && b.z < 0) return false;
    if (a.z < 0) a = mix(a, b, a.z / (a.z - b.z));
    if (b.z < 0) b = mix(b, a, b.z / (b.z - a.z));
    return true;
}

// segment_corner of helpers.wgsl.
inline bool segment_corner(F4 a, F4 b, int c, float width_px, float W, float H, F4* out) {
    if (!clip_near(a, b)) return false;
    const float hx = 0.5f * W, hy = 0.5f * H;
    const float sax = a.x / a.w * hx, say = a.y / a.w * hy, sbx = b.x / b.w * hx, sby = b.y / b.w * hy;
    const float dx = sbx - sax, dy = sby - say, len = std::sqrt(dx * dx + dy * dy);
    float ux = 1, uy = 0;
    if (len > 1e-6f) {
        ux = dx / len;
        uy = dy / len;
    }
    const float qx = kQuad[c][0], qy = kQuad[c][1];
    const float w = std::max(width_px, 1.0f) * 0.5f;
    const float s = qx > 0.5f ? w : -w;
    const float offx = -uy * (qy * w) + ux * s, offy = ux * (qy * w) + uy * s;
    const F4 e = qx > 0.5f ? b : a;
    *out = {e.x + offx / hx * e.w, e.y + offy / hy * e.w, e.z, e.w};
    return true;
}

struct RVert {
    double x = 0, y = 0, z = 0, iw = 1;   // pixels (top-left origin), depth, 1 / w
    float c[4] = {0, 0, 0, 0};            // colour
    float o[3] = {0, 0, 0};               // entity-space position
    float uv[2] = {0, 0};
};

inline RVert to_screen(F4 c, int W, int H) {
    RVert v;
    v.x = (double(c.x) / c.w + 1.0) * 0.5 * W;
    v.y = (1.0 - double(c.y) / c.w) * 0.5 * H;
    v.z = double(c.z) / c.w;
    v.iw = 1.0 / c.w;
    return v;
}

inline uint8_t unorm8(float v) { return uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }
inline uint32_t pack(const float c[4]) {
    return uint32_t(unorm8(c[0])) | uint32_t(unorm8(c[1])) << 8 | uint32_t(unorm8(c[2])) << 16 | uint32_t(unorm8(c[3])) << 24;
}
inline int max_diff(uint32_t a, uint32_t b) {
    int d = 0;
    for (int k = 0; k < 4; ++k) d = std::max(d, std::abs(int((a >> (8 * k)) & 0xffu) - int((b >> (8 * k)) & 0xffu)));
    return d;
}

// Perspective-correct barycentric weights of the pixel centre (px, py), inside or not; the depth
// is interpolated linearly. False for a degenerate triangle.
inline bool weights(const RVert (&v)[3], double px, double py, double q[3], double* z, bool* inside) {
    auto edge = [](const RVert& a, const RVert& b, double x, double y) {
        return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
    };
    double area = edge(v[0], v[1], v[2].x, v[2].y);
    if (area == 0) return false;
    const double sign = area > 0 ? 1.0 : -1.0;
    area *= sign;
    const int from[3] = {1, 2, 0}, to[3] = {2, 0, 1};
    double l[3];
    bool in = true;
    for (int k = 0; k < 3; ++k) {
        const RVert& a = v[from[k]];
        const RVert& b = v[to[k]];
        const double e = sign * edge(a, b, px, py);
        l[k] = e / area;
        if (e < 0) in = false;
        if (e == 0) {   // the top-left rule, antisymmetric so a shared edge is covered once
            const double dx = sign * (b.x - a.x), dy = sign * (b.y - a.y);
            if (!(dy < 0 || (dy == 0 && dx > 0))) in = false;
        }
    }
    *z = l[0] * v[0].z + l[1] * v[1].z + l[2] * v[2].z;
    const double s = l[0] * v[0].iw + l[1] * v[1].iw + l[2] * v[2].iw;
    for (int k = 0; k < 3; ++k) q[k] = l[k] * v[k].iw / s;
    *inside = in;
    return true;
}

struct Reference {
    int W, H;
    uint32_t clear;
    std::vector<uint32_t> rgba, id;   // id: (draw + 1) << 24 | (item + 1), 0 for the clear
    std::vector<float> depth;
    struct Tri { RVert v[3]; float lit; int32_t index[3]; };
    int mesh_draw = -1;
    std::vector<Tri> tris;            // the mesh draw's triangles, by item
    std::vector<uint32_t> flat;       // each draw's colour, where it has one
    size_t clipped = 0;               // vertices behind the eye: none in the row's scene

    Reference(int w, int h, uint32_t c)
        : W(w), H(h), clear(c), rgba(size_t(w) * h, c), id(size_t(w) * h, 0), depth(size_t(w) * h, 1.0f) {}

    static uint32_t make_id(int draw, uint32_t item) { return uint32_t(draw + 1) << 24 | (item + 1); }
    static int draw_of(uint32_t id) { return int(id >> 24) - 1; }
    static uint32_t item_of(uint32_t id) { return (id & 0xffffffu) - 1; }

    uint32_t mesh_color(const Tri& t, const double q[3]) const {
        float c[4];
        for (int k = 0; k < 4; ++k) c[k] = float(q[0] * t.v[0].c[k] + q[1] * t.v[1].c[k] + q[2] * t.v[2].c[k]);
        for (int k = 0; k < 3; ++k) c[k] *= t.lit;
        return pack(c);
    }

    // Covers the pixel centres inside v; shade returns false to discard.
    template <class Shade> void triangle(const RVert (&v)[3], uint32_t ident, Shade&& shade) {
        const double x0 = std::min({v[0].x, v[1].x, v[2].x}), x1 = std::max({v[0].x, v[1].x, v[2].x});
        const double y0 = std::min({v[0].y, v[1].y, v[2].y}), y1 = std::max({v[0].y, v[1].y, v[2].y});
        const int i0 = std::max(0, int(std::floor(x0 - 0.5))), i1 = std::min(W - 1, int(std::ceil(x1)));
        const int j0 = std::max(0, int(std::floor(y0 - 0.5))), j1 = std::min(H - 1, int(std::ceil(y1)));
        for (int j = j0; j <= j1; ++j)
            for (int i = i0; i <= i1; ++i) {
                double q[3], z;
                bool inside;
                if (!weights(v, i + 0.5, j + 0.5, q, &z, &inside) || !inside) continue;
                if (z < 0 || z > 1) continue;
                const size_t k = size_t(j) * W + i;
                if (!(float(z) <= depth[k])) continue;
                uint32_t out;
                if (!shade(q, &out)) continue;
                rgba[k] = out;
                id[k] = ident;
                depth[k] = float(z);
            }
    }

    // The colour candidate `ident` would give the pixel (i, j): its own colour, extrapolated over
    // the triangle's plane for a mesh face.
    uint32_t evaluate(uint32_t ident, int i, int j) const {
        if (ident == 0) return clear;
        const int d = draw_of(ident);
        if (d != mesh_draw) return flat[size_t(d)];
        const Tri& t = tris[item_of(ident)];
        double q[3], z;
        bool inside;
        if (!weights(t.v, i + 0.5, j + 0.5, q, &z, &inside)) return clear;
        return mesh_color(t, q);
    }
    int tolerance(uint32_t ident) const { return ident != 0 && draw_of(ident) == mesh_draw ? 2 : 0; }
};

inline Reference rasterise(const std::vector<DrawInfo>& draws, const scene_fixtures::SurfaceRowFixture& fx) {
    const int W = fx.kSize, H = fx.kSize;
    Reference ref(W, H, fx.kClear);
    ref.flat.assign(draws.size(), 0);
    for (size_t di = 0; di < draws.size(); ++di) {
        const DrawInfo& d = draws[di];
        const sh::PrimParams& p = d.p;
        const int draw = int(di);
        auto flat_shade = [&](const double*, uint32_t* out) {
            *out = p.flat_rgba;
            return true;
        };
        auto check_w = [&](const F4& c) {
            if (!(c.w > 0) || c.z < 0 || c.z > c.w) ++ref.clipped;
        };
        if (d.primitive == Primitive::Mesh) {
            ref.mesh_draw = draw;
            // Each vertex's colour from the CPU reference map, as the colormap block decides it.
            const size_t n = fx.values.size();
            std::vector<uint8_t> colors(n * 4);
            caliper_host::map_f32_to_rgba8(fx.values.data(), int(n), 1, caliper_host::colormap_lut(int32_t(fx.kColormap)),
                                           fx.vmin, fx.vmax, colors.data());
            ref.tris.resize(fx.indices.size() / 3);
            for (size_t t = 0; t < ref.tris.size(); ++t) {
                Reference::Tri& tri = ref.tris[t];
                const float* pv[3];
                for (int k = 0; k < 3; ++k) {
                    const int32_t vi = fx.indices[3 * t + k];
                    tri.index[k] = vi;
                    pv[k] = &fx.positions[3 * size_t(vi)];
                    const F4 c = mul(p.mvp, pv[k][0], pv[k][1], pv[k][2]);
                    check_w(c);
                    tri.v[k] = to_screen(c, W, H);
                    for (int ch = 0; ch < 4; ++ch) tri.v[k].c[ch] = colors[4 * size_t(vi) + ch] / 255.0f;
                    std::memcpy(tri.v[k].o, pv[k], 12);
                }
                // fs_mesh_flat: the face's normal through the normal matrix, lit on both sides.
                const float e1[3] = {pv[1][0] - pv[0][0], pv[1][1] - pv[0][1], pv[1][2] - pv[0][2]};
                const float e2[3] = {pv[2][0] - pv[0][0], pv[2][1] - pv[0][1], pv[2][2] - pv[0][2]};
                const float n3[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
                float vn[3];
                for (int k = 0; k < 3; ++k) vn[k] = n3[0] * p.nmat0[k] + n3[1] * p.nmat1[k] + n3[2] * p.nmat2[k];
                const float len = std::sqrt(vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]);
                tri.lit = 0.30f + 0.70f * std::fabs(vn[2] / len);
                ref.triangle(tri.v, Reference::make_id(draw, uint32_t(t)), [&](const double* q, uint32_t* out) {
                    *out = ref.mesh_color(tri, q);
                    return true;
                });
            }
        } else if (d.primitive == Primitive::Lines) {
            ref.flat[di] = p.flat_rgba;
            const bool strip = d.path == "path";
            const std::vector<float>& rows = strip ? fx.path : fx.box;
            const size_t n = rows.size() / 3, segments = strip ? n - 1 : n / 2;
            REQUIRE(d.record.instance_count == segments);
            for (size_t s = 0; s < segments; ++s) {
                if (strip && fx.keep[s] == 0) continue;
                const float* a = &rows[3 * (strip ? s : 2 * s)];
                const float* b = a + 3;
                const F4 ca = mul(p.mvp, a[0], a[1], a[2]), cb = mul(p.mvp, b[0], b[1], b[2]);
                check_w(ca);
                check_w(cb);
                RVert v[6];
                for (int c = 0; c < 6; ++c) {
                    F4 out;
                    REQUIRE(segment_corner(ca, cb, c, p.size_px, float(W), float(H), &out));
                    v[c] = to_screen(out, W, H);
                }
                const RVert t0[3] = {v[0], v[1], v[2]}, t1[3] = {v[3], v[4], v[5]};
                ref.triangle(t0, Reference::make_id(draw, uint32_t(s)), flat_shade);
                ref.triangle(t1, Reference::make_id(draw, uint32_t(s)), flat_shade);
            }
        } else if (d.primitive == Primitive::Points) {
            ref.flat[di] = p.flat_rgba;
            const F4 c = mul(p.mvp, fx.marker[0], fx.marker[1], fx.marker[2]);
            check_w(c);
            const float kx = p.size_px / float(W) * c.w, ky = p.size_px / float(H) * c.w;
            RVert v[6];
            for (int k = 0; k < 6; ++k) {
                const float cx = kQuad[k][0] * 2 - 1, cy = kQuad[k][1];
                v[k] = to_screen({c.x + cx * kx, c.y + cy * ky, c.z, c.w}, W, H);
                v[k].uv[0] = cx;
                v[k].uv[1] = cy;
            }
            // fs_points: a disc cut where |uv| > 1.
            auto disc = [&](const RVert (&t)[3]) {
                ref.triangle(t, Reference::make_id(draw, 0), [&](const double* q, uint32_t* out) {
                    const double u = q[0] * t[0].uv[0] + q[1] * t[1].uv[0] + q[2] * t[2].uv[0];
                    const double w = q[0] * t[0].uv[1] + q[1] * t[1].uv[1] + q[2] * t[2].uv[1];
                    if (p.point_shape == 1 && u * u + w * w > 1.0) return false;
                    *out = p.flat_rgba;
                    return true;
                });
            };
            const RVert t0[3] = {v[0], v[1], v[2]}, t1[3] = {v[3], v[4], v[5]};
            disc(t0);
            disc(t1);
        }
    }
    return ref;
}

// The comparison under web geometry's parity rules.
struct Comparison {
    size_t interior = 0, edge = 0, failed = 0;
    int max_interior_lit = 0;            // the largest difference on a lit interior pixel
    size_t lit = 0, lit_own = 0;         // lit pixels, and those within 2 levels of their own face
    std::map<int, size_t> interior_by_draw;
    std::string first;
};

inline Comparison compare(const Reference& ref, const std::vector<uint8_t>& got) {
    Comparison c;
    for (int j = 0; j < ref.H; ++j)
        for (int i = 0; i < ref.W; ++i) {
            const size_t k = size_t(j) * ref.W + i;
            uint32_t g;
            std::memcpy(&g, &got[4 * k], 4);
            uint32_t ids[9];
            int n = 0;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    const int x = std::clamp(i + di, 0, ref.W - 1), y = std::clamp(j + dj, 0, ref.H - 1);
                    const uint32_t v = ref.id[size_t(y) * ref.W + x];
                    if (std::find(ids, ids + n, v) == ids + n) ids[n++] = v;
                }
            if (ref.tolerance(ref.id[k]) > 0) {
                ++c.lit;
                c.lit_own += max_diff(g, ref.rgba[k]) <= ref.tolerance(ref.id[k]);
            }
            bool ok;
            if (n == 1) {
                ++c.interior;
                ++c.interior_by_draw[ids[0] ? Reference::draw_of(ids[0]) : -1];
                const int d = max_diff(g, ref.rgba[k]);
                if (ref.tolerance(ids[0]) > 0) c.max_interior_lit = std::max(c.max_interior_lit, d);
                ok = d <= ref.tolerance(ids[0]);
            } else {
                ++c.edge;
                ok = false;
                for (int m = 0; m < n && !ok; ++m) ok = max_diff(g, ref.evaluate(ids[m], i, j)) <= ref.tolerance(ids[m]);
            }
            if (!ok) {
                if (c.failed == 0) {
                    char buf[160];
                    std::snprintf(buf, sizeof buf, "pixel (%d, %d): got %08x, reference %08x, %d candidates", i, j, g,
                                  ref.rgba[k], n);
                    c.first = buf;
                }
                ++c.failed;
            }
        }
    return c;
}

}  // namespace scene_reference
