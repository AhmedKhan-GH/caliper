// draw_reference.h: the CPU reference of a scene drawn by the scene core, for every primitive, and
// the comparison under web geometry's parity rules (design/web-geometry.md, "Parity with the native
// renderers"). The graphics suite's scene view rows on Metal (tests/gfx/gfx_scene_view.cpp) and the
// browser module's scene self-test on WebGPU (web/wasm/src/scene_selftest.cpp) both use it, so it
// depends on no test framework.
//
// The reference rasterises the draws the scene core compiled, in their order, from two sources: each
// draw's parameter block and frame block as the draw list's arena holds them (the matrices, sizes,
// colour mode, flat colour and range the GPU reads), and the arrays of the part as the protocol 3.2
// messages carried them (positions, colours, values, class ids, radii, strip lengths, index pairs,
// centres, half sizes, rotations, normals, pixels), read with the protocol's meaning rather than
// through the scene view's buffers, so that the view's conversions (break masks, per-strip colours,
// u16 depths, class ids) are checked as well. It repeats each vertex stage of the generated programs
// in f32 (vs_points, vs_depth_points, vs_lines for segments and arrows, vs_box_wire, vs_box_solid,
// vs_mesh, vs_image), covers the pixel centres inside each triangle (shared edges by the top-left
// rule), interpolates depth linearly in screen space and the other attributes perspective-correctly,
// tests depth less-or-equal, blends with the layer's alpha equations, honours the scissor, and
// shades as the fragment stages do: a colormapped value through the CPU reference map_f32_to_rgba8,
// class ids through the class table the SCENE_CLASSES message sets over the 16-colour palette,
// images by nearest or bilinear lookups.
//
// Each pixel keeps the identity of the primitive that wrote it last (a draw and an item: a point, a
// segment, an arrow, a box, a triangle, or a texel of an image looked up by nearest) and how far its
// colour may move: 0 for a lookup (a flat or per-item colour, a colormapped value, a class colour, a
// texel by nearest), 1 for a colour interpolated between two ends of a line or sampled bilinearly
// and for any pixel that was blended, 2 for a lit or flat-shaded face.
#pragma once

#include "gfx/gfx.h"
#include "gfx/shaders/generated/uniforms.h"
#include "net/wire.h"
#include "scene/compile.h"
#include "scene/resources.h"
#include "scene/scene.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace draw_reference {

namespace wire = caliper_net::wire;
namespace sh = caliper_gfx::shaders;
using caliper_scene::DrawMap;
using caliper_scene::DrawMapEntry;
using caliper_scene::DrawPart;
using caliper_scene::Primitive;

// ---- what was sent -----------------------------------------------------------------------------------

// The parts a fixture sent: the latest update of each entity part, and the class table's texels.
struct SceneParts {
    std::map<std::pair<std::string, std::string>, const wire::SceneUpdate*> updates;   // (path, part)
    std::vector<uint32_t> class_texels;

    SceneParts() : class_texels(65536) {
        for (uint32_t id = 0; id < 65536; ++id) class_texels[id] = caliper_scene::kPalette[id % 16];
    }
    void add(const wire::SceneUpdate& u) { updates[{u.path, u.part}] = &u; }
    void classes(const wire::SceneClasses& c) {
        for (uint32_t id = 0; id < 65536; ++id) class_texels[id] = caliper_scene::kPalette[id % 16];
        for (const caliper_net::Json& e : c.classes.items()) {
            int64_t id = 0;
            if (!e.find("id") || !e.find("id")->get_i64(&id) || id < 0 || id > 65535) continue;
            const caliper_net::Json* col = e.find("color");
            uint32_t rgba = 0;
            for (size_t k = 0; col && k < 4; ++k) rgba |= uint32_t(col->at(k).as_double()) << (8 * k);
            class_texels[size_t(id)] = rgba;
        }
    }
    const wire::SceneUpdate* find(const std::string& path, const char* part) const {
        const auto it = updates.find({path, part});
        return it == updates.end() ? nullptr : it->second;
    }
};

// ---- small arithmetic, as the shaders do it -------------------------------------------------------------

struct F4 { float x = 0, y = 0, z = 0, w = 1; };
inline F4 mul(const float m[16], float x, float y, float z) {
    return {m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13],
            m[2] * x + m[6] * y + m[10] * z + m[14], m[3] * x + m[7] * y + m[11] * z + m[15]};
}
inline F4 mix(F4 a, F4 b, float t) {
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t};
}
inline constexpr float kQuad[6][2] = {{0, -1}, {1, -1}, {0, 1}, {0, 1}, {1, -1}, {1, 1}};   // quad_corner

inline bool is_nan_bits(float v) {
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}
inline uint32_t bits_of(float v) {
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
}
inline float float_of(uint32_t b) {
    float v;
    std::memcpy(&v, &b, 4);
    return v;
}

// clip_near of helpers.wgsl.
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

inline uint8_t unorm8(float v) { return uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }
inline uint32_t pack(const float c[4]) {
    return uint32_t(unorm8(c[0])) | uint32_t(unorm8(c[1])) << 8 | uint32_t(unorm8(c[2])) << 16 | uint32_t(unorm8(c[3])) << 24;
}
inline void unpack(uint32_t c, float out[4]) {
    for (int k = 0; k < 4; ++k) out[k] = float((c >> (8 * k)) & 0xffu) / 255.0f;
}
inline int max_diff(uint32_t a, uint32_t b) {
    int d = 0;
    for (int k = 0; k < 4; ++k) d = std::max(d, std::abs(int((a >> (8 * k)) & 0xffu) - int((b >> (8 * k)) & 0xffu)));
    return d;
}

// ---- the rasteriser --------------------------------------------------------------------------------------

struct RVert {
    double x = 0, y = 0, z = 0, iw = 1;   // pixels (top-left origin), depth, 1 / w
    float c[4] = {0, 0, 0, 0};            // colour
    float uv[2] = {0, 0};
    float o[3] = {0, 0, 0};               // entity-space position
    bool hidden = false;                  // hidden() in the shader: z > w
    bool clipped = false;                 // behind the eye or past the far plane
};

inline RVert to_screen(F4 c, int W, int H) {
    RVert v;
    v.hidden = c.w > 0 && c.z > c.w && c.z == 2.0f && c.w == 1.0f;
    v.clipped = !(c.w > 0) || c.z < 0 || c.z > c.w;
    v.x = (double(c.x) / c.w + 1.0) * 0.5 * W;
    v.y = (1.0 - double(c.y) / c.w) * 0.5 * H;
    v.z = double(c.z) / c.w;
    v.iw = 1.0 / c.w;
    return v;
}

// Perspective-correct barycentric weights of the pixel centre (px, py), inside or not; the depth
// is interpolated linearly. False for a degenerate triangle.
inline bool weights(const RVert (&v)[3], double px, double py, double q[3], double* z, bool* inside, double* minw) {
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
    *minw = std::min({l[0], l[1], l[2]});
    return true;
}

// The fragment stage of a triangle: false discards; out is straight RGBA in [0, 1].
using Shade = std::function<bool(const RVert (&v)[3], const double q[3], float out[4])>;

struct Tri {
    RVert v[3];
    uint32_t ident = 0;
    int shade = -1;   // index into Reference::shades
};

struct Reference {
    int W = 0, H = 0;
    uint32_t clear = 0;
    std::vector<uint32_t> rgba, id;   // id: (draw + 1) << 24 | (item + 1), 0 for the clear
    std::vector<uint32_t> base;       // the identity of the last opaque write, under any blended ones
    // The other side of the top identity's edge: the nearest colour of another identity that the top
    // write replaced or hid in depth (the clear at first), and that identity, which an edge pixel may
    // show although no neighbour does (web geometry, "Parity": an edge pixel takes either side).
    std::vector<uint32_t> behind, behind_id;
    std::vector<double> behind_z;
    std::vector<float> depth;
    std::vector<double> depth_d;      // the same depth before its rounding to f32
    std::vector<uint8_t> contested;   // a write tied in depth with another identity, within 1e-5
    std::vector<uint8_t> tol;         // how far each pixel's colour may move
    std::vector<int> tol_of_draw;     // per draw: the tolerance of its unblended pixels
    std::vector<bool> blended_draw;   // per draw: alpha blended
    std::vector<Shade> shades;
    std::vector<Tri> tris;            // every triangle drawn, for evaluating a candidate at an edge
    std::map<uint32_t, std::vector<uint32_t>> tris_of;   // ident to its triangles
    std::map<uint32_t, int> tol_of_ident;                 // an ident's own tolerance, where it differs
    size_t clipped = 0;               // triangles that would need clipping: none in a row's scene
    size_t triangles = 0;
    std::vector<size_t> writes;       // per draw: the pixels it wrote

    Reference(int w, int h, uint32_t c)
        : W(w), H(h), clear(c), rgba(size_t(w) * h, c), id(size_t(w) * h, 0), base(size_t(w) * h, 0),
          behind(size_t(w) * h, c), behind_id(size_t(w) * h, 0), behind_z(size_t(w) * h, 2.0),
          depth(size_t(w) * h, 1.0f), depth_d(size_t(w) * h, 1.0), contested(size_t(w) * h, 0),
          tol(size_t(w) * h, 0) {}

    static uint32_t make_id(int draw, uint32_t item) { return uint32_t(draw + 1) << 24 | ((item + 1) & 0xffffffu); }
    static int draw_of(uint32_t ident) { return int(ident >> 24) - 1; }

    int tolerance(uint32_t ident) const {
        if (ident == 0) return 0;
        const auto it = tol_of_ident.find(ident);
        if (it != tol_of_ident.end()) return it->second;
        return tol_of_draw[size_t(draw_of(ident))];
    }

    struct State {
        bool depth_test = false, depth_write = false, blend = false;
        int scissor[4] = {0, 0, 1 << 30, 1 << 30};
    };

    // Covers the pixel centres inside v with the state's tests and the shade.
    void triangle(const RVert (&v)[3], uint32_t ident, int shade, const State& st) {
        ++triangles;
        if (v[0].hidden && v[1].hidden && v[2].hidden) return;
        if (v[0].clipped || v[1].clipped || v[2].clipped) {
            ++clipped;
            return;
        }
        Tri t;
        for (int k = 0; k < 3; ++k) t.v[k] = v[k];
        t.ident = ident;
        t.shade = shade;
        tris_of[ident].push_back(uint32_t(tris.size()));
        tris.push_back(t);
        const double x0 = std::min({v[0].x, v[1].x, v[2].x}), x1 = std::max({v[0].x, v[1].x, v[2].x});
        const double y0 = std::min({v[0].y, v[1].y, v[2].y}), y1 = std::max({v[0].y, v[1].y, v[2].y});
        const int i0 = std::max({0, int(std::floor(x0 - 0.5)), st.scissor[0]});
        const int i1 = std::min({W - 1, int(std::ceil(x1)), st.scissor[0] + st.scissor[2] - 1});
        const int j0 = std::max({0, int(std::floor(y0 - 0.5)), st.scissor[1]});
        const int j1 = std::min({H - 1, int(std::ceil(y1)), st.scissor[1] + st.scissor[3] - 1});
        const int dtol = tol_of_draw.back();
        for (int j = j0; j <= j1; ++j)
            for (int i = i0; i <= i1; ++i) {
                double q[3], z, mw;
                bool inside;
                if (!weights(v, i + 0.5, j + 0.5, q, &z, &inside, &mw) || !inside) continue;
                if (z < 0 || z > 1) continue;
                const size_t k = size_t(j) * W + i;
                // Near ties in depth between two identities: the GPU may break them the other way.
                if (st.depth_test && id[k] != 0 && id[k] != ident && std::fabs(z - depth_d[k]) < 1e-5) contested[k] = 1;
                if (st.depth_test && !(float(z) <= depth[k])) {
                    // Hidden here: the nearest such fragment of another identity is the far side.
                    float c[4];
                    if (!st.blend && id[k] != ident && z < behind_z[k] && shades[size_t(shade)](v, q, c)) {
                        behind[k] = pack(c);
                        behind_id[k] = ident;
                        behind_z[k] = z;
                    }
                    continue;
                }
                float c[4];
                if (!shades[size_t(shade)](v, q, c)) continue;
                if (id[k] != ident) {   // what this write replaces is the far side of its edge
                    behind[k] = rgba[k];
                    behind_id[k] = id[k];
                    behind_z[k] = depth_d[k];
                }
                if (st.blend) {
                    float d[4];
                    unpack(rgba[k], d);
                    const float a = c[3];
                    float o[4];
                    for (int ch = 0; ch < 3; ++ch) o[ch] = c[ch] * a + d[ch] * (1.0f - a);
                    o[3] = a + d[3] * (1.0f - a);
                    rgba[k] = pack(o);
                    tol[k] = uint8_t(std::max(1, dtol));
                } else {
                    rgba[k] = pack(c);
                    tol[k] = uint8_t(tolerance(ident));
                    base[k] = ident;
                }
                id[k] = ident;
                ++writes[size_t(draw_of(ident))];
                if (st.depth_write) {
                    depth[k] = float(z);
                    depth_d[k] = z;
                }
            }
    }

    // The colour candidate `ident` would give the pixel (i, j) if it covered it, opaque: its
    // triangle nearest the pixel, its weights extrapolated there. False when it cannot say.
    bool evaluate(uint32_t ident, int i, int j, uint32_t* out) const {
        if (ident == 0) {
            *out = clear;
            return true;
        }
        if (blended_draw[size_t(draw_of(ident))]) return false;
        const auto it = tris_of.find(ident);
        if (it == tris_of.end()) return false;
        double best = -1e300;
        bool found = false;
        for (const uint32_t ti : it->second) {
            const Tri& t = tris[ti];
            double q[3], z, mw;
            bool inside;
            if (!weights(t.v, i + 0.5, j + 0.5, q, &z, &inside, &mw) || mw <= best) continue;
            float c[4];
            if (!shades[size_t(t.shade)](t.v, q, c)) continue;   // a disc's edge: the clear or another side
            best = mw;
            *out = pack(c);
            found = true;
        }
        return found;
    }
};

// ---- reading the messages -----------------------------------------------------------------------------

struct Rows {
    const uint8_t* data = nullptr;
    uint64_t rows = 0;
    uint32_t width = 0;     // components per row
    int64_t dtype = CALIPER_NET_DTYPE_F32;
    explicit operator bool() const { return data != nullptr; }
    float f(uint64_t r, uint32_t c) const {
        float v;
        std::memcpy(&v, data + (r * width + c) * 4, 4);
        return v;
    }
    uint32_t u(uint64_t r) const {
        uint32_t v;
        std::memcpy(&v, data + r * 4, 4);
        return v;
    }
};

inline Rows rows(const wire::SceneUpdate* u, const char* name) {
    Rows r;
    if (!u) return r;
    const wire::Section* s = u->array(name);
    if (!s || s->shape.empty()) return r;
    r.data = s->data;
    r.rows = uint64_t(s->shape[0]);
    r.width = s->shape.size() >= 2 ? uint32_t(s->shape[1]) : 1;
    r.dtype = s->dtype;
    return r;
}

inline std::string str(const wire::SceneUpdate& u, const char* key, const char* fallback) {
    const caliper_net::Json* v = u.params.find(key);
    return v && v->is_string() ? v->as_string() : std::string(fallback);
}

inline double num(const wire::SceneUpdate& u, const char* key, double fallback) {
    const caliper_net::Json* v = u.params.find(key);
    return v && v->is_number() ? v->as_double() : fallback;
}

// ---- the draws -------------------------------------------------------------------------------------

struct Rasteriser {
    const caliper_gfx::DrawList& list;
    const DrawMap& map;
    const caliper_scene::SceneTree& tree;   // paths of the map's entities
    const SceneParts& parts;
    Reference ref;
    std::vector<std::string> problems;
    // The pick pass instead of the colour pass: every draw opaque and writing depth where it tests
    // it, as compile_pick sets its pipelines; the identities are then what the pick window holds.
    bool pick = false;
    std::map<int, std::vector<std::array<uint32_t, 3>>> mesh_vertices;   // per mesh draw: each triangle's vertices

    Rasteriser(const caliper_gfx::DrawList& l, const DrawMap& m, const caliper_scene::SceneTree& t,
               const SceneParts& p, int w, int h)
        : list(l), map(m), tree(t), parts(p), ref(w, h, l.target().clear_rgba) {}

    // One colour source of 4 bytes by the draw's colour mode, as item_color does.
    uint32_t item_rgba(const sh::PrimParams& p, uint32_t src, uint32_t colormap) const {
        switch (p.color_mode) {
            case 1: {
                const float v = float_of(src);
                uint8_t out[4];
                caliper_host::map_f32_to_rgba8(&v, 1, 1, caliper_host::colormap_lut(int32_t(colormap)), p.vmin, p.vmax, out);
                return uint32_t(out[0]) | uint32_t(out[1]) << 8 | uint32_t(out[2]) << 16 | uint32_t(out[3]) << 24;
            }
            case 2: return src;
            case 3: return 0xffffffffu;
            case 4: return src < 65536 ? parts.class_texels[src] : caliper_scene::kDefaultRgba;
            default: return p.flat_rgba;
        }
    }

    // The colour sources of a part's items (colors, values or class ids); none: the flat colour.
    struct Source {
        Rows rows;
        bool any() const { return bool(rows); }
        uint32_t at(uint64_t i) const { return rows.rows == 1 ? rows.u(0) : rows.u(i); }
    };
    Source source(const wire::SceneUpdate* u) const {
        Source s;
        s.rows = rows(u, "colors");
        if (!s.rows) s.rows = rows(u, "values");
        if (!s.rows) s.rows = rows(u, "class_ids");
        return s;
    }

    static Shade flat_shade() {
        return [](const RVert (&v)[3], const double q[3], float out[4]) {
            for (int k = 0; k < 4; ++k) out[k] = float(q[0] * v[0].c[k] + q[1] * v[1].c[k] + q[2] * v[2].c[k]);
            return true;
        };
    }

    int add_shade(Shade s) {
        ref.shades.push_back(std::move(s));
        return int(ref.shades.size() - 1);
    }

    void quad(const RVert (&v)[6], uint32_t ident, int shade, const Reference::State& st) {
        const RVert t0[3] = {v[0], v[1], v[2]}, t1[3] = {v[3], v[4], v[5]};
        ref.triangle(t0, ident, shade, st);
        ref.triangle(t1, ident, shade, st);
    }

    // A segment from a to b with colours ca and cb, as vs_lines draws instance `item`.
    void segment(int draw, uint32_t item, F4 a, F4 b, const float ca[4], const float cb[4], float width,
                 int shade, const Reference::State& st) {
        RVert v[6];
        for (int c = 0; c < 6; ++c) {
            F4 out;
            if (!segment_corner(a, b, c, width, float(ref.W), float(ref.H), &out)) return;
            v[c] = to_screen(out, ref.W, ref.H);
            std::memcpy(v[c].c, kQuad[c][0] > 0.5f ? cb : ca, 16);
        }
        if (std::memcmp(ca, cb, 16) != 0) ref.tol_of_ident[Reference::make_id(draw, item)] = 1;   // interpolated
        quad(v, Reference::make_id(draw, item), shade, st);
    }

    void points(int draw, const sh::PrimParams& p, const sh::Frame& fb, const wire::SceneUpdate& u,
                const Reference::State& st) {
        const Rows pos = rows(&u, "positions");
        const Source src = source(&u);
        const Rows radii = rows(&u, "radii");
        const float one_radius = float(num(u, "radius", 0.0));
        const uint32_t cmap = uint32_t(num(u, "colormap", 0));
        const int shade = add_shade([shape = p.point_shape](const RVert (&v)[3], const double q[3], float out[4]) {
            const double s = q[0] * v[0].uv[0] + q[1] * v[1].uv[0] + q[2] * v[2].uv[0];
            const double t = q[0] * v[0].uv[1] + q[1] * v[1].uv[1] + q[2] * v[2].uv[1];
            if (shape == 1 && s * s + t * t > 1.0) return false;
            std::memcpy(out, v[0].c, 16);
            return true;
        });
        for (uint64_t i = 0; i < pos.rows; ++i) {
            const float x = pos.f(i, 0), y = pos.f(i, 1), z = pos.width == 3 ? pos.f(i, 2) : 0.0f;
            if (is_nan_bits(x) || is_nan_bits(y) || is_nan_bits(z)) continue;
            float col[4];
            unpack(item_rgba(p, src.any() ? src.at(i) : 0u, cmap), col);
            const F4 c = mul(p.mvp, x, y, z);
            float kx = (p.size_px / fb.viewport[0]) * c.w, ky = (p.size_px / fb.viewport[1]) * c.w;
            if (p.radius_mode == 1) {
                const float r = radii ? radii.f(radii.rows == 1 ? 0 : i, 0) : one_radius;
                kx = std::max(r * p.radius_scale * fb.proj_scale[0], c.w / fb.viewport[0]);
                ky = std::max(r * p.radius_scale * fb.proj_scale[1], c.w / fb.viewport[1]);
            }
            RVert v[6];
            for (int k = 0; k < 6; ++k) {
                const float cx = kQuad[k][0] * 2 - 1, cy = kQuad[k][1];
                v[k] = to_screen({c.x + cx * kx, c.y + cy * ky, c.z, c.w}, ref.W, ref.H);
                v[k].uv[0] = cx;
                v[k].uv[1] = cy;
                std::memcpy(v[k].c, col, 16);
            }
            quad(v, Reference::make_id(draw, uint32_t(i)), shade, st);
        }
    }

    void lines(int draw, const sh::PrimParams& p, const wire::SceneUpdate& u, const Reference::State& st) {
        const Rows pos = rows(&u, "positions");
        const Source src = source(&u);
        const uint32_t cmap = uint32_t(num(u, "colormap", 0));
        const std::string mode = str(u, "mode", CALIPER_NET_LINES_SEGMENTS);
        const std::string per = str(u, "color_per", CALIPER_NET_COLOR_PER_VERTEX);
        const int shade = add_shade(flat_shade());
        const uint64_t V = pos.rows;
        auto at = [&](uint64_t v) {
            const float z = pos.width == 3 ? pos.f(v, 2) : 0.0f;
            return std::array<float, 3>{pos.f(v, 0), pos.f(v, 1), z};
        };
        auto colour = [&](uint64_t row, float out[4]) { unpack(item_rgba(p, src.any() ? src.at(row) : 0u, cmap), out); };
        auto draw_seg = [&](uint32_t item, std::array<float, 3> a, std::array<float, 3> b, uint64_t ra, uint64_t rb) {
            if (is_nan_bits(a[0]) || is_nan_bits(a[1]) || is_nan_bits(a[2]) || is_nan_bits(b[0]) || is_nan_bits(b[1]) ||
                is_nan_bits(b[2]))
                return;
            float ca[4], cb[4];
            colour(ra, ca);
            colour(rb, cb);
            segment(draw, item, mul(p.mvp, a[0], a[1], a[2]), mul(p.mvp, b[0], b[1], b[2]), ca, cb, p.size_px, shade, st);
        };
        if (mode == CALIPER_NET_LINES_EDGES) {
            const Rows pairs = rows(&u, "indices");
            const bool per_edge = per == CALIPER_NET_COLOR_PER_EDGE;
            for (uint64_t e = 0; e < pairs.rows; ++e) {
                const uint64_t i0 = std::min<uint64_t>(pairs.u(2 * e), V - 1), i1 = std::min<uint64_t>(pairs.u(2 * e + 1), V - 1);
                draw_seg(uint32_t(e), at(i0), at(i1), per_edge ? e : i0, per_edge ? e : i1);
            }
        } else if (mode == CALIPER_NET_LINES_STRIPS) {
            const Rows lens = rows(&u, "strip_lengths");
            const bool per_strip = per == CALIPER_NET_COLOR_PER_STRIP;
            uint64_t v = 0;
            for (uint64_t s = 0; s < lens.rows; ++s) {
                const uint64_t l = uint64_t(std::max(int32_t(lens.u(s)), 0));
                for (uint64_t k = 0; k + 1 < l; ++k)
                    draw_seg(uint32_t(v + k), at(v + k), at(v + k + 1), per_strip ? s : v + k, per_strip ? s : v + k + 1);
                v += l;
            }
        } else {
            for (uint64_t k = 0; 2 * k + 1 < V; ++k) draw_seg(uint32_t(k), at(2 * k), at(2 * k + 1), 2 * k, 2 * k + 1);
        }
    }

    // A unit geometry of the scene core's constant streams (frustums, axes): segments with colours.
    void unit_segments(int draw, const sh::PrimParams& p, uint32_t offset, uint32_t count, const uint32_t* colors,
                       const Reference::State& st) {
        uint8_t g[caliper_scene::StaticGeometry::kBytes];
        caliper_scene::StaticGeometry::fill(g);
        const int shade = add_shade(flat_shade());
        for (uint32_t k = 0; k < count; ++k) {
            float a[3], b[3];
            std::memcpy(a, g + offset + 24 * k, 12);
            std::memcpy(b, g + offset + 24 * k + 12, 12);
            float ca[4], cb[4];
            unpack(colors ? colors[2 * k] : p.flat_rgba, ca);
            unpack(colors ? colors[2 * k + 1] : p.flat_rgba, cb);
            segment(draw, k, mul(p.mvp, a[0], a[1], a[2]), mul(p.mvp, b[0], b[1], b[2]), ca, cb, p.size_px, shade, st);
        }
    }

    void arrows(int draw, const sh::PrimParams& p, const sh::Frame& fb, const wire::SceneUpdate& u,
                const Reference::State& st) {
        const Rows o = rows(&u, "origins"), vec = rows(&u, "vectors");
        const Source src = source(&u);
        const uint32_t cmap = uint32_t(num(u, "colormap", 0));
        const int shade = add_shade(flat_shade());
        const float W = fb.viewport[0], H = fb.viewport[1];
        for (uint64_t i = 0; i < o.rows; ++i) {
            const float ax = o.f(i, 0), ay = o.f(i, 1), az = o.width == 3 ? o.f(i, 2) : 0.0f;
            const float tx = ax + vec.f(i, 0), ty = ay + vec.f(i, 1), tz = az + (vec.width == 3 ? vec.f(i, 2) : 0.0f);
            if (is_nan_bits(ax) || is_nan_bits(ay) || is_nan_bits(az) || is_nan_bits(tx) || is_nan_bits(ty) || is_nan_bits(tz))
                continue;
            float col[4];
            unpack(item_rgba(p, src.any() ? src.at(i) : 0u, cmap), col);
            F4 a = mul(p.mvp, ax, ay, az), b = mul(p.mvp, tx, ty, tz);
            if (!clip_near(a, b)) continue;
            const float hx = 0.5f * W, hy = 0.5f * H;
            const float sax = a.x / a.w * hx, say = a.y / a.w * hy, sbx = b.x / b.w * hx, sby = b.y / b.w * hy;
            const float dx = sbx - sax, dy = sby - say, len = std::sqrt(dx * dx + dy * dy);
            float ux = 1, uy = 0;
            if (len > 1e-6f) {
                ux = dx / len;
                uy = dy / len;
            }
            const float w = std::max(p.size_px, 1.0f);
            const float head = 0.2f * len;
            const float bx = sbx - ux * head, by = sby - uy * head;
            const uint32_t ident = Reference::make_id(draw, uint32_t(i));
            // The shaft: a segment to the head's base, w wide.
            RVert v[6];
            bool ok = true;
            for (int c = 0; c < 6 && ok; ++c) {
                F4 out;
                ok = segment_corner(a, F4{bx / hx * b.w, by / hy * b.w, b.z, b.w}, c, w, W, H, &out);
                v[c] = to_screen(out, ref.W, ref.H);
                std::memcpy(v[c].c, col, 16);
            }
            if (ok) quad(v, ident, shade, st);
            // The head: a flat triangle in screen space.
            const float side = std::max(1.5f * w, 0.4f * head);
            const float sx = -uy * side, sy = ux * side;
            const float hxs[3] = {sbx, bx + sx, bx - sx}, hys[3] = {sby, by + sy, by - sy};
            RVert t[3];
            for (int k = 0; k < 3; ++k) {
                t[k] = to_screen(F4{hxs[k] / hx * b.w, hys[k] / hy * b.w, b.z, b.w}, ref.W, ref.H);
                std::memcpy(t[k].c, col, 16);
            }
            ref.triangle(t, ident, shade, st);
        }
    }

    // box_pose of boxes.wgsl.
    static void box_pose(const float c[3], const float h[3], const float r[4], float m[16]) {
        float qx = 0, qy = 0, qz = 0, qw = 1;
        const float n2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
        if (n2 > 0) {
            const float k = 1.0f / std::sqrt(n2);
            qx = r[0] * k;
            qy = r[1] * k;
            qz = r[2] * k;
            qw = r[3] * k;
        }
        const float c0[3] = {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)};
        const float c1[3] = {2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qw * qx)};
        const float c2[3] = {2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)};
        for (int k = 0; k < 3; ++k) {
            m[k] = c0[k] * h[0];
            m[4 + k] = c1[k] * h[1];
            m[8 + k] = c2[k] * h[2];
            m[12 + k] = c[k];
        }
        m[3] = m[7] = m[11] = 0;
        m[15] = 1;
    }
    static void on_axis(uint32_t axis, float along, float s, float t, float out[3]) {
        if (axis == 0) { out[0] = along; out[1] = s; out[2] = t; }
        else if (axis == 1) { out[0] = s; out[1] = along; out[2] = t; }
        else { out[0] = s; out[1] = t; out[2] = along; }
    }

    void boxes(int draw, bool faces, const sh::PrimParams& p, const caliper_gfx::DrawRecord& r, const wire::SceneUpdate& u,
               const Reference::State& st) {
        const Rows c = rows(&u, "centers"), h = rows(&u, "half_sizes"), rot = rows(&u, "rotations");
        const Source src = source(&u);
        const int shade = add_shade(flat_shade());
        for (uint64_t i = 0; i < c.rows; ++i) {
            const float cc[3] = {c.f(i, 0), c.f(i, 1), c.width == 3 ? c.f(i, 2) : 0.0f};
            const float hh[3] = {h.f(i, 0), h.f(i, 1), h.width == 3 ? h.f(i, 2) : 0.0f};
            const float rr[4] = {rot ? rot.f(i, 0) : 0.0f, rot ? rot.f(i, 1) : 0.0f, rot ? rot.f(i, 2) : 0.0f,
                                 rot ? rot.f(i, 3) : 1.0f};
            float pose[16];
            box_pose(cc, hh, rr, pose);
            float col[4];
            unpack(item_rgba(p, src.any() ? src.at(i) : 0u, 0), col);
            const uint32_t ident = Reference::make_id(draw, uint32_t(i));
            auto clip_of = [&](const float l[3]) {
                const F4 w = mul(pose, l[0], l[1], l[2]);
                return mul(p.mvp, w.x, w.y, w.z);   // mvp * (pose * v), never folded
            };
            if (!faces) {
                for (uint32_t e = 0; e < r.count / 6; ++e) {
                    uint32_t axis = (e % 4) / 2;
                    const float s = (e & 1u) ? 1.0f : -1.0f;
                    float t = e >= 4 ? 1.0f : -1.0f;
                    if (e >= 8) {
                        axis = 2;
                        t = (e & 2u) ? 1.0f : -1.0f;
                    }
                    float la[3], lb[3];
                    on_axis(axis, -1, s, t, la);
                    on_axis(axis, 1, s, t, lb);
                    RVert v[6];
                    bool ok = true;
                    for (int k = 0; k < 6 && ok; ++k) {
                        F4 out;
                        ok = segment_corner(clip_of(la), clip_of(lb), k, p.size_px, float(ref.W), float(ref.H), &out);
                        v[k] = to_screen(out, ref.W, ref.H);
                        std::memcpy(v[k].c, col, 16);
                    }
                    if (ok) quad(v, ident, shade, st);
                }
            } else {
                float fc[4] = {col[0], col[1], col[2], col[3] * 0.25f};   // kFaceAlpha
                const uint32_t order[3] = {2, 0, 1};
                for (uint32_t f = 0; f < r.count / 6; ++f) {
                    const uint32_t axis = order[f / 2];
                    const float sgn = (f & 1u) ? -1.0f : 1.0f;
                    RVert v[6];
                    for (int k = 0; k < 6; ++k) {
                        float l[3];
                        on_axis(axis, sgn, kQuad[k][0] * 2 - 1, kQuad[k][1], l);
                        v[k] = to_screen(clip_of(l), ref.W, ref.H);
                        std::memcpy(v[k].c, fc, 16);
                    }
                    quad(v, ident, shade, st);
                }
            }
        }
    }

    // A texture as the image's draw reads it: RGBA8, R8 or r32uint bits, built from the message.
    struct Texels {
        uint32_t w = 0, h = 0;
        int kind = 0;                      // 0 RGBA8, 1 R8 (grey), 2 bits
        std::vector<uint32_t> v;           // RGBA8 packed, grey in the low byte, or the bits
        uint32_t at(uint32_t x, uint32_t y) const { return v[size_t(y) * w + x]; }
    };
    static Texels texels(const wire::SceneUpdate& u, const char* section) {
        Texels t;
        const wire::Section* s = u.array(section);
        if (!s) return t;
        t.h = uint32_t(s->shape[0]);
        t.w = uint32_t(s->shape[1]);
        t.v.resize(size_t(t.w) * t.h);
        const std::string form = str(u, "form", CALIPER_NET_FORM_COLOR);
        if (std::string(section) == "texture" || (form == CALIPER_NET_FORM_COLOR && s->shape.size() == 3)) {
            std::memcpy(t.v.data(), s->data, t.v.size() * 4);
        } else if (form == CALIPER_NET_FORM_COLOR) {
            t.kind = 1;
            for (size_t i = 0; i < t.v.size(); ++i) t.v[i] = s->data[i];
        } else if (form == CALIPER_NET_FORM_DEPTH) {
            t.kind = 2;
            const bool u16 = s->dtype == CALIPER_NET_DTYPE_U16;
            const float scale = float(num(u, "scale", u16 ? 0.001 : 1.0));
            for (size_t i = 0; i < t.v.size(); ++i) {
                float d;
                if (u16) {
                    uint16_t x;
                    std::memcpy(&x, s->data + 2 * i, 2);
                    d = float(x) * scale;
                } else {
                    std::memcpy(&d, s->data + 4 * i, 4);
                    if (scale != 1.0f) d = d * scale;
                }
                t.v[i] = bits_of(d);
            }
        } else {
            t.kind = 2;
            for (size_t i = 0; i < t.v.size(); ++i) {
                if (s->dtype == CALIPER_NET_DTYPE_U16) {
                    uint16_t x;
                    std::memcpy(&x, s->data + 2 * i, 2);
                    t.v[i] = x;
                } else {
                    t.v[i] = s->data[i];
                }
            }
        }
        return t;
    }
    // textureSample with the linear sampler, clamped to the edge, without mipmaps.
    static void bilinear(const Texels& t, double u, double v, float out[4]) {
        const double x = u * t.w - 0.5, y = v * t.h - 0.5;
        const double fx = std::floor(x), fy = std::floor(y);
        const double ax = x - fx, ay = y - fy;
        auto texel = [&](double xi, double yi, float c[4]) {
            const uint32_t cx = uint32_t(std::clamp(xi, 0.0, double(t.w - 1)));
            const uint32_t cy = uint32_t(std::clamp(yi, 0.0, double(t.h - 1)));
            const uint32_t p = t.at(cx, cy);
            if (t.kind == 1) {
                const float g = float(p & 0xffu) / 255.0f;
                c[0] = g;
                c[1] = c[2] = 0;
                c[3] = 1;
            } else {
                unpack(p, c);
            }
        };
        float c00[4], c10[4], c01[4], c11[4];
        texel(fx, fy, c00);
        texel(fx + 1, fy, c10);
        texel(fx, fy + 1, c01);
        texel(fx + 1, fy + 1, c11);
        for (int k = 0; k < 4; ++k)
            out[k] = float((c00[k] * (1 - ax) + c10[k] * ax) * (1 - ay) + (c01[k] * (1 - ax) + c11[k] * ax) * ay);
    }
    static uint32_t texel_index(const Texels& t, double u, double v) {   // texel_of of image.wgsl
        const uint32_t x = std::min(uint32_t(std::max(0.0f, float(u) * float(t.w))), t.w - 1);
        const uint32_t y = std::min(uint32_t(std::max(0.0f, float(v) * float(t.h))), t.h - 1);
        return y * t.w + x;
    }

    void image(int draw, const sh::PrimParams& p, const wire::SceneUpdate& u, const Reference::State& st) {
        auto tex = std::make_shared<Texels>(texels(u, "pixels"));
        if (tex->w == 0) return;
        const uint32_t cmap = uint32_t(num(u, "colormap", 0));
        float tint[4];
        unpack(p.flat_rgba, tint);
        const bool nearest = tex->kind == 2;
        // A texel looked up by nearest is its own item, so that texel boundaries count as edges.
        const std::string form = str(u, "form", CALIPER_NET_FORM_COLOR);
        const bool value = form == CALIPER_NET_FORM_DEPTH;
        auto uv_at = [](const RVert (&v)[3], const double q[3], double* s, double* t) {
            *s = q[0] * v[0].uv[0] + q[1] * v[1].uv[0] + q[2] * v[2].uv[0];
            *t = q[0] * v[0].uv[1] + q[1] * v[1].uv[1] + q[2] * v[2].uv[1];
        };
        RVert v[6];
        for (int k = 0; k < 6; ++k) {
            const float s = kQuad[k][0], t = kQuad[k][1] * 0.5f + 0.5f;
            v[k] = to_screen(mul(p.mvp, s, t, 0.0f), ref.W, ref.H);
            v[k].uv[0] = s;
            v[k].uv[1] = t;
            std::memcpy(v[k].c, tint, 16);
        }
        if (!nearest) {
            const bool grey = tex->kind == 1;
            const int shade = add_shade([tex, grey, uv_at](const RVert (&vv)[3], const double q[3], float out[4]) {
                double s, t;
                uv_at(vv, q, &s, &t);
                float c[4];
                bilinear(*tex, s, t, c);
                if (grey) c[1] = c[2] = c[0], c[3] = 1;
                for (int k = 0; k < 4; ++k) out[k] = c[k] * vv[0].c[k];
                return true;
            });
            ref.tol_of_ident[Reference::make_id(draw, 0)] = 1;   // bilinear
            quad(v, Reference::make_id(draw, 0), shade, st);
            return;
        }
        // Nearest: the quad is split per texel by the identity, so each pixel takes its texel's id.
        const int shade = add_shade([this, tex, value, cmap, p, uv_at](const RVert (&vv)[3], const double q[3], float out[4]) {
            double s, t;
            uv_at(vv, q, &s, &t);
            const uint32_t bits = tex->v[texel_index(*tex, s, t)];
            float c[4];
            if (value) {
                sh::PrimParams pp = p;
                pp.color_mode = 1;
                unpack(item_rgba(pp, bits, cmap), c);
            } else {
                unpack(bits < 65536 ? parts.class_texels[bits] : caliper_scene::kDefaultRgba, c);
            }
            out[0] = c[0];
            out[1] = c[1];
            out[2] = c[2];
            out[3] = c[3] * vv[0].c[3];
            return true;
        });
        // Coverage is the quad's; the identity per pixel is its texel's, set after drawing.
        const size_t before = ref.tris.size();
        quad(v, Reference::make_id(draw, 0), shade, st);
        for (int j = 0; j < ref.H; ++j)
            for (int i = 0; i < ref.W; ++i) {
                const size_t k = size_t(j) * ref.W + i;
                if (ref.id[k] != Reference::make_id(draw, 0)) continue;
                for (size_t ti = before; ti < ref.tris.size(); ++ti) {
                    double q[3], z, mw;
                    bool inside;
                    if (!weights(ref.tris[ti].v, i + 0.5, j + 0.5, q, &z, &inside, &mw) || !inside) continue;
                    double s, t;
                    uv_at(ref.tris[ti].v, q, &s, &t);
                    const uint32_t texel = texel_index(*tex, s, t);
                    if (ref.base[k] == ref.id[k]) ref.base[k] = Reference::make_id(draw, texel);
                    ref.id[k] = Reference::make_id(draw, texel);
                    ref.tris_of[ref.id[k]] = ref.tris_of[Reference::make_id(draw, 0)];
                    break;
                }
            }
    }

    void depth_cloud(int draw, const sh::PrimParams& p, const sh::Frame& fb, const wire::SceneUpdate& u,
                     const Reference::State& st) {
        const Texels tex = texels(u, "pixels");
        const uint32_t cmap = uint32_t(num(u, "colormap", 0));
        const int shade = add_shade([shape = p.point_shape](const RVert (&v)[3], const double q[3], float out[4]) {
            const double s = q[0] * v[0].uv[0] + q[1] * v[1].uv[0] + q[2] * v[2].uv[0];
            const double t = q[0] * v[0].uv[1] + q[1] * v[1].uv[1] + q[2] * v[2].uv[1];
            if (shape == 1 && s * s + t * t > 1.0) return false;
            std::memcpy(out, v[0].c, 16);
            return true;
        });
        for (uint32_t i = 0; i < tex.w * tex.h; ++i) {
            const uint32_t bits = tex.v[i];
            if ((bits >> 31) != 0 || bits == 0 || (bits & 0x7fffffffu) >= 0x7f800000u) continue;
            const float d = float_of(bits);
            const float x = (float(i % tex.w) + 0.5f) * d, y = (float(i / tex.w) + 0.5f) * d;
            const F4 c = mul(p.mvp, x, y, d);
            float col[4];
            unpack(p.color_mode == 1 ? item_rgba(p, bits, cmap) : p.flat_rgba, col);
            const float kx = (p.size_px / fb.viewport[0]) * c.w, ky = (p.size_px / fb.viewport[1]) * c.w;
            RVert v[6];
            for (int k = 0; k < 6; ++k) {
                const float cx = kQuad[k][0] * 2 - 1, cy = kQuad[k][1];
                v[k] = to_screen({c.x + cx * kx, c.y + cy * ky, c.z, c.w}, ref.W, ref.H);
                v[k].uv[0] = cx;
                v[k].uv[1] = cy;
                std::memcpy(v[k].c, col, 16);
            }
            quad(v, Reference::make_id(draw, i), shade, st);
        }
    }

    void mesh(int draw, const sh::PrimParams& p, const wire::SceneUpdate& u, const Reference::State& st) {
        const Rows pos = rows(&u, "positions"), idx = rows(&u, "indices"), nrm = rows(&u, "normals"), uvs = rows(&u, "uvs");
        const Source src = source(&u);
        const uint32_t cmap = uint32_t(num(u, "colormap", 0));
        const uint64_t V = pos.rows;
        const uint64_t T = idx ? idx.rows : V / 3;
        auto tex = std::make_shared<Texels>(uvs ? texels(u, "texture") : Texels{});
        // Each vertex's colour, lit by the headlight when the mesh has normals (vs_mesh).
        auto vertex_colour = [&](uint64_t vi, float out[4]) {
            unpack(item_rgba(p, src.any() ? src.at(vi) : 0u, cmap), out);
            if (p.shade_mode == 1) {
                float n[3] = {nrm.f(vi, 0), nrm.f(vi, 1), nrm.f(vi, 2)};
                const float ln = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                for (float& x : n) x /= ln;
                float vn[3];
                for (int k = 0; k < 3; ++k) vn[k] = n[0] * p.nmat0[k] + n[1] * p.nmat1[k] + n[2] * p.nmat2[k];
                const float lv = std::sqrt(vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]);
                const float lit = 0.30f + 0.70f * std::max(vn[2] / lv, 0.0f);
                for (int k = 0; k < 3; ++k) out[k] *= lit;
            }
        };
        const bool flat = !nrm && !uvs;
        const bool textured = bool(uvs);
        std::vector<std::array<uint32_t, 3>>& tv = mesh_vertices[draw];
        for (uint64_t t = 0; t < T; ++t) {
            RVert v[3];
            float pv[3][3];
            tv.push_back({});
            for (int k = 0; k < 3; ++k) {
                const uint64_t vi = idx ? std::min<uint64_t>(idx.u(3 * t + k), V - 1) : 3 * t + k;
                tv.back()[size_t(k)] = uint32_t(vi);
                for (int c = 0; c < 3; ++c) pv[k][c] = c < int(pos.width) ? pos.f(vi, uint32_t(c)) : 0.0f;
                v[k] = to_screen(mul(p.mvp, pv[k][0], pv[k][1], pv[k][2]), ref.W, ref.H);
                vertex_colour(vi, v[k].c);
                std::memcpy(v[k].o, pv[k], 12);
                if (textured) {
                    v[k].uv[0] = uvs.f(vi, 0);
                    v[k].uv[1] = uvs.f(vi, 1);
                }
            }
            Shade s;
            if (flat) {
                // fs_mesh_flat: the face's normal through the normal matrix, lit on both sides.
                const float e1[3] = {pv[1][0] - pv[0][0], pv[1][1] - pv[0][1], pv[1][2] - pv[0][2]};
                const float e2[3] = {pv[2][0] - pv[0][0], pv[2][1] - pv[0][1], pv[2][2] - pv[0][2]};
                const float n3[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
                float vn[3];
                for (int k = 0; k < 3; ++k) vn[k] = n3[0] * p.nmat0[k] + n3[1] * p.nmat1[k] + n3[2] * p.nmat2[k];
                const float len = std::sqrt(vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]);
                const float lit = 0.30f + 0.70f * std::fabs(vn[2] / len);
                s = [lit](const RVert (&vv)[3], const double q[3], float out[4]) {
                    for (int k = 0; k < 4; ++k) out[k] = float(q[0] * vv[0].c[k] + q[1] * vv[1].c[k] + q[2] * vv[2].c[k]);
                    for (int k = 0; k < 3; ++k) out[k] *= lit;
                    return true;
                };
            } else if (textured) {
                s = [tex](const RVert (&vv)[3], const double q[3], float out[4]) {
                    const double su = q[0] * vv[0].uv[0] + q[1] * vv[1].uv[0] + q[2] * vv[2].uv[0];
                    const double sv = q[0] * vv[0].uv[1] + q[1] * vv[1].uv[1] + q[2] * vv[2].uv[1];
                    float c[4];
                    bilinear(*tex, su, sv, c);
                    for (int k = 0; k < 3; ++k)
                        out[k] = c[k] * float(q[0] * vv[0].c[k] + q[1] * vv[1].c[k] + q[2] * vv[2].c[k]);
                    out[3] = c[3];
                    return true;
                };
            } else {
                s = flat_shade();
            }
            ref.triangle(v, Reference::make_id(draw, uint32_t(t)), add_shade(std::move(s)), st);
        }
    }

    // Rasterises every draw of the list in order.
    void run() {
        const auto recs = list.records();
        ref.tol_of_draw.clear();
        ref.blended_draw.clear();
        ref.writes.clear();
        for (size_t di = 0; di < recs.size; ++di) {
            const caliper_gfx::DrawRecord& r = recs[di];
            sh::PrimParams p;
            sh::Frame fb;
            std::memcpy(&p, list.arena().data + r.dynamic_offsets[1], sizeof p);
            std::memcpy(&fb, list.arena().data + r.dynamic_offsets[0], sizeof fb);
            const DrawMapEntry* e = map.find(r.pick_id);
            if (!e) {
                problems.push_back("draw " + std::to_string(di) + " is not in the draw map");
                continue;
            }
            const int draw = int(di);
            Reference::State st;
            st.depth_test = e->depth_test;
            st.depth_write = e->depth_write;
            st.blend = e->translucent;
            if (pick) {
                st.blend = false;
                st.depth_write = st.depth_test;
            }
            if (r.scissor[2] != 0xffff || r.scissor[3] != 0xffff)
                for (int k = 0; k < 4; ++k) st.scissor[k] = r.scissor[k];
            const bool lit = e->primitive == Primitive::Mesh && e->part == DrawPart::Batch;
            ref.tol_of_draw.push_back(lit ? 2 : 0);
            ref.blended_draw.push_back(st.blend);
            ref.writes.push_back(0);
            const std::string path = tree.path(e->entity);
            const char* part = e->part == DrawPart::Frustum || e->part == DrawPart::Axes
                                   ? nullptr
                                   : e->part == DrawPart::DepthCloud ? CALIPER_NET_PART_IMAGE
                                                                     : caliper_scene::primitive_name(e->primitive);
            const wire::SceneUpdate* u = part ? parts.find(path, part) : nullptr;
            if (part && !u) {
                problems.push_back("draw " + std::to_string(di) + " (" + path + " " + part + ") has no message");
                continue;
            }
            switch (e->part) {
                case DrawPart::Frustum: unit_segments(draw, p, caliper_scene::StaticGeometry::kFrustum, 8, nullptr, st); break;
                case DrawPart::Axes: {
                    uint8_t g[caliper_scene::StaticGeometry::kBytes];
                    caliper_scene::StaticGeometry::fill(g);
                    uint32_t colors[6];
                    std::memcpy(colors, g + caliper_scene::StaticGeometry::kAxesColors, sizeof colors);
                    unit_segments(draw, p, caliper_scene::StaticGeometry::kAxes, 3, colors, st);
                    break;
                }
                case DrawPart::DepthCloud: depth_cloud(draw, p, fb, *u, st); break;
                case DrawPart::BoxFaces: boxes(draw, true, p, r, *u, st); break;
                case DrawPart::Batch:
                    switch (e->primitive) {
                        case Primitive::Points: points(draw, p, fb, *u, st); break;
                        case Primitive::Lines: lines(draw, p, *u, st); break;
                        case Primitive::Arrows: arrows(draw, p, fb, *u, st); break;
                        case Primitive::Boxes: boxes(draw, false, p, r, *u, st); break;
                        case Primitive::Mesh: mesh(draw, p, *u, st); break;
                        case Primitive::Image: image(draw, p, *u, st); break;
                    }
                    break;
            }
        }
    }
};

// ---- the comparison --------------------------------------------------------------------------------------

struct Comparison {
    size_t pixels = 0;
    size_t interior = 0, edge = 0;      // pixels whose 3 x 3 neighbourhood shows one identity, or several
    size_t exact = 0;                   // pixels equal to the reference byte for byte
    size_t differ_edge = 0;             // differ, next to an edge, and take a colour a side gives
    size_t differ_lit = 0;              // differ, interior, lit or flat-shaded faces within 2 levels
    size_t differ_soft = 0;             // differ, interior, blended, bilinear or interpolated within 1 level
    size_t failed = 0;                  // outside the parity rules
    int max_interior = 0;               // the largest difference on an interior pixel
    std::map<int, size_t> interior_by_draw;   // -1: the clear
    std::string first;

    // One sentence: how many pixels differ and why.
    std::string sentence(const std::string& name) const {
        char t[512];
        std::snprintf(t, sizeof t,
                      "%s: %zu of %zu pixels differ from the CPU reference (%zu next to an edge took a colour of a side "
                      "beside them, %zu lit faces within 2 levels, %zu blended, bilinear or interpolated pixels within 1 "
                      "level), and %zu are outside the parity rules",
                      name.c_str(), pixels - exact, pixels, differ_edge, differ_lit, differ_soft, failed);
        return t;
    }
};

inline Comparison compare(const Reference& ref, const std::vector<uint8_t>& got) {
    Comparison c;
    for (int j = 0; j < ref.H; ++j)
        for (int i = 0; i < ref.W; ++i) {
            const size_t k = size_t(j) * ref.W + i;
            ++c.pixels;
            uint32_t g;
            std::memcpy(&g, &got[4 * k], 4);
            // The identities around the pixel, of the top write and of the opaque one under it, so
            // that an edge under a translucent layer counts as an edge too.
            uint32_t ids[9];
            uint64_t layers[9];
            int n = 0, nl = 0;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    const int x = std::clamp(i + di, 0, ref.W - 1), y = std::clamp(j + dj, 0, ref.H - 1);
                    const uint32_t v = ref.id[size_t(y) * ref.W + x];
                    const uint64_t l = uint64_t(v) << 32 | ref.base[size_t(y) * ref.W + x];
                    if (std::find(ids, ids + n, v) == ids + n) ids[n++] = v;
                    if (std::find(layers, layers + nl, l) == layers + nl) layers[nl++] = l;
                }
            const int d = max_diff(g, ref.rgba[k]);
            if (d == 0) ++c.exact;
            bool ok;
            if (nl == 1) {
                ++c.interior;
                ++c.interior_by_draw[ids[0] ? Reference::draw_of(ids[0]) : -1];
                c.max_interior = std::max(c.max_interior, d);
                ok = d <= ref.tol[k];
                const bool lit = ids[0] != 0 && ref.tol[k] >= 2 && !ref.blended_draw[size_t(Reference::draw_of(ids[0]))];
                if (ok && d > 0) ++(lit ? c.differ_lit : c.differ_soft);
            } else {
                ++c.edge;
                ok = d <= ref.tol[k];
                // Either side: the colour a candidate gives here, or the reference beside this pixel.
                for (int m = 0; m < n && !ok; ++m) {
                    uint32_t e;
                    if (ref.evaluate(ids[m], i, j, &e)) ok = max_diff(g, e) <= ref.tolerance(ids[m]);
                }
                for (int dj = -1; dj <= 1 && !ok; ++dj)
                    for (int di = -1; di <= 1 && !ok; ++di) {
                        const int x = std::clamp(i + di, 0, ref.W - 1), y = std::clamp(j + dj, 0, ref.H - 1);
                        const size_t kk = size_t(y) * ref.W + x;
                        ok = max_diff(g, ref.rgba[kk]) <= std::max(ref.tol[kk], ref.tol[k]);
                    }
                // Or the far side of the top primitive's edge, which it hides here and no neighbour need show.
                if (!ok) ok = max_diff(g, ref.behind[k]) <= std::max(int(ref.tol[k]), ref.tolerance(ref.behind_id[k]));
                if (ok && d > 0) ++c.differ_edge;
            }
            if (!ok) {
                if (c.failed == 0) {
                    char buf[200];
                    std::snprintf(buf, sizeof buf,
                                  "pixel (%d, %d): got %08x, reference %08x (identity %08x over %08x, %d candidates)", i, j,
                                  g, ref.rgba[k], ref.id[k], ref.base[k], n);
                    c.first = buf;
                }
                ++c.failed;
            }
        }
    return c;
}

}  // namespace draw_reference
