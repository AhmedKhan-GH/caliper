// primitive_fixtures.h: one scene per primitive of a scene, as wire protocol 3.2 messages, for the
// scene view's rows on Metal (tests/gfx/gfx_scene_view.cpp) and the browser module's scene
// self-test on WebGPU (web/wasm/src/scene_selftest.cpp), which draw each at 256 x 256 and compare it
// with the CPU reference of draw_reference.h under web geometry's parity rules. It depends on no
// test framework.
//
//   arrows    arrows with per-arrow colours and with values through magma, over a floor of segments
//   boxes     wire boxes, one rotated, coloured by a class table over the palette, a filled box,
//             and a chain of boxes whose entity transforms compose down the tree
//   edges     index-pair edges of a small graph with per-edge values (one index past the last
//             vertex, which the clamp draws as the last vertex) and with per-vertex colours, and
//             the graph's nodes as points coloured by class
//   mesh      a mesh lit by the headlight from its normals, and a textured quad
//   cameras   two cameras on posed rigs, drawn as frustums: one with a colour image filling its
//             image plane, one with a depth image and its back-projected cloud
//   through   the same rigs looked through the first camera: the room projected, and the camera's
//             pixel space (a translucent image over a quarter of it, and 2-D points, boxes and arrows
//             over the image) flat over it, letterboxed to the camera's aspect
//   images    a 2-D scene of a colour, a grey, a u16 depth and a segmentation image, placed by
//             entity transforms, with 2-D boxes and arrows
#pragma once

#include "net/wire.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace primitive_fixtures {

namespace wire = caliper_net::wire;
using caliper_net::Json;

inline constexpr uint32_t kSize = 256;
inline constexpr uint32_t kClear = 0xff201810u;   // a dark background, opaque

struct Fixture {
    std::string name;
    wire::SceneView info;
    std::vector<wire::Message> messages;      // their sections point into `store`
    std::string look_through;                 // a camera entity to look through, or ""
    std::shared_ptr<std::deque<std::vector<uint8_t>>> store = std::make_shared<std::deque<std::vector<uint8_t>>>();

    // Bytes the fixture owns, which a section may point into.
    template <class T> const T* own(const std::vector<T>& v) {
        store->emplace_back(reinterpret_cast<const uint8_t*>(v.data()),
                            reinterpret_cast<const uint8_t*>(v.data()) + v.size() * sizeof(T));
        return reinterpret_cast<const T*>(store->back().data());
    }
    template <class T> wire::Section section(const char* name, int64_t dtype, std::vector<int64_t> shape,
                                             const std::vector<T>& v) {
        return wire::Section::make(name, dtype, std::move(shape), own(v), v.size() * sizeof(T));
    }

    uint64_t seq = 0;
    wire::SceneUpdate update(const std::string& path, const char* part, bool is_static = false) {
        wire::SceneUpdate u;
        u.sub = 1;
        u.view = name;
        u.seq = ++seq;
        u.stamp_ns = 1000 + seq;
        u.path = path;
        u.part = part;
        u.is_static = is_static;
        return u;
    }
    void transform(const std::string& path, const std::vector<float>& m16, bool is_static = false) {
        wire::SceneTransforms t;
        t.sub = 1;
        t.view = name;
        t.seq = ++seq;
        t.paths = {path};
        t.is_static = is_static;
        t.matrices = section("matrices", CALIPER_NET_DTYPE_F32, {1, 16}, m16);
        messages.push_back(t);
    }
};

inline Json rgba(uint32_t c) {
    Json a = Json::array();
    for (int k = 0; k < 4; ++k) a.push(Json(int((c >> (8 * k)) & 0xffu)));
    return a;
}
inline uint32_t RGBA(uint32_t r, uint32_t g, uint32_t b, uint32_t a = 255) { return r | g << 8 | b << 16 | a << 24; }

// Column-major 4 x 4 matrices.
inline std::vector<float> translate(float x, float y, float z) { return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1}; }
inline std::vector<float> rotate_z(float angle, float x = 0, float y = 0, float z = 0) {
    const float c = std::cos(angle), s = std::sin(angle);
    return {c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, x, y, z, 1};
}
// A pose whose camera frame (x right, y down, z forward) looks along `forward` from `eye`, with
// world +z up: the columns are right, down and forward.
inline std::vector<float> look_from(const float eye[3], const float forward[3]) {
    const float up[3] = {0, 0, 1};
    float f[3] = {forward[0], forward[1], forward[2]};
    const float lf = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    for (float& x : f) x /= lf;
    float r[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]};
    const float lr = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    for (float& x : r) x /= lr;
    const float d[3] = {f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]};
    return {r[0], r[1], r[2], 0, d[0], d[1], d[2], 0, f[0], f[1], f[2], 0, eye[0], eye[1], eye[2], 1};
}

inline Json classes_json(const std::vector<std::pair<int, uint32_t>>& entries, const std::vector<std::string>& labels) {
    Json a = Json::array();
    for (size_t i = 0; i < entries.size(); ++i) {
        Json e = Json::object();
        e.set("id", Json(entries[i].first));
        e.set("label", Json(labels[i]));
        e.set("color", rgba(entries[i].second));
        a.push(e);
    }
    return a;
}

inline Json stats_from(const std::vector<float>& rows, uint32_t dims, double vmin = 0, double vmax = 0) {
    double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t i = 0; i + dims - 1 < rows.size(); i += dims)
        for (uint32_t k = 0; k < dims; ++k) {
            lo[k] = std::fmin(lo[k], rows[i + k]);
            hi[k] = std::fmax(hi[k], rows[i + k]);
        }
    Json s = Json::array();
    s.push(Json(vmin));
    s.push(Json(vmax));
    for (uint32_t k = 0; k < 3; ++k) {
        s.push(Json(k < dims ? lo[k] : 0.0));
        s.push(Json(k < dims ? hi[k] : 0.0));
    }
    return s;
}

// A floor of segments, a square with a cross, at z = 0 from -s to s.
inline std::vector<float> floor_segments(float s) {
    return {-s, -s, 0, s, -s, 0, s, -s, 0, s, s, 0, s, s, 0, -s, s, 0, -s, s, 0, -s, -s, 0,
            -s, 0, 0, s, 0, 0, 0, -s, 0, 0, s, 0};
}

inline Fixture arrows() {
    Fixture f;
    f.name = "arrows";
    f.info = {CALIPER_NET_SPACE_3D, "+z", "m"};
    std::vector<float> fl = floor_segments(1.2f);
    wire::SceneUpdate floor = f.update("", CALIPER_NET_PART_LINES, true);
    floor.params.set("mode", CALIPER_NET_LINES_SEGMENTS);
    floor.params.set("color", rgba(RGBA(90, 90, 100)));
    floor.params.set("width", Json(1.0));
    floor.arrays.push_back(f.section("positions", CALIPER_NET_DTYPE_F32, {int64_t(fl.size() / 3), 3}, fl));
    f.messages.push_back(floor);

    // A ring of five arrows pointing out and up, each with its own colour.
    std::vector<float> o, v;
    std::vector<uint32_t> c = {RGBA(230, 80, 60), RGBA(240, 200, 40), RGBA(60, 200, 90), RGBA(60, 140, 240),
                               RGBA(200, 90, 230)};
    for (int i = 0; i < 5; ++i) {
        const float a = float(i) * 2.0f * 3.14159265f / 5.0f;
        o.insert(o.end(), {0.4f * std::cos(a), 0.4f * std::sin(a), 0.05f});
        v.insert(v.end(), {0.6f * std::cos(a), 0.6f * std::sin(a), 0.5f});
    }
    wire::SceneUpdate ring = f.update("ring", CALIPER_NET_PART_ARROWS);
    ring.params.set("width", Json(3.0));
    ring.arrays.push_back(f.section("origins", CALIPER_NET_DTYPE_F32, {5, 3}, o));
    ring.arrays.push_back(f.section("vectors", CALIPER_NET_DTYPE_F32, {5, 3}, v));
    ring.arrays.push_back(f.section("colors", CALIPER_NET_DTYPE_U8, {5, 4}, c));
    f.messages.push_back(ring);

    // Four upright arrows coloured through magma, values on both sides of the range's middle.
    std::vector<float> o2 = {-0.8f, -0.8f, 0, 0.8f, -0.8f, 0, 0.8f, 0.8f, 0, -0.8f, 0.8f, 0};
    std::vector<float> v2 = {0, 0, 0.9f, 0, 0, 0.6f, 0, 0, 0.3f, 0, 0, 0.75f};
    std::vector<float> val = {0.0f, 0.37f, 0.5f, 1.0f};
    wire::SceneUpdate up = f.update("posts", CALIPER_NET_PART_ARROWS);
    up.params.set("width", Json(4.0));
    up.params.set("colormap", Json(1));
    up.params.set("flags", Json(0));
    up.params.set("vmin", Json(0.0));
    up.params.set("vmax", Json(1.0));
    up.arrays.push_back(f.section("origins", CALIPER_NET_DTYPE_F32, {4, 3}, o2));
    up.arrays.push_back(f.section("vectors", CALIPER_NET_DTYPE_F32, {4, 3}, v2));
    up.arrays.push_back(f.section("values", CALIPER_NET_DTYPE_F32, {4}, val));
    f.messages.push_back(up);
    return f;
}

inline Fixture boxes() {
    Fixture f;
    f.name = "boxes";
    f.info = {CALIPER_NET_SPACE_3D, "+z", "m"};
    wire::SceneClasses cl;
    cl.sub = 1;
    cl.view = f.name;
    cl.seq = ++f.seq;
    cl.is_static = true;
    cl.classes = classes_json({{1, RGBA(230, 60, 50)}, {2, RGBA(50, 200, 80)}}, {"car", "person"});
    f.messages.push_back(cl);

    // Three wire boxes by class: 1 and 2 from the table, 7 from the palette; the third rotated.
    std::vector<float> cen = {-0.7f, 0, 0.3f, 0, 0, 0.3f, 0.7f, 0, 0.3f};
    std::vector<float> half = {0.2f, 0.25f, 0.3f, 0.15f, 0.15f, 0.3f, 0.25f, 0.2f, 0.3f};
    const float h = std::sin(0.35f), w = std::cos(0.35f);
    std::vector<float> rot = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, h, w};
    std::vector<int32_t> ids = {1, 2, 7};
    wire::SceneUpdate crates = f.update("crates", CALIPER_NET_PART_BOXES);
    crates.params.set("width", Json(3.0));
    crates.params.set("labels", [] {
        Json l = Json::array();
        for (const char* s : {"a", "b", "c"}) l.push(Json(s));
        return l;
    }());
    crates.arrays.push_back(f.section("centers", CALIPER_NET_DTYPE_F32, {3, 3}, cen));
    crates.arrays.push_back(f.section("half_sizes", CALIPER_NET_DTYPE_F32, {3, 3}, half));
    crates.arrays.push_back(f.section("rotations", CALIPER_NET_DTYPE_F32, {3, 4}, rot));
    crates.arrays.push_back(f.section("class_ids", CALIPER_NET_DTYPE_I32, {3}, ids));
    f.messages.push_back(crates);
    f.transform("crates", translate(0, 0.5f, 0));

    // A filled box: its edges opaque, its faces at a quarter of the colour's alpha.
    std::vector<float> zc = {0, -0.6f, 0.25f}, zh = {0.5f, 0.2f, 0.25f};
    wire::SceneUpdate zone = f.update("zone", CALIPER_NET_PART_BOXES);
    zone.params.set("color", rgba(RGBA(60, 140, 230)));
    zone.params.set("fill", Json(true));
    zone.params.set("width", Json(2.0));
    zone.arrays.push_back(f.section("centers", CALIPER_NET_DTYPE_F32, {1, 3}, zc));
    zone.arrays.push_back(f.section("half_sizes", CALIPER_NET_DTYPE_F32, {1, 3}, zh));
    f.messages.push_back(zone);

    // A chain of two links: each a box from its origin up, each placed at the end of its parent and
    // turned about z, so that the second's pose is the product of both transforms.
    std::vector<float> lc = {0, 0, 0.2f}, lh = {0.05f, 0.05f, 0.2f};
    for (const char* path : {"arm/link0", "arm/link0/link1"}) {
        wire::SceneUpdate link = f.update(path, CALIPER_NET_PART_BOXES, true);
        link.params.set("color", rgba(RGBA(220, 220, 220)));
        link.arrays.push_back(f.section("centers", CALIPER_NET_DTYPE_F32, {1, 3}, lc));
        link.arrays.push_back(f.section("half_sizes", CALIPER_NET_DTYPE_F32, {1, 3}, lh));
        f.messages.push_back(link);
    }
    f.transform("arm", translate(0.9f, -0.6f, 0), true);
    std::vector<float> tilt = {1, 0, 0, 0, 0, std::cos(0.6f), std::sin(0.6f), 0, 0, -std::sin(0.6f), std::cos(0.6f), 0, 0, 0, 0, 1};
    f.transform("arm/link0", tilt);
    std::vector<float> next = tilt;
    next[14] = 0.4f;   // at the end of link0
    f.transform("arm/link0/link1", next);
    return f;
}

inline Fixture edges() {
    Fixture f;
    f.name = "edges";
    f.info = {CALIPER_NET_SPACE_3D, "+y", "m"};
    // Six nodes on a hexagon and its centre's lift.
    std::vector<float> pos;
    for (int i = 0; i < 6; ++i) {
        const float a = float(i) * 3.14159265f / 3.0f;
        pos.insert(pos.end(), {std::cos(a), 0.2f * float(i % 2), std::sin(a)});
    }
    wire::SceneClasses cl;
    cl.sub = 1;
    cl.view = f.name;
    cl.seq = ++f.seq;
    cl.classes = classes_json({{0, RGBA(250, 250, 250)}}, {"hub"});
    f.messages.push_back(cl);
    std::vector<int32_t> node_class = {0, 1, 2, 3, 300, 70000};   // 300 and 70000: palette and the default
    wire::SceneUpdate nodes = f.update("graph/nodes", CALIPER_NET_PART_POINTS);
    nodes.params.set("radius", Json(0.06));
    nodes.arrays.push_back(f.section("positions", CALIPER_NET_DTYPE_F32, {6, 3}, pos));
    nodes.arrays.push_back(f.section("class_ids", CALIPER_NET_DTYPE_I32, {6}, node_class));
    f.messages.push_back(nodes);

    // Seven edges with per-edge values through viridis; the last pair names vertex 9, which draws
    // as the last vertex, and vertex -1, which does too.
    std::vector<int32_t> pairs = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 9, -1};
    std::vector<float> w = {0.0f, 0.15f, 0.3f, 0.5f, 0.65f, 0.85f, 1.0f};
    wire::SceneUpdate e1 = f.update("graph/ring", CALIPER_NET_PART_LINES);
    e1.params.set("mode", CALIPER_NET_LINES_EDGES);
    e1.params.set("color_per", CALIPER_NET_COLOR_PER_EDGE);
    e1.params.set("width", Json(3.0));
    e1.params.set("colormap", Json(0));
    e1.params.set("flags", Json(0));
    e1.params.set("vmin", Json(0.0));
    e1.params.set("vmax", Json(1.0));
    e1.arrays.push_back(f.section("positions", CALIPER_NET_DTYPE_F32, {6, 3}, pos));
    e1.arrays.push_back(f.section("indices", CALIPER_NET_DTYPE_I32, {7, 2}, pairs));
    e1.arrays.push_back(f.section("values", CALIPER_NET_DTYPE_F32, {7}, w));
    f.messages.push_back(e1);

    // Spokes from vertex 0 to the others, coloured per vertex: the colours blend along each edge.
    std::vector<int32_t> spokes = {0, 2, 0, 3, 0, 4};
    std::vector<uint32_t> vc = {RGBA(255, 255, 255), RGBA(255, 0, 0), RGBA(0, 255, 0), RGBA(0, 0, 255),
                                RGBA(255, 255, 0), RGBA(0, 255, 255)};
    wire::SceneUpdate e2 = f.update("graph/spokes", CALIPER_NET_PART_LINES);
    e2.params.set("mode", CALIPER_NET_LINES_EDGES);
    e2.params.set("width", Json(2.0));
    e2.arrays.push_back(f.section("positions", CALIPER_NET_DTYPE_F32, {6, 3}, pos));
    e2.arrays.push_back(f.section("indices", CALIPER_NET_DTYPE_I32, {3, 2}, spokes));
    e2.arrays.push_back(f.section("colors", CALIPER_NET_DTYPE_U8, {6, 4}, vc));
    f.messages.push_back(e2);
    return f;
}

inline Fixture mesh() {
    Fixture f;
    f.name = "mesh";
    f.info = {CALIPER_NET_SPACE_3D, "+y", "m"};
    // A ridge of four triangles with smooth normals, lit by the headlight.
    std::vector<float> pos = {-1, 0, -0.5f, 0, 0.6f, -0.5f, 1, 0, -0.5f, -1, 0, 0.5f, 0, 0.6f, 0.5f, 1, 0, 0.5f};
    std::vector<float> nrm = {-0.5f, 0.8f, 0, 0, 1, 0, 0.5f, 0.8f, 0, -0.5f, 0.8f, 0, 0, 1, 0, 0.5f, 0.8f, 0};
    std::vector<int32_t> tri = {0, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4};
    wire::SceneUpdate ridge = f.update("ridge", CALIPER_NET_PART_MESH);
    ridge.params.set("color", rgba(RGBA(200, 170, 120)));
    ridge.arrays.push_back(f.section("positions", CALIPER_NET_DTYPE_F32, {6, 3}, pos));
    ridge.arrays.push_back(f.section("indices", CALIPER_NET_DTYPE_I32, {4, 3}, tri));
    ridge.arrays.push_back(f.section("normals", CALIPER_NET_DTYPE_F32, {6, 3}, nrm));
    f.messages.push_back(ridge);
    // A textured quad standing behind it: an 8 x 8 texture of smooth colours.
    std::vector<float> qp = {-0.8f, 0.1f, -1, 0.8f, 0.1f, -1, 0.8f, 1.1f, -1, -0.8f, 0.1f, -1, 0.8f, 1.1f, -1, -0.8f, 1.1f, -1};
    std::vector<float> uv = {0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0};
    std::vector<uint32_t> tex(64);
    for (uint32_t y = 0; y < 8; ++y)
        for (uint32_t x = 0; x < 8; ++x) tex[y * 8 + x] = RGBA(60 + 20 * x, 80 + 18 * y, 200 - 10 * x, 255);
    wire::SceneUpdate card = f.update("card", CALIPER_NET_PART_MESH);
    card.arrays.push_back(f.section("positions", CALIPER_NET_DTYPE_F32, {6, 3}, qp));
    card.arrays.push_back(f.section("uvs", CALIPER_NET_DTYPE_F32, {6, 2}, uv));
    card.arrays.push_back(f.section("texture", CALIPER_NET_DTYPE_U8, {8, 8, 4}, tex));
    f.messages.push_back(card);
    return f;
}

// The rigs of the camera fixtures: a room of points and a floor, two posed cameras, a colour image
// on the first's image plane and a depth image with its cloud under the second.
inline void rigs(Fixture& f, bool pixel_space) {
    std::vector<float> fl = floor_segments(1.5f);
    wire::SceneUpdate floor = f.update("room/floor", CALIPER_NET_PART_LINES, true);
    floor.params.set("mode", CALIPER_NET_LINES_SEGMENTS);
    floor.params.set("color", rgba(RGBA(110, 110, 120)));
    floor.arrays.push_back(f.section("positions", CALIPER_NET_DTYPE_F32, {int64_t(fl.size() / 3), 3}, fl));
    f.messages.push_back(floor);
    std::vector<float> cen = {0.2f, 0.3f, 0.25f}, half = {0.25f, 0.25f, 0.25f};
    wire::SceneUpdate block = f.update("room/block", CALIPER_NET_PART_BOXES);
    block.params.set("color", rgba(RGBA(240, 180, 60)));
    block.params.set("width", Json(3.0));
    block.arrays.push_back(f.section("centers", CALIPER_NET_DTYPE_F32, {1, 3}, cen));
    block.arrays.push_back(f.section("half_sizes", CALIPER_NET_DTYPE_F32, {1, 3}, half));
    f.messages.push_back(block);

    const float eye1[3] = {-1.6f, -1.6f, 1.0f}, fwd1[3] = {1.6f, 1.9f, -0.75f};
    f.transform("rig", translate(0, 0, 0), true);
    f.transform("rig/left", look_from(eye1, fwd1), true);
    wire::SceneUpdate cam = f.update("rig/left", CALIPER_NET_PART_CAMERA, true);
    cam.params.set("fx", Json(36.0));
    cam.params.set("fy", Json(36.0));
    cam.params.set("cx", Json(32.0));
    cam.params.set("cy", Json(24.0));
    cam.params.set("width", Json(64));
    cam.params.set("height", Json(48));
    cam.params.set("convention", Json(CALIPER_NET_CONVENTION_RDF));
    cam.params.set("plane_distance", Json(0.4));
    f.messages.push_back(cam);
    // The colour image: smooth gradients, 32 x 24. Looked through, it covers the top-left quarter of
    // the camera's pixels, so that the sketch over it and the room beside it can be picked; on the
    // image plane in 3-D it fills the plane at 2 pixels a texel, because a texture drawn smaller than
    // about a texel a pixel takes colours from a sample position that differs by a fraction of a texel
    // between rasterisers, more than the parity rules' one level for a bilinear sample allows.
    std::vector<uint32_t> px(32 * 24);
    for (uint32_t y = 0; y < 24; ++y)
        for (uint32_t x = 0; x < 32; ++x) px[y * 32 + x] = RGBA(40 + 6 * x, 60 + 7 * y, 160, 255);
    wire::SceneUpdate img = f.update("rig/left/rgb", CALIPER_NET_PART_IMAGE);
    img.params.set("form", Json(CALIPER_NET_FORM_COLOR));
    if (pixel_space) img.params.set("opacity", Json(0.6));
    img.arrays.push_back(f.section("pixels", CALIPER_NET_DTYPE_U8, {24, 32, 4}, px));
    f.messages.push_back(img);
    if (pixel_space) f.transform("rig/left/rgb", translate(4, 4, 0));   // a 2-D transform in the camera's pixels
    else f.transform("rig/left/rgb", {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});   // 2 camera pixels a texel

    const float eye2[3] = {1.7f, -1.4f, 0.9f}, fwd2[3] = {-1.5f, 1.7f, -0.6f};
    f.transform("rig/right", look_from(eye2, fwd2), true);
    wire::SceneUpdate cam2 = f.update("rig/right", CALIPER_NET_PART_CAMERA, true);
    for (const char* k : {"fx", "fy"}) cam2.params.set(k, Json(10.0));
    cam2.params.set("cx", Json(8.0));
    cam2.params.set("cy", Json(6.0));
    cam2.params.set("width", Json(16));
    cam2.params.set("height", Json(12));
    cam2.params.set("convention", Json(CALIPER_NET_CONVENTION_RDF));
    cam2.params.set("plane_distance", Json(0.3));
    f.messages.push_back(cam2);
    // A depth image of 16 x 12 in millimetres (u16): a plane two units away tilting to three, whose
    // pixels all have different depths, so that no two discs of the cloud that overlap tie in depth;
    // one zero, which the cloud skips.
    std::vector<uint16_t> dm(16 * 12);
    for (uint32_t y = 0; y < 12; ++y)
        for (uint32_t x = 0; x < 16; ++x) dm[y * 16 + x] = uint16_t(x == 3 && y == 4 ? 0 : 2000 + 60 * x + 7 * y);
    wire::SceneUpdate depth = f.update("rig/right/depth", CALIPER_NET_PART_IMAGE);
    depth.params.set("form", Json(CALIPER_NET_FORM_DEPTH));
    depth.params.set("colormap", Json(0));
    depth.params.set("flags", Json(0));
    depth.params.set("vmin", Json(2.0));
    depth.params.set("vmax", Json(3.1));
    depth.params.set("cloud", Json(true));
    depth.arrays.push_back(f.section("pixels", CALIPER_NET_DTYPE_U16, {12, 16}, dm));
    f.messages.push_back(depth);

    if (!pixel_space) return;
    // The first camera's pixel space: marks, a box and an arrow in pixels, drawn and picked over the
    // translucent image they annotate although "flow" and "marks" come before "rgb" in path order.
    std::vector<float> marks = {8, 8, 56, 8, 56, 40, 8, 40};
    std::vector<uint32_t> mc = {RGBA(255, 60, 60), RGBA(60, 255, 60), RGBA(60, 60, 255), RGBA(255, 255, 255)};
    wire::SceneUpdate pts = f.update("rig/left/marks", CALIPER_NET_PART_POINTS);
    pts.params.set("radius", Json(2.5));
    pts.arrays.push_back(f.section("positions", CALIPER_NET_DTYPE_F32, {4, 2}, marks));
    pts.arrays.push_back(f.section("colors", CALIPER_NET_DTYPE_U8, {4, 4}, mc));
    f.messages.push_back(pts);
    std::vector<float> bc = {32, 24}, bh = {10, 6};
    wire::SceneUpdate box = f.update("rig/left/roi", CALIPER_NET_PART_BOXES);
    box.params.set("color", rgba(RGBA(255, 230, 0)));
    box.params.set("width", Json(4.0));
    box.arrays.push_back(f.section("centers", CALIPER_NET_DTYPE_F32, {1, 2}, bc));
    box.arrays.push_back(f.section("half_sizes", CALIPER_NET_DTYPE_F32, {1, 2}, bh));
    f.messages.push_back(box);
    std::vector<float> ao = {12, 36}, av = {14, -10};
    wire::SceneUpdate arrow = f.update("rig/left/flow", CALIPER_NET_PART_ARROWS);
    arrow.params.set("color", rgba(RGBA(0, 220, 255)));
    arrow.params.set("width", Json(2.0));
    arrow.arrays.push_back(f.section("origins", CALIPER_NET_DTYPE_F32, {1, 2}, ao));
    arrow.arrays.push_back(f.section("vectors", CALIPER_NET_DTYPE_F32, {1, 2}, av));
    f.messages.push_back(arrow);
}

inline Fixture cameras() {
    Fixture f;
    f.name = "cameras";
    f.info = {CALIPER_NET_SPACE_3D, "+z", "m"};
    rigs(f, false);
    return f;
}

inline Fixture through() {
    Fixture f;
    f.name = "through";
    f.info = {CALIPER_NET_SPACE_3D, "+z", "m"};
    rigs(f, true);
    f.look_through = "rig/left";
    return f;
}

inline Fixture images() {
    Fixture f;
    f.name = "images";
    f.info = {CALIPER_NET_SPACE_2D, "-y", "px"};
    // A colour image of smooth gradients and a grey one, sampled bilinearly.
    std::vector<uint32_t> rgb(16 * 12);
    std::vector<uint8_t> grey(16 * 12);
    for (uint32_t y = 0; y < 12; ++y)
        for (uint32_t x = 0; x < 16; ++x) {
            rgb[y * 16 + x] = RGBA(30 + 12 * x, 40 + 14 * y, 220 - 8 * x, 255);
            grey[y * 16 + x] = uint8_t(20 + 10 * x + 4 * y);
        }
    wire::SceneUpdate a = f.update("image/rgb", CALIPER_NET_PART_IMAGE);
    a.params.set("form", Json(CALIPER_NET_FORM_COLOR));
    a.arrays.push_back(f.section("pixels", CALIPER_NET_DTYPE_U8, {12, 16, 4}, rgb));
    f.messages.push_back(a);
    wire::SceneUpdate g = f.update("image/grey", CALIPER_NET_PART_IMAGE);
    g.params.set("form", Json(CALIPER_NET_FORM_COLOR));
    g.arrays.push_back(f.section("pixels", CALIPER_NET_DTYPE_U8, {12, 16}, grey));
    f.messages.push_back(g);
    f.transform("image/grey", translate(20, 0, 0));
    // A u16 depth through viridis by nearest, in hundredths, with a zero and a far value.
    std::vector<uint16_t> d(16 * 12);
    for (uint32_t y = 0; y < 12; ++y)
        for (uint32_t x = 0; x < 16; ++x) d[y * 16 + x] = uint16_t(50 + 9 * x + 3 * y);
    d[0] = 0;
    d[17] = 65535;
    wire::SceneUpdate depth = f.update("image/depth", CALIPER_NET_PART_IMAGE);
    depth.params.set("form", Json(CALIPER_NET_FORM_DEPTH));
    depth.params.set("scale", Json(0.01));
    depth.params.set("colormap", Json(0));
    depth.params.set("flags", Json(0));
    depth.params.set("vmin", Json(0.5));
    depth.params.set("vmax", Json(2.5));
    depth.arrays.push_back(f.section("pixels", CALIPER_NET_DTYPE_U16, {12, 16}, d));
    f.messages.push_back(depth);
    f.transform("image/depth", translate(0, 16, 0));
    // A segmentation image over the class table, at its default opacity of one half.
    wire::SceneClasses cl;
    cl.sub = 1;
    cl.view = f.name;
    cl.seq = ++f.seq;
    cl.is_static = true;
    cl.classes = classes_json({{1, RGBA(255, 40, 40)}, {2, RGBA(40, 255, 40)}}, {"road", "tree"});
    f.messages.push_back(cl);
    std::vector<uint8_t> seg(16 * 12);
    for (uint32_t y = 0; y < 12; ++y)
        for (uint32_t x = 0; x < 16; ++x) seg[y * 16 + x] = uint8_t(x < 6 ? 1 : x < 11 ? 2 : (y < 6 ? 5 : 0));
    wire::SceneUpdate s = f.update("image/seg", CALIPER_NET_PART_IMAGE);
    s.params.set("form", Json(CALIPER_NET_FORM_SEGMENTATION));
    s.arrays.push_back(f.section("pixels", CALIPER_NET_DTYPE_U8, {12, 16}, seg));
    f.messages.push_back(s);
    f.transform("image/seg", translate(20, 16, 0));
    // Boxes and arrows in the plane: after the opaque images in the byte order of the paths, and under
    // the translucent segmentation, which is drawn after every opaque batch.
    std::vector<float> bc = {8, 6, 28, 22}, bh = {5, 3, 4, 4};
    std::vector<uint32_t> bcol = {RGBA(255, 255, 255), RGBA(255, 200, 0)};
    wire::SceneUpdate b = f.update("marks/boxes", CALIPER_NET_PART_BOXES);
    b.params.set("width", Json(4.0));
    b.arrays.push_back(f.section("centers", CALIPER_NET_DTYPE_F32, {2, 2}, bc));
    b.arrays.push_back(f.section("half_sizes", CALIPER_NET_DTYPE_F32, {2, 2}, bh));
    b.arrays.push_back(f.section("colors", CALIPER_NET_DTYPE_U8, {2, 4}, bcol));
    f.messages.push_back(b);
    std::vector<float> ao = {2, 26, 34, 2}, av = {10, -4, -6, 8};
    wire::SceneUpdate ar = f.update("marks/arrows", CALIPER_NET_PART_ARROWS);
    ar.params.set("color", rgba(RGBA(0, 255, 255)));
    ar.params.set("width", Json(2.0));
    ar.arrays.push_back(f.section("origins", CALIPER_NET_DTYPE_F32, {2, 2}, ao));
    ar.arrays.push_back(f.section("vectors", CALIPER_NET_DTYPE_F32, {2, 2}, av));
    f.messages.push_back(ar);
    return f;
}

inline std::vector<Fixture> all() { return {arrows(), boxes(), edges(), mesh(), cameras(), through(), images()}; }

}  // namespace primitive_fixtures
