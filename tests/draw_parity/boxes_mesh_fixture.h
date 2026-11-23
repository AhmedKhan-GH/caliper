// boxes_mesh_fixture.h: the library pieces boxes@1 and mesh@1 in the fixture of the pixel comparison
// (design/programmable-graphics.md, Pixel Comparison). Each is one instance as the app sends it in
// the instance form of PROGRAM set, with one version of each published resource as RESOURCE,
// written here as the pieces' derive halves write them, thus no training runs:
//
//   library_boxes  twelve boxes in 3-D, some rotated, with RGBA of each box, edges 3 pixels wide at a
//                  pixel ratio of 1.5 and a depth offset of 2. Without faces: the edges and the
//                  faces of a box have one ID, the box, thus the comparison would hold the border of
//                  an edge over a face to the levels of an interior pixel, and two renderers
//                  rasterize that border a pixel apart
//   library_mesh   a height field of 32 by 32 vertices lit from its normals, colored by values
//                  through viridis
//
// The plan draws both under the fixed camera of the arcs' row. parity_fixture.h calls
// add_boxes_mesh() from make_fixture().
#pragma once

#include "parity_draw.h"

#include "../gfx/example_pieces.h"

#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/spec/spec.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace draw_parity {

namespace boxes_mesh_detail {

namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
using caliper_net::Json;

// A piece of the built-in library, version 1, with its files and its hash.
inline bool library_piece(const std::string& name, cs::Piece* piece, std::string* why) {
    std::string_view manifest;
    std::map<std::string, std::string> files;
    const std::string prefix = "pieces/" + name + "/1/";
    for (const cp::LibraryFile& f : cp::library_files()) {
        const std::string_view path = f.path;
        if (path.substr(0, prefix.size()) != prefix) continue;
        const std::string file(path.substr(prefix.size()));
        if (file == "piece.toml") manifest = f.text;
        else files[file] = std::string(f.text);
    }
    if (manifest.empty()) return (*why = "the library has no " + prefix + "piece.toml", false);
    cs::PieceParse parse = cs::piece_from_toml(manifest);
    if (!parse.ok()) return (*why = prefix + "piece.toml does not parse", false);
    *piece = std::move(parse.piece);
    for (const char* file : {"common.wgsl", "derive.wgsl", "draw.wgsl"})
        if (files.count(file)) piece->files[file] = files[file];
    return true;
}

// PROGRAM set of an instance of a library piece, as the app's program host sends it: the pin, the
// canonical manifest, a binding of each tensor input, the values, the panel, the lib modules before
// common.wgsl, and the sections with the imports resolved.
inline bool program(const cs::Piece& piece, const std::string& id, const std::string& panel, const Json& values,
                    caliper_net::wire::Program* out, std::string* why) {
    const cp::Resolved resolved = cp::resolve_imports(piece);
    if (!resolved.verdict.ok()) return (*why = piece.name + ": " + resolved.verdict.message, false);
    auto file = [&](const char* name) {
        auto it = resolved.files.find(name);
        return it == resolved.files.end() ? std::string() : it->second;
    };
    caliper_net::wire::Program m;
    m.op = CALIPER_NET_PROGRAM_SET;
    m.id = id;
    m.piece = Json::object();
    m.piece.set("name", piece.name);
    m.piece.set("version", std::uint64_t(piece.version));
    m.piece.set("hash", cp::piece_hash(piece));
    const std::string manifest = cp::canonical_manifest(piece);
    std::string err;
    if (manifest.empty() || !Json::parse(manifest.data(), manifest.size(), &m.manifest, &err))
        return (*why = piece.name + ": the manifest has no canonical form " + err, false);
    m.bind = Json::object();
    for (const cs::PieceInput& in : piece.inputs) m.bind.set(in.name, "live:" + id + "." + in.name);
    m.params = values;
    m.panel = panel;
    m.common = resolved.lib_text + file("common.wgsl");
    m.segments = Json::array();
    for (const cp::Segment& seg : resolved.lib_segments) {
        Json j = Json::object();
        j.set("part", seg.part);
        j.set("first_line", std::uint64_t(seg.first_line));
        j.set("lines", std::uint64_t(seg.lines));
        m.segments.push(std::move(j));
    }
    m.derive = file("derive.wgsl");
    m.draw = file("draw.wgsl");
    *out = std::move(m);
    return true;
}

inline caliper_net::wire::Resource resource(const std::string& program, const std::string& id,
                                            const std::vector<std::uint8_t>& rows, std::uint32_t count,
                                            std::uint32_t stride, std::uint64_t step) {
    caliper_net::wire::Resource r;
    r.program = program;
    r.resource = id;
    r.step = step;
    r.count = count;
    r.stride = stride;
    r.data.name = "resource";
    r.data.dtype = CALIPER_NET_DTYPE_U8;
    r.data.shape = {std::int64_t(rows.size())};
    r.data.data = rows.data();
    r.data.size = rows.size();
    return r;
}

template <class T>
void put(std::vector<std::uint8_t>* rows, std::size_t at, const T& v) {
    std::memcpy(rows->data() + at, &v, sizeof v);
}

// The element rows of boxes@1's resource `boxes` (64 bytes each): center (x, y, z, 3), half sizes
// (x, y, z, and the flags in box 0), rotation, RGBA, class RGBA, index.
inline std::vector<std::uint8_t> box_rows() {
    constexpr int n = 12;
    std::vector<std::uint8_t> rows(std::size_t(n) * 64, 0);
    for (int i = 0; i < n; ++i) {
        const float a = float(i) * 0.5236f;
        const float c[4] = {0.9f * std::cos(a), 0.25f * float(i % 3) - 0.1f, 0.9f * std::sin(a), 3.0f};
        const float h[4] = {0.12f + 0.02f * float(i % 4), 0.1f + 0.03f * float(i % 3), 0.14f, 0.0f};
        const float s = std::sin(0.2f * float(i)), co = std::cos(0.2f * float(i));
        const float q[4] = {i % 2 ? s : 0.0f, 0.0f, i % 2 ? 0.0f : s, co};
        const std::uint32_t rgba = 0xff000000u | (std::uint32_t((i * 97 + 60) % 256) << 16) |
                                   (std::uint32_t((i * 53 + 120) % 256) << 8) | std::uint32_t((i * 31 + 200) % 256);
        const std::size_t at = std::size_t(i) * 64;
        put(&rows, at, c);
        put(&rows, at + 16, h);
        put(&rows, at + 32, q);
        put(&rows, at + 48, rgba);
        put(&rows, at + 52, 0xff7f7f7fu);
        put(&rows, at + 56, std::uint32_t(i));
    }
    return rows;
}

// The rows of mesh@1's resources: `vertices` (48 bytes each: position with the value in w, normal
// with the flags of vertex 0 in w, RGBA, index) of a height field of 32 by 32 vertices, and
// `faces` (the index buffer: three u32 indices for each triangle).
inline void mesh_rows(std::vector<std::uint8_t>* vertices, std::vector<std::uint8_t>* faces, std::uint32_t* nv,
                      std::uint32_t* nt) {
    constexpr int side = 32;
    *nv = side * side;
    *nt = (side - 1) * (side - 1) * 2;
    vertices->assign(std::size_t(*nv) * 48, 0);
    faces->assign(std::size_t(*nt) * 12, 0);
    for (int j = 0; j < side; ++j)
        for (int i = 0; i < side; ++i) {
            const float x = -1.0f + 2.0f * float(i) / float(side - 1), z = -1.0f + 2.0f * float(j) / float(side - 1);
            const float y = 0.25f * std::sin(3.0f * x) * std::cos(2.0f * z);
            const float dx = 0.75f * std::cos(3.0f * x) * std::cos(2.0f * z), dz = -0.5f * std::sin(3.0f * x) * std::sin(2.0f * z);
            const float len = std::sqrt(dx * dx + dz * dz + 1.0f);
            const int k = j * side + i;
            const float p[4] = {x, y, z, float((k * 37) % 256) / 255.0f};
            const float nrm[4] = {-dx / len, 1.0f / len, -dz / len, 0.0f};
            const std::size_t at = std::size_t(k) * 48;
            put(vertices, at, p);
            put(vertices, at + 16, nrm);
            put(vertices, at + 32, 0xffe6dcdcu);
            put(vertices, at + 36, std::uint32_t(k));
        }
    std::size_t t = 0;
    for (int j = 0; j + 1 < side; ++j)
        for (int i = 0; i + 1 < side; ++i) {
            const std::uint32_t a = std::uint32_t(j * side + i), b = a + 1, c = a + side, d = c + 1;
            const std::uint32_t two[2][3] = {{a, b, d}, {a, d, c}};
            for (const auto& tri : two) put(faces, 12 * t++, tri);
        }
}

}  // namespace boxes_mesh_detail

// Appends the programs, the resources and the plans of boxes@1 and mesh@1 to the fixture.
inline void add_boxes_mesh(std::vector<std::uint8_t>* frames, Plan* plan, std::string* error) {
    namespace bm = boxes_mesh_detail;
    namespace ex = caliper_test_pieces;
    using caliper_net::Json;
    auto add = [&](const caliper_net::wire::Message& m) {
        std::vector<std::uint8_t> bytes;
        std::string err;
        if (!caliper_net::wire::encode(m, &bytes, &err)) {
            if (error->empty()) *error = "a frame of the shapes does not encode: " + err;
            return;
        }
        append_frame(frames, bytes);
    };
    const ex::Mat4 view = ex::view_matrix({0.0f, 1.9f, 2.6f}, {0.0f, 0.0f, 0.0f});
    const ex::Mat4 proj = ex::perspective(0.8f, 1.0f, 0.1f, 20.0f);
    auto camera = [&](ProgramPlan* p, float ratio) {
        p->inputs.clip_from_world = ex::multiply(proj, view);
        p->inputs.view_from_world = view;
        p->inputs.world_from_clip = ex::Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        p->inputs.eye = {0.0f, 1.9f, 2.6f, 1.0f};
        p->inputs.viewport = {0, 0, float(plan->size), float(plan->size)};
        p->inputs.pixel_ratio = ratio;
    };
    const std::uint64_t step = std::uint64_t(plan->step);
    std::string why;

    caliper_builder::spec::Piece boxes;
    caliper_net::wire::Program m;
    Json values = Json::object();
    values.set("width", 3.0);
    values.set("fill", 0);
    values.set("color_mode", 1);
    values.set("depth_offset", 2);
    if (!bm::library_piece("boxes", &boxes, &why) || !bm::program(boxes, "library_boxes", "boxes", values, &m, &why)) {
        if (error->empty()) *error = why;
        return;
    }
    add(m);
    const std::vector<std::uint8_t> box_rows = bm::box_rows();
    add(bm::resource("library_boxes", "boxes", box_rows, std::uint32_t(box_rows.size() / 64), 64, step));
    add(bm::resource("library_boxes", "names", {}, 0, 64, step));   // no labels: the strings of the text table
    ProgramPlan bp;
    bp.id = "library_boxes";
    bp.pick_id = 101;
    camera(&bp, 1.5f);
    plan->programs.push_back(bp);

    caliper_builder::spec::Piece mesh;
    Json mv = Json::object();
    mv.set("color_mode", 1);
    mv.set("shading", 1);
    if (!bm::library_piece("mesh", &mesh, &why) || !bm::program(mesh, "library_mesh", "mesh", mv, &m, &why)) {
        if (error->empty()) *error = why;
        return;
    }
    add(m);
    std::vector<std::uint8_t> vertices, faces;
    std::uint32_t nv = 0, nt = 0;
    bm::mesh_rows(&vertices, &faces, &nv, &nt);
    add(bm::resource("library_mesh", "vertices", vertices, nv, 48, step));
    add(bm::resource("library_mesh", "faces", faces, 3 * nt, 4, step));
    ProgramPlan mp;
    mp.id = "library_mesh";
    mp.pick_id = 102;
    camera(&mp, 1.0f);
    plan->programs.push_back(mp);
}

}  // namespace draw_parity
