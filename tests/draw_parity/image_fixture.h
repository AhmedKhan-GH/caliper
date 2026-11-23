// image_fixture.h: the programs of the library pieces image@1, depth@1 and segmentation@1 in the
// fixture of the pixel comparison (parity_fixture.h). Each piece is one instance, sent as the app
// sends it (parity_fixture.h, instance_message), with one version of each published resource, made
// on the CPU as its derive half makes it: a color image with translucent pixels flat under a 2-D
// camera, a depth map on a camera's image plane with its back-projected cloud under a perspective
// camera, and a mask over a table of class colors at an opacity flat under a 2-D camera.
#pragma once

#include "parity_draw.h"

#include "../gfx/example_pieces.h"

#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/spec/spec.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace draw_parity {

namespace image_pieces {

// The library piece `name`@1 with its files and hash, or a piece without a name.
inline caliper_builder::spec::Piece library_piece(const std::string& name) {
    namespace cp = caliper_builder::programs;
    namespace cs = caliper_builder::spec;
    std::string manifest;
    std::map<std::string, std::string> files;
    const std::string prefix = "pieces/" + name + "/1/";
    for (const cp::LibraryFile& f : cp::library_files()) {
        const std::string path(f.path);
        if (path.rfind(prefix, 0) != 0) continue;
        const std::string file = path.substr(prefix.size());
        if (file == "piece.toml") manifest = std::string(f.text);
        else files[file] = std::string(f.text);
    }
    cs::PieceParse parse = cs::piece_from_toml(manifest);
    if (manifest.empty() || !parse.ok()) return cs::Piece{};
    cs::Piece piece = std::move(parse.piece);
    for (const auto& [file, text] : files) piece.files[file] = text;
    piece.hash = cp::piece_hash(piece);
    return piece;
}

// PROGRAM set of an instance as the app's program host sends it, with the pin of the piece's own
// files: the canonical manifest, the bindings, the values, the panel, the segments of the lib
// modules and common.wgsl, and the sections with the imports resolved.
inline bool program_message(const caliper_builder::spec::Piece& piece, const std::string& id,
                            const caliper_net::Json& values, const caliper_net::Json& bind,
                            caliper_net::wire::Program* out, std::string* why) {
    namespace cp = caliper_builder::programs;
    using caliper_net::Json;
    const cp::Resolved resolved = cp::resolve_imports(piece);
    if (!resolved.verdict.ok()) return (*why = piece.name + ": " + resolved.verdict.message, false);
    auto file = [&](const char* name) {
        auto it = resolved.files.find(name);
        return it == resolved.files.end() ? std::string() : it->second;
    };
    auto lines = [](const std::string& t) {
        std::uint32_t n = std::uint32_t(std::count(t.begin(), t.end(), '\n'));
        return t.empty() || t.back() == '\n' ? n : n + 1;
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
    m.bind = bind;
    m.params = values;
    m.panel = id;
    const std::string common_file = file("common.wgsl");
    m.common = resolved.lib_text + common_file;
    m.segments = Json::array();
    for (const cp::Segment& seg : resolved.lib_segments) {
        Json j = Json::object();
        j.set("part", seg.part);
        j.set("first_line", std::uint64_t(seg.first_line));
        j.set("lines", std::uint64_t(seg.lines));
        m.segments.push(std::move(j));
    }
    if (!common_file.empty()) {
        Json j = Json::object();
        j.set("part", "common.wgsl");
        j.set("first_line", std::uint64_t(lines(resolved.lib_text) + 1));
        j.set("lines", std::uint64_t(lines(common_file)));
        m.segments.push(std::move(j));
    }
    m.derive = file("derive.wgsl");
    m.draw = file("draw.wgsl");
    *out = std::move(m);
    return true;
}

// RESOURCE of one version: the elements of v, each of sizeof(T) bytes.
template <typename T>
caliper_net::wire::Resource resource(const std::string& program, const std::string& id, const std::vector<T>& v,
                                     std::uint64_t step) {
    caliper_net::wire::Resource r;
    r.program = program;
    r.resource = id;
    r.step = step;
    r.count = std::uint32_t(v.size());
    r.stride = std::uint32_t(sizeof(T));
    r.data.name = "resource";
    r.data.dtype = CALIPER_NET_DTYPE_U8;
    r.data.shape = {std::int64_t(v.size() * sizeof(T))};
    r.data.data = reinterpret_cast<const std::uint8_t*>(v.data());
    r.data.size = v.size() * sizeof(T);
    return r;
}

// The element `info` of image@1 and segmentation@1, and of depth@1 with its range.
struct Info {
    std::uint32_t size[4];
    float pose[16];   // column-major
};
struct DepthInfo {
    std::uint32_t size[4];
    float pose[16];
    float range[4];
};
static_assert(sizeof(Info) == 80 && sizeof(DepthInfo) == 96, "the layouts of the manifests");

inline void identity(float* m) {
    for (int i = 0; i < 16; ++i) m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
}

// A 2-D camera with y down over a world `width` units wide, from its top left corner.
inline caliper_draw::FrameInputs flat_inputs(float width, std::uint32_t size) {
    caliper_draw::FrameInputs in;
    const float s = 2.0f / width;
    in.clip_from_world = {s, 0, 0, 0, 0, -s, 0, 0, 0, 0, 0, 0, -1, 1, 0.5f, 1};
    in.world_from_clip = {1 / s, 0, 0, 0, 0, -1 / s, 0, 0, 0, 0, 1, 0, 1 / s, 1 / s, 0, 1};
    in.view_from_world = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    in.viewport = {0, 0, float(size), float(size)};
    in.pixel_ratio = 1.0f;
    return in;
}

}  // namespace image_pieces

// Adds the three programs as frames, with their versions, and their plans.
inline void add_image_pieces(std::vector<std::uint8_t>* frames, Plan* plan, std::string* error) {
    namespace cs = caliper_builder::spec;
    namespace ex = caliper_test_pieces;
    namespace ip = image_pieces;
    using caliper_net::Json;
    const std::uint64_t step = std::uint64_t(plan->step);
    auto add = [&](const caliper_net::wire::Message& m) {
        std::vector<std::uint8_t> bytes;
        std::string err;
        if (!caliper_net::wire::encode(m, &bytes, &err)) {
            if (error->empty()) *error = "a frame of the image pieces does not encode: " + err;
            return;
        }
        append_frame(frames, bytes);
    };
    auto program = [&](const std::string& piece_name, const std::string& id, const Json& values, const Json& bind) {
        const cs::Piece piece = ip::library_piece(piece_name);
        if (piece.name.empty()) {
            if (error->empty()) *error = "the library has no piece " + piece_name;
            return;
        }
        caliper_net::wire::Program m;
        std::string why;
        if (!ip::program_message(piece, id, values, bind, &m, &why)) {
            if (error->empty()) *error = why;
            return;
        }
        add(m);
    };
    using ip::resource;

    // image@1: 48 x 32 pixels of smooth gradients, a block of them at half alpha.
    {
        const std::uint32_t w = 48, h = 32;
        std::vector<std::uint32_t> pixels(std::size_t(w) * h);
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x) {
                const std::uint32_t a = (x >= 30 && x < 40 && y >= 8 && y < 20) ? 128u : 255u;
                pixels[std::size_t(y) * w + x] = (20 + 4 * x) | (30 + 6 * y) << 8 | (220 - 3 * x) << 16 | a << 24;
            }
        ip::Info info{{w, h, 4, 0}, {}};
        ip::identity(info.pose);
        Json bind = Json::object();
        bind.set("image", "live:image");
        bind.set("pose", "live:pose");
        Json values = Json::object();
        values.set("opacity", 1.0);
        program("image", "image_flat", values, bind);
        caliper_net::wire::Resource texels = resource("image_flat", "pixels", pixels, step);
        texels.width = w;   // image@1's pixels are an rgba8unorm texture of the image's size
        texels.height = h;
        add(texels);
        add(resource("image_flat", "info", std::vector<ip::Info>{info}, step));
        ProgramPlan p;
        p.id = "image_flat";
        p.pick_id = 4;
        p.inputs = ip::flat_inputs(float(w), plan->size);
        plan->programs.push_back(p);
    }

    // depth@1: 32 x 24 depths from 2 to 3 with holes, on the image plane of a camera at the origin
    // with its cloud, in magma, under a perspective camera.
    {
        const std::uint32_t w = 32, h = 24;
        std::vector<float> depth(std::size_t(w) * h);
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x)
                depth[std::size_t(y) * w + x] = 2.0f + float((x * 11 + y * 7) % 200 + 20) / 255.0f;
        for (std::uint32_t i : {7u, 300u, 555u}) depth[i] = 0.0f;
        ip::DepthInfo info{{w, h, 1, 0}, {}, {2.0f, 3.0f, 1.0f, 0.0f}};
        ip::identity(info.pose);
        Json values = Json::object();
        values.set("colormap", 1);
        values.set("vmin", 2.0);
        values.set("vmax", 3.0);
        values.set("auto_range", 0);
        values.set("cloud", 1);
        values.set("placement", 1);
        values.set("fx", 24.0);
        values.set("fy", 24.0);
        values.set("cx", 16.0);
        values.set("cy", 12.0);
        values.set("plane", 1.5);
        Json bind = Json::object();
        bind.set("depth", "live:depth");
        bind.set("pose", "live:pose");
        program("depth", "depth_cloud", values, bind);
        add(resource("depth_cloud", "pixels", depth, step));
        add(resource("depth_cloud", "info", std::vector<ip::DepthInfo>{info}, step));
        ProgramPlan p;
        p.id = "depth_cloud";
        p.pick_id = 5;
        const ex::Mat4 view = ex::view_matrix({2.6f, -1.4f, -1.0f}, {0.0f, 0.0f, 2.0f});
        const ex::Mat4 proj = ex::perspective(0.9f, 1.0f, 0.1f, 20.0f);
        p.inputs.clip_from_world = ex::multiply(proj, view);
        p.inputs.view_from_world = view;
        p.inputs.world_from_clip = ex::Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        p.inputs.eye = {2.6f, -1.4f, -1.0f, 1.0f};
        p.inputs.viewport = {0, 0, float(plan->size), float(plan->size)};
        p.inputs.pixel_ratio = 1.0f;
        plan->programs.push_back(p);
    }

    // segmentation@1: 32 x 24 class ids in bands, two of them in colors of the table, one of those
    // translucent, at an opacity of 0.6.
    {
        const std::uint32_t w = 32, h = 24;
        std::vector<std::uint32_t> classes(std::size_t(w) * h);
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x) classes[std::size_t(y) * w + x] = (x / 6 + 3 * (y / 8)) % 9;
        static const std::uint32_t kPalette[16] = {
            0xffb4771fu, 0xff0e7fffu, 0xff2ca02cu, 0xff2827d6u, 0xffbd6794u, 0xff4b568cu, 0xffc277e3u, 0xff7f7f7fu,
            0xff22bdbcu, 0xffcfbe17u, 0xffe8c7aeu, 0xff78bbffu, 0xff8adf98u, 0xff9698ffu, 0xffd5b0c5u, 0xff949cc4u};
        std::vector<std::uint32_t> palette(4096);
        for (std::uint32_t k = 0; k < palette.size(); ++k) palette[k] = kPalette[k % 16];
        palette[1] = 0xff2828ffu;   // red
        palette[4] = 0x8028ff28u;   // green at half alpha
        ip::Info info{{w, h, 1, 0}, {}};
        ip::identity(info.pose);
        Json values = Json::object();
        values.set("opacity", 0.6);
        Json bind = Json::object();
        bind.set("classes", "live:classes");
        bind.set("colors", "live:colors");
        bind.set("pose", "live:pose");
        program("segmentation", "mask_flat", values, bind);
        add(resource("mask_flat", "pixels", classes, step));
        add(resource("mask_flat", "palette", palette, step));
        add(resource("mask_flat", "info", std::vector<ip::Info>{info}, step));
        ProgramPlan p;
        p.id = "mask_flat";
        p.pick_id = 6;
        p.inputs = ip::flat_inputs(float(w), plan->size);
        plan->programs.push_back(p);
    }
}

}  // namespace draw_parity
