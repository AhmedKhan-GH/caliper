// parity_fixture.h: the fixture of the pixel comparison (design/programmable-graphics.md, Pixel
// Comparison): two instances, attention_field and attention_arcs, as the app sends them in the
// instance form of PROGRAM set, and one recorded version of each published resource as RESOURCE,
// from the worked examples of tests/gfx/example_pieces.h, thus no training runs. Two more
// programs are ones that a viewer must refuse: a copy of the field whose draw.wgsl WebGPU does not
// compile (program.browser_refused at a known line), and the field with a pin whose hash its files
// do not have (piece.hash_mismatch).
#pragma once

#include "parity_draw.h"

#include "../gfx/example_pieces.h"

#include <caliper_builder/programs/pieces.h>

#include <algorithm>
#include <string>
#include <vector>

namespace draw_parity {

namespace ex = caliper_test_pieces;

// The frames and the plan of the fixture.
struct Fixture {
    std::vector<std::uint8_t> frames;   // each frame with its length (split_frames)
    Plan plan;
    std::string error;                  // non-empty: the fixture could not be made
};

// PROGRAM set of an instance as the app's program host sends it (live/live_programs.h,
// AppProgramHost::wanted_instance and ProgramSync): the pin, the canonical manifest, the bindings,
// the values, the panel, the segments of the lib modules and common.wgsl, and the sections with the
// imports resolved. The section derive goes with it, so that a viewer can check the pin's hash.
inline bool instance_message(const caliper_builder::spec::Piece& piece, const std::string& id, const std::string& panel,
                             const caliper_net::Json& values, const std::string& pin_hash,
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
    m.piece.set("hash", pin_hash);
    const std::string manifest = cp::canonical_manifest(piece);
    std::string err;
    if (manifest.empty() || !Json::parse(manifest.data(), manifest.size(), &m.manifest, &err))
        return (*why = piece.name + ": the manifest has no canonical form " + err, false);
    m.bind = Json::object();
    m.bind.set("attn0", "live:blocks.0.attn.weights.act");
    m.bind.set("attn1", "live:blocks.1.attn.weights.act");
    m.params = values;
    m.panel = panel;
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

template <typename T>
caliper_net::wire::Resource resource_message(const std::string& program, const std::string& id,
                                             const std::vector<T>& v, std::uint64_t step) {
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

// The draw.wgsl of the field with one more function that WebGPU refuses: a u32 where an f32 goes.
// *line is the line of that error in the file.
inline std::string broken_field_draw(std::uint32_t* line) {
    std::string text = ex::kFieldDraw;
    if (!text.empty() && text.back() != '\n') text += '\n';
    const std::uint32_t before = std::uint32_t(std::count(text.begin(), text.end(), '\n'));
    text += "fn broken_gain() -> f32 {\n    let g: f32 = 2u;\n    return g;\n}\n";
    *line = before + 2;
    return text;
}

inline Fixture make_fixture(std::uint32_t size = 512) {
    namespace cp = caliper_builder::programs;
    namespace cs = caliper_builder::spec;
    using caliper_net::Json;
    Fixture fx;
    fx.plan.size = size;
    fx.plan.frames = 3;
    fx.plan.step = 100;
    auto add = [&](const caliper_net::wire::Message& m) {
        std::vector<std::uint8_t> bytes;
        std::string err;
        if (!caliper_net::wire::encode(m, &bytes, &err)) {
            if (fx.error.empty()) fx.error = "a frame of the fixture does not encode: " + err;
            return;
        }
        append_frame(&fx.frames, bytes);
    };
    auto program = [&](const cs::Piece& piece, const std::string& id, const std::string& panel, const Json& values,
                       const std::string& pin) {
        caliper_net::wire::Program m;
        std::string why;
        if (!instance_message(piece, id, panel, values, pin, &m, &why)) {
            if (fx.error.empty()) fx.error = why;
            return;
        }
        add(m);
    };

    // The field: layer 1, head 2 (the induction head), gain 2, as the graphics suite's row draws it.
    const cs::Piece field = ex::attention_field();
    Json field_values = Json::object();
    field_values.set("layer", 1);
    field_values.set("head", 2);
    field_values.set("gain", 2.0);
    program(field, "head_pattern", "pattern", field_values, cp::piece_hash(field));
    const std::vector<float> pattern = ex::field_pattern();
    add(resource_message("head_pattern", "pattern", pattern, std::uint64_t(fx.plan.step)));
    ProgramPlan fp;
    fp.id = "head_pattern";
    fp.pick_id = 1;
    fp.inputs.world_from_clip = ex::Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    fp.inputs.clip_from_world = fp.inputs.world_from_clip;
    fp.inputs.view_from_world = fp.inputs.world_from_clip;
    fp.inputs.viewport = {0, 0, float(size), float(size)};
    fp.inputs.pixel_ratio = 1.0f;
    fx.plan.programs.push_back(fp);

    // The arcs: threshold 0.05 and glow 1, under the graphics suite's fixed camera at a pixel ratio of 2.
    const cs::Piece arcs = ex::attention_arcs();
    Json arc_values = Json::object();
    arc_values.set("threshold", 0.05);
    arc_values.set("glow", 1.0);
    program(arcs, "tower_arcs", "tower", arc_values, cp::piece_hash(arcs));
    const ex::ArcResources res = ex::arc_resources();
    add(resource_message("tower_arcs", "heads", res.heads, std::uint64_t(fx.plan.step)));
    add(resource_message("tower_arcs", "arcs", res.arcs, std::uint64_t(fx.plan.step)));
    add(resource_message("tower_arcs", "labels", res.labels, std::uint64_t(fx.plan.step)));
    ProgramPlan ap;
    ap.id = "tower_arcs";
    ap.pick_id = 1u << 24;
    const ex::Mat4 view = ex::view_matrix({0.0f, 1.9f, 2.6f}, {0.0f, 0.3f, 0.0f});
    const ex::Mat4 proj = ex::perspective(0.8f, 1.0f, 0.1f, 20.0f);
    ap.inputs.clip_from_world = ex::multiply(proj, view);
    ap.inputs.view_from_world = view;
    ap.inputs.world_from_clip = ex::Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    ap.inputs.eye = {0.0f, 1.9f, 2.6f, 1.0f};
    ap.inputs.viewport = {0, 0, float(size), float(size)};
    ap.inputs.pixel_ratio = 2.0f;
    fx.plan.programs.push_back(ap);

    // A field whose draw.wgsl WebGPU refuses, with the pin of its own files.
    cs::Piece broken = field;
    broken.name = "attention_field_broken";
    std::uint32_t line = 0;
    broken.files["draw.wgsl"] = broken_field_draw(&line);
    program(broken, "broken_pattern", "broken", field_values, cp::piece_hash(broken));
    add(resource_message("broken_pattern", "pattern", pattern, std::uint64_t(fx.plan.step)));
    ProgramPlan bp = fp;
    bp.id = "broken_pattern";
    bp.pick_id = 2;
    bp.expect = cp::kProgramBrowserRefused;
    bp.expect_line = line;
    fx.plan.programs.push_back(bp);

    // The field under a pin whose hash its files do not have.
    program(field, "wrong_pin", "wrong", field_values, "sha256:" + std::string(64, '0'));
    ProgramPlan wp = fp;
    wp.id = "wrong_pin";
    wp.pick_id = 3;
    wp.expect = cp::kPieceHashMismatch;
    fx.plan.programs.push_back(wp);
    return fx;
}

}  // namespace draw_parity
