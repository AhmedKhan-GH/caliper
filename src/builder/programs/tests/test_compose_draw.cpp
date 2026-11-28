// Tests of the composer of a piece's draw half and of the reflection checks
// (programs/pieces.h, design/programmable-graphics.md, Generated Header and
// Check Order 7), on the two worked examples of the plan.
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/programs/sandbox.h>

#include <doctest/doctest.h>

#include "../../../../tests/gfx/example_pieces.h"

namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
namespace ex = caliper_test_pieces;

namespace {

// The header segment of a composed module.
std::string header_of(const cp::Composed& c) {
    for (const cp::Segment& s : c.segments) {
        if (s.part != "header") continue;
        std::size_t at = 0;
        for (std::uint32_t line = 1; line < s.first_line; ++line) at = c.text.find('\n', at) + 1;
        std::size_t end = at;
        for (std::uint32_t i = 0; i < s.lines; ++i) end = c.text.find('\n', end) + 1;
        return c.text.substr(at, end - at);
    }
    return {};
}

const cp::Segment* segment(const cp::Composed& c, const std::string& part) {
    for (const cp::Segment& s : c.segments)
        if (s.part == part) return &s;
    return nullptr;
}

std::string line_at(const std::string& text, std::uint32_t line) {
    std::size_t at = 0;
    for (std::uint32_t i = 1; i < line; ++i) at = text.find('\n', at) + 1;
    return text.substr(at, text.find('\n', at) - at);
}

}  // namespace

TEST_CASE("the draw header of the arcs' first pass is the plan's, with every target declared") {
    const cs::Piece piece = ex::ring_arcs();
    const cp::Composed c = cp::compose_draw(piece, ex::resolve(piece), 0);
    REQUIRE(c.verdict.ok());
    CHECK(header_of(c) ==
          "struct Params { threshold: f32, glow: f32 }\n"
          "@group(0) @binding(5) var<uniform> params: Params;\n"
          "struct CaliperCounts { heads: u32, arcs: u32, labels: u32 }\n"
          "struct Arc { from_token: u32, to_token: u32, layer: u32, head: u32, weight: f32, score: f32 }\n"
          "struct Label { position: vec4<f32>, string: u32 }\n"
          "@group(1) @binding(0) var<uniform> counts: CaliperCounts;\n"
          "@group(1) @binding(1) var<storage, read> heads: array<f32>;\n"
          "@group(1) @binding(2) var<storage, read> arcs: array<Arc>;\n"
          "@group(1) @binding(3) var<storage, read> labels: array<Label>;\n"
          "@group(2) @binding(4) var glow: texture_2d<f32>;\n");
    // Every pass declares the target glow, so that a function that samples it compiles in the
    // passes that write it; the host binds it only in the bloom, which reads it.
    for (std::size_t k = 1; k < 3; ++k)
        CHECK(header_of(cp::compose_draw(piece, ex::resolve(piece), k)) == header_of(c));
}

TEST_CASE("each pass keeps only its own entry points, with every line in its place") {
    const cs::Piece piece = ex::ring_arcs();
    const cp::Resolved r = ex::resolve(piece);
    const cp::Composed arcs = cp::compose_draw(piece, r, 0), bloom = cp::compose_draw(piece, r, 2);
    REQUIRE(arcs.verdict.ok());
    REQUIRE(bloom.verdict.ok());
    CHECK(arcs.text.find("fn arc_vs(") != std::string::npos);
    CHECK(arcs.text.find("fn arc_fs(") != std::string::npos);
    CHECK(arcs.text.find("fn arc_glow(") == std::string::npos);
    CHECK(arcs.text.find("fn bloom(") == std::string::npos);
    CHECK(arcs.text.find("fn arc_point(") != std::string::npos);   // no entry point: kept
    CHECK(bloom.text.find("fn bloom(") != std::string::npos);
    CHECK(bloom.text.find("fn arc_vs(") == std::string::npos);
    CHECK(bloom.text.find("struct ArcOut") != std::string::npos);
    // The blanked entries keep their lines, thus both modules map draw.wgsl the same way.
    const cp::Segment* a = segment(arcs, "draw.wgsl");
    const cp::Segment* b = segment(bloom, "draw.wgsl");
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    CHECK(a->lines == b->lines);
    CHECK(a->first_line == b->first_line);
    const std::uint32_t fn_bloom = 54;   // the line of `fn bloom(` in draw.wgsl
    CHECK(line_at(r.files.at("draw.wgsl"), fn_bloom).find("fn bloom(") == 0);
    CHECK(line_at(bloom.text, b->first_line + fn_bloom - 1).find("fn bloom(") == 0);
    CHECK(line_at(arcs.text, a->first_line + fn_bloom - 1).find_first_not_of(' ') == std::string::npos);
}

TEST_CASE("a composed pass is the prelude, the header, the lib modules, common.wgsl and draw.wgsl") {
    const cs::Piece piece = ex::ring_arcs();
    const cp::Resolved r = ex::resolve(piece);
    const cp::Composed c = cp::compose_draw(piece, r, 0);
    REQUIRE(c.verdict.ok());
    std::vector<std::string> parts;
    for (const cp::Segment& s : c.segments) parts.push_back(s.part);
    CHECK(parts == std::vector<std::string>{"prelude", "header", "lib/colormap@1", "lib/pick@1", "lib/camera@1",
                                            "common.wgsl", "draw.wgsl"});
    CHECK(c.text.rfind(std::string(cp::draw_prelude()), 0) == 0);
    // The segments follow each other, and a line of draw.wgsl maps back to its own line.
    std::uint32_t next = 1;
    for (const cp::Segment& s : c.segments) {
        CHECK(s.first_line == next);
        next += s.lines;
    }
    const cp::Segment* draw = segment(c, "draw.wgsl");
    REQUIRE(draw != nullptr);
    CHECK(line_at(c.text, draw->first_line) == "// draw.wgsl");
    CHECK(line_at(c.text, draw->first_line + 21) == line_at(r.files.at("draw.wgsl"), 22));
    const cp::Segment* camera = segment(c, "lib/camera@1");
    REQUIRE(camera != nullptr);
    CHECK(line_at(c.text, camera->first_line) == line_at(std::string(cp::lib_module("camera@1")), 1));
}

TEST_CASE("the field's header has its three params and one counted resource") {
    const cs::Piece piece = ex::attention_field();
    const cp::Composed c = cp::compose_draw(piece, ex::resolve(piece), 0);
    REQUIRE(c.verdict.ok());
    CHECK(header_of(c) ==
          "struct Params { layer: i32, head: i32, gain: f32 }\n"
          "@group(0) @binding(5) var<uniform> params: Params;\n"
          "struct CaliperCounts { pattern: u32 }\n"
          "@group(1) @binding(0) var<uniform> counts: CaliperCounts;\n"
          "@group(1) @binding(1) var<storage, read> pattern: array<f32>;\n");
}

TEST_CASE("draw buffers are read_write only in a compute pass that writes them; buffer inputs follow the resources") {
    cs::Piece piece = ex::ring_arcs();
    cs::PieceInput in;
    in.name = "other";
    in.kind = cs::InputKind::buffer;
    in.layout = ex::fields("Arc", piece.resources[1].layout.fields);
    piece.inputs.push_back(in);
    cs::DrawBuffer trail, age;
    trail.id = "trail";
    trail.layout = ex::fields("Trail", {{"value", "vec4<f32>"}});
    trail.count = 64;
    age.id = "age";
    age.layout = ex::element("u32");
    age.count = 64;
    piece.draw->buffers = {trail, age};
    cs::DrawPass step;
    step.kind = cs::PassKind::compute;
    step.entry = "step";
    step.writes = {"age"};
    piece.draw->passes.insert(piece.draw->passes.begin(), step);
    piece.files["draw.wgsl"] += "\n@compute @workgroup_size(64)\nfn step(@builtin(global_invocation_id) id: vec3<u32>) {}\n";

    const cp::Composed compute = cp::compose_draw(piece, ex::resolve(piece), 0);
    REQUIRE(compute.verdict.ok());
    const std::string h = header_of(compute);
    CHECK(h.find("@group(1) @binding(4) var<storage, read> other: array<Arc>;") != std::string::npos);
    CHECK(h.find("struct CaliperCounts { heads: u32, arcs: u32, labels: u32, other: u32 }") != std::string::npos);
    // The struct Arc of the resource and of the input is declared once.
    CHECK(h.find("struct Arc ") == h.rfind("struct Arc "));
    CHECK(h.find("@group(2) @binding(0) var<storage, read> trail: array<Trail>;") != std::string::npos);
    CHECK(h.find("@group(2) @binding(1) var<storage, read_write> age: array<u32>;") != std::string::npos);
    const std::string render = header_of(cp::compose_draw(piece, ex::resolve(piece), 1));
    CHECK(render.find("@group(2) @binding(1) var<storage, read> age: array<u32>;") != std::string::npos);
}

TEST_CASE("the composer refuses what it cannot compose") {
    const cs::Piece arcs = ex::ring_arcs();
    const cp::Resolved r = ex::resolve(arcs);
    SUBCASE("a pass past the last") {
        const cp::Composed c = cp::compose_draw(arcs, r, 3);
        CHECK(c.verdict.code == cp::kProgramBadPass);
        CHECK(c.verdict.where == "draw.passes[3]");
        CHECK(c.text.empty());
    }
    SUBCASE("a piece without a draw half") {
        cs::Piece p = arcs;
        p.draw.reset();
        CHECK(cp::compose_draw(p, r, 0).verdict.code == cp::kProgramBadPass);
    }
    SUBCASE("a draw half without draw.wgsl") {
        cp::Resolved without = r;
        without.files.erase("draw.wgsl");
        CHECK(cp::compose_draw(arcs, without, 0).verdict.code == cp::kProgramEntryStage);
    }
    SUBCASE("a resolution that failed") {
        cp::Resolved bad = r;
        bad.verdict.code = cp::kPieceUnknownImport;
        CHECK(cp::compose_draw(arcs, bad, 0).verdict.code == cp::kPieceUnknownImport);
    }
    SUBCASE("one struct name with two layouts") {
        cs::Piece p = arcs;
        p.resources[2].layout.type_name = "Arc";
        const cp::Composed c = cp::compose_draw(p, r, 0);
        CHECK(c.verdict.code == cp::kProgramBadLayout);
    }
    SUBCASE("a pass that reads no target") {
        cs::Piece p = arcs;
        p.draw->passes[2].reads = {"nothing"};
        const cp::Composed c = cp::compose_draw(p, r, 2);
        CHECK(c.verdict.code == cp::kProgramUnknownResource);
        CHECK(c.verdict.where == "draw.passes[2]");
    }
}

namespace {

cp::ReflectedIo io(std::uint32_t location, cp::IoScalar scalar, std::uint32_t components,
                   cp::IoInterpolation interpolation = cp::IoInterpolation::perspective) {
    cp::ReflectedIo v;
    v.location = location;
    v.scalar = scalar;
    v.components = components;
    v.interpolation = interpolation;
    v.sampling = interpolation == cp::IoInterpolation::flat ? cp::IoSampling::first : cp::IoSampling::center;
    return v;
}

// The entries of the arcs example as the compiler reflects them, with their inputs and outputs:
// ArcOut is a vec4<f32> color at location 0 and a flat u32 item at location 1, and the prelude's
// CaliperFullscreen a vec2<f32> uv at location 0.
std::vector<cp::ReflectedEntry> arcs_entries() {
    const cp::ReflectedIo color = io(0, cp::IoScalar::f32, 4);
    const cp::ReflectedIo item = io(1, cp::IoScalar::u32, 1, cp::IoInterpolation::flat);
    const cp::ReflectedIo uv = io(0, cp::IoScalar::f32, 2);
    const cp::ReflectedIo id = io(1, cp::IoScalar::u32, 2, cp::IoInterpolation::flat);
    std::vector<cp::ReflectedEntry> e = {{"caliper_fullscreen", cp::EntryStage::vertex, {}},
                                         {"arc_vs", cp::EntryStage::vertex, {}},
                                         {"arc_fs", cp::EntryStage::fragment, {}},
                                         {"arc_glow", cp::EntryStage::fragment, {}},
                                         {"bloom", cp::EntryStage::fragment, {}}};
    e[0].outputs = {uv};
    e[1].outputs = {color, item};
    e[2].inputs = {color, item};
    e[2].outputs = {color, id};
    e[3].inputs = {color, item};
    e[3].outputs = {color};
    e[4].inputs = {uv};
    e[4].outputs = {color};
    return e;
}

}  // namespace

TEST_CASE("the reflection check takes the right entries and refuses the rest") {
    const cs::Piece arcs = ex::ring_arcs();
    cp::Reflection r;
    r.entries = arcs_entries();
    cp::ReflectedBinding counts;
    counts.group = 1;
    counts.binding = 0;
    counts.kind = cp::BindingKind::uniform_buffer;
    counts.read = true;
    counts.name = "counts";
    r.bindings = {counts};
    for (std::size_t k = 0; k < 3; ++k) CHECK(cp::check_piece_reflection(arcs, cp::Half::draw, k, r).ok());

    SUBCASE("an entry of the wrong stage") {
        cp::Reflection bad = r;
        bad.entries[1].stage = cp::EntryStage::fragment;
        const cp::Verdict v = cp::check_piece_reflection(arcs, cp::Half::draw, 0, bad);
        CHECK(v.code == cp::kProgramEntryStage);
        CHECK(v.where == "draw.passes[0]");
    }
    SUBCASE("a missing entry") {
        cp::Reflection bad = r;
        bad.entries.pop_back();
        CHECK(cp::check_piece_reflection(arcs, cp::Half::draw, 2, bad).code == cp::kProgramEntryStage);
    }
    SUBCASE("an entry named main") {
        cp::Reflection bad = r;
        bad.entries.push_back({"main", cp::EntryStage::fragment, {}});
        CHECK(cp::check_piece_reflection(arcs, cp::Half::draw, 0, bad).code == cp::kProgramEntryStage);
    }
    SUBCASE("a write of a read-only binding") {
        cp::Reflection bad = r;
        bad.bindings[0].write = true;
        CHECK(cp::check_piece_reflection(arcs, cp::Half::draw, 0, bad).code == cp::kProgramBindingNotAllowed);
    }
    SUBCASE("a binding in group 3") {
        cp::Reflection bad = r;
        bad.bindings[0].group = 3;
        CHECK(cp::check_piece_reflection(arcs, cp::Half::draw, 0, bad).code == cp::kProgramBindingNotAllowed);
    }
}

TEST_CASE("the reflection check refuses stage interfaces that disagree, before any pipeline") {
    const cs::Piece arcs = ex::ring_arcs();
    cp::Reflection r;
    r.entries = arcs_entries();
    for (std::size_t k = 0; k < 3; ++k) CHECK(cp::check_piece_reflection(arcs, cp::Half::draw, k, r).ok());
    auto code = [&](const cp::Reflection& bad, std::size_t pass) {
        const cp::Verdict v = cp::check_piece_reflection(arcs, cp::Half::draw, pass, bad);
        CHECK(v.where == "draw.passes[" + std::to_string(pass) + "]");
        return v.code;
    };

    SUBCASE("a fragment input that the vertex entry does not write") {
        cp::Reflection bad = r;
        bad.entries[3].inputs.push_back(io(2, cp::IoScalar::f32, 1));
        CHECK(code(bad, 1) == cp::kProgramInterfaceMismatch);
    }
    SUBCASE("a fragment input of another type") {
        cp::Reflection bad = r;
        bad.entries[4].inputs[0].components = 3;
        CHECK(code(bad, 2) == cp::kProgramInterfaceMismatch);
        bad = r;
        bad.entries[2].inputs[1].scalar = cp::IoScalar::i32;
        CHECK(code(bad, 0) == cp::kProgramInterfaceMismatch);
    }
    SUBCASE("a fragment input with another interpolation") {
        cp::Reflection bad = r;
        bad.entries[3].inputs[0].interpolation = cp::IoInterpolation::linear;
        CHECK(code(bad, 1) == cp::kProgramInterfaceMismatch);
        bad = r;
        bad.entries[3].inputs[0].sampling = cp::IoSampling::centroid;
        CHECK(code(bad, 1) == cp::kProgramInterfaceMismatch);
    }
    SUBCASE("a vertex attribute, which a piece cannot feed") {
        cp::Reflection bad = r;
        bad.entries[1].inputs.push_back(io(0, cp::IoScalar::f32, 3));
        CHECK(code(bad, 0) == cp::kProgramInterfaceMismatch);
    }
    SUBCASE("more inter-stage variables than WebGPU allows") {
        cp::Reflection bad = r;
        bad.entries[1].outputs.clear();
        for (std::uint32_t l = 0; l < 17; ++l) bad.entries[1].outputs.push_back(io(l, cp::IoScalar::f32, 1));
        CHECK(code(bad, 0) == cp::kProgramOverWebLimits);
        bad = r;
        bad.entries[1].outputs.push_back(io(16, cp::IoScalar::f32, 1));
        CHECK(code(bad, 0) == cp::kProgramOverWebLimits);
    }
    SUBCASE("a pick pass whose fragment does not return CaliperPicked") {
        cp::Reflection bad = r;
        bad.entries[2].outputs.pop_back();
        CHECK(code(bad, 0) == cp::kProgramInterfaceMismatch);
        bad = r;
        bad.entries[2].outputs[1].scalar = cp::IoScalar::f32;
        CHECK(code(bad, 0) == cp::kProgramInterfaceMismatch);
    }
    SUBCASE("a target pass whose fragment does not return one vec4<f32>") {
        cp::Reflection bad = r;
        bad.entries[3].outputs[0].components = 3;
        CHECK(code(bad, 1) == cp::kProgramInterfaceMismatch);
        bad = r;
        bad.entries[3].outputs.push_back(io(1, cp::IoScalar::u32, 2, cp::IoInterpolation::flat));
        CHECK(code(bad, 1) == cp::kProgramInterfaceMismatch);
        bad = r;
        bad.entries[3].outputs.clear();
        CHECK(code(bad, 1) == cp::kProgramInterfaceMismatch);
    }
    SUBCASE("a host pass without pick that writes an ID") {
        cp::Reflection bad = r;
        bad.entries[4].outputs.push_back(io(1, cp::IoScalar::u32, 2, cp::IoInterpolation::flat));
        CHECK(code(bad, 2) == cp::kProgramInterfaceMismatch);
    }
}

TEST_CASE("the reflection check holds compute entries to WebGPU's workgroup limits") {
    cs::Piece p = ex::ring_arcs();
    cs::DrawBuffer age;
    age.id = "age";
    age.layout = ex::element("u32");
    age.count = 64;
    p.draw->buffers = {age};
    cs::DrawPass step;
    step.kind = cs::PassKind::compute;
    step.entry = "step";
    step.writes = {"age"};
    p.draw->passes = {step};
    cp::Reflection r;
    r.entries = {{"step", cp::EntryStage::compute, {64, 1, 1}}};
    cp::ReflectedBinding age_binding;
    age_binding.group = 2;
    age_binding.binding = 0;
    age_binding.write = true;
    age_binding.name = "age";
    r.bindings = {age_binding};
    CHECK(cp::check_piece_reflection(p, cp::Half::draw, 0, r).ok());
    r.entries[0].workgroup_size = {16, 16, 2};
    CHECK(cp::check_piece_reflection(p, cp::Half::draw, 0, r).code == cp::kProgramOverWebLimits);
    r.entries[0].workgroup_size = {1, 1, 65};
    CHECK(cp::check_piece_reflection(p, cp::Half::draw, 0, r).code == cp::kProgramOverWebLimits);
    r.entries[0].workgroup_size = {64, 1, 1};
    p.draw->passes[0].writes.clear();
    CHECK(cp::check_piece_reflection(p, cp::Half::draw, 0, r).code == cp::kProgramBindingNotAllowed);
}

TEST_CASE("the reflection check of a derive pass keeps the tensor inputs read-only") {
    const cs::Piece arcs = ex::ring_arcs();
    cp::Reflection r;
    r.entries = {{"scores", cp::EntryStage::compute, {8, 1, 1}}, {"build", cp::EntryStage::compute, {64, 1, 1}}};
    cp::ReflectedBinding attn0;
    attn0.group = 1;
    attn0.binding = 0;
    attn0.read = true;
    attn0.name = "attn0";
    r.bindings = {attn0};
    CHECK(cp::check_piece_reflection(arcs, cp::Half::derive, 0, r).ok());
    CHECK(cp::check_piece_reflection(arcs, cp::Half::derive, 2, r).code == cp::kProgramEntryStage);
    r.bindings[0].write = true;
    const cp::Verdict v = cp::check_piece_reflection(arcs, cp::Half::derive, 1, r);
    CHECK(v.code == cp::kProgramBindingNotAllowed);
    CHECK(v.where == "derive.passes[1]");
}

TEST_CASE("the compiler's reflection refuses a fragment that reads what its vertex entry does not write") {
    if (!cp::compiler_available()) {
        MESSAGE("this build has no WGSL compiler");
        return;
    }
    // The field example's one pass draws the full-screen triangle, whose CaliperFullscreen gives
    // a vec2<f32> uv at location 0. A fragment that reads a vec3<f32> there, or a location the
    // triangle does not write, is refused before any pipeline; the example itself passes.
    const cs::Piece field = ex::attention_field();
    const std::string fragment = field.draw->passes[0].fragment;
    auto verdict = [&](const std::string& draw_wgsl) {
        cs::Piece p = field;
        p.files["draw.wgsl"] = draw_wgsl;
        const cp::Composed c = cp::compose_draw(p, ex::resolve(p), 0);
        REQUIRE(c.verdict.ok());
        const cp::Compiled compiled = cp::compile_and_reflect(c.text);
        REQUIRE_MESSAGE(compiled.verdict.ok(), compiled.verdict.message);
        return cp::check_piece_reflection(p, cp::Half::draw, 0, compiled.reflection);
    };
    CHECK(verdict(field.files.at("draw.wgsl")).ok());
    const std::string head = "@fragment\nfn " + fragment;
    CHECK(verdict(head + "(@location(0) uv: vec3<f32>) -> @location(0) vec4<f32> { return vec4<f32>(uv, 1.0); }\n")
              .code == cp::kProgramInterfaceMismatch);
    CHECK(verdict(head + "(@location(3) t: f32) -> @location(0) vec4<f32> { return vec4<f32>(t); }\n").code ==
          cp::kProgramInterfaceMismatch);
    CHECK(verdict(head + "(@location(0) uv: vec2<f32>) -> @location(0) vec2<f32> { return uv; }\n").code ==
          cp::kProgramInterfaceMismatch);
    CHECK(verdict(head + "(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return vec4<f32>(uv, 0.0, 1.0); }\n")
              .ok());
}

TEST_CASE("every composed pass of both examples compiles with the bounds-checked compiler") {
    if (!cp::compiler_available()) {
        MESSAGE("this build has no WGSL compiler");
        return;
    }
    for (const cs::Piece& piece : {ex::attention_field(), ex::ring_arcs()}) {
        for (std::size_t k = 0; k < piece.draw->passes.size(); ++k) {
            CAPTURE(piece.name);
            CAPTURE(k);
            const cp::Composed c = cp::compose_draw(piece, ex::resolve(piece), k);
            REQUIRE(c.verdict.ok());
            const cp::Compiled compiled = cp::compile_and_reflect(c.text);
            CHECK_MESSAGE(compiled.verdict.ok(), compiled.verdict.message);
            if (compiled.verdict.ok()) CHECK(cp::check_piece_reflection(piece, cp::Half::draw, k, compiled.reflection).ok());
        }
    }
}
