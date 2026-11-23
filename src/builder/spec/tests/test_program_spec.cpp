// Tests of the programs, stages and provenance of spec version 1 (program.h,
// design/gpu-programs.md, Spec Additions) and of the toggle control: the
// codecs read and write them canonically, the validator holds them to the
// limits of program.h with the sandbox's codes, and an older document reads
// as before. Then the inline pieces and the instances of pieces
// (design/programmable-graphics.md, Pieces and Program Instances), and one
// piece alone in its JSON and piece.toml forms.
#include <doctest/doctest.h>

#include <caliper_builder/spec/instrument.h>

#include "helpers.h"

using namespace spec_test;
using nlohmann::json;

namespace {

cs::ParseResult programs_fixture() {
    cs::ParseResult r = cs::parse_toml(fixture("programs.spec.toml"));
    REQUIRE_MESSAGE(r.ok(), dump(r.diagnostics));
    return r;
}

// A version 1 document with one scene panel `world`, a series panel `chart`
// and the given TOML after them.
std::string with_panels(const std::string& rest) {
    return "caliper_spec = 1\n[applet]\nid = \"a\"\n"
           "[[source]]\nid = \"s\"\ncatalog = \"live:loss\"\n"
           "[[panel]]\nid = \"world\"\nprimitive = \"scene\"\nbind = { scene = \"live:model\" }\n"
           "[[panel]]\nid = \"chart\"\nprimitive = \"series\"\nbind = { y = \"s\" }\n" +
           rest;
}

const char* const kProgram =
    "[[program]]\nid = \"p\"\ninputs = [{ var = \"w\", source = \"live:w\" }]\n"
    "output = { kind = \"geometry\", panel = \"world\", max_count = 8, arrays = [\"positions\"] }\n"
    "wgsl = \"@compute @workgroup_size(8) fn main() {}\"\n";

}  // namespace

TEST_CASE("a version 1 spec with programs, a stage and provenance parses into the expected spec") {
    const cs::Spec s = programs_fixture().spec;
    CHECK(s.version == 1);
    REQUIRE(s.programs.size() == 2);
    const cs::Program& rows = s.programs[0];
    CHECK(rows.id == "rows");
    CHECK(rows.title == std::optional<std::string>("Rows of the first layer"));
    REQUIRE(rows.inputs.size() == 1);
    CHECK(rows.inputs[0] == cs::ProgramInput{"w", "live:layers.0.weight"});
    CHECK(rows.output.kind == cs::OutputKind::geometry);
    CHECK(rows.output.panel == "world");
    CHECK(rows.output.entity == std::optional<std::string>("programs/rows"));
    CHECK(rows.output.max_count == 128);
    CHECK(rows.output.arrays == std::vector<cs::GeometryArray>{cs::GeometryArray::positions, cs::GeometryArray::values});
    CHECK(rows.output.style.colormap == std::optional<std::string>("magma"));
    CHECK(rows.output.style.vmin == std::optional<double>(-1.0));
    CHECK(rows.limits.every == std::optional<std::uint32_t>(10));
    CHECK(rows.limits.time_ms == std::optional<double>(2.5));
    CHECK_FALSE(rows.limits.memory_mb);
    CHECK(rows.wgsl.rfind("@group(0) @binding(0) var<storage, read> w", 0) == 0);
    CHECK(rows.wgsl.back() == '\n');
    const cs::Program& image = s.programs[1];
    CHECK(image.output.kind == cs::OutputKind::pixels);
    CHECK(image.output.width == 64);
    CHECK(image.output.height == 32);
    CHECK(image.workgroups == std::optional<std::array<std::uint32_t, 3>>(std::array<std::uint32_t, 3>{8, 4, 1}));
    REQUIRE(s.stages.size() == 1);
    CHECK(s.stages[0].kind == cs::StageKind::material);
    CHECK(cs::stage_id(s.stages[0]) == "world/programs/rows/material");
    CHECK(s.provenance.foreign);
    REQUIRE(s.provenance.reviewed.size() == 1);
    CHECK(s.provenance.reviewed[0].target == "program:rows");
    CHECK(s.find_program("image") == &s.programs[1]);
    CHECK(cs::required_version(s) == 1);
    const cs::ValidationReport report = cs::validate(s);
    CHECK_MESSAGE(report.ok(), dump(report.diagnostics));
}

TEST_CASE("programs, stages and provenance round trip through canonical TOML and canonical JSON") {
    const cs::Spec first = programs_fixture().spec;
    const std::string toml = cs::to_toml(first);
    CHECK(toml.find("\n[[program]]\nid = \"rows\"\n") != std::string::npos);
    CHECK(toml.find("[[program]]") < toml.find("[[stage]]"));
    CHECK(toml.find("[[stage]]") < toml.find("[provenance]"));
    CHECK(toml.find("wgsl = \"\"\"\n@group(0)") != std::string::npos);
    auto second = cs::parse_toml(toml);
    REQUIRE_MESSAGE(second.ok(), dump(second.diagnostics));
    CHECK(second.spec == first);
    CHECK(cs::to_toml(second.spec) == toml);

    const std::string text = cs::to_json_text(first);
    auto back = cs::from_json_text(text);
    REQUIRE_MESSAGE(back.ok(), dump(back.diagnostics));
    CHECK(back.spec == first);
    CHECK(cs::to_json_text(back.spec) == text);
    const json j = json::parse(text);
    CHECK(j.at("program")[0].at("output").at("part") == "points");
    CHECK(j.at("program")[1].at("output") == json::parse(R"({"kind": "pixels", "width": 64, "height": 32, "format": "rgba8unorm"})"));
    CHECK_FALSE(j.at("program")[1].contains("title"));
    CHECK(j.at("provenance").at("foreign") == true);

    // The item codecs give the same program.
    auto one = cs::program_from_json(j.at("program")[0]);
    REQUIRE_MESSAGE(one.ok(), dump(one.diagnostics));
    CHECK(one.value == first.programs[0]);
    auto st = cs::stage_from_json(j.at("stage")[0]);
    REQUIRE(st.ok());
    CHECK(st.value == first.stages[0]);
}

TEST_CASE("the reader refuses the keys of programs in version 0 and the wrong keys in version 1") {
    const std::string v0 = "caliper_spec = 0\n[applet]\nid = \"a\"\n";
    CHECK(has(cs::parse_toml(v0 + kProgram).diagnostics, "spec.unknown_key", "program"));
    CHECK(has(cs::parse_toml(v0 + "[provenance]\nforeign = true\n").diagnostics, "spec.unknown_key", "provenance"));
    CHECK(cs::parse_toml(with_panels(kProgram)).ok());

    struct Case {
        const char* toml;
        const char* code;
        const char* path;
    };
    const Case cases[] = {
        {"[[program]]\nid = \"p\"\ninputs = [{ var = \"w\", source = \"live:w\", x = { a = 1 } }]\n"
         "output = { kind = \"pixels\", width = 4, height = 4 }\nwgsl = \"x\"\n",
         "spec.unknown_key", "program[0].inputs[0].x"},
        {"[[program]]\nid = \"p\"\ninputs = []\noutput = { kind = \"pixels\", width = 4, height = 4, panel = \"world\" }\n"
         "wgsl = \"x\"\n",
         "spec.unknown_key", "program[0].output.panel"},
        {"[[program]]\nid = \"p\"\ninputs = []\noutput = { kind = \"geometry\", panel = \"world\", arrays = [\"positions\"] }\n"
         "wgsl = \"x\"\n",
         "spec.missing_key", "program[0].output.max_count"},
        {"[[program]]\nid = \"p\"\ninputs = []\noutput = { kind = \"mesh\" }\nwgsl = \"x\"\n", "spec.bad_value",
         "program[0].output.kind"},
        {"[[program]]\nid = \"p\"\noutput = { kind = \"pixels\", width = 4, height = 4 }\nwgsl = \"x\"\n",
         "spec.missing_key", "program[0].inputs"},
        {"[[program]]\nid = \"p\"\ninputs = []\noutput = { kind = \"pixels\", width = -4, height = 4 }\nwgsl = \"x\"\n",
         "spec.out_of_range", "program[0].output.width"},
        {"[[program]]\nid = \"p\"\ninputs = []\noutput = { kind = \"pixels\", width = 4, height = 4 }\n"
         "workgroups = [1, 2]\nwgsl = \"x\"\n",
         "spec.wrong_type", "program[0].workgroups"},
        {"[[stage]]\npanel = \"world\"\nentity = \"e\"\nkind = \"compute\"\nwgsl = \"x\"\n", "spec.bad_value",
         "stage[0].kind"},
        {"[provenance]\nfrom = \"x\"\n", "spec.missing_key", "provenance.foreign"},
    };
    for (const Case& c : cases) {
        CAPTURE(c.toml);
        const auto r = cs::parse_toml(with_panels(c.toml));
        CHECK_MESSAGE(has(r.diagnostics, c.code, c.path), dump(r.diagnostics));
    }
}

TEST_CASE("the validator holds programs and stages to the limits of program.h and to the spec's panels") {
    struct Case {
        std::string toml;
        const char* code;
        const char* path;
    };
    const std::string pixels = "output = { kind = \"pixels\", width = 4, height = 4 }\nwgsl = \"x\"\n";
    const Case cases[] = {
        {std::string(kProgram) + kProgram, "spec.duplicate_id", "program[1].id"},
        {"[[program]]\nid = \"p\"\ninputs = []\noutput = { kind = \"geometry\", panel = \"chart\", max_count = 8, "
         "arrays = [\"positions\"] }\nwgsl = \"x\"\n",
         "program.bad_target", "program[0].output.panel"},
        {"[[program]]\nid = \"p\"\ninputs = []\noutput = { kind = \"geometry\", panel = \"nowhere\", max_count = 8, "
         "arrays = [\"positions\"] }\nwgsl = \"x\"\n",
         "spec.unknown_panel", "program[0].output.panel"},
        {"[[program]]\nid = \"p\"\ninputs = []\noutput = { kind = \"geometry\", panel = \"world\", max_count = 2000000, "
         "arrays = [\"positions\"] }\nwgsl = \"x\"\n",
         "program.over_limits", "program[0].output.max_count"},
        {"[[program]]\nid = \"p\"\ninputs = []\nlimits = { every = 0 }\n" + pixels, "program.over_limits",
         "program[0].limits.every"},
        // A program id of 56 bytes: its live view program.<id> would pass the wire's name limit.
        {"[[program]]\nid = \"" + std::string(56, 'p') + "\"\ninputs = []\n" + pixels, "spec.bad_value",
         "program[0].id"},
        {"[[program]]\nid = \"p\"\ninputs = [{ var = \"w\", source = \"w\" }]\n" + pixels, "program.over_limits",
         "program[0].inputs[0].source"},
        {"[[stage]]\npanel = \"chart\"\nentity = \"e\"\nkind = \"material\"\nwgsl = \"x\"\n", "stage.not_scene_panel",
         "stage[0].panel"},
        {"[[stage]]\npanel = \"world\"\nentity = \"e\"\nkind = \"vertex\"\nwgsl = \"x\"\n"
         "[[stage]]\npanel = \"world\"\nentity = \"e\"\nkind = \"vertex\"\nwgsl = \"y\"\n",
         "spec.duplicate_id", "stage[1]"},
        {"[[stage]]\npanel = \"world\"\nentity = \"a//b\"\nkind = \"vertex\"\nwgsl = \"x\"\n", "program.bad_target",
         "stage[0].entity"},
        {"[provenance]\nforeign = true\nreviewed = [{ target = \"panel:world\", sha256 = \"" + std::string(64, 'a') +
             "\" }]\n",
         "spec.bad_value", "provenance.reviewed[0].target"},
        {"[provenance]\nforeign = true\nreviewed = [{ target = \"program:p\", sha256 = \"ABC\" }]\n", "spec.bad_value",
         "provenance.reviewed[0].sha256"},
    };
    for (const Case& c : cases) {
        CAPTURE(c.toml);
        const cs::ValidationReport r = cs::check_toml(with_panels(c.toml));
        CHECK_MESSAGE(has(r.diagnostics, c.code, c.path), dump(r.diagnostics));
    }
    // At most 16 programs.
    std::string many;
    for (int i = 0; i < 17; ++i)
        many += "[[program]]\nid = \"p" + std::to_string(i) + "\"\ninputs = []\n" + pixels;
    CHECK(has(cs::check_toml(with_panels(many)).diagnostics, "spec.too_many", "program"));
    // A stage of another kind on the same entity is a second stage.
    const std::string two = "[[stage]]\npanel = \"world\"\nentity = \"e\"\nkind = \"vertex\"\nwgsl = \"x\"\n"
                            "[[stage]]\npanel = \"world\"\nentity = \"e\"\nkind = \"material\"\nwgsl = \"y\"\n";
    const cs::ValidationReport ok = cs::check_toml(with_panels(two));
    CHECK_MESSAGE(ok.ok(), dump(ok.diagnostics));
}

TEST_CASE("a toggle binds an intervention point or a parameter, and takes no options") {
    CHECK(cs::control_kind_built(cs::ControlKind::toggle));
    CHECK(cs::control_kind_built(cs::ControlKind::picker));
    CHECK(cs::control_kind_built(cs::ControlKind::transport));
    const auto t = cs::parse_control_target("intervene:blocks.0.attn#3");
    REQUIRE(t);
    CHECK(t->kind == cs::ControlTarget::Kind::intervene);
    CHECK(t->name == "blocks.0.attn#3");
    CHECK(cs::control_target_string(*t) == "intervene:blocks.0.attn#3");
    CHECK(cs::parse_control_target("intervene:" + std::string(255, 'a')));
    CHECK_FALSE(cs::parse_control_target("intervene:" + std::string(256, 'a')));
    CHECK_FALSE(cs::parse_control_target("intervene:"));
    CHECK(cs::control_takes_target(cs::ControlKind::toggle, *t));
    CHECK_FALSE(cs::control_takes_target(cs::ControlKind::slider, *t));
    CHECK_FALSE(cs::control_takes_target(cs::ControlKind::button, *t));

    const std::string head = "caliper_spec = 1\n[applet]\nid = \"a\"\n[[control]]\nid = \"k\"\nkind = \"toggle\"\n";
    for (const char* bind : {"intervene:blocks.0.attn", "param:use_bias"}) {
        CAPTURE(bind);
        const cs::ValidationReport r = cs::check_toml(head + "bind = \"" + bind + "\"\n");
        CHECK_MESSAGE(r.ok(), dump(r.diagnostics));
    }
    CHECK(has(cs::check_toml(head + "bind = \"run:pause\"\n").diagnostics, "spec.control_mismatch", "control[0].bind"));
    CHECK(has(cs::check_toml(head + "bind = \"param:x\"\noptions = { min = 0 }\n").diagnostics, "spec.control_mismatch",
              "control[0].options.min"));
    CHECK(has(cs::check_toml("caliper_spec = 1\n[applet]\nid = \"a\"\n[[control]]\nid = \"k\"\nkind = \"slider\"\n"
                             "bind = \"intervene:blocks.0\"\n")
                  .diagnostics,
              "spec.control_mismatch", "control[0].bind"));
    CHECK(has(cs::check_toml("caliper_spec = 1\n[applet]\nid = \"a\"\n[[control]]\nid = \"k\"\nkind = \"transport\"\n"
                             "bind = \"value\"\n")
                  .diagnostics,
              "spec.control_mismatch", "control[0].bind"));
}

TEST_CASE("every older fixture reads as before and has no programs") {
    for (const std::string& name : valid_fixtures()) {
        CAPTURE(name);
        const auto r = cs::parse_toml(fixture(name));
        REQUIRE(r.ok());
        CHECK(r.spec.programs.empty());
        CHECK(r.spec.stages.empty());
        CHECK(r.spec.provenance.empty());
        CHECK(cs::to_json(r.spec).count("program") == 0);
    }
    const auto v1 = cs::parse_toml(fixture("instruments.spec.toml"));
    REQUIRE(v1.ok());
    CHECK(v1.spec.programs.empty());
    CHECK(cs::validate(v1.spec).ok());
}

namespace {

// The piece attention_field of the plan, inline, and its instance in `world`.
const char* const kFieldPiece = R"toml(
[[program]]
id = "head_pattern"
piece = { name = "attention_field", version = 1, from = "inline" }
panel = "world"
bind = { attn0 = "live:blocks.0.attn.weights.act", attn1 = "live:blocks.1.attn.weights.act" }
params = { layer = 1, head = { state = "head" }, gain = 2.0 }

[[state]]
id = "head"
type = "integer"
default = 0

[[piece]]
name = "attention_field"
version = 1
description = "The attention pattern of one head of a two-layer model, as a field with contour lines"
imports = ["colormap@1"]

[[piece.inputs]]
name = "attn0"
kind = "tensor"
rank_min = 4
rank_max = 4

[[piece.inputs]]
name = "attn1"
kind = "tensor"
rank_min = 4
rank_max = 4

[[piece.params]]
name = "head"
type = "i32"
min = 0
max = 3

[[piece.params]]
name = "gain"
default = 1.0
min = 0.1
max = 10.0

[[piece.resources]]
id = "pattern"
kind = "buffer"
element = "f32"
count = 8192

[piece.derive]
passes = [{ entry = "copy_rows", workgroups = [16, 1, 1], writes = ["pattern"] }]

[piece.draw]
camera = "none"
passes = [{ kind = "render", fragment = "field" }]

[piece.files]
common = """
const H: u32 = 4u;
const T: u32 = 32u;
"""
derive = """
@compute @workgroup_size(256)
fn copy_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= H * T * T) { return; }
    pattern[id.x] = attn0[id.x];
}
"""
draw = """
@fragment
fn field(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    return colormap::magma(clamp(pattern[0] * params.gain, 0.0, 1.0));
}
"""
)toml";

cs::ParseResult field_spec(const std::string& extra = "") {
    return cs::parse_toml(with_panels(kFieldPiece + extra));
}

}  // namespace

TEST_CASE("a spec reads and writes inline pieces and the pin, bindings, panel and params of an instance") {
    const auto r = field_spec();
    REQUIRE_MESSAGE(r.ok(), dump(r.diagnostics));
    REQUIRE(r.spec.pieces.size() == 1);
    const cs::Piece& piece = r.spec.pieces[0];
    CHECK(piece.name == "attention_field");
    CHECK(piece.imports == std::vector<std::string>{"colormap@1"});
    CHECK(piece.inputs.size() == 2);
    CHECK(piece.params[0].type == cs::ParamType::i32);
    CHECK(piece.resources[0].count == 8192);
    REQUIRE(piece.derive);
    CHECK(piece.derive->passes[0].workgroups == std::array<std::uint32_t, 3>{16, 1, 1});
    REQUIRE(piece.draw);
    CHECK(piece.draw->camera == cs::CameraMode::none);
    CHECK(piece.files.at("draw.wgsl").find("colormap::magma") != std::string::npos);
    const cs::Program* p = r.spec.find_program("head_pattern");
    REQUIRE(p);
    REQUIRE(cs::is_piece_instance(*p));
    CHECK(p->piece->from == cs::PieceFrom::inline_piece);
    CHECK(p->piece->hash.empty());
    CHECK(p->panel == std::optional<std::string>("world"));
    CHECK(p->bind.at("attn1") == "live:blocks.1.attn.weights.act");
    CHECK(p->params.size() == 3);
    const auto v = cs::validate(r.spec);
    CHECK_MESSAGE(v.ok(), dump(v.diagnostics));

    const std::string text = cs::to_toml(r.spec);
    INFO(text);
    const auto again = cs::parse_toml(text);
    REQUIRE_MESSAGE(again.ok(), dump(again.diagnostics));
    CHECK(again.spec == r.spec);
    CHECK(text.find("[[piece.inputs]]") != std::string::npos);
    CHECK(text.find("[piece.files]") != std::string::npos);
    const auto from_json = cs::from_json(cs::to_json(r.spec));
    REQUIRE(from_json.ok());
    CHECK(from_json.spec == r.spec);
    CHECK(cs::to_json(r.spec)["program"][0]["piece"] ==
          json{{"name", "attention_field"}, {"version", 1}, {"from", "inline"}});
}

TEST_CASE("a program with a pin and a key of the one-pass form is program.form_mixed") {
    const auto r = cs::parse_toml(with_panels(
        "[[program]]\nid = \"p\"\npiece = { name = \"points\" }\nwgsl = \"fn f() {}\"\noutput = { kind = \"tensor\" }\n"));
    CHECK(has(r.diagnostics, "program.form_mixed", "program[0].wgsl"));
    CHECK(has(r.diagnostics, "program.form_mixed", "program[0].output"));
    CHECK(!has(r.diagnostics, "spec.unknown_key"));
    const auto none = cs::parse_toml(with_panels("[[program]]\nid = \"p\"\npanel = \"world\"\n"));
    CHECK(has(none.diagnostics, "spec.missing_key", "program[0].piece"));
    const auto pin = cs::parse_toml(with_panels("[[program]]\nid = \"p\"\npiece = { name = \"points\", from = \"disk\", path = \"x\" }\n"));
    CHECK(has(pin.diagnostics, "spec.bad_value", "program[0].piece.from"));
    CHECK(has(pin.diagnostics, "spec.unknown_key", "program[0].piece.path"));
}

TEST_CASE("the validator holds an instance's pin, bindings and panel to their forms") {
    auto check = [](const std::string& program, const char* code, const char* path) {
        const auto r = cs::parse_toml(with_panels(program));
        REQUIRE_MESSAGE(r.ok(), dump(r.diagnostics));
        const auto v = cs::validate(r.spec);
        CAPTURE(program);
        CHECK_MESSAGE(has(v.diagnostics, code, path), dump(v.diagnostics));
    };
    check("[[program]]\nid = \"p\"\npiece = { name = \"Points\" }\n", "piece.bad_name", "program[0].piece.name");
    check("[[program]]\nid = \"p\"\npiece = { name = \"points\", version = 0 }\n", "piece.bad_name",
          "program[0].piece.version");
    check("[[program]]\nid = \"p\"\npiece = { name = \"points\", hash = \"md5:1\" }\n", "spec.bad_value",
          "program[0].piece.hash");
    check("[[program]]\nid = \"p\"\npiece = { name = \"points\" }\npanel = \"chart\"\n", "program.bad_target",
          "program[0].panel");
    check("[[program]]\nid = \"p\"\npiece = { name = \"points\" }\npanel = \"nowhere\"\n", "spec.unknown_panel",
          "program[0].panel");
    check("[[program]]\nid = \"p\"\npiece = { name = \"points\" }\nbind = { positions = \"file:/etc/passwd\" }\n",
          "piece.bad_binding", "program[0].bind.positions");
    check("[[program]]\nid = \"p\"\npiece = { name = \"points\" }\nbind = { arcs = \"program:q/arcs\" }\n",
          "piece.bad_binding", "program[0].bind.arcs");
    check("[[program]]\nid = \"p\"\npiece = { name = \"points\" }\nbind = { arcs = \"program:p/arcs\" }\n",
          "piece.cycle", "program[0].bind.arcs");
    check("[[program]]\nid = \"p\"\npiece = { name = \"points\" }\nbind = { Arcs = \"live:x\" }\n",
          "piece.bad_binding", "program[0].bind.Arcs");
    // A pin to a piece that the spec lacks still validates: the program is refused when it runs.
    const auto r = cs::parse_toml(with_panels("[[program]]\nid = \"p\"\npiece = { name = \"gone\", from = \"inline\" }\n"));
    REQUIRE(r.ok());
    CHECK(cs::validate(r.spec).ok());
}

TEST_CASE("the validator checks each inline piece alone with the codes of pieces") {
    auto check = [](const std::string& from, const std::string& to, const char* code, const char* path) {
        std::string text = kFieldPiece;
        const std::size_t at = text.find(from);
        REQUIRE(at != std::string::npos);
        text.replace(at, from.size(), to);
        const auto r = cs::parse_toml(with_panels(text));
        REQUIRE_MESSAGE(r.ok(), dump(r.diagnostics));
        const auto v = cs::validate(r.spec);
        CAPTURE(to);
        CHECK_MESSAGE(has(v.diagnostics, code, path), dump(v.diagnostics));
    };
    check("writes = [\"pattern\"]", "writes = [\"patterns\"]", "program.unknown_resource",
          "piece[0].derive.passes[0].writes[0]");
    check("count = 8192", "count = 0", "program.bad_resource", "piece[0].resources[0].count");
    check("imports = [\"colormap@1\"]", "imports = [\"colormap@9\"]", "piece.unknown_import", "piece[0].imports[0]");
    check("colormap::magma", "colormap::plasma", "piece.unknown_import", "piece[0].files.draw");
    check("@fragment\nfn field", "@group(0) @binding(9) var<uniform> u: f32;\n@fragment\nfn field",
          "program.binding_not_allowed", "piece[0].files.draw");
    check("const T: u32", "const caliper_T: u32", "program.reserved_name", "piece[0].files.common");
    check("camera = \"none\"\npasses = [{ kind = \"render\", fragment = \"field\" }]",
          "passes = [{ kind = \"render\", fragment = \"field\", target = \"glow\" }]", "program.bad_pass",
          "piece[0].draw.passes[0].target");
    check("name = \"attention_field\"\nversion = 1\ndescription",
          "name = \"attention_field\"\nversion = 1\n[[piece]]\nname = \"attention_field\"\nversion = 1\n"
          "[piece.draw]\npasses = [{ kind = \"render\", fragment = \"f\" }]\n[piece.files]\ndraw = \"x\"\n"
          "[[piece]]\nname = \"attention_field\"\nversion = 1\ndescription",
          "spec.duplicate_id", "piece[1].name");
    check("panel = \"world\"\nbind", "bind", "spec.missing_key", "program[0].panel");
}

TEST_CASE("one piece reads and writes as JSON and as piece.toml with the same hash") {
    const auto r = field_spec();
    REQUIRE(r.ok());
    const cs::Piece& piece = r.spec.pieces[0];
    const std::string toml = cs::piece_to_toml(piece);
    INFO(toml);
    CHECK(toml.rfind("name = \"attention_field\"\nversion = 1\n", 0) == 0);
    CHECK(toml.find("[[inputs]]") != std::string::npos);
    CHECK(toml.find("[draw]") != std::string::npos);
    CHECK(toml.find("files") == std::string::npos);
    const cs::PieceParse back = cs::piece_from_toml(toml);
    REQUIRE_MESSAGE(back.ok(), dump(back.diagnostics));
    cs::Piece expected = piece;
    expected.files.clear();
    CHECK(back.piece == expected);
    const cs::PieceParse js = cs::piece_from_json_value(cs::to_json(piece));
    REQUIRE_MESSAGE(js.ok(), dump(js.diagnostics));
    CHECK(js.piece == piece);
    CHECK(cs::to_json(piece)["files"]["draw"] == piece.files.at("draw.wgsl"));
    // The key optional of an input survives both forms.
    cs::Piece loose = piece;
    loose.inputs[0].optional = true;
    const cs::PieceParse loose_toml = cs::piece_from_toml(cs::piece_to_toml(loose));
    REQUIRE_MESSAGE(loose_toml.ok(), dump(loose_toml.diagnostics));
    CHECK(loose_toml.piece.inputs[0].optional);
    CHECK(cs::piece_from_json_value(cs::to_json(loose)).piece == loose);

    const cs::PieceParse bad = cs::piece_from_toml("name = \"x\"\n\n[draw]\npasses = [{ kind = \"render\", colour = 1 }]\n");
    CHECK(has(bad.diagnostics, "spec.unknown_key", "draw.passes[0].colour"));
    const cs::Diagnostic* d = find(bad.diagnostics, "spec.unknown_key");
    REQUIRE(d);
    CHECK(d->line == 4);
    CHECK(has(cs::piece_from_toml("version = 2\n").diagnostics, "spec.missing_key", "name"));
    CHECK(has(cs::piece_from_toml("name = \"x\"\nfiles = { draw = 1 }\n").diagnostics, "spec.wrong_type", "files.draw"));
    CHECK(has(cs::piece_from_toml("name = \"x\"\n[[inputs]]\nname = \"a\"\nkind = \"tensor\"\nfields = [\"a f32\"]\n")
                  .diagnostics,
              "spec.bad_value", "inputs[0].fields[0]"));
    CHECK(has(cs::piece_from_toml("name = \"x\"\n[[sample]]\ninput = \"a\"\nshape = [4]\ndata = \"AAAA\"\n").diagnostics,
              "spec.unknown_key", "sample[0].data"));
}

#include "schema_check.h"

TEST_CASE("the schema of version 1 takes inline pieces and instances, and refuses a mixed program") {
    const SchemaChecker checker(
        json::parse(read_file(std::string(CALIPER_SPEC_FIXTURES) + "/../../schema/caliper-spec-v1.schema.json")));
    const auto r = field_spec();
    REQUIRE(r.ok());
    json doc = cs::to_json(r.spec);
    std::vector<std::string> errors = checker.errors(doc);
    std::string all;
    for (const std::string& e : errors) all += e + "\n";
    CHECK_MESSAGE(errors.empty(), all);
    doc["program"][0]["wgsl"] = "fn f() {}";
    CHECK(!checker.errors(doc).empty());
    doc = cs::to_json(r.spec);
    doc["piece"][0]["draw"]["passes"][0]["blend"] = "max";
    CHECK(!checker.errors(doc).empty());
}

// Once stages are gone, the reader gives spec.stage_replaced for each [[stage]] table of an old spec
// or saved instrument (format.h): an error at the stage's path that names it and says how to write
// the same look as a piece program.
TEST_CASE("a spec that names a stage gets spec.stage_replaced, which says how to write the same look as a piece") {
    const nlohmann::json material = {{"panel", "world"}, {"entity", "programs/grid"}, {"kind", "material"},
                                     {"wgsl", "fn material(in: MaterialIn) -> vec4<f32> { return vec4<f32>(1.0); }"}};
    const cs::Diagnostic d = cs::stage_replaced(material, "stage[0]");
    CHECK(d.is_error());
    CHECK(d.code == "spec.stage_replaced");
    CHECK(d.path == "stage[0]");
    for (const char* words : {"Stages were replaced by pieces", "the material stage on the entity \"programs/grid\" of "
                              "the panel \"world\" no longer runs", "piece.copy", "points@1", "point_color",
                              "program.create", "the panel \"world\"", "[[stage]]"})
        CHECK_MESSAGE(d.message.find(words) != std::string::npos, d.message, " lacks: ", words);
    const cs::Diagnostic v = cs::stage_replaced({{"panel", "world"}, {"entity", "e"}, {"kind", "vertex"}}, "stage[1]");
    CHECK(v.message.find("vertex entry point") != std::string::npos);
    // A table without its keys still gets the error, with no name to give.
    const cs::Diagnostic bare = cs::stage_replaced(nlohmann::json::object(), "stage[2]");
    CHECK(bare.code == "spec.stage_replaced");
    CHECK(bare.message.find("thus the stage no longer runs") != std::string::npos);
    CHECK(bare.message.find("the scene panel") != std::string::npos);
}
