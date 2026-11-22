// Tests of the programs, stages and provenance of spec version 1 (program.h,
// design/gpu-programs.md, Spec Additions) and of the toggle control: the
// codecs read and write them canonically, the validator holds them to the
// limits of program.h with the sandbox's codes, and an older document reads
// as before.
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
    CHECK_FALSE(cs::control_kind_built(cs::ControlKind::picker));
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
    CHECK(has(cs::check_toml("caliper_spec = 1\n[applet]\nid = \"a\"\n[[control]]\nid = \"k\"\nkind = \"picker\"\n"
                             "bind = \"value\"\n")
                  .diagnostics,
              "spec.unsupported_control", "control[0].kind"));
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
