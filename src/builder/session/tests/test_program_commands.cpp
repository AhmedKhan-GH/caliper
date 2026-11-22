// Tests of the stage and program commands (program_commands.h,
// design/gpu-programs.md, Commands): the session binds all seven, the edits
// check the source with the sandbox and file a diff and an undo step, the
// refusals carry the sandbox's codes with line and column, stage.check
// reflects a source, program.status merges both hosts, the status item has
// the keys of the command table's program_status, and a program event keeps
// the keys of its kind. The cases that compile WGSL run only in a build with
// the caliper-wgsl library; the others run in every build.
#include "helpers.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/programs/sandbox.h>
#include <caliper_builder/session/program_commands.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

#include <caliper_builder/session/chain_hooks.h>

#include <cstring>
#include <set>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

TEST_CASE("programs: the session binds the stage and program commands and has no hosts at first") {
    auto s = open_session(fresh_dir("program_commands_bound") + "/workspace");
    CHECK(s->program_host() == nullptr);
    CHECK(s->stage_host() == nullptr);
    for (const char* name : {"stage.set", "stage.remove", "stage.check", "program.create", "program.patch",
                             "program.remove", "program.status"}) {
        const CommandDescriptor* d = find_command_descriptor(name);
        REQUIRE(d);
        CAPTURE(name);
        const auto r = invoke(*s, name, d->examples.front());
        CHECK((r.ok || r.error.code != "command.not_bound"));
        CHECK(is_spec_edit_command(name) == d->mutates);
    }
}

TEST_CASE("programs: a status item has exactly the keys of the table's program_status") {
    const CommandDescriptor* d = find_command_descriptor("program.status");
    REQUIRE(d);
    const json& item = d->result["properties"]["programs"]["items"];
    std::set<std::string> want;
    for (const auto& k : item["required"]) want.insert(k.get<std::string>());
    ProgramStatus st;
    st.id = "rows";
    st.kind = "geometry";
    st.error = ProgramError{"program.over_time", "The program went over its time limit on 3 passes in a row.", 0, 0};
    const json j = program_status_json(st);
    std::set<std::string> have;
    for (const auto& kv : j.items()) have.insert(kv.key());
    CHECK(have == want);
    CHECK(j["error"]["code"] == "program.over_time");
}

TEST_CASE("programs: a program event carries its id, change, host and error") {
    EventRing ring;
    ProgramStatus st;
    st.id = "rows";
    st.host = "script";
    st.panel = "world";
    st.error = ProgramError{"program.writes_input", "The input w is not var<storage, read>.", 2, 1};
    push_program_event(ring, st, "refused");
    const auto read = ring.read(0, {}, 8);
    REQUIRE(read.events.size() == 1);
    const json e = event_json(read.events.front());
    CHECK(e["kind"] == "program");
    CHECK(e["panel"] == "world");
    CHECK(e["program"] == "rows");
    CHECK(e["change"] == "refused");
    CHECK(e["host"] == "script");
    CHECK(e["code"] == "program.writes_input");
    CHECK(e["line"] == 2);
}

namespace {

namespace cp = caliper_builder::programs;

const char* const kRows =
    "@group(0) @binding(0) var<storage, read> w: array<f32>;\n"
    "@group(0) @binding(1) var<storage, read_write> positions: array<f32>;\n"
    "@group(0) @binding(2) var<storage, read_write> count: atomic<u32>;\n"
    "@compute @workgroup_size(64) fn fill(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "  let i = id.x;\n"
    "  if (i >= 256u) { return; }\n"
    "  positions[3u * i] = w[3u * i];\n"
    "  positions[3u * i + 1u] = w[3u * i + 1u];\n"
    "  positions[3u * i + 2u] = w[3u * i + 2u];\n"
    "  atomicAdd(&count, 1u);\n"
    "}\n";

const char* const kMaterial = "fn material(in: MaterialIn) -> vec4<f32> {\n  return vec4<f32>(in.color.rgb * in.value, 1.0);\n}\n";

json rows_args() {
    return {{"id", "rows"},
            {"wgsl", kRows},
            {"inputs", json::array({{{"var", "w"}, {"source", "live:layers.0.weight"}}})},
            {"output",
             {{"kind", "geometry"}, {"panel", "world"}, {"max_count", 256}, {"arrays", json::array({"positions"})}}},
            {"limits", {{"every", 10}}}};
}

// A session with a new spec that holds the scene panel world and the series
// panel chart.
std::unique_ptr<Session> scene_session(const std::string& name) {
    auto s = open_session(fresh_dir(name) + "/workspace");
    ok(*s, "spec.new", {{"id", "gpu"}});
    ok(*s, "source.add", {{"id", "loss"}, {"catalog", "live:loss"}});
    ok(*s, "panel.create", {{"id", "world"}, {"primitive", "scene"}, {"bind", {{"scene", "live:model"}}}});
    ok(*s, "panel.create", {{"id", "chart"}, {"primitive", "series"}, {"bind", {{"y", "loss"}}}});
    return s;
}

CommandResult call(Session& s, const std::string& name, json args) { return invoke(s, name, std::move(args)); }

}  // namespace

TEST_CASE("programs: program.create adds a checked program with a diff and an undo step") {
    auto s = scene_session("program_create");
    const spec::Spec before = s->open_spec()->spec;
    CHECK(before.version == 1);   // the scene panel raised it
    json dry = rows_args();
    dry["dry_run"] = true;
    const json d = ok(*s, "program.create", dry);
    CHECK(d["applied"] == false);
    CHECK(s->open_spec()->spec == before);

    const json r = ok(*s, "program.create", rows_args());
    CHECK(r["id"] == "rows");
    CHECK(r["applied"] == true);
    CHECK(r["diff"].get<std::string>().find("+[[program]]") != std::string::npos);
    if (cp::compiler_available()) {
        CHECK(r["messages"].empty());
    } else {
        REQUIRE(r["messages"].size() == 1);
        CHECK(r["messages"][0]["code"] == cp::kProgramNotChecked);
        CHECK(r["messages"][0]["severity"] == "warning");
        CHECK(r["warnings"][0]["code"] == cp::kProgramNotChecked);
    }
    const spec::Spec after = s->open_spec()->spec;
    REQUIRE(after.programs.size() == 1);
    CHECK(after.programs[0].limits.every == std::optional<std::uint32_t>(10));
    CHECK(err(*s, "program.create", rows_args()) == cp::kProgramExists);
    ok(*s, "history.undo");
    CHECK(s->open_spec()->spec == before);
    ok(*s, "history.redo");
    CHECK(s->open_spec()->spec == after);
}

TEST_CASE("programs: the editor refuses a target, a limit and a directive with the sandbox's codes") {
    auto s = scene_session("program_refusals");
    json chart = rows_args();
    chart["output"]["panel"] = "chart";
    CHECK(err(*s, "program.create", chart) == cp::kProgramBadTarget);
    json nowhere = rows_args();
    nowhere["output"]["panel"] = "nowhere";
    CHECK(err(*s, "program.create", nowhere) == cp::kProgramBadTarget);
    json budget = rows_args();
    budget["output"]["max_count"] = 1000000;
    budget["limits"]["memory_mb"] = 1;
    CHECK(err(*s, "program.create", budget) == cp::kProgramOverLimits);
    json directive = rows_args();
    directive["wgsl"] = std::string("enable f16;\n") + kRows;
    const CommandResult r = call(*s, "program.create", directive);
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.code == cp::kProgramDirectiveNotAllowed);
    CHECK(r.error.where == "args.wgsl:1:1");
    CHECK(r.error.message.find("(line 1, column 1)") != std::string::npos);
    CHECK(s->open_spec()->spec.programs.empty());
}

TEST_CASE("programs: with the compiler the editor refuses a write to an input, an undeclared binding and a compile error") {
    if (!cp::compiler_available()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    auto s = scene_session("program_compiled_refusals");
    json writes = rows_args();
    std::string rw = kRows;
    rw.replace(rw.find("read> w"), 7, "read_write> w");
    writes["wgsl"] = rw;
    CommandResult r = call(*s, "program.create", writes);
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.code == cp::kProgramWritesInput);
    CHECK(r.error.where == "args.wgsl:1:48");

    json undeclared = rows_args();
    undeclared["inputs"] = json::array();
    CHECK(err(*s, "program.create", undeclared) == cp::kProgramBindingUndeclared);

    json broken = rows_args();
    std::string bad = kRows;
    bad.replace(bad.find("let i = id.x;"), 13, "let i = ;");
    broken["wgsl"] = bad;
    r = call(*s, "program.create", broken);
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.code == cp::kProgramCompileFailed);
    CHECK(r.error.where.rfind("args.wgsl:5:", 0) == 0);
    CHECK(s->open_spec()->spec.programs.empty());
}

TEST_CASE("programs: program.patch merges a patch and checks again, and program.remove removes") {
    auto s = scene_session("program_patch");
    ok(*s, "program.create", rows_args());
    json r = ok(*s, "program.patch", {{"id", "rows"}, {"patch", {{"limits", {{"every", 20}}}, {"output", {{"max_count", 128}}}}}});
    CHECK(r["program"]["limits"]["every"] == 20);
    CHECK(r["program"]["output"]["max_count"] == 128);
    CHECK(r["program"]["output"]["panel"] == "world");
    CHECK(r["messages"].empty());
    CHECK(s->open_spec()->spec.programs[0].output.max_count == 128);
    CHECK(err(*s, "program.patch", {{"id", "rows"}, {"patch", {{"output", {{"panel", "chart"}}}}}}) ==
          cp::kProgramBadTarget);
    CHECK(err(*s, "program.patch", {{"id", "other"}, {"patch", {{"title", "x"}}}}) == cp::kProgramNotFound);
    // A new kind replaces the output instead of merging into it.
    const std::string image =
        "@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;\n"
        "@compute @workgroup_size(8, 8) fn fill(@builtin(global_invocation_id) id: vec3<u32>) {\n"
        "  if (id.x < 16u && id.y < 16u) { pixels[id.y * 16u + id.x] = 0xff0000ffu; }\n"
        "}\n";
    r = ok(*s, "program.patch",
           {{"id", "rows"},
            {"patch", {{"wgsl", image}, {"inputs", json::array()}, {"output", {{"kind", "pixels"}, {"width", 16}, {"height", 16}}}}}});
    CHECK(r["program"]["output"] == json::parse(R"({"kind": "pixels", "width": 16, "height": 16, "format": "rgba8unorm"})"));
    CHECK(ok(*s, "program.remove", {{"id", "rows"}})["id"] == "rows");
    CHECK(s->open_spec()->spec.programs.empty());
    CHECK(err(*s, "program.remove", {{"id", "rows"}}) == cp::kProgramNotFound);
    ok(*s, "history.undo");
    CHECK(s->open_spec()->spec.programs.size() == 1);
}

TEST_CASE("stages: stage.set replaces the stage of one panel, entity and kind, and stage.remove removes it") {
    auto s = scene_session("stage_set");
    json args = {{"panel", "world"}, {"entity", "programs/rows"}, {"kind", "material"}, {"wgsl", kMaterial}};
    json r = ok(*s, "stage.set", args);
    CHECK(r["panel"] == "world");
    CHECK(r["kind"] == "material");
    CHECK(r["diff"].get<std::string>().find("+[[stage]]") != std::string::npos);
    args["wgsl"] = "fn material(in: MaterialIn) -> vec4<f32> {\n  return in.color;\n}\n";
    ok(*s, "stage.set", args);
    REQUIRE(s->open_spec()->spec.stages.size() == 1);
    CHECK(s->open_spec()->spec.stages[0].wgsl == args["wgsl"]);
    ok(*s, "stage.set",
       {{"panel", "world"}, {"entity", "programs/rows"}, {"kind", "vertex"},
        {"wgsl", "fn vertex(in: VertexIn) -> VertexOut {\n  return VertexOut(in.position * 2.0, in.radius);\n}\n"}});
    CHECK(s->open_spec()->spec.stages.size() == 2);

    json chart = args;
    chart["panel"] = "chart";
    CHECK(err(*s, "stage.set", chart) == cp::kStageNotScenePanel);
    json bound = args;
    bound["wgsl"] = std::string(kMaterial) + "@group(0) @binding(0) var<uniform> u: vec4<f32>;\n";
    const CommandResult b = call(*s, "stage.set", bound);
    REQUIRE_FALSE(b.ok);
    CHECK(b.error.code == cp::kStageBindingNotAllowed);
    CHECK(b.error.where == "args.wgsl:4:1");

    ok(*s, "stage.remove", {{"panel", "world"}, {"entity", "programs/rows"}, {"kind", "vertex"}});
    CHECK(s->open_spec()->spec.stages.size() == 1);
    CHECK(err(*s, "stage.remove", {{"panel", "world"}, {"entity", "programs/rows"}, {"kind", "vertex"}}) ==
          cp::kStageNotFound);
    // A panel's removal takes its stages along, in one undo step.
    const spec::Spec before = s->open_spec()->spec;
    ok(*s, "panel.remove", {{"id", "world"}});
    CHECK(s->open_spec()->spec.stages.empty());
    ok(*s, "history.undo");
    CHECK(s->open_spec()->spec == before);
}

TEST_CASE("stages: with the compiler stage.set gives the line of a compile error in the stage's own source") {
    if (!cp::compiler_available()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    auto s = scene_session("stage_compile");
    json args = {{"panel", "world"}, {"entity", "e"}, {"kind", "material"},
                 {"wgsl", "fn material(in: MaterialIn) -> vec4<f32> {\n  return in.colour;\n}\n"}};
    CommandResult r = call(*s, "stage.set", args);
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.code == cp::kStageCompileFailed);
    CHECK(r.error.where.rfind("args.wgsl:2:", 0) == 0);
    // A function with another signature fails at the viewer's call of it.
    args["wgsl"] = "fn material(in: MaterialIn) -> vec3<f32> {\n  return in.color.rgb;\n}\n";
    r = call(*s, "stage.set", args);
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.code == cp::kStageCompileFailed);
    CHECK(r.error.message.find("fn material(in: MaterialIn) -> vec4<f32>") != std::string::npos);
    args["wgsl"] = std::string(kMaterial) + "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }\n";
    CHECK(err(*s, "stage.set", args) == cp::kStageEntryNotAllowed);
    CHECK(s->open_spec()->spec.stages.empty());
}

TEST_CASE("stages: stage.check compiles a source and reflects its entry points and bindings") {
    auto s = scene_session("stage_check");
    const CommandDescriptor* d = find_command_descriptor("stage.check");
    REQUIRE(d);
    if (!cp::compiler_available()) {
        CHECK(err(*s, "stage.check", d->examples.front()) == cp::kProgramCompilerUnavailable);
        return;
    }
    json r = ok(*s, "stage.check", d->examples.front());
    CHECK(r["ok"] == true);
    REQUIRE(r["entry_points"].size() == 1);
    CHECK(r["entry_points"][0] == json::parse(R"({"name": "run", "stage": "compute", "workgroup_size": [64, 1, 1]})"));
    REQUIRE(r["bindings"].size() == 3);
    CHECK(r["bindings"][0]["access"] == "read");
    CHECK(r["bindings"][1]["access"] == "read_write");
    CHECK(r["bindings"][2]["kind"] == "storage");
    r = ok(*s, "stage.check", {{"kind", "material"}, {"wgsl", kMaterial}});
    CHECK(r["ok"] == true);
    CHECK(r["entry_points"].empty());
    r = ok(*s, "stage.check", {{"kind", "vertex"}, {"wgsl", "fn vertex(in: VertexIn) -> VertexOut {\n  return VertexOut(in.positon, in.radius);\n}\n"}});
    CHECK(r["ok"] == false);
    REQUIRE(r["messages"].size() == 1);
    CHECK(r["messages"][0]["code"] == cp::kStageCompileFailed);
    CHECK_MESSAGE(r["messages"][0]["line"] == 2, r["messages"][0].dump());
    r = ok(*s, "stage.check", {{"kind", "compute"}, {"wgsl", "fn f() {}"}});
    CHECK(r["ok"] == false);
    CHECK(r["messages"][0]["code"] == cp::kProgramNoEntry);
}

namespace {

// Hosts that report fixed items and count their reconciles.
struct FakeProgramHost : ProgramHost {
    std::vector<ProgramStatus> items;
    int reconciles = 0;
    std::size_t last_programs = 0;
    void reconcile(const spec::Spec& s) override {
        ++reconciles;
        last_programs = s.programs.size();
    }
    std::vector<ProgramStatus> status() const override { return items; }
};
struct FakeStageHost : StageHost {
    std::vector<ProgramStatus> items;
    int reconciles = 0;
    void reconcile(const spec::Spec&) override { ++reconciles; }
    std::vector<ProgramStatus> status() const override { return items; }
};

}  // namespace

TEST_CASE("programs: program.status lists the spec's items with what each host reports") {
    auto s = scene_session("program_status");
    ok(*s, "program.create", rows_args());
    ok(*s, "stage.set", {{"panel", "world"}, {"entity", "programs/rows"}, {"kind", "material"}, {"wgsl", kMaterial}});
    json r = ok(*s, "program.status");
    REQUIRE(r["programs"].size() == 1);
    CHECK(r["programs"][0]["host"] == "none");
    CHECK(r["programs"][0]["state"] == "waiting");
    CHECK(r["programs"][0]["every_min"] == 10);
    CHECK(r["programs"][0]["memory_limit_bytes"] == 64u << 20);
    CHECK(r["programs"][0]["kind"] == "geometry");
    REQUIRE(r["stages"].size() == 1);
    CHECK(r["stages"][0]["id"] == "world/programs/rows/material");
    CHECK(r["stages"][0]["host"] == "none");

    FakeProgramHost programs;
    ProgramStatus running;
    running.id = "rows";
    running.kind = "geometry";
    running.host = "script";
    running.state = "running";
    running.every = 12;
    running.runs = 3;
    programs.items = {running};
    FakeStageHost stages;
    ProgramStatus compiled;
    compiled.id = "world/programs/rows/material";
    compiled.kind = "material";
    compiled.host = "viewer";
    compiled.state = "compiled";
    stages.items = {compiled};
    s->set_program_host(&programs);
    s->set_stage_host(&stages);
    r = ok(*s, "program.status");
    CHECK(r["programs"][0]["host"] == "script");
    CHECK(r["programs"][0]["every"] == 12);
    CHECK(r["programs"][0]["runs"] == 3);
    CHECK(r["stages"][0]["state"] == "compiled");
    r = ok(*s, "program.status", {{"id", "world/programs/rows/material"}});
    CHECK(r["programs"].empty());
    CHECK(r["stages"].size() == 1);
    CHECK(err(*s, "program.status", {{"id", "nothing"}}) == cp::kProgramNotFound);
    s->set_program_host(nullptr);
    s->set_stage_host(nullptr);
}

TEST_CASE("programs: in a foreign spec an unreviewed source waits for a review and provenance stays as it is") {
    const std::string ws = fresh_dir("program_foreign") + "/workspace";
    auto s = open_session(ws);
    write_text(ws + "/foreign.spec.toml", read_text(fixture_path("programs.spec.toml")));
    ok(*s, "spec.open", {{"path", "foreign.spec.toml"}});
    const json r = ok(*s, "program.status");
    REQUIRE(r["programs"].size() == 2);
    CHECK(r["programs"][0]["state"] == "waiting");   // rows: its hash is reviewed
    CHECK(r["programs"][1]["state"] == "review");    // image: no review
    CHECK(r["stages"][0]["state"] == "review");
    // An edit keeps the provenance table, and no text can change it.
    ok(*s, "program.patch", {{"id", "image"}, {"patch", {{"title", "Gradient"}}}});
    CHECK(s->open_spec()->spec.provenance.foreign);
    std::string text = read_text(fixture_path("programs.spec.toml"));
    text = text.substr(0, text.find("[provenance]"));
    CHECK(err(*s, "spec.set_text", {{"text", text}}) == "spec.provenance_locked");
}

TEST_CASE("programs: a toggle bound to an intervention point enters the spec") {
    auto s = scene_session("toggle_spec");
    std::string text = spec::to_toml(s->open_spec()->spec);
    text += "\n[[control]]\nid = \"ablate\"\nkind = \"toggle\"\ntitle = \"Ablate head 3\"\nbind = \"intervene:blocks.0.attn#3\"\n";
    ok(*s, "spec.set_text", {{"text", text}});
    REQUIRE(s->open_spec()->spec.controls.size() == 1);
    CHECK(s->open_spec()->spec.controls[0].kind == spec::ControlKind::toggle);
    CHECK(s->open_spec()->spec.controls[0].bind == "intervene:blocks.0.attn#3");
}

// ---- the operator steps in the app (chain_hooks.h, operator_steps.cpp)

namespace {

TensorData floats(std::vector<int64_t> shape, const std::vector<float>& v) {
    std::vector<uint8_t> bytes(v.size() * 4);
    std::memcpy(bytes.data(), v.data(), bytes.size());
    TensorResult r = TensorData::from_bytes(DType::f32, std::move(shape), std::move(bytes));
    REQUIRE_FALSE(r.error);
    return std::move(r.tensor);
}

std::vector<float> values_of(const TensorData& t) {
    std::vector<float> v(static_cast<std::size_t>(t.numel()));
    std::memcpy(v.data(), t.data(), v.size() * 4);
    return v;
}

}  // namespace

TEST_CASE("operator steps: surface and path give their named arrays, with the window's steps as coordinates") {
    // A window of 4 kept steps of a row of 6: (4, 6), cell (r, c) = 10 r + c.
    std::vector<float> g;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 6; ++c) g.push_back(float(10 * r + c));
    const TensorData input = floats({4, 6}, g);
    const TensorData steps = floats({4}, {200, 201, 202, 203});

    spec::Transform surface;
    surface.op = spec::TransformOp::surface;
    surface.max_cells = 6;
    GeometryArrays out;
    REQUIRE_FALSE(apply_operator(surface, input, &steps, &out));
    REQUIRE(out.count("height"));
    CHECK(out["height"].shape() == std::vector<int64_t>{2, 3});
    CHECK(values_of(out["height"]) == std::vector<float>{5.5f, 7.5f, 9.5f, 25.5f, 27.5f, 29.5f});
    CHECK(values_of(out["x"]) == std::vector<float>{200.5f, 202.5f});
    CHECK(values_of(out["y"]) == std::vector<float>{0.5f, 2.5f, 4.5f});

    spec::Transform path;
    path.op = spec::TransformOp::path;
    path.columns = {5, 0};
    path.max_points = 2;
    REQUIRE_FALSE(apply_operator(path, input, &steps, &out));
    CHECK(out.count("height") == 0);
    CHECK(out["positions"].shape() == std::vector<int64_t>{2, 3});
    CHECK(values_of(out["positions"]) == std::vector<float>{5, 0, 0, 25, 20, 0});
    CHECK(values_of(out["values"]) == std::vector<float>{200, 202});

    // Another rank, and a column past the input.
    Error err = apply_operator(surface, floats({24}, g), nullptr, &out);
    CHECK(err.code == "resolve.operator_shape");
    path.columns = {6, 0};
    CHECK(apply_operator(path, input, nullptr, &out).code == "resolve.operator_shape");
}

TEST_CASE("operator steps: only the first step of a live tensor source runs in the script") {
    spec::Spec s;
    spec::Source live;
    live.id = "w";
    live.catalog = "live:blocks.0.attn.proj.weight";
    spec::Transform op;
    op.op = spec::TransformOp::surface;
    live.transforms = {op};
    spec::Source plain;
    plain.id = "act";
    plain.catalog = "live:blocks.0.attn.proj.act";
    spec::Source derived;
    derived.id = "terrain";
    derived.from = "act";
    derived.transforms = {op};
    spec::Source windowed = derived;
    windowed.id = "recent";
    spec::Transform window;
    window.op = spec::TransformOp::window;
    window.steps = 16;
    windowed.transforms = {window, op};
    spec::Source file = live;
    file.id = "f";
    file.catalog = "tensor:acts";
    s.sources = {live, plain, derived, windowed, file};
    CHECK(operator_runs_in_script(s, s.sources[0], 0));
    CHECK(operator_runs_in_script(s, s.sources[2], 0));    // from a live source without steps
    CHECK_FALSE(operator_runs_in_script(s, s.sources[3], 1));   // after a window: in the app
    CHECK_FALSE(operator_runs_in_script(s, s.sources[4], 0));   // not live
    s.sources[1].transforms = {window};
    CHECK_FALSE(operator_runs_in_script(s, s.sources[2], 0));   // its root has a step
}
