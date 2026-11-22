// The viewer's stages (spec_stage_host.h, scene/stage_cache.h, design/gpu-programs.md): the
// entry points of points and lines, the compile of a stage program with caliper-wgsl, the
// merge of the entries' MSL, the time watchdog, the host's checks, status and events, and a
// stage program drawn and picked on the GPU. The cases that need the caliper-wgsl library, or
// the sandbox's checks, say that they skip when the build lacks them.
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_stage_host.h"

#include "spec_test_canvas.h"

#include "gfx/metal_backend.h"
#include "gfx/recording_backend.h"
#include "live/live_scene.h"
#include "net/wire.h"
#include "gfx/shaders/generated/programs.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "scene/backend_resources.h"
#include "scene/pick.h"
#include "scene/stage_cache.h"

#include <caliper_builder/session/chain_hooks.h>
#include <caliper_builder/session/events.h>

#include <nlohmann/json.hpp>

#include <doctest/doctest.h>
#include <stb_image_write.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace cb = caliper_builder;
namespace pg = caliper_builder::programs;
namespace spec = caliper_builder::spec;
using namespace caliper_scene;
using caliper_host::AppStageHost;
using caliper_host::StageWatchdog;

const char* kRed = "fn material(in: MaterialIn) -> vec4<f32> { return vec4<f32>(1.0, 0.0, 1.0, 1.0); }\n";
const char* kSame = "fn material(in: MaterialIn) -> vec4<f32> { return in.color; }\n";
const char* kShift =
    "const SHIFT: f32 = 20.0;\n"
    "fn vertex(in: VertexIn) -> VertexOut { return VertexOut(in.position + vec3<f32>(SHIFT, 0.0, 0.0), in.radius); }\n";

bool have_compiler() {
    const std::string m = kRed;
    return caliper_host::compile_stage_program(Primitive::Points, &m, nullptr, "probe").verdict.code !=
           pg::kProgramCompilerUnavailable;
}

// Whether this tree's sandbox runs the checks; until it does, the host refuses every stage with
// the sandbox's own answer.
bool have_sandbox() {
    spec::Stage st;
    st.panel = "view";
    st.entity = "cloud";
    st.wgsl = kRed;
    return pg::check_stage_declaration(st).code != "command.not_implemented";
}

spec::Spec scene_spec(std::vector<spec::Stage> stages) {
    spec::Spec s;
    s.version = 1;
    spec::Panel scene;
    scene.id = "view";
    scene.primitive = spec::Primitive::scene;
    spec::Panel table;
    table.id = "rows";
    table.primitive = spec::Primitive::table;
    s.panels = {scene, table};
    s.stages = std::move(stages);
    return s;
}

spec::Stage stage(const char* panel, const char* entity, spec::StageKind kind, std::string wgsl) {
    spec::Stage st;
    st.panel = panel;
    st.entity = entity;
    st.kind = kind;
    st.wgsl = std::move(wgsl);
    return st;
}

struct Recorder final : cb::EventSink {
    std::vector<cb::Event> events;
    std::uint64_t push(cb::Event e) override {
        events.push_back(std::move(e));
        return events.size();
    }
    std::vector<std::string> changes() const {
        std::vector<std::string> out;
        for (const cb::Event& e : events) out.push_back(e.fields.value("change", "") + " " + e.fields.value("program", ""));
        return out;
    }
};

}  // namespace

TEST_CASE("stages: the entry points come for points and lines with the functions their stages define") {
    CHECK(stage_entries(Primitive::Mesh, true, true).empty());
    CHECK(stage_entries(Primitive::Points, false, false).empty());
    for (Primitive p : {Primitive::Points, Primitive::Lines}) {
        const std::string both = stage_entries(p, true, true);
        CHECK(both.find(std::string("fn ") + kStageVertexEntry + "(") != std::string::npos);
        CHECK(both.find(std::string("fn ") + kStageFragmentEntry + "(") != std::string::npos);
        CHECK(both.find(std::string("fn ") + kStagePickEntry + "(") != std::string::npos);
        CHECK(both.find("vertex(VertexIn(") != std::string::npos);
        CHECK(both.find("material(MaterialIn(") != std::string::npos);
        CHECK(stage_entries(p, true, false).find("vertex(VertexIn(") == std::string::npos);
        CHECK(stage_entries(p, false, true).find("material(MaterialIn(") == std::string::npos);
    }
}

TEST_CASE("stages: the MSL of the entry points is merged with the common lines once") {
    const std::string head = "#include <metal_stdlib>\nstruct A { int x; };\n";
    const std::string m =
        caliper_host::merge_msl({head + "vertex void vs() {}\n", head + "fragment void fs() {}\n", head + "fragment void pk() {}\n"});
    CHECK(m == head + "vertex void vs() {}\nfragment void fs() {}\nfragment void pk() {}\n");
    CHECK(caliper_host::merge_msl({}).empty());
}

TEST_CASE("stages: the watchdog probes after 30 slow frames and stops only stages slower than the panel without them") {
    StageWatchdog w;
    for (int i = 0; i < 29; ++i) CHECK(w.frames(20.0, 1, true) == StageWatchdog::Action::none);
    CHECK(w.frames(20.0, 1, true) == StageWatchdog::Action::probe);
    CHECK(w.probing());
    CHECK(w.frames(20.0, 1, true) == StageWatchdog::Action::none);   // a frame from before the probe
    CHECK(w.frames(5.0, 1, false) == StageWatchdog::Action::stop);

    StageWatchdog slow;   // as slow without the stages: they stay
    CHECK(slow.frames(20.0, 30, true) == StageWatchdog::Action::probe);
    CHECK(slow.frames(25.0, 1, false) == StageWatchdog::Action::none);
    CHECK(slow.over() == 0);
    CHECK_FALSE(slow.probing());

    StageWatchdog fast;   // a frame under the limit starts the count again
    CHECK(fast.frames(20.0, 29, true) == StageWatchdog::Action::none);
    CHECK(fast.frames(10.0, 1, true) == StageWatchdog::Action::none);
    CHECK(fast.frames(20.0, 29, true) == StageWatchdog::Action::none);
    CHECK(fast.over() == 29);
}

TEST_CASE("stages: a material and a vertex stage compose and translate for points and lines") {
    if (!have_compiler()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    const std::string mat = "fn material(in: MaterialIn) -> vec4<f32> {\n  return vec4<f32>(in.normal, sin(in.time_s));\n}\n";
    const std::string vtx = kShift;
    for (Primitive p : {Primitive::Points, Primitive::Lines}) {
        const caliper_host::StageCompile c = caliper_host::compile_stage_program(p, &mat, &vtx, "test");
        REQUIRE_MESSAGE(c.build, c.verdict.message);
        CHECK(c.build->material);
        CHECK(c.build->vertex);
        CHECK(c.build->animated);
        for (const char* e : {kStageVertexEntry, kStageFragmentEntry, kStagePickEntry})
            CHECK(c.build->msl.find(std::string(" ") + e + "(") != std::string::npos);
        const auto& d = caliper_gfx::shaders::kProgramDescriptions[uint32_t(c.build->builtin)];
        CHECK(c.build->bindings.size() == d.binding_count);
    }
    const caliper_host::StageCompile still = caliper_host::compile_stage_program(Primitive::Points, nullptr, &vtx, "test");
    REQUIRE(still.build);
    CHECK_FALSE(still.build->animated);
}

TEST_CASE("stages: a compile error has its code, line and column in the stage's own source") {
    if (!have_compiler()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    const std::string bad = "fn material(in: MaterialIn) -> vec4<f32> {\n  return in.colour;\n}\n";
    const std::string vtx = kShift;
    caliper_host::StageCompile c = caliper_host::compile_stage_program(Primitive::Points, &bad, &vtx, "test");
    CHECK_FALSE(c.build);
    CHECK(c.verdict.code == pg::kStageCompileFailed);
    CHECK(c.culprit == 0);
    CHECK(c.verdict.line == 2);
    CHECK(c.verdict.column > 0);
    const std::string mat = kSame;
    const std::string bad_vertex = "fn vertex(in: VertexIn) -> VertexOut {\n  let q = 1.0;\n  return VertexOut(in.position q);\n}\n";
    c = caliper_host::compile_stage_program(Primitive::Lines, &mat, &bad_vertex, "test");
    CHECK_FALSE(c.build);
    CHECK(c.culprit == 1);
    CHECK(c.verdict.line == 3);
    // A name of the viewer's program: refused, not drawn.
    const std::string clash = "fn hidden() -> vec4<f32> { return vec4<f32>(0.0); }\n" + std::string(kSame);
    c = caliper_host::compile_stage_program(Primitive::Points, &clash, nullptr, "test");
    CHECK_FALSE(c.build);
    CHECK(c.verdict.code == pg::kStageCompileFailed);
}

TEST_CASE("stages: the host refuses a stage of a panel that is not a scene panel, with a program event") {
    AppStageHost host;
    Recorder rec;
    host.set_events(&rec);
    host.reconcile(scene_spec({stage("rows", "cloud", spec::StageKind::material, kRed)}));
    const std::vector<cb::ProgramStatus> st = host.status();
    REQUIRE(st.size() == 1);
    CHECK(st[0].id == "rows/cloud/material");
    CHECK(st[0].kind == "material");
    CHECK(st[0].host == "viewer");
    CHECK(st[0].panel == "rows");
    CHECK(st[0].state == "refused");
    REQUIRE(st[0].error);
    if (have_sandbox()) CHECK(st[0].error->code == pg::kStageNotScenePanel);
    else MESSAGE("the sandbox's checks are not in this tree: the code is ", st[0].error->code);
    REQUIRE(rec.events.size() == 1);
    CHECK(rec.events[0].kind == cb::EventKind::program);
    CHECK(rec.events[0].panel == "rows");
    CHECK(rec.changes()[0] == "refused rows/cloud/material");
    CHECK(host.source("rows") == nullptr);
    CHECK(host.message("rows").find("Stage material on cloud") == 0);
    // The same spec again: nothing new.
    host.reconcile(scene_spec({stage("rows", "cloud", spec::StageKind::material, kRed)}));
    CHECK(rec.events.size() == 1);
    host.reconcile(scene_spec({}));
    CHECK(host.status().empty());
    CHECK(host.message("rows").empty());
}

namespace {

// Checks that pass, so that the host's own compile meets every stage (the sandbox's checks are
// in their own tests).
caliper_host::StageChecks passing_checks() {
    caliper_host::StageChecks c;
    c.declaration = [](const spec::Stage&) { return pg::Verdict{}; };
    c.provenance = [](const spec::Provenance&, std::string_view, std::string_view) { return pg::Verdict{}; };
    c.source = [](std::string_view, std::size_t) { return pg::Verdict{}; };
    c.compile = [](std::string_view) { return pg::Compiled{}; };
    c.stage = [](const spec::Stage&, const pg::Reflection&) { return pg::Verdict{}; };
    return c;
}

}  // namespace

TEST_CASE("stages: the host compiles the stages of a scene panel, refuses one with its line, and stops them over time") {
    if (!have_compiler()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    const bool sandbox = have_sandbox();
    if (!sandbox) MESSAGE("the sandbox's checks are not in this tree: the host's own compile meets each stage");
    AppStageHost host(sandbox ? caliper_host::StageChecks::sandbox() : passing_checks());
    Recorder rec;
    host.set_events(&rec);
    const std::string bad = "fn vertex(in: VertexIn) -> VertexOut {\n  return VertexOut(in.position in.radius);\n}\n";
    host.reconcile(scene_spec({stage("view", "cloud", spec::StageKind::material, kRed),
                               stage("view", "cloud", spec::StageKind::vertex, bad)}));
    std::vector<cb::ProgramStatus> st = host.status();
    REQUIRE(st.size() == 2);
    CHECK(st[0].state == "compiled");
    CHECK(st[1].state == "refused");
    REQUIRE(st[1].error);
    CHECK(st[1].error->line == 2);
    CHECK(host.message("view").find("at line 2, column") != std::string::npos);
    CHECK(rec.changes() == std::vector<std::string>{"compiled view/cloud/material", "refused view/cloud/vertex"});

    StageSource* src = host.source("view");
    REQUIRE(src);
    std::shared_ptr<const StageBuild> b = src->build("cloud", Primitive::Points);
    REQUIRE(b);
    CHECK(b->material);
    CHECK_FALSE(b->vertex);
    CHECK(src->build("cloud", Primitive::Lines));
    CHECK_FALSE(src->build("other", Primitive::Points));

    // Frames with the stages count as runs; the watchdog stops them after 30 slow frames.
    src->frames_done(2.0, 3, true);
    st = host.status();
    CHECK(st[0].state == "running");
    CHECK(st[0].runs == 3);
    const uint64_t rev = src->revision();
    src->frames_done(30.0, 30, true);
    CHECK(src->revision() != rev);
    CHECK_FALSE(src->build("cloud", Primitive::Points));   // the probe frame draws without them
    src->frames_done(3.0, 1, false);
    st = host.status();
    CHECK(st[0].state == "stopped");
    REQUIRE(st[0].error);
    CHECK(st[0].error->code == pg::kStageOverTime);
    CHECK(rec.changes().back() == "stopped view/cloud/material");
    CHECK_FALSE(src->build("cloud", Primitive::Points));

    // A fixed vertex stage compiles both again.
    host.reconcile(scene_spec({stage("view", "cloud", spec::StageKind::material, kRed),
                               stage("view", "cloud", spec::StageKind::vertex, kShift)}));
    st = host.status();
    CHECK(st[1].state == "compiled");
    REQUIRE(host.source("view"));
    b = host.source("view")->build("cloud", Primitive::Points);
    REQUIRE(b);
    CHECK(b->vertex);
}

TEST_CASE("stages: a frame with stages that does not finish stops them without a wait") {
    CHECK(StageWatchdog::pending(StageWatchdog::kHangMs / 2) == StageWatchdog::Action::none);
    CHECK(StageWatchdog::pending(StageWatchdog::kHangMs * 2) == StageWatchdog::Action::stop);
    if (!have_compiler()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    const bool sandbox = have_sandbox();
    AppStageHost host(sandbox ? caliper_host::StageChecks::sandbox() : passing_checks());
    Recorder rec;
    host.set_events(&rec);
    host.reconcile(scene_spec({stage("view", "cloud", spec::StageKind::material, kRed)}));
    StageSource* src = host.source("view");
    REQUIRE(src);
    REQUIRE(src->build("cloud", Primitive::Points));
    const uint64_t rev = src->revision();
    src->frame_pending(StageWatchdog::kHangMs / 2);   // a slow frame: the stages stay
    CHECK(src->revision() == rev);
    CHECK(host.status()[0].state == "compiled");
    src->frame_pending(StageWatchdog::kHangMs + 500.0);
    CHECK(src->revision() != rev);
    CHECK_FALSE(src->build("cloud", Primitive::Points));   // the panel draws without them
    const std::vector<cb::ProgramStatus> st = host.status();
    CHECK(st[0].state == "stopped");
    REQUIRE(st[0].error);
    CHECK(st[0].error->code == pg::kStageOverTime);
    CHECK(st[0].error->message.find("did not finish in 1500 ms") != std::string::npos);
    CHECK(st[0].error->message.find("cannot be stopped") != std::string::npos);
    CHECK(rec.changes().back() == "stopped view/cloud/material");
}

// A scene view over the recording backend, whose frames complete only when the case says: the
// view reports how long its oldest frame with a stage waits, and never waits for it itself.
TEST_CASE("stages: a scene view reports a frame with stages that has not completed") {
    if (!have_compiler()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    caliper_gfx::RecordingBackend gfx;
    gfx.set_auto_complete(false);
    caliper_live_views::SceneView v(gfx, caliper_gfx::shaders::kMetalPrograms);
    std::string why;
    REQUIRE_MESSAGE(v.init(&why), why);
    v.describe(caliper_net::wire::SceneView{CALIPER_NET_SPACE_3D, "+y", ""});
    const float xyz[9] = {-1.0f, -1.0f, 0.0f, 0.0f, 0.5f, 0.0f, 1.0f, -0.5f, 0.0f};
    caliper_net::wire::SceneUpdate u;
    u.sub = 1;
    u.view = "stages";
    u.seq = 1;
    u.path = "cloud";
    u.part = CALIPER_NET_PART_POINTS;
    u.params.set("size", caliper_net::Json(8.0));
    u.arrays.push_back(caliper_net::wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {3, 3}, xyz, sizeof xyz));
    REQUIRE_MESSAGE(v.apply(u, &why), why);

    struct Pending final : StageSource {
        std::shared_ptr<const StageBuild> b;
        std::vector<double> waits;
        std::shared_ptr<const StageBuild> build(std::string_view p, Primitive prim) const override {
            return b && p == "cloud" && prim == b->primitive ? b : nullptr;
        }
        uint64_t revision() const override { return 1; }
        float time_s() const override { return 0.0f; }
        void refused(const StageBuild&, const std::string&) override {}
        void frames_done(double, uint32_t, bool) override {}
        void frame_pending(double waited_ms) override { waits.push_back(waited_ms); }
    } src;
    const std::string red = kRed;
    const caliper_host::StageCompile c = caliper_host::compile_stage_program(Primitive::Points, &red, nullptr, "view");
    REQUIRE(c.build);
    src.b = c.build;
    REQUIRE_MESSAGE(v.render(64, 64, 1.0f, 0xff000000u, &why), why);   // a frame without the source
    v.set_stage_source(&src);
    REQUIRE_MESSAGE(v.render(64, 64, 1.0f, 0xff000000u, &why), why);
    REQUIRE(v.staged_draws() == 1);
    v.poll();
    v.poll();
    REQUIRE(src.waits.size() == 2);
    CHECK(src.waits[0] >= 0.0);
    CHECK(src.waits[1] >= src.waits[0]);   // the same frame, waiting longer
    gfx.finish(gfx.submitted().back().serial);
    v.poll();
    CHECK(src.waits.size() == 2);          // completed: nothing waits
    v.set_stage_source(nullptr);
}

TEST_CASE("stages: a stage reads its params from a uniform the program binds at group 3, and the host packs them each frame") {
    if (!have_compiler() || !have_sandbox()) {
        MESSAGE("skipped: this build has no caliper-wgsl library or sandbox");
        return;
    }
    const std::string level =
        "fn material(in: MaterialIn) -> vec4<f32> {\n"
        "  if (abs(in.value - params.level) < params.band) { return vec4<f32>(1.0, 0.6, 0.2, 1.0); }\n"
        "  return in.color;\n}\n";
    const caliper_host::StageCompile c =
        caliper_host::compile_stage_program(Primitive::Points, &level, nullptr, "params", {"band", "level"});
    INFO(c.verdict.message);
    REQUIRE(c.build);
    CHECK(c.build->params == std::vector<std::string>{"band", "level"});
    bool uniform = false;
    for (const caliper_gfx::BindingDesc& b : c.build->bindings)
        if (b.group == pg::kStageParamsGroup && b.binding == 0) uniform = b.type == caliper_gfx::BindingType::Uniform && b.size == 16;
    CHECK(uniform);
    // Without the names the source does not compile, and its line is the stage's own.
    const caliper_host::StageCompile bare = caliper_host::compile_stage_program(Primitive::Points, &level, nullptr, "bare");
    CHECK_FALSE(bare.build);
    CHECK(bare.verdict.line == 2);

    // The host: a literal and a reference to a control, read at each frame with no new build.
    spec::Stage st = stage("view", "cloud", spec::StageKind::material, level);
    st.params["band"] = 0.05;
    spec::Reference gain;
    gain.kind = spec::Reference::Kind::control;
    gain.target = "level";
    gain.fallback = 0.25;
    st.params["level"] = gain;
    AppStageHost host;
    host.reconcile(scene_spec({st}));
    REQUIRE(host.status().size() == 1);
    INFO(host.message("view"));
    CHECK(host.status()[0].state == "compiled");
    StageSource* src = host.source("view");
    REQUIRE(src);
    const std::shared_ptr<const StageBuild> before = src->build("cloud", Primitive::Points);
    REQUIRE(before);
    auto floats = [](const std::vector<uint8_t>& bytes) {
        std::vector<float> f(bytes.size() / 4);
        std::memcpy(f.data(), bytes.data(), f.size() * 4);
        return f;
    };
    std::vector<float> f = floats(src->params("cloud"));
    REQUIRE(f.size() == 4);   // two fields padded to 16 bytes, in name order: band, level
    CHECK(f[0] == doctest::Approx(0.05));
    CHECK(f[1] == doctest::Approx(0.25));   // the reference's default
    double value = 0.6;
    host.set_evaluate([&](const spec::Reference& r) { return r.target == "level" ? nlohmann::json(value) : r.fallback; });
    CHECK(floats(src->params("cloud"))[1] == doctest::Approx(0.6));
    value = 0.7;   // the next frame reads the new value
    CHECK(floats(src->params("cloud"))[1] == doctest::Approx(0.7));
    CHECK(src->build("cloud", Primitive::Points) == before);   // no compile, no new hash

    // A name the stage reads and does not declare is refused with the stage's line.
    spec::Stage wrong = st;
    wrong.params.erase("band");
    host.reconcile(scene_spec({wrong}));
    REQUIRE(host.status().size() == 1);
    CHECK(host.status()[0].state == "refused");
    REQUIRE(host.status()[0].error);
    CHECK(host.status()[0].error->code == pg::kStageParamsMismatch);
    CHECK(host.status()[0].error->line == 2);
    host.set_evaluate(nullptr);
}

TEST_SUITE("gpu") {

namespace {

// One source of one build, for the GPU cases.
struct OneSource final : StageSource {
    std::shared_ptr<const StageBuild> b;
    std::string path = "cloud";
    std::string refusal;
    uint64_t rev = 1;
    uint32_t staged_frames = 0, plain_frames = 0;
    std::shared_ptr<const StageBuild> build(std::string_view p, Primitive prim) const override {
        return b && p == path && prim == b->primitive ? b : nullptr;
    }
    uint64_t revision() const override { return rev; }
    float time_s() const override { return 0.0f; }
    void refused(const StageBuild&, const std::string& m) override { refusal = m; }
    void frames_done(double, uint32_t frames, bool staged) override { (staged ? staged_frames : plain_frames) += frames; }
};

// Three square points 4 pixels wide in a 2-D scene of 64 x 64 pixels, drawn through the scene
// core on Metal with the generated programs, with or without a stage program.
struct StageRig {
    static constexpr int kSide = 64;
    static constexpr uint32_t kClear = 0xff000000u;
    std::unique_ptr<caliper_gfx::Backend> gfx;
    std::unique_ptr<BackendResources> res;
    caliper_gfx::Buffer data;
    caliper_gfx::Texture color, ids;
    SceneTree tree{Space::k2D};
    SceneCompiler compiler;
    std::vector<std::pair<int, int>> at{{10, 12}, {30, 40}, {20, 50}};

    bool init() {
        caliper_host::SpecOffscreen* c = spec_test::canvas();
        if (!c || !c->native_device() || !c->native_queue()) return false;
        gfx = caliper_gfx::make_metal_backend(c->native_device(), c->native_queue());
        if (!gfx) return false;
        res = std::make_unique<BackendResources>(*gfx, caliper_gfx::shaders::kMetalPrograms);
        std::string err;
        REQUIRE_MESSAGE(res->init(&err), err);
        std::vector<uint8_t> bytes(1024, 0);
        const uint32_t rgba[3] = {0xff2040ffu, 0xff10e070u, 0xffe0e0e0u};
        for (size_t i = 0; i < at.size(); ++i) {
            const float xy[2] = {float(at[i].first) + 0.5f, float(at[i].second) + 0.5f};
            std::memcpy(bytes.data() + 8 * i, xy, 8);
            std::memcpy(bytes.data() + 512 + 4 * i, &rgba[i], 4);
        }
        data = gfx->create_buffer({bytes.size(), caliper_gfx::kVertex | caliper_gfx::kCopyDst, false, "stage points"});
        REQUIRE(gfx->write_buffer(data, 0, bytes.data(), bytes.size()));
        color = gfx->create_texture({uint32_t(kSide), uint32_t(kSide), caliper_gfx::Format::RGBA8Unorm,
                                     caliper_gfx::kRenderTarget | caliper_gfx::kCopySrcTex, "stage colour"});
        ids = gfx->create_texture({16, 16, caliper_gfx::Format::RG32Uint,
                                   caliper_gfx::kRenderTarget | caliper_gfx::kCopySrcTex, "stage pick"});
        REQUIRE((color && ids));
        Batch b;
        b.primitive = Primitive::Points;
        BatchArray pos;
        pos.role = Role::Positions;
        pos.buffer = data;
        pos.count = at.size();
        pos.stride = 8;
        pos.format = caliper_gfx::VertexFormat::F32x2;
        BatchArray col = pos;
        col.role = Role::Colors;
        col.offset = 512;
        col.stride = 4;
        col.format = caliper_gfx::VertexFormat::Unorm8x4;
        b.arrays = {pos, col};
        b.size_px = 4.0f;
        b.point_shape = PointShape::Square;
        b.bounds.add({0, 0, 0});
        b.bounds.add({double(kSide), double(kSide), 0});
        std::string why;
        REQUIRE_MESSAGE(tree.set_batch(tree.ensure("cloud"), b, false, &why), why);
        tree.update();
        res->update_class_table(tree);
        return true;
    }
    ~StageRig() {
        if (gfx) {
            res.reset();
            gfx->release(color);
            gfx->release(ids);
            gfx->release(data);
        }
    }
    ViewCamera camera() const {
        PanZoom p;
        p.center[0] = p.center[1] = kSide / 2.0;
        p.half_height = kSide / 2.0;
        p.y_down = true;
        return p;
    }
    CompileOptions options(const StageLookup* stages) const {
        CompileOptions o;
        o.width = o.height = kSide;
        o.clear_rgba = kClear;
        o.stages = stages;
        return o;
    }
    std::vector<uint8_t> read(caliper_gfx::Texture t, uint32_t w, uint32_t h) {
        const caliper_gfx::Readback r = gfx->read_texture(t, 0, 0, w, h);
        std::vector<uint8_t> out;
        REQUIRE(r);
        REQUIRE(gfx->wait(gfx->last_use(t)));
        REQUIRE(gfx->take_readback(r, &out));
        return out;
    }
    std::vector<uint8_t> draw(const StageLookup* stages, uint32_t* staged) {
        caliper_gfx::Frame frame;
        compiler.prepare(tree, *res, frame);
        caliper_gfx::RenderTarget rt;
        rt.color = color;
        rt.clear_rgba = kClear;
        caliper_gfx::DrawList list(rt);
        const CompileResult r = compiler.compile(tree, camera(), options(stages), *res, list, nullptr);
        REQUIRE_MESSAGE(r.ok, (r.notes.empty() ? std::string() : r.notes[0]));
        if (staged) *staged = r.staged;
        frame.render(std::move(list));
        REQUIRE_MESSAGE(gfx->submit(std::move(frame)) != 0, gfx->last_error());
        return read(color, kSide, kSide);
    }
    std::optional<Hit> pick(const StageLookup* stages, int x, int y) {
        Picker picker;
        const PickWindow w = picker.request(x + 0.5f, y + 0.5f, 8, kSide, kSide);
        const int32_t window[4] = {w.x0, w.y0, w.width, w.height};
        caliper_gfx::RenderTarget pt;
        pt.color = ids;
        caliper_gfx::DrawList list(pt);
        DrawMap map;
        const CompileResult r = compiler.compile_pick(tree, camera(), options(stages), window, *res, list, &map);
        REQUIRE(r.ok);
        caliper_gfx::Frame f;
        f.render(std::move(list));
        REQUIRE(gfx->submit(std::move(f)) != 0);
        const std::vector<uint8_t> id_bytes = read(ids, 16, 16);
        return picker.deliver(picker.current(), map.revision, id_bytes, nullptr, map);
    }
};

uint32_t pixel(const std::vector<uint8_t>& px, int x, int y) {
    uint32_t v;
    std::memcpy(&v, px.data() + (size_t(y) * StageRig::kSide + x) * 4, 4);
    return v;
}

void capture(const char* name, const std::vector<uint8_t>& rgba) {
    const char* dir = std::getenv("CALIPER_STAGE_CAPTURE_DIR");
    if (dir == nullptr || *dir == 0) return;
    constexpr int S = StageRig::kSide, k = 4;
    std::vector<uint8_t> big(size_t(S) * k * S * k * 4);
    for (int y = 0; y < S * k; ++y)
        for (int x = 0; x < S * k; ++x) std::memcpy(&big[(size_t(y) * S * k + x) * 4], &rgba[(size_t(y / k) * S + x / k) * 4], 4);
    const std::string path = std::string(dir) + "/" + name;
    CHECK(stbi_write_png(path.c_str(), S * k, S * k, 4, big.data(), S * k * 4) != 0);
    MESSAGE("capture written: ", path);
}

}  // namespace

TEST_CASE("stages: a material stage recolours points, a vertex stage moves them, and both pick where they draw") {
    if (!have_compiler()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    StageRig rig;
    if (!rig.init()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    uint32_t staged = 0;
    const std::vector<uint8_t> plain = rig.draw(nullptr, &staged);
    CHECK(staged == 0);
    capture("stage-builtin.png", plain);
    const auto [px, py] = rig.at[1];
    REQUIRE(pixel(plain, px, py) != StageRig::kClear);

    StageCache cache(*rig.gfx);
    OneSource src;
    cache.set_source(&src);
    auto make = [&](const std::string* mat, const std::string* vtx) {
        const caliper_host::StageCompile c = caliper_host::compile_stage_program(Primitive::Points, mat, vtx, "test");
        REQUIRE_MESSAGE(c.build, c.verdict.message);
        src.b = c.build;
    };
    auto drawn = [&](uint32_t* n) {
        cache.begin_frame();
        std::vector<uint8_t> out = rig.draw(&cache, n);
        cache.end_frame();
        CHECK_MESSAGE(src.refusal.empty(), src.refusal);
        return out;
    };

    // A material stage that keeps the colour draws the same bytes as the built-in program.
    const std::string same = kSame;
    make(&same, nullptr);
    CHECK(drawn(&staged) == plain);
    CHECK(staged == 1);
    CHECK(cache.size() == 1);

    // A material stage of one colour: each covered pixel takes it, the rest stay clear.
    const std::string red = kRed;
    make(&red, nullptr);
    const std::vector<uint8_t> recoloured = drawn(&staged);
    capture("stage-material.png", recoloured);
    size_t covered = 0, magenta = 0;
    for (int y = 0; y < StageRig::kSide; ++y)
        for (int x = 0; x < StageRig::kSide; ++x) {
            if (pixel(plain, x, y) != StageRig::kClear) ++covered;
            if (pixel(recoloured, x, y) == 0xffff00ffu) ++magenta;
            else CHECK(pixel(recoloured, x, y) == StageRig::kClear);
        }
    CHECK(covered > 0);
    CHECK(magenta == covered);

    // A vertex stage moves each point 20 pixels right, in the colour pass and in the pick pass.
    const std::string shift = kShift;
    make(nullptr, &shift);
    const std::vector<uint8_t> moved = drawn(&staged);
    capture("stage-vertex.png", moved);
    CHECK(pixel(moved, px, py) == StageRig::kClear);
    CHECK(pixel(moved, px + 20, py) == pixel(plain, px, py));
    cache.begin_frame();
    const std::optional<Hit> hit = rig.pick(&cache, px + 20, py);
    REQUIRE(hit);
    CHECK(rig.tree.path(hit->entity) == "cloud");
    CHECK(hit->primitive == Primitive::Points);
    CHECK(hit->item == 1);
    const std::optional<Hit> old = rig.pick(&cache, px, py);
    CHECK_FALSE(old);
    cache.end_frame();
    // Without the source, the built-in program again, and the cache lets its programs go.
    cache.set_source(nullptr);
    CHECK(drawn(&staged) == plain);
    CHECK(staged == 0);
    CHECK(cache.size() == 0);
}

TEST_CASE("stages: a stage that indexes out of range draws, with zero for each read past its resource") {
    if (!have_compiler()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    StageRig rig;
    if (!rig.init()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    uint32_t staged = 0;
    const std::vector<uint8_t> plain = rig.draw(nullptr, &staged);
    // A texel load far outside the class table and a column far past the uniform's matrix read
    // zero; the local array's index is clamped to its last element, 1.0. Thus magenta.
    const std::string far = R"(fn material(in: MaterialIn) -> vec4<f32> {
    let far = u32(in.color.a * 0.0) + 100000u;
    var local = array<f32, 4>(0.0, 0.0, 0.0, 1.0);
    let texel = textureLoad(class_table, vec2<u32>(far, far), 0);
    let column = p.mvp[far];
    return vec4<f32>(local[far] + texel.r + column.x, texel.g + column.y, 1.0 + column.z, 1.0);
}
)";
    const caliper_host::StageCompile c = caliper_host::compile_stage_program(Primitive::Points, &far, nullptr, "test");
    REQUIRE_MESSAGE(c.build, c.verdict.message);
    CHECK(c.build->msl.find("DefaultConstructible") != std::string::npos);
    StageCache cache(*rig.gfx);
    OneSource src;
    src.b = c.build;
    cache.set_source(&src);
    cache.begin_frame();
    const std::vector<uint8_t> drawn = rig.draw(&cache, &staged);
    cache.end_frame();
    CHECK_MESSAGE(src.refusal.empty(), src.refusal);
    CHECK(staged == 1);
    capture("stage-out-of-range.png", drawn);
    size_t covered = 0, magenta = 0;
    for (int y = 0; y < StageRig::kSide; ++y)
        for (int x = 0; x < StageRig::kSide; ++x) {
            if (pixel(plain, x, y) != StageRig::kClear) ++covered;
            if (pixel(drawn, x, y) == 0xffff00ffu) ++magenta;
        }
    CHECK(covered > 0);
    CHECK(magenta == covered);
    cache.set_source(nullptr);
}

TEST_CASE("stages: a stage's params reach its program as the uniform at group 3, a new value at the next frame") {
    if (!have_compiler()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    StageRig rig;
    if (!rig.init()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    uint32_t staged = 0;
    const std::vector<uint8_t> plain = rig.draw(nullptr, &staged);
    const auto [px, py] = rig.at[1];
    REQUIRE(pixel(plain, px, py) != StageRig::kClear);
    // Red scaled by the param `level`: 0 draws black, 1 full red, with one program.
    const std::string level = "fn material(in: MaterialIn) -> vec4<f32> { return vec4<f32>(params.level, 0.0, 0.0, 1.0); }\n";
    const caliper_host::StageCompile c =
        caliper_host::compile_stage_program(Primitive::Points, &level, nullptr, "level", {"level"});
    REQUIRE_MESSAGE(c.build, c.verdict.message);
    struct LevelSource final : StageSource {
        std::shared_ptr<const StageBuild> b;
        float level = 1.0f;
        std::string refusal;
        std::shared_ptr<const StageBuild> build(std::string_view p, Primitive prim) const override {
            return b && p == "cloud" && prim == b->primitive ? b : nullptr;
        }
        uint64_t revision() const override { return 1; }
        float time_s() const override { return 0.0f; }
        void refused(const StageBuild&, const std::string& m) override { refusal = m; }
        void frames_done(double, uint32_t, bool) override {}
        std::vector<uint8_t> params(std::string_view) const override {
            return pg::pack_params({pg::ParamField{"level", pg::ParamType::f32}}, {{"level", double(level)}});
        }
    } src;
    src.b = c.build;
    StageCache cache(*rig.gfx);
    cache.set_source(&src);
    cache.set_params_bound(true);
    auto drawn = [&] {
        cache.begin_frame();
        std::vector<uint8_t> out = rig.draw(&cache, &staged);
        cache.end_frame();
        CHECK_MESSAGE(src.refusal.empty(), src.refusal);
        return out;
    };
    const std::vector<uint8_t> full = drawn();
    CHECK(staged == 1);
    CHECK(pixel(full, px, py) == 0xff0000ffu);
    const uint64_t program = cache.find("cloud", Primitive::Points)->program.id;
    src.level = 0.0f;
    const std::vector<uint8_t> dark = drawn();
    CHECK(staged == 1);
    CHECK(pixel(dark, px, py) == 0xff000000u);
    CHECK(cache.find("cloud", Primitive::Points)->program.id == program);   // no new compile
    capture("stage-params.png", full);
}

TEST_CASE("stages: a scene view draws its panel's stages, again on a new revision, and reports their frames") {
    if (!have_compiler()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    caliper_host::SpecOffscreen* canvas = spec_test::canvas();
    if (!canvas || !canvas->native_device()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    std::unique_ptr<caliper_gfx::Backend> gfx = caliper_gfx::make_metal_backend(canvas->native_device(), canvas->native_queue());
    REQUIRE(gfx);
    {
        caliper_live_views::SceneView v(*gfx, caliper_gfx::shaders::kMetalPrograms);
        std::string why;
        REQUIRE_MESSAGE(v.init(&why), why);
        v.describe(caliper_net::wire::SceneView{CALIPER_NET_SPACE_3D, "+y", ""});
        const float xyz[9] = {-1.0f, -1.0f, 0.0f, 0.0f, 0.5f, 0.0f, 1.0f, -0.5f, 0.0f};
        caliper_net::wire::SceneUpdate u;
        u.sub = 1;
        u.view = "stages";
        u.seq = 1;
        u.path = "cloud";
        u.part = CALIPER_NET_PART_POINTS;
        u.params.set("size", caliper_net::Json(8.0));
        u.arrays.push_back(caliper_net::wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {3, 3}, xyz, sizeof xyz));
        REQUIRE_MESSAGE(v.apply(u, &why), why);

        constexpr uint32_t S = 64, clear = 0xff000000u;
        std::vector<uint8_t> plain, staged;
        REQUIRE_MESSAGE(v.render(S, S, 1.0f, clear, &why), why);
        CHECK(v.staged_draws() == 0);
        REQUIRE(v.read_pixels(&plain, &why));

        OneSource src;
        const std::string red = kRed;
        const caliper_host::StageCompile c = caliper_host::compile_stage_program(Primitive::Points, &red, nullptr, "view");
        REQUIRE(c.build);
        src.b = c.build;
        v.set_stage_source(&src);
        const uint64_t frames = v.cost().frames;
        REQUIRE_MESSAGE(v.render(S, S, 1.0f, clear, &why), why);
        CHECK(v.cost().frames == frames + 1);
        CHECK(v.staged_draws() == 1);
        REQUIRE(v.read_pixels(&staged, &why));
        size_t magenta = 0, lit = 0;
        for (size_t i = 0; i < staged.size(); i += 4) {
            uint32_t a, b;
            std::memcpy(&a, &plain[i], 4);
            std::memcpy(&b, &staged[i], 4);
            if (a != clear) ++lit;
            if (b == 0xffff00ffu) ++magenta;
        }
        CHECK(lit > 0);
        CHECK(magenta == lit);
        v.poll();
        CHECK(src.staged_frames >= 1);
        CHECK(src.refusal.empty());

        // Nothing changed: no new frame; a new revision of the source: a new frame.
        REQUIRE(v.render(S, S, 1.0f, clear, &why));
        CHECK(v.cost().frames == frames + 1);
        ++src.rev;
        REQUIRE(v.render(S, S, 1.0f, clear, &why));
        CHECK(v.cost().frames == frames + 2);

        // Without the source the view draws the built-in program again.
        v.set_stage_source(nullptr);
        std::vector<uint8_t> again;
        REQUIRE(v.render(S, S, 1.0f, clear, &why));
        CHECK(v.staged_draws() == 0);
        REQUIRE(v.read_pixels(&again, &why));
        CHECK(again == plain);
    }
}

}  // TEST_SUITE("gpu")

TEST_SUITE("gpu") {

TEST_CASE("stages: the cache gives a stage program its params at each find, and keeps the built-in program until the view binds them") {
    if (!have_compiler()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    caliper_host::SpecOffscreen* canvas = spec_test::canvas();
    if (!canvas || !canvas->native_device() || !canvas->native_queue()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    auto gfx = caliper_gfx::make_metal_backend(canvas->native_device(), canvas->native_queue());
    REQUIRE(gfx);
    const std::string level = "fn material(in: MaterialIn) -> vec4<f32> { return in.color * params.gain; }\n";
    const caliper_host::StageCompile c =
        caliper_host::compile_stage_program(Primitive::Points, &level, nullptr, "gain", {"gain"});
    REQUIRE(c.build);

    struct ParamSource final : StageSource {
        std::shared_ptr<const StageBuild> b;
        float gain = 0.5f;
        std::string refusal;
        std::shared_ptr<const StageBuild> build(std::string_view, Primitive) const override { return b; }
        uint64_t revision() const override { return 1; }
        float time_s() const override { return 0.0f; }
        void refused(const StageBuild&, const std::string& m) override { refusal = m; }
        void frames_done(double, uint32_t, bool) override {}
        std::vector<uint8_t> params(std::string_view) const override {
            return pg::pack_params({pg::ParamField{"gain", pg::ParamType::f32}}, {{"gain", double(gain)}});
        }
    } src;
    src.b = c.build;
    StageCache cache(*gfx);
    cache.set_source(&src);
    cache.begin_frame();
    CHECK(cache.find("cloud", Primitive::Points) == nullptr);   // the view does not bind params yet
    CHECK(src.refusal.find("params") != std::string::npos);
    cache.end_frame();

    StageCache bound(*gfx);
    bound.set_source(&src);
    bound.set_params_bound(true);
    bound.begin_frame();
    const StageProgram* p = bound.find("cloud", Primitive::Points);
    REQUIRE(p);
    float got = 0;
    REQUIRE(p->params.size() == 16);
    std::memcpy(&got, p->params.data(), 4);
    CHECK(got == 0.5f);
    const uint64_t program = p->program.id;
    bound.end_frame();
    src.gain = 2.0f;   // a new value: the next frame's find gives it, with the same program
    bound.begin_frame();
    p = bound.find("cloud", Primitive::Points);
    REQUIRE(p);
    std::memcpy(&got, p->params.data(), 4);
    CHECK(got == 2.0f);
    CHECK(p->program.id == program);
    bound.end_frame();
}

}  // TEST_SUITE("gpu")

// ---- the operators' geometry drawn in a scene view (design/instrument-interface.md, Operators)

TEST_SUITE("gpu") {

namespace {

// A scene view of S by S pixels in 3-D with z up, its camera fitted and tilted.
struct OperatorView {
    static constexpr uint32_t S = 256, kClear = 0xff1a1a1au;
    caliper_live_views::SceneView v;
    uint64_t seq = 0;
    explicit OperatorView(caliper_gfx::Backend& gfx) : v(gfx, caliper_gfx::shaders::kMetalPrograms) {}
    bool init() {
        std::string why;
        if (!v.init(&why)) return false;
        v.describe(caliper_net::wire::SceneView{CALIPER_NET_SPACE_3D, "+z", ""});
        return true;
    }
    void apply(const std::string& path, const char* part, std::vector<caliper_net::wire::Section> arrays,
               caliper_net::Json params) {
        caliper_net::wire::SceneUpdate u;
        u.sub = 1;
        u.view = "operators";
        u.seq = ++seq;
        u.path = path;
        u.part = part;
        u.params = std::move(params);
        u.arrays = std::move(arrays);
        std::string why;
        REQUIRE_MESSAGE(v.apply(u, &why), why);
    }
    std::vector<uint8_t> draw(float orbit_x, float orbit_y) {
        std::string why;
        REQUIRE_MESSAGE(v.render(S, S, 1.0f, kClear, &why), why);
        v.fit();
        v.orbit(orbit_x, orbit_y);
        REQUIRE_MESSAGE(v.render(S, S, 1.0f, kClear, &why), why);
        std::vector<uint8_t> px;
        REQUIRE(v.read_pixels(&px, &why));
        return px;
    }
};

caliper_net::Json colormap_params(double vmin, double vmax) {
    caliper_net::Json p = caliper_net::Json::object();
    p.set("colormap", caliper_net::Json(0.0));   // viridis
    p.set("vmin", caliper_net::Json(vmin));
    p.set("vmax", caliper_net::Json(vmax));
    return p;
}

// Writes `tiles` side by side as one PNG under CALIPER_STAGE_CAPTURE_DIR, when it is set.
void capture_tiles(const char* name, const std::vector<std::vector<uint8_t>>& tiles, uint32_t side) {
    const char* dir = std::getenv("CALIPER_STAGE_CAPTURE_DIR");
    if (dir == nullptr || *dir == 0 || tiles.empty()) return;
    const uint32_t w = side * uint32_t(tiles.size());
    std::vector<uint8_t> out(size_t(w) * side * 4);
    for (size_t t = 0; t < tiles.size(); ++t)
        for (uint32_t y = 0; y < side; ++y)
            std::memcpy(&out[(size_t(y) * w + t * side) * 4], &tiles[t][size_t(y) * side * 4], size_t(side) * 4);
    const std::string path = std::string(dir) + "/" + name;
    CHECK(stbi_write_png(path.c_str(), int(w), int(side), 4, out.data(), int(w) * 4) != 0);
    MESSAGE("capture written: ", path);
}

size_t lit_pixels(const std::vector<uint8_t>& px, uint32_t clear) {
    size_t n = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        uint32_t c;
        std::memcpy(&c, &px[i], 4);
        n += c != clear;
    }
    return n;
}

caliper_builder::TensorData f32_tensor(std::vector<int64_t> shape, const std::vector<float>& v) {
    std::vector<uint8_t> bytes(v.size() * 4);
    std::memcpy(bytes.data(), v.data(), bytes.size());
    caliper_builder::TensorResult r = caliper_builder::TensorData::from_bytes(caliper_builder::DType::f32, std::move(shape),
                                                                              std::move(bytes));
    REQUIRE_FALSE(r.error);
    return std::move(r.tensor);
}

std::vector<float> f32_values(const caliper_builder::TensorData& t) {
    std::vector<float> v(size_t(t.numel()));
    std::memcpy(v.data(), t.data(), v.size() * 4);
    return v;
}

}  // namespace

TEST_CASE("operators: a surface over a window of a tensor that grows by one row a step draws a growing terrain") {
    caliper_host::SpecOffscreen* canvas = spec_test::canvas();
    if (!canvas || !canvas->native_device() || !canvas->native_queue()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    auto gfx = caliper_gfx::make_metal_backend(canvas->native_device(), canvas->native_queue());
    REQUIRE(gfx);
    // The kept steps of a (24,) activation, one row a step: a ridge that drifts across the
    // features and a ripple that grows with the step.
    constexpr int kFeatures = 24, kSteps = 48;
    auto row = [](int step, int c) {
        const double centre = 4.0 + 16.0 * step / double(kSteps);
        return float(std::exp(-0.5 * std::pow((c - centre) / 2.5, 2.0)) + 0.15 * std::sin(0.5 * c + 0.3 * step));
    };
    spec::Transform surface;
    surface.op = spec::TransformOp::surface;
    surface.max_cells = 4096;
    std::vector<std::vector<uint8_t>> tiles;
    size_t last_lit = 0;
    for (int steps : {12, 24, 48}) {
        // The window over the kept steps: (steps, 24), and the step of each row.
        std::vector<float> window, stamps;
        for (int k = 0; k < steps; ++k) {
            stamps.push_back(float(1000 + k));
            for (int c = 0; c < kFeatures; ++c) window.push_back(row(k, c));
        }
        const caliper_builder::TensorData rows = f32_tensor({steps}, stamps);
        caliper_builder::GeometryArrays g;
        REQUIRE_FALSE(caliper_builder::apply_operator(surface, f32_tensor({steps, kFeatures}, window), &rows, &g));
        const std::vector<float> h = f32_values(g["height"]), x = f32_values(g["x"]), y = f32_values(g["y"]);
        REQUIRE(h.size() == size_t(steps) * kFeatures);
        CHECK(x.front() == 1000.0f);
        CHECK(x.back() == float(1000 + steps - 1));

        // The grid as a mesh: x the step over the run's 48 steps, y the feature, z the height.
        std::vector<float> pos, val;
        std::vector<int32_t> tri;
        for (int i = 0; i < steps; ++i)
            for (int j = 0; j < kFeatures; ++j) {
                pos.push_back(-1.0f + 2.0f * (x[size_t(i)] - 1000.0f) / float(kSteps - 1));
                pos.push_back(-1.0f + 2.0f * y[size_t(j)] / float(kFeatures - 1));
                pos.push_back(0.5f * h[size_t(i) * kFeatures + j]);
                val.push_back(h[size_t(i) * kFeatures + j]);
            }
        for (int i = 0; i + 1 < steps; ++i)
            for (int j = 0; j + 1 < kFeatures; ++j) {
                const int32_t a = i * kFeatures + j, b = a + 1, c = a + kFeatures, d = c + 1;
                for (int32_t k : {a, c, b, b, c, d}) tri.push_back(k);
            }
        OperatorView ov(*gfx);
        REQUIRE(ov.init());
        // The run's data box at z 0: the terrain grows inside it.
        const float box[24] = {-1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 1, 0, 1, 1, 0, -1, 1, 0, -1, 1, 0, -1, -1, 0};
        caliper_net::Json lp = caliper_net::Json::object();
        lp.set("mode", caliper_net::Json(std::string(CALIPER_NET_LINES_SEGMENTS)));
        lp.set("width", caliper_net::Json(1.5));
        ov.apply("box", CALIPER_NET_PART_LINES,
                 {caliper_net::wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {8, 3}, box, sizeof box)}, lp);
        ov.apply("terrain", CALIPER_NET_PART_MESH,
                 {caliper_net::wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {int64_t(pos.size() / 3), 3},
                                                   pos.data(), pos.size() * 4),
                  caliper_net::wire::Section::make("indices", CALIPER_NET_DTYPE_I32, {int64_t(tri.size() / 3), 3},
                                                   tri.data(), tri.size() * 4),
                  caliper_net::wire::Section::make("values", CALIPER_NET_DTYPE_F32, {int64_t(val.size())}, val.data(),
                                                   val.size() * 4)},
                 colormap_params(-0.2, 1.1));
        std::vector<uint8_t> px = ov.draw(40.0f, 25.0f);
        const size_t lit = lit_pixels(px, OperatorView::kClear);
        CHECK(lit > last_lit);   // more steps, more terrain
        last_lit = lit;
        tiles.push_back(std::move(px));
    }
    capture_tiles("operators-terrain-growing.png", tiles, OperatorView::S);
}

TEST_CASE("operators: a path through the rows of a tensor draws a line strip with its kept rows") {
    caliper_host::SpecOffscreen* canvas = spec_test::canvas();
    if (!canvas || !canvas->native_device() || !canvas->native_queue()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    auto gfx = caliper_gfx::make_metal_backend(canvas->native_device(), canvas->native_queue());
    REQUIRE(gfx);
    // 600 rows of 5 columns: columns 1, 3 and 4 trace a spiral that rises, with noise in the others.
    constexpr int kRows = 600, kCols = 5;
    std::vector<float> t;
    for (int r = 0; r < kRows; ++r) {
        const double a = r * 0.035, rad = 0.3 + 0.7 * r / double(kRows);
        t.push_back(float(std::sin(r * 1.7)));
        t.push_back(float(rad * std::cos(a)));
        t.push_back(float(std::cos(r * 0.9)));
        t.push_back(float(rad * std::sin(a)));
        t.push_back(float(-0.8 + 1.6 * r / double(kRows)));
    }
    spec::Transform path;
    path.op = spec::TransformOp::path;
    path.columns = {1, 3, 4};
    path.max_points = 96;
    caliper_builder::GeometryArrays g;
    REQUIRE_FALSE(caliper_builder::apply_operator(path, f32_tensor({kRows, kCols}, t), nullptr, &g));
    const std::vector<float> pos = f32_values(g["positions"]), val = f32_values(g["values"]);
    const int64_t points = int64_t(val.size());
    CHECK(points == 86);   // the stride is 7: rows 0, 7, ... 595
    CHECK(val[1] == 7.0f);

    OperatorView ov(*gfx);
    REQUIRE(ov.init());
    caliper_net::Json lp = colormap_params(0.0, double(kRows));
    lp.set("mode", caliper_net::Json(std::string(CALIPER_NET_LINES_STRIPS)));
    lp.set("width", caliper_net::Json(2.5));
    const int32_t strip[1] = {int32_t(points)};
    ov.apply("path", CALIPER_NET_PART_LINES,
             {caliper_net::wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {points, 3}, pos.data(), pos.size() * 4),
              caliper_net::wire::Section::make("strip_lengths", CALIPER_NET_DTYPE_I32, {1}, strip, sizeof strip),
              caliper_net::wire::Section::make("values", CALIPER_NET_DTYPE_F32, {points}, val.data(), val.size() * 4)},
             lp);
    caliper_net::Json pp = colormap_params(0.0, double(kRows));
    pp.set("radius", caliper_net::Json(0.04));
    ov.apply("rows", CALIPER_NET_PART_POINTS,
             {caliper_net::wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {points, 3}, pos.data(), pos.size() * 4),
              caliper_net::wire::Section::make("values", CALIPER_NET_DTYPE_F32, {points}, val.data(), val.size() * 4)},
             pp);
    std::vector<uint8_t> px = ov.draw(30.0f, 35.0f);
    CHECK(lit_pixels(px, OperatorView::kClear) > 500);
    capture_tiles("operators-path-through-rows.png", {px}, OperatorView::S);
}

}  // TEST_SUITE("gpu")
