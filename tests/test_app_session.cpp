// Tests of the builder session in the app (src/host/builder/app_session.h).
//
// The unit cases drive AppSession with a fake frame loop: a thread that calls
// frame() about every two milliseconds, as the app's frame loop does between
// two frames, hands over a "read-back" after a frame when a capture was asked
// for, and closes the session on its own thread at the end. Commands come from
// the test's thread through the session's queue, as Caliper API's will. A
// JobSystem of the test's own stands in for the host's.
//
// The process case runs the real app hidden on a workspace of its own and
// drives the session through the command script (CALIPER_COMMAND_SCRIPT),
// then stops it with SIGTERM and reads what it left in the workspace.
#include <doctest/doctest.h>

#include "api/command_gateway.h"
#include "builder/app_session.h"
#include "builder/panel_renderer.h"
#include "builder/spec_page.h"
#include "live/live_attach.h"
#include "live/live_programs.h"
#include "job_system.h"
#include "test_temp.h"
#ifndef _WIN32
#include "test_proc.h"
#endif

#include <caliper_builder/commands/approval.h>
#include <caliper_builder/commands/journal.h>
#include <caliper_builder/operators/operators.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/spec/format.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace cb = caliper_builder;
using caliper_host::AppSession;
using caliper_host::LaunchOutcome;
using nlohmann::json;
using namespace std::chrono_literals;

namespace {

// A folder of this run's own (test_temp.h), removed when the case ends.
struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name) : path(caliper_test::unique_temp_path("caliper-session-" + name)) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str(const std::string& sub) const { return (path / sub).string(); }
};

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_text(const std::string& path, const std::string& text) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path, std::ios::binary);
    out << text;
}

std::string b64(const std::string& in) {
    static const char* k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int val = 0, bits = -6;
    for (unsigned char c : in) {
        val = (val << 8) + c;
        bits += 8;
        while (bits >= 0) {
            out.push_back(k[(val >> bits) & 0x3F]);
            bits -= 6;
        }
    }
    if (bits > -6) out.push_back(k[((val << 8) >> (bits + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

std::string unb64(const std::string& in) {
    std::string out;
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        int d;
        if (c >= 'A' && c <= 'Z') d = c - 'A';
        else if (c >= 'a' && c <= 'z') d = c - 'a' + 26;
        else if (c >= '0' && c <= '9') d = c - '0' + 52;
        else if (c == '+') d = 62;
        else if (c == '/') d = 63;
        else break;
        val = (val << 6) + d;
        bits += 6;
        if (bits >= 0) {
            out.push_back(char((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

cb::CommandInvocation inv(const std::string& name, json args = json::object(),
                          cb::CommandOrigin origin = cb::CommandOrigin::ui) {
    cb::CommandInvocation i;
    i.name = name;
    i.args = std::move(args);
    i.origin = origin;
    return i;
}

cb::CommandDescriptor toy(const std::string& name, bool frame_thread) {
    cb::CommandDescriptor d;
    d.name = name;
    d.summary = "A test command.";
    d.args = json::parse(R"({"type": "object", "properties": {"n": {"type": "integer"}}})");
    d.result = json::parse(R"({"type": "object"})");
    d.frame_thread = frame_thread;
    return d;
}

// The app's frame loop, faked: frame() between two "frames" on a thread of its
// own, and the read-back of a capture after the frame that follows its request.
struct FrameLoop {
    caliper_host::JobSystem jobs;
    AppSession session;
    std::thread thread;
    std::thread::id frame_id;
    std::atomic<bool> stop{false};
    std::atomic<bool> started{false};
    std::atomic<uint64_t> frames{0};
    std::atomic<bool> capture_armed{false};
    std::atomic<int> capture_requests{0};
    std::mutex m;
    std::vector<double> frame_ms;
    std::vector<std::array<std::string, 3>> activity;
    std::ostringstream err;   // read only after finish()

    void run(AppSession::Options o, AppSession::Hooks h = {}) {
        h.err = &err;
        h.log = [this](const std::string& e, const std::string& ob, const std::string& d) {
            std::lock_guard<std::mutex> lock(m);
            activity.push_back({e, ob, d});
        };
        h.run_job = [this](const std::string& label, std::function<void()> fn) {
            auto* user = new std::function<void()>(std::move(fn));
            jobs.submit(label, [](void* u, const CaliperJobControl*) {
                std::unique_ptr<std::function<void()>> f(static_cast<std::function<void()>*>(u));
                (*f)();
            }, user);
        };
        if (!h.request_capture)
            h.request_capture = [this] {
                capture_armed = true;
                ++capture_requests;
            };
        if (!h.encode_png)
            h.encode_png = [](const uint8_t*, uint32_t w, uint32_t hh, std::string* png) {
                *png = "PNG:" + std::to_string(w) + "x" + std::to_string(hh);
                return true;
            };
        thread = std::thread([this, o, h]() mutable {
            frame_id = std::this_thread::get_id();
            session.start(o, h);
            started = true;
            while (!stop) {
                const auto t0 = std::chrono::steady_clock::now();
                session.frame();
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                {
                    std::lock_guard<std::mutex> lock(m);
                    frame_ms.push_back(ms);
                }
                if (capture_armed.exchange(false)) session.capture_ready(4, 2, std::vector<uint8_t>(32, 7));
                ++frames;
                std::this_thread::sleep_for(2ms);
            }
            session.close();   // on the frame thread, as the app does
        });
        while (!started) std::this_thread::sleep_for(1ms);
    }

    cb::Session& open() {
        cb::Session* s = session.wait_open(20s);
        REQUIRE_MESSAGE(s != nullptr, session.open_error());
        return *s;
    }

    void finish() {
        stop = true;
        if (thread.joinable()) thread.join();
    }

    std::vector<std::array<std::string, 3>> lines() {
        std::lock_guard<std::mutex> lock(m);
        return activity;
    }

    ~FrameLoop() { finish(); }
};

AppSession::Options options(const std::string& workspace, bool chosen = true) {
    AppSession::Options o;
    o.workspace = workspace;
    o.chosen = chosen;
    o.from = chosen ? "--workspace" : "default";
    return o;
}

cb::CommandResult call(cb::Session& s, cb::CommandInvocation i) {
    auto f = s.queue().submit(std::move(i));
    REQUIRE(f.wait_for(20s) == std::future_status::ready);
    return f.get();
}

}  // namespace

// ---- the program host: the programs of operators and params (live/live_programs.h)

namespace {

// A ProgramLink that records each PROGRAM and hands back the answers a case queues.
struct RecordingProgramLink final : caliper_live::ProgramLink {
    std::vector<caliper_net::wire::Program> sent;
    std::vector<caliper_live::ProgramAnswer> answers;
    uint32_t next = 1;
    bool ready() override { return true; }
    uint64_t epoch() override { return 1; }
    uint32_t send(caliper_net::wire::Program m) override {
        m.request_id = next++;
        sent.push_back(std::move(m));
        return sent.back().request_id;
    }
    std::vector<caliper_live::ProgramAnswer> take() override {
        std::vector<caliper_live::ProgramAnswer> out;
        out.swap(answers);
        return out;
    }
    std::vector<caliper_net::wire::Program> of(const std::string& op, const std::string& id) const {
        std::vector<caliper_net::wire::Program> out;
        for (const auto& m : sent)
            if (m.op == op && m.id == id) out.push_back(m);
        return out;
    }
};

const char* kProgramSpec = R"(caliper_spec = 1
[applet]
id = "programs"
[[source]]
id = "w"
catalog = "live:blocks.0.attn.proj.weight"
transform = [{ op = "surface", max_cells = 64 }]
[[program]]
id = "rows"
wgsl = """
struct Params { level: f32, count: u32 }
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.count) { out[id.x] = w[id.x] * params.level; }
}
"""
inputs = [{ var = "w", source = "live:layers.0.weight" }]
output = { kind = "tensor", shape = [16] }
params = { level = { control = "gain", default = 0.5 }, count = 16 }
)";

double number_of(const caliper_net::Json& o, const char* key) {
    const caliper_net::Json* j = o.is_object() ? o.find(key) : nullptr;
    return j && j->is_number() ? j->as_double() : -1.0;
}

}  // namespace

TEST_CASE("app session: the program host sends an operator's program once the catalog gives its shape, and params as op params") {
    const cb::spec::ParseResult parsed = cb::spec::parse_toml(kProgramSpec);
    REQUIRE(parsed.ok());
    caliper_live::AppProgramHost host;
    RecordingProgramLink link;
    host.reconcile(parsed.spec);
    host.pump(link, static_cast<cb::EventSink*>(nullptr));

    // The program goes out with its params at their defaults; the operator waits for a shape.
    auto sets = link.of(CALIPER_NET_PROGRAM_SET, "rows");
    REQUIRE(sets.size() == 1);
    const caliper_net::Json* params = sets[0].declaration.find("params");
    REQUIRE(params);
    CHECK(number_of(*params, "level") == 0.5);
    CHECK(number_of(*params, "count") == 16);
    CHECK(link.of(CALIPER_NET_PROGRAM_SET, "operator.w.0").empty());
    std::vector<cb::ProgramStatus> st = host.status();
    REQUIRE(st.size() == 1);
    CHECK(st[0].kind == "tensor");

    // A control moves, and the catalog gives (12, 40): op params for the program, a set for the operator.
    const cb::spec::SourceResolver resolver = [](const std::string& id) -> std::optional<cb::spec::ResolvedSource> {
        if (id != "live:blocks.0.attn.proj.weight") return std::nullopt;
        cb::spec::ResolvedSource r;
        r.kind = "live";
        r.dtype = "f32";
        r.shape = std::vector<std::int64_t>{12, 40};
        return r;
    };
    double gain = 0.75;
    const caliper_live::AppProgramHost::Evaluate evaluate = [&](const cb::spec::Reference& r) {
        return r.target == "gain" ? nlohmann::json(gain) : r.fallback;
    };
    CHECK(host.refresh(evaluate, &resolver));
    host.pump(link, static_cast<cb::EventSink*>(nullptr));
    CHECK(link.of(CALIPER_NET_PROGRAM_SET, "rows").size() == 1);   // no new set, no new hash
    auto ops = link.of(CALIPER_NET_PROGRAM_PARAMS, "rows");
    REQUIRE(ops.size() == 1);
    CHECK(number_of(ops[0].params, "level") == 0.75);
    auto op_sets = link.of(CALIPER_NET_PROGRAM_SET, "operator.w.0");
    REQUIRE(op_sets.size() == 1);
    const caliper_net::Json& d = op_sets[0].declaration;
    REQUIRE(d.find("builtin"));
    CHECK(d.find("builtin")->as_string() == "surface@1");
    const caliper_net::Json* shape = d.find("output")->find("shape");
    REQUIRE(shape);
    REQUIRE(shape->size() == 2);
    CHECK(shape->at(0).as_double() == 4);   // 12 by 40 in at most 64 cells pools with f = 3
    CHECK(shape->at(1).as_double() == 14);
    CHECK(op_sets[0].sha256.value_or("") == cb::operators::builtin_sha256("surface"));
    CHECK(number_of(*d.find("params"), "factor") == 3);

    // The same value again sends nothing; the script reports the operator running.
    CHECK_FALSE(host.refresh(evaluate, &resolver));
    caliper_live::ProgramAnswer a;
    a.programs = caliper_net::Json::array();
    caliper_net::Json item = caliper_net::Json::object();
    item.set("id", "operator.w.0");
    item.set("kind", "operator");
    item.set("state", "running");
    item.set("sha256", cb::operators::builtin_sha256("surface"));
    a.programs.push(std::move(item));
    link.answers.push_back(std::move(a));
    host.pump(link, static_cast<cb::EventSink*>(nullptr));
    st = host.status();
    REQUIRE(st.size() == 2);
    CHECK(st[1].id == "operator.w.0");
    CHECK(st[1].kind == "operator");
    CHECK(st[1].host == "script");
    CHECK(st[1].state == "running");

    // A new shape sends the operator again.
    const cb::spec::SourceResolver wider = [](const std::string&) -> std::optional<cb::spec::ResolvedSource> {
        cb::spec::ResolvedSource r;
        r.shape = std::vector<std::int64_t>{12, 50};
        return r;
    };
    CHECK(host.refresh(evaluate, &wider));
    host.pump(link, static_cast<cb::EventSink*>(nullptr));
    CHECK(link.of(CALIPER_NET_PROGRAM_SET, "operator.w.0").size() == 2);
}

TEST_CASE("app session: commands from another thread run on the frame thread, in order, a few per frame") {
    TempDir t("order");
    FrameLoop loop;
    AppSession::Options o = options(t.str("ws"));
    o.budget_ms = 4.0;
    loop.run(o);
    cb::Session& s = loop.open();

    std::mutex m;
    std::vector<std::pair<int, uint64_t>> runs;   // (n, the frame it ran in)
    std::vector<std::thread::id> threads;
    REQUIRE(s.registry().add(toy("toy.slow", true), [&](const cb::CommandContext& ctx) {
                             std::this_thread::sleep_for(3ms);
                             std::lock_guard<std::mutex> lock(m);
                             runs.emplace_back(ctx.invocation.args.value("n", -1), loop.frames.load());
                             threads.push_back(std::this_thread::get_id());
                             return cb::CommandOutput::ok({});
                         }).ok());
    std::vector<std::future<cb::CommandResult>> futures;
    for (int n = 0; n < 20; ++n) futures.push_back(s.queue().submit(inv("toy.slow", {{"n", n}})));
    for (auto& f : futures) {
        REQUIRE(f.wait_for(20s) == std::future_status::ready);
        CHECK(f.get().ok);
    }
    loop.finish();
    REQUIRE(runs.size() == 20);
    std::map<uint64_t, int> per_frame;
    for (size_t i = 0; i < runs.size(); ++i) {
        CHECK(runs[i].first == (int)i);   // in submission order
        CHECK(threads[i] == loop.frame_id);
        ++per_frame[runs[i].second];
    }
    // 3 ms each against a budget of 4 ms: at most two per frame, so the queue
    // took at least ten frames and never held one for long.
    for (const auto& f : per_frame) CHECK(f.second <= 2);
    CHECK(per_frame.size() >= 10);
}

TEST_CASE("app session: a long loader runs on the job system and never holds a frame") {
    TempDir t("worker");
    FrameLoop loop;
    loop.run(options(t.str("ws")));
    cb::Session& s = loop.open();

    std::atomic<bool> loader_running{false};
    std::thread::id loader_thread, mark_thread;
    std::atomic<bool> mark_ran_during_loader{false};
    REQUIRE(s.registry().bind("loader.run", [&](const cb::CommandContext& ctx) {
                             loader_running = true;
                             loader_thread = std::this_thread::get_id();
                             std::this_thread::sleep_for(600ms);
                             loader_running = false;
                             return cb::CommandOutput::ok({{"id", ctx.invocation.args.at("id")}, {"outputs", json::array()},
                                                           {"elapsed_ms", 600}});
                         }).ok());
    REQUIRE(s.registry().add(toy("toy.mark", true), [&](const cb::CommandContext&) {
                             mark_thread = std::this_thread::get_id();
                             if (loader_running) mark_ran_during_loader = true;
                             return cb::CommandOutput::ok({});
                         }).ok());

    auto ran = s.queue().submit(inv("loader.run", {{"id", "slow"}}));
    auto mark = s.queue().submit(inv("toy.mark"));
    // The worker starts at the next frame; while it runs, frames keep coming.
    const auto until = std::chrono::steady_clock::now() + 5s;
    while (!loader_running && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(1ms);
    REQUIRE(loader_running);
    CHECK(loop.session.worker_busy());
    bool labelled = false;
    for (const auto& v : loop.jobs.views()) labelled = labelled || (v.running && v.label == "loader.run slow");
    CHECK(labelled);
    size_t first;
    uint64_t frames_before;
    {
        std::lock_guard<std::mutex> lock(loop.m);
        first = loop.frame_ms.size();
        frames_before = loop.frames.load();
    }
    REQUIRE(ran.wait_for(10s) == std::future_status::ready);
    const cb::CommandResult r = ran.get();
    CHECK(r.ok);
    const uint64_t frames_during = loop.frames.load() - frames_before;
    REQUIRE(mark.wait_for(10s) == std::future_status::ready);
    CHECK(mark.get().ok);
    loop.finish();

    CHECK(loader_thread != loop.frame_id);
    CHECK(mark_thread == loop.frame_id);
    CHECK_FALSE(mark_ran_during_loader);   // the later command waited, in order
    CHECK(frames_during >= 20);            // about 600 ms of frames at 2 ms each
    double longest = 0.0;
    for (size_t i = first; i < loop.frame_ms.size(); ++i) longest = std::max(longest, loop.frame_ms[i]);
    CHECK(longest < 100.0);
}

TEST_CASE("app session: shutdown answers waiting commands and closes the journal and the database") {
    TempDir t("shutdown");
    const std::string ws = t.str("ws");
    std::future<cb::CommandResult> slow, a, b, c;
    std::string err;
    {
        FrameLoop loop;
        loop.run(options(ws));
        cb::Session& s = loop.open();
        CHECK(call(s, inv("spec.new", {{"id", "kept"}, {"title", "Kept"}})).ok);
        std::atomic<bool> running{false};
        REQUIRE(s.registry().bind("loader.run", [&](const cb::CommandContext&) {
                                 running = true;
                                 std::this_thread::sleep_for(400ms);
                                 return cb::CommandOutput::ok({{"id", "x"}, {"outputs", json::array()}, {"elapsed_ms", 400}});
                             }).ok());
        slow = s.queue().submit(inv("loader.run", {{"id", "x"}}));
        a = s.queue().submit(inv("spec.get"));
        b = s.queue().submit(inv("panel.create", {{"id", "late"}, {"primitive", "text"}, {"options", {{"text", "t"}}}}));
        c = s.queue().submit(inv("history.undo"));
        while (!running) std::this_thread::sleep_for(1ms);
        loop.finish();   // closes the session on the frame thread while the loader runs
        err = loop.err.str();
    }
    // The running command finished; the waiting ones were answered.
    REQUIRE(slow.wait_for(0s) == std::future_status::ready);
    CHECK(slow.get().ok);
    CHECK(a.get().error.code == "command.shutdown");
    CHECK(b.get().error.code == "command.shutdown");
    CHECK(c.get().error.code == "command.shutdown");
    CHECK(err.find("[session] workspace " + ws) != std::string::npos);
    CHECK(err.find("closed") != std::string::npos);

    // The journal holds what ran, and the database is closed: the workspace
    // opens again in this process.
    const auto read = cb::read_journal(ws + "/journal/commands.jsonl");
    REQUIRE(read.error.ok());
    std::vector<std::string> names;
    for (const auto& e : read.events) names.push_back(e.name);
    CHECK(names == std::vector<std::string>{"spec.new", "loader.run"});
    auto again = cb::Session::open(ws);
    REQUIRE_MESSAGE(again.session != nullptr, again.error.message);
    const cb::SpecReplay replay = cb::replay_spec_journal(ws);
    REQUIRE(replay.spec.has_value());
    CHECK(replay.spec->applet.id == "kept");
}

namespace {
struct FakeRenderer final : caliper_host::PanelRenderSeam {
    std::string panel;
    cb::CommandOutput render_panel(const cb::spec::Spec&, const cb::PanelData& data, int width, int height) override {
        panel = data.panel;
        return cb::CommandOutput::ok({{"png_base64", "AAAA"}, {"width", width}, {"height", height}});
    }
};
}  // namespace

TEST_CASE("app session: the host serves app.info, app.applets, app.launch, capture.frame and panel.render") {
    TempDir t("host");
    FrameLoop loop;
    AppSession::Hooks h;
    h.info_json = [] {
        return std::string(R"({"abi_epoch":2,"host":"0.6.0","libtorch":null,"sdk":"0.2.0",)"
                           R"("services":["caliper.log.v1"],"ui_stack":null})");
    };
    h.applets_json = [] {
        return std::string(R"({"applets":[{"status":"READY","id":"dev.example.one","version":"1.0.0",)"
                           R"("folder":"/x","reason":""}]})");
    };
    h.launch = [](const std::string& id, const std::string&) -> LaunchOutcome {
        if (id == "dev.example.one")
            return {200, R"({"id":"dev.example.one","version":"1.0.0","status":"ACTIVE","already_open":false})", ""};
        if (id == "dev.example.bad") return {409, "dev.example.bad cannot be launched (REFUSED): too new", "app.launch_refused"};
        return {404, "no applet with the id " + id, "app.applet_not_found"};
    };
    loop.run(options(t.str("ws")), h);
    cb::Session& s = loop.open();

    const cb::CommandResult info = call(s, inv("app.info"));
    REQUIRE(info.ok);
    CHECK(info.result["host"] == "0.6.0");
    CHECK(info.result["libtorch"].is_null());
    const cb::CommandResult applets = call(s, inv("app.applets"));
    REQUIRE(applets.ok);
    CHECK(applets.result["applets"][0]["id"] == "dev.example.one");
    const cb::CommandResult launched = call(s, inv("app.launch", {{"id", "dev.example.one"}}));
    REQUIRE(launched.ok);
    CHECK(launched.result["status"] == "ACTIVE");
    const cb::CommandResult refused = call(s, inv("app.launch", {{"id", "dev.example.bad"}}));
    CHECK(refused.error.code == "app.launch_refused");
    CHECK(refused.error.message.find("too new") != std::string::npos);
    CHECK(refused.error.where == "args.id");
    CHECK(call(s, inv("app.launch", {{"id", "dev.example.none"}})).error.code == "app.applet_not_found");

    // capture.frame waits for the read-back of the next frame.
    const cb::CommandResult frame = call(s, inv("capture.frame"));
    REQUIRE_MESSAGE(frame.ok, frame.error.message);
    CHECK(frame.result["png_base64"] == b64("PNG:4x2"));
    CHECK(frame.result["width"] == 4);
    CHECK(frame.result["height"] == 2);
    CHECK(loop.capture_requests == 1);

    // panel.render: command.not_available until the spec runtime gives a renderer.
    REQUIRE(call(s, inv("spec.new", {{"id", "demo"}, {"title", "Demo"}})).ok);
    REQUIRE(call(s, inv("panel.create", {{"id", "note"}, {"primitive", "text"}, {"options", {{"text", "a note"}}}})).ok);
    const cb::CommandResult none = call(s, inv("panel.render", {{"id", "note"}}));
    CHECK(none.error.code == "command.not_available");
    CHECK(none.error.message.find("spec runtime") != std::string::npos);
    CHECK(call(s, inv("panel.render", {{"id", "missing"}})).error.code == "panel.not_found");
    auto fake = std::make_shared<FakeRenderer>();
    loop.session.set_panel_renderer(fake);
    const cb::CommandResult drawn = call(s, inv("panel.render", {{"id", "note"}, {"width", 320}, {"height", 200}}));
    REQUIRE(drawn.ok);
    CHECK(drawn.result["width"] == 320);
    CHECK(fake->panel == "note");
    loop.session.set_panel_renderer(nullptr);
    CHECK(call(s, inv("panel.render", {{"id", "note"}})).error.code == "command.not_available");
}

TEST_CASE("app session: where the app cannot read its frame back, capture.frame says why at once") {
    TempDir t("nocapture");
    FrameLoop loop;
    AppSession::Hooks h;
    h.capture_unavailable = "the gl renderer cannot read the window's frame back";
    loop.run(options(t.str("ws")), h);
    cb::Session& s = loop.open();
    const cb::CommandResult r = call(s, inv("capture.frame"));
    CHECK(r.error.code == "capture.unavailable");
    CHECK(r.error.message == "the gl renderer cannot read the window's frame back");
    CHECK(loop.capture_requests == 0);
}

TEST_CASE("app session: one Activity line per journaled command, with its origin and its error") {
    TempDir t("activity");
    FrameLoop loop;
    loop.run(options(t.str("ws")));
    cb::Session& s = loop.open();
    const json note = {{"id", "note"}, {"primitive", "text"}, {"options", {{"text", "a note"}}}};
    REQUIRE(call(s, inv("spec.new", {{"id", "demo"}, {"title", "Demo"}})).ok);
    json dry = note;
    dry["dry_run"] = true;
    REQUIRE(call(s, inv("panel.create", dry, cb::CommandOrigin::mcp)).ok);   // a preview: no line
    // From api, whose spec edits are not reviewed, so the edit runs at once.
    REQUIRE(call(s, inv("panel.create", note, cb::CommandOrigin::api)).ok);
    CHECK(call(s, inv("panel.create", note, cb::CommandOrigin::agent)).error.code == "panel.exists");
    REQUIRE(call(s, inv("spec.get")).ok);   // a read: no line
    REQUIRE(call(s, inv("history.undo")).ok);
    // The lines reach the Activity pane at the frame after the command.
    const auto until = std::chrono::steady_clock::now() + 5s;
    while (loop.lines().size() < 5 && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(2ms);
    loop.finish();
    const auto lines = loop.lines();
    REQUIRE(lines.size() == 5);
    CHECK(lines[0] == std::array<std::string, 3>{"workspace", t.str("ws"), "--workspace"});
    CHECK(lines[1] == std::array<std::string, 3>{"ui command", "spec.new", ""});
    CHECK(lines[2] == std::array<std::string, 3>{"api command", "panel.create", ""});
    CHECK(lines[3] == std::array<std::string, 3>{"agent command", "panel.create", "panel.exists"});
    CHECK(lines[4] == std::array<std::string, 3>{"ui command", "history.undo", ""});
    const std::string err = loop.err.str();
    CHECK(err.find("[session] command spec.new (ui)\n") != std::string::npos);
    CHECK(err.find("[session] command panel.create (agent): panel.exists\n") != std::string::npos);
}

TEST_CASE("app session: the default workspace opens only on demand, a chosen one after the first frame") {
    TempDir t("lazy");
    {
        FrameLoop loop;
        loop.run(options(t.str("default-ws"), false));
        while (loop.frames < 30) std::this_thread::sleep_for(1ms);
        CHECK_FALSE(fs::exists(t.str("default-ws")));   // nothing is created until something needs it
        CHECK(loop.session.session() == nullptr);
        loop.open();
        CHECK(fs::exists(t.str("default-ws") + "/journal/commands.jsonl"));
    }
    // A chosen workspace: nothing before the first frame has been drawn, then
    // opened on a worker and adopted by a later frame. This thread is the
    // frame thread here.
    AppSession session;
    std::ostringstream err;
    AppSession::Hooks h;
    h.err = &err;
    session.start(options(t.str("chosen")), h);
    session.frame();
    CHECK(session.session() == nullptr);
    CHECK_FALSE(fs::exists(t.str("chosen")));
    const auto until = std::chrono::steady_clock::now() + 20s;
    while (!session.session() && std::chrono::steady_clock::now() < until) {
        session.frame();
        std::this_thread::sleep_for(2ms);
    }
    REQUIRE(session.session() != nullptr);
    CHECK(session.wait_open(0ms) == session.session());   // never waits on the frame thread
    session.close();
    CHECK(err.str().find("[session] workspace " + t.str("chosen") + " (--workspace) opened in ") != std::string::npos);
}

TEST_CASE("app session: a workspace that cannot be opened leaves the app running without the builder") {
    TempDir t("bad");
    write_text(t.str("file"), "not a folder");
    FrameLoop loop;
    loop.run(options(t.str("file")));
    CHECK(loop.session.wait_open(20s) == nullptr);
    CHECK_FALSE(loop.session.open_error().empty());
    const auto until = std::chrono::steady_clock::now() + 5s;
    while (loop.lines().empty() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(2ms);
    const uint64_t frames = loop.frames;
    std::this_thread::sleep_for(20ms);
    CHECK(loop.frames > frames);   // the frames go on
    loop.finish();
    const auto lines = loop.lines();
    REQUIRE(lines.size() == 1);
    CHECK(lines[0][0] == "workspace not opened");
    CHECK(lines[0][1] == t.str("file"));
    CHECK(lines[0][2] == loop.session.open_error());
    CHECK(loop.err.str().find("[session] workspace not opened: " + t.str("file") + ": ") != std::string::npos);
}

TEST_CASE("app session: the command script runs through the queue and writes each result") {
    TempDir t("script");
    write_text(t.str("script.jsonl"),
               "# a demo\n"
               "{\"name\": \"spec.new\", \"args\": {\"id\": \"demo\", \"title\": \"Demo\"}}\n"
               "\n"
               "not json\n"
               // An agent's spec edit is reviewed: approved by the script...
               "{\"name\": \"panel.create\", \"origin\": \"mcp\", \"approve\": true, \"args\": {\"id\": \"note\", "
               "\"primitive\": \"text\", \"options\": {\"text\": \"a note\"}}}\n"
               // ...or left pending, and the script goes on.
               "{\"name\": \"panel.create\", \"origin\": \"mcp\", \"args\": {\"id\": \"note2\", "
               "\"primitive\": \"text\", \"options\": {\"text\": \"another\"}}}\n"
               "{\"name\": \"capture.frame\"}\n");
    FrameLoop loop;
    AppSession::Options o = options(t.str("ws"));
    o.script = t.str("script.jsonl");
    o.results = t.str("results.jsonl");
    loop.run(o);
    CHECK(loop.session.script_running());
    const auto until = std::chrono::steady_clock::now() + 20s;
    while (loop.session.script_running() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(2ms);
    CHECK_FALSE(loop.session.script_running());
    loop.finish();

    std::vector<json> results;
    std::istringstream in(read_text(t.str("results.jsonl")));
    for (std::string line; std::getline(in, line);) results.push_back(json::parse(line));
    REQUIRE(results.size() == 6);
    CHECK(results[0]["line"] == 2);
    CHECK(results[0]["name"] == "spec.new");
    CHECK(results[0]["ok"] == true);
    CHECK(results[1]["line"] == 4);
    CHECK(results[1]["error"]["code"] == "command.invalid_invocation");
    CHECK(results[2]["name"] == "panel.create");
    CHECK(results[2]["ok"] == true);
    CHECK(results[2]["approved"].get<std::string>().rfind("ap-", 0) == 0);
    CHECK(results[3]["name"] == "panel.create");
    CHECK(results[3]["error"]["code"] == "command.needs_approval");
    CHECK(results[3]["error"]["approval"].get<std::string>().rfind("ap-", 0) == 0);
    CHECK(results[4]["name"] == "capture.frame");
    CHECK(results[4]["result"]["png_base64"] == b64("PNG:4x2"));
    CHECK(results[5] == json{{"done", true}, {"commands", 5}, {"failed", 2}});
    const auto read = cb::read_journal(t.str("ws") + "/journal/commands.jsonl");
    REQUIRE(read.events.size() >= 2);
    CHECK(read.events[1].name == "panel.create");
    CHECK(read.events[1].origin == cb::CommandOrigin::mcp);
    CHECK(read.events[1].ok);
    CHECK(loop.err.str().find("[session] script done: 5 commands, 2 failed") != std::string::npos);
}

TEST_CASE("app session: an agent's spec edit is held with its diff in the pending list and runs on approval") {
    TempDir t("review");
    FrameLoop loop;
    loop.run(options(t.str("ws")));
    cb::Session& s = loop.open();
    REQUIRE(call(s, inv("spec.new", {{"id", "demo"}, {"title", "Demo"}})).ok);
    const auto gateway = loop.session.command_gateway();
    REQUIRE(gateway);

    // From mcp, reviewed by default: the edit waits for the person.
    auto held = s.queue().submit(
        inv("panel.create", {{"id", "note"}, {"primitive", "text"}, {"options", {{"text", "a note"}}}},
            cb::CommandOrigin::mcp));
    std::string id;
    json pending;
    const auto until = std::chrono::steady_clock::now() + 10s;
    while (id.empty() && std::chrono::steady_clock::now() < until) {
        for (const auto& p : s.registry().pending_approvals()) {
            id = p.approval;
            pending = cb::pending_approval_to_json(p);
        }
        std::this_thread::sleep_for(2ms);
    }
    REQUIRE_FALSE(id.empty());
    CHECK(held.wait_for(50ms) == std::future_status::timeout);
    INFO(pending.dump());
    CHECK(pending["invocation"]["name"] == "panel.create");
    REQUIRE(pending["review"].is_object());
    const std::string diff = pending["review"]["diff"].get<std::string>();
    CHECK(diff.find("+id = \"note\"") != std::string::npos);
    CHECK(s.open_spec()->spec.panels.empty());

    // GET /v0/approvals/{id} gives the same review, read from the pending
    // approvals of the last frame.
    json looked = json::object();
    while (looked.value("state", "") != "pending" && std::chrono::steady_clock::now() < until) {
        looked = json::parse(gateway->approval_json(id));
        std::this_thread::sleep_for(2ms);
    }
    CHECK(looked["state"] == "pending");
    CHECK(looked["name"] == "panel.create");
    CHECK(looked["origin"] == "mcp");
    CHECK(looked["review"]["diff"] == diff);

    // Approved, it runs as it was shown.
    REQUIRE(s.queue().approve(id).get().ok);
    REQUIRE(held.wait_for(10s) == std::future_status::ready);
    CHECK(held.get().ok);
    CHECK(s.open_spec()->spec.panels.size() == 1);
    CHECK(json::parse(gateway->approval_json(id))["state"] == "decided");
}

// The app shows the spec page for every applet.open that succeeds, from any
// origin, and for a new or opened spec of an agent (the origins agent and
// mcp), since the person reviews an agent's edits on that page and would
// otherwise not see the spec the side chat's agent builds from the live page.
// The person's own spec.new (F3, a drop) shows the page by its own path, in
// build mode, and a program with the person's token (api) leaves the page alone.
TEST_CASE("app session: the spec page shows for every applet.open and for an agent's new or opened spec") {
    TempDir t("specs-opened");
    FrameLoop loop;
    loop.run(options(t.str("ws")));
    cb::Session& s = loop.open();
    auto opened = [&](uint64_t want) {
        const auto until = std::chrono::steady_clock::now() + 5s;
        while (loop.session.specs_opened() != want && std::chrono::steady_clock::now() < until)
            std::this_thread::sleep_for(2ms);
        return loop.session.specs_opened();
    };
    REQUIRE(call(s, inv("spec.new", {{"id", "mine"}, {"title", "Mine"}})).ok);   // ui
    CHECK(opened(0) == 0);
    REQUIRE(call(s, inv("applet.save")).ok);
    REQUIRE(call(s, inv("spec.new", {{"id", "theirs"}, {"title", "Theirs"}}, cb::CommandOrigin::agent)).ok);
    CHECK(opened(1) == 1);
    REQUIRE(call(s, inv("applet.save")).ok);
    REQUIRE(call(s, inv("spec.open", {{"path", "applets/mine/mine.spec.toml"}}, cb::CommandOrigin::mcp)).ok);
    CHECK(opened(2) == 2);
    REQUIRE(call(s, inv("spec.new", {{"id", "program"}, {"title", "Program"}}, cb::CommandOrigin::api)).ok);
    CHECK(opened(2) == 2);
    CHECK_FALSE(call(s, inv("spec.new", {{"id", "Bad Id"}}, cb::CommandOrigin::agent)).ok);
    CHECK(opened(2) == 2);
    REQUIRE(call(s, inv("applet.save")).ok);
    REQUIRE(call(s, inv("applet.open", {{"id", "theirs"}})).ok);   // ui
    CHECK(opened(3) == 3);
}

TEST_CASE("spec page: Command K and Escape have one meaning each, by whether a panel has the focus") {
    using caliper_host::spec_key_action;
    using A = caliper_host::SpecKeyAction;
    CHECK(spec_key_action(true, false, true) == A::panel_prompt);    // the inline prompt in the panel
    CHECK(spec_key_action(true, false, false) == A::page_command);   // the page's command input
    CHECK(spec_key_action(false, true, true) == A::unfocus);         // Escape gives the focus up first
    CHECK(spec_key_action(false, true, false) == A::home);           // then goes Home
    CHECK(spec_key_action(false, false, true) == A::none);
    CHECK(spec_key_action(true, true, true) == A::panel_prompt);     // both in one frame: the command key
}

TEST_CASE("app session: Caliper API's gateway opens the default workspace on the first command") {
    TempDir t("gateway");
    FrameLoop loop;
    loop.run(options(t.str("default-ws"), false));
    const auto gateway = loop.session.command_gateway();
    REQUIRE(gateway);
    CHECK(json::parse(gateway->table_json())["commands"].size() > 30);
    while (loop.frames < 10) std::this_thread::sleep_for(1ms);
    CHECK_FALSE(fs::exists(t.str("default-ws")));   // the table needs no workspace

    caliper_api::CommandRequest r;
    r.id = "api-1";
    r.name = "spec.new";
    r.args = R"({"id": "demo"})";
    r.origin = "api";
    std::unique_ptr<caliper_api::PendingCommand> p = gateway->submit(r);
    const auto until = std::chrono::steady_clock::now() + 20s;
    while (!p->ready() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(2ms);
    REQUIRE(p->ready());
    const json env = json::parse(p->envelope());
    CHECK(env["id"] == "api-1");
    CHECK(env["ok"] == true);
    CHECK(fs::exists(t.str("default-ws") + "/journal/commands.jsonl"));
    CHECK(json::parse(gateway->approval_json("ap-9")) ==
          json{{"approval", "ap-9"}, {"state", "unknown"}, {"result", nullptr}});

    // A refused request answers at once; after the session closes, commands
    // answer command.shutdown.
    r.origin = "ui";
    std::unique_ptr<caliper_api::PendingCommand> bad = gateway->submit(r);
    REQUIRE(bad->ready());
    CHECK(json::parse(bad->envelope())["error"]["code"] == "command.invalid_invocation");
    loop.finish();
    r.origin = "api";
    std::unique_ptr<caliper_api::PendingCommand> late = gateway->submit(r);
    REQUIRE(late->ready());
    CHECK(json::parse(late->envelope())["error"]["code"] == "command.shutdown");
}

#ifndef _WIN32
TEST_CASE("app session: the hidden app runs a command script on its workspace and closes it on SIGTERM") {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const char* applets = std::getenv("CALIPER_TEST_APPLETS");
    if (!exe || !*exe || !applets || !*applets) {
        MESSAGE("CALIPER_TEST_CALIPER_EXE or CALIPER_TEST_APPLETS is not set: run the ctest entry "
                "caliper_app_session_hidden");
        return;
    }
    TempDir t("app");
    const std::string ws = t.str("ws");
    const std::string data = t.str("data");
    const std::string csv = t.str("inputs/temps.csv");
    std::string rows = "step,temp,pressure\n";
    for (int i = 0; i < 50; ++i)
        rows += std::to_string(i) + "," + std::to_string(20.0 + 0.1 * i) + "," + std::to_string(1.0 + 0.01 * i) + "\n";
    write_text(csv, rows);
    const std::vector<json> script = {
        {{"name", "data.import"}, {"args", {{"paths", {csv}}, {"run", false}}}},
        {{"name", "loader.run"}, {"args", {{"id", "temps"}}}},
        {{"name", "spec.new"}, {"args", {{"id", "temps-board"}, {"title", "Temperatures"}}}},
        {{"name", "source.add"}, {"args", {{"id", "t"}, {"catalog", "table:temps"}}}},
        {{"name", "panel.create"},
         {"args", {{"id", "curve"}, {"primitive", "series"}, {"title", "Temperature"},
                   {"bind", {{"x", "t.step"}, {"y", {"t.temp"}}}}}}},
        {{"name", "panel.create"},
         {"args", {{"id", "rows"}, {"primitive", "table"}, {"bind", {{"rows", "t"}}},
                   {"options", {{"columns", {"step", "temp"}}}}}}},
        {{"name", "applet.save"}, {"args", json::object()}},
        {{"name", "app.info"}},
        {{"name", "app.applets"}},
        {{"name", "app.launch"}, {"args", {{"id", "dev.caliper.hello"}}}},
        {{"name", "capture.frame"}},
        {{"name", "panel.render"}, {"args", {{"id", "curve"}}}},
    };
    std::string text;
    for (const json& j : script) text += j.dump() + "\n";
    write_text(t.str("script.jsonl"), text);
    fs::create_directories(t.str("user-applets"));

    Proc p;
    REQUIRE(p.start(exe, {"--hidden", "--applets", applets, "--workspace", ws},
                    {"CALIPER_DATA_DIR=" + data, "CALIPER_USER_APPLETS_DIR=" + t.str("user-applets"),
                     "CALIPER_WINDOW_SIZE=640x400", "CALIPER_COMMAND_SCRIPT=" + t.str("script.jsonl"),
                     "CALIPER_COMMAND_RESULTS=" + t.str("results.jsonl")}));
    REQUIRE_MESSAGE(p.read_until("[session] script done:", 90000), p.text);
    const bool metal = p.text.find("[renderer] metal") != std::string::npos;

    std::vector<json> results;
    std::istringstream in(read_text(t.str("results.jsonl")));
    for (std::string line; std::getline(in, line);) results.push_back(json::parse(line));
    REQUIRE(results.size() == script.size() + 1);
    for (size_t i = 0; i + 4 < script.size(); ++i) {
        CAPTURE(results[i].dump());
        CHECK(results[i]["ok"] == true);
    }
    CHECK(results[1]["result"]["outputs"][0]["rows"] == 50);
    CHECK(results[6]["result"]["manifest"] == "applets/temps-board/temps-board.caliper.toml");
    CHECK(results[7]["result"]["services"].size() > 0);
    CHECK(results[8]["ok"] == true);
    // app.launch opens the fixture's page, as POST /v0/applets/{id}/launch does.
    REQUIRE_MESSAGE(results[9]["ok"] == true, results[9].dump());
    CHECK(results[9]["result"]["id"] == "dev.caliper.hello");
    CHECK(results[9]["result"]["status"] == "ACTIVE");
    if (metal) {
        REQUIRE_MESSAGE(results[10]["ok"] == true, results[10].dump());
        const std::string png = unb64(results[10]["result"]["png_base64"].get<std::string>());
        int w = 0, h = 0, n = 0;
        unsigned char* px = stbi_load_from_memory((const unsigned char*)png.data(), (int)png.size(), &w, &h, &n, 4);
        REQUIRE(px != nullptr);
        stbi_image_free(px);
        CHECK(((w == 640 && h == 400) || (w == 1280 && h == 800)));
        CHECK(results[10]["result"]["width"] == w);
    } else {
        CHECK(results[10]["error"]["code"] == "capture.unavailable");
    }
    // panel.render draws the panel with the spec runtime on a canvas that
    // shares the app's font atlas.
    if (metal) {
        REQUIRE_MESSAGE(results[11]["ok"] == true, results[11].dump());
        CHECK(results[11]["result"]["width"] == 800);
        CHECK(results[11]["result"]["height"] == 500);
    } else {
        CHECK(results[11]["error"]["code"] == "panel.render_failed");
    }
    CHECK(results.back() == json{{"done", true}, {"commands", (int)script.size()}, {"failed", metal ? 0 : 2}});

    // SIGTERM closes the hidden window as a click would; the session closes
    // its journal and its database before the process ends.
    REQUIRE(p.pid > 0);
    ::kill(p.pid, SIGTERM);
    CHECK_MESSAGE(p.wait(30000) == 0, p.text);
    CHECK(p.text.find("[session] workspace " + ws + " (--workspace) opened in ") != std::string::npos);
    CHECK(p.text.find("[session] command loader.run (ui)") != std::string::npos);
    CHECK(p.text.find("[session] command applet.save (ui)") != std::string::npos);
    CHECK(p.text.find("[session] workspace " + ws + " closed") != std::string::npos);

    CHECK(fs::is_regular_file(ws + "/applets/temps-board/temps-board.spec.toml"));
    CHECK(fs::is_regular_file(ws + "/applets/temps-board/temps-board.caliper.toml"));
    CHECK(fs::is_regular_file(ws + "/loaders/temps.loader.toml"));
    CHECK(fs::is_regular_file(ws + "/workspace.duckdb"));
    CHECK_FALSE(fs::exists(data + "/workspace"));   // the chosen workspace, not the default one
    const auto read = cb::read_journal(ws + "/journal/commands.jsonl");
    REQUIRE(read.error.ok());
    std::vector<std::string> names;
    for (const auto& e : read.events) {
        CHECK(e.origin == cb::CommandOrigin::ui);
        CHECK(e.ok);
        names.push_back(e.name);
    }
    CHECK(names == std::vector<std::string>{"data.import", "loader.run", "spec.new", "source.add", "panel.create",
                                            "panel.create", "applet.save"});
    // The workspace the app left opens here to the same spec.
    const cb::SpecReplay replay = cb::replay_spec_journal(ws);
    REQUIRE(replay.spec.has_value());
    CHECK(replay.spec->panels.size() == 2);
    auto again = cb::Session::open(ws);
    REQUIRE_MESSAGE(again.session != nullptr, again.error.message);
    const cb::CommandResult listed = again.session->invoke(inv("applet.list"));
    REQUIRE(listed.ok);
    CHECK(listed.result["applets"][0]["id"] == "temps-board");
}
#endif

// The command script may claim the ui origin and approve what waits, so the
// app runs it only where CALIPER_TEST_APPROVALS works: hidden, or in caliper
// shot. No test opens a visible window, so the gate is checked where it is
// decided, in app_session_options().
TEST_CASE("app session: the command script runs only in the hidden app or caliper shot, never in a visible app") {
    std::string refused;
    // A visible app: refused whole, with the reason.
    AppSession::Options visible;
    visible.script = "script.jsonl";
    visible.results = "results.jsonl";
    caliper_host::gate_command_script(&visible, /*scripts_allowed=*/false, &refused);
    CHECK(visible.script.empty());
    CHECK(visible.results.empty());
    CHECK(refused.find("--hidden") != std::string::npos);
    // The hidden app and caliper shot run it.
    refused.clear();
    AppSession::Options hidden;
    hidden.script = "script.jsonl";
    hidden.results = "results.jsonl";
    caliper_host::gate_command_script(&hidden, /*scripts_allowed=*/true, &refused);
    CHECK(hidden.script == "script.jsonl");
    CHECK(hidden.results == "results.jsonl");
    CHECK(refused.empty());
    // Without a script there is nothing to refuse.
    AppSession::Options none;
    caliper_host::gate_command_script(&none, false, &refused);
    CHECK(refused.empty());
}

// ---- the app's run control (design/instruments.md, Run Control in App) ----------------

namespace {

// The attached script's run, faked: its CONTROLS, its access, what each
// request answers, and what was sent. The frame thread and the test's thread
// both read it, so it has a lock.
struct FakeLink final : caliper_live::RunLink {
    mutable std::mutex m;
    bool is_attached = true;
    std::string name = "train.py";
    caliper_live::ControlAccess acc{true, ""};
    bool have = true;
    caliper_live::Controls ctl;
    std::string refuse;   // what a request answers; "" sends it
    struct Sent {
        std::string op, name, value;
        uint32_t count = 0;
    };
    std::vector<Sent> sent;
    std::map<std::string, ActionResult> results;

    FakeLink() {
        ctl.run_state = "running";
        ctl.has_step = true;
        ctl.step = 12;
        ctl.version = 1;
        caliper_live::ControlParam lr;
        lr.name = "lr";
        lr.type = "number";
        lr.value = caliper_net::Json(0.01);
        lr.has_min = lr.has_max = true;
        lr.min = 1e-5;
        lr.max = 0.1;
        lr.log = true;
        lr.help = "learning rate";
        caliper_live::ControlParam batch;
        batch.name = "batch";
        batch.type = "integer";
        batch.value = caliper_net::Json(int64_t(32));
        batch.pending = caliper_net::Json(int64_t(64));
        caliper_live::ControlParam opt;
        opt.name = "optimizer";
        opt.type = "choice";
        opt.value = caliper_net::Json("adam");
        opt.choices = {"adam", "sgd"};
        ctl.params = {lr, batch, opt};
        caliper_live::ControlAction save;
        save.name = "save";
        save.help = "writes a checkpoint";
        ctl.actions = {save};
    }
    bool attached() const override {
        std::lock_guard<std::mutex> lock(m);
        return is_attached;
    }
    std::string script() const override {
        std::lock_guard<std::mutex> lock(m);
        return name;
    }
    const caliper_live::Controls* controls() const override {
        std::lock_guard<std::mutex> lock(m);
        return have ? &ctl : nullptr;
    }
    caliper_live::ControlAccess access() const override {
        std::lock_guard<std::mutex> lock(m);
        return acc;
    }
    std::string send(const std::string& op, const std::string& n, const caliper_net::Json& value,
                     uint32_t count) override {
        std::lock_guard<std::mutex> lock(m);
        if (!refuse.empty()) return refuse;
        sent.push_back({op, n, value.dump(), count});
        return "";
    }
    ActionResult action_result(const std::string& n) const override {
        std::lock_guard<std::mutex> lock(m);
        const auto it = results.find(n);
        return it == results.end() ? ActionResult{} : it->second;
    }
};

struct Recorder final : cb::EventSink {
    std::vector<cb::Event> events;
    std::uint64_t push(cb::Event e) override {
        e.seq = events.size() + 1;
        events.push_back(std::move(e));
        return events.back().seq;
    }
    std::vector<json> fields() const {
        std::vector<json> out;
        for (const auto& e : events) out.push_back(e.fields);
        return out;
    }
};

}  // namespace

TEST_CASE("app session: the run control reads the attached script's CONTROLS as run.state") {
    caliper_host::AppRunControl rc;
    CHECK_FALSE(rc.state().attached);
    FakeLink link;
    rc.set_link(&link);
    cb::RunState st = rc.state();
    CHECK(st.attached);
    CHECK(st.script == "train.py");
    CHECK(st.control);
    CHECK(st.why_not.empty());
    CHECK(st.state == "running");
    CHECK(st.step == std::optional<std::int64_t>(12));
    CHECK_FALSE(st.steps_left);
    REQUIRE(st.params.size() == 3);
    CHECK(st.params[0].value == 0.01);
    CHECK(st.params[0].pending.is_null());
    CHECK(st.params[0].min == std::optional<double>(1e-5));
    CHECK(st.params[0].log);
    CHECK(st.params[0].help == "learning rate");
    CHECK(st.params[1].value == 32);
    CHECK(st.params[1].value.is_number_integer());
    CHECK(st.params[1].pending == 64);
    CHECK_FALSE(st.params[1].min);
    CHECK(st.params[2].value == "adam");
    CHECK(st.params[2].choices == std::vector<std::string>{"adam", "sgd"});
    REQUIRE(st.actions.size() == 1);
    CHECK(st.actions[0].name == "save");
    CHECK(cb::run_state_json(st)["params"][0]["max"] == 0.1);

    link.ctl.run_state = "stepping";
    link.ctl.steps_left = 3;
    CHECK(rc.state().steps_left == std::optional<std::int64_t>(3));
    // A connection that takes no control says why, as a sentence.
    link.acc = {false, "the server did not grant control"};
    st = rc.state();
    CHECK_FALSE(st.control);
    CHECK(st.why_not == "The server did not grant control.");
    link.acc = {true, ""};
    link.have = false;
    st = rc.state();
    CHECK_FALSE(st.control);
    CHECK(st.why_not == "The script has sent no controls yet.");
    CHECK(st.params.empty());
    link.is_attached = false;
    CHECK_FALSE(rc.state().attached);
}

TEST_CASE("app session: the run control sends CONTROL requests and maps the refusals to run errors") {
    caliper_host::AppRunControl rc;
    CHECK(rc.set_param("lr", 0.02).code == cb::kRunNotAttached);
    FakeLink link;
    rc.set_link(&link);
    CHECK(rc.set_param("lr", 0.02).ok());
    CHECK(rc.set_param("batch", 128).ok());
    CHECK(rc.set_param("optimizer", "sgd").ok());
    CHECK(rc.run_action("save").ok());
    CHECK(rc.pause(cb::RunPauseOp::pause, 0).ok());
    CHECK(rc.pause(cb::RunPauseOp::resume, 0).ok());
    CHECK(rc.pause(cb::RunPauseOp::step, 5).ok());
    REQUIRE(link.sent.size() == 7);
    CHECK(link.sent[0].op == "set");
    CHECK(link.sent[0].name == "lr");
    CHECK(link.sent[0].value == "0.02");
    CHECK(link.sent[1].value == "128");
    CHECK(link.sent[2].value == "\"sgd\"");
    CHECK(link.sent[3].op == "run");
    CHECK(link.sent[3].name == "save");
    CHECK(link.sent[4].op == "pause");
    CHECK(link.sent[5].op == "resume");
    CHECK(link.sent[6].op == "step");
    CHECK(link.sent[6].count == 5);

    link.refuse = "set lr: the hub refused it";
    cb::Error e = rc.set_param("lr", 0.03);
    CHECK(e.code == cb::kRunRefused);
    CHECK(e.message == "Set lr: the hub refused it.");
    // Before the first step, RunControls takes no pause, step or action.
    link.ctl.run_state = "none";
    link.refuse = "pause: the script has called no checkpoint yet";
    CHECK(rc.pause(cb::RunPauseOp::pause, 0).code == cb::kRunNoStep);
    CHECK(rc.run_action("save").code == cb::kRunNoStep);
    link.acc = {false, "the stream takes no control"};
    e = rc.pause(cb::RunPauseOp::resume, 0);
    CHECK(e.code == cb::kRunControlNotAllowed);
    CHECK(e.message == "The stream takes no control.");
}

TEST_CASE("app session: the run control turns CONTROLS and action results into run events") {
    caliper_host::AppRunControl rc;
    FakeLink link;
    Recorder rec;
    rc.feed(rec);
    CHECK(rec.events.empty());   // no link: nothing attached, nothing changed

    // The attach, then the first CONTROLS as the baseline: the run state is
    // news, the registered values are not.
    rc.set_link(&link);
    rc.feed(rec);
    REQUIRE(rec.events.size() == 2);
    CHECK(rec.events[0].kind == cb::EventKind::run);
    CHECK(rec.events[0].origin == "script");
    CHECK(rec.events[0].panel.empty());
    CHECK(rec.fields()[0] == json{{"change", "attach"}, {"attached", true}, {"script", "train.py"}});
    CHECK(rec.fields()[1] == json{{"change", "state"}, {"state", "running"}, {"step", 12}});
    rc.feed(rec);
    CHECK(rec.events.size() == 2);   // nothing new

    // The script applies a pending value and a new one, and pauses.
    link.ctl.version = 2;
    link.ctl.step = 13;
    link.ctl.params[0].value = caliper_net::Json(0.05);   // lr, clamped by the script
    link.ctl.params[1].value = caliper_net::Json(int64_t(64));
    link.ctl.params[1].pending = caliper_net::Json();
    link.ctl.run_state = "paused";
    rc.feed(rec);
    REQUIRE(rec.events.size() == 5);
    CHECK(rec.fields()[2] == json{{"change", "state"}, {"state", "paused"}, {"step", 13}});
    CHECK(rec.fields()[3] == json{{"change", "param"}, {"name", "lr"}, {"value", 0.05}, {"step", 13}});
    CHECK(rec.fields()[4] == json{{"change", "param"}, {"name", "batch"}, {"value", 64}, {"step", 13}});

    // A value the script holds as pending is not applied yet.
    link.ctl.version = 3;
    link.ctl.params[0].pending = caliper_net::Json(0.001);
    rc.feed(rec);
    CHECK(rec.events.size() == 5);
    link.ctl.version = 4;
    link.ctl.params[0].pending = caliper_net::Json();
    link.ctl.params[0].value = caliper_net::Json(0.001);
    rc.feed(rec);
    REQUIRE(rec.events.size() == 6);
    CHECK(rec.fields()[5]["value"] == 0.001);

    // An action's end, once.
    link.results["save"] = {"saved at step 13", false, 100};
    rc.feed(rec);
    rc.feed(rec);
    REQUIRE(rec.events.size() == 7);
    CHECK(rec.fields()[6] == json{{"change", "action"}, {"name", "save"}, {"status", "ok"}, {"text", "saved at step 13"}});
    link.results["save"] = {"disk full", true, 200};
    rc.feed(rec);
    REQUIRE(rec.events.size() == 8);
    CHECK(rec.fields()[7]["status"] == "error");

    // The detach names the script that went.
    link.is_attached = false;
    rc.feed(rec);
    REQUIRE(rec.events.size() == 9);
    CHECK(rec.fields()[8] == json{{"change", "attach"}, {"attached", false}, {"script", "train.py"}});
}

TEST_CASE("app session: the session holds the app's run control, and the run commands act through it") {
    TempDir t("run");
    FakeLink link;   // outlives the frame loop, which reads it until it stops
    FrameLoop loop;
    loop.session.set_run_link(&link);   // before the frame thread starts
    loop.run(options(t.str("ws")));
    cb::Session& s = loop.open();
    CHECK(s.run_control() == loop.session.run_control());

    cb::CommandResult r = call(s, inv("run.state"));
    REQUIRE(r.ok);
    CHECK(r.result["script"] == "train.py");
    CHECK(r.result["params"].size() == 3);
    r = call(s, inv("run.set_param", {{"name", "lr"}, {"value", 0.02}}));
    REQUIRE(r.ok);
    CHECK(r.result["pending"] == true);
    r = call(s, inv("run.action", {{"name", "evaluate"}}));
    CHECK(r.error.code == cb::kRunUnknownAction);

    // An mcp client's request waits for the person's card, then runs.
    auto held = s.queue().submit(inv("run.set_param", {{"name", "lr"}, {"value", 0.03}}, cb::CommandOrigin::mcp));
    std::string id;
    const auto until = std::chrono::steady_clock::now() + 10s;
    while (id.empty() && std::chrono::steady_clock::now() < until) {
        for (const auto& p : s.registry().pending_approvals())
            if (p.invocation.name == "run.set_param") id = p.approval;
        std::this_thread::sleep_for(2ms);
    }
    REQUIRE_FALSE(id.empty());
    CHECK(held.wait_for(50ms) == std::future_status::timeout);
    {
        std::lock_guard<std::mutex> lock(link.m);
        REQUIRE(link.sent.size() == 1);
        CHECK(link.sent[0].value == "0.02");
    }
    REQUIRE(s.queue().approve(id).get().ok);
    REQUIRE(held.wait_for(10s) == std::future_status::ready);
    CHECK(held.get().ok);
    std::lock_guard<std::mutex> lock(link.m);
    REQUIRE(link.sent.size() == 2);
    CHECK(link.sent[1].value == "0.03");
}

TEST_CASE("app session: the run control reads the CONTROLS key interventions in the run control's form") {
    // The key as the script sends it (tests/fixtures/wire_v3/v3_2/controls_interventions.json).
    const json iv = json::parse(R"({"points": [
        {"point": "blocks.0.attn", "kind": "module", "module": "blocks.0.attn", "head": null, "type": "Attention",
         "shape": [8, 64, 128]},
        {"point": "blocks.0.attn#3", "kind": "head", "module": "blocks.0.attn", "head": 3, "type": "Attention",
         "shape": null}],
      "active": [
        {"point": "blocks.0.attn", "op": "ablate", "from_step": null, "until_step": null, "since_step": 1190,
         "state": "active"},
        {"point": "blocks.0.attn#3", "op": "ablate", "from_step": 1201, "until_step": 1300, "since_step": null,
         "state": "pending"}]})");
    const std::vector<cb::InterventionPoint> pts = caliper_host::intervention_points_of(iv);
    REQUIRE(pts.size() == 2);
    CHECK(pts[0].point == "blocks.0.attn");
    CHECK(pts[0].kind == "module");
    CHECK_FALSE(pts[0].head);
    CHECK(pts[0].shape == std::optional<std::vector<std::int64_t>>(std::vector<std::int64_t>{8, 64, 128}));
    CHECK(pts[1].kind == "head");
    CHECK(pts[1].head == std::optional<std::int64_t>(3));
    CHECK(pts[1].module == "blocks.0.attn");
    CHECK(pts[1].type == "Attention");
    CHECK_FALSE(pts[1].shape);
    const std::vector<cb::Intervention> ivs = caliper_host::interventions_of(iv);
    REQUIRE(ivs.size() == 2);
    CHECK(ivs[0].point == "blocks.0.attn#3");   // the pending one first
    CHECK(ivs[0].from_step == std::optional<std::int64_t>(1201));
    CHECK(ivs[0].until_step == std::optional<std::int64_t>(1300));
    CHECK_FALSE(ivs[0].since_step);
    CHECK(ivs[1].state == "active");
    CHECK(ivs[1].since_step == std::optional<std::int64_t>(1190));
    CHECK(caliper_host::intervention_points_of(json()).empty());
    CHECK(caliper_host::interventions_of(json::array()).empty());
}

TEST_CASE("app session: the run control reports each applied and each stopped intervention as a run event") {
    caliper_host::AppRunControl rc;
    Recorder rec;
    const auto key = [](const std::string& active) {
        return json::parse(R"({"points": [], "active": )" + active + "}");
    };
    // The first key is the baseline: what was applied before is no event.
    rc.feed_interventions(key(R"([{"point": "head", "op": "ablate", "from_step": null, "until_step": null,
                                   "since_step": 4, "state": "active"}])"),
                          12, true, rec);
    CHECK(rec.events.empty());
    rc.feed_interventions(key(R"([{"point": "head", "op": "ablate", "from_step": null, "until_step": null,
                                   "since_step": 4, "state": "active"},
                                  {"point": "blocks.0.attn#1", "op": "ablate", "from_step": null, "until_step": null,
                                   "since_step": null, "state": "pending"}])"),
                          13, false, rec);
    CHECK(rec.events.empty());   // pending is no change yet
    rc.feed_interventions(key(R"([{"point": "head", "op": "ablate", "from_step": null, "until_step": null,
                                   "since_step": 4, "state": "active"},
                                  {"point": "blocks.0.attn#1", "op": "ablate", "from_step": null, "until_step": null,
                                   "since_step": 15, "state": "active"}])"),
                          14, false, rec);
    REQUIRE(rec.events.size() == 1);
    CHECK(rec.events[0].kind == cb::EventKind::run);
    CHECK(rec.events[0].origin == "script");
    CHECK(rec.fields()[0] ==
          json{{"change", "intervene"}, {"point", "blocks.0.attn#1"}, {"op", "ablate"}, {"step", 15}});
    // A cleared point and an ended one stop: op null at the script's step.
    rc.feed_interventions(key(R"([{"point": "blocks.0.attn#1", "op": "ablate", "from_step": null,
                                   "until_step": 20, "since_step": 15, "state": "ended"}])"),
                          20, false, rec);
    REQUIRE(rec.events.size() == 3);
    CHECK(rec.fields()[1] == json{{"change", "intervene"}, {"point", "blocks.0.attn#1"}, {"op", nullptr}, {"step", 20}});
    CHECK(rec.fields()[2] == json{{"change", "intervene"}, {"point", "head"}, {"op", nullptr}, {"step", 20}});
}

TEST_CASE("app session: an intervention that the script ended while the link was down is a run event at its return") {
    caliper_host::AppRunControl rc;
    FakeLink link;
    Recorder rec;
    const auto set = [&](const char* state) {
        const std::string text = std::string(R"({"points": [], "active": [{"point": "head", "op": "ablate",
            "from_step": null, "until_step": null, "since_step": 4, "state": ")") + state + R"("}]})";
        std::string err;
        REQUIRE(caliper_net::Json::parse(text.data(), text.size(), &link.ctl.interventions, &err));
        ++link.ctl.version;
    };
    set("active");
    rc.set_link(&link);
    rc.feed(rec);   // the attach and the baseline: no intervene event
    const auto intervene = [&]() {
        std::vector<json> out;
        for (const json& f : rec.fields())
            if (f.value("change", "") == "intervene") out.push_back(f);
        return out;
    };
    CHECK(intervene().empty());
    link.is_attached = false;   // the viewer's link drops; the script ends it after the linger time
    rc.feed(rec);
    set("ended");
    link.ctl.step = 30;
    link.is_attached = true;
    rc.feed(rec);
    REQUIRE(intervene().size() == 1);
    CHECK(intervene()[0] == json{{"change", "intervene"}, {"point", "head"}, {"op", nullptr}, {"step", 30}});

    // Another script after a drop reports nothing of the first one's.
    set("active");
    rc.feed(rec);
    REQUIRE(intervene().size() == 2);   // applied again, at its step 4
    CHECK(intervene()[1]["op"] == "ablate");
    link.is_attached = false;
    rc.feed(rec);
    link.name = "other.py";
    set("ended");
    link.is_attached = true;
    rc.feed(rec);
    CHECK(intervene().size() == 2);
}

TEST_CASE("app session: the run control refuses interventions from a script that did not announce them") {
    caliper_host::AppRunControl rc;
    cb::Intervention i;
    i.point = "head";
    CHECK_FALSE(rc.interventions_supported());
    CHECK(rc.intervene_set(i).code == cb::kRunNotAttached);
    FakeLink link;   // its CONTROLS carry no key interventions
    rc.set_link(&link);
    CHECK_FALSE(rc.interventions_supported());
    CHECK(rc.intervention_points().empty());
    CHECK(rc.interventions().empty());
    CHECK(rc.intervene_set(i).code == cb::kInterveneNotSupported);
    CHECK(rc.intervene_clear("").code == cb::kInterveneNotSupported);
    CHECK(link.sent.empty());
}

TEST_CASE("app session: the run control reads the link's key interventions and sends intervene through the link") {
    caliper_host::AppRunControl rc;
    FakeLink link;
    const std::string text = R"({"points": [{"point": "blocks.0.attn", "kind": "module", "module": "blocks.0.attn",
        "head": null, "type": "Attention", "shape": null}], "active": []})";
    std::string err;
    REQUIRE(caliper_net::Json::parse(text.data(), text.size(), &link.ctl.interventions, &err));
    rc.set_link(&link);
    CHECK(rc.interventions_supported());
    const std::vector<cb::InterventionPoint> pts = rc.intervention_points();
    REQUIRE(pts.size() == 1);
    CHECK(pts[0].point == "blocks.0.attn");
    CHECK(rc.interventions().empty());
    cb::Intervention i;
    i.point = "blocks.0.attn";
    CHECK(rc.intervene_set(i).ok());
    CHECK(rc.intervene_clear("blocks.0.attn").ok());
    CHECK(rc.intervene_clear("").ok());
    REQUIRE(link.sent.size() == 3);
    CHECK(link.sent[0].op == "intervene");
    CHECK(json::parse(link.sent[0].value) == json{{"set", {{"point", "blocks.0.attn"}, {"op", "ablate"},
                                                           {"from_step", nullptr}, {"until_step", nullptr}}}});
    CHECK(json::parse(link.sent[1].value) == json{{"clear", "blocks.0.attn"}});
    CHECK(json::parse(link.sent[2].value) == json{{"clear_all", true}});
}

TEST_CASE("app session: the status bar and the settings say when the file still holds Allow all") {
    const fs::path root = caliper_test::unique_temp_path("caliper-app-session-allow-all");
    fs::create_directories(root / "app-data");
    cb::SessionOptions o;
    o.permissions_file = (root / "app-data" / "agent-permissions.json").string();
    cb::Session::Open opened = cb::Session::open((root / "ws").string(), o);
    REQUIRE(opened.error.ok());
    cb::Session& s = *opened.session;
    CHECK(caliper_host::allow_all_status(s).empty());
    CHECK(caliper_host::permission_save_note(s).empty());
    REQUIRE(s.set_agent_mode(cb::AgentPrincipal::chat, cb::AgentMode::allow_all).ok());
    CHECK(caliper_host::allow_all_status(s) == "allow all for side chat");
    REQUIRE(s.set_agent_mode(cb::AgentPrincipal::mcp, cb::AgentMode::allow_all).ok());
    CHECK(caliper_host::allow_all_status(s) == "allow all for side chat and MCP clients");
    // The person ends the chat's Allow all and the save fails: both places say
    // that the file still holds it.
    std::error_code ec;
    fs::remove(o.permissions_file + ".lock", ec);
    fs::create_directories(o.permissions_file + ".lock/keep");
    CHECK_FALSE(s.set_agent_mode(cb::AgentPrincipal::chat, cb::AgentMode::ask).ok());
    CHECK(caliper_host::allow_all_status(s) == "allow all for MCP clients  \xc2\xb7  allow all in the saved file for side chat");
    const std::string note = caliper_host::permission_save_note(s);
    CHECK(note.find("could not be saved") != std::string::npos);
    CHECK(note.find("The file still says Allow all for Side chat, which returns at the next launch.") != std::string::npos);
    CHECK(note.find("Choose the mode again to save it.") != std::string::npos);
    // Choosing the mode again once the save works clears both.
    fs::remove_all(o.permissions_file + ".lock", ec);
    REQUIRE(s.set_agent_mode(cb::AgentPrincipal::chat, cb::AgentMode::ask).ok());
    CHECK(caliper_host::allow_all_status(s) == "allow all for MCP clients");
    CHECK(caliper_host::permission_save_note(s).empty());
    s.close();
    opened.session.reset();
    fs::remove_all(root, ec);
}
