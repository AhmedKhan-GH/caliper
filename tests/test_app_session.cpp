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

#include "builder/app_session.h"
#include "builder/panel_renderer.h"
#include "job_system.h"
#include "test_temp.h"
#ifndef _WIN32
#include "test_proc.h"
#endif

#include <caliper_builder/commands/journal.h>
#include <caliper_builder/session/session.h>

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
struct FakeRenderer final : caliper_host::PanelRenderer {
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
    h.launch = [](const std::string& id) -> LaunchOutcome {
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
    REQUIRE(call(s, inv("panel.create", note, cb::CommandOrigin::mcp)).ok);
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
    CHECK(lines[2] == std::array<std::string, 3>{"mcp command", "panel.create", ""});
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
               "{\"name\": \"panel.create\", \"origin\": \"mcp\", \"args\": {\"id\": \"note\", \"primitive\": \"text\", "
               "\"options\": {\"text\": \"a note\"}}}\n"
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
    REQUIRE(results.size() == 5);
    CHECK(results[0]["line"] == 2);
    CHECK(results[0]["name"] == "spec.new");
    CHECK(results[0]["ok"] == true);
    CHECK(results[1]["line"] == 4);
    CHECK(results[1]["error"]["code"] == "command.invalid_invocation");
    CHECK(results[2]["name"] == "panel.create");
    CHECK(results[2]["ok"] == true);
    CHECK(results[3]["name"] == "capture.frame");
    CHECK(results[3]["result"]["png_base64"] == b64("PNG:4x2"));
    CHECK(results[4] == json{{"done", true}, {"commands", 4}, {"failed", 1}});
    const auto read = cb::read_journal(t.str("ws") + "/journal/commands.jsonl");
    REQUIRE(read.events.size() == 2);
    CHECK(read.events[1].origin == cb::CommandOrigin::mcp);
    CHECK(loop.err.str().find("[session] script done: 4 commands, 1 failed") != std::string::npos);
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
    CHECK(results[11]["error"]["code"] == "command.not_available");
    CHECK(results.back() == json{{"done", true}, {"commands", (int)script.size()}, {"failed", metal ? 1 : 2}});

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
