// Tests of the model runner (src/host/runner/model_runner.h,
// design/hosted-runs.md): the child processes, the stream links, and managed
// runs of the stand-in script tests/stand_in_run.py, which follows the
// process contract with Caliper LIB: a start, the run sources, the run
// control through the runner's connection, a fork from a checkpoint with a
// lower learning rate whose loss continues from its parent's, a stop of both,
// a start that writes no link, a stop of a fork that waits for its
// checkpoint, a script that ends with another code than 0 during a stop
// (failed) and one that the runner ends after the limit (stopped), a fork
// whose parent exits just after its checkpoint reply, the end of the child's
// whole process group, the size limit and the mode of output.log, and no
// child left at the end. Every wait waits on the runner's state, never on a
// fixed sleep, and every child is ended by its own process id, also when a
// check fails. Where a slow machine could change the order of two events, a
// gate holds the runner's passes (PassGate) and forces the order.
#include "runner/child_process.h"
#include "runner/model_runner.h"
#include "runner/run_connection.h"
#include "net/wire.h"
#include "net/ws.h"
#include "../../../../tests/probe_metadata.h"

#include <caliper_builder/session/events.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/schema.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

namespace cb = caliper_builder;
namespace cr = caliper_runner;
namespace fs = std::filesystem;
using nlohmann::json;

TEST_CASE("runner: native interpretation validation and protocol errors need no model") {
    namespace wire = caliper_net::wire;
    caliper_net::Server server;
    std::string why;
    REQUIRE(server.listen(0, &why));
    cr::RunConnection connection;
    REQUIRE(connection.open("ws://127.0.0.1:" + std::to_string(server.port()) +
        "/v3/stream?token=native_fixture_0123456789", &why));
    caliper_net::ConnId peer = 0;
    std::uint32_t pong = 0;
    bool closure_announced = false;
    std::vector<wire::Control> received;
    const auto send = [&](const wire::Message& message) {
        std::vector<std::uint8_t> bytes;
        REQUIRE(wire::encode(message, &bytes));
        REQUIRE(server.send(peer, bytes.data(), bytes.size()));
    };
    const auto pump = [&] {
        const auto state_before = connection.state();
        const bool first_change = connection.poll(0);
        for (auto& event : server.poll(5)) {
            if (event.type == caliper_net::Event::Type::Request) REQUIRE(server.accept(event.conn));
            if (event.type != caliper_net::Event::Type::Message) continue;
            peer = event.conn;
            wire::Message message;
            REQUIRE(wire::decode(event.data.data(), event.data.size(), &message).ok());
            if (std::holds_alternative<wire::Hello>(message)) {
                wire::Welcome welcome;
                welcome.role = CALIPER_NET_ROLE_PRODUCER;
                welcome.session = wire::Identity{"native-fixture", "fixture.py", 1, 1}.to_json();
                welcome.capabilities = {CALIPER_NET_CAP_INTERPRETATION, CALIPER_NET_CAP_CONTROLS};
                welcome.control = true;
                send(welcome);
            } else if (auto* control = std::get_if<wire::Control>(&message)) received.push_back(*control);
            else if (auto* reply = std::get_if<wire::Pong>(&message)) pong = reply->nonce;
        }
        const bool last_change = connection.poll(5);
        if (state_before != cr::RunConnection::State::closed && connection.state() == cr::RunConnection::State::closed)
            closure_announced = first_change || last_change;
    };
    const auto until = [&](const std::function<bool()>& ready) {
        for (unsigned i = 0; i < 200 && !ready(); ++i) pump();
        REQUIRE(ready());
    };
    until([&] { return !connection.session().empty(); });
    CHECK(connection.send_control(CALIPER_NET_OP_SET, "prompt", json(std::string("\xc3\x28", 2)), 0, &why) == 0);
    CHECK(why.find("UTF-8") != std::string::npos);
    const auto raw_reply = [&](const caliper_net::Json& meta) {
        std::vector<std::uint8_t> bytes;
        REQUIRE(wire::encode_frame(CALIPER_NET_MSG_CONTROL_REPLY, meta, {}, &bytes));
        REQUIRE(server.send(peer, bytes.data(), bytes.size()));
    };
    std::string prompt;
    for (unsigned i = 0; i < 2048; ++i) prompt += "\xc3\xa9";
    for (const auto& invalid : {std::string("a\0b", 3), prompt + "\xc3\xa9", std::string("\xc3\x28", 2)}) {
        const auto result = connection.interpret("trace.run", {{"prompt", invalid}}, 0);
        REQUIRE(result.contains("error"));
        CHECK(result["error"]["code"] == "interpret.refused");
        CHECK(result["error"]["message"].get<std::string>().size() <= CALIPER_NET_TEXT_MAX);
    }
    const auto queued = connection.interpret("trace.run", {{"prompt", prompt}}, 0);
    REQUIRE(queued.contains("request_id"));
    const auto id = queued["request_id"].get<std::uint32_t>();
    CHECK(connection.interpret("trace.run", json::object(), id + 1)["error"]["code"] == "interpret.unknown_request");
    CHECK(connection.interpret("trace.validate", json::object(), id)["error"]["code"] == "interpret.unknown_request");
    CHECK(connection.interpret("trace.run", {{"prompt", "new"}}, id)["error"]["code"] == "command.invalid_args");
    until([&] { return !received.empty(); });
    REQUIRE(received.size() == 1);
    CHECK(received.front().request_id == id);
    CHECK(received.front().interpretation.find("params")->find("prompt")->as_string() == prompt);
    std::size_t expected_received = 1;
    SUBCASE("text parameters require the producer capability") {
        expected_received = 3;
        wire::Controls controls;
        controls.run = caliper_net::Json::object();
        controls.run.set("state", "running");
        controls.run.set("step", 1);
        auto parameter = caliper_net::Json::object();
        parameter.set("name", "prompt");
        parameter.set("type", "text");
        parameter.set("value", "initial");
        controls.params.push(parameter);
        send(controls);
        until([&] { return !connection.controls().is_null(); });
        for (const auto& value : {std::string("short"), std::string(), prompt}) {
            CHECK(connection.send_control("set", "prompt", value, 0, &why) == 0);
            CHECK(why.find("text_params") != std::string::npos);
        }
        const auto choice = connection.send_control("set", "choice", "line\n\tbreak", 0, &why);
        REQUIRE(choice != 0);
        until([&] { return received.size() == 2; });
        wire::Welcome welcome;
        welcome.role = CALIPER_NET_ROLE_PRODUCER;
        welcome.session = wire::Identity{"native-fixture", "fixture.py", 1, 1}.to_json();
        welcome.capabilities = {CALIPER_NET_CAP_INTERPRETATION, CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_TEXT_PARAMS};
        welcome.control = true;
        send(welcome);
        send(wire::Ping{19});
        until([&] { return pong == 19; });
        REQUIRE(connection.send_control("set", "prompt", prompt, 0, &why) != 0);
        until([&] { return received.size() == 3; });
        CHECK(received.back().value.as_string() == prompt);
        wire::Error error;
        error.in_reply_to = CALIPER_NET_MSG_CONTROL;
        error.request_id = id;
        error.code = CALIPER_NET_ERR_BAD_REQUEST;
        error.message = "The fixture ended this request.";
        send(error);
    }
    SUBCASE("CONTROL ERROR") {
        wire::Error error;
        error.in_reply_to = CALIPER_NET_MSG_CONTROL;
        error.request_id = id;
        error.code = CALIPER_NET_ERR_BAD_REQUEST;
        error.message = "The fixture refused this request.";
        send(error);
    }
    SUBCASE("malformed CONTROL_REPLY") {
        auto meta = caliper_net::Json::object();
        meta.set("request_id", id); meta.set("op", "interpret"); meta.set("status", "ok");
        raw_reply(meta);
    }
    SUBCASE("malformed CONTROL_REPLY without a usable request ID") {
        auto meta = caliper_net::Json::object();
        meta.set("request_id", "invalid"); meta.set("op", "interpret"); meta.set("status", "ok");
        raw_reply(meta);
    }
    SUBCASE("unrelated errors preserve the pending request") {
        for (const auto* op : {"set", "run", "pause", "intervene"}) {
            for (const auto& request : {caliper_net::Json(id), caliper_net::Json("invalid"), caliper_net::Json()}) {
                auto meta = caliper_net::Json::object();
                if (!request.is_null()) meta.set("request_id", request);
                meta.set("op", op); meta.set("status", "invalid");
                raw_reply(meta);
            }
        }
        wire::Error error;
        error.code = CALIPER_NET_ERR_BAD_REQUEST;
        error.message = "The fixture refused an unrelated message.";
        send(error);
        error.in_reply_to = CALIPER_NET_MSG_CONTROL;
        send(error);
        error.request_id = id;
        error.sub = 1;
        send(error);
        send(wire::Ping{17});
        until([&] { return pong == 17; });
        CHECK(connection.interpret("trace.run", json::object(), id) == queued);
        error.sub.reset();
        send(error);
    }
    SUBCASE("BYE ends pending queries before the socket closes") {
        send(wire::Bye{CALIPER_NET_BYE_NORMAL, "The fixture ended."});
        until([&] { return connection.state() == cr::RunConnection::State::closed; });
        CHECK(closure_announced);
        CHECK(connection.interpret("trace.run", json::object(), id)["error"]["code"] == "run.not_attached");
        return;
    }
    until([&] { return connection.interpret("trace.run", json::object(), id).contains("error"); });
    const auto final = connection.interpret("trace.run", json::object(), id);
    CHECK(final["error"]["code"] == CALIPER_NET_ERR_BAD_REQUEST);
    wire::ControlReply late;
    late.request_id = id; late.op = CALIPER_NET_OP_INTERPRET; late.status = "queued";
    send(late);
    pump();
    CHECK(connection.interpret("trace.run", json::object(), id) == final);
    connection.close();
    CHECK(connection.interpret("trace.status", json::object(), 0)["error"]["code"] == "run.not_attached");
    REQUIRE(connection.open("ws://127.0.0.1:" + std::to_string(server.port()) +
        "/v3/stream?token=native_fixture_0123456789", &why));
    until([&] { return connection.interpret("trace.run", json::object(), id).value("error", json::object())
        .value("code", std::string()) == "interpret.unknown_request"; });
    CHECK(received.size() == expected_received);
}

namespace {

std::string fresh_dir(const std::string& name) {
    const char* root = std::getenv("CALIPER_RUNNER_TEST_TMP");
    const fs::path p = fs::path(root && *root ? root : CALIPER_RUNNER_TEST_TMP) / name;
    std::error_code ec;
    fs::remove_all(p, ec);
    fs::create_directories(p);
    return fs::canonical(p).u8string();
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// A workspace with the stand-in script, and the Python that runs it.
struct Workspace {
    std::string ws;
    std::string python = CALIPER_RUNNER_TEST_PYTHON;
    explicit Workspace(const std::string& name) {
        ws = fresh_dir(name);
        fs::create_directories(ws + "/scripts");
        fs::copy_file(CALIPER_RUNNER_STAND_IN, ws + "/scripts/stand_in_run.py");
        // The children import Caliper LIB from this checkout.
        setenv("PYTHONPATH", CALIPER_RUNNER_PYTHONPATH, 1);
    }
    cb::RunStartRequest start(std::vector<std::string> args, json params = json::object(), std::string id = {}) const {
        cb::RunStartRequest r;
        r.script_path = ws + "/scripts/stand_in_run.py";
        r.script = "scripts/stand_in_run.py";
        r.args = std::move(args);
        r.params = std::move(params);
        r.id = std::move(id);
        return r;
    }
};

// A sink that keeps the events.
struct Events final : cb::EventSink {
    std::mutex m;
    std::vector<cb::Event> events;
    std::uint64_t push(cb::Event e) override {
        std::lock_guard<std::mutex> lock(m);
        events.push_back(std::move(e));
        return events.size();
    }
    std::vector<std::string> states(const std::string& run) {
        std::lock_guard<std::mutex> lock(m);
        std::vector<std::string> out;
        for (const auto& e : events)
            if (e.fields.value("run", std::string()) == run) out.push_back(e.fields.value("state", std::string()));
        return out;
    }
};

const cb::RunEntry* find(const std::vector<cb::RunEntry>& runs, const std::string& id) {
    for (const auto& e : runs)
        if (e.id == id) return &e;
    return nullptr;
}

auto state_is(const std::string& id, const std::string& state) {
    return [id, state](const std::vector<cb::RunEntry>& runs) {
        const cb::RunEntry* e = find(runs, id);
        return e && e->state == state;
    };
}

// The values of a run source, or none.
std::vector<float> values_of(cr::ModelRunner& runner, const std::string& id) {
    cb::ServedData d = runner.data_source()->read(id, cb::ServedRequest());
    std::vector<float> out;
    if (d.error || !d.tensor.valid()) return out;
    for (std::uint64_t i = 0; i < d.tensor.numel(); ++i) out.push_back(static_cast<float>(d.tensor.element(i)));
    return out;
}

// Holds the runner's passes (ModelRunner::Options::before_pass): from the
// pass close_at on, the runner's thread waits before each pass until open()
// or step(). Declare it before the runner, so that it outlives the runner's
// thread.
struct PassGate {
    std::mutex m;
    std::condition_variable cv;
    std::uint64_t passes = 0;
    std::uint64_t close_at = 0;   // 0: open
    std::uint64_t holds = 0;      // how often the runner's thread began to wait
    void before_pass() {
        std::unique_lock<std::mutex> lock(m);
        ++passes;
        if (close_at == 0 || passes < close_at) return;
        ++holds;
        cv.notify_all();
        cv.wait(lock, [&] { return close_at == 0 || passes < close_at; });
    }
    // Holds the runner's thread before its next pass, and returns once it waits.
    bool hold(int timeout_ms) { return let_through(timeout_ms); }
    // Lets exactly one pass of the held runner run, and returns once the
    // runner's thread waits before the next one.
    bool step(int timeout_ms) { return let_through(timeout_ms); }
    bool let_through(int timeout_ms) {
        std::unique_lock<std::mutex> lock(m);
        const std::uint64_t before = holds;
        close_at = passes + 1;
        cv.notify_all();
        return cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return holds > before; });
    }
    void open() {
        std::lock_guard<std::mutex> lock(m);
        close_at = 0;
        cv.notify_all();
    }
};

}  // namespace

TEST_CASE("runner: a child process writes its output to the file and reports its exit code") {
    const std::string dir = fresh_dir("child");
    cr::ChildSpec spec;
    spec.program = "/bin/sh";
    spec.args = {"-c", "echo out; echo err 1>&2; echo \"$RUNNER_TEST_VALUE\"; pwd; exit 3"};
    spec.env = {{"RUNNER_TEST_VALUE", "from the runner"}, {"PATH", "/usr/bin:/bin"}};
    spec.cwd = dir;
    spec.output = dir + "/output.log";
    std::string err;
    auto child = cr::start_child(spec, &err);
    REQUIRE_MESSAGE(child, err);
    std::int64_t code = 0;
    REQUIRE(child->wait(10000, &code));
    CHECK(code == 3);
    const std::string out = read_file(spec.output);
    CHECK(out.find("out\n") != std::string::npos);
    CHECK(out.find("err\n") != std::string::npos);
    CHECK(out.find("from the runner\n") != std::string::npos);
    CHECK(out.find(dir) != std::string::npos);
    // Only the person who runs the app reads the output.
    CHECK((fs::status(spec.output).permissions() & fs::perms::all) == (fs::perms::owner_read | fs::perms::owner_write));
    // Once ended, the process id is never used again.
    child->kill();
    CHECK(child->poll(&code));

    // A child that runs on ends at the request to end, as a signal.
    spec.args = {"-c", "while :; do sleep 1; done"};
    auto sleeper = cr::start_child(spec, &err);
    REQUIRE_MESSAGE(sleeper, err);
    CHECK_FALSE(sleeper->poll(&code));
    sleeper->terminate();
    REQUIRE(sleeper->wait(10000, &code));
    CHECK(code == -15);

    // A request to end reaches the child's whole process group: a process
    // that the child started, as a venv launcher starts the interpreter,
    // ends with it.
    const std::string pid_file = dir + "/grandchild.pid";
    spec.args = {"-c", "sleep 120 & echo $! > grandchild.pid.tmp; mv grandchild.pid.tmp grandchild.pid; wait"};
    auto launcher = cr::start_child(spec, &err);
    REQUIRE_MESSAGE(launcher, err);
    for (int i = 0; i < 500 && !fs::exists(pid_file); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    REQUIRE(fs::exists(pid_file));
    const pid_t grandchild = static_cast<pid_t>(std::stol(read_file(pid_file)));
    REQUIRE(grandchild > 1);
    launcher->terminate();
    REQUIRE(launcher->wait(10000, &code));
    bool gone = false;
    for (int i = 0; i < 500 && !gone; ++i) {
        gone = ::kill(grandchild, 0) != 0 && errno == ESRCH;
        if (!gone) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!gone) ::kill(grandchild, SIGKILL);   // by its own id, so that no process is left
    CHECK(gone);

    spec.program = dir + "/missing";
    CHECK_FALSE(cr::start_child(spec, &err));
    CHECK(err.find("could not start") != std::string::npos);
}

TEST_CASE("runner: a page link or a stream link of this computer becomes the stream URL") {
    std::string err;
    const std::string token = "OHD2DeXPUL-50Xq5BD4cNKIvVnuNIRt-";
    CHECK(cr::stream_url_of("http://127.0.0.1:51988/?token=" + token, &err) ==
          "ws://127.0.0.1:51988/v3/stream?token=" + token);
    CHECK(cr::stream_url_of("ws://localhost:80/v3/stream?token=" + token, &err) ==
          "ws://localhost:80/v3/stream?token=" + token);
    CHECK(cr::stream_url_of("http://10.0.0.2:51988/?token=" + token, &err).empty());
    CHECK(cr::stream_url_of("http://127.0.0.1:51988/", &err).empty());
    CHECK(cr::stream_url_of("ftp://127.0.0.1:1/?token=" + token, &err).empty());
    CHECK(cr::stream_url_of("http://127.0.0.1:x/?token=" + token, &err).empty());
}

TEST_CASE("runner: a start that writes no link fails at the start limit, and its process ends") {
    Workspace w("runner_no_link");
    std::ofstream(w.ws + "/scripts/silent.py") << "import time\nprint('silent', flush=True)\ntime.sleep(120)\n";
    // The runner's passes wait until the script has printed its line, so that
    // a slow start of the interpreter cannot meet the short start limit first.
    PassGate gate;
    gate.close_at = 1;
    cr::ModelRunner::Options o;
    o.workspace = w.ws;
    o.python = w.python.empty() ? "python3" : w.python;
    o.start_limit_s = 1.0;
    o.kill_grace_s = 1.0;
    o.before_pass = [&] { gate.before_pass(); };
    cr::ModelRunner runner(o);
    struct Opener {
        PassGate& g;
        ~Opener() { g.open(); }   // before the runner's shutdown, also when a check fails
    } opener{gate};
    cb::RunStartRequest r;
    r.script_path = w.ws + "/scripts/silent.py";
    r.script = "scripts/silent.py";
    cb::RunEntry e;
    REQUIRE(!runner.start(r, &e));
    CHECK(e.id == "silent-1");
    CHECK(e.state == "starting");
    CHECK(fs::is_directory(w.ws + "/runs/silent-1"));
    bool printed = false;
    for (int i = 0; i < 3000 && !printed; ++i) {
        printed = read_file(w.ws + "/runs/silent-1/output.log").find("silent\n") != std::string::npos;
        if (!printed) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    REQUIRE(printed);
    gate.open();
    REQUIRE(runner.wait_for(state_is("silent-1", "failed"), 20000));
    const cb::RunEntry got = runner.list().at(0);
    REQUIRE(got.error);
    CHECK(got.error->find("no stream link") != std::string::npos);
    REQUIRE(got.exit_code);
    CHECK(*got.exit_code < 0);
    CHECK(runner.running_pids().empty());
    // The id of an existing run folder is refused; another one is chosen.
    r.id = "silent-1";
    CHECK(runner.start(r, &e).code == cb::kRunExists);
    r.id = "";
    REQUIRE(!runner.start(r, &e));
    CHECK(e.id == "silent-2");
    runner.shutdown();
    CHECK(runner.running_pids().empty());
    std::vector<std::string> lines;
    REQUIRE(!runner.logs("silent-1", 10, &lines));
    REQUIRE(!lines.empty());
    CHECK(lines.front() == "silent");
}

TEST_CASE("runner: a managed run starts, forks from a checkpoint with a lower lr, and both stop") {
    Workspace w("runner_fork");
    if (w.python.empty()) {
        MESSAGE("skipped: the build names no Python that imports Caliper LIB");
        return;
    }
    cr::ModelRunner::Options o;
    o.workspace = w.ws;
    o.python = w.python;
    o.start_limit_s = 60.0;
    cr::ModelRunner runner(o);
    Events events;
    runner.set_event_sink(&events);
    struct Guard {   // every child ends, also when a check fails
        cr::ModelRunner& r;
        ~Guard() { r.shutdown(); }
    } guard{runner};

    cb::RunEntry base;
    REQUIRE(!runner.start(w.start({"100000", "0.01"}, json{{"lr", 0.05}}, "base"), &base));
    CHECK(base.state == "starting");
    REQUIRE(runner.wait_for(state_is("base", "running"), 60000));
    // The app attaches to the new run at its next frame.
    std::string link = runner.frame(false, "");
    CHECK(link.rfind("http://127.0.0.1:", 0) == 0);
    CHECK(runner.frame(true, link).empty());
    CHECK(runner.list().at(0).attached);

    // Its loss is a run source, and it grows.
    REQUIRE(runner.wait_for([&](const auto&) { return values_of(runner, "run:base/loss").size() >= 20; }, 60000));
    cb::CatalogFilter filter;
    filter.kind = cb::CatalogKind::run;
    const cb::CatalogListResult listed = runner.catalog_provider()->list(filter);
    REQUIRE(listed.entries.size() == 1);
    CHECK(listed.entries[0].id == "run:base/loss");
    CHECK(runner.data_source()->generation("run:base/loss") > 0);

    // The run commands act on it through the runner's connection, and the
    // app switches to it.
    runner.frame(true, "http://127.0.0.1:1/?token=another-scripts-token-xx");
    cb::Error err;
    cb::RunControl* control = runner.control_of("base", &err);
    REQUIRE(control);
    const cb::RunState st = control->state();
    CHECK(st.attached);
    CHECK(st.control);
    CHECK(st.script == "scripts/stand_in_run.py");
    bool has_checkpoint = false;
    for (const auto& a : st.actions) has_checkpoint = has_checkpoint || a.name == "caliper.checkpoint";
    CHECK(has_checkpoint);
    CHECK(runner.frame(true, "http://127.0.0.1:1/?token=another-scripts-token-xx") == link);
    CHECK(runner.control_of("nope", &err) == nullptr);
    CHECK(err.code == cb::kRunNotFound);

    // A fork with a lower learning rate starts from the parent's checkpoint.
    cb::RunForkRequest fr;
    fr.parent = "base";
    fr.params = json{{"lr", 0.01}};
    cb::RunEntry fork;
    REQUIRE(!runner.fork(fr, &fork));
    CHECK(fork.id == "stand-in-run-1");
    CHECK(fork.parent == std::optional<std::string>("base"));
    REQUIRE(runner.wait_for(state_is(fork.id, "running"), 60000));
    REQUIRE(runner.wait_for([&](const auto&) {
        std::size_t real = 0;
        for (float v : values_of(runner, "run:" + fork.id + "/loss")) real += std::isnan(v) ? 0 : 1;
        return real >= 5;
    }, 60000));
    // The fork resumed at the checkpoint's step: its first loss continues the
    // parent's at the lower rate, below the parent's first loss.
    const std::vector<std::string> checkpoints = [&] {
        std::vector<std::string> out;
        for (const auto& f : fs::directory_iterator(w.ws + "/runs/base"))
            if (f.path().filename().u8string().rfind("checkpoint-", 0) == 0) out.push_back(f.path().u8string());
        return out;
    }();
    REQUIRE(checkpoints.size() == 1);
    const json ck = json::parse(read_file(checkpoints[0]));
    const std::vector<float> forked = values_of(runner, "run:" + fork.id + "/loss");
    const std::vector<float> parent = values_of(runner, "run:base/loss");
    // The fork's source starts at the checkpoint's step, after NaN values,
    // so that its curve continues its parent's on one step axis.
    const std::size_t at = static_cast<std::size_t>(ck["step"].get<std::uint64_t>());
    REQUIRE(forked.size() > at);
    for (std::size_t i = 0; i < at; ++i) CHECK(std::isnan(forked[i]));
    CHECK(forked[at] == doctest::Approx(ck["loss"].get<double>() * (1.0 - 0.01)).epsilon(1e-4));
    CHECK(forked[at] < parent.front());
    std::vector<std::string> lines;
    REQUIRE(!runner.logs(fork.id, 200, &lines));
    bool resumed = false;
    for (const auto& l : lines) resumed = resumed || l.find("resumed from checkpoint-") == 0;
    CHECK(resumed);

    // Both stop at their next checkpoint and exit with code 0.
    cb::RunEntry stopped;
    REQUIRE(!runner.stop("base", 30, &stopped));
    CHECK(stopped.stopping);
    REQUIRE(!runner.stop(fork.id, 30, &stopped));
    REQUIRE(runner.wait_for(state_is("base", "exited"), 60000));
    REQUIRE(runner.wait_for(state_is(fork.id, "exited"), 60000));
    for (const auto& e : runner.list()) {
        REQUIRE(e.exit_code);
        CHECK(*e.exit_code == 0);
    }
    CHECK(runner.running_pids().empty());
    REQUIRE(!runner.logs("base", 3, &lines));
    REQUIRE(lines.size() >= 1);
    CHECK(lines.back().find("stopped at step") == 0);
    // The series stay after the runs end.
    CHECK(values_of(runner, "run:base/loss").size() >= 20);
    // The runner events gave each state in order.
    const std::vector<std::string> seen = events.states("base");
    REQUIRE(seen.size() >= 3);
    CHECK(seen.front() == "starting");
    CHECK(seen.back() == "exited");
    // A finished run has no control.
    CHECK(runner.control_of("base", &err) == nullptr);
    CHECK(err.code == cb::kRunNotReady);
    runner.set_event_sink(nullptr);
}

TEST_CASE("runner: run.list adds the person's own attached script as user") {
    const std::string ws = fresh_dir("runner_user");
    cr::ModelRunner::Options o;
    o.workspace = ws;
    cr::ModelRunner runner(o);
    struct App final : cb::RunControl {
        cb::RunState state() override {
            cb::RunState s;
            s.attached = true;
            s.script = "mine.py";
            s.state = "paused";
            s.step = 9;
            return s;
        }
        cb::Error set_param(const std::string&, const json&) override { return {}; }
        cb::Error run_action(const std::string&) override { return {}; }
        cb::Error pause(cb::RunPauseOp, std::int64_t) override { return {}; }
    } app;
    runner.set_app_control(&app);
    CHECK(runner.list().empty());
    runner.frame(true, "http://127.0.0.1:2/?token=the-persons-own-script-token");
    const auto runs = runner.list();
    REQUIRE(runs.size() == 1);
    CHECK(runs[0].id == "user");
    CHECK_FALSE(runs[0].managed);
    CHECK(runs[0].state == "paused");
    CHECK(runs[0].script == "mine.py");
    CHECK(runs[0].step == std::optional<std::uint64_t>(9));
    cb::Error err;
    CHECK(runner.control_of("user", &err) == &app);
    runner.frame(false, "");
    CHECK(runner.control_of("user", &err) == nullptr);
    runner.set_app_control(nullptr);
}

namespace {

// A runner of the stand-in that every check ends, also when one fails.
struct StandIn {
    Workspace w;
    cr::ModelRunner runner;
    explicit StandIn(const std::string& name, double start_limit_s = 60.0, std::function<void()> before_pass = {})
        : w(name), runner([&] {
              cr::ModelRunner::Options o;
              o.workspace = w.ws;
              o.python = w.python;
              o.start_limit_s = start_limit_s;
              o.before_pass = std::move(before_pass);
              return o;
          }()) {}
    ~StandIn() { runner.shutdown(); }
};

}  // namespace

TEST_CASE("runner: a stop ends a fork that waits for its checkpoint, and the checkpoint starts nothing") {
    if (std::string(CALIPER_RUNNER_TEST_PYTHON).empty()) {
        MESSAGE("skipped: the build names no Python that imports Caliper LIB");
        return;
    }
    StandIn t("runner_stop_waiting_fork");
    cr::ModelRunner& runner = t.runner;
    cb::RunEntry base;
    // One step a second, so that the fork waits for the checkpoint.
    REQUIRE(!runner.start(t.w.start({"100000", "1.0"}, json::object(), "base"), &base));
    REQUIRE(runner.wait_for(state_is("base", "running"), 60000));
    cb::RunForkRequest fr;
    fr.parent = "base";
    cb::RunEntry fork;
    REQUIRE(!runner.fork(fr, &fork));
    CHECK(fork.state == "starting");
    cb::RunEntry stopped;
    REQUIRE(!runner.stop(fork.id, 10, &stopped));
    CHECK(stopped.state == "exited");
    CHECK(stopped.stopping);
    // The parent saves the checkpoint at its next step and goes on; the fork
    // stays ended and starts no process.
    const std::string checkpoint_dir = t.w.ws + "/runs/base";
    REQUIRE(runner.wait_for([&](const auto&) {
        for (const auto& f : fs::directory_iterator(checkpoint_dir))
            if (f.path().filename().u8string().rfind("checkpoint-", 0) == 0) return values_of(runner, "run:base/loss").size() >= 4;
        return false;
    }, 60000));
    CHECK(runner.running_pids().size() == 1);
    REQUIRE(!runner.stop("base", 30, &stopped));
    REQUIRE(runner.wait_for(state_is("base", "exited"), 60000));
    // The parent's end read every reply that it sent, so the checkpoint's
    // reply has been handled here, also on a slow machine: it started nothing.
    const std::vector<cb::RunEntry> runs = runner.list();
    const cb::RunEntry* after = find(runs, fork.id);
    REQUIRE(after);
    CHECK(after->state == "exited");
    CHECK_FALSE(fs::exists(t.w.ws + "/runs/" + fork.id + "/output.log"));
    CHECK(runner.running_pids().empty());
}

TEST_CASE("runner: a script that ends with another code than 0 during a stop failed, and one that the runner ends stopped") {
    if (std::string(CALIPER_RUNNER_TEST_PYTHON).empty()) {
        MESSAGE("skipped: the build names no Python that imports Caliper LIB");
        return;
    }
    StandIn t("runner_stop_outcomes");
    // A script that serves without a checkpoint, and ends with 3 when the file quit is in its folder.
    std::ofstream(t.w.ws + "/scripts/quits.py")
        << "import os, sys, time\nimport caliper\nserver = caliper.serve()\n"
           "run_dir = os.environ['CALIPER_RUN_DIR']\n"
           "with open(os.path.join(run_dir, 'link.tmp'), 'w') as f:\n    f.write(server.url)\n"
           "os.replace(os.path.join(run_dir, 'link.tmp'), os.environ['CALIPER_RUN_LINK_FILE'])\n"
           "while not os.path.exists(os.path.join(run_dir, 'quit')):\n    time.sleep(0.02)\n"
           "sys.exit(3)\n";
    cr::ModelRunner& runner = t.runner;
    for (const std::string id : {"quits", "ignores"}) {
        cb::RunStartRequest r;
        r.script_path = t.w.ws + "/scripts/quits.py";
        r.script = "scripts/quits.py";
        r.id = id;
        cb::RunEntry e;
        REQUIRE(!runner.start(r, &e));
    }
    REQUIRE(runner.wait_for(state_is("quits", "running"), 60000));
    REQUIRE(runner.wait_for(state_is("ignores", "running"), 60000));
    // The stop waits for a checkpoint that never comes, and the script ends on its own with 3.
    cb::RunEntry stopped;
    REQUIRE(!runner.stop("quits", 60, &stopped));
    CHECK(stopped.stopping);
    std::ofstream(t.w.ws + "/runs/quits/quit") << "now";
    REQUIRE(runner.wait_for(state_is("quits", "failed"), 60000));
    // The runner ends a script that reaches no checkpoint after the limit: a stop, not a failure.
    REQUIRE(!runner.stop("ignores", 1, &stopped));
    REQUIRE(runner.wait_for(state_is("ignores", "exited"), 60000));
    const std::vector<cb::RunEntry> runs = runner.list();
    const cb::RunEntry* quits = find(runs, "quits");
    const cb::RunEntry* ignores = find(runs, "ignores");
    REQUIRE(quits);
    REQUIRE(ignores);
    REQUIRE(quits->exit_code);
    CHECK(*quits->exit_code == 3);
    REQUIRE(quits->error);
    CHECK(quits->error->find("exit code 3") != std::string::npos);
    REQUIRE(ignores->exit_code);
    CHECK(*ignores->exit_code < 0);
    CHECK_FALSE(ignores->error);
    CHECK(runner.running_pids().empty());
}

TEST_CASE("runner: a fork starts when its parent exits in the pass of its checkpoint reply") {
    if (std::string(CALIPER_RUNNER_TEST_PYTHON).empty()) {
        MESSAGE("skipped: the build names no Python that imports Caliper LIB");
        return;
    }
    PassGate gate;
    StandIn t("runner_fork_parent_exits", 60.0, [&] { gate.before_pass(); });
    struct Opener {
        PassGate& g;
        ~Opener() { g.open(); }   // before the runner's shutdown, also when a check fails
    } opener{gate};
    cr::ModelRunner& runner = t.runner;
    cb::RunEntry base;
    REQUIRE(!runner.start(t.w.start({"100000", "0.02", "exit-after-checkpoint"}, json::object(), "base"), &base));
    REQUIRE(runner.wait_for(state_is("base", "running"), 60000));
    cb::RunForkRequest fr;
    fr.parent = "base";
    fr.params = json{{"lr", 0.01}};
    cb::RunEntry fork;
    // The order that a slow machine gives: one pass writes the checkpoint
    // request, then the runner reads nothing until the parent has replied,
    // closed its stream and ended. The runner is held before the fork, so
    // that no pass can run between the request and the hold.
    REQUIRE(gate.hold(60000));
    REQUIRE(!runner.fork(fr, &fork));
    REQUIRE(gate.step(60000));
    bool ended = false;
    for (int i = 0; i < 3000 && !ended; ++i) {
        ended = runner.running_pids().empty();
        if (!ended) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    REQUIRE(ended);
    {
        const std::vector<cb::RunEntry> held = runner.list();
        const cb::RunEntry* f = find(held, fork.id);
        REQUIRE(f);
        CHECK(f->state == "starting");   // the runner has not read the reply yet
    }
    // Then the runner's first poll writes before it reads, into a socket
    // that the ended parent has reset: a frame waits in the runner's queue
    // (a pause for the parent), and the reset is there already. A ping
    // written beside the runner's connection reaches the parent's closed
    // socket, whose system answers with the reset; a later write fails once
    // the runner's socket took it. The reply is still unread, so the runner
    // starts the fork only if it reads what is left after its write failed
    // (src/host/net/ws.cpp).
    {
        const std::string link = read_file(t.w.ws + "/runs/base/link.txt");
        const std::size_t colon = link.find("127.0.0.1:");
        REQUIRE(colon != std::string::npos);
        const int port = std::atoi(link.c_str() + colon + 10);
        REQUIRE(port > 0);
        int sock = -1;
        for (int fd = 0; fd < 4096 && sock < 0; ++fd) {
            sockaddr_in theirs{};
            socklen_t tl = sizeof theirs;
            if (::getpeername(fd, (sockaddr*)&theirs, &tl) == 0 && theirs.sin_family == AF_INET &&
                ntohs(theirs.sin_port) == port)
                sock = fd;
        }
        REQUIRE(sock >= 0);
#ifdef MSG_NOSIGNAL
        const int flags = MSG_NOSIGNAL;
#else
        const int flags = 0;   // Apple: the runner's socket has SO_NOSIGPIPE
#endif
        const unsigned char ping[] = {0x89, 0x80, 0, 0, 0, 0};   // a masked ping with no payload
        bool reset = false;
        for (int i = 0; i < 5000 && !reset; ++i) {
            reset = ::send(sock, ping, sizeof ping, flags) < 0 && (errno == EPIPE || errno == ECONNRESET);
            if (!reset) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        REQUIRE(reset);
        cb::Error ce;
        cb::RunControl* control = runner.control_of("base", &ce);
        REQUIRE(control);
        REQUIRE(!control->pause(cb::RunPauseOp::pause, 1));
    }
    gate.open();
    REQUIRE(runner.wait_for([&](const std::vector<cb::RunEntry>& runs) {
        const cb::RunEntry* f = find(runs, fork.id);
        return f && f->state != "starting";
    }, 60000));
    REQUIRE(runner.wait_for(state_is("base", "exited"), 60000));
    const std::vector<cb::RunEntry> runs = runner.list();
    const cb::RunEntry* f = find(runs, fork.id);
    REQUIRE(f);
    INFO("the fork: " << f->state << " " << f->error.value_or(""));
    CHECK(f->state == "running");
    std::vector<std::string> lines;
    REQUIRE(!runner.logs("base", 200, &lines));
    REQUIRE(!lines.empty());
    CHECK(lines.back().find("exits after the checkpoint") == 0);
    // The process contract gives the child the app's process id.
    bool app_pid = false;
    for (const auto& l : lines) app_pid = app_pid || l.find("for the app " + std::to_string(::getpid())) != std::string::npos;
    CHECK(app_pid);
    cb::RunEntry stopped;
    REQUIRE(!runner.stop(fork.id, 30, &stopped));
    REQUIRE(runner.wait_for(state_is(fork.id, "exited"), 60000));
    CHECK(runner.running_pids().empty());
}

TEST_CASE("runner: output.log moves to output.log.1 at its size limit, and run.logs reads both") {
    Workspace w("runner_output_limit");
    std::ofstream(w.ws + "/scripts/noisy.py")
        << "import time\nfor i in range(3000):\n    print('line %05d ' % i + 'x' * 90)\nprint('done', flush=True)\n"
           "time.sleep(120)\n";
    cr::ModelRunner::Options o;
    o.workspace = w.ws;
    o.python = w.python.empty() ? "python3" : w.python;
    o.start_limit_s = 60.0;
    o.kill_grace_s = 1.0;
    o.output_limit_bytes = 64 * 1024;
    cr::ModelRunner runner(o);
    struct Guard {
        cr::ModelRunner& r;
        ~Guard() { r.shutdown(); }
    } guard{runner};
    cb::RunStartRequest r;
    r.script_path = w.ws + "/scripts/noisy.py";
    r.script = "scripts/noisy.py";
    cb::RunEntry e;
    REQUIRE(!runner.start(r, &e));
    const std::string out = w.ws + "/runs/" + e.id + "/output.log";
    // The script printed about 290 KiB; the runner keeps the file near the
    // limit at each pass.
    bool done = false;
    for (int i = 0; i < 1000 && !done; ++i) {
        std::vector<std::string> lines;
        std::error_code ec;
        done = !runner.logs(e.id, 3, &lines) && std::find(lines.begin(), lines.end(), "done") != lines.end() &&
               fs::file_size(out, ec) <= o.output_limit_bytes && !ec;
        if (!done) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    REQUIRE(done);
    REQUIRE(fs::exists(out + ".1"));
    CHECK(read_file(out + ".1").find("line ") != std::string::npos);
    const auto owner_only = fs::perms::owner_read | fs::perms::owner_write;
    CHECK((fs::status(out).permissions() & fs::perms::all) == owner_only);
    CHECK((fs::status(out + ".1").permissions() & fs::perms::all) == owner_only);
    CHECK(read_file(out).find("caliper: output.log reached 65536 bytes") != std::string::npos);
    // run.logs reads the older lines from output.log.1 when output.log has
    // fewer.
    std::vector<std::string> lines;
    REQUIRE(!runner.logs(e.id, 200, &lines));
    CHECK(lines.size() == 200);
    CHECK(std::find(lines.begin(), lines.end(), "done") != lines.end());
    CHECK(std::find(lines.begin(), lines.end(), "line 02999 " + std::string(90, 'x')) != lines.end());
    cb::RunEntry stopped;
    REQUIRE(!runner.stop(e.id, 0, &stopped));
    REQUIRE(runner.wait_for([&](const std::vector<cb::RunEntry>& runs) {
        const cb::RunEntry* x = find(runs, e.id);
        return x && (x->state == "exited" || x->state == "failed");
    }, 20000));
    CHECK(runner.running_pids().empty());
}

namespace {

// Starts the stand-in `base` at a pace of 0.02 seconds and waits until its step is at least `step`.
void start_parent_at(StandIn& t, std::uint64_t step) {
    cb::RunEntry base;
    REQUIRE(!t.runner.start(t.w.start({"100000", "0.02"}, json::object(), "base"), &base));
    REQUIRE(t.runner.wait_for(state_is("base", "running"), 60000));
    REQUIRE(t.runner.wait_for([&](const std::vector<cb::RunEntry>& runs) {
        const cb::RunEntry* e = find(runs, "base");
        return e && e->step && *e->step >= step;
    }, 60000));
}

json one_ablate() {
    return json::array({json{{"point", "blocks.1.attn#2"}, {"op", "ablate"}}});
}

bool has_line(const std::vector<std::string>& lines, const std::string& part) {
    return std::any_of(lines.begin(), lines.end(), [&](const std::string& l) { return l.find(part) != std::string::npos; });
}

}  // namespace

TEST_CASE("runner: a fork with a step starts at once from the newest kept checkpoint at or before it") {
    if (std::string(CALIPER_RUNNER_TEST_PYTHON).empty()) {
        MESSAGE("skipped: the build names no Python that imports Caliper LIB");
        return;
    }
    StandIn t("runner_fork_at_step");
    cr::ModelRunner& runner = t.runner;
    start_parent_at(t, 30);
    cb::RunForkRequest fr;
    fr.parent = "base";
    fr.step = 17;
    fr.interventions = one_ablate();
    cb::RunEntry fork;
    REQUIRE(!runner.fork(fr, &fork));
    CHECK(fork.parent == std::optional<std::string>("base"));
    REQUIRE(runner.wait_for(state_is(fork.id, "running"), 60000));
    CHECK(fs::is_regular_file(t.w.ws + "/runs/" + fork.id + "/resume-15.json"));
    // The loss source starts after 15 NaN values, as the step of the kept checkpoint.
    REQUIRE(runner.wait_for([&](const auto&) {
        std::size_t real = 0;
        for (float v : values_of(runner, "run:" + fork.id + "/loss")) real += std::isnan(v) ? 0 : 1;
        return real >= 3;
    }, 60000));
    const std::vector<float> forked = values_of(runner, "run:" + fork.id + "/loss");
    REQUIRE(forked.size() > 15);
    for (std::size_t i = 0; i < 15; ++i) CHECK(std::isnan(forked[i]));
    CHECK_FALSE(std::isnan(forked[15]));
    std::vector<std::string> lines;
    REQUIRE(!runner.logs(fork.id, 200, &lines));
    CHECK(has_line(lines, "resumed from resume-15.json at step 15"));
    CHECK(has_line(lines, "interventions [{\"op\":\"ablate\",\"point\":\"blocks.1.attn#2\"}]"));
    cb::RunEntry stopped;
    REQUIRE(!runner.stop(fork.id, 30, &stopped));
    REQUIRE(!runner.stop("base", 30, &stopped));
    REQUIRE(runner.wait_for(state_is(fork.id, "exited"), 60000));
    REQUIRE(runner.wait_for(state_is("base", "exited"), 60000));
    CHECK(runner.running_pids().empty());
}

TEST_CASE("runner: a fork before the oldest kept checkpoint or with an until_step that it reaches is refused") {
    if (std::string(CALIPER_RUNNER_TEST_PYTHON).empty()) {
        MESSAGE("skipped: the build names no Python that imports Caliper LIB");
        return;
    }
    StandIn t("runner_fork_refused");
    cr::ModelRunner& runner = t.runner;
    start_parent_at(t, 30);
    cb::RunEntry out;
    cb::RunForkRequest fr;
    fr.parent = "base";
    fr.step = 0;
    cb::Error e = runner.fork(fr, &out);
    CHECK(e.code == cb::kRunNotReady);
    CHECK(e.message.find("oldest kept checkpoint is at step 1") != std::string::npos);
    CHECK_FALSE(fs::exists(t.w.ws + "/runs/stand-in-run-1"));
    std::size_t folders = 0;
    for (const auto& f : fs::directory_iterator(t.w.ws + "/runs")) folders += f.is_directory() ? 1 : 0;
    CHECK(folders == 1);   // only the parent's

    fr.step = 17;
    fr.interventions = json::array({json{{"point", "blocks.1.attn#2"}, {"op", "ablate"}, {"until_step", std::uint64_t(15)}}});
    e = runner.fork(fr, &out);
    CHECK(e.code == cb::kInterveneBadRange);
    folders = 0;
    for (const auto& f : fs::directory_iterator(t.w.ws + "/runs")) folders += f.is_directory() ? 1 : 0;
    CHECK(folders == 1);
    cb::RunEntry stopped;
    REQUIRE(!runner.stop("base", 30, &stopped));
    REQUIRE(runner.wait_for(state_is("base", "exited"), 60000));
    CHECK(runner.running_pids().empty());
}

TEST_CASE("runner: a symbolic link among the kept checkpoints is never used for a fork") {
    if (std::string(CALIPER_RUNNER_TEST_PYTHON).empty()) {
        MESSAGE("skipped: the build names no Python that imports Caliper LIB");
        return;
    }
    StandIn t("runner_fork_symlink");
    cr::ModelRunner& runner = t.runner;
    start_parent_at(t, 30);
    const std::string real = t.w.ws + "/runs/base/other-checkpoint.json";
    {
        std::ofstream f(real);
        f << R"({"step": 16, "loss": 1.0, "params": {"lr": 0.05}})";
    }
    std::error_code ec;
    fs::create_symlink(real, t.w.ws + "/runs/base/kept/checkpoint-16.json", ec);
    REQUIRE_MESSAGE(!ec, ec.message());
    cb::RunForkRequest fr;
    fr.parent = "base";
    fr.step = 17;
    cb::RunEntry fork;
    REQUIRE(!runner.fork(fr, &fork));
    REQUIRE(runner.wait_for(state_is(fork.id, "running"), 60000));
    CHECK(fs::is_regular_file(t.w.ws + "/runs/" + fork.id + "/resume-15.json"));
    CHECK_FALSE(fs::exists(t.w.ws + "/runs/" + fork.id + "/resume-16.json"));
    std::vector<std::string> lines;
    REQUIRE(!runner.logs(fork.id, 200, &lines));
    CHECK(has_line(lines, "resumed from resume-15.json at step 15"));
    cb::RunEntry stopped;
    REQUIRE(!runner.stop(fork.id, 30, &stopped));
    REQUIRE(!runner.stop("base", 30, &stopped));
    REQUIRE(runner.wait_for(state_is(fork.id, "exited"), 60000));
    REQUIRE(runner.wait_for(state_is("base", "exited"), 60000));
    CHECK(runner.running_pids().empty());
}

TEST_CASE("runner: semantic probe metadata survives the managed connection") {
    const std::string pythonpath = std::getenv("PYTHONPATH") ? std::getenv("PYTHONPATH") : "";
    Workspace workspace("probe-metadata");
    // Retain an external CPU guard before the checkout path in child processes.
    if (!pythonpath.empty()) setenv("PYTHONPATH", pythonpath.c_str(), 1);
    REQUIRE(!workspace.python.empty());
    const json expected = caliper_test::probe_descriptors();
    std::ofstream script(workspace.ws + "/scripts/stand_in_run.py");
    script << "import json, os, threading\nfrom pathlib import Path\nimport torch\nimport caliper\n"
              "from caliper._controls import CONTROLS\n"
              "from caliper import _net_programs\n"
              "class Descriptors:\n"
              "    def points(self, unused):\n"
              "        return json.loads(" << json(expected.dump()).dump() << ")\n"
              "fixture = torch.nn.Module().to('cpu')\n"
              "CONTROLS.interventions_on(fixture, adapter=Descriptors())\n"
              // Announce the watch capability without a program manager or device.
              "_net_programs.enable()\n"
              "done = threading.Event()\n"
              "caliper.action('caliper.stop', done.set)\n"
              "server = caliper.serve(control=True)\n"
              "Path(os.environ['CALIPER_RUN_LINK_FILE']).write_text(server.url)\n"
              "while not done.wait(0.01):\n"
              "    caliper.step()\n"
              "caliper.close()\n"
              "_net_programs.disable()\n";
    script.close();
    cr::ModelRunner::Options options;
    options.workspace = workspace.ws;
    options.python = workspace.python;
    cr::ModelRunner runner(options);
    struct Guard { cr::ModelRunner& runner; ~Guard() { runner.shutdown(); } } guard{runner};
    cb::RunEntry entry;
    REQUIRE(!runner.start(workspace.start({}, json::object(), "probe-metadata"), &entry));
    REQUIRE(runner.wait_for(state_is(entry.id, "running"), 30000));
    cb::Error error;
    auto* control = runner.control_of(entry.id, &error);
    REQUIRE(control);
    REQUIRE(runner.wait_for([&](const auto&) { return control->intervention_points().size() == expected.size(); }, 30000));
    CHECK(runner.intervention_points(entry.id).size() == expected.size());
    auto opened = cb::Session::open(workspace.ws);
    REQUIRE(opened.session);
    opened.session->set_run_control(control);
    cb::CommandInvocation command;
    command.name = "intervene.points";
    command.args = json::object();
    command.origin = cb::CommandOrigin::mcp;
    const auto reply = opened.session->invoke(command);
    REQUIRE_MESSAGE(reply.ok, reply.error.message);
    CHECK(reply.result.at("points") == expected);
    const auto* descriptor = cb::find_command_descriptor(command.name);
    REQUIRE(descriptor);
    CHECK(cb::validate_command_value(descriptor->result, reply.result).ok());
}

TEST_CASE("runner: interpretation uses the managed connection without attachment") {
    Workspace w("interpretation");
    if (const char* python = std::getenv("CALIPER_INTERPRET_PYTHON")) w.python = python;
    REQUIRE(!w.python.empty());
    std::ofstream script(w.ws + "/scripts/stand_in_run.py");
    script << R"PY(import os
from pathlib import Path
import caliper
from caliper._controls import CONTROLS
from caliper.interp._service import register_service
service = register_service()
server = caliper.serve(control=True)
Path(os.environ["CALIPER_RUN_LINK_FILE"]).write_text(server.url)
while True:
    with CONTROLS.cond:
        CONTROLS.cond.wait_for(lambda: service.pending, timeout=1)
    if service.pending:
        caliper.step()
)PY";
    script.close();
    cr::ModelRunner::Options options;
    options.workspace = w.ws;
    options.python = w.python;
    cr::ModelRunner runner(options);
    struct Guard { cr::ModelRunner& runner; ~Guard() { runner.shutdown(); } } guard{runner};
    cb::RunEntry entry;
    REQUIRE(!runner.start(w.start({}, json::object(), "interpretation"), &entry));
    REQUIRE(runner.wait_for(state_is(entry.id, "running"), 30000));
    cb::Error error;
    auto* control = runner.control_of(entry.id, &error);
    REQUIRE(control);
    REQUIRE(runner.wait_for([&](const auto&) { return control->state().control; }, 30000));
    auto output = control->interpret("trace.status", json::object(), 0);
    REQUIRE(!output.error);
    const auto request = output.result.at("request_id").get<std::uint32_t>();
    REQUIRE(runner.wait_for([&](const auto&) {
        output = control->interpret("trace.status", json::object(), request);
        return output.error || !output.result.at("pending").get<bool>();
    }, 30000));
    REQUIRE(!output.error);
    CHECK(output.result.at("result").at("state") == "idle");
    output = control->interpret("feature.dictionary", json::object(), 0);
    REQUIRE(!output.error);
    const auto missing = output.result.at("request_id").get<std::uint32_t>();
    REQUIRE(runner.wait_for([&](const auto&) {
        output = control->interpret("feature.dictionary", json::object(), missing);
        return static_cast<bool>(output.error);
    }, 30000));
    CHECK(output.error.code == "interpret.missing_dictionary");
    CHECK(control->interpret("trace.read", json::object(), request).error.code == "interpret.unknown_request");
}

TEST_CASE("runner: interpretation final replies survive reordered transport replies") {
    Workspace w("interpretation-replies");
    if (const char* python = std::getenv("CALIPER_INTERPRET_PYTHON")) w.python = python;
    REQUIRE(!w.python.empty());
    std::ofstream script(w.ws + "/scripts/stand_in_run.py");
    script << R"PY(import os
import threading
from pathlib import Path
import caliper
from caliper import _wire
from caliper._controls import CONTROLS
from caliper.interp._service import register_service
register_service()
original = CONTROLS._on_control

def reordered(route, conn, meta, **kwargs):
    if meta["op"] != "interpret":
        return original(route, conn, meta, **kwargs)
    rid = meta["request_id"]
    method = meta["interpretation"]["method"]
    # These deferred frames follow the final reply returned to the sender.
    route._reply(conn, _wire.ControlReply(rid, "interpret", "queued").to_meta())
    route._reply(conn, _wire.ControlReply(rid, "interpret", "error",
        error={"code": "interpret.closed", "message": "late duplicate"}).to_meta())
    if method == "feature.dictionary":
        return _wire.ControlReply(rid, "interpret", "error",
            error={"code": "interpret.missing_dictionary", "message": "fixture refusal"}).to_meta()
    return _wire.ControlReply(rid, "interpret", "ok", result={"state": "idle"}).to_meta()

CONTROLS._on_control = reordered
server = caliper.serve(control=True)
Path(os.environ["CALIPER_RUN_LINK_FILE"]).write_text(server.url)
threading.Event().wait()
)PY";
    script.close();
    cr::ModelRunner::Options options;
    options.workspace = w.ws;
    options.python = w.python;
    cr::ModelRunner runner(options);
    struct Guard { cr::ModelRunner& runner; ~Guard() { runner.shutdown(); } } guard{runner};
    cb::RunEntry entry;
    REQUIRE(!runner.start(w.start({}, json::object(), "interpretation-replies"), &entry));
    REQUIRE(runner.wait_for(state_is(entry.id, "running"), 30000));
    cb::Error error;
    auto* control = runner.control_of(entry.id, &error);
    REQUIRE(control);
    REQUIRE(runner.wait_for([&](const auto&) { return control->state().control; }, 30000));
    std::vector<std::pair<std::string, std::uint32_t>> requests;
    for (const auto* method : {"trace.status", "feature.dictionary", "trace.read"}) {
        auto output = control->interpret(method, json::object(), 0);
        REQUIRE(!output.error);
        const auto request = output.result.at("request_id").get<std::uint32_t>();
        requests.emplace_back(method, request);
        REQUIRE(runner.wait_for([&](const auto&) {
            output = control->interpret(method, json::object(), request);
            return output.error || !output.result.at("pending").get<bool>();
        }, 30000));
    }
    // The third reply fences the late frames for the first two requests.
    auto success = control->interpret(requests[0].first, json::object(), requests[0].second);
    REQUIRE(!success.error);
    CHECK_FALSE(success.result.at("pending").get<bool>());
    CHECK(success.result.at("result").at("state") == "idle");
    auto refused = control->interpret(requests[1].first, json::object(), requests[1].second);
    CHECK(refused.error.code == "interpret.missing_dictionary");
}

TEST_CASE("runner: native inference keeps prompts out of arguments and excludes shadow packages") {
    Workspace w("inference-entry");
    if (const char* python = std::getenv("CALIPER_INTERPRET_PYTHON")) w.python = python;
    REQUIRE(!w.python.empty());
    std::string installation;
    SUBCASE("installed package") { installation = "installed"; }
    SUBCASE("editable package") { installation = "editable"; }
    cr::ChildSpec setup;
    setup.program = w.python;
    setup.args = {(fs::path(CALIPER_RUNNER_STAND_IN).parent_path() / "inference_probe.py").u8string(),
                  w.ws, CALIPER_RUNNER_PYTHONPATH, installation};
    setup.env = cr::current_environment();
    setup.output = w.ws + "/setup.log";
    std::string error;
    auto child = cr::start_child(setup, &error);
    REQUIRE_MESSAGE(child, error);
    struct ChildGuard {
        cr::ChildProcess& child;
        ~ChildGuard() { child.kill(); std::int64_t code = 0; child.wait(2000, &code); }
    } child_guard{*child};
    std::int64_t code = 0;
    REQUIRE(child->wait(15000, &code));
    REQUIRE_MESSAGE(code == 0, read_file(setup.output));

    struct Environment {
        std::vector<std::pair<std::string, std::optional<std::string>>> values;
        void set(const std::string& name, const std::string& value) {
            const char* old = std::getenv(name.c_str());
            values.emplace_back(name, old ? std::optional<std::string>(old) : std::nullopt);
            setenv(name.c_str(), value.c_str(), 1);
        }
        ~Environment() {
            for (const auto& [name, value] : values) {
                if (value) setenv(name.c_str(), value->c_str(), 1);
                else unsetenv(name.c_str());
            }
        }
    } environment;
    environment.set("PYTHONPATH", w.ws + "/shadow-path");
    environment.set("CALIPER_RUN_PROMPT", "stale inherited prompt");
    environment.set("CALIPER_RUN_PYTHON", "");
    cr::ModelRunner::Options options;
    options.workspace = w.ws;   // Exercise normal workspace interpreter selection.
    cr::ModelRunner runner(options);
    struct Guard { cr::ModelRunner& runner; ~Guard() { runner.shutdown(); } } guard{runner};
    cb::RunStartRequest request;
    request.kind = "inference";
    request.id = "inference-probe";
    request.model = "fixture model";
    request.dictionary = "fixture dictionary";
    request.prompt = u8"native prompt marker: 'quotes' \"double\" $value $(literal)\n\u03bb";
    request.params = json{{"prompt", "parameter override"}};
    cb::RunEntry entry;
    REQUIRE(!runner.start(request, &entry));
    REQUIRE(runner.wait_for(state_is(entry.id, "exited"), 15000));
    const auto observed = json::parse(read_file(w.ws + "/runs/" + entry.id + "/output.log"));
    CHECK(observed.at("prompt") == request.prompt);
    CHECK(observed.at("model") == request.model);
    CHECK(observed.at("dictionary") == request.dictionary);
    CHECK(observed.at("params") == request.params);
    CHECK(observed.at("prompt_removed") == true);
    CHECK(observed.at("executable") == w.ws + "/.venv/bin/python");
    const auto arguments = observed.at("original_argv").get<std::vector<std::string>>();
    CHECK(arguments == std::vector<std::string>{w.ws + "/.venv/bin/python", "-E", "-P", "-u", "-m",
          "caliper.interp._entry", "--model=" + request.model, "--dictionary=" + request.dictionary});
    for (const char* key : {"original_argv", "argv"}) {
        CHECK(observed.at(key).dump().find("native prompt marker") == std::string::npos);
        CHECK(observed.at(key).dump().find("--prompt") == std::string::npos);
    }
    CHECK(runner.running_pids().empty());

    auto script = w.start({"ordinary script argument"}, json::object(), "script-probe");
    script.script = "script_probe.py";
    script.script_path = w.ws + "/script_probe.py";
    REQUIRE(!runner.start(script, &entry));
    REQUIRE(runner.wait_for(state_is(entry.id, "exited"), 15000));
    const auto ordinary = json::parse(read_file(w.ws + "/runs/" + entry.id + "/output.log"));
    CHECK(ordinary.at("argv") == script.args);
    CHECK(ordinary.at("value") == "script path retained");
    CHECK(ordinary.at("prompt_removed") == true);
    CHECK(runner.running_pids().empty());
}
