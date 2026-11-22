// Tests of the model runner (src/host/runner/model_runner.h,
// design/hosted-runs.md): the child processes, the stream links, and managed
// runs of the stand-in script tests/stand_in_run.py, which follows the
// process contract with Caliper LIB: a start, the run sources, the run
// control through the runner's connection, a fork from a checkpoint with a
// lower learning rate whose loss continues from its parent's, a stop of both,
// a start that writes no link, and no child left at the end. Every wait
// waits on the runner's state, never on a fixed sleep, and every child is
// ended by its own process id, also when a check fails.
#include "runner/child_process.h"
#include "runner/model_runner.h"
#include "runner/run_connection.h"

#include <caliper_builder/session/events.h>

#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

namespace cb = caliper_builder;
namespace cr = caliper_runner;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

std::string fresh_dir(const std::string& name) {
    const fs::path p = fs::path(CALIPER_RUNNER_TEST_TMP) / name;
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
    cr::ModelRunner::Options o;
    o.workspace = w.ws;
    o.python = w.python.empty() ? "python3" : w.python;
    o.start_limit_s = 1.0;
    o.kill_grace_s = 1.0;
    cr::ModelRunner runner(o);
    cb::RunStartRequest r;
    r.script_path = w.ws + "/scripts/silent.py";
    r.script = "scripts/silent.py";
    cb::RunEntry e;
    REQUIRE(!runner.start(r, &e));
    CHECK(e.id == "silent-1");
    CHECK(e.state == "starting");
    CHECK(fs::is_directory(w.ws + "/runs/silent-1"));
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
        MESSAGE("skipped: no Python that imports Caliper LIB (CALIPER_M1_PYTHON)");
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
