// Tests of the runs panel (runs_panel.h, design/hosted-runs.md): the fields,
// the card text, the id it suggests, the workspace's scripts, the layout at
// two window sizes, and the panel's clicks over a real session, whose
// run.start, run.fork and run.stop reach a fake model runner with the origin
// ui; then, with a Python that imports Caliper LIB, the same clicks over the
// app's model runner and the stand-in run, from the start to the stop of a
// run and its fork, with no child left. Every wait waits on the runner's
// state, never on a fixed sleep, and every child ends by its process id, also
// when a check fails.
#include "builder/runs_panel.h"
#include "runner/model_runner.h"
#include "runs_panel_fake_runner.h"

#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace cb = caliper_builder;
namespace fs = std::filesystem;
using caliper_host::RunsPanel;
using nlohmann::json;
using runs_panel_test::FakeRunner;

namespace {

std::string fresh_dir(const std::string& name) {
    const fs::path p = fs::path(CALIPER_RUNS_PANEL_TEST_TMP) / name;
    std::error_code ec;
    fs::remove_all(p, ec);
    fs::create_directories(p);
    return fs::canonical(p).u8string();
}

void write_text(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << text;
}

std::unique_ptr<cb::Session> open_session(const std::string& ws) {
    cb::Session::Open o = cb::Session::open(ws);
    REQUIRE_MESSAGE(!o.error, o.error.message);
    return std::move(o.session);
}

// One frame of the app: the session's queue runs what waits, then the panel
// reads the answer, as AppSession::frame() and the panel's frame() do.
void tick(cb::Session& s, RunsPanel& panel, cb::ModelRunner* runner) {
    s.queue().drain();
    panel.frame(&s, runner);
}

// Six pixels a character, as a monospace face would measure.
float measure(const std::string& t) { return 6.0f * static_cast<float>(t.size()); }

}  // namespace

TEST_CASE("runs panel: the Parameters and Arguments fields") {
    json pairs;
    std::string problem;
    REQUIRE(caliper_host::parse_run_params("lr=0.01 momentum=0.9, opt=adam warm=true n=3 note=\"a b\"", &pairs,
                                           &problem));
    CHECK(pairs == json::parse(R"([{"name":"lr","value":0.01},{"name":"momentum","value":0.9},)"
                               R"({"name":"opt","value":"adam"},{"name":"warm","value":true},)"
                               R"({"name":"n","value":3},{"name":"note","value":"a b"}])"));
    CHECK(caliper_host::run_params_text(pairs) == "lr=0.01 momentum=0.9 opt=adam warm=true n=3 note=\"a b\"");
    CHECK(caliper_host::run_params_text(json{{"lr", 0.05}}) == "lr=0.05");
    REQUIRE(caliper_host::parse_run_params("", &pairs, &problem));
    CHECK(pairs.empty());

    CHECK_FALSE(caliper_host::parse_run_params("lr", &pairs, &problem));
    CHECK(problem == "Parameter lr has no value; write name=value.");
    CHECK_FALSE(caliper_host::parse_run_params("lr=1 lr=2", &pairs, &problem));
    CHECK(problem == "Parameter lr is given twice.");
    CHECK_FALSE(caliper_host::parse_run_params("=1", &pairs, &problem));
    CHECK_FALSE(caliper_host::parse_run_params("note=\"open", &pairs, &problem));

    std::vector<std::string> args;
    REQUIRE(caliper_host::split_run_args("200  0.05 \"two words\"", &args, &problem));
    CHECK(args == std::vector<std::string>{"200", "0.05", "two words"});
    CHECK_FALSE(caliper_host::split_run_args("\"open", &args, &problem));
}

// The card of a start or a fork, as the session's card and the panel's show it.
static std::string card_text(const char* name, json args, const std::string& script = {}) {
    cb::CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    return cb::run_card_text(inv, script);
}

TEST_CASE("runs panel: the card text of a start and a fork") {
    json pairs;
    std::string problem;
    REQUIRE(caliper_host::parse_run_params("lr=0.01", &pairs, &problem));
    CHECK(card_text("run.start", {{"script", "scripts/train.py"}, {"id", "train-1"}, {"args", {"200", "a b"}},
                                  {"params", pairs}}) ==
          "Start scripts/train.py as the managed run train-1, with the arguments 200 \"a b\" and the parameters "
          "lr=0.01. The script runs as Python code on this computer.");
    CHECK(card_text("run.start", {{"script", "train.py"}}) ==
          "Start train.py as a new managed run, with no arguments and no parameters. The script runs as Python code "
          "on this computer.");
    CHECK(card_text("run.fork", {{"run", "train-1"}, {"params", pairs}}, "scripts/train.py") ==
          "Fork the run train-1 at its next checkpoint, with the parameters lr=0.01. The fork runs scripts/train.py "
          "as Python code beside its parent.");
}

TEST_CASE("runs panel: the suggested id and the workspace's scripts") {
    const std::string ws = fresh_dir("scripts");
    write_text(fs::path(ws) / "train.py", "");
    write_text(fs::path(ws) / "scripts" / "Eval Model.py", "");
    write_text(fs::path(ws) / "scripts" / "notes.txt", "");
    write_text(fs::path(ws) / "a" / "b" / "c" / "deep.py", "");
    write_text(fs::path(ws) / "a" / "b" / "c" / "d" / "deeper.py", "");
    write_text(fs::path(ws) / "runs" / "train-1" / "copy.py", "");
    write_text(fs::path(ws) / ".venv" / "lib" / "site.py", "");
    write_text(fs::path(ws) / "scripts" / "__pycache__" / "x.py", "");
    CHECK(caliper_host::workspace_scripts(ws) ==
          std::vector<std::string>{"a/b/c/deep.py", "scripts/Eval Model.py", "train.py"});

    std::vector<cb::RunEntry> runs(1);
    runs[0].id = "train-2";
    // runs/train-1 exists and train-2 is listed: the next free id is train-3.
    CHECK(caliper_host::suggest_run_id("train.py", runs, ws) == "train-3");
    CHECK(caliper_host::suggest_run_id("scripts/Eval Model.py", runs, ws) == "eval-model-1");
    CHECK(caliper_host::suggest_run_id("___.py", {}, ws) == "run-1");
}

TEST_CASE("runs panel: the layout at two window sizes") {
    const caliper_host::RunsPanelMetrics m = caliper_host::RunsPanelMetrics::for_text_scale(1.0f);
    for (const auto& size : {std::pair<float, float>{2560, 1440}, std::pair<float, float>{1440, 900}}) {
        caliper_host::RunsPanelLayoutInput in;
        in.area = {0, 40, size.first, size.second - 80};
        in.runs = 4;
        in.scripts = {"scripts/eval.py", "scripts/train.py"};
        in.card = card_text("run.fork", {{"run", "train-1"}, {"params", {{{"name", "lr"}, {"value", 0.02}}}}},
                            "scripts/train.py");
        in.problem = true;
        const caliper_host::RunsPanelLayout L = caliper_host::layout_runs_panel(in, m, measure);
        CAPTURE(size.first);
        CHECK(L.window.w == doctest::Approx(std::round(size.first * m.share_w)));
        CHECK(L.window.x >= in.area.x);
        CHECK(L.window.right() <= in.area.right());
        CHECK(L.window.y >= in.area.y);
        CHECK(L.window.bottom() <= in.area.bottom());
        REQUIRE(L.rows.size() == 4);
        CHECK(L.rows_folded == 0);
        // Each part sits inside the window, below the one before.
        const float right = L.window.right() - m.pad;
        CHECK(L.rows.back().stop.right() <= right + 0.5f);
        CHECK(L.column_x[5] + L.column_w[5] <= L.rows[0].fork.x + 0.5f);
        CHECK(L.scripts.size() == 2);
        CHECK(L.script.y > L.rows.back().row.bottom());
        CHECK(L.params.y > L.args.bottom());
        CHECK(L.go.right() <= right + 0.5f);
        CHECK(L.card.y > L.id.bottom());
        CHECK_FALSE(L.card_lines.empty());
        for (const auto& line : L.card_lines) CHECK(measure(line) <= L.card.w - 2 * m.gap);
        CHECK(L.approve.bottom() <= L.card.bottom());
        CHECK(L.problem.bottom() <= L.window.bottom() - m.pad + 0.5f);
    }

    // A small work area folds the runs that do not fit into "+N".
    caliper_host::RunsPanelLayoutInput in;
    in.area = {0, 0, 900, 420};
    in.runs = 30;
    const caliper_host::RunsPanelLayout L = caliper_host::layout_runs_panel(in, m, measure);
    CHECK(L.rows_folded > 0);
    CHECK(L.rows.size() + L.rows_folded == 30);
    CHECK(L.window.bottom() <= in.area.bottom());
    CHECK(L.folded.w > 0);

    // The minimum grows with the text scale.
    CHECK(caliper_host::RunsPanelMetrics::for_text_scale(2.0f).min_w == 2 * m.min_w);
}

TEST_CASE("runs panel: start, fork and stop go through run.start, run.fork and run.stop with the origin ui") {
    const std::string ws = fresh_dir("panel_fake") + "/workspace";
    auto s = open_session(ws);
    write_text(fs::path(ws) / "scripts" / "train.py", "print('train')\n");
    FakeRunner runner;
    s->set_model_runner(&runner);
    RunsPanel panel;
    panel.frame(s.get(), &runner);
    panel.set_open(true);
    CHECK(panel.runs().empty());
    CHECK(panel.scripts() == std::vector<std::string>{"scripts/train.py"});
    CHECK(panel.script() == "scripts/train.py");
    CHECK(panel.id() == "train-1");

    // Start sends run.start, which the session parks on its one card; the
    // panel shows that card, and nothing starts before Approve.
    panel.set_args("200");
    panel.set_params("lr=0.05");
    REQUIRE(panel.review());
    CHECK(panel.busy());
    CHECK_FALSE(panel.review());           // one command in flight
    tick(*s, panel, &runner);
    const std::vector<cb::PendingApproval> waiting = s->registry().pending_approvals();
    REQUIRE(waiting.size() == 1);
    CHECK(waiting[0].invocation.name == "run.start");
    CHECK(waiting[0].invocation.origin == cb::CommandOrigin::ui);
    CHECK(panel.card() ==
          "Start scripts/train.py as the managed run train-1, with the arguments 200 and the parameters lr=0.05. "
          "The script runs as Python code on this computer.");
    CHECK(panel.card() == cb::run_card_text(waiting[0].invocation, ""));
    tick(*s, panel, &runner);
    CHECK(runner.starts.empty());
    REQUIRE(panel.approve());
    CHECK(panel.card().empty());
    CHECK_FALSE(panel.approve());          // decided once
    CHECK(panel.busy());
    panel.frame(s.get(), &runner);         // the decision waits for the drain; the card stays gone
    CHECK(panel.card().empty());
    tick(*s, panel, &runner);
    CHECK(s->registry().pending_approvals().empty());
    CHECK_FALSE(panel.busy());
    REQUIRE(runner.starts.size() == 1);
    CHECK(runner.starts[0].script == "scripts/train.py");
    CHECK(runner.starts[0].script_path == fs::canonical(fs::path(ws) / "scripts" / "train.py").u8string());
    CHECK(runner.starts[0].args == std::vector<std::string>{"200"});
    CHECK(runner.starts[0].params == json{{"lr", 0.05}});
    CHECK(runner.starts[0].id == "train-1");
    CHECK(panel.last_run() == "train-1");
    CHECK(panel.problem().empty());
    REQUIRE(panel.runs().size() == 1);
    // The next free id follows.
    CHECK(panel.id() == "train-2");

    // A starting run offers Stop and not Fork; a running one both.
    CHECK_FALSE(RunsPanel::can_fork(panel.runs()[0]));
    CHECK(RunsPanel::can_stop(panel.runs()[0]));
    runner.runs[0].state = "running";
    panel.frame(s.get(), &runner);
    panel.begin_fork("train-1");
    CHECK(panel.fork_of() == "train-1");
    CHECK(panel.params() == "lr=0.05");
    panel.set_params("lr=0.01");
    REQUIRE(panel.review());
    tick(*s, panel, &runner);
    CHECK(panel.card() == "Fork the run train-1 at its next checkpoint, with the parameters lr=0.01. The fork runs "
                          "scripts/train.py as Python code beside its parent.");
    CHECK(runner.forks.empty());
    REQUIRE(panel.approve());
    tick(*s, panel, &runner);
    REQUIRE(runner.forks.size() == 1);
    CHECK(runner.forks[0].parent == "train-1");
    CHECK(runner.forks[0].params == json{{"lr", 0.01}});
    CHECK(runner.forks[0].id == "train-2");
    CHECK(panel.fork_of().empty());
    CHECK(panel.last_run() == "train-2");

    // Stop: at once, with the default limit.
    REQUIRE(panel.stop("train-1"));
    tick(*s, panel, &runner);
    REQUIRE(runner.stops.size() == 1);
    CHECK(runner.stops[0].first == "train-1");
    CHECK(runner.stops[0].second == doctest::Approx(10.0));
    CHECK_FALSE(RunsPanel::can_stop(panel.runs()[0]));   // stopping
    CHECK_FALSE(panel.stop("train-1"));

    // Cancel denies the session's card: nothing starts.
    REQUIRE(panel.review());
    tick(*s, panel, &runner);
    REQUIRE_FALSE(panel.card().empty());
    REQUIRE(panel.cancel_review());
    tick(*s, panel, &runner);
    CHECK_FALSE(panel.busy());
    CHECK(panel.card().empty());
    CHECK(runner.starts.size() == 1);
    CHECK_FALSE(panel.problem().empty());

    // The session's refusal shows as the problem: an id that exists.
    panel.set_id("train-1");
    REQUIRE(panel.review());
    tick(*s, panel, &runner);
    if (!panel.card().empty()) {
        REQUIRE(panel.approve());
        tick(*s, panel, &runner);
    }
    CHECK(runner.starts.size() == 2);
    CHECK(panel.problem() == "A run train-1 exists in this workspace.");
    // A wrong field never reaches the session.
    panel.set_params("lr");
    CHECK_FALSE(panel.review());
    CHECK(panel.problem() == "Parameter lr has no value; write name=value.");
    panel.set_params("");
    panel.set_id("Bad Id");
    CHECK_FALSE(panel.review());
    CHECK(runner.starts.size() == 2);

    const std::vector<std::string> shown = panel.visible_text();
    for (const char* t : {"Runs", "Id", "State", "Script", "Parent", "Step", "Detail", "train-1", "train-2", "STARTING",
                          "RUNNING", "stopping", "Start", "Parameters"})
        CHECK_MESSAGE(std::find(shown.begin(), shown.end(), std::string(t)) != shown.end(), t);
    s->set_model_runner(nullptr);
}

TEST_CASE("runs panel: without a model runner or a workspace, Start says why") {
    RunsPanel panel;
    panel.frame(nullptr, nullptr);
    panel.set_open(true);
    panel.set_script("train.py");
    CHECK_FALSE(panel.review());
    CHECK(panel.problem() == "No workspace is open.");
    const std::string ws = fresh_dir("panel_none") + "/workspace";
    auto s = open_session(ws);
    panel.frame(s.get(), nullptr);
    CHECK_FALSE(panel.review());
    CHECK(panel.problem() == "This app has no model runner.");
}

TEST_CASE("runs panel: a run and its fork over the model runner, from the start to the stop") {
    const std::string python = CALIPER_RUNS_PANEL_TEST_PYTHON;
    if (python.empty()) {
        MESSAGE("skipped: no Python that imports Caliper LIB (CALIPER_ACCEPTANCE_PYTHON)");
        return;
    }
    const std::string ws = fresh_dir("panel_runner") + "/workspace";
    auto s = open_session(ws);
    fs::create_directories(fs::path(ws) / "scripts");
    fs::copy_file(CALIPER_RUNS_PANEL_STAND_IN, fs::path(ws) / "scripts" / "stand_in_run.py");
    setenv("PYTHONPATH", CALIPER_RUNS_PANEL_PYTHONPATH, 1);   // the children import Caliper LIB from this checkout
    caliper_runner::ModelRunner::Options o;
    o.workspace = s->workspace();
    o.python = python;
    o.start_limit_s = 60.0;
    caliper_runner::ModelRunner runner(o);
    s->set_model_runner(&runner);
    struct Guard {   // every child ends by its process id, also when a check fails
        cb::Session& s;
        caliper_runner::ModelRunner& r;
        ~Guard() {
            s.set_model_runner(nullptr);
            r.shutdown();
        }
    } guard{*s, runner};
    auto state_is = [&runner](const std::string& id, const std::string& state) {
        return runner.wait_for(
            [&](const std::vector<cb::RunEntry>& runs) {
                for (const auto& e : runs)
                    if (e.id == id) return e.state == state;
                return false;
            },
            60000);
    };

    RunsPanel panel;
    panel.frame(s.get(), &runner);
    panel.set_open(true);
    REQUIRE(panel.script() == "scripts/stand_in_run.py");
    panel.set_params("lr=0.05");
    REQUIRE(panel.review());
    tick(*s, panel, &runner);
    REQUIRE(panel.approve());
    tick(*s, panel, &runner);
    REQUIRE_MESSAGE(panel.problem().empty(), panel.problem());
    const std::string base = panel.last_run();
    CHECK(base == "stand-in-run-1");
    REQUIRE(state_is(base, "running"));

    panel.frame(s.get(), &runner);
    panel.begin_fork(base);
    REQUIRE(panel.fork_of() == base);
    CHECK(panel.params() == "lr=0.05");
    panel.set_params("lr=0.01");
    REQUIRE(panel.review());
    tick(*s, panel, &runner);
    REQUIRE(panel.approve());
    tick(*s, panel, &runner);
    REQUIRE_MESSAGE(panel.problem().empty(), panel.problem());
    const std::string fork = panel.last_run();
    CHECK(fork == "stand-in-run-2");
    REQUIRE(state_is(fork, "running"));
    panel.frame(s.get(), &runner);
    REQUIRE(panel.runs().size() == 2);
    CHECK(panel.runs()[1].parent == base);

    REQUIRE(panel.stop(fork));
    tick(*s, panel, &runner);
    REQUIRE(panel.stop(base));
    tick(*s, panel, &runner);
    CHECK(state_is(fork, "exited"));
    CHECK(state_is(base, "exited"));
    CHECK(runner.running_pids().empty());
}
