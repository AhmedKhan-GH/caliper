// Tests of the model runner's commands (runner_commands.h,
// design/hosted-runs.md): run.list, run.start, run.fork, run.stop and run.logs
// over a fake ModelRunner, the checks of the script, the id and the
// parameters, the card that a start and a fork show in every mode with no
// rule, and the argument `run` of the run commands of M1. Every result passes
// its schema (open_session checks results).
#include "helpers.h"

#include <caliper_builder/commands/permissions.h>
#include <caliper_builder/session/events.h>
#include <caliper_builder/session/runner_commands.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

#include <filesystem>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

// A run control with a fixed state that records its requests.
struct FakeControl final : RunControl {
    RunState run;
    std::vector<std::string> sent;
    RunState state() override { return run; }
    Error set_param(const std::string& name, const json&) override {
        sent.push_back("set " + name);
        return {};
    }
    Error run_action(const std::string& name) override {
        sent.push_back("run " + name);
        return {};
    }
    Error pause(RunPauseOp, std::int64_t) override {
        sent.push_back("pause");
        return {};
    }
};

// A runner that records each request and answers from a table the test sets.
struct FakeRunner final : ModelRunner {
    std::vector<RunEntry> runs;
    std::vector<RunStartRequest> starts;
    std::vector<RunForkRequest> forks;
    std::vector<std::pair<std::string, double>> stops;
    std::vector<std::pair<std::string, int>> log_calls;
    std::vector<std::string> controls_asked;
    FakeControl control;   // the control of the run "train-1"

    std::vector<RunEntry> list() override { return runs; }
    Error start(const RunStartRequest& r, RunEntry* out) override {
        starts.push_back(r);
        RunEntry e;
        e.id = r.id.empty() ? "train-1" : r.id;
        e.script = r.script;
        e.args = r.args;
        e.params = r.params;
        runs.push_back(e);
        *out = e;
        return {};
    }
    Error fork(const RunForkRequest& r, RunEntry* out) override {
        forks.push_back(r);
        if (r.parent != "train-1") return Error(kRunNotFound, "There is no run " + r.parent + ".");
        RunEntry e;
        e.id = r.id.empty() ? "train-2" : r.id;
        e.script = "train.py";
        e.parent = r.parent;
        e.params = r.params;
        *out = e;
        return {};
    }
    Error stop(const std::string& run, double after, RunEntry* out) override {
        stops.emplace_back(run, after);
        RunEntry e;
        e.id = run;
        e.state = "running";
        e.script = "train.py";
        e.stopping = true;
        e.step = 40;
        *out = e;
        return {};
    }
    Error logs(const std::string& run, int lines, std::vector<std::string>* out) override {
        log_calls.emplace_back(run, lines);
        *out = {"step 1 loss 2.3", "step 2 loss 2.1"};
        return {};
    }
    RunControl* control_of(const std::string& run, Error* error) override {
        controls_asked.push_back(run);
        if (run == "train-1") return &control;
        *error = Error(kRunNotFound, "There is no run " + run + ".");
        return nullptr;
    }
};

// A workspace with scripts/train.py, and a script outside it.
struct Fixture {
    std::string root, ws;
    std::unique_ptr<Session> s;
    FakeRunner runner;

    explicit Fixture(const std::string& name) {
        root = fresh_dir(name);
        ws = root + "/workspace";
        s = open_session(ws);
        std::filesystem::create_directories(ws + "/scripts");
        write_text(ws + "/scripts/train.py", "print('train')\n");
        write_text(root + "/outside.py", "print('outside')\n");
        s->set_model_runner(&runner);
    }
    ~Fixture() { s->set_model_runner(nullptr); }
};

CommandResult send(Session& s, const std::string& name, json args, CommandOrigin origin, const std::string& client) {
    CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = origin;
    inv.client = client;
    return s.invoke(std::move(inv));
}

// Runs a start or a fork that must wait for its card, approves the card as
// the person, and returns the result.
json approved(Session& s, const std::string& name, json args, CommandOrigin origin = CommandOrigin::ui) {
    const CommandResult r = send(s, name, std::move(args), origin, "");
    INFO(name << " answered " << (r.ok ? r.result.dump() : r.error.code + ": " + r.error.message));
    REQUIRE(r.error.code == "command.needs_approval");
    const CommandResult done = s.registry().approve(r.approval);
    INFO(name << " after approval: " << done.error.code << ": " << done.error.message);
    REQUIRE(done.ok);
    return done.result;
}

// A managed run that runs, as the fake runner lists it.
RunEntry running(const std::string& id) {
    RunEntry e;
    e.id = id;
    e.state = "running";
    e.script = "scripts/train.py";
    return e;
}

}  // namespace

TEST_CASE("runner commands: a session without a model runner answers command.not_available") {
    auto s = open_session(fresh_dir("runner_none") + "/workspace");
    CHECK(s->model_runner() == nullptr);
    CHECK(err(*s, "run.list") == "command.not_available");
    CHECK(err(*s, "run.start", {{"script", "train.py"}}) == "command.not_available");
    CHECK(err(*s, "run.fork", {{"run", "train-1"}}) == "command.not_available");
    CHECK(err(*s, "run.stop", {{"run", "train-1"}}) == "command.not_available");
    CHECK(err(*s, "run.logs", {{"run", "train-1"}}) == "command.not_available");
    // The run commands of M1 know only the attached script, user, without a runner.
    FakeControl control;
    control.run.attached = true;
    control.run.script = "mine.py";
    s->set_run_control(&control);
    CHECK(ok(*s, "run.state", {{"run", "user"}})["script"] == "mine.py");
    CHECK(err(*s, "run.state", {{"run", "train-1"}}) == kRunNotFound);
    s->set_run_control(nullptr);
}

TEST_CASE("runner commands: run.start takes only a file of the workspace and checks its id and parameters") {
    Fixture f("runner_start");
    CHECK(err(*f.s, "run.start", {{"script", "missing.py"}}) == kRunScriptRefused);
    CHECK(err(*f.s, "run.start", {{"script", "scripts"}}) == kRunScriptRefused);
    CHECK(err(*f.s, "run.start", {{"script", "../outside.py"}}) == kRunScriptRefused);
    CHECK(err(*f.s, "run.start", {{"script", f.root + "/outside.py"}}) == kRunScriptRefused);
    CHECK(err(*f.s, "run.start", {{"script", "scripts/train.py"}, {"id", "Train_1"}}) == "command.invalid_args");
    json many = json::array();
    for (int i = 0; i < 65; ++i) many.push_back(json{{"name", "p" + std::to_string(i)}, {"value", i}});
    CHECK(err(*f.s, "run.start", {{"script", "scripts/train.py"}, {"params", many}}) == "command.invalid_args");
    CHECK(err(*f.s, "run.start", {{"script", "scripts/train.py"}, {"params", json::array({{{"name", ""}, {"value", 1}}})}}) ==
          "command.invalid_args");
    const json twice = json::array({{{"name", "lr"}, {"value", 1}}, {{"name", "lr"}, {"value", 2}}});
    CHECK(err(*f.s, "run.start", {{"script", "scripts/train.py"}, {"params", twice}}) == "command.invalid_args");
    CHECK(f.runner.starts.empty());

    const json r = approved(*f.s, "run.start",
                            {{"script", "scripts/train.py"}, {"args", {"--epochs", "3"}},
                             {"params", json::array({{{"name", "lr"}, {"value", 0.01}}})}, {"id", "base"}});
    REQUIRE(f.runner.starts.size() == 1);
    const RunStartRequest& req = f.runner.starts[0];
    CHECK(req.script == "scripts/train.py");
    CHECK(req.script_path == std::filesystem::canonical(f.ws + "/scripts/train.py").u8string());
    CHECK(req.args == std::vector<std::string>{"--epochs", "3"});
    CHECK(req.params == json{{"lr", 0.01}});
    CHECK(req.id == "base");
    CHECK(r["run"]["id"] == "base");
    CHECK(r["run"]["state"] == "starting");
    CHECK(r["run"]["managed"] == true);
    CHECK(r["run"]["parent"].is_null());
    CHECK(r["run"]["params"] == json::array({{{"name", "lr"}, {"value", 0.01}}}));
    approved(*f.s, "run.start", {{"script", "scripts/train.py"}});
    CHECK(f.runner.starts.at(1).script == "scripts/train.py");
    CHECK(f.runner.starts.at(1).id.empty());

    const json list = ok(*f.s, "run.list");
    REQUIRE(list["runs"].size() == 2);
    CHECK(list["runs"][0]["id"] == "base");
}

TEST_CASE("runner commands: a start and a fork show a card in every mode, and no rule removes it") {
    Fixture f("runner_card");
    f.runner.runs.push_back(running("train-1"));
    const json start{{"script", "scripts/train.py"}};
    // mcp in Ask: the card, with no rule on it; approved, the start runs.
    CommandResult r = send(*f.s, "run.start", start, CommandOrigin::mcp, "");
    REQUIRE(r.error.code == "command.needs_approval");
    PendingApproval p = f.s->registry().pending_approvals().at(0);
    CHECK_FALSE(f.s->approval_offers_rules(p.invocation));
    CHECK(f.s->allow_always(p.invocation, RuleScope::session).code == "permissions.no_rule");
    CHECK(f.runner.starts.empty());
    REQUIRE(f.s->registry().approve(p.approval).ok);
    CHECK(f.runner.starts.size() == 1);

    // Allow all of the chat and Accept edits still show the card.
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::allow_all).ok());
    CHECK(send(*f.s, "run.start", start, CommandOrigin::agent, "chat").error.code == "command.needs_approval");
    CHECK(send(*f.s, "run.fork", {{"run", "train-1"}}, CommandOrigin::agent, "chat").error.code ==
          "command.needs_approval");
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::mcp, AgentMode::accept_edits).ok());
    CHECK(send(*f.s, "run.start", start, CommandOrigin::mcp, "").error.code == "command.needs_approval");
    // The API asks too, and so does the person's own panel (origin ui).
    CHECK(send(*f.s, "run.start", start, CommandOrigin::api, "").error.code == "command.needs_approval");
    CHECK(send(*f.s, "run.start", start, CommandOrigin::ui, "").error.code == "command.needs_approval");
    CHECK(send(*f.s, "run.fork", {{"run", "train-1"}}, CommandOrigin::builder, "").error.code ==
          "command.needs_approval");
    f.s->registry().cancel_pending();
    CHECK(f.runner.starts.size() == 1);
    CHECK(f.runner.forks.empty());
    approved(*f.s, "run.start", start, CommandOrigin::builder);
    CHECK(f.runner.starts.size() == 2);
    // A request that its handler refuses gets no card: the reason comes at once.
    CHECK(send(*f.s, "run.start", {{"script", "missing.py"}}, CommandOrigin::mcp, "").error.code == kRunScriptRefused);
    CHECK(send(*f.s, "run.fork", {{"run", "gone"}}, CommandOrigin::mcp, "").error.code == kRunNotFound);
    CHECK(f.s->registry().pending_approvals().empty());

    // Read only refuses them.
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::mcp, AgentMode::read_only).ok());
    CHECK(send(*f.s, "run.start", start, CommandOrigin::mcp, "").error.code == "command.denied");
    CHECK(f.runner.starts.size() == 2);

    // run.stop is a run command of M1's kind: it asks, and a rule can allow it.
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::mcp, AgentMode::ask).ok());
    r = send(*f.s, "run.stop", {{"run", "train-1"}}, CommandOrigin::mcp, "");
    REQUIRE(r.error.code == "command.needs_approval");
    p = f.s->registry().pending_approvals().at(0);
    CHECK(f.s->approval_offers_rules(p.invocation));
    f.s->registry().cancel_pending();
    // The reads run at once.
    CHECK(send(*f.s, "run.list", json::object(), CommandOrigin::mcp, "").ok);
    CHECK(send(*f.s, "run.logs", {{"run", "train-1"}}, CommandOrigin::mcp, "").ok);
}

TEST_CASE("runner commands: run.fork, run.stop and run.logs need a managed run and pass their arguments") {
    Fixture f("runner_fork");
    f.runner.runs.push_back(running("train-1"));
    RunEntry ended = running("ended-1");
    ended.state = "exited";
    f.runner.runs.push_back(ended);
    CHECK(err(*f.s, "run.fork", {{"run", "user"}}) == kRunNotManaged);
    CHECK(err(*f.s, "run.stop", {{"run", "user"}}) == kRunNotManaged);
    CHECK(err(*f.s, "run.logs", {{"run", "user"}}) == kRunNotManaged);
    CHECK(err(*f.s, "run.fork", {{"run", "nope"}}) == kRunNotFound);
    CHECK(err(*f.s, "run.fork", {{"run", "ended-1"}}) == kRunNotReady);
    CHECK(f.runner.forks.empty());

    const json fork =
        approved(*f.s, "run.fork", {{"run", "train-1"}, {"params", json::array({{{"name", "lr"}, {"value", 0.001}}})}});
    CHECK(fork["run"]["parent"] == "train-1");
    CHECK(fork["run"]["id"] == "train-2");
    REQUIRE(f.runner.forks.size() == 1);
    CHECK(f.runner.forks[0].params == json{{"lr", 0.001}});

    CHECK(ok(*f.s, "run.stop", {{"run", "train-1"}})["run"]["stopping"] == true);
    ok(*f.s, "run.stop", {{"run", "train-1"}, {"kill_after_s", 0}});
    CHECK(err(*f.s, "run.stop", {{"run", "train-1"}, {"kill_after_s", 601}}) == "command.invalid_args");
    REQUIRE(f.runner.stops.size() == 2);
    CHECK(f.runner.stops[0].second == 10.0);
    CHECK(f.runner.stops[1].second == 0.0);

    const json logs = ok(*f.s, "run.logs", {{"run", "train-1"}});
    CHECK(logs["lines"].size() == 2);
    ok(*f.s, "run.logs", {{"run", "train-1"}, {"lines", 200}});
    CHECK(err(*f.s, "run.logs", {{"run", "train-1"}, {"lines", 201}}) == "command.invalid_args");
    REQUIRE(f.runner.log_calls.size() == 2);
    CHECK(f.runner.log_calls[0].second == 50);
    CHECK(f.runner.log_calls[1].second == 200);
}

TEST_CASE("runner commands: the run commands of M1 act on the run that their argument run names") {
    Fixture f("runner_target");
    FakeControl attached;
    attached.run.attached = true;
    attached.run.script = "mine.py";
    f.s->set_run_control(&attached);
    f.runner.control.run.attached = true;
    f.runner.control.run.control = true;
    f.runner.control.run.script = "scripts/train.py";
    f.runner.control.run.state = "running";
    f.runner.control.run.step = 7;
    RunAction stop;
    stop.name = "caliper.stop";
    f.runner.control.run.actions = {stop};

    // Without run: the attached script, as in M1; the runner is not asked.
    CHECK(ok(*f.s, "run.state")["script"] == "mine.py");
    CHECK(f.runner.controls_asked.empty());
    // With run: the runner's control of that run.
    const json st = ok(*f.s, "run.state", {{"run", "train-1"}});
    CHECK(st["script"] == "scripts/train.py");
    CHECK(st["step"] == 7);
    ok(*f.s, "run.action", {{"name", "caliper.stop"}, {"run", "train-1"}});
    CHECK(f.runner.control.sent == std::vector<std::string>{"run caliper.stop"});
    CHECK(err(*f.s, "run.pause", {{"op", "pause"}, {"run", "other"}}) == kRunNotFound);
    CHECK(f.runner.controls_asked == std::vector<std::string>{"train-1", "train-1", "other"});
    CHECK(attached.sent.empty());
    f.s->set_run_control(nullptr);
}

TEST_CASE("runner commands: run ids are 1 to 40 characters of a-z, 0-9 and -") {
    CHECK(valid_run_id("train-1"));
    CHECK(valid_run_id(std::string(40, 'a')));
    CHECK_FALSE(valid_run_id(""));
    CHECK_FALSE(valid_run_id(std::string(41, 'a')));
    CHECK_FALSE(valid_run_id("Train"));
    CHECK_FALSE(valid_run_id("a_b"));
    CHECK_FALSE(valid_run_id("a/b"));
}

TEST_CASE("runner commands: a runner event keeps its run, parent, exit code and error in events.wait") {
    auto s = open_session(fresh_dir("runner_events") + "/workspace");
    Event e;
    e.kind = EventKind::run;
    e.origin = "script";
    e.fields = {{"change", "runner"}, {"run", "base-2"}, {"state", "failed"}, {"parent", "base"},
                {"exit_code", 1},      {"error", "The script ended with exit code 1."}};
    s->events().push(e);
    const json r = ok(*s, "events.wait", {{"cursor", 0}, {"kinds", {"run"}}, {"timeout_s", 0}});
    REQUIRE(r["events"].size() == 1);
    const json& got = r["events"][0];
    CHECK(got["change"] == "runner");
    CHECK(got["run"] == "base-2");
    CHECK(got["parent"] == "base");
    CHECK(got["exit_code"] == 1);
    CHECK(got["error"] == "The script ended with exit code 1.");
}

TEST_CASE("runner commands: run.start takes the script only as the path that the card shows and that runs") {
    Fixture f("runner_resolved");
    // Another path to the same file gets no card; the error names the path.
    for (const std::string& other : {f.ws + "/scripts/train.py", std::string("./scripts/train.py"),
                                     std::string("scripts/../scripts/train.py"), std::string("scripts//train.py")}) {
        const CommandResult r = send(*f.s, "run.start", {{"script", other}}, CommandOrigin::mcp, "");
        INFO(other << " answered " << r.error.code << ": " << r.error.message);
        CHECK(r.error.code == kRunScriptRefused);
        CHECK(r.error.message.find("scripts/train.py of the workspace") != std::string::npos);
    }
#ifndef _WIN32
    // A link in the workspace to the script is refused too, as the card
    // would name the link and not the file that runs.
    std::filesystem::create_symlink("scripts/train.py", f.ws + "/link.py");
    CHECK(send(*f.s, "run.start", {{"script", "link.py"}}, CommandOrigin::mcp, "").error.code == kRunScriptRefused);
    // A link made after the card was approved refuses the start.
    std::filesystem::create_directories(f.ws + "/real");
    write_text(f.ws + "/real/train.py", "print('real')\n");
    std::filesystem::create_directories(f.ws + "/swap");
    write_text(f.ws + "/swap/train.py", "print('swap')\n");
    const CommandResult r = send(*f.s, "run.start", {{"script", "swap/train.py"}}, CommandOrigin::mcp, "");
    REQUIRE(r.error.code == "command.needs_approval");
    std::filesystem::remove_all(f.ws + "/swap");
    std::filesystem::create_directory_symlink("real", f.ws + "/swap");
    const CommandResult done = f.s->registry().approve(r.approval);
    CHECK(done.error.code == kRunScriptRefused);
#endif
    CHECK(f.s->registry().pending_approvals().empty());
    CHECK(f.runner.starts.empty());
}

TEST_CASE("runner commands: a start past the limits gets no card, and the card shows each value in full") {
    Fixture f("runner_limits");
    f.runner.runs.push_back(running("train-1"));
    auto refused = [&f](const char* name, json args) {
        const CommandResult r = send(*f.s, name, std::move(args), CommandOrigin::mcp, "");
        INFO(name << " answered " << r.error.code << ": " << r.error.message);
        CHECK(r.error.code == "command.invalid_args");
    };
    std::string e200;   // 200 characters of 2 bytes: the schema's 256 characters pass, the 256 bytes do not
    for (int i = 0; i < 200; ++i) e200 += "\xC3\xA9";
    refused("run.start", {{"script", "scripts/train.py"}, {"args", {std::string(257, 'a')}}});
    refused("run.start", {{"script", "scripts/train.py"}, {"args", {e200}}});
    refused("run.start", {{"script", "scripts/train.py"}, {"args", json::array({std::string(256, 'a'), std::string(256, 'b'),
                                                                                 std::string(256, 'c'), std::string(256, 'd'),
                                                                                 "e"})}});
    refused("run.start", {{"script", "scripts/train.py"}, {"params", json::array({{{"name", "note"}, {"value", e200}}})}});
    json many = json::array();
    for (int i = 0; i < 5; ++i) many.push_back(json{{"name", "p" + std::to_string(i)}, {"value", std::string(250, 'x')}});
    refused("run.start", {{"script", "scripts/train.py"}, {"params", many}});
    refused("run.fork", {{"run", "train-1"}, {"params", many}});
    CHECK(f.s->registry().pending_approvals().empty());
    CHECK(f.runner.starts.empty());
    CHECK(f.runner.forks.empty());

    // At the limits the start gets its card, and runs once approved.
    const json full{{"script", "scripts/train.py"},
                    {"args", json::array({std::string(256, 'a'), std::string(256, 'b'), std::string(256, 'c'),
                                          std::string(256, 'd')})}};
    approved(*f.s, "run.start", full, CommandOrigin::mcp);
    REQUIRE(f.runner.starts.size() == 1);
    CHECK(f.runner.starts[0].args.size() == 4);

    // Quotes, backslashes, newlines and direction overrides cannot fake a
    // second argument, a parameter or the end of the card.
    CommandInvocation inv;
    inv.name = "run.start";
    inv.args = {{"script", "scripts/train.py"},
                {"args", {"a\" b", "x\ny. The script", "C:\\data", "--lr=0.1", "\xE2\x80\xAE" "cba"}},
                {"params", json::array({{{"name", "a=b"}, {"value", "1"}},
                                        {{"name", "note"}, {"value", "ok\" c=2"}},
                                        {{"name", "n"}, {"value", 2}}})}};
    CHECK(run_card_text(inv, "") ==
          "Start scripts/train.py as a new managed run, with the arguments \"a\\\" b\" \"x\\ny. The script\" "
          "\"C:\\\\data\" --lr=0.1 \"\\u202ecba\" and the parameters \"a=b\"=\"1\" note=\"ok\\\" c=2\" n=2. The "
          "script runs as Python code on this computer.");
    inv.args = {{"script", "my scripts/train.py"}, {"args", {std::string(256, 'z')}}};
    const std::string card = run_card_text(inv, "");
    CHECK(card.find("Start \"my scripts/train.py\" as") == 0);
    CHECK(card.find(std::string(256, 'z')) != std::string::npos);   // in full, not cut
    inv.name = "run.fork";
    inv.args = {{"run", "train-1"}};
    CHECK(run_card_text(inv, "a\nb.py") ==
          "Fork the run train-1 at its next checkpoint, with no changed parameters. The fork runs \"a\\nb.py\" as "
          "Python code beside its parent.");
}
