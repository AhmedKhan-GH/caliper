// Tests of the run commands (run_control.h, design/instruments.md, Run
// Commands): run.state, run.set_param, run.action and run.pause over a fake
// RunControl, with the error codes of run_control.h, the results checked
// against the command table's schemas, and the effect `run`, which asks an
// agent and runs for the person.
#include "helpers.h"

#include <caliper_builder/session/run_control.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

// A run control that records each request and answers from a state the test
// sets.
struct FakeRun final : RunControl {
    RunState run;
    Error answer;   // what each request answers
    struct Sent {
        std::string op, name;
        json value;
        std::int64_t count = 0;
    };
    std::vector<Sent> sent;

    RunState state() override { return run; }
    Error set_param(const std::string& name, const json& value) override {
        sent.push_back({"set", name, value, 0});
        return answer;
    }
    Error run_action(const std::string& name) override {
        sent.push_back({"run", name, json(), 0});
        return answer;
    }
    Error pause(RunPauseOp op, std::int64_t count) override {
        const char* n = op == RunPauseOp::pause ? "pause" : op == RunPauseOp::resume ? "resume" : "step";
        sent.push_back({n, "", json(), count});
        return answer;
    }
};

// A script that trains, at step 12, with a learning rate, a batch size, a
// switch and a choice, and one action.
RunState training() {
    RunState s;
    s.attached = true;
    s.script = "train.py";
    s.control = true;
    s.state = "running";
    s.step = 12;
    RunParam lr;
    lr.name = "lr";
    lr.type = "number";
    lr.value = 0.01;
    lr.min = 1e-5;
    lr.max = 0.1;
    lr.log = true;
    lr.help = "learning rate";
    RunParam batch;
    batch.name = "batch";
    batch.type = "integer";
    batch.value = 32;
    batch.pending = 64;
    RunParam aug;
    aug.name = "augment";
    aug.type = "boolean";
    aug.value = false;
    RunParam opt;
    opt.name = "optimizer";
    opt.type = "choice";
    opt.value = "adam";
    opt.choices = {"adam", "sgd"};
    s.params = {lr, batch, aug, opt};
    RunAction save;
    save.name = "save_checkpoint";
    save.help = "writes a checkpoint";
    save.queued = true;
    s.actions = {save};
    return s;
}

}  // namespace

TEST_CASE("run commands: a session without a run control answers command.not_available") {
    auto s = open_session(fresh_dir("run_commands_none") + "/workspace");
    CHECK(s->run_control() == nullptr);
    CHECK(err(*s, "run.state") == "command.not_available");
    CHECK(err(*s, "run.set_param", {{"name", "lr"}, {"value", 0.001}}) == "command.not_available");
    CHECK(err(*s, "run.action", {{"name", "save_checkpoint"}}) == "command.not_available");
    CHECK(err(*s, "run.pause", {{"op", "pause"}}) == "command.not_available");
}

TEST_CASE("run commands: run.state gives the script, its parameters and its actions in the table's form") {
    auto s = open_session(fresh_dir("run_commands_state") + "/workspace");
    FakeRun fake;
    s->set_run_control(&fake);
    CHECK(s->run_control() == &fake);

    // Nothing attached: the state says so, and every request is refused.
    json st = ok(*s, "run.state");
    CHECK(st["attached"] == false);
    CHECK_FALSE(st.contains("script"));
    CHECK(st["state"] == "none");
    CHECK(st["step"].is_null());
    CHECK(err(*s, "run.set_param", {{"name", "lr"}, {"value", 0.001}}) == kRunNotAttached);
    CHECK(err(*s, "run.action", {{"name", "save_checkpoint"}}) == kRunNotAttached);
    CHECK(err(*s, "run.pause", {{"op", "pause"}}) == kRunNotAttached);

    fake.run = training();
    st = ok(*s, "run.state");
    CHECK(st["attached"] == true);
    CHECK(st["script"] == "train.py");
    CHECK(st["control"] == true);
    CHECK_FALSE(st.contains("why_not"));
    CHECK(st["state"] == "running");
    CHECK(st["step"] == 12);
    CHECK(st["steps_left"].is_null());
    REQUIRE(st["params"].size() == 4);
    CHECK(st["params"][0] == json{{"name", "lr"}, {"type", "number"}, {"value", 0.01}, {"pending", nullptr},
                                  {"min", 1e-5}, {"max", 0.1}, {"log", true}, {"help", "learning rate"}});
    CHECK(st["params"][1]["pending"] == 64);
    CHECK_FALSE(st["params"][1].contains("min"));
    CHECK(st["params"][3]["choices"] == json{"adam", "sgd"});
    CHECK(st["actions"] == json::array({{{"name", "save_checkpoint"}, {"help", "writes a checkpoint"},
                                         {"running", false}, {"queued", true}}}));
    CHECK(fake.sent.empty());
    s->set_run_control(nullptr);
    CHECK(err(*s, "run.state") == "command.not_available");
}

TEST_CASE("run commands: a connection without control refuses with its reason") {
    auto s = open_session(fresh_dir("run_commands_access") + "/workspace");
    FakeRun fake;
    fake.run = training();
    fake.run.control = false;
    fake.run.why_not = "The script started its server with control=False.";
    s->set_run_control(&fake);
    CHECK(ok(*s, "run.state")["why_not"] == fake.run.why_not);
    const CommandResult r = invoke(*s, "run.set_param", {{"name", "lr"}, {"value", 0.001}});
    CHECK(r.error.code == kRunControlNotAllowed);
    CHECK(r.error.message == fake.run.why_not);
    CHECK(err(*s, "run.action", {{"name", "save_checkpoint"}}) == kRunControlNotAllowed);
    CHECK(err(*s, "run.pause", {{"op", "resume"}}) == kRunControlNotAllowed);
    CHECK(fake.sent.empty());
}

TEST_CASE("run commands: run.set_param checks the name and the type, sends the value and says it is pending") {
    auto s = open_session(fresh_dir("run_commands_set") + "/workspace");
    FakeRun fake;
    fake.run = training();
    s->set_run_control(&fake);

    CHECK(err(*s, "run.set_param", {{"name", "momentum"}, {"value", 0.9}}) == kRunUnknownParam);
    CHECK(err(*s, "run.set_param", {{"name", "lr"}, {"value", "fast"}}) == kRunBadValue);
    CHECK(err(*s, "run.set_param", {{"name", "lr"}, {"value", true}}) == kRunBadValue);
    CHECK(err(*s, "run.set_param", {{"name", "batch"}, {"value", 0.5}}) == kRunBadValue);
    CHECK(err(*s, "run.set_param", {{"name", "augment"}, {"value", 1}}) == kRunBadValue);
    CHECK(err(*s, "run.set_param", {{"name", "optimizer"}, {"value", "lbfgs"}}) == kRunBadValue);
    CHECK(fake.sent.empty());

    // The range is the script's to clamp: a value above it is sent as it is.
    CHECK(ok(*s, "run.set_param", {{"name", "lr"}, {"value", 0.5}}) ==
          json{{"name", "lr"}, {"value", 0.5}, {"pending", true}});
    // An integer parameter gets an integer.
    CHECK(ok(*s, "run.set_param", {{"name", "batch"}, {"value", 128.0}})["value"] == 128);
    ok(*s, "run.set_param", {{"name", "augment"}, {"value", true}});
    ok(*s, "run.set_param", {{"name", "optimizer"}, {"value", "sgd"}});
    REQUIRE(fake.sent.size() == 4);
    CHECK(fake.sent[0].op == "set");
    CHECK(fake.sent[0].name == "lr");
    CHECK(fake.sent[0].value == 0.5);
    CHECK(fake.sent[1].value.is_number_integer());
    CHECK(fake.sent[3].value == "sgd");

    // What the run control answers, the command answers.
    fake.answer = Error(kRunRefused, "The hub refused the request.");
    CHECK(err(*s, "run.set_param", {{"name", "lr"}, {"value", 0.02}}) == kRunRefused);
}

TEST_CASE("run commands: run.action queues a known action, and run.pause needs a first step") {
    auto s = open_session(fresh_dir("run_commands_action") + "/workspace");
    FakeRun fake;
    fake.run = training();
    s->set_run_control(&fake);

    CHECK(err(*s, "run.action", {{"name", "evaluate"}}) == kRunUnknownAction);
    CHECK(ok(*s, "run.action", {{"name", "save_checkpoint"}}) == json{{"name", "save_checkpoint"}, {"queued", true}});
    CHECK(ok(*s, "run.pause", {{"op", "pause"}}) == json{{"op", "pause"}, {"state", "running"}});
    CHECK(ok(*s, "run.pause", {{"op", "step"}, {"count", 5}})["op"] == "step");
    CHECK(ok(*s, "run.pause", {{"op", "step"}})["op"] == "step");
    REQUIRE(fake.sent.size() == 4);
    CHECK(fake.sent[0].op == "run");
    CHECK(fake.sent[0].name == "save_checkpoint");
    CHECK(fake.sent[1].op == "pause");
    CHECK(fake.sent[2].op == "step");
    CHECK(fake.sent[2].count == 5);
    CHECK(fake.sent[3].count == 1);

    // Before the script's first step, a pause or a step is refused; a resume
    // is sent.
    fake.run.state = "none";
    fake.run.step.reset();
    CHECK(err(*s, "run.pause", {{"op", "pause"}}) == kRunNoStep);
    CHECK(err(*s, "run.pause", {{"op", "step"}, {"count", 2}}) == kRunNoStep);
    CHECK(ok(*s, "run.pause", {{"op", "resume"}})["state"] == "none");
    CHECK(fake.sent.size() == 5);
}

TEST_CASE("run commands: the effect run asks an agent and an mcp client, and runs for the person") {
    auto s = open_session(fresh_dir("run_commands_policy") + "/workspace");
    FakeRun fake;
    fake.run = training();
    s->set_run_control(&fake);
    const json args{{"name", "lr"}, {"value", 0.02}};
    CHECK(err(*s, "run.set_param", args, CommandOrigin::mcp) == "command.needs_approval");
    CHECK(err(*s, "run.action", {{"name", "save_checkpoint"}}, CommandOrigin::agent) == "command.needs_approval");
    CHECK(err(*s, "run.pause", {{"op", "pause"}}, CommandOrigin::mcp) == "command.needs_approval");
    CHECK(fake.sent.empty());
    // run.state is a read.
    CHECK(ok(*s, "run.state", json::object(), CommandOrigin::mcp)["attached"] == true);
    for (CommandOrigin o : {CommandOrigin::ui, CommandOrigin::builder, CommandOrigin::api}) ok(*s, "run.set_param", args, o);
    CHECK(fake.sent.size() == 3);
}

TEST_CASE("run commands: free-text parameters preserve 4096 UTF-8 bytes") {
    auto s = open_session(fresh_dir("run_prompt_bytes") + "/workspace");
    FakeRun fake;
    fake.run = training();
    RunParam prompt;
    prompt.name = "prompt"; prompt.type = "text"; prompt.value = "";
    fake.run.params.push_back(prompt);
    s->set_run_control(&fake);
    std::string text;
    for (unsigned i = 0; i < 2048; ++i) text += "\xc3\xa9";
    REQUIRE(text.size() == 4096);
    const auto reply = ok(*s, "run.set_param", {{"name", "prompt"}, {"value", text}});
    CHECK(reply["value"] == text);
    REQUIRE(fake.sent.size() == 1);
    CHECK(fake.sent.back().value == text);
    for (const auto& value : {text + "a", std::string("a\0b", 3), std::string("\xc3\x28", 2)})
        CHECK(err(*s, "run.set_param", {{"name", "prompt"}, {"value", value}}) == kRunBadValue);
    CHECK(fake.sent.size() == 1);
}
