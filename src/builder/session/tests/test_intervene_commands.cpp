// Tests of the intervention commands (intervene_commands.h,
// design/gpu-programs.md, Intervention Commands): the session binds all four;
// intervene.points and intervene.list give the script's points and
// interventions in the table's form; intervene.set and intervene.clear check
// the request, send it through the run control and say it is pending; the
// refusals carry their codes; and the effect run asks an agent with the rule
// key of the point.
#include "helpers.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/permissions.h>
#include <caliper_builder/session/intervene_commands.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

struct PlainRun final : RunControl {
    RunState state() override { return {}; }
    Error set_param(const std::string&, const nlohmann::json&) override { return {}; }
    Error run_action(const std::string&) override { return {}; }
    Error pause(RunPauseOp, std::int64_t) override { return {}; }
};

// A run control with interventions that records what it sends.
struct FakeRun final : RunControl {
    RunState run;
    bool supported = true;
    std::vector<InterventionPoint> pts;
    std::vector<Intervention> ivs;
    std::vector<Intervention> sets;
    std::vector<std::string> clears;
    Error answer;

    RunState state() override { return run; }
    Error set_param(const std::string&, const json&) override { return {}; }
    Error run_action(const std::string&) override { return {}; }
    Error pause(RunPauseOp, std::int64_t) override { return {}; }
    bool interventions_supported() override { return supported; }
    std::vector<InterventionPoint> intervention_points() override { return pts; }
    std::vector<Intervention> interventions() override { return ivs; }
    Error intervene_set(const Intervention& i) override {
        sets.push_back(i);
        return answer;
    }
    Error intervene_clear(const std::string& point) override {
        clears.push_back(point);
        return answer;
    }
};

InterventionPoint point(const std::string& module, std::optional<std::int64_t> head = std::nullopt) {
    InterventionPoint p;
    p.module = module;
    p.type = head ? "Attention" : "Linear";
    if (head) {
        p.point = module + "#" + std::to_string(*head);
        p.kind = "head";
        p.head = head;
    } else {
        p.point = module;
        p.shape = std::vector<std::int64_t>{8, 64, 128};
    }
    return p;
}

FakeRun training() {
    FakeRun f;
    f.run.attached = true;
    f.run.script = "train.py";
    f.run.control = true;
    f.run.state = "running";
    f.run.step = 1200;
    f.pts = {point("blocks.0.attn"), point("blocks.0.attn", 0), point("blocks.0.attn", 1), point("blocks.0.mlp"),
             point("head")};
    Intervention active;
    active.point = "blocks.0.attn#1";
    active.since_step = 1201;
    active.state = "active";
    Intervention ended;
    ended.point = "head";
    ended.from_step = 10;
    ended.until_step = 20;
    ended.since_step = 10;
    ended.state = "ended";
    f.ivs = {active, ended};
    return f;
}

}  // namespace

TEST_CASE("interventions: the session binds the four commands") {
    auto s = open_session(fresh_dir("intervene_bound") + "/workspace");
    for (const char* name : {"intervene.points", "intervene.set", "intervene.clear", "intervene.list"}) {
        const CommandDescriptor* d = find_command_descriptor(name);
        REQUIRE(d);
        CAPTURE(name);
        const auto r = invoke(*s, name, d->examples.front());
        CHECK((r.ok || r.error.code != "command.not_bound"));
        CHECK(r.error.code == "command.not_available");   // no run control
    }
}

TEST_CASE("interventions: a run control without them has no points and refuses a change") {
    PlainRun run;
    CHECK_FALSE(run.interventions_supported());
    CHECK(run.intervention_points().empty());
    CHECK(run.interventions().empty());
    Intervention i;
    i.point = "blocks.0.attn#3";
    CHECK(run.intervene_set(i).code == "command.not_available");
    CHECK(run.intervene_clear("blocks.0.attn#3").code == "command.not_available");
}

#ifndef _WIN32

TEST_CASE("interventions: intervene.points and intervene.list read the script's state in the table's form") {
    auto s = open_session(fresh_dir("intervene_read") + "/workspace");
    FakeRun fake = training();
    s->set_run_control(&fake);

    const json all = ok(*s, "intervene.points");
    CHECK(all["step"] == 1200);
    REQUIRE(all["points"].size() == 5);
    CHECK(all["points"][0] == json{{"point", "blocks.0.attn"},
                                   {"kind", "module"},
                                   {"module", "blocks.0.attn"},
                                   {"head", nullptr},
                                   {"type", "Linear"},
                                   {"shape", {8, 64, 128}}});
    CHECK(all["points"][2] == json{{"point", "blocks.0.attn#1"},
                                   {"kind", "head"},
                                   {"module", "blocks.0.attn"},
                                   {"head", 1},
                                   {"type", "Attention"},
                                   {"shape", nullptr}});
    const json some = ok(*s, "intervene.points", {{"match", "blocks.*#?"}});
    CHECK(some["points"].size() == 2);
    CHECK(ok(*s, "intervene.points", {{"match", "*.mlp"}})["points"].size() == 1);
    CHECK(ok(*s, "intervene.points", {{"match", "nothing*"}})["points"].empty());

    const json list = ok(*s, "intervene.list", json::object(), CommandOrigin::mcp);   // a read
    REQUIRE(list["interventions"].size() == 2);
    CHECK(list["interventions"][0] == json{{"point", "blocks.0.attn#1"},
                                           {"op", "ablate"},
                                           {"from_step", nullptr},
                                           {"until_step", nullptr},
                                           {"since_step", 1201},
                                           {"state", "active"}});
    CHECK(list["interventions"][1]["state"] == "ended");

    // Without a script, or a script without interventions.
    fake.run.attached = false;
    CHECK(err(*s, "intervene.points") == kRunNotAttached);
    fake.run.attached = true;
    fake.supported = false;
    CHECK(err(*s, "intervene.list") == kInterveneNotSupported);
    CHECK(err(*s, "intervene.set", {{"point", "head"}, {"op", "ablate"}}) == kInterveneNotSupported);
    CHECK(fake.sets.empty());
    s->set_run_control(nullptr);
}

TEST_CASE("interventions: intervene.set checks the op, the point and the range, and says it is pending") {
    auto s = open_session(fresh_dir("intervene_set") + "/workspace");
    FakeRun fake = training();
    s->set_run_control(&fake);

    const json r = ok(*s, "intervene.set", {{"point", "blocks.0.attn#0"}, {"op", "ablate"}});
    CHECK(r == json{{"point", "blocks.0.attn#0"}, {"op", "ablate"}, {"pending", true}});
    REQUIRE(fake.sets.size() == 1);
    CHECK(fake.sets[0].point == "blocks.0.attn#0");
    CHECK(fake.sets[0].op == "ablate");
    CHECK_FALSE(fake.sets[0].from_step);
    CHECK_FALSE(fake.sets[0].until_step);

    ok(*s, "intervene.set", {{"point", "head"}, {"op", "ablate"}, {"from_step", 1300}, {"until_step", 1400}});
    REQUIRE(fake.sets.size() == 2);
    CHECK(fake.sets[1].from_step == 1300);
    CHECK(fake.sets[1].until_step == 1400);

    for (const char* op : {"scale", "steer", "patch"})
        CHECK(err(*s, "intervene.set", {{"point", "head"}, {"op", op}}) == kInterveneOpReserved);
    CHECK(err(*s, "intervene.set", {{"point", "blocks.9"}, {"op", "ablate"}}) == kInterveneUnknownPoint);
    CHECK(err(*s, "intervene.set", {{"point", "head"}, {"op", "ablate"}, {"from_step", 1500}, {"until_step", 1500}}) ==
          kInterveneBadRange);
    CHECK(err(*s, "intervene.set", {{"point", "head"}, {"op", "ablate"}, {"until_step", 1201}}) == kInterveneBadRange);
    CHECK(fake.sets.size() == 2);

    // Before the script lists its points, the script decides.
    fake.pts.clear();
    ok(*s, "intervene.set", {{"point", "blocks.9"}, {"op", "ablate"}});
    CHECK(fake.sets.size() == 3);

    // A connection without control, and a refusal of the run control.
    fake.run.control = false;
    fake.run.why_not = "The hub does not forward controls.";
    const CommandResult no = invoke(*s, "intervene.set", {{"point", "head"}, {"op", "ablate"}});
    CHECK(no.error.code == kRunControlNotAllowed);
    CHECK(no.error.message == "The hub does not forward controls.");
    fake.run.control = true;
    fake.answer = Error(kRunRefused, "too many controls wait to be sent");
    CHECK(err(*s, "intervene.set", {{"point", "head"}, {"op", "ablate"}}) == kRunRefused);
    s->set_run_control(nullptr);
}

TEST_CASE("interventions: intervene.clear names the points it clears, one or all") {
    auto s = open_session(fresh_dir("intervene_clear") + "/workspace");
    FakeRun fake = training();
    s->set_run_control(&fake);

    CHECK(ok(*s, "intervene.clear", {{"point", "blocks.0.attn#1"}}) ==
          json{{"cleared", {"blocks.0.attn#1"}}, {"pending", true}});
    CHECK(ok(*s, "intervene.clear", {{"point", "blocks.0.mlp"}})["cleared"].empty());
    // an ended intervention is not cleared again
    CHECK(ok(*s, "intervene.clear", {{"all", true}})["cleared"] == json{"blocks.0.attn#1"});
    CHECK(fake.clears == std::vector<std::string>{"blocks.0.attn#1", "blocks.0.mlp", ""});

    CHECK(err(*s, "intervene.clear") == "command.invalid_args");
    CHECK(err(*s, "intervene.clear", {{"point", "head"}, {"all", true}}) == "command.invalid_args");
    CHECK(err(*s, "intervene.clear", {{"point", "blocks.9"}}) == kInterveneUnknownPoint);
    CHECK(fake.clears.size() == 3);
    s->set_run_control(nullptr);
}

TEST_CASE("interventions: the effect run asks an agent, with a rule key for one point") {
    auto s = open_session(fresh_dir("intervene_policy") + "/workspace");
    FakeRun fake = training();
    s->set_run_control(&fake);
    const json args{{"point", "blocks.0.attn#0"}, {"op", "ablate"}};
    CHECK(err(*s, "intervene.set", args, CommandOrigin::mcp) == "command.needs_approval");
    CHECK(err(*s, "intervene.clear", {{"all", true}}, CommandOrigin::agent) == "command.needs_approval");
    CHECK(fake.sets.empty());
    CHECK(fake.clears.empty());
    // the reads run for an agent
    ok(*s, "intervene.points", json::object(), CommandOrigin::agent);
    ok(*s, "intervene.list", json::object(), CommandOrigin::mcp);

    CommandInvocation inv;
    inv.name = "intervene.set";
    inv.args = args;
    CHECK(permission_rule_key(inv) == "intervene.set:blocks.0.attn#0");
    s->set_run_control(nullptr);
}

#endif

TEST_CASE("interpretation: every command binds with bounded schema and correct effects") {
    auto session = open_session(fresh_dir("interpret_bound") + "/workspace");
    for (const char* name : {"feature.dictionary", "feature.show", "feature.examples", "trace.run", "trace.status", "trace.read"}) {
        CAPTURE(name);
        const auto* descriptor = find_command_descriptor(name);
        REQUIRE(descriptor);
        CHECK(descriptor->effect == (std::string(name) == "trace.run" ? CommandEffect::run : CommandEffect::read));
        CHECK(descriptor->frame_thread);
        auto result = invoke(*session, name, descriptor->examples.front());
        CHECK_FALSE(result.ok);
        CHECK(result.error.code == "interpret.unsupported");
    }
    CHECK(err(*session, "trace.read", {{"node_limit", 257}}) == "command.invalid_args");
    CHECK(err(*session, "feature.examples", {{"layer", 0}, {"feature", 0}, {"limit", 17}}) == "command.invalid_args");
    CHECK(err(*session, "trace.run", {{"prompt", "x"}, {"max_nodes", true}}) == "command.invalid_args");
}
