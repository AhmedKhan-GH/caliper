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
    std::vector<json> interpretations;
    std::map<std::uint32_t, std::string> interpretation_methods;

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
    CommandOutput interpret(const std::string& method, const json& params, std::uint32_t request) override {
        if (!request) {
            interpretations.push_back(params);
            request = static_cast<std::uint32_t>(interpretations.size());
            interpretation_methods[request] = method;
        }
        const auto found = interpretation_methods.find(request);
        if (found == interpretation_methods.end() || found->second != method)
            return CommandOutput::fail(Error("interpret.unknown_request", "No matching request is retained."));
        return CommandOutput::ok({{"request_id", request}, {"pending", true}, {"result", nullptr}});
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

    for (const char* op : {"steer"})
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
    for (const char* name : {"feature.dictionary", "feature.show", "feature.examples", "trace.run", "trace.status", "trace.read", "patch.sweep", "trace.validate"}) {
        CAPTURE(name);
        const auto* descriptor = find_command_descriptor(name);
        REQUIRE(descriptor);
        CHECK(descriptor->effect == (std::string(name) == "trace.run" || std::string(name) == "patch.sweep" || std::string(name) == "trace.validate" ? CommandEffect::run : CommandEffect::read));
        CHECK(descriptor->frame_thread);
        auto result = invoke(*session, name, descriptor->examples.front());
        CHECK_FALSE(result.ok);
        CHECK(result.error.code == "interpret.unsupported");
    }
    CHECK(err(*session, "trace.read", {{"node_limit", 257}}) == "command.invalid_args");
    CHECK(err(*session, "feature.examples", {{"layer", 0}, {"feature", 0}, {"limit", 17}}) == "command.invalid_args");
    CHECK(err(*session, "trace.run", {{"prompt", "x"}, {"max_nodes", true}}) == "command.invalid_args");
}

TEST_CASE("interpretation: invalid prompt bytes stop before the run transport") {
    auto session = open_session(fresh_dir("interpret_bytes") + "/workspace");
    auto run = training();
    session->set_run_control(&run);
    std::string multibyte;
    for (unsigned i = 0; i < 2048; ++i) multibyte += "\xc3\xa9";
    CHECK(ok(*session, "trace.run", {{"prompt", multibyte}})["pending"] == true);
    REQUIRE(run.interpretations.size() == 1);
    for (const auto& prompt : {std::string("a\0b", 3), multibyte + "\xc3\xa9", std::string("\xc3\x28", 2)}) {
        const auto result = invoke(*session, "trace.run", {{"prompt", prompt}});
        CHECK_FALSE(result.ok);
        CHECK(result.error.code == "command.invalid_args");
        CHECK(result.error.message.size() <= 1024);
        CHECK(run.interpretations.size() == 1);
    }
    session->set_run_control(nullptr);
}

TEST_CASE("interventions: all supported kinds retain their fields and reject incompatible fields") {
    auto session = open_session(fresh_dir("intervene_m8") + "/workspace");
    auto run = training();
    run.pts.clear();
    session->set_run_control(&run);
    std::vector<json> specifications = {
        {{"point","layers.0.mlp"},{"kind","ablate"}},
        {{"point","layers.0.mlp_out"},{"kind","add"},{"vector",{1.,-1.}},{"coefficient",.5}},
        {{"point","layers.0.mlp_out"},{"kind","scale"},{"feature",0},{"multiplier",2.},{"positions",{0,1}},{"dictionary","fixture"}},
        {{"point","layers.0.mlp_out"},{"kind","clamp"},{"feature",0},{"value",0.}},
        {{"point","layers.0.mlp_out"},{"kind","patch"},{"source_prompt","AB"},{"source_position",1}}};
    for (const auto& spec : specifications) {
        ok(*session,"intervene.set",spec);
        CHECK(run.sets.back().specification == spec);
        const auto sent = intervention_set_spec(run.sets.back());
        CHECK(sent["kind"] == spec["kind"]);
        CHECK(sent["op"] == spec["kind"]);
        CHECK_FALSE(check_intervention_spec(sent,"args"));
    }
    auto invalid = specifications[1];
    invalid["positions"] = {0,0};
    CHECK(err(*session,"intervene.set",invalid) == "command.invalid_args");
    invalid = specifications[3]; invalid["value"] = true;
    CHECK(err(*session,"intervene.set",invalid) == "command.invalid_args");
    invalid = specifications[4]; invalid["source_prompt"] = std::string(4097,'x');
    CHECK(err(*session,"intervene.set",invalid) == "command.invalid_args");
    invalid = specifications[1]; invalid["op"] = "scale";
    CHECK(err(*session,"intervene.set",invalid) == "command.invalid_args");
    CHECK(run.sets.size() == 5);
    CHECK(err(*session,"patch.sweep",{{"clean_prompt","A"}},CommandOrigin::ui) == "command.invalid_args");
    CHECK(err(*session,"patch.sweep",{{"request_id",1}},CommandOrigin::mcp) == "interpret.unknown_request");
    CHECK(err(*session,"trace.validate",json::object(),CommandOrigin::agent) == "command.needs_approval");
    session->set_run_control(nullptr);
}

TEST_CASE("interventions: command handlers refuse null discriminator fields") {
    auto run = training();
    run.pts.clear();
    const auto* descriptor = find_command_descriptor("intervene.set");
    REQUIRE(descriptor);
    for (const bool null_kind : {false, true}) {
        CommandInvocation request;
        request.name = "intervene.set";
        request.args = {{"point", "layers.0.mlp_out"}, {"feature", 0}, {"value", 0.},
                        {"kind", null_kind ? json() : json("clamp")},
                        {"op", null_kind ? json("clamp") : json()}};
        const auto result = intervene_command(CommandContext{request, *descriptor}, &run);
        CHECK(result.error.code == "command.invalid_args");
        CHECK(run.sets.empty());
    }
}

TEST_CASE("interpretation: one approval admits a request and polling cannot submit work") {
    auto session = open_session(fresh_dir("interpret_poll") + "/workspace");
    auto run = training();
    session->set_run_control(&run);
    const auto requested = invoke(*session, "trace.run", {{"prompt", "fixture"}}, CommandOrigin::mcp);
    REQUIRE(requested.error.code == "command.needs_approval");
    CHECK(run.interpretations.empty());
    const auto approved = session->registry().approve(requested.approval);
    REQUIRE(approved.ok);
    REQUIRE(run.interpretations.size() == 1);
    const auto id = approved.result["request_id"];
    CHECK(ok(*session, "trace.run", {{"request_id", id}}, CommandOrigin::mcp)["pending"] == true);
    CHECK(ok(*session, "trace.run", {{"request_id", id}}, CommandOrigin::agent)["pending"] == true);
    CHECK(session->registry().pending_approvals().empty());
    CHECK(run.interpretations.size() == 1);
    CHECK(err(*session, "trace.run", {{"request_id", 99}}, CommandOrigin::mcp) == "interpret.unknown_request");
    CHECK(err(*session, "trace.validate", {{"request_id", id}}, CommandOrigin::mcp) == "interpret.unknown_request");
    const auto mixed = invoke(*session, "trace.run", {{"request_id", id}, {"prompt", "new"}}, CommandOrigin::mcp);
    REQUIRE(mixed.error.code == "command.needs_approval");
    CHECK(session->registry().approve(mixed.approval).error.code == "command.invalid_args");
    CHECK(err(*session, "trace.run", {{"request_id", 0}, {"prompt", "new"}}, CommandOrigin::mcp) == "command.invalid_args");
    CHECK(err(*session, "trace.run", {{"prompt", "new"}}, CommandOrigin::mcp) == "command.needs_approval");
    CHECK(run.interpretations.size() == 1);
    auto other = training();
    session->set_run_control(&other);
    CHECK(err(*session, "trace.run", {{"request_id", id}}, CommandOrigin::mcp) == "interpret.unknown_request");
    CHECK(other.interpretations.empty());
    REQUIRE(session->set_agent_mode(AgentPrincipal::mcp, AgentMode::read_only).ok());
    CHECK(err(*session, "trace.run", {{"request_id", id}}, CommandOrigin::mcp) == "command.denied");
    session->set_run_control(nullptr);
}
