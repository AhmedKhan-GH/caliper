// Tests of the intervention commands (intervene_commands.h,
// design/gpu-programs.md, Intervention Commands): the session binds all four,
// and a run control without interventions keeps the defaults of
// run_control.h.
#include "helpers.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/intervene_commands.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

using namespace caliper_builder;
using namespace session_test;

namespace {

struct PlainRun final : RunControl {
    RunState state() override { return {}; }
    Error set_param(const std::string&, const nlohmann::json&) override { return {}; }
    Error run_action(const std::string&) override { return {}; }
    Error pause(RunPauseOp, std::int64_t) override { return {}; }
};

}  // namespace

TEST_CASE("interventions: the session binds the four commands") {
    auto s = open_session(fresh_dir("intervene_bound") + "/workspace");
    for (const char* name : {"intervene.points", "intervene.set", "intervene.clear", "intervene.list"}) {
        const CommandDescriptor* d = find_command_descriptor(name);
        REQUIRE(d);
        CAPTURE(name);
        const auto r = invoke(*s, name, d->examples.front());
        CHECK((r.ok || r.error.code != "command.not_bound"));
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
