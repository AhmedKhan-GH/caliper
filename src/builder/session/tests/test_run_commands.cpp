// Tests of the run commands (run_control.h, design/instruments.md, Run
// Commands). The M1 track "controls" owns this file; until it fills
// run_commands.cpp, the commands are bound and answer command.not_implemented.
#include "helpers.h"

#include <caliper_builder/session/run_control.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

TEST_CASE("the run commands are bound and not built yet, and the session holds no run control at first") {
    auto s = open_session(fresh_dir("run_commands") + "/workspace");
    CHECK(s->run_control() == nullptr);
    CHECK(err(*s, "run.state") == "command.not_implemented");
    CHECK(err(*s, "run.set_param", {{"name", "lr"}, {"value", 0.001}}) == "command.not_implemented");
    CHECK(err(*s, "run.action", {{"name", "save_checkpoint"}}) == "command.not_implemented");
    CHECK(err(*s, "run.pause", {{"op", "pause"}}) == "command.not_implemented");
}
