// Tests of the control commands of spec version 1 (design/instruments.md,
// Control Commands). The M1 track "controls" owns this file; until it fills
// control_edit.cpp, the commands are bound and answer command.not_implemented.
#include "helpers.h"

#include <caliper_builder/session/session.h>
#include <caliper_builder/session/spec_edit.h>

#include <doctest/doctest.h>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

TEST_CASE("the control commands are edits of the open spec, bound and not built yet") {
    for (const char* name : {"control.create", "control.patch", "control.remove"}) CHECK(is_spec_edit_command(name));
    auto s = open_session(fresh_dir("control_edit") + "/workspace");
    ok(*s, "spec.new", {{"id", "m1"}});
    CHECK(err(*s, "control.create", {{"id", "lr"}, {"kind", "slider"}, {"bind", "param:lr"}}) ==
          "command.not_implemented");
    CHECK(err(*s, "control.patch", {{"id", "lr"}, {"patch", {{"title", "Learning rate"}}}}) ==
          "command.not_implemented");
    CHECK(err(*s, "control.remove", {{"id", "lr"}}) == "command.not_implemented");
}
