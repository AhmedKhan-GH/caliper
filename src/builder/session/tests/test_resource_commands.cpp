// Tests of program.read (resource_commands.h,
// design/programmable-graphics.md, Program Commands). Until the handler is
// built, the command is bound and answers program.not_built.
#include "helpers.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

using namespace caliper_builder;
using namespace session_test;

TEST_CASE("resources: program.read is bound and answers program.not_built") {
    auto s = open_session(fresh_dir("resource_commands_bound") + "/workspace");
    const CommandDescriptor* d = find_command_descriptor("program.read");
    REQUIRE(d);
    CHECK(!d->mutates);
    const auto r = invoke(*s, "program.read", d->examples.front());
    CHECK(!r.ok);
    CHECK(r.error.code == "program.not_built");
}
