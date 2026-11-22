// Tests of the piece commands (piece_commands.h,
// design/programmable-graphics.md, Piece Commands). The descriptors and their
// schemas are the interface; until the handlers are built, each command is
// bound and answers program.not_built.
#include "helpers.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

using namespace caliper_builder;
using namespace session_test;

TEST_CASE("pieces: the piece commands are bound and answer program.not_built") {
    auto s = open_session(fresh_dir("piece_commands_bound") + "/workspace");
    for (const char* name : {"piece.check", "piece.list", "piece.show", "piece.save", "piece.copy", "piece.update",
                             "piece.remove"}) {
        const CommandDescriptor* d = find_command_descriptor(name);
        REQUIRE(d);
        CAPTURE(name);
        const auto r = invoke(*s, name, d->examples.front());
        CHECK(!r.ok);
        CHECK(r.error.code == "program.not_built");
    }
}
