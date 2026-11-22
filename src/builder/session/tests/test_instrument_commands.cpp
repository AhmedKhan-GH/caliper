// The commands of design/instrument-interface.md reach their handlers in a
// session: each one is in the table, bound, and answers with its own result
// or with command.not_implemented until it is built, never as an unbound or
// unknown command. camera.get, camera.set and camera.save need the app.
#include <doctest/doctest.h>

#include <algorithm>
#include <string>
#include <vector>

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/session.h>

#include "helpers.h"

using namespace caliper_builder;
using namespace session_test;

namespace {

const std::vector<std::string>& instrument_commands() {
    static const std::vector<std::string> names = {
        "state.create",  "state.patch",   "state.remove",   "state.set",         "state.get",  "control.set",
        "selection.set", "selection.get", "entity.set",     "entity.remove",     "annotation.set",
        "annotation.remove", "cursor.get", "cursor.set",    "cursor.step",       "camera.get", "camera.set",
        "camera.save"};
    return names;
}

}  // namespace

TEST_CASE("the eighteen instrument commands are in the table and reach their handlers") {
    const std::string ws = fresh_dir("instrument-commands");
    auto s = open_session(ws);
    ok(*s, "spec.new", {{"id", "bench"}});
    for (const std::string& name : instrument_commands()) {
        CAPTURE(name);
        const auto& table = command_table();
        const auto d = std::find_if(table.begin(), table.end(), [&](const auto& c) { return c.name == name; });
        REQUIRE(d != table.end());
        REQUIRE_FALSE(d->examples.empty());
        const bool app = std::find(session_host_commands().begin(), session_host_commands().end(), name) !=
                         session_host_commands().end();
        CHECK(s->registry().is_bound(name) == !app);
        // camera.save asks the app for its orbit first, so without the app it is unbound in effect.
        if (app || name == "camera.save") continue;
        const CommandResult r = invoke(*s, name, d->examples.front());
        INFO(r.error.code << ": " << r.error.message);
        CHECK(r.error.code != "command.not_bound");
        CHECK(r.error.code != "command.unknown");
        CHECK(r.error.code != "command.invalid_args");
    }
}
