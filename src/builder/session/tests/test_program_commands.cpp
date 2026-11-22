// Tests of the stage and program commands (program_commands.h,
// design/gpu-programs.md, Commands): the session binds all seven, the status
// item has the keys of the command table's program_status, and a program
// event keeps the keys of its kind.
#include "helpers.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/program_commands.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

#include <set>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

TEST_CASE("programs: the session binds the stage and program commands and has no hosts at first") {
    auto s = open_session(fresh_dir("program_commands_bound") + "/workspace");
    CHECK(s->program_host() == nullptr);
    CHECK(s->stage_host() == nullptr);
    for (const char* name : {"stage.set", "stage.remove", "stage.check", "program.create", "program.patch",
                             "program.remove", "program.status"}) {
        const CommandDescriptor* d = find_command_descriptor(name);
        REQUIRE(d);
        CAPTURE(name);
        const auto r = invoke(*s, name, d->examples.front());
        CHECK((r.ok || r.error.code != "command.not_bound"));
        CHECK(is_spec_edit_command(name) == d->mutates);
    }
}

TEST_CASE("programs: a status item has exactly the keys of the table's program_status") {
    const CommandDescriptor* d = find_command_descriptor("program.status");
    REQUIRE(d);
    const json& item = d->result["properties"]["programs"]["items"];
    std::set<std::string> want;
    for (const auto& k : item["required"]) want.insert(k.get<std::string>());
    ProgramStatus st;
    st.id = "rows";
    st.kind = "geometry";
    st.error = ProgramError{"program.over_time", "The program went over its time limit on 3 passes in a row.", 0, 0};
    const json j = program_status_json(st);
    std::set<std::string> have;
    for (const auto& kv : j.items()) have.insert(kv.key());
    CHECK(have == want);
    CHECK(j["error"]["code"] == "program.over_time");
}

TEST_CASE("programs: a program event carries its id, change, host and error") {
    EventRing ring;
    ProgramStatus st;
    st.id = "rows";
    st.host = "script";
    st.panel = "world";
    st.error = ProgramError{"program.writes_input", "The input w is not var<storage, read>.", 2, 1};
    push_program_event(ring, st, "refused");
    const auto read = ring.read(0, {}, 8);
    REQUIRE(read.events.size() == 1);
    const json e = event_json(read.events.front());
    CHECK(e["kind"] == "program");
    CHECK(e["panel"] == "world");
    CHECK(e["program"] == "rows");
    CHECK(e["change"] == "refused");
    CHECK(e["host"] == "script");
    CHECK(e["code"] == "program.writes_input");
    CHECK(e["line"] == 2);
}
