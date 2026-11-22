// Tests of the refusal of history.undo and history.redo to agents through
// the session and the stand-in's server: an agent reverts with an edit, which
// is reviewed; the person's undo takes back an agent's approved edit and the
// redo puts it back; and the descriptor table that the server serves to an
// agent's token leaves the two commands out.
#include <caliper_builder/session/command_server.h>
#include <caliper_builder/session/session.h>

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/journal.h>

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "helpers.h"

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

bool lists(const json& table, const std::string& name) {
    for (const auto& c : table["commands"])
        if (c["name"] == name) return true;
    return false;
}

}  // namespace

TEST_CASE("an agent cannot undo or redo, and the person's undo takes back an agent's approved edit") {
    const std::string ws = fresh_dir("history-refusal") + "/workspace";
    write_text(ws + "/data/metrics.csv", "step,loss\n0,1.0\n1,0.5\n");
    auto s = open_session(ws);
    ok(*s, "data.import", {{"paths", {"data/metrics.csv"}}});
    ok(*s, "spec.new", {{"id", "dash"}});
    ok(*s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
    ok(*s, "panel.create", {{"id", "loss"}, {"primitive", "series"}, {"bind", {{"y", "m.loss"}}}});
    // The agent's edit, approved by the person.
    const CommandResult edit =
        invoke(*s, "panel.patch", {{"id", "loss"}, {"patch", {{"title", "From the agent"}}}}, CommandOrigin::agent);
    REQUIRE(edit.error.code == "command.needs_approval");
    REQUIRE(s->registry().approve(edit.approval).ok);
    auto title = [&] { return s->open_spec()->spec.find_panel("loss")->title; };
    CHECK(title() == std::optional<std::string>("From the agent"));
    const std::size_t lines = read_journal(ws + "/journal/commands.jsonl").events.size();
    for (const char* name : {"history.undo", "history.redo"})
        for (CommandOrigin o : {CommandOrigin::agent, CommandOrigin::mcp}) {
            CAPTURE(name);
            CHECK(err(*s, name, json::object(), o) == "command.not_allowed");
        }
    CHECK(title() == std::optional<std::string>("From the agent"));
    CHECK(read_journal(ws + "/journal/commands.jsonl").events.size() == lines);
    CHECK(s->registry().pending_approvals().empty());
    CHECK(ok(*s, "history.list", json::object(), CommandOrigin::agent)["can_undo"] == true);
    // The person steps back over the agent's edit, and forward again.
    ok(*s, "history.undo");
    CHECK_FALSE(title());
    ok(*s, "history.redo");
    CHECK(title() == std::optional<std::string>("From the agent"));
}

#ifndef _WIN32
TEST_CASE("over the stand-in's server an agent's token neither sees nor runs history.undo and history.redo") {
    auto s = open_session(fresh_dir("history-refusal-http") + "/workspace");
    ok(*s, "spec.new", {{"id", "web"}});
    std::atomic<bool> stop{false};
    std::thread drainer([&] {
        while (!stop.load()) {
            s->queue().drain();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    for (CommandOrigin origin : {CommandOrigin::mcp, CommandOrigin::agent, CommandOrigin::api}) {
        CAPTURE(command_origin_name(origin));
        CommandServerOptions options;
        options.token = generate_session_token();
        options.origin = origin;
        CommandServer server(s->queue(), options);
        REQUIRE(server.start().ok());
        const bool agent = origin != CommandOrigin::api;
        const json table = http(server.port(), "GET", "/v0/commands", options.token).json_body();
        CHECK(lists(table, "history.undo") == !agent);
        CHECK(lists(table, "history.redo") == !agent);
        CHECK(lists(table, "history.list"));
        const HttpReply reply = http(server.port(), "POST", "/v0/commands/history.undo", options.token, "{}");
        INFO("HTTP " << reply.status << ": " << reply.body);
        const json undo = reply.json_body();
        REQUIRE(undo.is_object());
        if (agent) {
            CHECK(undo.at("error").at("code") == "command.not_allowed");
        } else {
            CHECK(undo.at("ok") == true);  // a program's token undoes the spec.new
            CHECK_FALSE(s->open_spec());
        }
        server.stop();
    }
    s->queue().shutdown();
    stop.store(true);
    drainer.join();
}
#endif
