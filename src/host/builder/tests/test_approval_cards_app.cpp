// The approval cards in the real app (the hidden-app pattern of
// docs/explanations/testing.md): `caliper --hidden --api --api-port 0
// --workspace <dir>` runs as a child, with a command script that imports a
// small table, builds a spec of one panel, saves it and opens it, which shows
// the spec page. The real caliper-mcp then sends panel.create through an MCP
// tool call. The app reviews an mcp edit of the open spec, so the call is
// held and a card waits; the test approves it through CALIPER_TEST_APPROVALS,
// the decisions file that only a hidden app's environment enables, and the
// tool call returns ok with the panel in the open spec. A second call, a
// panel.patch, is denied the same way and answers command.denied. POSIX
// only; without a Metal device the case says it is skipped.
#include <doctest/doctest.h>

#if !defined(_WIN32) && defined(CALIPER_MCP_EXE)

#include "mcp_child.h"
#include "spec_test_workspace.h"
#include "test_proc.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;
using caliper_builder::mcp::test::McpChild;
using caliper_builder::mcp::test::modern_meta;

namespace {

// Sends a tools/call without waiting for its answer.
void send_call(McpChild& child, int id, const std::string& tool, const json& args) {
    child.send({{"jsonrpc", "2.0"},
                {"id", id},
                {"method", "tools/call"},
                {"params", {{"name", tool}, {"arguments", args}, {"_meta", modern_meta()}}}});
}

void decide(const std::string& path, const std::string& line) {
    std::ofstream out(path, std::ios::app);
    out << line << "\n";
}

}  // namespace

TEST_SUITE("app") {
TEST_CASE("approval cards: an outside agent's panel.create over MCP waits in the hidden app, is approved and lands") {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        MESSAGE("CALIPER_TEST_CALIPER_EXE is not set: run the ctest entry caliper_approval_cards_app");
        return;
    }
    const std::string root = spec_test::fresh_dir("cards-app");
    const std::string ws = root + "/ws", data = root + "/data";
    spec_test::write_parquet(root + "/inputs/temps.parquet",
                             "SELECT range AS step, 20 + 3 * sin(range / 20.0) AS temp FROM range(300)");
    const std::vector<json> script = {
        {{"name", "data.import"}, {"args", {{"paths", {root + "/inputs/temps.parquet"}}, {"run", true}}}},
        {{"name", "spec.new"}, {"args", {{"id", "temps-board"}, {"title", "Temperatures"}}}},
        {{"name", "source.add"}, {"args", {{"id", "t"}, {"catalog", "table:temps"}}}},
        {{"name", "panel.create"},
         {"args", {{"id", "curve"}, {"primitive", "series"}, {"title", "Temperature"},
                   {"bind", {{"x", "t.step"}, {"y", {"t.temp"}}}}}}},
        {{"name", "applet.save"}},
        {{"name", "applet.open"}, {"args", {{"id", "temps-board"}}}},
    };
    {
        std::ofstream out(root + "/build.jsonl");
        for (const json& j : script) out << j.dump() << "\n";
    }
    const std::string decisions = root + "/decisions.txt";
    { std::ofstream touch(decisions); }
    fs::create_directories(root + "/user-applets");

    Proc p;
    REQUIRE(p.start(exe, {"--hidden", "--api", "--api-port", "0", "--workspace", ws},
                    {"CALIPER_DATA_DIR=" + data, "CALIPER_USER_APPLETS_DIR=" + root + "/user-applets",
                     "CALIPER_WINDOW_SIZE=800x500", "CALIPER_AGENT_OFFLINE=1",
                     "CALIPER_COMMAND_SCRIPT=" + root + "/build.jsonl",
                     "CALIPER_COMMAND_RESULTS=" + root + "/build-results.jsonl",
                     "CALIPER_TEST_APPROVALS=" + decisions}));
    REQUIRE_MESSAGE(p.read_until("[api] Caliper API on ", 60000), p.text);
    if (p.text.find("[renderer] metal") == std::string::npos) {
        const char* require = std::getenv("CALIPER_REQUIRE_GPU");
        REQUIRE_MESSAGE(!(require && std::string(require) == "1"), p.text);
        MESSAGE("caliper --hidden --api: run skipped: no GPU (no Metal device)");
        ::kill(p.pid, SIGTERM);
        p.wait(30000);
        return;
    }
    REQUIRE_MESSAGE(p.read_until("[session] script done: 6 commands, 0 failed", 60000), p.text);

    {
        McpChild child({"--api-file", data + "/api.json"}, {{"CALIPER_AGENT_OFFLINE", "1"}});
        // panel.create from mcp: the app reviews it, so the call is held and
        // a card waits.
        send_call(child, 2, "panel_create",
                  {{"id", "rows"}, {"primitive", "table"}, {"title", "Rows"}, {"bind", {{"rows", "t"}}}});
        REQUIRE_MESSAGE(p.read_until("panel.create (mcp) waiting", 30000), p.text);
        CHECK(child.next(500).is_null());   // still held
        // The open spec does not have the panel yet.
        const json before = child.call(3, "spec_get", json::object());
        INFO(before.dump());
        CHECK(before.at("structuredContent").at("json").at("panel").size() == 1);

        // The person approves; the held call returns with the command's result.
        decide(decisions, "approve panel.create mcp");
        const json created = child.await(2, 60000);
        INFO(created.dump());
        REQUIRE(created.contains("result"));
        CHECK(created["result"]["isError"] == false);
        CHECK(created["result"]["structuredContent"]["id"] == "rows");
        CHECK(created["result"]["structuredContent"]["applied"] == true);
        CHECK(p.read_until("panel.create (mcp) approved", 10000));
        const json after = child.call(4, "spec_get", json::object());
        bool found = false;
        for (const json& panel : after.at("structuredContent").at("json").at("panel"))
            found = found || panel.at("id") == "rows";
        CHECK(found);

        // A panel.patch, denied.
        send_call(child, 5, "panel_patch", {{"id", "curve"}, {"patch", {{"title", "Temperature in C"}}}});
        REQUIRE_MESSAGE(p.read_until("panel.patch (mcp) waiting", 30000), p.text);
        decide(decisions, "deny panel.patch mcp");
        const json denied = child.await(5, 60000);
        INFO(denied.dump());
        REQUIRE(denied.contains("result"));
        CHECK(denied["result"]["isError"] == true);
        CHECK(denied["result"]["content"][0]["text"].get<std::string>().find("command.denied") != std::string::npos);
        CHECK(p.read_until("panel.patch (mcp) denied", 10000));
        CHECK(child.finish().success());
    }

    ::kill(p.pid, SIGTERM);
    CHECK_MESSAGE(p.wait(30000) == 0, p.text);
    CHECK(p.text.find("[session] command panel.create (mcp)") != std::string::npos);
    CHECK(p.text.find("[session] command panel.patch (mcp): command.denied") != std::string::npos);
}
}

#endif
