// The acceptance test of the MCP route, headless. caliper-builder-host, the
// stand-in for the app's command endpoints, serves a fresh workspace that
// holds a Parquet file and a safetensors file; caliper-mcp runs against the
// API file it writes; and the test does through MCP alone what an outside
// agent would: list the tools, read the docs and the command table, import the
// two files, run their loaders (approved through the stand-in), read the
// catalog, query the table, build the three-panel dashboard of the spec
// fixture dashboard.spec.toml, save the applet and read the spec back. The
// saved spec on disk must equal the fixture. The arguments of the build come
// from the fixture through the session's conversions, so the test knows no
// command's schema. POSIX only.
#include "mcp_child.h"

#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include "../../session/tests/helpers.h"
#include "../../session/tests/parquet_writer.h"

#include <chrono>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace caliper_builder;
using caliper_builder::mcp::test::McpChild;
using caliper_builder::mcp::test::modern_meta;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

// The stand-in host as a child, with its approval lines read from its stdout.
struct Host {
    std::unique_ptr<acp::Process> process;
    std::string api_file;
    std::string pending;

    // `origin` is the origin of the host's token, which every call runs under.
    Host(const std::string& ws, const std::string& api_dir, const std::string& origin = "mcp") {
        acp::SpawnOptions so;
        so.argv = {CALIPER_BUILDER_HOST_EXE, "--workspace", ws, "--api-dir", api_dir, "--origin", origin,
                   "--approve", "ask"};
        Error e;
        process = acp::Process::spawn(so, &e);
        REQUIRE_MESSAGE(process, e.message);
        api_file = api_dir + "/api.json";
        for (int i = 0; i < 400 && !fs::exists(api_file); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(25));
        REQUIRE(fs::exists(api_file));
    }

    // The next approval the host asks for: {"approval", "name", "origin", "args"}.
    json next_approval(int timeout_ms = 20000) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            const auto eol = pending.find('\n');
            if (eol != std::string::npos) {
                const json j = json::parse(pending.substr(0, eol), nullptr, false);
                pending.erase(0, eol + 1);
                if (j.is_object() && j.contains("approval")) return j;
                continue;
            }
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - std::chrono::steady_clock::now()).count();
            if (left <= 0) return nullptr;
            char buf[4096];
            std::size_t got = 0;
            const acp::IoStatus st = process->read_stdout(buf, sizeof buf, &got, int(left));
            if (st == acp::IoStatus::ok) pending.append(buf, got);
            else if (st != acp::IoStatus::timeout) return nullptr;
        }
    }

    void decide(const json& approval, const std::string& decision) {
        const std::string line = decision + " " + approval["approval"].get<std::string>() + "\n";
        REQUIRE(process->write_stdin_all(line.data(), line.size(), 5000) == acp::IoStatus::ok);
    }
};

// The commands that build a spec from nothing, item by item, with the
// session's conversions from the spec's forms to the command table's.
std::vector<std::pair<std::string, json>> build_commands(const spec::Spec& s) {
    std::vector<std::pair<std::string, json>> out;
    json applet{{"id", s.applet.id}};
    if (s.applet.title) applet["title"] = *s.applet.title;
    if (s.applet.description) applet["description"] = *s.applet.description;
    out.emplace_back("spec.new", applet);
    for (const auto& src : s.sources) out.emplace_back("source.add", source_command_args(src));
    for (const auto& p : s.panels) out.emplace_back("panel.create", panel_command_args(p));
    if (s.layout) out.emplace_back("layout.set", json{{"layout", layout_to_flat(*s.layout, s)}});
    return out;
}

json structured(const json& tool_result, const std::string& what) {
    INFO(what << ": " << tool_result.dump().substr(0, 2000));
    REQUIRE(tool_result["isError"] == false);
    REQUIRE(tool_result.contains("structuredContent"));
    return tool_result["structuredContent"];
}

std::string error_code(const json& tool_result) {
    REQUIRE(tool_result["isError"] == true);
    return json::parse(tool_result["content"][0]["text"].get<std::string>())["error"]["code"].get<std::string>();
}

// Sends a tool call that the host holds for approval, answers the approval
// with `decision` when the host asks, and returns the tool result.
json call_with_approval(McpChild& child, Host& host, const json& id, const std::string& command, const json& args,
                        const std::string& decision, const std::string& origin = "mcp") {
    child.send({{"jsonrpc", "2.0"},
                {"id", id},
                {"method", "tools/call"},
                {"params", {{"name", command_tool_name(command)}, {"arguments", args}, {"_meta", modern_meta()}}}});
    const json asked = host.next_approval();
    REQUIRE_MESSAGE(asked.is_object(), "the host asked for no approval for " << command);
    CHECK(asked["name"] == command);
    CHECK(asked["origin"] == origin);
    CHECK(asked["args"] == args);
    host.decide(asked, decision);
    const json r = child.await(id, 30000);
    REQUIRE_MESSAGE(r.contains("result"), r.dump());
    return r["result"];
}

}  // namespace

TEST_CASE("acceptance over MCP: an outside agent builds the dashboard through caliper-mcp and the stand-in host") {
    const std::string root = session_test::fresh_dir("acceptance-mcp");
    const std::string ws = root + "/workspace";
    session_test::write_parquet(ws + "/data/run42_metrics.parquet",
                                "SELECT range AS step, 1.0 / CAST(range + 1 AS DOUBLE) AS loss, "
                                "CAST(range AS DOUBLE) / 10 AS accuracy FROM range(10)");
    std::vector<float> w(96);
    for (int i = 0; i < 96; ++i) w[i] = static_cast<float>(i);
    session_test::write_safetensors_f32(ws + "/data/attn.safetensors", "weights", {2, 3, 4, 4}, w);
    const spec::Spec dashboard = session_test::parse_fixture("dashboard.spec.toml");
    const std::string docs = fs::canonical(fs::path(CALIPER_BUILDER_SOURCE_DIR) / ".." / ".." / "docs").string();

    Host host(ws, root + "/api");
    const std::string token = json::parse(session_test::read_text(host.api_file))["token"].get<std::string>();
    McpChild child({"--api-file", host.api_file, "--docs", docs});

    // What the agent sees first: the server, its tools and its resources.
    const json discover = child.request("discover", "server/discover", json::object());
    CHECK(discover["result"]["supportedVersions"][0] == "2026-07-28");
    const json tools = child.request("tools", "tools/list", json::object())["result"]["tools"];
    // Every command an agent may run is a tool; the person's undo and redo are not.
    json table = command_table_to_json();
    json agents_may = json::array();
    for (const auto& c : table["commands"])
        if (c["agents"] == true) agents_may.push_back(c);
    table["commands"] = agents_may;
    REQUIRE(tools.size() == table["commands"].size());
    REQUIRE(tools.size() == command_table().size() - 2);
    std::set<std::string> names;
    for (const auto& t : tools) names.insert(t["name"].get<std::string>());
    for (const auto& c : table["commands"]) CHECK(names.count(command_tool_name(c["name"].get<std::string>())) == 1);
    CHECK(names.count("history_undo") == 0);

    const json resources = child.request("resources", "resources/list", json::object())["result"];
    std::set<std::string> uris;
    for (const auto& r : resources["resources"]) uris.insert(r["uri"].get<std::string>());
    CHECK(uris.count("caliper://docs/applets/manifest.md") == 1);
    CHECK(uris.count("caliper://docs/reference/cli.md") == 1);
    CHECK(uris.count("caliper://commands/command_table.json") == 1);
    CHECK(uris.count("caliper://schema/caliper-spec-v0.schema.json") == 1);
    const json page = child.request("doc", "resources/read", {{"uri", "caliper://docs/reference/cli.md"}});
    CHECK(page["result"]["contents"][0]["text"].get<std::string>().rfind("# CLI", 0) == 0);
    const json from_app = child.request("table", "resources/read", {{"uri", "caliper://commands/command_table.json"}});
    CHECK(json::parse(from_app["result"]["contents"][0]["text"].get<std::string>()) == table);
    const json schema = child.request("schema", "resources/read", {{"uri", "caliper://schema/caliper-spec-v0.schema.json"}});
    CHECK(json::parse(schema["result"]["contents"][0]["text"].get<std::string>(), nullptr, false).is_object());

    // Import the two files of the workspace, then run their loaders, which
    // the stand-in holds until the person allows them.
    const json imported = structured(
        child.call("import", "data_import",
                   {{"paths", {"data/run42_metrics.parquet", "data/attn.safetensors"}}, {"run", false}}),
        "data.import");
    REQUIRE(imported["refused"].empty());
    REQUIRE(imported["loaders"].size() == 2);
    int n = 0;
    for (const auto& l : imported["loaders"]) {
        const json ran = call_with_approval(child, host, "run-" + std::to_string(n++), "loader.run", {{"id", l["id"]}},
                                            "allow");
        CHECK(structured(ran, "loader.run")["outputs"].size() == 1);
    }

    // Read the catalog and query the table.
    const json listed = structured(child.call("catalog", "catalog_list", json::object()), "catalog.list");
    REQUIRE(listed["entries"].size() == 2);
    CHECK(listed["entries"][0]["id"] == "table:run42_metrics");
    CHECK(listed["entries"][1]["id"] == "tensor:attn_weights");
    const json described = structured(child.call("describe", "catalog_describe", {{"id", "tensor:attn_weights"}}),
                                      "catalog.describe");
    CHECK(described["entry"]["shape"] == json{2, 3, 4, 4});
    const json queried = structured(
        child.call("query", "data_query", {{"sql", "SELECT step, loss FROM run42_metrics ORDER BY step"}}), "data.query");
    REQUIRE(queried["rows"].size() == 10);
    CHECK(queried["rows"][3][0] == 3);

    // Build the dashboard command by command, save it, and read it back.
    int step = 0;
    for (const auto& command : build_commands(dashboard)) {
        CAPTURE(command.first);
        structured(child.call("build-" + std::to_string(step++), command_tool_name(command.first), command.second),
                   command.first);
    }
    const json saved = structured(child.call("save", "applet_save", json::object()), "applet.save");
    CHECK(saved["id"] == dashboard.applet.id);
    const json got = structured(child.call("get", "spec_get", json::object()), "spec.get");
    CHECK(got["text"] == spec::to_toml(dashboard));
    CHECK(got["dirty"] == false);

    // The saved applet's spec on disk is the fixture.
    const std::string saved_text = session_test::read_text(ws + "/" + saved["spec"].get<std::string>());
    const auto parsed = spec::parse_toml(saved_text);
    REQUIRE(parsed.ok());
    CHECK(parsed.spec == dashboard);
    CHECK(saved_text == spec::to_toml(dashboard));

    CHECK(child.finish().success());
    CHECK(child.all_out().find(token) == std::string::npos);
    CHECK(child.all_err().find(token) == std::string::npos);

    const acp::ExitStatus status = host.process->terminate(10000);
    CHECK(status.exited);
    CHECK(status.code == 0);
    CHECK(!fs::exists(host.api_file));

    // The same server under the origin agent, as the app's agent pane runs it,
    // against a host whose token is the agent's: the stand-in asks with that
    // origin, and a denial is a tool error.
    {
        Host agent_host(ws, root + "/api-agent", "agent");
        const std::string agent_token =
            json::parse(session_test::read_text(agent_host.api_file))["token"].get<std::string>();
        McpChild pane({"--api-file", agent_host.api_file, "--origin", "agent"});
        const json denied = call_with_approval(pane, agent_host, "delete", "loader.delete",
                                               {{"id", imported["loaders"][0]["id"]}}, "deny", "agent");
        CHECK(error_code(denied) == "command.denied");
        CHECK(pane.finish().success());
        CHECK(pane.all_out().find(agent_token) == std::string::npos);
        CHECK(pane.all_err().find(agent_token) == std::string::npos);
        CHECK(agent_host.process->terminate(10000).code == 0);
    }
}
