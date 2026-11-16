// The BYOK agent on the real command layer, end to end and headless:
// caliper-acp-harness (the real C++ ACP client) drives the caliper-agent
// sidecar, whose tools call a running caliper-builder-host over a fresh
// workspace that holds a Parquet and a safetensors file. The model is scripted
// through Pydantic AI's FunctionModel (the sidecar's test switch, route
// "function"), with a fake key: its conversation imports
// the files, runs their loaders, builds the three-panel dashboard of the spec
// fixture dashboard.spec.toml source by source and panel by panel, sets the
// layout and saves the applet, while the harness allows every permission
// request. The test checks the transcript (each spec edit shown as a diff
// before it runs), the saved spec on disk against the fixture, and the journal.
//
// It needs CALIPER_BUILDER_TEST_PYTHON, an interpreter with the sidecar's
// dependencies; without it the test says it is skipped.
#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/commands/journal.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <chrono>
#include <cstdio>
#include <map>
#include <thread>

#include "../../session/tests/helpers.h"
#include "../../session/tests/parquet_writer.h"

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

const char* const kFakeKey = "sk-ant-api03-TESTONLY-e2e-0123456789abcdefABCDEF-fakeAA";

json text_block(const std::string& t) { return {{"type", "text"}, {"text", t}}; }

json tool(const std::string& id, const std::string& name, const json& input) {
    return {{"type", "tool_use"}, {"id", id}, {"name", name}, {"input", input}};
}

json step(json content, const std::string& stop, int cache_read) {
    return {{"message",
             {{"content", std::move(content)},
              {"stop_reason", stop},
              {"usage", {{"input_tokens", 40}, {"output_tokens", 20}, {"cache_read_input_tokens", cache_read},
                         {"cache_creation_input_tokens", cache_read ? 50 : 9000}}}}}};
}

std::string tool_name(const std::string& command) {
    std::string n = command;
    n[n.find('.')] = '_';
    return n;
}

}  // namespace

#ifndef _WIN32

TEST_CASE("the BYOK agent builds the dashboard through caliper-builder-host") {
    const std::string python = CALIPER_BUILDER_TEST_PYTHON;
    if (python.empty()) {
        std::printf("builder_agent_e2e skipped: set CALIPER_BUILDER_TEST_PYTHON to a Python with pydantic-ai\n");
        std::fflush(stdout);
        return;
    }
    const std::string root = fresh_dir("byok-dashboard");
    const std::string ws = root + "/workspace";
    write_parquet(ws + "/data/run42_metrics.parquet",
                  "SELECT range AS step, 1.0 / CAST(range + 1 AS DOUBLE) AS loss, CAST(range AS DOUBLE) / 10 AS accuracy "
                  "FROM range(10)");
    std::vector<float> w(96);
    for (int i = 0; i < 96; ++i) w[i] = static_cast<float>(i);
    write_safetensors_f32(ws + "/data/attn.safetensors", "weights", {2, 3, 4, 4}, w);
    const spec::Spec dashboard = parse_fixture("dashboard.spec.toml");
    const std::string canonical = spec::to_toml(dashboard);

    // The scripted conversation: the tool calls a model would make.
    std::map<std::string, std::string> command_of;  // tool call id -> command
    int next = 1;
    auto call = [&](const std::string& command, const json& input) {
        const std::string id = "toolu_" + std::to_string(next++);
        command_of[id] = command;
        return tool(id, tool_name(command), input);
    };
    json steps = json::array();
    steps.push_back(step({{{"type", "thinking"}, {"thinking", "Import the files first."}, {"signature", "sig-1"}},
                          text_block("I will import the two files and run their loaders."),
                          call("data.import", {{"paths", {"data/run42_metrics.parquet", "data/attn.safetensors"}},
                                               {"run", false}})},
                         "tool_use", 0));
    steps.push_back(step({call("loader.run", {{"id", "run42_metrics"}}), call("loader.run", {{"id", "attn"}})},
                         "tool_use", 9000));
    json applet{{"id", dashboard.applet.id}};
    if (dashboard.applet.title) applet["title"] = *dashboard.applet.title;
    if (dashboard.applet.description) applet["description"] = *dashboard.applet.description;
    steps.push_back(step({text_block("Now the spec."), call("spec.new", applet)}, "tool_use", 9000));
    json sources = json::array();
    for (const auto& s : dashboard.sources) sources.push_back(call("source.add", source_command_args(s)));
    steps.push_back(step(sources, "tool_use", 9000));
    json panels = json::array();
    for (const auto& p : dashboard.panels) panels.push_back(call("panel.create", panel_command_args(p)));
    steps.push_back(step(panels, "tool_use", 9000));
    REQUIRE(dashboard.layout);
    steps.push_back(step({call("layout.set", {{"layout", layout_to_flat(*dashboard.layout, dashboard)}})},
                         "tool_use", 9000));
    steps.push_back(step({call("applet.save", json::object())}, "tool_use", 9000));
    steps.push_back(step({text_block("The dashboard is built and saved as run42-dashboard.")}, "end_turn", 9000));
    const std::string script = root + "/script.json";
    const std::string record = root + "/requests.jsonl";
    write_text(script, json{{"steps", steps}, {"record", record}, {"backoff_scale", 0}}.dump());

    // The stand-in for the app's command endpoints. It approves what the app
    // would hold for the person, since the harness answers the sidecar's own
    // permission requests.
    acp::SpawnOptions host_options;
    host_options.argv = {CALIPER_BUILDER_HOST_EXE, "--workspace", ws, "--api-dir", root + "/api", "--approve", "allow"};
    Error spawn_error;
    std::unique_ptr<acp::Process> host = acp::Process::spawn(host_options, &spawn_error);
    REQUIRE_MESSAGE(host, spawn_error.message);
    const std::string api_file = root + "/api/api.json";
    for (int i = 0; i < 400 && !fs::exists(api_file); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(25));
    REQUIRE(fs::exists(api_file));

    const std::string builder = CALIPER_BUILDER_SOURCE_DIR;
    acp::SpawnOptions harness;
    harness.argv = {CALIPER_ACP_HARNESS,
                    "--agent",
                    "'" + python + "' -P -m caliper_agent --command-table '" + builder +
                        "/commands/command_table.json' --api-file '" + api_file + "'",
                    "--cwd", ws,
                    "--env", std::string("ANTHROPIC_API_KEY=") + kFakeKey,
                    "--env", "CALIPER_AGENT_TEST_SCRIPT=" + script,
                    "--env", "PYTHONPATH=" + builder + "/agent",
                    "--env", "PYTHONDONTWRITEBYTECODE=1",
                    "--prompt", "Build the run 42 dashboard from the files in data, and save it.",
                    "--permission", "allow",
                    "--timeout", "150"};
    harness.env_remove = {"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "CALIPER_DATA_DIR"};
    acp::RunLimits limits;
    limits.timeout_ms = 180000;
    const acp::RunResult run = acp::run_process(harness, limits);
    REQUIRE_MESSAGE(run.error.ok(), run.error.message);
    const acp::ExitStatus host_status = host->terminate(10000);
    INFO("harness stdout tail: " << run.out.substr(run.out.size() > 6000 ? run.out.size() - 6000 : 0));
    INFO("harness stderr tail: " << run.err.substr(run.err.size() > 3000 ? run.err.size() - 3000 : 0));
    CHECK(run.out.find(kFakeKey) == std::string::npos);
    CHECK(run.err.find(kFakeKey) == std::string::npos);
    REQUIRE(run.status.exited);
    REQUIRE(run.status.code == 0);
    CHECK(host_status.exited);
    CHECK(host_status.code == 0);

    // The transcript.
    std::vector<json> lines;
    std::size_t start = 0;
    while (start < run.out.size()) {
        const std::size_t end = run.out.find('\n', start);
        const std::string line = run.out.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!line.empty()) lines.push_back(json::parse(line));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    REQUIRE(!lines.empty());
    CHECK(lines.back()["type"] == "result");
    CHECK(lines.back()["ok"] == true);
    CHECK(lines.back()["stop_reason"] == "end_turn");

    std::map<std::string, std::size_t> asked_at, done_at;
    std::map<std::string, json> preview_of, change_of;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const json& l = lines[i];
        const std::string type = l["type"];
        if (type != "permission_request" && type != "tool_call_update") continue;
        const json& tc = l["tool_call"];
        const std::string id = tc["id"];
        json diff;
        for (const auto& c : tc.value("content", json::array()))
            if (c["type"] == "diff") diff = c;
        if (type == "permission_request") {
            asked_at[id] = i;
            preview_of[id] = diff;
        } else if (tc.value("status", "") == "completed") {
            done_at[id] = i;
            change_of[id] = diff;
        } else if (tc.value("status", "") == "failed") {
            FAIL("the tool call " << id << " (" << command_of[id] << ") failed: " << tc.dump());
        }
    }
    CHECK(done_at.size() == command_of.size());

    // Every spec edit was shown as a diff before it ran, and the diffs chain
    // from the spec that spec.new made to the fixture's canonical text.
    std::string previous;
    int edits = 0;
    for (int n = 1; n < next; ++n) {
        const std::string id = "toolu_" + std::to_string(n);
        const std::string& command = command_of[id];
        CAPTURE(id);
        CAPTURE(command);
        const bool edit = command == "source.add" || command == "panel.create" || command == "layout.set";
        const bool destructive = command == "loader.run" || command == "applet.save";
        CHECK((asked_at.count(id) == 1) == (edit || destructive));
        if (!edit) continue;
        ++edits;
        REQUIRE(asked_at.count(id));
        REQUIRE(done_at.count(id));
        CHECK(asked_at[id] < done_at[id]);
        const json& preview = preview_of[id];
        REQUIRE_MESSAGE(preview.is_object(), "no diff in the permission request");
        CHECK(preview["path"] == ws + "/applets/run42-dashboard/run42-dashboard.spec.toml");
        CHECK(preview["new_text"] != preview["old_text"]);
        if (!previous.empty()) CHECK(preview["old_text"] == previous);
        previous = preview["new_text"];
        CHECK(change_of[id] == preview);  // what ran is what was shown
    }
    CHECK(edits == static_cast<int>(dashboard.sources.size() + dashboard.panels.size() + 1));
    CHECK(previous == canonical);

    // The saved applet on disk is the fixture.
    const std::string saved = read_text(ws + "/applets/run42-dashboard/run42-dashboard.spec.toml");
    CHECK(saved == canonical);
    auto parsed = spec::parse_toml(saved);
    REQUIRE(parsed.ok());
    CHECK(parsed.spec == dashboard);
    CHECK(fs::exists(ws + "/applets/run42-dashboard/run42-dashboard.caliper.toml"));

    // The journal: every command from the agent, and no line for a dry run.
    const JournalRead journal = read_journal(ws + "/journal/commands.jsonl");
    std::map<std::string, int> done;
    for (const auto& e : journal.events) {
        CHECK(e.origin == CommandOrigin::agent);
        if (e.event == "do" && e.ok) ++done[e.name];
    }
    CHECK(done["panel.create"] == 3);
    CHECK(done["source.add"] == static_cast<int>(dashboard.sources.size()));
    CHECK(done["layout.set"] == 1);

    // One model request per step, all answered by the scripted FunctionModel,
    // and the key in none of them.
    const std::string requests = read_text(record);
    CHECK(requests.find(kFakeKey) == std::string::npos);
    std::size_t count = 0;
    for (std::size_t p = 0; (p = requests.find('\n', p)) != std::string::npos; ++p) ++count;
    CHECK(count == steps.size());
    std::size_t function_route = 0;
    for (std::size_t p = 0; (p = requests.find("\"route\": \"function\"", p)) != std::string::npos; ++p)
        ++function_route;
    CHECK(function_route == steps.size());
}

#endif
