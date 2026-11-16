// A widget session end to end and headless: the caliper-agent sidecar edits
// the `curves` panel of the dashboard fixture on a stacked preview, through
// the real command layer, and the person's Accept commits it as one undo step.
//
// The workspace holds the Parquet table and the safetensors tensor of the
// dashboard, and the fixture dashboard.spec.toml is the open spec. The session
// runs in this process behind the stand-in's server (CommandServer, the
// command endpoints of caliper-builder-host), so that the test can commit the
// preview from the builder origin, which no HTTP origin can claim. The app's
// side opens the preview with preview.begin, scoped to the panel.
//
// The sidecar is driven by a small ACP client over the real JSON-RPC layer
// (acp::JsonRpcPeer, under AcpClient), because caliper-acp-harness and
// AcpClient cannot pass session/new's `_meta` on this branch yet. The model is
// scripted (Pydantic AI's FunctionModel behind the sidecar's test switch) with
// a fake key: "Use a linear y axis" clears the log scale, and the follow-up
// "Call it Training curves" sets the title. The test checks that the open spec
// does not change, that preview.diff holds both edits, that a commit from the
// builder origin applies them as one undo step journaled under the agent
// origin, and that one undo reverts both.
//
// It needs CALIPER_BUILDER_TEST_PYTHON, an interpreter with the sidecar's
// dependencies; without it the test says it is skipped.
#include <caliper_builder/acp/jsonrpc.h>
#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/commands/journal.h>
#include <caliper_builder/session/command_server.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>

#include "../../session/tests/helpers.h"
#include "../../session/tests/parquet_writer.h"

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

const char* const kFakeKey = "sk-ant-api03-TESTONLY-widget-0123456789abcdefABCDEF-fakeAA";
const char* const kPreview = "widget-curves";

json tool(const std::string& id, const std::string& name, const json& input) {
    return {{"type", "tool_use"}, {"id", id}, {"name", name}, {"input", input}};
}

json step(json content, const std::string& stop) {
    return {{"message", {{"content", std::move(content)}, {"stop_reason", stop},
                         {"usage", {{"input_tokens", 30}, {"output_tokens", 12}}}}}};
}

json text_block(const std::string& t) { return {{"type", "text"}, {"text", t}}; }

// What the sidecar sent while the client waited.
struct Transcript {
    std::mutex mutex;
    std::vector<json> updates;
    int permission_requests = 0;
};

}  // namespace

#ifndef _WIN32

TEST_CASE("a widget session edits the curves panel on a preview that the person commits as one undo step") {
    const std::string python = CALIPER_BUILDER_TEST_PYTHON;
    if (python.empty()) {
        std::printf("builder_agent_widget skipped: set CALIPER_BUILDER_TEST_PYTHON to a Python with pydantic-ai\n");
        std::fflush(stdout);
        return;
    }
    const std::string root = fresh_dir("widget-curves");
    const std::string ws = root + "/workspace";
    write_parquet(ws + "/data/run42_metrics.parquet",
                  "SELECT range AS step, 1.0 / CAST(range + 1 AS DOUBLE) AS loss, CAST(range AS DOUBLE) / 10 AS accuracy "
                  "FROM range(10)");
    std::vector<float> w(2 * 3 * 4 * 4);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<float>(i);
    write_safetensors_f32(ws + "/data/attn.safetensors", "weights", {2, 3, 4, 4}, w);
    write_text(ws + "/dashboard.spec.toml", read_text(fixture_path("dashboard.spec.toml")));

    auto opened = Session::open(ws);
    REQUIRE_MESSAGE(opened.error.ok(), opened.error.message);
    Session& s = *opened.session;
    auto invoke = [&](const std::string& name, json args, CommandOrigin origin) {
        CommandInvocation inv;
        inv.name = name;
        inv.args = std::move(args);
        inv.origin = origin;
        CommandResult r = s.invoke(std::move(inv));
        INFO(name << ": " << r.error.code << " " << r.error.message);
        REQUIRE(r.ok);
        return r.result;
    };
    invoke("data.import", {{"paths", {"data/run42_metrics.parquet", "data/attn.safetensors"}}}, CommandOrigin::ui);
    invoke("spec.open", {{"path", "dashboard.spec.toml"}}, CommandOrigin::ui);
    // The app's side of ⌘K: a preview of the panel, opened before the session.
    invoke("preview.begin", {{"id", kPreview}, {"panel", "curves"}}, CommandOrigin::ui);
    const std::string before = s.open_spec()->text;
    const std::size_t steps_before = s.registry().history_state().undo.size();
    REQUIRE(before.find("y_log = true") != std::string::npos);

    // The stand-in's server on the session's queue, drained as the app's frame
    // thread will, and its API file for the sidecar.
    CommandServerOptions server_options;
    server_options.token = generate_session_token();
    CommandServer server(s.queue(), server_options);
    REQUIRE(server.start().ok());
    std::atomic<bool> stop{false};
    std::thread drainer([&] {
        while (!stop.load()) {
            s.queue().drain();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    std::string api_file;
    REQUIRE(write_api_file(root + "/api", server.port(), server_options.token, &api_file).ok());

    // The scripted model: one edit for each prompt, then one sentence.
    const json steps = json::array(
        {step({text_block("Turning off the log scale."),
               tool("toolu_1", "panel_patch", {{"id", "curves"}, {"patch", {{"options", {{"y_log", nullptr}}}}}})},
              "tool_use"),
         step({text_block("The y axis of Loss and accuracy is linear now.")}, "end_turn"),
         step({tool("toolu_2", "panel_patch", {{"id", "curves"}, {"patch", {{"title", "Training curves"}}}})},
              "tool_use"),
         step({text_block("The panel is called Training curves.")}, "end_turn")});
    const std::string script = root + "/script.json";
    const std::string record = root + "/requests.jsonl";
    write_text(script, json{{"steps", steps}, {"record", record}, {"backoff_scale", 0}}.dump());

    const std::string builder = CALIPER_BUILDER_SOURCE_DIR;
    acp::SpawnOptions spawn;
    spawn.argv = {python, "-P", "-m", "caliper_agent", "--command-table", builder + "/commands/command_table.json",
                  "--api-file", api_file};
    spawn.cwd = ws;
    spawn.env_remove = {"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "CALIPER_DATA_DIR"};
    spawn.env = {{"ANTHROPIC_API_KEY", kFakeKey},
                 {"CALIPER_AGENT_TEST_SCRIPT", script},
                 {"PYTHONPATH", builder + "/agent"},
                 {"PYTHONDONTWRITEBYTECODE", "1"}};
    Error spawn_error;
    std::unique_ptr<acp::Process> agent = acp::Process::spawn(spawn, &spawn_error);
    REQUIRE_MESSAGE(agent, spawn_error.message);
    std::string agent_err;
    std::thread err_reader([&] {
        char buf[4096];
        for (;;) {
            std::size_t got = 0;
            const acp::IoStatus st = agent->read_stderr(buf, sizeof buf, &got, 200);
            if (st == acp::IoStatus::ok) agent_err.append(buf, got);
            if (st == acp::IoStatus::eof || st == acp::IoStatus::error) break;
        }
    });

    // The small ACP client.
    Transcript transcript;
    acp::JsonRpcPeer peer(acp::make_process_channel(*agent));
    peer.on_notification("session/update", [&](const json& params) {
        std::lock_guard<std::mutex> lock(transcript.mutex);
        transcript.updates.push_back(params.value("update", json::object()));
    });
    peer.on_request("session/request_permission", [&](const json& id, const json&) {
        {
            std::lock_guard<std::mutex> lock(transcript.mutex);
            ++transcript.permission_requests;
        }
        peer.respond(id, {{"outcome", {{"outcome", "cancelled"}}}});
    });
    peer.start();

    const acp::RpcResult init =
        peer.call_sync("initialize", {{"protocolVersion", 1}, {"clientCapabilities", json::object()}}, 30000);
    INFO("agent stderr: " << agent_err);
    REQUIRE(init.ok());
    CHECK(init.result["agentCapabilities"]["promptCapabilities"]["image"] == true);
    CHECK(init.result["agentCapabilities"]["promptCapabilities"]["embeddedContext"] == true);
    CHECK(init.result["agentCapabilities"]["sessionCapabilities"]["close"] == json::object());

    const json meta{{"caliper", {{"widget", {{"target", {{"kind", "panel"}, {"panel", "curves"}}}, {"preview", kPreview}}}}}};
    const acp::RpcResult created =
        peer.call_sync("session/new", {{"cwd", ws}, {"mcpServers", json::array()}, {"_meta", meta}}, 30000);
    REQUIRE_MESSAGE(created.ok(), created.error.message);
    const std::string sid = created.result["sessionId"];
    CHECK(sid.rfind("widget-", 0) == 0);
    const json scope = created.result["_meta"]["caliper"]["widget"]["scope"];
    CHECK(std::find(scope.begin(), scope.end(), "panel.patch") != scope.end());
    CHECK(std::find(scope.begin(), scope.end(), "preview.commit") == scope.end());

    auto prompt = [&](const std::string& text) {
        const acp::RpcResult r = peer.call_sync(
            "session/prompt", {{"sessionId", sid}, {"prompt", json::array({text_block(text)})}}, 120000);
        INFO("agent stderr: " << agent_err);
        REQUIRE_MESSAGE(r.ok(), r.error.message);
        CHECK(r.result["stopReason"] == "end_turn");
        const json widget = r.result["_meta"]["caliper"]["widget"];
        CHECK(widget["preview"] == kPreview);
        CHECK(widget["edits"] == 1);
        CHECK(widget["handoff"].is_null());
        return r.result;
    };
    prompt("Use a linear y axis");
    // The first edit is in the preview only.
    CHECK(s.open_spec()->text == before);
    CHECK(s.preview(kPreview).spec.text.find("y_log") == std::string::npos);
    prompt("Call it Training curves");
    CHECK(s.open_spec()->text == before);

    const acp::RpcResult closed = peer.call_sync("session/close", {{"sessionId", sid}}, 30000);
    CHECK(closed.ok());
    CHECK(closed.result == json::object());
    peer.close();  // closes the sidecar's input, after which it exits with 0
    acp::ExitStatus agent_status;
    const bool exited = agent->wait_for(15000, &agent_status);
    if (!exited) agent_status = agent->terminate(2000);
    err_reader.join();
    CHECK(agent_err.find(kFakeKey) == std::string::npos);
    CHECK(exited);
    CHECK(agent_status.success());
    s.queue().shutdown();
    server.stop();
    stop.store(true);
    drainer.join();

    {
        std::lock_guard<std::mutex> lock(transcript.mutex);
        CHECK(transcript.permission_requests == 0);  // the preview card is the review
        int completed = 0;
        for (const json& u : transcript.updates) {
            if (u.value("sessionUpdate", "") != "tool_call_update" || u.value("status", "") != "completed") continue;
            ++completed;
            bool diff = false;
            for (const json& c : u.value("content", json::array())) diff = diff || c.value("type", "") == "diff";
            CHECK(diff);
        }
        CHECK(completed == 2);
    }

    // The open spec, the history and the journal are as they were; the preview
    // holds both edits.
    CHECK(s.open_spec()->text == before);
    CHECK(s.registry().history_state().undo.size() == steps_before);
    const json diff = invoke("preview.diff", {{"id", kPreview}}, CommandOrigin::builder);
    CHECK(diff["edits"] == 2);
    const std::string d = diff["diff"];
    CHECK(d.find("-options = { x_label = \"Step\", y_log = true }") != std::string::npos);
    CHECK(d.find("+options = { x_label = \"Step\" }") != std::string::npos);
    CHECK(d.find("-title = \"Loss and accuracy\"") != std::string::npos);
    CHECK(d.find("+title = \"Training curves\"") != std::string::npos);
    const std::size_t lines_before_commit = read_journal(ws + "/journal/commands.jsonl").events.size();

    // Accept: the person's commit, as one undo step.
    const json committed = invoke("preview.commit", {{"id", kPreview}}, CommandOrigin::builder);
    CHECK(committed["edits"] == 2);
    const std::string after = s.open_spec()->text;
    CHECK(after.find("y_log") == std::string::npos);
    CHECK(after.find("title = \"Training curves\"") != std::string::npos);
    CHECK(s.registry().history_state().undo.size() == steps_before + 1);
    const JournalRead journal = read_journal(ws + "/journal/commands.jsonl");
    int agent_edits = 0;
    for (std::size_t i = lines_before_commit; i < journal.events.size(); ++i) {
        const auto& e = journal.events[i];
        if (e.name == "panel.patch") {
            ++agent_edits;
            CHECK(e.origin == CommandOrigin::agent);
            CHECK(e.group == kPreview);
        }
    }
    CHECK(agent_edits == 2);

    // One undo reverts both.
    invoke("history.undo", json::object(), CommandOrigin::ui);
    CHECK(s.open_spec()->text == before);
    CHECK(s.registry().history_state().undo.size() == steps_before);

    // Every model request saw the widget's tools and no key.
    const std::string requests = read_text(record);
    CHECK(requests.find(kFakeKey) == std::string::npos);
    CHECK(requests.find("\"handoff\"") != std::string::npos);
    CHECK(requests.find("\"preview_commit\"") == std::string::npos);
    CHECK(requests.find("\"applet_save\"") == std::string::npos);
}

#endif
