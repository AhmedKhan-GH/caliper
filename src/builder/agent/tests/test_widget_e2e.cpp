// A widget session end to end and headless: the caliper-agent sidecar edits
// the `curves` panel of the dashboard fixture on a stacked preview, through
// the real command layer, and the person's Accept commits it as one undo step.
//
// The workspace holds the Parquet table and the safetensors tensor of the
// dashboard, and the fixture dashboard.spec.toml is the open spec. The session
// runs in this process behind the stand-in's server (CommandServer, the
// command endpoints of caliper-builder-host), so that the test can commit the
// preview from the builder origin, which no HTTP origin can claim. The app's
// side opens the preview with preview.begin, scoped to the panel. The model is
// scripted (Pydantic AI's FunctionModel behind the sidecar's test switch) with
// a fake key: one edit clears the log scale ("Use a linear y axis") and one
// sets the title ("Call it Training curves").
//
// Two clients drive the sidecar:
// - caliper-acp-harness, the real AcpClient, with --meta for session/new's
//   `_meta` and one prompt in which the model makes both edits;
// - a small ACP client over the real JSON-RPC layer (acp::JsonRpcPeer, under
//   AcpClient), for what the harness does not do: a follow-up prompt that
//   refines the same preview, and session/close.
// Each case checks that the open spec does not change, that no permission
// request was sent, that preview.diff holds both edits, that a commit from the
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

const json kLinearY = tool("toolu_1", "panel_patch", {{"id", "curves"}, {"patch", {{"options", {{"y_log", nullptr}}}}}});
const json kTitle = tool("toolu_2", "panel_patch", {{"id", "curves"}, {"patch", {{"title", "Training curves"}}}});

json widget_meta() {
    return {{"caliper", {{"widget", {{"target", {{"kind", "panel"}, {"panel", "curves"}}}, {"preview", kPreview}}}}}};
}

// The dashboard's workspace, its session behind the stand-in's server, and
// the preview the app opened for the widget session.
struct WidgetWorkspace {
    std::string root, ws, api_file, script, record;
    std::unique_ptr<Session> session;
    std::unique_ptr<CommandServer> server;
    std::atomic<bool> stop{false};
    std::thread drainer;
    std::string before;
    std::size_t steps_before = 0;

    WidgetWorkspace(const std::string& name, const json& steps) {
        root = fresh_dir(name);
        ws = root + "/workspace";
        write_parquet(ws + "/data/run42_metrics.parquet",
                      "SELECT range AS step, 1.0 / CAST(range + 1 AS DOUBLE) AS loss, "
                      "CAST(range AS DOUBLE) / 10 AS accuracy FROM range(10)");
        std::vector<float> w(2 * 3 * 4 * 4);
        for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<float>(i);
        write_safetensors_f32(ws + "/data/attn.safetensors", "weights", {2, 3, 4, 4}, w);
        write_text(ws + "/dashboard.spec.toml", read_text(fixture_path("dashboard.spec.toml")));
        auto opened = Session::open(ws);
        REQUIRE_MESSAGE(opened.error.ok(), opened.error.message);
        session = std::move(opened.session);
        invoke("data.import", {{"paths", {"data/run42_metrics.parquet", "data/attn.safetensors"}}}, CommandOrigin::ui);
        invoke("spec.open", {{"path", "dashboard.spec.toml"}}, CommandOrigin::ui);
        // The app's side of ⌘K: a preview of the panel, opened before the session.
        invoke("preview.begin", {{"id", kPreview}, {"panel", "curves"}}, CommandOrigin::ui);
        before = session->open_spec()->text;
        steps_before = session->registry().history_state().undo.size();
        REQUIRE(before.find("y_log = true") != std::string::npos);

        CommandServerOptions options;
        options.token = generate_session_token();
        server = std::make_unique<CommandServer>(session->queue(), options);
        REQUIRE(server->start().ok());
        drainer = std::thread([this] {
            while (!stop.load()) {
                session->queue().drain();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
        REQUIRE(write_api_file(root + "/api", server->port(), options.token, &api_file).ok());
        script = root + "/script.json";
        record = root + "/requests.jsonl";
        write_text(script, json{{"steps", steps}, {"record", record}, {"backoff_scale", 0}}.dump());
    }

    ~WidgetWorkspace() { shut(); }

    void shut() {
        if (!server) return;
        session->queue().shutdown();
        server->stop();
        stop.store(true);
        drainer.join();
        server.reset();
    }

    json invoke(const std::string& name, json args, CommandOrigin origin) {
        CommandInvocation inv;
        inv.name = name;
        inv.args = std::move(args);
        inv.origin = origin;
        CommandResult r = session->invoke(std::move(inv));
        INFO(name << ": " << r.error.code << " " << r.error.message);
        REQUIRE(r.ok);
        return r.result;
    }

    std::vector<std::string> agent_argv(const std::string& python) const {
        const std::string builder = CALIPER_BUILDER_SOURCE_DIR;
        return {python, "-P", "-m", "caliper_agent", "--command-table", builder + "/commands/command_table.json",
                "--api-file", api_file};
    }

    std::vector<acp::EnvVar> agent_env() const {
        const std::string builder = CALIPER_BUILDER_SOURCE_DIR;
        return {{"ANTHROPIC_API_KEY", kFakeKey},
                {"CALIPER_AGENT_TEST_SCRIPT", script},
                {"PYTHONPATH", builder + "/agent"},
                {"PYTHONDONTWRITEBYTECODE", "1"}};
    }

    // After the sidecar is done: the open spec, the history and the journal
    // are as they were, the preview holds both edits, the person's commit is
    // one undo step of the agent's two edits, and one undo reverts both.
    void check_preview_and_commit() {
        shut();
        CHECK(session->open_spec()->text == before);
        CHECK(session->registry().history_state().undo.size() == steps_before);
        const json diff = invoke("preview.diff", {{"id", kPreview}}, CommandOrigin::builder);
        CHECK(diff["edits"] == 2);
        const std::string d = diff["diff"];
        CHECK(d.find("-options = { x_label = \"Step\", y_log = true }") != std::string::npos);
        CHECK(d.find("+options = { x_label = \"Step\" }") != std::string::npos);
        CHECK(d.find("-title = \"Loss and accuracy\"") != std::string::npos);
        CHECK(d.find("+title = \"Training curves\"") != std::string::npos);
        const std::size_t lines_before_commit = read_journal(ws + "/journal/commands.jsonl").events.size();

        const json committed = invoke("preview.commit", {{"id", kPreview}}, CommandOrigin::builder);
        CHECK(committed["edits"] == 2);
        const std::string after = session->open_spec()->text;
        CHECK(after.find("y_log") == std::string::npos);
        CHECK(after.find("title = \"Training curves\"") != std::string::npos);
        CHECK(session->registry().history_state().undo.size() == steps_before + 1);
        const JournalRead journal = read_journal(ws + "/journal/commands.jsonl");
        int agent_edits = 0;
        for (std::size_t i = lines_before_commit; i < journal.events.size(); ++i) {
            const auto& e = journal.events[i];
            if (e.name != "panel.patch") continue;
            ++agent_edits;
            CHECK(e.origin == CommandOrigin::agent);
            CHECK(e.group == kPreview);
        }
        CHECK(agent_edits == 2);

        invoke("history.undo", json::object(), CommandOrigin::ui);
        CHECK(session->open_spec()->text == before);
        CHECK(session->registry().history_state().undo.size() == steps_before);

        // Every model request saw the widget's tools and no key.
        const std::string requests = read_text(record);
        CHECK(requests.find(kFakeKey) == std::string::npos);
        CHECK(requests.find("\"handoff\"") != std::string::npos);
        CHECK(requests.find("\"preview_commit\"") == std::string::npos);
        CHECK(requests.find("\"applet_save\"") == std::string::npos);
    }
};

std::vector<json> json_lines(const std::string& text) {
    std::vector<json> lines;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t end = text.find('\n', start);
        const std::string line = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!line.empty()) lines.push_back(json::parse(line));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return lines;
}

bool skipped() {
    if (std::string(CALIPER_BUILDER_TEST_PYTHON).empty()) {
        std::printf("builder_agent_widget skipped: set CALIPER_BUILDER_TEST_PYTHON to a Python with pydantic-ai\n");
        std::fflush(stdout);
        return true;
    }
    return false;
}

}  // namespace

#ifndef _WIN32

TEST_CASE("caliper-acp-harness opens a widget session with --meta, and both edits land on the preview") {
    if (skipped()) return;
    WidgetWorkspace w("widget-harness",
                      json::array({step({text_block("Turning off the log scale and renaming the panel."), kLinearY, kTitle},
                                        "tool_use"),
                                   step({text_block("The y axis is linear and the panel is called Training curves.")},
                                        "end_turn")}));
    std::string agent;
    for (const std::string& a : w.agent_argv(CALIPER_BUILDER_TEST_PYTHON)) agent += (agent.empty() ? "'" : " '") + a + "'";
    acp::SpawnOptions harness;
    harness.argv = {CALIPER_ACP_HARNESS, "--agent", agent, "--cwd", w.ws, "--meta", widget_meta().dump(),
                    "--prompt", "Use a linear y axis and call it Training curves", "--permission", "deny",
                    "--timeout", "120"};
    for (const acp::EnvVar& e : w.agent_env()) {
        harness.argv.push_back("--env");
        harness.argv.push_back(e.name + "=" + e.value);
    }
    harness.env_remove = {"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "CALIPER_DATA_DIR"};
    acp::RunLimits limits;
    limits.timeout_ms = 150000;
    const acp::RunResult run = acp::run_process(harness, limits);
    INFO("harness stdout tail: " << run.out.substr(run.out.size() > 6000 ? run.out.size() - 6000 : 0));
    INFO("harness stderr tail: " << run.err.substr(run.err.size() > 3000 ? run.err.size() - 3000 : 0));
    REQUIRE(run.error.ok());
    REQUIRE(run.status.success());
    CHECK(run.out.find(kFakeKey) == std::string::npos);
    CHECK(run.err.find(kFakeKey) == std::string::npos);

    const std::vector<json> lines = json_lines(run.out);
    REQUIRE(!lines.empty());
    int completed = 0, permission_requests = 0;
    bool session_seen = false;
    for (const json& l : lines) {
        const std::string type = l.value("type", "");
        if (type == "session") {
            session_seen = true;
            CHECK(l["session_id"].get<std::string>().rfind("widget-", 0) == 0);
            const json scope = l["meta"]["caliper"]["widget"]["scope"];
            CHECK(std::find(scope.begin(), scope.end(), "panel.patch") != scope.end());
            CHECK(std::find(scope.begin(), scope.end(), "preview.commit") == scope.end());
        } else if (type == "permission_request") {
            ++permission_requests;
        } else if (type == "tool_call_update" && l["tool_call"].value("status", "") == "completed") {
            ++completed;
            bool diff = false;
            for (const json& c : l["tool_call"].value("content", json::array())) diff = diff || c.value("type", "") == "diff";
            CHECK(diff);
        }
    }
    CHECK(session_seen);
    CHECK(permission_requests == 0);  // the preview card is the review
    CHECK(completed == 2);
    CHECK(lines.back()["type"] == "result");
    CHECK(lines.back()["ok"] == true);
    CHECK(lines.back()["stop_reason"] == "end_turn");
    w.check_preview_and_commit();
}

TEST_CASE("a follow-up refines the same preview, and session/close ends the widget session") {
    if (skipped()) return;
    WidgetWorkspace w("widget-refine",
                      json::array({step({text_block("Turning off the log scale."), kLinearY}, "tool_use"),
                                   step({text_block("The y axis of Loss and accuracy is linear now.")}, "end_turn"),
                                   step({kTitle}, "tool_use"),
                                   step({text_block("The panel is called Training curves.")}, "end_turn")}));
    acp::SpawnOptions spawn;
    spawn.argv = w.agent_argv(CALIPER_BUILDER_TEST_PYTHON);
    spawn.cwd = w.ws;
    spawn.env_remove = {"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "CALIPER_DATA_DIR"};
    spawn.env = w.agent_env();
    Error spawn_error;
    std::unique_ptr<acp::Process> agent = acp::Process::spawn(spawn, &spawn_error);
    REQUIRE_MESSAGE(agent, spawn_error.message);
    std::mutex err_mutex;
    std::string agent_err;
    std::thread err_reader([&] {
        char buf[4096];
        for (;;) {
            std::size_t got = 0;
            const acp::IoStatus st = agent->read_stderr(buf, sizeof buf, &got, 200);
            if (st == acp::IoStatus::ok) {
                std::lock_guard<std::mutex> lock(err_mutex);
                agent_err.append(buf, got);
            }
            if (st == acp::IoStatus::eof || st == acp::IoStatus::error) break;
        }
    });
    auto stderr_text = [&] {
        std::lock_guard<std::mutex> lock(err_mutex);
        return agent_err;
    };

    // The small ACP client.
    std::mutex mutex;
    std::vector<json> updates;
    int permission_requests = 0;
    acp::JsonRpcPeer peer(acp::make_process_channel(*agent));
    peer.on_notification("session/update", [&](const json& params) {
        std::lock_guard<std::mutex> lock(mutex);
        updates.push_back(params.value("update", json::object()));
    });
    peer.on_request("session/request_permission", [&](const json& id, const json&) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++permission_requests;
        }
        peer.respond(id, {{"outcome", {{"outcome", "cancelled"}}}});
    });
    peer.start();

    const acp::RpcResult init =
        peer.call_sync("initialize", {{"protocolVersion", 1}, {"clientCapabilities", json::object()}}, 30000);
    INFO("agent stderr: " << stderr_text());
    REQUIRE(init.ok());
    CHECK(init.result["agentCapabilities"]["promptCapabilities"]["image"] == true);
    CHECK(init.result["agentCapabilities"]["promptCapabilities"]["embeddedContext"] == true);
    CHECK(init.result["agentCapabilities"]["sessionCapabilities"]["close"] == json::object());
    const acp::RpcResult created = peer.call_sync(
        "session/new", {{"cwd", w.ws}, {"mcpServers", json::array()}, {"_meta", widget_meta()}}, 30000);
    REQUIRE_MESSAGE(created.ok(), created.error.message);
    const std::string sid = created.result["sessionId"];

    auto prompt = [&](const std::string& text) {
        const acp::RpcResult r = peer.call_sync(
            "session/prompt", {{"sessionId", sid}, {"prompt", json::array({text_block(text)})}}, 120000);
        INFO("agent stderr: " << stderr_text());
        REQUIRE_MESSAGE(r.ok(), r.error.message);
        CHECK(r.result["stopReason"] == "end_turn");
        const json widget = r.result["_meta"]["caliper"]["widget"];
        CHECK(widget["preview"] == kPreview);
        CHECK(widget["edits"] == 1);
        CHECK(widget["handoff"].is_null());
    };
    prompt("Use a linear y axis");
    CHECK(w.session->open_spec()->text == w.before);  // the edit is in the preview only
    CHECK(w.session->preview(kPreview).spec.text.find("y_log") == std::string::npos);
    prompt("Call it Training curves");
    CHECK(w.session->open_spec()->text == w.before);

    const acp::RpcResult closed = peer.call_sync("session/close", {{"sessionId", sid}}, 30000);
    CHECK(closed.ok());
    CHECK(closed.result == json::object());
    const acp::RpcResult after_close = peer.call_sync(
        "session/prompt", {{"sessionId", sid}, {"prompt", json::array({text_block("Again")})}}, 30000);
    CHECK_FALSE(after_close.ok());
    peer.close();  // closes the sidecar's input, after which it exits with 0
    acp::ExitStatus agent_status;
    const bool exited = agent->wait_for(15000, &agent_status);
    if (!exited) agent_status = agent->terminate(2000);
    err_reader.join();
    CHECK(agent_err.find(kFakeKey) == std::string::npos);
    CHECK(exited);
    CHECK(agent_status.success());
    {
        std::lock_guard<std::mutex> lock(mutex);
        CHECK(permission_requests == 0);
        int completed = 0;
        for (const json& u : updates)
            if (u.value("sessionUpdate", "") == "tool_call_update" && u.value("status", "") == "completed") ++completed;
        CHECK(completed == 2);
    }
    w.check_preview_and_commit();
}

#endif
