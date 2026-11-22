// The widget agent over ACP: the inline surface drives the real caliper-agent
// sidecar through AcpWidgetAgent and AcpClient, with the sidecar's scripted
// model (its test switch, CALIPER_AGENT_TEST_SCRIPT) and a fake key. No real
// model and no real key: the script plays Claude's answers, and nothing
// leaves the sidecar's process.
//
// The fake key goes into the sidecar's environment directly, as
// builder_agent_widget does, and not through the key store: the key store
// removes CALIPER_AGENT_TEST_SCRIPT at spawn, so that the switch can never
// reach a sidecar that holds a real key, and a sidecar started through it
// would talk to the real Claude API. The rig refuses to start the sidecar
// unless the switch is in its environment and no key store is given, and a
// case below checks that the key store does strip the switch.
//
// The workspace holds the dashboard of dashboard.spec.toml over a Parquet
// table and a safetensors tensor; its session runs in this process behind the
// stand-in's command server, which the sidecar reaches over Caliper API with
// X-Caliper-Preview. The surface begins the preview from the ui origin,
// prompts, draws the preview and commits it from the ui origin.
//
// It needs CALIPER_BUILDER_TEST_PYTHON, an interpreter with the sidecar's
// dependencies; without it the cases say they are skipped.
#include "builder/widget_agent_acp.h"
#include "builder/widget_surface.h"

#include "spec_test_workspace.h"

#include <caliper_builder/acp/acp_client.h>
#include <caliper_builder/agent/key_store.h>
#include <caliper_builder/commands/journal.h>
#include <caliper_builder/session/command_server.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace caliper_host;
using namespace spec_test;
using caliper_builder::CommandOrigin;
using nlohmann::json;

namespace {

// Not a key: the sidecar's test switch takes any well-formed one.
const char* const kFakeKey = "sk-ant-api03-TESTONLY-inline-0123456789abcdefABCDEF-fakeAA";

json tool(const std::string& id, const std::string& name, const json& input) {
    return {{"type", "tool_use"}, {"id", id}, {"name", name}, {"input", input}};
}
json text_block(const std::string& t) { return {{"type", "text"}, {"text", t}}; }
json step(json content, const std::string& stop) {
    return {{"message", {{"content", std::move(content)}, {"stop_reason", stop},
                         {"usage", {{"input_tokens", 30}, {"output_tokens", 12}}}}}};
}

bool skipped() {
    if (std::string(CALIPER_BUILDER_TEST_PYTHON).empty()) {
        std::printf("caliper_widget_surface_agent skipped: set CALIPER_BUILDER_TEST_PYTHON to a Python with pydantic-ai\n");
        std::fflush(stdout);
        return true;
    }
    return false;
}

struct AcpRig {
    std::string root, ws, api_file, script, record;
    std::unique_ptr<caliper_builder::Session> s;
    std::unique_ptr<caliper_builder::CommandServer> server;
    std::atomic<bool> stop{false};
    std::thread drainer;
    std::unique_ptr<AcpWidgetAgent> agent;
    std::unique_ptr<SessionWidgetSession> host;
    std::unique_ptr<WidgetSurface> surface;
    std::vector<WidgetChatRequest> chats;
    std::string before;
    size_t steps_before = 0;

    AcpRig(const std::string& name, const json& steps, const std::vector<std::string>& extra_args = {}) {
        root = fresh_dir("widget-acp-" + name);
        ws = root + "/ws";
        s = dashboard_workspace(ws, root + "/data");
        before = s->open_spec()->text;
        steps_before = s->registry().history_state().undo.size();
        caliper_builder::CommandServerOptions options;
        options.token = caliper_builder::generate_session_token();
        options.origin = caliper_builder::CommandOrigin::agent;  // the token is the sidecar's
        server = std::make_unique<caliper_builder::CommandServer>(s->queue(), options);
        REQUIRE(server->start().ok());
        drainer = std::thread([this] {
            while (!stop.load()) {
                s->queue().drain();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
        REQUIRE(caliper_builder::write_api_file(root + "/api", server->port(), options.token, &api_file).ok());
        script = root + "/script.json";
        record = root + "/requests.jsonl";
        {
            std::ofstream out(script);
            out << json{{"steps", steps}, {"record", record}, {"backoff_scale", 0}}.dump();
        }
        const std::string builder = CALIPER_BUILDER_SOURCE_DIR;
        AcpWidgetAgentOptions o;
        o.argv = {CALIPER_BUILDER_TEST_PYTHON, "-P", "-m", "caliper_agent", "--command-table",
                  builder + "/commands/command_table.json", "--api-file", api_file};
        o.argv.insert(o.argv.end(), extra_args.begin(), extra_args.end());
        o.env = {{"ANTHROPIC_API_KEY", kFakeKey},
                 {"CALIPER_AGENT_OFFLINE", "1"},
                 {"CALIPER_AGENT_TEST_SCRIPT", script},
                 {"PYTHONPATH", builder + "/agent"},
                 {"PYTHONDONTWRITEBYTECODE", "1"}};
        o.env_remove = {"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "CALIPER_DATA_DIR"};
        o.cwd = ws;
        o.keys = nullptr;
        o.key_required = false;
        // The guards: only the scripted model, never the real API. The
        // sidecar inherits this process's environment, which ctest gives the
        // offline switch, and gets both switches in its own.
        bool scripted = false, offline = false;
        for (const auto& v : o.env) {
            scripted = scripted || (v.name == "CALIPER_AGENT_TEST_SCRIPT" && v.value == script);
            offline = offline || (v.name == "CALIPER_AGENT_OFFLINE" && v.value == "1");
        }
        REQUIRE(scripted);
        REQUIRE(offline);
        const char* inherited = std::getenv("CALIPER_AGENT_OFFLINE");
        REQUIRE_MESSAGE(inherited, "run this test through ctest, which sets CALIPER_AGENT_OFFLINE=1");
        REQUIRE(std::string(inherited) == "1");
        REQUIRE(o.keys == nullptr);
        REQUIRE(std::filesystem::is_regular_file(script));
        agent = std::make_unique<AcpWidgetAgent>(o);
        host = std::make_unique<SessionWidgetSession>(*s);
        WidgetSurface::Hooks hooks;
        hooks.ask_in_chat = [this](const WidgetChatRequest& r) { chats.push_back(r); };
        surface = std::make_unique<WidgetSurface>(*host, *agent, hooks);
    }

    ~AcpRig() {
        surface.reset();
        agent.reset();
        s->queue().shutdown();
        server->stop();
        stop.store(true);
        drainer.join();
        server.reset();
        s->close();
    }

    // Updates the surface until it leaves `running`, at most `seconds`.
    void wait(int seconds = 90) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        while (surface->state() == WidgetState::running && std::chrono::steady_clock::now() < end) {
            surface->update();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void ask(const std::string& text) {
        if (surface->state() == WidgetState::idle || surface->state() == WidgetState::applied)
            REQUIRE(surface->open(WidgetTarget::for_panel("curves")));
        surface->set_text(text);
        surface->submit();
        REQUIRE(surface->state() == WidgetState::running);
        wait();
    }

    size_t undo_steps() const { return s->registry().history_state().undo.size(); }
    std::string requests() const { return read_text(record); }
};

const json kLinearY = tool("toolu_1", "panel_patch", {{"id", "curves"}, {"patch", {{"options", {{"y_log", nullptr}}}}}});
const json kTitle = tool("toolu_2", "panel_patch", {{"id", "curves"}, {"patch", {{"title", "Training curves"}}}});

}  // namespace

#ifndef _WIN32

TEST_CASE("widget agent over ACP: a prompt, a refinement and Accept through the sidecar" * doctest::test_suite("agent")) {
    if (skipped()) return;
    AcpRig r("accept", json::array({step({text_block("Turning off the log scale."), kLinearY}, "tool_use"),
                                    step({text_block("The y axis of Loss and accuracy is linear.")}, "end_turn"),
                                    step({kTitle}, "tool_use"),
                                    step({text_block("The panel is called Training curves.")}, "end_turn")}));
    CHECK(r.agent->available(nullptr));
    r.ask("Use a linear y axis");
    // The scripted model answered: its record of the requests exists.
    REQUIRE(std::filesystem::is_regular_file(r.record));
    INFO("error: " << r.surface->error().code << " " << r.surface->error().message);
    REQUIRE(r.surface->state() == WidgetState::preview);
    // The session's config options name the model; the sidecar's widget default.
    CHECK(r.surface->model_label().find("Haiku") != std::string::npos);
    CHECK(r.surface->edits() == 1);
    CHECK(r.surface->reply().find("linear") != std::string::npos);
    REQUIRE(r.surface->diff().size() == 1);
    CHECK(r.surface->diff()[0].key == "y_log");
    CHECK(r.surface->diff()[0].kind == WidgetDiffLine::Kind::removed);
    CHECK(r.s->open_spec()->text == r.before);   // the edit is on the preview only

    // The session offers its models; choosing one goes through
    // session/set_config_option and the row's label follows.
    const auto choices = r.agent->models();
    REQUIRE(choices.size() > 1);
    std::string other, other_name;
    for (const auto& c : choices)
        if (!c.current && other.empty()) {
            other = c.value;
            other_name = c.name;
        }
    REQUIRE(!other.empty());
    r.agent->set_model(other);
    for (int i = 0; i < 500 && r.surface->model_label().find(other_name) == std::string::npos; ++i) {
        r.surface->update();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(r.surface->model_label().find(other_name) != std::string::npos);

    r.surface->refine();
    r.surface->set_text("Call it Training curves");
    r.surface->submit();
    r.wait();
    REQUIRE(r.surface->state() == WidgetState::preview);
    CHECK(r.surface->edits() == 2);
    CHECK(r.s->open_spec()->text == r.before);
    const std::string preview = r.surface->preview_id();

    r.surface->accept();
    REQUIRE(r.surface->state() == WidgetState::applied);
    const std::string after = r.s->open_spec()->text;
    CHECK(after.find("y_log") == std::string::npos);
    CHECK(after.find("Training curves") != std::string::npos);
    CHECK(r.undo_steps() == r.steps_before + 1);
    const auto journal = caliper_builder::read_journal(r.ws + "/journal/commands.jsonl");
    int agent_edits = 0;
    for (const auto& e : journal.events)
        if (e.name == "panel.patch" && e.group == preview) {
            ++agent_edits;
            CHECK(e.origin == CommandOrigin::agent);
        }
    CHECK(agent_edits == 2);
    // No request the model saw carries the key.
    const std::string requests = r.requests();
    CHECK(requests.find(kFakeKey) == std::string::npos);
    CHECK(requests.find("\"preview_commit\"") == std::string::npos);
    CHECK(requests.find("caliper-data") != std::string::npos);   // the context bundle reached the model
    CHECK(r.agent->running());   // the process stays for the next ⌘K
}

TEST_CASE("widget agent over ACP: a turn cut at its budget says the change is not finished, Refine goes on, and "
          "Accept applies what the card shows" * doctest::test_suite("agent")) {
    if (skipped()) return;
    // A budget of one model request: each reply's edit runs, and the request
    // after it is not made.
    AcpRig r("limit", json::array({step({kLinearY}, "tool_use"), step({kTitle}, "tool_use")}),
             {"--widget-requests", "1"});
    r.ask("Use a linear y axis and call the panel Training curves");
    INFO("error: " << r.surface->error().code << " " << r.surface->error().message);
    REQUIRE(r.surface->state() == WidgetState::preview);
    CHECK(r.surface->limit());
    CHECK(r.surface->tag() == "LIMIT");
    CHECK(r.surface->edits() == 1);
    CHECK(r.surface->reply() == "The change is not finished: the agent stopped at its limit of 1 model request after "
                                "1 edit (panel.patch). Accept applies the edits shown, or Refine lets it finish.");
    CHECK(r.surface->card_actions() == std::vector<WidgetAction>{WidgetAction::accept, WidgetAction::reject,
                                                                 WidgetAction::refine, WidgetAction::diff});
    const std::string preview = r.surface->preview_id();
    // Refine goes on in the same session, on the same preview.
    r.surface->refine();
    r.surface->set_text("Finish it");
    r.surface->submit();
    r.wait();
    REQUIRE(r.surface->state() == WidgetState::preview);
    CHECK(r.surface->preview_id() == preview);
    CHECK(r.surface->edits() == 2);
    REQUIRE(r.surface->diff().size() == 2);
    CHECK(r.s->open_spec()->text == r.before);
    // Accept applies the two edits the card shows, as one undo step.
    r.surface->accept();
    REQUIRE(r.surface->state() == WidgetState::applied);
    const std::string after = r.s->open_spec()->text;
    CHECK(after.find("y_log") == std::string::npos);
    CHECK(after.find("Training curves") != std::string::npos);
    CHECK(r.undo_steps() == r.steps_before + 1);
    int edits = 0;
    for (const auto& e : caliper_builder::read_journal(r.ws + "/journal/commands.jsonl").events)
        if (e.name == "panel.patch" && e.group == preview) ++edits;
    CHECK(edits == 2);
}

TEST_CASE("widget agent over ACP: a refusal, a handoff and Reject" * doctest::test_suite("agent")) {
    if (skipped()) return;
    json refused = step({text_block("I can't help with that.")}, "refusal");
    AcpRig r("declined",
             json::array({refused, step({text_block("Turning off the log scale."), kLinearY}, "tool_use"),
                          step({text_block("The y axis is linear.")}, "end_turn"),
                          step({tool("toolu_h", "handoff", {{"kind", "loader"}, {"reason", "A loader is work for the side chat."}})},
                               "tool_use")}));
    r.ask("Use a linear y axis");
    INFO("error: " << r.surface->error().code << " " << r.surface->error().message);
    REQUIRE(r.surface->state() == WidgetState::declined);
    r.surface->retry();
    r.wait();
    REQUIRE(r.surface->state() == WidgetState::preview);
    const std::string preview = r.surface->preview_id();
    r.surface->reject();
    CHECK(r.surface->state() == WidgetState::idle);
    CHECK(r.s->preview(preview).error);
    CHECK(r.s->open_spec()->text == r.before);
    CHECK(r.undo_steps() == r.steps_before);

    // A new ⌘K opens a new session in the same process (the old one closed).
    r.ask("Write a loader for the validation logs");
    REQUIRE(r.surface->state() == WidgetState::handoff);
    REQUIRE(r.surface->handoff());
    CHECK(r.surface->handoff()->kind == "loader");
    CHECK(r.surface->preview_id().empty());
    r.surface->ask_in_chat();
    REQUIRE(r.chats.size() == 1);
    CHECK(r.chats[0].request == "Write a loader for the validation logs");
    CHECK(r.s->open_spec()->text == r.before);
}

TEST_CASE("widget agent over ACP: the sidecar inherits the offline switch and refuses a real provider" * doctest::test_suite("agent")) {
    if (skipped()) return;
    // The child's environment has the switch: started with the fake key and
    // without the script switch, the sidecar would build the Claude provider
    // on its real transport, and it refuses session/new instead because
    // CALIPER_AGENT_OFFLINE is set, which it can only know from the
    // environment it inherited from this process. No prompt is ever sent, so
    // no model request could be made even without the guard.
    const char* inherited = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE(inherited, "run this test through ctest, which sets CALIPER_AGENT_OFFLINE=1");
    const std::string root = fresh_dir("widget-acp-offline");
    const std::string builder = CALIPER_BUILDER_SOURCE_DIR;
    std::string api;   // a Caliper API file on loopback that nothing serves; no command is called
    REQUIRE(caliper_builder::write_api_file(root + "/api", 9, caliper_builder::generate_session_token(), &api).ok());
    AcpWidgetAgentOptions o;
    o.argv = {CALIPER_BUILDER_TEST_PYTHON, "-P", "-m", "caliper_agent", "--command-table",
              builder + "/commands/command_table.json", "--api-file", api};
    o.env = {{"ANTHROPIC_API_KEY", kFakeKey}, {"PYTHONPATH", builder + "/agent"}, {"PYTHONDONTWRITEBYTECODE", "1"}};
    o.env_remove = {"CALIPER_AGENT_TEST_SCRIPT", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "CALIPER_DATA_DIR"};
    o.cwd = root;
    o.key_required = false;
    AcpWidgetAgent agent(o);
    agent.start(WidgetTarget::for_panel("curves"), "widget-offline", {});
    WidgetAgentEvent e;
    bool got = false;
    for (int i = 0; i < 3000 && !got; ++i) {
        got = agent.poll(&e);
        if (!got) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(got);
    INFO(e.error.code << ": " << e.error.message);
    CHECK(e.kind == WidgetAgentEvent::Kind::failed);
    CHECK(e.error.message.find("CALIPER_AGENT_OFFLINE") != std::string::npos);
}

TEST_CASE("widget agent over ACP: the key store strips the test switch, so it never reaches a real key" * doctest::test_suite("agent")) {
    caliper_builder::agent::MemoryKeyStore keys;
    REQUIRE(keys.set("anthropic", kFakeKey).ok());
    caliper_builder::acp::SpawnOptions spawn;
    spawn.argv = {"caliper-agent"};
    spawn.env = {{"CALIPER_AGENT_TEST_SCRIPT", "script.json"}, {"PYTHONPATH", "agent"}};
    REQUIRE(caliper_builder::agent::place_key_in_environment(keys, "anthropic", &spawn).ok());
    bool key = false, script = false;
    for (const auto& v : spawn.env) {
        key = key || v.name == "ANTHROPIC_API_KEY";
        script = script || v.name == "CALIPER_AGENT_TEST_SCRIPT";
    }
    CHECK(key);
    CHECK_FALSE(script);
    CHECK(std::find(spawn.env_remove.begin(), spawn.env_remove.end(), "CALIPER_AGENT_TEST_SCRIPT") != spawn.env_remove.end());
}

TEST_CASE("widget agent over ACP: no key means no provider, and the sidecar is not started" * doctest::test_suite("agent")) {
    caliper_builder::agent::MemoryKeyStore keys;
    AcpWidgetAgentOptions o;
    o.argv = {"/nonexistent/caliper-agent"};
    o.keys = &keys;
    AcpWidgetAgent agent(o);
    std::string why;
    CHECK_FALSE(agent.available(&why));
    CHECK_FALSE(why.empty());
    CHECK_FALSE(agent.running());
    o.key_required = false;   // a local server needs no key
    AcpWidgetAgent local(o);
    CHECK(local.available(nullptr));
    // A start that cannot spawn the sidecar fails the session.
    local.start(WidgetTarget::for_panel("curves"), "widget-x", {});
    WidgetAgentEvent e;
    bool got = false;
    for (int i = 0; i < 500 && !got; ++i) {
        got = local.poll(&e);
        if (!got) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(got);
    CHECK(e.kind == WidgetAgentEvent::Kind::failed);
    CHECK(!e.error.code.empty());
}

TEST_CASE("widget agent over ACP: the session's _meta and the label from config options" * doctest::test_suite("agent")) {
    WidgetTarget t;
    t.kind = WidgetTarget::Kind::property;
    t.panel = "attention";
    t.path = "options.colormap";
    const json meta = AcpWidgetAgent::session_meta(t, "widget-1", {"spec.get", "panel.patch"});
    CHECK(meta["caliper"]["widget"]["target"] == json{{"kind", "property"}, {"panel", "attention"}, {"path", "options.colormap"}});
    CHECK(meta["caliper"]["widget"]["preview"] == "widget-1");
    CHECK(meta["caliper"]["widget"]["scope"] == json::array({"spec.get", "panel.patch"}));
    CHECK_FALSE(AcpWidgetAgent::session_meta(t, "widget-1")["caliper"]["widget"].contains("scope"));
    const json content = AcpWidgetAgent::prompt_content("Use a linear y axis", json::array({{{"type", "text"}, {"text", "ctx"}}}));
    REQUIRE(content.size() == 2);
    CHECK(content[1]["text"] == "Use a linear y axis");
    CHECK(AcpWidgetAgent::prompt_content("x", json()).size() == 1);
    const json config = json::array(
        {{{"id", "model"}, {"currentValue", "claude-haiku-5-5"},
          {"options", json::array({{{"value", "claude-haiku-5-5"}, {"name", "Claude Haiku 5.5"}}})}},
         {{"id", "effort"}, {"currentValue", "low"}}});
    CHECK(AcpWidgetAgent::label_from_config(config) == "Claude Haiku 5.5 \xC2\xB7 low");
    CHECK(AcpWidgetAgent::label_from_config(json()).empty());
}

#endif

#ifndef _WIN32
TEST_CASE("AcpClient::close_session: refused before a session and by an agent that does not offer it" * doctest::test_suite("agent")) {
    // The stub agent of the ACP tests declares no sessionCapabilities.close.
    caliper_builder::acp::AcpClientOptions o;
    o.agent_argv = {CALIPER_ACP_STUB_AGENT};
    caliper_builder::acp::AcpClient client(o);
    CHECK(client.close_session().code == "acp.not_started");
    REQUIRE(client.start().ok());
    CHECK(client.close_session().code == "acp.no_session");
    REQUIRE(client.new_session(std::filesystem::current_path().string(), {}).ok());
    CHECK(client.close_session().code == "acp.unsupported");
    CHECK_FALSE(client.session_id().empty());   // still open
    client.shutdown();
}
#endif
