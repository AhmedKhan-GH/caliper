// AcpClient against caliper_acp_stub_agent, one test per script and per way
// of misbehaving. POSIX only.
#include <caliper_builder/acp/acp_client.h>

#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace caliper_builder::acp;
using caliper_builder::Error;
using nlohmann::json;

namespace {

const std::string kStub = CALIPER_ACP_STUB_AGENT;
const std::string kChild = CALIPER_ACP_TEST_CHILD;

std::string workspace() {
    std::filesystem::path p = std::filesystem::path(CALIPER_BUILDER_TEST_TMP) / "acp-workspace";
    std::filesystem::create_directories(p);
    return std::filesystem::canonical(p).string();
}

AcpClientOptions stub_options(std::vector<std::string> extra = {}) {
    AcpClientOptions o;
    o.agent_argv = {kStub};
    o.agent_argv.insert(o.agent_argv.end(), extra.begin(), extra.end());
    o.agent_cwd = workspace();
    o.request_timeout_ms = 5000;
    o.kill_grace_ms = 1000;
    return o;
}

void open_session(AcpClient& c, std::vector<McpServer> servers = {}) {
    AgentInfo info;
    Error e = c.start(&info);
    REQUIRE_MESSAGE(e.ok(), e.code << ": " << e.message);
    e = c.new_session(workspace(), servers);
    REQUIRE_MESSAGE(e.ok(), e.code << ": " << e.message);
}

// Collects events until the turn ends (or the agent exits), answering each
// permission request with `answer`, which may return "" to leave it open.
std::vector<AcpEvent> run_turn(AcpClient& c, std::function<std::string(const AcpEvent&)> answer = nullptr,
                               int timeout_ms = 10000) {
    std::vector<AcpEvent> out;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        AcpEvent ev;
        if (!c.wait(&ev, 50)) continue;
        out.push_back(ev);
        if (ev.kind == AcpEventKind::permission_request && answer) {
            const std::string option = answer(ev);
            if (!option.empty()) CHECK(c.respond_permission(ev.permission_id, option).ok());
        }
        if (ev.kind == AcpEventKind::turn_ended || ev.kind == AcpEventKind::agent_exited) break;
    }
    return out;
}

std::string message_text(const std::vector<AcpEvent>& events) {
    std::string s;
    for (const auto& e : events)
        if (e.kind == AcpEventKind::agent_message_chunk) s += e.text;
    return s;
}

int count(const std::vector<AcpEvent>& events, AcpEventKind kind) {
    int n = 0;
    for (const auto& e : events) n += e.kind == kind;
    return n;
}

int count_faults(const std::vector<AcpEvent>& events, const std::string& code) {
    int n = 0;
    for (const auto& e : events) n += e.kind == AcpEventKind::protocol_fault && e.error.code == code;
    return n;
}

const AcpEvent* find(const std::vector<AcpEvent>& events, AcpEventKind kind) {
    for (const auto& e : events)
        if (e.kind == kind) return &e;
    return nullptr;
}

}  // namespace

TEST_CASE("initialize negotiates version 1 and reads the agent's description") {
    AcpClient c(stub_options());
    AgentInfo info;
    REQUIRE(c.start(&info).ok());
    CHECK(info.protocol_version == 1);
    CHECK(info.name == "caliper-acp-stub");
    CHECK(info.title == "Caliper ACP stub agent");
    CHECK_FALSE(info.load_session);
    CHECK_FALSE(info.mcp_http);
    REQUIRE(info.auth_methods.is_array());
    CHECK(info.auth_methods.size() == 1);
    CHECK(c.agent().name == "caliper-acp-stub");
    CHECK(c.start().code == "acp.already_started");
    c.shutdown();
    CHECK_FALSE(c.agent_running());
}

TEST_CASE("the client declines the file-system and terminal capabilities") {
    AcpClient c(stub_options());
    open_session(c);
    REQUIRE(c.prompt("caps").ok());
    auto events = run_turn(c);
    json caps = json::parse(message_text(events));
    CHECK(caps["protocolVersion"] == 1);
    CHECK(caps["clientCapabilities"]["fs"]["readTextFile"] == false);
    CHECK(caps["clientCapabilities"]["fs"]["writeTextFile"] == false);
    CHECK(caps["clientCapabilities"]["terminal"] == false);
    CHECK(caps["clientInfo"]["name"] == "caliper");
    CHECK(caps["clientInfo"].contains("version"));
}

TEST_CASE("an agent that calls a declined method gets method-not-found") {
    AcpClient c(stub_options());
    open_session(c);
    REQUIRE(c.prompt("probe").ok());
    auto events = run_turn(c);
    json codes = json::parse(message_text(events));
    CHECK(codes["fs/read_text_file"] == -32601);
    CHECK(codes["terminal/create"] == -32601);
    CHECK(count_faults(events, "acp.declined_method") == 2);
    REQUIRE(find(events, AcpEventKind::turn_ended));
    CHECK(find(events, AcpEventKind::turn_ended)->stop_reason == "end_turn");
}

TEST_CASE("session/new hands the agent its folder and the MCP servers") {
    AcpClient c(stub_options());
    McpServer caliper;
    caliper.name = "caliper";
    caliper.command = kChild;
    caliper.args = {"hang", "--api-file", "a file.json"};
    caliper.env = {{"CALIPER_DATA_DIR", "/data"}};
    McpServer shell;
    shell.name = "other";
    shell.command = "sh";  // resolved on PATH to an absolute path
    open_session(c, {caliper, shell});
    CHECK(c.session_id() == "stub-session-1");

    REQUIRE(c.prompt("mcp").ok());
    auto events = run_turn(c);
    json report = json::parse(message_text(events));
    CHECK(report["cwd"] == workspace());
    REQUIRE(report["mcpServers"].size() == 2);
    const json& s0 = report["mcpServers"][0];
    CHECK(s0["name"] == "caliper");
    CHECK(s0["command"] == kChild);
    CHECK(s0["args"] == json::array({"hang", "--api-file", "a file.json"}));
    CHECK(s0["env"] == json::array({{{"name", "CALIPER_DATA_DIR"}, {"value", "/data"}}}));
    const json& s1 = report["mcpServers"][1];
    CHECK(s1["command"].get<std::string>().front() == '/');
    CHECK(s1["env"] == json::array());
    CHECK(report["_meta"].is_null());  // no _meta given, none sent
}

TEST_CASE("session/new passes the widget session's _meta through unchanged and bounds it") {
    const json meta = {{"caliper",
                        {{"widget",
                          {{"target", {{"kind", "panel"}, {"panel", "curves"}}},
                           {"scope", {"spec.get", "panel.patch", "layout.place"}},
                           {"preview", "widget-7f3a"},
                           {"note", "caf\u00e9 \u2713"},
                           {"ratio", 0.25},
                           {"empty", json::object()}}}}}};
    AcpClient c(stub_options());
    REQUIRE(c.start().ok());
    // Refused before anything is sent: not an object, too large, too deep, too many values.
    CHECK(c.new_session(workspace(), {}, nullptr, json::array({1})).code == "acp.bad_argument");
    CHECK(c.new_session(workspace(), {}, nullptr, json{{"big", std::string(kMaxSessionMetaBytes, 'x')}}).where ==
          "_meta");
    json deep = json::object();
    json* at = &deep;
    for (int i = 0; i < kMaxSessionMetaDepth; ++i) at = &((*at)["d"] = json::object());
    CHECK(c.new_session(workspace(), {}, nullptr, deep).code == "acp.bad_argument");
    CHECK(c.new_session(workspace(), {}, nullptr, json{{"many", json(std::vector<int>(kMaxSessionMetaValues, 1))}})
              .code == "acp.bad_argument");
    CHECK(c.session_id().empty());

    SessionInfo info;
    Error e = c.new_session(workspace(), {}, &info, meta);
    REQUIRE_MESSAGE(e.ok(), e.code << ": " << e.message);
    CHECK(info.meta == json{{"echo", meta}});
    REQUIRE(c.prompt("mcp").ok());
    const json report = json::parse(message_text(run_turn(c)));
    CHECK(report["_meta"] == meta);
}

TEST_CASE("session/new refuses bad folders and servers") {
    AcpClient c(stub_options());
    REQUIRE(c.start().ok());
    CHECK(c.new_session("relative/path", {}).code == "acp.bad_argument");
    CHECK(c.new_session(workspace() + "/does-not-exist", {}).code == "acp.bad_argument");
    McpServer a;
    a.name = "x";
    a.command = kChild;
    CHECK(c.new_session(workspace(), {a, a}).code == "acp.bad_argument");
    McpServer missing;
    missing.name = "m";
    missing.command = "caliper-no-such-mcp-server";
    CHECK(c.new_session(workspace(), {missing}).code == "acp.bad_argument");
    McpServer nameless;
    nameless.command = kChild;
    CHECK(c.new_session(workspace(), {nameless}).code == "acp.bad_argument");
    CHECK(c.prompt("hello").code == "acp.no_session");
}

TEST_CASE("a turn streams every kind of update") {
    AcpClient c(stub_options());
    open_session(c);
    REQUIRE(c.prompt("hello").ok());
    CHECK(c.turn_active());
    CHECK(c.prompt("again").code == "acp.turn_active");
    auto events = run_turn(c);
    CHECK(message_text(events) == "Hi, world.");
    const AcpEvent* thought = find(events, AcpEventKind::agent_thought_chunk);
    REQUIRE(thought);
    CHECK(thought->text == "Thinking about hello.");
    const AcpEvent* plan = find(events, AcpEventKind::plan);
    REQUIRE(plan);
    REQUIRE(plan->plan.size() == 2);
    CHECK(plan->plan[0].content == "Say hello");
    CHECK(plan->plan[0].priority == "high");
    CHECK(plan->plan[0].status == "in_progress");
    const AcpEvent* cmds = find(events, AcpEventKind::available_commands);
    REQUIRE(cmds);
    REQUIRE(cmds->commands.size() == 2);
    CHECK(cmds->commands[0].name == "spec");
    CHECK(cmds->commands[0].hint == "instrument id");
    const AcpEvent* mode = find(events, AcpEventKind::current_mode);
    REQUIRE(mode);
    CHECK(mode->mode_id == "plan");
    const AcpEvent* other = find(events, AcpEventKind::other_update);
    REQUIRE(other);
    CHECK(other->update_kind == "future_update_kind");
    CHECK(other->raw["value"] == 1);
    const AcpEvent* end = find(events, AcpEventKind::turn_ended);
    REQUIRE(end);
    CHECK(end->stop_reason == "end_turn");
    CHECK(end->error.ok());
    CHECK_FALSE(c.turn_active());

    // The session goes on.
    REQUIRE(c.prompt("echo me").ok());
    CHECK(message_text(run_turn(c)) == "You said: echo me");
}

TEST_CASE("a tool call's Caliper command comes from its _meta, when the agent names one") {
    AcpClient c(stub_options());
    open_session(c);
    REQUIRE(c.prompt("command").ok());
    const auto events = run_turn(c);
    std::vector<const AcpEvent*> calls;
    for (const auto& e : events)
        if (e.kind == AcpEventKind::tool_call || e.kind == AcpEventKind::tool_call_update) calls.push_back(&e);
    REQUIRE(calls.size() == 3);
    CHECK(calls[0]->tool_call.id == "call_c");
    CHECK(calls[0]->tool_call.caliper_command == "catalog.list");
    CHECK(event_to_json(*calls[0])["tool_call"]["caliper_command"] == "catalog.list");
    CHECK(calls[1]->tool_call.caliper_command.empty());   // an update without _meta names none
    CHECK(calls[2]->tool_call.caliper_command.empty());   // not a string
    CHECK(count(events, AcpEventKind::protocol_fault) == 0);
}

TEST_CASE("a proposed edit arrives as a diff and waits for permission") {
    AcpClient c(stub_options());
    open_session(c);
    REQUIRE(c.prompt("edit").ok());
    std::vector<PermissionOption> offered;
    auto events = run_turn(c, [&](const AcpEvent& ev) {
        offered = ev.options;
        CHECK(ev.tool_call.id == "call_1");
        CHECK(c.respond_permission(ev.permission_id + 100, "allow").code == "acp.unknown_permission");
        CHECK(c.respond_permission(ev.permission_id, "no-such-option").code == "acp.unknown_option");
        return std::string("allow");
    });
    const AcpEvent* call = find(events, AcpEventKind::tool_call);
    REQUIRE(call);
    CHECK(call->tool_call.kind == "edit");
    CHECK(call->tool_call.status == "pending");
    REQUIRE(call->tool_call.content.size() == 1);
    const ToolCallContent& diff = call->tool_call.content[0];
    CHECK(diff.type == "diff");
    CHECK(diff.path == workspace() + "/demo.spec.toml");
    CHECK(diff.has_old_text);
    CHECK(diff.old_text == "title = \"Old\"\n");
    CHECK(diff.new_text == "title = \"New\"\n");
    REQUIRE(call->tool_call.locations.size() == 1);
    CHECK(call->tool_call.locations[0].line == 1);
    CHECK(call->tool_call.raw_input["file"] == "demo.spec.toml");
    REQUIRE(offered.size() == 3);
    CHECK(offered[0].kind == "allow_once");
    CHECK(offered[2].kind == "reject_once");

    int completed = 0;
    for (const auto& e : events)
        if (e.kind == AcpEventKind::tool_call_update && e.tool_call.has_status && e.tool_call.status == "completed") {
            ++completed;
            REQUIRE(e.tool_call.content.size() == 1);
            CHECK(e.tool_call.content[0].text == "Applied.");
            CHECK_FALSE(e.tool_call.has_title);
        }
    CHECK(completed == 1);
    CHECK(message_text(events) == "Applied the change.");
    CHECK(find(events, AcpEventKind::turn_ended)->stop_reason == "end_turn");
    // An answered request cannot be answered twice.
    CHECK(c.respond_permission(1, "allow").code == "acp.unknown_permission");
}

TEST_CASE("a refused edit leaves the file alone") {
    AcpClient c(stub_options());
    open_session(c);
    REQUIRE(c.prompt("edit").ok());
    auto events = run_turn(c, [](const AcpEvent&) { return std::string("reject"); });
    CHECK(message_text(events) == "Left the spec unchanged.");
    bool failed = false;
    for (const auto& e : events)
        failed |= e.kind == AcpEventKind::tool_call_update && e.tool_call.status == "failed";
    CHECK(failed);
    CHECK(find(events, AcpEventKind::turn_ended)->stop_reason == "end_turn");
}

TEST_CASE("cancelling answers an open permission request as cancelled") {
    AcpClient c(stub_options());
    open_session(c);
    REQUIRE(c.prompt("edit").ok());
    auto events = run_turn(c, [&](const AcpEvent&) {
        CHECK(c.cancel().ok());
        CHECK(c.cancel().ok());  // a second cancel is harmless
        return std::string();
    });
    CHECK(count(events, AcpEventKind::permission_withdrawn) == 1);
    const AcpEvent* end = find(events, AcpEventKind::turn_ended);
    REQUIRE(end);
    CHECK(end->stop_reason == "cancelled");
    CHECK(end->error.ok());
    CHECK(c.cancel().code == "acp.no_turn");
}

TEST_CASE("cancelling a streaming turn ends it with the cancelled stop reason") {
    AcpClient c(stub_options());
    open_session(c);
    REQUIRE(c.prompt("slow").ok());
    std::vector<AcpEvent> events;
    bool cancelled = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        AcpEvent ev;
        if (!c.wait(&ev, 50)) continue;
        events.push_back(ev);
        if (!cancelled && count(events, AcpEventKind::agent_message_chunk) >= 3) {
            REQUIRE(c.cancel().ok());
            cancelled = true;
        }
        if (ev.kind == AcpEventKind::turn_ended) break;
    }
    const AcpEvent* end = find(events, AcpEventKind::turn_ended);
    REQUIRE(end);
    CHECK(end->stop_reason == "cancelled");
    CHECK(message_text(events).find("Stopped.") != std::string::npos);
}

TEST_CASE("an agent may withdraw its own permission request") {
    AcpClient c(stub_options());
    open_session(c);
    REQUIRE(c.prompt("withdraw").ok());
    auto events = run_turn(c);  // never answers the request itself
    CHECK(count(events, AcpEventKind::permission_request) == 1);
    CHECK(count(events, AcpEventKind::permission_withdrawn) == 1);
    CHECK(message_text(events) == "withdrawn: cancelled");
    CHECK(find(events, AcpEventKind::turn_ended)->stop_reason == "end_turn");
}

TEST_CASE("malformed lines are reported and the turn goes on") {
    AcpClient c(stub_options());
    open_session(c);
    REQUIRE(c.prompt("malformed").ok());
    auto events = run_turn(c);
    CHECK(count_faults(events, "rpc.parse_error") == 2);
    CHECK(count_faults(events, "rpc.invalid_message") == 3);
    CHECK(count_faults(events, "rpc.unknown_response") == 1);
    CHECK(count_faults(events, "rpc.too_deep") == 1);
    CHECK(count_faults(events, "acp.bad_update") == 3);
    CHECK(message_text(events) == "Still here.");
    CHECK(find(events, AcpEventKind::turn_ended)->stop_reason == "end_turn");
}

TEST_CASE("an oversized message is dropped and the turn goes on") {
    AcpClientOptions o = stub_options();
    o.max_message_bytes = 64 * 1024;
    AcpClient c(o);
    open_session(c);
    REQUIRE(c.prompt("oversize 300000").ok());
    auto events = run_turn(c);
    CHECK(count_faults(events, "rpc.message_too_large") == 1);
    CHECK(message_text(events) == "After.");
    CHECK(find(events, AcpEventKind::turn_ended)->stop_reason == "end_turn");
}

TEST_CASE("an agent that ignores the cancel gets a synthetic end of turn") {
    AcpClientOptions o = stub_options();
    o.cancel_timeout_ms = 300;
    AcpClient c(o);
    open_session(c);
    REQUIRE(c.prompt("hang").ok());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    REQUIRE(c.cancel().ok());
    auto events = run_turn(c, nullptr, 5000);
    const AcpEvent* end = find(events, AcpEventKind::turn_ended);
    REQUIRE(end);
    CHECK(end->stop_reason == "cancelled");
    CHECK(end->error.code == "acp.cancel_unanswered");
    CHECK_FALSE(c.turn_active());
}

TEST_CASE("a frozen agent is killed after the grace period") {
    AcpClientOptions o = stub_options();
    o.kill_grace_ms = 300;
    AcpClient c(o);
    open_session(c);
    REQUIRE(c.prompt("freeze").ok());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto t0 = std::chrono::steady_clock::now();
    c.shutdown();
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5));
    CHECK_FALSE(c.agent_running());
    CHECK(c.prompt("hello").code == "acp.agent_exited");
}

TEST_CASE("a silent turn is reported as stalled") {
    AcpClientOptions o = stub_options();
    o.stall_timeout_ms = 300;
    AcpClient c(o);
    open_session(c);
    REQUIRE(c.prompt("stall").ok());
    auto events = run_turn(c);
    CHECK(count(events, AcpEventKind::stalled) == 1);
    CHECK(message_text(events) == "Back.");
    CHECK(find(events, AcpEventKind::turn_ended)->stop_reason == "end_turn");
}

TEST_CASE("an agent that exits in the middle of a turn is reported") {
    AcpClient c(stub_options());
    open_session(c);
    REQUIRE(c.prompt("crash").ok());
    std::vector<AcpEvent> events = run_turn(c);
    // The turn ends first; the exit follows.
    AcpEvent ev;
    while (c.wait(&ev, 1500)) {
        events.push_back(ev);
        if (ev.kind == AcpEventKind::agent_exited) break;
    }
    CHECK(message_text(events) == "About to exit.");
    const AcpEvent* end = find(events, AcpEventKind::turn_ended);
    REQUIRE(end);
    CHECK(end->error.code == "acp.agent_exited");
    const AcpEvent* exited = find(events, AcpEventKind::agent_exited);
    REQUIRE(exited);
    REQUIRE(exited->has_exit_status);
    CHECK(exited->exit_status.exited);
    CHECK(exited->exit_status.code == 3);
    CHECK_FALSE(c.agent_running());
    CHECK(c.prompt("hello").code == "acp.agent_exited");
}

TEST_CASE("an agent that speaks another protocol version is refused") {
    AcpClient c(stub_options({"--version-reply", "2"}));
    Error e = c.start();
    CHECK(e.code == "acp.version_unsupported");
    CHECK(e.message.find("version 2") != std::string::npos);
    CHECK_FALSE(c.agent_running());
}

TEST_CASE("an agent that needs a login says so and gets no key") {
    AcpClient c(stub_options({"--auth-required"}));
    REQUIRE(c.start().ok());
    Error e = c.new_session(workspace(), {});
    CHECK(e.code == "acp.auth_required");
    CHECK(e.message.find("Stub login") != std::string::npos);
    CHECK(c.authenticate("not-advertised", 1000).code == "acp.bad_argument");
}

TEST_CASE("a turn that fails or ends badly is reported, and the session goes on") {
    AcpClient c(stub_options());
    open_session(c);
    REQUIRE(c.prompt("error").ok());
    auto events = run_turn(c);
    const AcpEvent* end = find(events, AcpEventKind::turn_ended);
    REQUIRE(end);
    CHECK(end->error.code == "acp.prompt_failed");
    CHECK(end->error.message.find("on purpose") != std::string::npos);

    REQUIRE(c.prompt("nostop").ok());
    events = run_turn(c);
    end = find(events, AcpEventKind::turn_ended);
    REQUIRE(end);
    CHECK(end->error.code == "acp.bad_response");

    REQUIRE(c.prompt("still alive").ok());
    CHECK(message_text(run_turn(c)) == "You said: still alive");
}

TEST_CASE("a missing agent or one that is not an ACP agent fails to start") {
    AcpClientOptions o;
    o.agent_argv = {"caliper-no-such-agent"};
    AcpClient missing(o);
    CHECK(missing.start().code == "process.not_found");

    AcpClientOptions none;
    AcpClient empty(none);
    CHECK(empty.start().code == "acp.bad_argument");

    // A program that writes noise and exits: initialize fails, and its stderr
    // is kept as a bounded log.
    AcpClientOptions noisy;
    noisy.agent_argv = {kChild, "spew", "100000"};
    noisy.stderr_log_bytes = 1000;
    noisy.request_timeout_ms = 5000;
    AcpClient bad(noisy);
    Error e = bad.start();
    CHECK(e.code == "acp.agent_exited");
    CHECK(bad.stderr_log().size() <= 1000);
    CHECK(bad.stderr_log().size() > 0);
}

TEST_CASE("the event queue is bounded and says when it dropped events") {
    AcpClientOptions o = stub_options();
    o.max_events = 4;
    AcpClient c(o);
    open_session(c);
    REQUIRE(c.prompt("hello").ok());
    for (int i = 0; i < 100 && c.turn_active(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    AcpEvent ev;
    REQUIRE(c.poll(&ev));
    CHECK(ev.kind == AcpEventKind::protocol_fault);
    CHECK(ev.error.code == "acp.events_dropped");
    std::vector<AcpEvent> rest;
    while (c.poll(&ev)) rest.push_back(ev);
    CHECK(rest.size() <= 4);
    REQUIRE(find(rest, AcpEventKind::turn_ended));
    CHECK(find(rest, AcpEventKind::current_mode));
}
