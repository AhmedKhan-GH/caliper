// caliper_acp_stub_agent: the agent side of ACP version 1, for tests. It needs
// no model and no network. It answers initialize and session/new, remembers
// what the client sent, and on session/prompt plays the script that the
// prompt's first word names:
//
//   hello          a thought, a plan, message chunks, a command list, a mode
//                  change and an update kind no client knows yet
//   edit           a tool call with a diff, then a permission request; it
//                  applies, refuses or stops according to the answer
//   mcp            reports the cwd, the MCP servers and the _meta of
//                  session/new as JSON (_meta is null when there was none)
//   caps           reports the clientCapabilities and clientInfo of initialize
//   slow           streams a chunk every 20 ms for 4 s, honouring session/cancel
//   probe          calls fs/read_text_file and terminal/create and reports the
//                  error codes the client answered with
//   withdraw       asks permission, then withdraws the request with
//                  $/cancel_request and reports the outcome it got
//   malformed      writes broken lines of many kinds, then carries on
//   oversize [N]   writes one message of N bytes (default 9 MiB), then carries on
//   stall          stays silent for 1.5 s, then ends the turn
//   hang           never answers and ignores session/cancel
//   freeze         like hang, and also ignores SIGTERM and the end of its input
//   crash          sends one chunk and exits with code 3 in the middle of the turn
//   error          answers session/prompt with a JSON-RPC error
//   nostop         ends the turn without a stop reason
//   anything else  echoes the prompt and ends the turn
//
// session/new answers with the request's _meta echoed in its own _meta, when
// the request had one.
//
// Options: --version-reply <n> answers initialize with protocol version n;
// --auth-required makes session/new fail with -32000.
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#ifndef _WIN32
#include <unistd.h>
#endif

using nlohmann::json;

namespace {

std::mutex g_out_mutex;
std::mutex g_state_mutex;
json g_init_params;
json g_session_params;
std::string g_session_id = "stub-session-1";
std::atomic<bool> g_cancelled{false};
std::atomic<bool> g_frozen{false};
std::atomic<int> g_next_id{1000};
std::map<int, std::promise<json>> g_waiting;
int g_version_reply = 1;
bool g_auth_required = false;

void write_raw(const std::string& line) {
    std::lock_guard<std::mutex> lock(g_out_mutex);
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void send(const json& msg) { write_raw(msg.dump()); }

void respond(const json& id, const json& result) { send({{"jsonrpc", "2.0"}, {"id", id}, {"result", result}}); }

void respond_error(const json& id, int code, const std::string& message) {
    send({{"jsonrpc", "2.0"}, {"id", id}, {"error", {{"code", code}, {"message", message}}}});
}

void update(const json& u) {
    send({{"jsonrpc", "2.0"},
          {"method", "session/update"},
          {"params", {{"sessionId", g_session_id}, {"update", u}}}});
}

void chunk(const std::string& text, const char* kind = "agent_message_chunk") {
    update({{"sessionUpdate", kind}, {"content", {{"type", "text"}, {"text", text}}}});
}

void end_turn(const json& id, const std::string& reason = "end_turn") { respond(id, {{"stopReason", reason}}); }

// Sends a request to the client and waits for its response (the whole
// response object), or returns null after `timeout_ms`.
json ask_client(const std::string& method, const json& params, int timeout_ms, std::atomic<int>* id_out = nullptr) {
    const int id = g_next_id++;
    std::future<json> f;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        f = g_waiting[id].get_future();
    }
    if (id_out) *id_out = id;
    send({{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}});
    if (f.wait_for(std::chrono::milliseconds(timeout_ms)) != std::future_status::ready) return nullptr;
    return f.get();
}

std::string session_cwd() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return g_session_params.value("cwd", std::string("/"));
}

json edit_tool_call(const std::string& id) {
    const std::string path = session_cwd() + "/demo.spec.toml";
    return {{"toolCallId", id},
            {"title", "Edit demo.spec.toml"},
            {"kind", "edit"},
            {"status", "pending"},
            {"content",
             json::array({{{"type", "diff"},
                           {"path", path},
                           {"oldText", "title = \"Old\"\n"},
                           {"newText", "title = \"New\"\n"}}})},
            {"locations", json::array({{{"path", path}, {"line", 1}}})},
            {"rawInput", {{"file", "demo.spec.toml"}}}};
}

json permission_options() {
    return json::array({{{"optionId", "allow"}, {"name", "Allow"}, {"kind", "allow_once"}},
                        {{"optionId", "allow-always"}, {"name", "Always allow"}, {"kind", "allow_always"}},
                        {{"optionId", "reject"}, {"name", "Reject"}, {"kind", "reject_once"}}});
}

void run_turn(json id, std::string text) {
    const std::string word = text.substr(0, text.find(' '));
    if (word == "hello") {
        chunk("Thinking about hello.", "agent_thought_chunk");
        update({{"sessionUpdate", "plan"},
                {"entries", json::array({{{"content", "Say hello"}, {"priority", "high"}, {"status", "in_progress"}},
                                         {{"content", "Stop"}, {"priority", "low"}, {"status", "pending"}}})}});
        chunk("Hi");
        chunk(", ");
        chunk("world.");
        update({{"sessionUpdate", "available_commands_update"},
                {"availableCommands",
                 json::array({{{"name", "spec"}, {"description", "Show the open spec"}, {"input", {{"hint", "instrument id"}}}},
                              {{"name", "help"}, {"description", "List the commands"}}})}});
        update({{"sessionUpdate", "current_mode_update"}, {"currentModeId", "plan"}});
        update({{"sessionUpdate", "future_update_kind"}, {"value", 1}});
        end_turn(id);
    } else if (word == "command") {
        // A call of a Caliper command made by the agent itself, which names it
        // in _meta (caliper-agent does), then one whose _meta is not usable.
        update({{"sessionUpdate", "tool_call"}, {"toolCallId", "call_c"}, {"title", "catalog_list"}, {"kind", "read"},
                {"status", "pending"}, {"_meta", {{"caliper", {{"command", "catalog.list"}}}}}});
        update({{"sessionUpdate", "tool_call_update"}, {"toolCallId", "call_c"}, {"title", "catalog.list"},
                {"status", "completed"}});
        update({{"sessionUpdate", "tool_call"}, {"toolCallId", "call_d"}, {"title", "spec_get"}, {"kind", "read"},
                {"status", "pending"}, {"_meta", {{"caliper", {{"command", 7}}}}}});
        end_turn(id);
    } else if (word == "edit") {
        json call = edit_tool_call("call_1");
        json tc = call;
        tc["sessionUpdate"] = "tool_call";
        update(tc);
        json answer = ask_client("session/request_permission",
                                 {{"sessionId", g_session_id}, {"toolCall", call}, {"options", permission_options()}},
                                 60000);
        const json outcome = answer.is_object() && answer.contains("result") ? answer["result"].value("outcome", json())
                                                                             : json();
        const std::string kind = outcome.is_object() ? outcome.value("outcome", std::string()) : std::string();
        const std::string option = outcome.is_object() ? outcome.value("optionId", std::string()) : std::string();
        if (kind == "selected" && (option == "allow" || option == "allow-always")) {
            update({{"sessionUpdate", "tool_call_update"}, {"toolCallId", "call_1"}, {"status", "in_progress"}});
            update({{"sessionUpdate", "tool_call_update"},
                    {"toolCallId", "call_1"},
                    {"status", "completed"},
                    {"content", json::array({{{"type", "content"}, {"content", {{"type", "text"}, {"text", "Applied."}}}}})}});
            chunk("Applied the change.");
            end_turn(id);
        } else if (kind == "selected") {
            update({{"sessionUpdate", "tool_call_update"}, {"toolCallId", "call_1"}, {"status", "failed"}});
            chunk("Left the spec unchanged.");
            end_turn(id);
        } else {
            update({{"sessionUpdate", "tool_call_update"}, {"toolCallId", "call_1"}, {"status", "failed"}});
            end_turn(id, "cancelled");
        }
    } else if (word == "mcp") {
        json report;
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            report = {{"cwd", g_session_params.value("cwd", "")},
                      {"mcpServers", g_session_params.value("mcpServers", json::array())},
                      {"_meta", g_session_params.value("_meta", json())}};
        }
        chunk(report.dump());
        end_turn(id);
    } else if (word == "caps") {
        json report;
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            report = {{"protocolVersion", g_init_params.value("protocolVersion", 0)},
                      {"clientCapabilities", g_init_params.value("clientCapabilities", json())},
                      {"clientInfo", g_init_params.value("clientInfo", json())}};
        }
        chunk(report.dump());
        end_turn(id);
    } else if (word == "slow") {
        for (int i = 0; i < 200; ++i) {
            if (g_cancelled) {
                chunk("Stopped.");
                end_turn(id, "cancelled");
                return;
            }
            chunk("tick " + std::to_string(i) + " ");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        end_turn(id);
    } else if (word == "probe") {
        const std::string cwd = session_cwd();
        json fs = ask_client("fs/read_text_file", {{"sessionId", g_session_id}, {"path", cwd + "/x.txt"}}, 5000);
        json term = ask_client("terminal/create", {{"sessionId", g_session_id}, {"command", "ls"}}, 5000);
        auto code_of = [](const json& r) {
            return r.is_object() && r.contains("error") ? r["error"].value("code", 0) : 0;
        };
        chunk(json({{"fs/read_text_file", code_of(fs)}, {"terminal/create", code_of(term)}}).dump());
        end_turn(id);
    } else if (word == "withdraw") {
        json call = edit_tool_call("call_2");
        std::atomic<int> request_id{0};
        std::thread asker([&] {
            json answer = ask_client("session/request_permission",
                                     {{"sessionId", g_session_id}, {"toolCall", call}, {"options", permission_options()}},
                                     10000, &request_id);
            const json outcome = answer.is_object() && answer.contains("result")
                                     ? answer["result"].value("outcome", json())
                                     : json();
            chunk("withdrawn: " + (outcome.is_object() ? outcome.value("outcome", std::string("?")) : std::string("none")));
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        send({{"jsonrpc", "2.0"}, {"method", "$/cancel_request"}, {"params", {{"requestId", request_id.load()}}}});
        asker.join();
        end_turn(id);
    } else if (word == "malformed") {
        write_raw("{not json");
        write_raw("42");
        write_raw("[1, 2]");
        write_raw(R"({"jsonrpc":"2.0","id":999999,"result":{}})");
        write_raw(R"({"jsonrpc":"1.0","method":"session/update","params":{}})");
        write_raw(std::string(300, '[') + std::string(300, ']'));
        write_raw(R"({"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"stub-session-1"}})");
        write_raw(R"({"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"stub-session-1","update":{"sessionUpdate":"tool_call","toolCallId":"x"}}})");
        write_raw(R"({"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"other","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"wrong session"}}}})");
        write_raw("\xff\xfe{\"jsonrpc\":\"2.0\"}");
        write_raw("");
        chunk("Still here.");
        end_turn(id);
    } else if (word == "oversize") {
        std::size_t n = 9u << 20;
        const auto sp = text.find(' ');
        if (sp != std::string::npos) n = std::strtoull(text.c_str() + sp + 1, nullptr, 10);
        chunk(std::string(n, 'x'));
        chunk("After.");
        end_turn(id);
    } else if (word == "stall") {
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        chunk("Back.");
        end_turn(id);
    } else if (word == "hang") {
        // Never answers.
    } else if (word == "freeze") {
#ifndef _WIN32
        std::signal(SIGTERM, SIG_IGN);
#endif
        g_frozen = true;
    } else if (word == "crash") {
        chunk("About to exit.");
        std::fflush(stdout);
        std::_Exit(3);
    } else if (word == "error") {
        respond_error(id, -32603, "The stub failed on purpose.");
    } else if (word == "nostop") {
        respond(id, json::object());
    } else {
        chunk("You said: " + text);
        end_turn(id);
    }
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--version-reply" && i + 1 < argc) g_version_reply = std::atoi(argv[++i]);
        else if (a == "--auth-required") g_auth_required = true;
    }
    std::thread turn;
    std::string line;
    while (std::getline(std::cin, line)) {
        json msg = json::parse(line, nullptr, false);
        if (!msg.is_object()) continue;
        if (msg.contains("method")) {
            const std::string method = msg.value("method", "");
            const json params = msg.value("params", json::object());
            const bool is_request = msg.contains("id");
            const json id = is_request ? msg["id"] : json();
            if (method == "initialize") {
                {
                    std::lock_guard<std::mutex> lock(g_state_mutex);
                    g_init_params = params;
                }
                respond(id, {{"protocolVersion", g_version_reply},
                             {"agentCapabilities",
                              {{"loadSession", false},
                               {"promptCapabilities", {{"image", false}, {"audio", false}, {"embeddedContext", false}}},
                               {"mcpCapabilities", {{"http", false}, {"sse", false}}}}},
                             {"authMethods", json::array({{{"id", "stub-login"}, {"name", "Stub login"}}})},
                             {"agentInfo", {{"name", "caliper-acp-stub"}, {"title", "Caliper ACP stub agent"}, {"version", "1"}}}});
            } else if (method == "session/new") {
                if (g_auth_required) {
                    respond_error(id, -32000, "Authentication required");
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lock(g_state_mutex);
                    g_session_params = params;
                }
                json answer = {{"sessionId", g_session_id},
                               {"modes", {{"currentModeId", "default"},
                                          {"availableModes", json::array({{{"id", "default"}, {"name", "Default"}},
                                                                          {{"id", "plan"}, {"name", "Plan"}}})}}}};
                if (params.contains("_meta")) answer["_meta"] = {{"echo", params["_meta"]}};
                respond(id, answer);
            } else if (method == "session/prompt") {
                if (turn.joinable()) turn.join();
                g_cancelled = false;
                std::string text;
                const json prompt = params.value("prompt", json::array());
                for (const auto& block : prompt)
                    if (block.is_object() && block.value("type", "") == "text") text += block.value("text", "");
                turn = std::thread(run_turn, id, text);
            } else if (method == "session/cancel") {
                g_cancelled = true;
            } else if (is_request) {
                respond_error(id, -32601, "Method not found: " + method);
            }
        } else if (msg.contains("id") && msg["id"].is_number_integer()) {
            const int id = msg["id"].get<int>();
            std::lock_guard<std::mutex> lock(g_state_mutex);
            auto it = g_waiting.find(id);
            if (it != g_waiting.end()) {
                it->second.set_value(msg);
                g_waiting.erase(it);
            }
        }
    }
    // The end of the input: a frozen stub keeps running until it is killed.
    while (g_frozen) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (turn.joinable()) turn.detach();
    std::fflush(stdout);
    std::_Exit(0);
}
