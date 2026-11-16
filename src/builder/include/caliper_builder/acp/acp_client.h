#pragma once
// caliper_builder::acp::AcpClient: the client side of the Agent Client
// Protocol (https://agentclientprotocol.com), protocol version 1, over the
// stdio of an agent that it starts as a child process. It knows nothing about
// which agent it runs: Claude Code through its adapter, Codex, Gemini CLI, a
// Caliper-built agent or the stub of the tests all look the same to it.
//
// The life of a client:
//   start()        spawns the agent and sends `initialize`, offering version 1
//                  and declining the file-system and terminal capabilities;
//   new_session()  sends `session/new` with an absolute working folder and the
//                  stdio MCP servers the agent should connect to;
//   prompt()       starts a turn and returns at once; the agent's
//                  `session/update` notifications, its permission requests and
//                  the end of the turn arrive as AcpEvents;
//   poll()/wait()  take events from a thread-safe queue (the app polls on its
//                  frame thread, between two frames);
//   respond_permission() answers a permission request with one of its options;
//   cancel()       sends `session/cancel`, answers every open permission request
//                  as cancelled and waits for the turn to end (an agent that
//                  does not end it within `cancel_timeout_ms` gets a synthetic
//                  end with acp.cancel_unanswered);
//   shutdown()     closes the agent's input, then terminates it after a grace
//                  period. The destructor calls it.
//
// start(), authenticate() and new_session() block for up to
// `request_timeout_ms`; call them off the frame thread. Every other function
// returns at once and may be called from any thread.
//
// The client never handles an API key: the agent uses its own login. Extra
// variables for the agent's environment (`agent_env`) are passed to the child
// and never logged. Requests the client declined (`fs/*`, `terminal/*`) are
// answered with method-not-found and reported as a protocol_fault event.
// The agent's stderr is kept as a bounded log.
//
// Error codes: acp.bad_argument, acp.not_started, acp.already_started,
// acp.version_unsupported, acp.bad_response, acp.auth_required,
// acp.no_session, acp.turn_active, acp.no_turn, acp.unknown_permission,
// acp.unknown_option, acp.agent_exited, acp.cancel_unanswered,
// acp.prompt_failed, acp.declined_method, acp.bad_update, acp.stalled, and
// the process.* and rpc.* codes of the layers below.
#include <caliper_builder/acp/jsonrpc.h>
#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caliper_builder::acp {

// The ACP version this client speaks. Version 2 is a draft (see
// docs/design/acp-integration.md); it is added beside version 1 when it is
// stable, never in place of it.
constexpr int kAcpProtocolVersion = 1;

// A stdio MCP server handed to the agent in `session/new`. A command without a
// slash is resolved on PATH before it is sent, since ACP asks for an absolute
// path. `env` values are sent to the agent, which starts the server; they are
// never logged here.
struct McpServer {
    std::string name;
    std::string command;
    std::vector<std::string> args;
    std::vector<EnvVar> env;
};

struct AcpClientOptions {
    std::vector<std::string> agent_argv;        // the agent's command and arguments
    std::vector<EnvVar> agent_env;              // added to the agent's environment
    std::vector<std::string> agent_env_remove;  // removed from the inherited environment
    std::string agent_cwd;                      // where the agent process starts; empty: this process's
    std::string client_name = "caliper";
    std::string client_title = "Caliper";
    std::string client_version = "0.0.0";
    int request_timeout_ms = 60000;   // initialize, session/new
    int cancel_timeout_ms = 10000;    // from session/cancel to the end of the turn
    int stall_timeout_ms = 120000;    // silence in a turn before a stalled event; <= 0: never
    int kill_grace_ms = 3000;         // from SIGTERM to SIGKILL in shutdown()
    std::size_t max_message_bytes = 8u << 20;
    std::size_t max_events = 20000;   // queued events; past it the oldest chunks are dropped
    std::size_t stderr_log_bytes = 64u << 10;
};

// What the agent said about itself in `initialize`.
struct AgentInfo {
    int protocol_version = 0;
    std::string name;
    std::string title;
    std::string version;
    bool load_session = false;
    bool mcp_http = false;
    bool mcp_sse = false;
    bool prompt_image = false;
    bool prompt_audio = false;
    bool prompt_embedded_context = false;
    nlohmann::json capabilities;   // agentCapabilities as received
    nlohmann::json auth_methods;   // authMethods as received (an array)
};

// What `session/new` returned.
struct SessionInfo {
    std::string id;
    nlohmann::json modes;           // the modes object, or null
    nlohmann::json config_options;  // the configOptions array, or null
};

// One item of a tool call's content.
struct ToolCallContent {
    std::string type;               // "content", "diff" or "terminal" (or what the agent sent)
    std::string content_type;       // for "content": the block's type ("text", "image", ...)
    std::string text;               // for a text block
    std::string path;               // for "diff": the absolute path of the file
    bool has_old_text = false;      // false for a new file
    std::string old_text;
    std::string new_text;
    std::string terminal_id;        // for "terminal"
};

struct ToolLocation {
    std::string path;
    int line = 0;                   // 1-based; 0 when absent
};

// A tool call or an update of one. In an update only the fields marked as
// present changed.
struct ToolCallInfo {
    std::string id;
    std::string title;
    std::string kind;               // read, edit, delete, move, search, execute, think, fetch, switch_mode, other
    std::string status;             // pending, in_progress, completed, failed
    bool has_title = false;
    bool has_kind = false;
    bool has_status = false;
    bool has_content = false;
    bool has_locations = false;
    std::vector<ToolCallContent> content;
    std::vector<ToolLocation> locations;
    nlohmann::json raw_input;
    nlohmann::json raw_output;
};

struct PlanEntry {
    std::string content;
    std::string priority;           // high, medium, low
    std::string status;             // pending, in_progress, completed
};

struct CommandInfo {
    std::string name;
    std::string description;
    std::string hint;               // the input hint, if the command takes input
};

struct PermissionOption {
    std::string id;
    std::string name;
    std::string kind;               // allow_once, allow_always, reject_once, reject_always
};

enum class AcpEventKind {
    user_message_chunk,
    agent_message_chunk,
    agent_thought_chunk,
    tool_call,
    tool_call_update,
    plan,
    available_commands,
    current_mode,
    other_update,          // any other session/update kind, kept raw (config options, usage, ...)
    permission_request,    // answer with respond_permission()
    permission_withdrawn,  // the agent cancelled it, or cancel() answered it
    turn_ended,            // stop_reason set; error set when the turn failed
    protocol_fault,        // a malformed, oversized or refused message; the session goes on
    stalled,               // no message for stall_timeout_ms in a turn
    agent_exited,          // the agent's process ended or closed its output
};

const char* event_kind_name(AcpEventKind kind);

struct AcpEvent {
    AcpEventKind kind = AcpEventKind::other_update;
    std::string session_id;
    std::string update_kind;        // the sessionUpdate value as received
    std::string content_type;       // of a chunk: "text", "image", "resource_link", ...
    std::string text;               // of a text chunk
    ToolCallInfo tool_call;         // tool_call, tool_call_update, permission_request
    std::vector<PlanEntry> plan;
    std::vector<CommandInfo> commands;
    std::string mode_id;
    std::uint64_t permission_id = 0;
    std::vector<PermissionOption> options;
    std::string stop_reason;        // end_turn, max_tokens, max_turn_requests, refusal, cancelled
    Error error;
    bool has_exit_status = false;
    ExitStatus exit_status;
    nlohmann::json raw;             // the update object, or the request's params
};

// The JSON form of an event, as the harness prints it: {"type": <kind>, ...}.
nlohmann::json event_to_json(const AcpEvent& event);

class AcpClient {
public:
    explicit AcpClient(AcpClientOptions options);
    ~AcpClient();
    AcpClient(const AcpClient&) = delete;
    AcpClient& operator=(const AcpClient&) = delete;

    Error start(AgentInfo* info = nullptr);
    // Asks the agent to run its own login flow for an advertised method of
    // type "agent". No key is ever sent.
    Error authenticate(const std::string& method_id, int timeout_ms);
    Error new_session(const std::string& cwd, const std::vector<McpServer>& mcp_servers,
                      SessionInfo* session = nullptr);

    Error prompt(const std::string& text);
    Error prompt_blocks(const nlohmann::json& content_blocks);
    Error cancel();
    Error respond_permission(std::uint64_t permission_id, const std::string& option_id);

    bool poll(AcpEvent* event);
    bool wait(AcpEvent* event, int timeout_ms);

    bool turn_active() const;
    bool agent_running() const;
    const AgentInfo& agent() const;
    std::string session_id() const;
    std::string stderr_log() const;
    std::string describe_command() const;  // the agent's command line, with variable names only

    void shutdown();

    struct State;

private:
    std::shared_ptr<State> state_;
};

}  // namespace caliper_builder::acp
