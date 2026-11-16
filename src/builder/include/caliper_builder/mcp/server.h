#pragma once
// caliper_builder::mcp::Server: the protocol engine of caliper-mcp, an MCP
// server over newline-delimited JSON-RPC 2.0 (docs/design/mcp-server.md).
//
// It serves both eras of MCP on one connection:
// - modern, revision 2026-07-28: every request carries
//   io.modelcontextprotocol/protocolVersion and clientCapabilities in
//   params._meta and is served on its own; server/discover and
//   subscriptions/listen are offered; results carry resultType, the server's
//   identity and, where the revision asks, ttlMs and cacheScope;
// - legacy, revisions 2025-11-25 and 2025-06-18: an `initialize` handshake
//   picks the revision for the rest of the process; an older request gets
//   2025-11-25 back, as the legacy negotiation rule says.
//
// Requests that may wait (tools/list, tools/call, resources/list and
// resources/read) run on worker threads, at most `max_in_flight` at once, so a
// tool call held for approval never blocks ping or a cancellation. Every other
// request is answered on the caller's thread.
//
// Every incoming line is hostile input: one nested deeper than 128 levels, one
// that is not JSON, a batch, and a message that is not JSON-RPC 2.0 are
// answered with the JSON-RPC error they call for (without an id when none can
// be read). The sink receives each outgoing message as one line of JSON
// without its newline; outgoing text is scrubbed of every session token the
// server has read, so a token cannot leave the process even by mistake.
//
// The stdio loop, run_stdio_server(), reads standard input, drops a line
// longer than `max_message_bytes` (answering -32600), writes messages to
// standard output and diagnostics to standard error, and returns when its
// input ends, after shutdown().
#include <caliper_builder/commands/command.h>
#include <caliper_builder/error.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder::mcp {

constexpr const char* kModernProtocolVersion = "2026-07-28";
// The revisions served, newest first; the first legacy one is the answer to
// an initialize that asks for a revision not in this list.
const std::vector<std::string>& supported_protocol_versions();

struct ServerOptions {
    std::string api_file;           // the API file; empty means default_api_file_path()
    std::string docs_dir;           // --docs; empty: no docs resources
    std::string sdk_dir;            // --sdk; empty: no SDK resources
    std::string origin = "mcp";     // X-Caliper-Origin: "mcp", or "agent" when the app's agent pane runs it
    int command_timeout_ms = 130000;
    int app_timeout_ms = 15000;
    std::size_t max_message_bytes = 8u << 20;
    std::size_t max_in_flight = 32;
    std::size_t max_subscriptions = 16;
    int shutdown_grace_ms = 2000;   // how long shutdown() lets running calls finish
    std::vector<CommandDescriptor> builtin_table;   // empty: command_table()
    std::string spec_schema_text;   // empty: the schema compiled in, when there is one
};

// The version of caliper-mcp that serverInfo names.
const char* server_version();

class Server {
public:
    // Receives one outgoing message, without its newline; returns false when
    // the output is closed. Called under the server's output lock.
    using Sink = std::function<bool(const std::string& line)>;
    // Receives one diagnostic line for a person; never a protocol message.
    using Log = std::function<void(const std::string& line)>;

    Server(ServerOptions options, Sink sink, Log log);
    ~Server();   // shutdown()
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Checks the options and opens the resource folders.
    Error open();

    // Handles one incoming line (without its newline).
    void handle_line(std::string_view line);
    // Answers a line that was dropped for being longer than max_message_bytes.
    void handle_oversized_line();

    // Lets running requests finish for up to shutdown_grace_ms, then cancels
    // the rest (which answer with the tool error mcp.cancelled), waits for
    // their threads, and ends every subscription with its closing result. A
    // request the client cancelled is never answered. Idempotent.
    void shutdown();

    // True once the sink has reported that the output is closed.
    bool output_closed() const;
    // Requests running on worker threads now.
    std::size_t in_flight() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// Runs the server on standard input and output until the input ends. Returns
// the process exit code.
int run_stdio_server(const ServerOptions& options);

}  // namespace caliper_builder::mcp
