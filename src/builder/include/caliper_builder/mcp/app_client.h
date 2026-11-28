#pragma once
// caliper_builder::mcp::AppClient: how caliper-mcp reaches the running app
// through Caliper API (design/mcp-server.md, "Reaching the app").
//
// - The API file is read at every call, so the client follows the app when it
//   restarts on another port with another token. When a request is refused
//   with 401 or the connection is refused, the file is read once more and the
//   request is sent again if the file now names another port or token; neither
//   fault means the request ran, so the retry cannot run a command twice.
// - table() gives the command table the tools come from: the app's, when
//   GET /v0/commands answers with a sound table, otherwise the one compiled
//   into caliper-mcp. The answer is kept for as long as the API file names the
//   same port and token.
// - call() runs one tool: it checks the arguments against the command's schema
//   with the commands library's validator, then sends app.info, app.instruments,
//   app.launch and capture.frame to their API v0 endpoints and every other
//   command to POST /v0/commands/{name} with X-Caliper-Origin. The outcome is
//   an MCP CallToolResult (tools.h), a tool error for every fault.
//
// The session token goes only into the Authorization header of requests to
// 127.0.0.1. It is never logged and never put into a result; known_tokens()
// lets the server strike it from anything it writes, as a second guard.
//
// Thread-safe: call() and table() may run on several threads at once.
#include <caliper_builder/commands/command.h>
#include <caliper_builder/mcp/api_file.h>
#include <caliper_builder/mcp/loopback_http.h>
#include <caliper_builder/mcp/tools.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace caliper_builder::mcp {

struct AppClientOptions {
    std::string api_file;                    // the API file to read
    std::string origin = "mcp";              // the X-Caliper-Origin of command calls: "mcp" or "agent"
    int command_timeout_ms = 130000;         // POST /v0/commands/{name}, which may wait for approval
    int app_timeout_ms = 15000;              // the API v0 endpoints
    int table_timeout_ms = 10000;            // GET /v0/commands
    std::size_t max_response_bytes = 64u << 20;
    std::size_t max_table_bytes = 8u << 20;
    std::vector<CommandDescriptor> builtin;  // empty: command_table()
};

class AppClient {
public:
    using Log = std::function<void(const std::string&)>;

    AppClient(AppClientOptions options, Log log);

    // The table the tools come from now. `changed` is set when its
    // fingerprint differs from the table this function returned before.
    std::shared_ptr<const ToolTable> table(bool* changed = nullptr);

    // The table compiled into caliper-mcp.
    std::shared_ptr<const ToolTable> builtin_table() const { return builtin_; }

    // Runs one tool call and returns its CallToolResult (without resultType).
    // `args` must be a JSON object.
    nlohmann::json call(const CommandDescriptor& descriptor, const nlohmann::json& args,
                        const std::atomic<bool>* cancel = nullptr);

    // Every session token read from the API file so far.
    std::vector<std::string> known_tokens() const;

    const AppClientOptions& options() const { return options_; }

private:
    // One request with the token added, after the retry rule above.
    struct Sent {
        Error file_error;        // the API file could not be read; nothing was sent
        ApiEndpoint endpoint;    // the endpoint the answer came from
        HttpResponse response;
    };
    ApiFileRead read_endpoint();
    Sent send(HttpRequest request, const HttpLimits& limits, const std::atomic<bool>* cancel);

    AppClientOptions options_;
    Log log_;
    std::shared_ptr<const ToolTable> builtin_;

    mutable std::mutex mu_;   // the fields below
    std::set<std::string> tokens_;
    bool have_cached_ = false;
    ApiEndpoint cached_endpoint_;
    std::shared_ptr<const ToolTable> cached_table_;
    std::string last_fingerprint_;
    std::string last_source_note_;

    std::mutex table_mu_;   // one table refresh at a time
};

}  // namespace caliper_builder::mcp
