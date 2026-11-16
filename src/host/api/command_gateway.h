#pragma once
// ============================================================================
// command_gateway.h: how Caliper API reaches the command layer
// (docs/references/api.md, "The command layer"). The server (api_server.h)
// serves GET /v0/commands, POST /v0/commands/{name} and GET /v0/approvals/{id}
// only while a CommandGateway is set on it; without one they answer 404, which
// tells caliper-mcp that the app has no command layer.
//
// The server knows nothing of the builder's types: commands, arguments and
// results cross this interface as JSON text, so caliper_api builds without the
// builder. session_gateway.h is the gateway over a caliper_builder::Session,
// compiled only when CALIPER_WITH_BUILDER is on.
//
// THREADS. table_json() runs on the server's thread and must not wait for the
// frame thread. submit() and approval_json() run on the frame thread, inside
// Server::take(), so a command request waits for the frame thread like every
// call of API v0. submit() hands the command to the command layer and returns
// at once; the server then asks the returned PendingCommand whether it is
// ready, from take() on the frame thread and from its own thread, never
// waiting, and answers the request once it is. A PendingCommand must therefore
// stay valid on its own, after the gateway and the command layer are gone.
// No function here may throw.
// ============================================================================
#include <memory>
#include <string>

namespace caliper_api {

// One command request, as the server hands it to the command layer.
struct CommandRequest {
    std::string id;            // the invocation id the server assigned ("api-<n>")
    std::string name;          // the command's name, from the path
    std::string args;          // the body: JSON text whose value is an object, at most 1 MiB
    std::string origin;        // "api", "mcp" or "agent"
    std::string group;         // from X-Caliper-Group; "" for none
    // How long a command that needs approval waits for the person's decision
    // (mcp and agent); 0 answers command.needs_approval at once (api).
    int approval_wait_ms = 0;
};

// A command handed to the command layer, whose result may not be there yet.
class PendingCommand {
public:
    virtual ~PendingCommand() = default;
    // True once the command layer has answered. Never waits.
    virtual bool ready() = 0;
    // The result envelope as JSON text: {"id", "ok", "result"} or
    // {"id", "ok", "error": {"code", "message", "where"[, "approval"]}}.
    // Called once, after ready() returned true.
    virtual std::string envelope() = 0;
};

// A result envelope that fails with `code`: {"id", "ok": false, "error":
// {"code", "message", "where"}} (in api_server.cpp).
std::string command_error_json(const std::string& id, const std::string& code, const std::string& message,
                               const std::string& where);

class CommandGateway {
public:
    virtual ~CommandGateway() = default;
    // The descriptor table, {"version": 0, "commands": [...]}, for
    // GET /v0/commands. On the server's thread.
    virtual std::string table_json() const = 0;
    // Hands one command to the command layer. On the frame thread.
    virtual std::unique_ptr<PendingCommand> submit(CommandRequest request) = 0;
    // {"approval": id, "state": "pending" | "decided" | "unknown",
    //  "result": <envelope> | null}, for GET /v0/approvals/{id}. On the
    // frame thread.
    virtual std::string approval_json(const std::string& approval_id) = 0;
};

}  // namespace caliper_api
