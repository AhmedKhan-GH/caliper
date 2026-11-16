// The command gateway over the builder session (session_gateway.h).
#include "api/session_gateway.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/session.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <future>
#include <string>
#include <utility>

namespace caliper_api {

namespace {

using caliper_builder::CommandInvocation;
using caliper_builder::CommandOrigin;
using caliper_builder::CommandResult;
using caliper_builder::Error;
using nlohmann::json;

// JSON text of a value; a string that is not UTF-8 is written with U+FFFD in
// its place rather than failing.
std::string dump(const json& value) {
    try {
        return value.dump(-1, ' ', false, json::error_handler_t::replace);
    } catch (...) {
        return "null";
    }
}

std::string envelope_text(const CommandResult& r) {
    try {
        return dump(caliper_builder::command_result_to_json(r));
    } catch (...) {
        return command_error_json(r.id, "command.handler_failed", "The result could not be written as JSON.", "");
    }
}

// An answer known when the command is handed over.
class ReadyCommand final : public PendingCommand {
public:
    explicit ReadyCommand(std::string text) : text_(std::move(text)) {}
    bool ready() override { return true; }
    std::string envelope() override { return std::move(text_); }

private:
    std::string text_;
};

// A command in the session's queue.
class QueuedCommand final : public PendingCommand {
public:
    explicit QueuedCommand(std::future<CommandResult> f) : f_(std::move(f)) {}
    bool ready() override {
        return f_.valid() && f_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
    std::string envelope() override {
        try {
            return envelope_text(f_.get());
        } catch (...) {
            return command_error_json("", "command.shutdown", "The command layer went away before this completed.",
                                      "");
        }
    }

private:
    std::future<CommandResult> f_;
};

class SessionGateway final : public CommandGateway {
public:
    explicit SessionGateway(caliper_builder::Session& session)
        : session_(session), table_(dump(caliper_builder::command_table_to_json())) {}

    std::string table_json() const override { return table_; }

    std::unique_ptr<PendingCommand> submit(CommandRequest request) override {
        try {
            json args = json::parse(request.args, nullptr, false);
            if (args.is_discarded() || !args.is_object())
                return failed(request.id, "command.invalid_args", "The arguments are a JSON object.", "args");
            const auto origin = caliper_builder::command_origin_from_name(request.origin);
            if (!origin || (*origin != CommandOrigin::api && *origin != CommandOrigin::mcp &&
                            *origin != CommandOrigin::agent))
                return failed(request.id, "command.invalid_invocation", "The origin is api, mcp or agent.", "origin");
            CommandInvocation inv;
            inv.id = std::move(request.id);
            inv.name = std::move(request.name);
            inv.args = std::move(args);
            inv.origin = *origin;
            inv.group = std::move(request.group);
            caliper_builder::SubmitOptions so;
            so.wait_for_approval = request.approval_wait_ms > 0;
            so.approval_timeout = std::chrono::milliseconds(request.approval_wait_ms > 0 ? request.approval_wait_ms : 0);
            return std::make_unique<QueuedCommand>(session_.queue().submit(std::move(inv), so));
        } catch (...) {
            return failed(request.id, "command.handler_failed", "The command could not be handed to the command layer.",
                          "");
        }
    }

    std::string approval_json(const std::string& approval_id) override {
        try {
            json j{{"approval", approval_id}, {"state", "unknown"}, {"result", nullptr}};
            if (const auto outcome = session_.queue().approval_outcome(approval_id)) {
                j["state"] = "decided";
                j["result"] = caliper_builder::command_result_to_json(*outcome);
            } else {
                for (const auto& p : session_.registry().pending_approvals()) {
                    if (p.approval == approval_id) {
                        j["state"] = "pending";
                        break;
                    }
                }
            }
            return dump(j);
        } catch (...) {
            return "{\"approval\":null,\"result\":null,\"state\":\"unknown\"}";
        }
    }

private:
    static std::unique_ptr<PendingCommand> failed(const std::string& id, const char* code, const char* message,
                                                  const char* where) {
        return std::make_unique<ReadyCommand>(envelope_text(CommandResult::failure(id, Error(code, message, where))));
    }

    caliper_builder::Session& session_;
    const std::string table_;
};

}  // namespace

std::shared_ptr<CommandGateway> make_session_gateway(caliper_builder::Session& session) {
    return std::make_shared<SessionGateway>(session);
}

}  // namespace caliper_api
