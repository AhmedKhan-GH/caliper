// The command gateway over the builder session (session_gateway.h).
#include "api/session_gateway.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/events.h>
#include <caliper_builder/session/session.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
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

std::string refused(const std::string& id, const char* code, const char* message, const char* where) {
    return envelope_text(CommandResult::failure(id, Error(code, message, where)));
}

class SessionGateway final : public CommandGateway {
public:
    explicit SessionGateway(caliper_builder::Session& session)
        : session_(session), table_(dump(caliper_builder::command_table_to_json())), agent_table_(agent_table()) {}

    std::string table_json() const override { return table_; }
    std::string agent_table_json() const override { return agent_table_; }

    std::unique_ptr<PendingCommand> submit(CommandRequest request) override {
        try {
            RequestInvocation r = request_invocation(std::move(request));
            if (!r.refused.empty()) return ready_command(std::move(r.refused));
            if (r.invocation.name == "events.wait") return long_poll(std::move(r.invocation));
            return queued_command(session_.queue().submit(std::move(r.invocation), r.options));
        } catch (...) {
            return ready_command(refused(request.id, "command.handler_failed",
                                         "The command could not be handed to the command layer.", ""));
        }
    }

    std::string approval_json(const std::string& approval_id) override {
        try {
            return approval_state_json(approval_id, session_.queue().approval_outcome(approval_id),
                                       session_.registry().pending_approvals());
        } catch (...) {
            return "{\"approval\":null,\"result\":null,\"state\":\"unknown\"}";
        }
    }

private:
    // events.wait as a long poll (design/instruments.md, Long Poll): the
    // server holds the request until its PendingCommand is ready, and a thread
    // of its own waits in wait_for_events(), so that neither the server nor
    // the frame thread blocks. The hold is taken here, while the session is
    // open, so that Session::close() waits until the thread has left
    // wait_for_events(); a closed session's queue and ring answer at once.
    std::unique_ptr<PendingCommand> long_poll(caliper_builder::CommandInvocation invocation) {
        auto done = std::make_shared<std::promise<CommandResult>>();
        std::future<CommandResult> result = done->get_future();
        auto hold = std::make_unique<caliper_builder::EventWaitHold>(session_.events());
        std::thread([done, hold = std::move(hold), queue = &session_.queue(), ring = &session_.events(),
                     inv = std::move(invocation)]() mutable {
            const std::string id = inv.id;
            CommandResult r = CommandResult::failure(id, Error("command.handler_failed", "events.wait failed.", ""));
            try {
                r = caliper_builder::wait_for_events(*queue, *ring, std::move(inv));
            } catch (...) {
            }
            hold.reset();   // the session may close from here on
            done->set_value(std::move(r));
        }).detach();
        return queued_command(std::move(result));
    }

    caliper_builder::Session& session_;
    const std::string table_;
    const std::string agent_table_;

    // The table without the commands agents may not run (`agents: false`).
    static std::string agent_table() {
        json table = caliper_builder::command_table_to_json();
        json kept = json::array();
        for (auto& c : table["commands"])
            if (c.value("agents", true)) kept.push_back(std::move(c));
        table["commands"] = std::move(kept);
        return dump(table);
    }
};

}  // namespace

std::shared_ptr<CommandGateway> make_session_gateway(caliper_builder::Session& session) {
    return std::make_shared<SessionGateway>(session);
}

RequestInvocation request_invocation(CommandRequest request) {
    RequestInvocation r;
    json args = json::parse(request.args, nullptr, false);
    if (args.is_discarded() || !args.is_object()) {
        r.refused = refused(request.id, "command.invalid_args", "The arguments are a JSON object.", "args");
        return r;
    }
    const auto origin = caliper_builder::command_origin_from_name(request.origin);
    if (!origin || (*origin != CommandOrigin::api && *origin != CommandOrigin::mcp && *origin != CommandOrigin::agent &&
                    *origin != CommandOrigin::viewer)) {
        r.refused =
            refused(request.id, "command.invalid_invocation", "The origin is api, mcp, agent or viewer.", "origin");
        return r;
    }
    r.invocation.id = std::move(request.id);
    r.invocation.name = std::move(request.name);
    r.invocation.args = std::move(args);
    r.invocation.origin = *origin;
    // The token's client names an agent the app started; only an agent token has one.
    if (*origin == CommandOrigin::agent) r.invocation.client = std::move(request.client);
    r.invocation.group = std::move(request.group);
    r.invocation.preview = std::move(request.preview);
    r.options.wait_for_approval = request.approval_wait_ms > 0;
    r.options.approval_timeout = std::chrono::milliseconds(request.approval_wait_ms > 0 ? request.approval_wait_ms : 0);
    return r;
}

std::unique_ptr<PendingCommand> ready_command(std::string envelope) {
    return std::make_unique<ReadyCommand>(std::move(envelope));
}

std::unique_ptr<PendingCommand> queued_command(std::future<CommandResult> result) {
    return std::make_unique<QueuedCommand>(std::move(result));
}

std::string approval_state_json(const std::string& approval_id, const std::optional<CommandResult>& outcome,
                                const std::vector<caliper_builder::PendingApproval>& pending) {
    try {
        json j{{"approval", approval_id}, {"state", "unknown"}, {"result", nullptr}};
        if (outcome) {
            j["state"] = "decided";
            j["result"] = caliper_builder::command_result_to_json(*outcome);
        } else {
            for (const auto& p : pending) {
                if (p.approval != approval_id) continue;
                // What the person is asked about: the command, who sent it,
                // and its review, such as a spec edit's diff and warnings.
                j["state"] = "pending";
                j["name"] = p.invocation.name;
                j["origin"] = caliper_builder::command_origin_name(p.invocation.origin);
                j["review"] = p.review;
                break;
            }
        }
        return dump(j);
    } catch (...) {
        return "{\"approval\":null,\"result\":null,\"state\":\"unknown\"}";
    }
}

}  // namespace caliper_api
