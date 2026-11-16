#pragma once
// ============================================================================
// session_gateway.h: the command gateway of Caliper API (command_gateway.h)
// over the builder's session (caliper_builder/session/session.h), so that the
// app's API server serves the session's command layer. It is compiled only
// when CALIPER_WITH_BUILDER is on, into the library caliper_api_session.
//
//   GET  /v0/commands         command_table_to_json(), made once, when the
//                             gateway is made
//   POST /v0/commands/{name}  an invocation with the server's id, the name,
//                             the args, the origin, the group and the
//                             preview, submitted
//                             to session.queue(): with wait_for_approval and
//                             approval_timeout for mcp and agent, and without
//                             for api, which gets command.needs_approval at
//                             once. The frame thread drains the queue, and
//                             the PendingCommand completes with the result.
//   GET  /v0/approvals/{id}   "decided" with the envelope when
//                             queue().approval_outcome() has it (the last 64
//                             decisions), "pending" while the registry lists
//                             the id among its pending approvals, with the
//                             command's name, its origin and its review (a
//                             spec edit's diff and warnings, else null), else
//                             "unknown"
//
// Wiring it into the app (on the frame thread, which drains the queue):
//   api.set_command_gateway(caliper_api::make_session_gateway(*session));
//   ...
//   api.set_command_gateway(nullptr);   // before the session closes
// The session must outlive the gateway's use. A PendingCommand holds only the
// queue's future, so a command still held when the session closes is answered
// with command.shutdown, as the queue's shutdown completes its future.
//
// The pieces below serve a gateway that reaches the session in its own way,
// such as the app's (src/host/builder/app_session.h), which opens its
// workspace on the first command and answers approval lookups without waiting
// for a loader that holds the registry.
// ============================================================================
#include "api/command_gateway.h"

#include <caliper_builder/commands/approval.h>
#include <caliper_builder/commands/command.h>
#include <caliper_builder/commands/queue.h>

#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace caliper_builder {
class Session;
}

namespace caliper_api {

std::shared_ptr<CommandGateway> make_session_gateway(caliper_builder::Session& session);

// The invocation and submit options of one command request, or, in
// `refused`, the envelope that refuses it (bad arguments or origin).
struct RequestInvocation {
    caliper_builder::CommandInvocation invocation;
    caliper_builder::SubmitOptions options;
    std::string refused;
};
RequestInvocation request_invocation(CommandRequest request);

// A PendingCommand answered at once with `envelope`, and one over the
// queue's future.
std::unique_ptr<PendingCommand> ready_command(std::string envelope);
std::unique_ptr<PendingCommand> queued_command(std::future<caliper_builder::CommandResult> result);

// The body of GET /v0/approvals/{id}, from the queue's outcome of the
// approval and the registry's pending approvals.
std::string approval_state_json(const std::string& approval_id,
                                const std::optional<caliper_builder::CommandResult>& outcome,
                                const std::vector<caliper_builder::PendingApproval>& pending);

}  // namespace caliper_api
