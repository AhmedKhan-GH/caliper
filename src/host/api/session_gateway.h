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
//                             the args, the origin and the group, submitted
//                             to session.queue(): with wait_for_approval and
//                             approval_timeout for mcp and agent, and without
//                             for api, which gets command.needs_approval at
//                             once. The frame thread drains the queue, and
//                             the PendingCommand completes with the result.
//   GET  /v0/approvals/{id}   "decided" with the envelope when
//                             queue().approval_outcome() has it (the last 64
//                             decisions), "pending" while the registry lists
//                             the id among its pending approvals, else
//                             "unknown"
//
// Wiring it into the app (on the frame thread, which drains the queue):
//   api.set_command_gateway(caliper_api::make_session_gateway(*session));
//   ...
//   api.set_command_gateway(nullptr);   // before the session closes
// The session must outlive the gateway's use. A PendingCommand holds only the
// queue's future, so a command still held when the session closes is answered
// with command.shutdown, as the queue's shutdown completes its future.
// ============================================================================
#include "api/command_gateway.h"

#include <memory>

namespace caliper_builder {
class Session;
}

namespace caliper_api {

std::shared_ptr<CommandGateway> make_session_gateway(caliper_builder::Session& session);

}  // namespace caliper_api
