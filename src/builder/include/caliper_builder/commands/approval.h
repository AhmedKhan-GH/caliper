#pragma once
// Approval (design/command-layer.md, Approval). An ApprovalPolicy decides
// allow, deny or ask for an invocation, given its descriptor and the
// invocation itself (its origin and its arguments). The default policy allows
// everything from the ui and builder origins and every command that is not
// destructive, and asks for a destructive command from agent, mcp and api.
//
// On ask, the registry parks the invocation under an approval id and answers
// "command.needs_approval"; only the app's own UI decides, through
// CommandRegistry::approve() and deny() (or the queue's), so an agent cannot
// approve its own request. A policy that throws is treated as a deny.
//
// An ApprovalReview, when the registry has one, is asked once for each
// invocation it parks, and what it returns is kept with the pending approval:
// the session gives the review of a spec edit, {"diff", "warnings"}, so that
// the card shows the change before the person decides. Null means none; a
// review that throws gives none.
#include <caliper_builder/commands/command.h>

#include <functional>
#include <string>

namespace caliper_builder {

enum class ApprovalDecision { allow, deny, ask };

using ApprovalPolicy = std::function<ApprovalDecision(const CommandDescriptor&, const CommandInvocation&)>;

ApprovalDecision default_approval_policy(const CommandDescriptor& descriptor, const CommandInvocation& invocation);

using ApprovalReview = std::function<nlohmann::json(const CommandDescriptor&, const CommandInvocation&)>;

// One invocation waiting for the person's decision, as the UI shows it.
struct PendingApproval {
    std::string approval;  // the approval id, "ap-<n>"
    CommandInvocation invocation;
    std::string summary;  // the command's summary
    std::string requested_at;  // UTC
    nlohmann::json review;     // what the ApprovalReview gave, such as a spec edit's diff; null when none
};

// {"approval", "invocation", "summary", "requested_at", "review"}.
nlohmann::json pending_approval_to_json(const PendingApproval& pending);

}  // namespace caliper_builder
