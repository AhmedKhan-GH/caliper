#pragma once
// Agent permissions (design/command-layer.md, Agent permissions): the modes
// and rules by which the person lets an agent's commands run without a card,
// or refuses them all. The command layer applies them, never the agent.
//
// Who they govern. The side chat's agent (the origin agent with the client
// "chat", which comes from the token the app issued for that chat) and the
// clients of caliper-mcp (the origin mcp) each have a mode, Ask by default.
// Nothing else is governed: the person's own UI never asks, the viewer only
// reads, the person's api token behaves as before, and the widget agent's
// sidecar (the client "widget") keeps its preview with Accept.
//
// The modes:
//   - Ask: the session's policy as it was, with a card for what asks.
//   - Accept edits: an edit of the open spec (CommandEffect::edit) that would
//     ask runs at once; it is still journaled, with "allowed_by", and undone
//     like any step. Everything else that asks still asks.
//   - Read only: reads and proposals run (a dry run, an edit into a stacked
//     preview, the previews' own commands); every other command that would
//     change something is refused with command.denied and a plain message.
//   - Allow all: what would ask runs without a card, except a data-loss
//     card (below). It is saved like the other modes, for the chat's agent
//     and for mcp clients each, and stays on, across launches, until the
//     person sets another mode.
// The rules: answering a card "for this session" or "always" allows that
// command (by its rule key, permission_rule_key()) for the chat's agent, or
// for mcp clients, in this session or, saved, until the person revokes it. The
// key is the command's name, and for run.set_param and run.action the name
// with its target ("run.set_param:lr"), so a rule allows one parameter or one
// action; for intervene.set and intervene.clear the name with the point
// ("intervene.set:blocks.0.attn#3"), so a rule allows one point, and
// intervene.clear with all true always asks. A saved rule is kept per agent:
// the chat's agent id, or "mcp".
//
// What no mode and no rule changes: the registry refuses before any policy a
// viewer's change and history.undo or history.redo from an agent; a data-loss
// card always asks, Allow all included: an approval bound to the content it
// would lose (an overwrite of a saved applet, the replacement of an open spec
// with unsaved changes), since the card shows what is lost. A command that
// deletes or overwrites saved work keeps its card under Allow all by binding
// its approval to that content in the session, as applet.save does. A rule
// never covers running Python code (a Python loader, a managed run's start or
// fork), which only a card or Allow all lets run; and Read only beats every
// rule. Nothing an agent sends sets a mode or a rule: there is no command for
// them, and only the app's UI calls the session's functions that change them.
//
// The saved file (agent-permissions.json in the app-data folder), written
// owner-only (0600) through a temporary file:
//   {"version": 1, "modes": {"chat": "ask", "mcp": "accept_edits"},
//    "rules": [{"agent": "claude", "command": "panel.patch"},
//              {"agent": "mcp", "command": "run.set_param:lr"}, ...]}
// read_agent_permissions() ignores, and reports, a malformed file, an unknown
// field, a mode it does not know, and a rule for an unknown agent id, for a
// command that is not an agent's, or for run.set_param, run.action,
// intervene.set or intervene.clear without its target. Allow all is read
// only from a file with no such problem; in any other file a mode of
// allow_all reads as ask. On POSIX a file that others may write, or that
// another user owns, grants nothing. A missing file is the defaults: Ask, no
// rules. No function here throws.
#include <caliper_builder/commands/approval.h>
#include <caliper_builder/commands/command.h>
#include <caliper_builder/error.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {

enum class AgentMode { ask, accept_edits, read_only, allow_all };

// "ask", "accept_edits", "read_only", "allow_all".
const char* agent_mode_name(AgentMode mode);
std::optional<AgentMode> agent_mode_from_name(std::string_view name);
// The word the app shows: "Ask", "Accept edits", "Read only", "Allow all".
const char* agent_mode_label(AgentMode mode);
// The mode after `mode` in the cycle of Shift-Tab: Ask, Accept edits, Read
// only, then Ask again. Allow all is not in it; from Allow all it gives Ask.
AgentMode next_agent_mode(AgentMode mode);

// Whose mode and rules an invocation falls under.
enum class AgentPrincipal { none, chat, mcp };
// chat for the origin agent with the client "chat", mcp for the origin mcp,
// none for everything else.
AgentPrincipal agent_principal(const CommandInvocation& invocation);

// Whether the invocation would change something, as Read only judges it: a
// command whose effect is `read` does not; an `edit` does unless it is a dry
// run or names a stacked preview; any other command, a `run` included, does.
bool invocation_changes(const CommandDescriptor& descriptor, const CommandInvocation& invocation);

// What decides one governed invocation besides its command.
struct PermissionContext {
    AgentMode mode = AgentMode::ask;  // the principal's mode
    bool session_rule = false;        // a rule of this session allows the command
    bool saved_rule = false;          // a saved rule of the agent allows it
    bool bound = false;               // a data-loss card, bound to content: only a card decides, in every mode
    bool python = false;              // it runs Python code (a loader, a run's start or fork): no rule covers it
};

struct PermissionVerdict {
    ApprovalDecision decision = ApprovalDecision::ask;
    std::string allowed_by;  // "accept_edits", "allow_all", "session_rule", "saved_rule" when they turned ask into allow
    std::string message;     // a denial's plain message
};

// The decision for an invocation whose policy, without modes, gave `base`.
// Not governed: `base`. Read only: deny what changes something. Otherwise
// only an ask can become an allow: never when bound; Allow all, for the chat
// and for mcp; Accept edits for an edit; a session rule, then a saved rule,
// unless it runs Python.
PermissionVerdict decide_agent_permission(const CommandDescriptor& descriptor, const CommandInvocation& invocation,
                                          ApprovalDecision base, const PermissionContext& context);

// One saved rule: the agent ("mcp" for caliper-mcp's clients, else the side
// chat's agent id) may run the command without a card.
struct AgentRule {
    std::string agent;
    std::string command;
    bool operator==(const AgentRule& o) const { return agent == o.agent && command == o.command; }
};

// The name under which a rule allows an invocation: the command's name, and
// for run.set_param and run.action the command's name, ":" and the parameter
// or the action it names ("run.set_param:lr"), so that a card's rule allows
// one parameter or one action and not the whole command
// (design/instruments.md, Run Effect Class); for intervene.set and
// intervene.clear the name, ":" and the argument `point`
// ("intervene.clear:blocks.0.attn#3"), or a key no rule holds for
// intervene.clear with all true (design/gpu-programs.md, Intervention
// Effect). The session looks session and saved rules up by this key.
std::string permission_rule_key(const CommandInvocation& invocation);

// The id a saved rule names: "mcp", or 1 to 64 characters of a-z 0-9 _ -
// for a chat agent ("claude", "custom").
bool valid_rule_agent(std::string_view agent);
// A command an agent may be allowed always: in the table, open to agents,
// and one that changes something. A rule key with a target passes when the
// command takes one (run.set_param, run.action, intervene.set,
// intervene.clear) and the target is a valid name (1 to 63 bytes, or 1 to 255
// for a point, no control characters). The bare name of such a
// command passes too, as the check of a card's command; the saved file
// ignores a rule of it, which would allow no target.
bool rule_command_allowed(std::string_view command);

struct AgentPermissionsFile {
    AgentMode chat = AgentMode::ask;
    AgentMode mcp = AgentMode::ask;
    std::vector<AgentRule> rules;
};

struct AgentPermissionsRead {
    AgentPermissionsFile file;
    std::vector<std::string> problems;  // one plain sentence for each thing ignored
};
AgentPermissionsRead read_agent_permissions(const std::string& path);
// The same, with the user id that must own the file on POSIX (the process's
// effective user id above); a test names another to check the owner rule.
AgentPermissionsRead read_agent_permissions(const std::string& path, long owner);
// The file's JSON text.
std::string agent_permissions_text(const AgentPermissionsFile& file);
// Writes it owner-only through a temporary file in the same folder.
Error write_agent_permissions(const std::string& path, const AgentPermissionsFile& file);

}  // namespace caliper_builder
