#pragma once
// The command model (design/command-layer.md, The command model): a
// CommandDescriptor describes one command, a CommandInvocation asks for one to
// run, and a CommandResult answers it. Each has its JSON form.
//
// A command name is `group.verb`: the group matches [a-z][a-z0-9]* and has no
// underscore, the verb matches [a-z][a-z0-9_]*, and the whole name has at most
// 64 characters, so the tool name `group_verb` maps back to it. An invocation
// id is up to 128 characters of A-Z a-z 0-9 . _ : - (empty means "assign one");
// a group, a merge key or a preview is 1 to 128 characters of
// A-Z a-z 0-9 . _ : / -.
// command_invocation_from_json() is strict: an unknown field, a missing name or
// origin, or a malformed field is "command.invalid_invocation"; arguments
// nested too deeply or too large are "command.invalid_args", found before they
// are copied, because copying, comparing and dumping a JSON value recurse.
//
// A result is {"id", "ok": true, "result"} or {"id", "ok": false, "error":
// {"code", "message", "where"}}, and a "command.needs_approval" error also
// carries "approval". The writers repair invalid UTF-8 in the envelope's own
// strings; `args` and `result` pass through as they are, so a caller dumps
// with nlohmann::json::error_handler_t::replace. No function here throws.
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {

// Who sent an invocation: the app's UI, the WYSIWYG builder, the agent pane,
// an outside agent through caliper-mcp, a program through Caliper API, or the
// browser's spec page, which the registry lets run reads only.
enum class CommandOrigin { ui, builder, agent, mcp, api, viewer };

const char* command_origin_name(CommandOrigin origin);
std::optional<CommandOrigin> command_origin_from_name(std::string_view name);

// What a command does, by which an agent's mode decides it (permissions.h):
// `read` reads, or proposes without changing the workspace (a stacked
// preview's begin, diff and discard); `edit` edits the open spec in memory,
// which undo reverses exactly; `change` does anything else that changes
// something: it writes or deletes saved data, runs a loader, replaces the open
// spec or the page the app shows, or moves the history; `run` acts on the
// running script that the app is attached to (a parameter, an action, a
// pause), which changes no workspace state and has no undo
// (design/instruments.md, Run Effect Class). Accept edits applies only `edit`
// without a card. Every command of the table states one (a test checks it);
// `unset`, which a descriptor made elsewhere may leave, counts as `change`, so
// a command nobody classified asks.
enum class CommandEffect { unset, read, edit, change, run };

// "read", "edit", "change" or "run" ("change" for unset).
const char* command_effect_name(CommandEffect effect);
std::optional<CommandEffect> command_effect_from_name(std::string_view name);

struct CommandDescriptor {
    std::string name;     // group.verb
    std::string summary;  // what the command does, for an agent that has never seen Caliper
    nlohmann::json args = nlohmann::json::object();    // JSON Schema of the arguments (an object schema)
    nlohmann::json result = nlohmann::json::object();  // JSON Schema of the result
    std::vector<nlohmann::json> examples;              // argument objects that pass `args`
    bool mutates = false;       // changes workspace state; journaled and undoable
    bool destructive = false;   // deletes or overwrites saved data, or runs a Python loader
    bool frame_thread = false;  // must run on the app's frame thread
    // Whether an agent may run it: the origins agent and mcp. history.undo
    // and history.redo are the person's tools and say false; the registry
    // refuses them to agents with command.not_allowed, and the clients that
    // build an agent's tools from the table leave them out.
    bool agents = true;
    CommandEffect effect = CommandEffect::unset;  // what it does (above)
};

Error check_command_name(std::string_view name);

// Checks the name, the summary, both schemas (within the subset; `args` an
// object schema) and that every example passes `args` ("command.bad_descriptor").
Error check_command_descriptor(const CommandDescriptor& descriptor);

// {"name", "summary", "args", "result", "examples", "mutates", "destructive",
// "frame_thread", "agents", "effect"}.
nlohmann::json command_descriptor_to_json(const CommandDescriptor& descriptor);

// "panel_create" for "panel.create", and back; an empty string when the input
// is not a well-formed name.
std::string command_tool_name(std::string_view command_name);
std::string command_name_from_tool(std::string_view tool_name);

struct CommandInvocation {
    std::string id;  // empty: the registry assigns "c-<n>"
    std::string name;
    nlohmann::json args = nlohmann::json::object();
    CommandOrigin origin = CommandOrigin::ui;
    std::string group;      // empty: none
    std::string merge_key;  // empty: none
    // The id of a stacked preview of the open spec (Session, preview.begin):
    // an edit of the open spec applies to the preview instead, and spec.get,
    // spec.validate and panel.render read it. Commands that do not read or
    // edit the open spec ignore it. Over Caliper API it is the header
    // X-Caliper-Preview. Empty: none.
    std::string preview;
    // Which agent the app started sent it, from the token of the request,
    // never from the request itself: "chat" for the side chat's agent,
    // "widget" for the widget agent's sidecar; empty for every other caller.
    // An agent's mode and rules (permissions.h) govern the origin agent with
    // the client "chat", and the origin mcp. 0 to 32 characters of a-z.
    std::string client;
};

// Checks the id, the name, the group and the merge key, and that `args` is an
// object ("command.invalid_invocation").
Error check_command_invocation(const CommandInvocation& invocation);

nlohmann::json command_invocation_to_json(const CommandInvocation& invocation);

struct CommandInvocationParse {
    CommandInvocation invocation;
    Error error;
};
CommandInvocationParse command_invocation_from_json(const nlohmann::json& value);

struct CommandResult {
    std::string id;
    bool ok = false;
    nlohmann::json result = nlohmann::json::object();  // when ok
    Error error;                                       // when not ok
    std::string approval;                              // with command.needs_approval

    static CommandResult success(std::string id, nlohmann::json result);
    static CommandResult failure(std::string id, Error error);
};

nlohmann::json command_result_to_json(const CommandResult& result);

struct CommandResultParse {
    CommandResult result;
    Error error;
};
CommandResultParse command_result_from_json(const nlohmann::json& value);

}  // namespace caliper_builder
