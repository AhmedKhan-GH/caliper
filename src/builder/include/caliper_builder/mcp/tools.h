#pragma once
// MCP tools from the command table (design/mcp-server.md, "Tools"). Each
// command of a table becomes one tool:
//
// - the name is command_tool_name(), `_` for the dot (panel_create), which
//   MCP's name rule (1 to 128 of A-Z a-z 0-9 _ - .) always accepts;
// - the title is the command's name and the description its summary;
// - the input schema is the command's `args`;
// - the output schema is the command's `result`, without `png_base64` when the
//   result carries a PNG, since the PNG travels as image content;
// - the annotations follow the flags: readOnlyHint when the command does not
//   mutate (except app.launch, which changes what the app shows),
//   destructiveHint from `destructive`, and openWorldHint false, since every
//   tool acts on the local app only.
//
// A table comes either from the app (GET /v0/commands) or from the descriptor
// table compiled into caliper-mcp. build_tool_table() checks every descriptor
// with the commands library and drops one that fails, reporting why;
// parse_app_command_table() also checks the envelope and its bounds. The
// result conversions turn a command's result or error into the content of an
// MCP CallToolResult. No function here throws.
#include <caliper_builder/commands/command.h>
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder::mcp {

// MCP's tool name rule: 1 to 128 characters of A-Z a-z 0-9 _ - and dot.
bool is_valid_mcp_tool_name(std::string_view name);

// True when the command's result schema has a `png_base64` property.
bool command_returns_png(const CommandDescriptor& descriptor);

// The result schema without `png_base64`, or null when the result schema is
// not an object schema.
nlohmann::json tool_output_schema(const CommandDescriptor& descriptor);

// The MCP Tool object of one command.
nlohmann::json tool_definition(const CommandDescriptor& descriptor);

struct ToolTable {
    std::vector<CommandDescriptor> commands;
    std::vector<std::string> tool_names;   // parallel to `commands`
    nlohmann::json tools = nlohmann::json::array();   // the Tool objects, in table order
    std::string source;        // "app" or "built-in"
    std::string fingerprint;   // SHA-256 of the table's JSON, to notice a change

    const CommandDescriptor* find_tool(std::string_view tool_name) const;
    const CommandDescriptor* find_command(std::string_view command_name) const;
    // The table as GET /v0/commands gives it: {"version": 0, "commands": [...]}.
    nlohmann::json to_json() const;
};

struct ToolTableBuild {
    std::shared_ptr<const ToolTable> table;   // null when `error` is set
    std::vector<Error> dropped;               // one per descriptor left out
    Error error;
};

struct ToolTableLimits {
    static constexpr std::size_t max_commands = 1024;
};

// Builds the tools of `descriptors`. A descriptor whose `agents` is false
// (history.undo, history.redo: the person's own tools) is left out without a
// report, since no agent may run it. A descriptor that fails
// check_command_descriptor(), repeats a name or makes a tool name MCP refuses
// is dropped and reported (mcp.bad_descriptor).
ToolTableBuild build_tool_table(std::vector<CommandDescriptor> descriptors, std::string source);

// Parses the body of GET /v0/commands ({"version": 0, "commands": [...]}).
// The envelope must be that, with version 0 and at most max_commands
// commands; otherwise the whole table is refused (mcp.bad_command_table). A
// command must hold the eight descriptor fields with their types (other
// fields are ignored); one that does not is dropped and reported. `agents` is
// read when present, and true when an older app sends none. Then
// build_tool_table(..., "app").
ToolTableBuild parse_app_command_table(const nlohmann::json& body);

// The CallToolResult of a successful command (without resultType): the
// result as structuredContent and as JSON text, with a PNG result's
// png_base64 moved into an image content block. When the result does not
// match the tool's output schema, or a png_base64 is not a PNG, the outcome is
// a tool error (mcp.invalid_result) whose text still carries the result.
nlohmann::json tool_result_from_command(const CommandDescriptor& descriptor, const nlohmann::json& result);

// The CallToolResult of a failed call: isError true, and one text block that
// holds {"error": {"code", "message", "where"}} (with "approval" when given).
nlohmann::json tool_error_result(const Error& error, const std::string& approval = {});

}  // namespace caliper_builder::mcp
