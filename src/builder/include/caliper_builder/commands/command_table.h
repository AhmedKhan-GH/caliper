#pragma once
// The descriptor table: every command of v0 (docs/plans/command-layer.md, The
// commands of v0) with its summary, its argument and result schemas, an
// example of its arguments and its flags, as data with no handlers. A
// registry adds it with CommandRegistry::add_all(command_table()), and the
// handlers are bound later. caliper-mcp links it to list its tools, and the
// Python sidecar reads its JSON form, src/builder/commands/command_table.json,
// which a test keeps equal to command_table_to_json().
//
// The summaries and property descriptions are written for an agent that has
// never seen Caliper. Every object schema is closed (additionalProperties
// false), lists its required properties and describes every property, so the
// schemas serve as strict tools; the one open object is the canonical JSON of
// a spec in the result of spec.get. The table is built once, on first use, and
// is immutable afterwards. No function here throws.
#include <caliper_builder/commands/command.h>

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {

// Every command of v0, in the order of the design page.
const std::vector<CommandDescriptor>& command_table();

// The descriptor of one command, or null.
const CommandDescriptor* find_command_descriptor(std::string_view name);

// The body of GET /v0/commands: {"version": 0, "commands": [descriptor, ...]}.
nlohmann::json command_table_to_json();

// The text of command_table.json: command_table_to_json() with two-space
// indents, sorted keys and a final newline.
std::string command_table_json_text();

// The result paths allowed to stay open, as check_closed_command_schema()
// names them, keyed by command: {"spec.get", "result.json"}.
std::vector<std::pair<std::string, std::string>> command_table_open_paths();

// Faults found while building the table from its text; empty when it is
// sound. The table's test checks it.
const std::vector<std::string>& command_table_build_errors();

}  // namespace caliper_builder
