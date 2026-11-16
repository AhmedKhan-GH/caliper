#pragma once
// The agents Caliper knows how to start, and the parsing of a custom agent
// command. The table holds the launch commands that the agents' own
// repositories documented on 15 November 2026 (docs/design/acp-integration.md
// records the sources); a person can always give a command of their own.
//
// split_command_line() turns one line of text into an argument vector the way
// a POSIX shell splits words, with single quotes, double quotes and
// backslashes, and nothing else: no variables, no globbing, no command
// substitution, no redirection. The result is run without a shell.
//
// Error codes: acp.bad_command.
#include <caliper_builder/error.h>

#include <string>
#include <vector>

namespace caliper_builder::acp {

struct AgentPreset {
    std::string id;                  // "claude", "codex", "gemini"
    std::string title;               // for a menu
    std::vector<std::string> argv;   // the launch command
    std::string login_hint;          // how a person logs in, in their own terminal
};

const std::vector<AgentPreset>& known_agents();
// Null when `id` names no preset.
const AgentPreset* find_agent_preset(const std::string& id);

Error split_command_line(const std::string& text, std::vector<std::string>* argv);

}  // namespace caliper_builder::acp
