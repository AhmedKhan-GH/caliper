#include <caliper_builder/acp/agents.h>

namespace caliper_builder::acp {

const std::vector<AgentPreset>& known_agents() {
    // The adapters are named without a version, so a person runs the release
    // its makers ship; a custom command pins one when that matters.
    static const std::vector<AgentPreset> presets = {
        {"claude", "Claude Code", {"npx", "-y", "@agentclientprotocol/claude-agent-acp"},
         "Log in once with `claude` and its /login command."},
        {"codex", "Codex", {"npx", "-y", "@agentclientprotocol/codex-acp"},
         "Log in once with `codex login`."},
        {"gemini", "Gemini CLI", {"gemini", "--acp"},
         "Log in once by running `gemini` and choosing a sign-in method."},
    };
    return presets;
}

const AgentPreset* find_agent_preset(const std::string& id) {
    for (const auto& p : known_agents())
        if (p.id == id) return &p;
    return nullptr;
}

Error split_command_line(const std::string& text, std::vector<std::string>* argv) {
    argv->clear();
    if (text.size() > 64 * 1024) return {"acp.bad_command", "The command is longer than 64 KiB."};
    std::string word;
    bool in_word = false;
    enum { plain, single_quoted, double_quoted } mode = plain;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '\0') return {"acp.bad_command", "The command contains a NUL byte."};
        switch (mode) {
            case plain:
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    if (in_word) {
                        argv->push_back(word);
                        word.clear();
                        in_word = false;
                    }
                } else if (c == '\'') {
                    mode = single_quoted;
                    in_word = true;
                } else if (c == '"') {
                    mode = double_quoted;
                    in_word = true;
                } else if (c == '\\') {
                    if (i + 1 >= text.size()) return {"acp.bad_command", "The command ends with a backslash."};
                    word += text[++i];
                    in_word = true;
                } else {
                    word += c;
                    in_word = true;
                }
                break;
            case single_quoted:
                if (c == '\'') mode = plain;
                else word += c;
                break;
            case double_quoted:
                if (c == '"') {
                    mode = plain;
                } else if (c == '\\' && i + 1 < text.size() &&
                           (text[i + 1] == '"' || text[i + 1] == '\\' || text[i + 1] == '$' || text[i + 1] == '`')) {
                    word += text[++i];
                } else {
                    word += c;
                }
                break;
        }
    }
    if (mode != plain) return {"acp.bad_command", "The command has a quote that is not closed."};
    if (in_word) argv->push_back(word);
    if (argv->empty()) return {"acp.bad_command", "The command is empty."};
    return {};
}

}  // namespace caliper_builder::acp
