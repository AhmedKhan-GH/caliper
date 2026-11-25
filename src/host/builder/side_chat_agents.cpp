// side_chat_agents.cpp: agent detection on PATH, custom agent command
// splitting, launch construction, caliper-mcp discovery, tool-to-command
// matching, and compact spec diffs (side_chat.h).
#include "builder/side_chat.h"
#include "builder/side_chat_internal.h"
#include "builder/widget_surface.h"

#include <caliper_builder/acp/agents.h>
#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/commands/command.h>
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/spec/format.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace caliper_host {

namespace fs = std::filesystem;
namespace acp = caliper_builder::acp;
using caliper_builder::Error;
using nlohmann::json;
using side_chat_detail::trimmed;

namespace {

std::string env_path() {
    const char* p = std::getenv("PATH");
    return p ? std::string(p) : std::string("/usr/bin:/bin");
}

}  // namespace

// ---- the agents ---------------------------------------------------------------------------

std::vector<std::string> chat_agent_needs(const acp::AgentPreset& preset) {
    if (preset.id == "claude") return {"npx", "claude"};
    if (preset.id == "codex") return {"npx", "codex"};
    if (preset.argv.empty()) return {};
    return {preset.argv[0]};
}

std::vector<ChatAgent> detect_chat_agents(const std::string& path_list) {
    const std::string path = path_list.empty() ? env_path() : path_list;
    std::vector<ChatAgent> out;
    for (const acp::AgentPreset& p : acp::known_agents()) {
        ChatAgent a;
        a.id = p.id;
        a.title = p.title;
        a.argv = p.argv;
        a.needs = chat_agent_needs(p);
        a.login_hint = p.login_hint;
        a.found = !a.needs.empty();
        for (const std::string& cmd : a.needs)
            if (acp::find_executable(cmd, path).empty()) a.found = false;
        if (a.found && !a.argv.empty()) a.argv[0] = acp::find_executable(a.argv[0], path);
        out.push_back(std::move(a));
    }
    return out;
}

Error custom_chat_agent(const std::string& name, const std::string& line, const std::string& path_list, ChatAgent* out) {
    std::vector<std::string> argv;
    if (Error e = acp::split_command_line(line, &argv)) return e;
    ChatAgent a;
    a.id = "custom";
    a.title = trimmed(name).empty() ? std::string("Custom agent") : trimmed(name);
    a.custom = true;
    a.needs = {argv[0]};
    const std::string found = acp::find_executable(argv[0], path_list.empty() ? env_path() : path_list);
    a.found = !found.empty();
    if (a.found) argv[0] = found;
    a.argv = std::move(argv);
    *out = std::move(a);
    return {};
}

std::string chat_gpu_note(bool live_attached, bool local_model) {
    return live_attached && local_model ? "Shared GPU" : std::string();
}

Error build_chat_launch(const ChatAgent& agent, const ChatEnvironment& env, ChatLaunch* out,
                        const acp::AcpClientOptions& base) {
    if (agent.argv.empty() || agent.argv[0].empty()) return {"chat.no_agent", "No agent command was given."};
    if (!agent.found)
        return {"chat.no_agent", "The command " + (agent.needs.empty() ? agent.argv[0] : agent.needs.front()) +
                                     " of " + agent.title + " is not on PATH."};
    std::error_code ec;
    if (env.workspace.empty() || !fs::path(env.workspace).is_absolute() || !fs::is_directory(env.workspace, ec))
        return {"chat.no_workspace", "The workspace must be an absolute path to a folder.", env.workspace};
    if (env.mcp_command.empty() || !fs::path(env.mcp_command).is_absolute() ||
        acp::find_executable(env.mcp_command, "").empty())
        return {"chat.no_mcp", "This build's caliper-mcp was not found.", env.mcp_command};
    if (env.api_file.empty() || !fs::is_regular_file(env.api_file, ec))
        return {"chat.no_api", "Caliper API is not running, so caliper-mcp has no API file to read.", env.api_file};
    ChatLaunch l;
    l.client = base;
    l.client.agent_argv = agent.argv;
    l.client.agent_cwd = env.workspace;
    l.client.agent_env.erase(std::remove_if(l.client.agent_env.begin(), l.client.agent_env.end(),
                                            [](const acp::EnvVar& v) { return v.name == "PATH"; }),
                             l.client.agent_env.end());
    if (!env.path.empty()) l.client.agent_env.push_back({"PATH", env.path});
    l.cwd = env.workspace;
    acp::McpServer mcp;
    mcp.name = kChatMcpServerName;
    mcp.command = env.mcp_command;
    mcp.args = {"--origin", "agent", "--api-file", env.api_file};
    l.mcp_servers.push_back(std::move(mcp));
    *out = std::move(l);
    return {};
}

std::string find_caliper_mcp(const std::string& exe_dir) {
#ifdef _WIN32
    const char* name = "caliper-mcp.exe";
#else
    const char* name = "caliper-mcp";
#endif
    std::error_code ec;
    if (!exe_dir.empty()) {
        const fs::path beside = fs::path(exe_dir) / name;
        if (fs::is_regular_file(beside, ec)) return beside.string();
    }
#ifdef CALIPER_MCP_BUILD_PATH
    if (fs::is_regular_file(CALIPER_MCP_BUILD_PATH, ec)) return CALIPER_MCP_BUILD_PATH;
#endif
    return {};
}

std::string caliper_command_of(const acp::ToolCallInfo& call, const std::string& server) {
    // A tool name of the table: "panel_create" -> "panel.create".
    auto command_of_tool = [](const std::string& tool) -> std::string {
        const std::string name = caliper_builder::command_name_from_tool(tool);
        if (name.empty()) return {};
        for (const auto& d : caliper_builder::command_table())
            if (d.name == name) return name;
        return {};
    };
    auto from_text = [&](const std::string& text) -> std::string {
        // "panel_create (caliper MCP Server)"
        const std::string suffix = " (" + server + " MCP Server)";
        const size_t at = text.find(suffix);
        if (at != std::string::npos) {
            const size_t start = text.find_last_of(" \t", at == 0 ? 0 : at - 1);
            const std::string tool = text.substr(start == std::string::npos ? 0 : start + 1,
                                                 at - (start == std::string::npos ? 0 : start + 1));
            if (std::string c = command_of_tool(tool); !c.empty()) return c;
        }
        // A word with the server's prefix.
        const std::vector<std::string> prefixes = {"mcp__" + server + "__", server + "__", server + ".",
                                                   server + "/", server + ":", server + "_"};
        size_t i = 0;
        while (i < text.size()) {
            while (i < text.size() && (std::isspace((unsigned char)text[i]) || text[i] == '(' || text[i] == '`' ||
                                       text[i] == '"' || text[i] == '\''))
                ++i;
            size_t j = i;
            while (j < text.size() && !std::isspace((unsigned char)text[j]) && text[j] != '(' && text[j] != ')' &&
                   text[j] != '`' && text[j] != '"' && text[j] != '\'')
                ++j;
            const std::string word = text.substr(i, j - i);
            for (const std::string& p : prefixes)
                if (word.size() > p.size() && word.compare(0, p.size(), p) == 0)
                    if (std::string c = command_of_tool(word.substr(p.size())); !c.empty()) return c;
            i = j + 1;
        }
        return {};
    };
    // An agent that calls Caliper API itself (caliper-agent) names the command
    // in the call's _meta; it counts when the table has it.
    if (!call.caliper_command.empty())
        for (const auto& d : caliper_builder::command_table())
            if (d.name == call.caliper_command) return d.name;
    if (std::string c = from_text(call.title); !c.empty()) return c;
    const json& raw = call.raw_input;
    if (raw.is_object()) {
        for (const char* sk : {"server", "server_name", "serverName"}) {
            auto s = raw.find(sk);
            if (s == raw.end() || !s->is_string() || s->get<std::string>() != server) continue;
            for (const char* tk : {"tool", "tool_name", "toolName", "name"}) {
                auto t = raw.find(tk);
                if (t != raw.end() && t->is_string())
                    if (std::string c = command_of_tool(t->get<std::string>()); !c.empty()) return c;
            }
        }
    }
    return {};
}

// ---- the compact diff of a spec file ------------------------------------------------------

std::vector<WidgetDiffLine> chat_compact_diff(const std::string& path, const std::string& before, const std::string& after) {
    const std::string suffix = ".spec.toml";
    if (path.size() < suffix.size() || path.compare(path.size() - suffix.size(), suffix.size(), suffix) != 0) return {};
    namespace spec = caliper_builder::spec;
    const spec::ParseResult a = spec::parse_toml(before), b = spec::parse_toml(after);
    if (!a.ok() || !b.ok()) return {};
    std::vector<WidgetDiffLine> out;
    std::set<std::string> added;
    for (const spec::Panel& q : b.spec.panels)
        if (!a.spec.find_panel(q.id)) {
            added.insert("panel " + q.id);
            for (WidgetDiffLine& l : widget_item_lines(spec::to_json(q))) out.push_back(std::move(l));
        }
    for (const spec::Source& q : b.spec.sources)
        if (!a.spec.find_source(q.id)) {
            added.insert("source " + q.id);
            for (WidgetDiffLine& l : widget_item_lines(spec::to_json(q))) out.push_back(std::move(l));
        }
    for (WidgetDiffLine& l : widget_compact_diff(a.spec, b.spec, std::string()))
        if (!added.count(l.key)) out.push_back(std::move(l));
    return out;
}

}  // namespace caliper_host
