#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/permissions.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace caliper_builder {

namespace {

using nlohmann::json;
namespace fs = std::filesystem;

constexpr std::size_t kMaxFileBytes = std::size_t(1) << 20;  // far more than any person's rules
constexpr std::size_t kMaxRules = 4096;

bool saved_mode(AgentMode m) { return m != AgentMode::allow_all; }

}  // namespace

const char* agent_mode_name(AgentMode mode) {
    switch (mode) {
        case AgentMode::ask: return "ask";
        case AgentMode::accept_edits: return "accept_edits";
        case AgentMode::read_only: return "read_only";
        case AgentMode::allow_all: return "allow_all";
    }
    return "ask";
}

std::optional<AgentMode> agent_mode_from_name(std::string_view name) {
    for (AgentMode m : {AgentMode::ask, AgentMode::accept_edits, AgentMode::read_only, AgentMode::allow_all})
        if (name == agent_mode_name(m)) return m;
    return std::nullopt;
}

const char* agent_mode_label(AgentMode mode) {
    switch (mode) {
        case AgentMode::ask: return "Ask";
        case AgentMode::accept_edits: return "Accept edits";
        case AgentMode::read_only: return "Read only";
        case AgentMode::allow_all: return "Allow all";
    }
    return "Ask";
}

AgentMode next_agent_mode(AgentMode mode) {
    switch (mode) {
        case AgentMode::ask: return AgentMode::accept_edits;
        case AgentMode::accept_edits: return AgentMode::read_only;
        case AgentMode::read_only:
        case AgentMode::allow_all: return AgentMode::ask;
    }
    return AgentMode::ask;
}

AgentPrincipal agent_principal(const CommandInvocation& inv) {
    if (inv.origin == CommandOrigin::mcp) return AgentPrincipal::mcp;
    if (inv.origin == CommandOrigin::agent && inv.client == "chat") return AgentPrincipal::chat;
    return AgentPrincipal::none;
}

bool invocation_changes(const CommandDescriptor& d, const CommandInvocation& inv) {
    switch (d.effect) {
        case CommandEffect::read: return false;
        case CommandEffect::edit: {
            const auto dry = inv.args.is_object() ? inv.args.find("dry_run") : inv.args.end();
            const bool dry_run = inv.args.is_object() && dry != inv.args.end() && dry->is_boolean() && dry->get<bool>();
            return !dry_run && inv.preview.empty();
        }
        case CommandEffect::change:
        case CommandEffect::unset: return true;
    }
    return true;
}

PermissionVerdict decide_agent_permission(const CommandDescriptor& d, const CommandInvocation& inv,
                                          ApprovalDecision base, const PermissionContext& c) {
    PermissionVerdict v;
    v.decision = base;
    if (agent_principal(inv) == AgentPrincipal::none) return v;
    // Read only beats every rule and Allow all.
    if (c.mode == AgentMode::read_only && invocation_changes(d, inv)) {
        v.decision = ApprovalDecision::deny;
        v.message = "Caliper is in Read only mode for this agent, so " + inv.name +
                    " was not run. Reads, dry runs and edits of a preview still work; the person can change the "
                    "mode in Caliper.";
        return v;
    }
    // Only what would ask can run without a card, and never what is bound
    // to the content its card shows.
    if (base != ApprovalDecision::ask || c.bound) return v;
    if (c.mode == AgentMode::allow_all && agent_principal(inv) == AgentPrincipal::chat) {
        v.decision = ApprovalDecision::allow;
        v.allowed_by = "allow_all";
        return v;
    }
    if (c.mode == AgentMode::accept_edits && d.effect == CommandEffect::edit) {
        v.decision = ApprovalDecision::allow;
        v.allowed_by = "accept_edits";
        return v;
    }
    if (!c.python && (c.session_rule || c.saved_rule)) {
        v.decision = ApprovalDecision::allow;
        v.allowed_by = c.session_rule ? "session_rule" : "saved_rule";
    }
    return v;
}

bool valid_rule_agent(std::string_view agent) {
    if (agent.empty() || agent.size() > 64) return false;
    for (const char ch : agent)
        if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-')) return false;
    return true;
}

bool rule_command_allowed(std::string_view command) {
    const CommandDescriptor* d = find_command_descriptor(command);
    return d && d->agents && d->effect != CommandEffect::read;
}

AgentPermissionsRead read_agent_permissions(const std::string& path) {
    AgentPermissionsRead out;
    std::error_code ec;
    const fs::path p = fs::u8path(path);
    if (!fs::exists(p, ec)) return out;
    auto give_up = [&](std::string why) {
        out.file = {};
        out.problems.push_back(std::move(why));
        return out;
    };
#ifndef _WIN32
    struct stat st {};
    if (::stat(p.c_str(), &st) != 0) return give_up("The agent permissions file cannot be read, so it grants nothing.");
    if (st.st_mode & (S_IWGRP | S_IWOTH))
        return give_up("The agent permissions file may be written by others, so it grants nothing.");
#endif
    const auto size = fs::file_size(p, ec);
    if (ec || size > kMaxFileBytes) return give_up("The agent permissions file is too large or unreadable, so it grants nothing.");
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const json j = json::parse(ss.str(), nullptr, false);
    if (j.is_discarded() || !j.is_object()) return give_up("The agent permissions file is not a JSON object, so it grants nothing.");
    const auto version = j.find("version");
    if (version == j.end() || !version->is_number_integer() || version->get<long long>() != 1)
        return give_up("The agent permissions file is not version 1, so it grants nothing.");
    for (const auto& item : j.items())
        if (item.key() != "version" && item.key() != "modes" && item.key() != "rules")
            out.problems.push_back("Ignored the unknown field " + item.key() + ".");
    if (const auto modes = j.find("modes"); modes != j.end()) {
        if (!modes->is_object()) {
            out.problems.push_back("Ignored modes, which is not an object.");
        } else {
            for (const auto& item : modes->items()) {
                const bool known = item.key() == "chat" || item.key() == "mcp";
                const auto mode = item.value().is_string() ? agent_mode_from_name(item.value().get<std::string>())
                                                           : std::nullopt;
                if (!known) {
                    out.problems.push_back("Ignored the mode of " + item.key() + ", which is neither chat nor mcp.");
                } else if (!mode || !saved_mode(*mode)) {
                    out.problems.push_back("Ignored the mode of " + item.key() +
                                           ", which is not ask, accept_edits or read_only.");
                } else {
                    (item.key() == "chat" ? out.file.chat : out.file.mcp) = *mode;
                }
            }
        }
    }
    if (const auto rules = j.find("rules"); rules != j.end()) {
        if (!rules->is_array()) {
            out.problems.push_back("Ignored rules, which is not a list.");
        } else {
            std::size_t index = 0;
            for (const auto& r : *rules) {
                const std::string where = "rules[" + std::to_string(index++) + "]";
                if (out.file.rules.size() >= kMaxRules) {
                    out.problems.push_back("Ignored the rules after " + std::to_string(kMaxRules) + ".");
                    break;
                }
                if (!r.is_object() || r.size() != 2 || !r.contains("agent") || !r.contains("command") ||
                    !r["agent"].is_string() || !r["command"].is_string()) {
                    out.problems.push_back("Ignored " + where + ", which is not {\"agent\", \"command\"}.");
                    continue;
                }
                AgentRule rule{r["agent"].get<std::string>(), r["command"].get<std::string>()};
                if (!valid_rule_agent(rule.agent)) {
                    out.problems.push_back("Ignored " + where + ", whose agent is not an agent id.");
                    continue;
                }
                if (!rule_command_allowed(rule.command)) {
                    out.problems.push_back("Ignored " + where + ", whose command is not one an agent may be allowed.");
                    continue;
                }
                if (std::find(out.file.rules.begin(), out.file.rules.end(), rule) == out.file.rules.end())
                    out.file.rules.push_back(std::move(rule));
            }
        }
    }
    return out;
}

std::string agent_permissions_text(const AgentPermissionsFile& file) {
    json rules = json::array();
    for (const auto& r : file.rules) rules.push_back({{"agent", r.agent}, {"command", r.command}});
    auto mode = [](AgentMode m) { return agent_mode_name(saved_mode(m) ? m : AgentMode::ask); };
    const json j = {{"version", 1}, {"modes", {{"chat", mode(file.chat)}, {"mcp", mode(file.mcp)}}}, {"rules", rules}};
    return j.dump(2, ' ', false, json::error_handler_t::replace) + "\n";
}

Error write_agent_permissions(const std::string& path, const AgentPermissionsFile& file) {
    const std::string text = agent_permissions_text(file);
    const fs::path p = fs::u8path(path);
    std::error_code ec;
    if (!p.parent_path().empty()) fs::create_directories(p.parent_path(), ec);
    fs::path tmp = p;
    tmp += ".tmp";
    fs::remove(tmp, ec);
    auto failed = [&](const std::string& why) {
        fs::remove(tmp, ec);
        return Error("permissions.write_failed", "The agent permissions could not be saved: " + why, path);
    };
#ifndef _WIN32
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_TRUNC, 0600);
    if (fd < 0) return failed("the temporary file cannot be made");
    std::size_t done = 0;
    while (done < text.size()) {
        const ssize_t n = ::write(fd, text.data() + done, text.size() - done);
        if (n <= 0) {
            ::close(fd);
            return failed("the temporary file cannot be written");
        }
        done += static_cast<std::size_t>(n);
    }
    ::fchmod(fd, 0600);
    if (::close(fd) != 0) return failed("the temporary file cannot be closed");
#else
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << text;
        if (!f) return failed("the temporary file cannot be written");
    }
#endif
    fs::rename(tmp, p, ec);
    if (ec) return failed(ec.message());
    return {};
}

}  // namespace caliper_builder
