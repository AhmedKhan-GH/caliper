// side_chat.cpp: the side chat (side_chat.h): the agents and their launch,
// the handoff draft, the state machine over AcpClient, the layout as plain
// rectangles, and the ImGui side in the house style.
#include "builder/side_chat.h"

#include "spec_runtime/spec_runtime.h"
#include "ui/sternbild_theme.h"

#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/commands/command.h>
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/line_diff.h>
#include <caliper_builder/spec/format.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <set>

namespace caliper_host {

namespace fs = std::filesystem;
namespace acp = caliper_builder::acp;
using caliper_builder::Error;
using nlohmann::json;

namespace {

constexpr size_t kMaxItemText = 256u << 10;   // one message, thought or prompt text kept
constexpr size_t kMaxItems = 4000;            // the conversation's items; the oldest go first
constexpr size_t kMaxFaults = 200;
constexpr int kStderrTail = 1;                // stderr lines shown when the agent exited, one elided
constexpr int kMaxCompactLines = 8;           // a permission card's compact lines before "+N"
constexpr int kMaxDiffLines = 16;             // a permission card's diff lines before "+N"

// The lines of a text, without their line feeds.
std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start < text.size()) {
        const size_t end = text.find('\n', start);
        out.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
        start = end == std::string::npos ? text.size() : end + 1;
    }
    return out;
}

std::string env_path() {
    const char* p = std::getenv("PATH");
    return p ? std::string(p) : std::string("/usr/bin:/bin");
}

std::string trimmed(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

void append_bounded(std::string* s, const std::string& more) {
    if (s->size() >= kMaxItemText) return;
    s->append(more, 0, std::min(more.size(), kMaxItemText - s->size()));
}

json for_assistant(json block) {
    if (block.is_object()) block["annotations"] = {{"audience", json::array({"assistant"})}};
    return block;
}

json text_block(const std::string& text) { return {{"type", "text"}, {"text", text}}; }

// The noun of a bundle block: its part and what it is about.
std::string chip_of(const json& block) {
    const std::string type = block.value("type", "");
    if (type == "image") return "render";
    std::string text;
    if (type == "resource") {
        const json r = block.value("resource", json::object());
        const std::string uri = r.value("uri", "");
        const std::string prefix = "caliper://context/";
        if (uri.rfind(prefix, 0) == 0) {
            std::string rest = uri.substr(prefix.size());
            std::replace(rest.begin(), rest.end(), '/', ' ');
            return rest;
        }
        text = r.value("text", "");
    } else {
        text = block.value("text", "");
    }
    const std::string marker = "<caliper-data part=\"";
    if (text.rfind(marker, 0) == 0) {
        const size_t end = text.find('"', marker.size());
        std::string chip = text.substr(marker.size(), end == std::string::npos ? std::string::npos : end - marker.size());
        const std::string about = " about=\"";
        const size_t a = text.find(about);
        const size_t close = text.find('>');
        if (a != std::string::npos && a < close) {
            const size_t ae = text.find('"', a + about.size());
            if (ae != std::string::npos) chip += " " + text.substr(a + about.size(), ae - a - about.size());
        }
        return chip;
    }
    return "context";
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

// ---- the handoff --------------------------------------------------------------------------

ChatDraft chat_handoff_draft(const WidgetChatRequest& request, const caliper_builder::ContextBundle& bundle,
                             const caliper_builder::spec::Spec* open_spec) {
    ChatDraft d;
    d.text = request.request;
    for (const json& b : bundle.blocks) {
        if (!b.is_object()) continue;
        d.blocks.push_back(for_assistant(b));
        d.chips.push_back(chip_of(b));
    }
    if (!request.conversation.empty() || request.handoff) {
        std::string t = "The widget agent's conversation about this target so far:\n";
        for (const WidgetExchange& x : request.conversation)
            t += "Request: " + x.request + "\nReply: " + (x.reply.empty() ? std::string("(none)") : x.reply) + "\n";
        if (request.handoff)
            t += "The widget agent handed the request to the chat (" + request.handoff->kind + "): " +
                 request.handoff->reason + "\n";
        d.blocks.push_back(for_assistant(text_block(t)));
        d.chips.push_back("conversation");
    }
    if (!request.diff.empty()) {
        d.blocks.push_back(for_assistant(text_block(
            "The widget agent's preview, NOT APPLIED: nothing of it is in the spec. Its diff of the spec's canonical "
            "TOML:\n" + request.diff)));
        d.chips.push_back("preview diff");
    }
    json ids = {{"target", request.target.to_json()}};
    if (open_spec && !request.target.panel.empty()) {
        if (const caliper_builder::spec::Panel* p = open_spec->find_panel(request.target.panel)) {
            json sources = json::array();
            std::set<std::string> seen;
            for (const auto& kv : p->bind) {
                for (const std::string& ref : kv.second.bindings) {
                    const std::string sid = ref.substr(0, ref.find('.'));
                    if (!seen.insert(sid).second) continue;
                    json one = {{"id", sid}};
                    if (const auto* src = open_spec->find_source(sid)) {
                        if (src->catalog) one["catalog"] = *src->catalog;
                        if (src->from) one["from"] = *src->from;
                    }
                    sources.push_back(one);
                }
            }
            ids["panel"] = p->id;
            ids["sources"] = sources;
        }
    }
    d.blocks.push_back(for_assistant(text_block(
        "The ids of the target, for the caliper tools: " + ids.dump())));
    d.chips.push_back("ids");
    return d;
}

json chat_prompt_blocks(const std::string& text, const ChatDraft* draft, const acp::AgentInfo& agent,
                        std::vector<std::string>* dropped) {
    json out = json::array();
    if (draft) {
        for (const json& b : draft->blocks) {
            const std::string type = b.value("type", "");
            if (type == "resource" && !agent.prompt_embedded_context) {
                json t = text_block(b.value("resource", json::object()).value("text", ""));
                if (b.contains("annotations")) t["annotations"] = b["annotations"];
                out.push_back(std::move(t));
            } else if (type == "image" && !agent.prompt_image) {
                if (dropped) dropped->push_back("render");
            } else {
                out.push_back(b);
            }
        }
    }
    if (!text.empty()) out.push_back(text_block(text));
    return out;
}

json chat_opening_block() {
    return for_assistant(text_block(
        "You are working in the side chat of Caliper, an app that shows machine learning data in panels. The MCP "
        "server named caliper gives you Caliper's commands as tools: change the open spec, its sources, panels and "
        "layout, and the workspace's loaders through them, since they are validated, journaled and undoable, rather "
        "than by editing spec files with your own tools. Caliper shows the person each spec edit and each "
        "destructive command for approval itself, so do not ask for permission before you call a caliper tool."));
}

// ---- names --------------------------------------------------------------------------------

const char* chat_state_name(ChatState state) {
    switch (state) {
        case ChatState::no_agent: return "no_agent";
        case ChatState::idle: return "idle";
        case ChatState::starting: return "starting";
        case ChatState::ready: return "ready";
        case ChatState::running: return "running";
        case ChatState::cancelling: return "cancelling";
        case ChatState::needs_login: return "needs_login";
        case ChatState::exited: return "exited";
        case ChatState::failed: return "failed";
    }
    return "?";
}

const char* chat_action_label(ChatAction action) {
    switch (action) {
        case ChatAction::send: return "Send";
        case ChatAction::cancel: return "Cancel";
        case ChatAction::restart: return "Restart";
        case ChatAction::custom: return "Custom command";
        case ChatAction::clear: return "Clear";
    }
    return "";
}

const char* chat_action_key(ChatAction action) { return action == ChatAction::send ? "\xE2\x8F\x8E" : ""; }   // ⏎

std::string chat_stop_word(const std::string& stop_reason, const Error& error) {
    if (stop_reason == "cancelled") return "STOPPED";
    if (!error.ok()) return "FAILED";
    if (stop_reason == "end_turn") return "DONE";
    if (stop_reason == "refusal") return "DECLINED";
    if (stop_reason == "max_tokens" || stop_reason == "max_turn_requests") return "LIMIT";
    return stop_reason.empty() ? std::string("DONE") : stop_reason;
}

std::string chat_tool_status_word(const std::string& status) {
    if (status == "in_progress") return "running";
    if (status == "completed") return "done";
    if (status == "failed") return "failed";
    if (status == "cancelled") return "stopped";
    return status.empty() ? std::string("pending") : status;
}

// ---- the state machine --------------------------------------------------------------------

struct SideChat::StartJob {
    std::mutex m;
    bool done = false;
    bool abandoned = false;   // the chat let go of it: the worker shuts its client down
    Error error;
    acp::AgentInfo info;
    acp::SessionInfo session;
};

SideChat::SideChat(Options options, Hooks hooks) : options_(std::move(options)), hooks_(std::move(hooks)) {
    if (options_.agents.empty()) refresh_agents();
    else agents_ = options_.agents;
    const ChatAgent* chosen = agent();
    if (!chosen) {
        // The settings' agent, else the custom agent, else the first found.
        for (const ChatAgent& a : agents_)
            if (a.found && a.custom) chosen = &a;
        if (!chosen)
            for (const ChatAgent& a : agents_)
                if (a.found) {
                    chosen = &a;
                    break;
                }
        if (chosen) options_.settings.agent = chosen->id;
    }
    state_ = chosen ? ChatState::idle : ChatState::no_agent;
}

SideChat::~SideChat() {
    if (job_) {
        std::lock_guard<std::mutex> lock(job_->m);
        job_->abandoned = true;
    }
    job_.reset();
    retire_client(/*ending=*/false);   // no hooks while the chat is destroyed
    for (std::thread& t : workers_)
        if (t.joinable()) t.join();
}

void SideChat::refresh_agents() {
    agents_ = detect_chat_agents(options_.env.path);
    if (!trimmed(options_.settings.custom_command).empty()) {
        ChatAgent custom;
        if (!custom_chat_agent(options_.settings.custom_name, options_.settings.custom_command, options_.env.path, &custom))
            agents_.push_back(std::move(custom));
    }
}

const ChatAgent* SideChat::agent() const {
    for (const ChatAgent& a : agents_)
        if (a.id == options_.settings.agent && a.found) return &a;
    return nullptr;
}

std::string SideChat::agent_title() const {
    const ChatAgent* a = agent();
    return a ? a->title : std::string();
}

std::string chat_gpu_note(bool live_attached, bool local_model) {
    return live_attached && local_model ? "Shared GPU" : std::string();
}

bool SideChat::local_model() const {
    const ChatAgent* a = agent();
    if (!a) return false;
    const std::vector<std::string>& marked = options_.settings.local_model_agents;
    return std::find(marked.begin(), marked.end(), a->id) != marked.end();
}

std::string SideChat::gpu_note() const {
    return chat_gpu_note(hooks_.live_attached && hooks_.live_attached(), local_model());
}

bool SideChat::mode_skips_permission() const {
    return mode_ == "bypassPermissions" || mode_ == "yolo" || mode_ == "agent-full-access";
}

ChatItem& SideChat::add(ChatItem::Kind kind) {
    if (items_.size() >= kMaxItems) items_.erase(items_.begin(), items_.begin() + (long)(kMaxItems / 10));
    items_.emplace_back();
    items_.back().kind = kind;
    items_.back().turn = turn_;
    return items_.back();
}

ChatItem* SideChat::find_tool(const std::string& id) {
    for (auto it = items_.rbegin(); it != items_.rend(); ++it)
        if (it->kind == ChatItem::Kind::tool && it->tool.id == id) return &*it;
    return nullptr;
}

ChatItem* SideChat::last_of(ChatItem::Kind kind) {
    for (auto it = items_.rbegin(); it != items_.rend(); ++it) {
        if (it->turn != turn_) return nullptr;
        if (it->kind == kind) return &*it;
    }
    return nullptr;
}

void SideChat::notice(const std::string& tag, const Error& error, std::vector<std::string> lines) {
    ChatItem& n = add(ChatItem::Kind::notice);
    n.text = tag;
    n.detail = error.code;
    n.lines = std::move(lines);
    if (!error.message.empty()) n.lines.insert(n.lines.begin(), error.message);
}

bool SideChat::choose_agent(const std::string& id) {
    const ChatAgent* next = nullptr;
    for (const ChatAgent& a : agents_)
        if (a.id == id && a.found) next = &a;
    if (!next) return false;
    if (id == options_.settings.agent && state_ != ChatState::no_agent) return true;
    retire_client();
    options_.settings.agent = id;
    state_ = ChatState::idle;
    stalled_ = false;
    offer_restart_ = false;
    error_ = {};
    info_ = {};
    mode_.clear();
    if (hooks_.settings_changed) hooks_.settings_changed(options_.settings);
    return true;
}

Error SideChat::set_custom_agent(const std::string& name, const std::string& line) {
    ChatAgent probe;
    if (Error e = custom_chat_agent(name, line, options_.env.path, &probe)) return e;
    options_.settings.custom_name = probe.title;
    options_.settings.custom_command = trimmed(line);
    agents_.erase(std::remove_if(agents_.begin(), agents_.end(), [](const ChatAgent& a) { return a.custom; }), agents_.end());
    agents_.push_back(probe);
    if (!probe.found) {
        if (hooks_.settings_changed) hooks_.settings_changed(options_.settings);
        return {"chat.no_agent", "The command " + probe.needs.front() + " is not on PATH."};
    }
    // The custom agent replaces the chosen one even when its id stays "custom".
    options_.settings.agent.clear();
    choose_agent("custom");
    return {};
}

bool SideChat::set_local_model(bool on) {
    const ChatAgent* a = agent();
    if (!a) return false;
    std::vector<std::string>& marked = options_.settings.local_model_agents;
    const auto it = std::find(marked.begin(), marked.end(), a->id);
    if (on == (it != marked.end())) return true;
    if (on) marked.push_back(a->id);
    else marked.erase(it);
    if (hooks_.settings_changed) hooks_.settings_changed(options_.settings);
    return true;
}

void SideChat::set_text(std::string text) { text_ = std::move(text); }

void SideChat::open_handoff(ChatDraft draft) {
    draft_ = std::move(draft);
    text_ = draft_.text;
    focus_input_ = true;
}

void SideChat::clear_draft() {
    draft_ = {};
}

void SideChat::send() {
    if (state_ == ChatState::no_agent || !agent()) return;
    if (state_ == ChatState::running || state_ == ChatState::cancelling) return;   // one turn at a time
    if (state_ == ChatState::starting && pending_send_) return;
    const std::string text = trimmed(text_);
    if (text.empty() && draft_.blocks.empty()) return;
    ++turn_;
    ChatItem& p = add(ChatItem::Kind::person);
    p.text = text.substr(0, std::min(text.size(), kMaxItemText));
    p.chips = draft_.chips;
    pending_text_ = text;
    pending_draft_ = std::move(draft_);
    pending_send_ = true;
    draft_ = {};
    text_.clear();
    if (state_ == ChatState::ready) send_now();
    else if (state_ != ChatState::starting) start_agent();
}

void SideChat::start_agent() {
    const ChatAgent* a = agent();
    if (!a) {
        state_ = ChatState::no_agent;
        return;
    }
    retire_client(/*ending=*/false);   // a new agent follows at once
    if (hooks_.before_start) {
        if (Error e = hooks_.before_start(&options_.env)) {
            state_ = ChatState::failed;
            error_ = e;
            notice("FAILED", e);
            return;
        }
    }
    ChatLaunch launch;
    if (Error e = build_chat_launch(*a, options_.env, &launch, options_.client)) {
        state_ = ChatState::failed;
        error_ = e;
        notice("FAILED", e);
        return;
    }
    client_ = std::make_shared<acp::AcpClient>(launch.client);
    job_ = std::make_shared<StartJob>();
    std::shared_ptr<StartJob> job = job_;
    std::shared_ptr<acp::AcpClient> client = client_;
    workers_.emplace_back([job, client, launch] {
        acp::AgentInfo info;
        acp::SessionInfo session;
        Error e = client->start(&info);
        if (e.ok()) e = client->new_session(launch.cwd, launch.mcp_servers, &session);
        bool abandoned = false;
        {
            std::lock_guard<std::mutex> lock(job->m);
            job->info = info;
            job->session = session;
            job->error = e;
            job->done = true;
            abandoned = job->abandoned;
        }
        if (abandoned || !e.ok()) client->shutdown();
    });
    state_ = ChatState::starting;
    stalled_ = false;
    offer_restart_ = false;
    opened_ = false;
    error_ = {};
    mode_.clear();
}

void SideChat::retire_client(bool ending) {
    const bool had = job_ || client_;
    if (job_) {
        // The start worker still uses the client: it shuts it down itself.
        std::lock_guard<std::mutex> lock(job_->m);
        job_->abandoned = true;
    } else if (client_) {
        std::shared_ptr<acp::AcpClient> c = client_;
        workers_.emplace_back([c] { c->shutdown(); });
    }
    job_.reset();
    client_.reset();
    if (had && ending && hooks_.agent_ended) hooks_.agent_ended();
}

void SideChat::note(const std::string& tag, const std::string& line) {
    ChatItem& n = add(ChatItem::Kind::notice);
    n.text = tag;
    n.lines = {line};
}

void SideChat::finish_start() {
    if (!job_) return;
    {
        std::lock_guard<std::mutex> lock(job_->m);
        if (!job_->done) return;
    }
    std::shared_ptr<StartJob> job = job_;
    job_.reset();
    info_ = job->info;
    if (!job->error.ok()) {
        // The worker has shut the client down.
        client_.reset();
        if (hooks_.agent_ended) hooks_.agent_ended();
        error_ = job->error;
        if (error_.code == "acp.auth_required") {
            state_ = ChatState::needs_login;
            // The agent's own login methods, and the commands of the preset's hint.
            std::vector<std::string> lines;
            if (info_.auth_methods.is_array())
                for (const json& m : info_.auth_methods) {
                    if (!m.is_object()) continue;
                    std::string line = m.value("name", m.value("id", std::string()));
                    const std::string desc = m.value("description", std::string());
                    if (!desc.empty()) line += ": " + desc;
                    if (!line.empty()) lines.push_back(line);
                }
            if (const ChatAgent* a = agent()) {
                const std::string& h = a->login_hint;
                for (size_t i = h.find('`'); i != std::string::npos; i = h.find('`', i + 1)) {
                    const size_t j = h.find('`', i + 1);
                    if (j == std::string::npos) break;
                    lines.push_back("$ " + h.substr(i + 1, j - i - 1));
                    i = j;
                }
            }
            ChatItem& n = add(ChatItem::Kind::notice);
            n.text = "LOGIN";
            n.detail = error_.code;
            n.lines = std::move(lines);
        } else {
            state_ = ChatState::failed;
            notice("FAILED", error_);
        }
        return;
    }
    if (job->session.modes.is_object()) mode_ = job->session.modes.value("currentModeId", std::string());
    state_ = ChatState::ready;
    if (pending_send_) send_now();
}

void SideChat::send_now() {
    if (!client_) return;
    std::vector<std::string> dropped;
    json blocks = chat_prompt_blocks(pending_text_, &pending_draft_, info_, &dropped);
    if (!opened_ && options_.opening_prompt) blocks.insert(blocks.begin(), chat_opening_block());
    pending_send_ = false;
    pending_draft_ = {};
    const Error e = client_->prompt_blocks(blocks);
    if (!e.ok()) {
        notice("FAILED", e);
        return;
    }
    last_prompt_ = std::move(blocks);
    opened_ = true;
    state_ = ChatState::running;
    stalled_ = false;
}

void SideChat::cancel() {
    if (state_ != ChatState::running || !client_) return;
    client_->cancel();
    state_ = ChatState::cancelling;
}

void SideChat::restart() {
    for (ChatItem& it : items_) {
        if (it.kind == ChatItem::Kind::permission && it.answer.empty()) it.answer = "withdrawn";
        if (it.kind == ChatItem::Kind::tool && (it.tool.status == "pending" || it.tool.status == "in_progress"))
            it.tool.status = "cancelled";
    }
    stalled_ = false;
    start_agent();
}

void SideChat::answer_permission(std::uint64_t permission, const std::string& option_id) {
    for (ChatItem& it : items_) {
        if (it.kind != ChatItem::Kind::permission || it.permission != permission || !it.answer.empty()) continue;
        const acp::PermissionOption* chosen = nullptr;
        for (const auto& o : it.options)
            if (o.id == option_id) chosen = &o;
        if (!chosen || !client_) return;
        if (client_->respond_permission(permission, option_id).ok()) it.answer = chosen->name.empty() ? chosen->id : chosen->name;
        return;
    }
}

void SideChat::act(ChatAction action) {
    switch (action) {
        case ChatAction::send: send(); break;
        case ChatAction::cancel: cancel(); break;
        case ChatAction::restart: restart(); break;
        case ChatAction::custom: custom_open_ = true; break;
        case ChatAction::clear: clear_draft(); break;
    }
}

void SideChat::update() {
    finish_start();
    if (client_ && state_ != ChatState::starting) {
        acp::AcpEvent e;
        for (int n = 0; n < 1024 && client_ && client_->poll(&e); ++n) on_event(e);
    }
    mirror_approvals();
}

namespace {

void merge_tool(acp::ToolCallInfo* into, const acp::ToolCallInfo& from) {
    if (from.has_title) into->title = from.title;
    if (from.has_kind) into->kind = from.kind;
    if (from.has_status) into->status = from.status;
    if (from.has_content) into->content = from.content;
    if (from.has_locations) into->locations = from.locations;
    if (!from.raw_input.is_null()) into->raw_input = from.raw_input;
    if (!from.raw_output.is_null()) into->raw_output = from.raw_output;
    into->has_title = into->has_title || from.has_title;
    into->has_kind = into->has_kind || from.has_kind;
    into->has_status = into->has_status || from.has_status;
    into->has_content = into->has_content || from.has_content;
}

// The first diff of a call's content as a unified diff, and its file.
void diff_of(const acp::ToolCallInfo& call, std::string* diff, std::string* path,
             std::vector<WidgetDiffLine>* compact = nullptr) {
    for (const acp::ToolCallContent& c : call.content) {
        if (c.type != "diff") continue;
        const std::string name = fs::path(c.path).filename().string();
        *diff = caliper_builder::unified_line_diff(c.has_old_text ? c.old_text : std::string(), c.new_text,
                                                   c.has_old_text ? name : std::string("/dev/null"), name);
        *path = name;
        if (compact) *compact = chat_compact_diff(name, c.has_old_text ? c.old_text : std::string(), c.new_text);
        return;
    }
}

}  // namespace

void SideChat::on_event(const acp::AcpEvent& e) {
    using K = acp::AcpEventKind;
    switch (e.kind) {
        case K::user_message_chunk: break;   // the chat shows the person's messages as it sent them
        case K::agent_message_chunk:
        case K::agent_thought_chunk: {
            const ChatItem::Kind kind = e.kind == K::agent_message_chunk ? ChatItem::Kind::agent : ChatItem::Kind::thought;
            const std::string piece = e.content_type == "text" || e.content_type.empty() ? e.text : "[" + e.content_type + "]";
            ChatItem* last = items_.empty() ? nullptr : &items_.back();
            if (last && last->kind == kind && last->turn == turn_) append_bounded(&last->text, piece);
            else add(kind).text = piece.substr(0, std::min(piece.size(), kMaxItemText));
            break;
        }
        case K::tool_call:
        case K::tool_call_update: {
            ChatItem* t = find_tool(e.tool_call.id);
            if (!t) {
                t = &add(ChatItem::Kind::tool);
                t->tool.id = e.tool_call.id;
            }
            merge_tool(&t->tool, e.tool_call);
            t->command = caliper_command_of(t->tool);
            diff_of(t->tool, &t->diff, &t->diff_path);
            break;
        }
        case K::plan: {
            ChatItem* p = last_of(ChatItem::Kind::plan);
            if (!p) p = &add(ChatItem::Kind::plan);
            p->plan = e.plan;
            break;
        }
        case K::current_mode: mode_ = e.mode_id; break;
        case K::available_commands:
        case K::other_update: break;
        case K::permission_request: {
            ChatItem& p = add(ChatItem::Kind::permission);
            p.permission = e.permission_id;
            p.tool = e.tool_call;
            if (ChatItem* t = find_tool(e.tool_call.id)) {
                acp::ToolCallInfo merged = t->tool;
                merge_tool(&merged, e.tool_call);
                p.tool = merged;
            }
            p.command = caliper_command_of(p.tool);
            for (const acp::PermissionOption& o : e.options)
                if (o.kind != "allow_always" || options_.settings.allow_always) p.options.push_back(o);
            diff_of(p.tool, &p.diff, &p.diff_path, &p.compact);
            break;
        }
        case K::permission_withdrawn:
            for (ChatItem& it : items_)
                if (it.kind == ChatItem::Kind::permission && it.permission == e.permission_id && it.answer.empty())
                    it.answer = "withdrawn";
            break;
        case K::turn_ended: end_turn(e); break;
        case K::protocol_fault:
            if (faults_.size() >= kMaxFaults) faults_.erase(faults_.begin());
            faults_.push_back(e.error);
            break;
        case K::stalled:
            if (state_ == ChatState::running || state_ == ChatState::cancelling) {
                stalled_ = true;
                notice("STALLED", e.error);
            }
            break;
        case K::agent_exited: {
            std::vector<std::string> tail;
            if (client_) {
                const std::vector<std::string> lines = split_lines(client_->stderr_log());
                for (auto it = lines.rbegin(); it != lines.rend() && (int)tail.size() < kStderrTail; ++it)
                    if (!trimmed(*it).empty()) tail.insert(tail.begin(), *it);
            }
            for (ChatItem& it : items_) {
                if (it.kind == ChatItem::Kind::permission && it.answer.empty()) it.answer = "withdrawn";
                if (it.kind == ChatItem::Kind::tool && (it.tool.status == "pending" || it.tool.status == "in_progress"))
                    it.tool.status = "cancelled";
            }
            Error why = e.error;
            if (e.has_exit_status) why.message = e.exit_status.describe();
            ChatItem& n = add(ChatItem::Kind::notice);
            n.text = "EXITED";
            n.detail = why.message;
            n.lines = std::move(tail);
            error_ = e.error;
            state_ = ChatState::exited;
            stalled_ = false;
            pending_send_ = false;
            retire_client();
            break;
        }
    }
}

void SideChat::end_turn(const acp::AcpEvent& e) {
    for (ChatItem& it : items_) {
        if (it.turn != turn_) continue;
        if (it.kind == ChatItem::Kind::permission && it.answer.empty()) it.answer = "withdrawn";
        if (e.stop_reason == "cancelled" && it.kind == ChatItem::Kind::tool &&
            (it.tool.status == "pending" || it.tool.status == "in_progress"))
            it.tool.status = "cancelled";
    }
    stalled_ = false;
    if (e.error.code == "acp.agent_exited") return;   // the agent_exited event follows and says it
    ChatItem& s = add(ChatItem::Kind::stop);
    s.text = chat_stop_word(e.stop_reason, e.error);
    s.detail = e.error.code;
    if (e.error.code == "acp.cancel_unanswered") offer_restart_ = true;
    state_ = ChatState::ready;
}

void SideChat::mirror_approvals() {
    if (!hooks_.pending_approvals) return;
    bool open = false;
    for (const ChatItem& it : items_)
        if (it.kind == ChatItem::Kind::approval && it.answer.empty()) open = true;
    if (!open && state_ != ChatState::running && state_ != ChatState::cancelling) return;
    const std::vector<caliper_builder::PendingApproval> pending = hooks_.pending_approvals();
    std::set<std::string> ids;
    for (const auto& p : pending)
        if (p.invocation.origin == caliper_builder::CommandOrigin::agent) ids.insert(p.approval);
    // A row whose approval was reissued for changed content follows the new
    // one, still waiting, so one change keeps one row.
    if (hooks_.reissued_as)
        for (ChatItem& it : items_)
            if (it.kind == ChatItem::Kind::approval && it.answer.empty() && !ids.count(it.approval))
                for (int hops = 0; hops < 8 && !ids.count(it.approval); ++hops) {
                    const std::string next = hooks_.reissued_as(it.approval);
                    if (next.empty() || next == it.approval) break;
                    it.approval = next;
                }
    for (const auto& p : pending) {
        if (p.invocation.origin != caliper_builder::CommandOrigin::agent) continue;
        bool known = false;
        for (const ChatItem& it : items_)
            if (it.kind == ChatItem::Kind::approval && it.approval == p.approval) known = true;
        if (known || (state_ != ChatState::running && state_ != ChatState::cancelling)) continue;
        ChatItem& a = add(ChatItem::Kind::approval);
        a.approval = p.approval;
        a.command = p.invocation.name;
    }
    for (ChatItem& it : items_)
        if (it.kind == ChatItem::Kind::approval && it.answer.empty() && !ids.count(it.approval)) {
            const std::string outcome = hooks_.approval_outcome ? hooks_.approval_outcome(it.approval) : std::string();
            it.answer = outcome.empty() ? "decided" : outcome;
        }
}

std::string SideChat::tag() const {
    switch (state_) {
        case ChatState::no_agent: return "NO AGENT";
        case ChatState::starting: return "STARTING";
        case ChatState::running: return stalled_ ? "STALLED" : "RUNNING";
        case ChatState::cancelling: return "STOPPING";
        case ChatState::needs_login: return "LOGIN";
        case ChatState::exited: return "EXITED";
        case ChatState::failed: return "FAILED";
        case ChatState::idle:
        case ChatState::ready: return "";
    }
    return "";
}

std::vector<ChatAction> SideChat::header_actions() const {
    switch (state_) {
        case ChatState::running:
            if (stalled_) return {ChatAction::restart};
            return {};
        case ChatState::cancelling:
        case ChatState::needs_login:
        case ChatState::exited:
        case ChatState::failed: return {ChatAction::restart};
        case ChatState::ready:
            if (offer_restart_) return {ChatAction::restart};
            return {};
        default: return {};
    }
}

ChatAction SideChat::input_action() const {
    return state_ == ChatState::running || state_ == ChatState::cancelling ? ChatAction::cancel : ChatAction::send;
}

// ---- the rows -----------------------------------------------------------------------------

std::vector<ChatRow> chat_rows(const std::vector<ChatItem>& items, const std::map<std::string, bool>& open) {
    std::vector<ChatRow> out;
    auto key_of = [](const ChatItem& it) { return it.command.empty() ? it.tool.title : it.command; };
    size_t i = 0;
    while (i < items.size()) {
        const ChatItem& it = items[i];
        if (it.kind == ChatItem::Kind::approval && it.answer.empty()) {
            ChatRow r;
            r.kind = ChatRow::Kind::waiting_approval;
            r.item = i++;
            out.push_back(std::move(r));
            continue;
        }
        if (it.kind != ChatItem::Kind::tool) {
            ChatRow r;
            r.item = i++;
            out.push_back(std::move(r));
            continue;
        }
        // A run of calls of one command and the decided approvals of that command.
        const std::string key = key_of(it);
        ChatRow g;
        g.kind = ChatRow::Kind::group;
        g.key = it.tool.id;
        g.command = key;
        g.caliper = !it.command.empty();
        std::vector<size_t> waiting;
        std::map<std::string, int> outcomes;   // a decided approval's answer, counted
        bool failed = false, running = false, pending = false, stopped = false;
        size_t j = i;
        for (; j < items.size(); ++j) {
            const ChatItem& m = items[j];
            if (m.kind == ChatItem::Kind::tool && key_of(m) == key) {
                g.members.push_back(j);
                ++g.count;
                const std::string& st = m.tool.status;
                failed = failed || st == "failed";
                stopped = stopped || st == "cancelled";
                running = running || st == "in_progress";
                pending = pending || st.empty() || st == "pending";
            } else if (m.kind == ChatItem::Kind::approval && g.caliper && m.command == key) {
                if (m.answer.empty()) {
                    waiting.push_back(j);
                } else {
                    g.members.push_back(j);
                    ++outcomes[m.answer];
                }
            } else {
                break;
            }
        }
        const bool approvals = !outcomes.empty();
        if (g.count == 1 && !approvals) {
            // A lone call: a plain row, as before.
            ChatRow r;
            r.item = i;
            out.push_back(std::move(r));
        } else {
            // "approved" when every approval was given; otherwise each outcome with its count.
            std::string a;
            const int approved = outcomes.count("approved") ? outcomes["approved"] : 0;
            for (const char* o : {"denied", "timed out", "failed", "decided"})
                if (outcomes.count(o)) a += (a.empty() ? "" : " \xC2\xB7 ") + std::to_string(outcomes[o]) + " " + o;
            if (approved) a += a.empty() ? std::string("approved") : " \xC2\xB7 " + std::to_string(approved) + " approved";
            g.approvals = a;
            const bool refused = outcomes.count("denied") || outcomes.count("timed out") || outcomes.count("failed");
            g.mark = failed ? "failed" : stopped ? "cancelled" : running ? "in_progress" : pending ? "pending" : "completed";
            g.status = chat_tool_status_word(g.mark);
            g.alert = failed || refused;
            const auto o = open.find(g.key);
            g.expanded = o != open.end() ? o->second : g.alert;
            out.push_back(std::move(g));
        }
        for (size_t w : waiting) {
            ChatRow r;
            r.kind = ChatRow::Kind::waiting_approval;
            r.item = w;
            out.push_back(std::move(r));
        }
        i = j;
    }
    return out;
}

std::string chat_group_summary(const ChatRow& row) {
    return row.approvals.empty() ? row.status : row.approvals + " \xC2\xB7 " + row.status;
}

// ---- the layout ---------------------------------------------------------------------------

ChatLayoutMetrics ChatLayoutMetrics::for_text_scale(float s) {
    s = std::max(0.5f, s);
    ChatLayoutMetrics m;
    auto r = [s](float v) { return std::round(v * s); };
    m.header = r(26);
    m.row = r(30);
    m.line = r(18);
    m.chip = r(22);
    m.pad = r(8);
    m.gap = r(6);
    m.button = r(22);
    m.button_pad = r(8);
    m.key_gap = r(6);
    m.picker_mark = r(14);
    m.min_input = r(120);
    m.min_width = r(300);
    m.min_height = r(220);
    m.min_canvas = r(480);
    m.min_upper = r(160);
    m.min_left = r(220);
    return m;
}

ChatColumnLayout layout_chat_column(const ChatColumnInput& in, const ChatLayoutMetrics& m) {
    ChatColumnLayout out;
    const WidgetRect& w = in.window;
    float col = std::max(m.min_width, std::round(w.w * std::clamp(in.share, 0.05f, 0.9f)));
    float left = in.left_open ? std::max(m.min_left, std::round(w.w * std::clamp(in.left_share, 0.05f, 0.9f))) : 0.0f;
    if (w.w - col - left < m.min_canvas) col = std::max(m.min_width, w.w - left - m.min_canvas);
    if (in.left_open && w.w - col - left < m.min_canvas) left = std::max(m.min_left, w.w - col - m.min_canvas);
    col = std::min(col, w.w);
    left = std::min(left, std::max(0.0f, w.w - col));
    if (in.left_open) out.left = {w.x, w.y, left, w.h};
    out.canvas = {w.x + left, w.y, w.w - col - left, w.h};
    out.column = {w.x + w.w - col, w.y, col, w.h};
    out.chat = out.column;
    if (in.upper_open) {
        float upper = std::round(w.h * std::clamp(in.upper_share, 0.0f, 1.0f));
        float chat = w.h - upper;
        if (chat < m.min_height) {
            chat = std::min(m.min_height, w.h);
            upper = w.h - chat;
        }
        if (upper >= m.min_upper) {
            out.upper = {out.column.x, w.y, col, upper};
            out.chat = {out.column.x, w.y + upper, col, chat};
        }
    }
    return out;
}

ChatColumnLayout layout_chat_page(const WidgetRect& window, float top_strips, float bottom_strips, bool chat_open,
                                  bool left_open, const ChatLayoutMetrics& m) {
    ChatColumnInput in;
    in.window = {window.x, window.y + top_strips, window.w, std::max(0.0f, window.h - top_strips - bottom_strips)};
    in.left_open = left_open;
    ChatColumnLayout out = layout_chat_column(in, m);
    if (!chat_open) {
        out.canvas.w += out.column.w;
        out.column = out.upper = out.chat = WidgetRect{};
    }
    return out;
}

namespace {

float button_width(ChatAction a, bool keys, const ChatLayoutMetrics& m, const ChatMeasure& measure) {
    float w = 2.0f * m.button_pad + measure(chat_action_label(a), ChatFont::label);
    const char* key = chat_action_key(a);
    if (keys && key[0]) w += measure(key, ChatFont::key) + m.key_gap;
    return std::ceil(w);
}

std::string upper(std::string s) {
    for (char& c : s) c = (char)std::toupper((unsigned char)c);
    return s;
}

}  // namespace

ChatPaneLayout layout_chat_pane(const ChatPaneInput& in, const ChatLayoutMetrics& m, const ChatMeasure& measure) {
    ChatPaneLayout out;
    const WidgetRect& P = in.pane;
    out.header = {P.x, P.y, P.w, std::min(m.header, P.h)};
    const int lines = std::clamp(in.input_lines, 1, std::max(1, m.max_input_lines));
    const float row_h = m.row + (float)(lines - 1) * m.line;

    // The input row and its collapses: the picker's name, then the keys.
    auto lay_input = [&](bool picker_label, bool keys) {
        out.input_row = {P.x + m.pad, P.bottom() - m.pad - row_h, std::max(0.0f, P.w - 2.0f * m.pad), row_h};
        const float by = std::round(out.input_row.bottom() - (m.row + m.button) * 0.5f);
        const float pw = 2.0f * m.button_pad + m.picker_mark + (picker_label ? measure(in.picker, ChatFont::label) : 0.0f);
        out.picker = {out.input_row.x + m.gap * 0.5f, by, std::ceil(pw), m.button};
        const float sw = button_width(in.input_action, keys, m, measure);
        out.send = {out.input_row.right() - m.gap * 0.5f - sw, by, sw, m.button};
        const float ix = out.picker.right() + m.gap;
        out.input = {ix, out.input_row.y, std::max(0.0f, out.send.x - m.gap - ix), row_h};
        out.picker_label = picker_label;
        out.keys = keys;
    };
    lay_input(true, true);
    if (out.input.w < m.min_input) lay_input(false, true);
    if (out.input.w < m.min_input) lay_input(false, false);

    // The draft's chips above the input row, with Clear at the right.
    if (!in.chips.empty()) {
        out.chips = {P.x + m.pad, out.input_row.y - m.gap - m.chip, std::max(0.0f, P.w - 2.0f * m.pad), m.chip};
        const float cw = std::ceil(2.0f * m.button_pad + measure(chat_action_label(ChatAction::clear), ChatFont::label));
        out.clear = {out.chips.right() - cw, out.chips.y, cw, m.chip};
        const float fold = std::ceil(measure("+" + std::to_string(in.chips.size()), ChatFont::label)) + m.gap;
        float x = out.chips.x;
        for (size_t i = 0; i < in.chips.size(); ++i) {
            const float w = std::ceil(2.0f * m.button_pad + measure(in.chips[i], ChatFont::label));
            const bool last = i + 1 == in.chips.size();
            const float limit = out.clear.x - m.gap - (last ? 0.0f : fold);
            if (x + w > limit) break;
            out.chip_rects.push_back({x, out.chips.y, w, m.chip});
            x += w + m.gap;
        }
        out.chips_shown = (int)out.chip_rects.size();
    }

    // The header: the title tag, the agent and its version at the left; the
    // buttons, the faults and the state's tag at the right.
    const WidgetRect& H = out.header;
    const float bh = std::min(m.button, H.h);
    const float by = std::round(H.y + (H.h - bh) * 0.5f);
    const float tw = std::ceil(measure(upper(in.title), ChatFont::tag));
    out.title = {H.x + m.pad, H.y, tw, H.h};
    float x = out.title.right() + 2.0f * m.gap;
    float right = H.right() - m.pad;
    for (size_t i = in.header_actions.size(); i-- > 0;) {
        const float w = button_width(in.header_actions[i], out.keys, m, measure);
        out.header_buttons.insert(out.header_buttons.begin(), WidgetRect{right - w, by, w, bh});
        right -= w + m.gap;
    }
    // The faults chip, then the state's tag, give way when the header is too narrow for them.
    const float fw = in.faults > 0 ? std::ceil(measure("FAULTS " + std::to_string(in.faults), ChatFont::tag) + 2.0f * m.button_pad) : 0.0f;
    const float tgw = in.tag.empty() ? 0.0f : std::ceil(measure(in.tag, ChatFont::tag) + m.picker_mark);
    const bool tag_fits = tgw > 0 && right - tgw >= x;
    const bool faults_fit = fw > 0 && right - fw - (tag_fits ? tgw + m.gap : 0.0f) >= x;
    if (faults_fit) {
        out.faults = {right - fw, by, fw, bh};
        right -= fw + m.gap;
    }
    if (tag_fits && right - tgw >= x) {
        out.tag = {right - tgw, H.y, tgw, H.h};
        right -= tgw + m.gap;
    }
    const float aw = in.agent.empty() ? 0.0f : std::ceil(measure(in.agent, ChatFont::label));
    const float vw = in.version.empty() ? 0.0f : std::ceil(measure(in.version, ChatFont::mono));
    const float mw = in.mode.empty() ? 0.0f : std::ceil(measure(in.mode, ChatFont::mono));
    // The GPU note: a framed noun after the agent's name, which gives way
    // after the version and before the name.
    const float nw = in.note.empty() ? 0.0f : std::ceil(measure(in.note, ChatFont::tag) + 2.0f * m.button_pad);
    if (aw > 0 && x + aw <= right) {
        out.agent = {x, H.y, aw, H.h};
        x += aw + m.gap;
        const bool note_fits = nw > 0 && x + nw <= right;
        if (note_fits) {
            out.note = {x, by, nw, bh};
            x += nw + m.gap;
        }
        if ((nw == 0 || note_fits) && vw > 0 && x + vw <= right) {
            out.version = {x, H.y, vw, H.h};
            x += vw + 2.0f * m.gap;
            if (mw > 0 && x + mw <= right) out.mode = {x, H.y, mw, H.h};
        }
    }

    const float top = H.bottom();
    const float bottom = in.chips.empty() ? out.input_row.y - m.pad : out.chips.y - m.gap;
    out.conversation = {P.x, top, P.w, std::max(0.0f, bottom - top)};
    return out;
}

ChatPaneInput chat_pane_input(const SideChat& chat, const WidgetRect& pane) {
    ChatPaneInput in;
    in.pane = pane;
    in.agent = chat.agent_title();
    in.version = chat.agent_info().version;
    in.mode = chat.mode();
    in.tag = chat.tag();
    in.note = chat.gpu_note();
    in.header_actions = chat.header_actions();
    in.faults = (int)chat.faults().size();
    in.picker = chat.agent() ? chat.agent_title() : std::string("Agent");
    in.input_lines = 1 + (int)std::count(chat.text().begin(), chat.text().end(), '\n');
    in.chips = chat.draft().chips;
    in.input_action = chat.input_action();
    return in;
}

// ---- the ImGui side -----------------------------------------------------------------------

namespace {

ImVec2 v2(float x, float y) { return ImVec2(x, y); }

ImVec2 extent(ImFont* f, float size, const std::string& s, float wrap = 0.0f) {
    if (!f) f = ImGui::GetFont();
    return f->CalcTextSizeA(size, FLT_MAX, wrap, s.c_str(), s.c_str() + s.size());
}

std::string fit(ImFont* f, float size, const std::string& s, float w) {
    if (extent(f, size, s).x <= w) return s;
    static const std::string kDots = "\xE2\x80\xA6";
    if (w <= 0.0f || extent(f, size, kDots).x > w) return {};
    size_t lo = 0, hi = s.size();
    while (lo < hi) {   // the longest prefix that fits with the dots
        size_t mid = (lo + hi + 1) / 2;
        while (mid > 0 && ((unsigned char)s[mid] & 0xC0) == 0x80) --mid;
        if (mid <= lo) break;
        if (extent(f, size, s.substr(0, mid) + kDots).x <= w) lo = mid;
        else hi = mid - 1;
    }
    while (lo > 0 && ((unsigned char)s[lo] & 0xC0) == 0x80) --lo;
    return s.substr(0, lo) + kDots;
}

void draw_str(ImDrawList* dl, ImFont* f, float size, ImVec2 p, ImU32 c, const std::string& s, float wrap = 0.0f) {
    dl->AddText(f ? f : ImGui::GetFont(), size, ImVec2(std::round(p.x), std::round(p.y)), c, s.c_str(), s.c_str() + s.size(),
                wrap);
}


std::string first_line(const std::string& s) {
    const std::string t = trimmed(s);
    const size_t nl = t.find('\n');
    return nl == std::string::npos ? t : t.substr(0, nl);
}

int resize_callback(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        auto* s = static_cast<std::string*>(data->UserData);
        s->resize((size_t)data->BufTextLen);
        data->Buf = const_cast<char*>(s->c_str());
    }
    return 0;
}

// The house's buttons: an outlined key button, or the cut plate of a primary.
bool house_button(const char* id, ImRect b, const std::string& label, const char* key, bool primary, bool keys,
                  const SpecLook& L, bool enabled = true) {
    const float s = L.text_scale;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGui::SetCursorScreenPos(b.Min);
    const bool clicked = ImGui::InvisibleButton(id, b.GetSize()) && enabled;
    const bool hovered = enabled && ImGui::IsItemHovered(), held = enabled && ImGui::IsItemActive();
    ImU32 outline = primary ? L.text : L.rule, fill = 0, key_ink = primary ? L.accent : L.secondary, ink = L.text;
    if (!enabled) {
        outline = L.rule_faint;
        key_ink = ink = L.faint;
    } else if (held && hovered) {
        fill = L.text;
        outline = fill;
        key_ink = ink = L.deep;
    } else if (hovered) {
        fill = L.plate_hover;
        key_ink = L.accent;
    }
    if (primary) sternbild::cut_plate(dl, b.Min, b.Max, outline, fill, std::round(4.0f * s));
    else {
        if (fill) dl->AddRectFilled(b.Min, b.Max, fill);
        dl->AddRect(b.Min, b.Max, outline, 0.0f, 0, 1.0f);
    }
    const float ui_h = extent(L.ui, L.ui_size, "Ag").y, mono_h = extent(L.mono, L.mono_size, "Ag").y;
    float x = b.Min.x + std::round(8.0f * s);
    if (keys && key && key[0]) {
        draw_str(dl, L.mono, L.mono_size, v2(x, b.Min.y + (b.GetHeight() - mono_h) * 0.5f), key_ink, key);
        x += extent(L.mono, L.mono_size, key).x + std::round(6.0f * s);
    }
    draw_str(dl, L.ui, L.ui_size, v2(x, b.Min.y + (b.GetHeight() - ui_h) * 0.5f), ink,
         fit(L.ui, L.ui_size, label, b.Max.x - x - std::round(4.0f * s)));
    return clicked;
}

// A small tracked uppercase tag; returns its width.
float draw_tag(ImDrawList* dl, const SpecLook& L, ImVec2 p, float line_h, ImU32 c, const std::string& t) {
    const float tracking = std::round(L.small_size * 0.08f);
    const float h = extent(L.ui, L.small_size, "Ag").y;
    sternbild::text_tracked(dl, L.ui, L.small_size, v2(std::round(p.x), std::round(p.y + (line_h - h) * 0.5f)), c, t.c_str(),
                            tracking);
    return sternbild::text_tracked_width(L.ui, L.small_size, t.c_str(), tracking);
}

// The mark of a tool call's or a plan entry's status.
void status_mark(ImDrawList* dl, const SpecLook& L, ImVec2 c, const std::string& status) {
    const float half = std::round(3.5f * L.text_scale);
    if (status == "in_progress") sternbild::diamond(dl, c, half, L.accent, true);
    else if (status == "completed") sternbild::diamond(dl, c, half, L.secondary, true);
    else if (status == "failed" || status == "cancelled") {
        sternbild::diamond(dl, c, half, L.faint, false);
        dl->AddRectFilled(v2(c.x - half - 1.0f, std::round(c.y)), v2(c.x + half + 1.0f, std::round(c.y) + 1.0f), L.faint);
    } else sternbild::diamond(dl, c, half, L.faint, false);
}

}  // namespace

void SideChat::draw(const ImRect& pane_rect, const SpecLook& L) {
    const float s = L.text_scale;
    const ChatLayoutMetrics m = ChatLayoutMetrics::for_text_scale(s);
    const float tracking = std::round(L.small_size * 0.08f);
    const ChatMeasure measure = [&L, tracking](const std::string& t, ChatFont font) {
        switch (font) {
            case ChatFont::label: return extent(L.ui, L.ui_size, t).x;
            case ChatFont::key:
            case ChatFont::mono: return extent(L.mono, L.mono_size, t).x;
            case ChatFont::tag: return sternbild::text_tracked_width(L.ui, L.small_size, t.c_str(), tracking);
        }
        return 0.0f;
    };
    const WidgetRect P{pane_rect.Min.x, pane_rect.Min.y, pane_rect.GetWidth(), pane_rect.GetHeight()};
    if (P.empty()) return;
    const ChatPaneInput in = chat_pane_input(*this, P);
    const ChatPaneLayout lay = layout_chat_pane(in, m, measure);
    auto rr = [](const WidgetRect& r) { return ImRect(v2(r.x, r.y), v2(r.right(), r.bottom())); };
    const float ui_h = extent(L.ui, L.ui_size, "Ag").y;
    const float small_h = extent(L.ui, L.small_size, "Ag").y;
    const float mono_h = extent(L.mono, L.mono_size, "Ag").y;

    ImGui::PushID("##side.chat");
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->PushClipRect(pane_rect.Min, pane_rect.Max, true);
    dl->AddRectFilled(pane_rect.Min, pane_rect.Max, L.ground);
    dl->AddRectFilled(pane_rect.Min, v2(pane_rect.Min.x + 1.0f, pane_rect.Max.y), L.rule);

    // ---- the header
    {
        const ImRect h = rr(lay.header);
        dl->AddRectFilled(h.Min, h.Max, L.plate);
        dl->AddRectFilled(v2(h.Min.x, h.Max.y - 1.0f), h.Max, L.rule);
        draw_tag(dl, L, v2(lay.title.x, lay.title.y), lay.title.h, L.text, upper(in.title));
        if (!lay.agent.empty())
            draw_str(dl, L.ui, L.ui_size, v2(lay.agent.x, lay.agent.y + (lay.agent.h - ui_h) * 0.5f), L.secondary, in.agent);
        if (!lay.note.empty()) {   // the GPU note, framed in bronze as the widget agent's settings show it
            const ImRect nr = rr(lay.note);
            dl->AddRect(nr.Min, nr.Max, L.accent, 0.0f, 0, 1.0f);
            const ImVec2 e = extent(L.ui, L.small_size, in.note);
            draw_str(dl, L.ui, L.small_size,
                     v2(std::round(nr.Min.x + (nr.GetWidth() - e.x) * 0.5f), std::round(nr.Min.y + (nr.GetHeight() - e.y) * 0.5f)),
                     L.accent, in.note);
        }
        if (!lay.version.empty())
            draw_str(dl, L.mono, L.mono_size, v2(lay.version.x, lay.version.y + (lay.version.h - mono_h) * 0.5f), L.faint,
                 in.version);
        if (!lay.mode.empty())   // the agent's mode, in bronze when it skips the agent's permission prompts
            draw_str(dl, L.mono, L.mono_size, v2(lay.mode.x, lay.mode.y + (lay.mode.h - mono_h) * 0.5f),
                     mode_skips_permission() ? L.accent : L.faint, in.mode);
        if (!lay.tag.empty()) {
            const bool live = state_ == ChatState::running && !stalled_;
            const bool alarm = state_ == ChatState::exited || state_ == ChatState::failed ||
                               state_ == ChatState::needs_login || stalled_;
            float x = lay.tag.x;
            const float half = std::round(3.5f * s);
            if (live || state_ == ChatState::starting) {
                sternbild::diamond(dl, v2(x + half, lay.tag.y + lay.tag.h * 0.5f), half, L.accent, live);
            }
            x += m.picker_mark;
            draw_tag(dl, L, v2(x, lay.tag.y), lay.tag.h, alarm ? L.accent : L.secondary, in.tag);
        }
        if (!lay.faults.empty()) {
            const ImRect f = rr(lay.faults);
            ImGui::SetCursorScreenPos(f.Min);
            if (ImGui::InvisibleButton("##chat.faults", f.GetSize())) diagnostics_open_ = !diagnostics_open_;
            const bool hov = ImGui::IsItemHovered();
            if (hov) dl->AddRectFilled(f.Min, f.Max, L.plate_hover);
            dl->AddRect(f.Min, f.Max, diagnostics_open_ ? L.accent : L.rule, 0.0f, 0, 1.0f);
            draw_tag(dl, L, v2(f.Min.x + m.button_pad, f.Min.y), f.GetHeight(), L.secondary, "FAULTS " + std::to_string(in.faults));
        }
        for (size_t i = 0; i < lay.header_buttons.size() && i < in.header_actions.size(); ++i) {
            const ChatAction a = in.header_actions[i];
            const std::string id = std::string("##chat.header.") + chat_action_label(a);
            if (house_button(id.c_str(), rr(lay.header_buttons[i]), chat_action_label(a), chat_action_key(a), false,
                             lay.keys, L))
                act(a);
        }
    }

    // ---- the conversation
    const ImRect conv = rr(lay.conversation);
    if (conv.GetHeight() > 1.0f && conv.GetWidth() > 1.0f) {
        ImGui::SetCursorScreenPos(conv.Min);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, v2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, v2(0, 0));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, 0);
        ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, 0);
        if (ImGui::BeginChild("##chat.conversation", conv.GetSize(), ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
            ImDrawList* cdl = ImGui::GetWindowDrawList();
            const bool at_bottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f;
            const float pad = m.pad + std::round(2.0f * s);
            const float x_left = ImGui::GetCursorScreenPos().x + pad;
            const float w_full = std::max(40.0f, ImGui::GetContentRegionAvail().x - 2.0f * pad);
            float x0 = x_left, w = w_full;   // a group's members are drawn indented
            const float line = std::round(22.0f * s);
            const float gap = std::round(8.0f * s);
            ImGui::Dummy(v2(1.0f, gap));
            // One item: draws with its top-left and returns its height.
            auto place = [&](float h, const std::function<void(ImVec2)>& fn) {
                const ImVec2 p = v2(x0, ImGui::GetCursorScreenPos().y);
                if (ImGui::IsRectVisible(v2(x0, p.y), v2(x0 + w, p.y + h))) fn(p);
                ImGui::SetCursorScreenPos(v2(x0 - pad, p.y));
                ImGui::Dummy(v2(w, h));
                ImGui::Dummy(v2(1.0f, gap));
            };

            if (state_ == ChatState::no_agent) {
                // No agent found: the agents as nouns, and one action.
                place(line, [&](ImVec2 p) { draw_tag(cdl, L, p, line, L.faint, "AGENTS"); });
                for (const ChatAgent& a : agents_) {
                    if (a.custom) continue;
                    // The name and whether it was found, as the picker shows them; never a command line.
                    place(line, [&](ImVec2 p) {
                        draw_str(cdl, L.ui, L.ui_size, v2(p.x, p.y + (line - ui_h) * 0.5f), a.found ? L.text : L.secondary, a.title);
                        const char* word = a.found ? "FOUND" : "NOT FOUND";
                        const float nw = sternbild::text_tracked_width(L.ui, L.small_size, word, tracking);
                        const float nx = p.x + std::max(std::round(140.0f * s), extent(L.ui, L.ui_size, a.title).x + gap);
                        if (nx + nw <= p.x + w) draw_tag(cdl, L, v2(nx, p.y), line, L.faint, word);
                    });
                }
                const float bw = std::ceil(2.0f * m.button_pad + extent(L.ui, L.ui_size, chat_action_label(ChatAction::custom)).x +
                                           std::round(8.0f * s));
                const ImVec2 p = v2(x0, ImGui::GetCursorScreenPos().y + gap);
                if (house_button("##chat.custom.open", ImRect(p, v2(p.x + bw, p.y + m.button)),
                                 chat_action_label(ChatAction::custom), "", true, true, L))
                    custom_open_ = true;
                ImGui::SetCursorScreenPos(v2(x0 - pad, p.y));
                ImGui::Dummy(v2(w, m.button + gap));
            }

            auto draw_item = [&](size_t idx) {
                ChatItem& it = items_[idx];
                ImGui::PushID((int)idx);
                switch (it.kind) {
                    case ChatItem::Kind::person: {
                        const float tw = w - 2.0f * m.pad;
                        const float th = it.text.empty() ? 0.0f : extent(L.ui, L.ui_size, it.text, tw).y;
                        const float ch = it.chips.empty() ? 0.0f : m.chip + (it.text.empty() ? 0.0f : m.gap);
                        place(m.pad * 2.0f + th + ch, [&](ImVec2 p) {
                            const float h = m.pad * 2.0f + th + ch;
                            cdl->AddRectFilled(p, v2(p.x + w, p.y + h), L.plate);
                            if (!it.text.empty()) draw_str(cdl, L.ui, L.ui_size, v2(p.x + m.pad, p.y + m.pad), L.text, it.text, tw);
                            float cx = p.x + m.pad;
                            const float cy = p.y + m.pad + th + (it.text.empty() ? 0.0f : m.gap);
                            for (size_t c = 0; c < it.chips.size(); ++c) {
                                const float cw = std::ceil(2.0f * m.button_pad + extent(L.ui, L.small_size, it.chips[c]).x);
                                if (cx + cw > p.x + w - m.pad) {
                                    draw_str(cdl, L.ui, L.small_size, v2(cx, cy + (m.chip - small_h) * 0.5f), L.faint,
                                         "+" + std::to_string(it.chips.size() - c));
                                    break;
                                }
                                cdl->AddRect(v2(cx, cy), v2(cx + cw, cy + m.chip), L.rule, 0.0f, 0, 1.0f);
                                draw_str(cdl, L.ui, L.small_size, v2(cx + m.button_pad, cy + (m.chip - small_h) * 0.5f), L.secondary,
                                     it.chips[c]);
                                cx += cw + m.gap;
                            }
                        });
                        break;
                    }
                    case ChatItem::Kind::agent: {
                        const float th = extent(L.ui, L.ui_size, it.text, w).y;
                        place(th, [&](ImVec2 p) { draw_str(cdl, L.ui, L.ui_size, p, L.text, it.text, w); });
                        break;
                    }
                    case ChatItem::Kind::thought: {
                        const float bx = sternbild::text_tracked_width(L.ui, L.small_size, "THOUGHT", tracking) + gap;
                        const float th = it.expanded ? extent(L.ui, L.small_size, trimmed(it.text), w).y : 0.0f;
                        place(line + th, [&](ImVec2 p) {
                            ImGui::SetCursorScreenPos(p);
                            if (ImGui::InvisibleButton("##thought", v2(w, line))) it.expanded = !it.expanded;
                            const bool hov = ImGui::IsItemHovered();
                            draw_tag(cdl, L, p, line, hov ? L.secondary : L.faint, "THOUGHT");
                            if (!it.expanded)
                                draw_str(cdl, L.ui, L.small_size, v2(p.x + bx, p.y + (line - small_h) * 0.5f), L.faint,
                                     fit(L.ui, L.small_size, first_line(it.text), w - bx));
                            else
                                draw_str(cdl, L.ui, L.small_size, v2(p.x, p.y + line), L.faint, trimmed(it.text), w);
                        });
                        break;
                    }
                    case ChatItem::Kind::tool: {
                        place(line, [&](ImVec2 p) {
                            const float half = std::round(3.5f * s);
                            status_mark(cdl, L, v2(p.x + half + 1.0f, p.y + line * 0.5f), it.tool.status);
                            float x = p.x + 2.0f * half + gap;
                            const std::string word = chat_tool_status_word(it.tool.status);
                            const float ww = extent(L.mono, L.mono_size, word).x;
                            const bool show_word = w > std::round(220.0f * s);
                            const float right = p.x + w - (show_word ? ww + gap : 0.0f);
                            if (!it.command.empty()) {
                                const std::string c = fit(L.mono, L.mono_size, it.command, right - x);
                                draw_str(cdl, L.mono, L.mono_size, v2(x, p.y + (line - mono_h) * 0.5f), L.text, c);
                                x += extent(L.mono, L.mono_size, c).x + gap;
                            }
                            // The title, unless it only names the tool the command already shows.
                            const bool redundant = !it.command.empty() &&
                                                   it.tool.title.find(caliper_builder::command_tool_name(it.command)) != std::string::npos &&
                                                   it.tool.title.size() <= caliper_builder::command_tool_name(it.command).size() + 32;
                            if (!redundant && !it.tool.title.empty() && right - x > 20.0f)
                                draw_str(cdl, L.ui, L.small_size, v2(x, p.y + (line - small_h) * 0.5f),
                                     it.command.empty() ? L.secondary : L.faint, fit(L.ui, L.small_size, it.tool.title, right - x));
                            if (show_word)
                                draw_str(cdl, L.mono, L.mono_size, v2(p.x + w - ww, p.y + (line - mono_h) * 0.5f),
                                     it.tool.status == "in_progress" ? L.secondary : L.faint, word);
                        });
                        break;
                    }
                    case ChatItem::Kind::plan: {
                        float h = line;
                        for (const auto& e : it.plan) h += std::max(line, extent(L.ui, L.small_size, e.content, w - line).y);
                        place(h, [&](ImVec2 p) {
                            draw_tag(cdl, L, p, line, L.faint, "PLAN");
                            float y = p.y + line;
                            for (const auto& e : it.plan) {
                                const float eh = std::max(line, extent(L.ui, L.small_size, e.content, w - line).y);
                                status_mark(cdl, L, v2(p.x + std::round(4.5f * s), y + line * 0.5f), e.status);
                                const ImU32 c = e.status == "completed" ? L.faint : e.status == "in_progress" ? L.text : L.secondary;
                                draw_str(cdl, L.ui, L.small_size, v2(p.x + line, y + (line - small_h) * 0.5f), c, e.content, w - line);
                                y += eh;
                            }
                        });
                        break;
                    }
                    case ChatItem::Kind::permission: {
                        // The tag and the command or the tool's title; while open, the
                        // file, the compact diff as the inline surface and the approval
                        // cards draw it (diff_card.h) or the unified diff on demand and
                        // for a file that is not a spec, and the agent's options.
                        const bool open = it.answer.empty();
                        const std::string label = !it.command.empty() ? it.command : it.tool.title;
                        const bool compact = !it.compact.empty() && !it.expanded;
                        std::string full;
                        int full_lines = 0;
                        if (open && !compact && !it.diff.empty())
                            for (const std::string& l : split_lines(it.diff)) {
                                if (l.rfind("---", 0) == 0 || l.rfind("+++", 0) == 0) continue;
                                if (full_lines < kMaxDiffLines) full += l + "\n";
                                ++full_lines;
                            }
                        const int total = compact ? (int)it.compact.size() : full_lines;
                        const int cap = compact ? kMaxCompactLines : kMaxDiffLines;
                        const int shown = std::min(total, cap);
                        const bool folded = total > shown;
                        const float dline = std::round(18.0f * s);
                        float h = line;
                        if (open) {
                            if (!it.diff_path.empty()) h += dline;
                            h += dline * (float)(shown + (folded ? 1 : 0));
                            h += m.gap + m.button + m.pad;
                        }
                        place(h + (open ? m.pad : 0.0f), [&](ImVec2 p) {
                            const ImVec2 q = v2(p.x + w, p.y + h + (open ? m.pad : 0.0f));
                            if (open) {
                                cdl->AddRectFilled(p, q, L.deep);
                                cdl->AddRect(p, q, L.accent, 0.0f, 0, 1.0f);
                            }
                            const float ix = p.x + (open ? m.pad : 0.0f);
                            const float iw = w - (open ? 2.0f * m.pad : 0.0f);
                            float y = p.y + (open ? m.pad * 0.5f : 0.0f);
                            const float tw = draw_tag(cdl, L, v2(ix, y), line, open ? L.secondary : L.faint, "PERMISSION");
                            const float x = ix + tw + gap;
                            if (!it.command.empty())
                                draw_str(cdl, L.mono, L.mono_size, v2(x, y + (line - mono_h) * 0.5f), open ? L.text : L.secondary,
                                         fit(L.mono, L.mono_size, label, ix + iw - x));
                            else
                                draw_str(cdl, L.ui, L.ui_size, v2(x, y + (line - ui_h) * 0.5f), open ? L.text : L.secondary,
                                         fit(L.ui, L.ui_size, label, ix + iw - x));
                            if (!open) {
                                const float ax = p.x + w - extent(L.ui, L.small_size, it.answer).x;
                                draw_str(cdl, L.ui, L.small_size, v2(ax, y + (line - small_h) * 0.5f), L.faint, it.answer);
                                return;
                            }
                            y += line;
                            if (!it.diff_path.empty()) {
                                draw_str(cdl, L.mono, L.mono_size, v2(ix, y + (dline - mono_h) * 0.5f), L.faint,
                                         fit(L.mono, L.mono_size, it.diff_path, iw));
                                y += dline;
                            }
                            const float dh = dline * (float)(shown + (folded ? 1 : 0));
                            if (compact) {
                                draw_compact_diff(cdl, L, ImRect(v2(ix, y), v2(ix + iw, y + dh)), it.compact, shown, folded);
                            } else if (shown > 0) {
                                cdl->PushClipRect(v2(ix, y), v2(ix + iw, y + dh), true);
                                ImGui::SetCursorScreenPos(v2(ix, y));
                                draw_unified_diff(L, full, dline);   // diff_card.h
                                cdl->PopClipRect();
                                if (folded)
                                    draw_str(cdl, L.mono, L.mono_size, v2(ix, y + dline * (float)shown + (dline - mono_h) * 0.5f), L.faint,
                                             "+" + std::to_string(total - shown));
                            }
                            y += dh + m.gap;
                            float bx = ix;
                            bool first_allow = true;
                            for (size_t o = 0; o < it.options.size(); ++o) {
                                const auto& opt = it.options[o];
                                const std::string name = opt.name.empty() ? opt.id : opt.name;
                                const float bw = std::ceil(2.0f * m.button_pad + extent(L.ui, L.ui_size, name).x + std::round(8.0f * s));
                                if (bx + bw > ix + iw) break;
                                const bool primary = first_allow && opt.kind.rfind("allow", 0) == 0;
                                if (primary) first_allow = false;
                                const std::string bid = "##perm." + std::to_string(o);
                                if (house_button(bid.c_str(), ImRect(v2(bx, y), v2(bx + bw, y + m.button)), name, "", primary,
                                                 false, L))
                                    answer_permission(it.permission, opt.id);
                                bx += bw + m.gap;
                            }
                            // Diff: the full unified diff in place of the compact one.
                            if (!it.compact.empty()) {
                                const float dw = std::ceil(2.0f * m.button_pad + extent(L.ui, L.ui_size, "Diff").x + std::round(8.0f * s));
                                const ImRect db(v2(ix + iw - dw, y), v2(ix + iw, y + m.button));
                                if (db.Min.x >= bx) {
                                    if (house_button("##perm.diff", db, "Diff", "", false, false, L)) it.expanded = !it.expanded;
                                    if (it.expanded) cdl->AddRect(db.Min, db.Max, L.accent, 0.0f, 0, 1.0f);
                                }
                            }
                        });
                        break;
                    }
                    case ChatItem::Kind::approval: {
                        // One row that points at the approval card in the strip.
                        place(line, [&](ImVec2 p) {
                            const bool waiting = it.answer.empty();
                            ImGui::SetCursorScreenPos(p);
                            if (ImGui::InvisibleButton("##approval", v2(w, line)) && hooks_.show_approval) hooks_.show_approval(it.approval);
                            const bool hov = ImGui::IsItemHovered();
                            if (hov && waiting) cdl->AddRectFilled(p, v2(p.x + w, p.y + line), L.plate_hover);
                            const float tw = draw_tag(cdl, L, p, line, waiting ? L.accent : L.faint, "APPROVAL");
                            const float x = p.x + tw + gap;
                            const std::string word = waiting ? "waiting" : it.answer;
                            const float ww = extent(L.mono, L.mono_size, word).x;
                            draw_str(cdl, L.mono, L.mono_size, v2(x, p.y + (line - mono_h) * 0.5f), waiting ? L.text : L.secondary,
                                     fit(L.mono, L.mono_size, it.command, p.x + w - ww - gap - x));
                            draw_str(cdl, L.mono, L.mono_size, v2(p.x + w - ww, p.y + (line - mono_h) * 0.5f),
                                     waiting ? L.secondary : L.faint, word);
                        });
                        break;
                    }
                    case ChatItem::Kind::stop: {
                        place(line, [&](ImVec2 p) {
                            float x = p.x + draw_tag(cdl, L, p, line, it.text == "FAILED" ? L.accent : L.faint, it.text) + gap;
                            if (!it.detail.empty()) {
                                const std::string d = fit(L.mono, L.mono_size, it.detail, p.x + w - x);
                                draw_str(cdl, L.mono, L.mono_size, v2(x, p.y + (line - mono_h) * 0.5f), L.faint, d);
                                x += extent(L.mono, L.mono_size, d).x + gap;
                            }
                            if (x < p.x + w) cdl->AddRectFilled(v2(x, std::round(p.y + line * 0.5f)), v2(p.x + w, std::round(p.y + line * 0.5f) + 1.0f), L.rule_faint);
                        });
                        break;
                    }
                    case ChatItem::Kind::notice: {
                        // The tag, the error or the exit status, and the lines: the
                        // agent's stderr tail or its login methods. Restart is in the header.
                        const bool login = it.text == "LOGIN";
                        const float lh0 = std::round(18.0f * s);
                        auto mono_line = [&](const std::string& l) { return it.text == "EXITED" || (login && l.rfind("$ ", 0) == 0); };
                        float h = line;
                        auto line_h = [&](const std::string& l) {
                            if (it.text == "EXITED") return lh0;
                            return std::max(lh0, mono_line(l) ? extent(L.mono, L.mono_size, l, w).y : extent(L.ui, L.small_size, l, w).y);
                        };
                        for (const std::string& l : it.lines) h += line_h(l);
                        place(h, [&](ImVec2 p) {
                            const float x = p.x + draw_tag(cdl, L, p, line, L.accent, it.text) + gap;
                            if (!it.detail.empty())
                                draw_str(cdl, L.mono, L.mono_size, v2(x, p.y + (line - mono_h) * 0.5f), L.faint,
                                         fit(L.mono, L.mono_size, it.detail, p.x + w - x));
                            float y = p.y + line;
                            for (const std::string& l : it.lines) {
                                const bool mono = mono_line(l);
                                const float lh = line_h(l);
                                if (mono && !login)   // the agent's last stderr line, elided to one line
                                    draw_str(cdl, L.mono, L.mono_size, v2(p.x, y + (lh0 - mono_h) * 0.5f), L.faint, fit(L.mono, L.mono_size, l, w));
                                else if (mono) draw_str(cdl, L.mono, L.mono_size, v2(p.x, y + (lh0 - mono_h) * 0.5f), L.text, l, w);
                                else draw_str(cdl, L.ui, L.small_size, v2(p.x, y + (lh0 - small_h) * 0.5f), L.secondary, l, w);
                                y += lh;
                            }
                        });
                        break;
                    }
                }
                ImGui::PopID();
            };
            // The rows: the items, with runs of one command folded into a group.
            for (const ChatRow& row : chat_rows(items_, group_open_)) {
                if (row.kind != ChatRow::Kind::group) {
                    draw_item(row.item);
                    continue;
                }
                ImGui::PushID(row.key.c_str());
                place(line, [&](ImVec2 p) {
                    ImGui::SetCursorScreenPos(p);
                    if (ImGui::InvisibleButton("##group", v2(w, line))) group_open_[row.key] = !row.expanded;
                    if (ImGui::IsItemHovered()) cdl->AddRectFilled(p, v2(p.x + w, p.y + line), L.plate_hover);
                    const float half = std::round(3.5f * s);
                    status_mark(cdl, L, v2(p.x + half + 1.0f, p.y + line * 0.5f), row.mark);
                    float x = p.x + 2.0f * half + gap;
                    // The fold's mark: right when closed, down when open.
                    const float t = std::round(3.0f * s);
                    const ImVec2 c = v2(std::round(x + t), std::round(p.y + line * 0.5f));
                    if (row.expanded) cdl->AddTriangleFilled(v2(c.x - t, c.y - t * 0.6f), v2(c.x + t, c.y - t * 0.6f), v2(c.x, c.y + t * 0.7f), L.faint);
                    else cdl->AddTriangleFilled(v2(c.x - t * 0.6f, c.y - t), v2(c.x - t * 0.6f, c.y + t), v2(c.x + t * 0.7f, c.y), L.faint);
                    x += 2.0f * t + gap;
                    const std::string summary = chat_group_summary(row);
                    const float sw = extent(L.mono, L.mono_size, summary).x;
                    const float right = p.x + w - sw - gap;
                    ImFont* f = row.caliper ? L.mono : L.ui;
                    const float fs = row.caliper ? L.mono_size : L.small_size;
                    const float fh = row.caliper ? mono_h : small_h;
                    const std::string name = fit(f, fs, row.command, right - x);
                    draw_str(cdl, f, fs, v2(x, p.y + (line - fh) * 0.5f), row.caliper ? L.text : L.secondary, name);
                    x += extent(f, fs, name).x;
                    if (row.count > 1 && x < right) {
                        const std::string times = " \xC3\x97" + std::to_string(row.count);   // ×
                        draw_str(cdl, L.mono, L.mono_size, v2(x, p.y + (line - mono_h) * 0.5f), L.faint, fit(L.mono, L.mono_size, times, right - x));
                    }
                    draw_str(cdl, L.mono, L.mono_size, v2(p.x + w - sw, p.y + (line - mono_h) * 0.5f),
                             row.alert ? L.accent : row.mark == "in_progress" ? L.secondary : L.faint, summary);
                });
                if (row.expanded) {
                    x0 = x_left + std::round(16.0f * s);
                    w = w_full - std::round(16.0f * s);
                    for (size_t mi : row.members) draw_item(mi);
                    x0 = x_left;
                    w = w_full;
                }
                ImGui::PopID();
            }
            if (follow_ || items_.size() != drawn_items_) {
                if (at_bottom || follow_) ImGui::SetScrollHereY(1.0f);
            }
            drawn_items_ = items_.size();
            follow_ = at_bottom;
        }
        ImGui::EndChild();
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(2);
    }

    // ---- the draft's chips
    if (!lay.chips.empty()) {
        for (int i = 0; i < lay.chips_shown; ++i) {
            const ImRect c = rr(lay.chip_rects[(size_t)i]);
            dl->AddRect(c.Min, c.Max, L.rule, 0.0f, 0, 1.0f);
            draw_str(dl, L.ui, L.ui_size, v2(c.Min.x + m.button_pad, c.Min.y + (c.GetHeight() - ui_h) * 0.5f), L.secondary,
                 fit(L.ui, L.ui_size, in.chips[(size_t)i], c.GetWidth() - 2.0f * m.button_pad));
        }
        if (lay.chips_shown < (int)in.chips.size()) {
            const float x = lay.chip_rects.empty() ? lay.chips.x : lay.chip_rects.back().right() + m.gap;
            draw_str(dl, L.ui, L.ui_size, v2(x, lay.chips.y + (lay.chips.h - ui_h) * 0.5f), L.faint,
                 "+" + std::to_string(in.chips.size() - (size_t)lay.chips_shown));
        }
        if (house_button("##chat.clear", rr(lay.clear), chat_action_label(ChatAction::clear), "", false, false, L))
            clear_draft();
    }

    // ---- the input row
    {
        const ImRect row = rr(lay.input_row);
        const bool usable = state_ != ChatState::no_agent;
        dl->AddRectFilled(row.Min, row.Max, L.deep);
        const ImGuiID input_id = ImGui::GetID("##chat.input");
        const bool typing = ImGui::GetActiveID() == input_id;
        dl->AddRect(row.Min, row.Max, typing ? L.accent : L.rule, 0.0f, 0, 1.0f);

        // The picker: the agent's name and its ▾ mark.
        const ImRect pk = rr(lay.picker);
        ImGui::SetCursorScreenPos(pk.Min);
        if (ImGui::InvisibleButton("##chat.picker", pk.GetSize())) picker_open_ = true;
        if (ImGui::IsItemHovered()) dl->AddRectFilled(pk.Min, pk.Max, L.plate_hover);
        dl->AddRect(pk.Min, pk.Max, L.rule, 0.0f, 0, 1.0f);
        float px = pk.Min.x + m.button_pad;
        if (lay.picker_label) {
            draw_str(dl, L.ui, L.ui_size, v2(px, pk.Min.y + (pk.GetHeight() - ui_h) * 0.5f), usable ? L.text : L.faint, in.picker);
            px += extent(L.ui, L.ui_size, in.picker).x + std::round(4.0f * s);
        }
        {
            const float t = std::round(3.5f * s);
            const ImVec2 c = v2(std::round(px + t + 1.0f), std::round(pk.GetCenter().y));
            dl->AddTriangleFilled(v2(c.x - t, c.y - t * 0.5f), v2(c.x + t, c.y - t * 0.5f), v2(c.x, c.y + t * 0.6f), L.secondary);
        }

        // The input: one line, more with ⇧⏎ or ⌘⏎ (up to the layout's maximum).
        const ImRect inp = rr(lay.input);
        if (!usable) ImGui::BeginDisabled();
        ImGui::SetCursorScreenPos(v2(inp.Min.x, inp.Min.y + std::round((m.row - ui_h) * 0.5f) - 2.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, 0);
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, 0);
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, 0);
        ImGui::PushStyleColor(ImGuiCol_Text, L.text);
        ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, sternbild::with_alpha(L.accent, 90));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, v2(0, 2));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
        ImGui::PushFont(L.ui, L.ui_size);
        if (focus_input_ && usable) {
            ImGui::SetKeyboardFocusHere();
            focus_input_ = false;
        }
        if (text_.capacity() < 64) text_.reserve(64);
        const float ih = std::max(ui_h + 4.0f, inp.GetHeight() - std::round((m.row - ui_h) * 0.5f) * 2.0f + 4.0f);
        const bool entered = ImGui::InputTextMultiline(
            "##chat.input", const_cast<char*>(text_.c_str()), text_.capacity() + 1, v2(inp.GetWidth(), ih),
            ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CtrlEnterForNewLine |
                ImGuiInputTextFlags_CallbackResize | ImGuiInputTextFlags_NoHorizontalScroll,
            resize_callback, &text_);
        ImGui::PopFont();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(5);
        if (!usable) ImGui::EndDisabled();
        if (entered && usable) {
            send();
            focus_input_ = true;
        }

        const ChatAction a = in.input_action;
        bool enabled = usable;
        if (a == ChatAction::send) enabled = usable && (!trimmed(text_).empty() || !draft_.blocks.empty()) &&
                                             state_ != ChatState::starting;
        if (house_button("##chat.send", rr(lay.send), chat_action_label(a), chat_action_key(a), a == ChatAction::send,
                         lay.keys, L, enabled))
            act(a);
    }
    dl->PopClipRect();

    // ---- the picker's list: the agents found, the others in the faint tone, and the custom command
    if (picker_open_) {
        ImGui::OpenPopup("##chat.agents");
        picker_open_ = false;
    }
    const ImRect pk = rr(lay.picker);
    ImGui::SetNextWindowPos(v2(pk.Min.x, pk.Min.y - 2.0f), ImGuiCond_Always, v2(0.0f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, v2(m.pad, m.pad));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, L.deep);
    ImGui::PushStyleColor(ImGuiCol_Border, L.rule);
    if (ImGui::BeginPopup("##chat.agents")) {
        ImGui::PushFont(L.ui, L.ui_size);
        const float cw = std::round(260.0f * s);
        for (const ChatAgent& ag : agents_) {
            ImGui::PushID(ag.id.c_str());
            const bool chosen = ag.found && ag.id == options_.settings.agent;
            const ImVec2 p = ImGui::GetCursorScreenPos();
            if (!ag.found) ImGui::BeginDisabled();
            if (ImGui::Selectable("##agent", chosen, 0, v2(cw, std::round(22.0f * s)))) choose_agent(ag.id);
            if (!ag.found) ImGui::EndDisabled();
            ImDrawList* pdl = ImGui::GetWindowDrawList();
            const float lh = std::round(22.0f * s);
            draw_str(pdl, L.ui, L.ui_size, v2(p.x + std::round(4.0f * s), p.y + (lh - ui_h) * 0.5f),
                 ag.found ? (chosen ? L.accent : L.text) : L.faint, ag.title);
            if (!ag.found) {   // the name and whether it was found; never its command line
                const float nw = sternbild::text_tracked_width(L.ui, L.small_size, "NOT FOUND", tracking);
                draw_tag(pdl, L, v2(p.x + cw - nw - std::round(4.0f * s), p.y), lh, L.faint, "NOT FOUND");
            }
            ImGui::PopID();
        }
        ImGui::Dummy(v2(1.0f, std::round(4.0f * s)));
        ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetCursorScreenPos(), v2(ImGui::GetCursorScreenPos().x + cw, ImGui::GetCursorScreenPos().y + 1.0f), L.rule_faint);
        ImGui::Dummy(v2(1.0f, std::round(4.0f * s)));
        if (ImGui::Selectable(chat_action_label(ChatAction::custom), false, 0, v2(cw, std::round(22.0f * s)))) custom_open_ = true;
        // Local model: the chosen agent's mark, a box filled in bronze while it
        // is set; the popup stays open so that the mark shows.
        {
            const bool can = agent() != nullptr, on = local_model();
            const float lh = std::round(22.0f * s);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            if (!can) ImGui::BeginDisabled();
            if (ImGui::Selectable("##chat.local_model", false, ImGuiSelectableFlags_NoAutoClosePopups, v2(cw, lh)))
                set_local_model(!on);
            if (!can) ImGui::EndDisabled();
            ImDrawList* pdl = ImGui::GetWindowDrawList();
            const float side = std::round(11.0f * s);
            const ImVec2 b(p.x + std::round(4.0f * s), std::round(p.y + (lh - side) * 0.5f));
            pdl->AddRect(b, v2(b.x + side, b.y + side), can ? L.secondary : L.faint, 0.0f, 0, 1.0f);
            if (on) {
                const float in = std::round(3.0f * s);
                pdl->AddRectFilled(v2(b.x + in, b.y + in), v2(b.x + side - in, b.y + side - in), L.accent);
            }
            draw_str(pdl, L.ui, L.ui_size, v2(b.x + side + std::round(8.0f * s), p.y + (lh - ui_h) * 0.5f),
                     !can ? L.faint : on ? L.accent : L.text, "Local model");
        }
        ImGui::PopFont();
        ImGui::EndPopup();
    }

    // ---- the custom command: a name and one command line
    if (custom_open_) {
        ImGui::OpenPopup("##chat.custom");
        if (custom_line_.empty()) custom_line_ = options_.settings.custom_command;
        if (custom_name_.empty()) custom_name_ = options_.settings.custom_name;
        custom_open_ = false;
    }
    ImGui::SetNextWindowPos(v2(pane_rect.Min.x + m.pad, lay.input_row.y - m.gap), ImGuiCond_Always, v2(0.0f, 1.0f));
    if (ImGui::BeginPopup("##chat.custom")) {
        ImGui::PushFont(L.ui, L.ui_size);
        const float fw = std::max(std::round(200.0f * s), P.w - 4.0f * m.pad);
        sternbild::label("Name");
        ImGui::PushItemWidth(fw);
        if (custom_name_.capacity() < 32) custom_name_.reserve(32);
        ImGui::InputText("##custom.name", const_cast<char*>(custom_name_.c_str()), custom_name_.capacity() + 1,
                         ImGuiInputTextFlags_CallbackResize, resize_callback, &custom_name_);
        sternbild::label("Command");
        if (custom_line_.capacity() < 64) custom_line_.reserve(64);
        const bool done = ImGui::InputText("##custom.line", const_cast<char*>(custom_line_.c_str()), custom_line_.capacity() + 1,
                                           ImGuiInputTextFlags_CallbackResize | ImGuiInputTextFlags_EnterReturnsTrue,
                                           resize_callback, &custom_line_);
        ImGui::PopItemWidth();
        if (!custom_error_.ok()) ImGui::TextColored(sternbild::to_vec4(L.faint), "%s", custom_error_.code.c_str());
        if (sternbild::primary_button("Save") || done) {
            custom_error_ = set_custom_agent(custom_name_, custom_line_);
            if (custom_error_.ok()) ImGui::CloseCurrentPopup();
        }
        ImGui::PopFont();
        ImGui::EndPopup();
    }

    // ---- the diagnostics: the protocol faults, out of the conversation
    if (diagnostics_open_ && !faults_.empty()) {
        ImGui::SetNextWindowPos(v2(lay.faults.right(), lay.header.bottom() + 2.0f), ImGuiCond_Always, v2(1.0f, 0.0f));
        ImGui::SetNextWindowSize(v2(std::min(P.w - 2.0f * m.pad, std::round(420.0f * s)), 0.0f));
        if (ImGui::Begin("##chat.diagnostics", &diagnostics_open_,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::PushFont(L.mono, L.mono_size);
            const size_t from = faults_.size() > 12 ? faults_.size() - 12 : 0;
            for (size_t i = from; i < faults_.size(); ++i) {
                ImGui::TextColored(sternbild::to_vec4(L.secondary), "%s", faults_[i].code.c_str());
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(sternbild::to_vec4(L.faint), "%s", faults_[i].message.c_str());
                ImGui::PopTextWrapPos();
            }
            ImGui::PopFont();
        }
        ImGui::End();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();
    ImGui::PopID();
}

}  // namespace caliper_host
