// side_chat.cpp: the side chat (side_chat.h): the handoff draft, the state
// machine over AcpClient, worker-thread lifecycle, event handling, approvals,
// permissions, stalls, and conversation row grouping.
#include "builder/side_chat.h"
#include "builder/side_chat_internal.h"
#include "builder/widget_surface.h"

#include <caliper_builder/acp/acp_client.h>
#include <caliper_builder/commands/approval.h>
#include <caliper_builder/commands/command.h>
#include <caliper_builder/error.h>
#include <caliper_builder/session/context_bundle.h>
#include <caliper_builder/session/line_diff.h>
#include <caliper_builder/spec/spec.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace caliper_host {

namespace fs = std::filesystem;
namespace acp = caliper_builder::acp;
using caliper_builder::Error;
using nlohmann::json;
using side_chat_detail::split_lines;
using side_chat_detail::trimmed;

namespace {

constexpr size_t kMaxItemText = 256u << 10;   // one message, thought or prompt text kept
constexpr size_t kMaxItems = 4000;            // the conversation's items; the oldest go first
constexpr size_t kMaxFaults = 200;
constexpr int kStderrTail = 1;                // stderr lines shown when the agent exited, one elided

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

void merge_tool(acp::ToolCallInfo* into, const acp::ToolCallInfo& from) {
    if (from.has_title) into->title = from.title;
    if (from.has_kind) into->kind = from.kind;
    if (from.has_status) into->status = from.status;
    if (from.has_content) into->content = from.content;
    if (from.has_locations) into->locations = from.locations;
    if (!from.raw_input.is_null()) into->raw_input = from.raw_input;
    if (!from.raw_output.is_null()) into->raw_output = from.raw_output;
    if (!from.caliper_command.empty()) into->caliper_command = from.caliper_command;
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

std::string chat_permission_label(const ChatItem& item) {
    if (!item.tool.title.empty()) return item.tool.title;
    if (!item.diff_path.empty()) return item.diff_path;
    return "Tool call";
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
    if (items_.size() >= kMaxItems) {
        items_.erase(items_.begin(), items_.begin() + (long)(kMaxItems / 10));
        // The drawn text of the items that went goes with them.
        markdown_.erase(markdown_.begin(), markdown_.lower_bound(items_.front().serial));
    }
    items_.emplace_back();
    items_.back().serial = ++next_serial_;
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

void SideChat::applied(std::uint64_t seq, const std::string& command, const std::string& allowed_by) {
    ChatItem& a = add(ChatItem::Kind::applied);
    a.seq = seq;
    a.command = command;
    // What allowed it, as the header names modes and the cards name rules.
    a.text = allowed_by == "accept_edits"   ? "Accept edits"
             : allowed_by == "allow_all"    ? "Allow all"
             : allowed_by == "session_rule" ? "Rule for this session"
             : allowed_by == "saved_rule"   ? "Saved rule"
                                            : allowed_by;
}

bool SideChat::undo_applied(std::uint64_t serial) {
    for (ChatItem& it : items_) {
        if (it.serial != serial || it.kind != ChatItem::Kind::applied || !it.answer.empty()) continue;
        if (!hooks_.can_undo || !hooks_.undo || !hooks_.can_undo(it.seq)) return false;
        hooks_.undo(it.seq);
        it.answer = "undone";
        return true;
    }
    return false;
}

std::string SideChat::permission_word() const { return hooks_.permission_mode ? hooks_.permission_mode() : std::string(); }

std::string SideChat::permission_note() const { return hooks_.permission_note ? hooks_.permission_note() : std::string(); }

void SideChat::cycle_permission_mode() {
    if (!hooks_.set_permission_mode) return;
    const std::string now = permission_word();
    hooks_.set_permission_mode(now == "Ask" ? "accept_edits" : now == "Accept edits" ? "read_only" : "ask");
}

void SideChat::set_allow_all(bool on) {
    if (!hooks_.set_permission_mode) return;
    const std::string word = permission_word();
    if (on) {
        hooks_.set_permission_mode("allow_all");
    } else if (word == "Allow all") {
        hooks_.set_permission_mode("ask");
    } else if (!permission_note().empty()) {
        // After a failed save: save the mode the person chose again, never a looser one.
        hooks_.set_permission_mode(word == "Read only" ? "read_only" : word == "Accept edits" ? "accept_edits" : "ask");
    }
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
        std::vector<size_t> waiting;   // waiting approvals and APPLIED rows, shown after the group
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
            } else if (m.kind == ChatItem::Kind::applied && g.caliper && m.command == key) {
                waiting.push_back(j);   // its own row, with Undo, after the group
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
            r.kind = items[w].kind == ChatItem::Kind::applied ? ChatRow::Kind::item : ChatRow::Kind::waiting_approval;
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

}  // namespace caliper_host
