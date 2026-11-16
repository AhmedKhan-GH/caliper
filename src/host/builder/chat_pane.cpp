// chat_pane.cpp: the side chat in the app (chat_pane.h).
#include "builder/chat_pane.h"

#include "builder/app_session.h"

#if CALIPER_WITH_BUILDER
#include "spec_runtime/spec_runtime.h"
#include "ui/sternbild_theme.h"
#include "ui/work_area.h"

#include <caliper_builder/session/session.h>

#include <nlohmann/json.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#endif

namespace caliper_host {

#if CALIPER_WITH_BUILDER

namespace {

namespace fs = std::filesystem;
using nlohmann::json;

#ifdef __APPLE__
const char* const kToggleKey = "\xe2\x8c\x98L";   // ⌘L
#else
const char* const kToggleKey = "Ctrl L";
#endif

}  // namespace

bool read_chat_settings(const std::string& path, ChatPaneSettings* out, std::string* why) {
    *out = ChatPaneSettings{};
    std::error_code ec;
    if (path.empty() || !fs::is_regular_file(path, ec)) return true;
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    const json j = json::parse(ss.str(), nullptr, false);
    if (!j.is_object()) {
        if (why) *why = "the file is not a JSON object";
        return false;
    }
    auto str = [&j](const char* k, const std::string& d) {
        const auto it = j.find(k);
        return it != j.end() && it->is_string() ? it->get<std::string>() : d;
    };
    auto flag = [&j](const char* k) {
        const auto it = j.find(k);
        return it != j.end() && it->is_boolean() && it->get<bool>();
    };
    out->chat.agent = str("agent", "");
    out->chat.custom_name = str("custom_name", out->chat.custom_name);
    out->chat.custom_command = str("custom_command", "");
    out->chat.allow_always = flag("allow_always");
    // The agents marked as running on a local model: their ids, once each.
    if (const auto it = j.find("local_model_agents"); it != j.end() && it->is_array()) {
        std::vector<std::string>& marked = out->chat.local_model_agents;
        for (const json& v : *it) {
            if (!v.is_string() || marked.size() >= 64) continue;
            const std::string& id = v.get_ref<const std::string&>();
            if (!id.empty() && id.size() <= 64 && std::find(marked.begin(), marked.end(), id) == marked.end())
                marked.push_back(id);
        }
    }
    out->open = flag("open");
    return true;
}

bool write_chat_settings(const std::string& path, const ChatPaneSettings& s, std::string* why) {
    if (path.empty()) return true;
    const json j = {{"agent", s.chat.agent},
                    {"custom_name", s.chat.custom_name},
                    {"custom_command", s.chat.custom_command},
                    {"allow_always", s.chat.allow_always},
                    {"local_model_agents", s.chat.local_model_agents},
                    {"open", s.open}};
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << j.dump(2) << "\n";
        if (!out) {
            if (why) *why = "cannot write " + tmp;
            return false;
        }
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        if (why) *why = "cannot replace " + path + ": " + ec.message();
        return false;
    }
    return true;
}

struct ChatPane::Impl {
    Options options;
    Hooks hooks;
    ChatPaneSettings settings;
    bool started = false;
    std::unique_ptr<SideChat> chat;
    std::vector<caliper_builder::PendingApproval> pending;   // as read while the registry was free
    std::optional<ChatDraft> draft;   // a handoff that came before the chat was made
    caliper_builder::Session* session = nullptr;   // the app's, as of the last frame
    bool busy = false;        // a long command holds the session's registry this frame
    std::string agent_file;   // the API file of the agent's session, with its own agent token

    // The agent's session ends with its token: Allow all and the rules of
    // this session go with it (Session::end_chat_session()).
    void end_agent_file() {
        if (!agent_file.empty() && hooks.end_api_file) hooks.end_api_file(agent_file);
        if (!agent_file.empty() && session) session->end_chat_session();
        agent_file.clear();
    }

    void log(const std::string& event, const std::string& object, const std::string& detail) {
        if (hooks.log) hooks.log(event, object, detail);
    }

    void save() {
        std::string why;
        if (!write_chat_settings(options.settings_path, settings, &why) && hooks.log)
            hooks.log("chat settings not saved", options.settings_path, why);
    }

    void make(AppSession& app) {
        SideChat::Options o;
        o.env.workspace = app.workspace();
        o.env.mcp_command = find_caliper_mcp(options.exe_dir);
        o.env.path = options.path;
        o.settings = settings.chat;
        SideChat::Hooks h;
        h.pending_approvals = [this] { return pending; };
        h.show_approval = hooks.show_approval;
        h.approval_outcome = [this](const std::string& id) -> std::string {
            if (!session) return {};
            const std::optional<caliper_builder::CommandResult> r = session->queue().approval_outcome(id);
            if (!r) return {};
            if (r->ok) return "approved";
            if (r->error.code == "command.denied") return "denied";
            if (r->error.code == "command.approval_timeout") return "timed out";
            if (r->error.code == "command.needs_approval") return "reissued";
            return "failed";
        };
        h.reissued_as = [this](const std::string& id) -> std::string {
            if (!session) return {};
            const std::optional<caliper_builder::CommandResult> r = session->queue().approval_outcome(id);
            if (r && r->error.code == "command.needs_approval" && r->approval != id) return r->approval;
            return {};
        };
        h.settings_changed = [this](const ChatSettings& s) {
            settings.chat = s;
            save();
        };
        // Caliper's mode for the agent, which the command layer applies to the
        // commands of this chat's token (permissions.h); only the person's
        // own UI changes it, here and in the settings.
        h.permission_mode = [this]() -> std::string {
            if (!session) return {};
            return caliper_builder::agent_mode_label(session->agent_mode(caliper_builder::AgentPrincipal::chat));
        };
        h.set_permission_mode = [this](const std::string& name) {
            namespace cb = caliper_builder;
            // "" ends Allow all: the saved mode beneath it holds again.
            const auto mode = name.empty() && session ? std::optional<cb::AgentMode>(session->saved_agent_mode(cb::AgentPrincipal::chat))
                                                      : cb::agent_mode_from_name(name);
            if (!session || !mode) return;
            const caliper_builder::Error e = session->set_agent_mode(caliper_builder::AgentPrincipal::chat, *mode);
            log(e ? "mode not set" : "side chat mode", caliper_builder::agent_mode_label(*mode), e ? e.code : "");
        };
        h.can_undo = [this](std::uint64_t seq) {
            if (!session || busy) return false;
            const auto state = session->registry().history_state(1);
            if (state.undo.empty()) return false;
            const auto& seqs = state.undo.front().seqs;
            return std::find(seqs.begin(), seqs.end(), seq) != seqs.end();
        };
        h.undo = [this](std::uint64_t) {
            if (!session) return;
            caliper_builder::CommandInvocation inv;
            inv.name = "history.undo";
            inv.origin = caliper_builder::CommandOrigin::ui;   // the person's own Undo
            session->queue().submit(std::move(inv));
        };
        h.before_start = [this](ChatEnvironment* env) -> caliper_builder::Error {
            if (!hooks.start_api) return {"chat.no_api", "Caliper API is not running."};
            end_agent_file();   // a restart's agent gets a new token; the old one ends
            // The saved rules that apply are this agent's.
            if (session && chat) session->set_chat_agent(chat->settings().agent);
            std::string why, started;
            const std::string file = hooks.start_api(&why, &started);
            if (file.empty()) return {"chat.no_api", "Caliper API did not start: " + why};
            env->api_file = file;
            agent_file = file;
            if (!started.empty() && chat) chat->note("API", "Caliper API started on " + started + " for this chat");
            return {};
        };
        h.agent_ended = [this] {
            end_agent_file();   // the session's token is revoked with it
            if (hooks.agent_ended) hooks.agent_ended();
        };
        h.live_attached = hooks.live_attached;
        chat = std::make_unique<SideChat>(std::move(o), std::move(h));
        if (draft) {
            chat->open_handoff(std::move(*draft));
            draft.reset();
        }
    }
};

ChatPane::ChatPane() : impl_(std::make_unique<Impl>()) {}
ChatPane::~ChatPane() = default;

bool ChatPane::built() { return true; }

void ChatPane::start(Options options, Hooks hooks) {
    Impl& im = *impl_;
    im.options = std::move(options);
    im.hooks = std::move(hooks);
    std::string why;
    if (!read_chat_settings(im.options.settings_path, &im.settings, &why) && im.hooks.log)
        im.hooks.log("chat settings not read", im.options.settings_path, why);
    im.started = true;
}

void ChatPane::frame(AppSession& app) {
    Impl& im = *impl_;
    if (!im.started) return;
    if (im.settings.open && !im.chat) {
        app.ensure_open();   // the workspace the agent works in
        im.make(app);
    }
    im.session = app.session();
    im.busy = app.worker_busy();
    if (!im.chat) return;
    // The pending approvals, read only while no long command holds the registry.
    if (caliper_builder::Session* s = app.session(); s && !app.worker_busy()) im.pending = s->registry().pending_approvals();
    // The agent's commands that a mode or a rule let run without a card.
    for (const AppSession::Applied& a : app.take_applied())
        if (a.origin == "agent" && !im.agent_file.empty()) im.chat->applied(a.seq, a.name, a.allowed_by);
    im.chat->update();
}

bool ChatPane::open() const { return impl_->settings.open; }

void ChatPane::set_open(bool open) {
    Impl& im = *impl_;
    if (im.settings.open == open) return;
    im.settings.open = open;
    im.save();
    if (open && im.chat) im.chat->focus_input();
}

float ChatPane::toggle_width() {
    return ImGui::CalcTextSize(kToggleKey).x + ImGui::CalcTextSize("Chat").x + 32.0f;
}

void ChatPane::draw_toggle() {
    const bool was_open = open();
    if (sternbild::key_button(kToggleKey, "Chat")) toggle();
    if (was_open)   // the pane is open: the toggle is outlined in bronze
        ImGui::GetWindowDrawList()->AddRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), sternbild::col::bronze, 0.0f,
                                            0, 1.0f);
}

void ChatPane::read_keys() {
    if (!impl_->started) return;
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_L)) toggle();
}

void ChatPane::draw_column() {
    Impl& im = *impl_;
    if (!im.started || !im.settings.open || !im.chat) return;
    const SpecLook look = house_spec_look();
    const ImRect area = work_area();
    ChatColumnInput in;
    in.window = {area.Min.x, area.Min.y, area.GetWidth(), area.GetHeight()};
    const ChatColumnLayout col = layout_chat_column(in, ChatLayoutMetrics::for_text_scale(look.text_scale));
    if (col.column.w < 1.0f || col.column.h < 1.0f) return;
    ImGui::PushStyleColor(ImGuiCol_WindowBg, look.ground);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(0.0f, 0.0f));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar |
                                   ImGuiWindowFlags_NoScrollWithMouse;
    const bool shown = ImGui::BeginViewportSideBar("##caliper_side_chat", ImGui::GetMainViewport(), ImGuiDir_Right,
                                                   col.column.w, flags);
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor();
    if (shown) {
        const ImVec2 o = ImGui::GetWindowPos(), size = ImGui::GetWindowSize();
        // The chat takes the column's part that layout_chat_column() gives it.
        const float y0 = o.y + (col.chat.y - col.column.y);
        im.chat->draw(ImRect(o, ImVec2(o.x + size.x, y0 + std::min(col.chat.h, size.y))), look);
    }
    ImGui::End();
}

void ChatPane::draw_in(float x, float y, float w, float h) {
    Impl& im = *impl_;
    if (!im.started || !im.settings.open || !im.chat || w < 1.0f || h < 1.0f) return;
    im.chat->draw(ImRect(x, y, x + w, y + h), house_spec_look());
}

std::string ChatPane::state() const {
    return impl_->chat ? chat_state_name(impl_->chat->state()) : std::string("none");
}

bool ChatPane::has_item(const std::string& kind, const std::string& command, const std::string& status) const {
    if (!impl_->chat) return false;
    static const char* kinds[] = {"person", "agent", "thought", "tool", "plan", "permission", "approval", "stop", "notice",
                                  "applied"};
    for (const ChatItem& it : impl_->chat->items()) {
        if (kind != kinds[(int)it.kind]) continue;
        if (!command.empty() && it.command != command) continue;
        if (!status.empty()) {
            const std::string st = it.kind == ChatItem::Kind::tool         ? it.tool.status
                                   : it.kind == ChatItem::Kind::approval  ? (it.answer.empty() ? "waiting" : it.answer)
                                   : it.kind == ChatItem::Kind::permission ? (it.answer.empty() ? "open" : it.answer)
                                   : it.kind == ChatItem::Kind::applied    ? (it.answer.empty() ? "applied" : it.answer)
                                                                           : it.text;
            if (st != status) continue;
        }
        return true;
    }
    return false;
}

void ChatPane::focus_input() {
    if (impl_->chat) impl_->chat->focus_input();
}

int ChatPane::draft_chips() const { return impl_->chat ? (int)impl_->chat->draft().chips.size() : 0; }

std::string ChatPane::note() const { return impl_->chat ? impl_->chat->gpu_note() : std::string(); }

bool ChatPane::set_local_model(bool on) { return impl_->chat && impl_->chat->set_local_model(on); }

void ChatPane::open_picker() {
    if (impl_->chat) impl_->chat->open_picker();
}

std::string ChatPane::permission_mode() const { return impl_->chat ? impl_->chat->permission_word() : std::string(); }

void ChatPane::set_allow_all(bool on) {
    if (impl_->chat) impl_->chat->set_allow_all(on);
}

bool ChatPane::undo_applied(const std::string& command) {
    if (!impl_->chat) return false;
    const auto& items = impl_->chat->items();
    for (auto it = items.rbegin(); it != items.rend(); ++it)
        if (it->kind == ChatItem::Kind::applied && it->command == command && it->answer.empty())
            return impl_->chat->undo_applied(it->serial);
    return false;
}

void ChatPane::shutdown() {
    impl_->chat.reset();
    impl_->end_agent_file();
    impl_->started = false;
}

void chat_pane_handoff(ChatPane& pane, const WidgetChatRequest& request, const caliper_builder::ContextBundle& bundle,
                       const caliper_builder::spec::Spec* open_spec) {
    pane.set_open(true);
    ChatPane::Impl& im = pane.impl();
    ChatDraft d = chat_handoff_draft(request, bundle, open_spec);
    // A pane opened for the first time makes its chat at the next frame; the
    // draft waits for it.
    if (im.chat) im.chat->open_handoff(std::move(d));
    else im.draft = std::move(d);
}

SideChat* chat_pane_chat(ChatPane& pane) { return pane.impl().chat.get(); }

#else  // !CALIPER_WITH_BUILDER

struct ChatPane::Impl {};
ChatPane::ChatPane() : impl_(std::make_unique<Impl>()) {}
ChatPane::~ChatPane() = default;
bool ChatPane::built() { return false; }
void ChatPane::start(Options, Hooks) {}
void ChatPane::frame(AppSession&) {}
bool ChatPane::open() const { return false; }
void ChatPane::set_open(bool) {}
void ChatPane::draw_toggle() {}
float ChatPane::toggle_width() { return 0.0f; }
void ChatPane::read_keys() {}
void ChatPane::draw_column() {}
void ChatPane::draw_in(float, float, float, float) {}
std::string ChatPane::state() const { return "none"; }
bool ChatPane::has_item(const std::string&, const std::string&, const std::string&) const { return false; }
void ChatPane::focus_input() {}
int ChatPane::draft_chips() const { return 0; }
std::string ChatPane::note() const { return {}; }
bool ChatPane::set_local_model(bool) { return false; }
void ChatPane::open_picker() {}
std::string ChatPane::permission_mode() const { return {}; }
void ChatPane::set_allow_all(bool) {}
bool ChatPane::undo_applied(const std::string&) { return false; }
void ChatPane::shutdown() {}

#endif

}  // namespace caliper_host
