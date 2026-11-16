// The spec page of the app and of `caliper shot` (spec_page.h).
#include "builder/spec_page.h"

#include "builder/app_session.h"

#include "builder/chat_pane.h"

#if CALIPER_WITH_BUILDER
#include "builder/builder_pane.h"
#include "builder/panel_renderer.h"
#include "builder/spec_page_layout.h"
#include "builder/widget_app.h"
#include "builder/widget_surface.h"
#include "applet_spec.h"
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_runtime.h"
#include "ui/applet_page.h"      // begin_page_bar, JobsTray
#include "ui/landing_screen.h"   // draw_host_status_items
#include "ui/sternbild_fonts.h"
#include "ui/sternbild_theme.h"
#include "ui/work_area.h"

#include <caliper_builder/commands/command.h>
#include <caliper_builder/session/context_bundle.h>
#include <caliper_builder/session/session.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <future>
#include <map>
#include <optional>
#include <vector>
#endif

namespace caliper_host {

#if CALIPER_WITH_BUILDER

namespace cb = caliper_builder;

namespace {

// The data of the open spec's panels through the session, kept off the
// session while its worker runs a long command (AppSession::worker_busy()):
// the worker holds the registry's and the store's locks, and only the frame
// thread starts a worker, so within a frame "busy" can only turn into "done".
class PageSource final : public SpecPanelSource {
public:
    PageSource(cb::Session& session, const AppSession& app) : session_(session), inner_(session), app_(app) {}

    cb::PanelData resolve(const cb::spec::Spec& spec, std::string_view panel_id) override {
        if (app_.worker_busy()) {
            missed_ = true;
            cb::PanelData d;
            d.panel = std::string(panel_id);
            d.error = cb::Error("resolve.busy", "A loader is running; the panel is resolved when it is done.",
                                "worker");
            return d;
        }
        return inner_.resolve(spec, panel_id);
    }

    // A panel's served generation (live views, feed channels, metrics), the
    // last one read while a worker runs, so that a frame never waits on it.
    uint64_t generation(const cb::spec::Spec& spec, std::string_view panel_id) override {
        std::string id(panel_id);
        if (app_.worker_busy()) {
            const auto it = generations_.find(id);
            return it == generations_.end() ? 0 : it->second;
        }
        const uint64_t g = inner_.generation(spec, panel_id);
        generations_[std::move(id)] = g;
        return g;
    }

    // The catalog's revision, read only while no worker runs, plus one for
    // each stretch in which a panel could not be resolved because one did, so
    // that those panels are resolved once it is done. Both only grow.
    uint64_t revision() const override {
        if (!app_.worker_busy()) {
            catalog_ = session_.catalog().revision();
            if (missed_) {
                missed_ = false;
                ++retries_;
            }
        }
        return catalog_ + retries_;
    }

private:
    cb::Session& session_;
    SessionPanelSource inner_;
    const AppSession& app_;
    mutable uint64_t catalog_ = 0;
    mutable uint64_t retries_ = 0;
    mutable bool missed_ = false;
    std::map<std::string, uint64_t> generations_;
};

// panel.render through the runtime's offscreen renderer.
class PanelRenderOffscreen final : public PanelRenderSeam {
public:
    cb::CommandOutput render_panel(const cb::spec::Spec& spec, const cb::PanelData& data, int width,
                                   int height) override {
        return panel_render_output(renderer_.render_panel(spec, data, width, height));
    }

private:
    OffscreenPanelRenderer renderer_;
};

// The folder's own name, which the status bar shows for the workspace.
std::string folder_name(const std::string& path) {
    std::filesystem::path p(path);
    if (!p.has_filename()) p = p.parent_path();
    const std::string name = p.filename().string();
    return name.empty() ? path : name;
}

}  // namespace

struct SpecPage::Impl {
    std::unique_ptr<SpecTextures> textures = std::make_unique<SpecTextures>();
    std::unique_ptr<SpecRuntime> runtime = std::make_unique<SpecRuntime>(*textures);
    std::unique_ptr<PageSource> source;
    cb::Session* bound = nullptr;   // the session `source` reads

    uint64_t opens_taken = 0;               // AppSession::applets_opened() at the last take_open()
    std::optional<uint64_t> journal_read;   // AppSession::journaled() when the open spec was last read

    std::string title;
    std::string applet_id;
    std::string workspace;
    size_t panels = 0, sources = 0;
    bool has_spec = false;
    bool dirty = false;
    bool from_file = false;   // enter_file(): a spec applet's file, not the open spec

    std::optional<std::future<cb::CommandResult>> request;
    std::string request_error;
    bool request_is_new = false;   // the request is spec.new, not applet.open
    bool new_done = false;         // a spec.new succeeded since the last take_new()

    Seams seams;
    PanelOverlay panel_overlay;
    std::string focused;   // the focused panel's id, "" for none

    // The widget agent's inline surface (widget_surface.h), made at the
    // first ⌘K on a panel, with the agent the app has (widget_app.h).
    std::unique_ptr<InlineWidget> widget;
    bool hidden = false;
    ChatPane* chat = nullptr;   // the side chat, when the app has one

    // The builder pane over the bound session (builder_pane.h), made with it;
    // build mode is its own.
    std::unique_ptr<BuilderPane> builder;
    bool building() const { return builder && builder->build_mode() && !from_file; }

    void open_widget(AppSession& app, const std::string& panel) {
        if (!bound || from_file) return;
        if (!widget) {
            InlineWidget::Options o;
            o.session = bound;
            o.busy = [&app] { return app.worker_busy(); };
            o.agent = make_app_widget_agent(app_widget_env(hidden, app.workspace()));
            // Ask in chat: the side chat opens with the request and its context
            // as a draft. The bundle is built here, inside the frame, so it
            // leaves the panel's render out (no offscreen draw mid-frame).
            o.hooks.ask_in_chat = [this](const WidgetChatRequest& r) {
                if (!chat || !bound) return;
                cb::ContextBundleOptions bo;
                bo.vision = false;
                const cb::ContextBundle bundle = cb::build_context_bundle(*bound, r.target.to_builder(), bo);
                const std::optional<cb::OpenSpec> open = bound->open_spec();
                chat_pane_handoff(*chat, r, bundle, open ? &open->spec : nullptr);
            };
            widget = std::make_unique<InlineWidget>(std::move(o));
            widget->attach(*runtime);
            install_overlay();   // the page's, which calls the widget's
        }
        widget->surface().open(WidgetTarget::for_panel(panel));
    }

    // The page's command input, which ⌘K opens when no panel is focused:
    // a ruled line in the page bar that finds a panel by its title as you
    // type and focuses it, and Enter opens that panel's inline prompt.
    bool command_open = false;
    bool command_focus = false;
    std::string command;

    // The first shown panel whose title or id holds every word of `text`.
    std::string find_panel(const std::string& text) const {
        auto lower = [](std::string x) {
            for (char& c : x) c = (char)std::tolower((unsigned char)c);
            return x;
        };
        std::vector<std::string> words;
        {
            std::string w;
            for (char c : lower(text) + " ") {
                if (std::isspace((unsigned char)c)) {
                    if (!w.empty()) words.push_back(w);
                    w.clear();
                } else {
                    w += c;
                }
            }
        }
        if (words.empty()) return {};
        for (const auto& lp : runtime->layout().panels) {
            const PreparedPanel* p = runtime->prepared(lp.id);
            const std::string hay = lower(lp.id) + "\n" + (p ? lower(p->title) : std::string());
            bool all = true;
            for (const auto& w : words) all = all && hay.find(w) != std::string::npos;
            if (all) return lp.id;
        }
        return {};
    }

    bool widget_open() const {
        if (!widget) return false;
        const WidgetState st = widget->surface().state();
        return st != WidgetState::idle && st != WidgetState::applied;
    }

    // The session the page reads: a new one (the workspace opened, or the
    // session closed) gets a new source.
    void bind(AppSession& app) {
        if (workspace.empty()) workspace = folder_name(app.workspace());
        cb::Session* s = app.session();
        if (s == bound) return;
        if (widget) {
            // Its session has gone or changed: the surface forgets its
            // preview without a command to it, then detaches.
            widget->surface().abandon();
            widget.reset();
            install_overlay();
        }
        bound = s;
        source = s ? std::make_unique<PageSource>(*s, app) : nullptr;
        runtime->set_source(source.get());
        journal_read.reset();
        builder.reset();
        if (s) {
            AppSession* a = &app;
            auto port = std::make_unique<SessionBuilderPort>(
                *s, [a] { return a->worker_busy(); }, [a] { return a->journaled(); });
            builder = std::make_unique<BuilderPane>(std::move(port));
            builder->set_layout_rule(spec_page_layout_rule);
        }
    }

    // The open spec, read again when the journal recorded a command since.
    void refresh(AppSession& app) {
        if (from_file || !bound || app.worker_busy()) return;
        const uint64_t j = app.journaled();
        if (journal_read && *journal_read == j) return;
        journal_read = j;
        const std::optional<cb::OpenSpec> open = bound->open_spec();
        if (!open) {
            runtime->clear_spec();
            has_spec = false;
            applet_id.clear();
            panels = sources = 0;
            return;
        }
        show(open->spec);
        dirty = open->dirty;
    }

    void show(const cb::spec::Spec& spec) {
        runtime->set_spec(spec);
        has_spec = true;
        const auto& a = spec.applet;
        title = a.title && !a.title->empty() ? *a.title : a.id;
        applet_id = a.id;
        panels = spec.panels.size();
        sources = spec.sources.size();
    }

    // `command_key`: the page bar's ⌘K Command was clicked this frame.
    bool draw_bars(const char* renderer_name, JobsTray* tray, bool* command_key) {
        bool home = false;
        // The page bar, as the applet page's: Home as a key button holding
        // its key, then the spec's title where an applet's name goes, the
        // page's command input while it is open, and at the right end ⌘K
        // Command, the same control as the landing screen's command bar.
        if (begin_page_bar()) {
            if (sternbild::key_button("esc", "Home")) home = true;
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
            ImGui::TextUnformatted(has_spec ? title.c_str() : "");
            ImGui::PopStyleColor();
            if (command_open) draw_command_input();
#ifdef __APPLE__
            const char* key = "\xe2\x8c\x98K";   // ⌘K
#else
            const char* key = "Ctrl K";
#endif
            const float w = ImGui::CalcTextSize(key).x + ImGui::CalcTextSize("Command").x + 32.0f;
            // ⌘L Chat, then ⌘K Command at the right end.
            const float cw = chat ? ChatPane::toggle_width() + ImGui::GetStyle().ItemSpacing.x : 0.0f;
            const float right = ImGui::GetWindowContentRegionMax().x - w - cw - sternbild::metric::pad;
            // The builder's items after the title, in the room the right end
            // leaves (only ⌘B Build when it is short); none for a spec
            // applet's file, which the builder does not edit.
            if (builder && has_spec && !from_file) {
                ImGui::Separator();
                builder->page_bar(right - ImGui::GetCursorPosX() - 2.0f * sternbild::metric::pad);
                ImGui::SameLine();
            }
            if (right > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(right);
            if (chat) chat->draw_toggle();
            if (sternbild::key_button(key, "Command")) *command_key = true;
            ImGui::EndMainMenuBar();
        }
        // The status bar: the machine, then the page's facts.
        sternbild::status_bar_begin();
        ::draw_host_status_items(renderer_name);
        {
            char text[96];
            std::snprintf(text, sizeof text, "workspace %s", workspace.c_str());
            sternbild::status_item(text);
            if (has_spec) {
                std::snprintf(text, sizeof text, panels == 1 ? "%zu panel" : "%zu panels", panels);
                sternbild::status_item(text);
                std::snprintf(text, sizeof text, sources == 1 ? "%zu source" : "%zu sources", sources);
                sternbild::status_item(text);
                if (dirty) sternbild::status_item("unsaved");
            }
        }
        sternbild::status_bar_end();
        // The jobs tray's strip above the status bar while it is open (and
        // the approval cards' above it), then the side chat's column at the
        // right: the spec fills what they leave.
        if (tray) tray->draw();
        // In build mode the chat is the builder's right column's lower part,
        // drawn inside the page (ChatPane::draw_in()); otherwise its own column.
        if (chat && !building()) chat->draw_column();
        return home;
    }

    // The command input: the house's ruled line (the landing's), with the
    // bronze prompt mark before it. Typing focuses the first panel whose
    // title holds the words; Enter opens its prompt; Escape closes the line.
    bool command_entered = false;
    void draw_command_input() {
        ImGui::Separator();
        ImGui::PushFont(sternbild::fonts().mono, sternbild::font_size::mono);
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::bronze);
        ImGui::TextUnformatted("\xe2\x80\xba");   // ›
        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, IM_COL32(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, IM_COL32(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_NavCursor, IM_COL32(0, 0, 0, 0));   // the rule marks it
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
        ImGui::SetNextItemWidth(240.0f);
        char buf[256];
        std::snprintf(buf, sizeof buf, "%s", command.c_str());
        const bool entered = ImGui::InputTextWithHint("##spec.command", "Panel", buf, sizeof buf,
                                                      ImGuiInputTextFlags_EnterReturnsTrue);
        if (command_focus) {
            // The page bar is a menu bar, where keyboard focus does not go;
            // the input is activated instead, from the next frame.
            ImGui::ActivateItemByID(ImGui::GetItemID());
            command_focus = false;
        }
        const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(a.x, b.y - 1.0f), b,
                                                  ImGui::IsItemActive() ? sternbild::col::parchment
                                                                        : sternbild::col::rule);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(4);
        if (command != buf) {
            command = buf;
            const std::string found = find_panel(command);
            if (!found.empty()) focused = found;
        }
        if (entered) command_entered = true;
    }

    // The spec over the work area the strips leave this frame.
    void draw_spec() {
        const ImRect area = work_area();
        if (area.GetWidth() < 1.0f || area.GetHeight() < 1.0f) return;
        ImGui::SetNextWindowPos(area.Min);
        ImGui::SetNextWindowSize(area.GetSize());
        ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                                       ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                                       ImGuiWindowFlags_NoScrollWithMouse;
        const bool open = ImGui::Begin("##caliper_spec_page", nullptr, flags);
        ImGui::PopStyleVar(3);
        if (open) {
            const SpecLook look = house_spec_look();
            // The page routes ⌘K and Escape (spec_key_action); the surface
            // takes Enter on its card and polls its agent.
            if (widget) widget->frame(look, focused, /*keys=*/false);
            if (building()) {
                // The builder's regions; it draws the runtime on its canvas,
                // takes the page's focus when the page changed it and gives
                // its own back.
                builder->set_chat(chat && chat->open(), [this](const SpecRect& r) {
                    if (chat) chat->draw_in(r.x, r.y, r.w, r.h);
                });
                builder->draw(area, *runtime, look, &focused);
            } else {
                runtime->draw(area, look);
                focus_and_frame();
            }
        }
        ImGui::End();
    }

    // The runtime's overlay, inside each panel's own window after its
    // content: the bronze frame of the focused panel, which the panel's
    // content would otherwise cover, the app's panel overlay (the approval
    // cards' mark), then the inline surface.
    void install_overlay() {
        runtime->set_overlay([this](const SpecPanelFrame& f) {
            if (f.id == focused)
                ImGui::GetWindowDrawList()->AddRect(f.rect.Min, f.rect.Max, sternbild::col::bronze, 0.0f, 0, 1.0f);
            if (panel_overlay) {
                const PreparedPanel* p = runtime->prepared(f.id);
                panel_overlay(f.id, p ? p->title : std::string(), f.rect.Min.x, f.rect.Min.y, f.rect.GetWidth(),
                              f.rect.GetHeight(), f.header.GetHeight());
            }
            if (widget) widget->overlay(f);
        });
    }

    // A click inside a shown panel focuses it, and the focused panel gets a
    // bronze frame, the palette's colour for focus, drawn by the overlay
    // above. A panel that is no longer shown loses the focus.
    void focus_and_frame() {
        const SpecLayout& layout = runtime->layout();
        const bool clicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                             ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
        const ImVec2 mouse = ImGui::GetMousePos();
        bool still_shown = false;
        for (const auto& lp : layout.panels) {
            const ImRect r(ImVec2(lp.rect.x, lp.rect.y), ImVec2(lp.rect.right(), lp.rect.bottom()));
            if (clicked && r.Contains(mouse)) focused = lp.id;
            if (lp.id == focused) still_shown = true;
        }
        if (!still_shown) focused.clear();
    }
};

SpecPage::SpecPage() : impl_(std::make_unique<Impl>()) { impl_->install_overlay(); }

SpecPage::~SpecPage() = default;

bool SpecPage::built() { return true; }

bool SpecPage::take_open(const AppSession& session) {
    const uint64_t n = session.applets_opened();
    if (n == impl_->opens_taken) return false;
    impl_->opens_taken = n;
    return true;
}

bool SpecPage::request_open(AppSession& session, const std::string& id) {
    cb::Session* s = session.session();
    if (!s) return false;
    cb::CommandInvocation inv;
    inv.name = "applet.open";
    inv.args = {{"id", id}};
    inv.origin = cb::CommandOrigin::ui;
    impl_->request = s->queue().submit(std::move(inv));
    impl_->request_is_new = false;
    impl_->request_error.clear();
    return true;
}

bool SpecPage::request_new(AppSession& session, const std::string& title) {
    cb::Session* s = session.session();
    if (!s) return false;
    const std::filesystem::path applets = std::filesystem::path(s->workspace()) / "applets";
    const std::string id = pane_new_spec_id(title, [&](const std::string& x) {
        std::error_code ec;
        return std::filesystem::exists(applets / x, ec);
    });
    cb::CommandInvocation inv;
    inv.name = "spec.new";
    inv.args = {{"id", id}, {"title", title.empty() ? id : title}};
    inv.origin = cb::CommandOrigin::ui;
    impl_->request = s->queue().submit(std::move(inv));
    impl_->request_is_new = true;
    impl_->request_error.clear();
    return true;
}

bool SpecPage::take_new() {
    const bool r = impl_->new_done;
    impl_->new_done = false;
    return r;
}

bool SpecPage::has_open_spec(AppSession& session) {
    cb::Session* s = session.session();
    return s && !session.worker_busy() && s->open_spec().has_value();
}

bool SpecPage::importing() const { return impl_->builder && impl_->builder->importing(); }

bool SpecPage::request_answered(std::string* error) {
    Impl& im = *impl_;
    if (!im.request) return false;
    if (im.request->wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
    const cb::CommandResult r = im.request->get();
    im.request.reset();
    if (im.request_is_new && r.ok) im.new_done = true;
    im.request_is_new = false;
    if (error) *error = r.ok ? std::string() : r.error.code + ": " + r.error.message;
    return true;
}

void SpecPage::enter(AppSession& session) {
    impl_->from_file = false;
    impl_->bind(session);
    impl_->journal_read.reset();
    impl_->refresh(session);
}

bool SpecPage::enter_file(AppSession& session, const std::string& spec_path, std::string* error) {
    SpecFile file = read_spec_file(spec_path);
    if (!file.ok) {
        if (error) *error = file.reason;
        return false;
    }
    Impl& im = *impl_;
    im.bind(session);
    im.from_file = true;
    im.dirty = false;
    im.show(file.spec);
    return true;
}

SpecPage::Result SpecPage::draw(AppSession& session, const char* renderer_name, JobsTray* tray, bool escape) {
    Impl& im = *impl_;
    im.bind(session);
    im.refresh(session);
    Result r;
    bool clicked_command = false;
    r.home = im.draw_bars(renderer_name, tray, &clicked_command);
    im.draw_spec();
    // ⌘K and Escape (spec_key_action). ImGuiMod_Ctrl is the Command key on
    // macOS, as on the landing screen; the page bar's ⌘K Command is the same
    // key. An open inline prompt, and then the page's command input, take
    // Escape first, also while their input has the keyboard; the page's
    // order follows only when neither is open.
    const bool command_key = clicked_command || ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_K);
    if (im.chat) im.chat->read_keys();   // ⌘L Chat
    const bool esc_key = ImGui::IsKeyPressed(ImGuiKey_Escape, false);
    if (im.widget_open() && esc_key) {
        im.widget->surface().escape();
        escape = false;
    } else if (im.command_open && esc_key) {
        im.command_open = false;
        im.command.clear();
        escape = false;
    }
    // The builder's keys (⌘B, and in build mode ⌘T, ⌘Z, ⇧⌘Z, ⌘S, ⌫), and
    // its Escape (a drag, the text editor, a source's focus) before the
    // page's (a panel's focus, then Home).
    if (im.builder && im.has_spec && !im.from_file && !im.widget_open() && im.builder->keys(escape)) escape = false;
    if (im.command_entered) {
        // Enter on the command input: the found panel's inline prompt.
        im.command_entered = false;
        im.command_open = false;
        im.command.clear();
        if (!im.focused.empty()) {
            if (im.seams.panel_prompt) im.seams.panel_prompt(im.focused);
            else im.open_widget(session, im.focused);
        }
    }
    switch (spec_key_action(command_key, escape, !im.focused.empty())) {
        case SpecKeyAction::panel_prompt:
            im.command_open = false;
            if (im.seams.panel_prompt) im.seams.panel_prompt(im.focused);
            else im.open_widget(session, im.focused);
            break;
        case SpecKeyAction::page_command:
            if (im.seams.page_command) {
                im.seams.page_command();
            } else {
                im.command_open = true;
                im.command_focus = true;
            }
            break;
        case SpecKeyAction::unfocus:
            im.focused.clear();
            break;
        case SpecKeyAction::home:
            r.home = true;
            break;
        case SpecKeyAction::none:
            break;
    }
    return r;
}

bool SpecPage::build_mode() const { return impl_->building(); }

void SpecPage::set_build_mode(bool on) {
    if (impl_->builder && !impl_->from_file) impl_->builder->set_build_mode(on);
}

bool SpecPage::drop_files(AppSession& session, const std::vector<std::string>& paths, std::string* why) {
    Impl& im = *impl_;
    im.bind(session);
    if (!im.builder || im.from_file || !im.has_spec) {
        if (why) *why = "no open spec is shown";
        return false;
    }
    im.builder->set_build_mode(true);
    if (!im.builder->drop_files(paths)) {
        if (why) *why = im.builder->error().message;
        return false;
    }
    return true;
}

BuilderPane* SpecPage::builder() { return impl_->builder.get(); }

void SpecPage::set_seams(Seams seams) { impl_->seams = std::move(seams); }

void SpecPage::set_hidden(bool hidden) { impl_->hidden = hidden; }

void SpecPage::set_chat(ChatPane* chat) { impl_->chat = chat; }

bool SpecPage::panel_rect(const std::string& panel_id, float rect[4]) const {
    for (const auto& lp : impl_->runtime->layout().panels)
        if (lp.id == panel_id) {
            rect[0] = lp.rect.x;
            rect[1] = lp.rect.y;
            rect[2] = lp.rect.w;
            rect[3] = lp.rect.h;
            return true;
        }
    return false;
}

std::string SpecPage::widget_state() const {
    return impl_->widget ? widget_state_name(impl_->widget->surface().state()) : "idle";
}

void SpecPage::set_panel_overlay(PanelOverlay overlay) { impl_->panel_overlay = std::move(overlay); }

const std::string& SpecPage::focused_panel() const { return impl_->focused; }

void SpecPage::focus_panel(const std::string& panel_id) { impl_->focused = panel_id; }

void SpecPage::leave() {
    // An open prompt closes with the page: its preview is discarded. The page
    // opens again out of build mode.
    if (impl_->widget) impl_->widget->surface().close();
    if (impl_->builder) impl_->builder->set_build_mode(false);
    impl_->focused.clear();
    impl_->runtime->clear_spec();
    impl_->journal_read.reset();
    impl_->has_spec = false;
    impl_->from_file = false;
    impl_->applet_id.clear();
}

const std::string& SpecPage::title() const { return impl_->title; }

const std::string& SpecPage::applet_id() const { return impl_->applet_id; }

bool SpecPage::facts(const std::string& spec_path, SpecAppletFacts* out) {
    // One reading per file and change, on the frame thread.
    struct Kept {
        std::filesystem::file_time_type mtime;
        SpecAppletFacts facts;
    };
    static std::map<std::string, Kept> kept;
    std::error_code ec;
    const auto mtime = std::filesystem::last_write_time(spec_path, ec);
    if (ec) return false;
    auto it = kept.find(spec_path);
    if (it == kept.end() || it->second.mtime != mtime) {
        Kept k;
        k.mtime = mtime;
        const SpecFile file = read_spec_file(spec_path);
        if (file.ok) {
            k.facts.format = file.spec.version;
            k.facts.panels = file.spec.panels.size();
            k.facts.sources = file.spec.sources.size();
        } else {
            k.facts.error = file.reason;
        }
        if (kept.size() > 256) kept.clear();
        it = kept.insert_or_assign(spec_path, std::move(k)).first;
    }
    if (out) *out = it->second.facts;
    return true;
}

void SpecPage::shutdown() {
    // The app has closed the builder session by now (main.cpp closes it
    // before the renderer, and the page's textures need the renderer gone
    // first): the inline surface forgets its preview without a command,
    // since a command into the closed session crashed the app on quitting
    // with a prompt open; then it goes, before the runtime it is attached to.
    if (impl_->widget) impl_->widget->surface().abandon();
    impl_->widget.reset();
    impl_->builder.reset();
    impl_->runtime.reset();
    impl_->source.reset();
    impl_->bound = nullptr;
    impl_->textures.reset();
}

std::shared_ptr<PanelRenderSeam> SpecPage::make_panel_renderer() {
    return std::make_shared<PanelRenderOffscreen>();
}

#else  // !CALIPER_WITH_BUILDER

// Without the builder there is no session and no spec: the page never opens.
struct SpecPage::Impl {
    std::string title;
    std::string focused;
};

SpecPage::SpecPage() : impl_(std::make_unique<Impl>()) {}
SpecPage::~SpecPage() = default;
bool SpecPage::built() { return false; }
bool SpecPage::take_open(const AppSession&) { return false; }
bool SpecPage::request_open(AppSession&, const std::string&) { return false; }
bool SpecPage::request_answered(std::string*) { return false; }
bool SpecPage::request_new(AppSession&, const std::string&) { return false; }
bool SpecPage::take_new() { return false; }
bool SpecPage::has_open_spec(AppSession&) { return false; }
bool SpecPage::importing() const { return false; }
void SpecPage::enter(AppSession&) {}
SpecPage::Result SpecPage::draw(AppSession&, const char*, JobsTray*, bool) { return {}; }
void SpecPage::set_seams(Seams) {}
bool SpecPage::build_mode() const { return false; }
void SpecPage::set_build_mode(bool) {}
bool SpecPage::drop_files(AppSession&, const std::vector<std::string>&, std::string* why) {
    if (why) *why = "the builder is not in this build";
    return false;
}
BuilderPane* SpecPage::builder() { return nullptr; }
void SpecPage::set_hidden(bool) {}
void SpecPage::set_chat(ChatPane*) {}
bool SpecPage::panel_rect(const std::string&, float*) const { return false; }
std::string SpecPage::widget_state() const { return "idle"; }
void SpecPage::set_panel_overlay(PanelOverlay) {}
const std::string& SpecPage::focused_panel() const { return impl_->focused; }
void SpecPage::focus_panel(const std::string&) {}
void SpecPage::leave() {}
const std::string& SpecPage::title() const { return impl_->title; }
const std::string& SpecPage::applet_id() const { return impl_->title; }
bool SpecPage::enter_file(AppSession&, const std::string&, std::string* error) {
    if (error) *error = "the builder is not in this build";
    return false;
}
bool SpecPage::facts(const std::string&, SpecAppletFacts*) { return false; }
void SpecPage::shutdown() {}
std::shared_ptr<PanelRenderSeam> SpecPage::make_panel_renderer() { return nullptr; }

#endif

}  // namespace caliper_host
