// The spec page of the app and of `caliper shot` (spec_page.h).
#include "builder/spec_page.h"

#include "builder/app_session.h"

#include "builder/chat_pane.h"

#if CALIPER_WITH_BUILDER
#include "builder/agent_settings.h"
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
#include <cmath>
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

// The label of ⌘O in the page bar, and what the page says of it while the
// open spec has unsaved changes.
constexpr const char* kBrowserLabel = "Open in browser";
constexpr const char* kBrowserNote = "the browser shows the saved dashboard";

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

    uint64_t opens_taken = 0;               // AppSession::specs_opened() at the last take_open()
    uint64_t agent_opens_taken = 0;         // AppSession::agent_specs_opened() then
    bool opened_by_agent = false;           // every open the last take_open() took was an agent's
    std::optional<uint64_t> journal_read;   // AppSession::journaled() when the open spec was last read

    std::string title;
    std::string applet_id;
    std::string workspace;
    size_t panels = 0, sources = 0;
    bool has_spec = false;
    bool dirty = false;
    bool from_file = false;   // enter_file(): a spec applet's file, not the open spec
    // The open spec's saved applet in the workspace (applets/<id>/), "" while
    // it was never saved: what ⌘O Open in browser shows.
    std::string saved_id;
    float browser_rect[4] = {0, 0, 0, 0};   // the button, as the last frame drew it
    // The page's named rectangles for the input driver (item_rect()), as the
    // last frame drew them: "panel:<id>", "remove:<id>" (the × on a panel's
    // strip) and "menu:remove" (Remove panel in a panel's menu).
    std::map<std::string, ImRect> items;
    std::string remove_request;   // the × clicked this frame, removed after drawing
    std::string menu_request;     // a panel right-clicked this frame
    std::string menu_panel;       // the panel whose menu is open

    std::optional<std::future<cb::CommandResult>> request;       // the last applet.open
    std::optional<std::future<cb::CommandResult>> request_new_f; // the last spec.new, apart, so an open never drops it
    std::string request_error;
    bool new_done = false;         // a spec.new succeeded since the last take_new()

    Seams seams;
    PanelOverlay panel_overlay;
    std::string focused;   // the focused panel's id, "" for none

    // The widget agent's inline surface (widget_surface.h), made at the
    // first ⌘K on a panel, with the agent the app has (widget_app.h).
    std::unique_ptr<InlineWidget> widget;
    bool hidden = false;
    ChatPane* chat = nullptr;   // the side chat, when the app has one
    // ⌘O Open in browser (set_open_in_browser()), and its last refusal.
    std::function<std::string(const std::string&)> open_in_browser;
    std::string browser_why;
    std::string browser_given;   // the saved dashboard whose link went out
    // The names of a catalog source's dims, for the builder's Properties.
    std::function<std::vector<std::string>(const std::string&)> dim_names;
    ApiFiles api_files;         // the sidecar's own API file
    std::string widget_api_file;

    // Drops the inline surface and ends its sidecar's token.
    void drop_widget() {
        widget.reset();
        if (!widget_api_file.empty() && api_files.revoke) api_files.revoke(widget_api_file);
        widget_api_file.clear();
    }
    AgentSettings* settings = nullptr;   // the widget agent's settings, when the app has them
    int widget_revision = -1;            // the settings' revision the widget's agent was made with
    AppSession* app = nullptr;           // the app's session, from the last bind()

    // A changed setting applies at once: the inline prompt closes as Close
    // closes it (a running turn is cancelled, a preview discarded) while its
    // session is the app's, and is forgotten without a command otherwise; then
    // the agent goes, so its sidecar ends with the key in its environment and
    // its token is revoked. The next ⌘K makes the agent from the settings.
    void retire_widget() {
        if (!widget) return;
        if (bound && app && app->session() == bound) widget->surface().close();
        else widget->surface().abandon();
        drop_widget();
        if (runtime) install_overlay();
    }

    // The builder pane over the bound session (builder_pane.h), made with it;
    // build mode is its own.
    std::unique_ptr<BuilderPane> builder;
    bool building() const { return builder && builder->build_mode() && !from_file; }

    // ⌘K on a panel: the prompt for the panel, or for its selection when it
    // has one (spec_runtime.h, SpecSelection), with the selected rows.
    void open_widget(AppSession& app, const std::string& panel) {
        const SpecSelection& sel = runtime->selection();
        if (!sel.empty() && sel.panel == panel) {
            if (const PreparedPanel* p = runtime->prepared(panel)) {
                const SpecSelectionRows rows = spec_selection_rows(*p, sel, 1000);
                cb::WidgetSelection ws;
                ws.kind = sel.kind == SpecSelection::Kind::points ? "points" : "range";
                ws.columns = rows.columns;
                ws.rows = nlohmann::json::array();
                for (const auto& row : rows.rows) {
                    nlohmann::json cells = nlohmann::json::array();
                    for (double v : row) cells.push_back(std::isfinite(v) ? nlohmann::json(v) : nlohmann::json(nullptr));
                    ws.rows.push_back(std::move(cells));
                }
                const std::string x = rows.columns.empty() ? "x" : rows.columns.front();
                ws.range = {{x, {sel.x0, sel.x1}}};
                if (sel.kind == SpecSelection::Kind::points && rows.columns.size() > 1)
                    ws.range[rows.columns[1]] = {sel.y0, sel.y1};
                open_widget_target(app, WidgetTarget::for_selection(panel, std::move(ws)));
                return;
            }
        }
        open_widget_target(app, WidgetTarget::for_panel(panel));
    }

    void open_widget_target(AppSession& app, const WidgetTarget& target) {
        if (!bound || from_file) return;
        // Settings changed since the agent was made (set_agent_settings()
        // retires it at each change; this catches a change it did not see).
        if (widget && settings && settings->revision() != widget_revision) retire_widget();
        if (!widget) {
            InlineWidget::Options o;
            o.session = bound;
            o.busy = [&app] { return app.worker_busy(); };
            widget_api_file = api_files.issue ? api_files.issue() : std::string();
            std::string none;
            AppWidgetEnv env = app_widget_env(hidden, app.workspace(), widget_api_file);
            if (settings) env.live_attached = settings->options().live_attached;
            o.agent = make_app_widget_agent(env, &none);
            if (!none.empty() && !widget_api_file.empty()) {   // no sidecar runs: its token ends now
                if (api_files.revoke) api_files.revoke(widget_api_file);
                widget_api_file.clear();
            }
            widget_revision = settings ? settings->revision() : -1;
            if (settings) {
                // The bundle takes at most a quarter of the model's context
                // length, and no render for a model that reads no images.
                const AgentCapabilities caps = settings->capabilities();
                if (caps.context && *caps.context > 0) o.bundle.context_tokens = (std::uint64_t)*caps.context;
                if (caps.vision && !*caps.vision) o.bundle.vision = false;
            }
            o.hooks.open_settings = [this] {
                if (settings) settings->set_open(true);
            };
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
        widget->surface().open(target);
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

    // Whether the person can remove the panel `id` from the page (the × on
    // its strip, its menu, ⌫): a panel of the open spec, not one drawn from
    // a file, and nothing proposed on the page. While the inline prompt is
    // open, or a ⌘K preview is shown, no panel is offered: the preview's
    // panels are only proposed, and removing the panel a preview is on
    // would leave the preview pointing at nothing; Accept or Reject first.
    bool can_remove(const std::string& id) const {
        if (!builder || !has_spec || from_file || widget_open() || !runtime) return false;
        if (runtime->has_spec_preview() || runtime->has_preview(id)) return false;
        const auto& shown = runtime->spec();   // the open spec as the page shows it
        return shown && shown->find_panel(id) != nullptr;
    }

    // The × on a panel's strip, at its right end (a tab's panel, whose tab
    // carries the title, at its top right corner), drawn while the pointer
    // is over the panel; a click removes the panel after the frame is drawn,
    // and a right click on the panel opens its menu. Inside the panel's own
    // window, after its content.
    void remove_control(const SpecPanelFrame& f) {
        if (f.preview || !can_remove(f.id)) return;
        const SpecLook L = house_spec_look();
        const float s = L.text_scale;
        const bool strip = f.header.GetHeight() >= 1.0f;
        const float side = strip ? f.header.GetHeight() : std::round(22.0f * s);
        const ImVec2 corner = strip ? ImVec2(f.header.Max.x, f.header.Min.y) : ImVec2(f.rect.Max.x, f.rect.Min.y);
        const ImRect x(ImVec2(corner.x - side, corner.y), ImVec2(corner.x, corner.y + side));
        items["remove:" + f.id] = x;
        if (building()) builder->claim_canvas(SpecRect{x.Min.x, x.Min.y, x.GetWidth(), x.GetHeight()});
        // The cursor stays after the button: moving it back past the panel's
        // content with nothing submitted there is an error in ImGui.
        ImGui::SetCursorScreenPos(x.Min);
        ImGui::PushID(f.id.c_str());
        const bool clicked = ImGui::InvisibleButton("##panel.remove", x.GetSize());
        const bool on_x = ImGui::IsItemHovered();
        ImGui::PopID();
        const bool menu = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
        const bool over = !menu && ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
                          ImGui::IsMouseHoveringRect(f.rect.Min, f.rect.Max);
        if (over || on_x) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(x.Min, x.Max, on_x ? L.plate_hover : L.plate);
            const ImVec2 c = x.GetCenter();
            const float a = std::round(4.0f * s);
            const ImU32 ink = on_x ? L.accent : L.secondary;
            dl->AddLine(ImVec2(c.x - a, c.y - a), ImVec2(c.x + a, c.y + a), ink, std::max(1.0f, std::round(1.5f * s)));
            dl->AddLine(ImVec2(c.x - a, c.y + a), ImVec2(c.x + a, c.y - a), ink, std::max(1.0f, std::round(1.5f * s)));
            if (on_x) ImGui::SetTooltip("Remove panel");
        }
        if (clicked) remove_request = f.id;
        if (over && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) menu_request = f.id;
    }

    // A panel's menu, opened by a right click on it: Remove panel. In the
    // page's window, after the panels, in the house style of the pane's menus.
    void panel_menu() {
        if (!menu_request.empty()) {
            menu_panel = menu_request;
            menu_request.clear();
            ImGui::OpenPopup("##panel.menu");
        }
        const SpecLook L = house_spec_look();
        const float s = L.text_scale;
        ImGui::PushStyleColor(ImGuiCol_PopupBg, L.deep);
        ImGui::PushStyleColor(ImGuiCol_Border, L.rule);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, std::round(4.0f * s)));
        if (ImGui::BeginPopup("##panel.menu")) {
            const float row = std::round(26.0f * s), w = std::round(160.0f * s), pad = std::round(10.0f * s);
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const ImRect r(at, ImVec2(at.x + w, at.y + row));
            const bool ok = can_remove(menu_panel);
            const bool click = ImGui::InvisibleButton("##panel.menu.remove", r.GetSize());
            items["menu:remove"] = r;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            if (ok && ImGui::IsItemHovered()) dl->AddRectFilled(r.Min, r.Max, L.plate_hover);
            dl->AddText(L.ui, L.ui_size, ImVec2(r.Min.x + pad, r.Min.y + std::round((row - L.ui_size) * 0.5f)),
                        ok ? L.text : L.faint, "Remove panel");
            if (click && ok) {
                remove_request = menu_panel;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        } else {
            menu_panel.clear();
            items.erase("menu:remove");
        }
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);
    }

    // The panel the × or the menu asked to remove, now that nothing draws it.
    void take_remove() {
        if (remove_request.empty()) return;
        const std::string id = remove_request;
        remove_request.clear();
        if (can_remove(id) && builder->remove_panel(id) && focused == id) focused.clear();
    }

    // The session the page reads: a new one (the workspace opened, or the
    // session closed) gets a new source.
    void bind(AppSession& app) {
        this->app = &app;
        if (workspace.empty()) workspace = folder_name(app.workspace());
        cb::Session* s = app.session();
        if (s == bound) return;
        if (widget) {
            // Its session has gone or changed: the surface forgets its
            // preview without a command to it, then detaches.
            widget->surface().abandon();
            drop_widget();
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
            port->set_dim_names([this](const std::string& id) {
                return dim_names ? dim_names(id) : std::vector<std::string>();
            });
            builder = std::make_unique<BuilderPane>(std::move(port));
            builder->set_layout_rule(spec_page_layout_rule);
            // ⌘K in the builder: a dropped file's row and a property's row, a
            // panel's or a spec source's, open the inline prompt there, and it
            // is drawn under that row.
            BuilderPane::WidgetHooks hooks;
            hooks.open_file = [this, a](const std::string& loader, const std::vector<std::string>& catalog,
                                        const std::string& anchor) {
                open_widget_target(*a, WidgetTarget::for_dropped_file(loader, catalog, anchor));
            };
            hooks.open_property = [this, a](const std::string& panel, const std::string& source, const std::string& path) {
                open_widget_target(*a, WidgetTarget::for_property(panel, source, path));
            };
            hooks.height = [this](const std::string& anchor, float w) {
                return widget ? widget->surface().row_height(anchor, w) : 0.0f;
            };
            hooks.draw = [this](const std::string& anchor, const SpecRect& r) {
                if (widget) widget->surface().draw_row(anchor, ImRect(ImVec2(r.x, r.y), ImVec2(r.right(), r.bottom())));
            };
            builder->set_widget_hooks(std::move(hooks));
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
            saved_id.clear();
            panels = sources = 0;
            return;
        }
        show(open->spec);
        dirty = open->dirty;
        saved_id = saved_applet(open->path);
    }

    // The workspace id of a spec saved at `path` (relative to the workspace,
    // applets/<id>/<file>), "" for one never saved or saved elsewhere.
    static std::string saved_applet(const std::optional<std::string>& path) {
        if (!path) return {};
        const std::filesystem::path p(*path);
        auto it = p.begin();
        if (it == p.end() || it->string() != "applets") return {};
        if (++it == p.end()) return {};
        return it->string();
    }

    // ⌘O Open in browser: disabled, with the reason, while there is nothing
    // the browser's page can show.
    std::string browser_disabled() const {
        if (from_file) return "the browser shows the workspace's saved dashboards";
        if (saved_id.empty()) return "save the dashboard first";
        return {};
    }
    void open_browser() {
        if (!open_in_browser || !browser_disabled().empty()) return;
        browser_why = open_in_browser(saved_id);
        if (browser_why.empty()) browser_given = saved_id;
    }
    // ⌘O opens the saved file: once its link is out, and the open spec has
    // changes since, the page says that the browser shows the saved one.
    std::string browser_note() const {
        return dirty && !saved_id.empty() && browser_given == saved_id ? kBrowserNote : std::string();
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
            // ⌘O Open in browser, ⌘L Chat, then ⌘K Command at the right end.
            const float cw = chat ? ChatPane::toggle_width() + ImGui::GetStyle().ItemSpacing.x : 0.0f;
#ifdef __APPLE__
            const char* okey = "\xe2\x8c\x98O";   // ⌘O
#else
            const char* okey = "Ctrl O";
#endif
            const bool browser = open_in_browser && has_spec;
            const float bw = browser ? ImGui::CalcTextSize(okey).x + ImGui::CalcTextSize(kBrowserLabel).x + 32.0f +
                                           ImGui::GetStyle().ItemSpacing.x
                                     : 0.0f;
            const float right = ImGui::GetWindowContentRegionMax().x - w - cw - bw - sternbild::metric::pad;
            // The builder's items after the title, in the room the right end
            // leaves (only ⌘B Build when it is short); none for a spec
            // applet's file, which the builder does not edit.
            if (builder && has_spec && !from_file) {
                ImGui::Separator();
                builder->page_bar(right - ImGui::GetCursorPosX() - 2.0f * sternbild::metric::pad);
                ImGui::SameLine();
            }
            if (right > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(right);
            browser_rect[2] = browser_rect[3] = 0.0f;
            if (browser) {
                const std::string off = browser_disabled();
                if (sternbild::key_button(okey, kBrowserLabel, off.empty(), off.c_str())) open_browser();
                const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
                browser_rect[0] = a.x, browser_rect[1] = a.y, browser_rect[2] = b.x - a.x, browser_rect[3] = b.y - a.y;
                // Why the last try did not reach the browser, or, with unsaved
                // changes, that the browser shows the saved dashboard.
                if (off.empty() && ImGui::IsItemHovered()) {
                    if (!browser_why.empty()) ImGui::SetTooltip("%s", browser_why.c_str());
                    else if (dirty) ImGui::SetTooltip("%s", kBrowserNote);
                }
            }
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
                if (const std::string note = browser_note(); !note.empty()) sternbild::status_item(note.c_str());
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
        for (auto it = items.begin(); it != items.end();)
            it = it->first == "menu:remove" ? std::next(it) : items.erase(it);
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
                runtime->set_empty_hint({});
                runtime->draw(area, look);
                focus_and_frame();
            }
            for (const auto& lp : runtime->layout().panels)
                items["panel:" + lp.id] = ImRect(ImVec2(lp.rect.x, lp.rect.y), ImVec2(lp.rect.right(), lp.rect.bottom()));
            panel_menu();
            take_remove();
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
            remove_control(f);
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
    const uint64_t n = session.specs_opened();
    if (n == impl_->opens_taken) return false;
    const uint64_t a = session.agent_specs_opened();
    impl_->opened_by_agent = a - impl_->agent_opens_taken >= n - impl_->opens_taken;
    impl_->opens_taken = n;
    impl_->agent_opens_taken = a;
    return true;
}

bool SpecPage::opened_by_agent() const { return impl_->opened_by_agent; }

SpecPage::OpenReview SpecPage::open_review(AppSession& session, const std::string& origin) {
    cb::Session* s = session.session();
    if (!s) return OpenReview::opens;   // no session yet: no open spec to replace
    if (!s->review().on(cb::command_origin_from_name(origin).value_or(cb::CommandOrigin::api))) return OpenReview::opens;
    if (session.worker_busy()) return OpenReview::unknown;
    const std::optional<cb::OpenSpec> open = s->open_spec();
    return open && open->dirty ? OpenReview::waits : OpenReview::opens;
}

bool SpecPage::request_open(AppSession& session, const std::string& id, const std::string& origin) {
    cb::Session* s = session.session();
    if (!s) return false;
    cb::CommandInvocation inv;
    inv.name = "applet.open";
    inv.args = {{"id", id}};
    // From who asked; a name that is not an origin is a program's (api), never
    // the person's.
    inv.origin = cb::command_origin_from_name(origin).value_or(cb::CommandOrigin::api);
    impl_->request = s->queue().submit(std::move(inv));
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
    impl_->request_new_f = s->queue().submit(std::move(inv));
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

std::string SpecPage::open_title(AppSession& session) {
    cb::Session* s = session.session();
    if (!s || session.worker_busy()) return {};
    const std::optional<cb::OpenSpec> open = s->open_spec();
    if (!open) return {};
    const auto& a = open->spec.applet;
    return a.title && !a.title->empty() ? *a.title : a.id;
}

bool SpecPage::has_unsaved_spec(AppSession& session) {
    cb::Session* s = session.session();
    if (!s || session.worker_busy()) return false;
    const std::optional<cb::OpenSpec> open = s->open_spec();
    return open && open->dirty;
}

bool SpecPage::importing() const { return impl_->builder && impl_->builder->importing(); }

bool SpecPage::request_answered(std::string* error, bool* was_new) {
    Impl& im = *impl_;
    for (const bool is_new : {true, false}) {
        std::optional<std::future<cb::CommandResult>>& f = is_new ? im.request_new_f : im.request;
        if (!f || f->wait_for(std::chrono::seconds(0)) != std::future_status::ready) continue;
        const cb::CommandResult r = f->get();
        f.reset();
        if (is_new && r.ok) im.new_done = true;
        if (error) *error = r.ok ? std::string() : r.error.code + ": " + r.error.message;
        if (was_new) *was_new = is_new;
        return true;
    }
    return false;
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
    // ⌘O Open in browser, unless a text field has the keyboard.
    if (im.open_in_browser && im.has_spec && !ImGui::GetIO().WantTextInput &&
        ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_O))
        im.open_browser();
    const bool esc_key = ImGui::IsKeyPressed(ImGuiKey_Escape, false);
    if (im.widget_open() && esc_key) {
        im.widget->surface().escape();
        escape = false;
    } else if (im.command_open && esc_key) {
        im.command_open = false;
        im.command.clear();
        escape = false;
    } else if (escape && !im.runtime->selection().empty()) {
        im.runtime->clear_selection();   // the selection goes before the focus
        escape = false;
    }
    // The builder's keys (⌘B, and in build mode ⌘T, ⌘Z, ⇧⌘Z, ⌘S, ⌫), and
    // its Escape (a drag, the text editor, a source's focus) before the
    // page's (a panel's focus, then Home).
    if (im.builder && im.has_spec && !im.from_file && !im.widget_open() && im.builder->keys(escape)) escape = false;
    // ⌫ or Delete on the focused panel of the dashboard page (in build mode
    // the pane's keys take it), unless a text field, the command input or
    // the inline prompt has the keyboard.
    if (!im.building() && !im.focused.empty() && !im.command_open && !ImGui::GetIO().WantTextInput &&
        (ImGui::IsKeyPressed(ImGuiKey_Delete, false) || ImGui::IsKeyPressed(ImGuiKey_Backspace, false)) &&
        im.can_remove(im.focused)) {
        im.remove_request = im.focused;
        im.take_remove();
    }
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
            } else if (im.building() && im.builder->open_file_prompt()) {
                // ⌘K on the row of a file just dropped, selected in Sources.
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

void SpecPage::set_api_files(ApiFiles files) { impl_->api_files = std::move(files); }

void SpecPage::set_chat(ChatPane* chat) { impl_->chat = chat; }

void SpecPage::set_agent_settings(AgentSettings* settings) {
    Impl& im = *impl_;
    if (im.settings) im.settings->set_on_change(nullptr);
    im.settings = settings;
    if (settings) settings->set_on_change([&im] { im.retire_widget(); });
}

void SpecPage::set_open_in_browser(std::function<std::string(const std::string&)> open) {
    impl_->open_in_browser = std::move(open);
}

void SpecPage::set_dim_names(std::function<std::vector<std::string>(const std::string&)> names) {
    impl_->dim_names = std::move(names);
}

std::string SpecPage::browser_note() const { return impl_->browser_note(); }

bool SpecPage::item_rect(const std::string& name, float rect[4]) const {
    const Impl& im = *impl_;
    if (const auto it = im.items.find(name); it != im.items.end()) {
        rect[0] = it->second.Min.x, rect[1] = it->second.Min.y;
        rect[2] = it->second.GetWidth(), rect[3] = it->second.GetHeight();
        return true;
    }
    if (name != "page:browser" || im.browser_rect[2] <= 0.0f) return false;
    for (int i = 0; i < 4; ++i) rect[i] = im.browser_rect[i];
    return true;
}

bool SpecPage::widget_action(const std::string& label) {
    if (!impl_->widget) return false;
    WidgetSurface& s = impl_->widget->surface();
    std::vector<WidgetAction> offered = s.row_actions();
    for (WidgetAction a : s.card_actions()) offered.push_back(a);
    for (WidgetAction a : offered) {
        if (label != widget_action_label(a)) continue;
        s.act(a);
        return true;
    }
    return false;
}

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

std::string SpecPage::widget_facts() const {
    if (!impl_->widget) return {};
    const WidgetSurface& w = impl_->widget->surface();
    auto one_line = [](std::string t, size_t most) {
        for (char& c : t)
            if (c == '\n' || c == '\r') c = ' ';
        if (t.size() > most) t = t.substr(0, most) + "...";
        return t;
    };
    std::string out = "tag " + (w.tag().empty() ? std::string("none") : w.tag()) + ", " + std::to_string(w.edits()) +
                      (w.edits() == 1 ? " edit" : " edits");
    if (!w.model_label().empty()) out += ", model " + w.model_label();
    if (w.error()) out += ", error " + w.error().code + ": " + one_line(w.error().message, 200);
    if (!w.turn_input_tokens().empty()) {
        out += ", tokens in";
        for (long long t : w.turn_input_tokens()) out += " " + std::to_string(t);
        out += ", out " + std::to_string(w.turn_output_tokens());
    }
    if (!w.reply().empty()) out += ", reply " + one_line(w.reply(), 200);
    return out;
}

std::string SpecPage::selection_json() const {
    const SpecSelection& sel = impl_->runtime->selection();
    if (sel.empty()) return {};
    return nlohmann::json{{"panel", sel.panel},
                          {"kind", sel.kind == SpecSelection::Kind::points ? "points" : "range"},
                          {"x0", sel.x0},
                          {"x1", sel.x1},
                          {"y0", sel.y0},
                          {"y1", sel.y1}}
        .dump();
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
    impl_->drop_widget();
    if (impl_->settings) impl_->settings->set_on_change(nullptr);
    impl_->settings = nullptr;
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
bool SpecPage::opened_by_agent() const { return false; }
std::string SpecPage::browser_note() const { return {}; }
bool SpecPage::request_open(AppSession&, const std::string&, const std::string&) { return false; }
SpecPage::OpenReview SpecPage::open_review(AppSession&, const std::string&) { return OpenReview::opens; }
bool SpecPage::request_answered(std::string*, bool*) { return false; }
bool SpecPage::request_new(AppSession&, const std::string&) { return false; }
bool SpecPage::take_new() { return false; }
bool SpecPage::has_open_spec(AppSession&) { return false; }
std::string SpecPage::open_title(AppSession&) { return {}; }
bool SpecPage::has_unsaved_spec(AppSession&) { return false; }
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
void SpecPage::set_api_files(ApiFiles) {}
void SpecPage::set_chat(ChatPane*) {}
void SpecPage::set_agent_settings(AgentSettings*) {}
bool SpecPage::widget_action(const std::string&) { return false; }
void SpecPage::set_open_in_browser(std::function<std::string(const std::string&)>) {}
void SpecPage::set_dim_names(std::function<std::vector<std::string>(const std::string&)>) {}
bool SpecPage::item_rect(const std::string&, float*) const { return false; }
bool SpecPage::panel_rect(const std::string&, float*) const { return false; }
std::string SpecPage::widget_state() const { return "idle"; }
std::string SpecPage::widget_facts() const { return {}; }
std::string SpecPage::selection_json() const { return {}; }
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
