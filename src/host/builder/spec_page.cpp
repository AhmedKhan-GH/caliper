// The spec page of the app and of `caliper shot` (spec_page.h).
#include "builder/spec_page.h"

#include "builder/app_session.h"

#if CALIPER_WITH_BUILDER
#include "builder/panel_renderer.h"
#include "applet_spec.h"
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_runtime.h"
#include "ui/applet_page.h"      // begin_page_bar, JobsTray
#include "ui/landing_screen.h"   // draw_host_status_items
#include "ui/sternbild_theme.h"
#include "ui/work_area.h"

#include <caliper_builder/commands/command.h>
#include <caliper_builder/session/session.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <future>
#include <map>
#include <optional>
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

    Seams seams;
    std::string focused;   // the focused panel's id, "" for none

    // The session the page reads: a new one (the workspace opened, or the
    // session closed) gets a new source.
    void bind(AppSession& app) {
        if (workspace.empty()) workspace = folder_name(app.workspace());
        cb::Session* s = app.session();
        if (s == bound) return;
        bound = s;
        source = s ? std::make_unique<PageSource>(*s, app) : nullptr;
        runtime->set_source(source.get());
        journal_read.reset();
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

    bool draw_bars(const char* renderer_name, JobsTray* tray) {
        bool home = false;
        // The page bar, as the applet page's: Home as a key button holding
        // its key, then the spec's title where an applet's name goes.
        if (begin_page_bar()) {
            if (sternbild::key_button("esc", "Home")) home = true;
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
            ImGui::TextUnformatted(has_spec ? title.c_str() : "");
            ImGui::PopStyleColor();
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
        // The jobs tray's strip above the status bar while it is open.
        if (tray) tray->draw();
        return home;
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
            runtime->draw(area, house_spec_look());
            focus_and_frame();
        }
        ImGui::End();
    }

    // A click inside a shown panel focuses it, and the focused panel gets a
    // bronze frame, the palette's colour for focus. A panel that is no longer
    // shown loses the focus.
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
        for (const auto& lp : layout.panels) {
            if (lp.id != focused) continue;
            ImGui::GetWindowDrawList()->AddRect(ImVec2(lp.rect.x, lp.rect.y),
                                                ImVec2(lp.rect.right(), lp.rect.bottom()), sternbild::col::bronze);
        }
    }
};

SpecPage::SpecPage() : impl_(std::make_unique<Impl>()) {}

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
    impl_->request_error.clear();
    return true;
}

bool SpecPage::request_answered(std::string* error) {
    Impl& im = *impl_;
    if (!im.request) return false;
    if (im.request->wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
    const cb::CommandResult r = im.request->get();
    im.request.reset();
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
    r.home = im.draw_bars(renderer_name, tray);
    im.draw_spec();
    // ⌘K and Escape (spec_key_action). ImGuiMod_Ctrl is the Command key on
    // macOS, as on the landing screen.
    const bool command_key = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_K);
    switch (spec_key_action(command_key, escape, !im.focused.empty())) {
        case SpecKeyAction::panel_prompt:
            if (im.seams.panel_prompt) im.seams.panel_prompt(im.focused);
            break;
        case SpecKeyAction::page_command:
            if (im.seams.page_command) im.seams.page_command();
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

void SpecPage::set_seams(Seams seams) { impl_->seams = std::move(seams); }

const std::string& SpecPage::focused_panel() const { return impl_->focused; }

void SpecPage::focus_panel(const std::string& panel_id) { impl_->focused = panel_id; }

void SpecPage::leave() {
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
void SpecPage::enter(AppSession&) {}
SpecPage::Result SpecPage::draw(AppSession&, const char*, JobsTray*, bool) { return {}; }
void SpecPage::set_seams(Seams) {}
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
