#include <iostream>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

#include "ui/applet_page.h"
#include "ui/dock_layout.h"
#include "ui/landing_screen.h"
#include "ui/sternbild_fonts.h"
#include "ui/sternbild_theme.h"
#include "ui/ui_recovery.h"
#include "host/applet_loader.h"
#include "host/host_cli.h"
#include "host/background_process.h"
#include "host/shot.h"
#include "host/host_services.h"
#include "host/core_lifecycle.h"
#include "host/job_system.h"
#include "host/host_version.h"
#include "host/renderer/host_renderer.h"
#include "host/embed/embed_core.h"   // caliper_live_set_event_tap
#include "host/live/live_attach.h"
#include "host/live/live_controls.h"
#include "host/live/live_ui_driver.h"
#include "host/live/live_views_scene.h"   // scene_view, for CALIPER_TEST_UI's spec page steps
#include "host/live/live_views_state.h"   // the views' header look
#include "host/live/live_scene.h"
#include "host/api/app_api.h"
#include "host/api/window_captures.h"
#include "host/open_url.h"
#include "host/builder/app_hooks.h"
#include "host/builder/app_providers.h"
#include "host/builder/approval_cards.h"
#include "host/builder/chat_pane.h"
#include "host/builder/save_prompt.h"
#include "host/builder/spec_page.h"
#if CALIPER_WITH_BUILDER
#include "host/builder/agent_settings.h"
#include "host/builder/runs_panel.h"
#include "host/builder/widget_app.h"
#include "host/runner/model_runner.h"
#include <caliper_builder/session/session.h>
#include "host/spec_runtime/spec_runtime.h"
#include "host/spec_runtime/spec_stages.h"
#include "live/live_programs.h"
#endif
#include "host/builder/spec_ui_driver.h"
#include "app_paths.h"

#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <system_error>

// The hidden window (--hidden, CALIPER_HIDDEN=1): SIGINT and SIGTERM ask for
// the normal close of the window, which nobody can click, so the teardown runs
// and the API file is removed.
static volatile std::sig_atomic_t g_hidden_stop = 0;
extern "C" void caliper_hidden_on_signal(int) { g_hidden_stop = 1; }

enum class AppPage {
    Landing,
    Applet,
    Live,     // an attached training script's views (F4 Attach to process)
    Spec,     // the builder session's open spec (host/builder/spec_page.h)
};

// The live page's run bar and selection details in the house style: key
// buttons, secondary labels, the dense booktabs table, and bronze for the
// shared selection's highlight.
static caliper_live::ControlsLook house_controls_look() {
    caliper_live::ControlsLook l;
    l.button = [](const char* label, bool enabled, const char* reason) {
        return sternbild::key_button("", label, enabled, reason);
    };
    l.label = [](const char* text) {
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
        ImGui::TextUnformatted(text);
        ImGui::PopStyleColor();
    };
    l.begin_table = [](const char* id, int columns) { return sternbild::begin_table(id, columns); };
    l.end_table = [] { sternbild::end_table(); };
    l.header_row = [] { sternbild::table_header_row(); };
    l.text = sternbild::col::parchment;
    l.secondary = sternbild::col::secondary;
    l.faint = sternbild::col::faint;
    return l;
}

// The live views' headers in the house style: the info string in mono at the
// small size, the tools in the UI face at the small size, bronze for a
// pressed tool and the live dot.
static caliper_live_views::Look house_view_look() {
    caliper_live_views::Look l;
    l.mono = sternbild::fonts().mono;
    l.mono_size = sternbild::font_size::ui_small;
    l.small_size = sternbild::font_size::ui_small;
    l.text = sternbild::col::parchment;
    l.faint = sternbild::col::faint;
    l.rule = sternbild::col::rule;
    l.accent = sternbild::col::bronze;
    l.plot = sternbild::col::parchment;
    return l;
}

class CaliperApp {
public:
    // `applet_dirs`: the --applets folders, searched first. `hidden`: the
    // window is never shown (--hidden, CALIPER_HIDDEN=1), a testing aid.
    explicit CaliperApp(std::vector<std::string> applet_dirs, bool hidden = false)
        : applet_dirs_(std::move(applet_dirs)), hidden_(hidden) {}

    // The builder session's workspace (app_session_options()), whose
    // applets/ folder the loader searches for spec applets. Before
    // initialize(); reading it creates nothing.
    void set_workspace(std::string dir) { workspace_dir_ = std::move(dir); }

    bool initialize() {
        // The hidden window takes the mechanism of the test binaries that open
        // hidden windows (tests/glfw_background.h): the background activation
        // policy before and after glfwInit(), and no menu bar from GLFW, so the
        // process has no Dock icon, takes no menu bar and is never activated.
        // The window is created invisible and unfocused (create_window_and_init)
        // and nothing shows it later; run() hides it again should anything do.
        if (hidden_) {
            caliper_host::use_background_activation_policy();
            glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
        }
        if (!glfwInit()) {
            std::cerr << "GLFW init failed" << std::endl;
            return false;
        }
        if (hidden_) caliper_host::use_background_activation_policy();

        // Renderer seam (docs/applets/rendering.md): the choice of the
        // platform's renderer lives in libcaliper's core_lifecycle so every
        // embedder gets the same policy. The host owns the window and starts
        // the renderer below. There is no fallback renderer: where none is
        // chosen or it cannot start, the app refuses with one message.
        std::string refusal;
        renderer_ = caliper_host::core_select_renderer(&refusal);
        if (!renderer_) {
            std::cerr << "caliper: " << refusal << std::endl;
            glfwTerminate();
            return false;
        }

        // Display asleep / headless-ish states can return NULL here — fall
        // back to a sane default window instead of crashing at startup.
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        int ww = 1440, wh = 900;
        if (hidden_) {
            // The same size on every display, for reproducible captures.
            ww = 1280;
            wh = 800;
        } else if (monitor) {
            int ax, ay, aw, ah;
            glfwGetMonitorWorkarea(monitor, &ax, &ay, &aw, &ah);
            // GLFW work areas and window sizes are both in screen coordinates.
            // Dividing by Retina scale here made the window half-sized.
            ww = (int)(aw * 0.95f);
            wh = (int)(ah * 0.95f);
        }

        // Dev hook: CALIPER_WINDOW_SIZE=<W>x<H> (e.g. 1440x900, screen
        // coordinates) overrides the 95%-of-work-area size, so screenshots and
        // layout checks are reproducible on any display. Malformed values are
        // ignored and the default stands.
        if (const char* ws = std::getenv("CALIPER_WINDOW_SIZE")) {
            int w = 0, h = 0;
            char tail = 0;
            if (std::sscanf(ws, "%dx%d%c", &w, &h, &tail) == 2 &&
                w > 0 && h > 0 && w <= 16384 && h <= 16384) {
                ww = w;
                wh = h;
            }
        }

        // Applet-canvas ImGui/ImPlot/ImPlot3D contexts + docking flag are
        // created by libcaliper (core_lifecycle) — they must exist before the
        // renderer wires its ImGui backend. Host chrome styling follows.
        caliper_host::core_create_ui_context();

        // imgui.ini lives in the app-data folder, not the working directory:
        // the landing panes and the applet dock layout persist per user, and
        // a launch from Finder (cwd "/") or a build dir no longer drops ini
        // files around. IniFilename is a raw pointer read until shutdown, so
        // the path string is static.
        static const std::string ini = caliper::app_data_path("imgui.ini");
        // The hidden window reads and writes no layout: every run is a first
        // run's, and the user's saved layout is left alone.
        ImGui::GetIO().IniFilename = hidden_ ? nullptr : ini.c_str();

        // Sternbild house style. Faces first: apply_style() reads the UI font
        // size, and the first NewFrame() fixes style.FontSizeBase from the
        // default font. The 1.92 dynamic atlas takes fonts before the backend
        // exists, so this precedes renderer_->init().
        sternbild::load_fonts();
        sternbild::apply_style();

        if (!create_window_and_init(ww, wh)) {
            std::cerr << "caliper: "
                      << caliper_host::renderer_start_refusal(
                             renderer_->name(), window_ ? renderer_->start_error()
                                                        : std::string("GLFW could not create a window"))
                      << std::endl;
            glfwTerminate();
            return false;
        }
        std::cerr << "[renderer] " << renderer_->name() << std::endl;
        landing_.log("renderer", renderer_->name());
        if (hidden_) {
            int fw = 0, fh = 0;
            glfwGetFramebufferSize(window_, &fw, &fh);
            std::cerr << "[hidden] the window is not shown (visible "
                      << glfwGetWindowAttrib(window_, GLFW_VISIBLE) << ", focused "
                      << glfwGetWindowAttrib(window_, GLFW_FOCUSED) << ", background policy "
                      << caliper_host::background_activation_policy_active() << ", framebuffer " << fw
                      << "x" << fh << ")" << std::endl;
        }
        caliper_host::services_init();
        // caliper.ui.v1_1 dock hints go to the applet page's layout.
        caliper_host::set_dock_hint_sink(
            [](void* layout, const char* title, uint32_t role) {
                static_cast<caliper_host::DockLayout*>(layout)->hint(title, role);
            },
            &page_view_.dock_layout());
        // Hand the live renderer to the bridge (caliper.tensor_bridge.v1); it is
        // cleared before renderer teardown in cleanup(). Do this right after the
        // renderer is up so the first applet frame can vend textures.
        caliper_host::services_set_renderer(renderer_.get());

        // Scan for applets in the search order: the development folders
        // (--applets, then CALIPER_APPLET_PATH), the workspace's spec
        // applets, the user applets folder, then the applets shipped next to
        // the executable. Of several copies of one id the loader keeps the
        // highest version that passes. The workspace's folder is only read,
        // so a workspace that is not open yet stays uncreated.
        loader_.set_workspace(workspace_dir_);
        for (const std::string& dir : caliper_host::applet_search_dirs(
                 applet_dirs_, std::getenv("CALIPER_APPLET_PATH"),
                 caliper::user_applets_dir(), caliper::shipped_applets_dir(), workspace_dir_))
            log_scan(dir, loader_.scan(dir));

        // Dev hook: CALIPER_AUTOLAUNCH=<manifest id> skips the landing page
        // and launches that applet immediately — reproduces a row launch for
        // headless debugging/CI and seeds the future `caliper dev` workflow.
        if (const char* want_id = std::getenv("CALIPER_AUTOLAUNCH")) {
            const int i = loader_.find(want_id);
            if (i >= 0) {
                std::cerr << "[autolaunch] " << want_id << std::endl;
                landing_.log("autolaunch", want_id);
                launch_applet(i);
            }
        }

        // CALIPER_CHROMELESS=1, for recordings: the applet page draws
        // neither its page bar nor its status bar, so the dock space and the
        // applet's windows fill the whole window. Esc still goes Home; the
        // landing and the live page keep their bars.
        if (const char* c = std::getenv("CALIPER_CHROMELESS"))
            chromeless_ = std::strcmp(c, "1") == 0;

        // Dev hook: CALIPER_AUTOATTACH=<pid> attaches to that process's live
        // session as soon as the landing lists it (F4, then Enter on its row),
        // for screenshots and headless checks of the live page. Once: a
        // session that cannot attach is left alone.
        if (const char* aa = std::getenv("CALIPER_AUTOATTACH"))
            autoattach_pid_ = std::atoi(aa);

        // Testing aid: CALIPER_TEST_UI=<script> plays scripted mouse and key
        // input into the live page (live/live_ui_driver.h), only while the
        // window is hidden, so a test clicks the real widgets and nothing
        // appears on the screen.
        if (const char* ui = std::getenv("CALIPER_TEST_UI")) {
            if (hidden_ && *ui) {
                ui_driver_ = std::make_unique<caliper_live::UiDriver>(ui);
                give_ui_driver_spec_hooks();
            }
            else if (*ui) std::cerr << "[ui] CALIPER_TEST_UI needs --hidden; ignored" << std::endl;
        }
        // The same for the spec page (builder/spec_ui_driver.h), whose shots
        // go to CALIPER_TEST_SPEC_UI_SHOTS.
        if (const char* su = std::getenv("CALIPER_TEST_SPEC_UI")) {
            const char* dir = std::getenv("CALIPER_TEST_SPEC_UI_SHOTS");
            if (hidden_ && *su) spec_ui_ = std::make_unique<caliper_host::SpecUiDriver>(su, dir ? dir : ".");
            else if (*su) std::cerr << "[ui] CALIPER_TEST_SPEC_UI needs --hidden; ignored" << std::endl;
        }

        return true;
    }

    void run() {
        // Dev hook: CALIPER_EXIT_AFTER=<seconds> requests a NORMAL window
        // close after N seconds — exercising the full teardown path
        // (on_cleanup, dlclose policy, renderer/context destruction) that
        // aliveness kill-checks skip. Clean-exit regressions hide there.
        double exit_after = 0.0;
        if (const char* ea = std::getenv("CALIPER_EXIT_AFTER"))
            exit_after = std::atof(ea);
        const double t0 = glfwGetTime();

        while (!glfwWindowShouldClose(window_)) {
            if (exit_after > 0.0 && glfwGetTime() - t0 >= exit_after)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            if (hidden_) {
                if (g_hidden_stop) glfwSetWindowShouldClose(window_, GLFW_TRUE);
                // A safety net: nothing in the app shows the window, but a
                // window that became visible is hidden again at once.
                if (glfwGetWindowAttrib(window_, GLFW_VISIBLE)) {
                    glfwHideWindow(window_);
                    std::cerr << "[hidden] the window became visible and was hidden again" << std::endl;
                }
            }
            if (page_ == AppPage::Live) wait_for_live_frame();
            glfwPollEvents();

            // An attached session: claim and apply the newest completed
            // snapshot of every view before the frame draws them (the live
            // core's order: poll, then frame). Never waits on the script. The
            // attachment lasts across every page until Detach, so it is
            // polled on every page, for the catalog's live sources.
            if (live_.attached()) live_.poll();
            if (autoattach_pid_ > 0 && page_ == AppPage::Landing) autoattach();
            note_live_events();
            note_control_lines();

            // Caliper API (--api) and the builder session's commands: the
            // calls that read or change the app run here, between frames, as
            // CALIPER_AUTOLAUNCH launches. A capture arms the read-back of the
            // frame drawn next (window_captures.h).
            if (api_.running()) serve_api();
            session_.frame();
#if CALIPER_WITH_BUILDER
            frame_runner();
            // GPU programs (design/gpu-programs.md, Attachment): the attached
            // script's programs follow the open spec's, and its PROGRAM_STATE
            // becomes program.status and the program events.
            // The session also gives the params' values and the operators' input shapes.
            program_host_.pump(program_link_, session_.session());
#endif
            save_prompt_.frame(session_);   // "Save changes to <name>?" and what waits on it
            approvals_.frame(session_.session(), session_.worker_busy(), glfwGetTime());
            chat_.frame(session_);   // the side chat polls its agent
            note_spec_applets();
            note_drops();
            captures_.before_frame(*renderer_, [this](uint64_t t, int status, const std::string& why) {
                capture_error(t, status, why);
            });

            int dw, dh;
            glfwGetFramebufferSize(window_, &dw, &dh);

            // CALIPER_TEST_UI with --hidden: the scripted clicks of a UI test,
            // queued into ImGui before the frame reads its input.
            if (ui_driver_ && (page_ == AppPage::Live || (page_ == AppPage::Spec && ui_driver_->spec_step())))
                feed_ui_driver();
            if (spec_ui_ && (page_ == AppPage::Spec || page_ == AppPage::Live || page_ == AppPage::Landing ||
                             page_ == AppPage::Applet))
                feed_spec_ui();

            renderer_->new_frame();

            // Jobs tray (§7.5), on every page while a job runs or a failure
            // is kept (ui/applet_page.h): this frame's jobs and whether its
            // strip is open. Each page draws the strip between its status
            // bar and its dock space, so the dock space ends above it.
            // Failures are kept until cleared, see note_failed_jobs.
            {
                auto views = caliper_host::host_job_system().views();
                note_failed_jobs(views);
                jobs_tray_.begin_frame(std::move(views), glfwGetTime());
            }

            if (page_ == AppPage::Landing) {
                // The workstation home, identical on every backend. It reads
                // loader_ live each frame, so statuses are always fresh.
                LandingScreen::Context ctx;
                ctx.loader = &loader_;
                ctx.renderer_name = renderer_->name();
                ctx.launch = [this](int i) { launch_from_landing(i); };
                ctx.restart = [this](int i) { restart_applet(i); };
                ctx.sessions = &sessions_;
                ctx.attach = [this](const std::string& s) { attach_session(s); };
                ctx.after_bars = [this] { jobs_tray_.draw(); };
                ctx.attach_url = [this](const std::string& u) { attach_stream(u); };
                // The attached session, which the command bar shows with F5
                // Live page, the way back to it.
                LandingScreen::LiveSession live_row;
                if (live_.attached()) {
                    live_row.session = live_.status().session;
                    live_row.script = live_script_;
                    live_row.ended = !live_.status().alive && !live_.status().pending;
                    ctx.live = &live_row;
                    ctx.show_live = [this] { show_live_page(); };
                }
                // F6 Dashboard: the session's open spec, the way back to its
                // page, as F5 is to the live page.
                ctx.open_spec = open_spec_title();
                if (!ctx.open_spec.empty()) ctx.show_spec = [this] { open_spec_page(); };
                ctx.spec_facts = [](const caliper_host::AppletEntry& e, SpecAppletFacts* f) {
                    return caliper_host::SpecPage::facts(e.spec_path, f);
                };
                ctx.shown_path = [this](const std::string& p) { return workspace_relative(p); };
                // F3 New dashboard asks "Save changes to <name>?" first when
                // the open spec has unsaved changes.
                if (caliper_host::SpecPage::built())
                    ctx.new_spec = [this](const std::string& title) {
                        save_prompt_.ask("new dashboard", [this, title] { new_spec(title); });
                    };
#if CALIPER_WITH_BUILDER
                ctx.agent_settings = [this] { agent_settings_.set_open(true); };
#endif
                landing_.draw(ctx);
            } else if (page_ == AppPage::Live) {
                draw_live_page();
            } else if (page_ == AppPage::Spec) {
                // The open spec in the work area the strips leave; Home and
                // esc go back to the landing after the frame is rendered.
                // Escape gives a focused panel's focus up first and goes
                // Home only when no panel has it (spec_key_action).
                bool go_back = false;
                // Dev hook: CALIPER_HOME_AFTER=<sec>, as on the other pages:
                // Home once, N sec after the spec page opened.
                if (const char* ha = std::getenv("CALIPER_HOME_AFTER")) {
                    static double page_t0 = -1.0;
                    static bool fired = false;   // once: a spec page opened later stays
                    if (page_t0 < 0) page_t0 = glfwGetTime();
                    if (!fired && glfwGetTime() - page_t0 >= std::atof(ha)) go_back = fired = true;
                }
                if (spec_page_.draw(session_, renderer_->name(), &jobs_tray_, esc_home_pressed()).home || go_back)
                    pending_spec_home_ = true;
            } else if (page_ == AppPage::Applet) {
                bool go_back = esc_home_pressed();

                // Dev hook: CALIPER_HOME_AFTER=<sec> forces Home once, N sec
                // after the applet page opened — reproduces "click Home during
                // training" headlessly (for the SIGSEGV repro).
                if (const char* ha = std::getenv("CALIPER_HOME_AFTER")) {
                    static double page_t0 = -1.0;
                    static bool fired = false;   // once: an applet page opened later stays
                    if (page_t0 < 0) page_t0 = glfwGetTime();
                    if (!fired && glfwGetTime() - page_t0 >= std::atof(ha)) go_back = fired = true;
                }

                double now = glfwGetTime();
                int ww = 0, wh = 0;
                glfwGetWindowSize(window_, &ww, &wh);
                CaliperFrameInfo fi{};
                fi.struct_size = sizeof fi;
                fi.fb_width = dw; fi.fb_height = dh;              // physical px
                fi.dpi_scale = (ww > 0) ? (float)dw / (float)ww : 1.0f;
                fi.time_sec = now;
                fi.delta_sec = now - last_frame_time_;
                last_frame_time_ = now;

                // The page bar and the status bar (unless CALIPER_CHROMELESS=1),
                // the dock space, docking by role and the applet's frame
                // (ui/applet_page.h, which `caliper shot` draws too).
                const caliper_host::AppletPage::Result page = page_view_.draw(
                    loader_, active_applet_, fi, renderer_->name(), chromeless_,
                    &jobs_tray_);
                if (page.home) go_back = true;

                if (!page.alive) {            // quarantined mid-frame
                    go_back = true;
                    // A failed job that caused it goes in Activity first.
                    note_failed_jobs(caliper_host::host_job_system().views());
                    const auto& a = loader_.at(active_applet_);
                    landing_.log("quarantined", a.manifest.name, a.status_text);
                }

                // Teardown is DEFERRED to after render() (see below): the
                // applet's draw list THIS frame may reference bridge textures,
                // and releasing them in cleanup() before the Metal render pass
                // consumes that draw list is a use-after-free (objc_retain on a
                // freed MTLTexture). We only flag it here; the applet keeps
                // drawing valid textures through this frame's render.
                if (go_back) pending_home_ = active_applet_;
            }

            // "Save changes to <name>?" over whatever page is shown, while open.
            save_prompt_.draw();

#if CALIPER_WITH_BUILDER
            // The widget agent's settings, over whatever page is shown, in the
            // work area; ⌘, opens them anywhere.
            agent_settings_.read_keys();
            agent_settings_.frame();
            if (agent_settings_.is_open()) {
                const ImGuiViewport* vp = ImGui::GetMainViewport();
                agent_settings_.draw(vp->WorkPos.x, vp->WorkPos.y, vp->WorkSize.x, vp->WorkSize.y,
                                     caliper_host::house_spec_look());
            }
            // The managed runs (design/hosted-runs.md), over whatever page is
            // shown; ⌘⇧R opens them anywhere. A start or a fork shows the
            // session's card in the panel, the same card as the tray's.
            runs_panel_.read_keys();
            runs_panel_.frame(session_.session(), runner_.get(), session_.worker_busy());
            if (runs_panel_.is_open()) {
                const ImGuiViewport* vp = ImGui::GetMainViewport();
                runs_panel_.draw(vp->WorkPos.x, vp->WorkPos.y, vp->WorkSize.x, vp->WorkSize.y,
                                 caliper_host::house_spec_look());
            }
#endif

            // The session's catalog providers: registered once the session is
            // open, and the live views handed over after the live poll and
            // the page's draw, on every page, so that a detach is seen.
            providers_.frame(session_, live_);

            // Sampled after every window of the frame: next frame's Esc test
            // needs to know a popup was open before NewFrame() closed it.
            popup_open_prev_ = ImGui::IsPopupOpen(
                nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);

            renderer_->render(dw, dh);

            // The captures: a read-back that has finished on the GPU answers
            // its calls; one still copying is picked up by a later frame.
            captures_.after_frame(
                *renderer_,
                [this](uint64_t t, uint32_t w, uint32_t h, std::vector<uint8_t> rgba) {
                    if (t == caliper_host::AppSession::kCaptureTicket) session_.capture_ready(w, h, std::move(rgba));
                    else if (t == caliper_host::SpecUiDriver::kShotTicket) { if (spec_ui_) spec_ui_->write_shot(w, h, rgba); }
                    else api_.reply_capture(t, w, h, std::move(rgba));
                },
                [this](uint64_t t, int status, const std::string& why) { capture_error(t, status, why); });

            // Deferred Home and Detach (⌘D) of the live page, for the same
            // reason as the applet teardown below: the views' textures and
            // imports are released only after the frame that drew them has
            // been rendered.
            if (pending_live_home_) {
                pending_live_home_ = false;
                leave_live_page();
            }
            if (pending_detach_) {
                pending_detach_ = false;
                detach_live();
            }
            // The spec page's textures go after the frame that drew them too.
            if (pending_spec_home_) {
                pending_spec_home_ = false;
                leave_spec_page();
            }

            // Deferred applet teardown — runs ONLY after this frame's draw list
            // has been rendered, so no just-released bridge texture is still
            // referenced by the Metal pass (leave_applet_page gives the order).
            if (pending_home_ >= 0) {
                const int idx = pending_home_;
                pending_home_ = -1;
                leave_applet_page(idx);
            }
        }
    }

    // --url: the stream to attach to once the app has started.
    void attach_at_start(const std::string& link) {
        std::cerr << "[live] attaching to " << link.substr(0, link.find('?')) << std::endl;
        attach_stream(link);
    }

    // Caliper API (--api): listens until cleanup(). A failure to start is
    // said on stderr and in Activity, and the app runs without it.
    void start_api(const caliper_host::CommandLine& cl) {
        std::string why;
        // The command endpoints serve the builder session's command layer; a
        // command opens the default workspace on first use. Set before the
        // server listens, so that no request finds it without one.
        api_.set_command_gateway(session_.command_gateway());
        // A capture reads the window's frame back where the renderer can
        // (Metal); elsewhere it is answered 501 with the reason.
        const std::string capture_unavailable =
            renderer_->window_capture_supported()
                ? std::string()
                : caliper_host::WindowCaptures::unsupported(renderer_->name());
        if (!api_.start(cl, capture_unavailable, caliper::app_data_location(), std::cout, &why)) {
            std::cerr << "[api] not started: " << why << std::endl;
            landing_.log("api not started", why);
            api_.set_command_gateway(nullptr);
            return;
        }
        landing_.log("api", api_.url());
    }

    // The builder session (host/builder/app_session.h) for the workspace of
    // the command line. It opens nothing yet.
    void start_session(const caliper_host::CommandLine& cl) {
        cl_ = cl;
        caliper_host::AppSession::Hooks h = caliper_host::app_session_hooks();
        h.applets_json = [this] { return caliper_host::applets_json(loader_); };
        h.launch = [this](const std::string& id, const std::string& origin) { return launch_by_id(id, origin); };
        if (!renderer_->window_capture_supported())
            h.capture_unavailable = caliper_host::WindowCaptures::unsupported(renderer_->name());
        h.request_capture = [this] { captures_.add(caliper_host::AppSession::kCaptureTicket); };
        h.log = [this](const std::string& e, const std::string& o, const std::string& d) { landing_.log(e, o, d); };
        save_prompt_.set_log([this](const std::string& e, const std::string& o, const std::string& d) { landing_.log(e, o, d); });
        std::string refused;   // a command script, which only a hidden app runs
        session_.start(caliper_host::app_session_options(cl, hidden_, &refused), std::move(h));
        if (!refused.empty()) {
            std::cerr << "[session] " << refused << std::endl;
            landing_.log("command script refused", "CALIPER_COMMAND_SCRIPT", "needs --hidden");
        }
        spec_page_.set_hidden(hidden_);
        // The attached script's run (design/instruments.md, Run Control in
        // App): run.state and the run commands act through the attachment,
        // and the spec page's scene panels draw its live scenes.
        session_.set_run_link(&run_link_);
#if CALIPER_WITH_BUILDER
        // The hosts of GPU programs: the programs run in the attached script,
        // the stages in the spec runtime's scene panels.
        session_.set_program_host(&program_host_);
        session_.set_stage_host(caliper_host::app_stage_host());
        // The model runner of Hosted Runs (design/hosted-runs.md): its runs
        // live in the workspace's runs folder, and run.list, run.start,
        // run.fork, run.stop and run.logs act through it.
        {
            caliper_runner::ModelRunner::Options ro;
            std::error_code ec;
            ro.workspace = std::filesystem::weakly_canonical(std::filesystem::path(session_.workspace()), ec).string();
            if (ec || ro.workspace.empty()) ro.workspace = session_.workspace();
            runner_ = std::make_unique<caliper_runner::ModelRunner>(ro);
            runner_->set_app_control(session_.run_control());
            session_.set_model_runner(runner_.get());
        }
#endif
        spec_page_.set_live_core([this] { return live_.core(); });
        // panel.render draws on an offscreen canvas that shares this
        // window's font atlas, made on the first call between two frames.
        session_.set_panel_renderer(caliper_host::SpecPage::make_panel_renderer([this] { return session_.session(); }));
        // The approval cards: a strip above the jobs tray on every page, and
        // a mark on the spec page's panel that a waiting edit changes. A
        // test decides through CALIPER_TEST_APPROVALS, read only when hidden.
        caliper_host::ApprovalTray::Hooks ah;
        ah.log = [this](const std::string& e, const std::string& o, const std::string& d) { landing_.log(e, o, d); };
        if (const char* d = std::getenv("CALIPER_TEST_APPROVALS"); d && hidden_) ah.decisions = d;
        approvals_.start(std::move(ah));
        jobs_tray_.set_next_strip([this] { approvals_.draw_strip(); });
        spec_page_.set_panel_overlay([this](const std::string& id, const std::string& title, float x, float y, float w,
                                            float h, float strip) { approvals_.mark_panel(id, title, x, y, w, h, strip); });
        // The side chat (host/builder/chat_pane.h) on the spec page and the
        // live page. Its agent's caliper-mcp needs Caliper API: when the app
        // runs without --api, the chat starts it at its first message, on a
        // free loopback port with a new token and the API file.
        caliper_host::ChatPane::Options co;
        co.settings_path = caliper::app_data_path("side-chat.json");
        co.exe_dir = caliper::executable_dir();
        caliper_host::ChatPane::Hooks chh;
        // One of the two cases where the API starts without --api: the side
        // chat starts it for its agent's caliper-mcp, on 127.0.0.1 at a free
        // port (the server binds loopback only, and no setting chooses
        // another address), and stops it when the chat's agent ends, unless
        // Open in browser uses it too (api_for()).
        chh.start_api = [this](std::string* why, std::string* started) -> std::string {
            if (!api_for(ApiUser::chat, started)) {
                *why = "see Activity";
                return {};
            }
            // The agent's own API file, with a token of the origin agent for
            // this session: never the person's token, nor the API file's.
            return api_.issue_agent_file(why, "chat");
        };
        chh.end_api_file = [this](const std::string& file) { api_.revoke_agent_file(file); };
        chh.agent_ended = [this] {
            api_for_chat_ = false;
            if (!api_started_here_ || api_for_browser_ || !api_.running()) return;
            const std::string address = api_address();
            api_.stop();
            api_started_here_ = false;
            landing_.log("api stopped", address, "the side chat's agent ended");
            std::cerr << "[chat] Caliper API stopped on " << address << ": the side chat's agent ended" << std::endl;
        };
        chh.show_approval = [this](const std::string& id) { approvals_.point_to(id); };
        // The header's GPU note: a live run is attached.
        chh.live_attached = [this] { return live_.attached(); };
        chh.log = [this](const std::string& e, const std::string& o, const std::string& d) { landing_.log(e, o, d); };
        chat_.start(std::move(co), std::move(chh));
        // Every page's status bar shows the API's address while it runs.
        set_host_status_note([this] {
            std::string note = api_.running() ? "api " + api_address() : std::string();
            if (page_ == AppPage::Applet && !agent_spec_note_.empty())
                note += (note.empty() ? "" : "  \xc2\xb7  ") + ("dashboard " + agent_spec_note_ + " open");
            return note;
        });
        spec_page_.set_chat(&chat_);
        // ⌘O Open in browser on the spec page, and the names of a live view's
        // dims for the builder's Properties.
        spec_page_.set_open_in_browser([this](const std::string& applet) { return open_in_browser(applet); });
        spec_page_.set_dim_names([this](const std::string& id) { return providers_.dim_names(id); });
        // ⌘K's sidecar gets an API file of its own, with an agent token.
        caliper_host::SpecPage::ApiFiles af;
        af.issue = [this]() -> std::string {
            std::string why;
            return api_.running() ? api_.issue_agent_file(&why, "widget") : std::string();
        };
        af.revoke = [this](const std::string& file) { api_.revoke_agent_file(file); };
        spec_page_.set_api_files(std::move(af));
#if CALIPER_WITH_BUILDER
        // The widget agent's settings (host/builder/agent_settings.h): a popup
        // from ⌘K's Setup and the landing's ⌘, Agent, kept in widget-agent.json
        // in the app-data folder. A hidden app's key store is an in-memory one.
        {
            caliper_host::AgentSettings::Options ao;
            ao.settings_path = caliper::app_data_path("widget-agent.json");
            ao.cache_dir = caliper::app_data_dir();
            const caliper_host::AppWidgetEnv we = caliper_host::app_widget_env(hidden_, std::string());
            ao.python = we.python;
            ao.source = we.source;
            ao.keys = caliper_host::app_key_store(hidden_);
            ao.live_attached = [this] { return live_.attached(); };
            ao.log = [this](const std::string& e, const std::string& o, const std::string& d) { landing_.log(e, o, d); };
            agent_settings_.start(std::move(ao));
            spec_page_.set_agent_settings(&agent_settings_);
            // Its Permissions section: the modes of the side chat's agent and
            // of mcp clients, and the saved rules, through the session as the
            // person's own UI (design/command-layer.md, Agent permissions).
            namespace cb = caliper_builder;
            caliper_host::AgentPermissionHooks ph;
            auto who_of = [](const std::string& who) {
                return who == "mcp" ? cb::AgentPrincipal::mcp : cb::AgentPrincipal::chat;
            };
            ph.mode = [this, who_of](const std::string& who) -> std::string {
                cb::Session* s = session_.session();
                if (!s) return "ask";
                // The saved mode: Allow all is the chat's session's, above it, and not here.
                return cb::agent_mode_name(s->saved_agent_mode(who_of(who)));
            };
            ph.set_mode = [this, who_of](const std::string& who, const std::string& mode) {
                cb::Session* s = session_.session();
                const auto m = cb::agent_mode_from_name(mode);
                if (!s || !m || *m == cb::AgentMode::allow_all) return;
                const cb::Error e = s->set_agent_mode(who_of(who), *m);
                landing_.log(e ? "mode not set" : (who == "mcp" ? "mcp mode" : "side chat mode"), cb::agent_mode_label(*m),
                             e ? e.code : "");
            };
            ph.rules = [this]() {
                std::vector<std::pair<std::string, std::string>> out;
                if (cb::Session* s = session_.session())
                    for (const cb::AgentRule& r : s->saved_agent_rules()) out.push_back({r.agent, r.command});
                return out;
            };
            ph.revoke = [this](const std::string& agent, const std::string& command) {
                if (cb::Session* s = session_.session(); s && s->revoke_agent_rule({agent, command}))
                    landing_.log("rule revoked", command, agent);
            };
            agent_settings_.set_permissions(std::move(ph));
        }
#endif
    }

    // Who uses a Caliper API that the app started without --api: the side
    // chat's agent, or Open in browser.
    enum class ApiUser { chat, browser };

    // Starts the API for `user` when it does not run, on a free port of
    // 127.0.0.1 with new tokens and the API file, as --api does, and says so
    // in Activity and on standard error; `started` gets its address then.
    // An API the app started is the users' (api_for_chat_, api_for_browser_);
    // one started with --api is left alone. False when it does not run.
    bool api_for(ApiUser user, std::string* started) {
        if (!start_api_here(user, started)) return false;
        if (api_started_here_) (user == ApiUser::chat ? api_for_chat_ : api_for_browser_) = true;
        return true;
    }

    // api_for() without taking the API: it starts it when it does not run,
    // and the caller takes it once it knows it keeps it.
    bool start_api_here(ApiUser user, std::string* started) {
        if (!api_.running()) {
            caliper_host::CommandLine c = cl_;
            c.api_port = 0;
            start_api(c);
            if (api_.running()) {
                api_started_here_ = true;
                *started = api_address();
                const bool chat = user == ApiUser::chat;
                landing_.log("api started", *started, chat ? "for the side chat" : "for the browser");
                std::cerr << (chat ? "[chat]" : "[browser]") << " Caliper API started on " << *started
                          << (chat ? " for the side chat" : " for the browser") << std::endl;
            }
        }
        return api_.running();
    }

    // ⌘O Open in browser on the spec page: the saved dashboard `applet` (its
    // workspace id) in the person's default browser, at the spec page of
    // Caliper API with the viewer token. Without --api the API starts for
    // it as for the side chat, and stays until the app quits, since a page
    // open in a browser keeps reading. A hidden app opens no browser: it
    // writes the link to the file CALIPER_TEST_OPEN_URL names, and without
    // that refuses. Whatever can refuse the link is asked before the API
    // starts, and an API that this request alone started stops again when
    // no link is handed over. "" when the link went out, or why not. The
    // token is never written to Activity or standard error.
    std::string open_in_browser(const std::string& applet) {
        const char* hook = hidden_ ? std::getenv("CALIPER_TEST_OPEN_URL") : nullptr;
        if (hidden_ && (!hook || !*hook)) {
            std::cerr << "[browser] not opened: the app runs hidden, and CALIPER_TEST_OPEN_URL names no file for the link"
                      << std::endl;
            landing_.log("browser refused", applet, "the app runs hidden");
            return "the app runs hidden";
        }
        std::string started, why;
        if (!caliper_host::AppApi::viewer_link_possible(applet, &why)) {
            landing_.log("browser refused", applet, why);
            return why;
        }
        const bool was_running = api_.running();
        if (!start_api_here(ApiUser::browser, &started)) return "Caliper API did not start; see Activity";
        const auto refuse = [&](const std::string& reason) {
            landing_.log("browser refused", applet, reason);
            if (!was_running && api_started_here_ && !api_for_chat_ && !api_for_browser_ && api_.running()) {
                const std::string address = api_address();
                api_.stop();
                api_started_here_ = false;
                landing_.log("api stopped", address, "no link was handed over");
                std::cerr << "[browser] Caliper API stopped on " << address << ": no link was handed over" << std::endl;
            }
            return reason;
        };
        const std::string url = api_.viewer_link(applet, &why);
        if (url.empty()) return refuse(why);
        if (hook) {
            std::ofstream out(hook, std::ios::app);
            if (out) out << url << "\n";
            out.close();
            if (!out) return refuse("the test hook's file cannot be written");
            std::cerr << "[browser] the link of " << applet << " on " << api_address() << " went to CALIPER_TEST_OPEN_URL"
                      << std::endl;
        } else if (!caliper_host::open_in_default_browser(url, &why)) {
            return refuse(why);
        } else {
            std::cerr << "[browser] opened " << applet << " on " << api_address() << std::endl;
        }
        // A link is out: the API is the browser's too, until the app quits.
        if (api_started_here_) api_for_browser_ = true;
        landing_.log("browser", applet, api_address() + (hook ? " (test hook)" : ""));
        return {};
    }

    void cleanup() {
        // The side chat's agent first, while the API it calls still answers.
        // Then the API: calls still waiting for a frame get 503. Then the
        // builder session: its worker, its journal and its database close
        // here, before the host's own stores.
        chat_.shutdown();
#if CALIPER_WITH_BUILDER
        agent_settings_.shutdown();
#endif
        api_.stop();
        api_.set_command_gateway(nullptr);
        session_.set_panel_renderer(nullptr);
        providers_.close(session_);   // before the session and the host's stores
#if CALIPER_WITH_BUILDER
        close_runner();   // every managed run's process ends here, by its process id
#endif
        approvals_.close();
        if (live_.core()) caliper_live_set_event_tap(live_.core(), {});
        session_.set_run_link(nullptr);   // before the attachment goes
#if CALIPER_WITH_BUILDER
        session_.set_program_host(nullptr);
        session_.set_stage_host(nullptr);
#endif
        session_.close();
        // Workers FIRST: a job may hold a pointer into applet state, so every
        // worker must be joined before close_all() destroys that state (the
        // 139-at-exit was a worker touching a freed applet mid-teardown).
        // Stores are still open here, so a finishing worker's last metrics
        // writes land safely.
        caliper_host::host_job_system().cancel_all_and_join();
        loader_.close_all();
        caliper_host::set_dock_hint_sink(nullptr, nullptr);
        // An attached session's views go while the bridge still has the
        // renderer; the loop ended after rendering the last frame that drew
        // them. Quitting while attached, on any page, is a detach.
        live_.detach();
        note_live_events();   // "detached"
        // Applets are torn down first (they may release bridge textures while the
        // renderer is still live); THEN drop the renderer from the bridge before
        // renderer teardown, so no bridge thunk touches a destroyed renderer.
        caliper_host::services_set_renderer(nullptr);
        // Close the DuckDB stores NOW — leaving them to static destructors
        // races DuckDB's globals and aborts in malloc at exit.
        caliper_host::services_shutdown();
        renderer_->shutdown();
        // The spec page's textures were the renderer's to destroy; their
        // records go while the UI context still exists.
        spec_page_.shutdown();
        caliper_host::core_destroy_ui_context();
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }

private:
    // The applet page's teardown, run only after the frame that drew the
    // applet has been rendered. Order: hard-join workers (they reference
    // applet state and the applet's own cleanup wait is a timeout, not a
    // join), tear the applet down, then return to the landing page (it reads
    // statuses live, so nothing to refresh).
    void leave_applet_page(int idx) {
        const std::string left = loader_.at(idx).manifest.name;
        // Joining drops every job entry, failed ones included, so a
        // failure since the tray last looked is recorded first.
        note_failed_jobs(caliper_host::host_job_system().views());
        caliper_host::host_job_system().cancel_all_and_join();
        loader_.teardown(idx);
        active_applet_ = -1;
        agent_spec_note_.clear();
        page_ = AppPage::Landing;
        glfwSetWindowTitle(window_, "Caliper");
        landing_.log("home", left);
    }

    // Home from the live page, after the frame that drew its views. The
    // session stays attached, its views polled for the catalog's live
    // sources, until the person detaches it (⌘D Detach on the live page) or
    // attaches another; F5 Live page on the landing comes back to it.
    void leave_live_page() {
        page_ = AppPage::Landing;
        glfwSetWindowTitle(window_, "Caliper");
        landing_.log("home", live_script_);
    }

    // F5 Live page on the landing (or Enter on the attached session's row of
    // F4): the live page of the attached session, which never detached.
    void show_live_page() {
        if (!live_.attached() || page_ != AppPage::Landing) return;
        live_frame_t_ = 0.0;
        page_ = AppPage::Live;
        set_live_title();
    }

    // ⌘D Detach on the live page, after the frame that drew its views: the
    // only way the person ends an attachment, apart from attaching another
    // session. The live page goes Home; the spec's live sources go offline.
    void detach_live() {
        live_.detach();
        note_live_events();   // "detached"
        if (page_ == AppPage::Live) {
            page_ = AppPage::Landing;
            glfwSetWindowTitle(window_, "Caliper");
        }
    }

    // The spec page, for an applet.open that succeeded: an applet's page goes
    // first, as Home takes it, between two frames. An attached live session
    // stays attached on every page, so that the spec's live sources keep
    // their data. Opening another spec applet while the page shows only
    // changes what it shows.
    void open_spec_page() {
        if (page_ == AppPage::Applet) leave_applet_page(active_applet_);
        spec_page_.enter(session_);   // reads the open spec
        if (page_ != AppPage::Spec) landing_.log("open", spec_page_.title(), "spec");
        page_ = AppPage::Spec;
        glfwSetWindowTitle(window_, ("Caliper - " + spec_page_.title()).c_str());
        // The workspace's spec applet that the page shows is ACTIVE, also
        // when it already was (an agent's open of the dashboard on screen).
        mark_spec_entry(workspace_spec_entry(spec_page_.applet_id()));
        agent_spec_note_.clear();
    }

    // Home from the spec page, after the frame that drew it. An attached
    // live session stays attached.
    void leave_spec_page() {
        const std::string left = spec_page_.title();
        spec_page_.leave();
        mark_spec_entry(-1);
        page_ = AppPage::Landing;
        glfwSetWindowTitle(window_, "Caliper");
        landing_.log("home", left);
    }

    // The spec applet the spec page shows is ACTIVE while it shows it, and
    // READY again after: `idx`, or none for -1. Home returns to its row.
    void mark_spec_entry(int idx) {
        for (int i = 0; i < loader_.count(); i++) {
            caliper_host::AppletEntry& e = loader_.at(i);
            if (e.manifest.kind == caliper_host::AppletKind::Spec && e.status == caliper_host::AppletStatus::Active)
                e.status = caliper_host::AppletStatus::Ready;
        }
        spec_entry_.clear();
        if (idx < 0 || idx >= loader_.count()) return;
        caliper_host::AppletEntry& e = loader_.at(idx);
        e.status = caliper_host::AppletStatus::Active;
        e.status_text.clear();
        spec_entry_ = e.manifest.id;
        landing_.select(idx);
    }

    // A launch from the landing screen (a row, Enter, ⏎ Launch, the command
    // line): a dashboard of the workspace replaces the open spec, so it asks
    // "Save changes to <name>?" first when the open spec has unsaved changes,
    // and the open spec's own dashboard shows its page, unsaved changes and
    // all, instead of reading its file again. Other applets launch at once.
    // app.launch and the API's launch never come here: an agent's request is
    // not the person's.
    void launch_from_landing(int idx) {
        if (idx < 0 || idx >= loader_.count()) return;
        const caliper_host::AppletEntry& a = loader_.at(idx);
        const std::string ws_id = a.manifest.kind == caliper_host::AppletKind::Spec ? workspace_applet_id(a) : "";
        if (ws_id.empty()) {
            launch_applet(idx);
            return;
        }
        save_prompt_.ask(
            "open " + a.manifest.name,
            [this, id = a.manifest.id] {
                const int i = loader_.find(id);
                if (i >= 0) launch_applet(i);
            },
            ws_id, [this] { open_spec_page(); });
    }

    // The window's close request (its close button, ⌘Q, the input driver's
    // `close`), which GLFW makes after it has set the close flag: with unsaved
    // changes in the open spec it asks first, and the window closes after
    // Save or Don't Save. While the session's worker runs a command the open
    // spec cannot be read, so the close waits for the command, saying so with
    // Cancel, and then asks if there is anything to save (save_prompt.h,
    // hold_close). A hidden app's SIGTERM, CALIPER_EXIT_AFTER and the
    // driver's `quit` set the flag without a request and never ask.
    void on_close_request() {
        if (quit_confirmed_) return;
        if (save_prompt_.hold_close(session_, [this] {
                quit_confirmed_ = true;
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            }))
            glfwSetWindowShouldClose(window_, GLFW_FALSE);
    }

    // The workspace id of a spec applet found in the workspace's applets/
    // folder (its folder's name, which applet.open takes), or "" for one
    // found elsewhere.
    std::string workspace_applet_id(const caliper_host::AppletEntry& a) const {
        if (workspace_dir_.empty() || a.folder.empty()) return {};
        const std::filesystem::path folder(a.folder);
        std::error_code ec;
        if (!std::filesystem::equivalent(folder.parent_path(), std::filesystem::path(workspace_dir_) / "applets", ec))
            return {};
        return folder.filename().string();
    }

    // A launch of a spec applet (a row, Enter, CALIPER_AUTOLAUNCH, app.launch,
    // POST /v0/applets/{id}/launch): its spec is checked, then one in the
    // workspace opens through the session's applet.open, from the origin of
    // who asked (`origin`: "ui" for the person's own launch, else the
    // app.launch's or the API token's, so that an agent's replacement of
    // unsaved changes waits on an approval card), as soon as the session is
    // open, and the page follows that command as for any origin; one found elsewhere is drawn from its file without
    // becoming the open spec. False, with the reason in Activity, when its
    // spec does not validate or its file does not read.
    bool launch_spec_applet(int idx, std::string* why, const std::string& origin = "ui") {
        const std::string name = loader_.at(idx).manifest.name;
        landing_.log("launch", name);
        landing_.select(idx);
        if (!loader_.check_spec(idx)) {
            *why = loader_.at(idx).status_text;
            landing_.log("launch failed", name, *why);
            return false;
        }
        session_.ensure_open();
        const std::string ws_id = workspace_applet_id(loader_.at(idx));
        if (!ws_id.empty()) {
            pending_spec_open_ = ws_id;
            pending_spec_origin_ = origin;
            return true;
        }
        if (page_ == AppPage::Applet) leave_applet_page(active_applet_);
        if (!spec_page_.enter_file(session_, loader_.at(idx).spec_path, why)) {
            landing_.log("launch failed", name, *why);
            return false;
        }
        landing_.log("open", spec_page_.title(), "spec");
        page_ = AppPage::Spec;
        glfwSetWindowTitle(window_, ("Caliper - " + spec_page_.title()).c_str());
        mark_spec_entry(idx);
        return true;
    }

    // Once per frame, after the session's: a pending launch of a workspace
    // spec applet goes out as applet.open, its refusal goes to Activity,
    // every applet.open that succeeded, from any origin, shows the spec
    // page, and every applet.save lists the workspace's applets again, while
    // no compiled applet runs (the loader moves its entries).
    void note_spec_applets() {
        if (!pending_spec_open_.empty()) {
            if (session_.session()) {
                spec_page_.request_open(session_, pending_spec_open_, pending_spec_origin_);
                pending_spec_open_.clear();
            } else if (!session_.open_error().empty()) {
                landing_.log("launch failed", pending_spec_open_, session_.open_error());
                pending_spec_open_.clear();
            } else {
                // A launch before the session started (CALIPER_AUTOLAUNCH)
                // opens the default workspace here.
                session_.ensure_open();
            }
        }
        std::string error;
        bool was_new = false;
        while (spec_page_.request_answered(&error, &was_new)) {
            if (error.empty()) continue;
            landing_.log(was_new ? "new failed" : "launch failed", was_new ? "spec.new" : "applet.open", error);
            if (was_new) {
                new_pending_ = false;
                drop_new_error_ = error;   // a drop that asked for it says why it goes
            }
        }
        if (spec_page_.take_open(session_)) {
            // An agent never ends the person's running applet: on a compiled
            // applet's page its spec opens in the session, and the page stays.
            if (page_ == AppPage::Applet && spec_page_.opened_by_agent()) note_agent_spec();
            else open_spec_page();
        }
        // A new spec (F3 New dashboard, or a drop with none open): the page in build mode.
        if (spec_page_.take_new()) {
            open_spec_page();
            spec_page_.set_build_mode(true);
            new_pending_ = false;
            drop_new_asked_ = false;   // the drop's new spec is made: it goes into it, wherever the person is now
        }
        if (session_.applets_saved() != applets_saved_ && page_ != AppPage::Applet) {
            applets_saved_ = session_.applets_saved();
            const int sel = landing_.selected_index();
            const std::string selected = sel >= 0 && sel < loader_.count() ? loader_.at(sel).manifest.id : "";
            const std::string shown = spec_entry_;
            const int listed = loader_.rescan_workspace();
            if (listed >= 0) {
                if (!selected.empty()) landing_.select(loader_.find(selected));
                // The spec the page shows is ACTIVE and the landing's
                // selection, so that Home returns to its row: also once it
                // is saved for the first time, which lists it only now.
                if (page_ == AppPage::Spec) {
                    const int now = workspace_spec_entry(spec_page_.applet_id());
                    if (now >= 0) mark_spec_entry(now);
                    else if (!shown.empty()) mark_spec_entry(loader_.find(shown));
                }
            }
        }
    }

    // The session's open spec's title, read again only when the journal
    // moved and no long command holds the session; "" when none is open.
    const std::string& open_spec_title() {
        const uint64_t j = session_.journaled();
        if (j != open_spec_read_ && !session_.worker_busy()) {
            open_spec_title_ = caliper_host::SpecPage::open_title(session_);
            open_spec_read_ = j;
        }
        return open_spec_title_;
    }

    // An agent's spec.new, spec.open or applet.open while the person is on a
    // compiled applet's page: the page stays, Activity and standard error say
    // that a dashboard was opened and how to get to it, and the page's status
    // bar names it until the person leaves the page. The person's own opens
    // show the spec page as before.
    void note_agent_spec() {
        const std::string title = caliper_host::SpecPage::open_title(session_);
        agent_spec_note_ = title.empty() ? std::string("a dashboard") : title;
        landing_.log("agent opened", agent_spec_note_, "esc Home, then F6 Dashboard");
        std::cerr << "[spec] an agent opened " << agent_spec_note_
                  << "; the applet's page stays (esc Home, then F6 Dashboard)" << std::endl;
    }

    // The entry of the workspace's spec applet `id` (its folder's name), or -1.
    int workspace_spec_entry(const std::string& id) const {
        if (id.empty()) return -1;
        for (int i = 0; i < loader_.count(); i++) {
            const caliper_host::AppletEntry& e = loader_.at(i);
            if (e.manifest.kind == caliper_host::AppletKind::Spec &&
                (e.status == caliper_host::AppletStatus::Ready || e.status == caliper_host::AppletStatus::Active) &&
                workspace_applet_id(e) == id)
                return i;
        }
        return -1;
    }

    // F3 New dashboard on the landing, or a drop with no open spec: spec.new
    // from the ui origin once the session is open, with an id derived from
    // the title; note_spec_applets() shows the page in build mode when it
    // succeeds.
    void new_spec(const std::string& title) {
        session_.ensure_open();
        pending_new_title_ = title.empty() ? std::string("Untitled") : title;
        new_requested_ = true;
        landing_.log("new", pending_new_title_, "spec");
    }

    // Files dropped on the window (GLFW's drop callback, or the hidden input
    // driver's drop step, which calls this the same way).
    void on_drop(std::vector<std::string> paths) {
        if (paths.empty()) return;
        if (!caliper_host::SpecPage::built()) {
            landing_.log("drop", std::to_string(paths.size()) + (paths.size() == 1 ? " file" : " files"),
                         "the builder is not in this build");
            return;
        }
        pending_drops_.push_back(std::move(paths));
    }

    // Once per frame: a new spec that waits for the session, then the files
    // dropped. On the spec page they go to its builder (build mode on); on
    // the landing the open spec's page opens first, or a new spec when none
    // is open; on the live page a new spec, beside the attached session's
    // live sources in the builder's Sources, unless the open spec has
    // unsaved changes, which takes the files as on the landing; the applet
    // page keeps its windows and says so in Activity.
    void note_drops() {
        if (new_requested_ && !new_pending_) {
            if (session_.session() && !session_.worker_busy()) {
                if (spec_page_.request_new(session_, pending_new_title_)) new_pending_ = true;
                new_requested_ = false;
            } else if (!session_.open_error().empty()) {
                landing_.log("new failed", pending_new_title_, session_.open_error());
                new_requested_ = false;
            }
        }
        if (pending_drops_.empty()) return;
        const std::vector<std::string>& paths = pending_drops_.front();
        const std::string what = paths.size() == 1 ? std::filesystem::path(paths[0]).filename().string()
                                                    : std::to_string(paths.size()) + " files";
        if (page_ == AppPage::Spec) {
            if (!session_.session() || session_.worker_busy() || spec_page_.importing()) return;   // wait
            std::string why;
            if (spec_page_.drop_files(session_, paths, &why)) landing_.log("drop", what, "import");
            else landing_.log("drop refused", what, why);
            pending_drops_.erase(pending_drops_.begin());
            drop_new_asked_ = false;
            return;
        }
        if (page_ == AppPage::Landing || page_ == AppPage::Live) {
            session_.ensure_open();
            if (!session_.open_error().empty()) {
                landing_.log("drop refused", what, session_.open_error());
                pending_drops_.clear();
                return;
            }
            if (!session_.session() || session_.worker_busy() || new_requested_ || new_pending_) return;   // wait
            // One new spec per drop: when its spec.new failed (said in
            // Activity), the drop goes instead of asking again every frame.
            if (drop_new_asked_) {
                landing_.log("drop refused", what,
                             drop_new_error_.empty() ? std::string("no new spec was made") : drop_new_error_);
                pending_drops_.clear();
                drop_new_asked_ = false;
                drop_new_error_.clear();
                return;
            }
            const bool into_open = page_ == AppPage::Landing ? caliper_host::SpecPage::has_open_spec(session_)
                                                             : caliper_host::SpecPage::has_unsaved_spec(session_);
            if (into_open) {
                open_spec_page();
                spec_page_.set_build_mode(true);
            } else {
                new_spec("");
                drop_new_asked_ = true;
            }
            return;
        }
        landing_.log("drop refused", what, "open a spec to import files");
        pending_drops_.clear();
        drop_new_asked_ = false;
        drop_new_error_.clear();
    }

    // Caliper API's calls of this frame (api/api_server.h). A capture waits
    // for the read-back of a presented frame (captures_).
    void serve_api() {
        for (const caliper_api::Call& c : api_.take()) {
            switch (c.kind) {
                case caliper_api::Call::Kind::Applets:
                    api_.reply(c.ticket, 200, caliper_host::applets_json(loader_));
                    break;
                case caliper_api::Call::Kind::Launch:
                    api_launch(c);
                    break;
                case caliper_api::Call::Kind::Capture:
                    captures_.add(c.ticket);
                    break;
            }
        }
    }

    // POST /v0/applets/{id}/launch and the session's app.launch: opens that
    // applet's page as a click on its row does. The page shown goes first, as
    // Home takes it: an applet is torn down; an attached live session stays
    // attached.
    void api_launch(const caliper_api::Call& c) {
        const caliper_host::LaunchOutcome o = launch_by_id(c.applet_id, c.origin.empty() ? "api" : c.origin);
        if (o.status == 200) api_.reply(c.ticket, 200, o.body);
        else api_.reply_error(c.ticket, o.status, o.body);
    }

    // `origin`: who asked, "ui" for the person (and CALIPER_AUTOLAUNCH, the
    // person's own setting), else the app.launch's or the API token's.
    caliper_host::LaunchOutcome launch_by_id(const std::string& id, const std::string& origin = "ui") {
        const int i = loader_.find(id);
        if (i < 0)
            return {404, "no applet with the id " + id + " in the search folders (GET /v0/applets lists them)",
                    "app.applet_not_found"};
        if (page_ == AppPage::Applet && active_applet_ == i) {
            const auto& a = loader_.at(i);
            return {200, caliper_host::launched_json(a.manifest.id, a.manifest.version, true), ""};
        }
        if (loader_.at(i).manifest.kind == caliper_host::AppletKind::Spec) {
            const auto& a = loader_.at(i);
            if (page_ == AppPage::Spec && a.status == caliper_host::AppletStatus::Active)
                return {200, caliper_host::launched_json(a.manifest.id, a.manifest.version, true), ""};
            if (a.status != caliper_host::AppletStatus::Ready)
                return {409,
                        id + " cannot be launched (" + caliper_host::applet_status_word(a.status) + "): " +
                            a.status_text,
                        "app.launch_refused"};
            // The page opens with the applet.open that this sends, a frame
            // or two later. One of the workspace that would replace unsaved
            // changes from an origin under review waits for the person's
            // approval, and the answer says so rather than "launched"; it
            // reads the state now, so it claims no more than that.
            const caliper_host::SpecPage::OpenReview review =
                workspace_applet_id(a).empty() ? caliper_host::SpecPage::OpenReview::opens
                                               : caliper_host::SpecPage::open_review(session_, origin);
            std::string why;
            if (!launch_spec_applet(i, &why, origin)) return {409, id + " failed to launch: " + why, "app.launch_failed"};
            const auto& b = loader_.at(loader_.find(id));
            if (review == caliper_host::SpecPage::OpenReview::waits)
                return {200,
                        caliper_host::launch_pending_json(
                            b.manifest.id, b.manifest.version, "PENDING_APPROVAL",
                            "Accepted, not open yet: opening it would replace the open spec's unsaved changes, so it "
                            "waits for the person's approval in the app. spec.get names the open spec once it has "
                            "opened; if the person denies it, the open spec stays as it is."),
                        ""};
            if (review == caliper_host::SpecPage::OpenReview::unknown)
                return {200,
                        caliper_host::launch_pending_json(
                            b.manifest.id, b.manifest.version, "PENDING",
                            "Accepted, not open yet: the app is running a long command, so the open waits for it, and "
                            "then for the person's approval if the open spec has unsaved changes by then. spec.get "
                            "names the open spec once it has opened."),
                        ""};
            return {200, caliper_host::launched_json(b.manifest.id, b.manifest.version, false), ""};
        }
        if (loader_.at(i).status != caliper_host::AppletStatus::Ready) {
            const auto& a = loader_.at(i);
            return {409,
                    id + " cannot be launched (" + caliper_host::applet_status_word(a.status) + "): " + a.status_text,
                    "app.launch_refused"};
        }
        if (page_ == AppPage::Applet) leave_applet_page(active_applet_);
        else if (page_ == AppPage::Live) leave_live_page();
        else if (page_ == AppPage::Spec) leave_spec_page();
        launch_applet(i);
        const auto& a = loader_.at(i);
        if (page_ == AppPage::Applet && active_applet_ == i)
            return {200, caliper_host::launched_json(a.manifest.id, a.manifest.version, false), ""};
        return {409, id + " failed to launch: " + a.status_text, "app.launch_failed"};
    }

    // A capture that cannot be served: the API's call gets the status, the
    // session's capture.frame the reason.
    void capture_error(uint64_t t, int status, const std::string& why) {
        if (t == caliper_host::AppSession::kCaptureTicket) session_.capture_failed(why);
        else if (t == caliper_host::SpecUiDriver::kShotTicket) { if (spec_ui_) spec_ui_->shot_failed(why); }
        else api_.reply_error(t, status, why);
    }

    void launch_applet(int idx) {
        if (idx < 0 || idx >= loader_.count()) return;
        if (loader_.at(idx).manifest.kind == caliper_host::AppletKind::Spec) {
            std::string why;
            launch_spec_applet(idx, &why);
            return;
        }
        const std::string name = loader_.at(idx).manifest.name;
        landing_.log("launch", name);
        // Home returns to this row, however the launch came (a row, Enter,
        // CALIPER_AUTOLAUNCH), so a quarantine lands on its reason.
        landing_.select(idx);
        // initialize() runs inside the landing's frame, after its windows
        // have ended: checkpointed like frame(), in case it throws mid-UI.
        const auto ui_checkpoint = caliper_host::UiCheckpoint::store();
        if (loader_.launch(idx, caliper_host::applet_host_proto())) {
            enter_applet(idx);
        } else {
            ui_checkpoint.recover();
            // The loader marked the entry Failed with the reason; say it in
            // Activity too, so the attempt is not silent.
            landing_.log("launch failed", name, loader_.at(idx).status_text);
        }
    }

    // The Inspector's Restart (C++ exception quarantines only, see
    // AppletLoader::restartable): unload the stale instance, launch afresh.
    void restart_applet(int idx) {
        if (!loader_.restartable(idx)) return;
        const std::string name = loader_.at(idx).manifest.name;
        landing_.log("restart", name);
        landing_.select(idx);
        const auto ui_checkpoint = caliper_host::UiCheckpoint::store();
        if (loader_.restart(idx, caliper_host::applet_host_proto())) {
            enter_applet(idx);
        } else {
            ui_checkpoint.recover();
            landing_.log("restart failed", name, loader_.at(idx).status_text);
        }
    }

    void enter_applet(int idx) {
        active_applet_ = idx;
        page_view_.enter();
        last_frame_time_ = glfwGetTime();
        page_ = AppPage::Applet;
        glfwSetWindowTitle(window_,
                           ("Caliper - " + loader_.at(idx).manifest.name).c_str());
    }

    // Esc goes Home (applet and live pages) only when it is not cancelling
    // something else: a text field being edited (an applet's search box) or an
    // open popup gets Esc first, as in any desktop app. Both reads are of the
    // previous frame on purpose: WantTextInput is computed from last frame's
    // active field, and ImGui closes a popup on Esc inside NewFrame(), before
    // this code can see it open.
    // The widget agent's settings and the runs panel take Escape while they
    // are open.
    bool agent_settings_open() const {
#if CALIPER_WITH_BUILDER
        return agent_settings_.is_open() || runs_panel_.is_open();
#else
        return false;
#endif
    }

    bool esc_home_pressed() const {
        const bool esc_free =
            !ImGui::GetIO().WantTextInput && !popup_open_prev_ && !agent_settings_open() &&
            !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId |
                                             ImGuiPopupFlags_AnyPopupLevel);
        return ImGui::IsKeyPressed(ImGuiKey_Escape, false) && esc_free;
    }

    // ---- Live page (F4 Attach to process) ---------------------------------

    // An attach while a session is attached: the attached one again shows
    // its live page; another replaces it, the attached one detached first
    // (one attachment at a time). True when the attach should go on.
    // One hosted core per process, so the attached session is detached
    // before another is tried; what it was is kept in `previous`, so that a
    // failed attach (a stale row, a refusal) can attach it again
    // (restore_previous), and nothing is lost.
    struct Previous {
        bool attached = false;
        std::string session, url, script;
        int32_t pid = 0;
    };
    bool make_room_for(const std::string& session, Previous* previous) {
        if (!live_.attached()) return true;
        if (!session.empty() && session == live_.status().session) {
            show_live_page();
            return false;
        }
        previous->attached = true;
        previous->session = live_.status().session;
        previous->url = live_url_;
        previous->script = live_script_;
        previous->pid = live_pid_;
        live_.detach();
        note_live_events();   // "detached"
        return true;
    }

    // After a failed attach: the session that was attached before, attached
    // again, on the page shown now.
    void restore_previous(const Previous& p) {
        if (!p.attached) return;
        std::string err;
        const bool ok = p.url.empty() ? live_.attach(renderer_.get(), p.session, &err)
                                      : live_.attach_url(renderer_.get(), p.url, &err);
        if (!ok) {
            landing_.log("attach failed", p.script, err);
            return;
        }
        caliper_live_views::set_look(live_.core(), house_view_look());
        const auto& st = live_.status();
        live_script_ = !st.script.empty() ? st.script : p.script;
        live_pid_ = st.pid ? st.pid : p.pid;
        live_url_ = p.url;
        note_live_events();   // "attached"
    }

    // The landing's Attach (and CALIPER_AUTOATTACH): attach through a hosted
    // core over this window's renderer and open the live page. A refusal (the
    // session went, another viewer holds it, no Metal device) stays on the
    // landing and says why in Activity.
    void attach_session(const std::string& token) {
        Previous previous;
        if (!make_room_for(token, &previous)) return;
        const int i = sessions_.find(token);
        const caliper_live::SessionEntry* e = i >= 0 ? &sessions_.sessions()[(size_t)i] : nullptr;
        std::string script = e && !e->script.empty() ? e->script : token;
        std::string err;
        if (!live_.attach(renderer_.get(), token, &err)) {
            landing_.log("attach failed", script, err);
            std::cerr << "[live] attach failed: " << script << ": " << err << std::endl;
            restore_previous(previous);
            return;
        }
        caliper_live_views::set_look(live_.core(), house_view_look());
        const auto& st = live_.status();
        live_script_ = st.script.empty() ? script : st.script;
        live_pid_ = st.pid ? st.pid : (e ? e->pid : 0);
        live_frame_t_ = 0.0;
        page_ = AppPage::Live;
        set_live_title();
        note_live_events();   // "attached"
    }

#if CALIPER_WITH_BUILDER
    // The model runner between two frames (design/hosted-runs.md): once the
    // session is adopted, its run sources join the catalog and the data
    // sources and its runner events go to the session's ring; then a managed
    // run that became ready, or that a run command named, is attached.
    void frame_runner() {
        if (!runner_) return;
        if (!runner_session_) {
            caliper_builder::Session* s = session_.session();
            if (s && !session_.worker_busy()) {
                const caliper_builder::Error e = s->catalog().add_provider("runs", runner_->catalog_provider());
                if (!e.ok()) std::cerr << "[session] run sources: " << e.message << std::endl;
                s->data_sources().set(caliper_builder::CatalogKind::run, runner_->data_source());
                runner_->set_event_sink(&s->events());
                runner_session_ = s;
            }
        }
        const std::string next = runner_->frame(live_.attached(), live_.attached() ? live_url_ : std::string());
        if (next.empty()) return;
        // The spec page stays: its panels draw the runs. From another page the
        // app shows the run's live page, as --url does.
        const bool on_spec = page_ == AppPage::Spec;
        attach_stream(next);
        if (on_spec && page_ == AppPage::Live) {
            page_ = AppPage::Spec;
            glfwSetWindowTitle(window_, ("Caliper - " + spec_page_.title()).c_str());
        }
    }

    // Before the session closes: the session lets go of the runner, and the
    // runner ends every child by its process id, so that none outlives the app.
    void close_runner() {
        if (!runner_) return;
        if (runner_session_ && runner_session_ == session_.session()) {
            runner_session_->catalog().remove_provider("runs");
            runner_session_->data_sources().set(caliper_builder::CatalogKind::run, nullptr);
        }
        runner_->set_event_sink(nullptr);
        runner_session_ = nullptr;
        session_.set_model_runner(nullptr);
        runner_->set_app_control(nullptr);
        runner_->shutdown();
    }
#endif

    // The attach pane's stream link (wire protocol v3, live_net_source.h):
    // attach through a hosted core as for a session; a refusal (nothing
    // listens, a wrong token) stays on the landing and says why in Activity.
    // The link holds the session token, so only the part before '?' is shown.
    void attach_stream(const std::string& link) {
        const std::string shown = link.substr(0, link.find('?'));
        // A listed stream's row is matched by its session id; a pasted link
        // to the attached stream itself shows its page too.
        const int i = [&] {
            for (size_t k = 0; k < sessions_.sessions().size(); ++k)
                if (sessions_.sessions()[k].stream_url == link) return (int)k;
            return -1;
        }();
        if (live_.attached() && link == live_url_) {
            show_live_page();
            return;
        }
        Previous previous;
        if (!make_room_for(i >= 0 ? sessions_.sessions()[(size_t)i].session : std::string(), &previous)) return;
        std::string err;
        if (!live_.attach_url(renderer_.get(), link, &err)) {
            landing_.log("attach failed", shown, err);
            std::cerr << "[live] attach failed: " << shown << ": " << err << std::endl;
            restore_previous(previous);
            return;
        }
        caliper_live_views::set_look(live_.core(), house_view_look());
        const auto& st = live_.status();
        live_script_ = !st.script.empty() ? st.script : !st.session.empty() ? st.session : shown;
        live_pid_ = st.pid;
        live_url_ = link;
        live_frame_t_ = 0.0;
        page_ = AppPage::Live;
        set_live_title();
        note_live_events();   // "attached"
    }

    // What the live page's controls did (live_controls.h), one stderr line
    // each like [live] attached, for headless runs and tests.
    void note_control_lines() {
        caliper_live::RunControls* rc = live_.controls();
        if (!rc) return;
        for (const std::string& line : rc->take_log()) std::cerr << "[live] " << line << std::endl;
    }

    void feed_spec_ui() {
        spec_ui_->set_page(page_ == AppPage::Spec      ? "spec"
                           : page_ == AppPage::Landing ? "landing"
                           : page_ == AppPage::Applet  ? "applet"
                                                       : "live");
        spec_ui_->set_drop([this](std::vector<std::string> paths) { on_drop(std::move(paths)); });
        spec_ui_->set_command_line(landing_.command_text());
        spec_ui_->set_cards([this] { return approvals_.shown_waiting(); });
        {
            const int sel = landing_.selected_index();
            std::vector<std::string> projects;
            for (const int i : landing_.project_rows())
                if (i >= 0 && i < loader_.count()) projects.push_back(loader_.at(i).manifest.id);
            spec_ui_->set_landing(sel >= 0 && sel < loader_.count() ? loader_.at(sel).manifest.id : std::string(),
                                  std::move(projects));
        }
        spec_ui_->set_save_prompt(&save_prompt_);
        spec_ui_->set_close([this] {   // as GLFW does: the flag, then the request
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
            on_close_request();
        });
#if CALIPER_WITH_BUILDER
        spec_ui_->set_agent_settings(&agent_settings_);
#endif
        spec_ui_->feed(spec_page_, session_, &chat_);
        if (spec_ui_->take_shot_request()) captures_.add(caliper_host::SpecUiDriver::kShotTicket);
        for (const std::string& line : spec_ui_->take_log()) std::cerr << "[ui] " << line << std::endl;
        if (spec_ui_->quit()) glfwSetWindowShouldClose(window_, GLFW_TRUE);
        if (spec_ui_->done() && spec_ui_->released()) {   // a last click is released first
            std::cerr << "[ui] " << (spec_ui_->error().empty() ? "done" : "stopped: " + spec_ui_->error()) << std::endl;
            spec_ui_.reset();
        }
    }

    // The picks and selections a person makes in the live views outside a
    // spec go to the agent's events.wait as events without a panel
    // (design/instruments.md, Event Feeds). Each attachment has a core of its
    // own, so the tap is set at each attach.
    void tap_live_events() {
#if CALIPER_WITH_BUILDER
        caliper_live_set_event_tap(live_.core(), [this](const caliper_live_views::LiveEvent& e) {
            caliper_builder::Session* s = session_.session();
            if (!s) return;
            caliper_builder::Event ev;
            ev.kind = std::strcmp(e.type, "pick") == 0 ? caliper_builder::EventKind::pick
                                                       : caliper_builder::EventKind::select;
            ev.view = e.view;
            ev.origin = "ui";
            ev.fields = nlohmann::json::parse(e.detail, nullptr, false);
            if (!ev.fields.is_object()) ev.fields = nlohmann::json::object();
            s->events().push(std::move(ev));
        });
#endif
    }

    // CALIPER_TEST_UI's steps on the spec page (live_ui_driver.h): a scene
    // panel of the open spec that draws its live scene, a pick in it, and a
    // long poll of events.wait that waits on the event ring.
    void give_ui_driver_spec_hooks() {
#if CALIPER_WITH_BUILDER
        caliper_live::UiDriver::SpecHooks h;
        h.scene_ready = [this](const std::string& id) { return scene_panel_ready(id); };
        h.scene_pick = [this](const std::string& id, float u, float v) {
            return scene_panel_ready(id) && spec_page_.pick_scene(id, u, v);
        };
        h.long_poll_held = [this](const std::string& kind) {
            caliper_builder::Session* s = session_.session();
            if (!s) return false;
            if (kind.empty()) return s->events().waiting() > 0;
            const std::optional<caliper_builder::EventKind> k = caliper_builder::event_kind_from_name(kind);
            return k && s->events().waiting(k) > 0;
        };
        ui_driver_->set_spec_hooks(std::move(h));
#endif
    }

#if CALIPER_WITH_BUILDER
    bool scene_panel_ready(const std::string& id) {
        namespace spec = caliper_builder::spec;
        caliper_builder::Session* s = session_.session();
        CaliperCore* core = live_.core();
        if (page_ != AppPage::Spec || !s || !core) return false;
        const std::optional<caliper_builder::OpenSpec> open = s->open_spec();
        const spec::Panel* p = open ? open->spec.find_panel(id) : nullptr;
        if (!p || p->primitive != spec::Primitive::scene) return false;
        const auto b = p->bind.find("scene");
        if (b == p->bind.end() || b->second.bindings.size() != 1) return false;
        const std::string& ref = b->second.bindings[0];
        if (ref.rfind("live:", 0) != 0) return false;
        const caliper_live_views::SceneView* v = caliper_live_views::scene_view(core, ref.substr(5).c_str());
        return v && v->generation() > 0;
    }
#endif

    void feed_ui_driver() {
        ui_driver_->feed(live_.core(), live_.controls());
        for (const std::string& line : ui_driver_->take_log()) std::cerr << "[ui] " << line << std::endl;
        if (ui_driver_->done()) {
            std::cerr << "[ui] " << (ui_driver_->error().empty() ? "done" : "stopped") << std::endl;
            ui_driver_.reset();
        }
    }

    void autoattach() {
        sessions_.refresh();
        const int i = sessions_.find_pid(autoattach_pid_);
        if (i < 0) return;
        const caliper_live::SessionEntry& e = sessions_.sessions()[(size_t)i];
        if (!caliper_live::can_attach(e)) return;
        autoattach_pid_ = 0;
        const std::string token = e.session;
        std::cerr << "[autoattach] " << token << std::endl;
        landing_.log("autoattach", token);
        if (!e.stream_url.empty()) attach_stream(e.stream_url);   // a caliper.serve() stream
        else attach_session(token);
    }

    // The attachment's transitions as Activity rows (and one stderr line
    // each, like [autoattach], for headless runs): attached; session ended
    // (the script exited or died, and the page keeps its last frames);
    // detached (⌘D Detach, another session attached, or quitting).
    void note_live_events() {
        using Event = caliper_live::Attachment::Event;
        for (const Event ev : live_.take_events()) {
            const std::string pid = "pid " + std::to_string(live_pid_);
            const char* what = "";
            switch (ev) {
                case Event::Attached:
                    what = "attached";
                    landing_.log(what, live_script_, pid);
                    tap_live_events();
                    break;
                case Event::Ended:
                    what = "session ended";
                    landing_.log(what, live_script_, pid + " exited; last frames kept");
                    if (page_ == AppPage::Live) set_live_title();
                    break;
                case Event::Detached:
                    what = "detached";
                    landing_.log(what, live_script_, pid);
                    live_url_.clear();
                    break;
            }
            std::cerr << "[live] " << what << ": " << live_script_ << " (" << pid << ")"
                      << std::endl;
        }
    }

    void set_live_title() {
        std::string t = "Caliper - " + live_script_ + " (pid " + std::to_string(live_pid_) + ")";
        if (live_.attached() && !live_.status().alive && !live_.status().pending)
            t += " - session ended";
        glfwSetWindowTitle(window_, t.c_str());
    }

    // The live page draws at up to 60 fps, like caliper-viewer's 60 Hz timer:
    // instead of drawing at the display's rate (120 Hz on ProMotion), the loop
    // waits out the rest of each 1/60 s for events, so attaching costs the
    // training script's GPU no more than the standalone viewer does.
    void wait_for_live_frame() {
        constexpr double kFrame = 1.0 / 60.0;
        const double next = live_frame_t_ + kFrame;
        for (double t = glfwGetTime(); t < next; t = glfwGetTime())
            glfwWaitEventsTimeout(next - t);
        live_frame_t_ = glfwGetTime();
    }

    // The live page: the session's views as windows on a docked desktop like
    // the applet page's (they tile into the work area on first appearance and
    // can be moved, resized or docked; the layout persists in imgui.ini), under
    // the same page bar and status bar. esc Home goes to the landing with the
    // session still attached; ⌘D Detach ends the attachment. Both act after
    // the frame.
    void draw_live_page() {
        const auto& st = live_.status();
        bool go_back = esc_home_pressed();
        bool detach = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_D) && !agent_settings_open();
        // Dev hook: CALIPER_HOME_AFTER=<sec>, as on the applet page: Home once,
        // N sec after the live page first opened; the session stays attached,
        // and the live page opened again (F5) stays.
        if (const char* ha = std::getenv("CALIPER_HOME_AFTER")) {
            static double page_t0 = -1.0;
            static bool fired = false;
            if (page_t0 < 0) page_t0 = glfwGetTime();
            if (!fired && glfwGetTime() - page_t0 >= std::atof(ha)) go_back = fired = true;
        }

        // Page bar: esc Home, ⌘D Detach, the script, then its state. The live
        // diamond is filled and breathes while the script runs and is
        // outlined once it has gone (SPEC §6.5); SESSION ENDED is in capitals,
        // in parchment, as the jobs tray writes FAILED (bronze stays for focus
        // and selection).
        if (caliper_host::begin_page_bar()) {
            if (sternbild::key_button("esc", "Home")) go_back = true;
#ifdef __APPLE__
            if (sternbild::key_button("\xe2\x8c\x98" "D", "Detach")) detach = true;   // ⌘D
#else
            if (sternbild::key_button("Ctrl D", "Detach")) detach = true;
#endif
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
            ImGui::TextUnformatted(live_script_.c_str());
            ImGui::PopStyleColor();
            ImGui::Separator();
            const bool running = st.alive && !st.pending;
            ImGui::Dummy(ImVec2(8.0f, ImGui::GetFrameHeight()));
            {
                const ImVec2 mn = ImGui::GetItemRectMin();
                const ImVec2 mx = ImGui::GetItemRectMax();
                const float phase = (float)std::fmod(ImGui::GetTime(), 2.4) / 2.4f;
                const float alpha = running ? 0.725f + 0.275f * std::cos(phase * 6.2831853f)
                                            : 1.0f;
                sternbild::diamond(ImGui::GetWindowDrawList(),
                                   ImVec2(std::floor((mn.x + mx.x) * 0.5f),
                                          std::floor((mn.y + mx.y) * 0.5f)),
                                   4.0f,
                                   sternbild::with_alpha(sternbild::col::parchment,
                                                         (int)(255.0f * alpha)),
                                   running);
            }
            if (st.pending && st.alive) {
                ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
                ImGui::TextUnformatted("waiting for the session to start");
                ImGui::PopStyleColor();
            } else if (!st.alive) {
                ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::parchment);
                ImGui::TextUnformatted("SESSION ENDED");
                ImGui::PopStyleColor();
                ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
                ImGui::TextUnformatted("the script has exited; its last frames stay");
                ImGui::PopStyleColor();
            } else {
                ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
                ImGui::TextUnformatted("live");
                ImGui::PopStyleColor();
            }
            // ⌘L Chat at the right end: the side chat, which can build a spec
            // over this stream's live sources.
            if (caliper_host::ChatPane::built()) {
                const float right = ImGui::GetWindowContentRegionMax().x - caliper_host::ChatPane::toggle_width() -
                                    sternbild::metric::pad;
                if (right > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(right);
                chat_.draw_toggle();
            }
            ImGui::EndMainMenuBar();
        }
        chat_.read_keys();

        // The run bar (wire protocol 3.1) under the page bar, for a stream:
        // the run state, pause, resume and step, the script's parameters and
        // actions, and why they are off when they are.
        caliper_live::RunControls* rc = live_.controls();
        const caliper_live::ControlsLook look = house_controls_look();
        if (rc) {
            sternbild::control_bar_begin();
            caliper_live::draw_run_bar(*rc, look, ImGui::GetContentRegionAvail().x);
            sternbild::control_bar_end();
        }

        // Status bar: the machine, then the session's facts.
        sternbild::status_bar_begin();
        draw_host_status_items(renderer_->name());
        {
            char text[64];
            std::snprintf(text, sizeof text, "pid %d", live_pid_);
            sternbild::status_item(text);
            std::snprintf(text, sizeof text, st.views == 1 ? "%d view" : "%d views", st.views);
            sternbild::status_item(text);
            if (st.fps > 0.0) {
                std::snprintf(text, sizeof text, "%.0f fps", st.fps);
                sternbild::status_item(text);
            }
            if (st.latency_ms > 0.0) {
                std::snprintf(text, sizeof text, "%.0f ms to screen", st.latency_ms);
                sternbild::status_item(text);
            }
            if (!st.alive) sternbild::status_item("session ended");
        }
        sternbild::status_bar_end();
        jobs_tray_.draw();   // above the status bar while it is open
        chat_.draw_column();   // the side chat's column at the right, while open

        // Its own dock space id, so the applet page's docked layout never
        // shows through on this page (and the reverse). The views tile the
        // work area the bars leave this frame.
        const ImRect area = caliper_host::work_area();
        caliper_host::dock_space_over_work_area(ImGui::GetID("caliper_live_dockspace"),
                                                ImGuiDockNodeFlags_PassthruCentralNode);
        // The selection's details take a pane at the right edge while a
        // selection has them; the views tile what is left.
        float views_w = area.GetWidth();
        if (rc && rc->details()) {
            const float pane_w = std::floor(std::min(380.0f, area.GetWidth() * 0.4f));
            views_w -= pane_w;
            ImGui::SetNextWindowPos(ImVec2(area.Min.x + views_w, area.Min.y));
            ImGui::SetNextWindowSize(ImVec2(pane_w, area.GetHeight()));
            const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                           ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                                           ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings;
            if (ImGui::Begin("##caliper_live_selection", nullptr, flags)) {
                const caliper_live::LinkedSelection* sel = rc->details();
                sternbild::pane_header("Selection", sel->space.c_str(), nullptr, false);
                caliper_live::draw_selection_details(*rc, look);
            }
            ImGui::End();
        }
        live_.draw(area.Min.x, area.Min.y, views_w, area.GetHeight(),
                   ImGui::GetIO().DisplayFramebufferScale.x);

        if (detach) pending_detach_ = true;
        else if (go_back) pending_live_home_ = true;
    }

    // A failed job (JobView::failed) is kept by the tray until the user
    // clears finished jobs, and written to Activity once. The job system's
    // own entry does not last that long: going Home joins the applet's jobs
    // and drops every entry, and a failed applet job is what quarantines its
    // applet and sends the host Home. Job ids never repeat.
    void note_failed_jobs(const std::vector<caliper_host::JobSystem::JobView>& views) {
        jobs_tray_.note_failures(views, [this](const caliper_host::JobsTray::Failure& f) {
            landing_.log("job failed", f.label, f.reason);
        });
    }

    // One Activity line per applet directory scanned, the path written from
    // the home folder ("~/...") so the OBJECT column shows the part that
    // differs between the two scan locations.
    void log_scan(const std::string& dir, int added) {
        landing_.log("scanned", home_relative(dir),
                     std::to_string(added) + (added == 1 ? " applet" : " applets"));
    }

    // `path` relative to the workspace when it is in it, such as
    // applets/run42-dashboard/run42-dashboard.spec.toml; otherwise as
    // home_relative() writes it.
    std::string workspace_relative(const std::string& path) const {
        namespace fs = std::filesystem;
        if (!workspace_dir_.empty()) {
            std::error_code ec;
            for (const bool canonical : {false, true}) {
                const fs::path p = canonical ? fs::weakly_canonical(path, ec) : fs::path(path);
                const fs::path w = canonical ? fs::weakly_canonical(workspace_dir_, ec) : fs::path(workspace_dir_);
                if (ec) break;
                const fs::path rel = p.lexically_relative(w);
                if (!rel.empty() && *rel.begin() != "..") return rel.generic_string();
            }
        }
        return home_relative(path);
    }

    // `path` with the user's home folder written as "~"; unchanged when it is
    // not under home or home is unknown.
    static std::string home_relative(const std::string& path) {
#ifdef _WIN32
        const char* home = std::getenv("USERPROFILE");
#else
        const char* home = std::getenv("HOME");
#endif
        if (!home || !*home) return path;
        std::string h = home;
        while (h.size() > 1 && (h.back() == '/' || h.back() == '\\')) h.pop_back();
        if (path.compare(0, h.size(), h) != 0) return path;
        if (path.size() == h.size()) return "~";
        const char next = path[h.size()];
        if (next != '/' && next != '\\') return path;   // "/Users/namex" is not home
        return "~" + path.substr(h.size());
    }

    // Creates the GLFW window with the renderer's hints and starts the
    // renderer on it; false when either fails (window_ says which).
    bool create_window_and_init(int ww, int wh) {
        glfwDefaultWindowHints();
        renderer_->window_hints();
        if (hidden_) {
            // Never shown, never focused (GLFW shows and focuses a window at
            // creation only when it is visible).
            glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
            glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
            glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
        }
        window_ = glfwCreateWindow(ww, wh, "Caliper", nullptr, nullptr);
        if (!window_) return false;
        // Files dropped on the window (the Finder, a file manager) are
        // imported into the workspace by the spec page's builder.
        glfwSetWindowUserPointer(window_, this);
        glfwSetDropCallback(window_, [](GLFWwindow* w, int count, const char** paths) {
            auto* app = static_cast<CaliperApp*>(glfwGetWindowUserPointer(w));
            if (!app || count <= 0) return;
            std::vector<std::string> list;
            for (int i = 0; i < count && i < 256; ++i)
                if (paths[i] && *paths[i]) list.emplace_back(paths[i]);
            app->on_drop(std::move(list));
        });
        // The close button and ⌘Q: "Save changes to <name>?" first when the
        // open spec has unsaved changes (on_close_request).
        glfwSetWindowCloseCallback(window_, [](GLFWwindow* w) {
            if (auto* app = static_cast<CaliperApp*>(glfwGetWindowUserPointer(w))) app->on_close_request();
        });
        // A press at the pointer's place: GLFW's macOS backend gives a mouse press without the
        // place of its event (cocoa_window.m, mouseDown:), and the window gets no cursor moves
        // while it is not the key window or while moves do not reach it, so ImGui would put the
        // press at the last place it was given, often hundreds of points away, and the drag's
        // first move would then carry that whole distance in one frame. The pointer's place goes
        // to ImGui first: ImGui's GLFW backend (installed by the renderer below) calls the
        // callback it finds here before it adds the button.
        glfwSetMouseButtonCallback(window_, [](GLFWwindow* w, int, int action, int) {
            if (action != GLFW_PRESS || !ImGui::GetCurrentContext()) return;
            double x = 0.0, y = 0.0;
            glfwGetCursorPos(w, &x, &y);
            ImGui::GetIO().AddMousePosEvent(float(x), float(y));
        });
        return renderer_->init(window_);
    }

    std::vector<std::string> applet_dirs_;   // --applets, in order
    bool hidden_ = false;                    // --hidden: the window is never shown
    GLFWwindow* window_ = nullptr;
    std::unique_ptr<caliper_host::HostRenderer> renderer_;
    AppPage page_ = AppPage::Landing;
    LandingScreen landing_;
    caliper_host::AppletLoader loader_{caliper_host::host_caps(),
                                       caliper::app_data_path("data")};
    int active_applet_ = -1;
    int pending_home_ = -1;   // applet idx awaiting post-render teardown (Home)
    double last_frame_time_ = 0.0;
    caliper_host::JobsTray jobs_tray_;      // every page; failures kept until cleared
    caliper_host::AppletPage page_view_;    // the applet page: bars, dock space, applet
    bool popup_open_prev_ = false;    // a popup was open at the end of last frame
    // F4 Attach to process: the live sessions the landing lists, the one
    // attached session (hosted core) and what Activity names it by.
    caliper_live::SessionList sessions_;
    caliper_live::Attachment live_;
    caliper_live::AttachmentRunLink run_link_{live_};   // the session's run control acts through it
    std::string live_script_;
    int32_t live_pid_ = 0;
    bool pending_live_home_ = false;  // esc Home on the live page: the landing after render
    bool pending_detach_ = false;     // ⌘D Detach on the live page: detach after render
    std::string live_url_;            // the attached stream's link (attach_stream), "" otherwise
    caliper_host::SpecPage spec_page_;  // the session's open spec (applet.open)
    caliper_host::AppProviders providers_;  // live views, feed channels and metrics as catalog sources
    caliper_host::ApprovalTray approvals_;  // the agents' requests that wait for the person
    caliper_host::SavePrompt save_prompt_;  // "Save changes to <name>?" before the person's replacement or quit
    bool quit_confirmed_ = false;           // the person answered the close's question
    caliper_host::ChatPane chat_;           // the side chat, on the spec and live pages
#if CALIPER_WITH_BUILDER
    std::unique_ptr<caliper_runner::ModelRunner> runner_;   // the managed runs (design/hosted-runs.md)
    caliper_builder::Session* runner_session_ = nullptr;    // the session that holds its run sources
    caliper_host::AgentSettings agent_settings_;   // the widget agent's settings, a popup over any page
    caliper_host::RunsPanel runs_panel_;           // the managed runs, a popup over any page
    caliper_live::AppProgramHost program_host_;    // the open spec's programs in the attached script
    caliper_live::AttachmentProgramLink program_link_{live_};
#endif
    caliper_host::CommandLine cl_;          // for starting Caliper API later, for the chat
    bool api_started_here_ = false;         // the app started the API without --api (api_for())
    bool api_for_chat_ = false;             // ... and the side chat's agent uses it
    bool api_for_browser_ = false;          // ... and Open in browser gave its link out
    // The dashboard an agent opened while a compiled applet's page showed,
    // which that page's status bar names (note_agent_spec()); "" for none.
    std::string agent_spec_note_;
    // The open spec's title for the landing's F6 Dashboard, read again when
    // the journal moved (open_spec_title()).
    std::string open_spec_title_;
    uint64_t open_spec_read_ = ~uint64_t(0);

    // The API's address as the status bar and Activity show it: 127.0.0.1:<port>.
    std::string api_address() const {
        const std::string url = api_.url();   // http://127.0.0.1:<port>/v0
        const size_t a = url.find("://");
        if (a == std::string::npos) return url;
        const size_t b = url.find('/', a + 3);
        return url.substr(a + 3, b == std::string::npos ? std::string::npos : b - a - 3);
    }
    bool pending_spec_home_ = false;  // Home on the spec page: leave after render
    std::string workspace_dir_;       // the session's workspace, whose spec applets are listed
    std::string pending_spec_open_;   // a workspace spec applet to open once the session is open
    std::string pending_spec_origin_ = "ui";   // ... from the origin of who asked for it
    std::vector<std::vector<std::string>> pending_drops_;   // files dropped on the window, in order
    bool new_requested_ = false;      // a new spec waits for the session
    bool new_pending_ = false;        // its spec.new is in the queue
    bool drop_new_asked_ = false;     // the first pending drop asked for a new spec
    std::string drop_new_error_;      // why that spec.new failed, for the drop's refusal
    std::string pending_new_title_;
    std::string spec_entry_;          // the manifest id of the spec applet the page shows, or ""
    uint64_t applets_saved_ = 0;      // AppSession::applets_saved() when last listed
    caliper_host::AppApi api_;        // Caliper API (--api); not running without it
    caliper_host::WindowCaptures captures_;   // the API's and the session's captures of the window's frame
    caliper_host::AppSession session_;        // the builder session of the workspace
    double live_frame_t_ = 0.0;       // start of the live page's current frame
    int autoattach_pid_ = 0;          // CALIPER_AUTOATTACH, until attached
    std::unique_ptr<caliper_live::UiDriver> ui_driver_;   // CALIPER_TEST_UI with --hidden
    std::unique_ptr<caliper_host::SpecUiDriver> spec_ui_;   // CALIPER_TEST_SPEC_UI with --hidden
    bool chromeless_ = false;         // CALIPER_CHROMELESS=1: applet page without bars
};

int main(int argc, char** argv) {
    // The command line is read before anything opens a window.
    const caliper_host::CommandLine cl = caliper_host::parse_command_line(argc, argv);
    if (!cl.error.empty()) {
        std::fprintf(stderr, "caliper: %s\n%s", cl.error.c_str(),
                     caliper_host::usage_text());
        return 2;
    }
    using Command = caliper_host::CommandLine::Command;
    if (cl.command == Command::Help) {
        std::fputs(caliper_host::usage_text(), stdout);
        return 0;
    }
#ifdef _WIN32
    // What the Windows build refuses for now (host_cli.h), before anything
    // starts. CALIPER_API, a setting rather than this command line's request,
    // is said to be ignored, and the app starts without the API.
    if (const char* what = caliper_host::windows_unavailable(cl)) {
        std::fprintf(stderr, "caliper: %s\n", caliper_host::windows_unavailable_message(what).c_str());
        return 1;
    }
    if (const char* v = std::getenv("CALIPER_API"); v && std::strcmp(v, "1") == 0)
        std::fprintf(stderr, "caliper: CALIPER_API is ignored: %s\n",
                     caliper_host::windows_unavailable_message("Caliper API").c_str());
#endif
    if (cl.command == Command::Shot) {
        // The applet page drawn into an offscreen canvas and written as a
        // PNG (host/shot.h). No window, no GLFW, no Dock icon.
        return caliper_host::run_shot(cl);
    }
    if (cl.command != Command::Run && cl.command != Command::Dev) {
        // A subcommand runs and exits here; the app never starts, and no
        // store opens. Only install writes, and only to the user applets
        // folder; the others create nothing in the app-data folder.
        const bool reads_applets = cl.command == Command::List ||
                                   cl.command == Command::Check ||
                                   cl.command == Command::Install ||
                                   cl.command == Command::Build;
        caliper_host::CliContext ctx;
        ctx.caps = caliper_host::host_caps();
        ctx.sdk_version = caliper_host::host_sdk_version();
        if (reads_applets) {
            ctx.user_dir = caliper::user_applets_dir();
            ctx.shipped_dir = caliper::shipped_applets_dir();
            // Read only: nothing is created in the app-data folder.
            ctx.data_root =
                (std::filesystem::path(caliper::app_data_location()) / "data").string();
        }
        ctx.applet_path_env = std::getenv("CALIPER_APPLET_PATH");
        if (cl.command == Command::List)   // the spec applets the app would list
            ctx.workspace_dir =
                caliper_host::workspace_choice(cl, std::getenv("CALIPER_WORKSPACE"), caliper::default_workspace_dir()).dir;
        ctx.exe_dir = caliper::executable_dir();   // build: the SDK may be installed beside it
        if (cl.command == Command::Serve) {
            // Producers on this machine find the hub here (0600, removed on exit).
            const char* hub_file = std::getenv("CALIPER_HUB_FILE");
            ctx.hub_file = hub_file && *hub_file ? std::string(hub_file)
                                                 : caliper::app_data_path("hub.json");
        }
        return caliper_host::run_subcommand(cl, ctx, std::cout, std::cerr);
    }
    if (cl.command == Command::Dev) {
        std::error_code ec;
        if (!std::filesystem::is_directory(cl.applet_dirs.front(), ec)) {
            std::fprintf(stderr, "caliper: not a folder: %s\n",
                         cl.applet_dirs.front().c_str());
            return 1;
        }
    }

    std::cout << "=== Caliper ===" << std::endl;

    // --hidden or CALIPER_HIDDEN=1: the real app with its window never shown,
    // a testing aid, which never uses your own app-data folder: without
    // CALIPER_DATA_DIR it stops at the first lookup (app_paths.h, THE GUARD).
    const char* hidden_env = std::getenv("CALIPER_HIDDEN");
    const bool hidden = cl.hidden || (hidden_env && std::strcmp(hidden_env, "1") == 0);
    if (hidden)
        caliper::app_data_guard(cl.hidden ? "the hidden app (--hidden)" : "the hidden app (CALIPER_HIDDEN=1)");
    CaliperApp app(cl.applet_dirs, hidden);
    app.set_workspace(caliper_host::app_session_options(cl).workspace);
    // initialize() prints its own refusal, such as the renderer's.
    if (!app.initialize()) return 1;
    app.start_session(cl);
    if (caliper_host::api_wanted(cl)) app.start_api(cl);
    if (!cl.url.empty()) app.attach_at_start(cl.url);

    void (*old_int)(int) = SIG_DFL;
    void (*old_term)(int) = SIG_DFL;
    if (hidden) {
        old_int = std::signal(SIGINT, caliper_hidden_on_signal);
        old_term = std::signal(SIGTERM, caliper_hidden_on_signal);
    }
    app.run();
    if (hidden) {
        std::signal(SIGINT, old_int);
        std::signal(SIGTERM, old_term);
    }
    app.cleanup();
    return 0;
}
