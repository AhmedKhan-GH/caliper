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
#include "host/live/live_attach.h"
#include "host/live/live_controls.h"
#include "host/live/live_ui_driver.h"
#include "host/live/live_views_state.h"   // the views' header look
#include "host/api/app_api.h"
#include "host/api/window_captures.h"
#include "app_paths.h"

#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
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

        // Renderer seam (docs/explanations/rendering.md): the choice of the
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
        // (--applets, then CALIPER_APPLET_PATH), the user applets folder,
        // then the applets shipped next to the executable. Of several copies
        // of one id the loader keeps the highest version that passes.
        for (const std::string& dir : caliper_host::applet_search_dirs(
                 applet_dirs_, std::getenv("CALIPER_APPLET_PATH"),
                 caliper::user_applets_dir(), caliper::shipped_applets_dir()))
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
            if (hidden_ && *ui) ui_driver_ = std::make_unique<caliper_live::UiDriver>(ui);
            else if (*ui) std::cerr << "[ui] CALIPER_TEST_UI needs --hidden; ignored" << std::endl;
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
            // core's order: poll, then frame). Never waits on the script.
            if (page_ == AppPage::Live) live_.poll();
            if (autoattach_pid_ > 0 && page_ == AppPage::Landing) autoattach();
            note_live_events();
            note_control_lines();

            // Caliper API (--api): the calls that read or change the app run
            // here, between frames, as CALIPER_AUTOLAUNCH launches. A capture
            // arms the read-back of the frame drawn next (window_captures.h).
            if (api_.running()) {
                serve_api();
                captures_.before_frame(*renderer_, [this](uint64_t t, int status, const std::string& why) {
                    api_.reply_error(t, status, why);
                });
            }

            int dw, dh;
            glfwGetFramebufferSize(window_, &dw, &dh);

            // CALIPER_TEST_UI with --hidden: the scripted clicks of a UI test,
            // queued into ImGui before the frame reads its input.
            if (ui_driver_ && page_ == AppPage::Live) feed_ui_driver();

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
                ctx.launch = [this](int i) { launch_applet(i); };
                ctx.restart = [this](int i) { restart_applet(i); };
                ctx.sessions = &sessions_;
                ctx.attach = [this](const std::string& s) { attach_session(s); };
                ctx.after_bars = [this] { jobs_tray_.draw(); };
                ctx.attach_url = [this](const std::string& u) { attach_stream(u); };
                landing_.draw(ctx);
            } else if (page_ == AppPage::Live) {
                draw_live_page();
            } else if (page_ == AppPage::Applet) {
                bool go_back = esc_home_pressed();

                // Dev hook: CALIPER_HOME_AFTER=<sec> forces Home once, N sec
                // after the applet page opened — reproduces "click Home during
                // training" headlessly (for the SIGSEGV repro).
                if (const char* ha = std::getenv("CALIPER_HOME_AFTER")) {
                    static double page_t0 = -1.0;
                    if (page_t0 < 0) page_t0 = glfwGetTime();
                    if (glfwGetTime() - page_t0 >= std::atof(ha)) go_back = true;
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

            // Sampled after every window of the frame: next frame's Esc test
            // needs to know a popup was open before NewFrame() closed it.
            popup_open_prev_ = ImGui::IsPopupOpen(
                nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);

            renderer_->render(dw, dh);

            // Caliper API's captures: a read-back that has finished on the GPU
            // answers its calls; one still copying is picked up by a later frame.
            if (api_.running())
                captures_.after_frame(
                    *renderer_,
                    [this](uint64_t t, uint32_t w, uint32_t h, std::vector<uint8_t> rgba) {
                        api_.reply_capture(t, w, h, std::move(rgba));
                    },
                    [this](uint64_t t, int status, const std::string& why) { api_.reply_error(t, status, why); });

            // Deferred detach (esc Home on the live page), for the same reason
            // as the applet teardown below: the views' textures and imports are
            // released only after the frame that drew them has been rendered.
            if (pending_detach_) {
                pending_detach_ = false;
                leave_live_page();
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
        // A capture reads the window's frame back where the renderer can
        // (Metal); elsewhere it is answered 501 with the reason.
        const std::string capture_unavailable =
            renderer_->window_capture_supported()
                ? std::string()
                : caliper_host::WindowCaptures::unsupported(renderer_->name());
        if (!api_.start(cl, capture_unavailable, caliper::app_data_location(), std::cout, &why)) {
            std::cerr << "[api] not started: " << why << std::endl;
            landing_.log("api not started", why);
            return;
        }
        landing_.log("api", api_.url());
    }

    void cleanup() {
        // The API first: calls still waiting for a frame get 503.
        api_.stop();
        // Workers FIRST: a job may hold a pointer into applet state, so every
        // worker must be joined before close_all() destroys that state (the
        // 139-at-exit was a worker touching a freed applet mid-teardown).
        // Stores are still open here, so a finishing worker's last metrics
        // writes land safely.
        caliper_host::host_job_system().cancel_all_and_join();
        loader_.close_all();
        caliper_host::set_dock_hint_sink(nullptr, nullptr);
        // An attached session's views go while the bridge still has the
        // renderer; the loop ended after rendering the last frame that drew them.
        live_.detach();
        // Applets are torn down first (they may release bridge textures while the
        // renderer is still live); THEN drop the renderer from the bridge before
        // renderer teardown, so no bridge thunk touches a destroyed renderer.
        caliper_host::services_set_renderer(nullptr);
        // Close the DuckDB stores NOW — leaving them to static destructors
        // races DuckDB's globals and aborts in malloc at exit.
        caliper_host::services_shutdown();
        renderer_->shutdown();
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
        page_ = AppPage::Landing;
        glfwSetWindowTitle(window_, "Caliper");
        landing_.log("home", left);
    }

    // The live page's detach, after the frame that drew its views.
    void leave_live_page() {
        live_.detach();
        note_live_events();   // "detached"
        page_ = AppPage::Landing;
        glfwSetWindowTitle(window_, "Caliper");
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

    // POST /v0/applets/{id}/launch: opens that applet's page as a click on
    // its row does. The page shown goes first, as Home takes it: an applet
    // is torn down, a live session detached.
    void api_launch(const caliper_api::Call& c) {
        const int i = loader_.find(c.applet_id);
        if (i < 0) {
            api_.reply_error(c.ticket, 404,
                             "no applet with the id " + c.applet_id +
                                 " in the search folders (GET /v0/applets lists them)");
            return;
        }
        if (page_ == AppPage::Applet && active_applet_ == i) {
            const auto& a = loader_.at(i);
            api_.reply(c.ticket, 200,
                       caliper_host::launched_json(a.manifest.id, a.manifest.version, true));
            return;
        }
        if (loader_.at(i).status != caliper_host::AppletStatus::Ready) {
            const auto& a = loader_.at(i);
            api_.reply_error(c.ticket, 409,
                             c.applet_id + " cannot be launched (" +
                                 caliper_host::applet_status_word(a.status) + "): " +
                                 a.status_text);
            return;
        }
        if (page_ == AppPage::Applet) leave_applet_page(active_applet_);
        else if (page_ == AppPage::Live) leave_live_page();
        launch_applet(i);
        const auto& a = loader_.at(i);
        if (page_ == AppPage::Applet && active_applet_ == i)
            api_.reply(c.ticket, 200,
                       caliper_host::launched_json(a.manifest.id, a.manifest.version, false));
        else
            api_.reply_error(c.ticket, 409, c.applet_id + " failed to launch: " + a.status_text);
    }

    void launch_applet(int idx) {
        if (idx < 0 || idx >= loader_.count()) return;
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
    bool esc_home_pressed() const {
        const bool esc_free =
            !ImGui::GetIO().WantTextInput && !popup_open_prev_ &&
            !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId |
                                             ImGuiPopupFlags_AnyPopupLevel);
        return ImGui::IsKeyPressed(ImGuiKey_Escape, false) && esc_free;
    }

    // ---- Live page (F4 Attach to process) ---------------------------------

    // The landing's Attach (and CALIPER_AUTOATTACH): attach through a hosted
    // core over this window's renderer and open the live page. A refusal (the
    // session went, another viewer holds it, no Metal device) stays on the
    // landing and says why in Activity.
    void attach_session(const std::string& token) {
        const int i = sessions_.find(token);
        const caliper_live::SessionEntry* e = i >= 0 ? &sessions_.sessions()[(size_t)i] : nullptr;
        std::string script = e && !e->script.empty() ? e->script : token;
        std::string err;
        if (!live_.attach(renderer_.get(), token, &err)) {
            landing_.log("attach failed", script, err);
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

    // The attach pane's stream link (wire protocol v3, live_net_source.h):
    // attach through a hosted core as for a session; a refusal (nothing
    // listens, a wrong token) stays on the landing and says why in Activity.
    // The link holds the session token, so only the part before '?' is shown.
    void attach_stream(const std::string& link) {
        const std::string shown = link.substr(0, link.find('?'));
        std::string err;
        if (!live_.attach_url(renderer_.get(), link, &err)) {
            landing_.log("attach failed", shown, err);
            return;
        }
        caliper_live_views::set_look(live_.core(), house_view_look());
        const auto& st = live_.status();
        live_script_ = !st.script.empty() ? st.script : !st.session.empty() ? st.session : shown;
        live_pid_ = st.pid;
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
        attach_session(token);
    }

    // The attachment's transitions as Activity rows (and one stderr line
    // each, like [autoattach], for headless runs): attached; session ended
    // (the script exited or died, and the page keeps its last frames);
    // detached (esc Home).
    void note_live_events() {
        using Event = caliper_live::Attachment::Event;
        for (const Event ev : live_.take_events()) {
            const std::string pid = "pid " + std::to_string(live_pid_);
            const char* what = "";
            switch (ev) {
                case Event::Attached:
                    what = "attached";
                    landing_.log(what, live_script_, pid);
                    break;
                case Event::Ended:
                    what = "session ended";
                    landing_.log(what, live_script_, pid + " exited; last frames kept");
                    set_live_title();
                    break;
                case Event::Detached:
                    what = "detached";
                    landing_.log(what, live_script_, pid);
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
    // the same page bar and status bar. esc Home detaches after the frame.
    void draw_live_page() {
        const auto& st = live_.status();
        bool go_back = esc_home_pressed();
        // Dev hook: CALIPER_HOME_AFTER=<sec>, as on the applet page: Home once,
        // N sec after the live page opened (a headless detach).
        if (const char* ha = std::getenv("CALIPER_HOME_AFTER")) {
            static double page_t0 = -1.0;
            if (page_t0 < 0) page_t0 = glfwGetTime();
            if (glfwGetTime() - page_t0 >= std::atof(ha)) go_back = true;
        }

        // Page bar: esc Home, the script, then its state. The live diamond is
        // filled and breathes while the script runs and is outlined once it
        // has gone (SPEC §6.5); SESSION ENDED is in capitals, in parchment, as
        // the jobs tray writes FAILED (bronze stays for focus and selection).
        if (caliper_host::begin_page_bar()) {
            if (sternbild::key_button("esc", "Home")) go_back = true;
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
            ImGui::EndMainMenuBar();
        }

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

        if (go_back) pending_detach_ = true;
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
    std::string live_script_;
    int32_t live_pid_ = 0;
    bool pending_detach_ = false;     // esc Home on the live page: detach after render
    caliper_host::AppApi api_;        // Caliper API (--api); not running without it
    caliper_host::WindowCaptures captures_;   // the API's captures of the window's frame
    double live_frame_t_ = 0.0;       // start of the live page's current frame
    int autoattach_pid_ = 0;          // CALIPER_AUTOATTACH, until attached
    std::unique_ptr<caliper_live::UiDriver> ui_driver_;   // CALIPER_TEST_UI with --hidden
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

    // --hidden or CALIPER_HIDDEN=1: the real app with its window never shown.
    const char* hidden_env = std::getenv("CALIPER_HIDDEN");
    const bool hidden = cl.hidden || (hidden_env && std::strcmp(hidden_env, "1") == 0);
    CaliperApp app(cl.applet_dirs, hidden);
    // initialize() prints its own refusal, such as the renderer's.
    if (!app.initialize()) return 1;
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
