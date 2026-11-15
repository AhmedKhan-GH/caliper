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
#include "host/shot.h"
#include "host/host_services.h"
#include "host/core_lifecycle.h"
#include "host/job_system.h"
#include "host/host_version.h"
#include "host/renderer/host_renderer.h"
#include "host/live/live_attach.h"
#include "app_paths.h"

#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

enum class AppPage {
    Landing,
    Applet,
    Live,     // an attached training script's views (F4 Attach to process)
};

class CaliperApp {
public:
    // `applet_dirs`: the --applets folders, searched first.
    explicit CaliperApp(std::vector<std::string> applet_dirs)
        : applet_dirs_(std::move(applet_dirs)) {}

    bool initialize() {
        if (!glfwInit()) {
            std::cerr << "GLFW init failed" << std::endl;
            return false;
        }

        // Renderer seam (docs/explanations/rendering.md): backend selection (env
        // CALIPER_RENDERER + platform default, GL fallback) now lives in
        // libcaliper's core_lifecycle so every embedder gets the same policy.
        // The host still owns the window and drives init()/the fallback below.
        renderer_ = caliper_host::core_select_renderer();

        // Display asleep / headless-ish states can return NULL here — fall
        // back to a sane default window instead of crashing at startup.
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        int ww = 1440, wh = 900;
        if (monitor) {
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
        ImGui::GetIO().IniFilename = ini.c_str();

        // Sternbild house style. Faces first: apply_style() reads the UI font
        // size, and the first NewFrame() fixes style.FontSizeBase from the
        // default font. The 1.92 dynamic atlas takes fonts before the backend
        // exists, so this precedes renderer_->init().
        sternbild::load_fonts();
        sternbild::apply_style();

        // Backends whose window hints differ (Metal = NO_API, GL = a GL
        // context) cannot share a window, so a failed backend init recreates
        // the window for the fallback. GL is the guaranteed fallback.
        if (!create_window_and_init(ww, wh)) {
            if (std::string(renderer_->name()) != "gl") {
                std::cerr << "[renderer] " << renderer_->name()
                          << " init failed; falling back to gl" << std::endl;
                renderer_ = caliper_host::make_renderer("gl");
                if (!create_window_and_init(ww, wh)) {
                    glfwTerminate();
                    return false;
                }
            } else {
                glfwTerminate();
                return false;
            }
        }
        std::cerr << "[renderer] " << renderer_->name() << std::endl;
        landing_.log("renderer", renderer_->name());
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
            if (page_ == AppPage::Live) wait_for_live_frame();
            glfwPollEvents();

            // An attached session: claim and apply the newest completed
            // snapshot of every view before the frame draws them (the live
            // core's order: poll, then frame). Never waits on the script.
            if (page_ == AppPage::Live) live_.poll();
            if (autoattach_pid_ > 0 && page_ == AppPage::Landing) autoattach();
            note_live_events();

            int dw, dh;
            glfwGetFramebufferSize(window_, &dw, &dh);

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

            // Deferred detach (esc Home on the live page), for the same reason
            // as the applet teardown below: the views' textures and imports are
            // released only after the frame that drew them has been rendered.
            if (pending_detach_) {
                pending_detach_ = false;
                live_.detach();
                note_live_events();   // "detached"
                page_ = AppPage::Landing;
                glfwSetWindowTitle(window_, "Caliper");
            }

            // Deferred applet teardown — runs ONLY after this frame's draw list
            // has been rendered, so no just-released bridge texture is still
            // referenced by the Metal pass. Order: hard-join workers (they
            // reference applet state and the applet's own cleanup wait is a
            // timeout, not a join), tear the applet down, then return to the
            // landing page (it reads statuses live, so nothing to refresh).
            if (pending_home_ >= 0) {
                const std::string left = loader_.at(pending_home_).manifest.name;
                // Joining drops every job entry, failed ones included, so a
                // failure since the tray last looked is recorded first.
                note_failed_jobs(caliper_host::host_job_system().views());
                caliper_host::host_job_system().cancel_all_and_join();
                loader_.teardown(pending_home_);
                pending_home_ = -1;
                active_applet_ = -1;
                page_ = AppPage::Landing;
                glfwSetWindowTitle(window_, "Caliper");
                landing_.log("home", left);
            }
        }
    }

    void cleanup() {
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
        const auto& st = live_.status();
        live_script_ = st.script.empty() ? script : st.script;
        live_pid_ = st.pid ? st.pid : (e ? e->pid : 0);
        live_frame_t_ = 0.0;
        page_ = AppPage::Live;
        set_live_title();
        note_live_events();   // "attached"
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
        live_.draw(area.Min.x, area.Min.y, area.GetWidth(), area.GetHeight(),
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

    // (Re)create the GLFW window for the current renderer and init the backend.
    // glfwDefaultWindowHints() clears sticky hints from a prior attempt (e.g.
    // Metal's GLFW_NO_API) so the fallback backend gets a clean slate.
    bool create_window_and_init(int ww, int wh) {
        if (window_) { glfwDestroyWindow(window_); window_ = nullptr; }
        glfwDefaultWindowHints();
        renderer_->window_hints();
        window_ = glfwCreateWindow(ww, wh, "Caliper", nullptr, nullptr);
        if (!window_) return false;
        return renderer_->init(window_);
    }

    std::vector<std::string> applet_dirs_;   // --applets, in order
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
    double live_frame_t_ = 0.0;       // start of the live page's current frame
    int autoattach_pid_ = 0;          // CALIPER_AUTOATTACH, until attached
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
                                   cl.command == Command::Install;
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

    CaliperApp app(cl.applet_dirs);
    if (!app.initialize()) {
        std::cerr << "Initialization failed" << std::endl;
        return 1;
    }

    app.run();
    app.cleanup();
    return 0;
}
