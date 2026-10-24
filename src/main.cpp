#include <iostream>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>   // DockBuilder* — first-run applet dock layout
#include <implot.h>
#include <implot3d.h>

#include "ui/landing_screen.h"
#include "ui/sternbild_fonts.h"
#include "ui/sternbild_theme.h"
#include "ui/ui_recovery.h"
#include "host/applet_loader.h"
#include "host/host_services.h"
#include "host/core_lifecycle.h"
#include "host/job_system.h"
#include "host/host_version.h"
#include "host/frame_watchdog.h"
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
#ifdef __APPLE__
  #include <mach-o/dyld.h>
#elif defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

enum class AppPage {
    Landing,
    Applet,
    Live,     // an attached training script's views (F4 Attach to process)
};

class CaliperApp {
public:
    CaliperApp() = default;

    bool initialize() {
        if (!glfwInit()) {
            std::cerr << "GLFW init failed" << std::endl;
            return false;
        }

        // Renderer seam (PLATFORM.md §5.4): backend selection (env
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
        // Hand the live renderer to the bridge (caliper.tensor_bridge.v1); it is
        // cleared before renderer teardown in cleanup(). Do this right after the
        // renderer is up so the first applet frame can vend textures.
        caliper_host::services_set_renderer(renderer_.get());

        // Scan for applet shared libraries
        std::string applets_dir = caliper::app_data_path("applets");
        log_scan(applets_dir, loader_.scan(applets_dir));

        // Also scan applets next to the executable (dev + prod)
        {
            namespace fs = std::filesystem;
            std::string exe_dir;
#ifdef __APPLE__
            char path_buf[1024];
            uint32_t path_size = sizeof(path_buf);
            if (_NSGetExecutablePath(path_buf, &path_size) == 0)
                exe_dir = fs::path(path_buf).parent_path().string();
#elif defined(_WIN32)
            char path_buf[MAX_PATH];
            GetModuleFileNameA(nullptr, path_buf, MAX_PATH);
            exe_dir = fs::path(path_buf).parent_path().string();
#else
            exe_dir = fs::read_symlink("/proc/self/exe", std::error_code{})
                          .parent_path().string();
#endif
            if (!exe_dir.empty()) {
                const std::string dir = (fs::path(exe_dir) / "applets").string();
                log_scan(dir, loader_.scan(dir));
            }
        }

        // Dev hook: CALIPER_AUTOLAUNCH=<manifest id> skips the landing page
        // and launches that applet immediately — reproduces a row launch for
        // headless debugging/CI and seeds the future `caliper dev` workflow.
        if (const char* want_id = std::getenv("CALIPER_AUTOLAUNCH")) {
            for (int i = 0; i < loader_.count(); i++) {
                if (loader_.at(i).manifest.id == want_id) {
                    std::cerr << "[autolaunch] " << want_id << std::endl;
                    landing_.log("autolaunch", want_id);
                    launch_applet(i);
                    break;
                }
            }
        }

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

                // Applet bar: Home as one key button holding its key, esc
                // (recognition over recall, nothing printed beside it), the
                // applet's name as metadata, and the watchdog warning in
                // bronze, the palette's only accent. The page's one action
                // lives here; the status bar below carries only facts.
                if (begin_page_bar()) {
                    if (sternbild::key_button("esc", "Home")) go_back = true;
                    ImGui::Separator();
                    ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
                    ImGui::TextUnformatted(
                        loader_.at(active_applet_).manifest.name.c_str());
                    ImGui::PopStyleColor();
                    if (watchdog_.flagged()) {
                        ImGui::Separator();
                        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::bronze);
                        ImGui::TextUnformatted(
                            "slow: long work belongs in background jobs");
                        ImGui::PopStyleColor();
                    }
                    ImGui::EndMainMenuBar();
                }

                // Status bar (VS Code): the same persistent strip as the
                // landing's, so the machine is always in the same place; facts
                // only, as there (Home is in the applet bar above). Like the
                // menu bar it is a viewport side bar, drawn before the dock
                // space that fills what it leaves.
                sternbild::status_bar_begin();
                draw_host_status_items(renderer_->name());
                {
                    int running = 0;
                    for (const auto& v : caliper_host::host_job_system().views())
                        running += v.running ? 1 : 0;
                    char jobs[48];
                    if (running == 0)
                        std::snprintf(jobs, sizeof jobs, "no jobs");
                    else
                        std::snprintf(jobs, sizeof jobs,
                                      running == 1 ? "%d job running"
                                                   : "%d jobs running",
                                      running);
                    sternbild::status_item(jobs);
                }
                sternbild::status_bar_end();

                // Docked desktop (applet page only). A passthru dockspace fills
                // the viewport's work area (between the main menu bar and the
                // status bar) so the applet's windows can be docked. The
                // central node stays transparent while empty, so a
                // full-viewport applet docked into it looks exactly as before.
                ImGuiID dock_id = ImGui::DockSpaceOverViewport(
                    0, ImGui::GetMainViewport(),
                    ImGuiDockNodeFlags_PassthruCentralNode);

                // Dock-by-title layout, attempted once per process. Two cases:
                //  - fresh dockspace (no imgui.ini history): build the full
                //    tiled layout (showcase central, panels in a side column,
                //    Runs right).
                //  - restored dockspace: dock ONLY windows that have no saved
                //    settings yet (e.g. an applet grew new windows since the
                //    ini was written). Anything the user ever placed is
                //    untouched — their layout wins.
                if (!dock_layout_built_) {
                    // Titles are the exact Begin() strings from applet source.
                    static const char* const central_windows[] = {
                        "EmbedScope: Cloud",   // the showcase
                        "GPTScope: Logit Lens",// the mech-interp centerpiece
                        "GPTScope: ThoughtSpace", // residual constellation (3-D)
                        "FlowScope: Field",    // the geometry.v1 exemplar
                        "FieldScope: EM Field",// PIC plasma — the physics exemplar
                        "SculptScope: Learned Cloud", // net's own tensor drawn live
                        "MeshScope: Surface", // indexed triangles + lines via geometry.v1_1
                        "Hello, Caliper",      // examples/hello (tutorial)
                        "SineScope",           // examples/sine_scope (tutorial)
                    };
                    static const char* const side_top_windows[] = {
                        "EmbedScope: Training", "GPTScope: Training"};
                    static const char* const side_bot_windows[] = {
                        "EmbedScope: Tensors", "EmbedScope: Data",
                        "GPTScope: Heads", "GPTScope: Embeddings",
                        "GPTScope: Residual", "GPTScope: Sample"};
                    auto unseen = [](const char* w) {
                        return ImGui::FindWindowSettingsByID(ImHashStr(w)) ==
                               nullptr;
                    };
                    ImGuiDockNode* node = ImGui::DockBuilderGetNode(dock_id);
                    const bool fresh = node && node->IsEmpty();
                    bool any_unseen = false;
                    for (const char* w : central_windows) any_unseen |= unseen(w);
                    for (const char* w : side_top_windows) any_unseen |= unseen(w);
                    for (const char* w : side_bot_windows) any_unseen |= unseen(w);
                    if (node && (fresh || any_unseen)) {
                        ImGuiID central = dock_id;
                        ImGuiID right = 0;
                        if (fresh) {
                            ImGui::DockBuilderSetNodeSize(
                                dock_id, ImGui::GetMainViewport()->WorkSize);
                            right = ImGui::DockBuilderSplitNode(
                                central, ImGuiDir_Right, 0.32f, nullptr,
                                &central);
                        } else if (ImGuiDockNode* c =
                                       ImGui::DockBuilderGetCentralNode(dock_id)) {
                            central = c->ID;
                        }
                        ImGuiID side = ImGui::DockBuilderSplitNode(
                            central, ImGuiDir_Right, 0.30f, nullptr, &central);
                        ImGuiID side_bot = ImGui::DockBuilderSplitNode(
                            side, ImGuiDir_Down, 0.40f, nullptr, &side);
                        for (const char* w : central_windows)
                            if (fresh || unseen(w))
                                ImGui::DockBuilderDockWindow(w, central);
                        for (const char* w : side_top_windows)
                            if (fresh || unseen(w))
                                ImGui::DockBuilderDockWindow(w, side);
                        for (const char* w : side_bot_windows)
                            if (fresh || unseen(w))
                                ImGui::DockBuilderDockWindow(w, side_bot);
                        ImGui::DockBuilderFinish(dock_id);
                    }
                    dock_layout_built_ = true;
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

                // The UI stacks are checkpointed around the applet's frame
                // (ui_recovery.h): a throw or fault partway through it leaves
                // its windows, plots and pushes open, and they are unwound
                // before the host draws anything more into this frame.
                const caliper_host::UiCheckpoint ui_checkpoint =
                    caliper_host::UiCheckpoint::store();
                double t0 = glfwGetTime();
                bool alive = loader_.frame(active_applet_, fi);
                watchdog_.feed((glfwGetTime() - t0) * 1000.0);

                if (!alive) {                 // quarantined mid-frame
                    ui_checkpoint.recover();
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

            // Jobs tray (§7.5): visible on every page while jobs exist. A
            // plate pinned to the bottom-right of the WORK area, so it sits
            // above the status bar instead of over it; sized to its content,
            // never docked, never saved. Actions are house words, not boxes.
            // Running jobs first, then failures (kept until cleared, see
            // note_failed_jobs), then the jobs that finished.
            {
                auto views = caliper_host::host_job_system().views();
                note_failed_jobs(views);
                if (!views.empty() || !failed_jobs_.empty()) {
                    const ImGuiViewport* vp = ImGui::GetMainViewport();
                    constexpr float inset = 12.0f;
                    constexpr float bar_w = 240.0f;
                    ImGui::SetNextWindowPos(
                        {vp->WorkPos.x + vp->WorkSize.x - inset,
                         vp->WorkPos.y + vp->WorkSize.y - inset},
                        ImGuiCond_Always, {1.0f, 1.0f});
                    ImGui::SetNextWindowViewport(vp->ID);
                    ImGui::PushStyleColor(ImGuiCol_WindowBg, sternbild::col::plate);
                    ImGui::Begin("Jobs", nullptr,
                                 ImGuiWindowFlags_NoTitleBar |
                                     ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoCollapse |
                                     ImGuiWindowFlags_NoDocking |
                                     ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_AlwaysAutoResize);
                    ImGui::PopStyleColor();
                    sternbild::label("Jobs");
                    bool any_done = !failed_jobs_.empty();
                    for (auto& v : views) any_done |= !v.running;
                    if (any_done) {
                        ImGui::SameLine(0.0f, sternbild::metric::gap);
                        if (sternbild::word_button("clear finished")) {
                            caliper_host::host_job_system().clear_finished();
                            failed_jobs_.clear();
                        }
                    }
                    auto job_row = [&](const caliper_host::JobSystem::JobView& v) {
                        ImGui::PushID((int)v.id);
                        ImGui::TextUnformatted(v.label.c_str());
                        // Fixed width: the tray auto-sizes, so a bar sized
                        // from the window width would feed back into it.
                        ImGui::ProgressBar(v.progress, {bar_w, 0.0f},
                                           v.message.empty() ? nullptr
                                                             : v.message.c_str());
                        if (v.running) {
                            ImGui::SameLine(0.0f, sternbild::metric::gap);
                            if (sternbild::word_button("cancel"))
                                caliper_host::host_job_system().request_cancel(v.id);
                        }
                        ImGui::PopID();
                    };
                    for (auto& v : views)
                        if (v.running) job_row(v);
                    // A failure in place of its bar: the status word in
                    // capitals, as the Instruments STATUS column writes it,
                    // in parchment, the strongest ink after bronze (which
                    // stays for focus and selection; the palette has no red),
                    // then the reason in secondary, wrapped at the bar's
                    // width so a long one never widens the tray. Newest first.
                    for (auto it = failed_jobs_.rbegin(); it != failed_jobs_.rend();
                         ++it) {
                        ImGui::PushID((int)it->id);
                        ImGui::TextUnformatted(it->label.c_str());
                        const float row_x = ImGui::GetCursorPosX();
                        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::parchment);
                        ImGui::TextUnformatted("FAILED");
                        ImGui::PopStyleColor();
                        ImGui::SameLine(0.0f, sternbild::metric::pad);
                        ImGui::PushTextWrapPos(row_x + bar_w);
                        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
                        ImGui::TextUnformatted(it->reason.c_str());
                        ImGui::PopStyleColor();
                        ImGui::PopTextWrapPos();
                        ImGui::PopID();
                    }
                    for (auto& v : views)
                        if (!v.running && !v.failed) job_row(v);
                    ImGui::End();
                }
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
    static CaliperHost host_proto() {
        CaliperHost proto{};
        proto.struct_size  = sizeof(CaliperHost);
        proto.abi_epoch    = 2;
        proto.host_version = caliper_host::kHostVersionU32;
        proto.applet_data_dir = nullptr;   // loader fills per-applet
        proto.get_service = [](const CaliperHost*, const char* id) {
            return caliper_host::services_get(id);
        };
        return proto;
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
        if (loader_.launch(idx, host_proto())) {
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
        if (loader_.restart(idx, host_proto())) {
            enter_applet(idx);
        } else {
            ui_checkpoint.recover();
            landing_.log("restart failed", name, loader_.at(idx).status_text);
        }
    }

    void enter_applet(int idx) {
        active_applet_ = idx;
        watchdog_.reset();
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

    // The page bar of the applet and live pages: a main menu bar one frame
    // tall, so a taller frame padding gives it the landing command bar's
    // height and its key button the same margins as there. EndMainMenuBar()
    // when it returns true.
    static bool begin_page_bar() {
        const float bar_pad_y =
            (sternbild::metric::command_bar - ImGui::GetFontSize()) * 0.5f;
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                            ImVec2(ImGui::GetStyle().FramePadding.x, bar_pad_y));
        const bool open = ImGui::BeginMainMenuBar();
        ImGui::PopStyleVar();
        return open;
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
        if (begin_page_bar()) {
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

        // Its own dock space id, so the applet page's docked layout never
        // shows through on this page (and the reverse).
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::DockSpaceOverViewport(ImGui::GetID("caliper_live_dockspace"), vp,
                                     ImGuiDockNodeFlags_PassthruCentralNode);
        live_.draw(vp->WorkPos.x, vp->WorkPos.y, vp->WorkSize.x, vp->WorkSize.y,
                   ImGui::GetIO().DisplayFramebufferScale.x);

        if (go_back) pending_detach_ = true;
    }

    // A failed job (JobView::failed) is kept here, for the tray, until the
    // user clears finished jobs, and written to Activity once. The job
    // system's own entry does not last that long: going Home joins the
    // applet's jobs and drops every entry, and a failed applet job is what
    // quarantines its applet and sends the host Home. Job ids never repeat.
    void note_failed_jobs(const std::vector<caliper_host::JobSystem::JobView>& views) {
        for (const auto& v : views) {
            if (!v.failed) continue;
            bool known = false;
            for (const auto& f : failed_jobs_) known |= f.id == v.id;
            if (known) continue;
            // JobSystem writes the reason as the message, "failed: <why>".
            std::string reason = v.message;
            constexpr const char* kPrefix = "failed: ";
            if (reason.rfind(kPrefix, 0) == 0) reason.erase(0, std::strlen(kPrefix));
            if (reason.empty()) reason = "no reason given";
            landing_.log("job failed", v.label, reason);
            failed_jobs_.push_back({v.id, v.label, std::move(reason)});
        }
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
        if (next != '/' && next != '\\') return path;   // "/Users/ahmedx" is not home
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

    GLFWwindow* window_ = nullptr;
    std::unique_ptr<caliper_host::HostRenderer> renderer_;
    AppPage page_ = AppPage::Landing;
    LandingScreen landing_;
    caliper_host::AppletLoader loader_{
        caliper_host::HostCaps{2, caliper_host::kHostVersionStr,
                               caliper_host::service_ids()},
        caliper::app_data_path("data")};
    caliper_host::FrameWatchdog watchdog_;
    int active_applet_ = -1;
    int pending_home_ = -1;   // applet idx awaiting post-render teardown (Home)
    double last_frame_time_ = 0.0;
    struct FailedJob {
        uint64_t id;
        std::string label;
        std::string reason;   // the message without its "failed: " prefix
    };
    std::vector<FailedJob> failed_jobs_;   // oldest first; see note_failed_jobs
    bool dock_layout_built_ = false;  // first-run applet dock layout attempted
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
};

int main() {
    std::cout << "=== Caliper ===" << std::endl;

    CaliperApp app;
    if (!app.initialize()) {
        std::cerr << "Initialization failed" << std::endl;
        return 1;
    }

    app.run();
    app.cleanup();
    return 0;
}
