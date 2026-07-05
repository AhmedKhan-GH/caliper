#include <iostream>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>   // DockBuilder* — first-run applet dock layout
#include <implot.h>
#include <implot3d.h>

#include "intro_screen.h"
#include "host/applet_loader.h"
#include "host/host_services.h"
#include "host/job_system.h"
#include "host/host_version.h"
#include "host/frame_watchdog.h"
#include "host/renderer/host_renderer.h"
#include "app_paths.h"

#include <string>
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
};

class CaliperApp {
public:
    CaliperApp() = default;

    bool initialize() {
        if (!glfwInit()) {
            std::cerr << "GLFW init failed" << std::endl;
            return false;
        }

        // Renderer seam (PLATFORM.md §5.4): backend hints run before the
        // window exists; init() runs after. On Apple the default is now Metal
        // (device-resident tensors, zero CPU staging) — the opengllama migration
        // retired the last raw-GL applet, clearing the flip gate. CALIPER_RENDERER=gl
        // selects the frozen GL fallback; off Apple, GL is the only backend.
        const char* want = std::getenv("CALIPER_RENDERER");
        bool want_gl = want && std::strcmp(want, "gl") == 0;
#ifdef __APPLE__
        if (!want_gl) renderer_ = caliper_host::make_metal_renderer();
#elif defined(_WIN32)
        // Windows default is Vulkan (Phase 4: device-resident CUDA tensors
        // via external-memory interop). Same fallback contract as Metal:
        // if init() fails (no Vulkan driver, RDP, ...) GL takes over below.
        if (!want_gl) renderer_ = caliper_host::make_vulkan_renderer();
#else
        (void)want_gl;
#endif
        if (!renderer_) renderer_ = caliper_host::make_renderer("gl");

        // Display asleep / headless-ish states can return NULL here — fall
        // back to a sane default window instead of crashing at startup.
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        int ww = 1440, wh = 900;
        if (monitor) {
            int ax, ay, aw, ah;
            glfwGetMonitorWorkarea(monitor, &ax, &ay, &aw, &ah);
            float sx = 1.0f, sy = 1.0f;
            glfwGetMonitorContentScale(monitor, &sx, &sy);
            ww = (int)((aw / sx) * 0.95f);
            wh = (int)((ah / sy) * 0.95f);
        }

        // Host-owned ImGui/ImPlot contexts must exist before the renderer
        // initializes its ImGui backends. None of these touch GL/Metal.
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImPlot3D::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        // Docking (ImGui docking branch): the applet page hosts a docked
        // desktop — applet windows tile into a central node + side column.
        // Persisted per-window/dockspace layout lives in imgui.ini (host-owned).
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        ImGui::StyleColorsDark();
        style_ui();

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
        caliper_host::services_init();
        // Hand the live renderer to the bridge (caliper.tensor_bridge.v1); it is
        // cleared before renderer teardown in cleanup(). Do this right after the
        // renderer is up so the first applet frame can vend textures.
        caliper_host::services_set_renderer(renderer_.get());

        // IntroScreen is raw-GL end to end (its initialize/render_3d/cleanup all
        // issue GL). On non-GL backends there is no GL context, so skip init:
        // every other IntroScreen entry point early-outs when its state is null,
        // so update()/draw_ui()/cleanup() stay crash-free. The 3D landing bg and
        // card launcher return with the GL->2D migration (2D). render_3d is
        // already guarded below; initialize() needs the same guard.
        if (std::string(renderer_->name()) == "gl") {
            if (!intro_.initialize()) {
                std::cerr << "Intro screen init failed" << std::endl;
                return false;
            }
            intro_active_ = true;
        }
        // On non-GL backends the launcher must not depend on IntroScreen:
        // draw_fallback_launcher() renders a plain-ImGui card list instead.

        // Scan for applet shared libraries
        std::string applets_dir = caliper::app_data_path("applets");
        loader_.scan(applets_dir);

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
            if (!exe_dir.empty())
                loader_.scan((fs::path(exe_dir) / "applets").string());
        }

        // Populate intro screen cards from loaded applets
        std::vector<AppletCard> cards;
        for (int i = 0; i < loader_.count(); i++) {
            const auto& e = loader_.at(i);
            std::string desc = e.manifest.summary;
            if (e.status != caliper_host::AppletStatus::Ready &&
                e.status != caliper_host::AppletStatus::Active)
                desc = "[unavailable] " + e.status_text + "\n\n" + desc;
            cards.push_back({e.manifest.name, e.manifest.summary, desc,
                             e.manifest.tag});
        }
        intro_.set_applets(std::move(cards));

        // Dev hook: CALIPER_AUTOLAUNCH=<manifest id> skips the landing page
        // and launches that applet immediately — reproduces a card click for
        // headless debugging/CI and seeds the future `caliper dev` workflow.
        if (const char* want_id = std::getenv("CALIPER_AUTOLAUNCH")) {
            for (int i = 0; i < loader_.count(); i++) {
                if (loader_.at(i).manifest.id == want_id) {
                    std::cerr << "[autolaunch] " << want_id << std::endl;
                    launch_applet(i);
                    break;
                }
            }
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
            glfwPollEvents();

            int dw, dh;
            glfwGetFramebufferSize(window_, &dw, &dh);

            renderer_->new_frame();

            if (page_ == AppPage::Landing) {
                intro_.update(window_);
                // TODO(2D): dies with the backend flip — IntroScreen issues raw
                // GL, so it only runs on the GL backend; Metal skips the 3D bg.
                if (std::string(renderer_->name()) == "gl")
                    intro_.render_3d(dw, dh);
            }

            if (page_ == AppPage::Landing) {
                if (intro_active_) {
                    intro_.draw_ui(dw, dh);
                    if (intro_.should_launch()) {
                        intro_.reset_launch_flag();
                        launch_applet(intro_.selected_index());
                    }
                } else {
                    // Metal (and any future non-GL backend): the GL-only
                    // IntroScreen never initialized, so it owns no UI here.
                    // Reads loader_ live each frame — statuses always fresh.
                    draw_fallback_launcher();
                }
            } else if (page_ == AppPage::Applet) {
                // Docked desktop (applet page only). A passthru dockspace fills
                // the viewport's work area (below the main menu bar) so the
                // applet's windows can be docked. The central node stays
                // transparent while empty, so a full-viewport applet docked
                // into it looks exactly as before.
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
                        "Hello, Caliper",      // examples/hello
                        "MyScope",             // examples/my_scope (tutorial)
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

                bool go_back = glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS;

                // Dev hook: CALIPER_HOME_AFTER=<sec> forces Home once, N sec
                // after the applet page opened — reproduces "click Home during
                // training" headlessly (for the SIGSEGV repro).
                if (const char* ha = std::getenv("CALIPER_HOME_AFTER")) {
                    static double page_t0 = -1.0;
                    if (page_t0 < 0) page_t0 = glfwGetTime();
                    if (glfwGetTime() - page_t0 >= std::atof(ha)) go_back = true;
                }

                if (ImGui::BeginMainMenuBar()) {
                    if (ImGui::MenuItem("< Home")) go_back = true;
                    ImGui::Separator();
                    ImGui::TextDisabled("%s",
                        loader_.at(active_applet_).manifest.name.c_str());
                    if (watchdog_.flagged()) {
                        ImGui::Separator();
                        ImGui::TextColored({1.0f, 0.6f, 0.2f, 1.0f},
                            "slow: long work belongs in background jobs");
                    }
                    ImGui::EndMainMenuBar();
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

                double t0 = glfwGetTime();
                bool alive = loader_.frame(active_applet_, fi);
                watchdog_.feed((glfwGetTime() - t0) * 1000.0);

                if (!alive) go_back = true;   // quarantined mid-frame

                // Teardown is DEFERRED to after render() (see below): the
                // applet's draw list THIS frame may reference bridge textures,
                // and releasing them in cleanup() before the Metal render pass
                // consumes that draw list is a use-after-free (objc_retain on a
                // freed MTLTexture). We only flag it here; the applet keeps
                // drawing valid textures through this frame's render.
                if (go_back) pending_home_ = active_applet_;
            }

            // Jobs tray (§7.5): visible on every page while jobs exist.
            {
                auto views = caliper_host::host_job_system().views();
                if (!views.empty()) {
                    ImGuiIO& tio = ImGui::GetIO();
                    ImGui::SetNextWindowPos(
                        {tio.DisplaySize.x - 330.0f, tio.DisplaySize.y - 10.0f},
                        ImGuiCond_Always, {0.0f, 1.0f});
                    ImGui::SetNextWindowSize({320.0f, 0.0f});
                    ImGui::Begin("Jobs", nullptr,
                                 ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoCollapse);
                    for (auto& v : views) {
                        ImGui::PushID((int)v.id);
                        ImGui::Text("%s", v.label.c_str());
                        ImGui::ProgressBar(v.progress, {-60.0f, 0.0f},
                                           v.message.empty() ? nullptr
                                                             : v.message.c_str());
                        if (v.running) {
                            ImGui::SameLine();
                            if (ImGui::SmallButton("cancel"))
                                caliper_host::host_job_system().request_cancel(v.id);
                        }
                        ImGui::PopID();
                    }
                    ImGui::End();
                }
            }

            renderer_->render(dw, dh);

            // Deferred applet teardown — runs ONLY after this frame's draw list
            // has been rendered, so no just-released bridge texture is still
            // referenced by the Metal pass. Order: hard-join workers (they
            // reference applet state and the applet's own cleanup wait is a
            // timeout, not a join), tear the applet down, then return to the
            // landing page and refresh the cards.
            if (pending_home_ >= 0) {
                caliper_host::host_job_system().cancel_all_and_join();
                loader_.teardown(pending_home_);
                pending_home_ = -1;
                active_applet_ = -1;
                page_ = AppPage::Landing;
                glfwSetWindowTitle(window_, "Caliper");
                std::vector<AppletCard> cards;
                for (int i = 0; i < loader_.count(); i++) {
                    const auto& e = loader_.at(i);
                    std::string desc = e.manifest.summary;
                    if (e.status != caliper_host::AppletStatus::Ready)
                        desc = "[unavailable] " + e.status_text + "\n\n" + desc;
                    cards.push_back({e.manifest.name, e.manifest.summary,
                                     desc, e.manifest.tag});
                }
                intro_.set_applets(std::move(cards));
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
        // Applets are torn down first (they may release bridge textures while the
        // renderer is still live); THEN drop the renderer from the bridge before
        // renderer teardown, so no bridge thunk touches a destroyed renderer.
        caliper_host::services_set_renderer(nullptr);
        // Close the DuckDB stores NOW — leaving them to static destructors
        // races DuckDB's globals and aborts in malloc at exit.
        caliper_host::services_shutdown();
        intro_.cleanup();
        renderer_->shutdown();
        ImPlot3D::DestroyContext();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }

private:
    void launch_applet(int idx) {
        if (idx < 0 || idx >= loader_.count()) return;
        CaliperHost proto{};
        proto.struct_size  = sizeof(CaliperHost);
        proto.abi_epoch    = 2;
        proto.host_version = caliper_host::kHostVersionU32;
        proto.applet_data_dir = nullptr;   // loader fills per-applet
        proto.get_service = [](const CaliperHost*, const char* id) {
            return caliper_host::services_get(id);
        };
        if (loader_.launch(idx, proto)) {
            active_applet_ = idx;
            watchdog_.reset();
            last_frame_time_ = glfwGetTime();
            page_ = AppPage::Applet;
            glfwSetWindowTitle(window_,
                ("Caliper - " + loader_.at(idx).manifest.name).c_str());
        }
    }

    // Renderer-agnostic launcher for backends where the GL IntroScreen is
    // inactive. Plain ImGui: name/tag/summary/status + Launch per applet.
    void draw_fallback_launcher() {
        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("##launcher", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::TextDisabled("C A L I P E R");
        ImGui::Separator();
        if (loader_.count() == 0)
            ImGui::TextWrapped("No applets found. Drop a dylib + "
                               "<name>.caliper.toml into the applets folder.");
        for (int i = 0; i < loader_.count(); i++) {
            const auto& e = loader_.at(i);
            const bool ready = e.status == caliper_host::AppletStatus::Ready;
            ImGui::PushID(i);
            ImGui::Text("%s", e.manifest.name.c_str());
            if (!e.manifest.tag.empty()) {
                ImGui::SameLine();
                ImGui::TextDisabled("[%s]", e.manifest.tag.c_str());
            }
            ImGui::TextWrapped("%s", e.manifest.summary.c_str());
            if (!ready)
                ImGui::TextColored({1.0f, 0.55f, 0.35f, 1.0f},
                                   "[unavailable] %s", e.status_text.c_str());
            ImGui::BeginDisabled(!ready);
            if (ImGui::Button("Launch")) launch_applet(i);
            ImGui::EndDisabled();
            ImGui::Separator();
            ImGui::PopID();
        }
        ImGui::End();
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

    void style_ui() {
        ImGuiStyle& st = ImGui::GetStyle();
        st.WindowRounding = 6.0f;
        st.FrameRounding = 4.0f;
        st.GrabRounding = 3.0f;
        st.ScrollbarRounding = 4.0f;
        st.ItemSpacing = ImVec2(8, 5);

        auto* c = st.Colors;
        c[ImGuiCol_WindowBg]        = {0.09f, 0.09f, 0.12f, 0.97f};
        c[ImGuiCol_ChildBg]         = {0.11f, 0.11f, 0.15f, 1.00f};
        c[ImGuiCol_Header]          = {0.18f, 0.22f, 0.32f, 1.00f};
        c[ImGuiCol_HeaderHovered]   = {0.26f, 0.30f, 0.42f, 1.00f};
        c[ImGuiCol_HeaderActive]    = {0.22f, 0.26f, 0.38f, 1.00f};
        c[ImGuiCol_Button]          = {0.18f, 0.22f, 0.32f, 1.00f};
        c[ImGuiCol_ButtonHovered]   = {0.28f, 0.32f, 0.44f, 1.00f};
        c[ImGuiCol_ButtonActive]    = {0.14f, 0.18f, 0.28f, 1.00f};
        c[ImGuiCol_FrameBg]         = {0.14f, 0.14f, 0.20f, 1.00f};
        c[ImGuiCol_FrameBgHovered]  = {0.20f, 0.20f, 0.28f, 1.00f};
        c[ImGuiCol_SliderGrab]      = {0.40f, 0.55f, 0.80f, 1.00f};
        c[ImGuiCol_SliderGrabActive]= {0.50f, 0.65f, 0.90f, 1.00f};
        c[ImGuiCol_ScrollbarBg]     = {0.08f, 0.08f, 0.10f, 1.00f};
        c[ImGuiCol_ScrollbarGrab]   = {0.25f, 0.25f, 0.35f, 1.00f};
    }

    GLFWwindow* window_ = nullptr;
    std::unique_ptr<caliper_host::HostRenderer> renderer_;
    AppPage page_ = AppPage::Landing;
    IntroScreen intro_;
    bool intro_active_ = false;   // GL-only IntroScreen initialized (owns landing UI)
    caliper_host::AppletLoader loader_{
        caliper_host::HostCaps{2, caliper_host::kHostVersionStr,
                               caliper_host::service_ids()},
        caliper::app_data_path("data")};
    caliper_host::FrameWatchdog watchdog_;
    int active_applet_ = -1;
    int pending_home_ = -1;   // applet idx awaiting post-render teardown (Home)
    double last_frame_time_ = 0.0;
    bool dock_layout_built_ = false;  // first-run applet dock layout attempted
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
