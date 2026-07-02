#include <iostream>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

#include "intro_screen.h"
#include "host/applet_loader.h"
#include "host/host_services.h"
#include "host/job_system.h"
#include "host/host_version.h"
#include "host/frame_watchdog.h"
#include "host/runs_dashboard.h"
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
        // window exists; init() runs after. CALIPER_RENDERER=metal selects the
        // Metal backend (Apple only); GL stays the default until the 2D flip.
        const char* want = std::getenv("CALIPER_RENDERER");
        bool want_metal = want && std::strcmp(want, "metal") == 0;
#ifdef __APPLE__
        if (want_metal) renderer_ = caliper_host::make_metal_renderer();
#else
        (void)want_metal;
#endif
        if (!renderer_) renderer_ = caliper_host::make_renderer("gl");

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        int ax, ay, aw, ah;
        glfwGetMonitorWorkarea(monitor, &ax, &ay, &aw, &ah);
        float sx = 1.0f, sy = 1.0f;
        glfwGetMonitorContentScale(monitor, &sx, &sy);
        int ww = (int)((aw / sx) * 0.95f);
        int wh = (int)((ah / sy) * 0.95f);

        // Host-owned ImGui/ImPlot contexts must exist before the renderer
        // initializes its ImGui backends. None of these touch GL/Metal.
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImPlot3D::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

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
        }

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

        return true;
    }

    void run() {
        while (!glfwWindowShouldClose(window_)) {
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
                // The Landing page has no menu bar of its own; add a minimal one
                // solely to host the always-reachable "Runs" toggle (the Applet
                // page toggles it from its existing menu bar below).
                if (ImGui::BeginMainMenuBar()) {
                    if (ImGui::MenuItem("Runs", nullptr, runs_open_))
                        runs_open_ = !runs_open_;
                    ImGui::EndMainMenuBar();
                }
                intro_.draw_ui(dw, dh);
                if (intro_.should_launch()) {
                    intro_.reset_launch_flag();
                    int idx = intro_.selected_index();
                    if (idx >= 0 && idx < loader_.count()) {
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
                }
            } else if (page_ == AppPage::Applet) {
                bool go_back = glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS;

                if (ImGui::BeginMainMenuBar()) {
                    if (ImGui::MenuItem("< Home")) go_back = true;
                    if (ImGui::MenuItem("Runs", nullptr, runs_open_))
                        runs_open_ = !runs_open_;
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

                if (go_back) {
                    loader_.teardown(active_applet_);
                    active_applet_ = -1;
                    page_ = AppPage::Landing;
                    glfwSetWindowTitle(window_, "Caliper");
                    // refresh cards so refusal/quarantine text shows up
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

            // Runs dashboard (B5): host UI over the metrics store, toggled from
            // the menu bar on either page; queries only while runs_open_.
            caliper_host::render_runs_dashboard(
                caliper_host::host_metrics_store(), &runs_open_);

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
        }
    }

    void cleanup() {
        loader_.close_all();
        intro_.cleanup();
        renderer_->shutdown();
        ImPlot3D::DestroyContext();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }

private:
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
    caliper_host::AppletLoader loader_{
        caliper_host::HostCaps{2, caliper_host::kHostVersionStr,
                               caliper_host::service_ids()},
        caliper::app_data_path("data")};
    caliper_host::FrameWatchdog watchdog_;
    int active_applet_ = -1;
    double last_frame_time_ = 0.0;
    bool runs_open_ = false;
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
