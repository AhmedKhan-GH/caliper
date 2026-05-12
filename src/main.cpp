#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <implot.h>
#include <implot3d.h>

#include "intro_screen.h"
#include "applet_host.h"
#include "app_paths.h"

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

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        int ax, ay, aw, ah;
        glfwGetMonitorWorkarea(monitor, &ax, &ay, &aw, &ah);
        float sx = 1.0f, sy = 1.0f;
        glfwGetMonitorContentScale(monitor, &sx, &sy);
        int ww = (int)((aw / sx) * 0.95f);
        int wh = (int)((ah / sy) * 0.95f);

        window_ = glfwCreateWindow(ww, wh, "Caliper", nullptr, nullptr);
        if (!window_) { glfwTerminate(); return false; }

        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1);

        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) { glfwTerminate(); return false; }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImPlot3D::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        ImGui::StyleColorsDark();
        style_ui();

        ImGui_ImplGlfw_InitForOpenGL(window_, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        if (!intro_.initialize()) {
            std::cerr << "Intro screen init failed" << std::endl;
            return false;
        }

        // Scan for applet shared libraries
        std::string applets_dir = caliper::app_data_path("applets");
        host_.scan(applets_dir);

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
                host_.scan((fs::path(exe_dir) / "applets").string());
        }

        // Populate intro screen cards from loaded applets
        std::vector<AppletCard> cards;
        for (int i = 0; i < host_.count(); i++) {
            auto& info = host_[i].info;
            cards.push_back({
                info.name ? info.name : "",
                info.description ? info.description : "",
                info.description ? info.description : "",
                info.tag ? info.tag : "",
            });
        }
        intro_.set_applets(std::move(cards));

        return true;
    }

    void run() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();

            int dw, dh;
            glfwGetFramebufferSize(window_, &dw, &dh);
            glViewport(0, 0, dw, dh);
            glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            if (page_ == AppPage::Landing) {
                intro_.update(window_);
                intro_.render_3d(dw, dh);
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            if (page_ == AppPage::Landing) {
                intro_.draw_ui(dw, dh);
                if (intro_.should_launch()) {
                    intro_.reset_launch_flag();
                    int idx = intro_.selected_index();
                    if (idx >= 0 && idx < host_.count()) {
                        CaliperHostContext ctx{};
                        ctx.imgui    = ImGui::GetCurrentContext();
                        ctx.implot   = ImPlot::GetCurrentContext();
                        ctx.implot3d = ImPlot3D::GetCurrentContext();
                        ctx.data_dir = caliper::app_data_dir().c_str();

                        if (host_.launch(idx, ctx)) {
                            active_applet_ = idx;
                            page_ = AppPage::Applet;
                            glfwSetWindowTitle(window_,
                                ("Caliper - " + std::string(host_[idx].info.name)).c_str());
                        }
                    }
                }
            } else if (page_ == AppPage::Applet) {
                host_.draw(active_applet_, dw, dh);

                // Check for back-to-landing (Escape key for now)
                if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                    host_.teardown(active_applet_);
                    active_applet_ = -1;
                    page_ = AppPage::Landing;
                    glfwSetWindowTitle(window_, "Caliper");
                }
            }

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window_);
        }
    }

    void cleanup() {
        host_.close_all();
        intro_.cleanup();
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot3D::DestroyContext();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }

private:
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
    AppPage page_ = AppPage::Landing;
    IntroScreen intro_;
    AppletHost host_;
    int active_applet_ = -1;
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
