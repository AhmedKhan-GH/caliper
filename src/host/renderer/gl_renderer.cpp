// GL backend for the HostRenderer seam (PLATFORM.md §5.4). This is the
// extraction of the OpenGL/ImGui-backend code that used to live inline in
// src/main.cpp — behavior-identical. The Metal backend (C2) is a sibling
// translation unit implementing the same interface; make_renderer() picks.
#include "host_renderer.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <unordered_map>

namespace caliper_host {
namespace {

class GLRenderer final : public HostRenderer {
public:
    const char* name() const override { return "gl"; }

    // GL profile hints — must run before glfwCreateWindow (Metal uses NO_API).
    void window_hints() override {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    }

    bool init(GLFWwindow* window) override {
        window_ = window;
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) return false;

        // The ImGui core/ImPlot contexts are host-owned and already created by
        // the caller; only the GL+GLFW *backends* belong to this renderer.
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");
        return true;
    }

    // Start-of-frame clear + backend NewFrame. The clear lives here (not in
    // render()) because IntroScreen::render_3d draws a full-screen composite
    // straight to the default framebuffer between new_frame() and render();
    // clearing at present-time would wipe it. Applet page has no render_3d, so
    // this clear is its background — identical to the old top-of-loop clear.
    void new_frame() override {
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void render(int fb_w, int fb_h) override {
        glViewport(0, 0, fb_w, fb_h);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window_);
    }

    void shutdown() override {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
    }

    // ---- Texture ops (classic GL donor pattern). id table maps an opaque
    // uint64 → GL name; the raw GL handle never leaves this file. ----
    uint64_t tex_create_rgba8(int w, int h) override {
        if (w <= 0 || h <= 0) return 0;
        GLuint gl_name = 0;
        glGenTextures(1, &gl_name);
        if (gl_name == 0) return 0;
        glBindTexture(GL_TEXTURE_2D, gl_name);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        uint64_t id = next_id_++;
        textures_[id] = Tex{gl_name, w, h};
        return id;
    }

    bool tex_upload_rgba8(uint64_t tex, const void* data, int w, int h) override {
        auto it = textures_.find(tex);
        if (it == textures_.end() || data == nullptr) return false;
        glBindTexture(GL_TEXTURE_2D, it->second.gl_name);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA,
                        GL_UNSIGNED_BYTE, data);
        glBindTexture(GL_TEXTURE_2D, 0);
        return true;
    }

    void tex_release(uint64_t tex) override {
        auto it = textures_.find(tex);
        if (it == textures_.end()) return;
        glDeleteTextures(1, &it->second.gl_name);
        textures_.erase(it);
    }

    uint64_t tex_imtexture_id(uint64_t tex) override {
        auto it = textures_.find(tex);
        return it == textures_.end() ? 0 : (uint64_t)it->second.gl_name;
    }

    // GL is the frozen staged fallback: it never reads device memory directly,
    // so the caller always stages through CPU (tex_upload_rgba8).
    bool tex_update_from_device(uint64_t, const CaliperTensor&,
                                const uint32_t*, float, float) override {
        return false;
    }

private:
    struct Tex { GLuint gl_name; int w; int h; };
    GLFWwindow* window_ = nullptr;
    std::unordered_map<uint64_t, Tex> textures_;
    uint64_t next_id_ = 1;   // 0 is the invalid id
};

}  // namespace

std::unique_ptr<HostRenderer> make_renderer(const char* /*name*/) {
    // C1 ships only the GL backend; the Metal backend and "metal"/nullptr
    // dispatch arrive in C2. Default (and every request) is GL for now.
    return std::make_unique<GLRenderer>();
}

}  // namespace caliper_host
