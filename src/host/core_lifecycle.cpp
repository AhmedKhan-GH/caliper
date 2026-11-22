#include "core_lifecycle.h"

#include <cstdlib>
#include <cstring>

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

namespace caliper_host {

namespace {

// What a refusal calls each renderer.
const char* display_name(const char* name) {
    if (std::strcmp(name, "metal") == 0) return "Metal";
    if (std::strcmp(name, "vulkan") == 0) return "Vulkan";
    return name;
}

}  // namespace

std::unique_ptr<HostRenderer> core_select_renderer(std::string* why) {
    // Renderer seam (docs/applets/rendering.md): backend hints run before
    // the window exists; init() runs after. Metal on Apple and Vulkan on
    // Windows (device-resident CUDA tensors through external-memory interop).
    const char* want = std::getenv("CALIPER_RENDERER");
    if (want && std::strcmp(want, "gl") == 0) {
        if (why)
            *why = "CALIPER_RENDERER=gl asks for the OpenGL renderer, which was removed; "
                   "unset it, and Caliper draws with Metal on macOS and Vulkan on Windows";
        return nullptr;
    }
#ifdef __APPLE__
    return make_metal_renderer();
#elif defined(_WIN32)
    return make_vulkan_renderer();
#else
    if (why)
        *why = "this platform has no renderer: Caliper draws with Metal on macOS and "
               "Vulkan on Windows";
    return nullptr;
#endif
}

std::string renderer_start_refusal(const char* name, const std::string& why) {
    return std::string("the ") + display_name(name) + " renderer could not start (" +
           (why.empty() ? std::string("no reason given") : why) +
           "), and there is no fallback renderer until the WebGPU fallback through Dawn lands";
}

void core_create_ui_context() {
    // Host-owned ImGui/ImPlot contexts must exist before the renderer
    // initializes its ImGui backends. None of these touch the GPU.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImPlot3D::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    // Docking (ImGui docking branch): the applet page hosts a docked desktop —
    // applet windows tile into a central node + side column. Persisted
    // per-window/dockspace layout lives in imgui.ini (host-owned).
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
}

void core_destroy_ui_context() {
    ImPlot3D::DestroyContext();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}

}  // namespace caliper_host
