#include "core_lifecycle.h"

#include <cstdlib>
#include <cstring>

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

namespace caliper_host {

std::unique_ptr<HostRenderer> core_select_renderer() {
    // Renderer seam (PLATFORM.md §5.4): backend hints run before the window
    // exists; init() runs after. On Apple the default is Metal (device-resident
    // tensors, zero CPU staging). CALIPER_RENDERER=gl selects the frozen GL
    // fallback; off Apple, GL is the only backend besides Vulkan on Win32.
    const char* want = std::getenv("CALIPER_RENDERER");
    bool want_gl = want && std::strcmp(want, "gl") == 0;
    std::unique_ptr<HostRenderer> renderer;
#ifdef __APPLE__
    if (!want_gl) renderer = make_metal_renderer();
#elif defined(_WIN32)
    // Windows default is Vulkan (Phase 4: device-resident CUDA tensors via
    // external-memory interop). Same fallback contract as Metal: if init()
    // fails (no Vulkan driver, RDP, ...) GL takes over in the host.
    if (!want_gl) renderer = make_vulkan_renderer();
#else
    (void)want_gl;
#endif
    if (!renderer) renderer = make_renderer("gl");
    return renderer;
}

void core_create_ui_context() {
    // Host-owned ImGui/ImPlot contexts must exist before the renderer
    // initializes its ImGui backends. None of these touch GL/Metal.
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
