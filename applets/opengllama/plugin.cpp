#include <caliper/abi_v1.h>
#include "opengllama.h"

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

extern "C" {

APPLET_API CaliperAppletInfo applet_info() {
    return {
        "OpenGllama",
        "0.1",
        "Load GGUF models via llama.cpp and visualize layer activations "
        "with OpenGL-rendered heatmaps on Metal/CUDA backends.",
        "LLM",
        CALIPER_APPLET_ABI
    };
}

APPLET_API void* applet_create() {
    return new OpenGllamaApplet();
}

APPLET_API void applet_destroy(void* ctx) {
    delete static_cast<OpenGllamaApplet*>(ctx);
}

APPLET_API bool applet_initialize(void* ctx, const CaliperHostContext* host) {
    ImGui::SetCurrentContext(host->imgui);
    ImPlot::SetCurrentContext(host->implot);
    ImPlot3D::SetCurrentContext(host->implot3d);
    return static_cast<OpenGllamaApplet*>(ctx)->initialize();
}

APPLET_API void applet_draw_ui(void* ctx, int w, int h) {
    static_cast<OpenGllamaApplet*>(ctx)->draw_ui(w, h);
}

APPLET_API void applet_cleanup(void* ctx) {
    static_cast<OpenGllamaApplet*>(ctx)->cleanup();
}

} // extern "C"
