#include <caliper/abi_v1.h>
#include "repnet_demo.h"

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

extern "C" {

APPLET_API CaliperAppletInfo applet_info() {
    return {
        "RepNet Demo",
        "1.0",
        "Load and visualize the UCDH Senior Design dataset, run "
        "signal preprocessing, inspect raw data via DuckDB, and "
        "run model inference on ECG recordings.",
        "ECG",
        CALIPER_APPLET_ABI
    };
}

APPLET_API void* applet_create() {
    return new RepNetDemoApplet();
}

APPLET_API void applet_destroy(void* ctx) {
    delete static_cast<RepNetDemoApplet*>(ctx);
}

APPLET_API bool applet_initialize(void* ctx, const CaliperHostContext* host) {
    ImGui::SetCurrentContext(host->imgui);
    ImPlot::SetCurrentContext(host->implot);
    ImPlot3D::SetCurrentContext(host->implot3d);
    return static_cast<RepNetDemoApplet*>(ctx)->initialize();
}

APPLET_API void applet_draw_ui(void* ctx, int w, int h) {
    static_cast<RepNetDemoApplet*>(ctx)->draw_ui(w, h);
}

APPLET_API void applet_cleanup(void* ctx) {
    static_cast<RepNetDemoApplet*>(ctx)->cleanup();
}

} // extern "C"
