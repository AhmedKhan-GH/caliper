#include "applet_api.h"
#include "ucdh_workbench.h"

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

extern "C" {

APPLET_API CaliperAppletInfo applet_info() {
    return {
        "UCDH Workbench",
        "1.0",
        "Load and visualize the UCDH Senior Design dataset, run "
        "signal preprocessing, inspect raw data via DuckDB, and "
        "run model inference on ECG recordings.",
        "ECG",
        CALIPER_APPLET_ABI
    };
}

APPLET_API void* applet_create() {
    return new UCDHWorkbenchApplet();
}

APPLET_API void applet_destroy(void* ctx) {
    delete static_cast<UCDHWorkbenchApplet*>(ctx);
}

APPLET_API bool applet_initialize(void* ctx, const CaliperHostContext* host) {
    ImGui::SetCurrentContext(host->imgui);
    ImPlot::SetCurrentContext(host->implot);
    ImPlot3D::SetCurrentContext(host->implot3d);
    return static_cast<UCDHWorkbenchApplet*>(ctx)->initialize();
}

APPLET_API void applet_draw_ui(void* ctx, int w, int h) {
    static_cast<UCDHWorkbenchApplet*>(ctx)->draw_ui(w, h);
}

APPLET_API void applet_cleanup(void* ctx) {
    static_cast<UCDHWorkbenchApplet*>(ctx)->cleanup();
}

} // extern "C"
