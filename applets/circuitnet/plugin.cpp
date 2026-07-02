#include <caliper/abi_v1.h>
#include "circuitnet.h"

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

extern "C" {

APPLET_API CaliperAppletInfo applet_info() {
    return {
        "CircuitNet 3.0",
        "1.0",
        "Gate-level circuit architecture explorer with DuckDB-powered "
        "querying, Verilog netlist parsing, and interactive graph visualization.",
        "EDA",
        CALIPER_APPLET_ABI
    };
}

APPLET_API void* applet_create() {
    return new CircuitNetApplet();
}

APPLET_API void applet_destroy(void* ctx) {
    delete static_cast<CircuitNetApplet*>(ctx);
}

APPLET_API bool applet_initialize(void* ctx, const CaliperHostContext* host) {
    ImGui::SetCurrentContext(host->imgui);
    ImPlot::SetCurrentContext(host->implot);
    ImPlot3D::SetCurrentContext(host->implot3d);
    return static_cast<CircuitNetApplet*>(ctx)->initialize();
}

APPLET_API void applet_draw_ui(void* ctx, int w, int h) {
    static_cast<CircuitNetApplet*>(ctx)->draw_ui(w, h);
}

APPLET_API void applet_cleanup(void* ctx) {
    static_cast<CircuitNetApplet*>(ctx)->cleanup();
}

} // extern "C"
