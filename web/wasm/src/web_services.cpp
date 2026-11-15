// web_services.cpp: the one piece of host_services.cpp that the browser's view
// layer needs. host_services.cpp itself pulls in the DuckDB stores, the job
// system and every service table, none of which exist in a browser, so it is
// not compiled for web/wasm.
//
// services_set_renderer and services_bridge build the real TensorBridge
// (tensor_bridge.cpp) on the WebGPU renderer, as host_services.cpp does
// natively; the embed v1.2 views (embed_views.cpp) reach it through
// services_bridge().
#include "host_services.h"
#include "tensor_bridge.h"

#include <memory>

namespace caliper_host {

namespace {
HostRenderer* g_renderer = nullptr;
std::unique_ptr<TensorBridge> g_bridge;
}  // namespace

void services_set_renderer(HostRenderer* renderer) {
    g_bridge.reset();
    g_renderer = renderer;
}
HostRenderer* services_renderer() { return g_renderer; }
TensorBridge* services_bridge() {
    if (!g_renderer) return nullptr;
    if (!g_bridge) g_bridge = std::make_unique<TensorBridge>(*g_renderer);
    return g_bridge.get();
}

}  // namespace caliper_host
