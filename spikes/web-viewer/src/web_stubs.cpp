// web_stubs.cpp: the one piece of host_services.cpp that the web viewer spike
// needs (docs/plans/web-viewer-spike.md). host_services.cpp itself pulls in
// the DuckDB stores, the job system and every service table, none of which
// exist in a browser, so it is not compiled.
//
// embed_core.cpp compiles unchanged. Its create/shutdown/applet paths
// reference the applet loader, the service registry, the job system, the app
// paths and the core lifecycle (22 symbols, listed in the report), but the
// page host only uses the hosted core (caliper_core_create_hosted), so the
// linker drops those paths and no stand-in is needed for them. A full viewer
// that exported caliper_core_create would need them stubbed or split out.
//
// What is left is the tensor bridge vending: services_set_renderer and
// services_bridge build the real TensorBridge (tensor_bridge.cpp, compiled
// unchanged) on the WebGPU renderer, as host_services.cpp does natively.
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
