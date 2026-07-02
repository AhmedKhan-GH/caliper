#pragma once
#include <set>
#include <string>

namespace caliper_host {
// Host-side service registry (PLATFORM.md §6b). Call services_init() once
// after the ImGui/ImPlot/ImPlot3D contexts exist; tables are static and live
// for the process lifetime (the ABI's pointer-validity guarantee).
void services_init();
const void* services_get(const char* service_id);   // NULL for unknown ids
const std::set<std::string>& service_ids();

// The process-wide job system backing caliper.jobs.v1 (§7.5); the host jobs
// tray reads its views(). Cancelled + joined at shutdown by its own dtor.
class JobSystem;
JobSystem& host_job_system();

// The process-wide metrics store backing caliper.metrics.v1 (§7.6/§11); the
// host dashboard (B5) reads it. Opened by services_init() at
// caliper::app_data_path("metrics.duckdb"); may be unopened if the disk failed
// (the service is still vended, its thunks no-op on the null store).
class MetricsStore;
MetricsStore& host_metrics_store();

// The active renderer backing caliper.tensor_bridge.v1 (§7.4). main owns the
// HostRenderer; it hands it in right after renderer init and clears it (nullptr)
// before renderer teardown. The bridge is constructed lazily on the first thunk
// call once a renderer is bound; every bridge thunk no-ops (0/false) until then,
// so a pre-renderer/headless call never crashes (the metrics-open pattern).
class HostRenderer;
void services_set_renderer(HostRenderer* renderer);
}
