#pragma once
#include <set>
#include <string>
#include <caliper/services/log_v1.h>   // CaliperLogLevel (applet log sink)

namespace caliper_host {

// Applet log routing (embed v1.1, §3.3 — the ledgered de-singletonization).
// caliper.log.v1 was a process-wide stderr singleton. Install a sink and every
// log.v1 line is delivered to it INSTEAD of stderr; pass nullptr to restore the
// built-in timestamped stderr writer (the caliper exe path, which never installs
// a sink — unchanged). Set once at create on the frame thread; the embed core
// installs it when the embedder provides a log_fn and clears it at shutdown
// AFTER joining workers, so the read on worker threads (log.v1 is callable from
// workers) is ordered by thread create/join — no lock needed (mirrors the
// existing set_bridge_log_sink pattern).
void set_applet_log_sink(void (*sink)(void* userdata, CaliperLogLevel level,
                                      const char* message_utf8),
                         void* userdata);
// Host-side service registry (PLATFORM.md §6b). Call services_init() once
// after the ImGui/ImPlot/ImPlot3D contexts exist; tables are static and live
// for the process lifetime (the ABI's pointer-validity guarantee).
void services_init();
const void* services_get(const char* service_id);   // NULL for unknown ids
const std::set<std::string>& service_ids();

// Deterministic service teardown, called from the host's cleanup() BEFORE
// main returns: joins all job workers, then closes the DuckDB-backed stores.
// Leaving those to static destructors races DuckDB's own globals (undefined
// cross-TU destruction order) and aborts in malloc at exit. After this call
// every store-backed thunk no-ops; the tables themselves remain valid.
void services_shutdown();

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

// The process-wide artifact store backing caliper.artifacts.v1 (§7.8).
// Opened by services_init() under the app data dir; may be unopened if the
// disk failed (the service is still vended, its thunks no-op).
class ArtifactStore;
ArtifactStore& host_artifact_store();

// The process-wide data store backing caliper.data.v1 (§7.7). Opened by
// services_init() at caliper::app_data_path("data.duckdb"); may be unopened
// (thunks fail with last_error, never crash).
class DataStore;
DataStore& host_data_store();

// The active renderer backing caliper.tensor_bridge.v1 (§7.4). main owns the
// HostRenderer; it hands it in right after renderer init and clears it (nullptr)
// before renderer teardown. The bridge is constructed lazily on the first thunk
// call once a renderer is bound; every bridge thunk no-ops (0/false) until then,
// so a pre-renderer/headless call never crashes (the metrics-open pattern).
class HostRenderer;
void services_set_renderer(HostRenderer* renderer);

// The HostRenderer currently backing caliper.tensor_bridge.v1 (or nullptr).
// White-box use (the §7 host-axis byte-compare in caliper_embed_tests): read a
// bridge texture back with debug_readback_rgba8, which is renderer-side. Not on
// the applet-facing surface — applets never see the renderer type.
HostRenderer* services_renderer();
}
