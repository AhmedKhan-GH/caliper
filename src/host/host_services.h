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
}
