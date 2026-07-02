#include "host_services.h"
#include "job_system.h"
#include "device_query.h"
#include <caliper/services/log_v1.h>
#include <caliper/services/ui_v1.h>
#include <caliper/services/jobs_v1.h>
#include <caliper/services/device_v1.h>
#include <imgui.h>
#include <implot.h>
#include <implot3d.h>
#include <cstdio>
#include <cstring>
#include <ctime>

namespace caliper_host {
namespace {

// --- caliper.log.v1: timestamped console lines (console panel = later) ---
// Reentrant time formatting: the docs promise log() is callable from applet
// worker threads, and plain std::localtime shares a static buffer.
void log_impl(CaliperLogLevel level, const char* msg) {
    static const char* kTag[] = {"DEBUG", "INFO ", "WARN ", "ERROR"};
    int idx = (level >= 0 && level <= 3) ? (int)level : 1;
    std::time_t t = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char ts[16];
    std::strftime(ts, sizeof ts, "%H:%M:%S", &tm_buf);
    std::fprintf(stderr, "[%s] [%s] %s\n", ts, kTag[idx], msg ? msg : "");
}
const CaliperLogV1 kLog = {sizeof(CaliperLogV1), &log_impl};

// --- caliper.ui.v1: contexts + allocator handoff (§6d) ---
ImGuiContext*    ui_imgui()    { return ImGui::GetCurrentContext(); }
ImPlotContext*   ui_implot()   { return ImPlot::GetCurrentContext(); }
ImPlot3DContext* ui_implot3d() { return ImPlot3D::GetCurrentContext(); }
void ui_allocators(CaliperImGuiAllocFn* out_alloc, CaliperImGuiFreeFn* out_free,
                   void** out_user) {
    ImGuiMemAllocFunc a = nullptr; ImGuiMemFreeFunc f = nullptr; void* u = nullptr;
    ImGui::GetAllocatorFunctions(&a, &f, &u);
    *out_alloc = reinterpret_cast<CaliperImGuiAllocFn>(a);
    *out_free  = reinterpret_cast<CaliperImGuiFreeFn>(f);
    *out_user  = u;
}
const CaliperUiV1 kUi = {sizeof(CaliperUiV1), &ui_imgui, &ui_implot,
                         &ui_implot3d, &ui_allocators};

// --- caliper.jobs.v1: background compute with progress + cancel (§7.5) ---
// Backed by one process-wide JobSystem; its dtor cancels + joins at shutdown.
JobSystem g_jobs;

uint64_t jobs_submit(const char* label, CaliperJobFn fn, void* user) {
    return g_jobs.submit(label ? label : "(job)", fn, user);
}
void jobs_cancel(uint64_t id)   { g_jobs.request_cancel(id); }
bool jobs_running(uint64_t id)  { return g_jobs.is_running(id); }
float jobs_progress(uint64_t id){ return g_jobs.progress_of(id); }
const CaliperJobsV1 kJobs = {sizeof(CaliperJobsV1), &jobs_submit, &jobs_cancel,
                             &jobs_running, &jobs_progress};

// --- caliper.device.v1: negotiated compute device (§7.3) ---
CaliperDeviceKind dev_kind(void) { return device_info().kind; }
int32_t dev_index(void)          { return device_info().index; }
const char* dev_name(void)       { return device_info().name.c_str(); }
uint64_t dev_hint(void)          { return device_info().free_memory_hint; }
const CaliperDeviceV1 kDevice = {sizeof(CaliperDeviceV1), &dev_kind, &dev_index,
                                 &dev_name, &dev_hint};

const std::set<std::string> kIds = {CALIPER_UI_V1, CALIPER_LOG_V1,
                                    CALIPER_JOBS_V1, CALIPER_DEVICE_V1};

} // namespace

void services_init() { /* tables are static; hook kept for later services */ }

JobSystem& host_job_system() { return g_jobs; }

const void* services_get(const char* id) {
    if (!id) return nullptr;
    if (std::strcmp(id, CALIPER_UI_V1) == 0)     return &kUi;
    if (std::strcmp(id, CALIPER_LOG_V1) == 0)    return &kLog;
    if (std::strcmp(id, CALIPER_JOBS_V1) == 0)   return &kJobs;
    if (std::strcmp(id, CALIPER_DEVICE_V1) == 0) return &kDevice;
    return nullptr;   // unknown ids: NULL, never UB (§6b)
}

const std::set<std::string>& service_ids() { return kIds; }

} // namespace caliper_host
