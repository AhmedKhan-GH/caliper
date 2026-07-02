#include "host_services.h"
#include "job_system.h"
#include "device_query.h"
#include "metrics_store.h"
#include "../app_paths.h"   // host_services.cpp compiles into the caliper exe,
                            // which also compiles app_paths.cpp (CMakeLists)
#include <caliper/services/log_v1.h>
#include <caliper/services/ui_v1.h>
#include <caliper/services/jobs_v1.h>
#include <caliper/services/device_v1.h>
#include <caliper/services/metrics_v1.h>
#include <caliper/tensor.h>
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

// --- caliper.metrics.v1: DuckDB-backed run/tag/step store (§7.6/§11) ---
// One process-wide store, opened in services_init(). If the disk open fails we
// still vend the table (never crash the host over a bad disk): g_metrics_open
// stays false and every thunk no-ops on the unopened store.
//
// DECLARATION ORDER IS LOAD-BEARING: g_metrics must be declared BEFORE
// g_jobs so it is destroyed AFTER g_jobs's destructor joins the worker
// threads — jobs may write metrics until the moment they are joined.
MetricsStore g_metrics;
bool         g_metrics_open = false;

uint64_t met_begin_run(const char* experiment, const char* run_name) {
    return g_metrics_open
        ? g_metrics.begin_run(experiment ? experiment : "", run_name ? run_name : "")
        : 0;
}
void met_end_run(uint64_t run) { if (g_metrics_open) g_metrics.end_run(run); }
void met_scalar(uint64_t run, const char* tag, int64_t step, double value) {
    if (g_metrics_open) g_metrics.scalar(run, tag ? tag : "", step, value);
}
void met_histogram(uint64_t run, const char* tag, int64_t step,
                   const float* values, int64_t count) {
    if (g_metrics_open) g_metrics.histogram(run, tag ? tag : "", step, values, count);
}
// v1 accepts only CPU-resident HWC u8 tensors (documented in metrics_v1.h);
// a non-conforming tensor is logged and dropped rather than misinterpreted.
void met_image(uint64_t run, const char* tag, int64_t step,
              const CaliperTensor* t) {
    if (!g_metrics_open) return;
    if (!(t && t->struct_size >= sizeof(CaliperTensor) &&
          t->dtype == CALIPER_DT_U8 && t->ndim == 3 &&
          t->device == CALIPER_DEV_CPU &&
          t->strides[2] == 1 && t->strides[1] == t->shape[2] &&
          t->strides[0] == t->shape[1] * t->shape[2])) {
        log_impl(CALIPER_LOG_WARN,
                 "metrics.v1: image() dropped a non-contiguous or non-CPU-u8-HWC tensor");
        return;
    }
    g_metrics.image(run, tag ? tag : "", step, t->data,
                    (int32_t)t->shape[1], (int32_t)t->shape[0], (int32_t)t->shape[2]);
}
void met_hparams_json(uint64_t run, const char* json_utf8) {
    if (g_metrics_open) g_metrics.hparams_json(run, json_utf8 ? json_utf8 : "");
}
const CaliperMetricsV1 kMetrics = {sizeof(CaliperMetricsV1), &met_begin_run,
                                   &met_end_run, &met_scalar, &met_histogram,
                                   &met_image, &met_hparams_json};

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
                                    CALIPER_JOBS_V1, CALIPER_DEVICE_V1,
                                    CALIPER_METRICS_V1};

} // namespace

void services_init() {
    // Open the metrics store; on failure log and carry on (the table is vended
    // either way, its thunks no-op on the unopened store — §6b, never crash).
    const std::string path = caliper::app_data_path("metrics.duckdb");
    g_metrics_open = g_metrics.open(path);
    if (!g_metrics_open)
        std::fprintf(stderr,
                     "[metrics] failed to open %s; metrics.v1 will no-op\n",
                     path.c_str());
}

JobSystem& host_job_system() { return g_jobs; }

MetricsStore& host_metrics_store() { return g_metrics; }

const void* services_get(const char* id) {
    if (!id) return nullptr;
    if (std::strcmp(id, CALIPER_UI_V1) == 0)     return &kUi;
    if (std::strcmp(id, CALIPER_LOG_V1) == 0)    return &kLog;
    if (std::strcmp(id, CALIPER_JOBS_V1) == 0)   return &kJobs;
    if (std::strcmp(id, CALIPER_DEVICE_V1) == 0) return &kDevice;
    if (std::strcmp(id, CALIPER_METRICS_V1) == 0) return &kMetrics;
    return nullptr;   // unknown ids: NULL, never UB (§6b)
}

const std::set<std::string>& service_ids() { return kIds; }

} // namespace caliper_host
