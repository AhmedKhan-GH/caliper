#include "host_services.h"
#include "job_system.h"
#include "device_query.h"
#include "metrics_store.h"
#include "feed_store.h"
#ifdef __APPLE__
#include "feed_provider_mac.h"   // macOS telemetry provider (T2); Apple-only
#elif defined(_WIN32)
#include "feed_provider_win.h"   // Windows telemetry provider (feed spec §6.2)
#endif
#include "artifact_store.h"
#include "data_store.h"
#include "tensor_bridge.h"
#include "renderer/host_renderer.h"  // export.v1 reads the view back off the renderer
#include "export_service.h"          // export.v1 PNG/sidecar/atomic-write helpers
#include "host_version.h"            // kHostVersionStr for the export sidecar
#include "../app_paths.h"   // host_services.cpp compiles into the caliper exe,
                            // which also compiles app_paths.cpp (CMakeLists)
#include <caliper/services/log_v1.h>
#include <caliper/services/ui_v1.h>
#include <caliper/services/jobs_v1.h>
#include <caliper/services/device_v1.h>
#include <caliper/services/metrics_v1.h>
#include <caliper/services/metrics_v1_1.h>
#include <caliper/services/artifacts_v1.h>
#include <caliper/services/data_v1.h>
#include <caliper/services/feed_v1.h>
#include <caliper/services/tensor_bridge_v1.h>
#include <caliper/services/tensor_bridge_v1_1.h>
#include <caliper/services/tensor_bridge_v1_2.h>
#include <caliper/services/geometry_v1.h>
#include <caliper/services/geometry_v1_1.h>
#include <caliper/services/geometry_v1_2.h>
#include <caliper/services/geometry_v1_3.h>
#include <caliper/services/export_v1.h>
#include <caliper/tensor.h>
#include <imgui.h>
#include <implot.h>
#include <implot3d.h>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <optional>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace caliper_host {

// Defined in tensor_bridge.cpp (the exe / gfx-test link scope, NOT the frozen
// tensor_bridge.h). Installs a caliper.log.v1 sink for the bridge's rejection
// reasons; the unit/gfx test binaries link tensor_bridge.cpp WITHOUT this TU,
// so they simply never call it and fall back to its built-in stderr sink.
void set_bridge_log_sink(void (*sink)(CaliperLogLevel, const char*));

namespace {

// --- caliper.log.v1: timestamped console lines (console panel = later) ---
// Installable sink (embed v1.1): when set, log.v1 lines route here instead of
// stderr. A plain pointer, not atomic: it is installed at core create (before
// any applet/worker exists) and cleared at shutdown AFTER workers join, so the
// worker-thread reads are ordered by thread create/join (same discipline as
// g_bridge's log sink). NULL => the built-in stderr writer below.
void (*g_applet_log_sink)(void*, CaliperLogLevel, const char*) = nullptr;
void*  g_applet_log_userdata = nullptr;

// Reentrant time formatting: the docs promise log() is callable from applet
// worker threads, and plain std::localtime shares a static buffer.
void log_impl(CaliperLogLevel level, const char* msg) {
    if (g_applet_log_sink) {   // embedder installed a sink: route, don't touch stderr
        g_applet_log_sink(g_applet_log_userdata, level, msg ? msg : "");
        return;
    }
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

// metrics.v1_1 (C0b): the SAME six writers plus the read surface query() — a
// consumer (Compass via caliper_core_get_service) lists runs / streams scalars
// as Arrow against the store's OWN live connection. Read-only + Arrow ownership
// enforced inside MetricsStore::query (see metrics_v1_1.h). No-ops on the
// unopened store, the same discipline as the writers above.
bool met_query(const char* sql, struct ArrowArrayStream* out) {
    return g_metrics_open && g_metrics.query(sql ? sql : "", out);
}
const char* met_last_error(void) {
    // metrics.v1_1 promises never-NULL; the store's error is thread-local, so a
    // thread-local holder honors "valid until the next metrics.v1_1 call on the
    // same thread" (mirrors data.v1's last_error).
    static thread_local std::string held;
    held = g_metrics_open ? g_metrics.last_error() : "metrics store is not open";
    return held.c_str();
}
const CaliperMetricsV1_1 kMetrics11 = {sizeof(CaliperMetricsV1_1),
    &met_begin_run, &met_end_run, &met_scalar, &met_histogram,
    &met_image, &met_hparams_json, &met_query, &met_last_error};

// --- caliper.feed.v1: live telemetry ring buffers (feed spec §4) ---
// One process-wide store, vended for applets AND embedders (feed.v1 is an
// any-thread service). In T1 the store owns NO provider thread and NO channels:
// caps() reports 0 (CALIPER_FEED_CAP_LIVE unset) and every read yields nothing
// until a provider (T2 macOS sensors) or a test registers channels — honest
// degradation, never fake data. The thunks are trivial forwarders onto
// FeedStore, which carries the tested per-channel ring / cursor logic.
FeedStore g_feed;

uint32_t feed_caps(void)          { return g_feed.caps(); }
uint32_t feed_channel_count(void) { return g_feed.channel_count(); }
uint32_t feed_channel_info(uint32_t index, CaliperFeedChannelInfo* info) {
    return g_feed.channel_info(index, info);
}
uint32_t feed_read(const char* id, CaliperFeedSample* buf, uint32_t max,
                   uint64_t* cursor) {
    return g_feed.read(id, buf, max, cursor);
}
const CaliperFeedV1 kFeed = {sizeof(CaliperFeedV1), &feed_caps,
                             &feed_channel_count, &feed_channel_info,
                             &feed_read, nullptr};

// --- caliper.artifacts.v1: content-addressed checkpoints (§7.8) ---
// Same non-fatal-open + no-op-thunks discipline as metrics above, and the
// same DECLARATION ORDER rule: g_artifacts is declared BEFORE g_jobs so job
// threads can save checkpoints until the moment they are joined.
ArtifactStore g_artifacts;
bool          g_artifacts_open = false;

bool art_put(const char* name, const void* bytes, uint64_t len,
             uint64_t run, char out_digest[65]) {
    return g_artifacts_open &&
           g_artifacts.put(name ? name : "", bytes, len, run, out_digest);
}
const char* art_path_of(const char* digest_or_name) {
    if (!g_artifacts_open || !digest_or_name) return nullptr;
    // ArtifactStore keeps the returned string alive until the next call
    // (the artifacts.v1 "host-owned, valid until next call" contract).
    static thread_local std::string held;
    held = g_artifacts.path_of(digest_or_name);
    return held.empty() ? nullptr : held.c_str();
}
bool art_exists(const char* digest_or_name) {
    return g_artifacts_open && digest_or_name &&
           g_artifacts.exists(digest_or_name);
}
const CaliperArtifactsV1 kArtifacts = {sizeof(CaliperArtifactsV1), &art_put,
                                       &art_path_of, &art_exists};

// --- caliper.data.v1: SQL over the host store, Arrow streams out (§7.7) ---
// Same non-fatal-open + declaration-before-g_jobs discipline as above.
DataStore g_data;
bool      g_data_open = false;

bool dat_query(const char* sql, struct ArrowArrayStream* out) {
    return g_data_open && g_data.query(sql ? sql : "", out);
}
bool dat_register_dataset(const char* name, const char* uri) {
    return g_data_open &&
           g_data.register_dataset(name ? name : "", uri ? uri : "");
}
bool dat_open_dataset(const char* name, struct ArrowArrayStream* out) {
    return g_data_open && g_data.open_dataset(name ? name : "", out);
}
const char* dat_last_error(void) {
    // data.v1 promises never-NULL; the store's error is thread-local, so a
    // thread-local holder honors "valid until the next data.v1 call on the
    // same thread" without cross-thread stomping.
    static thread_local std::string held;
    held = g_data_open ? g_data.last_error() : "data store is not open";
    return held.c_str();
}
const CaliperDataV1 kData = {sizeof(CaliperDataV1), &dat_query,
                             &dat_register_dataset, &dat_open_dataset,
                             &dat_last_error};

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

// --- caliper.tensor_bridge.v1: CaliperTensor -> live texture (§7.4) ---
// The bridge builds on the live HostRenderer, which main owns and hands in via
// services_set_renderer(). It is constructed lazily on the first thunk call
// once a renderer is bound (TensorBridge holds a HostRenderer&); before that,
// or on a headless host, every thunk no-ops on the null bridge — never a crash
// (the metrics-open pattern).
//
// STATIC DESTRUCTION ORDER: g_bridge holds only its HostRenderer& (no textures
// are released in its dtor), and it is reset the moment main clears the renderer
// (services_set_renderer(nullptr)) BEFORE renderer teardown — so the renderer
// reference is dropped while the renderer is still alive and the process-exit
// dtor of g_bridge never touches a destroyed renderer. Same load-bearing-order
// discipline as g_metrics/g_jobs above.
HostRenderer*             g_renderer = nullptr;
std::optional<TensorBridge> g_bridge;

TensorBridge* bridge() {
    if (!g_renderer) return nullptr;
    if (!g_bridge) g_bridge.emplace(*g_renderer);
    return &*g_bridge;
}

CaliperTextureId br_texture_from_tensor(const CaliperTensor* t, uint32_t flags) {
    TensorBridge* b = bridge();
    return b ? b->texture_from_tensor(t, flags) : 0;
}
bool br_update_texture(CaliperTextureId tex, const CaliperTensor* t) {
    TensorBridge* b = bridge();
    return b ? b->update_texture(tex, t) : false;
}
void br_release_texture(CaliperTextureId tex) {
    if (TensorBridge* b = bridge()) b->release_texture(tex);
}
CaliperTextureId br_texture_from_tensor_mapped(const CaliperTensor* t,
        int32_t colormap, float vmin, float vmax, uint32_t flags) {
    TensorBridge* b = bridge();
    return b ? b->texture_from_tensor_mapped(t, colormap, vmin, vmax, flags) : 0;
}
bool br_alloc_shared(CaliperDType dtype, int32_t ndim, const int64_t* shape,
                     CaliperTensor* out_tensor, CaliperTextureId* out_texture) {
    TensorBridge* b = bridge();
    return b ? b->alloc_shared(dtype, ndim, shape, out_tensor, out_texture)
             : false;
}
void br_free_shared(CaliperTextureId tex) {
    if (TensorBridge* b = bridge()) b->free_shared(tex);
}
const CaliperTensorBridgeV1 kBridge = {sizeof(CaliperTensorBridgeV1),
    &br_texture_from_tensor, &br_update_texture, &br_release_texture,
    &br_texture_from_tensor_mapped, &br_alloc_shared, &br_free_shared};

// v1.1 (D24): the same six thunks plus caps(). Bit 0 reflects the ACTIVE
// renderer (Metal/Vulkan honor it once M2 lands there; GL and headless never
// do) — 0 with no renderer bound, so adapters keep draining.
uint32_t br_caps(void) {
    TensorBridge* b = bridge();
    return b ? b->caps() : 0u;
}
const CaliperTensorBridgeV1_1 kBridge11 = {sizeof(CaliperTensorBridgeV1_1),
    &br_texture_from_tensor, &br_update_texture, &br_release_texture,
    &br_texture_from_tensor_mapped, &br_alloc_shared, &br_free_shared,
    &br_caps};

// v1.2 (zero-copy imported allocations): the seven v1.1 members plus the three
// import ops. Null-bridge (no renderer / headless) → 0 / no-op / false, the
// same discipline as the v1 thunks above.
CaliperAllocId br_import_allocation(void* os_handle, uint64_t size_bytes,
                                    uint32_t handle_type) {
    TensorBridge* b = bridge();
    return b ? b->import_allocation(os_handle, size_bytes, handle_type) : 0;
}
void br_release_allocation(CaliperAllocId alloc) {
    if (TensorBridge* b = bridge()) b->release_allocation(alloc);
}
bool br_update_texture_from_alloc(CaliperTextureId tex, CaliperAllocId alloc,
                                  uint64_t offset_bytes, const CaliperTensor* desc) {
    TensorBridge* b = bridge();
    return b ? b->update_texture_from_alloc(tex, alloc, offset_bytes, desc) : false;
}
const CaliperTensorBridgeV1_2 kBridge12 = {sizeof(CaliperTensorBridgeV1_2),
    &br_texture_from_tensor, &br_update_texture, &br_release_texture,
    &br_texture_from_tensor_mapped, &br_alloc_shared, &br_free_shared,
    &br_caps, &br_import_allocation, &br_release_allocation,
    &br_update_texture_from_alloc};

// caliper.geometry.v1: imported 3-D points. Vended from the SAME TensorBridge
// object (one backing object, two tables) so views share the texture id space
// and points address v1.2 imported allocations coherently. Null-bridge
// (headless / no renderer) → 0 / no-op / false, same discipline as above.
uint32_t geo_caps(void) {
    TensorBridge* b = bridge();
    return b ? b->geom_caps() : 0u;
}
CaliperTextureId geo_create_view(uint32_t w, uint32_t h) {
    TensorBridge* b = bridge();
    return b ? b->geom_create_view(w, h) : 0;
}
CaliperTextureId geo_create_view_ex(uint32_t w, uint32_t h, uint32_t flags) {
    TensorBridge* b = bridge();
    return b ? b->geom_create_view_ex(w, h, flags) : 0;
}
void geo_release_view(CaliperTextureId view) {
    if (TensorBridge* b = bridge()) b->geom_release_view(view);
}
bool geo_draw_points(CaliperTextureId view, const CaliperGeomCamera* cam,
                     CaliperAllocId pos_alloc, uint64_t pos_offset,
                     uint64_t count,
                     CaliperAllocId attr_alloc, uint64_t attr_offset,
                     int32_t colormap, float vmin, float vmax,
                     float size_px, uint32_t clear_rgba) {
    TensorBridge* b = bridge();
    return b ? b->geom_draw_points(view, cam, pos_alloc, pos_offset, count,
                                   attr_alloc, attr_offset, colormap, vmin,
                                   vmax, size_px, clear_rgba)
             : false;
}
bool geo_draw_primitives(CaliperTextureId view, const CaliperGeomCamera* cam,
                         const CaliperGeomDraw* draws, uint32_t draw_count,
                         uint32_t draw_stride, uint32_t clear_rgba) {
    TensorBridge* b = bridge();
    return b ? b->geom_draw_primitives(view, cam, draws, draw_count,
                                       draw_stride, clear_rgba)
             : false;
}
bool geo_draw_primitives_v12(CaliperTextureId view,
                             const CaliperGeomCamera* cam,
                             const CaliperGeomDrawV1_2* draws,
                             uint32_t draw_count, uint32_t draw_stride,
                             uint32_t clear_rgba) {
    TensorBridge* b = bridge();
    return b ? b->geom_draw_primitives_v1_2(view, cam, draws, draw_count,
                                            draw_stride, clear_rgba)
             : false;
}
bool geo_draw_primitives_v13(CaliperTextureId view,
                             const CaliperGeomCamera* cam,
                             const CaliperGeomDrawV1_3* draws,
                             uint32_t draw_count, uint32_t draw_stride,
                             uint32_t clear_rgba) {
    TensorBridge* b = bridge();
    return b ? b->geom_draw_primitives_v1_3(view, cam, draws, draw_count,
                                            draw_stride, clear_rgba)
             : false;
}
const CaliperGeometryV1 kGeom1 = {sizeof(CaliperGeometryV1),
    &geo_caps, &geo_create_view, &geo_release_view, &geo_draw_points};
const CaliperGeometryV1_1 kGeom11 = {sizeof(CaliperGeometryV1_1),
    &geo_caps, &geo_create_view, &geo_release_view, &geo_draw_points,
    &geo_create_view_ex, &geo_draw_primitives, nullptr};
const CaliperGeometryV1_2 kGeom12 = {sizeof(CaliperGeometryV1_2),
    &geo_caps, &geo_create_view, &geo_release_view, &geo_draw_points,
    &geo_create_view_ex, &geo_draw_primitives_v12, nullptr};
const CaliperGeometryV1_3 kGeom13 = {sizeof(CaliperGeometryV1_3),
    &geo_caps, &geo_create_view, &geo_release_view, &geo_draw_points,
    &geo_create_view_ex, &geo_draw_primitives_v13, nullptr};

// --- caliper.export.v1: the terminal sink (PUBLISHING.md §3, Rung E) ---------
// A veneer by composition: temp geom_create_view_ex(DEPTH) → the EXISTING
// draw_primitives host path (every gate, every byte-exact behavior reused) →
// debug_readback_rgba8 (renderer-side) → stb PNG + JSON sidecar → destroy the
// view. No new render code, no retained draw state. Refusal purity extends to
// the filesystem (the PNG lands via temp-then-rename, the sidecar after), and
// caps() tracks the geometry primitives cap so export degrades in lockstep on a
// headless / no-renderer host.
#ifndef CALIPER_GIT_COMMIT
#define CALIPER_GIT_COMMIT "unknown"   // configured in at build time (one CMake line)
#endif

uint32_t ex_caps(void) {
    TensorBridge* b = bridge();
    const bool live = b && (b->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES);
    return live ? CALIPER_EXPORT_CAP_VIEW_PNG : 0u;
}

// Distinct colormap ids used by the draws, first-seen order. Reads only the
// frozen v1.1 prefix (color_mode/colormap), so it is safe at ANY draw_stride
// (v1.1/v1.2/v1.3 records the caller may hand in, stride-widened or not).
std::vector<int32_t> ex_colormaps(const CaliperGeomDrawV1_3* draws,
                                  uint32_t count, uint32_t stride) {
    std::vector<int32_t> out;
    if (!draws || count == 0 || stride == 0) return out;
    const auto* p = reinterpret_cast<const uint8_t*>(draws);
    for (uint32_t i = 0; i < count; ++i) {
        const auto* d = reinterpret_cast<const CaliperGeomDrawV1_3*>(
            p + static_cast<size_t>(i) * stride);
        if (d->base.base.color_mode == CALIPER_GEOM_COLOR_COLORMAP) {
            const int32_t id = d->base.base.colormap;
            bool seen = false;
            for (int32_t s : out) if (s == id) { seen = true; break; }
            if (!seen) out.push_back(id);
        }
    }
    return out;
}

// Render one frame to a fresh offscreen (w,h) and read it back TOP-DOWN. Returns
// false (nothing written, nothing to clean up on the FS — this touches no files)
// on any gate: no renderer/bridge, primitives cap absent, null cam, bad dims, a
// draw the geometry gate battery rejects, or a short/failed readback.
bool ex_render_readback(uint32_t w, uint32_t h, const CaliperGeomCamera* cam,
                        const CaliperGeomDrawV1_3* draws, uint32_t draw_count,
                        uint32_t draw_stride, uint32_t clear_rgba,
                        std::vector<uint8_t>& out_px) {
    TensorBridge* b = bridge();
    HostRenderer* r = g_renderer;
    if (!b || !r || !cam) return false;
    if (!(b->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES)) return false;
    if (w == 0 || h == 0 || w > CALIPER_EXPORT_MAX_DIM || h > CALIPER_EXPORT_MAX_DIM)
        return false;
    CaliperTextureId view = b->geom_create_view_ex(w, h, CALIPER_GEOM_VIEW_DEPTH);
    if (view == 0) return false;
    const bool drew = b->geom_draw_primitives_v1_3(view, cam, draws, draw_count,
                                                   draw_stride, clear_rgba);
    if (!drew) { b->geom_release_view(view); return false; }
    out_px = r->debug_readback_rgba8(view, static_cast<int>(w),
                                     static_cast<int>(h));
    b->geom_release_view(view);
    return out_px.size() == static_cast<size_t>(w) * h * 4u;
}

ExportProvenance ex_provenance(uint32_t w, uint32_t h, uint32_t clear_rgba,
                               const CaliperGeomCamera* cam,
                               const CaliperGeomDrawV1_3* draws,
                               uint32_t draw_count, uint32_t draw_stride,
                               const char* state_json) {
    ExportProvenance p;
    p.version      = kHostVersionStr;
    p.git_commit   = CALIPER_GIT_COMMIT;
    p.backend      = (g_renderer && g_renderer->name()) ? g_renderer->name() : "none";
    p.platform     = export_platform_string();
    p.timestamp_utc = export_utc_timestamp();
    p.width        = w;
    p.height       = h;
    p.clear_rgba   = clear_rgba;
    p.draw_count   = draw_count;
    p.view16       = cam ? cam->view : nullptr;
    p.proj16       = cam ? cam->proj : nullptr;
    p.colormaps    = ex_colormaps(draws, draw_count, draw_stride);
    p.state_json   = state_json;
    return p;
}

uint32_t ex_view_png(const char* path, uint32_t w, uint32_t h,
                     const CaliperGeomCamera* cam,
                     const CaliperGeomDrawV1_3* draws, uint32_t draw_count,
                     uint32_t draw_stride, uint32_t clear_rgba,
                     const char* state_json) {
    if (!path) return 0u;
    std::vector<uint8_t> px;
    if (!ex_render_readback(w, h, cam, draws, draw_count, draw_stride,
                            clear_rgba, px))
        return 0u;
    // The pixels are in hand; the target file is only ever touched by the atomic
    // rename below — a refusal above left the disk exactly as it was.
    if (!export_write_png_atomic(path, px.data(), w, h)) return 0u;
    const ExportProvenance prov = ex_provenance(w, h, clear_rgba, cam, draws,
                                                draw_count, draw_stride, state_json);
    const std::string json = export_build_sidecar_json(prov);
    if (!export_write_text_atomic(std::string(path) + ".json", json)) {
        // Refusal purity extends to the filesystem: 0 must mean the disk is
        // exactly as it was, and a PNG without its sidecar is a screenshot,
        // not a figure (PUBLISHING.md invariant) — roll the PNG back rather
        // than orphan it. (E1-review LOW finding.)
        std::error_code ec;
        std::filesystem::remove(path, ec);
        return 0u;
    }
    return 1u;
}

// One sequence live at a time (v0). The mutex guards this bookkeeping ONLY —
// the active/handle/frame_count fields and the last-frame provenance — NOT the
// renderer maps the frames actually render through. Export composes the
// frame-thread-owned renderer (ex_render_readback → the geometry draw path), so
// every export entry point MUST be called from the frame thread; the mutex
// serializes the bookkeeping but does not make export safe against a renderer
// running on another thread. The E2 exemplars call it inline, on the frame
// thread, for exactly this reason.
struct ExportSequence {
    std::mutex mtx;
    bool     active = false;
    uint64_t handle = 0;
    uint64_t next_handle = 1;
    std::string dir;
    std::string state_json;
    bool        has_state = false;
    uint32_t w = 0, h = 0;
    uint32_t frame_count = 0;
    // Last frame's provenance (a video has a per-frame camera): the sequence
    // sidecar records the most recent one alongside frame_count.
    CaliperGeomCamera last_cam{};
    uint32_t last_clear = 0;
    uint32_t last_draw_count = 0;
    std::vector<int32_t> last_colormaps;
};
ExportSequence g_seq;

uint64_t ex_begin_sequence(const char* dir, uint32_t w, uint32_t h,
                           const char* state_json) {
    if (!dir) return 0u;
    if (w == 0 || h == 0 || w > CALIPER_EXPORT_MAX_DIM || h > CALIPER_EXPORT_MAX_DIM)
        return 0u;
    TensorBridge* b = bridge();
    if (!b || !(b->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES)) return 0u;
    std::lock_guard<std::mutex> lk(g_seq.mtx);
    if (g_seq.active) return 0u;   // one at a time (v0)
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return 0u;
    g_seq.active = true;
    g_seq.handle = g_seq.next_handle++;
    g_seq.dir = dir;
    g_seq.has_state = state_json != nullptr;
    g_seq.state_json = state_json ? state_json : "";
    g_seq.w = w;
    g_seq.h = h;
    g_seq.frame_count = 0;
    return g_seq.handle;
}

uint32_t ex_frame(uint64_t seq, const CaliperGeomCamera* cam,
                  const CaliperGeomDrawV1_3* draws, uint32_t draw_count,
                  uint32_t draw_stride, uint32_t clear_rgba) {
    std::lock_guard<std::mutex> lk(g_seq.mtx);
    if (!g_seq.active || seq == 0 || seq != g_seq.handle) return 0u;
    std::vector<uint8_t> px;
    if (!ex_render_readback(g_seq.w, g_seq.h, cam, draws, draw_count,
                            draw_stride, clear_rgba, px))
        return 0u;
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%06u.png", g_seq.frame_count);
    const std::string frame_path =
        (std::filesystem::path(g_seq.dir) / name).string();
    if (!export_write_png_atomic(frame_path, px.data(), g_seq.w, g_seq.h))
        return 0u;
    if (cam) g_seq.last_cam = *cam;
    g_seq.last_clear = clear_rgba;
    g_seq.last_draw_count = draw_count;
    g_seq.last_colormaps = ex_colormaps(draws, draw_count, draw_stride);
    g_seq.frame_count++;
    return 1u;
}

void ex_end_sequence(uint64_t seq) {
    std::lock_guard<std::mutex> lk(g_seq.mtx);
    if (!g_seq.active || seq == 0 || seq != g_seq.handle) return;
    ExportProvenance p;
    p.version       = kHostVersionStr;
    p.git_commit    = CALIPER_GIT_COMMIT;
    p.backend       = (g_renderer && g_renderer->name()) ? g_renderer->name() : "none";
    p.platform      = export_platform_string();
    p.timestamp_utc = export_utc_timestamp();
    p.width         = g_seq.w;
    p.height        = g_seq.h;
    p.clear_rgba    = g_seq.last_clear;
    p.draw_count    = g_seq.last_draw_count;
    p.view16        = g_seq.last_cam.view;
    p.proj16        = g_seq.last_cam.proj;
    p.colormaps     = g_seq.last_colormaps;
    p.state_json    = g_seq.has_state ? g_seq.state_json.c_str() : nullptr;
    p.is_sequence   = true;
    p.frame_count   = g_seq.frame_count;
    const std::string json = export_build_sidecar_json(p);
    const std::string sidecar =
        (std::filesystem::path(g_seq.dir) / "sequence.json").string();
    export_write_text_atomic(sidecar, json);
    g_seq.active = false;
    g_seq.handle = 0;
    g_seq.dir.clear();
    g_seq.state_json.clear();
    g_seq.has_state = false;
}

const CaliperExportV1 kExport = {sizeof(CaliperExportV1), &ex_caps, &ex_view_png,
    &ex_begin_sequence, &ex_frame, &ex_end_sequence, nullptr};

const std::set<std::string> kIds = {CALIPER_UI_V1, CALIPER_LOG_V1,
                                    CALIPER_JOBS_V1, CALIPER_DEVICE_V1,
                                    CALIPER_METRICS_V1,
                                    CALIPER_METRICS_V1_1,
                                    CALIPER_TENSOR_BRIDGE_V1,
                                    CALIPER_TENSOR_BRIDGE_V1_1,
                                    CALIPER_TENSOR_BRIDGE_V1_2,
                                    CALIPER_GEOMETRY_V1,
                                    CALIPER_GEOMETRY_V1_1,
                                    CALIPER_GEOMETRY_V1_2,
                                    CALIPER_GEOMETRY_V1_3,
                                    CALIPER_ARTIFACTS_V1, CALIPER_DATA_V1,
                                    CALIPER_FEED_V1, CALIPER_EXPORT_V1};

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

    // Open the artifact store rooted in the app data dir; same non-fatal
    // discipline as metrics (thunks no-op if this fails).
    const std::string art_root = caliper::app_data_path("");
    g_artifacts_open = g_artifacts.open(art_root);
    if (!g_artifacts_open)
        std::fprintf(stderr,
                     "[artifacts] failed to open store under %s; "
                     "artifacts.v1 will no-op\n",
                     art_root.c_str());

    // Open the data store; same non-fatal discipline.
    const std::string data_path = caliper::app_data_path("data.duckdb");
    g_data_open = g_data.open(data_path);
    if (!g_data_open)
        std::fprintf(stderr,
                     "[data] failed to open %s; data.v1 will no-op\n",
                     data_path.c_str());

    // Route the bridge's acceptance-rule rejections through caliper.log.v1
    // (retires the C4 stderr placeholder inside tensor_bridge.cpp).
    set_bridge_log_sink(&log_impl);

#if defined(__APPLE__) || defined(_WIN32)
    // Start the platform telemetry provider (feed spec §4 / T2 on macOS, §6.2
    // on Windows): it probes the privilege-free sensors and registers the
    // readable ones into g_feed, then samples at 10 Hz. g_feed is a
    // process-lifetime static, so it exists here; the provider is JOINED in
    // services_shutdown BEFORE any teardown (below). Both the exe (main.cpp)
    // and the embed core (embed_core.cpp) reach this, and the start/stop pair
    // survives the embed create/shutdown/create cycling. Hosts with no
    // provider (Linux/other) — g_feed keeps zero channels (honest degradation,
    // the T1 default).
    feed_provider_start(g_feed);
#endif
}

void services_shutdown() {
    // Workers first (they may still be writing metrics/artifacts), then the
    // stores. Flags flip first so any thunk racing the close no-ops.
#if defined(__APPLE__) || defined(_WIN32)
    // Stop + JOIN the telemetry provider before anything else: its thread writes
    // into g_feed, so it must be joined here (BEFORE teardown, and before g_feed's
    // own process-exit static dtor). Safe if never started; re-startable on the
    // next services_init — so the embed create/shutdown/create battery cycles the
    // provider cleanly, twice, with no lingering thread across a shutdown.
    feed_provider_stop();
#endif
    g_jobs.cancel_all_and_join();
    g_metrics_open = false;
    g_artifacts_open = false;
    g_data_open = false;
    g_data.close();
    g_artifacts.close();
    g_metrics.close();
}

ArtifactStore& host_artifact_store() { return g_artifacts; }

DataStore& host_data_store() { return g_data; }

JobSystem& host_job_system() { return g_jobs; }

MetricsStore& host_metrics_store() { return g_metrics; }

FeedStore& host_feed_store() { return g_feed; }

void services_set_renderer(HostRenderer* renderer) {
    g_renderer = renderer;
    // Dropping the renderer drops the bridge's HostRenderer& before teardown;
    // a later set_renderer() rebuilds it lazily on the next thunk call.
    if (!renderer) g_bridge.reset();
}

HostRenderer* services_renderer() { return g_renderer; }

void set_applet_log_sink(void (*sink)(void*, CaliperLogLevel, const char*),
                         void* userdata) {
    g_applet_log_sink = sink;
    g_applet_log_userdata = userdata;
}

const void* services_get(const char* id) {
    if (!id) return nullptr;
    if (std::strcmp(id, CALIPER_UI_V1) == 0)     return &kUi;
    if (std::strcmp(id, CALIPER_LOG_V1) == 0)    return &kLog;
    if (std::strcmp(id, CALIPER_JOBS_V1) == 0)   return &kJobs;
    if (std::strcmp(id, CALIPER_DEVICE_V1) == 0) return &kDevice;
    if (std::strcmp(id, CALIPER_METRICS_V1) == 0) return &kMetrics;
    if (std::strcmp(id, CALIPER_METRICS_V1_1) == 0) return &kMetrics11;
    if (std::strcmp(id, CALIPER_TENSOR_BRIDGE_V1) == 0) return &kBridge;
    if (std::strcmp(id, CALIPER_TENSOR_BRIDGE_V1_1) == 0) return &kBridge11;
    if (std::strcmp(id, CALIPER_TENSOR_BRIDGE_V1_2) == 0) return &kBridge12;
    if (std::strcmp(id, CALIPER_GEOMETRY_V1) == 0) return &kGeom1;
    if (std::strcmp(id, CALIPER_GEOMETRY_V1_1) == 0) return &kGeom11;
    if (std::strcmp(id, CALIPER_GEOMETRY_V1_2) == 0) return &kGeom12;
    if (std::strcmp(id, CALIPER_GEOMETRY_V1_3) == 0) return &kGeom13;
    if (std::strcmp(id, CALIPER_ARTIFACTS_V1) == 0) return &kArtifacts;
    if (std::strcmp(id, CALIPER_DATA_V1) == 0) return &kData;
    if (std::strcmp(id, CALIPER_FEED_V1) == 0) return &kFeed;
    if (std::strcmp(id, CALIPER_EXPORT_V1) == 0) return &kExport;
    return nullptr;   // unknown ids: NULL, never UB (§6b)
}

const std::set<std::string>& service_ids() { return kIds; }

} // namespace caliper_host
