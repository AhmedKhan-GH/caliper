#pragma once
// Caliper C++ sugar (PLATFORM.md §8). Header-only, optional by design: a C
// applet can implement abi.h by hand. Requires C++20 (designated inits).
#include <caliper/abi.h>
#include <caliper/services/log_v1.h>
#include <caliper/services/ui_v1.h>
#include <caliper/services/jobs_v1.h>
#include <caliper/services/device_v1.h>
#include <caliper/services/metrics_v1.h>
#include <caliper/services/artifacts_v1.h>
#include <caliper/services/data_v1.h>
#include <caliper/services/tensor_bridge_v1.h>
#include <caliper/services/tensor_bridge_v1_1.h>

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

#include <string>
#include <vector>

namespace caliper {

struct Frame {
    int32_t fb_width = 0, fb_height = 0;   // PHYSICAL pixels (§6a)
    float   dpi_scale = 1.0f;
    double  time_sec = 0.0, delta_sec = 0.0;
    static Frame from(const CaliperFrameInfo& fi) {
        Frame f;
        f.fb_width = fi.fb_width;  f.fb_height = fi.fb_height;
        f.dpi_scale = fi.dpi_scale;
        f.time_sec = fi.time_sec;  f.delta_sec = fi.delta_sec;
        return f;
    }
};

class Host {
public:
    Host() = default;
    explicit Host(const CaliperHost* raw) : raw_(raw) {
        if (raw_ && raw_->get_service)
            log_ = static_cast<const CaliperLogV1*>(
                raw_->get_service(raw_, CALIPER_LOG_V1));
    }
    const CaliperHost* raw() const { return raw_; }
    const void* service(const char* id) const {
        return (raw_ && raw_->get_service) ? raw_->get_service(raw_, id) : nullptr;
    }
    const char* data_dir() const {
        return (raw_ && raw_->applet_data_dir) ? raw_->applet_data_dir : "";
    }
    void log(CaliperLogLevel lv, const char* msg) const {
        if (log_ && log_->log) log_->log(lv, msg);
    }
    void log_info(const char* m) const  { log(CALIPER_LOG_INFO, m); }
    void log_error(const char* m) const { log(CALIPER_LOG_ERROR, m); }

private:
    const CaliperHost* raw_ = nullptr;
    const CaliperLogV1* log_ = nullptr;
};

// Typed wrapper over caliper.jobs.v1 (§7.5). Falsy when the host doesn't vend
// the service; every method null-guards its fn pointer so it stays inert (not
// UB) on a headless/older host. Job fns run UNGUARDED on host worker threads.
class Jobs {
public:
    Jobs() = default;
    explicit Jobs(const Host& host)
        : t_(static_cast<const CaliperJobsV1*>(host.service(CALIPER_JOBS_V1))) {}
    explicit operator bool() const { return t_ && t_->submit; }
    uint64_t submit(const char* label, CaliperJobFn fn, void* user) const {
        return (t_ && t_->submit) ? t_->submit(label, fn, user) : 0;
    }
    void request_cancel(uint64_t id) const {
        if (t_ && t_->request_cancel) t_->request_cancel(id);
    }
    bool is_running(uint64_t id) const {
        return (t_ && t_->is_running) ? t_->is_running(id) : false;
    }
    float progress_of(uint64_t id) const {
        return (t_ && t_->progress_of) ? t_->progress_of(id) : 0.0f;
    }
private:
    const CaliperJobsV1* t_ = nullptr;
};

// Typed wrapper over caliper.metrics.v1 (§7.6). Falsy when the host doesn't
// vend the service; every method null-guards its fn pointer so it stays inert
// (not UB) on a headless/older host. The writers are callable from job threads
// (the host serializes internally). image() takes a CPU-resident HWC u8 tensor;
// a non-conforming tensor is dropped by the host thunk (see host_services.cpp).
class Metrics {
public:
    Metrics() = default;
    explicit Metrics(const Host& host)
        : t_(static_cast<const CaliperMetricsV1*>(host.service(CALIPER_METRICS_V1))) {}
    explicit operator bool() const { return t_ && t_->begin_run; }
    uint64_t begin_run(const char* experiment, const char* run_name) const {
        return (t_ && t_->begin_run) ? t_->begin_run(experiment, run_name) : 0;
    }
    void end_run(uint64_t run) const {
        if (t_ && t_->end_run) t_->end_run(run);
    }
    void scalar(uint64_t run, const char* tag, int64_t step, double value) const {
        if (t_ && t_->scalar) t_->scalar(run, tag, step, value);
    }
    void histogram(uint64_t run, const char* tag, int64_t step,
                   const float* values, int64_t count) const {
        if (t_ && t_->histogram) t_->histogram(run, tag, step, values, count);
    }
    void image(uint64_t run, const char* tag, int64_t step,
               const CaliperTensor* hwc_u8) const {
        if (t_ && t_->image) t_->image(run, tag, step, hwc_u8);
    }
    void hparams_json(uint64_t run, const char* json_utf8) const {
        if (t_ && t_->hparams_json) t_->hparams_json(run, json_utf8);
    }
private:
    const CaliperMetricsV1* t_ = nullptr;
};

// Typed wrapper over caliper.artifacts.v1 (§7.8): content-addressed checkpoint
// storage. Falsy when the host doesn't vend the service; every call is inert
// then (put returns "", path_of nullptr, exists false).
class Artifacts {
public:
    Artifacts() = default;
    explicit Artifacts(const Host& host)
        : t_(static_cast<const CaliperArtifactsV1*>(
              host.service(CALIPER_ARTIFACTS_V1))) {}
    explicit operator bool() const { return t_ && t_->put; }
    // Returns the 64-hex digest, or "" on failure/absence.
    std::string put(const char* name, const void* bytes, uint64_t len,
                    uint64_t run = 0) const {
        if (!(t_ && t_->put)) return {};
        char digest[65] = {};
        return t_->put(name, bytes, len, run, digest) ? std::string(digest)
                                                      : std::string();
    }
    // Host-owned string, valid until the next artifacts.v1 call; nullptr if
    // unknown or the service is absent.
    const char* path_of(const char* digest_or_name) const {
        return (t_ && t_->path_of) ? t_->path_of(digest_or_name) : nullptr;
    }
    bool exists(const char* digest_or_name) const {
        return t_ && t_->exists && t_->exists(digest_or_name);
    }
private:
    const CaliperArtifactsV1* t_ = nullptr;
};

// Typed wrapper over caliper.data.v1 (§7.7): SQL in, Arrow streams out.
// Falsy-inert when absent. The raw stream API is fully exposed; the
// drain_numeric helper covers the common all-numeric-columns case (each
// column widened to double) so simple consumers never touch Arrow buffers.
class Data {
public:
    Data() = default;
    explicit Data(const Host& host)
        : t_(static_cast<const CaliperDataV1*>(
              host.service(CALIPER_DATA_V1))) {}
    explicit operator bool() const { return t_ && t_->query; }
    bool query(const char* sql, ArrowArrayStream* out) const {
        return t_ && t_->query && t_->query(sql, out);
    }
    bool register_dataset(const char* name, const char* uri) const {
        return t_ && t_->register_dataset && t_->register_dataset(name, uri);
    }
    bool open_dataset(const char* name, ArrowArrayStream* out) const {
        return t_ && t_->open_dataset && t_->open_dataset(name, out);
    }
    const char* last_error() const {
        return (t_ && t_->last_error) ? t_->last_error()
                                      : "data.v1 is not available";
    }

    // Drain a stream of numeric columns (int8..int64 / uint8..uint64 /
    // float / double / bool) into column-major doubles. Releases the stream.
    // Returns false (and releases what it took) on a non-numeric column or
    // stream error. NULL values read as whatever the producer left in the
    // data buffer — callers that care about NULLs should walk the raw stream.
    static bool drain_numeric(ArrowArrayStream* stream,
                              std::vector<std::string>* names,
                              std::vector<std::vector<double>>* cols) {
        if (!stream || !stream->get_schema || !cols) return false;
        ArrowSchema schema = {};
        if (stream->get_schema(stream, &schema) != 0) {
            stream->release(stream);
            return false;
        }
        const int64_t n = schema.n_children;
        std::vector<char> fmt(static_cast<size_t>(n < 0 ? 0 : n), 0);
        cols->assign(fmt.size(), {});
        if (names) names->assign(fmt.size(), "");
        bool ok = true;
        for (int64_t c = 0; c < n; c++) {
            const char* f = schema.children[c]->format;
            // Arrow primitive format strings: c/C s/S i/I l/L = ints,
            // f/g = float32/64, b = bool. Anything else is non-numeric.
            const bool numeric = f && f[0] != '\0' && f[1] == '\0' &&
                                 std::string("cCsSiIlLfgb").find(f[0]) !=
                                     std::string::npos;
            if (!numeric) ok = false;
            fmt[static_cast<size_t>(c)] = numeric ? f[0] : 0;
            if (names && schema.children[c]->name)
                (*names)[static_cast<size_t>(c)] = schema.children[c]->name;
        }
        if (schema.release) schema.release(&schema);
        while (ok) {
            ArrowArray array = {};
            if (stream->get_next(stream, &array) != 0) { ok = false; break; }
            if (!array.release) break;  // end of stream
            for (int64_t c = 0; ok && c < array.n_children &&
                                c < static_cast<int64_t>(cols->size()); c++) {
                const ArrowArray* col = array.children[c];
                const void* buf = col->buffers[1];
                auto& out = (*cols)[static_cast<size_t>(c)];
                for (int64_t i = 0; i < array.length; i++) {
                    const int64_t at = col->offset + i;
                    switch (fmt[static_cast<size_t>(c)]) {
                        case 'c': out.push_back(static_cast<const int8_t*>(buf)[at]); break;
                        case 'C': out.push_back(static_cast<const uint8_t*>(buf)[at]); break;
                        case 's': out.push_back(static_cast<const int16_t*>(buf)[at]); break;
                        case 'S': out.push_back(static_cast<const uint16_t*>(buf)[at]); break;
                        case 'i': out.push_back(static_cast<const int32_t*>(buf)[at]); break;
                        case 'I': out.push_back(static_cast<const uint32_t*>(buf)[at]); break;
                        case 'l': out.push_back(static_cast<double>(static_cast<const int64_t*>(buf)[at])); break;
                        case 'L': out.push_back(static_cast<double>(static_cast<const uint64_t*>(buf)[at])); break;
                        case 'f': out.push_back(static_cast<const float*>(buf)[at]); break;
                        case 'g': out.push_back(static_cast<const double*>(buf)[at]); break;
                        case 'b': {  // bit-packed booleans
                            const uint8_t* bits = static_cast<const uint8_t*>(buf);
                            out.push_back((bits[at / 8] >> (at % 8)) & 1);
                            break;
                        }
                        default: ok = false; break;
                    }
                }
            }
            array.release(&array);
        }
        if (stream->release) stream->release(stream);
        return ok;
    }

private:
    const CaliperDataV1* t_ = nullptr;
};

// Typed wrapper over caliper.tensor_bridge.v1 (§7.4): a CaliperTensor becomes a
// live texture, crossing the ABI as an opaque CaliperTextureId. Falsy when the
// host doesn't vend the service; every method null-guards its fn pointer so it
// stays inert (not UB) on a headless/older host. FRAME-THREAD ONLY in v1: a
// tensor produced by a background job is consumed at frame time, so call these
// on the UI/frame thread, never from a job worker (the C8 pattern).
class Bridge {
public:
    Bridge() = default;
    explicit Bridge(const Host& host)
        : t_(static_cast<const CaliperTensorBridgeV1*>(
              host.service(CALIPER_TENSOR_BRIDGE_V1))),
          t11_(static_cast<const CaliperTensorBridgeV1_1*>(
              host.service(CALIPER_TENSOR_BRIDGE_V1_1))) {}
    explicit operator bool() const { return t_ && t_->texture_from_tensor; }
    CaliperTextureId texture_from_tensor(const CaliperTensor* t,
                                         uint32_t flags = 0) const {
        return (t_ && t_->texture_from_tensor)
            ? t_->texture_from_tensor(t, flags) : 0;
    }
    bool update_texture(CaliperTextureId tex, const CaliperTensor* t) const {
        return (t_ && t_->update_texture) ? t_->update_texture(tex, t) : false;
    }
    void release_texture(CaliperTextureId tex) const {
        if (t_ && t_->release_texture) t_->release_texture(tex);
    }
    CaliperTextureId texture_from_tensor_mapped(const CaliperTensor* t,
                                                int32_t colormap, float vmin,
                                                float vmax,
                                                uint32_t flags = 0) const {
        return (t_ && t_->texture_from_tensor_mapped)
            ? t_->texture_from_tensor_mapped(t, colormap, vmin, vmax, flags) : 0;
    }
    bool alloc_shared(CaliperDType dtype, int32_t ndim, const int64_t* shape,
                      CaliperTensor* out_tensor,
                      CaliperTextureId* out_texture) const {
        return (t_ && t_->alloc_shared)
            ? t_->alloc_shared(dtype, ndim, shape, out_tensor, out_texture)
            : false;
    }
    void free_shared(CaliperTextureId tex) const {
        if (t_ && t_->free_shared) t_->free_shared(tex);
    }
    // v1.1 capability bits — 0 on a v1-only or headless host (D24). Query
    // once per handoff site and pass to adapters::stream_to_tensor; bit
    // CALIPER_BRIDGE_CAP_STREAM_ORDERED means a non-NULL CaliperTensor.stream
    // replaces the adapter's device drain.
    uint32_t caps() const { return (t11_ && t11_->caps) ? t11_->caps() : 0u; }
    // Opaque id -> ImTextureID for ImGui::Image (§5.4: the id's value IS the
    // host's ImGui-compatible handle for this backend; applets never interpret
    // it, they only cast it here).
    static ImTextureID imtex(CaliperTextureId tex) { return (ImTextureID)tex; }
private:
    const CaliperTensorBridgeV1* t_ = nullptr;
    const CaliperTensorBridgeV1_1* t11_ = nullptr;
};

// Snapshot of caliper.device.v1 (§7.3). Defaults to CPU when the host doesn't
// vend the service; name is host-owned (valid for the process lifetime).
struct Device {
    CaliperDeviceKind kind = CALIPER_DEV_CPU;
    int32_t index = 0;
    const char* name = "CPU";              // host-owned string
    uint64_t free_memory_hint = 0;
    static Device query(const Host& host) {
        Device d;
        auto* t = static_cast<const CaliperDeviceV1*>(
            host.service(CALIPER_DEVICE_V1));
        if (t && t->kind) {
            d.kind = t->kind();
            d.index = t->index ? t->index() : 0;
            d.name = t->name ? t->name() : "";
            d.free_memory_hint = t->free_memory_hint ? t->free_memory_hint() : 0;
        }
        return d;
    }
};

class Applet {
public:
    virtual ~Applet() = default;
    virtual bool on_init(Host& host) = 0;
    virtual void on_frame(const Frame& frame) = 0;
    virtual void on_cleanup() {}
};

namespace ui {
// SetAllocatorFunctions + SetCurrentContext x3, in one call authors cannot
// get wrong (§6d). Returns false when the host has no ui.v1 (headless).
inline bool connect(const CaliperHost* h) {
    if (!h || !h->get_service) return false;
    auto* ui = static_cast<const CaliperUiV1*>(h->get_service(h, CALIPER_UI_V1));
    if (!ui) return false;
    CaliperImGuiAllocFn alloc = nullptr;
    CaliperImGuiFreeFn  free_fn = nullptr;
    void* user = nullptr;
    ui->imgui_allocators(&alloc, &free_fn, &user);
    if (alloc && free_fn)
        ImGui::SetAllocatorFunctions(reinterpret_cast<ImGuiMemAllocFunc>(alloc),
                                     reinterpret_cast<ImGuiMemFreeFunc>(free_fn),
                                     user);
    ImGui::SetCurrentContext(ui->imgui_context());
    ImPlot::SetCurrentContext(ui->implot_context());
    ImPlot3D::SetCurrentContext(ui->implot3d_context());
    return true;
}
} // namespace ui

struct AppletMeta {
    const char* id;
    const char* version;
    const char* name;
    const char* summary;
    const char* tag;
    const char* services[15];   // NULL-terminated by aggregate zero-init
};

} // namespace caliper

// Generates: descriptor + the five C bridge functions + the single export.
// Field order is fixed: id, version, name, summary, tag, services.
#define CALIPER_APPLET(CLASS, ...)                                             \
    namespace caliper_applet_gen {                                             \
    static const ::caliper::AppletMeta kMeta{__VA_ARGS__};                     \
    struct Holder {                                                            \
        CLASS obj;                                                             \
        ::caliper::Host host;                                                  \
    };                                                                         \
    static void* cal_create(void) {                                            \
        try { return new Holder(); } catch (...) { return nullptr; }           \
    }                                                                          \
    static void cal_destroy(void* s) {                                         \
        try { delete static_cast<Holder*>(s); } catch (...) {}                 \
    }                                                                          \
    static bool cal_initialize(void* s, const CaliperHost* h) {                \
        auto* hold = static_cast<Holder*>(s);                                  \
        hold->host = ::caliper::Host(h);                                       \
        ::caliper::ui::connect(h);                                             \
        try { return hold->obj.on_init(hold->host); }                          \
        catch (...) {                                                          \
            hold->host.log_error("unhandled exception in on_init");            \
            return false;                                                      \
        }                                                                      \
    }                                                                          \
    static void cal_frame(void* s, const CaliperFrameInfo* fi) {               \
        auto* hold = static_cast<Holder*>(s);                                  \
        try { hold->obj.on_frame(::caliper::Frame::from(*fi)); }               \
        catch (...) {                                                          \
            hold->host.log_error("unhandled exception in on_frame");           \
        }                                                                      \
    }                                                                          \
    static void cal_cleanup(void* s) {                                         \
        auto* hold = static_cast<Holder*>(s);                                  \
        try { hold->obj.on_cleanup(); }                                        \
        catch (...) {                                                          \
            hold->host.log_error("unhandled exception in on_cleanup");         \
        }                                                                      \
    }                                                                          \
    } /* namespace caliper_applet_gen */                                       \
    extern "C" CALIPER_EXPORT const CaliperAppletDescriptor*                   \
    caliper_applet_descriptor(void) {                                          \
        static const CaliperAppletDescriptor kDesc = {                         \
            (uint32_t)sizeof(CaliperAppletDescriptor),                         \
            CALIPER_ABI_EPOCH,                                                 \
            ::caliper_applet_gen::kMeta.id,                                    \
            ::caliper_applet_gen::kMeta.version,                               \
            ::caliper_applet_gen::kMeta.name,                                  \
            ::caliper_applet_gen::kMeta.summary,                               \
            ::caliper_applet_gen::kMeta.tag,                                   \
            ::caliper_applet_gen::kMeta.services,                              \
            { (uint32_t)sizeof(CaliperAppletAPI),                              \
              &::caliper_applet_gen::cal_create,                               \
              &::caliper_applet_gen::cal_destroy,                              \
              &::caliper_applet_gen::cal_initialize,                           \
              &::caliper_applet_gen::cal_frame,                                \
              &::caliper_applet_gen::cal_cleanup } };                          \
        return &kDesc;                                                         \
    }
