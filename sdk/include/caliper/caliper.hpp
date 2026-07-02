#pragma once
// Caliper C++ sugar (PLATFORM.md §8). Header-only, optional by design: a C
// applet can implement abi.h by hand. Requires C++20 (designated inits).
#include <caliper/abi.h>
#include <caliper/services/log_v1.h>
#include <caliper/services/ui_v1.h>

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

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
