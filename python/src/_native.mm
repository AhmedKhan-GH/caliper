// caliper._native — the Python binding of the embed C ABI (caliper/embed.h) and
// of the typed live views (caliper/live_views.h), plus the in-process window
// transport's render thread.
//
// One `Core` per process (the embed ABI's rule). It owns:
//   - the CaliperCore (create/shutdown, a log ring fed by CaliperLogFn),
//   - either a GLFW window whose NSView is handed to attach_canvas(WINDOW), or
//     an OFFSCREEN canvas (tests / CI / benchmarks / caliper.render()),
//   - the GLFW -> CaliperInputEvent translation (callbacks fire inside poll()),
//   - optionally a RENDER THREAD (start_render_thread) for caliper.window(): it
//     owns every core call while it runs.
//
// Two threading modes:
//
//   Synchronous (caliper.offscreen(), the test mode). Every core call runs on
//   the calling thread; publish_live() applies the snapshot before returning.
//
//   Render thread (caliper.window()). A std::thread ticks at `fps`: it settles
//   GPU-read claims, takes the newest published snapshot per view, applies
//   queued input events, applies the snapshots and renders a frame. ALL core
//   calls then run on that thread; calls made from Python are forwarded to it
//   and waited for. AppKit still needs the MAIN thread for window events: GLFW
//   callbacks fire inside poll()/wait_events()/run_event_loop() on the main
//   thread and only queue CaliperInputEvents. Metal encoding and CAMetalLayer
//   presentation from the render thread are legal, so the canvas repaints
//   whatever the main thread does; input, move/resize and close are processed
//   only while it pumps.
//
// Snapshot memory (SharedBuffer): every slot the window transport hands to
// torch is a caliper-made MTLBuffer with shared storage, exported as a DLPack
// capsule (kDLMetal for MPS producers, kDLCPU over -contents for CPU ones), so
// torch writes it in place and the core reads the same pages: on the GPU for
// heatmap/image/points (imported zero-copy), on the CPU for stats, series and
// histogram values. publish_live() takes the buffers themselves (ARC strong
// references), so nothing here needs the GIL to release them.
//
// Slot safety (the tearing-free rule): a slot the core may still read is
// CLAIMED and reported by reading_slots(); the producer never rewrites a
// claimed slot nor its last published one. Claims (live_views.h rules): the
// slot being applied; heatmap/image: the current and previous payloads until
// caliper_core_live_view_read_done() is 1 (their colormap/blit retired); points:
// the current payload while it is current (drawn every frame) and the previous
// one until read_done; series/histogram: none (copied at apply). A queued
// snapshot of a view is taken only once its previous claim cleared. The v1.2
// route (switch off: caliper_core_view_tensor + WAIT_GPU on this thread) has
// retired its read when the call returns, so it leaves no claim.
#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#include <caliper/embed.h>
#include <caliper/live_views.h>
#include <caliper/tensor.h>
#include <imgui.h>   // ImGuiKey values only (CaliperInputEvent.key == ImGuiKey)

#include "dlpack.h"

#include <pthread.h>
#include <signal.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// CALIPER_PY_VIEWS: embed.h declares the v1.2 views AND the build did not opt out.
#if defined(CALIPER_EMBED_HAVE_VIEWS) && !defined(CALIPER_PY_DISABLE_VIEWS)
#define CALIPER_PY_VIEWS 1
#endif
#ifndef CALIPER_PY_VIEWS
#error "caliper._native needs the embed v1.2 views (CALIPER_EMBED_HAVE_VIEWS)"
#endif

// THE SWITCH. true: heatmap and image snapshots go through
// caliper_core_live_apply like points/series/histogram (the typed view layer
// delegates them to the embed v1.2 view of the same name, so view_info /
// view_readback keep working). false (or CALIPER_PY_LIVE_APPLY=0): the embed
// v1.2 route, caliper_core_view_tensor + WAIT_GPU on the frame thread.
constexpr bool kHeatmapImageViaLiveApply = true;

// The heartbeat view's name (start_render_thread(heartbeat=True)): a strip that
// scrolls one texel per rendered frame, so a screenshot shows whether the
// canvas is still being repainted.
constexpr const char* kHeartbeatView = "caliper.render";

// DLPack >= 1.0 managed tensor (the vendored dlpack.h predates it). Layout per
// dlpack.h v1.0: version, manager_ctx, deleter, flags, dl_tensor.
struct CaliperDLPackVersion { uint32_t major, minor; };
struct CaliperDLManagedTensorVersioned {
    CaliperDLPackVersion version;
    void* manager_ctx;
    void (*deleter)(CaliperDLManagedTensorVersioned* self);
    uint64_t flags;
    DLTensor dl_tensor;
};

namespace {

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

// Same clock as Python's time.monotonic_ns() on macOS (mach_absolute_time).
uint64_t mono_ns() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

void require_main_thread(const char* what) {
    if (!pthread_main_np())
        throw std::runtime_error(std::string("caliper: ") + what +
                                 " must be called from the main thread (AppKit/GLFW)");
}

// --- log ring --------------------------------------------------------------
struct LogRing {
    std::mutex mu;
    std::deque<std::string> lines;
    size_t cap = 512;
    bool echo = false;
    void push(int level, const char* msg) {
        static const char* kTag[] = {"debug", "info", "warn", "error"};
        std::string s = std::string(kTag[(level >= 0 && level <= 3) ? level : 1]) + ": " +
                        (msg ? msg : "");
        if (echo) std::fprintf(stderr, "[caliper] %s\n", s.c_str());
        std::lock_guard<std::mutex> lk(mu);
        lines.push_back(std::move(s));
        while (lines.size() > cap) lines.pop_front();
    }
};

void log_trampoline(void* ud, int level, const char* msg) {
    static_cast<LogRing*>(ud)->push(level, msg);
}

// --- GLFW -> ImGuiKey (subset; text input arrives separately as TEXT) -------
int glfw_to_imgui_key(int k) {
    if (k >= GLFW_KEY_A && k <= GLFW_KEY_Z) return ImGuiKey_A + (k - GLFW_KEY_A);
    if (k >= GLFW_KEY_0 && k <= GLFW_KEY_9) return ImGuiKey_0 + (k - GLFW_KEY_0);
    if (k >= GLFW_KEY_F1 && k <= GLFW_KEY_F12) return ImGuiKey_F1 + (k - GLFW_KEY_F1);
    switch (k) {
        case GLFW_KEY_TAB: return ImGuiKey_Tab;
        case GLFW_KEY_LEFT: return ImGuiKey_LeftArrow;
        case GLFW_KEY_RIGHT: return ImGuiKey_RightArrow;
        case GLFW_KEY_UP: return ImGuiKey_UpArrow;
        case GLFW_KEY_DOWN: return ImGuiKey_DownArrow;
        case GLFW_KEY_PAGE_UP: return ImGuiKey_PageUp;
        case GLFW_KEY_PAGE_DOWN: return ImGuiKey_PageDown;
        case GLFW_KEY_HOME: return ImGuiKey_Home;
        case GLFW_KEY_END: return ImGuiKey_End;
        case GLFW_KEY_INSERT: return ImGuiKey_Insert;
        case GLFW_KEY_DELETE: return ImGuiKey_Delete;
        case GLFW_KEY_BACKSPACE: return ImGuiKey_Backspace;
        case GLFW_KEY_SPACE: return ImGuiKey_Space;
        case GLFW_KEY_ENTER: return ImGuiKey_Enter;
        case GLFW_KEY_KP_ENTER: return ImGuiKey_KeypadEnter;
        case GLFW_KEY_ESCAPE: return ImGuiKey_Escape;
        case GLFW_KEY_MINUS: return ImGuiKey_Minus;
        case GLFW_KEY_EQUAL: return ImGuiKey_Equal;
        case GLFW_KEY_COMMA: return ImGuiKey_Comma;
        case GLFW_KEY_PERIOD: return ImGuiKey_Period;
        case GLFW_KEY_SLASH: return ImGuiKey_Slash;
        case GLFW_KEY_SEMICOLON: return ImGuiKey_Semicolon;
        case GLFW_KEY_APOSTROPHE: return ImGuiKey_Apostrophe;
        case GLFW_KEY_LEFT_BRACKET: return ImGuiKey_LeftBracket;
        case GLFW_KEY_RIGHT_BRACKET: return ImGuiKey_RightBracket;
        case GLFW_KEY_BACKSLASH: return ImGuiKey_Backslash;
        case GLFW_KEY_GRAVE_ACCENT: return ImGuiKey_GraveAccent;
        case GLFW_KEY_LEFT_SHIFT: return ImGuiKey_LeftShift;
        case GLFW_KEY_RIGHT_SHIFT: return ImGuiKey_RightShift;
        case GLFW_KEY_LEFT_CONTROL: return ImGuiKey_LeftCtrl;
        case GLFW_KEY_RIGHT_CONTROL: return ImGuiKey_RightCtrl;
        case GLFW_KEY_LEFT_ALT: return ImGuiKey_LeftAlt;
        case GLFW_KEY_RIGHT_ALT: return ImGuiKey_RightAlt;
        case GLFW_KEY_LEFT_SUPER: return ImGuiKey_LeftSuper;
        case GLFW_KEY_RIGHT_SUPER: return ImGuiKey_RightSuper;
        default: return ImGuiKey_None;
    }
}

int glfw_mods_to_caliper(int m) {
    int r = 0;
    if (m & GLFW_MOD_CONTROL) r |= CALIPER_MOD_CTRL;
    if (m & GLFW_MOD_SHIFT) r |= CALIPER_MOD_SHIFT;
    if (m & GLFW_MOD_ALT) r |= CALIPER_MOD_ALT;
    if (m & GLFW_MOD_SUPER) r |= CALIPER_MOD_SUPER;
    return r;
}

template <typename T>
nb::ndarray<nb::numpy, T> owned_array(std::vector<size_t> shape) {
    size_t n = 1;
    for (size_t s : shape) n *= s;
    T* buf = new T[n > 0 ? n : 1];
    std::memset(buf, 0, sizeof(T) * (n > 0 ? n : 1));
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T>(buf, shape.size(), shape.data(), owner);
}

// --- DLPack capsule -> CaliperTensor (Core.view_tensor) ---------------------
struct ParsedCapsule {
    DLTensor* dl = nullptr;
    void* managed = nullptr;
    bool versioned = false;
};

ParsedCapsule open_capsule(nb::handle cap, const char* what) {
    ParsedCapsule p;
    PyObject* o = cap.ptr();
    if (PyCapsule_IsValid(o, "dltensor")) {
        auto* m = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(o, "dltensor"));
        p.dl = &m->dl_tensor;
        p.managed = m;
    } else if (PyCapsule_IsValid(o, "dltensor_versioned")) {
        auto* m = static_cast<CaliperDLManagedTensorVersioned*>(
            PyCapsule_GetPointer(o, "dltensor_versioned"));
        if (m->version.major > 1)
            throw std::invalid_argument(std::string(what) +
                                        ": unsupported DLPack major version " +
                                        std::to_string(m->version.major));
        p.dl = &m->dl_tensor;
        p.managed = m;
        p.versioned = true;
    } else {
        throw std::invalid_argument(std::string(what) +
                                    ": expected an unconsumed DLPack capsule (tensor.__dlpack__())");
    }
    return p;
}

// Consume the capsule and run the producer's deleter (the core retained what
// it needs: the MTLBuffer for imports; CPU data is staged during the call).
void consume_capsule(nb::handle cap, const ParsedCapsule& p) {
    PyCapsule_SetName(cap.ptr(), p.versioned ? "used_dltensor_versioned" : "used_dltensor");
    if (!p.managed) return;
    if (p.versioned) {
        auto* m = static_cast<CaliperDLManagedTensorVersioned*>(p.managed);
        if (m->deleter) m->deleter(m);
    } else {
        auto* m = static_cast<DLManagedTensor*>(p.managed);
        if (m->deleter) m->deleter(m);
    }
}

CaliperTensor to_caliper_tensor(const DLTensor& dl, uint64_t extra_offset,
                                uint64_t* out_byte_offset) {
    CaliperTensor t{};
    t.struct_size = sizeof(CaliperTensor);
    if (dl.ndim < 0 || dl.ndim > 8)
        throw std::invalid_argument("view_tensor: ndim must be in [0, 8]");
    if (dl.dtype.lanes != 1)
        throw std::invalid_argument("view_tensor: vector dtypes (lanes != 1) unsupported");
    const int code = dl.dtype.code, bits = dl.dtype.bits;
    if (code == kDLFloat && bits == 32) t.dtype = CALIPER_DT_F32;
    else if (code == kDLFloat && bits == 16) t.dtype = CALIPER_DT_F16;
    else if (code == kDLBfloat && bits == 16) t.dtype = CALIPER_DT_BF16;
    else if (code == kDLInt && bits == 64) t.dtype = CALIPER_DT_I64;
    else if (code == kDLInt && bits == 32) t.dtype = CALIPER_DT_I32;
    else if (code == kDLUInt && bits == 8) t.dtype = CALIPER_DT_U8;
    else
        throw std::invalid_argument("view_tensor: unsupported dtype (code " +
                                    std::to_string(code) + ", bits " + std::to_string(bits) +
                                    "); pass float32 heatmaps or uint8 RGBA images");
    t.ndim = dl.ndim;
    for (int i = 0; i < dl.ndim; ++i) t.shape[i] = dl.shape[i];
    if (dl.strides) {
        for (int i = 0; i < dl.ndim; ++i) t.strides[i] = dl.strides[i];
    } else {
        int64_t s = 1;
        for (int i = dl.ndim - 1; i >= 0; --i) { t.strides[i] = s; s *= dl.shape[i]; }
    }
    t.device_index = dl.device.device_id;
    t.stream = nullptr;
    const uint64_t off = dl.byte_offset + extra_offset;
    if (dl.device.device_type == kDLMetal) {
        t.device = CALIPER_DEV_METAL;
        t.data = dl.data;               // the MTLBuffer object
        *out_byte_offset = off;
    } else if (dl.device.device_type == kDLCPU) {
        t.device = CALIPER_DEV_CPU;
        t.data = static_cast<char*>(dl.data) + off;
        *out_byte_offset = 0;
    } else {
        throw std::invalid_argument("view_tensor: only CPU and Metal (MPS) tensors are supported "
                                    "(DLPack device_type " +
                                    std::to_string((int)dl.device.device_type) + ")");
    }
    return t;
}

// ---------------------------------------------------------------------------
// SharedBuffer: a caliper-made shared-storage MTLBuffer, exported to torch
// ---------------------------------------------------------------------------
id<MTLDevice> metal_device() {
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev;
}

struct BufferCapsuleCtx {
    DLManagedTensor managed{};
    int64_t shape[8] = {0};
    int64_t strides[8] = {0};
    void* retained = nullptr;   // +1 id<MTLBuffer> (CFBridgingRetain)
};

void buffer_capsule_deleter(DLManagedTensor* m) {
    auto* ctx = static_cast<BufferCapsuleCtx*>(m->manager_ctx);
    if (ctx->retained) CFRelease(ctx->retained);
    delete ctx;
}

void dltensor_capsule_destructor(PyObject* cap) {
    // Only an unconsumed capsule still owns its tensor (the DLPack protocol).
    if (PyCapsule_IsValid(cap, "dltensor")) {
        auto* m = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
        if (m && m->deleter) m->deleter(m);
    }
}

bool dtype_from_name(const std::string& s, DLDataType* dl, CaliperDType* cd, size_t* item) {
    if (s == "float32") { *dl = DLDataType{kDLFloat, 32, 1}; *cd = CALIPER_DT_F32; *item = 4; return true; }
    if (s == "uint8") { *dl = DLDataType{kDLUInt, 8, 1}; *cd = CALIPER_DT_U8; *item = 1; return true; }
    return false;
}

class SharedBuffer {
public:
    explicit SharedBuffer(size_t nbytes) {
        if (nbytes == 0 || nbytes > (size_t(1) << 34))
            throw std::invalid_argument("SharedBuffer: nbytes must be in [1, 2^34]");
        id<MTLDevice> dev = metal_device();
        if (!dev) throw std::runtime_error("SharedBuffer: no Metal device");
        buf_ = [dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
        if (!buf_) throw std::runtime_error("SharedBuffer: allocation of " +
                                            std::to_string(nbytes) + " bytes failed");
        std::memset([buf_ contents], 0, nbytes);
        nbytes_ = nbytes;
    }

    // A "dltensor" capsule over the buffer for torch.from_dlpack: kDLMetal
    // (data = the MTLBuffer object) for device "mps", kDLCPU (data = -contents)
    // for "cpu". Contiguous `shape` of `dtype` ("float32" | "uint8"). The
    // capsule's tensor holds its own reference to the buffer.
    nb::object dlpack(const std::string& device, const std::string& dtype,
                      const std::vector<int64_t>& shape) const {
        DLDataType dl{};
        CaliperDType cd;
        size_t item = 0;
        if (!dtype_from_name(dtype, &dl, &cd, &item))
            throw std::invalid_argument("SharedBuffer.dlpack: dtype must be 'float32' or 'uint8'");
        if (shape.size() > 8) throw std::invalid_argument("SharedBuffer.dlpack: ndim > 8");
        size_t n = 1;
        for (int64_t s : shape) {
            if (s < 0) throw std::invalid_argument("SharedBuffer.dlpack: negative dim");
            n *= (size_t)s;
        }
        if (n * item > nbytes_)
            throw std::invalid_argument("SharedBuffer.dlpack: shape exceeds the buffer");
        const bool metal = device == "mps" || device == "metal";
        if (!metal && device != "cpu")
            throw std::invalid_argument("SharedBuffer.dlpack: device must be 'mps' or 'cpu'");
        auto* ctx = new BufferCapsuleCtx();
        ctx->retained = (void*)CFBridgingRetain(buf_);
        int64_t st = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            ctx->shape[i] = shape[i];
            ctx->strides[i] = st;
            st *= shape[i];
        }
        DLTensor& t = ctx->managed.dl_tensor;
        t.data = metal ? (void*)(__bridge void*)buf_ : [buf_ contents];
        t.device = DLDevice{metal ? kDLMetal : kDLCPU, 0};
        t.ndim = (int)shape.size();
        t.dtype = dl;
        t.shape = ctx->shape;
        t.strides = ctx->strides;
        t.byte_offset = 0;
        ctx->managed.manager_ctx = ctx;
        ctx->managed.deleter = &buffer_capsule_deleter;
        PyObject* cap = PyCapsule_New(&ctx->managed, "dltensor", &dltensor_capsule_destructor);
        if (!cap) {
            buffer_capsule_deleter(&ctx->managed);
            throw nb::python_error();
        }
        return nb::steal(cap);
    }

    size_t nbytes() const { return nbytes_; }
    uintptr_t address() const { return (uintptr_t)[buf_ contents]; }
    uintptr_t handle() const { return (uintptr_t)(__bridge void*)buf_; }
    id<MTLBuffer> buffer() const { return buf_; }

    // Test helper: copy of the first `count` float32 values (CPU side).
    std::vector<float> read_f32(size_t count) const {
        count = std::min(count, nbytes_ / sizeof(float));
        const float* p = static_cast<const float*>([buf_ contents]);
        return std::vector<float>(p, p + count);
    }

private:
    id<MTLBuffer> buf_ = nil;
    size_t nbytes_ = 0;
};

// ---------------------------------------------------------------------------
// A completed snapshot on its way to the core. Holds strong references to the
// slot buffers until it has been applied (ARC; no GIL needed to release).
// ---------------------------------------------------------------------------
struct LiveSnap {
    std::string name;
    int32_t kind = 0;
    int slot = -1;                       // the producer's ring slot (claims)
    id<MTLBuffer> payload = nil;
    CaliperDType dtype = CALIPER_DT_F32;
    int32_t ndim = 0;
    int64_t shape[4] = {0, 0, 0, 0};
    id<MTLBuffer> aux = nil;             // POINTS labels (N,) f32 or nil
    int64_t aux_n = 0;
    id<MTLBuffer> stats = nil;           // CALIPER_LIVE stats floats, CPU-readable
    int32_t stats_count = 0;
    int32_t colormap = 0;
    float vmin = 0.0f, vmax = 1.0f;
    uint32_t flags = 0;                  // CALIPER_LIVE_APPLY_*
    uint64_t series_total = 0;
    int32_t points_dims = 0;
    int32_t label_count = 0;
    uint64_t seq = 0;
    uint64_t stamp_ns = 0;
};

// Slots of one view the core may still read (caliper/live_views.h claim rules):
//   heatmap/image  cur and prev until read_done (then every applied payload,
//                  the current one included, is free);
//   points         cur while it is current (drawn every frame), prev until
//                  read_done (every superseded payload retired);
//   series/hist    copied at apply: nothing.
struct Claim {
    int taking = -1;   // being applied right now
    int cur = -1;      // the view's current payload
    int prev = -1;     // the payload before it
    int32_t kind = 0;
};

// Bounded sample buffer (telemetry).
struct Samples {
    std::vector<double> v;
    size_t cap = 1 << 16;
    void push(double x) { if (v.size() < cap) v.push_back(x); }
};

constexpr int kMaxStats = 8;

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------
class Core {
public:
    explicit Core(std::optional<std::string> data_dir) {
        const char* e = std::getenv("CALIPER_PY_LOG");
        log_.echo = e && *e && std::strcmp(e, "0") != 0;
        const char* la = std::getenv("CALIPER_PY_LIVE_APPLY");
        heatmap_live_apply_ = la && *la ? std::strcmp(la, "0") != 0 : kHeatmapImageViaLiveApply;
        if (data_dir) data_dir_ = *data_dir;
        CaliperCoreDesc d{};
        d.struct_size = sizeof d;
        d.renderer = CALIPER_RENDERER_DEFAULT;
        d.data_dir = data_dir ? data_dir_.c_str() : nullptr;
        d.applets_dir = nullptr;
        d.log_fn = &log_trampoline;
        d.crash_fn = nullptr;
        d.userdata = &log_;
        @autoreleasepool {
            core_ = caliper_core_create(&d);
        }
        if (!core_)
            throw std::runtime_error(
                "caliper_core_create failed (one live core per process; see log_lines()): " +
                last_log_line());
    }
    ~Core() { close(); }

    Core(const Core&) = delete;
    Core& operator=(const Core&) = delete;

    // --- canvas ------------------------------------------------------------
    // activate=False keeps the script's terminal frontmost (no app activation,
    // no key focus); the window is still shown.
    void open_window(int width, int height, const std::string& title, bool activate) {
        require_main_thread("open_window");
        check_live();
        if (attached_) throw std::runtime_error("a canvas is already attached (one per core)");
        if (width <= 0 || height <= 0) throw std::invalid_argument("window size must be positive");
        @autoreleasepool {
            glfwInitHint(GLFW_COCOA_CHDIR_RESOURCES, GLFW_FALSE);
            if (!glfwInit()) throw std::runtime_error("glfwInit failed");
            glfw_inited_ = true;
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
            glfwWindowHint(GLFW_SCALE_FRAMEBUFFER, GLFW_TRUE);
            glfwWindowHint(GLFW_FOCUSED, activate ? GLFW_TRUE : GLFW_FALSE);
            glfwWindowHint(GLFW_FOCUS_ON_SHOW, activate ? GLFW_TRUE : GLFW_FALSE);
            win_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
            if (!win_) throw std::runtime_error("glfwCreateWindow failed");
            glfwSetWindowUserPointer(win_, this);

            id view = glfwGetCocoaView(win_);
            if (!view) {
                NSWindow* nsw = (NSWindow*)glfwGetCocoaWindow(win_);
                view = nsw.contentView;
            }
            int fbw = 0, fbh = 0;
            glfwGetFramebufferSize(win_, &fbw, &fbh);
            float sx = 1.f, sy = 1.f;
            glfwGetWindowContentScale(win_, &sx, &sy);
            scale_ = sx > 0 ? sx : 1.f;

            CaliperCanvasDesc c{};
            c.struct_size = sizeof c;
            c.mode = CALIPER_CANVAS_WINDOW;
            c.width = fbw;
            c.height = fbh;
            c.content_scale = scale_;
            if (!caliper_core_attach_canvas(core_, (__bridge void*)view, &c)) {
                std::string err = caliper_core_last_error(core_);
                glfwDestroyWindow(win_);
                win_ = nullptr;
                throw std::runtime_error("attach_canvas(window) failed: " + err);
            }
            width_ = fbw;
            height_ = fbh;
            attached_ = true;
            offscreen_ = false;
            install_callbacks();
            watch_occlusion();
            // Opening a window makes this process an AppKit app, and macOS
            // naps an app whose windows are all occluded while its main thread
            // idles in the event loop (caliper.run): timers coalesce and EVERY
            // thread is throttled, the training thread included (measured:
            // render 9 fps, training -40 %). A training script is user work.
            if (!no_app_nap_activity_)
                no_app_nap_activity_ = [[NSProcessInfo processInfo]
                    beginActivityWithOptions:(NSActivityUserInitiatedAllowingIdleSystemSleep |
                                              NSActivityLatencyCritical)
                                      reason:@"caliper: live view of a training run"];
            if (activate) {
                NSApplication* app = [NSApplication sharedApplication];
                [app activateIgnoringOtherApps:YES];
                glfwFocusWindow(win_);
            }
        }
    }

    void open_offscreen(int width, int height) {
        check_live();
        if (attached_) throw std::runtime_error("a canvas is already attached (one per core)");
        CaliperCanvasDesc c{};
        c.struct_size = sizeof c;
        c.mode = CALIPER_CANVAS_OFFSCREEN;
        c.width = width;
        c.height = height;
        c.content_scale = 1.0f;
        int ok;
        @autoreleasepool {
            ok = caliper_core_attach_canvas(core_, nullptr, &c);
        }
        if (!ok)
            throw std::runtime_error(std::string("attach_canvas(offscreen) failed: ") +
                                     caliper_core_last_error(core_));
        width_ = width;
        height_ = height;
        scale_ = 1.f;
        attached_ = true;
        offscreen_ = true;
    }

    // --- main-thread event pumping ------------------------------------------
    void poll() {
        if (!win_) return;
        require_main_thread("poll");
        nb::gil_scoped_release nogil;
        glfwPollEvents();
    }

    // Block in the AppKit event loop up to `timeout` seconds (returns early on
    // an event). The GIL is released while waiting.
    void wait_events(double timeout) {
        if (!win_) return;
        require_main_thread("wait_events");
        nb::gil_scoped_release nogil;
        glfwWaitEventsTimeout(timeout > 0 ? timeout : 0.0);
    }

    // Run the AppKit event loop on the main thread until stop_event_loop() is
    // called (from any thread), the window is closed, `timeout` seconds passed
    // (< 0: no limit) or a Python signal handler raises (KeyboardInterrupt).
    // The GIL is held only for a signal check every `signal_check` seconds, so
    // a training loop on another Python thread runs at full speed. Returns True
    // if the window was closed.
    bool run_event_loop(double signal_check, double timeout) {
        if (!win_) throw std::runtime_error("run_event_loop needs a window");
        require_main_thread("run_event_loop");
        loop_stop_.store(false);
        const double dt = signal_check > 0 ? signal_check : 0.1;
        const double deadline = timeout >= 0 ? now_ms() + timeout * 1e3 : -1.0;
        for (;;) {
            {
                nb::gil_scoped_release nogil;
                double until = now_ms() + dt * 1e3;
                if (deadline >= 0) until = std::min(until, deadline);
                while (!loop_stop_.load() && !glfwWindowShouldClose(win_)) {
                    const double left = (until - now_ms()) * 1e-3;
                    if (left <= 0) break;
                    glfwWaitEventsTimeout(left);
                }
            }
            if (PyErr_CheckSignals() != 0) throw nb::python_error();
            if (glfwWindowShouldClose(win_)) return true;
            if (loop_stop_.load()) return false;
            if (deadline >= 0 && now_ms() >= deadline) return false;
        }
    }

    // Any thread: make run_event_loop() return.
    void stop_event_loop() {
        loop_stop_.store(true);
        if (win_) glfwPostEmptyEvent();
    }

    double frame() {
        check_live();
        if (!offscreen_ && !rendering()) require_main_thread("frame");
        const double t0 = now_ms();
        on_render_thread([&] {
            @autoreleasepool {
                caliper_core_frame(core_);
            }
            if (!rendering_.load()) settle_prev_claims();
            return 0;
        });
        return now_ms() - t0;
    }

    bool should_close() const { return win_ ? glfwWindowShouldClose(win_) != 0 : false; }

    nb::ndarray<nb::numpy, uint8_t> read_pixels() {
        check_live();
        if (!offscreen_) throw std::runtime_error("read_pixels needs an offscreen canvas");
        auto arr = owned_array<uint8_t>({(size_t)height_, (size_t)width_, 4});
        uint8_t* dst = arr.data();
        const int stride = width_ * 4;
        std::string err;
        const int ok = on_render_thread([&] {
            int r;
            @autoreleasepool {
                r = caliper_core_read_pixels(core_, dst, stride);
            }
            if (!r) err = caliper_core_last_error(core_);
            return r;
        });
        if (!ok) throw std::runtime_error("read_pixels failed: " + err);
        return arr;
    }

    std::string last_error() {
        if (!core_) return "";
        return on_render_thread([&] { return std::string(caliper_core_last_error(core_)); });
    }

    std::vector<std::string> log_lines() {
        std::lock_guard<std::mutex> lk(log_.mu);
        return std::vector<std::string>(log_.lines.begin(), log_.lines.end());
    }

    // --- views (embed v1.2), synchronous: a DLPack capsule straight in -------
    void view_tensor(const std::string& name, nb::handle capsule, uint64_t byte_offset_extra,
                     int colormap, float vmin, float vmax, uint32_t flags) {
        check_live();
        if (!offscreen_ && !rendering()) require_main_thread("view_tensor");
        ParsedCapsule p = open_capsule(capsule, "view_tensor");
        int ok = 0;
        std::string err;
        try {
            uint64_t boff = 0;
            CaliperTensor t = to_caliper_tensor(*p.dl, byte_offset_extra, &boff);
            CaliperViewDesc d{};
            d.struct_size = sizeof d;
            d.name = name.c_str();
            d.tensor = &t;
            d.byte_offset = boff;
            d.colormap = colormap;
            d.vmin = vmin;
            d.vmax = vmax;
            d.flags = flags;
            ok = on_render_thread([&] {
                int r;
                @autoreleasepool {
                    r = caliper_core_view_tensor(core_, &d);
                }
                if (!r) err = core_error("view_tensor", name);
                return r;
            });
        } catch (...) {
            consume_capsule(capsule, p);
            throw;
        }
        consume_capsule(capsule, p);
        if (!ok) throw std::runtime_error(err);
    }

    nb::dict view_info(const std::string& name) {
        check_live();
        CaliperViewInfo info{};
        info.struct_size = sizeof info;
        std::string path, err;
        const int ok = on_render_thread([&] {
            const int r = caliper_core_view_info(core_, name.c_str(), &info);
            if (r) path = info.path ? info.path : "";   // valid until the next core call
            else err = core_error("view_info", name);
            return r;
        });
        if (!ok) throw std::runtime_error(err);
        nb::dict d;
        d["path"] = path;
        d["width"] = info.width;
        d["height"] = info.height;
        d["device"] = info.device == CALIPER_DEV_METAL ? "metal"
                      : info.device == CALIPER_DEV_CPU ? "cpu"
                                                       : "other";
        d["updates"] = info.updates;
        d["last_update_ms"] = info.last_update_ms;
        d["imported"] = info.imported != 0;
        d["creates"] = info.creates;
        return d;
    }

    void view_remove(const std::string& name) {
        check_live();
        std::string err;
        const int ok = on_render_thread([&] {
            const int r = caliper_core_view_remove(core_, name.c_str());
            if (!r) err = core_error("view_remove", name);
            return r;
        });
        if (!ok) throw std::runtime_error(err);
        forget(name);
    }

    int view_count() {
        check_live();
        return on_render_thread([&] { return caliper_core_view_count(core_); });
    }

    nb::ndarray<nb::numpy, uint8_t> view_readback(const std::string& name) {
        check_live();
        CaliperViewInfo info{};
        info.struct_size = sizeof info;
        std::string err;
        if (!on_render_thread([&] {
                const int r = caliper_core_view_info(core_, name.c_str(), &info);
                if (!r) err = core_error("view_readback", name);
                return r;
            }))
            throw std::runtime_error(err);
        auto arr = owned_array<uint8_t>({(size_t)info.height, (size_t)info.width, 4});
        uint8_t* dst = arr.data();
        const size_t bytes = (size_t)info.width * info.height * 4;
        const int ok = on_render_thread([&] {
            int r;
            @autoreleasepool {
                r = caliper_core_view_readback(core_, name.c_str(), dst, bytes);
            }
            if (!r) err = core_error("view_readback", name);
            return r;
        });
        if (!ok) throw std::runtime_error(err);
        return arr;
    }

    // --- typed live views (caliper/live_views.h) -------------------------------
    // Hand a COMPLETED snapshot to the core. Render thread running: queue it
    // (newest per view wins; one replaced before it was taken counts as
    // superseded) and return at once. Otherwise apply it now on this thread and
    // raise the core's error on refusal.
    void publish_live(const std::string& name, int kind, int slot, const SharedBuffer& payload,
                      const std::vector<int64_t>& shape, const std::string& dtype,
                      const SharedBuffer* aux, int64_t aux_n, const SharedBuffer* stats,
                      int stats_count, int colormap, float vmin, float vmax, uint32_t flags,
                      uint64_t series_total, int points_dims, int label_count, uint64_t seq,
                      uint64_t stamp_ns) {
        check_live();
        if (name.empty()) throw std::invalid_argument("publish_live: empty view name");
        if (shape.empty() || shape.size() > 4)
            throw std::invalid_argument("publish_live: payload ndim must be 1..4");
        DLDataType dl{};
        size_t item = 0;
        LiveSnap s;
        if (!dtype_from_name(dtype, &dl, &s.dtype, &item))
            throw std::invalid_argument("publish_live: dtype must be 'float32' or 'uint8'");
        size_t n = 1;
        for (size_t i = 0; i < shape.size(); ++i) { s.shape[i] = shape[i]; n *= (size_t)shape[i]; }
        if (n * item > payload.nbytes())
            throw std::invalid_argument("publish_live: shape exceeds the payload buffer");
        s.name = name;
        s.kind = kind;
        s.slot = slot;
        s.payload = payload.buffer();
        s.ndim = (int32_t)shape.size();
        if (aux) {
            if ((size_t)aux_n * sizeof(float) > aux->nbytes())
                throw std::invalid_argument("publish_live: aux_n exceeds the aux buffer");
            s.aux = aux->buffer();
            s.aux_n = aux_n;
        }
        if (stats) {
            s.stats = stats->buffer();
            s.stats_count = std::max(0, std::min<int>(stats_count, (int)(stats->nbytes() / sizeof(float))));
        }
        s.colormap = colormap;
        s.vmin = vmin;
        s.vmax = vmax;
        s.flags = flags;
        s.series_total = series_total;
        s.points_dims = points_dims;
        s.label_count = label_count;
        s.seq = seq;
        s.stamp_ns = stamp_ns;
        if (rendering_.load()) {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = queued_.find(name);
            if (it != queued_.end()) {
                it->second = std::move(s);
                ++superseded_;
            } else {
                queued_.emplace(name, std::move(s));
            }
            ++published_;
            return;
        }
        // Synchronous mode: apply here.
        if (!offscreen_) require_main_thread("publish_live (window without a render thread)");
        std::string err;
        bool ok;
        {
            nb::gil_scoped_release nogil;
            {
                std::lock_guard<std::mutex> lk(mu_);
                ++published_;
                claims_[name].taking = slot;
            }
            @autoreleasepool {
                settle_sync(name);
                s.flags |= CALIPER_LIVE_APPLY_WAIT_GPU;   // the test mode: reads retired on return
                ok = apply_live(s, err);
                after_apply(s, ok);
                settle_prev_claims();
            }
            if (ok) {
                std::lock_guard<std::mutex> lk(mu_);
                ++drawn_;
            }
        }
        if (!ok) throw std::runtime_error(err);
    }

    // The slots of `name` the core may still read (claimed), oldest last.
    std::vector<int> reading_slots(const std::string& name) {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<int> r;
        auto it = claims_.find(name);
        if (it == claims_.end()) return r;
        for (int s : {it->second.taking, it->second.cur, it->second.prev})
            if (s >= 0 && std::find(r.begin(), r.end(), s) == r.end()) r.push_back(s);
        return r;
    }

    int reading_slot(const std::string& name) {
        auto r = reading_slots(name);
        return r.empty() ? -1 : r.front();
    }

    // Drop the claims and any queued snapshot of `name` (the producer retired
    // the view's slots).
    void forget(const std::string& name) {
        std::lock_guard<std::mutex> lk(mu_);
        claims_.erase(name);
        queued_.erase(name);
        last_range_.erase(name);
    }

    nb::dict live_view_info(const std::string& name) {
        check_live();
        CaliperLiveViewInfo info{};
        info.struct_size = sizeof info;
        std::string path, err;
        const int ok = on_render_thread([&] {
            const int r = caliper_core_live_view_info(core_, name.c_str(), &info);
            if (r) path = info.path ? info.path : "";
            else err = core_error("live_view_info", name);
            return r;
        });
        if (!ok) throw std::runtime_error(err);
        nb::dict d;
        d["kind"] = info.kind;
        d["path"] = path;
        d["width"] = info.width;
        d["height"] = info.height;
        d["updates"] = info.updates;
        d["creates"] = info.creates;
        d["dropped"] = info.dropped;
        d["seq"] = info.seq;
        d["stamp_ns"] = info.stamp_ns;
        d["last_apply_ms"] = info.last_apply_ms;
        d["latency_ms"] = info.latency_ms;
        d["imported"] = info.imported != 0;
        d["read_done"] = info.read_done != 0;
        d["count"] = info.count;
        return d;
    }

    nb::ndarray<nb::numpy, float> live_view_values(const std::string& name) {
        check_live();
        size_t count = 0;
        std::string err;
        if (!on_render_thread([&] {
                const int r = caliper_core_live_view_values(core_, name.c_str(), nullptr, 0, &count);
                if (!r) err = core_error("live_view_values", name);
                return r;
            }))
            throw std::runtime_error(err);
        auto arr = owned_array<float>({count});
        float* dst = arr.data();
        size_t got = 0;
        if (!on_render_thread([&] {
                const int r = caliper_core_live_view_values(core_, name.c_str(), dst, count, &got);
                if (!r) err = core_error("live_view_values", name);
                return r;
            }))
            throw std::runtime_error(err);
        if (got < count) {
            auto shorter = owned_array<float>({got});
            std::memcpy(shorter.data(), dst, got * sizeof(float));
            return shorter;
        }
        return arr;
    }

    nb::ndarray<nb::numpy, uint8_t> live_view_readback(const std::string& name) {
        check_live();
        CaliperLiveViewInfo info{};
        info.struct_size = sizeof info;
        std::string err;
        if (!on_render_thread([&] {
                const int r = caliper_core_live_view_info(core_, name.c_str(), &info);
                if (!r) err = core_error("live_view_readback", name);
                return r;
            }))
            throw std::runtime_error(err);
        if (info.width <= 0 || info.height <= 0)
            throw std::runtime_error("live_view_readback('" + name + "'): view has no texture");
        auto arr = owned_array<uint8_t>({(size_t)info.height, (size_t)info.width, 4});
        uint8_t* dst = arr.data();
        const size_t bytes = (size_t)info.width * info.height * 4;
        const int ok = on_render_thread([&] {
            int r;
            @autoreleasepool {
                r = caliper_core_live_view_readback(core_, name.c_str(), dst, bytes);
            }
            if (!r) err = core_error("live_view_readback", name);
            return r;
        });
        if (!ok) throw std::runtime_error(err);
        return arr;
    }

    bool live_view_remove(const std::string& name) {
        check_live();
        const int r = on_render_thread([&] { return caliper_core_live_view_remove(core_, name.c_str()); });
        forget(name);
        return r != 0;
    }

    int live_view_count() {
        check_live();
        return on_render_thread([&] { return caliper_core_live_view_count(core_); });
    }

    bool live_view_read_done(const std::string& name) {
        check_live();
        return on_render_thread([&] { return caliper_core_live_view_read_done(core_, name.c_str()); }) != 0;
    }

    void live_set_paused(bool paused) {
        check_live();
        on_render_thread([&] { caliper_core_live_set_paused(core_, paused ? 1 : 0); return 0; });
    }

    bool live_paused() {
        check_live();
        return on_render_thread([&] { return caliper_core_live_paused(core_); }) != 0;
    }

    bool heatmap_live_apply() const { return heatmap_live_apply_; }

    // --- render thread ----------------------------------------------------------
    void start_render_thread(double fps, bool heartbeat) {
        check_live();
        if (!attached_) throw std::runtime_error("start_render_thread: attach a canvas first");
        if (render_.joinable()) throw std::runtime_error("the render thread is already running");
        if (!(fps > 0 && fps <= 1000)) throw std::invalid_argument("fps must be in (0, 1000]");
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = false;
            fps_ = fps;
            heartbeat_ = heartbeat;
            render_tid_ready_ = false;
        }
        render_ = std::thread([this] { render_main(); });
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return render_tid_ready_; });
    }

    void stop_render_thread() {
        if (!render_.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        {
            nb::gil_scoped_release nogil;
            render_.join();
        }
        std::lock_guard<std::mutex> lk(mu_);
        queued_.clear();
        claims_.clear();
        input_q_.clear();
        rendering_.store(false);
    }

    bool rendering() const { return rendering_.load(); }

    nb::dict render_stats(bool reset) {
        nb::dict d;
        std::vector<double> lat, tick, gap;
        {
            std::lock_guard<std::mutex> lk(mu_);
            d["running"] = render_.joinable();
            d["fps"] = fps_;
            d["frames"] = frames_;
            d["published"] = published_;
            d["superseded"] = superseded_;
            d["drawn"] = drawn_;
            d["deferred_takes"] = deferred_takes_;
            d["errors"] = errors_;
            d["stats_unreadable"] = stats_unreadable_;
            d["nonfinite_range"] = nonfinite_range_;
            d["queued"] = (uint64_t)queued_.size();
            d["last_error"] = last_render_error_;
            d["max_frame_gap_ms"] = max_gap_ms_;
            d["events_applied"] = events_applied_;
            d["heatmap_live_apply"] = heatmap_live_apply_;
            lat = latency_ms_.v;
            tick = tick_ms_.v;
            gap = gap_ms_.v;
            if (reset) {
                latency_ms_.v.clear(); tick_ms_.v.clear(); gap_ms_.v.clear();
                max_gap_ms_ = 0; frames_ = published_ = superseded_ = drawn_ = 0;
                errors_ = stats_unreadable_ = nonfinite_range_ = events_applied_ = 0;
                deferred_takes_ = 0;
            }
        }
        d["latency_ms"] = lat;      // snapshot taken -> first frame drawn with it
        d["tick_ms"] = tick;        // render-thread CPU time per tick
        d["frame_gap_ms"] = gap;    // time between consecutive rendered frames
        return d;
    }

    // --- main-thread responsiveness probes (measurement surface) -------------
    // Each probe is posted from the calling (any) thread and stamped when the
    // MAIN thread services it, which happens only while it runs the AppKit
    // event loop (poll / wait_events / run_event_loop):
    //   kind 0: a block on the main dispatch queue (the run loop is serviced);
    //   kind 1: a synthetic mouse-move NSEvent at the window centre, stamped in
    //           GLFW's cursor callback (the input path ImGui hover rides on);
    //   kind 2: -[NSWindow performClose:] on the main queue (the close button's
    //           effect: GLFW's should-close flag).
    // probe_results() returns [(kind, posted_ns, handled_ns)] for handled probes.
    void post_probe(int kind) {
        if (!win_) throw std::runtime_error("post_probe needs a window");
        const uint64_t t0 = mono_ns();
        NSWindow* w = (NSWindow*)glfwGetCocoaWindow(win_);
        Core* self = this;
        if (kind == 0) {
            dispatch_async(dispatch_get_main_queue(), ^{ self->probe_done(0, t0); });
        } else if (kind == 1) {
            {
                std::lock_guard<std::mutex> lk(probe_mu_);
                probe_mouse_pending_.push_back(t0);
            }
            const NSRect r = w.contentView.bounds;
            NSEvent* ev = [NSEvent mouseEventWithType:NSEventTypeMouseMoved
                                             location:NSMakePoint(NSMidX(r), NSMidY(r))
                                        modifierFlags:0
                                            timestamp:[[NSProcessInfo processInfo] systemUptime]
                                         windowNumber:w.windowNumber
                                              context:nil
                                          eventNumber:0
                                           clickCount:0
                                             pressure:0];
            [NSApp postEvent:ev atStart:NO];   // documented as callable from subthreads
        } else if (kind == 2) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [w performClose:nil];
                self->probe_done(2, t0);
            });
        } else {
            throw std::invalid_argument("post_probe: kind must be 0, 1 or 2");
        }
    }

    std::vector<std::tuple<int, uint64_t, uint64_t>> probe_results(bool reset) {
        std::lock_guard<std::mutex> lk(probe_mu_);
        auto r = probes_;
        if (reset) probes_.clear();
        return r;
    }

    // Occlusion transitions of the window, [(mono_ns, visible)], the first entry
    // being the state at open. macOS throttles drawables of an occluded window
    // (the render thread's frame then waits in -nextDrawable), so this tells a
    // throttled window from a slow renderer. Updated only while the main thread
    // runs the event loop (AppKit delivers the notification there).
    std::vector<std::tuple<uint64_t, bool>> occlusion_log() {
        std::lock_guard<std::mutex> lk(probe_mu_);
        return occlusion_;
    }

    int64_t window_number() const {
        if (!win_) return -1;
        NSWindow* w = (NSWindow*)glfwGetCocoaWindow(win_);
        return (int64_t)w.windowNumber;
    }

    void set_window_pos(int x, int y) {
        if (!win_) return;
        require_main_thread("set_window_pos");
        glfwSetWindowPos(win_, x, y);
    }

    // --- teardown ----------------------------------------------------------
    void close() {
        stop_render_thread();
        if (core_) {
            @autoreleasepool {
                caliper_core_shutdown(core_);
            }
            core_ = nullptr;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            queued_.clear();
            claims_.clear();
            last_range_.clear();
        }
        if (occlusion_observer_) {
            [[NSNotificationCenter defaultCenter] removeObserver:occlusion_observer_];
            occlusion_observer_ = nil;
        }
        if (no_app_nap_activity_) {
            [[NSProcessInfo processInfo] endActivity:no_app_nap_activity_];
            no_app_nap_activity_ = nil;
        }
        if (win_) {
            glfwDestroyWindow(win_);
            win_ = nullptr;
        }
        if (glfw_inited_) {
            // Pump once so AppKit actually removes the window from screen.
            glfwPollEvents();
            glfwTerminate();
            glfw_inited_ = false;
        }
        attached_ = false;
    }

    bool closed() const { return core_ == nullptr; }
    bool is_offscreen() const { return offscreen_; }
    bool has_window() const { return win_ != nullptr; }
    int width() const { return width_; }
    int height() const { return height_; }
    float content_scale() const { return scale_; }

private:
    void check_live() const {
        if (!core_) throw std::runtime_error("caliper core is closed");
    }

    // The core's last_error verbatim (it prefixes "<fn>('<name>'): "); a
    // generic message only when the core left it empty.
    std::string core_error(const char* fn, const std::string& name) const {
        std::string e = caliper_core_last_error(core_);
        return e.empty() ? std::string(fn) + "('" + name + "') failed" : e;
    }

    std::string last_log_line() {
        std::lock_guard<std::mutex> lk(log_.mu);
        return log_.lines.empty() ? std::string() : log_.lines.back();
    }

    // Run f() where core calls belong: inline when no render thread runs (or on
    // it), else on the render thread, waiting with the GIL released. f must not
    // touch Python objects.
    template <class F>
    auto on_render_thread(F&& f) -> decltype(f()) {
        using R = decltype(f());
        std::packaged_task<R()> task(std::forward<F>(f));
        auto fut = task.get_future();
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (!rendering_.load() || std::this_thread::get_id() == render_id_) {
                lk.unlock();
                task();
                return fut.get();
            }
            tasks_.emplace_back([&task] { task(); });
        }
        cv_.notify_all();
        {
            nb::gil_scoped_release nogil;
            fut.wait();
        }
        return fut.get();
    }

    void probe_done(int kind, uint64_t posted) {
        std::lock_guard<std::mutex> lk(probe_mu_);
        probes_.emplace_back(kind, posted, mono_ns());
    }

    // Input from GLFW callbacks (main thread): straight into the core in the
    // synchronous mode, queued for the render thread otherwise.
    void send(CaliperInputEvent ev) {
        if (!core_) return;
        ev.struct_size = sizeof ev;
        if (rendering_.load()) {
            std::lock_guard<std::mutex> lk(mu_);
            input_q_.push_back(ev);
            return;
        }
        caliper_core_event(core_, &ev);
    }

    // --- applying a snapshot (frame thread) ---------------------------------------
    bool via_live_apply(int32_t kind) const {
        return heatmap_live_apply_ ||
               (kind != CALIPER_LIVE_KIND_HEATMAP && kind != CALIPER_LIVE_KIND_IMAGE);
    }

    // Resolve the AUTO ends from the producer's stats (CPU side of the stats
    // buffer): a non-finite range (NaN/inf in the tensor) keeps the view's last
    // finite range, a constant one is widened. Writes the floats the core sees.
    void resolve_stats(const LiveSnap& s, float* st, int* count, float* lo, float* hi) {
        *count = 0;
        *lo = s.vmin;
        *hi = s.vmax;
        if (s.stats && s.stats_count > 0) {
            const float* p = static_cast<const float*>([s.stats contents]);
            if (p) {
                *count = std::min(s.stats_count, kMaxStats);
                std::memcpy(st, p, sizeof(float) * (size_t)*count);
            }
        }
        const uint32_t autos = s.flags & (CALIPER_LIVE_APPLY_AUTO_VMIN | CALIPER_LIVE_APPLY_AUTO_VMAX);
        if (!autos) return;
        if (*count < 2) {
            std::lock_guard<std::mutex> lk(mu_);
            ++stats_unreadable_;
        } else {
            if (s.flags & CALIPER_LIVE_APPLY_AUTO_VMIN) *lo = st[0];
            if (s.flags & CALIPER_LIVE_APPLY_AUTO_VMAX) *hi = st[1];
        }
        std::lock_guard<std::mutex> lk(mu_);
        if (!std::isfinite(*lo) || !std::isfinite(*hi) || *count < 2) {
            auto it = last_range_.find(s.name);
            if (it != last_range_.end()) { *lo = it->second.first; *hi = it->second.second; }
            else {
                *lo = std::isfinite(s.vmin) ? s.vmin : 0.0f;
                *hi = std::isfinite(s.vmax) ? s.vmax : 1.0f;
            }
            ++nonfinite_range_;
        }
        if (*lo == *hi) { *lo -= 0.5f; *hi += 0.5f; }
        if (*lo > *hi) std::swap(*lo, *hi);
        last_range_[s.name] = {*lo, *hi};
        if (*count < 2) *count = 2;
        st[0] = *lo;
        st[1] = *hi;
    }

    bool apply_live(const LiveSnap& s, std::string& err) {
        float st[kMaxStats] = {0};
        int count = 0;
        float lo = s.vmin, hi = s.vmax;
        resolve_stats(s, st, &count, &lo, &hi);

        CaliperTensor t{};
        t.struct_size = sizeof t;
        t.data = (__bridge void*)s.payload;
        t.dtype = s.dtype;
        t.ndim = s.ndim;
        int64_t stride = 1;
        for (int i = s.ndim - 1; i >= 0; --i) {
            t.shape[i] = s.shape[i];
            t.strides[i] = stride;
            stride *= s.shape[i];
        }
        t.device = CALIPER_DEV_METAL;
        t.device_index = 0;
        t.stream = nullptr;

        if (!via_live_apply(s.kind)) {
            // embed v1.2 route; WAIT_GPU: the read has retired when this returns.
            // (CALIPER_LIVE_APPLY_WAIT_GPU is not a v1.2 flag: mapped below.)
            CaliperViewDesc d{};
            d.struct_size = sizeof d;
            d.name = s.name.c_str();
            d.tensor = &t;
            d.byte_offset = 0;
            d.colormap = s.colormap;
            d.vmin = lo;
            d.vmax = hi;
            d.flags = (s.flags & CALIPER_LIVE_APPLY_RAW ? CALIPER_VIEW_FLAG_RAW : 0u) |
                      CALIPER_VIEW_FLAG_WAIT_GPU;
            if (!caliper_core_view_tensor(core_, &d)) {
                err = core_error("view_tensor", s.name);
                return false;
            }
            return true;
        }

        CaliperTensor a{};
        if (s.aux) {
            a.struct_size = sizeof a;
            a.data = (__bridge void*)s.aux;
            a.dtype = CALIPER_DT_F32;
            a.ndim = 1;
            a.shape[0] = s.aux_n;
            a.strides[0] = 1;
            a.device = CALIPER_DEV_METAL;
        }
        CaliperLiveSnapshot snap{};
        snap.struct_size = sizeof snap;
        snap.name = s.name.c_str();
        snap.kind = s.kind;
        snap.data = &t;
        snap.data_offset = 0;
        snap.aux = s.aux ? &a : nullptr;
        snap.aux_offset = 0;
        snap.stats = count > 0 ? st : nullptr;
        snap.stats_count = count;
        snap.colormap = s.colormap;
        snap.vmin = lo;
        snap.vmax = hi;
        snap.flags = s.flags & (CALIPER_LIVE_APPLY_RAW | CALIPER_LIVE_APPLY_AUTO_VMIN |
                                CALIPER_LIVE_APPLY_AUTO_VMAX | CALIPER_LIVE_APPLY_WAIT_GPU);
        snap.series_total = s.series_total;
        snap.points_dims = s.points_dims;
        snap.label_count = s.label_count;
        snap.seq = s.seq;
        snap.stamp_ns = s.stamp_ns;
        if (!caliper_core_live_apply(core_, &snap)) {
            err = core_error("live_apply", s.name);
            return false;
        }
        return true;
    }

    // Claims after an apply attempt (frame thread).
    void after_apply(const LiveSnap& s, bool ok) {
        std::lock_guard<std::mutex> lk(mu_);
        Claim& c = claims_[s.name];
        c.taking = -1;
        if (!ok) return;    // nothing read; the view keeps its previous payload
        const bool copies = s.kind == CALIPER_LIVE_KIND_SERIES || s.kind == CALIPER_LIVE_KIND_HISTOGRAM;
        if (!via_live_apply(s.kind) || copies) {
            c.cur = -1;
            c.prev = -1;
            c.kind = s.kind;
            return;
        }
        if (c.cur >= 0 && c.cur != s.slot) c.prev = c.cur;
        c.cur = s.slot;
        c.kind = s.kind;
    }

    // Release claims whose GPU reads retired (frame thread; never waits):
    // read_done == 1 frees every applied heatmap/image payload, and every
    // superseded points payload (the current one stays claimed).
    void settle_prev_claims() {
        std::vector<std::string> names;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto& kv : claims_) {
                const Claim& c = kv.second;
                const bool texture = c.kind == CALIPER_LIVE_KIND_HEATMAP || c.kind == CALIPER_LIVE_KIND_IMAGE;
                if (c.prev >= 0 || (texture && c.cur >= 0)) names.push_back(kv.first);
            }
        }
        for (auto& n : names) {
            if (caliper_core_live_view_read_done(core_, n.c_str())) {
                std::lock_guard<std::mutex> lk(mu_);
                auto it = claims_.find(n);
                if (it == claims_.end()) continue;
                Claim& c = it->second;
                c.prev = -1;
                if (c.kind != CALIPER_LIVE_KIND_POINTS) c.cur = -1;
            }
        }
    }

    // Synchronous mode, before applying a new snapshot of `name`: the previous
    // claim must clear first (a points payload is read until the next one's
    // first frame retired), so render frames until it does (bounded, 2 s).
    void settle_sync(const std::string& name) {
        for (int i = 0; i < 2000; ++i) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                auto it = claims_.find(name);
                if (it == claims_.end() || it->second.prev < 0) return;
            }
            if (caliper_core_live_view_read_done(core_, name.c_str())) {
                std::lock_guard<std::mutex> lk(mu_);
                claims_[name].prev = -1;
                return;
            }
            caliper_core_frame(core_);
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        log_.push(2, ("publish_live('" + name + "'): previous payload read did not retire in 2 s").c_str());
        std::lock_guard<std::mutex> lk(mu_);
        claims_[name].prev = -1;
    }

    void record_error(const std::string& e) {
        std::lock_guard<std::mutex> lk(mu_);
        ++errors_;
        if (e != last_render_error_) log_.push(2, e.c_str());
        last_render_error_ = e;
    }

    void heartbeat_update() {
        constexpr int W = 1024, H = 8;   // wide enough for the info line (updates = frames)
        if (hb_.empty()) hb_.resize((size_t)W * H);
        const uint64_t f = frames_seen_ * 4;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                hb_[(size_t)y * W + x] = (float)((x + f) % W) / (float)(W - 1);
        CaliperTensor t{};
        t.struct_size = sizeof t;
        t.dtype = CALIPER_DT_F32;
        t.device = CALIPER_DEV_CPU;
        t.ndim = 2;
        t.shape[0] = H; t.shape[1] = W;
        t.strides[0] = W; t.strides[1] = 1;
        t.data = hb_.data();
        CaliperViewDesc d{};
        d.struct_size = sizeof d;
        d.name = kHeartbeatView;
        d.tensor = &t;
        d.colormap = 0;
        d.vmin = 0.0f;
        d.vmax = 1.0f;
        if (!caliper_core_view_tensor(core_, &d)) record_error(caliper_core_last_error(core_));
    }

    void render_main() {
        pthread_setname_np("caliper.render");
        // Leave process-directed signals (SIGINT, SIGALRM, ...) to the main
        // thread, where Python runs its handlers.
        sigset_t all;
        sigfillset(&all);
        pthread_sigmask(SIG_BLOCK, &all, nullptr);
        {
            std::lock_guard<std::mutex> lk(mu_);
            render_id_ = std::this_thread::get_id();
            rendering_.store(true);
            render_tid_ready_ = true;
        }
        cv_.notify_all();

        using clk = std::chrono::steady_clock;
        // Fixed ticks at `fps`. Drawing a snapshot as soon as it is published
        // (frames above the display rate) was measured to RAISE latency: the
        // frame then blocks in -nextDrawable at the display's vsync after the
        // snapshot was already taken.
        const auto period = std::chrono::nanoseconds((int64_t)(1e9 / fps_));
        auto next = clk::now();
        uint64_t last_frame_ns = 0;
        std::deque<std::function<void()>> tasks;
        std::vector<LiveSnap> draw;
        std::vector<CaliperInputEvent> events;

        for (;;) {
            bool tick = false, stopping = false;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait_until(lk, next, [&] { return stop_ || !tasks_.empty(); });
                tasks.swap(tasks_);
                stopping = stop_;
                tick = !stopping && clk::now() >= next;
            }
            for (auto& t : tasks) t();
            tasks.clear();
            if (stopping) break;
            if (!tick) continue;

            const uint64_t t_tick = mono_ns();
            next += period;
            if (next < clk::now()) next = clk::now() + period;   // no burst after a stall
            @autoreleasepool {
                settle_prev_claims();
                {
                    // Take the newest snapshot per view whose previous payload's
                    // read retired; from here until its read is done the
                    // producer must not rewrite that slot.
                    std::lock_guard<std::mutex> lk(mu_);
                    for (auto it = queued_.begin(); it != queued_.end();) {
                        Claim& c = claims_[it->first];
                        if (c.prev >= 0) {
                            ++deferred_takes_;
                            ++it;
                            continue;
                        }
                        c.taking = it->second.slot;
                        draw.push_back(std::move(it->second));
                        it = queued_.erase(it);
                    }
                    events.swap(input_q_);
                }
                if (!events.empty()) {
                    // Resize touches the CAMetalLayer: explicit transaction off
                    // the main thread.
                    [CATransaction begin];
                    for (auto& ev : events) caliper_core_event(core_, &ev);
                    [CATransaction commit];
                }
                size_t ok_count = 0;
                std::vector<uint64_t> stamps;
                for (auto& s : draw) {
                    std::string err;
                    const bool ok = apply_live(s, err);
                    after_apply(s, ok);
                    if (ok) {
                        ++ok_count;
                        stamps.push_back(s.stamp_ns);
                    } else {
                        record_error(err);
                    }
                }
                draw.clear();   // releases the buffer references (ARC)
                if (heartbeat_) heartbeat_update();
                caliper_core_frame(core_);
                const uint64_t t_done = mono_ns();
                ++frames_seen_;
                std::lock_guard<std::mutex> lk(mu_);
                ++frames_;
                drawn_ += ok_count;
                events_applied_ += events.size();
                tick_ms_.push((t_done - t_tick) * 1e-6);
                if (last_frame_ns) {
                    const double gap = (t_done - last_frame_ns) * 1e-6;
                    gap_ms_.push(gap);
                    max_gap_ms_ = std::max(max_gap_ms_, gap);
                }
                for (uint64_t st : stamps)
                    if (st && t_done > st) latency_ms_.push((t_done - st) * 1e-6);
                last_frame_ns = t_done;
            }
            events.clear();
        }
        // Exit: run tasks posted while stopping, then hand core calls back to
        // the calling threads (rendering_ = false under the same lock).
        for (;;) {
            std::deque<std::function<void()>> rest;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (tasks_.empty()) {
                    rendering_.store(false);
                    break;
                }
                rest.swap(tasks_);
            }
            for (auto& t : rest) t();
        }
    }

    static Core* self(GLFWwindow* w) { return static_cast<Core*>(glfwGetWindowUserPointer(w)); }

    void install_callbacks() {
        glfwSetCursorPosCallback(win_, [](GLFWwindow* w, double x, double y) {
            Core* c = self(w);
            c->note_mouse_probe();
            CaliperInputEvent ev{};
            ev.type = CALIPER_EVENT_MOUSE_MOVE;
            ev.x = (float)(x * c->scale_);
            ev.y = (float)(y * c->scale_);
            c->send(ev);
        });
        glfwSetMouseButtonCallback(win_, [](GLFWwindow* w, int button, int action, int) {
            if (button > 2) return;
            CaliperInputEvent ev{};
            ev.type = CALIPER_EVENT_MOUSE_BUTTON;
            ev.button = button;  // GLFW left/right/middle == 0/1/2
            ev.down = action == GLFW_PRESS;
            self(w)->send(ev);
        });
        glfwSetScrollCallback(win_, [](GLFWwindow* w, double dx, double dy) {
            CaliperInputEvent ev{};
            ev.type = CALIPER_EVENT_MOUSE_SCROLL;
            ev.dx = (float)dx;
            ev.dy = (float)dy;
            self(w)->send(ev);
        });
        glfwSetKeyCallback(win_, [](GLFWwindow* w, int key, int, int action, int mods) {
            if (action == GLFW_REPEAT) action = GLFW_PRESS;
            const int k = glfw_to_imgui_key(key);
            if (k == ImGuiKey_None) return;
            CaliperInputEvent ev{};
            ev.type = CALIPER_EVENT_KEY;
            ev.key = k;
            ev.down = action == GLFW_PRESS;
            ev.mods = glfw_mods_to_caliper(mods);
            self(w)->send(ev);
        });
        glfwSetCharCallback(win_, [](GLFWwindow* w, unsigned int cp) {
            CaliperInputEvent ev{};
            ev.type = CALIPER_EVENT_TEXT;
            ev.codepoint = cp;
            self(w)->send(ev);
        });
        glfwSetFramebufferSizeCallback(win_, [](GLFWwindow* w, int fw, int fh) {
            Core* c = self(w);
            if (fw <= 0 || fh <= 0) return;  // minimized
            c->width_ = fw;
            c->height_ = fh;
            CaliperInputEvent ev{};
            ev.type = CALIPER_EVENT_RESIZE;
            ev.width = fw;
            ev.height = fh;
            c->send(ev);
        });
        glfwSetWindowContentScaleCallback(win_, [](GLFWwindow* w, float sx, float) {
            Core* c = self(w);
            if (sx <= 0) return;
            c->scale_ = sx;
            CaliperInputEvent ev{};
            ev.type = CALIPER_EVENT_CONTENT_SCALE;
            ev.scale = sx;
            c->send(ev);
        });
        glfwSetWindowFocusCallback(win_, [](GLFWwindow* w, int focused) {
            CaliperInputEvent ev{};
            ev.type = CALIPER_EVENT_FOCUS;
            ev.focused = focused;
            self(w)->send(ev);
        });
        // Live resize: AppKit runs a nested loop while the user drags the
        // corner; GLFW reports refresh requests from inside it, so render there
        // to keep the canvas live during the drag. With the render thread
        // running it repaints on its own; rendering here would race it.
        glfwSetWindowRefreshCallback(win_, [](GLFWwindow* w) {
            Core* c = self(w);
            if (c->core_ && !c->rendering_.load()) {
                @autoreleasepool {
                    caliper_core_frame(c->core_);
                }
            }
        });
    }

    void watch_occlusion() {
        NSWindow* nsw = (NSWindow*)glfwGetCocoaWindow(win_);
        Core* self = this;
        {
            std::lock_guard<std::mutex> lk(probe_mu_);
            occlusion_.clear();
            occlusion_.emplace_back(mono_ns(), ([nsw occlusionState] & NSWindowOcclusionStateVisible) != 0);
        }
        occlusion_observer_ = [[NSNotificationCenter defaultCenter]
            addObserverForName:NSWindowDidChangeOcclusionStateNotification
                        object:nsw
                         queue:nil
                    usingBlock:^(NSNotification*) {
                        const bool vis = ([nsw occlusionState] & NSWindowOcclusionStateVisible) != 0;
                        std::lock_guard<std::mutex> lk(self->probe_mu_);
                        self->occlusion_.emplace_back(mono_ns(), vis);
                    }];
    }

    void note_mouse_probe() {
        std::lock_guard<std::mutex> lk(probe_mu_);
        if (probe_mouse_pending_.empty()) return;
        probes_.emplace_back(1, probe_mouse_pending_.front(), mono_ns());
        probe_mouse_pending_.pop_front();
    }

    CaliperCore* core_ = nullptr;
    GLFWwindow* win_ = nullptr;
    bool glfw_inited_ = false;
    bool attached_ = false;
    bool offscreen_ = false;
    bool heatmap_live_apply_ = false;
    int width_ = 0, height_ = 0;
    float scale_ = 1.f;
    std::string data_dir_;
    LogRing log_;
    std::atomic<bool> loop_stop_{false};

    // Render thread. mu_ guards everything below it except the frame-thread
    // private fields (hb_, frames_seen_).
    std::thread render_;
    std::thread::id render_id_;
    std::atomic<bool> rendering_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    bool render_tid_ready_ = false;
    double fps_ = 60.0;
    bool heartbeat_ = false;
    std::map<std::string, LiveSnap> queued_;           // newest per view
    std::map<std::string, Claim> claims_;              // slots the core may read
    std::vector<CaliperInputEvent> input_q_;
    std::deque<std::function<void()>> tasks_;
    uint64_t frames_ = 0, published_ = 0, superseded_ = 0, drawn_ = 0, errors_ = 0;
    uint64_t stats_unreadable_ = 0, nonfinite_range_ = 0, events_applied_ = 0;
    uint64_t deferred_takes_ = 0;
    double max_gap_ms_ = 0.0;
    std::string last_render_error_;
    Samples latency_ms_, tick_ms_, gap_ms_;
    std::vector<float> hb_;
    std::map<std::string, std::pair<float, float>> last_range_;
    uint64_t frames_seen_ = 0;

    // Main-thread probes.
    std::mutex probe_mu_;
    std::deque<uint64_t> probe_mouse_pending_;
    std::vector<std::tuple<int, uint64_t, uint64_t>> probes_;
    std::vector<std::tuple<uint64_t, bool>> occlusion_;
    id occlusion_observer_ = nil;
    id<NSObject> no_app_nap_activity_ = nil;
};

}  // namespace

NB_MODULE(_native, m) {
    m.doc() = "Caliper embed C ABI binding (one Core per process; window calls on the main thread)";
    // A Core kept alive by a script global at interpreter exit is closed by
    // caliper's atexit hook already; nanobind's leak report would only be noise.
    nb::set_leak_warnings(false);
    m.attr("HAVE_VIEWS") = true;
    m.attr("HAVE_LIVE_VIEWS") = true;
    m.attr("EMBED_API_VERSION") = CALIPER_EMBED_API_VERSION;
    m.attr("VIEW_FLAG_RAW") = (uint32_t)CALIPER_VIEW_FLAG_RAW;
    m.attr("VIEW_FLAG_WAIT_GPU") = (uint32_t)CALIPER_VIEW_FLAG_WAIT_GPU;
    m.attr("LIVE_APPLY_RAW") = (uint32_t)CALIPER_LIVE_APPLY_RAW;
    m.attr("LIVE_APPLY_AUTO_VMIN") = (uint32_t)CALIPER_LIVE_APPLY_AUTO_VMIN;
    m.attr("LIVE_APPLY_AUTO_VMAX") = (uint32_t)CALIPER_LIVE_APPLY_AUTO_VMAX;
    m.attr("LIVE_APPLY_WAIT_GPU") = (uint32_t)CALIPER_LIVE_APPLY_WAIT_GPU;
    m.attr("KIND_HEATMAP") = (int)CALIPER_LIVE_KIND_HEATMAP;
    m.attr("KIND_IMAGE") = (int)CALIPER_LIVE_KIND_IMAGE;
    m.attr("KIND_POINTS") = (int)CALIPER_LIVE_KIND_POINTS;
    m.attr("KIND_SERIES") = (int)CALIPER_LIVE_KIND_SERIES;
    m.attr("KIND_HISTOGRAM") = (int)CALIPER_LIVE_KIND_HISTOGRAM;
    m.attr("HEATMAP_IMAGE_VIA_LIVE_APPLY") = kHeatmapImageViaLiveApply;
    m.attr("HEARTBEAT_VIEW") = kHeartbeatView;
    m.attr("CMAP_VIRIDIS") = 0;
    m.attr("CMAP_MAGMA") = 1;
    m.attr("CMAP_RDBU") = 2;

    m.def("colormap_lut", [](int cmap) {
        auto arr = owned_array<uint32_t>({256});
        if (!caliper_core_colormap_lut(cmap, arr.data()))
            throw std::invalid_argument("colormap_lut: unknown colormap " + std::to_string(cmap));
        return arr;
    }, "cmap"_a, "The core's 256-entry RGBA8 LUT (little-endian packed R | G<<8 | B<<16 | A<<24).");

    m.def("is_main_thread", []() { return pthread_main_np() != 0; });
    m.def("monotonic_ns", []() { return mono_ns(); },
          "clock_gettime_nsec_np(CLOCK_UPTIME_RAW): the clock of the render thread's stamps.");

    nb::class_<SharedBuffer>(m, "SharedBuffer",
                             "A caliper-made shared-storage MTLBuffer (zero-filled).")
        .def(nb::init<size_t>(), "nbytes"_a)
        .def("dlpack", &SharedBuffer::dlpack, "device"_a, "dtype"_a, "shape"_a,
             "A 'dltensor' capsule for torch.from_dlpack: kDLMetal for 'mps', kDLCPU for 'cpu'.")
        .def("read_f32", &SharedBuffer::read_f32, "count"_a)
        .def_prop_ro("nbytes", &SharedBuffer::nbytes)
        .def_prop_ro("address", &SharedBuffer::address)
        .def_prop_ro("handle", &SharedBuffer::handle);

    nb::class_<Core>(m, "Core")
        .def(nb::init<std::optional<std::string>>(), "data_dir"_a = nb::none())
        .def("open_window", &Core::open_window, "width"_a = 1280, "height"_a = 800,
             "title"_a = "caliper", "activate"_a = true)
        .def("open_offscreen", &Core::open_offscreen, "width"_a, "height"_a)
        .def("poll", &Core::poll)
        .def("wait_events", &Core::wait_events, "timeout"_a)
        .def("run_event_loop", &Core::run_event_loop, "signal_check"_a = 0.1, "timeout"_a = -1.0)
        .def("stop_event_loop", &Core::stop_event_loop)
        .def("frame", &Core::frame, "Pump one frame; returns the wall time of the call in ms.")
        .def("should_close", &Core::should_close)
        .def("read_pixels", &Core::read_pixels)
        .def("last_error", &Core::last_error)
        .def("log_lines", &Core::log_lines)
        .def("view_tensor", &Core::view_tensor, "name"_a, "capsule"_a,
             "byte_offset_extra"_a = 0, "colormap"_a = 0, "vmin"_a = 0.0f, "vmax"_a = 1.0f,
             "flags"_a = (uint32_t)CALIPER_VIEW_FLAG_WAIT_GPU)
        .def("view_info", &Core::view_info, "name"_a)
        .def("view_remove", &Core::view_remove, "name"_a)
        .def("view_count", &Core::view_count)
        .def("view_readback", &Core::view_readback, "name"_a)
        .def("publish_live", &Core::publish_live, "name"_a, "kind"_a, "slot"_a, "payload"_a,
             "shape"_a, "dtype"_a, "aux"_a.none(), "aux_n"_a, "stats"_a.none(), "stats_count"_a,
             "colormap"_a, "vmin"_a, "vmax"_a, "flags"_a, "series_total"_a, "points_dims"_a,
             "label_count"_a, "seq"_a, "stamp_ns"_a)
        .def("reading_slots", &Core::reading_slots, "name"_a)
        .def("reading_slot", &Core::reading_slot, "name"_a)
        .def("forget", &Core::forget, "name"_a)
        .def("live_view_info", &Core::live_view_info, "name"_a)
        .def("live_view_values", &Core::live_view_values, "name"_a)
        .def("live_view_readback", &Core::live_view_readback, "name"_a)
        .def("live_view_remove", &Core::live_view_remove, "name"_a)
        .def("live_view_count", &Core::live_view_count)
        .def("live_view_read_done", &Core::live_view_read_done, "name"_a)
        .def("live_set_paused", &Core::live_set_paused, "paused"_a)
        .def("live_paused", &Core::live_paused)
        .def_prop_ro("heatmap_live_apply", &Core::heatmap_live_apply)
        .def("start_render_thread", &Core::start_render_thread, "fps"_a = 60.0,
             "heartbeat"_a = false)
        .def("stop_render_thread", &Core::stop_render_thread)
        .def("render_stats", &Core::render_stats, "reset"_a = false)
        .def("post_probe", &Core::post_probe, "kind"_a)
        .def("probe_results", &Core::probe_results, "reset"_a = false)
        .def("set_window_pos", &Core::set_window_pos, "x"_a, "y"_a)
        .def("occlusion_log", &Core::occlusion_log)
        .def("close", &Core::close)
        .def_prop_ro("rendering", &Core::rendering)
        .def_prop_ro("window_number", &Core::window_number)
        .def_prop_ro("closed", &Core::closed)
        .def_prop_ro("offscreen", &Core::is_offscreen)
        .def_prop_ro("has_window", &Core::has_window)
        .def_prop_ro("width", &Core::width)
        .def_prop_ro("height", &Core::height)
        .def_prop_ro("content_scale", &Core::content_scale);
}
