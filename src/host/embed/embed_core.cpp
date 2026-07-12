// ===========================================================================
// embed_core.cpp — implementation of the embed C ABI (caliper/embed.h) inside
// libcaliper (Compass R4, L2a). This is the seam that makes "embeddable" true:
// it COMPOSES the existing pieces the L1 extraction proved out — core_lifecycle
// (renderer selection + applet-canvas ImGui context), the service registry
// (services_init/set_renderer/shutdown), the applet loader, and a frame arc
// carved from main.cpp::run() — behind a toolkit-free C surface. No new
// rendering, no new service; just the ordered wiring, GLFW-free.
//
// ORDERING IS LOAD-BEARING (l1-survey §6, main.cpp:76-114 / 417-439). Every
// step below cites the exe path it mirrors; the documented crashes behind each
// reordering (MTLTexture UAF, DuckDB static-dtor malloc abort, worker-touching-
// freed-applet 139) bind here identically. Do not reorder.
//
// This TU is plain C++: all Metal lives behind the HostRenderer canvas_* seam
// (metal_renderer.mm), so ObjC stays scoped to the backend .mm files.
// ===========================================================================
#include "caliper/embed.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <imgui.h>

#include "app_paths.h"
#include "applet_loader.h"
#include "core_lifecycle.h"
#include "host_services.h"
#include "job_system.h"   // complete JobSystem (host_job_system().cancel_all_and_join)
#include "host_version.h"
#include "frame_watchdog.h"
#include "renderer/host_renderer.h"

#include <caliper/abi.h>

#ifdef __APPLE__
#  include <mach-o/dyld.h>
#  include <filesystem>
#elif defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <filesystem>
#else
#  include <filesystem>
#endif

using namespace caliper_host;

namespace {

// One CaliperCore per process in v0 (D5 already binds the process to one
// libtorch; full de-singletonization is not L2a). This lock makes the refusal
// honest — a second create() returns NULL while one is live.
//
// ATOMIC, not a plain pointer (M2): two threads racing caliper_core_create must
// not BOTH pass the guard and each half-build a core over the shared process
// globals (ImGui context, service registry). The slot is claimed with a single
// compare-exchange BEFORE any construction — the loser refuses immediately, and
// the winner holds a sentinel reservation until it either publishes the real
// pointer or releases the slot on a build failure.
std::atomic<CaliperCore*> g_live_core{nullptr};
CaliperCore* const kCoreReserved = reinterpret_cast<CaliperCore*>(1);

double now_sec() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

}  // namespace

// The core is a plain struct (opaque to the ABI). It holds ONLY instance state
// the ABI forces; the service registry, crash guard, colormap LUTs, ImGui
// context, and GLFW-analog remain process-global exactly as L1 left them (see
// l1-survey §7 / the report). That is why v0 is one-core-per-process.
struct CaliperCore {
    std::unique_ptr<HostRenderer>        renderer;
    std::unique_ptr<AppletLoader>        loader;
    FrameWatchdog                        watchdog;

    CaliperLogFn   log_fn = nullptr;
    CaliperCrashFn crash_fn = nullptr;
    void*          userdata = nullptr;

    bool                     canvas_ready = false;
    HostRenderer::CanvasMode canvas_mode = HostRenderer::CANVAS_OFFSCREEN;
    int                      canvas_w = 0, canvas_h = 0;
    float                    scale = 1.0f;

    int    active_applet = -1;
    double last_frame_time = 0.0;
    std::string last_error;

    void log(int level, const std::string& msg) {
        if (log_fn) log_fn(userdata, level, msg.c_str());
        else std::fprintf(stderr, "[embed] %s\n", msg.c_str());
    }
    void fail(const std::string& why) { last_error = why; log(2, why); }
};

// ---------------------------------------------------------------------------
// create / shutdown
// ---------------------------------------------------------------------------
CaliperCore* caliper_core_create(const CaliperCoreDesc* desc) {
    if (!desc || desc->struct_size < sizeof(CaliperCoreDesc)) {
        std::fprintf(stderr, "[embed] create: null/short CaliperCoreDesc\n");
        return nullptr;
    }
    // Claim the one-core-per-process slot atomically BEFORE building anything
    // (M2). A racing second create sees the CAS fail and refuses; only the
    // winner proceeds, holding kCoreReserved until it publishes or releases.
    CaliperCore* expected = nullptr;
    if (!g_live_core.compare_exchange_strong(expected, kCoreReserved,
                                             std::memory_order_acq_rel)) {
        // Honest refusal (§4.3): one core per process in v0.
        if (desc->log_fn)
            desc->log_fn(desc->userdata, 2,
                         "caliper_core_create: a core is already live "
                         "(one per process in v0); shut it down first");
        else
            std::fprintf(stderr, "[embed] create refused: a core is already "
                                 "live (one per process in v0)\n");
        return nullptr;
    }

    // Release the one-core slot if we bail before publishing — by an early
    // return OR by an exception unwinding out of this function (M2). The CAS
    // above pinned kCoreReserved; if services_init() or the AppletLoader ctor
    // (or any step between here and the publish below) throws, this guard puts
    // the slot back to nullptr — otherwise it stays pinned forever and every
    // future caliper_core_create() refuses. Disarmed only once the real pointer
    // is about to be published.
    struct SlotGuard {
        bool armed = true;
        ~SlotGuard() {
            if (armed) g_live_core.store(nullptr, std::memory_order_release);
        }
    } slot_guard;

    auto core = std::make_unique<CaliperCore>();
    core->log_fn   = desc->log_fn;
    core->crash_fn = desc->crash_fn;
    core->userdata = desc->userdata;

    // STEP 1 (mirrors main.cpp:67): applet-canvas ImGui/ImPlot/ImPlot3D contexts
    // FIRST — they must exist before the renderer wires its ImGui backend.
    core_create_ui_context();

    // STEP 2 (mirrors main.cpp:49 / renderer factory selection). DEFAULT honors
    // CALIPER_RENDERER + the platform default (Metal/Vulkan, GL fallback). An
    // explicit backend that this OS can't build refuses cleanly. GL is never an
    // embed target — a GL-resolved renderer refuses later at attach_canvas.
    switch (desc->renderer) {
        case CALIPER_RENDERER_DEFAULT:
            core->renderer = core_select_renderer();
            break;
        case CALIPER_RENDERER_METAL:
#ifdef __APPLE__
            core->renderer = make_metal_renderer();
#endif
            break;
        case CALIPER_RENDERER_VULKAN:
#ifdef _WIN32
            core->renderer = make_vulkan_renderer();
#endif
            break;
    }
    if (core->renderer) {
        // Core-owned diagnostic through log_fn (NOT the applet log service): the
        // resolved backend. The exe prints its own "[renderer] metal"; the embed
        // core routes the equivalent through the embedder's sink.
        core->log(1, std::string("renderer: ") + core->renderer->name());
    }
    if (!core->renderer) {
        core_destroy_ui_context();
        // slot_guard releases the one-core slot on this return.
        if (desc->log_fn)
            desc->log_fn(desc->userdata, 2,
                         "caliper_core_create: requested renderer not available "
                         "on this platform");
        else
            std::fprintf(stderr, "[embed] create: renderer unavailable\n");
        return nullptr;
    }

    // STEP 3 (mirrors main.cpp:90 / :94): open the service registry, then bind
    // the live renderer to the bridge. The bridge is lazy (constructed on first
    // thunk), so binding before the renderer's device exists (canvas_init) is
    // safe — the first bridge call happens during an applet frame, post-attach.
    services_init();
    services_set_renderer(core->renderer.get());

    // STEP 4 (mirrors main.cpp:113-135): manifest-first discovery. The loader
    // never dlopens here — scan() only reads manifests. service_ids() is a
    // static set, valid independent of services_init.
    core->loader = std::make_unique<AppletLoader>(
        HostCaps{2, kHostVersionStr, service_ids()},
        caliper::app_data_path("data"));
    core->loader->scan(caliper::app_data_path("applets"));
    {
        namespace fs = std::filesystem;
        std::string exe_dir;
#ifdef __APPLE__
        char buf[1024]; uint32_t sz = sizeof(buf);
        if (_NSGetExecutablePath(buf, &sz) == 0)
            exe_dir = fs::path(buf).parent_path().string();
#elif defined(_WIN32)
        char buf[MAX_PATH]; GetModuleFileNameA(nullptr, buf, MAX_PATH);
        exe_dir = fs::path(buf).parent_path().string();
#else
        exe_dir = fs::read_symlink("/proc/self/exe", std::error_code{})
                      .parent_path().string();
#endif
        if (!exe_dir.empty())
            core->loader->scan((fs::path(exe_dir) / "applets").string());
    }
    if (desc->applets_dir && desc->applets_dir[0])
        core->loader->scan(desc->applets_dir);

    // Publish the real pointer, replacing the sentinel reservation. Disarm the
    // guard first: from here on the slot legitimately holds a live core.
    slot_guard.armed = false;
    g_live_core.store(core.get(), std::memory_order_release);
    return core.release();
}

void caliper_core_shutdown(CaliperCore* core) {
    if (!core) return;
    // Teardown reverses create (main.cpp:397-417). Workers FIRST: a job may hold
    // a pointer into applet state, so join before close_all destroys it.
    host_job_system().cancel_all_and_join();
    if (core->loader) core->loader->close_all();
    // Applets are down; drop the renderer from the bridge BEFORE renderer
    // teardown so no bridge thunk touches a destroyed renderer.
    services_set_renderer(nullptr);
    // Close DuckDB stores now — leaving them to static dtors races DuckDB's
    // globals and aborts in malloc at exit.
    services_shutdown();
    if (core->renderer) core->renderer->canvas_shutdown();  // no-op if unattached
    core->renderer.reset();
    // ImGui contexts last, AFTER the renderer's ImGui backend is gone.
    core_destroy_ui_context();

    CaliperCore* expected = core;
    g_live_core.compare_exchange_strong(expected, nullptr,
                                        std::memory_order_acq_rel);
    delete core;
}

// ---------------------------------------------------------------------------
// canvas
// ---------------------------------------------------------------------------
int caliper_core_attach_canvas(CaliperCore* core, void* native_view,
                               const CaliperCanvasDesc* desc) {
    if (!core) return 0;
    if (!desc || desc->struct_size < sizeof(CaliperCanvasDesc)) {
        core->fail("attach_canvas: null/short CaliperCanvasDesc");
        return 0;
    }
    if (core->canvas_ready) {
        core->fail("attach_canvas: a canvas is already attached (one per core)");
        return 0;
    }
    if (desc->width <= 0 || desc->height <= 0) {
        core->fail("attach_canvas: canvas size must be positive");
        return 0;
    }
    if (!core->renderer->canvas_supported()) {
        core->fail("attach_canvas: embed requires Metal or Vulkan (the "
                   "GL fallback is not an embed target)");
        return 0;
    }
    HostRenderer::CanvasMode mode = (desc->mode == CALIPER_CANVAS_OFFSCREEN)
                                        ? HostRenderer::CANVAS_OFFSCREEN
                                        : HostRenderer::CANVAS_WINDOW;
    if (!core->renderer->canvas_init(native_view, mode, desc->width, desc->height)) {
        core->fail("attach_canvas: backend canvas_init failed");
        return 0;
    }
    core->canvas_ready = true;
    core->canvas_mode = mode;
    core->canvas_w = desc->width;
    core->canvas_h = desc->height;
    core->scale = desc->content_scale > 0.0f ? desc->content_scale : 1.0f;
    core->last_frame_time = now_sec();
    core->last_error.clear();
    core->log(1, "canvas attached: " + std::to_string(core->canvas_w) + "x" +
                     std::to_string(core->canvas_h) + " @" +
                     std::to_string(core->scale) + "x (" +
                     (mode == HostRenderer::CANVAS_WINDOW ? "window" : "offscreen") +
                     ")");
    return 1;
}

// ---------------------------------------------------------------------------
// frame — the carved core arc of main.cpp::run() (new_frame -> guarded applet
// draw -> render -> deferred teardown), minus every chrome item.
// ---------------------------------------------------------------------------
void caliper_core_frame(CaliperCore* core) {
    if (!core) return;
    if (!core->canvas_ready) {
        core->fail("frame: no canvas attached");
        return;
    }

    // The embed layer owns timing + size (there is no imgui_impl_glfw to set
    // them). DisplaySize is logical points; DisplayFramebufferScale carries the
    // DPI so ImGui_ImplMetal projects/viewports to the physical target.
    ImGuiIO& io = ImGui::GetIO();
    io.DisplayFramebufferScale = ImVec2(core->scale, core->scale);
    io.DisplaySize = ImVec2((float)core->canvas_w / core->scale,
                            (float)core->canvas_h / core->scale);
    const double now = now_sec();
    float dt = (float)(now - core->last_frame_time);
    io.DeltaTime = dt > 0.0f ? dt : (1.0f / 60.0f);   // ImGui asserts DeltaTime>0
    core->last_frame_time = now;

    core->renderer->canvas_new_frame();

    bool alive = true;
    if (core->active_applet >= 0) {
        CaliperFrameInfo fi{};
        fi.struct_size = sizeof fi;
        fi.fb_width = core->canvas_w;    // physical px
        fi.fb_height = core->canvas_h;
        fi.dpi_scale = core->scale;
        fi.time_sec = now;
        fi.delta_sec = dt;   // same clock as io.DeltaTime
        double t0 = now_sec();
        alive = core->loader->frame(core->active_applet, fi);  // crash-guarded inside
        core->watchdog.feed((now_sec() - t0) * 1000.0);
    }

    // Render BEFORE any teardown: the applet's draw list this frame may still
    // reference bridge textures; releasing them before the pass consumes the
    // draw list is a use-after-free (main.cpp:370). Deferred teardown follows.
    core->renderer->canvas_render();

    if (!alive) {
        // Applet faulted mid-frame and was quarantined by the loader's crash
        // guard (existing machinery — we route it, we don't rebuild it). Tear
        // it down AFTER render, then surface it to the embedder.
        int crashed = core->active_applet;
        std::string id = core->loader->at(crashed).manifest.id;
        std::string fault = core->loader->at(crashed).status_text;
        host_job_system().cancel_all_and_join();
        core->loader->teardown(crashed);
        core->active_applet = -1;
        if (core->crash_fn)
            core->crash_fn(core->userdata, id.c_str(), fault.c_str());
        else
            core->log(2, "applet " + id + " faulted: " + fault);
    }
}

void caliper_core_event(CaliperCore* core, const CaliperInputEvent* ev) {
    if (!core || !ev) return;
    // struct_size gate (embed.h promises every struct is size-checked): a caller
    // built against an older, smaller CaliperInputEvent is dropped rather than
    // misread field-by-field. Same exact rule as create/attach_canvas.
    if (ev->struct_size < sizeof(CaliperInputEvent)) return;
    if (!core->canvas_ready) return;   // event before a canvas: no-op, no crash
    ImGuiIO& io = ImGui::GetIO();
    switch (ev->type) {
        case CALIPER_EVENT_MOUSE_MOVE:
            io.AddMousePosEvent(ev->x / core->scale, ev->y / core->scale);
            break;
        case CALIPER_EVENT_MOUSE_BUTTON:
            io.AddMouseButtonEvent(ev->button, ev->down != 0);
            break;
        case CALIPER_EVENT_MOUSE_SCROLL:
            io.AddMouseWheelEvent(ev->dx, ev->dy);
            break;
        case CALIPER_EVENT_KEY:
            io.AddKeyEvent(ImGuiMod_Ctrl,  (ev->mods & CALIPER_MOD_CTRL)  != 0);
            io.AddKeyEvent(ImGuiMod_Shift, (ev->mods & CALIPER_MOD_SHIFT) != 0);
            io.AddKeyEvent(ImGuiMod_Alt,   (ev->mods & CALIPER_MOD_ALT)   != 0);
            io.AddKeyEvent(ImGuiMod_Super, (ev->mods & CALIPER_MOD_SUPER) != 0);
            io.AddKeyEvent((ImGuiKey)ev->key, ev->down != 0);
            break;
        case CALIPER_EVENT_TEXT:
            io.AddInputCharacter(ev->codepoint);
            break;
        case CALIPER_EVENT_RESIZE:
            if (ev->width > 0 && ev->height > 0) {
                core->canvas_w = ev->width;
                core->canvas_h = ev->height;
                core->renderer->canvas_resize(ev->width, ev->height);
            }
            break;
        case CALIPER_EVENT_CONTENT_SCALE:
            if (ev->scale > 0.0f) core->scale = ev->scale;
            break;
        case CALIPER_EVENT_FOCUS:
            io.AddFocusEvent(ev->focused != 0);
            break;
    }
}

// ---------------------------------------------------------------------------
// applet control (reuses the loader's manifest discovery)
// ---------------------------------------------------------------------------
int caliper_core_load_applet(CaliperCore* core, const char* manifest_id) {
    if (!core || !manifest_id) return 0;
    if (!core->loader) { core->fail("load_applet: no loader"); return 0; }
    // W1: an applet's launch + first frame touch the renderer's ImGui backend
    // (canvas_init wired it). Loading before a canvas is attached is an honest
    // refusal — the header contract says attach first — not a deferred crash.
    if (!core->canvas_ready) {
        core->fail("load_applet: attach a canvas before loading applets");
        return 0;
    }
    int idx = -1;
    for (int i = 0; i < core->loader->count(); ++i)
        if (core->loader->at(i).manifest.id == manifest_id) { idx = i; break; }
    if (idx < 0) {
        // An unknown id refuses WITHOUT disturbing a running applet — a typo
        // must not kill the live session. Only a resolvable target proceeds to
        // the teardown-then-launch arc below.
        core->fail(std::string("load_applet: no applet with id ") + manifest_id);
        return 0;
    }

    // Tear down the currently-active applet FIRST — same id (a clean restart) or
    // different id (a swap) — via the exact ordered arc the exe uses
    // (main.cpp:377 / :403): hard-join every worker (a job may hold a pointer
    // into applet state — the worker-touching-freed-applet 139 crash class),
    // THEN destroy the instance. Doing this BEFORE launch fixes two faults the
    // final review flagged:
    //   (1) AppletLoader::launch would teardown a same-id Active entry with NO
    //       cancel_all_and_join first (applet_loader.cpp:133) — the documented
    //       worker-touching-freed-applet crash;
    //   (2) the previous order ran cancel_all_and_join AFTER launch, cancelling
    //       the NEW applet's freshly-scheduled init jobs.
    // ACCEPTED CONSEQUENCE (documented in embed.h load_applet): if the launch
    // below fails, the old applet is already gone — a failed load leaves NO
    // applet, not the previous one. load_applet is called between frames (no
    // live draw list), so this teardown is immediate, mirroring unload_applet.
    if (core->active_applet >= 0) {
        host_job_system().cancel_all_and_join();
        core->loader->teardown(core->active_applet);
        core->active_applet = -1;
    }

    // Build the CaliperHost prototype exactly as main.cpp::launch_applet does;
    // the loader fills applet_data_dir per entry before initialize().
    CaliperHost proto{};
    proto.struct_size  = sizeof(CaliperHost);
    proto.abi_epoch    = 2;
    proto.host_version = kHostVersionU32;
    proto.applet_data_dir = nullptr;
    proto.get_service = [](const CaliperHost*, const char* id) {
        return services_get(id);
    };
    if (!core->loader->launch(idx, proto)) {
        core->fail(std::string("load_applet: launch refused/failed: ") +
                   core->loader->at(idx).status_text);
        return 0;   // active_applet stays -1: a failed load leaves NO applet.
    }

    core->active_applet = idx;
    core->watchdog.reset();
    core->last_frame_time = now_sec();
    core->last_error.clear();
    core->log(1, std::string("applet loaded: ") + manifest_id);
    return 1;
}

void caliper_core_unload_applet(CaliperCore* core) {
    if (!core || core->active_applet < 0) return;
    // Called between frames (no live draw list), so teardown immediately.
    // Workers first (they may reference applet state), then the instance.
    host_job_system().cancel_all_and_join();
    core->loader->teardown(core->active_applet);
    core->active_applet = -1;
}

int caliper_core_read_pixels(CaliperCore* core, void* buf, int stride) {
    if (!core) return 0;
    if (!core->canvas_ready || !buf) {
        if (core) core->fail("read_pixels: no canvas / null buffer");
        return 0;
    }
    return core->renderer->canvas_read_pixels((uint8_t*)buf, stride) ? 1 : 0;
}

const char* caliper_core_last_error(CaliperCore* core) {
    return core ? core->last_error.c_str() : "";
}

// ---------------------------------------------------------------------------
// service consumption (v1.1) — vend from the SAME process registry an applet
// gets. embed_core already builds the CaliperHost proto's get_service as a
// thunk over services_get() (load_applet above); this is that same registry,
// exposed to the host directly. No new state, no per-core table: the service
// registry is process-global (host note: one core per process in v0), and the
// tables are process-static, so the returned pointer is valid until shutdown.
// ---------------------------------------------------------------------------
const void* caliper_core_get_service(CaliperCore* core, const char* id) {
    if (!core || !id) return nullptr;
    return services_get(id);   // NULL for unknown ids (services_get contract)
}
