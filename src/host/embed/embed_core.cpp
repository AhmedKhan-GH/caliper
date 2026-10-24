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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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
#include "tensor_bridge.h"   // TensorBridge (views), colormap_lut, bridge_last_reject

#include <caliper/abi.h>
#include <caliper/services/log_v1.h>   // CaliperLogLevel (applet log routing)

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

// The CaliperCore struct and the v1.2 TensorView record live in embed_core.h
// so the live-streaming module (src/host/live) shares them.
#include "embed_core.h"

// Applet log.v1 sink (embed v1.1, §3.3): route each applet log line to the
// embedder's log_fn, TAGGED "[applet] " so a host can tell applet lines apart
// from core diagnostics (core->log delivers those UNtagged). Installed at create
// only when the embedder provided a log_fn; the caliper exe installs nothing and
// keeps stderr. `ud` is the CaliperCore*; it is only dereferenced while the sink
// is installed, which is strictly between create (this core alive) and shutdown's
// post-join clear — no worker can call it after log_fn is gone.
static void embed_applet_log_sink(void* ud, CaliperLogLevel level,
                                  const char* msg) {
    CaliperCore* core = static_cast<CaliperCore*>(ud);
    if (!core->log_fn) return;
    std::string tagged = "[applet] ";
    tagged += (msg ? msg : "");
    core->log_fn(core->userdata, static_cast<int>(level), tagged.c_str());
}

// v1.2 view helpers (defined with the view ABI at the end of this file).
static void views_destroy_all(CaliperCore* core);
static void views_draw(CaliperCore* core);

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
            if (armed) {
                // Restore the default app-data root if data_dir routing was armed
                // for a create that then bailed — otherwise a failed create would
                // leak its per-core root into the next one.
                caliper::set_app_data_dir_override("");
                g_live_core.store(nullptr, std::memory_order_release);
            }
        }
    } slot_guard;

    // v1.1 (§3.3): route CaliperCoreDesc.data_dir into the app-data root BEFORE
    // the stores open (services_init) and the loader scans, so metrics/artifacts/
    // applet-data land under the embedder's directory. NULL/empty => the OS
    // default is used, byte-for-byte as before (the caliper exe passes NULL).
    if (desc->data_dir && desc->data_dir[0])
        caliper::set_app_data_dir_override(desc->data_dir);

    auto core = std::make_unique<CaliperCore>();
    core->log_fn   = desc->log_fn;
    core->crash_fn = desc->crash_fn;
    core->userdata = desc->userdata;

    // STEP 1 (mirrors main.cpp:67): applet-canvas ImGui/ImPlot/ImPlot3D contexts
    // FIRST — they must exist before the renderer wires its ImGui backend.
    core_create_ui_context();

    // imgui.ini placement. ImGui's default writes ./imgui.ini into the process
    // cwd — for an embedder (a Python script, Compass) that is the USER's working
    // directory, never ours to write. With a data_dir the window layout persists
    // at <data_dir>/imgui.ini (the string lives in the core, so the pointer is
    // valid until shutdown); without one there is NO layout persistence — an
    // anonymous embedder gets fresh default placement every run.
    {
        ImGuiIO& io = ImGui::GetIO();
        if (desc->data_dir && desc->data_dir[0]) {
            std::error_code ec;
            std::filesystem::create_directories(desc->data_dir, ec);   // usually exists
            core->ini_path =
                (std::filesystem::path(desc->data_dir) / "imgui.ini").string();
            io.IniFilename = core->ini_path.c_str();
        } else {
            io.IniFilename = nullptr;
        }
    }

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
        ImGui::GetIO().IniFilename = nullptr;   // nothing to persist; write no file
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

    // Route applet log.v1 lines to the embedder's log_fn (§3.3). Installed here,
    // AFTER every fallible create step, so no failure path can leave a sink
    // pointing at a half-built core; cleared in shutdown after workers join. No
    // log_fn -> no sink -> stderr, exactly as the caliper exe runs.
    if (core->log_fn)
        set_applet_log_sink(&embed_applet_log_sink, core.get());

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
    // v1.2 views: release their bridge textures and MTLBuffer imports while the
    // bridge still has its renderer — the same discipline as the applet
    // teardown above (no GPU resource may outlive the renderer that owns it).
    views_destroy_all(core);
    // Applets are down; drop the renderer from the bridge BEFORE renderer
    // teardown so no bridge thunk touches a destroyed renderer.
    services_set_renderer(nullptr);
    // Close DuckDB stores now — leaving them to static dtors races DuckDB's
    // globals and aborts in malloc at exit.
    services_shutdown();
    if (core->renderer) core->renderer->canvas_shutdown();  // no-op if unattached
    core->renderer.reset();
    // ImGui contexts last, AFTER the renderer's ImGui backend is gone. Flush the
    // layout to <data_dir>/imgui.ini explicitly (DestroyContext's own final save
    // is skipped below), then null IniFilename so the context never holds a
    // pointer into this core's ini_path past this point.
    {
        ImGuiIO& io = ImGui::GetIO();
        if (io.IniFilename) ImGui::SaveIniSettingsToDisk(io.IniFilename);
        io.IniFilename = nullptr;
    }
    core_destroy_ui_context();

    // Detach the applet log sink LAST, once no applet code can run again: workers
    // were joined above, and the loader's close_all() ran on_cleanup on THIS
    // (shutdown) thread with core->log_fn still valid — so the applet's cleanup
    // log lines still reached the embedder. Clear it before the core is freed
    // (the sink trampoline holds this core pointer). No log_fn -> never installed.
    set_applet_log_sink(nullptr, nullptr);

    // Restore the default app-data root: the stores are closed, and the next
    // core (NULL data_dir) must resolve the OS default again — the override is
    // tied to this core's lifetime.
    caliper::set_app_data_dir_override("");

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

    // v1.2 built-in tensor views draw AFTER the applet (so they work with no
    // applet loaded and float alongside one). Skipped on a fault frame: the
    // quarantined applet may have left the ImGui window stack mid-Begin.
    if (alive) views_draw(core);

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

// ===========================================================================
// v1.2 — built-in tensor views (embed.h). The core draws a host's tensor itself:
// texture through the process TensorBridge (the SAME bridge applets use, driven
// host-side so it can reach the host-only set_mapping), an ImGui window per view
// composited after the applet. Frame-thread only, like every bridge op.
//
// ZERO-COPY HONESTY. A METAL tensor goes through the v1.2 import path only
// (import_allocation + update_texture_from_alloc: the renderer's compute/blit
// reads the MTLBuffer in place at byte_offset). If that path refuses, the call
// fails — there is deliberately NO fallback that downloads the buffer and
// CPU-stages it, so info.imported/path never lie about what happened.
// ===========================================================================
using caliper_embed_views::TensorView;

namespace {

constexpr int64_t kMaxViewDim = 16384;   // Metal's 2-D texture limit

const char* view_device_name(int32_t d) {
    switch (d) {
        case CALIPER_DEV_CPU:   return "cpu";
        case CALIPER_DEV_METAL: return "metal";
        case CALIPER_DEV_CUDA:  return "cuda";
        default:                return "device?";
    }
}

const char* view_dtype_name(CaliperDType d) {
    switch (d) {
        case CALIPER_DT_F32:  return "f32";
        case CALIPER_DT_F16:  return "f16";
        case CALIPER_DT_BF16: return "bf16";
        case CALIPER_DT_I64:  return "i64";
        case CALIPER_DT_I32:  return "i32";
        case CALIPER_DT_U8:   return "u8";
    }
    return "unknown";
}

TensorView* find_view(CaliperCore* core, const char* name) {
    for (auto& v : core->views)
        if (v.name == name) return &v;
    return nullptr;
}

// Release a view's GPU resources through the bridge (no-op pieces when already
// zero). The bridge may be null only if no renderer is bound — then the
// renderer is gone and its tables went with it.
void release_view(TensorBridge* b, TensorView& v) {
    if (b) {
        if (v.tex)   b->release_texture(v.tex);
        if (v.alloc) b->release_allocation(v.alloc);
    }
    v.tex = 0;
    v.alloc = 0;
    v.alloc_buf = nullptr;
    v.alloc_bytes = 0;
}

// " (bridge: <reason>)" when the bridge logged a rejection during this call.
std::string bridge_reason() {
    const char* r = bridge_last_reject();
    return (r && r[0]) ? std::string(" (bridge: ") + r + ")" : std::string();
}

// First placement of a windowed view: image at 1 texel = 1 physical pixel,
// small tensors magnified by an integer factor up to a ~192 pt long side (the
// draw samples nearest when magnifying, so cells stay crisp), big ones capped
// to 90% of the canvas; windows tiled left-to-right, wrapping rows.
void place_view(CaliperCore* core, TensorView& v, ImVec2 img) {
    const float lw = (float)core->canvas_w / core->scale;
    const float lh = (float)core->canvas_h / core->scale;
    const ImGuiStyle& st = ImGui::GetStyle();
    const float chrome_w = st.WindowPadding.x * 2.0f;
    const float chrome_h = ImGui::GetFrameHeight() + st.WindowPadding.y * 2.0f +
                           ImGui::GetTextLineHeightWithSpacing();
    float s = 1.0f;
    const float longest = std::max(img.x, img.y);
    if (longest > 0.0f && longest < 192.0f) s = std::floor(192.0f / longest);
    const float max_w = std::max(64.0f, lw * 0.9f - chrome_w);
    const float max_h = std::max(64.0f, lh * 0.9f - chrome_h);
    s = std::min(s, std::min(max_w / img.x, max_h / img.y));
    const float ww = std::max(img.x * s + chrome_w, 240.0f);
    const float wh = img.y * s + chrome_h;

    const float gap = 8.0f;
    if (core->tile_x > gap && core->tile_x + ww > lw) {          // wrap the row
        core->tile_x = gap;
        core->tile_y += core->tile_row_h + gap;
        core->tile_row_h = 0.0f;
    }
    if (core->tile_y > gap && core->tile_y + wh > lh) {          // out of rows
        core->tile_x = gap;
        core->tile_y = gap;
        core->tile_row_h = 0.0f;
    }
    v.pos_x = core->tile_x;
    v.pos_y = core->tile_y;
    v.size_x = ww;
    v.size_y = wh;
    core->tile_x += ww + gap;
    core->tile_row_h = std::max(core->tile_row_h, wh);
    v.placed = true;
}

}  // namespace

static void views_destroy_all(CaliperCore* core) {
    if (core->views.empty()) return;
    TensorBridge* b = services_bridge();
    for (auto& v : core->views) release_view(b, v);
    core->views.clear();
}

// Draw every view into the current ImGui frame (between canvas_new_frame and
// canvas_render). The ImTextureID is the bridge's public CaliperTextureId: the
// bridge hands out renderer->tex_imtexture_id(tex), which on Metal is the
// id<MTLTexture> pointer imgui_impl_metal binds directly (tensor_bridge.h §5.4).
static void views_draw(CaliperCore* core) {
    if (core->views.empty()) return;
    const ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
    const ImDrawCallback nearest = pio.DrawCallback_SetSamplerNearest;
    const ImDrawCallback linear  = pio.DrawCallback_SetSamplerLinear;

    for (auto& v : core->views) {
        if (!v.tex) continue;
        // Logical size at which one texel covers one PHYSICAL pixel.
        const ImVec2 img((float)v.w / core->scale, (float)v.h / core->scale);

        if (v.flags & CALIPER_VIEW_FLAG_RAW) {
            // Borderless, padless, backgroundless, pinned at logical (0,0): at
            // content_scale 1.0 texel (i,j) lands on canvas pixel (i,j), sampled
            // nearest — the byte-compare surface.
            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(img, ImGuiCond_Always);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(1.0f, 1.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ImageBorderSize, 0.0f);
            const ImGuiWindowFlags f =
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
                ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoInputs |
                ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoBringToFrontOnFocus;
            const std::string id = "##caliper.view.raw." + v.name;
            if (ImGui::Begin(id.c_str(), nullptr, f)) {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                if (nearest) dl->AddCallback(nearest, nullptr);
                ImGui::Image((ImTextureID)v.tex, img);
                if (linear) dl->AddCallback(linear, nullptr);
            }
            ImGui::End();
            ImGui::PopStyleVar(5);
            continue;
        }

        if (!v.placed) place_view(core, v, img);
        ImGui::SetNextWindowPos(ImVec2(v.pos_x, v.pos_y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(v.size_x, v.size_y), ImGuiCond_FirstUseEver);
        // Title = the view name; the ### id keeps it apart from applet windows.
        const std::string label = v.name + "###caliper.view." + v.name;
        if (ImGui::Begin(label.c_str(), nullptr,
                         ImGuiWindowFlags_NoScrollbar |
                         ImGuiWindowFlags_NoScrollWithMouse)) {
            ImGui::Text("%dx%d %s %s %s updates=%llu last=%.2fms", v.w, v.h,
                        v.mapped ? "f32" : "u8", view_device_name(v.device),
                        v.path.c_str(), (unsigned long long)v.updates,
                        v.last_update_ms);
            const ImVec2 avail = ImGui::GetContentRegionAvail();
            const float s = std::min(avail.x / img.x, avail.y / img.y);
            if (s > 0.0f) {
                const ImVec2 sz(img.x * s, img.y * s);
                const float pad = (avail.x - sz.x) * 0.5f;      // center horizontally
                if (pad > 0.0f) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + pad);
                // Magnifying (>= 1 physical px per texel): nearest, so tensor
                // cells read as cells. Minifying: the default linear filter.
                const bool magnify = s >= 1.0f;
                ImDrawList* dl = ImGui::GetWindowDrawList();
                if (magnify && nearest) dl->AddCallback(nearest, nullptr);
                ImGui::Image((ImTextureID)v.tex, sz);
                if (magnify && linear) dl->AddCallback(linear, nullptr);
            }
        }
        ImGui::End();
    }
}

int caliper_core_view_tensor(CaliperCore* core, const CaliperViewDesc* desc) {
    if (!core) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    if (!desc || desc->struct_size < sizeof(CaliperViewDesc)) {
        core->fail("view_tensor: null/short CaliperViewDesc");
        return 0;
    }
    if (!desc->name || !desc->name[0]) {
        core->fail("view_tensor: name must be a non-empty string");
        return 0;
    }
    const std::string name = desc->name;
    auto vfail = [&](const std::string& why) {
        core->fail("view_tensor('" + name + "'): " + why);
        return 0;
    };
    if (!desc->tensor) return vfail("null tensor");
    if (!core->canvas_ready)
        return vfail("attach a canvas before showing tensors (the renderer's "
                     "device exists only after attach_canvas)");
    TensorBridge* bridge = services_bridge();
    if (!bridge || !core->renderer) return vfail("no renderer bound to the tensor bridge");

    // ---- validation (clear reasons up front; the bridge re-checks) ----------
    const CaliperTensor& src = *desc->tensor;
    bool mapped = false;
    if (src.dtype == CALIPER_DT_F32 && src.ndim == 2) {
        mapped = true;
    } else if (src.dtype == CALIPER_DT_U8 && src.ndim == 3) {
        if (src.shape[2] != 4)
            return vfail("a u8 image must be (H,W,4) RGBA; got C=" +
                         std::to_string(src.shape[2]) +
                         " (convert RGB/gray to RGBA before showing)");
    } else {
        return vfail(std::string("want a 2-D f32 (H,W) or a 3-D u8 (H,W,4) RGBA "
                                 "tensor; got ndim=") +
                     std::to_string(src.ndim) + " dtype=" + view_dtype_name(src.dtype));
    }
    const int64_t H = src.shape[0], W = src.shape[1];
    if (H < 1 || W < 1 || H > kMaxViewDim || W > kMaxViewDim)
        return vfail("dims must be in 1..16384; got " + std::to_string(H) + "x" +
                     std::to_string(W));
    {
        int64_t expect = 1;
        for (int i = src.ndim - 1; i >= 0; --i) {
            if (src.strides[i] != expect)
                return vfail("tensor must be row-major contiguous (strides in "
                             "elements); make it contiguous first");
            expect *= src.shape[i];
        }
    }
    const uint64_t extent_bytes =
        (uint64_t)H * (uint64_t)W * (mapped ? sizeof(float) : 4u);
    const bool metal = (src.device == CALIPER_DEV_METAL);
    if (src.device != CALIPER_DEV_CPU && !metal)
        return vfail(std::string("device must be CPU or METAL; got ") +
                     view_device_name(src.device));
    if (src.data == nullptr)
        return vfail(metal ? "null MTLBuffer (tensor->data)" : "null data");
    if (metal) {
        if (core->renderer->interop_device() != CALIPER_DEV_METAL)
            return vfail("METAL tensor, but the active renderer is not Metal");
        if (desc->byte_offset % 4 != 0)
            return vfail("METAL byte_offset must be a multiple of 4; got " +
                         std::to_string(desc->byte_offset));
        if (desc->byte_offset > UINT64_MAX - extent_bytes)
            return vfail("byte_offset overflows");
    }
    if (mapped) {
        if (colormap_lut(desc->colormap) == nullptr)
            return vfail("bad colormap " + std::to_string(desc->colormap) +
                         " (0 viridis, 1 magma, 2 rdbu)");
        if (!std::isfinite(desc->vmin) || !std::isfinite(desc->vmax))
            return vfail("vmin/vmax must be finite");
    }

    // What the bridge sees. METAL: data = the MTLBuffer; byte_offset rides the
    // import window (update_texture_from_alloc). CPU: bytes at data+byte_offset.
    CaliperTensor t = src;
    t.struct_size = sizeof(CaliperTensor);
    if (!metal) t.data = (uint8_t*)src.data + desc->byte_offset;
    const int32_t cm   = mapped ? desc->colormap : 0;
    const float   vmin = mapped ? desc->vmin : 0.0f;
    const float   vmax = mapped ? desc->vmax : 1.0f;

    TensorView* v = find_view(core, name.c_str());
    const bool need_create = !v || v->mapped != mapped || v->w != (int)W ||
                             v->h != (int)H;
    bridge_clear_last_reject();

    // ---- 1. METAL: import first, so the device path is proven before any
    // texture exists. Cached per view by buffer pointer (our strong retain keeps
    // that object alive, so an equal pointer IS the same buffer).
    CaliperAllocId alloc = 0;
    bool fresh_import = false;
    const uint64_t window = desc->byte_offset + extent_bytes;
    if (metal) {
        if (v && v->alloc && v->alloc_buf == src.data && v->alloc_bytes >= window) {
            alloc = v->alloc;
        } else {
            alloc = bridge->import_allocation(src.data, window,
                                              CALIPER_ALLOC_HANDLE_MTLBUFFER);
            if (!alloc)
                return vfail("MTLBuffer import refused — the buffer is shorter than "
                             "byte_offset + extent (" + std::to_string(window) +
                             " bytes), lives on another GPU, or the renderer "
                             "cannot import" + bridge_reason());
            fresh_import = true;
        }
    }
    auto drop_fresh_import = [&]() {
        if (fresh_import) bridge->release_allocation(alloc);
    };

    // ---- 2. texture: (re)create on a new view or a shape/dtype change — the new
    // one is built BEFORE the old is released, so a failed recreate keeps the old
    // view intact. A METAL create runs the bridge's direct device pass at offset
    // 0 (in bounds: window <= buffer length was just proven); the imported update
    // in step 3 overwrites it in queue order before any frame samples it. A CPU
    // create uploads the data itself (it IS this call's update).
    CaliperTextureId tex = need_create ? 0 : v->tex;
    bool cpu_uploaded = false;
    if (need_create) {
        tex = mapped ? bridge->texture_from_tensor_mapped(&t, cm, vmin, vmax, 0)
                     : bridge->texture_from_tensor(&t, 0);
        if (!tex) {
            drop_fresh_import();
            return vfail("texture creation failed" + bridge_reason());
        }
        cpu_uploaded = !metal;
    } else if (mapped && (v->colormap != cm || v->vmin != vmin || v->vmax != vmax)) {
        // Mapping change on a live view: re-pin, keep the texture.
        if (!bridge->set_mapping(tex, cm, vmin, vmax)) {
            drop_fresh_import();
            return vfail("mapping change refused" + bridge_reason());
        }
        v->colormap = cm; v->vmin = vmin; v->vmax = vmax;   // pinned now
    }

    // ---- 3. the update ------------------------------------------------------
    bool ok = true;
    bool imported = false;
    std::string path = "cpu-staged";
    if (metal) {
        ok = bridge->update_texture_from_alloc(tex, alloc, desc->byte_offset, &t);
        if (ok) {
            imported = true;
            const char* p = core->renderer->last_device_path();
            path = p ? p : "";
            if (desc->flags & CALIPER_VIEW_FLAG_WAIT_GPU)
                core->renderer->device_update_wait();
        }
    } else if (!cpu_uploaded) {
        ok = bridge->update_texture(tex, &t);
    }
    if (!ok) {
        if (need_create) bridge->release_texture(tex);
        drop_fresh_import();
        return vfail(metal
            ? "device update from the imported MTLBuffer was refused (METAL "
              "tensors never fall back to a host copy)" + bridge_reason()
            : "CPU-staged upload refused" + bridge_reason());
    }

    // ---- 4. commit -----------------------------------------------------------
    if (!v) {
        core->views.emplace_back();
        v = &core->views.back();
        v->name = name;
    } else if (need_create && v->tex && v->tex != tex) {
        bridge->release_texture(v->tex);
    }
    if (metal) {
        if (fresh_import) {
            if (v->alloc) bridge->release_allocation(v->alloc);   // stale import
            v->alloc = alloc;
            v->alloc_buf = src.data;
            v->alloc_bytes = window;
        }
    } else if (v->alloc) {
        // A CPU update: stop pinning the previous MTLBuffer.
        bridge->release_allocation(v->alloc);
        v->alloc = 0; v->alloc_buf = nullptr; v->alloc_bytes = 0;
    }
    if (need_create) ++v->creates;
    v->tex = tex;
    v->mapped = mapped;
    v->w = (int)W;
    v->h = (int)H;
    v->colormap = cm;
    v->vmin = vmin;
    v->vmax = vmax;
    v->flags = desc->flags;
    v->device = src.device;
    v->imported = imported;
    v->path = path;
    ++v->updates;
    v->last_update_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
    core->last_error.clear();
    return 1;
}

int caliper_core_view_remove(CaliperCore* core, const char* name) {
    if (!core || !name) return 0;
    for (auto it = core->views.begin(); it != core->views.end(); ++it) {
        if (it->name == name) {
            release_view(services_bridge(), *it);
            core->views.erase(it);
            return 1;
        }
    }
    core->fail(std::string("view_remove: no view named '") + name + "'");
    return 0;
}

int caliper_core_view_count(CaliperCore* core) {
    return core ? (int)core->views.size() : 0;
}

int caliper_core_view_info(CaliperCore* core, const char* name, CaliperViewInfo* out) {
    if (!core) return 0;
    if (!name || !out || out->struct_size < sizeof(CaliperViewInfo)) {
        core->fail("view_info: null name or null/short CaliperViewInfo");
        return 0;
    }
    const TensorView* v = find_view(core, name);
    if (!v) {
        core->fail(std::string("view_info: no view named '") + name + "'");
        return 0;
    }
    out->path = v->path.c_str();
    out->width = v->w;
    out->height = v->h;
    out->device = v->device;
    out->updates = v->updates;
    out->last_update_ms = v->last_update_ms;
    out->imported = v->imported ? 1 : 0;
    out->creates = v->creates;
    return 1;
}

int caliper_core_view_readback(CaliperCore* core, const char* name, void* buf,
                               size_t buf_bytes) {
    if (!core) return 0;
    if (!name || !buf) {
        core->fail("view_readback: null name/buffer");
        return 0;
    }
    const TensorView* v = find_view(core, name);
    if (!v) {
        core->fail(std::string("view_readback: no view named '") + name + "'");
        return 0;
    }
    const size_t need = (size_t)v->w * (size_t)v->h * 4;
    if (buf_bytes < need) {
        core->fail("view_readback: buffer too small (need " + std::to_string(need) +
                   " bytes)");
        return 0;
    }
    // Same-queue readback: commit order retires every prior update first.
    std::vector<uint8_t> px = core->renderer->debug_readback_rgba8(v->tex, v->w, v->h);
    if (px.size() != need) {
        core->fail("view_readback: renderer readback failed");
        return 0;
    }
    std::memcpy(buf, px.data(), need);
    return 1;
}

int caliper_core_colormap_lut(int32_t colormap, uint32_t out256[256]) {
    const uint32_t* lut = colormap_lut(colormap);
    if (!lut || !out256) return 0;
    std::memcpy(out256, lut, 256 * sizeof(uint32_t));
    return 1;
}
