// embed_core.h — the CaliperCore instance struct behind the embed C ABI
// (caliper/embed.h), shared between embed_core.cpp (lifecycle, frame, v1.2
// tensor views) and src/host/live/ (the live-streaming attach: typed views fed
// from a training process's shared-memory snapshots). Host-internal: nothing
// here crosses the C seam.
#pragma once

#include "caliper/embed.h"

#include <atomic>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "applet_loader.h"
#include "frame_watchdog.h"
#include "renderer/host_renderer.h"
#include "tensor_bridge.h"

// v1.2 built-in tensor views (embed.h "v1.2 — built-in tensor views"). One
// record per named view; CaliperCore keeps them in a vector so draw order is
// creation order. Every GPU resource here (the bridge texture, the MTLBuffer
// import) is released through the process TensorBridge — by view_remove, by a
// recreate, or by caliper_core_shutdown BEFORE the renderer is unbound.
namespace caliper_embed_views {
struct TensorView {
    std::string                    name;
    CaliperTextureId tex = 0;          // the bridge's public id == ImTextureID
    bool             mapped = false;   // 2-D f32 colormapped vs (H,W,4) u8
    int              w = 0, h = 0;     // texels
    int32_t          colormap = 0;     // pinned mapping (mapped views)
    float            vmin = 0.0f, vmax = 1.0f;
    uint32_t         flags = 0;        // CALIPER_VIEW_FLAG_* of the last update

    // METAL import cache: the MTLBuffer this view last read in place (strong-
    // retained by the renderer while alloc != 0). Re-imported when the buffer
    // pointer changes or the addressed window outgrows alloc_bytes.
    CaliperAllocId   alloc = 0;
    void*            alloc_buf = nullptr;
    uint64_t         alloc_bytes = 0;

    // Telemetry surfaced through caliper_core_view_info.
    std::string      path;
    int32_t          device = CALIPER_DEV_CPU;
    bool             imported = false;
    uint64_t         updates = 0;
    uint64_t         creates = 0;
    double           last_update_ms = 0.0;

    // Windowed layout: computed on the first draw (needs ImGui metrics), then
    // handed to ImGui as FirstUseEver so the user may move/resize freely.
    bool             placed = false;
    float            pos_x = 0.0f, pos_y = 0.0f, size_x = 0.0f, size_y = 0.0f;

    // The typed live view of the same name draws the image itself (its pan
    // and zoom, src/host/live/live_views_draw.cpp); the window stays ours.
    bool             drawn_by_live = false;
};
}  // namespace caliper_embed_views

// Opaque per-core state of the live-streaming attach (src/host/live). Owned by
// the live module: created by caliper_core_live_attach, destroyed by detach or
// by caliper_live_shutdown (called from caliper_core_shutdown).
struct CaliperLiveState;
// Opaque per-core state of the typed live views (caliper/live_views.h,
// src/host/live/live_views.cpp): points/series/histogram views and the typed
// bookkeeping over the v1.2 heatmap/image views. Created lazily by
// caliper_core_live_apply, destroyed by caliper_live_views_shutdown.
struct CaliperLiveViewsState;

// The core is a plain struct (opaque to the ABI). It holds ONLY instance state
// the ABI forces; the service registry, crash guard, colormap LUTs, ImGui
// context, and GLFW-analog remain process-global exactly as L1 left them (see
// l1-survey §7 / the report). That is why v0 is one-core-per-process.
struct CaliperCore {
    std::unique_ptr<caliper_host::HostRenderer> renderer;
    // The deleter is a function pointer set where the loader is made
    // (embed_core.cpp), so destroying a core without one (a hosted core, the
    // browser's views) references nothing of the applet loader.
    std::unique_ptr<caliper_host::AppletLoader, void (*)(caliper_host::AppletLoader*)> loader{
        nullptr, nullptr};
    caliper_host::FrameWatchdog                 watchdog;

    CaliperLogFn   log_fn = nullptr;
    CaliperCrashFn crash_fn = nullptr;
    void*          userdata = nullptr;

    bool                                   canvas_ready = false;
    caliper_host::HostRenderer::CanvasMode canvas_mode =
        caliper_host::HostRenderer::CANVAS_OFFSCREEN;
    int                                    canvas_w = 0, canvas_h = 0;
    float                                  scale = 1.0f;

    int    active_applet = -1;
    double last_frame_time = 0.0;
    std::string last_error;

    // "<data_dir>/imgui.ini" when the embedder gave a data_dir; ImGuiIO's
    // IniFilename points at its c_str() from create until shutdown nulls it
    // just before the ImGui context is destroyed. Empty = no persistence.
    std::string ini_path;

    // v1.2 views, in creation (= draw) order, plus the windowed-layout tiling
    // cursor (logical points) for first placement.
    std::vector<caliper_embed_views::TensorView> views;
    float tile_x = 8.0f, tile_y = 8.0f, tile_row_h = 0.0f;

    // Live-streaming attach state (src/host/live); null when not attached.
    CaliperLiveState* live = nullptr;
    // Typed live views state (caliper/live_views.h); null until the first apply.
    CaliperLiveViewsState* live_views = nullptr;

    // A hosted core (caliper_core_create_hosted, below): `renderer` is the
    // host's, borrowed (released, never deleted, at destroy); there is no
    // loader; the host draws the chrome. Views tile into the host's work area,
    // whose top-left (logical points) is (area_x, area_y); 0 for every other core.
    bool  hosted = false;
    float area_x = 0.0f, area_y = 0.0f;
    // A host that draws each view's name, status and data view picker in its
    // own header (the browser element, web/wasm) sets this: view windows then
    // show the view alone, with no title bar, no diagnostics line (size,
    // path, seq, latency, updates, drops) and no Pause or Fit button; hover
    // values and the plot stay. Every other core keeps all of them.
    bool  views_bare = false;
    // The views' diagnostics under each header (caliper_live_views::views_debug):
    // also on with CALIPER_VIEWS_DEBUG=1.
    bool  views_debug = false;

    // Copy counters per view name and per frame (embed/render_stats.h, which
    // writes and reads them; the core-wide total is the renderer's).
    struct CopyStats {
        struct View {
            caliper_host::HostRenderer::CopyCount total, last;
        };
        std::unordered_map<std::string, View> views;
        caliper_host::HostRenderer::CopyCount at_frame_end, last_frame, window_start;
        uint64_t frames = 0, window_frames = 0;
    } copy_stats;

    void log(int level, const std::string& msg) {
        if (log_fn) log_fn(userdata, level, msg.c_str());
        else std::fprintf(stderr, "[embed] %s\n", msg.c_str());
    }
    void fail(const std::string& why) { last_error = why; log(2, why); }
};

// The one-core-per-process slot (embed_views.cpp). caliper_core_create and
// caliper_core_create_hosted claim it with a compare-exchange from nullptr to
// the reserved sentinel before building anything, publish the built core, and
// release it to nullptr on failure or destroy.
std::atomic<CaliperCore*>& caliper_core_slot();
inline CaliperCore* caliper_core_slot_reserved() { return reinterpret_cast<CaliperCore*>(1); }

// v1.2 view hooks (embed_views.cpp) for embed_core.cpp: draw every v1.2 view
// into the current ImGui frame; release every v1.2 view through the bridge.
void caliper_embed_views_draw(CaliperCore* core);
void caliper_embed_views_destroy_all(CaliperCore* core);

// Live-module hooks called by embed_core.cpp (defined in src/host/live/
// live_core.cpp). frame: draw the live typed views into the current ImGui
// frame, after the v1.2 views. shutdown: release every live GPU resource while
// the bridge still has its renderer.
void caliper_live_frame(CaliperCore* core);
void caliper_live_shutdown(CaliperCore* core);
// The attached stream's wire protocol 3.1 state (live_controls.h) and its
// source, for a host's live page: null unless the core is attached to a
// network stream (caliper_core_live_attach_url). Frame-thread only.
namespace caliper_live { class Source; class RunControls; }
caliper_live::RunControls* caliper_live_controls(CaliperCore* core);
caliper_live::Source* caliper_live_stream(CaliperCore* core);
// The source's key of the attached view `name` (Source::send_event,
// request_data_view, data_view_of); 0 when there is none.
uint64_t caliper_live_view_key(CaliperCore* core, const char* name);
// The host's tap on each pick and selection that a person makes in a scene
// view of the attached stream (live_core.cpp), which the app gives to the
// agent as an event without a panel (design/instruments.md, Event Feeds). A
// hover never reaches it. An empty tap removes it; the core's shutdown removes
// it too. Frame-thread only.
namespace caliper_live_views { struct LiveEvent; }
void caliper_live_set_event_tap(CaliperCore* core, std::function<void(const caliper_live_views::LiveEvent&)> tap);
// Typed-views hooks (src/host/live/live_views.cpp): draw the points/series/
// histogram windows after the v1.2 views; release their GPU resources before
// the v1.2 views are destroyed.
void caliper_live_views_frame(CaliperCore* core);
void caliper_live_views_shutdown(CaliperCore* core);

// View-window grid and diagnostics hooks (defined in src/host/live/live_views.cpp),
// declared here so embed_core.cpp and embed_views.cpp do not include live headers.
namespace caliper_live_views {
bool views_debug(const CaliperCore* core);
void layout_views(CaliperCore* core);
bool window_before(CaliperCore* core, const std::string& name);
void window_after(CaliperCore* core, const std::string& name);
int  window_flags();
}  // namespace caliper_live_views

// ---- Hosted core (host-internal; the Caliper app's live page) ---------------
// For a host that already owns the window, the renderer (bound to the tensor
// bridge), the ImGui contexts and the services: the core borrows `renderer`
// and has no loader and no canvas of its own; its views draw inside the
// host's ImGui frame, and the caliper_core_live_* / view calls work on it as
// on any core. It holds the one-core-per-process slot like any core: null when
// the slot is taken or `renderer` is null. Frame-thread only.
CaliperCore* caliper_core_create_hosted(caliper_host::HostRenderer* renderer);
// Inside the host's ImGui frame: the v1.2 views, the typed live views and the
// live bookkeeping, tiled into the work area (x, y, w, h in logical points;
// `scale` physical px per point). No chrome.
void caliper_core_hosted_frame(CaliperCore* core, float x, float y, float w, float h,
                               float scale);
// Releases every view's GPU resources while the renderer is still bound (after
// the host rendered the last frame that drew them), gives the renderer back
// and frees the slot. Never touches the host's contexts or services.
void caliper_core_destroy_hosted(CaliperCore* core);
