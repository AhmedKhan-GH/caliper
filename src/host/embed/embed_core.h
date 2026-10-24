// embed_core.h — the CaliperCore instance struct behind the embed C ABI
// (caliper/embed.h), shared between embed_core.cpp (lifecycle, frame, v1.2
// tensor views) and src/host/live/ (the live-streaming attach: typed views fed
// from a training process's shared-memory snapshots). Host-internal: nothing
// here crosses the C seam.
#pragma once

#include "caliper/embed.h"

#include <cstdio>
#include <memory>
#include <string>
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
    std::unique_ptr<caliper_host::AppletLoader> loader;
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

    void log(int level, const std::string& msg) {
        if (log_fn) log_fn(userdata, level, msg.c_str());
        else std::fprintf(stderr, "[embed] %s\n", msg.c_str());
    }
    void fail(const std::string& why) { last_error = why; log(2, why); }
};

// Live-module hooks called by embed_core.cpp (defined in src/host/live/
// live_core.cpp). frame: draw the live typed views into the current ImGui
// frame, after the v1.2 views. shutdown: release every live GPU resource while
// the bridge still has its renderer.
void caliper_live_frame(CaliperCore* core);
void caliper_live_shutdown(CaliperCore* core);
// Typed-views hooks (src/host/live/live_views.cpp): draw the points/series/
// histogram windows after the v1.2 views; release their GPU resources before
// the v1.2 views are destroyed.
void caliper_live_views_frame(CaliperCore* core);
void caliper_live_views_shutdown(CaliperCore* core);
