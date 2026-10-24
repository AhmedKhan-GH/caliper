// live_views_state.h: the per-core state of the typed live views
// (caliper/live_views.h), shared by live_views.cpp (C ABI, apply, lifetime),
// live_views_draw.cpp (ImGui/ImPlot windows, camera, hover) and
// live_views_metal.mm (the few MTLBuffer queries a .cpp cannot make).
// Host-internal: nothing here crosses the C seam.
//
// GPU resources of a POINTS view live in the RENDERER's tables directly
// (HostRenderer::geom_create_view_ex / import_external_allocation /
// geom_draw_primitives), not in the TensorBridge's: the bridge's point draw
// only takes a built-in colormap id, and a categorical palette needs its own
// 256-entry LUT, which the renderer seam accepts. Every such resource is
// released through the renderer by view removal or caliper_live_views_shutdown,
// which embed_core.cpp calls while the renderer is still alive. HEATMAP and
// IMAGE views are embed v1.2 TensorViews (core->views, same name); this file
// only adds the typed bookkeeping around them.
#pragma once

#include "caliper/live_views.h"
#include "embed/embed_core.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caliper_live_views {

// ---- platform helpers (live_views_metal.mm; stubs elsewhere) --------------
struct BufferInfo {
    bool     ok = false;
    uint64_t length = 0;          // [buffer length]
    void*    contents = nullptr;  // [buffer contents]; NULL for private storage
};
BufferInfo metal_buffer_info(void* mtl_buffer);
// +1 shared-storage MTLBuffer holding a copy of `bytes` (CPU-staged payloads).
void* metal_new_shared_buffer(void* mtl_device, const void* bytes, uint64_t len);
void  metal_release(void* obj);   // balances metal_new_shared_buffer; NULL-safe

uint64_t now_ns();   // CLOCK_UPTIME_RAW ns on Apple (== Python time.monotonic_ns)

// ---- points --------------------------------------------------------------
// One renderer import of an MTLBuffer, shared by every payload of the view
// that lives in the same buffer (a shm segment holds all slots of a view in one
// buffer; the window transport may use one buffer per slot). Refcounted by the
// payloads that use it; released when the last one retires.
struct Import {
    void*    buf = nullptr;       // id<MTLBuffer> (cache key; kept alive by the import)
    uint64_t rid = 0;             // renderer import id
    uint64_t bytes = 0;           // imported length (the whole buffer)
    void*    owned = nullptr;     // +1 buffer we made for a CPU payload (released after rid)
    int      refs = 0;
};

struct PointsPayload {
    Import*        pos = nullptr;
    uint64_t       pos_off = 0;
    Import*        aux = nullptr;
    uint64_t       aux_off = 0;
    const uint8_t* pos_cpu = nullptr;   // CPU view of the positions (NULL: no hover)
    const uint8_t* aux_cpu = nullptr;   // CPU view of aux (NULL: hover shows no value)
    uint64_t       n = 0;
    void*          last_draw = nullptr; // +1 handle on the last frame that drew it
};

struct Camera {
    // 3-D orbit around (tx,ty,tz); 2-D: centre (tx,ty), half height half_h.
    float tx = 0.0f, ty = 0.0f, tz = 0.0f;
    float dist = 3.0f, az = 0.785398f, el = 0.5f;
    float radius = 1.0f;
    float half_h = 1.0f;
};

struct View {
    std::string name;
    int32_t     kind = 0;

    // Telemetry (caliper_core_live_view_info).
    uint64_t    updates = 0, creates = 0, dropped = 0, seq = 0, stamp_ns = 0;
    double      last_apply_ms = 0.0, latency_ms = 0.0;
    bool        imported = false;
    std::string path;
    int64_t     count = 0;

    // HEATMAP / IMAGE: the v1.2 view of the same name is ours (created or
    // adopted by an apply), and the +1 completion handles of its device
    // updates that have not retired yet (polled, never waited on).
    bool               owns_v12 = false;
    std::vector<void*> pending;
    float              vmin = 0.0f, vmax = 1.0f;
    bool               have_range = false;    // last finite AUTO range
    void*              tex_buf = nullptr;     // MTLBuffer of the last METAL payload
    const uint8_t*     tex_cpu = nullptr;     // its CPU view at data_offset (hover)

    // POINTS.
    uint64_t    view_rid = 0;                 // renderer id of the render target
    int         tex_w = 0, tex_h = 0;
    std::vector<std::unique_ptr<Import>> imports;
    PointsPayload cur;
    bool        have_cur = false;
    std::vector<PointsPayload> retired;       // superseded, last frame still in flight
    int32_t     dims = 3, label_count = 0, colormap = 0;
    float       pvmin = 0.0f, pvmax = 1.0f;
    uint32_t    lut[256] = {};
    bool        has_bbox = false;
    float       bbox[6] = {};                 // xmin,xmax,ymin,ymax,zmin,zmax (stats)
    bool        fit_pending = true;
    Camera      cam;
    bool        dirty = true;                 // redraw the render target this frame
    // hover cache
    float       hover_mx = -1e30f, hover_my = -1e30f;
    int64_t     hover_idx = -1;
    float       hover_sx = 0.0f, hover_sy = 0.0f;
    bool        hover_dirty = true;
    uint64_t    hover_stride = 1;             // >1: hover searched a sample
    bool        draw_failed = false;          // last render-target draw was refused

    // SERIES / HISTOGRAM.
    std::vector<float>  values;
    uint64_t            first = 0;            // series: global index of values[0]
    float               lo = 0.0f, hi = 1.0f; // histogram range
    std::vector<double> xs, ys;               // draw cache (series decimation / bar centres)
    bool                draw_dirty = true;

    // Windowed layout (first placement tiles like the v1.2 views).
    bool  placed = false;
    float pos_x = 0.0f, pos_y = 0.0f, size_x = 0.0f, size_y = 0.0f;
};

}  // namespace caliper_live_views

struct CaliperLiveViewsState {
    std::vector<std::unique_ptr<caliper_live_views::View>> views;  // creation order
    bool paused = false;
};

namespace caliper_live_views {

// live_views.cpp
View* find(CaliperCore* core, const char* name);
void  poll_handles(CaliperCore* core, View& v);   // retire done handles (non-blocking)
void  fit_points(View& v, float aspect);          // camera from bbox (stats or CPU scan)

// live_views_draw.cpp
void draw_view(CaliperCore* core, CaliperLiveViewsState* S, View& v);
// Column-major matrices for the current camera (also used by hover).
void camera_matrices(const View& v, float aspect, float view16[16], float proj16[16]);
// The fixed 16-colour categorical palette (r | g<<8 | b<<16 | a<<24).
extern const uint32_t kPalette[16];

}  // namespace caliper_live_views
