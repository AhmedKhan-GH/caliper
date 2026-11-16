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
#include "live/live_source.h"   // PayloadMap

#include <cstdint>
#include <functional>
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
    // A CPU payload the renderer copied with import_host_allocation: our own
    // copy of the bytes, the CPU view hover reads.
    std::vector<uint8_t> host;
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

// What a person did in a view, for the host's event sink (set_event_sink):
// `type` is "select", "pick" or "hover" and `detail` a JSON object in source
// indices, exactly as wire protocol v3 ("Events") and the browser element's
// own views report it. A select with detail "{}" clears the selection.
struct LiveEvent {
    std::string view;
    const char* type = "select";
    std::string detail;
};
using EventSink = std::function<void(const LiveEvent&)>;

// The pointer gesture on a view while a host listens for events: the browser
// element's rule (web/src/views/interact.ts). A shift-drag selects; a drag
// without shift pans or orbits; a press and release that moved at most
// kClickSlop pixels is a click (a pick), and with shift it also clears the
// selection.
struct Gesture {
    bool down = false, moved = false, selecting = false;
    bool shift = false, alt = false, ctrl = false;
    int button = 0;                  // ImGuiMouseButton of the press
    float x0 = 0.0f, y0 = 0.0f;      // press position (screen)
    float lx = 0.0f, ly = 0.0f;      // the pointer when the drag last moved the view
    double d0x = 0.0, d0y = 0.0, d1x = 0.0, d1y = 0.0;   // drag ends in data coordinates
    std::vector<float> lasso;        // points: the lasso (x, y pairs, screen)
    bool rect = false;               // points: shift + alt selects a rectangle
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
    bool        latency_relative = false;   // CALIPER_LIVE_APPLY_STAMP_RELATIVE
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
    // CPU payloads: a copy of the last one's bytes (the caller may free its own
    // on return), so hover shows values for them too; tex_cpu points here.
    std::vector<uint8_t> tex_copy;

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

    // ---- events (only while the host has an event sink) ----------------------
    caliper_live::PayloadMap map;     // of the last applied snapshot (set_payload_map)
    Gesture gesture;
    double  last_hover_ms = -1e9;     // hovers reach the sink at most 30 times a second
    int64_t hover_sent = -2;          // the last hovered item reported (-1: none)
    // HEATMAP / IMAGE: the selected payload cells [r0, r1) x [c0, c1), r1 == 0: none.
    int64_t sel_r0 = 0, sel_r1 = 0, sel_c0 = 0, sel_c1 = 0;
    int64_t sel_rows = 0, sel_cols = 0;   // the payload size it was made on
    // SERIES / HISTOGRAM: the selected steps [s0, s1) or bins [b0, b1), s1 == s0: none.
    int64_t sel_s0 = 0, sel_s1 = 0;
    // HEATMAP / IMAGE pan and zoom: the cell at the centre of the area (cx, cy)
    // and the screen units per cell across and down (kx, ky). fitted: follow
    // the fit to the area.
    bool    zoom_fitted = true;
    double  zoom_cx = 0.0, zoom_cy = 0.0, zoom_kx = 1.0, zoom_ky = 1.0;
    int     zoom_w = 0, zoom_h = 0;       // the payload size the zoom was made on
    // SERIES / HISTOGRAM: follow (the axes fit the data every frame) or keep
    // the user's pan and zoom, and a log scale for y. fit_once: fit next frame.
    bool    follow = true, log_y = false, fit_once = false;
    bool    log_fit_now = false;          // fit a log y axis to the positive values this frame
    bool    pan_pending = false;          // a drag that stopped following: pan by it next frame
    float   pan_dx = 0.0f, pan_dy = 0.0f;
    bool    zoom_limits = false;          // set these limits (x0, x1, y0, y1) this frame
    double  zoom_lim[4] = {};
    float   zoom_rate_saved = 0.0f;       // ImPlot's own wheel zoom, off while the view zooms
    // Linked selection (wire protocol 3.1): the items of an item space that the
    // host says one axis of this view carries, sorted. Points are drawn over
    // the dimmed rest, heatmap rows or columns are marked, series steps get
    // markers. link_axis: 0 none, 1 points, 2 rows, 3 columns, 4 steps.
    int     link_axis = 0;
    std::vector<int64_t> link_items;
    uint64_t link_version = 0, link_for_version = 0, link_for_updates = 0;
    // The item last hovered, for tests: a point's index, a series step, a
    // histogram bin, or a cell row * columns + column; -1 for none.
    int64_t hover_item = -1;
    // POINTS: one flag per point of the current sample (empty: no selection),
    // and the selected points' copy that is drawn over the dimmed rest.
    std::vector<uint8_t> selected;
    uint64_t    sel_count = 0;
    uint64_t    sel_version = 0;          // grows with every change of the selection
    PointsPayload sel_draw;               // the selected points' positions and aux
    bool        have_sel_draw = false;
    uint64_t    sel_draw_for = 0, sel_draw_updates = 0;   // sel_version and updates it was built for

    // Where the view drew its data last frame, for tests and the page host:
    // area in screen units (x, y, w, h) and the data window it shows
    // (x0, x1, y0, y1: columns and rows for a heatmap, rows growing down).
    bool   geo_valid = false;
    float  geo_area[4] = {};
    double geo_window[4] = {};
    bool   geo_ydown = false;
};

}  // namespace caliper_live_views

struct CaliperLiveViewsState {
    std::vector<std::unique_ptr<caliper_live_views::View>> views;  // creation order
    bool paused = false;
    // Set by a host that sends what people do in the views to a producer (the
    // browser's NetSource). Without it the views neither select nor report.
    caliper_live_views::EventSink events;
};

namespace caliper_live_views {

// live_views.cpp
View* find(CaliperCore* core, const char* name);
// The host's event sink (empty: no selections, no events). Creates the state.
void set_event_sink(CaliperCore* core, EventSink sink);
// A header toggle of view `name`, set by the page: "follow" (series) or
// "log" (series, histogram). False: no such view or toggle. The views report
// their own changes of a toggle to the sink as type "toggle" with detail
// {"name": ..., "on": ...}, which a host keeps from the producer.
bool set_toggle(CaliperCore* core, const char* name, const char* toggle, bool on);
// The linked selection of view `name`: the items (any order) of the item
// space that its axis "points", "rows", "columns" or "steps" carries, or no
// axis ("" or null) to clear it. False: no such view.
bool set_link(CaliperCore* core, const char* name, const char* axis, std::vector<int64_t> items);
// How the last applied snapshot of view `name` maps onto its source; the
// host calls it right after caliper_core_live_apply. False: no such view.
bool set_payload_map(CaliperCore* core, const char* name, const caliper_live::PayloadMap& map);
// A copy of `len` host bytes as a renderer allocation of view v, as a CPU
// payload is staged (released with drop_import).
bool stage_host(CaliperCore* core, View& v, const void* bytes, uint64_t len, Import** out,
                const uint8_t** cpu, std::string* why);
void drop_payload(CaliperCore* core, View& v, PointsPayload& p);
void  poll_handles(CaliperCore* core, View& v);   // retire done handles (non-blocking)
void  fit_points(View& v, float aspect);          // camera from bbox (stats or CPU scan)

// live_views_draw.cpp
void draw_view(CaliperCore* core, CaliperLiveViewsState* S, View& v);
// Column-major matrices for the current camera (also used by hover).
void camera_matrices(const View& v, float aspect, float view16[16], float proj16[16]);
// The fixed 16-colour categorical palette (r | g<<8 | b<<16 | a<<24).
extern const uint32_t kPalette[16];
// The side in physical pixels of the square a points view draws per point.
float point_size_px(uint64_t n, float scale);

}  // namespace caliper_live_views
