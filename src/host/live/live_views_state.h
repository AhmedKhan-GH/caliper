// live_views_state.h: the per-core state of the typed live views
// (caliper/live_views.h), shared by live_views.cpp (C ABI, apply, lifetime),
// live_views_draw.cpp (ImGui/ImPlot windows, camera, hover) and
// live_views_metal.mm (the MTLBuffer query a .cpp cannot make).
// Host-internal: nothing here crosses the C seam.
//
// A POINTS view draws through the scene core and the graphics layer
// (live_points.h): its payload buffers are imports of the layer, each made by one
// import_buffer call whose route sets the view's path label, and every such
// resource goes back to the layer by view removal or caliper_live_views_shutdown,
// which embed_core.cpp calls while the renderer is still alive. HEATMAP and
// IMAGE views are embed v1.2 TensorViews (core->views, same name); this file
// only adds the typed bookkeeping around them.
#pragma once

#include "caliper/live_views.h"
#include "embed/embed_core.h"
#include "gfx/gfx.h"
#include "live/live_source.h"   // PayloadMap
#include "scene/camera_defaults.h"   // the 3-D views' default camera
#include "scene/scene.h"             // kPalette

#include <imgui.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace caliper_live_views {

// The scene view (live_views_scene.h): kind 6 of the view registry, CALIPER_NET_KIND_SCENE of
// wire protocol 3.2, fed with decoded scene messages instead of snapshots.
constexpr int32_t kKindScene = 6;
struct SceneHolder;

// What a scene view keeps of itself across a close and a reopen, by its name (live_scene.h,
// SceneView::Settings): the up mode the person chose and the scale bar option.
struct SceneSettings {
    bool up_chosen = false;
    uint8_t up = 0;            // caliper_scene::Axis
    bool scale_bar = true;
};

// ---- platform helpers (live_views_metal.mm; stubs elsewhere) --------------
struct BufferInfo {
    bool     ok = false;
    uint64_t length = 0;          // [buffer length]
    void*    contents = nullptr;  // [buffer contents]; NULL for private storage
};
BufferInfo metal_buffer_info(void* mtl_buffer);

uint64_t now_ns();   // CLOCK_UPTIME_RAW ns on Apple (== Python time.monotonic_ns)

// ---- the look of a view window's header ---------------------------------------
// A host sets it (set_look) to draw the headers in its own type and colours;
// the defaults read as ImGui's own. `mono` (may be null: the current font) at
// `mono_size` writes the info string, the current font at `small_size` the
// header's tool buttons.
struct Look {
    ImFont* mono = nullptr;
    float   mono_size = 0.0f;               // 0: the current size
    float   small_size = 0.0f;              // 0: the current size
    ImU32   text = IM_COL32(230, 230, 235, 255);
    ImU32   faint = IM_COL32(140, 140, 150, 255);
    ImU32   rule = IM_COL32(90, 90, 100, 255);
    ImU32   accent = IM_COL32(110, 170, 255, 255);   // a pressed tool, the live dot
    ImU32   plot = 0;                                // a series' line and a histogram's bars (0: ImPlot's)
};

// ---- points --------------------------------------------------------------
// One import of the graphics layer. An MTLBuffer is imported in place and shared
// by every payload of the view that lives in the same buffer up to the end of
// its window (a shm segment holds all slots of a view in one buffer; the window
// transport may use one buffer per slot); host memory is copied. Refcounted by
// the payloads that use it; released to the layer when the last one retires,
// and the layer frees it once the last frame that read it completed.
struct Import {
    void*    key = nullptr;       // the MTLBuffer of an in-place import (cache key); null for a copy
    caliper_gfx::Buffer buffer;   // the layer's buffer
    uint64_t bytes = 0;           // imported length, from byte 0 of the handle
    caliper_gfx::Route route = caliper_gfx::Route::Refused;
    const uint8_t* cpu = nullptr; // its CPU view at byte 0 (the layer's mapping), or null
    int      refs = 0;
    // A copy the layer maps nothing for (a browser): our own copy of the bytes, the CPU view hover reads.
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
    caliper_gfx::Serial last_draw = 0;  // the layer's submit of the last frame that drew it (0: none)
};

class PointsScene;
// What a points view draws through: the layer's backend, owned or the renderer's, and the scene.
// Without a backend the view shows `notice` instead.
struct PointsHolder {
    std::unique_ptr<caliper_gfx::Backend> gfx;   // owned; null when the view borrows the renderer's
    std::unique_ptr<PointsScene> scene;
    std::string notice;
    ~PointsHolder();
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

// The 3-D points camera's default, the same in every renderer
// (web/src/views/camera.ts, CAMERA) and in the scene view's orbit, from the
// one definition in scene/camera_defaults.h: a three-quarter view from
// azimuth kCamAz and elevation kCamEl (radians; the eye at target + dist *
// (cos el cos az, sin el, cos el sin az)), at the distance that fits the
// data's bounding sphere into the narrower field of view with kCamFit to
// spare, for the box's labels. The points view's camera is the scene core's
// orbit (live_points.h), which a fit returns to this view. The rotate tool
// turns the azimuth at kRotateRate radians a second.
constexpr float kCamAz = (float)caliper_scene::kDefaultAzimuth;
constexpr float kCamEl = (float)caliper_scene::kDefaultElevation;
constexpr float kCamFit = (float)caliper_scene::kDefaultFitMargin;
static_assert(kCamAz == 0.95f && kCamEl == 0.42f && kCamFit == 1.15f, "the points view's camera as before");
constexpr float kRotateRate = 0.35f;

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
    std::shared_ptr<PointsHolder> points;     // its scene and the layer it draws through
    bool        content_dirty = true;         // the scene's batches need setting again
    uint64_t    content_sel = 0, content_updates = 0;   // the selection and update the batches show
    uint32_t    ground = 0;                   // the plot's clear colour the batches were set with
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
    // 3-D: turning slowly about the up axis (the header's rotate tool, off
    // until pressed; any drag stops it), and the depth cues (nearer points
    // larger, farther points dimmer; on unless set_toggle "depth" turns them
    // off). The data's box, for the frame around the points, per update, and
    // the box's tick labels last frame (tests).
    bool        auto_rotate = false;
    bool        depth_cues = true;
    float       box[6] = {};
    bool        have_box = false;
    uint64_t    box_for = ~0ull;
    std::string box_labels;
    bool        dirty = true;                 // redraw the render target this frame
    // hover cache
    float       hover_mx = -1e30f, hover_my = -1e30f;
    int64_t     hover_idx = -1;
    float       hover_sx = 0.0f, hover_sy = 0.0f;
    bool        hover_dirty = true;
    uint64_t    hover_stride = 1;             // >1: hover searched a sample
    bool        draw_failed = false;          // last render-target draw was refused
    // The CPU time of the view's last redraw of its render target (encode and submit), and how
    // many redraws it made: the step's timing row and the benchmarks read them.
    double      draw_cpu_ms = 0.0;
    uint64_t    redraws = 0;

    // SCENE: its scene view and the layer it draws through (live_views_scene.h).
    std::shared_ptr<SceneHolder> scene;

    // SERIES / HISTOGRAM.
    std::vector<float>  values;
    uint64_t            first = 0;            // series: global index of values[0]
    float               lo = 0.0f, hi = 1.0f; // histogram range
    std::vector<double> xs, ys;               // draw cache (series decimation / bar centres)
    bool                draw_dirty = true;

    // Windowed layout (first placement tiles like the v1.2 views).
    bool  placed = false;
    float pos_x = 0.0f, pos_y = 0.0f, size_x = 0.0f, size_y = 0.0f;
    // The grid (layout_views): this view's cell, the layout it was last set
    // to, and whether the person moved, resized or docked its window since,
    // which takes it off the grid for good.
    float    cell[4] = {};                 // x, y, w, h in screen units
    uint64_t cell_gen = 0, applied_gen = 0;
    float    applied[4] = {};
    bool     applied_now = false;
    bool     off_grid = false;

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
    std::string geo_text;   // more of it in words (a scene's camera), for the UI driver's `where`
};

}  // namespace caliper_live_views

struct CaliperLiveViewsState {
    std::vector<std::unique_ptr<caliper_live_views::View>> views;  // creation order
    bool paused = false;
    // Set by a host that sends what people do in the views to a producer (the
    // browser's NetSource). Without it the views neither select nor report.
    caliper_live_views::EventSink events;
    // The headers' look (set_look), and the stream's state for each header's
    // dot (set_stream_state): 0 no stream known (no dot), 1 live, 2 ended.
    caliper_live_views::Look look;
    int stream = 0;
    // The grid of the view windows (layout_views): the layout's signature
    // (area x, y, w, h and the number of views) and its generation.
    float grid_sig[5] = {};
    uint64_t grid_gen = 0;
    // What a host draws in every view window under the header (a stream's
    // data view picker, live_controls.h): called with the view and the
    // width it may take, at the cursor; returns the height it took (0: none).
    std::function<float(caliper_live_views::View&, float width)> under_header;
    // The settings of the scene views that closed, by name, given back when one opens again.
    std::map<std::string, caliper_live_views::SceneSettings> scene_settings;
};

namespace caliper_live_views {

// live_views.cpp
View* find(CaliperCore* core, const char* name);
// The host's event sink (empty: no selections, no events). Creates the state.
void set_event_sink(CaliperCore* core, EventSink sink);
// The headers' look and the stream state their dots show (see the state).
void set_look(CaliperCore* core, const Look& look);
// What a host draws under each view's header (see the state); empty: nothing.
void set_under_header(CaliperCore* core, std::function<float(View&, float)> fn);
void set_stream_state(CaliperCore* core, int state);
// Whether the views show their diagnostics (size, path, seq, latency,
// updates, drops) under their header: CALIPER_VIEWS_DEBUG=1, or a host that
// sets core->views_debug. Off by default.
bool views_debug(const CaliperCore* core);
// The grid of the view windows, before the frame draws them: every view of
// this core gets a cell of the area (the host's work area for a hosted core,
// else the canvas), at least kMinCell large when the area allows, in creation
// order, laid out again whenever the area or the number of views changes.
// A window the person moved, resized or docked keeps its own place.
void layout_views(CaliperCore* core);
// Around ImGui::Begin of view `name`'s window (the typed views' own windows
// and the v1.2 window a heatmap or image draws in): window_before sets the
// window's cell when the grid changed and returns false when the view has no
// cell (the caller places it itself); window_after notes a window the person
// moved, resized or docked.
bool window_before(CaliperCore* core, const std::string& name);
void window_after(CaliperCore* core, const std::string& name);
// The flags every view window takes: no collapse, and no focus when it
// appears, so a view that appears never takes a click from the page.
constexpr ImGuiWindowFlags kWindowFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoFocusOnAppearing;

// A header toggle of view `name`, set by the page: "follow" (series), "log"
// (series, histogram) or "rotate" (points), or "depth", the depth cues of 3-D
// points, which no header shows. False: no such view or toggle. The views report
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
// The points view's scene and the layer it draws through, made on first use; null with *why (may
// be null) where no backend of the layer can start, and the view's notice says why.
PointsScene* points_scene(CaliperCore* core, View& v, std::string* why);
void  poll_handles(CaliperCore* core, View& v);   // retire done handles (non-blocking)
void  fit_points(View& v, float aspect);          // the scene core's fit to the scene's bounds

// live_views_draw.cpp
void draw_view(CaliperCore* core, CaliperLiveViewsState* S, View& v);
// The text of a view's header after its name ("1,000 points  3-D  10 classes"), as the header draws it.
std::string info_text(const CaliperCore* core, const View& v);
// The f32 clip-from-world matrix of a points view's camera at its render target's size, as the
// draws get it from the scene core (hover, lasso and the browser's self-check project with it).
bool points_mvp(const View& v, float out[16]);
// The fixed 16-colour categorical palette (r | g<<8 | b<<16 | a<<24), the scene core's.
using caliper_scene::kPalette;
// The side in physical pixels of the square a points view draws per point.
float point_size_px(uint64_t n, float scale);

}  // namespace caliper_live_views
