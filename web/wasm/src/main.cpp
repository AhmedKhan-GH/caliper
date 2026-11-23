// main.cpp: the page host of Caliper's views in the browser (web/wasm). It
// plays the part the Caliper app plays natively: it owns the canvas, the ImGui
// and ImPlot contexts and the WebGPU renderer, and draws Caliper's real view
// layer through a hosted core (caliper_core_create_hosted). The data comes
// from a producer over wire protocol v3: a NetSource (net_source.h) is the
// caliper_live::Source, and every frame source_poll.h applies what is new
// through caliper_core_live_apply, as live_core.cpp does for shared memory.
//
// One module instance draws on one canvas (Module.canvas) and shows one view,
// or every view of the session when no view is named. The instance reads its
// settings from Module.caliperConfig: {src, view, kind, header, compat, feed,
// scene}; compat "true" asks for a WebGPU device in compatibility mode.
//
// With feed "page" there is no stream: the instance draws one scene of its own
// over the whole canvas, as the app's spec runtime draws a scene panel that
// holds programs alone (a scene view of its own with box fit, through
// draw_scene_view_in_rect, with the panel's `labels` and `scale_bar` from
// `scene`, a JSON object), and the page hands it the instances of pieces
// named for the panel `view` and their resources through
// caliper_views_feed_program() and caliper_views_feed_resource(). The
// browser's spec page does this with what panel.data gives a viewer, which
// never opens the stream. Picks and selections go to Module.onEvent.
//
// What a person does in a view (a selection, a pick, a hover) comes out of
// the view layer's event sink: it goes to the producer as an EVENT through
// the NetSource and to the page through Module.onEvent(view, type, detail).
// The facts the page's header shows about a view (its size, range, last
// value) go to Module.onInfo(view, facts) when they change, at most ten times
// a second.
//
// Start-up is asynchronous because a browser hands out a WebGPU device only
// through promises: main() requests the adapter, the adapter callback requests
// the device, and the device callback builds everything, connects and starts
// the frame loop. No Asyncify is needed.
#include "draw_parity.h"
#include "fonts.h"
#include "gfx_selftest.h"
#include "scene_parity.h"
#include "scene_selftest.h"
#include "gfx/shaders/generated/programs_webgpu.h"
#include "net_source.h"
#include "net/json.h"
#include "page_style.h"
#include "points_parity.h"
#include "source_poll.h"
#include "web_input.h"
#include "web_programs.h"

#include <caliper_builder/programs/pieces.h>   // kMaxSelected

#include "caliper/embed.h"
#include "caliper/live_net.h"   // CALIPER_NET_BYE_NORMAL
#include "caliper/live_views.h"
#include "embed/embed_core.h"
#include "gfx/colormap_keys.h"   // derive_keys, for the test of the keys this build derives
#include "live/live_scene.h"         // a scene view's facts
#include "live/live_views_scene.h"
#include "live/live_points.h"        // the points check reads the view's scene, camera and payload
#include "live/live_views_state.h"
#include "host_services.h"
#include "renderer/webgpu_renderer.h"
#include "tensor_bridge.h"   // colormap_lut, map_f32_to_rgba8 (the CPU reference)
#include "ui/sternbild_fonts.h"
#include "ui/sternbild_theme.h"

#include <imgui.h>
#include <implot.h>

#include <emscripten/emscripten.h>
#include <emscripten/heap.h>
#include <emscripten/html5.h>
#include <malloc.h>
#include <webgpu/webgpu.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using caliper_host::WebGpuRenderer;
using caliper_web::NetSource;

namespace {

struct Config {
    std::string src;    // ws://127.0.0.1:<port>/v3/stream?token=<token>
    std::string view;   // "" = every view
    int32_t kind = 0;   // 0 = any
    bool header = true; // false: the page draws the view's header (no title bars)
    bool compat = false;// a compatibility-mode device
    bool page_feed = false;   // feed "page": no stream, the page feeds the programs
    std::string scene;        // feed "page": the scene view's description, JSON
};

struct Page {
    Config cfg;
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WebGpuRenderer* renderer = nullptr;   // owned by the hosted core
    CaliperCore* core = nullptr;
    std::unique_ptr<NetSource> net;
    // The draw halves of pieces that the script relays, and the feeds of the scene views that
    // draw them (web_programs.h).
    std::unique_ptr<caliper_web::WebPrograms> programs;
    uint64_t programs_published = 0;
    // feed "page": the scene that the page's programs draw in, with its own backend.
    std::unique_ptr<caliper_gfx::Backend> page_gfx;
    std::unique_ptr<caliper_live_views::SceneView> page_view;
    caliper_live_views::SceneRectOptions page_options;
    caliper_live_views::SceneRectState page_state;
    uint64_t page_events = 0;           // picks and selections of the page's scene, for the page's tests
    std::string page_last_event;
    caliper_web::PollState poll;
    std::string adapter_text;
    std::string error;

    // Timing (milliseconds).
    uint64_t frames = 0;
    double t_mark = 0.0;
    uint64_t frames_mark = 0;
    double cpu_sum = 0.0, poll_sum = 0.0, ui_sum = 0.0, enc_sum = 0.0;
    uint64_t cpu_n = 0;
    double fps = 0.0;

    // The byte check of the WGSL colormap against the CPU reference.
    // Every texture view is checked once: its texture read back and compared,
    // byte for byte, with what the CPU makes of the same snapshot.
    struct Check {
        std::string view;
        int kind = 0;
        long long mismatch = -1;          // texels that differ (-2: readback failed)
        int maxdiff = -1;                 // largest channel difference
    };
    std::vector<Check> checks;            // done, in order
    bool check_pending = false;
    std::vector<uint8_t> check_expect;
    std::string check_view;
    int check_kind = 0;
    uint64_t check_next = 10;             // the frame of the next check
    double last_publish = 0.0;            // when the counters were last published (ms)

    std::string selftest;                 // the graphics layer's self-test: "" or its report
    std::string scene_selftest;           // the last scene row: "" or its report
    std::string draw_parity;              // the pixel comparison's draw: "" or its report
    std::string scene_parity;             // the scene comparison's draw: "" or its report
    std::string points_parity;            // points@1 initial and replacement payloads: "" or its report

    bool ready = false;                   // the first snapshot was applied
    bool destroyed = false;

    float css_w = 0.0f, css_h = 0.0f;     // the canvas's CSS box
    int fb_w = 0, fb_h = 0;               // its backing store

    // The header facts last sent to the page, per view, and when.
    struct Info {
        std::string view, facts;
        bool pending = false;
        double sent_ms = -1e9;
    };
    std::vector<Info> infos;
    uint64_t events_sent = 0;
    bool publish_soon = false;            // a selection or pick: publish after this frame
};
Page g;

std::string config_string(const char* key) {
    char* s = (char*)EM_ASM_PTR({
        var c = Module["caliperConfig"] || {};
        var v = c[UTF8ToString($0)];
        return v == null ? 0 : stringToNewUTF8(String(v));
    }, key);
    if (!s) return {};
    std::string out(s);
    std::free(s);
    return out;
}

int32_t kind_of(const std::string& k) {
    if (k == "heatmap") return CALIPER_LIVE_KIND_HEATMAP;
    if (k == "image") return CALIPER_LIVE_KIND_IMAGE;
    if (k == "points") return CALIPER_LIVE_KIND_POINTS;
    if (k == "series") return CALIPER_LIVE_KIND_SERIES;
    if (k == "histogram") return CALIPER_LIVE_KIND_HISTOGRAM;
    if (k == "scene") return caliper_live_views::kKindScene;   // the protocol 3.2 draft
    return 0;
}

// ---- the report to the page and the tests: Module.caliperStats --------------
// Keys are passed as strings so that Closure's property renaming leaves them.
void put(const char* key, double v) {
    EM_ASM({ (Module["caliperStats"] = Module["caliperStats"] || {})[UTF8ToString($0)] = $1; },
           key, v);
}
void put(const char* key, const char* v) {
    EM_ASM({
        (Module["caliperStats"] = Module["caliperStats"] || {})[UTF8ToString($0)] =
            UTF8ToString($1);
    }, key, v);
}

void publish() {
    const double n = g.cpu_n ? (double)g.cpu_n : 1.0;
    put("frames", (double)g.frames);
    put("fps", g.fps);
    put("cpuMs", g.cpu_sum / n);
    put("pollMs", g.poll_sum / n);
    put("uiMs", g.ui_sum / n);
    put("encodeMs", g.enc_sum / n);
    put("gpuMs", g.renderer ? g.renderer->gpu_ms_mean() : 0.0);
    put("textures", g.renderer ? (double)g.renderer->texture_count() : 0.0);
    put("allocations", g.renderer ? (double)g.renderer->allocation_count() : 0.0);
    put("retiredViews", g.renderer ? (double)g.renderer->retired_view_count() : 0.0);
    put("dispatches", g.renderer ? (double)g.renderer->colormap_dispatches() : 0.0);
    put("rendererPath", g.renderer ? g.renderer->last_device_path() : "");
    // The graphics layer: the device's feature level, what the backend holds
    // and the errors WebGPU reported to it.
    caliper_gfx::WebGpuBackend* gb = g.renderer ? g.renderer->gfx() : nullptr;
    put("gfxReady", gb ? 1.0 : 0.0);
    put("featureLevel", !gb ? "" : gb->compatibility() ? "compatibility" : "core");
    put("gfxBuffers", gb ? (double)gb->live_buffer_count() : 0.0);
    put("gfxSpare", gb ? (double)gb->spare_buffer_count() : 0.0);
    std::string gerr;
    if (gb)
        for (const auto& m : gb->device_errors()) gerr += (gerr.empty() ? "" : " | ") + m;
    put("gfxErrors", gerr.c_str());
    put("gfxSelftest", g.selftest.c_str());
    put("sceneSelftest", g.scene_selftest.c_str());
    put("drawParity", g.draw_parity.c_str());
    put("sceneParity", g.scene_parity.c_str());
    put("pointsParity", g.points_parity.c_str());
    put("programs", g.programs ? g.programs->status_json().c_str() : "[]");
    put("state", g.net ? g.net->state_text().c_str() : "starting");
    put("messages", g.net ? (double)g.net->messages() : 0.0);
    put("bytes", g.net ? (double)g.net->bytes() : 0.0);
    put("updates", (double)g.poll.updates);
    put("applyErrors", (double)g.poll.errors);
    put("views", g.core ? caliper_core_live_view_count(g.core) : 0);
    std::string per;
    for (const auto& v : g.poll.views) {
        CaliperLiveViewInfo info{};
        info.struct_size = sizeof info;
        if (!v.shown || !caliper_core_live_view_info(g.core, v.name.c_str(), &info)) continue;
        char b[256];
        std::snprintf(b, sizeof b, "%s%s:kind=%d,updates=%llu,count=%lld,path=%s,w=%d,h=%d,latency=%.2f,creates=%llu",
                      per.empty() ? "" : ";", v.name.c_str(), info.kind,
                      (unsigned long long)info.updates, (long long)info.count, info.path, info.width,
                      info.height, info.latency_ms, (unsigned long long)info.creates);
        per += b;
        // A points view: the index of the point under the mouse, -1 for none.
        if (info.kind == CALIPER_LIVE_KIND_POINTS)
            if (const caliper_live_views::View* lv = caliper_live_views::find(g.core, v.name.c_str()))
                per += ",hover=" + std::to_string(lv->hover_idx);
    }
    put("viewInfo", per.c_str());
    // Where each view drew its data (CSS pixels in the canvas, and the data
    // window there), what it has selected and hovers, whether a plot follows
    // its data, and how many events went out: for tests, as JSON {"view":
    // {"area": [x, y, w, h], "window": [x0, x1, y0, y1], "yDown": bool,
    // "selected": n, "hover": item, "cols": n, "follow": bool, "log": bool}},
    // and for a scene view "scaleBar": [x0, y0, x1, y1], the band the scale
    // bar drew over its picture in the same CSS pixels, or null without one.
    std::string geo = "{";
    double selected = 0.0;
    if (g.core && g.core->live_views)
        for (const auto& v : g.core->live_views->views) {
            if (!v->geo_valid) continue;
            const double sel = v->kind == CALIPER_LIVE_KIND_POINTS ? (double)v->sel_count
                               : (v->kind == CALIPER_LIVE_KIND_HEATMAP || v->kind == CALIPER_LIVE_KIND_IMAGE)
                                   ? (double)((v->sel_r1 - v->sel_r0) * (v->sel_c1 - v->sel_c0))
                                   : (double)(v->sel_s1 - v->sel_s0);
            if (v->name == g.cfg.view || g.cfg.view.empty()) selected += sel;
            int cols = 0;
            for (const auto& tv : g.core->views)
                if (tv.name == v->name) cols = tv.w;
            char b[640];
            std::snprintf(b, sizeof b,
                          "%s\"%s\":{\"area\":[%.9g,%.9g,%.9g,%.9g],\"window\":[%.17g,%.17g,%.17g,%.17g],"
                          "\"yDown\":%s,\"selected\":%.0f,\"hover\":%lld,\"cols\":%d,\"follow\":%s,\"log\":%s",
                          geo.size() > 1 ? "," : "", v->name.c_str(), v->geo_area[0], v->geo_area[1],
                          v->geo_area[2], v->geo_area[3], v->geo_window[0], v->geo_window[1], v->geo_window[2],
                          v->geo_window[3], v->geo_ydown ? "true" : "false", sel, (long long)v->hover_item, cols,
                          v->follow ? "true" : "false", v->log_y ? "true" : "false");
            geo += b;
            // A points view: its camera, the rotate toggle, the depth cues and
            // the box's tick labels (digits, signs, dots, e, = and separators).
            if (v->kind == CALIPER_LIVE_KIND_POINTS) {
                // The scene core's orbit in 3-D (az, el, distance, the fitted radius); zero angles and
                // distance in 2-D, whose camera is a pan and zoom.
                double cam[4] = {0.0, 0.0, 0.0, 0.0};
                if (const caliper_live_views::PointsScene* sc = v->points ? v->points->scene.get() : nullptr) {
                    if (const caliper_scene::Orbit* o = std::get_if<caliper_scene::Orbit>(&sc->camera())) {
                        cam[0] = o->azimuth;
                        cam[1] = o->elevation;
                        cam[2] = o->distance;
                    }
                    cam[3] = sc->radius();
                }
                std::snprintf(b, sizeof b,
                              ",\"dims\":%d,\"camera\":[%.9g,%.9g,%.9g,%.9g],\"rotate\":%s,\"cues\":%s,\"box\":\"",
                              v->dims, cam[0], cam[1], cam[2], cam[3],
                              v->auto_rotate ? "true" : "false", v->depth_cues ? "true" : "false");
                geo += b;
                if (v->dims == 3) geo += v->box_labels;
                geo += "\"";
            }
            if (v->kind == caliper_live_views::kKindScene) {
                const float* bar = v->geo_bar;
                if (bar[2] > bar[0] && bar[3] > bar[1])
                    std::snprintf(b, sizeof b, ",\"scaleBar\":[%.9g,%.9g,%.9g,%.9g]", bar[0], bar[1], bar[2], bar[3]);
                else
                    std::snprintf(b, sizeof b, ",\"scaleBar\":null");
                geo += b;
                // The orbit the view draws with, [target x, y, z, distance], null for a pan and zoom or a
                // pinhole: a test checks that the fit framed the scene's bounds with finite numbers.
                const caliper_live_views::SceneView* sv = caliper_live_views::scene_view(*v);
                const caliper_scene::Orbit* o = sv ? std::get_if<caliper_scene::Orbit>(&sv->camera()) : nullptr;
                const bool finite = o && std::isfinite(o->target[0]) && std::isfinite(o->target[1]) &&
                                    std::isfinite(o->target[2]) && std::isfinite(o->distance);
                if (o && sv->fitted() && finite)
                    std::snprintf(b, sizeof b, ",\"orbit\":[%.17g,%.17g,%.17g,%.17g]", o->target[0], o->target[1],
                                  o->target[2], o->distance);
                else if (o && sv->fitted())
                    std::snprintf(b, sizeof b, ",\"orbit\":\"not finite\"");
                else
                    std::snprintf(b, sizeof b, ",\"orbit\":null");
                geo += b;
                // The scale bar the view measures for its picture: [length in scene units, logical pixels].
                const caliper_scene::ScaleBar sb = sv ? sv->scale_bar(v->geo_area[2], v->geo_area[3]) : caliper_scene::ScaleBar{};
                if (sb.visible && std::isfinite(sb.length) && std::isfinite(sb.pixels))
                    std::snprintf(b, sizeof b, ",\"scaleBarLength\":[%.17g,%.17g]", sb.length, sb.pixels);
                else
                    std::snprintf(b, sizeof b, sb.visible ? ",\"scaleBarLength\":\"not finite\"" : ",\"scaleBarLength\":null");
                geo += b;
            }
            geo += "}";
        }
    geo += "}";
    put("geometry", geo.c_str());
    // Library pieces in each scene view, as JSON keyed by view id.
    caliper_net::Json shapes = caliper_net::Json::object();
    for (const auto& pv : g.poll.views) {
        if (!pv.shown) continue;
        const caliper_live_views::View* v = caliper_live_views::find(g.core, pv.name.c_str());
        const caliper_live_views::SceneView* sv = v ? caliper_live_views::scene_view(*v) : nullptr;
        if (!sv) continue;
        const caliper_live_views::SceneShapeCounts c = sv->shape_counts();
        caliper_net::Json one = caliper_net::Json::object();
        one.set("draw", "pieces");
        one.set("pieces", double(c.pieces));
        shapes.set(pv.name, one);
    }
    put("sceneShapes", shapes.dump().c_str());
    put("selected", selected);
    put("events", (double)g.events_sent);
    put("pageEvents", (double)g.page_events);
    put("pageLastEvent", g.page_last_event.c_str());
    if (g.page_view) {
        // The page scene's orbit, [target x, y, z, distance, azimuth, elevation] (angles in radians), and
        // the labels of its programs' text tables, for the page's tests of its camera and its labels.
        const caliper_scene::Orbit* o = std::get_if<caliper_scene::Orbit>(&g.page_view->camera());
        char ob[256];
        if (o && g.page_view->fitted())
            std::snprintf(ob, sizeof ob, "[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g]", o->target[0], o->target[1], o->target[2],
                          o->distance, o->azimuth, o->elevation);
        else
            std::snprintf(ob, sizeof ob, "null");
        put("pageOrbit", ob);
        // [[text, x0, y0, x1, y1], ...]: the labels drawn at the last frame with their text boxes, in CSS pixels
        // of the canvas, which the page's dashboard comparison checks as text.
        caliper_net::Json labels = caliper_net::Json::array();
        for (const caliper_live_views::DrawnLabel& l : g.page_state.labels) {
            caliper_net::Json e = caliper_net::Json::array();
            e.push(l.text);
            e.push((double)l.x0);
            e.push((double)l.y0);
            e.push((double)l.x1);
            e.push((double)l.y1);
            labels.push(std::move(e));
        }
        put("pageLabels", labels.dump().c_str());
    }
    const caliper_web::InputCounts ic = caliper_web::input_counts();
    put("inputPointer", ic.pointer);
    put("inputButtons", ic.buttons);
    put("inputWheel", ic.wheel);
    put("inputKeys", ic.keys);
    put("inputChars", ic.chars);
    put("lastKey", ic.last_key);
    put("cssW", g.css_w);
    put("cssH", g.css_h);
    put("fbW", g.fb_w);
    put("fbH", g.fb_h);
    // Memory: the WebAssembly heap and the bytes malloc has handed out.
    const struct mallinfo mi = mallinfo();
    put("heapSize", (double)emscripten_get_heap_size());
    put("heapUsed", (double)mi.uordblks);
    // checkState 2 once every texture view drawn so far has been checked;
    // checkMismatch / checkMaxDiff sum and take the maximum over them, and
    // checks lists each one as "view:kind:mismatch:maxdiff".
    bool all = !g.checks.empty() && !g.check_pending;
    long long mism = 0;
    int maxd = 0;
    std::string list;
    for (const auto& c : g.checks) {
        mism = c.mismatch < 0 || mism < 0 ? -1 : mism + c.mismatch;
        maxd = std::max(maxd, c.maxdiff);
        list += (list.empty() ? "" : ";") + c.view + ":" + std::to_string(c.kind) + ":" +
                std::to_string(c.mismatch) + ":" + std::to_string(c.maxdiff);
    }
    auto unchecked = [&](const std::string& name) {
        return std::none_of(g.checks.begin(), g.checks.end(),
                            [&](const Page::Check& c) { return c.view == name; });
    };
    for (const auto& tv : g.core ? g.core->views : std::vector<caliper_embed_views::TensorView>{})
        if (tv.tex && unchecked(tv.name)) all = false;
    if (g.core && g.core->live_views)
        for (const auto& v : g.core->live_views->views)
            if (v->kind == CALIPER_LIVE_KIND_POINTS && v->label_count > 0 && v->points && v->points->scene &&
                v->points->scene->has_target() && unchecked(v->name))
                all = false;
    put("checkState", all ? 2 : g.checks.empty() && !g.check_pending ? 0 : 1);
    put("checkMismatch", g.checks.empty() ? -1.0 : (double)mism);
    put("checkMaxDiff", g.checks.empty() ? -1 : maxd);
    put("checks", list.c_str());
    put("adapter", g.adapter_text.c_str());
    put("error", g.error.c_str());
}

void fail(const std::string& why) {
    g.error = why;
    std::fprintf(stderr, "[caliper-views] %s\n", why.c_str());
    publish();
}

void finish_check(Page::Check c) {
    g.checks.push_back(std::move(c));
    g.check_pending = false;
    publish();
}

// A points view with categorical labels: its render target read back, and at
// the pixel under each of up to 256 sampled points that no other point covers
// in front of it (later in draw order in 2-D, nearer in 3-D), the palette
// colour of its label. With the depth cues of 3-D points, only points in front
// of the camera's target are compared, which the fade leaves at full strength,
// and a neighbour covers a point within the largest size the cues may draw.
// Records mismatches and, in maxdiff, how many points were compared. False when
// there is nothing to check yet.
bool start_points_check(caliper_live_views::View& v) {
    caliper_live_views::PointsScene* sc = v.points ? v.points->scene.get() : nullptr;
    if (!v.have_cur || !v.cur.pos_cpu || !v.cur.aux_cpu || !sc || !sc->has_target() || v.dirty) return false;
    const int W = (int)sc->width(), H = (int)sc->height();
    const uint64_t n = v.cur.n;
    float M[16];   // the scene core's clip-from-world matrix, as the draw gets it
    if (!caliper_live_views::points_mvp(v, M)) return false;
    struct Proj { float x, y, z, w; bool ok; };
    std::vector<Proj> pr(n);
    const float* pos = (const float*)v.cur.pos_cpu;
    for (uint64_t i = 0; i < n; ++i) {
        const float x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
        const float cw = M[3] * x + M[7] * y + M[11] * z + M[15];
        if (!(cw > 1e-9f)) { pr[i] = {0, 0, 0, 0, false}; continue; }
        pr[i] = {((M[0] * x + M[4] * y + M[8] * z + M[12]) / cw * 0.5f + 0.5f) * (float)W,
                 (0.5f - (M[1] * x + M[5] * y + M[9] * z + M[13]) / cw * 0.5f) * (float)H,
                 (M[2] * x + M[6] * y + M[10] * z + M[14]) / cw, cw, true};
    }
    // Half the side a point is drawn with, plus a pixel: with the depth cues
    // its size follows its distance, as the renderer draws it.
    float att_ref = 0.0f, fade_w0 = 0.0f, fade_w1 = 0.0f;   // the scene's cues, zero when off
    sc->cue_values(&att_ref, &fade_w0, &fade_w1);
    const bool cues = att_ref > 0.0f;
    const float side = caliper_live_views::point_size_px(n, g.core->scale);
    auto half_of = [&](const Proj& p) {
        const float k = cues ? std::clamp(att_ref / p.w, caliper_scene::kPointAttenMin, caliper_scene::kPointAttenMax) : 1.0f;
        return 0.5f * side * k + 1.0f;
    };
    struct Want { int px, py; uint32_t rgba; };
    auto wants = std::make_shared<std::vector<Want>>();
    const float* aux = (const float*)v.cur.aux_cpu;
    // Up to 1,024 candidates, evenly spread, of which those no other point
    // covers are compared (at most 256).
    const uint64_t stride = std::max<uint64_t>(1, n / 1024);
    for (uint64_t i = 0; i < n && wants->size() < 256; i += stride) {
        const Proj& a = pr[i];
        if (!a.ok || a.x < 1 || a.y < 1 || a.x >= W - 1 || a.y >= H - 1 || a.z < 0 || a.z > 1) continue;
        if (fade_w1 > fade_w0 && a.w > fade_w0) continue;   // dimmed by the fade
        bool covered = false;
        for (uint64_t j = 0; j < n && !covered; ++j) {
            if (j == i || !pr[j].ok) continue;
            const float h = half_of(pr[j]) + 1.0f;   // a pixel more for rounding
            if (std::fabs(pr[j].x - a.x) > h || std::fabs(pr[j].y - a.y) > h) continue;
            covered = v.dims == 2 ? j > i : (pr[j].z <= a.z + 1e-6f);
        }
        if (covered) continue;
        const float t = std::min(std::max((aux[i] - v.pvmin) / (v.pvmax - v.pvmin), 0.0f), 1.0f);
        wants->push_back({(int)a.x, (int)a.y, v.lut[(uint32_t)(t * 255.0f + 0.5f)]});
    }
    if (wants->empty()) return false;
    g.check_pending = true;
    const std::string name = v.name;
    g.renderer->readback_async(sc->color(), (uint32_t)W, (uint32_t)H, [wants, name, W, H](std::vector<uint8_t> got) {
        Page::Check c;
        c.view = name;
        c.kind = CALIPER_LIVE_KIND_POINTS;
        if (got.size() != (size_t)W * H * 4) {
            c.mismatch = -2;
        } else {
            long long bad = 0;
            for (const Want& w : *wants) {
                const uint8_t* q = got.data() + ((size_t)w.py * W + w.px) * 4;
                const uint32_t px = q[0] | (q[1] << 8) | (q[2] << 16) | ((uint32_t)q[3] << 24);
                if (px != w.rgba) ++bad;
            }
            c.mismatch = bad;
            c.maxdiff = (int)wants->size();   // for points: how many were compared
        }
        finish_check(c);
    });
    return true;
}

// Read the next unchecked texture view back and compare it, byte for byte,
// with what the CPU makes of the snapshot it was drawn from: the reference
// colormap map_f32_to_rgba8 for a heatmap, the bytes themselves for an image.
// Points views with categorical labels are checked by start_points_check.
void start_check() {
    if (g.core->live_views)
        for (auto& pv : g.core->live_views->views)
            if (pv->kind == CALIPER_LIVE_KIND_POINTS && pv->label_count > 0 &&
                std::none_of(g.checks.begin(), g.checks.end(),
                             [&](const Page::Check& c) { return c.view == pv->name; }) &&
                start_points_check(*pv))
                return;
    for (auto& tv : g.core->views) {
        if (!tv.tex || std::any_of(g.checks.begin(), g.checks.end(),
                                   [&](const Page::Check& c) { return c.view == tv.name; }))
            continue;
        const caliper_live::Snapshot* s = g.net ? g.net->shown(tv.name) : nullptr;
        if (!s) continue;
        const uint8_t* bytes = (const uint8_t*)s->host_base + s->byte_offset;
        const size_t n = (size_t)tv.w * tv.h * 4;
        if (tv.mapped && s->kind == CALIPER_LIVE_KIND_HEATMAP) {
            g.check_expect.assign(n, 0);
            caliper_host::map_f32_to_rgba8((const float*)bytes, tv.w, tv.h,
                                           caliper_host::colormap_lut(tv.colormap), tv.vmin, tv.vmax,
                                           g.check_expect.data());
        } else if (!tv.mapped && s->kind == CALIPER_LIVE_KIND_IMAGE) {
            g.check_expect.assign(bytes, bytes + n);
        } else {
            continue;
        }
        g.check_pending = true;
        g.check_view = tv.name;
        g.check_kind = (int)s->kind;
        g.renderer->readback_async(tv.tex, [](std::vector<uint8_t> got) {
            Page::Check c;
            c.view = g.check_view;
            c.kind = g.check_kind;
            if (got.size() != g.check_expect.size()) {
                c.mismatch = -2;   // readback failed
            } else {
                long long bad = 0;
                int maxd = 0;
                for (size_t i = 0; i < got.size(); i += 4) {
                    int d = 0;
                    for (int k = 0; k < 4; ++k)
                        d = std::max(d, std::abs((int)got[i + k] - (int)g.check_expect[i + k]));
                    if (d) ++bad;
                    maxd = std::max(maxd, d);
                }
                c.mismatch = bad;
                c.maxdiff = maxd;
            }
            finish_check(c);
        });
        return;
    }
}

// One named view fills the canvas: its window is pinned to the whole display.
// With no view named, the views tile as they do in the native app.
void pin_single_view(float w, float h) {
    if (g.cfg.view.empty()) return;
    for (const auto& v : g.poll.views) {
        if (!v.shown || v.name != g.cfg.view) continue;
        const std::string label = v.name + "###caliper.view." + v.name;
        ImGui::SetWindowPos(label.c_str(), ImVec2(0.0f, 0.0f), ImGuiCond_Always);
        ImGui::SetWindowSize(label.c_str(), ImVec2(w, h), ImGuiCond_Always);
    }
}

// The connection state, while there is nothing to draw or the session ended.
void draw_status(float w, float h) {
    const bool any = std::any_of(g.poll.views.begin(), g.poll.views.end(),
                                 [](const caliper_web::PolledView& v) { return v.shown; });
    std::string text = g.error.empty() ? (g.net ? g.net->state_text() : "starting") : g.error;
    if (g.net && !g.net->views_text().empty() && !any) text += "\n" + g.net->views_text();
    const bool ended = g.net && !g.poll.producer_alive;
    if (any && !ended && g.error.empty()) return;
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ImVec2 ts = ImGui::CalcTextSize(text.c_str());
    const ImVec2 p = any ? ImVec2(8.0f, h - ts.y - 8.0f) : ImVec2((w - ts.x) * 0.5f, (h - ts.y) * 0.5f);
    dl->AddRectFilled(ImVec2(p.x - 6.0f, p.y - 4.0f), ImVec2(p.x + ts.x + 6.0f, p.y + ts.y + 4.0f),
                      IM_COL32(0x18, 0x17, 0x15, 230));
    dl->AddText(p, IM_COL32(0xd0, 0xc9, 0xba, 255), text.c_str());
}

// ---- the header facts ---------------------------------------------------------
std::string num(double v) {
    if (!std::isfinite(v)) return "null";
    char b[40];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}

// What the page's header says about a view, as the element's own views say
// it (web/src/views/info.ts formats it): JSON facts by kind.
std::string facts(const caliper_live_views::View& v) {
    char b[256];
    switch (v.kind) {
        case CALIPER_LIVE_KIND_HEATMAP:
        case CALIPER_LIVE_KIND_IMAGE:
            for (const auto& tv : g.core->views)
                if (tv.name == v.name && tv.tex) {
                    std::snprintf(b, sizeof b, "{\"kind\":\"%s\",\"rows\":%d,\"cols\":%d,\"lo\":%s,\"hi\":%s}",
                                  v.kind == CALIPER_LIVE_KIND_HEATMAP ? "heatmap" : "image", tv.h, tv.w,
                                  num(tv.vmin).c_str(), num(tv.vmax).c_str());
                    return b;
                }
            return {};
        case CALIPER_LIVE_KIND_POINTS:
            std::snprintf(b, sizeof b, "{\"kind\":\"points\",\"n\":%lld,\"dims\":%d,\"classes\":%d,\"values\":%s}",
                          (long long)v.count, v.dims, v.label_count, v.cur.aux ? "true" : "false");
            return b;
        case CALIPER_LIVE_KIND_SERIES:
            if (v.values.empty()) return {};
            std::snprintf(b, sizeof b, "{\"kind\":\"series\",\"end\":%llu,\"last\":%s}",
                          (unsigned long long)(v.first + v.values.size()), num(v.values.back()).c_str());
            return b;
        case CALIPER_LIVE_KIND_HISTOGRAM: {
            double total = 0.0;
            for (float c : v.values) total += c;
            std::snprintf(b, sizeof b, "{\"kind\":\"histogram\",\"bins\":%zu,\"lo\":%s,\"hi\":%s,\"total\":%s}",
                          v.values.size(), num(v.lo).c_str(), num(v.hi).c_str(), num(total).c_str());
            return b;
        }
        case caliper_live_views::kKindScene: {
            const caliper_live_views::SceneView* sv = caliper_live_views::scene_view(v);
            if (!sv) return v.scene ? "{\"kind\":\"scene\",\"notice\":" + caliper_net::Json(v.scene->notice).dump() + "}"
                                    : std::string();
            if (!sv->notes().empty()) {
                std::string notice;
                for (const auto& note : sv->notes()) {
                    if (!notice.empty()) notice += '\n';
                    notice += note;
                }
                return "{\"kind\":\"scene\",\"notice\":" + caliper_net::Json(notice).dump() + "}";
            }
            const caliper_live_views::SceneCounts c = sv->counts();
            std::snprintf(b, sizeof b, "{\"kind\":\"scene\",\"entities\":%u,\"items\":%llu,\"copied\":%llu,\"gpu_ms\":%s}",
                          c.entities, (unsigned long long)c.items, (unsigned long long)c.bytes_last,
                          num(sv->cost().gpu_ms).c_str());
            return b;
        }
        default:
            return {};
    }
}

// Sends each view's facts when they changed, at most every 100 ms per view.
void send_infos() {
    if (!g.core || !g.core->live_views) return;
    const double t = emscripten_get_now();
    for (const auto& v : g.core->live_views->views) {
        auto it = std::find_if(g.infos.begin(), g.infos.end(), [&](const Page::Info& i) { return i.view == v->name; });
        if (it == g.infos.end()) {
            g.infos.push_back({v->name, {}, false, -1e9});
            it = g.infos.end() - 1;
        }
        std::string f = facts(*v);
        if (f.empty() || f == it->facts) continue;
        if (t - it->sent_ms < 100.0) continue;   // the next frame after the interval sends the newest
        it->facts = std::move(f);
        it->sent_ms = t;
        EM_ASM({
            var cb = Module["onInfo"];
            if (cb) cb(UTF8ToString($0), JSON.parse(UTF8ToString($1)));
        }, it->view.c_str(), it->facts.c_str());
    }
}

// The view layer's events: to the producer, and to the page. A toggle a
// view changed itself (a plot stopped or started following) goes to the page
// only, for its header.
void on_event(const caliper_live_views::LiveEvent& e) {
    if (std::strcmp(e.type, "toggle") == 0) {
        EM_ASM({
            var cb = Module["onToggle"];
            var d = JSON.parse(UTF8ToString($1));
            if (cb) cb(UTF8ToString($0), d["name"], d["on"]);
        }, e.view.c_str(), e.detail.c_str());
        g.publish_soon = true;
        return;
    }
    // The page may add to a selection's detail what the view layer does not
    // know: the item space of the axis and the selected items (3.1).
    std::string detail = e.detail;
    if (std::strcmp(e.type, "select") == 0) {
        char* aug = (char*)EM_ASM_PTR({
            var f = Module["onAugment"];
            if (!f) return 0;
            var d = f(UTF8ToString($0), JSON.parse(UTF8ToString($1)));
            return d ? stringToNewUTF8(JSON.stringify(d)) : 0;
        }, e.view.c_str(), e.detail.c_str());
        if (aug) {
            detail = aug;
            std::free(aug);
        }
    }
    if (g.net) {
        const uint64_t key = g.net->key_of(e.view);
        if (key && g.net->send_event(key, e.type, detail)) ++g.events_sent;
    }
    EM_ASM({
        var cb = Module["onEvent"];
        if (cb) cb(UTF8ToString($0), UTF8ToString($1), JSON.parse(UTF8ToString($2)));
    }, e.view.c_str(), e.type, detail.c_str());
    if (std::strcmp(e.type, "hover") != 0) g.publish_soon = true;
}

// The self-test's pictures, for the browser tests' captures:
// Module.caliperGfxImages = {name: {w, h, data: Uint8Array}}.
void publish_selftest_images() {
    for (const auto& im : caliper_web::gfx_selftest_images())
        EM_ASM({
            var m = Module["caliperGfxImages"] = Module["caliperGfxImages"] || {};
            m[UTF8ToString($0)] = ({"w": $1, "h": $2, "data": HEAPU8.slice($3, $3 + $4)});
        }, im.name.c_str(), im.w, im.h, im.rgba.data(), (int)im.rgba.size());
}

// The pixel comparison's results, for the browser test:
// Module.caliperDrawParity = {id: {code, message, where, line, column, color: Uint8Array, ids: Uint8Array}}.
void publish_parity() {
    EM_ASM({ Module["caliperDrawParity"] = {}; });
    for (const caliper_web::ParityResult& r : caliper_web::draw_parity_results())
        EM_ASM({
            Module["caliperDrawParity"][UTF8ToString($0)] = ({
                "code": UTF8ToString($1), "message": UTF8ToString($2), "where": UTF8ToString($3),
                "line": $4, "column": $5,
                "color": HEAPU8.slice($6, $6 + $7), "ids": HEAPU8.slice($8, $8 + $9)});
        }, r.id.c_str(), r.code.c_str(), r.message.c_str(), r.where.c_str(), (int)r.line, (int)r.column,
           r.color.data(), (int)r.color.size(), r.ids.data(), (int)r.ids.size());
}

// The draw halves of pieces: the finished browser checks and the refusals that WebGPU reported,
// then each scene view whose name is a program's panel gets that panel's feed.
void pump_programs() {
    if (!g.programs || !g.renderer || !g.renderer->gfx()) return;
    g.programs->pump(g.renderer->gfx()->device_errors());
    for (const std::string& panel : g.programs->panels())
        if (caliper_live_views::SceneView* sv = caliper_live_views::scene_view(g.core, panel.c_str()))
            sv->set_draw_feed(g.programs->feed(panel));
    if (g.programs->revision() != g.programs_published) {
        g.programs_published = g.programs->revision();
        g.publish_soon = true;
    }
}

// The panel's starting camera from its options `view_camera` and `views`, as the app's spec runtime
// reads them (spec_entities.cpp, spec_starting_camera): an object is an orbit with the keys it
// gives, a string names a saved view or else a camera entity, and anything else is the fit. A
// camera that the checks of scenes.md refuse (not finite, a distance not above 0, an elevation
// outside -89 to 89 degrees) is the fit too, as the spec check refuses it in the app.
caliper_scene::StartingCamera page_starting_camera(const caliper_net::Json& scene) {
    using caliper_scene::StartingCamera;
    StartingCamera s;
    const caliper_net::Json* v = scene.is_object() ? scene.find("view_camera") : nullptr;
    auto num = [](const caliper_net::Json& o, const char* key, std::optional<double>* out) {
        if (const caliper_net::Json* n = o.find(key); n && n->is_number()) *out = n->as_double();
    };
    auto vec3 = [](const caliper_net::Json* t, caliper_scene::Vec3* out) {
        if (!t || !t->is_array() || t->size() != 3) return false;
        for (size_t k = 0; k < 3; ++k)
            if (!t->at(k).is_number()) return false;
        *out = caliper_scene::Vec3{t->at(0).as_double(), t->at(1).as_double(), t->at(2).as_double()};
        return true;
    };
    if (v && v->is_object() && v->size() > 0) {
        s.form = StartingCamera::Form::Orbit;
        caliper_scene::Vec3 t;
        if (vec3(v->find("target"), &t)) s.target = t;
        std::optional<double> d;
        num(*v, "distance", &d);
        if (d && *d > 0) s.distance = d;
        num(*v, "azimuth", &s.azimuth_deg);
        num(*v, "elevation", &s.elevation_deg);
    } else if (v && v->is_string()) {
        const std::string name = v->as_string();
        const caliper_net::Json* views = scene.find("views");
        for (size_t i = 0; views && views->is_array() && i < views->size() && i < 64; ++i) {
            const caliper_net::Json& sv = views->at(i);
            const caliper_net::Json* n = sv.is_object() ? sv.find("name") : nullptr;
            if (!n || !n->is_string() || n->as_string() != name) continue;
            s.form = StartingCamera::Form::Orbit;
            caliper_scene::Vec3 t{0, 0, 0};
            vec3(sv.find("target"), &t);
            s.target = t;
            s.distance = 1.0;
            s.azimuth_deg = 0.0;
            s.elevation_deg = 0.0;
            num(sv, "distance", &s.distance);
            num(sv, "azimuth", &s.azimuth_deg);
            num(sv, "elevation", &s.elevation_deg);
            break;
        }
        if (s.form != StartingCamera::Form::Orbit) {
            s.form = StartingCamera::Form::Entity;
            s.entity = name;
        }
    }
    if (!caliper_scene::validate_starting_camera(s, caliper_scene::Space::k3D)) return StartingCamera{};
    return s;
}

// feed "page": the panel's scene over the whole canvas, drawn as the app's spec runtime draws
// a scene panel's content, with the page's programs. A pick or a selection goes to the page.
void draw_page_scene(float w, float h, float scale) {
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(w, h));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    // The scene clears to the window's colour: the ground of a spec panel, as in the app.
    ImGui::PushStyleColor(ImGuiCol_WindowBg, sternbild::col::ground);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollWithMouse;
    if (ImGui::Begin("##caliper.page.scene", nullptr, flags)) {
        g.page_options.draw_feed = g.programs ? g.programs->feed(g.cfg.view) : nullptr;
        auto sink = [](const char* type, const std::string& detail) {
            ++g.page_events;
            g.page_last_event = std::string(type) + " " + detail.substr(0, 200);
            EM_ASM({
                var cb = Module["onEvent"];
                if (cb) cb(UTF8ToString($0), UTF8ToString($1), JSON.parse(UTF8ToString($2)));
            }, g.cfg.view.c_str(), type, detail.c_str());
            g.publish_soon = true;
        };
        caliper_live_views::draw_scene_view_in_rect(*g.page_view, scale, "##page.scene", ImVec2(0.0f, 0.0f), ImVec2(w, h),
                                                    g.page_options, g.page_state, sink);
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void frame() {
    const double t0 = emscripten_get_now();
    if (caliper_web::gfx_selftest_running() && g.renderer && g.renderer->gfx() &&
        caliper_web::gfx_selftest_poll(*g.renderer->gfx(), &g.selftest)) {
        publish_selftest_images();
        publish();
    }
    if (caliper_web::scene_selftest_running() && g.renderer && g.renderer->gfx() &&
        caliper_web::scene_selftest_poll(*g.renderer->gfx(), &g.scene_selftest)) {
        const std::vector<uint8_t>& px = caliper_web::scene_selftest_pixels();
        EM_ASM({ Module["caliperScenePixels"] = HEAPU8.slice($0, $0 + $1); }, px.data(), (int)px.size());
        publish();
    }

    if (caliper_web::points_parity_running() && g.renderer && g.renderer->gfx() &&
        caliper_web::points_parity_poll(*g.renderer->gfx(), &g.points_parity))
        publish();

    if (caliper_web::draw_parity_running() && g.renderer && g.renderer->gfx() &&
        caliper_web::draw_parity_poll(*g.renderer->gfx(), g.renderer->gfx()->device_errors(), &g.draw_parity)) {
        publish_parity();
        publish();
    }
    // The scene comparison's targets and facts: Module.caliperSceneParity = {facts, color, ids}.
    if (caliper_web::scene_parity_running() && g.renderer && g.renderer->gfx() &&
        caliper_web::scene_parity_poll(*g.renderer->gfx(), &g.scene_parity)) {
        const caliper_web::SceneParityResult& r = caliper_web::scene_parity_result();
        EM_ASM({
            Module["caliperSceneParity"] = ({"facts": UTF8ToString($0), "color": HEAPU8.slice($1, $1 + $2),
                                             "ids": HEAPU8.slice($3, $3 + $4)});
        }, r.facts.c_str(), r.color.data(), (int)r.color.size(), r.ids.data(), (int)r.ids.size());
        publish();
    }

    if (g.net && caliper_web::poll_source(g.core, *g.net, g.poll) > 0 && !g.ready) {
        g.ready = true;   // the first snapshot is drawn this frame
        EM_ASM({
            var f = Module["onReady"];
            if (f) f();
        });
    }
    pump_programs();
    const double t_poll = emscripten_get_now();

    const caliper_web::FrameSize fs = caliper_web::input_new_frame();
    if (fs.fb_w <= 0 || fs.fb_h <= 0) {   // hidden (no CSS box): nothing to draw
        g.renderer->release_retired();    // what this poll's applies released
        return;
    }
    const float scale = (float)fs.fb_w / fs.css_w;
    g.css_w = fs.css_w;
    g.css_h = fs.css_h;
    g.fb_w = fs.fb_w;
    g.fb_h = fs.fb_h;

    g.renderer->new_frame();
    ImGui::NewFrame();
    if (g.page_view) {
        draw_page_scene(fs.css_w, fs.css_h, scale);
    } else {
        pin_single_view(fs.css_w, fs.css_h);
        caliper_core_hosted_frame(g.core, 0.0f, 0.0f, fs.css_w, fs.css_h, scale);
        draw_status(fs.css_w, fs.css_h);
    }
    ImGui::Render();
    caliper_web::input_end_frame();
    const double t_ui = emscripten_get_now();
    g.renderer->render(fs.fb_w, fs.fb_h);

    const double t1 = emscripten_get_now();
    ++g.frames;
    if (g.frames > 30) {   // skip start-up frames (pipeline creation, first uploads)
        g.cpu_sum += t1 - t0;
        g.poll_sum += t_poll - t0;
        g.ui_sum += t_ui - t_poll;
        g.enc_sum += t1 - t_ui;
        ++g.cpu_n;
    }
    if (g.frames == 30) { g.t_mark = t1; g.frames_mark = g.frames; }
    if (g.frames > 30 && t1 > g.t_mark)
        g.fps = (double)(g.frames - g.frames_mark) * 1000.0 / (t1 - g.t_mark);
    // From frame 10 on (past the start-up frames that create the pipelines),
    // one view is checked every 5 frames, so that the checks do not wait on a
    // frame rate: on a software adapter a frame can take most of a second.
    if (!g.check_pending && g.frames >= g.check_next) {
        g.check_next = g.frames + 5;
        start_check();
    }
    send_infos();
    // The counters every 15 frames, and past the start-up frames every 250 ms
    // as well when frames are slower. Not sooner: the views' windows are laid
    // out over the first frames, and a page reads the first geometry published
    // as the settled one.
    if (g.frames % 15 == 0 || g.publish_soon || (g.frames > 30 && t1 - g.last_publish >= 250.0)) {
        publish();
        g.last_publish = t1;
    }
    g.publish_soon = false;
}

void start() {
    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    if (wgpuAdapterGetInfo(g.adapter, &info) == WGPUStatus_Success) {
        auto s = [](WGPUStringView v) {
            return v.data ? std::string(v.data, v.length == WGPU_STRLEN ? std::strlen(v.data) : v.length)
                          : std::string();
        };
        g.adapter_text = "vendor=" + s(info.vendor) + " architecture=" + s(info.architecture) +
                         " device=" + s(info.device) + " description=" + s(info.description);
        wgpuAdapterInfoFreeMembers(info);
    }

    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;   // no file system in the page
    // The native app's look (src/main.cpp): its faces, then its style.
    const caliper_web::EmbeddedFont ui = caliper_web::ui_font(), mono = caliper_web::mono_font();
    sternbild::load_fonts_from_memory(ui.data, ui.size, mono.data, mono.size);
    sternbild::apply_style();
    caliper_web::apply_page_style(!g.cfg.header);
    caliper_web::input_install();

    auto r = std::make_unique<WebGpuRenderer>(g.instance, g.adapter, g.device, "!canvas");
    if (!r->init(nullptr)) return fail("WebGPU renderer init failed");
    g.renderer = r.get();
    caliper_host::services_set_renderer(g.renderer);
    g.core = caliper_core_create_hosted(r.release());
    if (!g.core) return fail("caliper_core_create_hosted refused");
    g.core->views_bare = !g.cfg.header;
    caliper_live_views::set_event_sink(g.core, on_event);
    g.programs = std::make_unique<caliper_web::WebPrograms>(g.device);

    if (g.cfg.page_feed) {
        // The scene of a panel that holds programs alone, as the app's spec runtime makes it
        // (spec_runtime.cpp, entity_view): a scene view of its own on a backend of the graphics
        // layer, fitted to its box, with the panel's labels and scale bar.
        caliper_net::Json scene;
        std::string err;
        if (!g.cfg.scene.empty() && !caliper_net::Json::parse(g.cfg.scene.data(), g.cfg.scene.size(), &scene, &err))
            return fail("caliperConfig.scene is not JSON: " + err);
        std::string up = "+z";   // the panel's option up, +z as the app's spec runtime takes it
        if (scene.is_object()) {
            if (const caliper_net::Json* l = scene.find("labels"); l && l->is_bool()) g.page_options.labels = l->as_bool();
            if (const caliper_net::Json* b = scene.find("scale_bar"); b && b->is_bool()) g.page_options.scale_bar = b->as_bool();
            if (const caliper_net::Json* u = scene.find("up"); u && u->is_string()) up = u->as_string();
        }
        if (g.cfg.view.empty()) return fail("feed page needs caliperConfig.view, the panel's id");
        caliper_gfx::Backend* backend = nullptr;
        const caliper_gfx::ProgramDesc* table = nullptr;
        std::string why;
        if (!caliper_live_views::open_layer(g.core, "scenes", &g.page_gfx, &backend, &table, &why))
            return fail("the scene cannot start here: " + why);
        auto view = std::make_unique<caliper_live_views::SceneView>(*backend, table);
        if (!view->init(&why)) return fail("the scene view did not start: " + why);
        view->set_box_fit(true);
        // A 3-D scene with the panel's up axis and the panel's starting camera, as apply_entity_scene
        // describes it and SpecRuntime::entity_view starts it.
        caliper_net::wire::SceneView info;
        info.space = CALIPER_NET_SPACE_3D;
        info.up = up;
        info.units = "m";
        view->describe(info);
        view->set_starting(page_starting_camera(scene));
        g.page_view = std::move(view);
        g.ready = true;
        EM_ASM({
            var f = Module["onReady"];
            if (f) f();
        });
    } else if (g.cfg.src.empty()) {
        fail("no stream: set caliperConfig.src to the ws:// URL of a Caliper stream");
    } else {
        NetSource::Options o;
        o.url = g.cfg.src;
        o.view = g.cfg.view;
        o.kind = g.cfg.kind;
        NetSource::Callbacks cb;
        cb.data_view = [](const std::string& view, const std::string& json) {
            EM_ASM({
                var f = Module["onDataView"];
                if (f) f(UTF8ToString($0), JSON.parse(UTF8ToString($1)));
            }, view.c_str(), json.c_str());
        };
        cb.program = [](const caliper_net::wire::Program& m) {
            if (g.programs) g.programs->on_program(m);
        };
        cb.resource = [](const caliper_net::wire::Resource& r) {
            if (g.programs) g.programs->on_resource(r);
        };
        g.net = std::make_unique<NetSource>(o, cb);
        g.poll.relative_stamps = true;   // live_clock.h: latency above the fastest message
        g.net->connect();
    }

    publish();
    emscripten_set_main_loop(frame, 0, false);
}

void on_device(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView msg, void*, void*) {
    if (status != WGPURequestDeviceStatus_Success)
        return fail(std::string("requestDevice failed: ") + (msg.data ? msg.data : ""));
    g.device = device;
    start();
}

void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView msg, void*,
                void*) {
    if (status != WGPURequestAdapterStatus_Success)
        return fail(std::string("requestAdapter failed: ") + (msg.data ? msg.data : ""));
    g.adapter = adapter;
    WGPUDeviceDescriptor dd = WGPU_DEVICE_DESCRIPTOR_INIT;
    dd.uncapturedErrorCallbackInfo.callback = [](WGPUDevice const*, WGPUErrorType type,
                                                 WGPUStringView m, void*, void*) {
        std::fprintf(stderr, "[webgpu] uncaptured error %d: %.*s\n", (int)type,
                     (int)(m.length == WGPU_STRLEN ? std::strlen(m.data) : m.length), m.data);
        g.error = "WebGPU validation error (see the console)";
    };
    dd.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    dd.deviceLostCallbackInfo.callback = [](WGPUDevice const*, WGPUDeviceLostReason reason,
                                            WGPUStringView m, void*, void*) {
        if (g.destroyed) return;   // caliper_views_destroy() destroyed it
        std::fprintf(stderr, "[webgpu] device lost (%d): %.*s\n", (int)reason,
                     (int)(m.length == WGPU_STRLEN ? std::strlen(m.data) : m.length), m.data);
    };
    WGPURequestDeviceCallbackInfo cb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = on_device;
    wgpuAdapterRequestDevice(adapter, &dd, cb);
}

}  // namespace

// ---- calls from JavaScript --------------------------------------------------------
extern "C" {

// A new data view (a JSON object) for the shown view. 1 when it was sent.
EMSCRIPTEN_KEEPALIVE int caliper_views_set_data_view(const char* json) {
    if (!g.net || !json) return 0;
    std::string err;
    if (!g.net->request_data_view(json, &err)) {
        std::fprintf(stderr, "[caliper-views] set_data_view: %s\n", err.c_str());
        return 0;
    }
    return 1;
}

// A header toggle ("follow" or "log") of the shown view, or of every view
// when the instance shows them all. 1 when a view took it.
EMSCRIPTEN_KEEPALIVE int caliper_views_set_toggle(const char* name, int on) {
    if (!g.core || !g.core->live_views || !name) return 0;
    int n = 0;
    for (const auto& v : g.core->live_views->views)
        if ((g.cfg.view.empty() || v->name == g.cfg.view) &&
            caliper_live_views::set_toggle(g.core, v->name.c_str(), name, on != 0))
            ++n;
    g.publish_soon = true;
    return n > 0 ? 1 : 0;
}

// The linked selection of the shown view (or of every view): the axis that
// carries the selection's item space ("points", "rows", "columns", "steps"),
// or "" to clear, and the items as a JSON array of integers. 1 when a view took it.
EMSCRIPTEN_KEEPALIVE int caliper_views_set_link(const char* axis, const char* items_json) {
    if (!g.core || !g.core->live_views) return 0;
    std::vector<int64_t> items;
    caliper_net::Json j;
    if (items_json && caliper_net::Json::parse(items_json, std::strlen(items_json), &j, nullptr) && j.is_array())
        for (size_t i = 0; i < j.size(); ++i) {
            int64_t x = 0;
            if (j.at(i).get_i64(&x)) items.push_back(x);
        }
    int n = 0;
    for (const auto& v : g.core->live_views->views)
        if ((g.cfg.view.empty() || v->name == g.cfg.view) &&
            caliper_live_views::set_link(g.core, v->name.c_str(), axis, items))
            ++n;
    return n > 0 ? 1 : 0;
}

// Starts the graphics layer's self-test (gfx_selftest.h); its report comes in
// caliperStats.gfxSelftest. 1 when it started.
EMSCRIPTEN_KEEPALIVE int caliper_views_gfx_selftest() {
    if (!g.renderer || !g.renderer->gfx()) {
        g.selftest = "fail not started: the renderer is not up";
        publish();
        return 0;
    }
    std::string why;
    if (!caliper_web::gfx_selftest_start(*g.renderer->gfx(), &why)) {
        g.selftest = "fail not started: " + why;
        publish();
        return 0;
    }
    g.selftest.clear();
    publish();
    return 1;
}

// Starts the scene row of fixture `name` (scene_selftest.h); its report comes in
// caliperStats.sceneSelftest and its picture in Module.caliperScenePixels (256 x 256 RGBA8).
// caliperStats.sceneFixtures names the fixtures. 1 when it started.
EMSCRIPTEN_KEEPALIVE int caliper_views_scene_selftest(const char* name) {
    put("sceneFixtures", caliper_web::scene_selftest_names().c_str());
    if (!g.renderer || !g.renderer->gfx()) {
        g.scene_selftest = "fail not started: the renderer is not up";
        publish();
        return 0;
    }
    std::string why;
    if (!caliper_web::scene_selftest_start(*g.renderer->gfx(), caliper_gfx::shaders::kWebgpuPrograms,
                                           name ? name : "", &why)) {
        g.scene_selftest = "fail not started: " + why;
        publish();
        return 0;
    }
    g.scene_selftest.clear();
    publish();
    return 1;
}

// Starts the browser's side of the scene comparison (scene_parity.h) from its config JSON: the
// fixture, the switch of the shapes and the points to pick at. The report comes in
// caliperStats.sceneParity and the targets and facts in Module.caliperSceneParity. 1 when it started.
EMSCRIPTEN_KEEPALIVE int caliper_views_scene_parity(const char* config) {
    if (!g.renderer || !g.renderer->gfx()) {
        g.scene_parity = "fail not started: the renderer is not up";
        publish();
        return 0;
    }
    std::string why;
    if (!caliper_web::scene_parity_start(*g.renderer->gfx(), caliper_gfx::shaders::kWebgpuPrograms,
                                         config ? config : "", &why)) {
        g.scene_parity = "fail not started: " + why;
        publish();
        return 0;
    }
    g.scene_parity.clear();
    publish();
    return 1;
}

// Starts the browser's side of the pixel comparison (draw_parity.h) on the fixture that the app's
// side wrote: `frames` (n bytes of length-prefixed PROGRAM and RESOURCE frames) and `plan` (its
// JSON). The report comes in caliperStats.drawParity and the targets in Module.caliperDrawParity.
// 1 when it started.
EMSCRIPTEN_KEEPALIVE int caliper_views_draw_parity(const uint8_t* frames, int n, const char* plan) {
    if (!g.renderer || !g.renderer->gfx() || !g.device || n < 0) {
        g.draw_parity = "fail not started: the renderer is not up";
        publish();
        return 0;
    }
    std::string why;
    if (!caliper_web::draw_parity_start(*g.renderer->gfx(), g.device, frames, size_t(n), plan ? plan : "", &why)) {
        g.draw_parity = "fail not started: " + why;
        publish();
        return 0;
    }
    g.draw_parity.clear();
    publish();
    return 1;
}

// Starts the comparison of initial and replacement points@1 payloads
// (points_parity.h); its report comes in caliperStats.pointsParity. 1 when it started.
EMSCRIPTEN_KEEPALIVE int caliper_views_points_parity() {
    if (!g.renderer || !g.renderer->gfx()) {
        g.points_parity = "fail not started: the renderer is not up";
        publish();
        return 0;
    }
    std::string why;
    if (!caliper_web::points_parity_start(*g.renderer->gfx(), caliper_gfx::shaders::kWebgpuPrograms, &why)) {
        caliper_web::points_parity_stop();
        g.points_parity = "fail not started: " + why;
        publish();
        return 0;
    }
    g.points_parity.clear();
    publish();
    return 1;
}

// The colormap keys that this build derives at run time for the range (vmin, vmax), from its own
// CPU reference (src/host/gfx/colormap_keys.h), as the scene views derive them: the browser test
// compares them with the case file that the native tool writes. Module.caliperColormapKeys =
// Uint32Array [flip, key[0], ..., key[256]]. Returns 1.
EMSCRIPTEN_KEEPALIVE int caliper_views_colormap_keys(float vmin, float vmax) {
    const caliper_gfx::cmap::Keys k = caliper_gfx::cmap::derive_keys(vmin, vmax);
    uint32_t out[1 + caliper_gfx::cmap::kKeyCount];
    out[0] = k.flip;
    std::memcpy(out + 1, k.key, sizeof k.key);
    EM_ASM({ Module["caliperColormapKeys"] = new Uint32Array(HEAPU8.slice($0, $0 + 4 * $1).buffer); }, out,
           (int)(1 + caliper_gfx::cmap::kKeyCount));
    return 1;
}

// Ends this instance: BYE to the producer, no more frames, every view, GPU
// resource and context released, the input listeners removed. The page drops
// the canvas and the module afterwards; an instance is not reused.
// feed "page": one PROGRAM of an instance of a piece, as JSON {op, id, piece,
// manifest, bind, params, panel, segments, common, derive, draw} with op
// "set", "params" or "remove", which the module checks as it checks a relay
// of the stream (web_programs.h: the pin's hash, the manifest, the files and
// WebGPU's compile). 1 when it was handed on, 0 for JSON that is no object or
// an op or id that is not a string.
EMSCRIPTEN_KEEPALIVE int caliper_views_feed_program(const char* json) {
    if (!g.cfg.page_feed || !g.programs || !json) return 0;
    caliper_net::Json j;
    std::string err;
    const size_t n = std::strlen(json);
    if (n > (size_t)64 << 20 || !caliper_net::Json::parse(json, n, &j, &err) || !j.is_object()) return 0;
    const caliper_net::Json* op = j.find("op");
    const caliper_net::Json* id = j.find("id");
    if (!op || !op->is_string() || !id || !id->is_string()) return 0;
    caliper_net::wire::Program m;
    m.op = op->as_string();
    m.id = id->as_string();
    if (m.op != CALIPER_NET_PROGRAM_SET && m.op != CALIPER_NET_PROGRAM_PARAMS && m.op != CALIPER_NET_PROGRAM_REMOVE)
        return 0;
    auto obj = [&](const char* k, caliper_net::Json* out) {
        if (const caliper_net::Json* v = j.find(k)) *out = *v;
    };
    auto text = [&](const char* k, std::optional<std::string>* out) {
        if (const caliper_net::Json* v = j.find(k); v && v->is_string()) *out = v->as_string();
    };
    obj("piece", &m.piece);
    obj("manifest", &m.manifest);
    obj("bind", &m.bind);
    obj("segments", &m.segments);
    obj("params", &m.params);
    text("panel", &m.panel);
    text("common", &m.common);
    text("derive", &m.derive);
    text("draw", &m.draw);
    g.programs->on_program(m);
    return 1;
}

// feed "page": one version of a published resource: `n` bytes of `count`
// elements of `stride` bytes, or a texture of width by height texels. The
// module keeps it only when the piece of its program publishes that resource
// within its declared count and bytes (web_programs.h).
EMSCRIPTEN_KEEPALIVE int caliper_views_feed_resource(const char* program, const char* resource, double step,
                                                    uint32_t count, uint32_t stride, uint32_t width, uint32_t height,
                                                    const uint8_t* bytes, uint32_t n, const char* format, double elements) {
    if (!g.cfg.page_feed || !g.programs || !program || !resource || (!bytes && n)) return 0;
    if (!(step >= 0.0) || step > 9.007199254740992e15) return 0;
    const uint64_t limit = width || height ? CALIPER_NET_RESOURCE_TEXTURE_VERSION_MAX_BYTES
                                          : CALIPER_NET_RESOURCE_VERSION_MAX_BYTES;
    if ((uint64_t)count * stride != n || n > limit) return 0;
    if (elements != -1.0 && (!(elements >= 0.0) || elements > UINT32_MAX || elements != std::floor(elements))) return 0;
    caliper_net::wire::Resource r;
    r.program = program;
    r.resource = resource;
    r.step = (uint64_t)step;
    r.count = count;
    r.stride = stride;
    if (width || height) {
        r.width = width;
        r.height = height;
    }
    if (format && *format) r.format = format;
    if (elements >= 0.0) r.elements = uint32_t(elements);
    r.data.name = "resource";
    r.data.dtype = CALIPER_NET_DTYPE_U8;
    r.data.shape = {(int64_t)n};
    r.data.data = bytes;
    r.data.size = n;
    g.programs->on_resource(r);
    return 1;
}

// feed "page": the linked selection of the page's picks, as JSON
// [{program, item, index, newest}], each pick of a program of this module as
// the app's draw host gives it to every program (at most kMaxSelected).
// feed "page": the panel's options view_camera and views changed in the spec, as JSON text
// {view_camera, views}. A new starting camera moves a camera the person has not moved, as in
// the app; the same one changes nothing. Returns 1 when it was read.
EMSCRIPTEN_KEEPALIVE int caliper_views_set_starting(const char* json) {
    if (!g.cfg.page_feed || !g.page_view || !json) return 0;
    caliper_net::Json j;
    std::string err;
    const size_t n = std::strlen(json);
    if (n > (size_t)1 << 16 || !caliper_net::Json::parse(json, n, &j, &err) || !j.is_object()) return 0;
    const caliper_scene::StartingCamera s = page_starting_camera(j);
    const caliper_scene::StartingCamera& now = g.page_view->starting();
    auto same_vec = [](const std::optional<caliper_scene::Vec3>& a, const std::optional<caliper_scene::Vec3>& b) {
        return a.has_value() == b.has_value() && (!a || (a->x == b->x && a->y == b->y && a->z == b->z));
    };
    if (s.form != now.form || !same_vec(s.target, now.target) || s.distance != now.distance ||
        s.azimuth_deg != now.azimuth_deg || s.elevation_deg != now.elevation_deg || s.entity != now.entity)
        g.page_view->set_starting(s);
    return 1;
}

EMSCRIPTEN_KEEPALIVE int caliper_views_select(const char* json) {
    if (!g.cfg.page_feed || !g.programs || !json) return 0;
    caliper_net::Json j;
    std::string err;
    const size_t n = std::strlen(json);
    if (n > (size_t)1 << 16 || !caliper_net::Json::parse(json, n, &j, &err) || !j.is_array()) return 0;
    std::vector<caliper_draw::Selected> sel;
    for (size_t i = 0; i < j.size() && sel.size() < caliper_builder::programs::kMaxSelected; ++i) {
        const caliper_net::Json& e = j.at(i);
        const caliper_net::Json* prog = e.find("program");
        const caliper_net::Json* item = e.find("item");
        uint64_t it = 0;
        if (!prog || !prog->is_string() || !item || !item->get_u64(&it) || it > 0xffffffffull) continue;
        caliper_draw::Selected s;
        s.pick = g.programs->pick_id_of(prog->as_string());
        if (!s.pick) continue;
        s.item = (uint32_t)it;
        if (const caliper_net::Json* nw = e.find("newest"); nw && nw->is_bool() && nw->as_bool()) s.flags = 1u;
        if (const caliper_net::Json* ix = e.find("index"); ix && ix->is_array() && ix->size() == 4)
            for (size_t k = 0; k < 4; ++k) {
                const caliper_net::Json& c = ix->at(k);
                const double v = c.is_number() ? c.as_double() : 0.0;
                if (c.is_number() && v >= -2147483648.0 && v <= 2147483647.0) s.index[k] = (int32_t)v;
            }
        sel.push_back(s);
    }
    g.programs->set_selection(std::move(sel));
    return 1;
}

EMSCRIPTEN_KEEPALIVE void caliper_views_destroy() {
    if (g.destroyed) return;
    g.destroyed = true;
    emscripten_cancel_main_loop();
    if (g.net) g.net->close(CALIPER_NET_BYE_NORMAL);
    caliper_web::input_uninstall();
    g.page_view.reset();                            // the page's scene, before its backend and the feeds
    g.page_gfx.reset();
    if (g.core) {
        caliper_web::remove_polled_views(g.core, g.poll);
        caliper_core_destroy_hosted(g.core);   // releases the views; the renderer is ours
        g.core = nullptr;
    }
    g.programs.reset();                             // the feeds, after the scene views that read them
    caliper_host::services_set_renderer(nullptr);   // drops the bridge before its renderer
    caliper_web::scene_selftest_stop();             // its scene view, before the backend
    caliper_web::draw_parity_stop();                // its runtimes and targets, before the backend
    caliper_web::scene_parity_stop();               // its scene view, before the backend
    caliper_web::points_parity_stop();              // its scenes, before the backend
    delete g.renderer;                              // shutdown(): textures, ImGui backend, surface
    g.renderer = nullptr;
    g.net.reset();
    if (ImPlot::GetCurrentContext()) ImPlot::DestroyContext();
    if (ImGui::GetCurrentContext()) ImGui::DestroyContext();
    if (g.device) {
        wgpuDeviceDestroy(g.device);
        wgpuDeviceRelease(g.device);
        g.device = nullptr;
    }
    if (g.adapter) wgpuAdapterRelease(g.adapter);
    if (g.instance) wgpuInstanceRelease(g.instance);
    g.adapter = nullptr;
    g.instance = nullptr;
}

}  // extern "C"

int main() {
    // "!canvas" names this module instance's own canvas (Module.canvas), so
    // several instances on one page never look a canvas up in the document.
    EM_ASM({ specialHTMLTargets["!canvas"] = Module["canvas"]; });
    g.cfg.src = config_string("src");
    g.cfg.view = config_string("view");
    g.cfg.kind = kind_of(config_string("kind"));
    g.cfg.header = config_string("header") != "false";
    g.cfg.compat = config_string("compat") == "true";
    g.cfg.page_feed = config_string("feed") == "page";
    g.cfg.scene = config_string("scene");

    g.instance = wgpuCreateInstance(nullptr);
    if (!g.instance) {
        fail("wgpuCreateInstance failed (no navigator.gpu?)");
        return 0;
    }
    WGPURequestAdapterOptions opts = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    // A compatibility-mode adapter gives a compatibility-mode device as long as
    // the device does not ask for core-features-and-limits, which it does not.
    if (g.cfg.compat) opts.featureLevel = WGPUFeatureLevel_Compatibility;
    WGPURequestAdapterCallbackInfo cb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = on_adapter;
    wgpuInstanceRequestAdapter(g.instance, &opts, cb);
    return 0;   // the runtime stays alive; the callbacks continue the start-up
}
