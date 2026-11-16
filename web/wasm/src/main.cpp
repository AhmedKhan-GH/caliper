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
// settings from Module.caliperConfig: {src, view, kind, header, compat};
// compat "true" asks for a WebGPU device in compatibility mode.
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
#include "fonts.h"
#include "gfx_selftest.h"
#include "net_source.h"
#include "net/json.h"
#include "page_style.h"
#include "source_poll.h"
#include "web_input.h"

#include "caliper/embed.h"
#include "caliper/live_net.h"   // CALIPER_NET_BYE_NORMAL
#include "caliper/live_views.h"
#include "embed/embed_core.h"
#include "live/live_views_state.h"   // the points check reads the view's camera and payload
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
};

struct Page {
    Config cfg;
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WebGpuRenderer* renderer = nullptr;   // owned by the hosted core
    CaliperCore* core = nullptr;
    std::unique_ptr<NetSource> net;
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
    uint64_t check_next = 90;             // the frame of the next check

    std::string selftest;                 // the graphics layer's self-test: "" or its report

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
    // "selected": n, "hover": item, "cols": n, "follow": bool, "log": bool}}.
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
                          "\"yDown\":%s,\"selected\":%.0f,\"hover\":%lld,\"cols\":%d,\"follow\":%s,\"log\":%s}",
                          geo.size() > 1 ? "," : "", v->name.c_str(), v->geo_area[0], v->geo_area[1],
                          v->geo_area[2], v->geo_area[3], v->geo_window[0], v->geo_window[1], v->geo_window[2],
                          v->geo_window[3], v->geo_ydown ? "true" : "false", sel, (long long)v->hover_item, cols,
                          v->follow ? "true" : "false", v->log_y ? "true" : "false");
            geo += b;
        }
    geo += "}";
    put("geometry", geo.c_str());
    put("selected", selected);
    put("events", (double)g.events_sent);
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
            if (v->kind == CALIPER_LIVE_KIND_POINTS && v->label_count > 0 && v->view_rid &&
                unchecked(v->name))
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
// colour of its label. Records mismatches and, in maxdiff, how many points were
// compared. False when there is nothing to check yet.
bool start_points_check(caliper_live_views::View& v) {
    if (!v.have_cur || !v.cur.pos_cpu || !v.cur.aux_cpu || !v.view_rid || v.dirty) return false;
    const int W = v.tex_w, H = v.tex_h;
    const uint64_t n = v.cur.n;
    float V[16], P[16], M[16];
    caliper_live_views::camera_matrices(v, (float)W / (float)H, V, P);
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float acc = 0.0f;
            for (int k = 0; k < 4; ++k) acc += P[k * 4 + r] * V[c * 4 + k];
            M[c * 4 + r] = acc;
        }
    struct Proj { float x, y, z; bool ok; };
    std::vector<Proj> pr(n);
    const float* pos = (const float*)v.cur.pos_cpu;
    for (uint64_t i = 0; i < n; ++i) {
        const float x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
        const float cw = M[3] * x + M[7] * y + M[11] * z + M[15];
        if (!(cw > 1e-9f)) { pr[i] = {0, 0, 0, false}; continue; }
        pr[i] = {((M[0] * x + M[4] * y + M[8] * z + M[12]) / cw * 0.5f + 0.5f) * (float)W,
                 (0.5f - (M[1] * x + M[5] * y + M[9] * z + M[13]) / cw * 0.5f) * (float)H,
                 (M[2] * x + M[6] * y + M[10] * z + M[14]) / cw, true};
    }
    const float half = 0.5f * caliper_live_views::point_size_px(n, g.core->scale) + 1.0f;
    struct Want { int px, py; uint32_t rgba; };
    auto wants = std::make_shared<std::vector<Want>>();
    const float* aux = (const float*)v.cur.aux_cpu;
    // Up to 1,024 candidates, evenly spread, of which those no other point
    // covers are compared (at most 256).
    const uint64_t stride = std::max<uint64_t>(1, n / 1024);
    for (uint64_t i = 0; i < n && wants->size() < 256; i += stride) {
        const Proj& a = pr[i];
        if (!a.ok || a.x < 1 || a.y < 1 || a.x >= W - 1 || a.y >= H - 1 || a.z < 0 || a.z > 1) continue;
        bool covered = false;
        for (uint64_t j = 0; j < n && !covered; ++j) {
            if (j == i || !pr[j].ok) continue;
            if (std::fabs(pr[j].x - a.x) > 2 * half || std::fabs(pr[j].y - a.y) > 2 * half) continue;
            covered = v.dims == 2 ? j > i : (pr[j].z <= a.z + 1e-6f);
        }
        if (covered) continue;
        const float t = std::min(std::max((aux[i] - v.pvmin) / (v.pvmax - v.pvmin), 0.0f), 1.0f);
        wants->push_back({(int)a.x, (int)a.y, v.lut[(uint32_t)(t * 255.0f + 0.5f)]});
    }
    if (wants->empty()) return false;
    g.check_pending = true;
    const std::string name = v.name;
    g.renderer->readback_async(v.view_rid, [wants, name, W, H](std::vector<uint8_t> got) {
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

void frame() {
    const double t0 = emscripten_get_now();
    if (caliper_web::gfx_selftest_running() && g.renderer && g.renderer->gfx() &&
        caliper_web::gfx_selftest_poll(*g.renderer->gfx(), &g.selftest)) {
        publish_selftest_images();
        publish();
    }

    if (g.net && caliper_web::poll_source(g.core, *g.net, g.poll) > 0 && !g.ready) {
        g.ready = true;   // the first snapshot is drawn this frame
        EM_ASM({
            var f = Module["onReady"];
            if (f) f();
        });
    }
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
    pin_single_view(fs.css_w, fs.css_h);
    caliper_core_hosted_frame(g.core, 0.0f, 0.0f, fs.css_w, fs.css_h, scale);
    draw_status(fs.css_w, fs.css_h);
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
    // From frame 90 on, one texture view is checked every 30 frames.
    if (!g.check_pending && g.frames >= g.check_next) {
        g.check_next = g.frames + 30;
        start_check();
    }
    send_infos();
    if (g.frames % 15 == 0 || g.publish_soon) publish();
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

    if (g.cfg.src.empty()) {
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

// Ends this instance: BYE to the producer, no more frames, every view, GPU
// resource and context released, the input listeners removed. The page drops
// the canvas and the module afterwards; an instance is not reused.
EMSCRIPTEN_KEEPALIVE void caliper_views_destroy() {
    if (g.destroyed) return;
    g.destroyed = true;
    emscripten_cancel_main_loop();
    if (g.net) g.net->close(CALIPER_NET_BYE_NORMAL);
    caliper_web::input_uninstall();
    if (g.core) {
        caliper_web::remove_polled_views(g.core, g.poll);
        caliper_core_destroy_hosted(g.core);   // releases the views; the renderer is ours
        g.core = nullptr;
    }
    caliper_host::services_set_renderer(nullptr);   // drops the bridge before its renderer
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
