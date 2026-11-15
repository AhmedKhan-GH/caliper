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
// settings from Module.caliperConfig: {src, view, kind}.
//
// Start-up is asynchronous because a browser hands out a WebGPU device only
// through promises: main() requests the adapter, the adapter callback requests
// the device, and the device callback builds everything, connects and starts
// the frame loop. No Asyncify is needed.
#include "fonts.h"
#include "net_source.h"
#include "source_poll.h"
#include "web_input.h"

#include "caliper/embed.h"
#include "caliper/live_views.h"
#include "embed/embed_core.h"
#include "host_services.h"
#include "renderer/webgpu_renderer.h"
#include "tensor_bridge.h"   // colormap_lut, map_f32_to_rgba8 (the CPU reference)
#include "ui/sternbild_fonts.h"
#include "ui/sternbild_theme.h"

#include <imgui.h>
#include <implot.h>

#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
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
    int check_state = 0;                  // 0 not run, 1 pending, 2 done
    long long check_mismatch = -1;        // texels that differ
    int check_maxdiff = -1;               // largest channel difference
    std::vector<uint8_t> check_expect;
    uint64_t check_frame = 0;

    float css_w = 0.0f, css_h = 0.0f;     // the canvas's CSS box
    int fb_w = 0, fb_h = 0;               // its backing store
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
    put("dispatches", g.renderer ? (double)g.renderer->colormap_dispatches() : 0.0);
    put("rendererPath", g.renderer ? g.renderer->last_device_path() : "");
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
        std::snprintf(b, sizeof b, "%s%s:kind=%d,updates=%llu,count=%lld,path=%s,w=%d,h=%d,latency=%.2f",
                      per.empty() ? "" : ";", v.name.c_str(), info.kind,
                      (unsigned long long)info.updates, (long long)info.count, info.path, info.width,
                      info.height, info.latency_ms);
        per += b;
    }
    put("viewInfo", per.c_str());
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
    put("checkState", g.check_state);
    put("checkMismatch", (double)g.check_mismatch);
    put("checkMaxDiff", g.check_maxdiff);
    put("adapter", g.adapter_text.c_str());
    put("error", g.error.c_str());
}

void fail(const std::string& why) {
    g.error = why;
    std::fprintf(stderr, "[caliper-views] %s\n", why.c_str());
    publish();
}

// Read the first heatmap's texture back and compare it, byte for byte, with
// the CPU reference map_f32_to_rgba8 over the snapshot it was mapped from.
void start_check() {
    for (auto& tv : g.core->views) {
        const caliper_live::Snapshot* s = g.net ? g.net->shown(tv.name) : nullptr;
        if (!tv.tex || !tv.mapped || !s || s->kind != CALIPER_LIVE_KIND_HEATMAP) continue;
        g.check_expect.assign((size_t)tv.w * tv.h * 4, 0);
        caliper_host::map_f32_to_rgba8((const float*)((const uint8_t*)s->host_base + s->byte_offset),
                                       tv.w, tv.h, caliper_host::colormap_lut(tv.colormap), tv.vmin,
                                       tv.vmax, g.check_expect.data());
        g.check_state = 1;
        g.renderer->readback_async(tv.tex, [](std::vector<uint8_t> got) {
            if (got.size() != g.check_expect.size()) {
                g.check_mismatch = -2;   // readback failed
            } else {
                long long bad = 0;
                int maxd = 0;
                for (size_t i = 0; i < got.size(); i += 4) {
                    int d = 0;
                    for (int c = 0; c < 4; ++c)
                        d = std::max(d, std::abs((int)got[i + c] - (int)g.check_expect[i + c]));
                    if (d) ++bad;
                    maxd = std::max(maxd, d);
                }
                g.check_mismatch = bad;
                g.check_maxdiff = maxd;
            }
            g.check_state = 2;
            publish();
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

void frame() {
    const double t0 = emscripten_get_now();

    if (g.net) {
        g.poll.stamp_offset_ns = g.net->have_clock_offset() ? g.net->clock_offset_ns() : 0;
        caliper_web::poll_source(g.core, *g.net, g.poll);
    }
    const double t_poll = emscripten_get_now();

    const caliper_web::FrameSize fs = caliper_web::input_new_frame();
    if (fs.fb_w <= 0 || fs.fb_h <= 0) return;   // hidden (no CSS box): nothing to draw
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
    // The colormap check runs once, 60 frames after the first heatmap update.
    if (g.check_state == 0 && g.renderer->colormap_dispatches() > 0) {
        if (!g.check_frame) g.check_frame = g.frames + 60;
        else if (g.frames >= g.check_frame) start_check();
    }
    if (g.frames % 15 == 0) publish();
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
    caliper_web::input_install();

    auto r = std::make_unique<WebGpuRenderer>(g.instance, g.adapter, g.device, "!canvas");
    if (!r->init(nullptr)) return fail("WebGPU renderer init failed");
    g.renderer = r.get();
    caliper_host::services_set_renderer(g.renderer);
    g.core = caliper_core_create_hosted(r.release());
    if (!g.core) return fail("caliper_core_create_hosted refused");

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

}  // extern "C"

int main() {
    // "!canvas" names this module instance's own canvas (Module.canvas), so
    // several instances on one page never look a canvas up in the document.
    EM_ASM({ specialHTMLTargets["!canvas"] = Module["canvas"]; });
    g.cfg.src = config_string("src");
    g.cfg.view = config_string("view");
    g.cfg.kind = kind_of(config_string("kind"));

    g.instance = wgpuCreateInstance(nullptr);
    if (!g.instance) {
        fail("wgpuCreateInstance failed (no navigator.gpu?)");
        return 0;
    }
    WGPURequestAdapterOptions opts = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    WGPURequestAdapterCallbackInfo cb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = on_adapter;
    wgpuInstanceRequestAdapter(g.instance, &opts, cb);
    return 0;   // the runtime stays alive; the callbacks continue the start-up
}
