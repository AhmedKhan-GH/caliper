// web_main.cpp: the page host of the web viewer spike
// (docs/plans/web-viewer-spike.md). It plays the part the Caliper app plays
// natively: it owns the window (a canvas through Emscripten's GLFW), the ImGui
// and ImPlot contexts and the renderer, and it draws Caliper's real view layer
// through a hosted core (caliper_core_create_hosted). A synthetic source
// compiled in stands in for the wire protocol client: every frame it applies a
// heatmap snapshot (a moving 2-D pattern) and a series snapshot (a noisy
// decaying curve) through caliper_core_live_apply, exactly as the shared-memory
// transport does natively.
//
// Start-up is asynchronous because a browser hands out a WebGPU device only
// through promises: main() requests the adapter, the adapter callback requests
// the device, and the device callback builds everything and starts the frame
// loop. No Asyncify is needed.
#include "webgpu_renderer.h"

#include "caliper/embed.h"
#include "caliper/live_views.h"
#include "embed/embed_core.h"
#include "host_services.h"
#include "live/live_views_state.h"   // caliper_live_views::find (no error logging)
#include "tensor_bridge.h"   // colormap_lut, map_f32_to_rgba8 (the CPU reference)

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <implot.h>

#include <GLFW/glfw3.h>
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using caliper_host::WebGpuRenderer;

namespace {

// ---- the synthetic source ----------------------------------------------------
// Heatmap size: 256x192 (non-square, so a transpose would show), or the page
// URL's ?w=&h= for the timing runs.
int g_heat_w = 256, g_heat_h = 192;
constexpr uint64_t kRing = 4096;              // series ring capacity (values)
constexpr int kStepsPerFrame = 4;             // series values appended per frame

struct Synthetic {
    std::vector<float> heat;
    std::vector<float> ring = std::vector<float>(kRing);
    uint64_t total = 0;                       // series values appended so far
    uint64_t seq = 0;
    uint32_t rng = 12345u;
    float stats[2] = {0.0f, 1.0f};            // the producer's min/max of the heatmap

    float uniform() {                         // [0,1), deterministic
        rng = rng * 1664525u + 1013904223u;
        return (float)(rng >> 8) * (1.0f / 16777216.0f);
    }
    float gauss() {                           // ~N(0,1) from four uniforms
        return (uniform() + uniform() + uniform() + uniform() - 2.0f) * 1.7320508f;
    }

    // A travelling interference pattern plus a Gaussian blob on a circle.
    void step_heatmap(double t) {
        heat.resize((size_t)g_heat_w * g_heat_h);
        const float cx = g_heat_w * (0.5f + 0.3f * (float)std::cos(t * 0.9));
        const float cy = g_heat_h * (0.5f + 0.3f * (float)std::sin(t * 0.9));
        float lo = INFINITY, hi = -INFINITY;
        for (int y = 0; y < g_heat_h; ++y) {
            for (int x = 0; x < g_heat_w; ++x) {
                const float dx = (x - cx) / 22.0f, dy = (y - cy) / 22.0f;
                const float v = 0.6f * std::sin(0.07f * x + 1.4f * (float)t) *
                                    std::cos(0.09f * y - 0.8f * (float)t) +
                                0.3f * std::sin(0.025f * (x + 2 * y) + 0.5f * (float)t) +
                                1.4f * std::exp(-(dx * dx + dy * dy));
                heat[(size_t)y * g_heat_w + x] = v;
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
        }
        stats[0] = lo;
        stats[1] = hi;
    }

    // A training-loss-like curve: exponential decay to a floor, with noise.
    void step_series() {
        for (int k = 0; k < kStepsPerFrame; ++k) {
            const double s = (double)total;
            const float v = (float)(2.2 * std::exp(-s / 450.0) + 0.12 +
                                    0.06 * std::exp(-s / 1500.0) * gauss() + 0.01 * gauss());
            ring[total % kRing] = v;
            ++total;
        }
    }
};

CaliperTensor cpu_tensor(void* data, int ndim, const int64_t* shape) {
    CaliperTensor t{};
    t.struct_size = sizeof t;
    t.data = data;
    t.dtype = CALIPER_DT_F32;
    t.ndim = ndim;
    int64_t st = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.strides[i] = st;
        st *= shape[i];
    }
    t.device = CALIPER_DEV_CPU;
    return t;
}

// ---- page state ----------------------------------------------------------------
struct Page {
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    GLFWwindow* window = nullptr;
    WebGpuRenderer* renderer = nullptr;   // owned by the hosted core
    CaliperCore* core = nullptr;
    Synthetic src;
    std::string adapter_text;
    std::string error;

    // Timing (milliseconds).
    uint64_t frames = 0;
    double t_start = 0.0, t_mark = 0.0;
    uint64_t frames_mark = 0;
    // Per frame: generating the synthetic data (not viewer work), the two
    // caliper_core_live_apply calls, the ImGui frame (NewFrame .. Render, with
    // the live views' draw code), and the renderer's encode + submit.
    double cpu_sum = 0.0, gen_sum = 0.0, apply_sum = 0.0, ui_sum = 0.0, enc_sum = 0.0;
    uint64_t cpu_n = 0;
    double fps = 0.0;

    // The byte-exact check of the WGSL colormap against the CPU reference.
    int check_state = 0;                  // 0 not run, 1 pending, 2 done
    long long check_mismatch = -1;        // texels that differ
    int check_maxdiff = -1;               // largest channel difference
    std::vector<uint8_t> check_expect;
};
Page g;

// The module's report to the page and the test: globalThis.caliperSpike. Keys
// are passed as strings so that Closure's property renaming leaves them alone.
void put(const char* key, double v) {
    EM_ASM({ (globalThis["caliperSpike"] = globalThis["caliperSpike"] || {})[UTF8ToString($0)] = $1; },
           key, v);
}
void put(const char* key, const char* v) {
    EM_ASM({
        (globalThis["caliperSpike"] = globalThis["caliperSpike"] || {})[UTF8ToString($0)] =
            UTF8ToString($1);
    }, key, v);
}

void publish() {
    const double n = g.cpu_n ? (double)g.cpu_n : 1.0;
    CaliperLiveViewInfo hi{};
    hi.struct_size = sizeof hi;
    const bool have_heat = g.core && caliper_live_views::find(g.core, "activations") &&
                           caliper_core_live_view_info(g.core, "activations", &hi);
    CaliperLiveViewInfo si{};
    si.struct_size = sizeof si;
    const bool have_series = g.core && caliper_live_views::find(g.core, "loss") &&
                             caliper_core_live_view_info(g.core, "loss", &si);
    put("frames", (double)g.frames);
    put("fps", g.fps);
    put("cpuMs", g.cpu_sum / n);
    put("genMs", g.gen_sum / n);
    put("applyMs", g.apply_sum / n);
    put("uiMs", g.ui_sum / n);
    put("encodeMs", g.enc_sum / n);
    put("gpuMs", g.renderer ? g.renderer->gpu_ms_mean() : 0.0);
    put("gpuFrames", g.renderer ? (double)g.renderer->gpu_frames() : 0.0);
    put("dispatches", g.renderer ? (double)g.renderer->colormap_dispatches() : 0.0);
    put("rendererPath", g.renderer ? g.renderer->last_device_path() : "");
    put("heatPath", have_heat ? hi.path : "");
    put("heatUpdates", have_heat ? (double)hi.updates : 0.0);
    put("seriesCount", have_series ? (double)si.count : 0.0);
    put("checkState", g.check_state);
    put("checkMismatch", (double)g.check_mismatch);
    put("checkMaxDiff", g.check_maxdiff);
    put("adapter", g.adapter_text.c_str());
    put("error", g.error.c_str());
    put("views", g.core ? caliper_core_live_view_count(g.core) : 0);
    put("heatW", g_heat_w);
    put("heatH", g_heat_h);
}

void fail(const std::string& why) {
    g.error = why;
    std::fprintf(stderr, "[spike] %s\n", why.c_str());
    publish();
}

// Read the heatmap texture back and compare it, byte for byte, with the CPU
// reference map_f32_to_rgba8 over the very data and range the GPU mapped.
void start_check() {
    caliper_embed_views::TensorView* tv = nullptr;
    for (auto& v : g.core->views)
        if (v.name == "activations") tv = &v;
    if (!tv || !tv->tex) return;
    g.check_expect.assign((size_t)g_heat_w * g_heat_h * 4, 0);
    caliper_host::map_f32_to_rgba8(g.src.heat.data(), g_heat_w, g_heat_h,
                                   caliper_host::colormap_lut(tv->colormap), tv->vmin, tv->vmax,
                                   g.check_expect.data());
    g.check_state = 1;
    g.renderer->readback_async(tv->tex, [](std::vector<uint8_t> got) {
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
}

void generate(double t) {
    g.src.step_heatmap(t);
    g.src.step_series();
    ++g.src.seq;
}

void apply_snapshots() {
    const int64_t hshape[2] = {g_heat_h, g_heat_w};
    CaliperTensor ht = cpu_tensor(g.src.heat.data(), 2, hshape);
    CaliperLiveSnapshot hs{};
    hs.struct_size = sizeof hs;
    hs.name = "activations";
    hs.kind = CALIPER_LIVE_KIND_HEATMAP;
    hs.data = &ht;
    hs.stats = g.src.stats;
    hs.stats_count = 2;
    hs.colormap = 0;   // viridis
    hs.flags = CALIPER_LIVE_APPLY_AUTO_VMIN | CALIPER_LIVE_APPLY_AUTO_VMAX;
    hs.seq = g.src.seq;
    if (!caliper_core_live_apply(g.core, &hs))
        fail(std::string("heatmap apply: ") + caliper_core_last_error(g.core));

    const int64_t sshape[1] = {(int64_t)kRing};
    CaliperTensor st = cpu_tensor(g.src.ring.data(), 1, sshape);
    CaliperLiveSnapshot ss{};
    ss.struct_size = sizeof ss;
    ss.name = "loss";
    ss.kind = CALIPER_LIVE_KIND_SERIES;
    ss.data = &st;
    ss.series_total = g.src.total;
    ss.seq = g.src.seq;
    if (!caliper_core_live_apply(g.core, &ss))
        fail(std::string("series apply: ") + caliper_core_last_error(g.core));
}

void frame() {
    const double t0 = emscripten_get_now();
    if (g.frames == 0) g.t_start = t0;

    generate((t0 - g.t_start) * 1e-3);
    const double t_gen = emscripten_get_now();
    apply_snapshots();
    const double t_applied = emscripten_get_now();

    int fb_w = 0, fb_h = 0, win_w = 0, win_h = 0;
    glfwGetFramebufferSize(g.window, &fb_w, &fb_h);
    glfwGetWindowSize(g.window, &win_w, &win_h);
    const float scale = win_w > 0 ? (float)fb_w / (float)win_w : 1.0f;

    g.renderer->new_frame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Caliper's view layer, tiled into the top of the page.
    caliper_core_hosted_frame(g.core, 0.0f, 0.0f, (float)win_w, (float)win_h - 96.0f, scale);

    // The page host's own status line (not a live view).
    ImGui::SetNextWindowPos(ImVec2(8.0f, (float)win_h - 88.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2((float)win_w - 16.0f, 80.0f), ImGuiCond_Always);
    ImGui::Begin("web viewer spike", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    ImGui::Text("adapter: %s", g.adapter_text.c_str());
    const double n = g.cpu_n ? (double)g.cpu_n : 1.0;
    ImGui::Text("frames %llu  %.1f fps  cpu %.2f ms/frame (synthetic %.2f, apply %.2f, ui %.2f, "
                "encode %.2f)  gpu submit->done %.2f ms  WGSL colormap dispatches %llu",
                (unsigned long long)g.frames, g.fps, g.cpu_sum / n, g.gen_sum / n,
                g.apply_sum / n, g.ui_sum / n, g.enc_sum / n, g.renderer->gpu_ms_mean(),
                (unsigned long long)g.renderer->colormap_dispatches());
    if (g.check_state == 2)
        ImGui::Text("WGSL colormap vs CPU reference: %lld texels differ (max channel diff %d)",
                    g.check_mismatch, g.check_maxdiff);
    else
        ImGui::Text("WGSL colormap vs CPU reference: pending");
    ImGui::End();

    ImGui::Render();
    const double t_ui = emscripten_get_now();
    g.renderer->render(fb_w, fb_h);

    const double t1 = emscripten_get_now();
    ++g.frames;
    if (g.frames > 30) {   // skip start-up frames (pipeline creation, first uploads)
        g.cpu_sum += t1 - t0;
        g.gen_sum += t_gen - t0;
        g.apply_sum += t_applied - t_gen;
        g.ui_sum += t_ui - t_applied;
        g.enc_sum += t1 - t_ui;
        ++g.cpu_n;
    }
    if (g.frames == 30) { g.t_mark = t1; g.frames_mark = g.frames; }
    if (g.frames > 30 && t1 > g.t_mark)
        g.fps = (double)(g.frames - g.frames_mark) * 1000.0 / (t1 - g.t_mark);
    if (g.frames == 90) start_check();
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

    if (!glfwInit()) return fail("glfwInit failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // the canvas gets a WebGPU context
    g.window = glfwCreateWindow(1280, 800, "Caliper web viewer spike", nullptr, nullptr);
    if (!g.window) return fail("glfwCreateWindow failed");

    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;   // no file system in the page
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOther(g.window, true);
    ImGui_ImplGlfw_InstallEmscriptenCallbacks(g.window, "#canvas");

    auto r = std::make_unique<WebGpuRenderer>(g.instance, g.adapter, g.device, "#canvas");
    if (!r->init(g.window)) return fail("WebGPU renderer init failed");
    g.renderer = r.get();
    caliper_host::services_set_renderer(g.renderer);
    g.core = caliper_core_create_hosted(r.release());
    if (!g.core) return fail("caliper_core_create_hosted refused");

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

int main() {
    const int w = EM_ASM_INT({ return Number(new URLSearchParams(location.search).get('w')) || 0; });
    const int h = EM_ASM_INT({ return Number(new URLSearchParams(location.search).get('h')) || 0; });
    if (w > 0 && h > 0 && w <= 4096 && h <= 4096) { g_heat_w = w; g_heat_h = h; }
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
