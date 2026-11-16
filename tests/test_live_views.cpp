// caliper_live_views_tests: the typed live views (caliper/live_views.h).
//
// Drives caliper_core_live_apply the way both transports do (a CaliperLiveSnapshot
// over a CPU pointer or an in-process shared MTLBuffer at a byte offset) against an
// OFFSCREEN core, then reads textures (live_view_readback), values (live_view_values),
// telemetry (live_view_info) and the slot-claim probe (live_view_read_done) back.
// Heatmaps are byte-exact against the SAME CPU reference the v1.2 views use
// (map_f32_to_rgba8 + colormap_lut). Compiled as ObjC++ with ARC, so the cases make
// real MTLBuffers directly.
//
// GPU SKIP: every GPU case starts with exactly one CALIPER_SKIP_NO_GPU line right
// after building its core: counted and reported as Skipped, a failure under
// CALIPER_REQUIRE_GPU=1 (gpu_skip.h).
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "gpu_skip.h"

#include <caliper/embed.h>
#include <caliper/live_views.h>

#include "live/live_clock.h"         // StampClock (network stamps into the viewer's clock)
#include "live/live_points.h"        // the points view's scene and its orbit camera (white-box)
#include "live/live_views_state.h"   // kPalette (white-box: the exact categorical colours)
#include "tensor_bridge.h"           // map_f32_to_rgba8 / colormap_lut (CPU reference)
#include "embed/render_stats.h"      // copy counters (white-box)

#include <imgui.h>
#include <imgui_internal.h>          // ImGui::FindWindowByName: where a view's window is

// The PNG writer for the screenshots of CALIPER_TEST_SHOTS, private to this file.
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "../third_party/stb/stb_image_write.h"
#pragma clang diagnostic pop

#import <Metal/Metal.h>

#include <time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

uint64_t now_ns() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

// One offscreen core per case (one core per process); shutdown in the dtor.
struct Core {
    CaliperCore* core = nullptr;
    bool         gpu = false;
    Core(int w, int h, bool attach = true) {
        CaliperCoreDesc d{};
        d.struct_size = sizeof d;
        d.renderer = CALIPER_RENDERER_DEFAULT;
        core = caliper_core_create(&d);
        REQUIRE(core != nullptr);
        if (attach && !caliper_test::force_no_gpu()) {
            CaliperCanvasDesc c{};
            c.struct_size = sizeof c;
            c.mode = CALIPER_CANVAS_OFFSCREEN;
            c.width = w;
            c.height = h;
            c.content_scale = 1.0f;
            gpu = caliper_core_attach_canvas(core, nullptr, &c) == 1 && MTLCreateSystemDefaultDevice() != nil;
        }
    }
    ~Core() { if (core) caliper_core_shutdown(core); }
    void frames(int n) { for (int i = 0; i < n; ++i) caliper_core_frame(core); }
    const char* err() const { return caliper_core_last_error(core); }
};

CaliperTensor tensor(void* data, CaliperDType dt, std::vector<int64_t> shape,
                     CaliperDeviceKind dev) {
    CaliperTensor t{};
    t.struct_size = sizeof t;
    t.data = data;
    t.dtype = dt;
    t.ndim = (int32_t)shape.size();
    for (int i = 0; i < t.ndim; ++i) t.shape[i] = shape[i];
    int64_t st = 1;
    for (int i = t.ndim - 1; i >= 0; --i) { t.strides[i] = st; st *= t.shape[i]; }
    t.device = dev;
    return t;
}

CaliperLiveSnapshot snap(const char* name, int32_t kind, const CaliperTensor* data,
                         uint64_t off = 0, uint32_t flags = 0) {
    CaliperLiveSnapshot s{};
    s.struct_size = sizeof s;
    s.name = name;
    s.kind = kind;
    s.data = data;
    s.data_offset = off;
    s.colormap = 0;
    s.vmin = 0.0f;
    s.vmax = 1.0f;
    s.flags = flags;
    s.points_dims = 3;
    return s;
}

CaliperLiveViewInfo info(CaliperCore* core, const char* name, int* ok = nullptr) {
    CaliperLiveViewInfo vi{};
    vi.struct_size = sizeof vi;
    const int r = caliper_core_live_view_info(core, name, &vi);
    if (ok) *ok = r;
    return vi;
}

std::vector<float> ramp(int W, int H, float lo = 0.0f, float hi = 1.0f, bool reversed = false) {
    std::vector<float> v((size_t)W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float r = (float)(x + y * W) / (float)(W * H - 1);
            if (reversed) r = 1.0f - r;
            v[(size_t)y * W + x] = lo + r * (hi - lo);
        }
    return v;
}

std::vector<uint8_t> map_ref(const std::vector<float>& v, int W, int H, int32_t cm, float vmin,
                             float vmax) {
    std::vector<uint8_t> out((size_t)W * H * 4);
    caliper_host::map_f32_to_rgba8(v.data(), W, H, caliper_host::colormap_lut(cm), vmin, vmax,
                                   out.data());
    return out;
}

std::vector<uint8_t> readback(CaliperCore* core, const char* name, int W, int H) {
    std::vector<uint8_t> px((size_t)W * H * 4, 0);
    if (!caliper_core_live_view_readback(core, name, px.data(), px.size())) px.clear();
    return px;
}

// Shared-storage MTLBuffer: `junk` bytes of 0xAB, then `bytes` of payload.
id<MTLBuffer> shared_buffer(id<MTLDevice> dev, size_t junk, const void* data, size_t bytes) {
    std::vector<uint8_t> b(junk + bytes, 0xAB);
    if (data) std::memcpy(b.data() + junk, data, bytes);
    return [dev newBufferWithBytes:b.data() length:b.size() options:MTLResourceStorageModeShared];
}

void* ptr(id<MTLBuffer> b) { return (__bridge void*)b; }

// Poll read_done with frames in between (bounded); never blocks the GPU.
bool wait_read_done(Core& c, const char* name, int max_frames = 240) {
    for (int i = 0; i < max_frames; ++i) {
        if (caliper_core_live_view_read_done(c.core, name)) return true;
        c.frames(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return caliper_core_live_view_read_done(c.core, name) == 1;
}

bool has(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }

}  // namespace

// ---------------------------------------------------------------------------
// The ground a points view clears its plot to (the window's background
// colour): the most common pixel of its render target.
// Whether p is one of the n colours scaled by a factor in [0.5, 1] with its
// hue kept, as the depth fade draws a point behind the camera's target.
static bool dimmed_of(uint32_t p, const uint32_t* colours, int n) {
    const int r = (int)(p & 255), g = (int)((p >> 8) & 255), b = (int)((p >> 16) & 255);
    for (int i = 0; i < n; ++i) {
        const uint32_t c = colours[i];
        const int cr = (int)(c & 255), cg = (int)((c >> 8) & 255), cb = (int)((c >> 16) & 255);
        const int cm = std::max({cr, cg, cb});
        if (cm == 0) continue;
        const double f = (double)std::max({r, g, b}) / cm;
        if (f < 0.49 || f > 1.0) continue;
        if (std::fabs(r - cr * f) <= 1.5 && std::fabs(g - cg * f) <= 1.5 && std::fabs(b - cb * f) <= 1.5) return true;
    }
    return false;
}

// The 3-D scene of the screenshots (web/test/scene.ts makes the same points):
// eight classes in clusters at different depths, placed by golden-ratio
// sequences so that every renderer draws the same points.
static void cluster_scene(int n, std::vector<float>* pos, std::vector<float>* lab) {
    pos->resize((size_t)3 * n);
    lab->resize((size_t)n);
    auto frac = [](double x) { return x - std::floor(x); };
    for (int i = 0; i < n; ++i) {
        const int k = i % 8, j = i / 8;
        const double a = 2.0 * M_PI * k / 8.0;
        const double cx = 4.0 * std::cos(a), cy = 2.2 * ((k % 3) - 1), cz = 4.0 * std::sin(a);
        const double u = frac((j + 0.37 * k) * 0.6180339887498949), v = frac((j + 0.53 * k) * 0.7548776662466927),
                     w = frac((j + 0.29 * k) * 0.5698402909980532);
        const double r = 0.95 * std::cbrt(u), th = 2.0 * M_PI * v, ph = std::acos(1.0 - 2.0 * w);
        (*pos)[3 * i] = (float)(cx + r * std::sin(ph) * std::cos(th));
        (*pos)[3 * i + 1] = (float)(cy + r * std::cos(ph));
        (*pos)[3 * i + 2] = (float)(cz + r * std::sin(ph) * std::sin(th));
        (*lab)[i] = (float)k;
    }
}

static uint32_t ground_of(const std::vector<uint8_t>& px) {
    std::map<uint32_t, int> counts;
    for (size_t i = 0; i + 4 <= px.size(); i += 4) {
        uint32_t p;
        std::memcpy(&p, &px[i], 4);
        ++counts[p];
    }
    uint32_t best = 0;
    int n = -1;
    for (const auto& kv : counts)
        if (kv.second > n) {
            n = kv.second;
            best = kv.first;
        }
    return best;
}

TEST_CASE("live_views/heatmap: CPU payload at an offset is byte-exact; telemetry; v1.2 "
          "view_info/readback/count keep working") {
    const int W = 64, H = 48;
    Core c(W, H);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());

    std::vector<float> r = ramp(W, H);
    std::vector<uint8_t> blob(16 + r.size() * 4, 0xAB);
    std::memcpy(blob.data() + 16, r.data(), r.size() * 4);
    CaliperTensor t = tensor(blob.data(), CALIPER_DT_F32, {H, W}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot s = snap("act", CALIPER_LIVE_KIND_HEATMAP, &t, 16, CALIPER_LIVE_APPLY_RAW);
    s.seq = 7;
    s.stamp_ns = now_ns() - 2000000;   // 2 ms ago
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &s) == 1, c.err());

    CHECK(caliper_core_live_view_count(c.core) == 1);
    CHECK(caliper_core_view_count(c.core) == 1);   // it IS a v1.2 view of the same name
    int ok = 0;
    CaliperLiveViewInfo vi = info(c.core, "act", &ok);
    REQUIRE(ok == 1);
    CHECK(vi.kind == CALIPER_LIVE_KIND_HEATMAP);
    CHECK(std::string(vi.path) == "cpu-staged");
    CHECK(vi.width == W);
    CHECK(vi.height == H);
    CHECK(vi.updates == 1);
    CHECK(vi.creates == 1);
    CHECK(vi.dropped == 0);
    CHECK(vi.seq == 7);
    CHECK(vi.imported == 0);
    CHECK(vi.read_done == 1);
    CHECK(vi.latency_ms >= 2.0);
    CHECK(vi.latency_ms < 1000.0);
    CHECK(caliper_core_live_view_read_done(c.core, "act") == 1);

    const std::vector<uint8_t> ref = map_ref(r, W, H, 0, 0.0f, 1.0f);
    CHECK(readback(c.core, "act", W, H) == ref);
    // The v1.2 surface sees the same view.
    CaliperViewInfo v12{};
    v12.struct_size = sizeof v12;
    REQUIRE(caliper_core_view_info(c.core, "act", &v12) == 1);
    CHECK(v12.width == W);
    std::vector<uint8_t> px((size_t)W * H * 4);
    REQUIRE(caliper_core_view_readback(c.core, "act", px.data(), px.size()) == 1);
    CHECK(px == ref);
    // RAW composites 1:1 at the canvas origin.
    c.frames(2);
    std::vector<uint8_t> canvas((size_t)W * H * 4);
    REQUIRE(caliper_core_read_pixels(c.core, canvas.data(), W * 4) == 1);
    CHECK(canvas == ref);

    // Same name updates in place; seq/updates move.
    std::vector<float> r2 = ramp(W, H, 0.0f, 1.0f, /*reversed=*/true);
    CaliperTensor t2 = tensor(r2.data(), CALIPER_DT_F32, {H, W}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot s2 = snap("act", CALIPER_LIVE_KIND_HEATMAP, &t2, 0, CALIPER_LIVE_APPLY_RAW);
    s2.seq = 8;
    s2.colormap = 1;
    REQUIRE(caliper_core_live_apply(c.core, &s2) == 1);
    vi = info(c.core, "act");
    CHECK(vi.updates == 2);
    CHECK(vi.seq == 8);
    CHECK(vi.latency_ms == 0.0);   // stamp unknown
    CHECK(readback(c.core, "act", W, H) == map_ref(r2, W, H, 1, 0.0f, 1.0f));

    // Removing it through the live API drops the v1.2 view too.
    CHECK(caliper_core_live_view_remove(c.core, "act") == 1);
    CHECK(caliper_core_live_view_count(c.core) == 0);
    CHECK(caliper_core_view_count(c.core) == 0);
    CHECK(caliper_core_live_view_read_done(c.core, "act") == 1);   // unknown view
}

TEST_CASE("live_views/heatmap: METAL shared MTLBuffer at a nonzero offset is zero-copy; "
          "read_done polls the GPU read; in-place rewrite") {
    const int W = 64, H = 48;
    Core c(W, H);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();

    std::vector<float> r = ramp(W, H);
    id<MTLBuffer> buf = shared_buffer(dev, 256, r.data(), r.size() * 4);
    REQUIRE(ptr(buf) != nullptr);
    CaliperTensor t = tensor(ptr(buf), CALIPER_DT_F32, {H, W}, CALIPER_DEV_METAL);
    CaliperLiveSnapshot s = snap("m", CALIPER_LIVE_KIND_HEATMAP, &t, 256, CALIPER_LIVE_APPLY_RAW);
    s.seq = 1;
    s.stamp_ns = now_ns();
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &s) == 1, c.err());
    CaliperLiveViewInfo vi = info(c.core, "m");
    CHECK(std::string(vi.path) == "compute-imported");
    CHECK(vi.imported == 1);
    CHECK(vi.seq == 1);
    // Never waited on: the probe turns 1 once the colormap kernel retired.
    CHECK(wait_read_done(c, "m"));
    CHECK(readback(c.core, "m", W, H) == map_ref(r, W, H, 0, 0.0f, 1.0f));
    c.frames(1);
    std::vector<uint8_t> canvas((size_t)W * H * 4);
    REQUIRE(caliper_core_read_pixels(c.core, canvas.data(), W * 4) == 1);
    CHECK(canvas == map_ref(r, W, H, 0, 0.0f, 1.0f));

    // The slot is free: rewrite the same bytes in place and apply again.
    std::vector<float> r2 = ramp(W, H, 0.0f, 1.0f, true);
    std::memcpy((uint8_t*)buf.contents + 256, r2.data(), r2.size() * 4);
    s.seq = 2;
    REQUIRE(caliper_core_live_apply(c.core, &s) == 1);
    CHECK(wait_read_done(c, "m"));
    CHECK(readback(c.core, "m", W, H) == map_ref(r2, W, H, 0, 0.0f, 1.0f));
    CHECK(info(c.core, "m").creates == 1);   // same shape: the texture is reused

    // WAIT_GPU (tests): on return the read already retired.
    s.seq = 3;
    s.flags |= CALIPER_LIVE_APPLY_WAIT_GPU;
    REQUIRE(caliper_core_live_apply(c.core, &s) == 1);
    CHECK(caliper_core_live_view_read_done(c.core, "m") == 1);
    CHECK(info(c.core, "m").updates == 3);
}

TEST_CASE("live_views/heatmap: AUTO flags take vmin/vmax from stats") {
    const int W = 32, H = 16;
    Core c(W, H);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    std::vector<float> r = ramp(W, H, -2.0f, 3.0f);
    CaliperTensor t = tensor(r.data(), CALIPER_DT_F32, {H, W}, CALIPER_DEV_CPU);
    const float stats[2] = {-2.0f, 3.0f};
    CaliperLiveSnapshot s = snap("auto", CALIPER_LIVE_KIND_HEATMAP, &t, 0,
                                 CALIPER_LIVE_APPLY_RAW | CALIPER_LIVE_APPLY_AUTO_VMIN |
                                     CALIPER_LIVE_APPLY_AUTO_VMAX);
    s.stats = stats;
    s.stats_count = 2;
    s.colormap = 2;
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &s) == 1, c.err());
    CHECK(readback(c.core, "auto", W, H) == map_ref(r, W, H, 2, -2.0f, 3.0f));

    // Only the AUTO end moves: vmin from stats, vmax fixed.
    const float stats2[2] = {-1.0f, 99.0f};
    s.stats = stats2;
    s.flags = CALIPER_LIVE_APPLY_RAW | CALIPER_LIVE_APPLY_AUTO_VMIN;
    s.vmax = 2.0f;
    REQUIRE(caliper_core_live_apply(c.core, &s) == 1);
    CHECK(readback(c.core, "auto", W, H) == map_ref(r, W, H, 2, -1.0f, 2.0f));

    // Non-finite stats (an all-NaN tensor) keep the last finite AUTO range.
    const float nan_stats[2] = {NAN, NAN};
    s.stats = nan_stats;
    s.flags = CALIPER_LIVE_APPLY_RAW | CALIPER_LIVE_APPLY_AUTO_VMIN | CALIPER_LIVE_APPLY_AUTO_VMAX;
    REQUIRE(caliper_core_live_apply(c.core, &s) == 1);
    CHECK(readback(c.core, "auto", W, H) == map_ref(r, W, H, 2, -1.0f, 2.0f));

    // AUTO without stats is refused, and the view keeps its pixels.
    s.stats = nullptr;
    s.stats_count = 0;
    CHECK(caliper_core_live_apply(c.core, &s) == 0);
    CHECK(has(c.err(), "AUTO_VMIN needs stats[0]"));
    CHECK(readback(c.core, "auto", W, H) == map_ref(r, W, H, 2, -1.0f, 2.0f));
}

TEST_CASE("live_views/image: u8 RGBA from CPU and from a METAL buffer (blit-imported)") {
    const int W = 20, H = 10;
    Core c(W, H);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    std::vector<uint8_t> rgba((size_t)W * H * 4);
    for (size_t i = 0; i < rgba.size(); ++i) rgba[i] = (uint8_t)(i * 7 + 3);
    CaliperTensor t = tensor(rgba.data(), CALIPER_DT_U8, {H, W, 4}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot s = snap("img", CALIPER_LIVE_KIND_IMAGE, &t, 0, CALIPER_LIVE_APPLY_RAW);
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &s) == 1, c.err());
    CHECK(readback(c.core, "img", W, H) == rgba);
    CHECK(std::string(info(c.core, "img").path) == "cpu-staged");

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLBuffer> buf = shared_buffer(dev, 512, rgba.data(), rgba.size());
    CaliperTensor tm = tensor(ptr(buf), CALIPER_DT_U8, {H, W, 4}, CALIPER_DEV_METAL);
    CaliperLiveSnapshot sm = snap("img", CALIPER_LIVE_KIND_IMAGE, &tm, 512, CALIPER_LIVE_APPLY_RAW);
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &sm) == 1, c.err());
    CaliperLiveViewInfo vi = info(c.core, "img");
    CHECK(std::string(vi.path) == "blit-imported");
    CHECK(vi.imported == 1);
    CHECK(wait_read_done(c, "img"));
    CHECK(readback(c.core, "img", W, H) == rgba);
}

TEST_CASE("live_views/points: categorical labels draw exact palette colours; the "
          "superseded payload's slot is released after frames") {
    Core c(480, 360);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();

    const int N = 8;
    std::vector<float> pos(N * 3), lab(N);
    for (int i = 0; i < N; ++i) {
        pos[3 * i + 0] = (float)(i % 4);
        pos[3 * i + 1] = (float)(i / 4);
        pos[3 * i + 2] = 0.0f;
        lab[i] = (float)i;
    }
    id<MTLBuffer> pbuf = shared_buffer(dev, 64, pos.data(), pos.size() * 4);
    id<MTLBuffer> lbuf = shared_buffer(dev, 128, lab.data(), lab.size() * 4);
    CaliperTensor tp = tensor(ptr(pbuf), CALIPER_DT_F32, {N, 3}, CALIPER_DEV_METAL);
    CaliperTensor tl = tensor(ptr(lbuf), CALIPER_DT_F32, {N}, CALIPER_DEV_METAL);
    const float stats[8] = {0, 7, 0, 3, 0, 1, 0, 0};   // aux min/max, then the bbox
    CaliperLiveSnapshot s = snap("emb", CALIPER_LIVE_KIND_POINTS, &tp, 64);
    s.aux = &tl;
    s.aux_offset = 128;
    s.label_count = N;
    s.points_dims = 2;
    s.stats = stats;
    s.stats_count = 8;
    s.seq = 1;
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &s) == 1, c.err());
    CaliperLiveViewInfo vi = info(c.core, "emb");
    CHECK(vi.kind == CALIPER_LIVE_KIND_POINTS);
    CHECK(std::string(vi.path) == "points-imported");
    CHECK(vi.imported == 1);
    CHECK(vi.count == N);
    CHECK(vi.read_done == 1);           // nothing superseded yet
    CHECK(caliper_core_view_count(c.core) == 0);   // not a v1.2 view

    c.frames(3);
    vi = info(c.core, "emb");
    REQUIRE(vi.width > 0);
    REQUIRE(vi.height > 0);
    CHECK(vi.creates >= 1);
    std::vector<uint8_t> px = readback(c.core, "emb", vi.width, vi.height);
    REQUIRE(px.size() == (size_t)vi.width * vi.height * 4);
    const uint32_t bg = ground_of(px);
    int fg = 0;
    std::set<uint32_t> seen;
    for (size_t i = 0; i < px.size(); i += 4) {
        uint32_t p;
        std::memcpy(&p, &px[i], 4);
        if (p != bg) ++fg;
        seen.insert(p);
    }
    int palette_hits = 0;
    for (int i = 0; i < N; ++i) palette_hits += seen.count(caliper_live_views::kPalette[i]) ? 1 : 0;
    MESSAGE("points target " << vi.width << "x" << vi.height << ": " << fg
                             << " foreground px, " << palette_hits << "/" << N << " palette colours");
    CHECK(fg > 0);
    CHECK(palette_hits == N);

    // A new payload in another buffer: the previous one stays claimed until the
    // last frame that drew it retired, then read_done turns 1 (no wait).
    std::vector<float> pos2 = pos;
    for (float& f : pos2) f *= 0.5f;
    id<MTLBuffer> pbuf2 = shared_buffer(dev, 0, pos2.data(), pos2.size() * 4);
    CaliperTensor tp2 = tensor(ptr(pbuf2), CALIPER_DT_F32, {N, 3}, CALIPER_DEV_METAL);
    s.data = &tp2;
    s.data_offset = 0;
    s.seq = 2;
    REQUIRE(caliper_core_live_apply(c.core, &s) == 1);
    CHECK(info(c.core, "emb").seq == 2);
    CHECK(wait_read_done(c, "emb"));
    c.frames(2);
    CHECK(caliper_core_live_view_read_done(c.core, "emb") == 1);   // the current payload alone is not "previous"

    // WAIT_GPU (tests): return only once the superseded payload's last frame retired.
    c.frames(1);
    s.data = &tp;
    s.data_offset = 64;
    s.flags = CALIPER_LIVE_APPLY_WAIT_GPU;
    s.seq = 3;
    REQUIRE(caliper_core_live_apply(c.core, &s) == 1);
    CHECK(caliper_core_live_view_read_done(c.core, "emb") == 1);
    CHECK(info(c.core, "emb").updates == 3);
}

TEST_CASE("live_views/points: 3-D continuous colormap and CPU-staged payloads draw") {
    Core c(400, 320);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    const int N = 500;
    std::vector<float> pos(N * 3), val(N);
    for (int i = 0; i < N; ++i) {
        const float a = (float)i * 0.1f;
        pos[3 * i + 0] = std::cos(a);
        pos[3 * i + 1] = (float)i / N - 0.5f;
        pos[3 * i + 2] = std::sin(a);
        val[i] = (float)i;
    }
    CaliperTensor tp = tensor(pos.data(), CALIPER_DT_F32, {N, 3}, CALIPER_DEV_CPU);
    CaliperTensor tv = tensor(val.data(), CALIPER_DT_F32, {N}, CALIPER_DEV_CPU);
    const float stats[2] = {0.0f, (float)(N - 1)};   // no bbox: fit scans the payload
    CaliperLiveSnapshot s = snap("cloud", CALIPER_LIVE_KIND_POINTS, &tp, 0,
                                 CALIPER_LIVE_APPLY_AUTO_VMIN | CALIPER_LIVE_APPLY_AUTO_VMAX);
    s.aux = &tv;
    s.colormap = 1;
    s.stats = stats;
    s.stats_count = 2;
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &s) == 1, c.err());
    CaliperLiveViewInfo vi = info(c.core, "cloud");
    CHECK(std::string(vi.path) == "points-cpu-staged");
    CHECK(vi.imported == 0);
    // The CPU bytes were copied: scribbling them now changes nothing drawn.
    std::fill(pos.begin(), pos.end(), 1e9f);
    c.frames(3);
    vi = info(c.core, "cloud");
    std::vector<uint8_t> px = readback(c.core, "cloud", vi.width, vi.height);
    REQUIRE(!px.empty());
    const uint32_t* lut = caliper_host::colormap_lut(1);
    std::set<uint32_t> lut_colours(lut, lut + 256);
    int fg = 0, in_lut = 0;
    const uint32_t bg = ground_of(px);
    for (size_t i = 0; i < px.size(); i += 4) {
        uint32_t p;
        std::memcpy(&p, &px[i], 4);
        if (p == bg) continue;
        ++fg;
        in_lut += lut_colours.count(p) ? 1 : 0;
    }
    // With the depth cues, the points behind the camera's target are dimmed
    // with their hue kept: every lit pixel is a LUT colour, exact or scaled.
    int dimmed = 0;
    for (size_t i = 0; i < px.size(); i += 4) {
        uint32_t p;
        std::memcpy(&p, &px[i], 4);
        if (p != bg && !lut_colours.count(p) && dimmed_of(p, lut, 256)) ++dimmed;
    }
    MESSAGE("3-D cloud: " << fg << " foreground px, " << in_lut << " exact and " << dimmed
                          << " dimmed magma colours");
    CHECK(fg > 100);
    CHECK(in_lut > 0);
    CHECK(dimmed > 0);
    CHECK(in_lut + dimmed == fg);   // opaque points: no other colour
    // The depth cues off: every lit pixel is an exact LUT colour.
    REQUIRE(caliper_live_views::set_toggle(c.core, "cloud", "depth", false));
    c.frames(2);
    px = readback(c.core, "cloud", vi.width, vi.height);
    fg = in_lut = 0;
    for (size_t i = 0; i < px.size(); i += 4) {
        uint32_t p;
        std::memcpy(&p, &px[i], 4);
        if (p == bg) continue;
        ++fg;
        in_lut += lut_colours.count(p) ? 1 : 0;
    }
    CHECK(fg > 100);
    CHECK(in_lut == fg);
    CHECK(caliper_core_live_view_read_done(c.core, "cloud") == 1);
}

// The orbit camera of a 3-D points view, which the scene core keeps (live_points.h).
static const caliper_scene::Orbit& orbit_of(caliper_live_views::View* pv) {
    REQUIRE(pv->points);
    REQUIRE(pv->points->scene);
    const caliper_scene::Orbit* o = std::get_if<caliper_scene::Orbit>(&pv->points->scene->camera());
    REQUIRE(o != nullptr);
    return *o;
}

TEST_CASE("live_views/points: 3-D points draw a box with ticks, nearer points larger and farther ones "
          "dimmer with their hue kept, and turn with the rotate toggle until a drag") {
    Core c(640, 480);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    // Eight anchors at the box's corners (label 2), one point in front of the
    // centre as the default camera sees it (label 1) and one behind it (label 3).
    std::vector<float> pos, lab;
    for (int i = 0; i < 8; ++i) {
        pos.insert(pos.end(), {(i & 1) ? 1.0f : -1.0f, (i & 2) ? 1.0f : -1.0f, (i & 4) ? 1.0f : -1.0f});
        lab.push_back(2.0f);
    }
    pos.insert(pos.end(), {0.45f, 0.45f, 0.45f});
    lab.push_back(1.0f);
    pos.insert(pos.end(), {-0.45f, -0.45f, -0.45f});
    lab.push_back(3.0f);
    const int N = (int)lab.size();
    CaliperTensor tp = tensor(pos.data(), CALIPER_DT_F32, {N, 3}, CALIPER_DEV_CPU);
    CaliperTensor tl = tensor(lab.data(), CALIPER_DT_F32, {N}, CALIPER_DEV_CPU);
    const float stats[8] = {0, 7, -1, 1, -1, 1, -1, 1};   // aux min/max, then the bbox
    CaliperLiveSnapshot s = snap("box3", CALIPER_LIVE_KIND_POINTS, &tp);
    s.aux = &tl;
    s.label_count = 8;
    s.stats = stats;
    s.stats_count = 8;
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &s) == 1, c.err());
    c.frames(4);
    caliper_live_views::View* pv = caliper_live_views::find(c.core, "box3");
    REQUIRE(pv != nullptr);
    // The default three-quarter camera, as every renderer's.
    CHECK(orbit_of(pv).azimuth == doctest::Approx(caliper_live_views::kCamAz));
    CHECK(orbit_of(pv).elevation == doctest::Approx(caliper_live_views::kCamEl));
    // The box: tick labels on x, y and z.
    MESSAGE("box labels: " << pv->box_labels);
    CHECK(pv->box_labels.find("x=") != std::string::npos);
    CHECK(pv->box_labels.find("y=") != std::string::npos);
    CHECK(pv->box_labels.find("z=") != std::string::npos);
    CHECK(pv->box_labels == "x=-1.0|-0.5|0.0|0.5|1.0;y=-1.0|-0.5|0.0|0.5|1.0;z=-1.0|-0.5|0.0|0.5|1.0");

    CaliperLiveViewInfo vi = info(c.core, "box3");
    std::vector<uint8_t> px = readback(c.core, "box3", vi.width, vi.height);
    REQUIRE(!px.empty());
    // The ground is transparent in 3-D (the box's back layer shows through it).
    int ground = 0, near = 0, far_exact = 0, far_dim = 0;
    uint32_t far_colour = 0;
    const uint32_t red = caliper_live_views::kPalette[3];
    for (size_t i = 0; i < px.size(); i += 4) {
        uint32_t p;
        std::memcpy(&p, &px[i], 4);
        if ((p >> 24) == 0) ++ground;
        if (p == caliper_live_views::kPalette[1]) ++near;
        if (p == red) ++far_exact;
        else if ((p >> 24) == 255 && dimmed_of(p, &red, 1)) { ++far_dim; far_colour = p; }
    }
    char hex[16];
    std::snprintf(hex, sizeof hex, "%06x", (unsigned)(far_colour & 0xffffffu));
    MESSAGE("near point " << near << " px, far point " << far_dim << " px dimmed to rgb " << hex << " (abgr)");
    CHECK(ground > 0);
    CHECK(near > 0);                 // in front of the target: full strength
    CHECK(far_exact == 0);           // behind it: dimmed
    CHECK(far_dim > 0);
    CHECK(near > far_dim);           // and nearer is larger
    // The depth cues off: the far point in its exact colour, the two the same size.
    REQUIRE(caliper_live_views::set_toggle(c.core, "box3", "depth", false));
    c.frames(2);
    px = readback(c.core, "box3", vi.width, vi.height);
    int near2 = 0, far2 = 0;
    for (size_t i = 0; i < px.size(); i += 4) {
        uint32_t p;
        std::memcpy(&p, &px[i], 4);
        near2 += p == caliper_live_views::kPalette[1];
        far2 += p == red;
    }
    CHECK(far2 > 0);
    CHECK(near2 == far2);
    REQUIRE(caliper_live_views::set_toggle(c.core, "box3", "depth", true));

    // The rotate toggle turns the view until a drag in it.
    REQUIRE(caliper_live_views::set_toggle(c.core, "box3", "rotate", true));
    const double az0 = orbit_of(pv).azimuth;
    for (int i = 0; i < 6; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        c.frames(1);
    }
    CHECK(orbit_of(pv).azimuth > az0);
    auto send = [&](CaliperEventType type, float x, float y, int down) {
        CaliperInputEvent e{};
        e.struct_size = sizeof e;
        e.type = type;
        e.x = x;
        e.y = y;
        e.button = 0;
        e.down = down;
        caliper_core_event(c.core, &e);
    };
    const float cx = pv->geo_area[0] + pv->geo_area[2] * 0.5f, cy = pv->geo_area[1] + pv->geo_area[3] * 0.5f;
    send(CALIPER_EVENT_MOUSE_MOVE, cx, cy, 0);
    c.frames(1);
    send(CALIPER_EVENT_MOUSE_BUTTON, cx, cy, 1);
    c.frames(1);
    send(CALIPER_EVENT_MOUSE_MOVE, cx + 30.0f, cy, 0);
    c.frames(1);
    send(CALIPER_EVENT_MOUSE_BUTTON, cx + 30.0f, cy, 0);
    c.frames(2);
    CHECK_FALSE(pv->auto_rotate);
    const double az1 = orbit_of(pv).azimuth;
    c.frames(4);
    CHECK(orbit_of(pv).azimuth == az1);
}

TEST_CASE("live_views/points: screenshots of a 3-D cluster scene from three angles (CALIPER_TEST_SHOTS)") {
    const char* dir = std::getenv("CALIPER_TEST_SHOTS");
    if (!dir || !*dir) return;
    Core c(900, 640);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    std::vector<float> pos, lab;
    cluster_scene(3000, &pos, &lab);
    CaliperTensor tp = tensor(pos.data(), CALIPER_DT_F32, {3000, 3}, CALIPER_DEV_CPU);
    CaliperTensor tl = tensor(lab.data(), CALIPER_DT_F32, {3000}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot s = snap("clusters", CALIPER_LIVE_KIND_POINTS, &tp);
    s.aux = &tl;
    s.label_count = 8;
    s.vmin = 0.0f;
    s.vmax = 7.0f;
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &s) == 1, c.err());
    c.frames(4);
    caliper_live_views::View* pv = caliper_live_views::find(c.core, "clusters");
    REQUIRE(pv != nullptr);
    const struct { const char* name; float daz, del; } angles[] = {{"a", 0.0f, 0.0f}, {"b", 1.6f, 0.0f}, {"c", -0.9f, 0.55f}};
    for (const auto& a : angles) {
        pv->points->scene->set_angles(caliper_live_views::kCamAz + a.daz, caliper_live_views::kCamEl + a.del);
        pv->dirty = true;
        c.frames(3);
        std::vector<uint8_t> canvas(900 * 640 * 4);
        REQUIRE(caliper_core_read_pixels(c.core, canvas.data(), 900 * 4) == 1);
        const std::string file = std::string(dir) + "/native-" + a.name + ".png";
        CHECK(stbi_write_png(file.c_str(), 900, 640, 4, canvas.data(), 900 * 4) != 0);
    }
}

// The colormap edge cases of web/test/mock_server.py's edge_values(): NaN, the
// infinities, zeros, the range's ends and +-1e30, then each side of every
// index boundary of [lo, hi] (float32 arithmetic throughout).
static std::vector<float> edge_values(float lo, float hi) {
    std::vector<float> out = {NAN, INFINITY, -INFINITY, 0.0f, -0.0f, lo, hi, 1e30f, -1e30f};
    std::vector<float> b(256);
    for (int k = 0; k < 256; ++k) b[k] = lo + ((float)k - 0.5f) / 255.0f * (hi - lo);
    for (float x : b) out.push_back(x);
    for (float x : b) out.push_back(std::nextafter(x, INFINITY));
    for (float x : b) out.push_back(std::nextafter(x, -INFINITY));
    for (float x : b) out.push_back(std::nextafter(std::nextafter(x, INFINITY), INFINITY));
    return out;
}

TEST_CASE("live_views/points: value-coloured points have map_f32_to_rgba8's colour for every edge case, over "
          "increasing, decreasing and very wide ranges") {
    Core c(900, 700);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    const uint32_t* lut = caliper_host::colormap_lut(1);
    const float ranges[][2] = {{-3.0f, 5.0f}, {1.0f, -1.0f}, {-3e38f, 3e38f}, {0.0f, 3e38f}};
    for (const auto& r : ranges) {
        CAPTURE(r[0]);
        CAPTURE(r[1]);
        // 1,024 2-D points on a 32 x 32 grid, coloured by the edge values.
        const std::vector<float> vals = edge_values(r[0], r[1]);
        const int N = 1024;
        std::vector<float> pos(N * 3), val(N);
        for (int i = 0; i < N; ++i) {
            pos[3 * i] = (float)(i % 32);
            pos[3 * i + 1] = (float)(i / 32);
            pos[3 * i + 2] = 0.0f;
            val[i] = vals[i % vals.size()];
        }
        CaliperTensor tp = tensor(pos.data(), CALIPER_DT_F32, {N, 3}, CALIPER_DEV_CPU);
        CaliperTensor tv = tensor(val.data(), CALIPER_DT_F32, {N}, CALIPER_DEV_CPU);
        const float stats[8] = {0, 0, 0, 31, 0, 31, 0, 0};
        CaliperLiveSnapshot s = snap("edge_points", CALIPER_LIVE_KIND_POINTS, &tp);
        s.aux = &tv;
        s.colormap = 1;
        s.vmin = r[0];
        s.vmax = r[1];
        s.points_dims = 2;
        s.stats = stats;
        s.stats_count = 8;
        REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &s) == 1, c.err());
        c.frames(3);
        caliper_live_views::View* pv = caliper_live_views::find(c.core, "edge_points");
        REQUIRE(pv != nullptr);
        CaliperLiveViewInfo vi = info(c.core, "edge_points");
        std::vector<uint8_t> px = readback(c.core, "edge_points", vi.width, vi.height);
        REQUIRE(px.size() == (size_t)vi.width * vi.height * 4);
        float M[16];   // the clip-from-world matrix the draw gets (2-D: an orthographic pan and zoom)
        REQUIRE(caliper_live_views::points_mvp(*pv, M));
        int wrong = 0, compared = 0;
        std::string first;
        for (int i = 0; i < N; ++i) {
            // The point's centre on the render target (2-D: the view matrix is the identity).
            const float x = pos[3 * i], y = pos[3 * i + 1];
            const float cw = M[3] * x + M[7] * y + M[15];
            const float cx = (M[0] * x + M[4] * y + M[12]) / cw, cy = (M[1] * x + M[5] * y + M[13]) / cw;
            const int sx = (int)((cx * 0.5f + 0.5f) * vi.width), sy = (int)((0.5f - cy * 0.5f) * vi.height);
            if (sx < 0 || sy < 0 || sx >= vi.width || sy >= vi.height) continue;
            uint8_t want[4];
            caliper_host::map_f32_to_rgba8(&val[i], 1, 1, lut, r[0], r[1], want);
            const uint8_t* got = &px[((size_t)sy * vi.width + sx) * 4];
            ++compared;
            if (std::memcmp(got, want, 4) != 0) {
                if (!wrong) first = std::to_string(i) + " (" + std::to_string(val[i]) + ")";
                ++wrong;
            }
        }
        MESSAGE("[" << r[0] << ", " << r[1] << "]: " << compared << " points compared, " << wrong
                    << " differ" << (wrong ? ", the first " + first : std::string()));
        CHECK(compared == N);
        CHECK(wrong == 0);
    }
}

TEST_CASE("live_views/series: ring wrap copies the window oldest first") {
    Core c(320, 240);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    const int M = 8;
    std::vector<float> ring(M);
    for (int i = 0; i < 13; ++i) ring[i % M] = (float)i;   // values 5..12 survive
    CaliperTensor t = tensor(ring.data(), CALIPER_DT_F32, {M}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot s = snap("loss", CALIPER_LIVE_KIND_SERIES, &t);
    s.series_total = 13;
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &s) == 1, c.err());
    float out[16] = {};
    size_t n = 0;
    REQUIRE(caliper_core_live_view_values(c.core, "loss", out, 16, &n) == 1);
    REQUIRE(n == 8);
    for (int j = 0; j < 8; ++j) CHECK(out[j] == (float)(5 + j));
    CaliperLiveViewInfo vi = info(c.core, "loss");
    CHECK(vi.count == 8);
    CHECK(std::string(vi.path) == "cpu-read");
    CHECK(vi.read_done == 1);
    // Count-only query, and a short cap.
    n = 0;
    REQUIRE(caliper_core_live_view_values(c.core, "loss", nullptr, 0, &n) == 1);
    CHECK(n == 8);
    float two[2] = {};
    REQUIRE(caliper_core_live_view_values(c.core, "loss", two, 2, &n) == 1);
    CHECK(two[0] == 5.0f);
    CHECK(two[1] == 6.0f);

    // Not yet wrapped (total < M), from a shared METAL buffer at an offset.
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    const float first3[M] = {0.5f, 1.5f, 2.5f, 0, 0, 0, 0, 0};
    id<MTLBuffer> buf = shared_buffer(dev, 32, first3, sizeof first3);
    CaliperTensor tm = tensor(ptr(buf), CALIPER_DT_F32, {M}, CALIPER_DEV_METAL);
    CaliperLiveSnapshot sm = snap("lr", CALIPER_LIVE_KIND_SERIES, &tm, 32);
    sm.series_total = 3;
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &sm) == 1, c.err());
    REQUIRE(caliper_core_live_view_values(c.core, "lr", out, 16, &n) == 1);
    REQUIRE(n == 3);
    CHECK(out[0] == 0.5f);
    CHECK(out[2] == 2.5f);
    c.frames(3);   // ImPlot windows draw
    CHECK(caliper_core_live_view_count(c.core) == 2);
}

TEST_CASE("live_views/histogram: bin counts round-trip; range from stats") {
    Core c(320, 240);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    const int B = 16;
    std::vector<float> bins(B);
    for (int i = 0; i < B; ++i) bins[i] = (float)(i * i);
    CaliperTensor t = tensor(bins.data(), CALIPER_DT_F32, {B}, CALIPER_DEV_CPU);
    const float stats[2] = {-1.0f, 1.0f};
    CaliperLiveSnapshot s = snap("grad", CALIPER_LIVE_KIND_HISTOGRAM, &t);
    s.stats = stats;
    s.stats_count = 2;
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &s) == 1, c.err());
    bins[0] = 1234.0f;   // copied at apply: the payload is reclaimable at once
    std::vector<float> out(B);
    size_t n = 0;
    REQUIRE(caliper_core_live_view_values(c.core, "grad", out.data(), out.size(), &n) == 1);
    REQUIRE(n == (size_t)B);
    for (int i = 0; i < B; ++i) CHECK(out[i] == (float)(i * i));
    CHECK(info(c.core, "grad").count == B);
    CHECK(caliper_core_live_view_read_done(c.core, "grad") == 1);
    c.frames(3);
}

TEST_CASE("live_views/pause: applies are accepted and dropped; the display is frozen") {
    const int W = 16, H = 8;
    Core c(W, H);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    std::vector<float> a = ramp(W, H), b = ramp(W, H, 0.0f, 1.0f, true);
    CaliperTensor ta = tensor(a.data(), CALIPER_DT_F32, {H, W}, CALIPER_DEV_CPU);
    CaliperTensor tb = tensor(b.data(), CALIPER_DT_F32, {H, W}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot s = snap("p", CALIPER_LIVE_KIND_HEATMAP, &ta, 0, CALIPER_LIVE_APPLY_RAW);
    s.seq = 1;
    REQUIRE(caliper_core_live_apply(c.core, &s) == 1);

    CHECK(caliper_core_live_paused(c.core) == 0);
    caliper_core_live_set_paused(c.core, 1);
    CHECK(caliper_core_live_paused(c.core) == 1);
    s.data = &tb;
    s.seq = 2;
    CHECK(caliper_core_live_apply(c.core, &s) == 1);
    s.seq = 3;
    CHECK(caliper_core_live_apply(c.core, &s) == 1);
    CaliperLiveViewInfo vi = info(c.core, "p");
    CHECK(vi.dropped == 2);
    CHECK(vi.updates == 1);
    CHECK(vi.seq == 1);
    CHECK(readback(c.core, "p", W, H) == map_ref(a, W, H, 0, 0.0f, 1.0f));
    // Gates still apply while paused.
    CaliperLiveSnapshot bad = snap("p", 9, &tb);
    CHECK(caliper_core_live_apply(c.core, &bad) == 0);

    // A series paused: values frozen too.
    std::vector<float> ring = {1, 2, 3, 4};
    CaliperTensor tr = tensor(ring.data(), CALIPER_DT_F32, {4}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot ss = snap("s", CALIPER_LIVE_KIND_SERIES, &tr);
    ss.series_total = 4;
    CHECK(caliper_core_live_apply(c.core, &ss) == 1);
    size_t n = 99;
    REQUIRE(caliper_core_live_view_values(c.core, "s", nullptr, 0, &n) == 1);
    CHECK(n == 0);
    CHECK(info(c.core, "s").dropped == 1);

    caliper_core_live_set_paused(c.core, 0);
    s.seq = 4;
    REQUIRE(caliper_core_live_apply(c.core, &s) == 1);
    vi = info(c.core, "p");
    CHECK(vi.updates == 2);
    CHECK(vi.seq == 4);
    CHECK(readback(c.core, "p", W, H) == map_ref(b, W, H, 0, 0.0f, 1.0f));
    REQUIRE(caliper_core_live_apply(c.core, &ss) == 1);
    REQUIRE(caliper_core_live_view_values(c.core, "s", nullptr, 0, &n) == 1);
    CHECK(n == 4);
}

TEST_CASE("live_views/kinds: a name may change kind; heatmap <-> points <-> series") {
    Core c(320, 240);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    std::vector<float> h(16, 0.5f);
    CaliperTensor th = tensor(h.data(), CALIPER_DT_F32, {4, 4}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot s = snap("x", CALIPER_LIVE_KIND_HEATMAP, &th);
    REQUIRE(caliper_core_live_apply(c.core, &s) == 1);
    CHECK(caliper_core_view_count(c.core) == 1);
    c.frames(2);

    std::vector<float> pos = {0, 0, 0, 1, 1, 1};
    CaliperTensor tp = tensor(pos.data(), CALIPER_DT_F32, {2, 3}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot sp = snap("x", CALIPER_LIVE_KIND_POINTS, &tp);
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &sp) == 1, c.err());
    CHECK(caliper_core_view_count(c.core) == 0);   // the heatmap's v1.2 view went with it
    CHECK(info(c.core, "x").kind == CALIPER_LIVE_KIND_POINTS);
    CHECK(info(c.core, "x").updates == 1);
    c.frames(2);

    // A failed kind change keeps the current view.
    CaliperLiveSnapshot bad = snap("x", CALIPER_LIVE_KIND_SERIES, &th);   // 2-D: refused
    CHECK(caliper_core_live_apply(c.core, &bad) == 0);
    CHECK(info(c.core, "x").kind == CALIPER_LIVE_KIND_POINTS);

    std::vector<float> ring = {1, 2};
    CaliperTensor tr = tensor(ring.data(), CALIPER_DT_F32, {2}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot ss = snap("x", CALIPER_LIVE_KIND_SERIES, &tr);
    ss.series_total = 2;
    REQUIRE(caliper_core_live_apply(c.core, &ss) == 1);
    CHECK(info(c.core, "x").kind == CALIPER_LIVE_KIND_SERIES);
    CHECK(caliper_core_live_view_count(c.core) == 1);
    c.frames(2);

    REQUIRE(caliper_core_live_apply(c.core, &s) == 1);   // back to a heatmap
    CHECK(caliper_core_view_count(c.core) == 1);
    CHECK(info(c.core, "x").width == 4);
}

TEST_CASE("live_views/windows: every kind draws windowed with hover and drag input") {
    Core c(960, 720);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    std::vector<float> hm = ramp(32, 24);
    id<MTLBuffer> hb = shared_buffer(dev, 0, hm.data(), hm.size() * 4);
    CaliperTensor th = tensor(ptr(hb), CALIPER_DT_F32, {24, 32}, CALIPER_DEV_METAL);
    CaliperLiveSnapshot sh = snap("heat", CALIPER_LIVE_KIND_HEATMAP, &th);
    REQUIRE(caliper_core_live_apply(c.core, &sh) == 1);

    const int N = 3000;
    std::vector<float> pos(N * 3), lab(N);
    for (int i = 0; i < N; ++i) {
        pos[3 * i] = std::sin(i * 0.37f);
        pos[3 * i + 1] = std::cos(i * 0.11f);
        pos[3 * i + 2] = std::sin(i * 0.05f);
        lab[i] = (float)(i % 5);
    }
    id<MTLBuffer> pb = shared_buffer(dev, 0, pos.data(), pos.size() * 4);
    id<MTLBuffer> lb = shared_buffer(dev, 0, lab.data(), lab.size() * 4);
    CaliperTensor tp = tensor(ptr(pb), CALIPER_DT_F32, {N, 3}, CALIPER_DEV_METAL);
    CaliperTensor tl = tensor(ptr(lb), CALIPER_DT_F32, {N}, CALIPER_DEV_METAL);
    CaliperLiveSnapshot sp = snap("pts", CALIPER_LIVE_KIND_POINTS, &tp);
    sp.aux = &tl;
    sp.label_count = 5;
    REQUIRE(caliper_core_live_apply(c.core, &sp) == 1);

    std::vector<float> ring(20000);
    for (size_t i = 0; i < ring.size(); ++i) ring[i] = std::sin((float)i * 0.01f);
    CaliperTensor tr = tensor(ring.data(), CALIPER_DT_F32, {(int64_t)ring.size()}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot sr = snap("loss", CALIPER_LIVE_KIND_SERIES, &tr);
    sr.series_total = ring.size();   // decimated path (> 8192)
    REQUIRE(caliper_core_live_apply(c.core, &sr) == 1);

    std::vector<float> bins = {1, 4, 9, 16, 9, 4, 1};
    CaliperTensor tb = tensor(bins.data(), CALIPER_DT_F32, {7}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot sb = snap("hist", CALIPER_LIVE_KIND_HISTOGRAM, &tb);
    REQUIRE(caliper_core_live_apply(c.core, &sb) == 1);

    c.frames(3);
    // Sweep the mouse over the canvas and drag (orbit) everywhere: exercises
    // hover picking, ImPlot tooltips and the camera paths.
    auto ev_move = [&](float x, float y) {
        CaliperInputEvent e{};
        e.struct_size = sizeof e;
        e.type = CALIPER_EVENT_MOUSE_MOVE;
        e.x = x;
        e.y = y;
        caliper_core_event(c.core, &e);
    };
    auto ev_button = [&](int b, bool down) {
        CaliperInputEvent e{};
        e.struct_size = sizeof e;
        e.type = CALIPER_EVENT_MOUSE_BUTTON;
        e.button = b;
        e.down = down ? 1 : 0;
        caliper_core_event(c.core, &e);
    };
    auto ev_wheel = [&](float dy) {
        CaliperInputEvent e{};
        e.struct_size = sizeof e;
        e.type = CALIPER_EVENT_MOUSE_SCROLL;
        e.dy = dy;
        caliper_core_event(c.core, &e);
    };
    for (float y = 60; y < 700; y += 90)
        for (float x = 40; x < 940; x += 110) {
            ev_move(x, y);
            c.frames(1);
        }
    for (float y = 80; y < 700; y += 160)
        for (float x = 60; x < 940; x += 200) {
            ev_move(x, y);
            c.frames(1);
            ev_button(1, true);
            c.frames(1);
            ev_move(x + 30, y + 10);
            c.frames(1);
            ev_button(1, false);
            ev_wheel(1.0f);
            c.frames(1);
        }
    std::vector<uint8_t> canvas((size_t)960 * 720 * 4);
    REQUIRE(caliper_core_read_pixels(c.core, canvas.data(), 960 * 4) == 1);
    int bright = 0;
    for (size_t i = 0; i < canvas.size(); i += 4)
        if (std::max(canvas[i], std::max(canvas[i + 1], canvas[i + 2])) > 80) ++bright;
    CHECK(bright > 1000);
    CHECK(caliper_core_live_view_count(c.core) == 4);
    // Live points: the current payload is drawn every redraw, still read_done
    // (nothing superseded).
    CHECK(caliper_core_live_view_read_done(c.core, "pts") == 1);
}

// ---------------------------------------------------------------------------
TEST_CASE("live_views/interaction: a drag in the plot pans the view and leaves its window; "
          "the title bar moves the window; the wheel zooms; a double click fits and follows") {
    Core c(960, 720);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    std::vector<float> hm = ramp(32, 24);
    CaliperTensor th = tensor(hm.data(), CALIPER_DT_F32, {24, 32}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot sh = snap("heat", CALIPER_LIVE_KIND_HEATMAP, &th);
    REQUIRE(caliper_core_live_apply(c.core, &sh) == 1);
    std::vector<float> ring(4096);
    for (size_t i = 0; i < ring.size(); ++i) ring[i] = std::sin((float)i * 0.05f);
    CaliperTensor tr = tensor(ring.data(), CALIPER_DT_F32, {(int64_t)ring.size()}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot sr = snap("loss", CALIPER_LIVE_KIND_SERIES, &tr);
    sr.series_total = 500;
    REQUIRE(caliper_core_live_apply(c.core, &sr) == 1);
    c.frames(4);

    auto move = [&](float x, float y) {
        CaliperInputEvent e{};
        e.struct_size = sizeof e;
        e.type = CALIPER_EVENT_MOUSE_MOVE;
        e.x = x;
        e.y = y;
        caliper_core_event(c.core, &e);
        c.frames(1);
    };
    auto button = [&](bool down) {
        CaliperInputEvent e{};
        e.struct_size = sizeof e;
        e.type = CALIPER_EVENT_MOUSE_BUTTON;
        e.button = 0;
        e.down = down ? 1 : 0;
        caliper_core_event(c.core, &e);
        c.frames(1);
    };
    auto wheel = [&](float dy) {
        CaliperInputEvent e{};
        e.struct_size = sizeof e;
        e.type = CALIPER_EVENT_MOUSE_SCROLL;
        e.dy = dy;
        caliper_core_event(c.core, &e);
        c.frames(1);
    };
    auto drag = [&](float x, float y, float dx, float dy) {
        move(x, y);
        button(true);
        for (int k = 1; k <= 4; ++k) move(x + dx * k / 4, y + dy * k / 4);
        button(false);
    };
    auto window = [](const char* label) { return ImGui::FindWindowByName(label); };
    auto window_of = [](const caliper_live_views::View* v) {
        return std::vector<double>(v->geo_window, v->geo_window + 4);
    };

    // The heatmap: a drag inside the image pans it; its window stays put.
    caliper_live_views::View* hv = caliper_live_views::find(c.core, "heat");
    REQUIRE(hv != nullptr);
    REQUIRE(hv->geo_valid);
    ImGuiWindow* hw = window("heat###caliper.view.heat");
    REQUIRE(hw != nullptr);
    const ImVec2 hpos = hw->Pos;
    const std::vector<double> fit = window_of(hv);
    const float cx = hv->geo_area[0] + hv->geo_area[2] * 0.5f, cy = hv->geo_area[1] + hv->geo_area[3] * 0.5f;
    drag(cx, cy, 60.0f, 20.0f);
    CHECK(hv->geo_window[0] < fit[0] - 0.5);      // the content followed the pointer right: the window of cells moved left
    CHECK(hv->geo_window[2] < fit[2] - 0.1);
    CHECK(hw->Pos.x == hpos.x);
    CHECK(hw->Pos.y == hpos.y);
    // The wheel zooms about the pointer, and a double click fits again.
    move(cx, cy);
    const double span = hv->geo_window[1] - hv->geo_window[0];
    wheel(2.0f);
    CHECK(hv->geo_window[1] - hv->geo_window[0] < span * 0.9);
    button(true);
    button(false);
    button(true);
    button(false);
    CHECK(window_of(hv) == fit);
    CHECK(hw->Pos.x == hpos.x);
    // A drag on the title bar still moves the window.
    const float tx = hw->Pos.x + 40.0f, ty = hw->Pos.y + hw->TitleBarHeight * 0.5f;
    drag(tx, ty, 50.0f, 30.0f);
    CHECK(hw->Pos.x == doctest::Approx(hpos.x + 50.0f).epsilon(0.01));
    CHECK(hw->Pos.y == doctest::Approx(hpos.y + 30.0f).epsilon(0.01));

    // The series follows its data until a drag in the plot, which holds the
    // view while values arrive; a double click follows again.
    caliper_live_views::View* lv = caliper_live_views::find(c.core, "loss");
    REQUIRE(lv != nullptr);
    REQUIRE(lv->geo_valid);
    CHECK(lv->follow);
    ImGuiWindow* lw = window("loss###caliper.view.loss");
    REQUIRE(lw != nullptr);
    const ImVec2 lpos = lw->Pos;
    const std::vector<double> lfit = window_of(lv);
    const float lx = lv->geo_area[0] + lv->geo_area[2] * 0.5f, ly = lv->geo_area[1] + lv->geo_area[3] * 0.5f;
    drag(lx, ly, -80.0f, 0.0f);
    c.frames(2);
    CHECK_FALSE(lv->follow);
    CHECK(lv->geo_window[0] > lfit[0]);           // panned toward later steps
    CHECK(lw->Pos.x == lpos.x);
    CHECK(lw->Pos.y == lpos.y);
    const std::vector<double> held = window_of(lv);
    sr.series_total = 1500;                       // more values: the held view does not move
    REQUIRE(caliper_core_live_apply(c.core, &sr) == 1);
    c.frames(3);
    CHECK(window_of(lv) == held);
    move(lx, ly);
    button(true);
    button(false);
    button(true);
    button(false);
    c.frames(2);
    CHECK(lv->follow);
    CHECK(lv->geo_window[1] > held[1] + 500.0);   // following again: the window reaches step 1499
}

TEST_CASE("live_views/points: a drag whose movement and release land in one frame still pans") {
    Core c(800, 600);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    std::vector<float> pts = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0};
    CaliperTensor tp = tensor(pts.data(), CALIPER_DT_F32, {4, 3}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot sp = snap("pts2", CALIPER_LIVE_KIND_POINTS, &tp);
    sp.points_dims = 2;
    REQUIRE(caliper_core_live_apply(c.core, &sp) == 1);
    c.frames(4);
    auto send = [&](CaliperEventType type, float x, float y, int down) {
        CaliperInputEvent e{};
        e.struct_size = sizeof e;
        e.type = type;
        e.x = x;
        e.y = y;
        e.button = 0;
        e.down = down;
        caliper_core_event(c.core, &e);
    };
    caliper_live_views::View* pv = caliper_live_views::find(c.core, "pts2");
    REQUIRE(pv != nullptr);
    REQUIRE(pv->geo_valid);
    const double x0 = pv->geo_window[0], span = pv->geo_window[1] - pv->geo_window[0];
    const float cx = pv->geo_area[0] + pv->geo_area[2] * 0.5f, cy = pv->geo_area[1] + pv->geo_area[3] * 0.5f;
    send(CALIPER_EVENT_MOUSE_MOVE, cx, cy, 0);
    c.frames(1);
    send(CALIPER_EVENT_MOUSE_BUTTON, cx, cy, 1);
    c.frames(1);
    // The whole movement and the release in one frame.
    send(CALIPER_EVENT_MOUSE_MOVE, cx + 80.0f, cy, 0);
    send(CALIPER_EVENT_MOUSE_BUTTON, cx + 80.0f, cy, 0);
    c.frames(3);
    // The points followed the pointer 80 px right: the window moved left by as much.
    CHECK(pv->geo_window[0] == doctest::Approx(x0 - span * 80.0 / pv->geo_area[2]).epsilon(0.02));
}

TEST_CASE("live_views/gates: every refusal is specific; nothing is created") {
    Core c(64, 64, /*attach=*/false);
    std::vector<float> h(16, 0.0f);
    CaliperTensor th = tensor(h.data(), CALIPER_DT_F32, {4, 4}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot s = snap("g", CALIPER_LIVE_KIND_HEATMAP, &th);
    CHECK(caliper_core_live_apply(c.core, &s) == 0);
    CHECK(has(c.err(), "attach a canvas first"));
    CHECK(caliper_core_live_apply(nullptr, &s) == 0);

    CaliperCanvasDesc cd{};
    cd.struct_size = sizeof cd;
    cd.mode = CALIPER_CANVAS_OFFSCREEN;
    cd.width = 64;
    cd.height = 64;
    cd.content_scale = 1.0f;
    c.gpu = !caliper_test::force_no_gpu() &&
            caliper_core_attach_canvas(c.core, nullptr, &cd) == 1;
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();

    auto refuse = [&](CaliperLiveSnapshot x, const char* sub) {
        const int r = caliper_core_live_apply(c.core, &x);
        CHECK_MESSAGE(r == 0, sub);
        CHECK_MESSAGE(has(c.err(), sub), "want '" << sub << "' in: " << c.err());
    };
    CHECK(caliper_core_live_apply(c.core, nullptr) == 0);
    CHECK(has(c.err(), "null/short CaliperLiveSnapshot"));
    CaliperLiveSnapshot shortsz = s;
    shortsz.struct_size = sizeof(CaliperLiveSnapshot) - 8;
    refuse(shortsz, "null/short CaliperLiveSnapshot");
    CaliperLiveSnapshot noname = s;
    noname.name = "";
    refuse(noname, "name must be a non-empty string");
    refuse(snap("g", 0, &th), "bad kind 0");
    refuse(snap("g", 6, &th), "bad kind 6");
    refuse(snap("g", CALIPER_LIVE_KIND_HEATMAP, &th, 0, 1u << 7), "unknown flag bits 0x80");
    refuse(snap("g", CALIPER_LIVE_KIND_HEATMAP, nullptr), "null data tensor");
    {
        CaliperLiveSnapshot x = s;
        x.stats_count = 2;
        refuse(x, "stats is NULL but stats_count > 0");
        x.stats_count = -1;
        refuse(x, "stats_count must be >= 0");
    }
    // dtype / ndim / shape per kind
    std::vector<uint8_t> u8(4 * 4 * 3);
    CaliperTensor rgb = tensor(u8.data(), CALIPER_DT_U8, {4, 4, 3}, CALIPER_DEV_CPU);
    refuse(snap("g", CALIPER_LIVE_KIND_HEATMAP, &rgb), "heatmap wants a 2-D f32 (H,W)");
    refuse(snap("g", CALIPER_LIVE_KIND_IMAGE, &rgb), "image wants a 3-D u8 (H,W,4)");
    refuse(snap("g", CALIPER_LIVE_KIND_IMAGE, &th), "image wants a 3-D u8 (H,W,4)");
    std::vector<float> p2(8);
    CaliperTensor n2 = tensor(p2.data(), CALIPER_DT_F32, {4, 2}, CALIPER_DEV_CPU);
    refuse(snap("g", CALIPER_LIVE_KIND_POINTS, &n2), "points wants a 2-D f32 (N,3)");
    refuse(snap("g", CALIPER_LIVE_KIND_SERIES, &th), "series wants a 1-D f32");
    refuse(snap("g", CALIPER_LIVE_KIND_HISTOGRAM, &rgb), "histogram wants a 1-D f32");
    {
        CaliperLiveSnapshot x = s;
        x.colormap = 7;
        refuse(x, "bad colormap 7");
    }
    std::vector<float> p3(12), aux3(3);
    CaliperTensor n3 = tensor(p3.data(), CALIPER_DT_F32, {4, 3}, CALIPER_DEV_CPU);
    CaliperTensor a3 = tensor(aux3.data(), CALIPER_DT_F32, {3}, CALIPER_DEV_CPU);
    {
        CaliperLiveSnapshot x = snap("g", CALIPER_LIVE_KIND_POINTS, &n3);
        x.points_dims = 4;
        refuse(x, "points_dims must be 2 or 3");
        x.points_dims = 2;
        x.label_count = 3;
        refuse(x, "label_count > 0 needs aux");
        x.aux = &a3;
        refuse(x, "aux must be (N,) f32 with N = 4");
        x.label_count = -1;
        refuse(x, "label_count must be >= 0");
    }
    // non-contiguous
    {
        CaliperTensor nc = th;
        nc.strides[0] = 8;
        refuse(snap("g", CALIPER_LIVE_KIND_HEATMAP, &nc), "row-major contiguous");
    }
    // misaligned offsets (METAL and CPU f32)
    id<MTLBuffer> mb = shared_buffer(dev, 0, h.data(), h.size() * 4 + 64);
    CaliperTensor tm = tensor(ptr(mb), CALIPER_DT_F32, {4, 4}, CALIPER_DEV_METAL);
    refuse(snap("g", CALIPER_LIVE_KIND_HEATMAP, &tm, 2), "data_offset must be a multiple of 4");
    refuse(snap("g", CALIPER_LIVE_KIND_HEATMAP, &th, 6), "data_offset must be a multiple of 4");
    {
        CaliperTensor nullbuf = tm;
        nullbuf.data = nullptr;
        refuse(snap("g", CALIPER_LIVE_KIND_HEATMAP, &nullbuf), "null MTLBuffer");
        CaliperTensor cuda = th;
        cuda.device = CALIPER_DEV_CUDA;
        refuse(snap("g", CALIPER_LIVE_KIND_HEATMAP, &cuda), "device must be CPU or METAL");
    }
    // METAL window beyond the buffer (the layer's import refuses it for points; series check it up front)
    {
        std::vector<float> big(300);
        CaliperTensor tbig = tensor(ptr(mb), CALIPER_DT_F32, {100, 3}, CALIPER_DEV_METAL);
        refuse(snap("g", CALIPER_LIVE_KIND_POINTS, &tbig), "the MTLBuffer is shorter than the size claimed");
        CaliperTensor sbig = tensor(ptr(mb), CALIPER_DT_F32, {1000}, CALIPER_DEV_METAL);
        CaliperLiveSnapshot x = snap("g", CALIPER_LIVE_KIND_SERIES, &sbig);
        x.series_total = 5;
        refuse(x, "exceeds the MTLBuffer length");
    }
    // NULL contents for the CPU-read kinds: a private-storage buffer
    {
        id<MTLBuffer> priv = [dev newBufferWithLength:256 options:MTLResourceStorageModePrivate];
        REQUIRE(ptr(priv) != nullptr);
        CaliperTensor tp = tensor(ptr(priv), CALIPER_DT_F32, {16}, CALIPER_DEV_METAL);
        CaliperLiveSnapshot x = snap("g", CALIPER_LIVE_KIND_SERIES, &tp);
        x.series_total = 16;
        refuse(x, "needs a CPU-visible payload");
        refuse(snap("g", CALIPER_LIVE_KIND_HISTOGRAM, &tp), "needs a CPU-visible payload");
        // ... but a heatmap from private storage is fine (the GPU reads it).
        CaliperTensor th2 = tensor(ptr(priv), CALIPER_DT_F32, {4, 16}, CALIPER_DEV_METAL);
        CaliperLiveSnapshot ok = snap("priv", CALIPER_LIVE_KIND_HEATMAP, &th2);
        CHECK_MESSAGE(caliper_core_live_apply(c.core, &ok) == 1, c.err());
        CHECK(caliper_core_live_view_remove(c.core, "priv") == 1);
    }
    // mixed devices for points aux
    {
        std::vector<float> aux4(4);
        CaliperTensor a4 = tensor(aux4.data(), CALIPER_DT_F32, {4}, CALIPER_DEV_CPU);
        id<MTLBuffer> pb = shared_buffer(dev, 0, p3.data(), p3.size() * 4);
        CaliperTensor tp = tensor(ptr(pb), CALIPER_DT_F32, {4, 3}, CALIPER_DEV_METAL);
        CaliperLiveSnapshot x = snap("g", CALIPER_LIVE_KIND_POINTS, &tp);
        x.aux = &a4;
        refuse(x, "same device as data");
    }
    // a name owned by a plain v1.2 view (caliper.show) is not taken over by points
    {
        CaliperViewDesc vd{};
        vd.struct_size = sizeof vd;
        vd.name = "shown";
        vd.tensor = &th;
        vd.vmin = 0.0f;
        vd.vmax = 1.0f;
        REQUIRE(caliper_core_view_tensor(c.core, &vd) == 1);
        refuse(snap("shown", CALIPER_LIVE_KIND_POINTS, &n3), "taken by an embed v1.2 view");
        caliper_core_view_remove(c.core, "shown");
    }
    CHECK(caliper_core_live_view_count(c.core) == 0);

    // per-view queries on unknown names and wrong kinds
    CaliperLiveViewInfo vi{};
    vi.struct_size = sizeof vi - 4;
    CHECK(caliper_core_live_view_info(c.core, "g", &vi) == 0);
    CHECK(has(c.err(), "null/short CaliperLiveViewInfo"));
    vi.struct_size = sizeof vi;
    CHECK(caliper_core_live_view_info(c.core, "nope", &vi) == 0);
    CHECK(has(c.err(), "no live view named 'nope'"));
    CHECK(caliper_core_live_view_remove(c.core, "nope") == 0);
    float out[4];
    size_t n = 0;
    CHECK(caliper_core_live_view_values(c.core, "nope", out, 4, &n) == 0);
    REQUIRE(caliper_core_live_apply(c.core, &s) == 1);
    CHECK(caliper_core_live_view_values(c.core, "g", out, 4, &n) == 0);
    CHECK(has(c.err(), "holds no values"));
    CHECK(caliper_core_live_view_values(c.core, "g", nullptr, 4, &n) == 0);
    std::vector<float> ring = {1, 2};
    CaliperTensor tr = tensor(ring.data(), CALIPER_DT_F32, {2}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot ss = snap("s", CALIPER_LIVE_KIND_SERIES, &tr);
    REQUIRE(caliper_core_live_apply(c.core, &ss) == 1);
    uint8_t px[64];
    CHECK(caliper_core_live_view_readback(c.core, "s", px, sizeof px) == 0);
    CHECK(has(c.err(), "has no texture"));
    CHECK(caliper_core_live_view_readback(c.core, "g", px, 8) == 0);   // too small (4x4x4)
    CaliperLiveSnapshot sp = snap("p", CALIPER_LIVE_KIND_POINTS, &n3);
    REQUIRE(caliper_core_live_apply(c.core, &sp) == 1);
    CHECK(caliper_core_live_view_readback(c.core, "p", px, sizeof px) == 0);   // no frame yet
    CHECK(has(c.err(), "no render target yet"));
}

TEST_CASE("live_views/shutdown: live views of every kind tear down clean; a fresh core works") {
    {
        Core c(320, 240);
        if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        std::vector<float> h = ramp(8, 8);
        id<MTLBuffer> hb = shared_buffer(dev, 0, h.data(), h.size() * 4);
        CaliperTensor th = tensor(ptr(hb), CALIPER_DT_F32, {8, 8}, CALIPER_DEV_METAL);
        CaliperLiveSnapshot sh = snap("h", CALIPER_LIVE_KIND_HEATMAP, &th);
        REQUIRE(caliper_core_live_apply(c.core, &sh) == 1);   // in flight: not waited
        std::vector<float> pos = {0, 0, 0, 1, 0, 0, 0, 1, 0};
        id<MTLBuffer> pb = shared_buffer(dev, 0, pos.data(), pos.size() * 4);
        CaliperTensor tp = tensor(ptr(pb), CALIPER_DT_F32, {3, 3}, CALIPER_DEV_METAL);
        CaliperLiveSnapshot sp = snap("p", CALIPER_LIVE_KIND_POINTS, &tp);
        REQUIRE(caliper_core_live_apply(c.core, &sp) == 1);
        c.frames(1);
        sp.data_offset = 12;   // same buffer, another "slot": shares the import
        CaliperTensor tp2 = tensor(ptr(pb), CALIPER_DT_F32, {2, 3}, CALIPER_DEV_METAL);
        sp.data = &tp2;
        REQUIRE(caliper_core_live_apply(c.core, &sp) == 1);   // previous still claimed
        std::vector<float> ring = {1, 2, 3};
        CaliperTensor tr = tensor(ring.data(), CALIPER_DT_F32, {3}, CALIPER_DEV_CPU);
        CaliperLiveSnapshot ss = snap("s", CALIPER_LIVE_KIND_SERIES, &tr);
        ss.series_total = 3;
        REQUIRE(caliper_core_live_apply(c.core, &ss) == 1);
        CaliperLiveSnapshot sg = snap("hg", CALIPER_LIVE_KIND_HISTOGRAM, &tr);
        REQUIRE(caliper_core_live_apply(c.core, &sg) == 1);
        caliper_core_live_set_paused(c.core, 1);
        c.frames(2);
        CHECK(caliper_core_live_view_count(c.core) == 4);
    }   // shutdown with everything live
    Core c2(32, 32);
    if (!c2.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c2.err());
    CHECK(caliper_core_live_view_count(c2.core) == 0);
    CHECK(caliper_core_live_paused(c2.core) == 0);   // pause is per core
    std::vector<float> h(16, 0.25f);
    CaliperTensor th = tensor(h.data(), CALIPER_DT_F32, {4, 4}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot s = snap("again", CALIPER_LIVE_KIND_HEATMAP, &th);
    CHECK(caliper_core_live_apply(c2.core, &s) == 1);
    c2.frames(1);
}

TEST_CASE("live_views/hover: a CPU payload is copied at apply, so hover reads values the "
          "caller may already have freed") {
    const int W = 8, H = 4;
    Core c(32, 32);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    std::vector<float> r = ramp(W, H, -1.0f, 1.0f);
    std::vector<float> caller = r;
    CaliperTensor t = tensor(caller.data(), CALIPER_DT_F32, {H, W}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot s = snap("hov", CALIPER_LIVE_KIND_HEATMAP, &t);
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &s) == 1, c.err());
    std::fill(caller.begin(), caller.end(), 99.0f);   // the caller reuses its bytes
    const caliper_live_views::View* v = caliper_live_views::find(c.core, "hov");
    REQUIRE(v != nullptr);
    REQUIRE(v->tex_cpu != nullptr);
    CHECK(v->tex_cpu != (const uint8_t*)caller.data());
    CHECK(std::memcmp(v->tex_cpu, r.data(), r.size() * 4) == 0);
    CHECK(std::string(info(c.core, "hov").path) == "cpu-staged");   // a native backend's path
    CHECK_FALSE(v->latency_relative);
    // A network source marks its mapped stamps; the view labels the latency.
    CaliperLiveSnapshot sr = snap("hov", CALIPER_LIVE_KIND_HEATMAP, &t, 0, CALIPER_LIVE_APPLY_STAMP_RELATIVE);
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &sr) == 1, c.err());
    CHECK(caliper_live_views::find(c.core, "hov")->latency_relative);

    // An RGBA image keeps its bytes the same way.
    std::vector<uint8_t> px((size_t)W * H * 4);
    for (size_t i = 0; i < px.size(); ++i) px[i] = (uint8_t)i;
    std::vector<uint8_t> ref = px;
    CaliperTensor ti = tensor(px.data(), CALIPER_DT_U8, {H, W, 4}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot si = snap("img", CALIPER_LIVE_KIND_IMAGE, &ti);
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &si) == 1, c.err());
    std::fill(px.begin(), px.end(), 0);
    const caliper_live_views::View* vi = caliper_live_views::find(c.core, "img");
    REQUIRE(vi != nullptr);
    REQUIRE(vi->tex_cpu != nullptr);
    CHECK(std::memcmp(vi->tex_cpu, ref.data(), ref.size()) == 0);
}

// ---------------------------------------------------------------------------
// Copy counters (embed/render_stats.h; docs/explanations/rendering.md, "Copy
// counters and frame statistics"): each route reads its known copies. A METAL
// payload is read in place (0, also while a points view redraws it every
// frame); a CPU heatmap or image is uploaded once per update; CPU points are
// staged once per buffer; series and histogram values are copied at apply.
namespace {
caliper_stats::ViewCopies copies_of(CaliperCore* core, const char* name) {
    caliper_stats::ViewCopies v{};
    caliper_stats::view_copies(core, name, &v);
    return v;
}
}  // namespace

TEST_CASE("live_views/copies: each route counts its known copies; METAL payloads read 0") {
    const int W = 32, H = 16;
    Core c(320, 240);
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << c.err());
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    caliper_stats::FrameCopies fc{};
    caliper_stats::frame_copies(c.core, &fc, true);   // a fresh window

    // METAL heatmap (what an MPS tensor is): in place, twice, with frames.
    std::vector<float> r = ramp(W, H);
    id<MTLBuffer> hb = shared_buffer(dev, 0, r.data(), r.size() * 4);
    CaliperTensor th = tensor(ptr(hb), CALIPER_DT_F32, {H, W}, CALIPER_DEV_METAL);
    CaliperLiveSnapshot sh = snap("hm_gpu", CALIPER_LIVE_KIND_HEATMAP, &th);
    for (int i = 0; i < 2; ++i) {
        REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &sh) == 1, c.err());
        CHECK(wait_read_done(c, "hm_gpu"));
    }
    CHECK(std::string(info(c.core, "hm_gpu").path) == "compute-imported");
    CHECK(copies_of(c.core, "hm_gpu").total.copies == 0);
    CHECK(copies_of(c.core, "hm_gpu").total.bytes == 0);

    // METAL RGBA8 image: the blit into the view's texture is the update itself.
    std::vector<uint8_t> rgba((size_t)W * H * 4, 0x7F);
    id<MTLBuffer> ib = shared_buffer(dev, 0, rgba.data(), rgba.size());
    CaliperTensor ti = tensor(ptr(ib), CALIPER_DT_U8, {H, W, 4}, CALIPER_DEV_METAL);
    CaliperLiveSnapshot si = snap("img_gpu", CALIPER_LIVE_KIND_IMAGE, &ti);
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &si) == 1, c.err());
    CHECK(std::string(info(c.core, "img_gpu").path) == "blit-imported");
    CHECK(copies_of(c.core, "img_gpu").total.copies == 0);

    // METAL points with a per-point value: imported in place, redrawn each frame.
    const int N = 200;
    std::vector<float> pos(N * 3), val(N);
    for (int i = 0; i < N; ++i) {
        pos[3 * i + 0] = (float)i / N - 0.5f;
        pos[3 * i + 1] = 0.25f;
        pos[3 * i + 2] = 0.0f;
        val[i] = (float)i;
    }
    id<MTLBuffer> pb = shared_buffer(dev, 0, pos.data(), pos.size() * 4);
    id<MTLBuffer> vb = shared_buffer(dev, 0, val.data(), val.size() * 4);
    CaliperTensor tp = tensor(ptr(pb), CALIPER_DT_F32, {N, 3}, CALIPER_DEV_METAL);
    CaliperTensor tv = tensor(ptr(vb), CALIPER_DT_F32, {N}, CALIPER_DEV_METAL);
    CaliperLiveSnapshot sp = snap("pts_gpu", CALIPER_LIVE_KIND_POINTS, &tp);
    sp.aux = &tv;
    sp.vmin = 0.0f;
    sp.vmax = (float)(N - 1);
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &sp) == 1, c.err());
    c.frames(3);
    CHECK(std::string(info(c.core, "pts_gpu").path) == "points-imported");
    CHECK(copies_of(c.core, "pts_gpu").total.copies == 0);

    // Every view so far is zero-copy, and so is every frame that drew them.
    c.frames(2);
    REQUIRE(caliper_stats::frame_copies(c.core, &fc, true));
    CHECK(fc.window_frames >= 5);
    CHECK(fc.window.copies == 0);
    CHECK(fc.window.bytes == 0);
    CHECK(fc.last_frame.copies == 0);
    CHECK(caliper_stats::renderer_total(c.core).copies == 0);

    // CPU heatmap: one upload of the RGBA8 texels per update.
    CaliperTensor tc = tensor(r.data(), CALIPER_DT_F32, {H, W}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot sc = snap("hm_cpu", CALIPER_LIVE_KIND_HEATMAP, &tc);
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &sc) == 1, c.err());
    CHECK(copies_of(c.core, "hm_cpu").last.copies == 1);
    CHECK(copies_of(c.core, "hm_cpu").last.bytes == (uint64_t)W * H * 4);
    REQUIRE(caliper_core_live_apply(c.core, &sc) == 1);
    CHECK(copies_of(c.core, "hm_cpu").last.copies == 1);
    CHECK(copies_of(c.core, "hm_cpu").total.copies == 2);
    CHECK(copies_of(c.core, "hm_cpu").total.bytes == 2ull * W * H * 4);
    c.frames(1);
    REQUIRE(caliper_stats::frame_copies(c.core, &fc, true));
    CHECK(fc.window_frames == 1);
    CHECK(fc.window.copies == 2);   // both updates land in the frame that follows them
    CHECK(fc.last_frame.copies == 2);
    c.frames(1);
    REQUIRE(caliper_stats::frame_copies(c.core, &fc, true));
    CHECK(fc.window.copies == 0);   // a frame with no update copies nothing
    CHECK(fc.last_frame.copies == 0);

    // CPU image: one upload.
    CaliperTensor tci = tensor(rgba.data(), CALIPER_DT_U8, {H, W, 4}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot sci = snap("img_cpu", CALIPER_LIVE_KIND_IMAGE, &tci);
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &sci) == 1, c.err());
    CHECK(copies_of(c.core, "img_cpu").total.copies == 1);
    CHECK(copies_of(c.core, "img_cpu").total.bytes == (uint64_t)W * H * 4);

    // CPU points: positions and values are each staged once; drawing copies nothing more.
    CaliperTensor tpc = tensor(pos.data(), CALIPER_DT_F32, {N, 3}, CALIPER_DEV_CPU);
    CaliperTensor tvc = tensor(val.data(), CALIPER_DT_F32, {N}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot spc = snap("pts_cpu", CALIPER_LIVE_KIND_POINTS, &tpc);
    spc.aux = &tvc;
    spc.vmin = 0.0f;
    spc.vmax = (float)(N - 1);
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &spc) == 1, c.err());
    c.frames(3);
    CHECK(std::string(info(c.core, "pts_cpu").path) == "points-cpu-staged");
    CHECK(copies_of(c.core, "pts_cpu").total.copies == 2);
    CHECK(copies_of(c.core, "pts_cpu").total.bytes == (uint64_t)N * 12 + (uint64_t)N * 4);

    // Series and histogram: the values are copied into the view at apply.
    const int M = 16;
    std::vector<float> ring(M, 1.0f);
    CaliperTensor ts = tensor(ring.data(), CALIPER_DT_F32, {M}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot ss = snap("loss", CALIPER_LIVE_KIND_SERIES, &ts);
    ss.series_total = 10;
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &ss) == 1, c.err());
    CHECK(copies_of(c.core, "loss").total.copies == 1);
    CHECK(copies_of(c.core, "loss").total.bytes == 10u * 4u);
    CaliperLiveSnapshot sg = snap("grad", CALIPER_LIVE_KIND_HISTOGRAM, &ts);
    REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &sg) == 1, c.err());
    CHECK(copies_of(c.core, "grad").total.copies == 1);
    CHECK(copies_of(c.core, "grad").total.bytes == (uint64_t)M * 4);

    // The zero-copy views still read 0, and the renderer's total is the sum.
    CHECK(copies_of(c.core, "hm_gpu").total.copies == 0);
    CHECK(copies_of(c.core, "pts_gpu").total.copies == 0);
    CHECK(caliper_stats::renderer_total(c.core).copies == 2 + 1 + 2 + 1 + 1);
}

// Frame statistics (CALIPER_METAL_STATS, read when the renderer is made): frame
// times, the GPU time of each update and of each view draw, and the updates by
// path, over a window that a read resets. Without the variable nothing is kept.
TEST_CASE("live_views/frame stats: CALIPER_METAL_STATS records frame and GPU times by path") {
    {
        Core off(64, 48);
        if (!off.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU: " << off.err());
        caliper_host::HostRenderer::FrameStats fs;
        CHECK_FALSE(caliper_stats::frame_stats(off.core, &fs, false));   // off by default
    }
    setenv("CALIPER_METAL_STATS", "1", 1);
    {
        Core c(320, 240);
        REQUIRE(c.gpu);
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        const int W = 64, H = 32, N = 300;
        std::vector<float> r = ramp(W, H);
        id<MTLBuffer> hb = shared_buffer(dev, 0, r.data(), r.size() * 4);
        CaliperTensor th = tensor(ptr(hb), CALIPER_DT_F32, {H, W}, CALIPER_DEV_METAL);
        CaliperLiveSnapshot sh = snap("hm", CALIPER_LIVE_KIND_HEATMAP, &th);
        std::vector<float> pos(N * 3, 0.1f);
        for (int i = 0; i < N; ++i) pos[3 * i] = (float)i / N - 0.5f;
        id<MTLBuffer> pb = shared_buffer(dev, 0, pos.data(), pos.size() * 4);
        CaliperTensor tp = tensor(ptr(pb), CALIPER_DT_F32, {N, 3}, CALIPER_DEV_METAL);
        CaliperLiveSnapshot sp = snap("pts", CALIPER_LIVE_KIND_POINTS, &tp);
        caliper_host::HostRenderer::FrameStats fs;
        REQUIRE(caliper_stats::frame_stats(c.core, &fs, true));   // a fresh window
        for (int i = 0; i < 5; ++i) {
            REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &sh) == 1, c.err());
            REQUIRE_MESSAGE(caliper_core_live_apply(c.core, &sp) == 1, c.err());
            c.frames(1);
        }
        CHECK(wait_read_done(c, "hm"));
        // GPU times arrive on Metal's completion threads: wait for the counts they publish, for
        // up to 10 s, which only a failure reaches.
        for (int i = 0; i < 2000; ++i) {
            REQUIRE(caliper_stats::frame_stats(c.core, &fs, false));
            if (fs.update_gpu_n >= 5 && fs.draw_gpu_n >= 1 && fs.frame_gpu_n >= 4) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        REQUIRE(caliper_stats::frame_stats(c.core, &fs, true));
        CHECK(fs.frames >= 5);
        CHECK(fs.frame_ms_p50 > 0.0);
        CHECK(fs.frame_ms_p99 >= fs.frame_ms_p50);
        CHECK(fs.frame_ms_max >= fs.frame_ms_p99);
        CHECK(fs.update_gpu_n >= 5);
        CHECK(fs.update_gpu_ms_p50 > 0.0);
        CHECK(fs.draw_gpu_n >= 1);
        CHECK(fs.frame_gpu_n >= 4);
        std::map<std::string, uint64_t> upd(fs.updates.begin(), fs.updates.end());
        CHECK(upd["compute-imported"] >= 5);
        // The points draw through the graphics layer, not as a renderer update: their GPU time
        // reaches the view-draw series above (HostRenderer::note_draw_gpu_ms), and no update by path.
        CHECK(upd.count("primitives-imported") == 0);
        // The read reset the window.
        REQUIRE(caliper_stats::frame_stats(c.core, &fs, false));
        CHECK(fs.frames == 0);
        CHECK(fs.updates.empty());
    }
    unsetenv("CALIPER_METAL_STATS");
}

TEST_CASE("live_clock: producer stamps map into the viewer's clock from the fastest message") {
    caliper_live::StampClock clk;
    CHECK_FALSE(clk.ready());
    CHECK(clk.to_viewer(123) == 0);
    // The producer's clock reads 5 s more than the viewer's; messages take 3,
    // 1 and 2 ms to arrive.
    const uint64_t base = 100ull * 1000000000ull, skew = 5ull * 1000000000ull, ms = 1000000ull;
    clk.observe(base + skew, base + 3 * ms);
    clk.observe(base + skew + 10 * ms, base + 11 * ms);
    clk.observe(base + skew + 20 * ms, base + 22 * ms);
    REQUIRE(clk.ready());
    CHECK(clk.offset_ns() == (int64_t)(skew - 1 * ms));   // the 1 ms message is the fastest
    // A snapshot stamped 20 ms (producer) maps to 1 ms before it arrived at 22.
    CHECK(clk.to_viewer(base + skew + 20 * ms) == base + 21 * ms);
    CHECK(clk.to_viewer(0) == 0);
    // After the window, an old fast message no longer counts.
    const uint64_t later = base + (uint64_t)(caliper_live::StampClock::kWindow + 2) * 1000000000ull;
    clk.observe(later + skew, later + 4 * ms);
    CHECK(clk.offset_ns() == (int64_t)(skew - 4 * ms));
}

CALIPER_GPU_TEST_MAIN()
