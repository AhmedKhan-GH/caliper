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

#include "live/live_views_state.h"   // kPalette (white-box: the exact categorical colours)
#include "tensor_bridge.h"           // map_f32_to_rgba8 / colormap_lut (CPU reference)

#import <Metal/Metal.h>

#include <time.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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
    const uint32_t bg = 0xff000000u | (20u << 16) | (16u << 8) | 16u;
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
    const uint32_t bg = 0xff000000u | (20u << 16) | (16u << 8) | 16u;
    for (size_t i = 0; i < px.size(); i += 4) {
        uint32_t p;
        std::memcpy(&p, &px[i], 4);
        if (p == bg) continue;
        ++fg;
        in_lut += lut_colours.count(p) ? 1 : 0;
    }
    MESSAGE("3-D cloud: " << fg << " foreground px, " << in_lut << " in the magma LUT");
    CHECK(fg > 100);
    CHECK(in_lut == fg);   // opaque points: every lit pixel is an exact LUT colour
    CHECK(caliper_core_live_view_read_done(c.core, "cloud") == 1);
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
    // METAL window beyond the buffer (points and series check it up front)
    {
        std::vector<float> big(300);
        CaliperTensor tbig = tensor(ptr(mb), CALIPER_DT_F32, {100, 3}, CALIPER_DEV_METAL);
        refuse(snap("g", CALIPER_LIVE_KIND_POINTS, &tbig), "exceeds the MTLBuffer length");
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

CALIPER_GPU_TEST_MAIN()
