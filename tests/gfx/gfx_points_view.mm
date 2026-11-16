// gfx_points_view.mm: the points live view's rows of the graphics suite (caliper_gfx_views_tests).
//
// Each case applies a points snapshot to an offscreen core through caliper_core_live_apply, as both
// transports do, runs frames, and reads the view's render target back. The cases cover what the
// points view draws: 2-D and 3-D; flat colour, categorical labels through the fixed 16-colour
// palette, and values through each colormap with fixed and automatic ranges, NaN values among them;
// every point size class by count (7, 5.5, 4 and 3 pixels); the dimmed rest of a selection; a
// single point; clouds of up to 4,000,000 points; and the three routes of a payload: a CPU tensor,
// a shared-storage MTLBuffer and a private-storage one, as an MPS tensor is.
//
// The 2-D lattices place every point at a chosen sub-pixel phase (0.5 is a pixel centre), so that
// the cases where a square's edges fall exactly on pixel centres are drawn on purpose. For lattice
// cases the row also checks each point's centre pixel against the CPU reference colour: the flat
// grey, the palette colour of the label's level, or the colormap's colour of the value's level as
// map_f32_to_rgba8 decides it, dimmed toward the ground for the unselected points of a selection.
//
// The reference mode (gfx_reference.h, tests/gfx/references/points_view.txt), recorded at the first
// commit of migration step 7 from the old path (hardware point sprites through the Metal
// renderer's geometry draw), holds each case's readback hash, its path label, whether it imported,
// the copies of its update and its header text. The rows that the step promises unchanged must
// match it byte for byte. The scene core composes the camera in double where the old view
// composed it in float, so a 3-D cloud of a million points or more may differ from the old view in
// a few pixels where a point's edge lies within rounding of a pixel centre: those cases
// ("near_ties") are compared with the old path's images in the second row, under a bound.
//
//   CALIPER_GFX_POINTS_CASES=<substring>   runs only the cases whose names contain it
//   CALIPER_GFX_POINTS_BENCH=1             runs the timing row (draw cost by count)
#define DOCTEST_CONFIG_IMPLEMENT
#include "../gpu_skip.h"

#include "gfx_reference.h"

#include <caliper/embed.h>
#include <caliper/live_views.h>

#include "embed/embed_core.h"
#include "embed/render_stats.h"
#include "gfx/colormap_keys.h"      // reference_index: the CPU reference's colormap level
#include "live/live_views_state.h"
#include "renderer/host_renderer.h"
#include "tensor_bridge.h"          // colormap_lut

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <stb_image.h>
#include <stb_image_write.h>
#pragma clang diagnostic pop

#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

namespace ref = gfx_reference;

constexpr int kCanvasW = 800, kCanvasH = 600;
constexpr uint32_t kFlat = 0xff000000u | (230u << 16) | (220u << 8) | 220u;   // the points view's grey

// One offscreen core, shut down when the case ends.
struct Core {
    CaliperCore* core = nullptr;
    bool gpu = false;
    Core() {
        CaliperCoreDesc d{};
        d.struct_size = sizeof d;
        d.renderer = CALIPER_RENDERER_DEFAULT;
        core = caliper_core_create(&d);
        REQUIRE(core != nullptr);
        if (caliper_test::force_no_gpu()) return;
        CaliperCanvasDesc c{};
        c.struct_size = sizeof c;
        c.mode = CALIPER_CANVAS_OFFSCREEN;
        c.width = kCanvasW;
        c.height = kCanvasH;
        c.content_scale = 1.0f;
        gpu = caliper_core_attach_canvas(core, nullptr, &c) == 1 && MTLCreateSystemDefaultDevice() != nil;
    }
    ~Core() { caliper_core_shutdown(core); }
    void frames(int n) { for (int i = 0; i < n; ++i) caliper_core_frame(core); }
};

CaliperTensor tensor(void* data, std::vector<int64_t> shape, CaliperDeviceKind dev) {
    CaliperTensor t{};
    t.struct_size = sizeof t;
    t.data = data;
    t.dtype = CALIPER_DT_F32;
    t.ndim = (int32_t)shape.size();
    for (int i = 0; i < t.ndim; ++i) t.shape[i] = shape[i];
    int64_t st = 1;
    for (int i = t.ndim - 1; i >= 0; --i) { t.strides[i] = st; st *= t.shape[i]; }
    t.device = dev;
    return t;
}

// ---- The cases -----------------------------------------------------------------------------------

enum class Layout { Lattice, Random, Single };
enum class Colour { Flat, Values, Classes };
enum class Route { Shared, Private, Cpu };

struct Case {
    std::string name;
    int dims = 3;
    uint64_t n = 0;
    Layout layout = Layout::Random;
    float phase = 0.5f;           // lattices: the sub-pixel position of every point
    Colour colour = Colour::Flat;
    int colormap = 0;
    bool auto_range = false;      // values: the range from stats[0..1]
    float vmin = 0.0f, vmax = 1.0f;
    bool nans = false;            // values: every 37th value is NaN
    int classes = 0;              // labels: label_count
    Route route = Route::Shared;
    bool selection = false;       // a linked selection of every third point
    bool bbox = true;             // the producer's bounding box in stats[2..7]
    bool near_ties = false;       // a 3-D cloud of a million points or more (see the file's comment)
    unsigned seed = 1;
};

std::vector<Case> cases() {
    std::vector<Case> out;
    const struct { const char* size; uint64_t n; } sizes[] = {{"7px", 400}, {"5.5px", 3000}, {"4px", 20000}, {"3px", 60000}};
    const struct { const char* tag; float phase; } phases[] = {{"p50", 0.5f}, {"p0", 0.0f}, {"p25", 0.25f}, {"p12", 0.125f}};
    for (const auto& s : sizes)
        for (const auto& p : phases) {
            Case c;
            c.name = std::string("lattice-") + s.size + "-" + p.tag;
            c.dims = 2;
            c.n = s.n;
            c.layout = Layout::Lattice;
            c.phase = p.phase;
            out.push_back(c);
        }
    auto lattice = [&](const char* name, uint64_t n) {
        Case c;
        c.name = name;
        c.dims = 2;
        c.n = n;
        c.layout = Layout::Lattice;
        return c;
    };
    auto values = [](Case c, int cm, bool autor, float lo, float hi, bool nans) {
        c.colour = Colour::Values;
        c.colormap = cm;
        c.auto_range = autor;
        c.vmin = lo;
        c.vmax = hi;
        c.nans = nans;
        return c;
    };
    auto classes = [](Case c, int k) {
        c.colour = Colour::Classes;
        c.classes = k;
        return c;
    };
    auto route = [](Case c, Route r) { c.route = r; return c; };
    auto select = [](Case c) { c.selection = true; return c; };
    out.push_back(values(lattice("2d-viridis-fixed-nan", 3000), 0, false, -1.0f, 2.0f, true));
    out.push_back(route(values(lattice("2d-magma-auto-cpu", 3000), 1, true, 0, 0, false), Route::Cpu));
    out.push_back(route(values(lattice("2d-rdbu-auto-nan-private", 3000), 2, true, 0, 0, true), Route::Private));
    out.push_back(classes(lattice("2d-classes", 400), 20));
    out.push_back(select(lattice("2d-flat-selection", 3000)));
    out.push_back(route(select(values(lattice("2d-viridis-selection-cpu", 3000), 0, true, 0, 0, false)), Route::Cpu));
    out.push_back(select(classes(lattice("2d-classes-selection", 400), 20)));
    auto random = [](const char* name, int dims, uint64_t n, unsigned seed) {
        Case c;
        c.name = name;
        c.dims = dims;
        c.n = n;
        c.seed = seed;
        return c;
    };
    Case single2 = random("2d-single", 2, 1, 2);
    single2.layout = Layout::Single;
    out.push_back(single2);
    Case single3 = random("3d-single", 3, 1, 3);
    single3.layout = Layout::Single;
    out.push_back(single3);
    out.push_back(values(random("2d-viridis-auto-1000000", 2, 1000000, 4), 0, true, 0, 0, false));
    Case scan = route(random("3d-flat-400-cpu-scan", 3, 400, 5), Route::Cpu);
    scan.bbox = false;   // the fit scans the CPU payload
    out.push_back(scan);
    out.push_back(classes(random("3d-classes-1000", 3, 1000, 6), 10));
    out.push_back(route(classes(random("3d-classes-1000-cpu", 3, 1000, 6), 10), Route::Cpu));
    out.push_back(route(values(random("3d-viridis-auto-3000-private", 3, 3000, 7), 0, true, 0, 0, false), Route::Private));
    out.push_back(route(values(random("3d-magma-fixed-nan-20000-cpu", 3, 20000, 8), 1, false, -1.5f, 1.5f, true), Route::Cpu));
    out.push_back(values(random("3d-rdbu-auto-60000", 3, 60000, 9), 2, true, 0, 0, false));
    out.push_back(values(random("3d-viridis-auto-nan-100000", 3, 100000, 10), 0, true, 0, 0, true));
    out.push_back(select(classes(random("3d-classes-selection-3000", 3, 3000, 11), 10)));
    Case m1 = values(random("3d-viridis-auto-1000000", 3, 1000000, 12), 0, true, 0, 0, false);
    m1.near_ties = true;
    out.push_back(m1);
    Case m4 = values(random("3d-viridis-auto-4000000", 3, 4000000, 13), 0, true, 0, 0, false);
    m4.near_ties = true;
    out.push_back(m4);
    return out;
}

bool selected_case(const Case& c) {
    const char* only = std::getenv("CALIPER_GFX_POINTS_CASES");
    return !only || !*only || c.name.find(only) != std::string::npos;
}

// The pitch of a lattice for a size class: wider than half the square plus half a pixel, so that
// no point covers its neighbours' centre pixels.
int lattice_pitch(uint64_t n) { return n > 50000 ? 2 : n > 5000 ? 3 : n > 500 ? 4 : 5; }

// A lattice label: the label count's classes in turn, with labels above 255, below 0, at 255, at 256
// and NaN among them.
float lattice_label(uint64_t i, int classes) {
    if (i % 61 == 7) return NAN;
    if (i % 53 == 3) return 300.0f;
    if (i % 59 == 4) return -3.0f;
    if (i % 67 == 5) return 255.0f;
    if (i % 71 == 6) return 256.0f;
    return (float)(i % (uint64_t)classes);
}

struct Payload {
    std::vector<float> pos, aux;
    std::vector<float> px;            // lattices: each point's intended pixel position (x, y)
    float stats[8] = {};
    int stats_count = 0;
};

// The case's positions and aux. Lattices need the render target's size, which the first frame sets.
Payload make_payload(const Case& c, int W, int H) {
    Payload p;
    p.pos.assign(c.n * 3, 0.0f);
    std::mt19937 rng(c.seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    if (c.layout == Layout::Lattice) {
        // The 2-D fit to the box [-1, 1]^2: half height max(1, 1 / aspect) * 1.08, centred.
        const double aspect = double(W) / double(H);
        const double hh = std::max(1.0, 1.0 / aspect) * 1.08, hw = hh * aspect;
        const int pitch = lattice_pitch(c.n);
        const int cols = (W - 16) / pitch, rows = (H - 16) / pitch;
        p.px.resize(c.n * 2);
        for (uint64_t i = 0; i < c.n; ++i) {
            const uint64_t k = i % (uint64_t)(cols * rows);
            const double x = 8.0 + double(k % (uint64_t)cols) * pitch + c.phase;
            const double y = 8.0 + double(k / (uint64_t)cols) * pitch + c.phase;
            p.px[2 * i] = (float)x;
            p.px[2 * i + 1] = (float)y;
            p.pos[3 * i] = (float)((2.0 * x / W - 1.0) * hw);
            p.pos[3 * i + 1] = (float)((1.0 - 2.0 * y / H) * hh);
        }
    } else if (c.layout == Layout::Single) {
        p.pos[0] = 0.25f;
        p.pos[1] = -0.5f;
        p.pos[2] = c.dims == 3 ? 0.75f : 0.0f;
    } else {
        for (uint64_t i = 0; i < c.n; ++i)
            for (int k = 0; k < 3; ++k) p.pos[3 * i + k] = (k == 2 && c.dims == 2) ? 0.0f : g(rng);
    }
    if (c.colour == Colour::Values) {
        p.aux.resize(c.n);
        const float span = c.auto_range ? 2.0f : c.vmax - c.vmin;
        const float lo = c.auto_range ? -1.0f : c.vmin - 0.25f * span, hi = c.auto_range ? 1.0f : c.vmax + 0.25f * span;
        for (uint64_t i = 0; i < c.n; ++i) {
            if (c.layout == Layout::Random) p.aux[i] = g(rng);
            else p.aux[i] = lo + (hi - lo) * (float)std::fmod(double(i) * 0.6180339887498949, 1.0);
            if (c.nans && i % 37 == 5) p.aux[i] = NAN;
        }
    } else if (c.colour == Colour::Classes) {
        p.aux.resize(c.n);
        for (uint64_t i = 0; i < c.n; ++i)
            p.aux[i] = c.layout == Layout::Lattice ? lattice_label(i, c.classes) : (float)((i * 7919u) % (uint64_t)c.classes);
    }
    // stats: the aux range (finite values), then the bounding box.
    float lo = INFINITY, hi = -INFINITY;
    for (float v : p.aux)
        if (std::isfinite(v)) { lo = std::min(lo, v); hi = std::max(hi, v); }
    p.stats[0] = lo;
    p.stats[1] = hi;
    float b[6] = {INFINITY, -INFINITY, INFINITY, -INFINITY, INFINITY, -INFINITY};
    for (uint64_t i = 0; i < c.n; ++i)
        for (int k = 0; k < 3; ++k) {
            b[2 * k] = std::min(b[2 * k], p.pos[3 * i + k]);
            b[2 * k + 1] = std::max(b[2 * k + 1], p.pos[3 * i + k]);
        }
    if (c.layout == Layout::Lattice) {   // the box the lattice was placed for
        b[0] = b[2] = -1.0f;
        b[1] = b[3] = 1.0f;
        b[4] = b[5] = 0.0f;
    }
    std::memcpy(p.stats + 2, b, sizeof b);
    p.stats_count = c.bbox ? 8 : (c.colour == Colour::Values ? 2 : 0);
    return p;
}

// A buffer of the payload's bytes after `junk` bytes of 0xAB, in shared or private storage.
id<MTLBuffer> metal_buffer(id<MTLDevice> dev, const std::vector<float>& v, size_t junk, bool priv) {
    std::vector<uint8_t> b(junk + v.size() * 4, 0xAB);
    std::memcpy(b.data() + junk, v.data(), v.size() * 4);
    id<MTLBuffer> shared = [dev newBufferWithBytes:b.data() length:b.size() options:MTLResourceStorageModeShared];
    if (!priv) return shared;
    id<MTLBuffer> out = [dev newBufferWithLength:b.size() options:MTLResourceStorageModePrivate];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:shared sourceOffset:0 toBuffer:out destinationOffset:0 size:b.size()];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    return out;
}

// What one case drew.
struct Result {
    int width = 0, height = 0;
    std::vector<uint8_t> rgba;
    ref::Line line;
    Payload payload;
    std::string error;
};

uint32_t pixel(const Result& r, int x, int y) {
    uint32_t p;
    std::memcpy(&p, &r.rgba[((size_t)y * r.width + x) * 4], 4);
    return p;
}

// Applies the case to a fresh core and reads its view back.
bool draw_case(const Case& c, Result* out) {
    Core core;
    if (!core.gpu) return false;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)core.core->renderer->native_device();
    REQUIRE(bool(dev != nil));
    const char* name = "pts";
    int W = 0, H = 0;
    if (c.layout == Layout::Lattice) {
        // One placeholder point fixes the render target and the fit to the lattice's box.
        float one[3] = {0, 0, 0};
        const float box[8] = {0, 1, -1, 1, -1, 1, 0, 0};
        CaliperTensor t = tensor(one, {1, 3}, CALIPER_DEV_CPU);
        CaliperLiveSnapshot s{};
        s.struct_size = sizeof s;
        s.name = name;
        s.kind = CALIPER_LIVE_KIND_POINTS;
        s.data = &t;
        s.points_dims = 2;
        s.stats = box;
        s.stats_count = 8;
        REQUIRE_MESSAGE(caliper_core_live_apply(core.core, &s) == 1, caliper_core_last_error(core.core));
        core.frames(2);
        CaliperLiveViewInfo vi{};
        vi.struct_size = sizeof vi;
        REQUIRE(caliper_core_live_view_info(core.core, name, &vi) == 1);
        W = vi.width;
        H = vi.height;
        REQUIRE((W > 0 && H > 0));
    }
    Payload p = make_payload(c, W, H);
    id<MTLBuffer> pbuf = nil, abuf = nil;
    CaliperTensor tp{}, ta{};
    CaliperLiveSnapshot s{};
    s.struct_size = sizeof s;
    s.name = name;
    s.kind = CALIPER_LIVE_KIND_POINTS;
    s.points_dims = c.dims;
    s.seq = 7;
    if (c.route == Route::Cpu) {
        tp = tensor(p.pos.data(), {(int64_t)c.n, 3}, CALIPER_DEV_CPU);
        if (!p.aux.empty()) ta = tensor(p.aux.data(), {(int64_t)c.n}, CALIPER_DEV_CPU);
    } else {
        const bool priv = c.route == Route::Private;
        pbuf = metal_buffer(dev, p.pos, 64, priv);
        tp = tensor((__bridge void*)pbuf, {(int64_t)c.n, 3}, CALIPER_DEV_METAL);
        s.data_offset = 64;
        if (!p.aux.empty()) {
            abuf = metal_buffer(dev, p.aux, 128, priv);
            ta = tensor((__bridge void*)abuf, {(int64_t)c.n}, CALIPER_DEV_METAL);
            s.aux_offset = 128;
        }
    }
    s.data = &tp;
    if (!p.aux.empty()) s.aux = &ta;
    s.colormap = c.colormap;
    s.vmin = c.vmin;
    s.vmax = c.vmax;
    if (c.colour == Colour::Values && c.auto_range) s.flags = CALIPER_LIVE_APPLY_AUTO_VMIN | CALIPER_LIVE_APPLY_AUTO_VMAX;
    if (c.colour == Colour::Classes) s.label_count = c.classes;
    s.stats = p.stats_count ? p.stats : nullptr;
    s.stats_count = p.stats_count;
    REQUIRE_MESSAGE(caliper_core_live_apply(core.core, &s) == 1, caliper_core_last_error(core.core));
    if (c.route == Route::Cpu) {   // the view copied the bytes: scribbling ours changes nothing drawn
        std::vector<float> keep = p.pos;
        std::fill(p.pos.begin(), p.pos.end(), 1e9f);
        p.pos.swap(keep);
    }
    if (c.selection) {
        std::vector<int64_t> items;
        for (uint64_t i = 0; i < c.n; i += 3) items.push_back((int64_t)i);
        REQUIRE(caliper_live_views::set_link(core.core, name, "points", items));
    }
    core.frames(3);
    CaliperLiveViewInfo vi{};
    vi.struct_size = sizeof vi;
    REQUIRE(caliper_core_live_view_info(core.core, name, &vi) == 1);
    out->width = vi.width;
    out->height = vi.height;
    out->rgba.assign((size_t)vi.width * vi.height * 4, 0);
    REQUIRE_MESSAGE(caliper_core_live_view_readback(core.core, name, out->rgba.data(), out->rgba.size()) == 1,
                    caliper_core_last_error(core.core));
    caliper_stats::ViewCopies vc{};
    caliper_stats::view_copies(core.core, name, &vc);
    caliper_live_views::View* v = caliper_live_views::find(core.core, name);
    REQUIRE(v != nullptr);
    ref::Line& l = out->line;
    l.name = c.name;
    l.width = vi.width;
    l.height = vi.height;
    l.hash = ref::hex16(ref::fnv1a64(out->rgba));
    l.fields = {{"path", vi.path ? vi.path : ""},
                {"imported", std::to_string(vi.imported)},
                {"copies", std::to_string(vc.last.copies)},
                {"bytes", std::to_string(vc.last.bytes)},
                {"count", std::to_string(vi.count)}};
    l.info = caliper_live_views::info_text(core.core, *v);
    out->payload = std::move(p);
    return true;
}

// rgba blended toward `ground` as the points view dims the unselected points of a selection.
uint32_t dimmed(uint32_t c, uint32_t ground) {
    uint32_t out = 0xff000000u;
    for (int k = 0; k < 3; ++k) {
        const float a = (float)((c >> (8 * k)) & 0xff), b = (float)((ground >> (8 * k)) & 0xff);
        out |= (uint32_t)std::lround(a * 0.28f + b * 0.72f) << (8 * k);
    }
    return out;
}

uint32_t level(float v, float vmin, float vmax) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return caliper_gfx::cmap::reference_index(bits, vmin, vmax);
}

// The reference colour of lattice point i, before dimming.
uint32_t reference_colour(const Case& c, const Payload& p, uint64_t i) {
    if (c.colour == Colour::Values) {
        const float vmin = c.auto_range ? p.stats[0] : c.vmin, vmax = c.auto_range ? p.stats[1] : c.vmax;
        return caliper_host::colormap_lut(c.colormap)[level(p.aux[i], vmin, vmax)];
    }
    if (c.colour == Colour::Classes) return caliper_live_views::kPalette[level(p.aux[i], 0.0f, 255.0f) % 16];
    return kFlat;
}

// A lattice's centre pixels against the CPU reference colours. Returns the mismatches, and in
// *points the indices of the points whose centre pixel differs.
size_t check_lattice_colours(const Case& c, const Result& r, std::string* first, std::vector<int64_t>* points = nullptr) {
    const Payload& p = r.payload;
    const uint32_t ground = pixel(r, 0, 0);
    // Every lattice site's last point is what shows there.
    std::vector<int64_t> last((size_t)r.width * r.height, -1);
    for (uint64_t i = 0; i < c.n; ++i) {
        const int x = (int)std::floor(p.px[2 * i]), y = (int)std::floor(p.px[2 * i + 1]);
        if (x >= 0 && y >= 0 && x < r.width && y < r.height) last[(size_t)y * r.width + x] = (int64_t)i;
    }
    size_t bad = 0;
    for (int y = 0; y < r.height; ++y)
        for (int x = 0; x < r.width; ++x) {
            const int64_t i = last[(size_t)y * r.width + x];
            if (i < 0) continue;
            uint32_t want = reference_colour(c, p, (uint64_t)i);
            if (c.selection && i % 3 != 0) want = dimmed(want, ground);
            const uint32_t got = pixel(r, x, y);
            if (got != want) {
                if (!bad && first) {
                    char t[160];
                    std::snprintf(t, sizeof t, "point %lld at (%d, %d): %08x, the reference %08x", (long long)i, x, y,
                                  got, want);
                    *first = t;
                }
                if (points) points->push_back(i);
                ++bad;
            }
        }
    return bad;
}

void capture(const Case& c, const Result& r) {
    const char* dir = ref::env("CALIPER_GFX_CAPTURE_DIR");
    if (!dir) return;
    const std::string path = std::string(dir) + "/points-view-" + c.name + ".png";
    CHECK(stbi_write_png(path.c_str(), r.width, r.height, 4, r.rgba.data(), r.width * 4) != 0);
}

std::string references_file() { return std::string(CALIPER_TEST_SOURCE_ROOT) + "/tests/gfx/references/points_view.txt"; }

// The bound on the pixels in which a near-tie case may differ from the old path, from the counts
// measured over seven clouds each (0 to 3 at 1,000,000 points, 0 to 6 at 4,000,000) with about
// two and a half times the largest as headroom.
size_t near_tie_bound(uint64_t n) { return n >= 4000000 ? 16 : 8; }

}  // namespace

TEST_CASE("gfx/Metal points view: every case draws the pixels, path label, copies and header of its reference") {
    {
        Core probe;
        if (!probe.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU");
    }
    ref::Set refs(references_file());
    std::string hash_of_classes;
    for (const Case& c : cases()) {
        if (!selected_case(c)) continue;
        CAPTURE(c.name);
        Result r;
        REQUIRE(draw_case(c, &r));
        capture(c, r);
        std::vector<int64_t> differs;
        if (c.layout == Layout::Lattice) {
            std::string first;
            const size_t bad = check_lattice_colours(c, r, &first, &differs);
            const ref::Line* want = refs.recording() ? nullptr : refs.find(c.name);
            if (refs.recording()) {
                // The old path is recorded as it draws. Points whose colour differs from the CPU
                // reference are noted in the line: there the reference decides, not the record.
                if (bad) MESSAGE(c.name, ": the old path differs from the CPU reference colour at ", bad, " points; ", first);
            } else if (want && r.line.hash == want->hash && bad) {
                // Still the recorded path itself (the step's first commit): its noted points may
                // keep their colour; any other point must have the reference's.
                const std::string listed = "," + want->field("cpu_ref_differs") + ",";
                size_t unlisted = 0;
                for (int64_t i : differs) unlisted += listed.find("," + std::to_string(i) + ",") == std::string::npos;
                CHECK_MESSAGE(unlisted == 0, c.name, ": ", unlisted, " centre pixels differ from the CPU reference colour "
                              "at points the record does not note; ", first);
            } else {
                CHECK_MESSAGE(bad == 0, c.name, ": ", bad, " centre pixels differ from the CPU reference colour; ", first);
            }
        }
        std::string differs_text;
        for (int64_t i : differs) differs_text += (differs_text.empty() ? "" : ",") + std::to_string(i);
        r.line.fields.emplace_back("cpu_ref_differs", differs_text.empty() ? "none" : differs_text);
        if (c.name == "3d-classes-1000") hash_of_classes = r.line.hash;
        if (c.name == "3d-classes-1000-cpu" && !hash_of_classes.empty())
            CHECK_MESSAGE(r.line.hash == hash_of_classes, "a CPU payload draws what the same MTLBuffer payload draws");
        if (refs.recording()) {
            refs.record(r.line);
            MESSAGE("recorded ", r.line.text());
            continue;
        }
        const ref::Line* want = refs.find(c.name);
        REQUIRE_MESSAGE(want != nullptr, "no reference for ", c.name, " in ", refs.path(),
                        " (record one with CALIPER_GFX_RECORD at the step's first commit)");
        CHECK(r.line.width == want->width);
        CHECK(r.line.height == want->height);
        for (const char* key : {"path", "imported", "copies", "bytes", "count"})
            CHECK_MESSAGE(r.line.field(key) == want->field(key), c.name, ": ", key, " is ", r.line.field(key),
                          ", the reference ", want->field(key));
        CHECK_MESSAGE(r.line.info == want->info, c.name, ": the header reads '", r.line.info, "', the reference '",
                      want->info, "'");
        const std::string old_differs = want->field("cpu_ref_differs");
        if (c.near_ties) {
            MESSAGE(c.name, ": readback ", (r.line.hash == want->hash ? "equals" : "differs from"),
                    " the reference (compared pixel by pixel with the old path's images in the next row)");
        } else if (!old_differs.empty() && old_differs != "none") {
            // The record is the old path's, which drew points ", old_differs, " in another colour than
            // the CPU reference; the check above holds this readback to the reference there, and the
            // next row holds every other pixel to the old path's image.
            MESSAGE(c.name, ": readback ", (r.line.hash == want->hash ? "equals" : "differs from"),
                    " the old path's record, whose points ", old_differs,
                    " differ from the CPU reference (the reference decides them; the next row compares the rest)");
        } else {
            CHECK_MESSAGE(r.line.hash == want->hash, c.name, ": the readback's hash ", r.line.hash,
                          " differs from the reference ", want->hash);
        }
    }
}

// The old path's images, captured by a build of the step's first commit, against the view now:
// every case but the near-tie clouds byte-identical, except inside the squares of the points that
// the old path drew in another colour than the CPU reference (cpu_ref_differs in the record), where
// the view must draw the reference colour; a near-tie cloud within its bound, with every differing
// pixel an edge of a point that its counterpart image has one pixel over, so that no point is
// missing or added.
TEST_CASE("gfx/Metal points view: each case against the old path's images") {
    const char* dir = ref::env("CALIPER_GFX_REFERENCE_IMAGES");
    if (!dir) {
        MESSAGE("CALIPER_GFX_REFERENCE_IMAGES is not set: no images of the old path to compare with");
        return;
    }
    {
        Core probe;
        if (!probe.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU");
    }
    ref::Set refs(references_file());
    for (const Case& c : cases()) {
        if (!selected_case(c)) continue;
        CAPTURE(c.name);
        // The points the old path drew in another colour than the CPU reference.
        std::vector<int64_t> decided;
        if (const ref::Line* want = refs.find(c.name)) {
            const std::string list = want->field("cpu_ref_differs");
            for (size_t a = 0; list != "none" && a < list.size();) {
                const size_t b = list.find(',', a);
                decided.push_back(std::atoll(list.substr(a, b - a).c_str()));
                a = b == std::string::npos ? list.size() : b + 1;
            }
        }
        const std::string path = std::string(dir) + "/points-view-" + c.name + ".png";
        int w = 0, h = 0, comp = 0;
        uint8_t* img = stbi_load(path.c_str(), &w, &h, &comp, 4);
        REQUIRE_MESSAGE(img != nullptr, "no image ", path);
        std::vector<uint8_t> old(img, img + (size_t)w * h * 4);
        stbi_image_free(img);
        Result r;
        REQUIRE(draw_case(c, &r));
        capture(c, r);
        REQUIRE(w == r.width);
        REQUIRE(h == r.height);
        auto at = [&](const std::vector<uint8_t>& px, int x, int y) {
            uint32_t p;
            std::memcpy(&p, &px[((size_t)y * w + x) * 4], 4);
            return p;
        };
        auto beside = [&](const std::vector<uint8_t>& px, int x, int y, uint32_t colour) {
            const int d[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (const auto& k : d) {
                const int xx = x + k[0], yy = y + k[1];
                if (xx >= 0 && yy >= 0 && xx < w && yy < h && at(px, xx, yy) == colour) return true;
            }
            return false;
        };
        // Whether (x, y) lies in the square of a decided point and the view draws its reference colour.
        const float half = 0.5f * caliper_live_views::point_size_px(c.n, 1.0f);
        const uint32_t ground = at(old, 0, 0);
        auto decided_by_reference = [&](int x, int y, uint32_t got) {
            for (int64_t i : decided) {
                const float px = r.payload.px[2 * (size_t)i], py = r.payload.px[2 * (size_t)i + 1];
                if (std::fabs(x + 0.5f - px) > half || std::fabs(y + 0.5f - py) > half) continue;
                uint32_t want = reference_colour(c, r.payload, (uint64_t)i);
                if (c.selection && i % 3 != 0) want = dimmed(want, ground);
                if (got == want) return true;
            }
            return false;
        };
        size_t differ = 0, unexplained = 0, by_reference = 0;
        std::string where;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const uint32_t a = at(old, x, y), b = at(r.rgba, x, y);
                if (a == b) continue;
                if (!decided.empty() && decided_by_reference(x, y, b)) {
                    ++by_reference;
                    continue;
                }
                ++differ;
                const bool edge = beside(r.rgba, x, y, a) || beside(old, x, y, b);
                if (!edge) ++unexplained;
                if (differ <= 32) {
                    char t[96];
                    std::snprintf(t, sizeof t, " (%d,%d) old %08x new %08x%s", x, y, a, b, edge ? "" : " unexplained");
                    where += t;
                }
            }
        MESSAGE(c.name, ": ", differ, " of ", (size_t)w * h, " pixels differ from the old path", where,
                (by_reference ? "; " + std::to_string(by_reference) +
                                    " more lie in the squares of points the old path coloured unlike the CPU reference, "
                                    "and draw the reference colour"
                              : std::string()));
        if (c.near_ties) {
            CHECK_MESSAGE(differ <= near_tie_bound(c.n), c.name, ": ", differ, " pixels differ, above the bound ",
                          near_tie_bound(c.n));
            CHECK_MESSAGE(unexplained == 0, c.name, ": ", unexplained, " differing pixels are not the edge of a point");
        } else {
            CHECK_MESSAGE(differ == 0, c.name, ": ", differ, " pixels differ from the old path");
        }
    }
}

// The cost of a redraw by count: the view's CPU time to encode and submit its draw, and the GPU
// time of the draw (CALIPER_METAL_STATS), with the cloud reapplied every frame so that every frame
// redraws, as a training run's points do.
TEST_CASE("gfx/Metal points view: timing — a redraw by count") {
    if (!ref::env("CALIPER_GFX_POINTS_BENCH")) return;
    setenv("CALIPER_METAL_STATS", "1", 1);
    for (const uint64_t n : {100000ull, 1000000ull, 4000000ull}) {
        Core core;
        if (!core.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU");
        id<MTLDevice> dev = (__bridge id<MTLDevice>)core.core->renderer->native_device();
        Case c;
        c.n = n;
        c.seed = 21;
        c.colour = Colour::Values;
        c.auto_range = true;
        Payload p = make_payload(c, 0, 0);
        id<MTLBuffer> pbuf = metal_buffer(dev, p.pos, 0, false), abuf = metal_buffer(dev, p.aux, 0, false);
        CaliperTensor tp = tensor((__bridge void*)pbuf, {(int64_t)n, 3}, CALIPER_DEV_METAL);
        CaliperTensor ta = tensor((__bridge void*)abuf, {(int64_t)n}, CALIPER_DEV_METAL);
        CaliperLiveSnapshot s{};
        s.struct_size = sizeof s;
        s.name = "bench";
        s.kind = CALIPER_LIVE_KIND_POINTS;
        s.data = &tp;
        s.aux = &ta;
        s.points_dims = 3;
        s.stats = p.stats;
        s.stats_count = 8;
        s.flags = CALIPER_LIVE_APPLY_AUTO_VMIN | CALIPER_LIVE_APPLY_AUTO_VMAX;
        std::vector<double> cpu;
        caliper_host::HostRenderer::FrameStats fs{};
        for (int f = 0; f < 140; ++f) {
            s.seq = (uint64_t)f + 1;
            REQUIRE(caliper_core_live_apply(core.core, &s) == 1);
            core.frames(1);
            if (f == 19) caliper_stats::frame_stats(core.core, &fs, true);   // 20 frames of warm-up
            if (f >= 20) cpu.push_back(caliper_live_views::find(core.core, "bench")->draw_cpu_ms);
        }
        REQUIRE(caliper_stats::frame_stats(core.core, &fs, true));
        std::sort(cpu.begin(), cpu.end());
        const auto q = [&](double f) { return cpu[(size_t)(f * (double)(cpu.size() - 1))]; };
        double load[3] = {0, 0, 0};
        getloadavg(load, 3);
        std::printf("[points-view-bench] %llu points: draw cpu ms p50 %.4f p99 %.4f; draw gpu ms p50 %.4f p99 %.4f (%llu); "
                    "frame ms p50 %.2f p99 %.2f; load %.2f\n",
                    (unsigned long long)n, q(0.5), q(0.99), fs.draw_gpu_ms_p50, fs.draw_gpu_ms_p99,
                    (unsigned long long)fs.draw_gpu_n, fs.frame_ms_p50, fs.frame_ms_p99, load[0]);
    }
}

CALIPER_GPU_TEST_MAIN()
