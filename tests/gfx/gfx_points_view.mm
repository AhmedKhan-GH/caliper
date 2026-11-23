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
// match it byte for byte. The view now draws every point as an exact square, two triangles, on Metal
// as it already did in the browser, where the old Metal path drew hardware point sprites. With the
// depth cues of 3-D points a point's size is fractional, and a sprite's edge can then sit up to about
// 1/32 of a pixel from the exact square's (the sprite row below measures it), so a 3-D case may
// differ from the old path's images in pixels at a point's edge: the row against those images
// bounds every such pixel, and the view must equal the CPU reference rasterisation of its draws byte
// for byte in every case.
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
#include "gfx/metal_backend.h"
#include "gfx/recording_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "gfx/shaders/generated/uniforms.h"
#include "live/live_points.h"
#include "live/live_views_state.h"
#include "renderer/host_renderer.h"
#include "tensor_bridge.h"          // colormap_lut

#include <imgui.h>                  // the key of a modifier in an input event

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <stb_image.h>
#include <stb_image_write.h>
#include "../match_reference.h"
#pragma clang diagnostic pop

#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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
    out.push_back(values(random("3d-viridis-auto-1000000", 3, 1000000, 12), 0, true, 0, 0, false));
    out.push_back(values(random("3d-viridis-auto-4000000", 3, 4000000, 13), 0, true, 0, 0, false));
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
    bool drew_pieces = false;
    // What the gestures after the readback gave (draw_case's `after`): the events of the view's sink,
    // the hovered point, the selection's count, and the picture with the selection.
    std::vector<std::string> events;
    int64_t hover = -1;
    uint64_t sel_count = 0;
    std::vector<uint8_t> after_rgba;
};

uint32_t pixel(const Result& r, int x, int y) {
    uint32_t p;
    std::memcpy(&p, &r.rgba[((size_t)y * r.width + x) * 4], 4);
    return p;
}

// The draw of the points view's scene: the rows of the reference draw hold it there, and the
// comparison rows draw each case both ways.


// Applies the case to a fresh core and reads its view back, drawn by `draw`.
struct Core;
using After = std::function<void(Core& core, const char* name, Result* out)>;
bool draw_case(const Case& c, Result* out, const After& after = {}) {
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
    // A payload that the CPU cannot read reaches points@1 after its readback: frames until it drew.
    if (caliper_live_views::View* kv = caliper_live_views::find(core.core, name); kv && kv->points && kv->points->scene)
        for (int f = 0; f < 600 && kv->points->scene->pending(); ++f) core.frames(1);
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
    if (v->points && v->points->scene) {
        const caliper_live_views::PointsScene& sc = *v->points->scene;
        out->drew_pieces = sc.drew_pieces();
        out->error = sc.error() + (sc.error_detail().empty() ? "" : " (" + sc.error_detail() + ")");
    }
    out->payload = std::move(p);
    if (after) after(core, name, out);
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

// ---- The CPU reference rasterisation of the scene core's points draws ----------------------------
//
// Every draw of the last frame is repeated on the CPU from the bytes the GPU read: the frame block,
// the draw's parameter block and its colormap block from the draw list's arena, and the payload's
// rows. vs_points in f32 (the scene programs compile without fast math): clip = mvp * (x, y, z, 1)
// summed column by column, each corner c.xy + corner * (size_px / viewport) * c.w, then the
// target's NDC scale and offset; the perspective divide and the viewport transform; the corners
// snapped to 1/256 of a pixel with a half rounded up, which is the rasteriser's fixed point (found
// against these rows' readbacks: 4 or 16 bits, or a half rounded to even or down, leave edge rows
// and columns apart); coverage of pixel centres inside each quad by the top-left rule; depth z / w
// tested less-or-equal and written, in instance order and draw order; the colour by the
// colormap's keys (colormap_keys.h) or the flat colour; NaN positions hidden. With the 3-D view's
// depth cues (att_ref, fade_w0 and fade_w1 in the parameter block) the size is size_px *
// clamp(att_ref / max(w, 1e-6), 0.5, 2) and each channel c / 255 is scaled by
// 1 - 0.5 * clamp((w - fade_w0) / (fade_w1 - fade_w0), 0, 1) and written to the target rounded to
// the nearest of its 255 steps, a half to even.
}  // namespace


// The view's readback against the CPU reference rasterisation of the draws the scene core compiled
// (above), byte for byte, for every case: the scene core's camera composed in double and rounded to
// f32 once is what both the GPU and the reference read, so they agree to the bit wherever the
// reference repeats the GPU's arithmetic.


// ---- points@1 against the reference draw ----------------------------------------------------------
//
// The view draws its points with the library piece points@1 by default (live_points.h). Each case is
// drawn both ways and held to the tolerances of the match tests (tests/points_match/): a pixel whose
// 3 x 3 neighbourhood in the reference picture holds one colour, inside a point or on the ground, has
// that colour in the piece's picture, and any other pixel, at a point's edge, has a colour within 1
// level of a reference colour of its neighbourhood. The lattices' centre pixels also keep the CPU
// reference colours exactly.
struct Match {
    size_t interior = 0, edge = 0, differing = 0, failed = 0;
    int largest = 0;
    std::string first;
};

int channel_gap(uint32_t a, uint32_t b) {
    int d = 0;
    for (int k = 0; k < 4; ++k) d = std::max(d, std::abs(int((a >> (8 * k)) & 0xffu) - int((b >> (8 * k)) & 0xffu)));
    return d;
}

Match match_pictures(int W, int H, const std::vector<uint8_t>& reference, const std::vector<uint8_t>& piece) {
    auto at = [&](const std::vector<uint8_t>& px, int x, int y) {
        uint32_t v;
        std::memcpy(&v, &px[((size_t)std::clamp(y, 0, H - 1) * W + std::clamp(x, 0, W - 1)) * 4], 4);
        return v;
    };
    Match m;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const uint32_t b = at(reference, x, y), p = at(piece, x, y);
            m.differing += b != p;
            m.largest = std::max(m.largest, channel_gap(b, p));
            bool uniform = true;
            for (int dy = -1; dy <= 1 && uniform; ++dy)
                for (int dx = -1; dx <= 1 && uniform; ++dx) uniform = at(reference, x + dx, y + dy) == b;
            bool ok = false;
            if (uniform) {
                ++m.interior;
                ok = p == b;
            } else {
                ++m.edge;
                for (int dy = -1; dy <= 1 && !ok; ++dy)
                    for (int dx = -1; dx <= 1 && !ok; ++dx) ok = channel_gap(p, at(reference, x + dx, y + dy)) <= 1;
            }
            if (!ok && m.failed++ == 0) {
                char t[160];
                std::snprintf(t, sizeof t, "; first at (%d, %d): piece %08x, reference %08x, %s", x, y, p, b,
                              uniform ? "interior" : "edge");
                m.first = t;
            }
        }
    return m;
}

void capture_named(const std::string& name, int W, int H, const std::vector<uint8_t>& rgba) {
    const char* dir = ref::env("CALIPER_GFX_CAPTURE_DIR");
    if (!dir) return;
    const std::string path = std::string(dir) + "/points-view-" + name + ".png";
    CHECK(stbi_write_png(path.c_str(), W, H, 4, rgba.data(), W * 4) != 0);
}

TEST_CASE("gfx/Metal points view: points@1 matches the saved scene references") {
    {
        Core probe;
        if (!probe.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU");
    }
    for (const Case& c : cases()) {
        if (!selected_case(c)) continue;
        CAPTURE(c.name);
        Result b, p;
        REQUIRE(draw_case(c, &p));
        b.width = p.width;
        b.height = p.height;
        b.rgba = match_reference::picture("compare", uint32_t(b.width), uint32_t(b.height), nullptr);
        CHECK_MESSAGE(p.drew_pieces, c.name, ": the reference drew the case of points@1: ", p.error);
        REQUIRE(b.width == p.width);
        REQUIRE(b.height == p.height);
        const Match m = match_pictures(b.width, b.height, b.rgba, p.rgba);
        MESSAGE(c.name, ": points@1 ", m.differing, " of ", (size_t)b.width * b.height, " pixels differ from the reference (largest ",
                m.largest, " levels); ", m.interior, " interior and ", m.edge, " edge pixels, ", m.failed,
                " outside the tolerances", m.first);
        CHECK_MESSAGE(m.failed == 0, c.name, ": ", m.failed, " pixels outside the tolerances", m.first);
        if (c.layout == Layout::Lattice) {
            std::string first;
            const size_t bad = check_lattice_colours(c, p, &first);
            CHECK_MESSAGE(bad == 0, c.name, ": points@1 ", bad, " centre pixels differ from the CPU reference colour; ", first);
        }
        capture_named(c.name + "-piece", p.width, p.height, p.rgba);
        capture_named(c.name + "-reference", b.width, b.height, b.rgba);
    }
}

// The picks, hovers and selections of the view come from the CPU payload under the camera, as the
// browser element's do, and the draw changes none of them: the same gestures on a case drawn by
// points@1 and by the reference give the same events, the same hovered point and the same selection
// count, and the pictures with the selection drawn match within the tolerances above. The gestures:
// a hover and a click at the point nearest the plot's centre, a shift-alt drag of a rectangle and a
// shift drag of a lasso around the centre.
void gestures(Core& core, const char* name, Result* out) {
    caliper_live_views::set_event_sink(core.core, [out](const caliper_live_views::LiveEvent& e) {
        if (std::string(e.type) != "hover") out->events.push_back(std::string(e.type) + " " + e.detail);
    });
    caliper_live_views::View* v = caliper_live_views::find(core.core, name);
    REQUIRE(v != nullptr);
    core.frames(2);
    REQUIRE(v->geo_valid);
    const float* a = v->geo_area;   // x, y, width and height of the plot
    const float cx = a[0] + a[2] * 0.5f, cy = a[1] + a[3] * 0.5f;
    float M[16];
    REQUIRE(caliper_live_views::points_mvp(*v, M));
    // The point nearest the centre, among those projected inside the plot.
    float px = cx, py = cy, best = INFINITY;
    const std::vector<float>& pos = out->payload.pos;
    for (size_t i = 0; i < out->payload.pos.size() / 3 && i < 200000; ++i) {
        const float x = pos[3 * i], y = pos[3 * i + 1], z = v->dims == 3 ? pos[3 * i + 2] : 0.0f;
        const float w = M[3] * x + M[7] * y + M[11] * z + M[15];
        if (!(w > 1e-6f)) continue;
        const float sx = a[0] + ((M[0] * x + M[4] * y + M[8] * z + M[12]) / w * 0.5f + 0.5f) * a[2];
        const float sy = a[1] + (0.5f - (M[1] * x + M[5] * y + M[9] * z + M[13]) / w * 0.5f) * a[3];
        const float d = (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy);
        if (d < best) {
            best = d;
            px = sx;
            py = sy;
        }
    }
    auto mouse = [&](CaliperEventType type, float x, float y, int down) {
        CaliperInputEvent e{};
        e.struct_size = sizeof e;
        e.type = type;
        e.x = x;
        e.y = y;
        e.button = 0;
        e.down = down;
        caliper_core_event(core.core, &e);
    };
    auto keys = [&](int mods, bool down) {
        CaliperInputEvent e{};
        e.struct_size = sizeof e;
        e.type = CALIPER_EVENT_KEY;
        e.key = ImGuiKey_LeftShift;
        e.down = down ? 1 : 0;
        e.mods = mods;
        caliper_core_event(core.core, &e);
    };
    auto drag = [&](const std::vector<std::pair<float, float>>& path, int mods) {
        keys(mods, true);
        mouse(CALIPER_EVENT_MOUSE_MOVE, path[0].first, path[0].second, 0);
        core.frames(1);
        mouse(CALIPER_EVENT_MOUSE_BUTTON, path[0].first, path[0].second, 1);
        core.frames(1);
        for (size_t k = 1; k < path.size(); ++k) {
            mouse(CALIPER_EVENT_MOUSE_MOVE, path[k].first, path[k].second, 0);
            core.frames(1);
        }
        mouse(CALIPER_EVENT_MOUSE_BUTTON, path.back().first, path.back().second, 0);
        core.frames(2);
        keys(0, false);
        core.frames(1);
    };
    // Hover, then a click.
    mouse(CALIPER_EVENT_MOUSE_MOVE, px, py, 0);
    core.frames(3);
    out->hover = v->hover_item;
    mouse(CALIPER_EVENT_MOUSE_BUTTON, px, py, 1);
    core.frames(1);
    mouse(CALIPER_EVENT_MOUSE_BUTTON, px, py, 0);
    core.frames(2);
    // A rectangle (shift and alt), then a lasso (shift).
    const float rw = a[2] * 0.15f, rh = a[3] * 0.15f;
    drag({{cx - rw, cy - rh}, {cx, cy}, {cx + rw, cy + rh}}, CALIPER_MOD_SHIFT | CALIPER_MOD_ALT);
    drag({{cx - rw, cy}, {cx, cy - rh}, {cx + rw * 1.5f, cy}, {cx, cy + rh * 0.8f}, {cx - rw, cy}}, CALIPER_MOD_SHIFT);
    out->sel_count = v->sel_count;
    // The picture with the selection drawn.
    core.frames(2);
    if (caliper_live_views::View* kv = caliper_live_views::find(core.core, name); kv && kv->points && kv->points->scene)
        for (int f = 0; f < 600 && kv->points->scene->pending(); ++f) core.frames(1);
    out->after_rgba.assign((size_t)out->width * out->height * 4, 0);
    REQUIRE(caliper_core_live_view_readback(core.core, name, out->after_rgba.data(), out->after_rgba.size()) == 1);
    caliper_live_views::set_event_sink(core.core, {});
}

TEST_CASE("gfx/Metal points view: picks, hovers and selections are the same for CPU and device payloads") {
    {
        Core probe;
        if (!probe.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU");
    }
    for (const Case& c : cases()) {
        if (!selected_case(c)) continue;
        // A point to pick and a selection to draw, at a test's cost; a payload that the CPU cannot read
        // has no hover, picks or selection in either draw.
        if (c.n < 400 || c.n > 100000 || c.route == Route::Private) continue;
        CAPTURE(c.name);
        Result b, p;
        Case cpu = c;
        cpu.route = Route::Cpu;
        REQUIRE(draw_case(cpu, &b, gestures));
        REQUIRE(draw_case(c, &p, gestures));
        CHECK(p.drew_pieces);
        CHECK_MESSAGE(b.hover == p.hover, c.name, ": hovered ", b.hover, " with the reference and ", p.hover, " with points@1");
        CHECK_MESSAGE(b.sel_count == p.sel_count, c.name, ": selected ", b.sel_count, " and ", p.sel_count);
        CHECK_MESSAGE(b.events == p.events, c.name, ": the events differ");
        size_t picks = 0, selects = 0;
        for (const std::string& e : p.events) {
            picks += e.rfind("pick ", 0) == 0;
            selects += e.rfind("select ", 0) == 0;
        }
        CHECK_MESSAGE(picks >= 1, c.name, ": the click picked nothing");
        CHECK_MESSAGE(selects >= 2, c.name, ": the drags selected nothing");
        const Match m = match_pictures(b.width, b.height, b.after_rgba, p.after_rgba);
        MESSAGE(c.name, ": hover ", p.hover, ", ", p.events.size(), " events (", picks, " picks, ", selects,
                " selections), ", p.sel_count, " selected; with the selection drawn ", m.differing,
                " pixels differ (largest ", m.largest, " levels), ", m.failed, " outside the tolerances", m.first);
        CHECK_MESSAGE(m.failed == 0, c.name, ": with the selection drawn ", m.failed, " pixels outside the tolerances", m.first);
        capture_named(c.name + "-selection-piece", p.width, p.height, p.after_rgba);
    }
}

// The old path's images, captured by a build of the step's first commit on the merged base, against
// the view now. Every 2-D case and the single 3-D point are byte-identical. In a 3-D cloud the old
// path drew hardware point sprites of the depth cues' fractional sizes, whose edges can sit up to
// kSpriteSlack from the exact square's: every pixel that differs there must lie within kSpriteSlack
// of an edge of one of the CPU reference's squares drawn at or in front of it, and the row fails on
// any pixel outside. The same cases rasterised by the CPU reference with the old view's float camera
// differ from the old images in exactly the same pixels, so the camera's arithmetic is not the cause.


// How Metal rasterises a point sprite: one sprite of size s at a known sub-pixel centre, read back
// along its centre row and column, against the exact square [c - s/2, c + s/2] with the centre
// snapped to 1/256 of a pixel and a pixel covered when its centre lies inside (top-left rule). For
// every whole and half size the sprite is the exact square; for a fractional size a pixel may differ
// from it only within kSpriteSlack of the square's edge. This is the bound of the row above: the old
// path drew sprites, and the depth cues of 3-D points made their sizes fractional.
constexpr float kSpriteSlack = 1.0f / 32.0f + 1.0f / 256.0f;

TEST_CASE("gfx/Metal point sprites: whole and half sizes are exact squares, other sizes within the bound") {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev == nil || caliper_test::force_no_gpu()) CALIPER_SKIP_NO_GPU("no Metal device");
    static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
struct P { float4 pos [[position]]; float size [[point_size]]; };
vertex P vs(uint vid [[vertex_id]], constant float4* a [[buffer(0)]]) {
    P o; o.pos = float4(a[vid].x, a[vid].y, 0.5, 1.0); o.size = a[vid].z; return o;
}
fragment float4 fs(P in [[stage_in]]) { return float4(1.0); }
)";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:nil error:&err];
    REQUIRE_MESSAGE(bool(lib != nil), (err ? err.localizedDescription.UTF8String : "no library"));
    MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
    id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
    REQUIRE(bool(ps != nil));
    // One sprite per 32 x 32 cell of a 1,024 x 1,024 target, 1,024 at a time.
    constexpr int kSide = 1024, kCell = 32, kPer = (kSide / kCell) * (kSide / kCell);
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:kSide height:kSide mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [dev newTextureWithDescriptor:td];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    // The sizes: every whole and half size from 1.5, the smallest the points view draws (3 pixels at
    // the depth cues' least scale), to 14, then 3 to 14 in steps of 1/256; at eight sub-pixel phases
    // of the centre on each axis.
    struct Sprite { float cx, cy, s; };
    std::vector<Sprite> all;
    const float phases[] = {0.0f, 0.0039f, 0.123f, 0.31f, 0.5f, 0.7731f, 0.9f, 0.4805f};
    for (float s = 1.5f; s <= 14.0f; s += 0.5f)
        for (int k = 0; k < 8; ++k) all.push_back({phases[k], phases[(k + 3) % 8], s});
    for (int i = 0; i <= 11 * 256; ++i)
        for (int k = 0; k < 8; ++k) all.push_back({phases[k], phases[(k + 5) % 8], 3.0f + (float)i / 256.0f});
    std::vector<uint8_t> px((size_t)kSide * kSide * 4);
    size_t whole_differ = 0, frac_cases = 0, frac_differ = 0, beyond = 0;
    std::string first, whole_first;
    const float e = kSpriteSlack;
    for (size_t base = 0; base < all.size(); base += kPer) {
        const size_t n = std::min<size_t>(kPer, all.size() - base);
        std::vector<float> verts(n * 4);
        std::vector<float> centre(n * 2);
        for (size_t k = 0; k < n; ++k) {
            const Sprite& sp = all[base + k];
            const float x = (float)((k % (kSide / kCell)) * kCell + kCell / 2) + sp.cx;
            const float y = (float)((k / (kSide / kCell)) * kCell + kCell / 2) + sp.cy;
            verts[4 * k] = x / kSide * 2.0f - 1.0f;
            verts[4 * k + 1] = 1.0f - y / kSide * 2.0f;
            verts[4 * k + 2] = sp.s;
            // The centre the rasteriser places, from the NDC it is given, snapped to 1/256.
            centre[2 * k] = std::floor((verts[4 * k] * 0.5f + 0.5f) * kSide * 256.0f + 0.5f) / 256.0f;
            centre[2 * k + 1] = std::floor((0.5f - verts[4 * k + 1] * 0.5f) * kSide * 256.0f + 0.5f) / 256.0f;
        }
        @autoreleasepool {
            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture = tex;
            rp.colorAttachments[0].loadAction = MTLLoadActionClear;
            rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLBuffer> vb = [dev newBufferWithBytes:verts.data() length:verts.size() * 4 options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
            [enc setRenderPipelineState:ps];
            [enc setVertexBuffer:vb offset:0 atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:n];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            [tex getBytes:px.data() bytesPerRow:kSide * 4 fromRegion:MTLRegionMake2D(0, 0, kSide, kSide) mipmapLevel:0];
        }
        for (size_t k = 0; k < n; ++k) {
            const Sprite& sp = all[base + k];
            const bool whole = sp.s == std::floor(sp.s * 2.0f) / 2.0f;
            const int gx = (int)(k % (kSide / kCell)) * kCell, gy = (int)(k / (kSide / kCell)) * kCell;
            const float cx = centre[2 * k], cy = centre[2 * k + 1], h = sp.s * 0.5f;
            bool differs = false;
            // Along the centre row (x) and the centre column (y).
            for (int axis = 0; axis < 2; ++axis) {
                const float c = axis == 0 ? cx : cy;
                const int line = axis == 0 ? (int)cy : (int)cx;
                for (int t = (axis == 0 ? gx : gy); t < (axis == 0 ? gx : gy) + kCell; ++t) {
                    const size_t at = axis == 0 ? (size_t)line * kSide + t : (size_t)t * kSide + line;
                    const bool hw = px[at * 4] > 0;
                    const float centre_t = (float)t + 0.5f;
                    const bool exact = centre_t >= c - h && centre_t < c + h;
                    if (hw == exact) continue;
                    differs = true;
                    const bool inside = std::fabs(centre_t - (c - h)) <= e || std::fabs(centre_t - (c + h)) <= e;
                    if (!whole && !inside) {
                        ++beyond;
                        if (first.empty()) {
                            char m[128];
                            std::snprintf(m, sizeof m, "size %.6f centre %.5f: pixel centre %.1f, edges %.5f and %.5f",
                                          sp.s, c, centre_t, c - h, c + h);
                            first = m;
                        }
                    }
                }
            }
            if (whole && differs && whole_first.empty()) {
                char m[96];
                std::snprintf(m, sizeof m, "size %.1f centre (%.5f, %.5f)", sp.s, cx, cy);
                whole_first = m;
            }
            if (whole) whole_differ += differs;
            else {
                ++frac_cases;
                frac_differ += differs;
            }
        }
    }
    MESSAGE("point sprites: ", all.size() - frac_cases, " of whole and half sizes, ", whole_differ,
            " differ from the exact square; ", frac_cases, " of fractional sizes, ", frac_differ,
            " differ from it, and of their differing pixels ", beyond, " lie farther than ", e, " px from its edge",
            first.empty() ? std::string() : " (the first: " + first + ")");
    CHECK_MESSAGE(whole_differ == 0, whole_differ, " sprites of whole or half sizes differ from the exact square, the first ",
                  whole_first);
    CHECK_MESSAGE(beyond == 0, beyond, " pixels of fractional sprites differ farther than ", e, " px from the edge");
}

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
