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
#include "scene/compile.h"
#include "scene/uniform_check.h"
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
    // The scene core's draws of the last frame, when the view drew through it: the draw list (its
    // arena holds every block the GPU read) and each draw's entity path.
    bool has_draws = false;
    caliper_gfx::DrawList draws;
    std::vector<std::string> draw_paths;
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
    if (caliper_live_views::View* kv = caliper_live_views::find(core.core, name);
        kv && kv->points && kv->points->scene)
        kv->points->scene->keep_draws(true);
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
    if (v->points && v->points->scene) {
        const caliper_live_views::PointsScene& sc = *v->points->scene;
        out->has_draws = sc.last_draws().records().size > 0;
        out->draws = sc.last_draws();
        for (const caliper_scene::DrawMapEntry& e : sc.last_map().entries) out->draw_paths.push_back(sc.tree().path(e.entity));
    }
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
namespace sh = caliper_gfx::shaders;

// A square the CPU reference drew: its edges in screen space (y down) before the snap, and its depth.
struct Square { float x0, x1, y0, y1, dz; };

// The camera of a draw, in place of the one in its parameter block (the old view's float camera).
struct CameraOverride {
    float mvp[16];
    float att_ref = 0.0f, fade_w0 = 0.0f, fade_w1 = 0.0f;
};

struct Raster {
    int w = 0, h = 0;
    std::vector<uint32_t> rgba;
    std::vector<float> depth;
    std::vector<int64_t> top;   // the payload row drawn at each pixel, -1 for the ground
    std::vector<int32_t> lvl;   // its colormap level, -1 for a flat colour
};

template <class T> T block_at(const caliper_gfx::DrawList& list, uint32_t offset) {
    T t;
    std::memcpy(&t, list.arena().data + offset, sizeof t);
    return t;
}

bool nan_bits(float v) {
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

// The rows a draw reads: the payload's, or the selection's subset (every third point), from the
// first item of the draw's chunk.
struct Rows { std::vector<uint64_t> index; };
Rows rows_of(const Case& c, const std::string& path, uint64_t first_item, uint64_t count) {
    Rows r;
    const bool sel = path.rfind("selection", 0) == 0;
    std::vector<uint64_t> all;
    if (sel)
        for (uint64_t i = 0; i < c.n; i += 3) all.push_back(i);
    for (uint64_t k = 0; k < count; ++k) r.index.push_back(sel ? all[first_item + k] : first_item + k);
    return r;
}

uint64_t chunk_of(const std::string& path) {
    const size_t at = path.rfind('c');
    if (path.empty() || path == "selection" || at == std::string::npos || at + 1 >= path.size()) return 0;
    return std::strtoull(path.c_str() + at + 1, nullptr, 10);
}

// A colour faded by the depth cues as vs_points fades it, and written to an RGBA8 target.
uint32_t faded(uint32_t rgba, float w, float w0, float w1) {
    const float f = std::clamp((w - w0) / (w1 - w0), 0.0f, 1.0f);
    const float k = 1.0f - 0.5f * f;
    uint32_t out = rgba & 0xff000000u;
    for (int ch = 0; ch < 3; ++ch) {
        const float v = (float)((rgba >> (8 * ch)) & 0xffu) / 255.0f * k;
        out |= (uint32_t)std::nearbyint(std::clamp(v, 0.0f, 1.0f) * 255.0f) << (8 * ch);
    }
    return out;
}

Raster rasterise(const Case& c, const Result& r, std::vector<Square>* squares = nullptr,
                 const CameraOverride* camera = nullptr) {
    Raster out;
    out.w = r.width;
    out.h = r.height;
    const caliper_gfx::DrawList& list = r.draws;
    out.rgba.assign((size_t)out.w * out.h, list.target().clear_rgba);
    out.depth.assign((size_t)out.w * out.h, 1.0f);
    out.top.assign((size_t)out.w * out.h, -1);
    out.lvl.assign((size_t)out.w * out.h, -1);
    const bool depth_test = bool(list.target().depth);
    const float W = (float)out.w, H = (float)out.h;

    for (size_t d = 0; d < list.records().size; ++d) {
        const caliper_gfx::DrawRecord& rec = list.records()[d];
        const sh::Frame fb = block_at<sh::Frame>(list, rec.dynamic_offsets[0]);
        sh::PrimParams pp = block_at<sh::PrimParams>(list, rec.dynamic_offsets[1]);
        if (camera) {
            std::memcpy(pp.mvp, camera->mvp, sizeof pp.mvp);
            if (pp.att_ref > 0.0f) pp.att_ref = camera->att_ref;
            if (pp.fade_w1 > pp.fade_w0) {
                pp.fade_w0 = camera->fade_w0;
                pp.fade_w1 = camera->fade_w1;
            }
        }
        const sh::Cmap cm = block_at<sh::Cmap>(list, rec.dynamic_offsets[2]);
        const Rows rows = rows_of(c, r.draw_paths[d], chunk_of(r.draw_paths[d]) * caliper_scene::kMaxItems,
                                  rec.instance_count);
        const float* m = pp.mvp;
        for (uint64_t i : rows.index) {
            const float x = r.payload.pos[3 * i], y = r.payload.pos[3 * i + 1];
            const float z = c.dims == 2 ? 0.0f : r.payload.pos[3 * i + 2];
            if (nan_bits(x) || nan_bits(y) || nan_bits(z)) continue;
            float cl[4];
            for (int k = 0; k < 4; ++k) cl[k] = m[k] * x + m[4 + k] * y + m[8 + k] * z + m[12 + k] * 1.0f;
            float size = pp.size_px;
            if (pp.att_ref > 0.0f) size = size * std::clamp(pp.att_ref / std::max(cl[3], 1e-6f), 0.5f, 2.0f);
            const float kx = (size / fb.viewport[0]) * cl[3], ky = (size / fb.viewport[1]) * cl[3];
            float qx[2] = {cl[0] - kx, cl[0] + kx}, qy[2] = {cl[1] - ky, cl[1] + ky};
            for (int e = 0; e < 2; ++e) {
                qx[e] = qx[e] * fb.ndc_scale[0] + fb.ndc_offset[0] * cl[3];
                qy[e] = qy[e] * fb.ndc_scale[1] + fb.ndc_offset[1] * cl[3];
            }
            const float wv = cl[3];
            if (!(wv > 0.0f)) continue;
            const float dz = cl[2] / wv;
            if (!(dz >= 0.0f && dz <= 1.0f)) continue;   // the quad is clipped whole: its depth is constant
            // Screen space, y down; x0 < x1 and ytop < ybottom.
            auto vp = [&](float q, float half, float sign) { return (q / wv) * (sign * half) + half; };
            float sx0 = vp(qx[0], W * 0.5f, 1.0f), sx1 = vp(qx[1], W * 0.5f, 1.0f);
            float syt = vp(qy[1], H * 0.5f, -1.0f), syb = vp(qy[0], H * 0.5f, -1.0f);
            if (squares) squares->push_back({sx0, sx1, syt, syb, dz});
            // The rasteriser's fixed-point vertex positions: 8 bits below the pixel, a half rounded up.
            auto snapped = [](float v) { return std::floor(v * 256.0f + 0.5f) / 256.0f; };
            sx0 = snapped(sx0);
            sx1 = snapped(sx1);
            syt = snapped(syt);
            syb = snapped(syb);
            const int xa = std::max(0, (int)std::ceil(sx0 - 0.5f)), xb = std::min(out.w - 1, (int)std::ceil(sx1 - 0.5f) - 1);
            const int ya = std::max(0, (int)std::ceil(syt - 0.5f)), yb = std::min(out.h - 1, (int)std::ceil(syb - 0.5f) - 1);
            if (xa > xb || ya > yb) continue;
            uint32_t colour = pp.flat_rgba;
            int32_t level = -1;
            if (pp.color_mode == 1) {
                uint32_t bits;
                std::memcpy(&bits, &r.payload.aux[i], 4);
                level = (int32_t)caliper_gfx::cmap::colormap_index(bits, cm);
                colour = caliper_gfx::cmap::level_rgba(cm, (uint32_t)level);
            }
            if (pp.fade_w1 > pp.fade_w0) colour = faded(colour, cl[3], pp.fade_w0, pp.fade_w1);
            for (int py = ya; py <= yb; ++py)
                for (int px = xa; px <= xb; ++px) {
                    const size_t at = (size_t)py * out.w + px;
                    if (depth_test) {
                        if (!(dz <= out.depth[at])) continue;
                        out.depth[at] = dz;
                    }
                    out.rgba[at] = colour;
                    out.top[at] = (int64_t)i;
                    out.lvl[at] = level;
                }
        }
    }
    return out;
}

// The pixels of a readback that differ from the CPU reference rasterisation, and their places.
size_t raster_mismatches(const Result& r, const Raster& ref, std::string* where) {
    size_t bad = 0;
    for (int y = 0; y < r.height; ++y)
        for (int x = 0; x < r.width; ++x) {
            const uint32_t got = pixel(r, x, y), want = ref.rgba[(size_t)y * r.width + x];
            if (got == want) continue;
            if (++bad <= 24 && where) {
                char t[80];
                std::snprintf(t, sizeof t, " (%d,%d) %08x, the reference %08x", x, y, got, want);
                *where += t;
            }
        }
    return bad;
}

// How far a hardware point sprite's edge may sit from the exact square's, in pixels, plus the
// rasteriser's 1/256 snap: the bound of the row against the old path's images in 3-D, where the depth
// cues make sizes fractional. The sprite row below holds the device to it.
constexpr float kSpriteSlack = 1.0f / 32.0f + 1.0f / 256.0f;

// Whether the centre of pixel (x, y) lies within kSpriteSlack of an edge of one of the CPU reference's
// squares drawn at or in front of what the reference shows there, for every pixel flagged in
// `differs`; the flags of the pixels that do are cleared, so that what is left set lies outside.
void clear_sprite_edges(const Raster& ref, const std::vector<Square>& squares, std::vector<uint8_t>* differs) {
    const float e = kSpriteSlack;
    for (const Square& q : squares) {
        const int xa = std::max(0, (int)std::floor(q.x0 - e - 0.5f)), xb = std::min(ref.w - 1, (int)std::ceil(q.x1 + e - 0.5f));
        const int ya = std::max(0, (int)std::floor(q.y0 - e - 0.5f)), yb = std::min(ref.h - 1, (int)std::ceil(q.y1 + e - 0.5f));
        for (int y = ya; y <= yb; ++y)
            for (int x = xa; x <= xb; ++x) {
                const size_t at = (size_t)y * ref.w + x;
                if (!(*differs)[at] || !(q.dz <= ref.depth[at])) continue;
                const float cx = (float)x + 0.5f, cy = (float)y + 0.5f;
                const bool near_x = std::fabs(cx - q.x0) <= e || std::fabs(cx - q.x1) <= e;
                const bool near_y = std::fabs(cy - q.y0) <= e || std::fabs(cy - q.y1) <= e;
                const bool in_x = cx >= q.x0 - e && cx <= q.x1 + e, in_y = cy >= q.y0 - e && cy <= q.y1 + e;
                if ((near_x && in_y) || (near_y && in_x)) (*differs)[at] = 0;
            }
    }
}

// The old view's camera, as the merged base before step 7 composed it in float (its fit_points,
// look_at, perspective and the renderer's matrix product, kept here verbatim), and its depth cues:
// the row against the old path's images rasterises each 3-D case with it too, to show that the
// camera's float arithmetic is not what makes the view differ from the old path there.
namespace old_camera {
struct V3 { float x, y, z; };
V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 norm3(V3 a) {
    const float l = std::sqrt(dot(a, a));
    return l > 0 ? V3{a.x / l, a.y / l, a.z / l} : V3{0, 0, 1};
}
void look_at(V3 eye, V3 at, V3 up, float* m) {
    const V3 f = norm3(sub(at, eye));
    const V3 s = norm3(cross(f, up));
    const V3 u = cross(s, f);
    const float t[16] = {s.x, u.x, -f.x, 0, s.y, u.y, -f.y, 0, s.z, u.z, -f.z, 0,
                         -dot(s, eye), -dot(u, eye), dot(f, eye), 1};
    std::memcpy(m, t, sizeof(t));
}
void perspective(float fovy, float aspect, float zn, float zf, float* m) {
    const float f = 1.0f / std::tan(fovy * 0.5f);
    std::memset(m, 0, 16 * sizeof(float));
    m[0] = f / aspect;
    m[5] = f;
    m[10] = zf / (zn - zf);
    m[11] = -1.0f;
    m[14] = (zn * zf) / (zn - zf);
}
void mul_cm(const float* a, const float* b, float* out) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float acc = 0.f;
            for (int k = 0; k < 4; ++k) acc += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = acc;
        }
}
// The 3-D camera of a case after its fit, on a target of the result's size.
CameraOverride of(const Case& c, const Result& r) {
    float b[6];
    if (c.bbox) {
        std::memcpy(b, r.payload.stats + 2, sizeof b);
    } else {   // the fit scanned the CPU payload
        float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
        for (uint64_t i = 0; i < c.n; ++i)
            for (int k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], r.payload.pos[3 * i + k]);
                hi[k] = std::max(hi[k], r.payload.pos[3 * i + k]);
            }
        for (int k = 0; k < 3; ++k) { b[2 * k] = lo[k]; b[2 * k + 1] = hi[k]; }
    }
    const float aspect = (float)r.width / (float)r.height;
    const float cx = 0.5f * (b[0] + b[1]), cy = 0.5f * (b[2] + b[3]), cz = 0.5f * (b[4] + b[5]);
    const float ex = 0.5f * std::fabs(b[1] - b[0]), ey = 0.5f * std::fabs(b[3] - b[2]), ez = 0.5f * std::fabs(b[5] - b[4]);
    float rad = std::sqrt(ex * ex + ey * ey + ez * ez);
    if (!(rad > 1e-12f) || !std::isfinite(rad)) rad = 1.0f;
    const float half_fovy = 22.5f * 3.14159265f / 180.0f;
    const float half_fovx = std::atan(std::tan(half_fovy) * aspect);
    const float dist = rad / std::sin(std::min(half_fovy, half_fovx)) * 1.15f;
    const float az = 0.95f, el = 0.42f;
    const float ce = std::cos(el), se = std::sin(el), ca = std::cos(az), sa = std::sin(az);
    const V3 eye{cx + dist * ce * ca, cy + dist * se, cz + dist * ce * sa};
    float V[16], P[16], VM[16];
    look_at(eye, V3{cx, cy, cz}, V3{0, 1, 0}, V);
    const float rr = std::max(rad, 1e-6f);
    float zn = std::max(dist - 2.0f * rr, std::max(dist * 1e-3f, 1e-6f));
    float zf = dist + 2.0f * rr;
    if (zf <= zn * 1.001f) zf = zn * 2.0f;
    perspective(45.0f * 3.14159265f / 180.0f, aspect, zn, zf, P);
    const float I[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};   // the identity model matrix
    mul_cm(V, I, VM);
    CameraOverride o;
    mul_cm(P, VM, o.mvp);
    o.att_ref = dist;
    o.fade_w0 = dist;
    o.fade_w1 = dist + rad;
    return o;
}
}  // namespace old_camera

std::string references_file() { return std::string(CALIPER_TEST_SOURCE_ROOT) + "/tests/gfx/references/points_view.txt"; }

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
            if (refs.recording()) {
                // The old path is recorded as it draws. Points whose colour differs from the CPU
                // reference are noted in the line (on the merged base none are: the old path's
                // colormap was exact there).
                if (bad) MESSAGE(c.name, ": the old path differs from the CPU reference colour at ", bad, " points; ", first);
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
        CHECK_MESSAGE(want->field("cpu_ref_differs") == "none", c.name, ": the record notes points ",
                      want->field("cpu_ref_differs"), " that the old path coloured unlike the CPU reference");
        const std::string same = r.line.hash == want->hash ? "equals" : "differs from";
        if (r.line.hash != want->hash && c.dims == 3 && c.n > 1 && r.has_draws) {
            // The record is the old path's, whose 3-D points were hardware sprites of fractional
            // sizes: this readback must be the CPU reference rasterisation of its draws, byte for
            // byte, and the row against the old path's images bounds every pixel that differs.
            std::string where;
            const size_t bad = raster_mismatches(r, rasterise(c, r), &where);
            CHECK_MESSAGE(bad == 0, c.name, ": the readback differs from the old record and from the CPU reference in ",
                          bad, " pixels", where);
            MESSAGE(c.name, ": the readback ", same, " the old path's record (3-D sprites of fractional sizes) and is "
                    "the CPU reference's");
        } else {
            CHECK_MESSAGE(r.line.hash == want->hash, c.name, ": the readback's hash ", r.line.hash,
                          " differs from the reference ", want->hash);
        }
    }
}

// The view's readback against the CPU reference rasterisation of the draws the scene core compiled
// (above), byte for byte, for every case: the scene core's camera composed in double and rounded to
// f32 once is what both the GPU and the reference read, so they agree to the bit wherever the
// reference repeats the GPU's arithmetic.
TEST_CASE("gfx/Metal points view: every case matches the CPU reference rasterisation of its draws") {
    {
        Core probe;
        if (!probe.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU");
    }
    for (const Case& c : cases()) {
        if (!selected_case(c)) continue;
        CAPTURE(c.name);
        Result r;
        REQUIRE(draw_case(c, &r));
        REQUIRE_MESSAGE(r.has_draws, c.name, ": the view kept no draws of the scene core");
        const Raster ref = rasterise(c, r);
        std::string where;
        const size_t bad = raster_mismatches(r, ref, &where);
        MESSAGE(c.name, ": ", bad, " of ", (size_t)r.width * r.height, " pixels differ from the CPU reference", where);
        CHECK_MESSAGE(bad == 0, c.name, ": ", bad, " pixels differ from the CPU reference rasterisation");
    }
}

// The old path's images, captured by a build of the step's first commit on the merged base, against
// the view now. Every 2-D case and the single 3-D point are byte-identical. In a 3-D cloud the old
// path drew hardware point sprites of the depth cues' fractional sizes, whose edges can sit up to
// kSpriteSlack from the exact square's: every pixel that differs there must lie within kSpriteSlack
// of an edge of one of the CPU reference's squares drawn at or in front of it, and the row fails on
// any pixel outside. The same cases rasterised by the CPU reference with the old view's float camera
// differ from the old images in exactly the same pixels, so the camera's arithmetic is not the cause.
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
    for (const Case& c : cases()) {
        if (!selected_case(c)) continue;
        CAPTURE(c.name);
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
        REQUIRE_MESSAGE(r.has_draws, c.name, ": the view kept no draws of the scene core");
        auto at = [&](const std::vector<uint8_t>& px, size_t i) {
            uint32_t p;
            std::memcpy(&p, &px[i * 4], 4);
            return p;
        };
        // The pixels that differ from the old path's image.
        std::vector<uint8_t> differs((size_t)w * h, 0);
        size_t differ = 0;
        for (size_t i = 0; i < (size_t)w * h; ++i)
            if (at(old, i) != at(r.rgba, i)) {
                differs[i] = 1;
                ++differ;
            }
        const bool sprites = c.dims == 3 && c.n > 1;   // fractional sizes from the depth cues
        std::vector<uint8_t> outside = differs;
        if (sprites) {
            std::vector<Square> squares;
            const Raster ref = rasterise(c, r, &squares);
            clear_sprite_edges(ref, squares, &outside);
            // The old view's float camera: the same pixels differ from the old image.
            const CameraOverride oc = old_camera::of(c, r);
            const Raster with_old = rasterise(c, r, nullptr, &oc);
            size_t same = 0, other = 0;
            for (size_t i = 0; i < (size_t)w * h; ++i) {
                const bool d = at(old, i) != with_old.rgba[i];
                (d == bool(differs[i]) ? same : other) += d || differs[i];
            }
            MESSAGE(c.name, ": with the old view's float camera the CPU reference differs from the old image in ", same + other,
                    " pixels, ", other, " of them not among the view's");
            CHECK_MESSAGE(other == 0, c.name, ": with the old view's float camera ", other,
                          " pixels differ from the old image other than the view's: the camera changes pixels");
        }
        size_t beyond = 0;
        std::string near, far;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const size_t i = (size_t)y * w + x;
                if (!differs[i]) continue;
                char t[64];
                std::snprintf(t, sizeof t, " (%d,%d)", x, y);
                if (outside[i]) {
                    ++beyond;
                    far += t;
                } else {
                    near += t;
                }
            }
        MESSAGE(c.name, ": ", differ, " of ", (size_t)w * h, " pixels differ from the old path: ", differ - beyond,
                " within ", kSpriteSlack, " px of an edge of a point drawn at or in front of them, where the old path's "
                "sprite of a fractional size and the exact square may differ, and ", beyond, " otherwise",
                (differ ? ";" : ""), near.empty() ? "" : " within:", near, far.empty() ? "" : " otherwise:", far);
        if (sprites)
            CHECK_MESSAGE(beyond == 0, c.name, ": ", beyond, " pixels differ from the old path farther than ",
                          kSpriteSlack, " px from a point's edge");
        else
            CHECK_MESSAGE(differ == 0, c.name, ": ", differ, " pixels differ from the old path");
    }
}

// How Metal rasterises a point sprite: one sprite of size s at a known sub-pixel centre, read back
// along its centre row and column, against the exact square [c - s/2, c + s/2] with the centre
// snapped to 1/256 of a pixel and a pixel covered when its centre lies inside (top-left rule). For
// every whole and half size the sprite is the exact square; for a fractional size a pixel may differ
// from it only within kSpriteSlack of the square's edge. This is the bound of the row above: the old
// path drew sprites, and the depth cues of 3-D points made their sizes fractional.
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

// One camera, one set of uniform bytes: the scene core's fixture (scene/uniform_check.h) compiles to
// the same arena bytes against the recording backend and the Metal backend, and to the pinned hash,
// which the browser module's self-test checks on WebGPU too (web/test/wasm.spec.ts).
TEST_CASE("gfx/Metal points view: the scene core's uniform bytes are the same on the recording and Metal backends") {
    Core core;
    if (!core.gpu) CALIPER_SKIP_NO_GPU("no embeddable Metal GPU");
    std::unique_ptr<caliper_gfx::Backend> metal =
        caliper_gfx::make_metal_backend(core.core->renderer->native_device(), core.core->renderer->native_queue());
    REQUIRE(metal);
    caliper_gfx::RecordingBackend recording;
    uint64_t hm = 0, hr = 0;
    std::string why;
    std::string detail;
    REQUIRE_MESSAGE(caliper_scene::uniform_check(*metal, caliper_gfx::shaders::kMetalPrograms, &hm, &why, &detail), why);
    MESSAGE("Metal's blocks: ", detail);
    REQUIRE_MESSAGE(caliper_scene::uniform_check(recording, caliper_gfx::shaders::kMetalPrograms, &hr, &why), why);
    MESSAGE("uniform arena hash: Metal ", ref::hex16(hm), ", recording ", ref::hex16(hr), ", pinned ",
            ref::hex16(caliper_scene::kUniformCheckFnv1a));
    CHECK(hm == hr);
    CHECK(hm == caliper_scene::kUniformCheckFnv1a);
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
