// test_surface_match.mm: the match of the library piece surface@1 with the built-in surface draw
// kind (design/programmable-graphics.md, Library Pieces; design/instrument-interface.md, Draw Kinds).
//
// The built-in draw of a surface entity is the spec runtime's mesh of its grid
// (build_entity_scene: the data box, two triangles a cell, holes, normals), drawn by the scene
// core's mesh program. Each case below builds that mesh from resolved arrays as the app does,
// draws it on Metal offscreen with the scene core, and draws the same grid with the piece in a
// DrawRuntime: the rows of its resource `cells` are what an app that attaches the piece writes
// (cells_of), and the camera matrix is the one that the built-in draw used. The images are held
// to the tolerances of the plan: a pixel inside one item of the piece's ID image is within
// 1 level of the built-in's color there, and a pixel at an edge is within 1 level of a built-in
// color in its 3 x 3 neighborhood. The values of the fixtures keep away from the boundary between
// two colormap levels, because the device's division may round such a value to the next level.
// Picks at pixels, with the first and the last item that a pick can name, must name the same
// item, and that item's fields must equal the built-in's pick fields.
//
// Two more cases need no image: the piece passes piece.check, its manifest and web limits and
// both composers, and its derive half on the producer runtime pools a tensor as the surface
// operator's CPU reference does. Without a Metal device the GPU cases skip (exit 77).
//
// With CALIPER_SURFACE_MATCH_CAPTURE_DIR set, the test writes surface-match-<case>-builtin.png,
// surface-match-<case>-piece.png and surface-match-<case>-diff.png (the difference scaled up 32
// times) there.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../../src/host/spec_runtime/tests/spec_test_data.h"
#include "../draw_parity/parity_fixture.h"
#include "../gfx/scene_reference.h"
#include "draw/draw_runtime.h"
#include "gfx/colormap_keys.h"
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "producer/caliper_producer.h"
#include "producer/caliper_producer_derive.h"
#include "scene/backend_resources.h"
#include "scene/camera.h"
#include "scene/pick.h"
#include "spec_runtime/spec_prepare.h"

#include <caliper_builder/operators/operators.h>
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/session/piece_commands.h>
#include <caliper_builder/spec/spec.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#include <stb_image_write.h>
#include "../match_reference.h"
#pragma clang diagnostic pop

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace cd = caliper_draw;
namespace ch = caliper_host;
namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
namespace dp = draw_parity;
namespace gfx = caliper_gfx;
namespace ops = caliper_builder::operators;
namespace sc = caliper_scene;
namespace sh = caliper_gfx::shaders;
namespace sr = scene_reference;
namespace st = spec_test;

constexpr std::uint32_t kSize = 256;   // the target, square
constexpr int kS = int(kSize);
constexpr std::uint32_t kPiecePick = 1u << 24;
constexpr std::uint32_t kClear = dp::kBackground;
constexpr std::uint32_t kFitAll = 31;   // the param fit: every range from the data

// ---- The device, images and captures ------------------------------------------------------------

struct Metal {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::unique_ptr<gfx::Backend> backend;
    std::string why;

    Metal() {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            why = "no Metal device";
            return;
        }
        if (!MPSSupportsMTLDevice(device)) {
            why = "no Metal device that MPS supports";
            return;
        }
        queue = [device newCommandQueue];
        backend = gfx::make_metal_backend((__bridge void*)device, (__bridge void*)queue);
        if (!backend) why = "the graphics layer made no Metal backend";
    }
};

std::vector<std::uint8_t> read_back(gfx::Backend& g, gfx::Texture t, std::uint32_t w, std::uint32_t h) {
    const gfx::Readback rb = g.read_texture(t, 0, 0, w, h);
    REQUIRE_MESSAGE(bool(rb), g.last_error());
    std::vector<std::uint8_t> out;
    REQUIRE(g.wait(g.last_use(t)));
    REQUIRE(g.take_readback(rb, &out));
    return out;
}

std::uint32_t pixel_of(const std::vector<std::uint8_t>& img, int x, int y) {
    std::uint32_t v = 0;
    std::memcpy(&v, img.data() + (std::size_t(y) * kSize + std::size_t(x)) * 4, 4);
    return v;
}

float depth_of(const std::vector<std::uint8_t>& img, int x, int y) {
    float v = 0;
    std::memcpy(&v, img.data() + (std::size_t(y) * kSize + std::size_t(x)) * 4, 4);
    return v;
}

int channel_gap(std::uint32_t a, std::uint32_t b) {
    int d = 0;
    for (int k = 0; k < 4; ++k) d = std::max(d, std::abs(int((a >> (8 * k)) & 0xffu) - int((b >> (8 * k)) & 0xffu)));
    return d;
}

// The identity of pixel (x, y) in an RG32Uint ID image: 0 for the background, else item + 1.
std::uint64_t identity_of(const std::vector<std::uint8_t>& ids, int x, int y) {
    std::uint32_t pick = 0, item = 0;
    const std::uint8_t* p = ids.data() + (std::size_t(y) * kSize + std::size_t(x)) * 8;
    std::memcpy(&pick, p, 4);
    std::memcpy(&item, p + 4, 4);
    return pick == 0 ? 0 : std::uint64_t(item) + 1;
}

bool is_inside(const std::vector<std::uint8_t>& ids, int x, int y) {
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (identity_of(ids, std::clamp(x + dx, 0, kS - 1), std::clamp(y + dy, 0, kS - 1)) == 0) return false;
    return true;
}

bool is_interior(const std::vector<std::uint8_t>& ids, int x, int y) {
    const std::uint64_t own = identity_of(ids, x, y);
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (identity_of(ids, std::clamp(x + dx, 0, kS - 1), std::clamp(y + dy, 0, kS - 1)) != own) return false;
    return true;
}

void capture(const std::string& name, const std::vector<std::uint8_t>& rgba) {
    const char* dir = std::getenv("CALIPER_SURFACE_MATCH_CAPTURE_DIR");
    if (dir == nullptr || *dir == 0) return;
    const std::string path = std::string(dir) + "/" + name;
    CHECK(stbi_write_png(path.c_str(), kS, kS, 4, rgba.data(), kS * 4) != 0);
    MESSAGE("capture written: ", path);
}

// The host target of a scene panel as parity_draw.h makes it, with a depth that reads back.
struct PieceTarget {
    gfx::Backend& g;
    cd::HostTarget host;
    bool ok = false;

    explicit PieceTarget(gfx::Backend& backend) : g(backend) {
        gfx::TextureDesc d;
        d.width = d.height = kSize;
        d.format = gfx::Format::RGBA8Unorm;
        d.usage = gfx::kRenderTarget | gfx::kSampled | gfx::kCopySrcTex;
        d.label = "surface color";
        host.color = g.create_texture(d);
        d.format = gfx::Format::RG32Uint;
        d.usage = gfx::kRenderTarget | gfx::kCopySrcTex;
        d.label = "surface id";
        host.id = g.create_texture(d);
        d.format = gfx::Format::Depth32Float;
        d.usage = gfx::kRenderTarget | gfx::kCopySrcTex;
        d.label = "surface depth";
        host.depth = g.create_texture(d);
        host.width = host.height = kSize;
        gfx::TextureDesc c;
        c.width = 256;
        c.height = gfx::cmap::kBuiltinColormaps;
        c.format = gfx::Format::RGBA8Unorm;
        c.usage = gfx::kSampled | gfx::kCopyDstTex;
        c.label = "surface colormaps";
        host.colormaps = g.create_texture(c);
        std::vector<std::uint32_t> rows(256u * gfx::cmap::kBuiltinColormaps);
        for (int id = 0; id < gfx::cmap::kBuiltinColormaps; ++id)
            std::memcpy(rows.data() + 256u * std::uint32_t(id), gfx::cmap::builtin_lut(id), 256 * 4);
        ok = host.color && host.id && host.depth && host.colormaps && g.write_texture(host.colormaps, rows.data(), 256 * 4);
    }
    ~PieceTarget() {
        for (gfx::Texture t : {host.color, host.id, host.depth, host.colormaps})
            if (t) g.release(t);
    }
    PieceTarget(const PieceTarget&) = delete;
    PieceTarget& operator=(const PieceTarget&) = delete;

    // One frame: the clear of the host target, then the program's frames, as the scene view submits them.
    bool frame(cd::DrawRuntime& rt, const cd::FrameInputs& in, std::string* why) {
        gfx::RenderTarget clear;
        clear.color = host.color;
        clear.id = host.id;
        clear.depth = host.depth;
        clear.clear_rgba = kClear;
        gfx::Frame f;
        f.render(gfx::DrawList(clear));
        const gfx::Serial s = g.submit(std::move(f));
        if (s == 0) return (*why = "the clear was refused: " + g.last_error(), false);
        std::vector<gfx::Frame> frames;
        if (!rt.encode(&frames, in, host)) {
            *why = "the program drew nothing: " + rt.status().error.code + " " + rt.status().error.message;
            return false;
        }
        gfx::Serial last = s;
        for (gfx::Frame& pf : frames) {
            last = g.submit(std::move(pf));
            if (last == 0) return (*why = "a frame of the program was refused: " + g.last_error(), false);
        }
        return g.wait(last) || (*why = g.last_error(), false);
    }
};

// ---- The piece ----------------------------------------------------------------------------------

bool load_surface_piece(cs::Piece* piece, std::string* why) {
    std::string_view manifest;
    std::map<std::string, std::string> files;
    for (const cp::LibraryFile& f : cp::library_files()) {
        const std::string_view path = f.path;
        const std::string_view prefix = "pieces/surface/1/";
        if (path.substr(0, prefix.size()) != prefix) continue;
        const std::string name(path.substr(prefix.size()));
        if (name == "piece.toml") manifest = f.text;
        else files[name] = std::string(f.text);
    }
    if (manifest.empty()) return (*why = "the library has no pieces/surface/1/piece.toml", false);
    cs::PieceParse parse = cs::piece_from_toml(manifest);
    if (!parse.ok()) return (*why = "piece.toml does not parse: " + parse.diagnostics.at(0).message, false);
    *piece = std::move(parse.piece);
    for (const char* name : {"common.wgsl", "derive.wgsl", "draw.wgsl"})
        if (files.count(name)) piece->files[name] = files[name];
    piece->hash = cp::piece_hash(*piece);
    return true;
}

// The params of surface@1 in the manifest's order, as the runtime reads them.
struct SurfaceParams {
    std::uint32_t max_cells = 32768, pool = 0, colormap = 0;
    float vmin = 0, vmax = 1;
    float x_lo = 0, x_hi = 1, y_lo = 0, y_hi = 1, z_lo = 0, z_hi = 1;
    float box_x = 0, box_y = 0, box_z = 0;
    std::int32_t depth_offset = 0;
    std::uint32_t fit = kFitAll;

    std::vector<std::uint8_t> bytes() const {
        std::vector<std::uint8_t> out(16 * 4);
        int i = 0;
        auto put = [&](auto v) { std::memcpy(out.data() + 4 * i++, &v, 4); };
        put(max_cells), put(pool), put(colormap), put(vmin), put(vmax), put(x_lo), put(x_hi), put(y_lo), put(y_hi);
        put(z_lo), put(z_hi), put(box_x), put(box_y), put(box_z), put(depth_offset), put(fit);
        return out;
    }
};

// One element of the resource `cells`, 32 bytes.
struct Cell {
    float x, y, z, value;
    std::int32_t row, column, step;
    std::uint32_t cols;
};
static_assert(sizeof(Cell) == 32, "a Cell is 32 bytes");

// The rows of `cells` for a built surface entity, as an app that attaches surface@1 to the entity
// writes them: the data units of each vertex, its color value, and the pick fields that
// entity_pick_fields() gives for it.
std::vector<Cell> cells_of(const ch::EntityDraw& d) {
    std::vector<Cell> out;
    if (d.error || d.draw != cs::DrawKind::surface) return out;
    out.resize(std::size_t(d.rows * d.cols));
    for (std::int64_t r = 0; r < d.rows; ++r)
        for (std::int64_t c = 0; c < d.cols; ++c) {
            const std::size_t i = std::size_t(r * d.cols + c);
            Cell& e = out[i];
            e.x = float(d.data[0][i]);
            e.y = float(d.data[1][i]);
            e.z = float(d.data[2][i]);
            e.value = i < d.values.size() ? float(d.values[i]) : std::numeric_limits<float>::quiet_NaN();
            e.row = std::int32_t(d.row_from_x && d.steps.empty() ? std::llround(d.row_coord[std::size_t(r)]) : r);
            e.column = std::int32_t(d.column_from_y ? std::llround(d.col_coord[std::size_t(c)]) : c);
            e.step = std::size_t(r) < d.steps.size() ? std::int32_t(d.steps[std::size_t(r)]) : -1;
            e.cols = std::uint32_t(d.cols);
        }
    return out;
}

// ---- The fixtures -------------------------------------------------------------------------------

struct Grid {
    int rows = 0, cols = 0;
    std::vector<float> height;   // rows x cols
    std::vector<float> values;   // empty, or rows x cols
    std::vector<float> x;        // empty, or rows: the x of each row, which are also its steps
    std::vector<float> y;        // empty, or cols
};

// Moves each finite value that sits within a thousandth of a level of the boundary between two
// colormap levels of [lo, hi] by two thousandths of a level.
void keep_off_boundaries(std::vector<float>* v, double lo, double hi) {
    for (float& x : *v) {
        if (!std::isfinite(x) || !(hi > lo)) continue;
        const double f = (double(x) - lo) / (hi - lo) * 255.0 + 0.5;
        const double frac = f - std::floor(f);
        if (f > 0.5 && f < 255.5 && (frac < 1e-3 || frac > 1.0 - 1e-3)) x = float(double(x) + 2e-3 / 255.0 * (hi - lo));
    }
}

void finite_range(const std::vector<float>& v, double* lo, double* hi) {
    *lo = 1e300, *hi = -1e300;
    for (float x : v)
        if (std::isfinite(x)) *lo = std::min(*lo, double(x)), *hi = std::max(*hi, double(x));
}

Grid terrain(int rows, int cols) {
    Grid g;
    g.rows = rows, g.cols = cols;
    g.height.resize(std::size_t(rows) * std::size_t(cols));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const double u = double(r) / std::max(1, rows - 1), v = double(c) / std::max(1, cols - 1);
            const double bump = 0.45 * std::exp(-((u - 0.55) * (u - 0.55) + (v - 0.4) * (v - 0.4)) / 0.03);
            g.height[std::size_t(r) * cols + c] = float(0.35 * std::sin(6.0 * u) * std::cos(4.5 * v) + 0.25 * u + bump);
        }
    double lo, hi;
    finite_range(g.height, &lo, &hi);
    keep_off_boundaries(&g.height, lo, hi);
    return g;
}

// One case: the grid, the options of the panel and the entity, and whether the piece takes its
// box and value range from the scene (an app's attachment into a panel of several entities) or
// from the data (fit = 31).
struct Case {
    std::string name;
    Grid grid;
    std::string panel_options = "up = \"+z\"";
    std::string entity_options = "colormap = \"viridis\"";
    std::uint32_t fit = kFitAll;
    double azimuth = -2.3, elevation = 0.95;   // radians
    bool depth = false;                        // also compare the depth images
    bool picks = true;
};

std::string spec_text(const Case& k) {
    return "caliper_spec = 1\n[applet]\nid = \"surface\"\n[[source]]\nid = \"grid\"\ncatalog = \"live:grid\"\n"
           "[[panel]]\nid = \"world\"\nprimitive = \"scene\"\noptions = { " +
           k.panel_options + " }\n[[panel.entity]]\npath = \"terrain\"\ndraw = \"surface\"\nbind = \"grid\"\noptions = { " +
           k.entity_options + " }\n";
}

caliper_builder::PanelData panel_data(const Grid& g) {
    caliper_builder::PanelData d;
    d.panel = "world";
    d.primitive = cs::Primitive::scene;
    caliper_builder::EntityData e;
    e.path = "terrain";
    e.generation = 1;
    e.slots.push_back(st::f32("height", "grid", {g.rows, g.cols}, g.height));
    if (!g.values.empty()) e.slots.push_back(st::f32("values", "grid", {g.rows, g.cols}, g.values));
    if (!g.x.empty()) {
        e.slots.push_back(st::f32("x", "grid", {g.rows}, g.x));
        e.slots[0].steps = st::f32("steps", "grid", {g.rows}, g.x).tensor;   // the rows came from a window
    }
    if (!g.y.empty()) e.slots.push_back(st::f32("y", "grid", {g.cols}, g.y));
    d.entities.push_back(std::move(e));
    return d;
}

// ---- One case: both draws, the comparison and the picks -----------------------------------------

struct Outcome {
    bool builtin_drew = false;
    std::size_t interior = 0, lit = 0, edge = 0, failed = 0;
    int max_interior = 0;
    // Inside the surface: the 3 x 3 neighborhood shows some item in every pixel of the piece's ID
    // image. There the color of the same pixel is held to 1 level too, across triangle edges.
    std::size_t inside = 0, equal = 0, lit_pixels = 0;
    int max_inside = 0;
    std::size_t depth_pixels = 0;
    double max_depth = 0;
    int picks = 0, agree = 0, fields = 0;
};

void check_fields(const ch::EntityScene& scene, const std::string& path, cd::DrawRuntime& rt, std::uint64_t item,
                  Outcome* out) {
    CAPTURE(item);
    const std::optional<caliper_builder::PickFields> f = ch::entity_pick_fields(scene, path, item);
    REQUIRE(f.has_value());
    REQUIRE(f->position.has_value());
    const std::optional<cd::PickedItem> p = rt.pick(std::uint32_t(item));
    REQUIRE(p.has_value());
    CHECK(p->item == item);
    std::map<std::string, double> got(p->fields.begin(), p->fields.end());
    REQUIRE(got.size() == 7);
    CHECK(float(got["x"]) == float((*f->position)[0]));
    CHECK(float(got["y"]) == float((*f->position)[1]));
    CHECK(float(got["z"]) == float((*f->position)[2]));
    if (f->value) CHECK(float(got["value"]) == float(*f->value));
    else CHECK(std::isnan(got["value"]));
    CHECK(std::int64_t(got["row"]) == f->row.value_or(-2));
    CHECK(std::int64_t(got["column"]) == f->column.value_or(-2));
    CHECK(std::int64_t(got["step"]) == f->step.value_or(-1));
    CHECK(p->index[0] == f->row.value_or(-2));
    CHECK(p->index[1] == f->column.value_or(-2));
    CHECK(p->index[2] == f->step.value_or(-1));
    CHECK(p->index[3] == -1);
    ++out->fields;
}

void run_case(Metal& m, const Case& k, Outcome* out) {
    gfx::Backend& g = *m.backend;
    const cs::Spec spec = st::parse(spec_text(k));
    REQUIRE(spec.panels.size() == 1);
    const ch::EntityScene scene = ch::build_entity_scene(spec.panels[0], panel_data(k.grid));
    REQUIRE(scene.draws.size() == 1);
    const ch::EntityDraw& d = scene.draws[0];
    const ch::EntityBox& box = scene.box;
    const std::vector<Cell> rows = cells_of(d);

    // ---- The built-in draw: the mesh of apply_entity_scene() through the scene core --------------
    struct Rig {
        gfx::Backend& g;
        std::unique_ptr<sc::BackendResources> res;
        std::vector<gfx::Texture> textures;
        std::vector<gfx::Buffer> buffers;
        ~Rig() {
            res.reset();
            for (gfx::Texture t : textures) g.release(t);
            for (gfx::Buffer b : buffers) g.release(b);
        }
    } rig{g, nullptr, {}, {}};
    rig.res = std::make_unique<sc::BackendResources>(
        g, sh::kMetalPrograms, gfx::cmap::colormap_source(std::make_shared<gfx::cmap::KeyBlockCache>()));
    {
        std::string err;
        REQUIRE_MESSAGE(rig.res->init(&err), err);
    }
    gfx::Texture color = g.create_texture({kSize, kSize, gfx::Format::RGBA8Unorm, gfx::kRenderTarget | gfx::kCopySrcTex,
                                           "surface match color"});
    gfx::Texture depth = g.create_texture({kSize, kSize, gfx::Format::Depth32Float, gfx::kRenderTarget | gfx::kCopySrcTex,
                                           "surface match depth"});
    REQUIRE_MESSAGE((color && depth), g.last_error());
    rig.textures = {color, depth};

    sc::SceneTree tree(sc::Space::k3D);
    REQUIRE(tree.set_up(sc::Axis::PosZ));
    sc::Aabb bounds;
    bounds.add({-box.half[0], -box.half[1], -box.half[2]});
    bounds.add({box.half[0], box.half[1], box.half[2]});
    out->builtin_drew = !d.error && !d.indices.empty();
    if (out->builtin_drew) {
        const std::uint64_t n = d.positions.size() / 3, t = d.indices.size() / 3;
        auto align = [](std::uint64_t x) { return (x + 255) / 256 * 256; };
        const std::uint64_t off_idx = align(n * 12), off_nrm = off_idx + align(t * 12), off_val = off_nrm + align(n * 12);
        std::vector<std::uint8_t> bytes(off_val + n * 4, 0);
        std::vector<float> values(d.values.begin(), d.values.end());
        std::memcpy(bytes.data(), d.positions.data(), n * 12);
        std::memcpy(bytes.data() + off_idx, d.indices.data(), t * 12);
        std::memcpy(bytes.data() + off_nrm, d.normals.data(), n * 12);
        std::memcpy(bytes.data() + off_val, values.data(), n * 4);
        gfx::Buffer data = g.create_buffer(
            {bytes.size(), gfx::kVertex | gfx::kIndex | gfx::kStorage | gfx::kCopyDst, false, "surface match mesh"});
        REQUIRE_MESSAGE(bool(data), g.last_error());
        rig.buffers.push_back(data);
        REQUIRE(g.write_buffer(data, 0, bytes.data(), bytes.size()));
        sc::Batch batch;
        batch.primitive = sc::Primitive::Mesh;
        auto array = [&](sc::Role role, std::uint64_t offset, std::uint64_t count, std::uint32_t stride,
                         gfx::VertexFormat format) {
            sc::BatchArray a;
            a.role = role;
            a.buffer = data;
            a.offset = offset;
            a.count = count;
            a.stride = stride;
            a.format = format;
            batch.arrays.push_back(a);
        };
        array(sc::Role::Positions, 0, n, 12, gfx::VertexFormat::F32x3);
        array(sc::Role::Indices, off_idx, t, 12, gfx::VertexFormat::S32);
        array(sc::Role::Normals, off_nrm, n, 12, gfx::VertexFormat::F32x3);
        array(sc::Role::Values, off_val, n, 4, gfx::VertexFormat::F32);
        batch.colormap = std::uint32_t(d.colormap);
        batch.vmin = float(d.vmin);
        batch.vmax = float(d.vmax);
        batch.depth_offset = std::int8_t(d.depth_offset);
        batch.bounds = bounds;
        batch.seq = 1;
        REQUIRE(tree.set_batch(tree.ensure(d.path), batch));
    }
    tree.update();
    rig.res->update_class_table(tree);
    sc::Orbit orbit;
    orbit.up = sc::Axis::PosZ;
    orbit.azimuth = k.azimuth;
    orbit.elevation = k.elevation;
    sc::ViewCamera camera = orbit;
    sc::fit(camera, bounds, 1.0);
    sc::CompileOptions options;
    options.width = options.height = kSize;
    options.pixel_scale = 1.0f;
    options.clear_rgba = kClear;
    sc::SceneCompiler compiler;

    std::array<float, 16> mvp{};       // the camera without the depth offset: the piece's clip_from_world
    std::array<float, 16> view{};
    {
        const sc::CameraMatrices cm = sc::camera_matrices(camera, tree, kSize, kSize);
        REQUIRE(cm.ok);
        const sc::Mat4 clip = cm.proj * cm.view;
        for (int i = 0; i < 16; ++i) mvp[std::size_t(i)] = float(clip.m[i]), view[std::size_t(i)] = float(cm.view.m[i]);
    }
    std::vector<std::uint8_t> builtin_color, builtin_depth;
    {
        gfx::Frame frame;
        compiler.prepare(tree, *rig.res, frame);
        gfx::RenderTarget target;
        target.color = color;
        target.depth = depth;
        target.clear_rgba = kClear;
        gfx::DrawList list(target);
        sc::DrawMap map;
        const sc::CompileResult r = compiler.compile(tree, camera, options, *rig.res, list, &map);
        REQUIRE_MESSAGE(r.ok, (r.notes.empty() ? std::string() : r.notes[0]));
        CHECK(r.skipped == 0);
        CHECK_MESSAGE(rig.res->error().empty(), rig.res->error());
        const std::vector<sr::DrawInfo> draws = sr::draws_of(list, map, tree);
        REQUIRE(draws.size() == (out->builtin_drew ? 1u : 0u));
        if (out->builtin_drew) {
            const sh::PrimParams& p = draws[0].p;
            CHECK(p.color_mode == 1);
            CHECK(p.shade_mode == 1);
            // The draw's matrix holds the depth offset; the piece applies it itself, so it takes
            // the matrix without it: row 2 of D * M is row 2 of M less k / 8192 times row 3.
            const double kk = double(d.depth_offset) / 8192.0;
            for (int c = 0; c < 4; ++c) {
                for (int r2 = 0; r2 < 4; ++r2) {
                    double want = p.mvp[c * 4 + r2];
                    if (r2 == 2) want += kk * double(p.mvp[c * 4 + 3]);
                    CHECK(std::fabs(want - double(mvp[std::size_t(c * 4 + r2)])) <= 1e-5 * (1.0 + std::fabs(want)));
                }
            }
            if (d.depth_offset == 0) std::memcpy(mvp.data(), p.mvp, sizeof(float) * 16);
        }
        frame.render(std::move(list));
        const gfx::Serial s = g.submit(std::move(frame));
        REQUIRE_MESSAGE(s != 0, g.last_error());
        REQUIRE(g.wait(s));
        const std::vector<std::uint8_t> px = read_back(g, color, kSize, kSize);
        builtin_color = match_reference::picture("color", kSize, kSize, &px);
        if (k.depth) {
            const std::vector<std::uint8_t> dz = read_back(g, depth, kSize, kSize);
            builtin_depth = match_reference::picture("depth", kSize, kSize, &dz);
        }
    }

    // ---- The piece's draw ------------------------------------------------------------------------
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_surface_piece(&piece, &why), why);
    cd::DrawRuntime rt(g, false);
    cd::Instance instance;
    instance.program = "terrain";
    instance.panel = "world";
    const cd::Verdict installed = rt.install(piece, instance, kPiecePick);
    REQUIRE_MESSAGE(installed.ok(), installed.code << " " << installed.message);
    cd::ResourceVersion version;
    version.resource = "cells";
    version.step = 1;
    version.count = std::uint32_t(rows.size());
    version.bytes = rows.empty() ? nullptr : reinterpret_cast<const std::uint8_t*>(rows.data());
    version.size = rows.size() * sizeof(Cell);
    {
        const cd::Verdict v = rt.set_resource(version);
        REQUIRE_MESSAGE(v.ok(), v.code << " " << v.message);
    }
    PieceTarget target(g);
    REQUIRE_MESSAGE(target.ok, g.last_error());
    SurfaceParams params;
    params.colormap = std::uint32_t(d.colormap);
    params.fit = k.fit;
    params.vmin = float(d.vmin);
    params.vmax = float(d.vmax);
    params.depth_offset = d.depth_offset;
    if (k.fit != kFitAll) {   // the scene's box, as an app writes it for a panel of several entities
        params.x_lo = float(box.lo[0]), params.x_hi = float(box.hi[0]);
        params.y_lo = float(box.lo[1]), params.y_hi = float(box.hi[1]);
        params.z_lo = float(box.lo[2]), params.z_hi = float(box.hi[2]);
        params.box_x = float(box.half[0]), params.box_y = float(box.half[1]), params.box_z = float(box.half[2]);
    }
    cd::FrameInputs in;
    in.clip_from_world = mvp;
    in.view_from_world = view;
    in.viewport = {0, 0, float(kSize), float(kSize)};
    in.pixel_ratio = 1.0f;
    in.cursor_step = 1;
    in.latest_step = 1;
    in.live = true;
    in.params = params.bytes();
    REQUIRE_MESSAGE(target.frame(rt, in, &why), why);
    CHECK_MESSAGE(rt.status().state == cd::DrawState::running,
                  rt.status().error.code << " " << rt.status().error.message);
    const std::vector<std::uint8_t> piece_color = read_back(g, target.host.color, kSize, kSize);
    const std::vector<std::uint8_t> piece_ids = read_back(g, target.host.id, kSize, kSize);
    REQUIRE(builtin_color.size() == piece_color.size());

    // ---- The captures ----------------------------------------------------------------------------
    capture("surface-match-" + k.name + "-builtin.png", builtin_color);
    capture("surface-match-" + k.name + "-piece.png", piece_color);
    std::vector<std::uint8_t> diff(piece_color.size());
    for (std::size_t i = 0; i < diff.size(); ++i)
        diff[i] = (i % 4 == 3) ? 255 : std::uint8_t(std::min(255, std::abs(int(piece_color[i]) - int(builtin_color[i])) * 32));
    capture("surface-match-" + k.name + "-diff.png", diff);

    // ---- The comparison --------------------------------------------------------------------------
    std::string first;
    for (int y = 0; y < kS; ++y)
        for (int x = 0; x < kS; ++x) {
            const std::uint32_t pc = pixel_of(piece_color, x, y), bc = pixel_of(builtin_color, x, y);
            out->lit_pixels += pc != kClear;
            out->equal += pc == bc;
            bool ok;
            if (is_inside(piece_ids, x, y)) {
                ++out->inside;
                const int gap = channel_gap(pc, bc);
                out->max_inside = std::max(out->max_inside, gap);
                if (gap > 1) {
                    ++out->failed;
                    if (first.empty()) {
                        char buf[160];
                        std::snprintf(buf, sizeof buf, "first at (%d, %d): piece %08x, built-in %08x, inside", x, y, pc, bc);
                        first = buf;
                    }
                }
            }
            if (is_interior(piece_ids, x, y)) {
                ++out->interior;
                out->lit += pc != kClear;
                const int gap = channel_gap(pc, bc);
                out->max_interior = std::max(out->max_interior, gap);
                ok = gap <= 1;
            } else {
                ++out->edge;
                ok = false;
                for (int dy = -1; dy <= 1 && !ok; ++dy)
                    for (int dx = -1; dx <= 1 && !ok; ++dx)
                        ok = channel_gap(pc, pixel_of(builtin_color, std::clamp(x + dx, 0, kS - 1),
                                                      std::clamp(y + dy, 0, kS - 1))) <= 1;
            }
            if (!ok) {
                ++out->failed;
                if (first.empty()) {
                    char buf[160];
                    std::snprintf(buf, sizeof buf, "first at (%d, %d): piece %08x, built-in %08x, %s", x, y, pc, bc,
                                  is_interior(piece_ids, x, y) ? "interior" : "edge");
                    first = buf;
                }
            }
        }
    MESSAGE(k.name, ": ", out->lit_pixels, " lit pixels, ", out->equal, " of ", kS * kS, " equal; ", out->interior,
            " interior pixels (", out->lit, " lit, largest gap ", out->max_interior, "), ", out->inside,
            " inside the surface (largest gap ", out->max_inside, "), ", out->edge, " edge pixels, ", out->failed,
            " failed");
    CHECK_MESSAGE(out->failed == 0, k.name, ": ", out->failed, " pixels outside the tolerances; ", first);
    if (!out->builtin_drew) CHECK(out->lit_pixels == 0);

    if (k.depth && out->builtin_drew) {
        const std::vector<std::uint8_t> piece_depth = read_back(g, target.host.depth, kSize, kSize);
        for (int y = 0; y < kS; ++y)
            for (int x = 0; x < kS; ++x) {
                if (!is_inside(piece_ids, x, y)) continue;
                const double gap = std::fabs(double(depth_of(piece_depth, x, y)) - double(depth_of(builtin_depth, x, y)));
                out->max_depth = std::max(out->max_depth, gap);
                ++out->depth_pixels;
            }
        MESSAGE(k.name, ": depth of ", out->depth_pixels, " pixels inside the surface, largest difference ", out->max_depth);
        CHECK(out->depth_pixels > 1000);
        CHECK(out->max_depth <= 1e-5);   // the offset moves depth by 3 / 8192, about 3.7e-4
    }

    // ---- The picks -------------------------------------------------------------------------------
    if (out->builtin_drew && k.picks) {
        // The first item that a pick can name is vertex 0, the first corner of cell (0, 0); the last is
        // vertex (R - 2, C - 1), the corner b of the second triangle of cell (R - 2, C - 2).
        const std::uint64_t first_item = 0, last_item = std::uint64_t((d.rows - 2) * d.cols + (d.cols - 1));
        struct Probe {
            int x, y;
            std::uint64_t item;
        };
        std::vector<Probe> candidates, chosen;
        std::optional<Probe> at_first, at_last;
        for (int y = 1; y < kS - 1; ++y)
            for (int x = 1; x < kS - 1; ++x) {
                const std::uint64_t id = identity_of(piece_ids, x, y);
                if (id == 0) continue;
                const bool interior = is_interior(piece_ids, x, y);
                if (interior) candidates.push_back({x, y, id - 1});
                if (id - 1 == first_item && (!at_first || interior)) at_first = Probe{x, y, id - 1};
                if (id - 1 == last_item && (!at_last || interior)) at_last = Probe{x, y, id - 1};
            }
        REQUIRE(candidates.size() >= 14);
        CHECK_MESSAGE(at_first.has_value(), k.name, ": no interior pixel shows the first item");
        CHECK_MESSAGE(at_last.has_value(), k.name, ": no interior pixel shows the last item");
        if (at_first) chosen.push_back(*at_first);
        if (at_last) chosen.push_back(*at_last);
        for (int i = 0; i < 14; ++i) chosen.push_back(candidates[std::size_t(i) * candidates.size() / 14]);

        gfx::Texture pick_ids =
            g.create_texture({16, 16, gfx::Format::RG32Uint, gfx::kRenderTarget | gfx::kCopySrcTex, "pick ids"});
        gfx::Texture pick_depth =
            g.create_texture({16, 16, gfx::Format::Depth32Float, gfx::kRenderTarget | gfx::kCopySrcTex, "pick depth"});
        REQUIRE_MESSAGE((pick_ids && pick_depth), g.last_error());
        rig.textures.push_back(pick_ids);
        rig.textures.push_back(pick_depth);
        sc::Picker picker;
        for (const Probe& pr : chosen) {
            CAPTURE(pr.x);
            CAPTURE(pr.y);
            const sc::PickWindow w = picker.request(pr.x + 0.5f, pr.y + 0.5f, 8, kSize, kSize);
            const std::int32_t window[4] = {w.x0, w.y0, w.width, w.height};
            gfx::RenderTarget pt;
            pt.color = pick_ids;
            pt.depth = pick_depth;
            gfx::DrawList pick(pt);
            sc::DrawMap pick_map;
            const sc::CompileResult r = compiler.compile_pick(tree, camera, options, window, *rig.res, pick, &pick_map);
            REQUIRE_MESSAGE(r.ok, (r.notes.empty() ? std::string() : r.notes[0]));
            gfx::Frame f;
            f.render(std::move(pick));
            REQUIRE_MESSAGE(g.submit(std::move(f)) != 0, g.last_error());
            const std::vector<std::uint8_t> id_bytes = read_back(g, pick_ids, 16, 16);
            const std::vector<std::uint8_t> depth_bytes = read_back(g, pick_depth, 16, 16);
            const std::optional<sc::Hit> hit =
                picker.deliver(picker.current(), pick_map.revision, id_bytes, &depth_bytes, pick_map);
            CHECK(hit.has_value());
            if (!hit) continue;
            ++out->picks;
            CHECK_MESSAGE(hit->item == pr.item, k.name, ": pixel (", pr.x, ", ", pr.y, "): built-in item ", hit->item,
                          ", piece item ", pr.item);
            out->agree += hit->item == pr.item;
            check_fields(scene, d.path, rt, pr.item, out);
        }
        // A pick without a pointer, as selection.set makes it: the first and the last element.
        check_fields(scene, d.path, rt, 0, out);
        check_fields(scene, d.path, rt, std::uint64_t(d.rows * d.cols - 1), out);
        MESSAGE(k.name, ": ", out->agree, " of ", out->picks, " picks name the piece's item; ", out->fields,
                " items have the built-in's fields");
    }

}

// ---- The derive half on the producer runtime ----------------------------------------------------

// A tensor in a shared MTLBuffer, imported in place.
struct Tensor {
    id<MTLBuffer> buffer = nil;
    std::uint32_t handle = 0;
};

Tensor import_tensor(id<MTLDevice> device, caliper_producer* rt, const char* name, const std::vector<float>& v,
                     std::int64_t rows, std::int64_t cols) {
    Tensor t;
    t.buffer = [device newBufferWithLength:v.size() * 4 options:MTLResourceStorageModeShared];
    std::memcpy([t.buffer contents], v.data(), v.size() * 4);
    caliper_producer_tensor d{};
    d.name = name;
    d.buffer = (__bridge void*)t.buffer;
    d.byte_size = v.size() * 4;
    d.dtype = CALIPER_PRODUCER_F32;
    d.ndim = 2;
    d.shape[0] = rows;
    d.shape[1] = cols;
    caliper_producer_error e{};
    REQUIRE_MESSAGE(caliper_producer_import(rt, &d, &t.handle, &e) == CALIPER_PRODUCER_OK, (e.message ? e.message : ""));
    caliper_producer_error_free(&e);
    return t;
}

// The derive declaration of surface@1 as the script gets it from PROGRAM, with its strings kept alive.
struct DeriveDecl {
    std::string wgsl, common;
    std::vector<caliper_producer_input> inputs;
    std::vector<const char*> fields;
    caliper_producer_resource_desc cells{};
    std::vector<const char*> writes = {"cells"};
    caliper_producer_derive_pass pass{};
    std::vector<caliper_producer_param> params;
    std::vector<const char*> param_types;

    DeriveDecl(const cs::Piece& piece, std::uint32_t height, std::uint32_t values) {
        const cp::Resolved resolved = cp::resolve_imports(piece);
        REQUIRE(resolved.verdict.ok());
        wgsl = resolved.files.at("derive.wgsl");
        common = resolved.lib_text;
        if (auto it = resolved.files.find("common.wgsl"); it != resolved.files.end()) common += it->second;
        inputs = {{"height", height}, {"values", values}};
        cells.id = "cells";
        cells.kind = CALIPER_PRODUCER_RESOURCE_BUFFER;
        cells.struct_name = "Cell";
        static const char* const kFields[8] = {"x: f32",   "y: f32",      "z: f32",    "value: f32",
                                               "row: i32", "column: i32", "step: i32", "cols: u32"};
        fields.assign(kFields, kFields + 8);
        cells.fields = fields.data();
        cells.field_count = fields.size();
        cells.count = 262144;
        cells.counted = 1;
        cells.publish = 1;
        pass.entry = "pool_cells";
        pass.workgroups[0] = 1024;
        pass.workgroups[1] = pass.workgroups[2] = 1;
        pass.writes = writes.data();
        pass.write_count = writes.size();
        for (const cs::PieceParam& p : piece.params) {
            params.push_back({p.name.c_str(), p.default_value});
            param_types.push_back(cs::param_type_name(p.type));
        }
    }
    caliper_producer_derive_desc desc() const {
        caliper_producer_derive_desc d{};
        d.id = "terrain";
        d.wgsl = wgsl.c_str();
        d.wgsl_size = wgsl.size();
        d.common = common.empty() ? nullptr : common.c_str();
        d.common_size = common.size();
        d.inputs = inputs.data();
        d.input_count = inputs.size();
        d.resources = &cells;
        d.resource_count = 1;
        d.passes = &pass;
        d.pass_count = 1;
        d.params = params.data();
        d.param_types = param_types.data();
        d.param_count = params.size();
        d.time_limit_ms = 50.0;
        d.memory_limit_bytes = 64ull << 20;
        return d;
    }
};

bool same_float(float a, float b, float rel) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return std::fabs(a - b) <= rel * std::max(1.0f, std::fabs(b));
}

}  // namespace

TEST_CASE("surface_match: surface@1 passes piece.check, its manifest and web limits, and both composers") {
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_surface_piece(&piece, &why), why);
    const cp::Verdict manifest = cp::check_piece_manifest(piece);
    CHECK_MESSAGE(manifest.ok(), manifest.code, " ", manifest.message, " at ", manifest.where);
    CHECK(cp::check_web_limits(piece).ok());
    const cp::Resolved resolved = cp::resolve_imports(piece);
    REQUIRE(resolved.verdict.ok());
    const cp::Composed derive = cp::compose_derive(piece, resolved);
    CHECK_MESSAGE(derive.verdict.ok(), derive.verdict.message);
    REQUIRE(piece.draw.has_value());
    CHECK(piece.draw->passes.size() == 4);
    for (std::size_t i = 0; i < piece.draw->passes.size(); ++i) {
        const cp::Composed pass = cp::compose_draw(piece, resolved, i);
        CHECK_MESSAGE(pass.verdict.ok(), "draw pass ", i, ": ", pass.verdict.message);
    }
    // The published bytes: 262,144 cells of 32 bytes, the most cells of the surface operator.
    cp::ElementLayout layout;
    REQUIRE(cp::element_layout(piece.resources.at(0).layout, &layout).ok());
    CHECK(layout.stride == 32);
    CHECK(piece.resources.at(0).count == 262144u);
    CHECK(std::uint64_t(piece.resources.at(0).count) * layout.stride <= cs::piece_limits::kPublishedResourceBytesMax);
    const nlohmann::json check = caliper_builder::check_piece_json(cs::to_json(piece), false, nullptr);
    CHECK_MESSAGE(check.value("ok", false), check["messages"].dump());
    CHECK(check["hash"] == piece.hash);
    MESSAGE("piece.check: ", check["passes"].size(), " passes, ", check["messages"].dump());
    // The pixel comparison of the app and the browser builds the same piece without the TOML reader.
    const cs::Piece parity = dp::surface_piece();
    CHECK(cp::canonical_manifest(parity) == cp::canonical_manifest(piece));
    CHECK(parity.files == piece.files);
    CHECK(parity.hash == piece.hash);
}

TEST_CASE("surface_match: the derive half pools a tensor as the surface operator's CPU reference does") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) CALIPER_SKIP_NO_GPU("no Metal device");
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_surface_piece(&piece, &why), why);
    constexpr std::int64_t R = 301, C = 203;
    std::vector<float> height(std::size_t(R * C)), values(std::size_t(R * C)), other(35, 1.0f);
    for (std::int64_t r = 0; r < R; ++r)
        for (std::int64_t c = 0; c < C; ++c) {
            float h = float(std::sin(r * 0.07) * std::cos(c * 0.05) + r * 0.001);
            if (r >= 40 && r < 52 && c >= 60 && c < 80) h = std::numeric_limits<float>::quiet_NaN();   // whole blocks
            if ((r * 7 + c * 3) % 97 == 0) h = std::numeric_limits<float>::quiet_NaN();                // single cells
            height[std::size_t(r * C + c)] = h;
            values[std::size_t(r * C + c)] = float(std::cos(r * 0.03 + c * 0.02));
        }
    caliper_producer* rt = nullptr;
    caliper_producer_error e{};
    REQUIRE(caliper_producer_create(nullptr, &rt, &e) == CALIPER_PRODUCER_OK);
    const Tensor th = import_tensor(device, rt, "grid.height", height, R, C);
    const Tensor tv = import_tensor(device, rt, "grid.values", values, R, C);
    const Tensor to = import_tensor(device, rt, "grid.other", other, 5, 7);

    struct Program {
        std::uint32_t id = 0;
        bool own_values = false;
    };
    std::vector<Program> programs;
    for (bool own : {true, false}) {
        DeriveDecl decl(piece, th.handle, own ? tv.handle : to.handle);
        const caliper_producer_derive_desc desc = decl.desc();
        Program p;
        p.own_values = own;
        const caliper_producer_status s = caliper_producer_compile_derive(rt, &desc, &p.id, &e);
        if (s == CALIPER_PRODUCER_UNSUPPORTED) {
            MESSAGE("no WGSL compiler or composer in this build: ", (e.message ? e.message : ""));
            caliper_producer_error_free(&e);
            caliper_producer_destroy(rt);
            return;
        }
        REQUIRE_MESSAGE(s == CALIPER_PRODUCER_OK, (e.code ? e.code : ""), ": ", (e.message ? e.message : ""));
        programs.push_back(p);
    }
    int runs = 0;
    std::size_t compared = 0, mean_rounded = 0;
    for (const Program& prog : programs)
        for (std::uint32_t pool : {0u, 1u, 2u})
            for (std::int64_t max_cells : {std::int64_t(4096), std::int64_t(262144), std::int64_t(1)}) {
                CAPTURE(prog.own_values);
                CAPTURE(pool);
                CAPTURE(max_cells);
                const caliper_producer_param set[2] = {{"pool", double(pool)}, {"max_cells", double(max_cells)}};
                REQUIRE(caliper_producer_set_params(rt, prog.id, set, 2, &e) == CALIPER_PRODUCER_OK);
                caliper_producer_frame frame{100, 0.0, 3};
                caliper_producer_pass pass{};
                REQUIRE_MESSAGE(caliper_producer_run(rt, &prog.id, 1, &frame, 0, &pass, &e) == CALIPER_PRODUCER_OK,
                                (e.message ? e.message : ""));
                CHECK(pass.ran == 1);
                ++runs;
                const ops::Pool which = pool == 1 ? ops::Pool::max : pool == 2 ? ops::Pool::min : ops::Pool::mean;
                const ops::Grid want = ops::surface_reference(height.data(), R, C, nullptr, max_cells, which);
                const ops::Grid want_values = ops::surface_reference(values.data(), R, C, nullptr, max_cells, which);
                caliper_producer_output out{};
                REQUIRE(caliper_producer_read_resource(rt, prog.id, "cells", &out, &e) == CALIPER_PRODUCER_OK);
                REQUIRE(out.count == std::uint32_t(want.rows * want.cols));
                REQUIRE(out.byte_size == std::uint64_t(out.count) * sizeof(Cell));
                std::vector<Cell> got(out.count);
                std::memcpy(got.data(), static_cast<const std::uint8_t*>([(__bridge id<MTLBuffer>)out.buffer contents]) +
                                            out.byte_offset,
                            out.byte_size);
                for (std::int64_t i = 0; i < want.rows; ++i)
                    for (std::int64_t j = 0; j < want.cols; ++j) {
                        const Cell& c = got[std::size_t(i * want.cols + j)];
                        const float h = want.height[std::size_t(i * want.cols + j)];
                        const float v = prog.own_values ? want_values.height[std::size_t(i * want.cols + j)] : h;
                        // A mean divides on the device, which may round the last bit otherwise.
                        const float rel = pool == 0 ? 2e-7f : 0.0f;
                        if (!same_float(c.z, h, rel) || !same_float(c.value, v, rel)) {
                            CHECK_MESSAGE(false, "cell (", i, ", ", j, "): z ", c.z, " value ", c.value, ", want ", h,
                                          " and ", v);
                        }
                        mean_rounded += pool == 0 && !(c.z == h || (std::isnan(c.z) && std::isnan(h)));
                        CHECK(c.x == want.x[std::size_t(i)]);
                        CHECK(c.y == want.y[std::size_t(j)]);
                        CHECK(c.row == std::int32_t(std::llround(want.x[std::size_t(i)])));
                        CHECK(c.column == std::int32_t(std::llround(want.y[std::size_t(j)])));
                        CHECK(c.step == -1);
                        CHECK(c.cols == std::uint32_t(want.cols));
                        ++compared;
                    }
            }
    MESSAGE("derive: ", runs, " runs, ", compared, " cells against the reference, ", mean_rounded,
            " means that differ from it in the last bits");
    caliper_producer_error_free(&e);
    caliper_producer_destroy(rt);
}

TEST_CASE("surface_match: surface@1 draws and picks as the built-in surface draw kind") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);

    std::vector<Case> cases;
    {
        Case k;
        k.name = "terrain";
        k.grid = terrain(40, 56);
        cases.push_back(k);
    }
    {
        // Holes from NaN and infinite heights, a values slot with a NaN, magma over a given range,
        // and the box from the data.
        Case k;
        k.name = "holes";
        k.grid = terrain(33, 47);
        Grid& g = k.grid;
        for (int r = 10; r < 15; ++r)
            for (int c = 20; c < 27; ++c) g.height[std::size_t(r) * 47 + c] = std::numeric_limits<float>::quiet_NaN();
        g.height[25 * 47 + 5] = std::numeric_limits<float>::infinity();
        g.height[3 * 47 + 40] = -std::numeric_limits<float>::infinity();
        g.values.resize(g.height.size());
        for (int r = 0; r < 33; ++r)
            for (int c = 0; c < 47; ++c) g.values[std::size_t(r) * 47 + c] = float(std::cos(r * 0.21) * std::sin(c * 0.13));
        g.values[7 * 47 + 7] = std::numeric_limits<float>::quiet_NaN();
        keep_off_boundaries(&g.values, -0.4, 0.9);
        k.entity_options = "colormap = \"magma\", vmin = -0.4, vmax = 0.9";
        k.fit = 4 | 8 | 16;
        cases.push_back(k);
    }
    {
        // The x of each row from a window, with its step, an uneven y, the box option, turbo, a
        // depth offset, and the box and range from the scene as an app writes them.
        Case k;
        k.name = "coords";
        k.grid = terrain(24, 30);
        for (int r = 0; r < 24; ++r) k.grid.x.push_back(float(100 + 16 * r + (r % 3) * 5));
        for (int c = 0; c < 30; ++c) k.grid.y.push_back(float(c * c) * 0.05f);
        k.panel_options = "up = \"+z\", box = [3.0, 1.0, 0.8]";
        k.entity_options = "colormap = \"turbo\", depth_offset = 3";
        k.fit = 0;
        k.depth = true;
        cases.push_back(k);
    }
    {
        // One height everywhere: the z range and the value range widen by a half on each side.
        Case k;
        k.name = "flat";
        k.grid.rows = 12, k.grid.cols = 12;
        k.grid.height.assign(144, 0.5f);
        k.entity_options = "colormap = \"greys\"";
        k.picks = false;   // a flat grid seen from above shows its far corner small
        cases.push_back(k);
    }
    {
        // The largest grid: 262,144 cells, the limit of a surface entity and the 8 MiB that the
        // piece's resource may publish.
        Case k;
        k.name = "largest";
        k.grid = terrain(512, 512);
        k.picks = false;   // cells are smaller than a pixel: no item holds a 3 x 3 window
        cases.push_back(k);
    }
    for (const auto& [name, r, c] : std::vector<std::tuple<std::string, int, int>>{
             {"one-cell", 1, 1}, {"one-row", 1, 9}, {"one-column", 7, 1}}) {
        Case k;
        k.name = name;
        k.grid = terrain(r, c);
        cases.push_back(k);
    }

    for (const Case& k : cases) {
        SUBCASE(k.name.c_str()) {
            Outcome out;
            run_case(m, k, &out);
            if (k.name == "terrain" || k.name == "holes" || k.name == "coords" || k.name == "largest") {
                CHECK(out.builtin_drew);
                CHECK(out.lit_pixels >= 2000);
            }
            if (k.name.rfind("one-", 0) == 0) CHECK_FALSE(out.builtin_drew);
        }
    }
}

TEST_CASE("surface_match: an empty grid publishes no cells and draws nothing") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    gfx::Backend& g = *m.backend;
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_surface_piece(&piece, &why), why);
    cd::DrawRuntime rt(g, false);
    cd::Instance instance;
    instance.program = "terrain";
    instance.panel = "world";
    REQUIRE(rt.install(piece, instance, kPiecePick).ok());
    cd::ResourceVersion version;
    version.resource = "cells";
    version.step = 1;
    version.count = 0;
    const cd::Verdict v = rt.set_resource(version);
    REQUIRE_MESSAGE(v.ok(), v.code << " " << v.message);
    PieceTarget target(g);
    REQUIRE(target.ok);
    cd::FrameInputs in;
    in.clip_from_world = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    in.view_from_world = in.clip_from_world;
    in.viewport = {0, 0, float(kSize), float(kSize)};
    in.params = SurfaceParams().bytes();
    REQUIRE_MESSAGE(target.frame(rt, in, &why), why);
    const std::vector<std::uint8_t> color = read_back(g, target.host.color, kSize, kSize);
    std::size_t lit = 0;
    for (int y = 0; y < kS; ++y)
        for (int x = 0; x < kS; ++x) lit += pixel_of(color, x, y) != kClear;
    CHECK(lit == 0);
    CHECK_FALSE(rt.pick(0).has_value());
    // The built-in draw has no entity to draw either: a grid with no rows is over the entity's limits.
    const cs::Spec spec = st::parse(spec_text(Case{}));
    Grid empty;
    empty.rows = 0, empty.cols = 5;
    const ch::EntityScene scene = ch::build_entity_scene(spec.panels[0], panel_data(empty));
    REQUIRE(scene.draws.size() == 1);
    CHECK(bool(scene.draws[0].error));
}

CALIPER_GPU_TEST_MAIN()
