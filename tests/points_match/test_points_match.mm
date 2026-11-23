// test_points_match.mm: the match of the library piece points@1 with the built-in points
// primitive (design/programmable-graphics.md, Library Pieces).
//
// One cloud of 256 points in 3-D is drawn twice on Metal, offscreen and without a window: once by
// the scene core as a points batch (SceneCompiler over BackendResources), and once by the piece in
// a DrawRuntime, with the camera matrix that the built-in draw used. The values sit at exact
// colormap levels, so that no value is on the boundary between two levels. The test then holds the
// two images to the tolerances of the plan: an interior pixel has the same color in both images,
// and a pixel at an edge has a piece color within 1 level of a built-in color in its 3 x 3
// neighborhood. The picks of the built-in draw at 16 pixels must name the item that the piece's ID
// image shows there. Without a Metal device the test skips (exit 77).
//
// With CALIPER_POINTS_MATCH_CAPTURE_DIR set, the test writes points-match-builtin.png,
// points-match-piece.png and points-match-diff.png (the difference scaled up 32 times) there.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../draw_parity/parity_draw.h"
#include "../gfx/scene_reference.h"
#include "draw/draw_runtime.h"
#include "gfx/colormap_keys.h"
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "scene/backend_resources.h"
#include "scene/pick.h"

#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/spec/spec.h>

// The derive half of a piece in the producer runtime, and the stb writer, come from the rig of the
// boxes and mesh matches.
#include "../boxes_match/match_rig.h"

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
#include <vector>

namespace {

namespace cd = caliper_draw;
namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
namespace dp = draw_parity;
namespace gfx = caliper_gfx;
namespace sc = caliper_scene;
namespace sh = caliper_gfx::shaders;
namespace sr = scene_reference;

constexpr std::uint32_t kSize = 256;        // the target, square
constexpr int kPoints = 256;
constexpr int kMostPoints = 4194304;   // the most that the piece publishes: 128 MiB of 32 bytes
constexpr float kPointSize = 10.0f;         // pixels
constexpr std::uint32_t kPiecePick = 1u << 24;
constexpr std::uint32_t kClear = dp::kBackground;

// What the values of the cloud are: the f32 values of the colormap, whose rows the case builds by
// hand, or a tensor of colors or class ids that the derive half of the piece turns into rows.
enum class Values { f32, u8_rows, u32_packed, u16_classes };

// The packed RGBA of point i of the color cases, r | g << 8 | b << 16 | a << 24, with whole levels.
std::uint32_t packed_color(int i) {
    return std::uint32_t((i * 53 + 40) % 256) | std::uint32_t((i * 97 + 90) % 256) << 8 |
           std::uint32_t((i * 31 + 200) % 256) << 16 | 0xff000000u;
}

// The class id of point i of the class cases.
std::uint32_t class_of(int i) { return std::uint32_t((i * 7) % 40); }

// The fixture: positions in [-1, 1]^3 from a fixed LCG, and values at exact colormap levels.
struct Cloud {
    int n;
    std::vector<float> positions;   // n rows of (x, y, z)
    std::vector<float> values;      // n values, L / 255 with L = (i * 37) % 256

    // The first 256 points are the fixture. A larger cloud lifts them in front of the others, which
    // lie in a slab behind and, being many, cover the target with points that no pixel can tell apart.
    explicit Cloud(int count = kPoints) : n(count) {
        std::uint32_t state = 12345u;
        auto next = [&]() {
            state = state * 1664525u + 1013904223u;
            return float(double(state >> 8) / double(1u << 24)) * 2.0f - 1.0f;
        };
        for (int i = 0; i < n; ++i) {
            const bool bulk = i >= kPoints;
            for (int k = 0; k < 3; ++k) {
                float v = next();
                if (n > kPoints && k == 2) v = bulk ? -0.75f + 0.25f * v : 0.2f + 0.5f * v;
                positions.push_back(v);
            }
            values.push_back(float((i * 37) % 256) / 255.0f);
        }
    }
};

// The device and the backend of the graphics layer on it, or nothing without a usable device.
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

// Reads one texture back as bytes, top row first.
std::vector<std::uint8_t> read_back(gfx::Backend& g, gfx::Texture t, std::uint32_t w, std::uint32_t h) {
    const gfx::Readback rb = g.read_texture(t, 0, 0, w, h);
    REQUIRE_MESSAGE(bool(rb), g.last_error());
    std::vector<std::uint8_t> out;
    REQUIRE(g.wait(g.last_use(t)));
    REQUIRE(g.take_readback(rb, &out));
    return out;
}

// The color of pixel (x, y) of an RGBA8 image as one packed word.
std::uint32_t pixel_of(const std::vector<std::uint8_t>& img, int x, int y) {
    std::uint32_t v = 0;
    std::memcpy(&v, img.data() + (std::size_t(y) * kSize + std::size_t(x)) * 4, 4);
    return v;
}

// The largest difference of one channel between two packed RGBA words.
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

// Whether the 3 x 3 neighborhood of (x, y), clamped to the image, shows one identity in the ID image.
bool is_interior(const std::vector<std::uint8_t>& ids, int x, int y) {
    const std::uint64_t own = identity_of(ids, x, y);
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (identity_of(ids, std::clamp(x + dx, 0, int(kSize) - 1), std::clamp(y + dy, 0, int(kSize) - 1)) != own)
                return false;
    return true;
}

// Writes an RGBA8 image as a PNG into the capture folder, when the environment names one.
void capture(const char* name, const std::vector<std::uint8_t>& rgba) {
    const char* dir = std::getenv("CALIPER_POINTS_MATCH_CAPTURE_DIR");
    if (dir == nullptr || *dir == 0) return;
    const std::string path = std::string(dir) + "/" + name;
    CHECK(stbi_write_png(path.c_str(), int(kSize), int(kSize), 4, rgba.data(), int(kSize) * 4) != 0);
    MESSAGE("capture written: ", path);
}

// The piece points@1 of the built-in library, with its files and its hash.
bool load_points_piece(cs::Piece* piece, std::string* why) {
    std::string_view manifest;
    std::map<std::string, std::string> files;
    for (const cp::LibraryFile& f : cp::library_files()) {
        const std::string_view path = f.path;
        const std::string_view prefix = "pieces/points/1/";
        if (path.substr(0, prefix.size()) != prefix) continue;
        const std::string name(path.substr(prefix.size()));
        if (name == "piece.toml") manifest = f.text;
        else files[name] = std::string(f.text);
    }
    if (manifest.empty()) return (*why = "the library has no pieces/points/1/piece.toml", false);
    cs::PieceParse parse = cs::piece_from_toml(manifest);
    if (!parse.ok()) return (*why = "piece.toml does not parse", false);
    *piece = std::move(parse.piece);
    for (const char* name : {"common.wgsl", "derive.wgsl", "draw.wgsl"})
        if (files.count(name)) piece->files[name] = files[name];
    piece->hash = cp::piece_hash(*piece);
    return true;
}

// The seven params of points@1 in the manifest's order, packed as the runtime reads them.
std::vector<std::uint8_t> points_params(float size, std::uint32_t shape, std::uint32_t color_mode, std::uint32_t color,
                                        std::uint32_t colormap, float vmin, float vmax) {
    std::vector<std::uint8_t> out(7 * 4);
    auto put = [&](int i, auto v) { std::memcpy(out.data() + 4 * i, &v, 4); };
    put(0, size);
    put(1, shape);
    put(2, color_mode);
    put(3, color);
    put(4, colormap);
    put(5, vmin);
    put(6, vmax);
    return out;
}

// The element rows of the piece's resource `points`: position (x, y, z, 1), bits, index, two pad words.
std::vector<std::uint8_t> point_rows(const Cloud& c) {
    std::vector<std::uint8_t> out(std::size_t(c.n) * 32, 0);
    for (int i = 0; i < c.n; ++i) {
        std::uint8_t* row = out.data() + std::size_t(i) * 32;
        const float pos[4] = {c.positions[3 * i], c.positions[3 * i + 1], c.positions[3 * i + 2], 1.0f};
        std::uint32_t bits = 0, index = std::uint32_t(i);
        std::memcpy(&bits, &c.values[i], 4);
        std::memcpy(row, pos, 16);
        std::memcpy(row + 16, &bits, 4);
        std::memcpy(row + 20, &index, 4);
    }
    return out;
}

void run_points_case(int n, Values kind = Values::f32, const char* name = "") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    gfx::Backend& g = *m.backend;
    const Cloud cloud(n);
    const int S = int(kSize);

    // ---- The built-in draw ----------------------------------------------------------------------
    sh::PrimParams builtin_params{};
    std::vector<std::uint8_t> builtin_color;
    struct Rig {
        gfx::Backend& g;
        std::unique_ptr<sc::BackendResources> res;
        ~Rig() { res.reset(); }
    } rig{g, nullptr};
    rig.res = std::make_unique<sc::BackendResources>(
        g, sh::kMetalPrograms, gfx::cmap::colormap_source(std::make_shared<gfx::cmap::KeyBlockCache>()));
    {
        std::string err;
        REQUIRE_MESSAGE(rig.res->init(&err), err);
    }

    // One buffer: the positions at 0 and the values after them, at a multiple of 256 bytes.
    const std::uint64_t off_values = (std::uint64_t(n) * 12 + 255) / 256 * 256;
    std::vector<std::uint8_t> bytes(off_values + std::uint64_t(n) * 4, 0);
    std::memcpy(bytes.data(), cloud.positions.data(), std::size_t(n) * 12);
    if (kind == Values::f32) {
        std::memcpy(bytes.data() + off_values, cloud.values.data(), std::size_t(n) * 4);
    } else {
        for (int i = 0; i < n; ++i) {
            const std::uint32_t v = kind == Values::u16_classes ? class_of(i) : packed_color(i);
            std::memcpy(bytes.data() + off_values + std::size_t(i) * 4, &v, 4);
        }
    }
    gfx::Buffer data = g.create_buffer({bytes.size(), gfx::kVertex | gfx::kIndex | gfx::kStorage | gfx::kCopyDst, false,
                                        "points match"});
    REQUIRE_MESSAGE(bool(data), g.last_error());
    REQUIRE(g.write_buffer(data, 0, bytes.data(), bytes.size()));
    gfx::Texture color = g.create_texture({kSize, kSize, gfx::Format::RGBA8Unorm, gfx::kRenderTarget | gfx::kCopySrcTex,
                                           "points match color"});
    gfx::Texture depth = g.create_texture({kSize, kSize, gfx::Format::Depth32Float, gfx::kRenderTarget, "points match depth"});
    REQUIRE_MESSAGE((color && depth), g.last_error());

    sc::SceneTree tree(sc::Space::k3D);
    sc::Batch batch;
    batch.primitive = sc::Primitive::Points;
    batch.arrays.resize(2);
    batch.arrays[0].role = sc::Role::Positions;
    batch.arrays[0].buffer = data;
    batch.arrays[0].offset = 0;
    batch.arrays[0].count = n;
    batch.arrays[0].stride = 12;
    batch.arrays[0].format = gfx::VertexFormat::F32x3;
    batch.arrays[1].role = kind == Values::f32 ? sc::Role::Values
                           : kind == Values::u16_classes ? sc::Role::ClassIds : sc::Role::Colors;
    batch.arrays[1].buffer = data;
    batch.arrays[1].offset = off_values;
    batch.arrays[1].count = n;
    batch.arrays[1].stride = 4;
    batch.arrays[1].format = kind == Values::f32 ? gfx::VertexFormat::F32
                             : kind == Values::u16_classes ? gfx::VertexFormat::S32 : gfx::VertexFormat::Unorm8x4;
    batch.colormap = 0;   // viridis
    batch.vmin = 0.0f;
    batch.vmax = 1.0f;
    batch.size_px = kPointSize;
    batch.point_shape = sc::PointShape::Disc;
    for (int i = 0; i < n; ++i)
        batch.bounds.add({cloud.positions[3 * i], cloud.positions[3 * i + 1], cloud.positions[3 * i + 2]});
    batch.seq = 1;
    REQUIRE(tree.set_batch(tree.ensure("cloud"), batch));
    tree.update();
    rig.res->update_class_table(tree);
    sc::Orbit orbit;
    sc::ViewCamera camera = orbit;
    sc::fit(camera, tree.scene_bounds(), 1.0);
    sc::CompileOptions options;
    options.width = options.height = kSize;
    options.pixel_scale = 1.0f;
    options.clear_rgba = kClear;
    sc::SceneCompiler compiler;

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
        REQUIRE(draws.size() == 1);
        builtin_params = draws[0].p;
        CHECK(builtin_params.color_mode == (kind == Values::f32 ? 1u : kind == Values::u16_classes ? 4u : 2u));
        frame.render(std::move(list));
        const gfx::Serial s = g.submit(std::move(frame));
        REQUIRE_MESSAGE(s != 0, g.last_error());
        REQUIRE(g.wait(s));
        builtin_color = read_back(g, color, kSize, kSize);
    }
    std::array<float, 16> mvp{};
    std::memcpy(mvp.data(), builtin_params.mvp, sizeof(float) * 16);   // column-major, as the runtime's clip_from_world

    // ---- The piece's draw -----------------------------------------------------------------------
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_points_piece(&piece, &why), why);
    cd::DrawRuntime rt(g, false);
    cd::Instance instance;
    instance.program = "cloud";
    instance.panel = "cloud";
    const cd::Verdict installed = rt.install(piece, instance, kPiecePick);
    REQUIRE_MESSAGE(installed.ok(), installed.code << " " << installed.message);
    std::vector<std::uint8_t> rows;
    std::uint32_t row_count = std::uint32_t(n);
    if (kind == Values::f32) {
        rows = point_rows(cloud);
    } else {
        // The derive half runs over the tensors, as a script's producer runtime runs it.
        std::vector<match_rig::HostTensor> tensors;
        tensors.push_back(match_rig::HostTensor::of("positions", "f32", {n, 3}, cloud.positions));
        if (kind == Values::u8_rows) {
            std::vector<std::uint8_t> v;
            for (int i = 0; i < n; ++i)
                for (int k = 0; k < 4; ++k) v.push_back(std::uint8_t((packed_color(i) >> (8 * k)) & 0xffu));
            tensors.push_back(match_rig::HostTensor::of("values", "u8", {n, 4}, v));
        } else if (kind == Values::u32_packed) {
            std::vector<std::uint32_t> v;
            for (int i = 0; i < n; ++i) v.push_back(packed_color(i));
            tensors.push_back(match_rig::HostTensor::of("values", "u32", {n}, v));
        } else {
            std::vector<std::uint16_t> v;
            for (int i = 0; i < n; ++i) v.push_back(std::uint16_t(class_of(i)));
            tensors.push_back(match_rig::HostTensor::of("values", "u16", {n}, v));
        }
        const std::vector<match_rig::Published> all =
            match_rig::run_derive(m.device, piece, tensors, {{"size", kPointSize}, {"color_mode", 2}});
        REQUIRE(all.size() == 1);
        rows = all[0].bytes;
        row_count = all[0].count;
        CHECK(row_count == std::uint32_t(n));
        REQUIRE(rows.size() == std::size_t(n) * 32);
    }
    cd::ResourceVersion version;
    version.resource = "points";
    version.step = 1;
    version.count = row_count;
    version.bytes = rows.data();
    version.size = rows.size();
    {
        const cd::Verdict v = rt.set_resource(version);
        REQUIRE_MESSAGE(v.ok(), v.code << " " << v.message);
    }
    dp::Target target(g, kSize);
    REQUIRE_MESSAGE(target.ok, g.last_error());
    cd::FrameInputs in;
    in.clip_from_world = mvp;
    in.viewport = {0, 0, float(kSize), float(kSize)};
    in.pixel_ratio = 1.0f;
    in.cursor_step = 1;
    in.latest_step = 1;
    in.live = true;
    in.params = points_params(kPointSize, 1, kind == Values::f32 ? 1 : 2, 0xffffffffu, 0, 0.0f, 1.0f);
    std::vector<gfx::Serial> serials;
    {
        const bool drawn = target.frame(rt, in, &serials, &why);
        REQUIRE_MESSAGE(drawn, why);
    }
    for (gfx::Serial s : serials) REQUIRE_MESSAGE(g.wait(s), g.last_error());
    CHECK_MESSAGE(rt.status().state == cd::DrawState::running,
                  rt.status().error.code << " " << rt.status().error.message);
    const std::vector<std::uint8_t> piece_color = read_back(g, target.host.color, kSize, kSize);
    const std::vector<std::uint8_t> piece_ids = read_back(g, target.host.id, kSize, kSize);
    REQUIRE(piece_color.size() == std::size_t(S) * S * 4);
    REQUIRE(piece_ids.size() == std::size_t(S) * S * 8);
    REQUIRE(builtin_color.size() == piece_color.size());

    // ---- The comparison -------------------------------------------------------------------------
    std::size_t interior = 0, lit_interior = 0, edge = 0, failed = 0, differ = 0;
    int largest = 0;
    std::string first;
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x) {
            const std::uint32_t pc = pixel_of(piece_color, x, y), bc = pixel_of(builtin_color, x, y);
            bool ok;
            if (is_interior(piece_ids, x, y)) {
                ++interior;
                lit_interior += pc != kClear;
                ok = pc == bc;
                differ += pc != bc;
                largest = std::max(largest, channel_gap(pc, bc));
            } else {
                ++edge;
                ok = false;
                for (int dy = -1; dy <= 1 && !ok; ++dy)
                    for (int dx = -1; dx <= 1 && !ok; ++dx)
                        ok = channel_gap(pc, pixel_of(builtin_color, std::clamp(x + dx, 0, S - 1),
                                                      std::clamp(y + dy, 0, S - 1))) <= 1;
            }
            if (!ok) {
                ++failed;
                if (first.empty()) {
                    char buf[160];
                    std::snprintf(buf, sizeof buf, "first at (%d, %d): piece %08x, built-in %08x, %s", x, y, pc, bc,
                                  is_interior(piece_ids, x, y) ? "interior" : "edge");
                    first = buf;
                }
            }
        }
    MESSAGE("points match ", std::string(name), ": ", interior, " interior pixels (", lit_interior, " lit), ", edge, " edge pixels, ",
            differ, " interior pixels differ, largest interior gap ", largest, ", ", failed, " failed");
    CHECK_MESSAGE(failed == 0, failed, " pixels outside the tolerances; ", first);
    CHECK(lit_interior >= 2000);

    // ---- The picks ------------------------------------------------------------------------------
    // Candidates: a point whose projected center pixel is interior in the piece's ID image and
    // shows that same point.
    struct Probe {
        int x, y, item;
    };
    std::vector<Probe> candidates;
    for (int i = 0; i < n; ++i) {
        const sr::F4 c = sr::mul(mvp.data(), cloud.positions[3 * i], cloud.positions[3 * i + 1], cloud.positions[3 * i + 2]);
        if (!(c.w > 0)) continue;
        const int x = int(std::floor((c.x / c.w + 1) * 0.5 * S)), y = int(std::floor((1 - c.y / c.w) * 0.5 * S));
        if (x < 1 || y < 1 || x > S - 2 || y > S - 2) continue;
        if (is_interior(piece_ids, x, y) && identity_of(piece_ids, x, y) == std::uint64_t(i) + 1) candidates.push_back({x, y, i});
    }
    MESSAGE("points match: ", candidates.size(), " points show their own center pixel");
    REQUIRE(candidates.size() >= 16);
    std::vector<Probe> probes;
    for (int k = 0; k < 16; ++k) probes.push_back(candidates[std::size_t(k) * candidates.size() / 16]);

    gfx::Texture pick_ids = g.create_texture({16, 16, gfx::Format::RG32Uint, gfx::kRenderTarget | gfx::kCopySrcTex, "pick ids"});
    gfx::Texture pick_depth =
        g.create_texture({16, 16, gfx::Format::Depth32Float, gfx::kRenderTarget | gfx::kCopySrcTex, "pick depth"});
    REQUIRE_MESSAGE((pick_ids && pick_depth), g.last_error());
    sc::Picker picker;
    int agree = 0;
    for (const Probe& pr : probes) {
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
        const std::optional<sc::Hit> hit = picker.deliver(picker.current(), pick_map.revision, id_bytes, &depth_bytes, pick_map);
        REQUIRE(hit.has_value());
        const std::uint64_t piece_item = identity_of(piece_ids, pr.x, pr.y) - 1;
        CHECK_MESSAGE(hit->item == piece_item, "pixel (", pr.x, ", ", pr.y, "): built-in item ", hit->item, ", piece item ",
                      piece_item);
        agree += hit->item == piece_item;
    }
    MESSAGE("points match: ", agree, " of 16 picks name the piece's item");

    // ---- The captures ---------------------------------------------------------------------------
    capture((std::string("points-match") + name + "-builtin.png").c_str(), builtin_color);
    capture((std::string("points-match") + name + "-piece.png").c_str(), piece_color);
    std::vector<std::uint8_t> diff(piece_color.size());
    for (std::size_t i = 0; i < diff.size(); ++i)
        diff[i] = (i % 4 == 3) ? 255 : std::uint8_t(std::min(255, std::abs(int(piece_color[i]) - int(builtin_color[i])) * 32));
    capture((std::string("points-match") + name + "-diff.png").c_str(), diff);

    g.release(pick_ids);
    g.release(pick_depth);
    g.release(color);
    g.release(depth);
    g.release(data);
}

}  // namespace

TEST_CASE("points_match: points@1 draws as the built-in points primitive") { run_points_case(kPoints); }

TEST_CASE("points_match: u8 color rows (N, 4) bound as bytes") { run_points_case(kPoints, Values::u8_rows, "-u8-rows"); }

TEST_CASE("points_match: u32 packed RGBA") { run_points_case(kPoints, Values::u32_packed, "-u32-packed"); }

TEST_CASE("points_match: u16 class ids through the default class colors") {
    run_points_case(kPoints, Values::u16_classes, "-u16-classes");
}

TEST_CASE("points_match: the most points that the piece publishes") { run_points_case(kMostPoints); }

CALIPER_GPU_TEST_MAIN()
