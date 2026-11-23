// Points library tests compare Metal output with saved images and pick identities.
// The original renderer supplied the reference cameras and pixel positions.
// Interior colors are exact. Edge colors retain the one-level neighborhood tolerance.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../draw_parity/parity_draw.h"
#include "draw/draw_runtime.h"
#include "gfx/colormap_keys.h"
#include "gfx/metal_backend.h"
#include "scene/camera.h"

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

constexpr std::uint32_t kSize = 256;        // the target, square
constexpr int kPoints = 256;
constexpr int kMostPoints = 4194304;   // the most that the piece publishes: 128 MiB of 32 bytes
constexpr float kPointSize = 10.0f;         // pixels
constexpr std::uint32_t kPiecePick = 1u << 24;
constexpr std::uint32_t kClear = dp::kBackground;

struct ClipPoint { float x, y, z, w; };
ClipPoint project(const float m[16], float x, float y, float z) {
    return {m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13],
            m[2] * x + m[6] * y + m[10] * z + m[14], m[3] * x + m[7] * y + m[11] * z + m[15]};
}


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

    const auto builtin_color = match_reference::picture("color", kSize, kSize, nullptr);
    const auto& reference = match_reference::metadata(match_reference::last_name("color"));
    const auto mvp = reference.at("mvp").get<std::array<float, 16>>();

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
    const auto& probes = reference.at("picks");
    REQUIRE(probes.size() == 16);
    int agree = 0;
    for (const auto& probe : probes) {
        const int x = probe.at("x").get<int>(), y = probe.at("y").get<int>();
        const auto item = probe.at("item").get<std::uint64_t>();
        const auto id = identity_of(piece_ids, x, y);
        CHECK_MESSAGE(id == item + 1, "pixel (", x, ", ", y, "): saved item ", item, ", piece identity ", id);
        const auto picked = rt.pick(std::uint32_t(item));
        REQUIRE(picked.has_value());
        CHECK(picked->item == item);
        agree += id == item + 1;
    }
    MESSAGE("points match: ", agree, " of 16 picks name the piece's item");

    // ---- The captures ---------------------------------------------------------------------------
    capture((std::string("points-match") + name + "-builtin.png").c_str(), builtin_color);
    capture((std::string("points-match") + name + "-piece.png").c_str(), piece_color);
    std::vector<std::uint8_t> diff(piece_color.size());
    for (std::size_t i = 0; i < diff.size(); ++i)
        diff[i] = (i % 4 == 3) ? 255 : std::uint8_t(std::min(255, std::abs(int(piece_color[i]) - int(builtin_color[i])) * 32));
    capture((std::string("points-match") + name + "-diff.png").c_str(), diff);

}

}  // namespace

TEST_CASE("points_match: points@1 draws as the built-in points primitive") { run_points_case(kPoints); }

TEST_CASE("points_match: u8 color rows (N, 4) bound as bytes") { run_points_case(kPoints, Values::u8_rows, "-u8-rows"); }

TEST_CASE("points_match: u32 packed RGBA") { run_points_case(kPoints, Values::u32_packed, "-u32-packed"); }

TEST_CASE("points_match: u16 class ids through the default class colors") {
    run_points_case(kPoints, Values::u16_classes, "-u16-classes");
}

TEST_CASE("points_match: the most points that the piece publishes") { run_points_case(kMostPoints); }

namespace {

// The eleven params of points@1 in the manifest's order with radius_mode 1.
std::vector<std::uint8_t> radii_params(float size, std::uint32_t shape, float radius_scale, float proj_x, float proj_y) {
    std::vector<std::uint8_t> out = points_params(size, shape, 1, 0xffffffffu, 0, 0.0f, 1.0f);
    out.resize(11 * 4);
    const std::uint32_t mode = 1;
    std::memcpy(out.data() + 28, &mode, 4);
    std::memcpy(out.data() + 32, &radius_scale, 4);
    std::memcpy(out.data() + 36, &proj_x, 4);
    std::memcpy(out.data() + 40, &proj_y, 4);
    return out;
}

const match_rig::Published& published(const std::vector<match_rig::Published>& all, const char* id) {
    for (const match_rig::Published& p : all)
        if (p.id == id) return p;
    FAIL("no published resource ", id);
    return all.front();
}

// position.w of row i of the published `points` rows (32 bytes each).
float row_radius(const match_rig::Published& p, std::uint32_t i) {
    float w = 0;
    std::memcpy(&w, p.bytes.data() + std::size_t(i) * 32 + 12, 4);
    return w;
}

}  // namespace

TEST_CASE("points_match: points@1 with radii in scene units draws as the built-in radius mode") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    gfx::Backend& g = *m.backend;
    const Cloud cloud;
    const int S = int(kSize);
    std::vector<float> radii(kPoints);
    // Every eighth point is far below half a pixel, so that the floor of the radius mode is drawn too.
    for (int i = 0; i < kPoints; ++i)
        radii[std::size_t(i)] = i % 8 == 0 ? 0.0005f : 0.01f + 0.05f * float((i * 7) % 16) / 15.0f;

    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_points_piece(&piece, &why), why);

    // ---- The derive half: radii per point, one for all, and unbound -------------------------------
    using match_rig::HostTensor;
    const HostTensor positions = HostTensor::of<float>("positions", "f32", {kPoints, 3}, cloud.positions);
    const HostTensor values = HostTensor::of<float>("values", "f32", {kPoints}, cloud.values);
    const std::vector<match_rig::Published> per_point = match_rig::run_derive(
        m.device, piece, {positions, values, HostTensor::of<float>("radii", "f32", {kPoints}, radii)}, {});
    const std::vector<match_rig::Published> one_for_all = match_rig::run_derive(
        m.device, piece, {positions, values, HostTensor::of<float>("radii", "f32", {1}, std::vector<float>{0.037f})}, {});
    const std::vector<match_rig::Published> unbound = match_rig::run_derive(m.device, piece, {positions, values}, {});
    const match_rig::Published& rows_per_point = published(per_point, "points");
    const match_rig::Published& rows_one = published(one_for_all, "points");
    const match_rig::Published& rows_unbound = published(unbound, "points");
    REQUIRE(rows_per_point.count == std::uint32_t(kPoints));
    REQUIRE(rows_one.count == std::uint32_t(kPoints));
    REQUIRE(rows_unbound.count == std::uint32_t(kPoints));
    REQUIRE(rows_per_point.bytes.size() == std::size_t(kPoints) * 32);
    std::size_t wrong = 0;
    for (std::uint32_t i = 0; i < std::uint32_t(kPoints); ++i) {
        wrong += row_radius(rows_per_point, i) != radii[i];
        wrong += row_radius(rows_one, i) != 0.037f;
        wrong += row_radius(rows_unbound, i) != 1.0f;
    }
    CHECK_MESSAGE(wrong == 0, wrong, " rows with a position.w other than their radius, the one radius or 1.0");

    const auto builtin_color = match_reference::picture("color", kSize, kSize, nullptr);
    const auto& reference = match_reference::metadata(match_reference::last_name("color"));
    const auto mvp = reference.at("mvp").get<std::array<float, 16>>();
    const float radius_scale = reference.at("radius_scale").get<float>();
    const auto projection = reference.at("projection").get<std::array<float, 2>>();
    const float proj_x = projection[0], proj_y = projection[1];

    // How many points sit at the half-pixel floor and how many above it, for the message.
    int at_floor = 0, above_floor = 0;
    for (int i = 0; i < kPoints; ++i) {
        const auto c = project(mvp.data(), cloud.positions[3 * i], cloud.positions[3 * i + 1], cloud.positions[3 * i + 2]);
        const double half_px = double(c.w) / double(kSize);
        (double(radii[std::size_t(i)]) * double(radius_scale) * double(proj_x) > half_px ? above_floor : at_floor)++;
    }
    MESSAGE("points radii match: ", at_floor, " points at the half-pixel floor, ", above_floor, " above it");
    CHECK(at_floor > 0);
    CHECK(above_floor > 0);

    // ---- The piece's draw, from the rows of the derive half --------------------------------------
    cd::DrawRuntime rt(g, false);
    cd::Instance instance;
    instance.program = "cloud";
    instance.panel = "cloud";
    const cd::Verdict installed = rt.install(piece, instance, kPiecePick);
    REQUIRE_MESSAGE(installed.ok(), installed.code << " " << installed.message);
    cd::ResourceVersion version;
    version.resource = "points";
    version.step = 1;
    version.count = rows_per_point.count;
    version.bytes = rows_per_point.bytes.data();
    version.size = rows_per_point.bytes.size();
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
    in.params = radii_params(kPointSize, 1, radius_scale, proj_x, proj_y);
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

    // ---- The comparison, under the rules of the case above ----------------------------------------
    std::size_t differing = 0, interior = 0, lit_interior = 0, edge = 0, failed = 0;
    std::string first;
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x) {
            const std::uint32_t pc = pixel_of(piece_color, x, y), bc = pixel_of(builtin_color, x, y);
            differing += pc != bc;
            bool ok;
            if (is_interior(piece_ids, x, y)) {
                ++interior;
                lit_interior += pc != kClear;
                ok = pc == bc;
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
    MESSAGE("points radii match: ", differing, " differing pixels, ", interior, " interior pixels (", lit_interior,
            " lit), ", edge, " edge pixels, ", failed, " failed");
    CHECK_MESSAGE(failed == 0, failed, " pixels outside the tolerances; ", first);
    CHECK(lit_interior >= 500);

    // ---- The picks ------------------------------------------------------------------------------
    const auto& probes = reference.at("picks");
    REQUIRE(probes.size() == 16);
    int agree = 0;
    for (const auto& probe : probes) {
        const int x = probe.at("x").get<int>(), y = probe.at("y").get<int>();
        const auto item = probe.at("item").get<std::uint64_t>();
        const auto id = identity_of(piece_ids, x, y);
        CHECK_MESSAGE(id == item + 1, "pixel (", x, ", ", y, "): saved item ", item, ", piece identity ", id);
        const auto picked = rt.pick(std::uint32_t(item));
        REQUIRE(picked.has_value());
        CHECK(picked->item == item);
        agree += id == item + 1;
    }
    MESSAGE("points radii match: ", agree, " of 16 picks name the piece's item");

    // ---- The captures ---------------------------------------------------------------------------
    capture("points-radii-builtin.png", builtin_color);
    capture("points-radii-piece.png", piece_color);
    std::vector<std::uint8_t> diff(piece_color.size());
    for (std::size_t i = 0; i < diff.size(); ++i)
        diff[i] = (i % 4 == 3) ? 255 : std::uint8_t(std::min(255, std::abs(int(piece_color[i]) - int(builtin_color[i])) * 32));
    capture("points-radii-diff.png", diff);

}

CALIPER_GPU_TEST_MAIN()
