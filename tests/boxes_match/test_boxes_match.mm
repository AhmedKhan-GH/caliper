// test_boxes_match.mm: the match of the library piece boxes@1 with the built-in boxes primitive
// (design/programmable-graphics.md, Library Pieces).
//
// Each case draws one set of boxes twice on Metal, offscreen and without a window: once by the scene
// core as a boxes batch (SceneCompiler over BackendResources), and once by the piece: its derive
// half runs on the same boxes as tensors in the script's producer runtime, and its draw half draws
// the published resource in a DrawRuntime with the camera that the scene view gives a program. The
// two images are held to the tolerances of match_rig.h, and the picks of the built-in draw at
// interior pixels of the first, a middle and the last box must name the item that the piece's ID
// image shows there, with the field `index` of its item map. The cases cover the color modes (one
// color, RGBA of each box, class ids with ids outside the table), widths and the pixel ratio,
// faces, translucent boxes (an alpha below 255, of one color or of a box), rotations, the depth
// offset, boxes in 2-D, a box with a NaN center, empty input, one box, and the capacity of the
// piece, 131,072 boxes of 64 bytes in its 8 MiB of published resources (the built-in takes
// 1,048,576). Without a Metal device the test skips (exit 77).
//
// With CALIPER_BOXES_MATCH_CAPTURE_DIR set, each case writes boxes-<case>-builtin.png,
// boxes-<case>-piece.png and boxes-<case>-diff.png (the difference scaled up 32 times) there.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#include "match_rig.h"

namespace {

using namespace match_rig;

constexpr std::uint32_t kSize = 256;
constexpr std::uint32_t kDefaultRgba = 0xffe6dcdcu;   // the scene's default color
constexpr std::uint32_t kCapacity = 131072;

// One set of boxes with the built-in's options.
struct Boxes {
    int dims = 3;
    std::vector<float> centers, half;   // n * dims
    std::vector<float> rot;             // n * 4, or none
    std::vector<std::uint8_t> rgba;     // n * 4, or none
    std::vector<std::int32_t> classes;  // n, or none
    std::uint32_t flat = kDefaultRgba;
    float width = 2.0f;
    bool fill = false;
    int depth_offset = 0;
    float pixel_ratio = 1.0f;
    int n() const { return int(centers.size()) / dims; }
};

// A fixed LCG in [-1, 1).
struct Lcg {
    std::uint32_t state;
    float next() {
        state = state * 1664525u + 1013904223u;
        return float(double(state >> 8) / double(1u << 24)) * 2.0f - 1.0f;
    }
};

// n boxes spread over [-1, 1]^dims with half sizes from 0.06 to 0.2, and with rotations of every
// third box about a random axis when `rotate`.
Boxes scatter(int n, int dims, std::uint32_t seed, bool rotate) {
    Boxes b;
    b.dims = dims;
    Lcg r{seed};
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < dims; ++k) b.centers.push_back(r.next());
        for (int k = 0; k < dims; ++k) b.half.push_back(0.13f + 0.07f * r.next());
        if (rotate) {
            float q[4] = {r.next(), r.next(), r.next(), r.next()};
            if (i % 3 != 0) q[0] = q[1] = q[2] = 0, q[3] = 1;
            b.rot.insert(b.rot.end(), q, q + 4);
        }
    }
    return b;
}

// RGBA of each box at whole levels, with an alpha of `alpha` on every fourth box.
void color_boxes(Boxes& b, std::uint8_t alpha) {
    for (int i = 0; i < b.n(); ++i) {
        b.rgba.push_back(std::uint8_t((i * 53 + 40) % 256));
        b.rgba.push_back(std::uint8_t((i * 97 + 90) % 256));
        b.rgba.push_back(std::uint8_t((i * 31 + 200) % 256));
        b.rgba.push_back(i % 4 == 3 ? alpha : 255);
    }
}

// The built-in's batch of the boxes, as live_scene.cpp makes it from a SCENE_UPDATE part.
void builtin_boxes(Builtin& bt, const Boxes& b) {
    const int n = b.n();
    if (n == 0) return;   // an empty part removes the batch
    sc::Batch batch;
    batch.primitive = sc::Primitive::Boxes;
    batch.width_px = b.width;
    batch.fill = b.fill;
    batch.flat_rgba = b.flat;
    batch.depth_offset = std::int8_t(b.depth_offset);
    batch.translucent = (b.flat >> 24) < 255;
    for (int i = 3; i < int(b.rgba.size()); i += 4) batch.translucent = batch.translucent || b.rgba[std::size_t(i)] < 255;
    const gfx::VertexFormat f = b.dims == 3 ? gfx::VertexFormat::F32x3 : gfx::VertexFormat::F32x2;
    Upload up;
    up.add(sc::Role::Centers, b.centers, std::uint32_t(n), std::uint32_t(b.dims * 4), f);
    up.add(sc::Role::HalfSizes, b.half, std::uint32_t(n), std::uint32_t(b.dims * 4), f);
    if (!b.rot.empty()) up.add(sc::Role::Rotations, b.rot, std::uint32_t(n), 16, gfx::VertexFormat::F32x4);
    if (!b.rgba.empty()) up.add(sc::Role::Colors, b.rgba, std::uint32_t(n), 4, gfx::VertexFormat::Unorm8x4);
    if (!b.classes.empty()) up.add(sc::Role::ClassIds, b.classes, std::uint32_t(n), 4, gfx::VertexFormat::S32);
    for (int i = 0; i < n; ++i) {
        double c[3] = {0, 0, 0}, h[3] = {0, 0, 0};
        for (int k = 0; k < b.dims; ++k) {
            c[k] = b.centers[std::size_t(i * b.dims + k)];
            h[k] = std::fabs(b.half[std::size_t(i * b.dims + k)]);
        }
        if (!b.rot.empty()) h[0] = h[1] = h[2] = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
        if (!std::isfinite(c[0] + c[1] + c[2])) continue;
        batch.bounds.add({c[0] - h[0], c[1] - h[1], c[2] - h[2]});
        batch.bounds.add({c[0] + h[0], c[1] + h[1], c[2] + h[2]});
    }
    bt.set("boxes", batch, up);
}

// The tensors of the piece's inputs and its params. A tensor that the boxes lack binds another of
// the same rank, as a program binds an input that its mode does not read.
std::vector<HostTensor> piece_tensors(const Boxes& b, caliper_net::Json* params) {
    const std::int64_t n = b.n(), d = b.dims;
    std::vector<HostTensor> t;
    t.push_back(HostTensor::of("centers", "f32", {n, d}, b.centers));
    t.push_back(HostTensor::of("half_sizes", "f32", {n, d}, b.half));
    if (!b.rot.empty()) t.push_back(HostTensor::of("rotations", "f32", {n, 4}, b.rot));
    else t.push_back(HostTensor::of("rotations", "f32", {n, d}, b.centers));
    std::uint32_t mode = 0;
    if (!b.rgba.empty()) {
        std::vector<float> c;
        for (std::uint8_t v : b.rgba) c.push_back(float(v) / 255.0f);
        t.push_back(HostTensor::of("colors", "f32", {n, 4}, c));
        mode = 1;
    } else if (!b.classes.empty()) {
        std::vector<float> c(b.classes.begin(), b.classes.end());
        t.push_back(HostTensor::of("colors", "f32", {n}, c));
        mode = 2;
    } else {
        t.push_back(HostTensor::of("colors", "f32", {n, d}, b.half));
    }
    *params = caliper_net::Json::object();
    params->set("width", double(b.width));
    params->set("fill", b.fill ? 1.0 : 0.0);
    params->set("color_mode", double(mode));
    params->set("color", double(b.flat));
    params->set("depth_offset", double(b.depth_offset));
    return t;
}

std::map<std::string, double> as_map(const caliper_net::Json& j) {
    std::map<std::string, double> m;
    for (const char* k : {"width", "fill", "color_mode", "color", "depth_offset"})
        if (const caliper_net::Json* v = j.find(k)) m[k] = v->as_double();
    return m;
}

struct Outcome {
    Comparison c;
    std::uint32_t published = 0;   // the valid boxes of the piece's resource
    int picks = 0, asked = 0;
};

// Draws the boxes both ways, compares, picks `items`, and writes the captures of the case.
Outcome run(Metal& m, const std::string& name, const Boxes& b, std::vector<std::uint64_t> items) {
    gfx::Backend& g = *m.backend;
    Outcome o;
    Builtin bt(g, kSize, b.dims == 3 ? sc::Space::k3D : sc::Space::k2D);
    bt.options.pixel_scale = b.pixel_ratio;
    builtin_boxes(bt, b);
    bt.draw();
    if (b.n() > 0) {
        REQUIRE(!bt.draws.empty());
        CHECK(bt.draws[0].p.color_mode == (!b.rgba.empty() ? 2u : !b.classes.empty() ? 4u : 0u));
        CHECK(bt.draws.size() == (b.fill ? 2u : 1u));
    }

    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("boxes", &piece, &why), why);
    caliper_net::Json params;
    const std::vector<HostTensor> tensors = piece_tensors(b, &params);
    const std::vector<Published> res = run_derive(m.device, piece, tensors, as_map(params));
    REQUIRE(res.size() == 1);
    o.published = res[0].count;
    if (res[0].count > 0 && !b.rgba.empty()) {
        // Box 0 holds the flags of all boxes: bit 0 when the RGBA of a box has an alpha below 255,
        // which only the mode of RGBA reads.
        float flags = 0;
        std::memcpy(&flags, res[0].bytes.data() + 28, 4);
        bool alpha = false;
        for (std::size_t i = 3; i < b.rgba.size(); i += 4) alpha = alpha || b.rgba[i] < 255;
        CHECK(flags == (alpha ? 1.0f : 0.0f));
    }

    PieceDraw pd(g, kSize);
    pd.install(piece, "boxes");
    if (res[0].count > 0) pd.set("boxes", res[0].bytes, res[0].count);
    cd::FrameInputs in = bt.frame_inputs();
    in.pixel_ratio = b.pixel_ratio;
    in.params = caliper_web::pack_params(piece, params);
    const bool drawn = pd.draw(in);
    CHECK(drawn == (res[0].count > 0));

    o.c = compare(pd.color, pd.ids, bt.image, kSize);
    MESSAGE("boxes ", name, ": ", b.n(), " boxes, ", o.published, " published; ", o.c.interior, " interior pixels (",
            o.c.lit_interior, " lit), ", o.c.edge, " edge pixels; ", o.c.differ, " differ, largest gap ",
            o.c.max_interior, " inside and ", o.c.max_edge, " at edges; ", o.c.failed, " outside the tolerances");
    CHECK_MESSAGE(o.c.failed == 0, name, ": ", o.c.failed, " pixels outside the tolerances; ", o.c.first);
    for (std::uint64_t it : items) {
        if (it >= std::uint64_t(b.n())) continue;
        ++o.asked;
        o.picks += check_picks(bt, pd, {it}, sc::Primitive::Boxes);
    }
    if (o.asked) MESSAGE("boxes ", name, ": ", o.picks, " of ", o.asked, " picks name the piece's item and field");

    capture("CALIPER_BOXES_MATCH_CAPTURE_DIR", "boxes-" + name + "-builtin.png", kSize, bt.image);
    capture("CALIPER_BOXES_MATCH_CAPTURE_DIR", "boxes-" + name + "-piece.png", kSize, pd.color);
    capture("CALIPER_BOXES_MATCH_CAPTURE_DIR", "boxes-" + name + "-diff.png", kSize, difference(pd.color, bt.image));
    return o;
}

}  // namespace

TEST_CASE("boxes_match: boxes@1 passes the checks of piece.check") {
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("boxes", &piece, &why), why);
    check_piece(piece);
}

TEST_CASE("boxes_match: wire boxes in 3-D with RGBA, rotations and a NaN center") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    Boxes b = scatter(24, 3, 12345u, true);
    color_boxes(b, 255);
    b.width = 4.0f;
    b.centers[5 * 3] = std::nanf("");
    const Outcome o = run(m, "wire", b, {0, 11, 23});
    CHECK(o.published == 24);
    CHECK(o.c.lit_interior >= 500);
    CHECK(o.picks == o.asked);
    CHECK(o.asked == 3);
}

TEST_CASE("boxes_match: filled boxes of one color, faces at a quarter of the alpha") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    Boxes b = scatter(12, 3, 777u, true);
    b.flat = 0xff3c8ce6u;
    b.fill = true;
    b.width = 3.0f;
    const Outcome o = run(m, "fill", b, {0, 5, 11});
    CHECK(o.c.lit_interior >= 2000);
    CHECK(o.picks == o.asked);
    CHECK(o.asked == 3);
}

TEST_CASE("boxes_match: translucent boxes by the alpha of a box, with faces") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    Boxes b = scatter(16, 3, 4242u, false);
    color_boxes(b, 120);
    b.fill = true;
    b.width = 5.0f;
    const Outcome o = run(m, "translucent", b, {0, 7, 15});
    CHECK(o.c.lit_interior >= 2000);
    CHECK(o.picks == o.asked);
}

TEST_CASE("boxes_match: one color with an alpha below 255, edges only") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    Boxes b = scatter(16, 3, 99u, true);
    b.flat = 0x80ffc040u;
    b.width = 6.0f;
    const Outcome o = run(m, "flat_alpha", b, {0, 15});
    CHECK(o.c.lit_interior >= 500);
    CHECK(o.picks == o.asked);
}

TEST_CASE("boxes_match: class ids through the palette, with ids outside the table") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    Boxes b = scatter(20, 3, 31337u, false);
    for (int i = 0; i < b.n(); ++i) b.classes.push_back(i == 3 ? -1 : i == 4 ? 70000 : i * 5 + 1);
    b.fill = true;
    b.width = 1.0f;
    const Outcome o = run(m, "classes", b, {0, 19});
    CHECK(o.c.lit_interior >= 2000);
    CHECK(o.picks == o.asked);
}

TEST_CASE("boxes_match: wide edges at a pixel ratio of 2, and the depth offset of the edges") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    Boxes b = scatter(10, 3, 2024u, true);
    color_boxes(b, 255);
    b.width = 5.0f;
    b.pixel_ratio = 2.0f;
    b.fill = true;
    b.depth_offset = 6;
    const Outcome o = run(m, "wide_offset", b, {0, 9});
    CHECK(o.c.lit_interior >= 2000);
    CHECK(o.picks == o.asked);
}

TEST_CASE("boxes_match: boxes in 2-D, filled, with RGBA") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    Boxes b = scatter(20, 2, 5150u, false);
    color_boxes(b, 255);
    b.fill = true;
    b.width = 3.0f;
    const Outcome o = run(m, "2d", b, {0, 10, 19});
    CHECK(o.c.lit_interior >= 2000);
    CHECK(o.picks == o.asked);
}

TEST_CASE("boxes_match: one box") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    Boxes one;
    one.centers = {0.1f, -0.2f, 0.3f};
    one.half = {0.5f, 0.3f, 0.4f};
    one.fill = true;
    one.width = 4.0f;
    const Outcome o = run(m, "one", one, {0});
    CHECK(o.published == 1);
    CHECK(o.picks == 1);
}

TEST_CASE("boxes_match: no box draws nothing") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    gfx::Backend& g = *m.backend;
    // The script's runtime imports no tensor without bytes (producer.mm refuses it), thus a viewer
    // gets no version of the boxes, or a version without a valid element. The piece then draws
    // nothing, as the built-in draws nothing for an empty part, which removes the batch.
    Builtin bt(g, kSize, sc::Space::k3D);
    bt.draw();
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("boxes", &piece, &why), why);
    PieceDraw pd(g, kSize);
    pd.install(piece, "boxes");
    cd::FrameInputs in;
    in.clip_from_world = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    in.viewport = {0, 0, float(kSize), float(kSize)};
    in.params = caliper_web::pack_params(piece, caliper_net::Json::object());
    CHECK_FALSE(pd.draw(in));
    CHECK(compare(pd.color, pd.ids, bt.image, kSize).differ == 0);
    pd.set("boxes", {}, 0);
    CHECK_FALSE(pd.draw(in));
    const Comparison c = compare(pd.color, pd.ids, bt.image, kSize);
    CHECK(c.differ == 0);
    MESSAGE("boxes empty: ", c.differ, " pixels differ from the built-in's empty scene");
}

TEST_CASE("boxes_match: the capacity of 131,072 boxes, and the first boxes of a larger set") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    // A grid of 64 x 64 x 32 small boxes, with the first and the last boxes large in front.
    auto grid = [](int n) {
        Boxes b;
        for (int i = 0; i < n; ++i) {
            const int x = i % 64, y = (i / 64) % 64, z = i / 4096;
            b.centers.insert(b.centers.end(), {float(x) / 32.0f - 1.0f, float(y) / 32.0f - 1.0f, float(z % 32) / 16.0f - 1.0f});
            b.half.insert(b.half.end(), {0.006f, 0.006f, 0.006f});
        }
        for (int i : {0, n - 1}) {
            b.centers[std::size_t(3 * i)] = i == 0 ? -0.5f : 0.5f;
            b.centers[std::size_t(3 * i + 1)] = 0.0f;
            b.centers[std::size_t(3 * i + 2)] = 1.3f;
            b.half[std::size_t(3 * i)] = b.half[std::size_t(3 * i + 1)] = b.half[std::size_t(3 * i + 2)] = 0.2f;
        }
        b.fill = true;
        b.width = 1.0f;
        color_boxes(b, 255);
        return b;
    };
    const Boxes full = grid(int(kCapacity));
    const Outcome o = run(m, "capacity", full, {0, kCapacity - 1});
    CHECK(o.published == kCapacity);
    CHECK(o.picks == o.asked);
    CHECK(o.asked == 2);

    // One box more than the capacity: the piece publishes the first 131,072.
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("boxes", &piece, &why), why);
    caliper_net::Json params;
    const Boxes over = grid(int(kCapacity) + 1);
    const std::vector<Published> res = run_derive(m.device, piece, piece_tensors(over, &params), as_map(params));
    REQUIRE(res.size() == 1);
    CHECK(res[0].count == kCapacity);
    CHECK(res[0].bytes.size() == std::size_t(kCapacity) * 64);
}

CALIPER_GPU_TEST_MAIN()

