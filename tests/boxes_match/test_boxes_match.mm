// The boxes piece derives and draws each fixture on Metal. Saved original images
// and pick identities define the reference. CPU scene helpers check label anchors.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#include "match_rig.h"
#include "scene/labels.h"

namespace {

using namespace match_rig;

constexpr std::uint32_t kSize = 256;
constexpr std::uint32_t kDefaultRgba = 0xffe6dcdcu;   // the scene's default color
constexpr std::uint32_t kCapacity = 126976;          // the boxes that the capacity case draws
constexpr std::uint32_t kPieceCapacity = 1048576;    // the boxes that the piece publishes

// One set of boxes with the built-in's options.
struct Boxes {
    int dims = 3;
    std::vector<float> centers, half;   // n * dims
    std::vector<float> rot;             // n * 4, or none
    std::vector<std::uint8_t> rgba;     // n * 4, or none
    std::vector<std::int32_t> classes;  // n, or none
    std::vector<std::string> labels;    // n, one for all, or none
    bool native_rgba = false;           // the colors reach the piece as u8 bytes, not as f32
    std::string class_dtype = "f32";    // the class ids reach the piece as "f32", "i32" or "u16"
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
void add_boxes(ReferenceScene& bt, const Boxes& b) {
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
    batch.labels = b.labels;
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

// The tensors of the piece's inputs and its params. An input that the boxes lack stays unbound.
// Each label is 16 u32 of its UTF-8 bytes, four in each element in order, zero after its end.
std::vector<HostTensor> piece_tensors(const Boxes& b, caliper_net::Json* params) {
    const std::int64_t n = b.n(), d = b.dims;
    std::vector<HostTensor> t;
    t.push_back(HostTensor::of("centers", "f32", {n, d}, b.centers));
    t.push_back(HostTensor::of("half_sizes", "f32", {n, d}, b.half));
    if (!b.rot.empty()) t.push_back(HostTensor::of("rotations", "f32", {n, 4}, b.rot));
    if (!b.labels.empty()) {
        std::vector<std::uint32_t> words(b.labels.size() * 16, 0u);
        for (std::size_t i = 0; i < b.labels.size(); ++i)
            std::memcpy(words.data() + i * 16, b.labels[i].data(), std::min<std::size_t>(b.labels[i].size(), 64));
        t.push_back(HostTensor::of("labels", "u32", {std::int64_t(b.labels.size()), 16}, words));
    }
    std::uint32_t mode = 0;
    if (!b.rgba.empty() && b.native_rgba) {
        t.push_back(HostTensor::of("colors", "u8", {n, 4}, b.rgba));
        mode = 1;
    } else if (!b.rgba.empty()) {
        std::vector<float> c;
        for (std::uint8_t v : b.rgba) c.push_back(float(v) / 255.0f);
        t.push_back(HostTensor::of("colors", "f32", {n, 4}, c));
        mode = 1;
    } else if (!b.classes.empty() && b.class_dtype == "i32") {
        t.push_back(HostTensor::of("colors", "i32", {n}, b.classes));
        mode = 2;
    } else if (!b.classes.empty() && b.class_dtype == "u16") {
        std::vector<std::uint16_t> c(b.classes.begin(), b.classes.end());
        t.push_back(HostTensor::of("colors", "u16", {n}, c));
        mode = 2;
    } else if (!b.classes.empty()) {
        std::vector<float> c(b.classes.begin(), b.classes.end());
        t.push_back(HostTensor::of("colors", "f32", {n}, c));
        mode = 2;
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
    std::size_t labels = 0;        // the built-in's labels, each matched by one of the piece
};

// The piece's labels at pixels, as the scene view projects a program's anchors (live_views_scene.cpp).
std::vector<sc::LabelAnchor> piece_labels(const cd::DrawRuntime& rt, const cd::FrameInputs& in, std::uint32_t size) {
    std::vector<sc::LabelAnchor> out;
    for (const cd::Label& l : rt.labels()) {
        double h[4];
        for (int r = 0; r < 4; ++r)
            h[r] = double(in.clip_from_world[std::size_t(r)]) * l.position[0] +
                   double(in.clip_from_world[std::size_t(4 + r)]) * l.position[1] +
                   double(in.clip_from_world[std::size_t(8 + r)]) * l.position[2] + double(in.clip_from_world[std::size_t(12 + r)]);
        if (!(h[3] > 0)) continue;
        const double x = h[0] / h[3], y = h[1] / h[3];
        if (std::fabs(x) > 1 || std::fabs(y) > 1) continue;
        sc::LabelAnchor a;
        a.x = float((x + 1) * 0.5 * size);
        a.y = float((1 - y) * 0.5 * size);
        a.text = l.text;
        out.push_back(a);
    }
    return out;
}

const Published& find(const std::vector<Published>& res, const char* id) {
    for (const Published& p : res)
        if (p.id == id) return p;
    FAIL("no published resource ", id);
    return res.front();
}

// Draws the boxes both ways, compares, picks `items`, and writes the captures of the case.
Outcome run(Metal& m, const std::string& name, const Boxes& b, std::vector<std::uint64_t> items,
            bool same_labels = true) {
    gfx::Backend& g = *m.backend;
    Outcome o;
    ReferenceScene bt(g, kSize, b.dims == 3 ? sc::Space::k3D : sc::Space::k2D);
    add_boxes(bt, b);
    bt.load();


    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("boxes", &piece, &why), why);
    caliper_net::Json params;
    const std::vector<HostTensor> tensors = piece_tensors(b, &params);
    const std::vector<Published> all = run_derive(m.device, piece, tensors, as_map(params));
    REQUIRE(all.size() == 2);
    const std::vector<Published> res{find(all, "boxes")};
    const Published& names = find(all, "names");
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
    pd.set("names", names.bytes, names.count);
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

    // The labels: each of the built-in's, at the center of its box, has one of the piece's with the
    // same text within a tenth of a pixel, and the piece has no other.
    const std::vector<sc::LabelAnchor> want = sc::project_labels(bt.tree, bt.camera, kSize, kSize);
    std::vector<sc::LabelAnchor> got = piece_labels(pd.rt, in, kSize);
    CHECK(got.size() == want.size());
    if (!same_labels) {   // more than 512 labels: the built-in keeps the nearest, the piece the first
        o.labels = std::min(got.size(), want.size());
        MESSAGE("boxes ", name, ": ", got.size(), " labels of the piece and ", want.size(), " of the built-in");
        return o;
    }
    for (const sc::LabelAnchor& w : want) {
        const auto it = std::find_if(got.begin(), got.end(), [&](const sc::LabelAnchor& g) {
            return g.text == w.text && std::fabs(g.x - w.x) <= 0.1f && std::fabs(g.y - w.y) <= 0.1f;
        });
        CHECK_MESSAGE(it != got.end(), name, ": the label ", w.text, " at ", w.x, ", ", w.y, " has no match");
        if (it != got.end()) {
            got.erase(it);
            ++o.labels;
        }
    }
    if (!want.empty()) MESSAGE("boxes ", name, ": ", o.labels, " of ", want.size(), " labels match, ", names.count, " strings published");

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

TEST_CASE("boxes_match: u8 colors of each box (N, 4) bound as bytes") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    Boxes b = scatter(16, 3, 4242u, false);
    color_boxes(b, 120);
    b.native_rgba = true;
    b.fill = true;
    b.width = 5.0f;
    const Outcome o = run(m, "u8_colors", b, {0, 7, 15});
    CHECK(o.c.lit_interior >= 2000);
    CHECK(o.picks == o.asked);
}

TEST_CASE("boxes_match: class ids of i32 and of u16 through the palette") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    for (const char* type : {"i32", "u16"}) {
        Boxes b = scatter(20, 3, 31337u, false);
        for (int i = 0; i < b.n(); ++i) b.classes.push_back(i == 3 ? 65535 : i == 4 ? 4096 : i * 5 + 1);
        if (std::string(type) == "i32") b.classes[3] = -1;
        b.class_dtype = type;
        b.fill = true;
        b.width = 1.0f;
        const Outcome o = run(m, std::string("classes_") + type, b, {0, 19});
        CHECK(o.c.lit_interior >= 2000);
        CHECK(o.picks == o.asked);
    }
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
    ReferenceScene bt(g, kSize, sc::Space::k3D);
    bt.load();
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

TEST_CASE("boxes_match: labels of each box, one label for all, and labels of every step-th box of a large set") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    Boxes b = scatter(40, 3, 777u, true);
    for (int i = 0; i < b.n(); ++i) b.labels.push_back(i % 5 == 4 ? std::string() : "box " + std::to_string(i) + " \xC3\xA9");
    const Outcome o = run(m, "labels", b, {0});
    CHECK(o.labels > 20);
    Boxes one = scatter(12, 2, 99u, false);
    one.labels = {"all"};
    CHECK(run(m, "labels_one", one, {}).labels == 12);
    // 1,200 labels of 1,200 boxes in 2-D, and one label for 9,000 boxes, which the built-in and the
    // piece give every third box, 3,000; of those both keep 512, the built-in the nearest and the
    // piece the first, thus only the count is held here.
    Boxes each;
    each.dims = 2;
    for (int i = 0; i < 1200; ++i) {
        each.centers.insert(each.centers.end(), {float(i % 40) / 20.0f - 0.975f, float(i / 40) / 15.0f - 0.97f});
        each.half.insert(each.half.end(), {0.005f, 0.005f});
        each.labels.push_back(i % 3 == 0 ? "e" + std::to_string(i) : std::string());
    }
    CHECK(run(m, "labels_each", each, {}).labels == 400);
    Boxes many;
    many.dims = 2;
    for (int i = 0; i < 9000; ++i) {
        many.centers.insert(many.centers.end(), {float(i % 90) / 45.0f - 0.99f, float(i / 90) / 50.0f - 0.99f});
        many.half.insert(many.half.end(), {0.002f, 0.002f});
    }
    many.labels = {"m"};
    const Outcome big = run(m, "labels_step", many, {}, false);
    CHECK(big.labels == 512);
}

TEST_CASE("boxes_match: the capacity of 126,976 boxes, and the first boxes of a larger set") {
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

    // One box more than the capacity of the piece, 1,048,576: the piece publishes the first 1,048,576.
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("boxes", &piece, &why), why);
    caliper_net::Json params;
    const Boxes over = grid(int(kPieceCapacity) + 1);
    const std::vector<Published> res = run_derive(m.device, piece, piece_tensors(over, &params), as_map(params));
    REQUIRE(res.size() == 2);
    CHECK(find(res, "boxes").count == kPieceCapacity);
    CHECK(find(res, "boxes").bytes.size() == std::size_t(kPieceCapacity) * 64);
    CHECK(find(res, "names").count == 0);
}

// Optional capacity check of the derive output. The default suite uses bounded image fixtures.
TEST_CASE("boxes_match: boxes@1 at its capacity" * doctest::skip()) {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    const int n = int(kPieceCapacity) / capacity_fraction();
    Boxes b;
    b.centers.reserve(std::size_t(n) * 3);
    b.half.reserve(std::size_t(n) * 3);
    for (int i = 0; i < n; ++i) {
        const int x = i % 64, y = (i / 64) % 64, z = i / 4096;
        b.centers.insert(b.centers.end(), {float(x) / 32.0f - 1.0f, float(y) / 32.0f - 1.0f, float(z % 32) / 16.0f - 1.0f});
        b.half.insert(b.half.end(), {0.006f, 0.006f, 0.006f});
    }
    for (int i : {0, n - 1}) {   // the first and the last box large and in front
        b.centers[std::size_t(3 * i)] = i == 0 ? -0.5f : 0.5f;
        b.centers[std::size_t(3 * i + 1)] = 0.0f;
        b.centers[std::size_t(3 * i + 2)] = 1.3f;
        b.half[std::size_t(3 * i)] = b.half[std::size_t(3 * i + 1)] = b.half[std::size_t(3 * i + 2)] = 0.2f;
    }
    b.fill = true;
    b.width = 1.0f;
    color_boxes(b, 255);
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("boxes", &piece, &why), why);
    caliper_net::Json params;
    const auto tensors = piece_tensors(b, &params);
    const auto output = run_derive(m.device, piece, tensors, as_map(params));
    CHECK(find(output, "boxes").count == std::uint32_t(n));
    CHECK(find(output, "boxes").bytes.size() == std::size_t(n) * 64);
}

CALIPER_GPU_TEST_MAIN()

