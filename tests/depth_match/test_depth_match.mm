// test_depth_match.mm: the match of the library piece depth@1 with the built-in depth image of the
// scene view (design/programmable-graphics.md, Library Pieces), through
// tests/image_match/image_match_rig.h. The cases cover each option of a depth map: f32 depths
// with NaN, zero, negative and infinite values through a fixed range, u16 depths in hundredths
// through the range of the data in another colormap, a range from the data with one fixed end, a
// flat one and an empty one, the image plane of a camera in 3-D with the back-projected cloud,
// one pixel, a minified map, invalid dimensions and an empty map. Without a Metal device the test skips (exit 77).
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "../image_match/image_match_rig.h"

namespace {

using namespace image_match;
using caliper_net::Json;
using pf::RGBA;


// The built-in's params of a depth image.
struct DepthParams {
    double scale = 1.0;
    int colormap = 0;
    std::uint32_t flags = 0;   // CALIPER_NET_SNAP_AUTO_VMIN | CALIPER_NET_SNAP_AUTO_VMAX
    double vmin = 0.0, vmax = 1.0;
    std::optional<std::pair<double, double>> stats;   // absent with an AUTO flag: no valid depth
    std::optional<bool> cloud;
    std::optional<double> opacity;
};

void set_depth(wire::SceneUpdate& u, const DepthParams& p) {
    u.params.set("form", Json(CALIPER_NET_FORM_DEPTH));
    u.params.set("scale", Json(p.scale));
    u.params.set("colormap", Json(p.colormap));
    u.params.set("flags", Json(std::uint64_t(p.flags)));
    u.params.set("vmin", Json(p.vmin));
    u.params.set("vmax", Json(p.vmax));
    if (p.flags) {
        Json s = Json::array();
        if (p.stats) {
            s.push(Json(p.stats->first));
            s.push(Json(p.stats->second));
        } else {
            s.push(Json());
            s.push(Json());
        }
        u.params.set("stats", s);
    }
    if (p.cloud) u.params.set("cloud", Json(*p.cloud));
    if (p.opacity) u.params.set("opacity", Json(*p.opacity));
}

// A 2-D scene of one depth map of f32 or u16 values.
template <class T>
pf::Fixture flat_scene(const std::string& name, const std::vector<T>& d, std::uint32_t w, std::uint32_t h,
                       const DepthParams& p) {
    pf::Fixture f;
    f.name = name;
    f.info = {CALIPER_NET_SPACE_2D, "-y", "px"};
    wire::SceneUpdate u = f.update("depth", CALIPER_NET_PART_IMAGE);
    set_depth(u, p);
    const std::int64_t dtype = sizeof(T) == 2 ? CALIPER_NET_DTYPE_U16 : CALIPER_NET_DTYPE_F32;
    u.arrays.push_back(f.section("pixels", dtype, {std::int64_t(h), std::int64_t(w)}, d));
    f.messages.push_back(u);
    return f;
}

// The range of the data as the producer's stats give it: the least and the greatest depth in scene
// units that is finite and not zero, or nothing.
std::optional<std::pair<double, double>> stats_of(const std::vector<float>& v, float scale) {
    std::optional<std::pair<double, double>> out;
    for (float raw : v) {
        const float x = raw * scale;
        if (!std::isfinite(x) || x == 0.0f) continue;
        if (!out) out = std::make_pair(double(x), double(x));
        out->first = std::min(out->first, double(x));
        out->second = std::max(out->second, double(x));
    }
    return out;
}

// The room with its camera at a pose, and the depth map under the camera unless `with_depth` is
// false.
pf::Fixture camera_scene(const std::vector<float>& pose, const std::vector<float>& d, std::uint32_t w,
                         std::uint32_t h, const DepthParams& p, bool with_depth) {
    pf::Fixture f = camera_room(pose);
    if (with_depth) {
        wire::SceneUpdate u = f.update("rig/cam/depth", CALIPER_NET_PART_IMAGE);
        set_depth(u, p);
        u.arrays.push_back(f.section("pixels", CALIPER_NET_DTYPE_F32, {std::int64_t(h), std::int64_t(w)}, d));
        f.messages.push_back(u);
    }
    return f;
}

struct Fixture {
    Metal m;
    cs::Piece piece;
    bool ready = false;
    Fixture() {
        if (!m.backend) return;
        std::string why;
        REQUIRE_MESSAGE(load_piece("depth", &piece, &why), why);
        ready = true;
    }
};

// The level L of a range's colormap at its center: vmin + (vmax - vmin) L / 255.
float at_level(float vmin, float vmax, int level) { return vmin + (vmax - vmin) * float(level) / 255.0f; }

}  // namespace

TEST_CASE("depth_match: depth@1 passes piece.check, and each of its passes composes and compiles") {
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("depth", &piece, &why), why);
    const Checked c = check_piece(piece);
    CHECK_MESSAGE(c.ok, c.messages);
    CHECK(c.passes == 4);   // two derive passes and two draw passes
    std::uint64_t bytes = 0;
    std::size_t textures = 0;
    for (const cs::Resource& r : piece.resources) {
        if (r.kind == cs::ResourceKind::texture) {
            ++textures;
            CHECK(r.id == "pixels");
            CHECK(r.width == 8192);
            CHECK(r.height == 8192);
            CHECK(cs::texture_texels(r) == 67108864ull);
            CHECK(cs::texture_bytes(r) == (256ull << 20));
            if (r.publish) bytes += cs::texture_bytes(r);
        } else {
            cp::ElementLayout l;
            REQUIRE(cp::element_layout(r.layout, &l).ok());
            CHECK(std::uint64_t(r.count) * l.stride <= (128ull << 20));
            if (r.publish) bytes += std::uint64_t(r.count) * l.stride;
        }
    }
    CHECK(textures == 1);
    CHECK(bytes > (256ull << 20));
    CHECK(bytes <= (257ull << 20));
}

TEST_CASE("depth_match: depth@1 draws each depth map as the built-in depth image draws it") {
    Fixture fx;
    if (!fx.ready) CALIPER_SKIP_NO_GPU(fx.m.why);

    SUBCASE("f32 flat in 2-D through a fixed range, with NaN, zero, negative and infinite depths") {
        const std::uint32_t w = 16, h = 12;
        std::vector<float> d(std::size_t(w) * h);
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x)
                d[std::size_t(y) * w + x] = at_level(0.5f, 2.5f, int((x * 16 + y * 5) % 256));
        d[0] = std::nanf("");
        d[17] = 0.0f;
        d[40] = -1.0f;
        d[55] = INFINITY;
        d[70] = -INFINITY;
        DepthParams p;
        p.colormap = 0;
        p.vmin = 0.5;
        p.vmax = 2.5;
        Case k;
        k.name = case_name("depth", "f32-fixed");
        k.scene = flat_scene("f32", d, w, h, p);
        k.tensors = {tensor("depth", {std::int64_t(h), std::int64_t(w)}, d)};
        k.params = {{"vmin", 0.5}, {"vmax", 2.5}, {"auto_range", 0}};
        k.width = w;
        k.height = h;
        k.field = [d](std::uint32_t i) { return double(d[i]); };
        const Outcome o = run_case(fx.m, fx.piece, k);
        CHECK(o.first);
        CHECK(o.last);
    }

    SUBCASE("u16 in hundredths through the range of the data, in magma, at an opacity of 0.8") {
        const std::uint32_t w = 16, h = 12;
        std::vector<std::uint16_t> raw(std::size_t(w) * h);
        std::vector<float> values(raw.size());
        for (std::uint32_t i = 0; i < raw.size(); ++i) {
            raw[i] = std::uint16_t(50 + (i * 37) % 256);
            if (i % 29 == 5) raw[i] = 0;   // holes, which the range leaves out
            values[i] = float(raw[i]);
        }
        raw[1] = 50;    // the ends of the range: 0.5 and 3.05
        raw[2] = 305;
        values[1] = 50.0f;
        values[2] = 305.0f;
        DepthParams p;
        p.scale = 0.01;
        p.colormap = 1;
        p.flags = CALIPER_NET_SNAP_AUTO_VMIN | CALIPER_NET_SNAP_AUTO_VMAX;
        p.stats = stats_of(values, 0.01f);
        p.opacity = 0.8;
        Case k;
        k.name = case_name("depth", "u16-auto");
        k.scene = flat_scene("u16", raw, w, h, p);
        k.tensors = {tensor("depth", {std::int64_t(h), std::int64_t(w)}, values)};
        k.params = {{"scale", 0.01}, {"colormap", 1}, {"auto_range", 3}, {"opacity", 0.8}};
        k.width = w;
        k.height = h;
        k.field = [values](std::uint32_t i) { return double(values[i] * 0.01f); };
        run_case(fx.m, fx.piece, k);
    }

    SUBCASE("u16 in millimeters bound as u16, through a fixed range at a scale of 0.001") {
        const std::uint32_t w = 16, h = 12;
        std::vector<std::uint16_t> raw(std::size_t(w) * h);
        for (std::uint32_t i = 0; i < raw.size(); ++i) {
            raw[i] = std::uint16_t(500 + (i * 97) % 2000);
            if (i % 31 == 7) raw[i] = 0;   // holes
        }
        DepthParams p;
        p.scale = 0.001;
        p.colormap = 0;
        p.vmin = 0.5;
        p.vmax = 2.5;
        Case k;
        k.name = case_name("depth", "u16-native");
        k.scene = flat_scene("u16n", raw, w, h, p);
        k.tensors = {tensor("depth", {std::int64_t(h), std::int64_t(w)}, raw, CALIPER_PRODUCER_U16)};
        k.params = {{"scale", 0.001}, {"vmin", 0.5}, {"vmax", 2.5}, {"auto_range", 0}};
        k.width = w;
        k.height = h;
        k.field = [raw](std::uint32_t i) { return double(float(raw[i]) * 0.001f); };
        const Outcome o = run_case(fx.m, fx.piece, k);
        CHECK(o.first);
        CHECK(o.last);
    }

    SUBCASE("a range from the data at its start and fixed at its end, a flat range and no valid depth") {
        const std::uint32_t w = 16, h = 12;
        for (int which = 0; which < 3; ++which) {
            CAPTURE(which);
            std::vector<float> d(std::size_t(w) * h);
            for (std::uint32_t i = 0; i < d.size(); ++i) {
                if (which == 0) d[i] = at_level(0.8f, 3.0f, int((i * 53) % 256));
                if (which == 1) d[i] = i % 7 == 3 ? 0.0f : 1.25f;
                if (which == 2) d[i] = i % 5 == 1 ? std::nanf("") : 0.0f;
            }
            if (which == 0) d[3] = 0.8f;   // the start of the data
            DepthParams p;
            p.colormap = 2;
            p.flags = which == 0 ? CALIPER_NET_SNAP_AUTO_VMIN : CALIPER_NET_SNAP_AUTO_VMIN | CALIPER_NET_SNAP_AUTO_VMAX;
            p.vmax = 3.0;
            p.stats = stats_of(d, 1.0f);
            Case k;
            k.name = case_name("depth", which == 0 ? "auto-start" : which == 1 ? "auto-flat" : "auto-none");
            k.scene = flat_scene("ranges", d, w, h, p);
            k.tensors = {tensor("depth", {std::int64_t(h), std::int64_t(w)}, d)};
            k.params = {{"colormap", 2}, {"auto_range", double(p.flags)}, {"vmax", 3.0}};
            k.width = w;
            k.height = h;
            k.field = [d](std::uint32_t i) { return double(d[i]); };
            run_case(fx.m, fx.piece, k);
        }
    }

    SUBCASE("on a camera's image plane in 3-D, with the back-projected cloud") {
        const std::uint32_t w = 16, h = 12;
        std::vector<float> d(std::size_t(w) * h);
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x)
                d[std::size_t(y) * w + x] = at_level(1.0f, 2.0f, int((x * 13 + y * 7) % 200 + 20));
        for (std::uint32_t i : {5u, 60u, 133u}) d[i] = 0.0f;   // holes: no point, level 0 on the plane
        DepthParams p;
        p.vmin = 1.0;
        p.vmax = 2.0;
        p.cloud = true;
        const std::vector<float> pose = facing_pose(fx.m);
        Case k;
        k.name = case_name("depth", "camera-cloud");
        k.scene = camera_scene(pose, d, w, h, p, true);
        k.plain = camera_scene(pose, d, w, h, p, false);
        k.tensors = {tensor("depth", {std::int64_t(h), std::int64_t(w)}, d), pose_tensor(pose)};
        k.params = kCameraParams;
        k.params.insert({{"vmin", 1.0}, {"vmax", 2.0}, {"auto_range", 0}, {"cloud", 1}});
        k.width = w;
        k.height = h;
        k.min_lit = 500;
        k.field = [d](std::uint32_t i) { return double(d[i]); };
        run_case(fx.m, fx.piece, k);
    }

    SUBCASE("one pixel") {
        const std::vector<float> d = {1.5f};
        DepthParams p;
        p.vmin = 1.0;
        p.vmax = 2.0;
        Case k;
        k.name = case_name("depth", "one-pixel");
        k.scene = flat_scene("one", d, 1, 1, p);
        k.tensors = {tensor("depth", {1, 1}, d)};
        k.params = {{"vmin", 1.0}, {"vmax", 2.0}, {"auto_range", 0}};
        k.width = k.height = 1;
        k.picks = 1;
        k.field = [d](std::uint32_t i) { return double(d[i]); };
        const Outcome o = run_case(fx.m, fx.piece, k);
        CHECK(o.first);
    }

    SUBCASE("a 1448 x 1448 map, minified") {
        const std::uint32_t w = 1448, h = 1448;
        std::vector<float> d(std::size_t(w) * h);
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x) d[std::size_t(y) * w + x] = 1.0f + float(x + y) / 2896.0f;
        DepthParams p;
        p.vmin = 1.0;
        p.vmax = 2.0;
        Case k;
        k.name = case_name("depth", "largest");
        k.scene = flat_scene("large", d, w, h, p);
        k.tensors = {tensor("depth", {std::int64_t(h), std::int64_t(w)}, d)};
        k.params = {{"vmin", 1.0}, {"vmax", 2.0}, {"auto_range", 0}};
        k.width = w;
        k.height = h;
        k.min_lit = 0;
        const Outcome o = run_case(fx.m, fx.piece, k);
        CHECK(o.c.edge > 50000);
    }
}

TEST_CASE("depth_match: depth@1 draws nothing for a map above its limit or an empty one") {
    Fixture fx;
    if (!fx.ready) CALIPER_SKIP_NO_GPU(fx.m.why);
    const std::vector<float> large(8193, 1.5f);   // one side above the image limit
    // On a camera with its cloud, so that neither pass draws.
    std::map<std::string, double> values = kCameraParams;
    values["cloud"] = 1;
    check_refusals(fx.m, fx.piece, {tensor("depth", {8193, 1}, large)},
                   {tensor("depth", {0, 16}, std::vector<float>{})}, 96, values);
}

CALIPER_GPU_TEST_MAIN()
