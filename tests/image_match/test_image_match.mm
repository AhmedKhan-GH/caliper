// test_image_match.mm: the match of the library piece image@1 with the built-in image of the
// scene view (design/programmable-graphics.md, Library Pieces), through image_match_rig.h. The
// cases cover each form and option of a color image: RGBA with translucent pixels flat in 2-D at
// an offset, grey at an opacity, RGB on a camera's image plane in 3-D in both camera conventions,
// one pixel, the largest image that the 8 MiB of published resources hold, the next size, which
// the piece leaves undrawn, and an empty image. Without a Metal device the test skips (exit 77).
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "image_match_rig.h"

namespace {

using namespace image_match;
using pf::RGBA;

// A smooth color image: RGBA bytes, row by row, from a function of the pixel.
std::vector<std::uint32_t> color_pixels(std::uint32_t w, std::uint32_t h,
                                        const std::function<std::uint32_t(std::uint32_t, std::uint32_t)>& f) {
    std::vector<std::uint32_t> px(std::size_t(w) * h);
    for (std::uint32_t y = 0; y < h; ++y)
        for (std::uint32_t x = 0; x < w; ++x) px[std::size_t(y) * w + x] = f(x, y);
    return px;
}

// The tensor of an RGBA image with `channels` of its channels, each byte as byte / 255.
Tensor image_tensor(const std::vector<std::uint32_t>& px, std::uint32_t w, std::uint32_t h, int channels) {
    std::vector<float> v;
    v.reserve(px.size() * std::size_t(channels));
    for (std::uint32_t c : px)
        for (int k = 0; k < channels; ++k) v.push_back(float((c >> (8 * k)) & 0xffu) / 255.0f);
    return tensor("image", {std::int64_t(h), std::int64_t(w), std::int64_t(channels)}, v);
}

// The same image as u8 bytes, the channels as they are.
Tensor image_tensor_u8(const std::vector<std::uint32_t>& px, std::uint32_t w, std::uint32_t h, int channels) {
    std::vector<std::uint8_t> v;
    v.reserve(px.size() * std::size_t(channels));
    for (std::uint32_t c : px)
        for (int k = 0; k < channels; ++k) v.push_back(std::uint8_t((c >> (8 * k)) & 0xffu));
    return tensor("image", {std::int64_t(h), std::int64_t(w), std::int64_t(channels)}, v, CALIPER_PRODUCER_U8);
}

// The packed RGBA of a pixel as the item map's field rgba gives it.
double field_of(std::uint32_t rgba) { return double(rgba); }

// A 2-D scene of one color image at `path`, translated by (tx, ty).
pf::Fixture flat_scene(const std::string& name, const std::vector<std::uint32_t>& px, std::uint32_t w,
                       std::uint32_t h, bool grey, const std::vector<std::uint8_t>& grey_px, float tx, float ty,
                       std::optional<double> opacity) {
    pf::Fixture f;
    f.name = name;
    f.info = {CALIPER_NET_SPACE_2D, "-y", "px"};
    wire::SceneUpdate u = f.update("image", CALIPER_NET_PART_IMAGE);
    u.params.set("form", caliper_net::Json(CALIPER_NET_FORM_COLOR));
    if (opacity) u.params.set("opacity", caliper_net::Json(*opacity));
    if (grey) u.arrays.push_back(f.section("pixels", CALIPER_NET_DTYPE_U8, {std::int64_t(h), std::int64_t(w)}, grey_px));
    else u.arrays.push_back(f.section("pixels", CALIPER_NET_DTYPE_U8, {std::int64_t(h), std::int64_t(w), 4}, px));
    f.messages.push_back(u);
    f.transform("image", pf::translate(tx, ty, 0));
    return f;
}

// The room with its camera at a pose, and the color image under the camera unless `with_image` is
// false.
pf::Fixture camera_scene(const std::vector<float>& pose, const char* convention, const std::vector<std::uint32_t>& px,
                         std::uint32_t w, std::uint32_t h, bool with_image) {
    pf::Fixture f = camera_room(pose, convention);
    if (with_image) {
        wire::SceneUpdate img = f.update("rig/cam/rgb", CALIPER_NET_PART_IMAGE);
        img.params.set("form", caliper_net::Json(CALIPER_NET_FORM_COLOR));
        img.arrays.push_back(f.section("pixels", CALIPER_NET_DTYPE_U8, {std::int64_t(h), std::int64_t(w), 4}, px));
        f.messages.push_back(img);
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
        REQUIRE_MESSAGE(load_piece("image", &piece, &why), why);
        ready = true;
    }
};

}  // namespace

TEST_CASE("image_match: image@1 passes piece.check, and each of its passes composes and compiles") {
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("image", &piece, &why), why);
    const Checked c = check_piece(piece);
    CHECK_MESSAGE(c.ok, c.messages);
    CHECK(c.passes == 3);   // two derive passes and one draw pass
    CHECK(cp::canonical_manifest(piece).find("\"image\"") != std::string::npos);
    // The published buffers fit the 8 MiB that a pass may send to a viewer, and the texture holds at
    // most 33,554,432 texels of 4 bytes, 128 MiB, the most that a published resource holds.
    std::uint64_t bytes = 0, texture_bytes = 0;
    for (const cs::Resource& r : piece.resources) {
        if (r.kind == cs::ResourceKind::texture) {
            if (r.publish) texture_bytes += std::uint64_t(r.texels.value_or(0)) * 4;
            continue;
        }
        cp::ElementLayout l;
        REQUIRE(cp::element_layout(r.layout, &l).ok());
        if (r.publish) bytes += std::uint64_t(r.count) * l.stride;
    }
    MESSAGE("image@1 publishes at most ", bytes, " bytes of buffers and ", texture_bytes, " bytes of texture");
    CHECK(bytes <= 8388608u);
    CHECK(texture_bytes <= (128ull << 20));
}

TEST_CASE("image_match: image@1 draws each color image as the built-in image draws it") {
    Fixture fx;
    if (!fx.ready) CALIPER_SKIP_NO_GPU(fx.m.why);

    SUBCASE("rgba flat in 2-D, translated, with translucent pixels") {
        const std::uint32_t w = 16, h = 12;
        const auto px = color_pixels(w, h, [](std::uint32_t x, std::uint32_t y) {
            const std::uint32_t a = (x >= 10 && x <= 13 && y >= 3 && y <= 6) ? 128 : 255;
            return RGBA(30 + 12 * x, 40 + 14 * y, 220 - 8 * x, a);
        });
        Case k;
        k.name = case_name("image", "rgba-flat");
        k.scene = flat_scene("rgba", px, w, h, false, {}, 4.0f, 2.0f, std::nullopt);
        k.tensors = {image_tensor(px, w, h, 4), pose_tensor(pf::translate(4, 2, 0))};
        k.width = w;
        k.height = h;
        k.interior_tolerance = 0;
        k.field = [px](std::uint32_t i) { return field_of(px[i]); };
        const Outcome o = run_case(fx.m, fx.piece, k);
        CHECK(o.first);
        CHECK(o.last);
    }

    SUBCASE("grey flat in 2-D at an opacity of 0.6") {
        const std::uint32_t w = 16, h = 12;
        std::vector<std::uint8_t> grey(std::size_t(w) * h);
        std::vector<float> values(grey.size());
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x) {
                grey[std::size_t(y) * w + x] = std::uint8_t(20 + 10 * x + 4 * y);
                values[std::size_t(y) * w + x] = float(grey[std::size_t(y) * w + x]) / 255.0f;
            }
        Case k;
        k.name = case_name("image", "grey-opacity");
        k.scene = flat_scene("grey", {}, w, h, true, grey, 0.0f, 0.0f, 0.6);
        k.tensors = {tensor("image", {std::int64_t(h), std::int64_t(w)}, values)};
        k.params = {{"opacity", 0.6}};
        k.width = w;
        k.height = h;
        k.interior_tolerance = 0;
        k.field = [grey](std::uint32_t i) {
            const std::uint32_t g = grey[i];
            return field_of(0xff000000u | g | g << 8 | g << 16);
        };
        run_case(fx.m, fx.piece, k);
    }

    SUBCASE("rgb on a camera's image plane in 3-D, in both camera conventions") {
        const std::uint32_t w = 16, h = 12;
        const auto px = color_pixels(w, h, [](std::uint32_t x, std::uint32_t y) {
            return RGBA(40 + 12 * x, 60 + 14 * y, 160, 255);
        });
        const std::vector<float> rdf = facing_pose(fx.m);
        // The same camera with right, up and back axes: the second and third columns negated.
        std::vector<float> rub = rdf;
        for (int k = 4; k < 12; ++k) rub[std::size_t(k)] = -rub[std::size_t(k)];
        for (const bool use_rub : {false, true}) {
            CAPTURE(use_rub);
            const std::vector<float>& pose = use_rub ? rub : rdf;
            const char* conv = use_rub ? CALIPER_NET_CONVENTION_RUB : CALIPER_NET_CONVENTION_RDF;
            Case k;
            k.name = case_name("image", use_rub ? "rgb-camera-rub" : "rgb-camera-rdf");
            k.scene = camera_scene(pose, conv, px, w, h, true);
            k.plain = camera_scene(pose, conv, px, w, h, false);
            k.tensors = {image_tensor(px, w, h, 3), pose_tensor(pose)};
            k.params = kCameraParams;
            k.params["convention"] = use_rub ? 1 : 0;
            k.width = w;
            k.height = h;
            k.interior_tolerance = 0;
            k.min_lit = 500;
            k.field = [px](std::uint32_t i) { return field_of(px[i]); };
            run_case(fx.m, fx.piece, k);   // the corners lie under the frustum, thus the 2-D cases pick them
        }
    }

    SUBCASE("u8 rgba flat in 2-D, bound as bytes, with translucent pixels") {
        const std::uint32_t w = 16, h = 12;
        const auto px = color_pixels(w, h, [](std::uint32_t x, std::uint32_t y) {
            const std::uint32_t a = (x >= 10 && x <= 13 && y >= 3 && y <= 6) ? 128 : 255;
            return RGBA(30 + 12 * x, 40 + 14 * y, 220 - 8 * x, a);
        });
        Case k;
        k.name = case_name("image", "u8-rgba-flat");
        k.scene = flat_scene("rgba8", px, w, h, false, {}, 4.0f, 2.0f, std::nullopt);
        k.tensors = {image_tensor_u8(px, w, h, 4), pose_tensor(pf::translate(4, 2, 0))};
        k.width = w;
        k.height = h;
        k.interior_tolerance = 0;
        k.field = [px](std::uint32_t i) { return field_of(px[i]); };
        const Outcome o = run_case(fx.m, fx.piece, k);
        CHECK(o.first);
        CHECK(o.last);
    }

    SUBCASE("u8 rgb on a camera's image plane in 3-D, bound as bytes") {
        const std::uint32_t w = 16, h = 12;
        const auto px = color_pixels(w, h, [](std::uint32_t x, std::uint32_t y) {
            return RGBA(40 + 12 * x, 60 + 14 * y, 160, 255);
        });
        const std::vector<float> pose = facing_pose(fx.m);
        Case k;
        k.name = case_name("image", "u8-rgb-camera-rdf");
        k.scene = camera_scene(pose, CALIPER_NET_CONVENTION_RDF, px, w, h, true);
        k.plain = camera_scene(pose, CALIPER_NET_CONVENTION_RDF, px, w, h, false);
        k.tensors = {image_tensor_u8(px, w, h, 3), pose_tensor(pose)};
        k.params = kCameraParams;
        k.params["convention"] = 0;
        k.width = w;
        k.height = h;
        k.interior_tolerance = 0;
        k.min_lit = 500;
        k.field = [px](std::uint32_t i) { return field_of(px[i]); };
        run_case(fx.m, fx.piece, k);
    }

    SUBCASE("u8 grey flat in 2-D at an opacity of 0.6, bound as bytes") {
        const std::uint32_t w = 16, h = 12;
        std::vector<std::uint8_t> grey(std::size_t(w) * h);
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x) grey[std::size_t(y) * w + x] = std::uint8_t(20 + 10 * x + 4 * y);
        Case k;
        k.name = case_name("image", "u8-grey-opacity");
        k.scene = flat_scene("grey8", {}, w, h, true, grey, 0.0f, 0.0f, 0.6);
        k.tensors = {tensor("image", {std::int64_t(h), std::int64_t(w)}, grey, CALIPER_PRODUCER_U8)};
        k.params = {{"opacity", 0.6}};
        k.width = w;
        k.height = h;
        k.interior_tolerance = 0;
        k.field = [grey](std::uint32_t i) {
            const std::uint32_t g = grey[i];
            return field_of(0xff000000u | g | g << 8 | g << 16);
        };
        run_case(fx.m, fx.piece, k);
    }

    SUBCASE("one pixel") {
        const std::vector<std::uint32_t> px = {RGBA(200, 90, 30, 255)};
        Case k;
        k.name = case_name("image", "one-pixel");
        k.scene = flat_scene("one", px, 1, 1, false, {}, 0.0f, 0.0f, std::nullopt);
        k.tensors = {image_tensor(px, 1, 1, 4)};
        k.width = k.height = 1;
        k.picks = 1;
        k.field = [px](std::uint32_t i) { return field_of(px[i]); };
        const Outcome o = run_case(fx.m, fx.piece, k);
        CHECK(o.first);
    }

    SUBCASE("the largest image of the published limit, 1448 x 1448, minified") {
        const std::uint32_t w = 1448, h = 1448;
        const auto px = color_pixels(w, h, [](std::uint32_t x, std::uint32_t y) {
            return RGBA(20 + x / 8, 30 + y / 8, 200 - (x + y) / 16, 255);
        });
        Case k;
        k.name = case_name("image", "largest");
        k.scene = flat_scene("large", px, w, h, false, {}, 0.0f, 0.0f, std::nullopt);
        k.tensors = {image_tensor(px, w, h, 4)};
        k.width = w;
        k.height = h;
        k.interior_tolerance = 0;
        k.min_lit = 0;   // a pixel of the image is smaller than a pixel of the view
        k.field = [px](std::uint32_t i) { return field_of(px[i]); };
        const Outcome o = run_case(fx.m, fx.piece, k);
        CHECK(o.c.edge > 50000);
    }
}

TEST_CASE("image_match: image@1 draws nothing for an image above its limit or an empty one") {
    Fixture fx;
    if (!fx.ready) CALIPER_SKIP_NO_GPU(fx.m.why);
    // A side of 8,193, one above the largest side of the texture.
    const std::vector<float> large(std::size_t(8193) * 4 * 4, 0.5f);
    check_refusals(fx.m, fx.piece, {tensor("image", {8193, 4, 4}, large)},
                   {tensor("image", {0, 16, 4}, std::vector<float>{})}, 80, {});
    // The built-in takes an empty image as the removal of the part, and draws nothing either.
    Builtin b(*fx.m.backend);
    pf::Fixture empty;
    empty.name = "empty";
    empty.info = {CALIPER_NET_SPACE_2D, "-y", "px"};
    wire::SceneUpdate u = empty.update("image", CALIPER_NET_PART_IMAGE);
    u.params.set("form", caliper_net::Json(CALIPER_NET_FORM_COLOR));
    u.arrays.push_back(empty.section("pixels", CALIPER_NET_DTYPE_U8, {0, 16, 4}, std::vector<std::uint8_t>{}));
    empty.messages.push_back(u);
    std::string why;
    REQUIRE_MESSAGE(b.draw(empty, &why), why);
    CHECK(b.refusal.empty());
    const std::vector<std::uint8_t> px = b.pixels();
    std::size_t lit = 0;
    for (int y = 0; y < int(kSize); ++y)
        for (int x = 0; x < int(kSize); ++x) lit += pixel_of(px, x, y) != kClear;
    CHECK(lit == 0);
}

// The largest image of the piece, 8,192 x 4,096 u8 RGBA, or its 1/CALIPER_CAPACITY_FRACTION in rows. It
// is heavy, thus it is skipped unless run with --no-skip --test-case="*at its capacity".
TEST_CASE("image_match: image@1 at its capacity" * doctest::skip()) {
    Fixture fx;
    if (!fx.ready) CALIPER_SKIP_NO_GPU(fx.m.why);
    const std::uint32_t w = 8192, h = 4096 / std::uint32_t(capacity_fraction());
    const auto px = color_pixels(w, h, [](std::uint32_t x, std::uint32_t y) {
        return RGBA(x / 32, y / 16, 255 - (x + y) / 64, 255);
    });
    Case k;
    k.name = case_name("image", "capacity");
    k.scene = flat_scene("capacity", px, w, h, false, {}, 0.0f, 0.0f, std::nullopt);
    k.tensors = {image_tensor_u8(px, w, h, 4)};
    k.width = w;
    k.height = h;
    k.interior_tolerance = 0;
    k.min_lit = 0;   // a pixel of the image is much smaller than a pixel of the view
    run_case(fx.m, fx.piece, k);
}

CALIPER_GPU_TEST_MAIN()
