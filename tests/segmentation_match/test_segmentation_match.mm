// test_segmentation_match.mm: the match of the library piece segmentation@1 with the built-in
// segmentation image of the scene view (design/programmable-graphics.md, Library Pieces), through
// tests/image_match/image_match_rig.h. The cases cover each option of a mask: u8 class ids over a
// class table at the default opacity, u16 ids up to 65,535 with a translucent class color and ids
// throughout the palette at another opacity, the image plane of a camera in 3-D, one pixel,
// a minified mask, invalid dimensions and an empty mask. Without a Metal device the test skips (exit 77).
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "../image_match/image_match_rig.h"

namespace {

using namespace image_match;
using caliper_net::Json;
using pf::RGBA;

const std::vector<float> kIdentity = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

// A class table: ids with their colors, as SCENE_CLASSES and as the rows of the piece's input
// `colors` (a row with r -1 keeps the palette).
struct Classes {
    std::vector<std::pair<int, std::uint32_t>> colors;
    wire::SceneClasses message(pf::Fixture& f) const {
        wire::SceneClasses cl;
        cl.sub = 1;
        cl.view = f.name;
        cl.seq = ++f.seq;
        cl.is_static = true;
        std::vector<std::string> labels;
        for (std::size_t i = 0; i < colors.size(); ++i) labels.push_back("class" + std::to_string(colors[i].first));
        cl.classes = pf::classes_json(colors, labels);
        return cl;
    }
    Tensor tensor() const {
        int rows = 1;
        for (const auto& [id, c] : colors) rows = std::max(rows, id + 1);
        std::vector<float> v(std::size_t(rows) * 4, -1.0f);
        for (const auto& [id, c] : colors)
            for (int k = 0; k < 4; ++k) v[std::size_t(id) * 4 + std::size_t(k)] = float((c >> (8 * k)) & 0xffu) / 255.0f;
        return image_match::tensor("colors", {rows, 4}, v);
    }
};

// The built-in palette, as packed RGBA.
const std::uint32_t kPalette[16] = {
    RGBA(31, 119, 180),  RGBA(255, 127, 14),  RGBA(44, 160, 44),   RGBA(214, 39, 40),
    RGBA(148, 103, 189), RGBA(140, 86, 75),   RGBA(227, 119, 194), RGBA(127, 127, 127),
    RGBA(188, 189, 34),  RGBA(23, 190, 207),  RGBA(174, 199, 232), RGBA(255, 187, 120),
    RGBA(152, 223, 138), RGBA(255, 152, 150), RGBA(197, 176, 213), RGBA(196, 156, 148)};

template <class T>
void add_mask(pf::Fixture& f, const std::string& path, const std::vector<T>& ids, std::uint32_t w, std::uint32_t h,
              std::optional<double> opacity) {
    wire::SceneUpdate u = f.update(path, CALIPER_NET_PART_IMAGE);
    u.params.set("form", Json(CALIPER_NET_FORM_SEGMENTATION));
    if (opacity) u.params.set("opacity", Json(*opacity));
    const std::int64_t dtype = sizeof(T) == 2 ? CALIPER_NET_DTYPE_U16 : CALIPER_NET_DTYPE_U8;
    u.arrays.push_back(f.section("pixels", dtype, {std::int64_t(h), std::int64_t(w)}, ids));
    f.messages.push_back(u);
}

// A 2-D scene of one mask with a class table.
template <class T>
pf::Fixture flat_scene(const std::string& name, const std::vector<T>& ids, std::uint32_t w, std::uint32_t h,
                       const Classes& classes, std::optional<double> opacity) {
    pf::Fixture f;
    f.name = name;
    f.info = {CALIPER_NET_SPACE_2D, "-y", "px"};
    if (!classes.colors.empty()) f.messages.push_back(classes.message(f));
    add_mask(f, "mask", ids, w, h, opacity);
    return f;
}

// The ids as the i32 tensor that the piece binds.
template <class T>
Tensor ids_tensor(const std::vector<T>& ids, std::uint32_t w, std::uint32_t h) {
    std::vector<std::int32_t> v(ids.begin(), ids.end());
    return tensor("classes", {std::int64_t(h), std::int64_t(w)}, v, CALIPER_PRODUCER_I32);
}

// The ids as they are: a u8 or u16 tensor.
template <class T>
Tensor native_tensor(const std::vector<T>& ids, std::uint32_t w, std::uint32_t h) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2);
    return tensor("classes", {std::int64_t(h), std::int64_t(w)}, ids,
                  sizeof(T) == 1 ? CALIPER_PRODUCER_U8 : CALIPER_PRODUCER_U16);
}

struct Fixture {
    Metal m;
    cs::Piece piece;
    bool ready = false;
    Fixture() {
        if (!m.backend) return;
        std::string why;
        REQUIRE_MESSAGE(load_piece("segmentation", &piece, &why), why);
        ready = true;
    }
};

}  // namespace

TEST_CASE("segmentation_match: segmentation@1 passes piece.check, and each of its passes composes and compiles") {
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("segmentation", &piece, &why), why);
    const Checked c = check_piece(piece);
    CHECK_MESSAGE(c.ok, c.messages);
    CHECK(c.passes == 4);   // three derive passes and one draw pass
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

TEST_CASE("segmentation_match: segmentation@1 draws each mask as the built-in segmentation image draws it") {
    Fixture fx;
    if (!fx.ready) CALIPER_SKIP_NO_GPU(fx.m.why);

    SUBCASE("u8 ids over a class table at the default opacity") {
        const std::uint32_t w = 16, h = 12;
        std::vector<std::uint8_t> ids(std::size_t(w) * h);
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x)
                ids[std::size_t(y) * w + x] = std::uint8_t(x < 6 ? 1 : x < 11 ? 2 : (y < 6 ? 5 : 0));
        Classes classes;
        classes.colors = {{1, RGBA(255, 40, 40)}, {2, RGBA(40, 255, 40)}};
        Case k;
        k.name = case_name("segmentation", "u8-table");
        k.scene = flat_scene("u8", ids, w, h, classes, std::nullopt);
        k.tensors = {ids_tensor(ids, w, h), classes.tensor()};
        k.width = w;
        k.height = h;
        k.field = [ids](std::uint32_t i) { return double(ids[i]); };
        const Outcome o = run_case(fx.m, fx.piece, k);
        CHECK(o.first);
        CHECK(o.last);
    }

    SUBCASE("u16 IDs through 65,535 with sparse class colors at an opacity of 0.75") {
        const std::uint32_t w = 16, h = 12;
        std::vector<std::uint16_t> ids(std::size_t(w) * h);
        for (std::uint32_t i = 0; i < ids.size(); ++i) ids[i] = std::uint16_t((i * 7) % 20);
        ids[0] = 65535;
        ids[20] = 4096;
        ids[33] = 4097;
        ids[50] = 40000;
        Classes classes;
        classes.colors = {{3, RGBA(250, 250, 20, 128)}, {7, RGBA(20, 200, 250)}, {19, RGBA(90, 10, 160, 40)},
                          {4096, RGBA(30, 220, 170)}, {40000, RGBA(120, 30, 180)},
                          {65535, RGBA(210, 100, 40, 128)}};
        Case k;
        k.name = case_name("segmentation", "u16-alpha");
        k.scene = flat_scene("u16", ids, w, h, classes, 0.75);
        k.tensors = {ids_tensor(ids, w, h), classes.tensor()};
        k.params = {{"opacity", 0.75}};
        k.width = w;
        k.height = h;
        k.field = [ids](std::uint32_t i) { return double(ids[i]); };
        run_case(fx.m, fx.piece, k);
    }

    SUBCASE("u8 ids bound as u8 over a class table") {
        const std::uint32_t w = 16, h = 12;
        std::vector<std::uint8_t> ids(std::size_t(w) * h);
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x)
                ids[std::size_t(y) * w + x] = std::uint8_t(x < 6 ? 1 : x < 11 ? 2 : (y < 6 ? 255 : 0));
        Classes classes;
        classes.colors = {{1, RGBA(255, 40, 40)}, {2, RGBA(40, 255, 40)}, {255, RGBA(40, 40, 255)}};
        Case k;
        k.name = case_name("segmentation", "u8-native");
        k.scene = flat_scene("u8n", ids, w, h, classes, std::nullopt);
        k.tensors = {native_tensor(ids, w, h), classes.tensor()};
        k.width = w;
        k.height = h;
        k.field = [ids](std::uint32_t i) { return double(ids[i]); };
        const Outcome o = run_case(fx.m, fx.piece, k);
        CHECK(o.first);
        CHECK(o.last);
    }

    SUBCASE("u8 ids bound as u8, 7 x 5, 35 bytes, not a multiple of 4") {
        const std::uint32_t w = 7, h = 5;
        std::vector<std::uint8_t> ids(std::size_t(w) * h);
        for (std::uint32_t i = 0; i < ids.size(); ++i) ids[i] = std::uint8_t(1 + i % 3);
        Classes classes;
        classes.colors = {{1, RGBA(255, 40, 40)}, {2, RGBA(40, 255, 40)}, {3, RGBA(40, 40, 255)}};
        Case k;
        k.name = case_name("segmentation", "u8-odd-size");
        k.scene = flat_scene("u8o", ids, w, h, classes, std::nullopt);
        k.tensors = {native_tensor(ids, w, h), classes.tensor()};
        k.width = w;
        k.height = h;
        k.field = [ids](std::uint32_t i) { return double(ids[i]); };
        run_case(fx.m, fx.piece, k);
    }

    SUBCASE("u16 ids bound as u16, a translucent class color, at an opacity of 0.75") {
        const std::uint32_t w = 16, h = 12;
        std::vector<std::uint16_t> ids(std::size_t(w) * h);
        for (std::uint32_t i = 0; i < ids.size(); ++i) ids[i] = std::uint16_t((i * 7) % 20);
        ids[0] = 65535;
        ids[20] = 4096;
        ids[33] = 4097;
        ids[50] = 40000;
        Classes classes;
        classes.colors = {{3, RGBA(250, 250, 20, 128)}, {7, RGBA(20, 200, 250)}, {19, RGBA(90, 10, 160, 40)},
                          {4096, RGBA(30, 220, 170)}, {40000, RGBA(120, 30, 180)},
                          {65535, RGBA(210, 100, 40, 128)}};
        Case k;
        k.name = case_name("segmentation", "u16-native");
        k.scene = flat_scene("u16n", ids, w, h, classes, 0.75);
        k.tensors = {native_tensor(ids, w, h), classes.tensor()};
        k.params = {{"opacity", 0.75}};
        k.width = w;
        k.height = h;
        k.field = [ids](std::uint32_t i) { return double(ids[i]); };
        run_case(fx.m, fx.piece, k);
    }

    SUBCASE("on a camera's image plane in 3-D") {
        const std::uint32_t w = 16, h = 12;
        std::vector<std::uint8_t> ids(std::size_t(w) * h);
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x) ids[std::size_t(y) * w + x] = std::uint8_t((x / 4 + y / 3) % 6);
        const std::vector<float> pose = facing_pose(fx.m);
        auto scene = [&](bool with_mask) {
            pf::Fixture f = camera_room(pose);
            if (with_mask) add_mask(f, "rig/cam/mask", ids, w, h, 0.9);
            return f;
        };
        Case k;
        k.name = case_name("segmentation", "camera");
        k.scene = scene(true);
        k.plain = scene(false);
        k.tensors = {ids_tensor(ids, w, h), pose_tensor(pose)};
        k.params = kCameraParams;
        k.params["opacity"] = 0.9;
        k.width = w;
        k.height = h;
        k.min_lit = 500;
        k.field = [ids](std::uint32_t i) { return double(ids[i]); };
        run_case(fx.m, fx.piece, k);   // the corners lie under the frustum, thus the 2-D cases pick them
    }

    SUBCASE("one pixel") {
        const std::vector<std::uint8_t> ids = {9};
        Case k;
        k.name = case_name("segmentation", "one-pixel");
        k.scene = flat_scene("one", ids, 1, 1, Classes{}, std::nullopt);
        k.tensors = {ids_tensor(ids, 1, 1), Classes{}.tensor()};
        k.width = k.height = 1;
        k.picks = 1;
        k.field = [ids](std::uint32_t i) { return double(ids[i]); };
        const Outcome o = run_case(fx.m, fx.piece, k);
        CHECK(o.first);
    }

    SUBCASE("the largest mask of the published limit, 1446 x 1446, minified") {
        const std::uint32_t w = 1446, h = 1446;
        std::vector<std::uint16_t> ids(std::size_t(w) * h);
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x) ids[std::size_t(y) * w + x] = std::uint16_t((x / 181) + 8 * (y / 181));
        Case k;
        k.name = case_name("segmentation", "largest");
        k.scene = flat_scene("large", ids, w, h, Classes{}, std::nullopt);
        k.tensors = {ids_tensor(ids, w, h), Classes{}.tensor()};
        k.width = w;
        k.height = h;
        k.min_lit = 0;
        const Outcome o = run_case(fx.m, fx.piece, k);
        CHECK(o.c.edge + o.c.interior > 50000);
    }
}

TEST_CASE("segmentation_match: the palette holds the class colors of the input and the built-in palette") {
    Fixture fx;
    if (!fx.ready) CALIPER_SKIP_NO_GPU(fx.m.why);
    Classes classes;
    classes.colors = {{2, RGBA(1, 2, 3, 4)}, {4095, RGBA(200, 100, 50)},
                      {40000, RGBA(60, 90, 180)}, {65535, RGBA(10, 210, 50, 128)}};
    const std::vector<std::int32_t> ids = {-5, 0, 70000};
    std::vector<Version> versions;
    std::string why;
    REQUIRE_MESSAGE(run_derive(fx.m.device, fx.piece,
                               {tensor("classes", {1, 3}, ids, CALIPER_PRODUCER_I32), classes.tensor()},
                               {}, &versions, &why),
                    why);
    const Version* palette = find_version(versions, "palette");
    const Version* pixels = find_version(versions, "pixels");
    REQUIRE(palette);
    REQUIRE(pixels);
    REQUIRE(palette->bytes.size() == 65536 * 4);
    std::vector<std::uint32_t> p(65536);
    std::memcpy(p.data(), palette->bytes.data(), palette->bytes.size());
    CHECK(p[0] == kPalette[0]);
    CHECK(p[2] == RGBA(1, 2, 3, 4));
    CHECK(p[3] == kPalette[3]);
    CHECK(p[17] == kPalette[1]);
    CHECK(p[4095] == RGBA(200, 100, 50));
    CHECK(p[4096] == kPalette[0]);
    CHECK(p[40000] == RGBA(60, 90, 180));
    CHECK(p[65535] == RGBA(10, 210, 50, 128));
    CHECK(pixels->format == "r32uint");
    CHECK(pixels->width == 3);
    CHECK(pixels->height == 1);
    // The ids clamp to 0 to 65,535, as the producer clamps them.
    REQUIRE(pixels->count == 3);
    std::uint32_t got[3];
    std::memcpy(got, pixels->bytes.data(), 12);
    CHECK(got[0] == 0);
    CHECK(got[1] == 0);
    CHECK(got[2] == 65535);
}

TEST_CASE("segmentation_match: segmentation@1 draws nothing for a mask above its limit or an empty one") {
    Fixture fx;
    if (!fx.ready) CALIPER_SKIP_NO_GPU(fx.m.why);
    const std::vector<std::int32_t> large(8193, 1);   // one side above the image limit
    check_refusals(fx.m, fx.piece,
                   {tensor("classes", {8193, 1}, large, CALIPER_PRODUCER_I32), Classes{}.tensor()},
                   {tensor("classes", {0, 16}, std::vector<std::int32_t>{}, CALIPER_PRODUCER_I32), Classes{}.tensor(),
                    pose_tensor(kIdentity)},
                   80, {});
}

CALIPER_GPU_TEST_MAIN()
