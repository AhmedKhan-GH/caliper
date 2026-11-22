// Picking: the pick pass's draw list (pick entries into a 16 x 16 RG32Uint window reached by
// zooming clip space), the map from pick id to entity and primitive, and the decoding of
// read-back ID texels into hits (src/host/scene/pick.h); and the pick of a program, read from the
// ID target of a frame that the view drew, with the refusals of its pointer, window and bytes.
#include "scene/pick.h"
#include "gfx/recording_backend.h"
#include "fake_resources.h"
#include "scene_fixtures.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>

using namespace caliper_scene;
using namespace scene_fixtures;
namespace sh = caliper_gfx::shaders;

namespace {

Batch bounded(Batch b, double lo = -1, double hi = 1) {
    b.bounds.add({lo, lo, lo});
    b.bounds.add({hi, hi, hi});
    return b;
}

// The window pixel a clip position lands in, as the shaders' to_target and the rasteriser put it.
void window_pixel(const sh::Frame& f, const double clip[4], int32_t ww, int32_t wh, double* x, double* y) {
    const double nx = (clip[0] * f.ndc_scale[0] + f.ndc_offset[0] * clip[3]) / clip[3];
    const double ny = (clip[1] * f.ndc_scale[1] + f.ndc_offset[1] * clip[3]) / clip[3];
    *x = (nx + 1) * 0.5 * ww;
    *y = (1 - ny) * 0.5 * wh;
}

}  // namespace

TEST_CASE("the pick window: 2r x 2r pixels with the pointer's pixel at texel (r, r)") {
    const PickWindow w = pick_window(100.7f, 50.2f, 8, 400, 300);
    CHECK(w.width == 16);
    CHECK(w.height == 16);
    CHECK(w.x0 == 92);
    CHECK(w.y0 == 42);
    CHECK(int(std::floor(w.px)) - w.x0 == 8);
}

TEST_CASE("the pick pass: pick entries into an RG32Uint window, clip space zoomed onto the pointer") {
    SceneTree t;
    Batch translucent = bounded(points3(20));
    translucent.flat_rgba = 0x40ffffffu;
    REQUIRE(t.set_batch(t.ensure("cloud"), translucent));
    REQUIRE(t.set_batch(t.ensure("mesh"), bounded(mesh3(6, 0))));
    REQUIRE(t.set_batch(t.ensure("lines"), bounded(lines3(4, LineMode::Segments))));
    t.update();
    ViewCamera cam = Orbit{};
    fit(cam, t.scene_bounds(), 400.0 / 300.0);
    CompileOptions o;
    o.width = 400;
    o.height = 300;
    FakeResources res;
    SceneCompiler compiler;
    // The colour pass, for its map.
    DrawList colour{RenderTarget{Texture{1}, {}, Texture{2}, 0, true}};
    DrawMap colour_map;
    REQUIRE(compiler.compile(t, cam, o, res, colour, &colour_map).ok);
    // The pick pass around the view's centre, where the orbit's target projects.
    Picker picker;
    const PickWindow w = picker.request(200.0f, 150.0f, 8, 400, 300);
    const int32_t window[4] = {w.x0, w.y0, w.width, w.height};
    DrawList pick{RenderTarget{Texture{3}, {}, Texture{4}, 0, true}};
    DrawMap map;
    const CompileResult r = compiler.compile_pick(t, cam, o, window, res, pick, &map);
    REQUIRE(r.ok);
    REQUIRE(pick.records().size == 3);
    // The same draws in the same order with the same pick ids as the colour pass.
    REQUIRE(map.entries.size() == colour_map.entries.size());
    for (size_t i = 0; i < map.entries.size(); ++i) {
        CHECK(map.entries[i].entity == colour_map.entries[i].entity);
        CHECK(map.entries[i].part == colour_map.entries[i].part);
        CHECK(pick.records()[i].pick_id == colour.records()[i].pick_id);
    }
    for (size_t i = 0; i < pick.records().size; ++i) {
        const PipelineRecord& p = res.pipeline(pick.records()[i].pipeline);
        CHECK(p.desc.color_format == Format::RG32Uint);
        CHECK(p.desc.blend == Blend::Opaque);
        CHECK(p.desc.depth_test);
        CHECK(p.desc.depth_write);   // translucent draws too: the front item wins
        CHECK(p.fs == (p.program == ProgramId::scene_points ? "fs_points_pick" : "fs_pick"));
        CHECK(params_of(pick, pick.records()[i]).pick_id == i + 1);
    }
    // The frame block's pick window: the view's own viewport, zoomed by 400 / 16 and 300 / 16.
    const sh::Frame f = frame_of(pick, pick.records()[0]);
    CHECK(f.viewport[0] == 400.0f);
    CHECK(f.ndc_scale[0] == doctest::Approx(25.0));
    CHECK(f.ndc_scale[1] == doctest::Approx(18.75));
    // The orbit's target lands on the pointer's texel, the window's centre.
    const sh::PrimParams q = params_of(pick, pick.records()[0]);
    double clip[4];
    const Orbit& orbit = std::get<Orbit>(cam);
    transform_h(Mat4::from(q.mvp), {orbit.target[0], orbit.target[1], orbit.target[2]}, clip);
    double x, y;
    window_pixel(f, clip, w.width, w.height, &x, &y);
    CHECK(x == doctest::Approx(200.0 - w.x0));
    CHECK(y == doctest::Approx(150.0 - w.y0));
    // A point at view pixel (205.5, 147.5) lands on texel (13, 5).
    const Mat4 inv_vp = map.inverse_view_proj;
    const double ndc[3] = {2 * 205.5 / 400 - 1, 1 - 2 * 147.5 / 300, 0.5};
    double wp[4];
    transform_h(inv_vp, {ndc[0], ndc[1], ndc[2]}, wp);
    const Vec3 world{wp[0] / wp[3], wp[1] / wp[3], wp[2] / wp[3]};
    transform_h(Mat4::from(q.mvp), world, clip);
    window_pixel(f, clip, w.width, w.height, &x, &y);
    CHECK(x == doctest::Approx(13.5).epsilon(1e-4));
    CHECK(y == doctest::Approx(5.5).epsilon(1e-4));
}

TEST_CASE("decoding ID texels: the nearest pickable texel in the view, its entity, primitive and item") {
    DrawMap map;
    map.revision = 7;
    DrawMapEntry pts;
    pts.entity = EntityId{3, 1};
    pts.primitive = Primitive::Points;
    pts.items = 100;
    DrawMapEntry mesh;
    mesh.entity = EntityId{4, 1};
    mesh.primitive = Primitive::Mesh;
    map.entries = {pts, mesh};
    const PickWindow w = pick_window(10.5f, 10.5f, 8, 400, 300);   // texel (8, 8) holds the pointer
    std::vector<uint32_t> ids(16 * 16 * 2, 0);
    std::vector<float> depth(16 * 16, 1.0f);
    auto put = [&](int i, int j, uint32_t id, uint32_t item, float z) {
        ids[2 * (j * 16 + i)] = id;
        ids[2 * (j * 16 + i) + 1] = item;
        depth[j * 16 + i] = z;
    };
    // Nothing: no hit.
    CHECK_FALSE(decode_pick(ids.data(), depth.data(), w, map));
    put(11, 8, 1, 42, 0.5f);   // 3 pixels right
    put(8, 6, 2, 17, 0.4f);    // 2 pixels up: nearer
    std::optional<Hit> h = decode_pick(ids.data(), depth.data(), w, map);
    REQUIRE(h);
    CHECK(h->entity == mesh.entity);
    CHECK(h->primitive == Primitive::Mesh);
    CHECK(h->item == 17);
    CHECK(h->texel[0] == 8);
    CHECK(h->texel[1] == 6);
    // At equal distance the nearer depth wins.
    put(6, 8, 1, 43, 0.3f);    // 2 pixels left, nearer depth
    h = decode_pick(ids.data(), depth.data(), w, map);
    CHECK(h->item == 43);
    // The pointer's own texel wins over all.
    put(8, 8, 1, 99, 0.9f);
    h = decode_pick(ids.data(), depth.data(), w, map);
    CHECK(h->item == 99);
    CHECK(h->entity == pts.entity);
    // Ids the map does not know, and texels outside the view, are skipped.
    std::fill(ids.begin(), ids.end(), 0u);
    put(8, 8, 9, 1, 0.5f);
    CHECK_FALSE(decode_pick(ids.data(), depth.data(), w, map));
    const PickWindow corner = pick_window(1.0f, 1.0f, 8, 400, 300);   // x0 = y0 = -7
    std::fill(ids.begin(), ids.end(), 0u);
    put(2, 2, 1, 5, 0.5f);     // view pixel (-5, -5): outside
    CHECK_FALSE(decode_pick(ids.data(), depth.data(), corner, map));
    put(8, 8, 1, 6, 0.5f);     // view pixel (1, 1)
    CHECK(decode_pick(ids.data(), depth.data(), corner, map)->item == 6);
    // Beyond the radius: skipped.
    std::fill(ids.begin(), ids.end(), 0u);
    put(0, 0, 1, 7, 0.5f);     // 8 pixels left and up: 11.3 > 8
    CHECK_FALSE(decode_pick(ids.data(), depth.data(), w, map));
}

TEST_CASE("decoding finds a world position from the depth, and an image's pixel from its quad") {
    SceneTree t(Space::k2D);
    Batch img;
    img.primitive = Primitive::Image;
    img.texture = Texture{5};
    img.image_width = 64;
    img.image_height = 32;
    const EntityId e = t.ensure("img");
    REQUIRE(t.set_batch(e, img));
    t.update();
    ViewCamera cam = starting_camera(StartingCamera{}, t, 2.0).camera;
    CompileOptions o;
    o.width = 400;
    o.height = 200;
    FakeResources res;
    SceneCompiler compiler;
    Picker picker;
    // The view pixel over image pixel (40, 10): the image spans the fitted pan and zoom.
    const CameraMatrices cm = camera_matrices(cam, t, 400, 200);
    double h4[4];
    transform_h(cm.proj, {40.5, 10.5, 0}, h4);
    const float px = float((h4[0] + 1) * 0.5 * 400), py = float((1 - h4[1]) * 0.5 * 200);
    const PickWindow w = picker.request(px, py, 8, 400, 200);
    const int32_t window[4] = {w.x0, w.y0, w.width, w.height};
    DrawList pick{RenderTarget{Texture{3}, {}, {}, 0, true}};
    DrawMap map;
    REQUIRE(compiler.compile_pick(t, cam, o, window, res, pick, &map).ok);
    std::vector<uint8_t> bytes(16 * 16 * 8, 0);
    const uint32_t texel[2] = {1, 0};   // pick id 1, item 0: images write no item
    std::memcpy(bytes.data() + (8 * 16 + 8) * 8, texel, 8);
    std::optional<Hit> hit = picker.deliver(picker.current(), map.revision, bytes, nullptr, map);
    REQUIRE(hit);
    CHECK(hit->entity == e);
    CHECK(hit->primitive == Primitive::Image);
    REQUIRE(hit->has_pixel);
    CHECK(hit->pixel[0] == 40);
    CHECK(hit->pixel[1] == 10);
    CHECK_FALSE(hit->has_position);

    // 3-D: the depth at the texel gives the world position.
    SceneTree s;
    REQUIRE(s.set_batch(s.ensure("p"), bounded(points3(1))));
    s.update();
    ViewCamera orbit = Orbit{};
    fit(orbit, s.scene_bounds(), 1.0);
    const CameraMatrices m = camera_matrices(orbit, s, 300, 300);
    const Vec3 target{std::get<Orbit>(orbit).target[0], std::get<Orbit>(orbit).target[1], std::get<Orbit>(orbit).target[2]};
    double c[4];
    transform_h(m.proj * m.view, target, c);
    const float zt = float(c[2] / c[3]);
    DrawMap map3;
    map3.revision = s.revision();
    DrawMapEntry entry;
    entry.entity = s.find("p");
    map3.entries = {entry};
    map3.has_inverse = inverse(m.proj * m.view, &map3.inverse_view_proj);
    const PickWindow w3 = pick_window(150.0f, 150.0f, 8, 300, 300);
    std::vector<uint32_t> ids(16 * 16 * 2, 0);
    std::vector<float> depth(16 * 16, 1.0f);
    ids[2 * (8 * 16 + 8)] = 1;
    depth[8 * 16 + 8] = zt;
    hit = decode_pick(ids.data(), depth.data(), w3, map3);
    REQUIRE(hit);
    REQUIRE(hit->has_position);
    // The texel's centre is half a pixel from the target's projection; the position is within that.
    CHECK(length(hit->position - target) < 0.02);
}

TEST_CASE("the picker drops superseded and stale results and short readbacks") {
    DrawMap map;
    map.revision = 3;
    DrawMapEntry e;
    e.entity = EntityId{1, 1};
    map.entries = {e};
    Picker picker;
    picker.request(8.0f, 8.0f, 8, 100, 100);
    const uint64_t first = picker.current();
    picker.request(9.0f, 9.0f, 8, 100, 100);   // supersedes the first
    CHECK(picker.pending());
    std::vector<uint8_t> bytes(16 * 16 * 8, 0);
    const uint32_t texel[2] = {1, 5};
    std::memcpy(bytes.data() + (8 * 16 + 8) * 8, texel, 8);
    CHECK_FALSE(picker.deliver(first, 3, bytes, nullptr, map));
    CHECK(picker.pending());
    // The draw list changed since the request was compiled: dropped.
    CHECK_FALSE(picker.deliver(picker.current(), 2, bytes, nullptr, map));
    CHECK_FALSE(picker.pending());
    picker.request(9.0f, 9.0f, 8, 100, 100);
    std::vector<uint8_t> short_bytes(100, 0);
    CHECK_FALSE(picker.deliver(picker.current(), 3, short_bytes, nullptr, map));
    picker.request(9.0f, 9.0f, 8, 100, 100);
    const std::optional<Hit> hit = picker.deliver(picker.current(), 3, bytes, nullptr, map);
    REQUIRE(hit);
    CHECK(hit->item == 5);
    CHECK(hit->entity == e.entity);
}

// ---- Picks in a drawn frame ---------------------------------------------------------------------------

namespace {

// An ID target of w x h on the recording backend, every texel (0, 0) but those of `marks`.
struct FrameTarget {
    caliper_gfx::RecordingBackend gfx;
    caliper_gfx::Texture ids, depth;
    uint32_t w, h;
    std::vector<uint32_t> id;
    std::vector<float> dz;
    FrameTarget(uint32_t width, uint32_t height, bool with_depth) : w(width), h(height) {
        ids = gfx.create_texture({w, h, caliper_gfx::Format::RG32Uint,
                                  caliper_gfx::kRenderTarget | caliper_gfx::kCopySrcTex | caliper_gfx::kCopyDstTex, "frame ids"});
        if (with_depth)
            depth = gfx.create_texture({w, h, caliper_gfx::Format::Depth32Float,
                                        caliper_gfx::kRenderTarget | caliper_gfx::kCopySrcTex | caliper_gfx::kCopyDstTex, "frame depth"});
        id.assign(size_t(w) * h * 2, 0);
        dz.assign(size_t(w) * h, 1.0f);
    }
    void mark(uint32_t x, uint32_t y, uint32_t pick, uint32_t item, float d = 0.5f) {
        id[(size_t(y) * w + x) * 2] = pick;
        id[(size_t(y) * w + x) * 2 + 1] = item;
        dz[size_t(y) * w + x] = d;
    }
    void upload() {
        REQUIRE(gfx.write_texture(ids, id.data(), w * 8));
        if (depth) REQUIRE(gfx.write_texture(depth, dz.data(), w * 4));
    }
};

const auto kPrograms = [](uint32_t pick) { return pick >= 1000 && pick < 1004; };

}  // namespace

TEST_CASE("a frame pick reads the window of the frame's ID target and takes a program's texel nearest the pointer") {
    FrameTarget t(64, 48, false);
    t.mark(30, 20, 1001, 7);   // two texels from the pointer
    t.mark(33, 21, 1002, 9);   // one texel away: the nearest
    t.mark(32, 20, 5, 3);      // nearer still, but an entity's id, not a program's
    t.upload();
    FramePick pick(t.gfx);
    std::string why;
    REQUIRE_MESSAGE(pick.request(t.ids, t.depth, t.w, t.h, 32.5f, 20.5f, &why), why);
    FrameHit hit;
    PickPass::Result r = PickPass::Result::Pending;
    for (int i = 0; i < 4 && r == PickPass::Result::Pending; ++i) r = pick.poll(kPrograms, &hit);
    REQUIRE(r == PickPass::Result::Hit);
    CHECK(hit.pick_id == 1002);
    CHECK(hit.item == 9);
    CHECK(hit.depth == 1.0f);   // no depth target
    CHECK(hit.texel[0] == 8 + 1);
    CHECK(hit.texel[1] == 8 + 1);
    CHECK(pick.poll(kPrograms, &hit) == PickPass::Result::None);
}

TEST_CASE("decoding a frame's window takes the nearer depth between texels as near the pointer, and the entity wins when nearer") {
    const PickWindow w = pick_window(8.0f, 8.0f, 8, 16, 16);
    const WindowRegion r = window_region(w, 16, 16);
    std::vector<uint8_t> ids(16 * 16 * 8, 0), depth(16 * 16 * 4, 0);
    const auto put = [&](int x, int y, uint32_t pick, uint32_t item, float d) {
        const uint32_t texel[2] = {pick, item};
        std::memcpy(ids.data() + size_t(y * 16 + x) * 8, texel, 8);
        std::memcpy(depth.data() + size_t(y * 16 + x) * 4, &d, 4);
    };
    put(7, 8, 1001, 1, 0.6f);   // as near the pointer (8.0, 8.0) as the next, but deeper
    put(8, 8, 1002, 2, 0.4f);
    put(12, 8, 1003, 3, 0.1f);  // nearest in depth, but farther from the pointer
    const std::optional<FrameHit> h = decode_frame_pick(ids, &depth, w, r, kPrograms);
    REQUIRE(h);
    CHECK(h->pick_id == 1002);
    CHECK(h->item == 2);
    CHECK(h->depth == doctest::Approx(0.4f));
    Hit entity;
    entity.depth = 0.3f;
    CHECK(entity_hit_wins(entity, *h));
    entity.depth = 0.4f;
    CHECK_FALSE(entity_hit_wins(entity, *h));
}

TEST_CASE("a frame pick at the frame's corner reads only the part of the window on the target") {
    FrameTarget t(40, 30, false);
    t.mark(0, 0, 1000, 2);
    t.upload();
    const PickWindow w = pick_window(1.0f, 1.0f, 8, t.w, t.h);
    const WindowRegion r = window_region(w, t.w, t.h);
    CHECK(r.x == 0);
    CHECK(r.y == 0);
    CHECK(r.width == 9);
    CHECK(r.height == 9);
    CHECK(r.i0 == 7);
    CHECK(r.j0 == 7);
    FramePick pick(t.gfx);
    std::string why;
    REQUIRE_MESSAGE(pick.request(t.ids, {}, t.w, t.h, 1.0f, 1.0f, &why), why);
    FrameHit hit;
    PickPass::Result res = PickPass::Result::Pending;
    for (int i = 0; i < 4 && res == PickPass::Result::Pending; ++i) res = pick.poll(kPrograms, &hit);
    REQUIRE(res == PickPass::Result::Hit);
    CHECK(hit.pick_id == 1000);
    CHECK(hit.item == 2);
    CHECK(hit.depth == 1.0f);   // no depth target in 2-D
    CHECK(hit.texel[0] == 7);
    // Beyond the radius, or with no program's id, it misses.
    REQUIRE(pick.request(t.ids, {}, t.w, t.h, 20.0f, 20.0f, &why));
    res = PickPass::Result::Pending;
    for (int i = 0; i < 4 && res == PickPass::Result::Pending; ++i) res = pick.poll(kPrograms, &hit);
    CHECK(res == PickPass::Result::Miss);
}

TEST_CASE("a frame pick refuses a pointer off the frame or not finite, no ID target, and a size the target lacks") {
    FrameTarget t(16, 16, false);
    t.upload();
    FramePick pick(t.gfx);
    std::string why;
    CHECK_FALSE(pick.request(t.ids, {}, t.w, t.h, -1.0f, 4.0f, &why));
    CHECK(why == "the pointer is not on the frame");
    CHECK_FALSE(pick.request(t.ids, {}, t.w, t.h, 4.0f, 16.0f, &why));
    CHECK_FALSE(pick.request(t.ids, {}, t.w, t.h, std::nanf(""), 4.0f, &why));
    CHECK_FALSE(pick.request(t.ids, {}, t.w, t.h, 4.0f, INFINITY, &why));
    CHECK_FALSE(pick.request({}, {}, t.w, t.h, 4.0f, 4.0f, &why));
    CHECK(why == "the frame has no ID target");
    // A size larger than the texture: the layer refuses the region, and nothing is read.
    CHECK_FALSE(pick.request(t.ids, {}, 4096, 4096, 100.0f, 100.0f, &why));
    CHECK(why.rfind("no pick readback", 0) == 0);
    CHECK_FALSE(pick.pending());
    CHECK(pick.poll(kPrograms, nullptr) == PickPass::Result::None);
}

TEST_CASE("decoding a frame's window refuses short bytes, a missing test of ids and an empty region") {
    const PickWindow w = pick_window(8.0f, 8.0f, 8, 16, 16);
    const WindowRegion r = window_region(w, 16, 16);
    REQUIRE(r.width == 16);
    std::vector<uint8_t> ids(16 * 16 * 8, 0);
    const uint32_t texel[2] = {1000, 4};
    std::memcpy(ids.data() + (8 * 16 + 8) * 8, texel, 8);
    CHECK(decode_frame_pick(ids, nullptr, w, r, kPrograms));
    // Bytes one texel short, and a depth one texel short, which is left out.
    std::vector<uint8_t> short_ids(ids.begin(), ids.end() - 8);
    CHECK_FALSE(decode_frame_pick(short_ids, nullptr, w, r, kPrograms));
    std::vector<uint8_t> short_depth(16 * 16 * 4 - 4, 0);
    const std::optional<FrameHit> h = decode_frame_pick(ids, &short_depth, w, r, kPrograms);
    REQUIRE(h);
    CHECK(h->depth == 1.0f);
    CHECK_FALSE(decode_frame_pick(ids, nullptr, w, r, nullptr));
    CHECK_FALSE(decode_frame_pick(ids, nullptr, w, WindowRegion{}, kPrograms));
    // A window wholly off the target has no region.
    const WindowRegion off = window_region(pick_window(-40.0f, 5.0f, 8, 16, 16), 16, 16);
    CHECK(off.width == 0);
}
