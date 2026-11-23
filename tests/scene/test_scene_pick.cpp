// Picking: the pick pass's draw list (pick entries into a 16 x 16 RG32Uint window reached by
// zooming clip space), the map from pick id to entity and primitive, and the decoding of
// read-back ID texels into hits (src/host/scene/pick.h); and the pick of a program, read from the
// ID target of a frame that the view drew, with the refusals of its pointer, window and bytes.
#include "scene/pick.h"
#include "gfx/recording_backend.h"
#include "scene_fixtures.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>

using namespace caliper_scene;
using namespace scene_fixtures;

TEST_CASE("the pick window: 2r x 2r pixels with the pointer's pixel at texel (r, r)") {
    const PickWindow w = pick_window(100.7f, 50.2f, 8, 400, 300);
    CHECK(w.width == 16);
    CHECK(w.height == 16);
    CHECK(w.x0 == 92);
    CHECK(w.y0 == 42);
    CHECK(int(std::floor(w.px)) - w.x0 == 8);
}

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
    PickResult r = PickResult::Pending;
    for (int i = 0; i < 4 && r == PickResult::Pending; ++i) r = pick.poll(kPrograms, &hit);
    REQUIRE(r == PickResult::Hit);
    CHECK(hit.pick_id == 1002);
    CHECK(hit.item == 9);
    CHECK(hit.depth == 1.0f);   // no depth target
    CHECK(hit.texel[0] == 8 + 1);
    CHECK(hit.texel[1] == 8 + 1);
    CHECK(pick.poll(kPrograms, &hit) == PickResult::None);
}

TEST_CASE("decoding a frame's window takes the nearer depth between texels as near the pointer") {
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
    PickResult res = PickResult::Pending;
    for (int i = 0; i < 4 && res == PickResult::Pending; ++i) res = pick.poll(kPrograms, &hit);
    REQUIRE(res == PickResult::Hit);
    CHECK(hit.pick_id == 1000);
    CHECK(hit.item == 2);
    CHECK(hit.depth == 1.0f);   // no depth target in 2-D
    CHECK(hit.texel[0] == 7);
    // Beyond the radius, or with no program's id, it misses.
    REQUIRE(pick.request(t.ids, {}, t.w, t.h, 20.0f, 20.0f, &why));
    res = PickResult::Pending;
    for (int i = 0; i < 4 && res == PickResult::Pending; ++i) res = pick.poll(kPrograms, &hit);
    CHECK(res == PickResult::Miss);
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
    CHECK(pick.poll(kPrograms, nullptr) == PickResult::None);
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
