// test_frustum_match.mm: the match of the library piece frustum@1 with the built-in frustum of a
// camera entity in the scene view (design/programmable-graphics.md, Library Pieces), through
// tests/image_match/image_match_rig.h. The piece's derive half runs in the producer runtime over a
// pose and a pinhole tensor, its draw half in the draw runtime with the scene view's camera, and the
// built-in draws the same camera as protocol 3.2 messages. The cases: a camera that faces the view
// at a slant in each convention, a wide pinhole off its centre with a far plane, and a camera that
// looks away, whose frustum crosses the near plane. The pixels of the room that gives the scene its
// bounds are masked; every other pixel follows the rules of the match tests, and a pick at a pixel
// of the piece's frustum gives the camera's frustum in the built-in. Without a Metal device the test
// skips (exit 77).
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "../image_match/image_match_rig.h"

namespace {

using namespace image_match;
using caliper_net::Json;

struct Pinhole {
    double fx = 8, fy = 8, cx = 8, cy = 6, width = 16, height = 12, plane = 0.9;
};

// A room large enough to hold the camera's centre, so that the scene's bounds, and thus the
// view's fit, are the same with the camera and without it: the room alone is the mask.
pf::Fixture big_room(const std::string& name) {
    pf::Fixture f;
    f.name = name;
    f.info = {CALIPER_NET_SPACE_3D, "+z", "m"};
    std::vector<float> cen = {0.0f, 0.0f, 0.0f}, half = {3.0f, 3.0f, 3.0f};
    wire::SceneUpdate box = f.update("room/box", CALIPER_NET_PART_BOXES, true);
    box.params.set("color", pf::rgba(pf::RGBA(240, 180, 60)));
    box.arrays.push_back(f.section("centers", CALIPER_NET_DTYPE_F32, {1, 3}, cen));
    box.arrays.push_back(f.section("half_sizes", CALIPER_NET_DTYPE_F32, {1, 3}, half));
    f.messages.push_back(box);
    return f;
}

// The room with a camera "rig/cam" at `pose` with pinhole `p`.
pf::Fixture camera_scene(const std::vector<float>& pose, const Pinhole& p, const char* convention) {
    pf::Fixture f = big_room("cam");
    f.transform("rig", pf::translate(0, 0, 0), true);
    f.transform("rig/cam", pose, true);
    wire::SceneUpdate cam = f.update("rig/cam", CALIPER_NET_PART_CAMERA, true);
    cam.params.set("fx", Json(p.fx));
    cam.params.set("fy", Json(p.fy));
    cam.params.set("cx", Json(p.cx));
    cam.params.set("cy", Json(p.cy));
    cam.params.set("width", Json(int64_t(p.width)));
    cam.params.set("height", Json(int64_t(p.height)));
    cam.params.set("convention", Json(convention));
    cam.params.set("plane_distance", Json(p.plane));
    f.messages.push_back(cam);
    return f;
}

struct Outcome {
    Comparison c;
    std::size_t lit = 0, probes = 0, agree = 0;
};

Outcome run_case(Metal& m, const cs::Piece& piece, const std::string& name, const std::vector<float>& pose,
                 const Pinhole& p, bool rub) {
    Outcome o;
    gfx::Backend& g = *m.backend;
    std::string why;
    const std::vector<Tensor> tensors = {
        pose_tensor(pose),
        tensor("pinhole", {7}, std::vector<float>{float(p.fx), float(p.fy), float(p.cx), float(p.cy), float(p.width),
                                                  float(p.height), float(p.plane)})};
    const std::map<std::string, double> params = {{"convention", rub ? 1.0 : 0.0}};
    std::vector<Version> versions;
    REQUIRE_MESSAGE(run_derive(m.device, piece, tensors, params, &versions, &why), why);
    const pf::Fixture scene = camera_scene(pose, p, rub ? CALIPER_NET_CONVENTION_RUB : CALIPER_NET_CONVENTION_RDF);
    Builtin b(g);
    REQUIRE_MESSAGE(b.draw(scene, &why), why);
    CHECK_MESSAGE(b.refusal.empty(), b.refusal);
    const std::vector<std::uint8_t> builtin = b.pixels();
    Builtin plain(g);
    REQUIRE_MESSAGE(plain.draw(big_room("cam"), &why), why);
    const std::vector<std::uint8_t> mask = mask_of(plain.pixels());
    PieceDraw d;
    REQUIRE_MESSAGE(draw_piece(g, piece, versions, b.inputs(), params, &d), d.error);
    o.c = compare(d.color, d.ids, builtin, &mask, 0);
    for (int y = 0; y < int(kSize); ++y)
        for (int x = 0; x < int(kSize); ++x)
            o.lit += identity_of(d.ids, x, y) != 0 && !mask[std::size_t(y) * kSize + std::size_t(x)];
    MESSAGE(name, ": ", o.lit, " frustum pixels outside the room, ", o.c.edge, " edge pixels, ", o.c.masked,
            " masked, ", o.c.failed, " outside the rules");
    CHECK_MESSAGE(o.c.failed == 0, name, ": ", o.c.first);
    capture_pair(name, builtin, d.color);

    // Picks: frustum pixels of the piece away from the room, where both pictures show the same
    // color; the built-in gives the camera's frustum, and the piece's item map the camera, element 0.
    for (int y = 2; y < int(kSize) - 2 && o.probes < 12; y += 3)
        for (int x = 2; x < int(kSize) - 2 && o.probes < 12; x += 3) {
            if (identity_of(d.ids, x, y) != 1 || pixel_of(d.color, x, y) != pixel_of(builtin, x, y)) continue;
            bool clear = true;
            for (int dy = -9; dy <= 9 && clear; ++dy)
                for (int dx = -9; dx <= 9 && clear; ++dx)
                    clear = !mask[std::size_t(std::clamp(y + dy, 0, int(kSize) - 1)) * kSize +
                                  std::size_t(std::clamp(x + dx, 0, int(kSize) - 1))];
            if (!clear) continue;
            ++o.probes;
            const std::optional<sc::Hit> hit = b.pick(x, y);
            const bool same = hit && hit->part == sc::DrawPart::Frustum && b.view->tree().path(hit->entity) == "rig/cam";
            CHECK_MESSAGE(same, name, ": the built-in pick at (", x, ", ", y, ") is not the camera's frustum");
            const std::optional<cd::PickedItem> picked = d.rt->pick(0);
            REQUIRE(picked.has_value());
            CHECK(picked->fields.size() == 1);
            o.agree += same && picked->fields.size() == 1 && picked->fields[0].second == 0.0;
        }
    CHECK_MESSAGE(o.probes >= 4, name, ": ", o.probes, " frustum pixels to pick");
    return o;
}

}  // namespace

TEST_CASE("frustum match: piece.check accepts frustum@1") {
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("frustum", &piece, &why), why);
    const Checked c = check_piece(piece);
    CHECK_MESSAGE(c.ok, c.messages);
    CHECK(c.passes == 2);
}

TEST_CASE("frustum match: frustum@1 draws the frustum of a camera entity as the built-in does") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("frustum", &piece, &why), why);
    const std::vector<float> facing = facing_pose(m);
    run_case(m, piece, "camera-rdf", facing, Pinhole{}, false);
    run_case(m, piece, "camera-rub", facing, Pinhole{}, true);
    Pinhole wide;
    wide.fx = 12;
    wide.fy = 9;
    wide.cx = 3;
    wide.cy = 10;
    wide.width = 32;
    wide.height = 18;
    wide.plane = 1.6;
    run_case(m, piece, "camera-wide-far", facing, wide, false);
}

CALIPER_GPU_TEST_MAIN()
