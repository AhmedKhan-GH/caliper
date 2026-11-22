// World matrices: composition down a path in double, 2-D transforms, the camera's pixel-space
// matrix, hidden entities for non-finite poses, and the bounds that follow
// (src/host/scene/scene.h, SceneTree::update).
#include "scene/scene.h"
#include "scene_fixtures.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace caliper_scene;
using namespace scene_fixtures;

namespace {

void check_mat(const Mat4& a, const Mat4& b, double tol = 1e-12) {
    for (int i = 0; i < 16; ++i) CHECK(std::fabs(a.m[i] - b.m[i]) <= tol * std::max(1.0, std::fabs(b.m[i])));
}

bool near(Vec3 a, Vec3 b, double tol = 1e-9) {
    return std::fabs(a.x - b.x) <= tol && std::fabs(a.y - b.y) <= tol && std::fabs(a.z - b.z) <= tol;
}

Mat4 rot_y(double a) {
    Transform t;
    t.rotation[1] = std::sin(a / 2);
    t.rotation[3] = std::cos(a / 2);
    return transform_matrix(t);
}

}  // namespace

TEST_CASE("the parts form maps p to T * R * S * p, the quaternion normalised") {
    Transform t;
    t.translation[0] = 1;
    t.translation[1] = 2;
    t.translation[2] = 3;
    // 90 degrees about z, given unnormalised (scaled by 2).
    t.rotation[2] = 2 * std::sqrt(0.5);
    t.rotation[3] = 2 * std::sqrt(0.5);
    t.scale[0] = 2;
    t.scale[1] = 3;
    t.scale[2] = 4;
    const Mat4 m = transform_matrix(t);
    // (1, 0, 0) -> scaled (2, 0, 0) -> rotated (0, 2, 0) -> translated (1, 4, 3).
    CHECK(near(transform_point(m, {1, 0, 0}), {1, 4, 3}));
    CHECK(near(transform_point(m, {0, 1, 0}), {-2, 2, 3}));
    CHECK(near(transform_point(m, {0, 0, 1}), {1, 2, 7}));
    // A zero quaternion gives a matrix that is not finite.
    Transform z;
    z.rotation[3] = 0;
    CHECK_FALSE(finite(transform_matrix(z)));
}

TEST_CASE("the 2-D form: a counter-clockwise angle when y is up, then a translation") {
    Transform2D t;
    t.translation[0] = 10;
    t.translation[1] = -5;
    t.angle = kPi / 2;
    t.scale[0] = 2;
    t.scale[1] = 0.5;
    const Mat4 m = transform_matrix(t);
    CHECK(near(transform_point(m, {1, 0, 0}), {10, -3, 0}));
    CHECK(near(transform_point(m, {0, 1, 0}), {9.5, -5, 0}));
    CHECK(m.at(2, 2) == 1.0);
    CHECK(m.at(2, 3) == 0.0);
}

TEST_CASE("world matrices compose down the path, M1 * M2 * ... * Mn, in double") {
    SceneTree t;
    const EntityId l0 = t.ensure("arm/link0");
    const EntityId l1 = t.ensure("arm/link0/link1");
    const EntityId l2 = t.ensure("arm/link0/link1/link2");
    const Mat4 a = translation({0, 0, 0.5}) * rot_y(0.3);
    const Mat4 b = translation({0, 0, 0.4}) * rot_y(-0.7);
    const Mat4 c = translation({1e6, 0, 0.3}) * scaling({2, 2, 2});
    REQUIRE(t.set_transform(t.find("arm"), translation({1, 2, 3})));
    REQUIRE(t.set_transform(l0, a));
    REQUIRE(t.set_transform(l1, b));
    REQUIRE(t.set_transform(l2, c));
    t.update();
    const Mat4 want = translation({1, 2, 3}) * a * b * c;
    check_mat(t.world(l2), want);
    check_mat(t.world(l1), translation({1, 2, 3}) * a * b);
    // A missing transform is the identity.
    const EntityId bare = t.ensure("arm/link0/bare");
    t.update();
    check_mat(t.world(bare), t.world(l0));
    // A change rewrites the subtree only.
    REQUIRE(t.set_transform(l1, Mat4{}));
    t.update();
    check_mat(t.world(l2), translation({1, 2, 3}) * a * c);
    check_mat(t.world(l0), translation({1, 2, 3}) * a);
    // Floats from a device slot are read as given.
    float f[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1};
    REQUIRE(t.set_transform(l0, f));
    t.update();
    CHECK(near(transform_point(t.world(l0), {0, 0, 0}), {6, 8, 10}));
}

TEST_CASE("a non-finite pose hides its entity and its subtree, and a finite one brings them back") {
    SceneTree t;
    const EntityId a = t.ensure("a");
    const EntityId b = t.ensure("a/b");
    const EntityId other = t.ensure("other");
    REQUIRE(t.set_batch(b, points3(3)));
    REQUIRE(t.set_batch(other, points3(3)));
    Mat4 bad;
    bad.m[12] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE(t.set_transform(a, bad));
    t.update();
    CHECK_FALSE(t.drawable(a));
    CHECK_FALSE(t.drawable(b));
    CHECK_FALSE(t.pose_finite(b));
    CHECK(t.drawable(other));
    Transform zero_q;
    zero_q.rotation[3] = 0;
    REQUIRE(t.set_transform(other, transform_matrix(zero_q)));
    t.update();
    CHECK_FALSE(t.drawable(other));
    REQUIRE(t.set_transform(a, translation({1, 0, 0})));
    t.update();
    CHECK(t.drawable(b));
    // Hiding by the person hides the subtree without touching the pose.
    t.set_visible(a, false);
    t.update();
    CHECK_FALSE(t.drawable(b));
    CHECK(t.pose_finite(b));
    t.set_visible(a, true);
    t.update();
    CHECK(t.drawable(b));
}

TEST_CASE("a camera's children live in its pixel space: world = camera frame * pixel-to-plane * 2-D transforms") {
    SceneTree t(Space::k3D, Axis::PosZ, "m");
    const EntityId cam = t.ensure("rig/left");
    Pinhole pin = pinhole(640, 480, 500);
    pin.cx = 300;
    pin.cy = 200;
    pin.plane_distance = 0.5;
    const Mat4 pose = translation({0.1, -2.5, 1.2}) * rot_y(0.25);
    REQUIRE(t.set_transform(t.find("rig"), translation({1, 0, 0})));
    REQUIRE(t.set_transform(cam, pose));
    REQUIRE(t.set_camera(cam, pin));
    const EntityId marks = t.ensure("rig/left/marks");
    const EntityId inner = t.ensure("rig/left/marks/inner");
    Transform2D shift;
    shift.translation[0] = 20;
    shift.translation[1] = 10;
    Transform2D turn;
    turn.angle = kPi / 2;
    REQUIRE(t.set_transform(marks, transform_matrix(shift)));
    REQUIRE(t.set_transform(inner, transform_matrix(turn)));
    t.update();
    CHECK(t.pixel_space_of(inner) == cam);
    CHECK_FALSE(t.pixel_space_of(cam));
    // A pixel point (u, v) lies at ((u - cx) / fx * d, (v - cy) / fy * d, d) in the camera frame.
    const Mat4 cam_world = translation({1, 0, 0}) * pose;
    auto expect = [&](double u, double v) {
        return transform_point(cam_world, {(u - 300) / 500 * 0.5, (v - 200) / 500 * 0.5, 0.5});
    };
    CHECK(near(transform_point(t.world(marks), {0, 0, 0}), expect(20, 10)));
    CHECK(near(transform_point(t.world(marks), {640, 480, 0}), expect(660, 490)));
    // inner turns (1, 0) to (0, 1) in pixels, then marks shifts it.
    CHECK(near(transform_point(t.world(inner), {1, 0, 0}), expect(20, 11)));
    check_mat(t.pixel_local(inner), transform_matrix(shift) * transform_matrix(turn));
    // The "rub" convention flips y and z of the camera frame.
    pin.convention = Convention::Rub;
    REQUIRE(t.set_camera(cam, pin));
    t.update();
    const Vec3 p = transform_point(t.world(marks), {0, 0, 0});
    const Vec3 q = transform_point(cam_world, {(20.0 - 300) / 500 * 0.5, -(10.0 - 200) / 500 * 0.5, -0.5});
    CHECK(near(p, q));
}

TEST_CASE("the default plane distance is a twentieth of the scene's bounding radius, and follows it") {
    SceneTree t;
    const EntityId cam = t.ensure("cam");
    REQUIRE(t.set_camera(cam, pinhole()));
    const EntityId img = t.ensure("cam/img");
    Batch cloud = points3(2);
    cloud.bounds.add({-3, -4, 0});
    cloud.bounds.add({3, 4, 0});   // radius 5, with the camera centre at the origin inside
    REQUIRE(t.set_batch(t.ensure("cloud"), cloud));
    t.update();
    CHECK(t.plane_distance(cam) == doctest::Approx(0.25));
    CHECK(transform_point(t.world(img), {320, 240, 0}).z == doctest::Approx(0.25));
    cloud.bounds = Aabb{};
    cloud.bounds.add({-30, -40, 0});
    cloud.bounds.add({30, 40, 0});
    REQUIRE(t.set_batch(t.find("cloud"), cloud));
    t.update();
    CHECK(t.plane_distance(cam) == doctest::Approx(2.5));
    CHECK(transform_point(t.world(img), {320, 240, 0}).z == doctest::Approx(2.5));
}

TEST_CASE("bounds: per batch from the producer or a scan, by the eight corners into world space") {
    SceneTree t;
    const EntityId e = t.ensure("box");
    Batch b = points3(2);
    b.bounds.add({-1, -1, -1});
    b.bounds.add({1, 1, 1});
    REQUIRE(t.set_batch(e, b));
    Transform tr;
    tr.translation[0] = 10;
    tr.rotation[2] = std::sin(kPi / 8);   // 45 degrees about z
    tr.rotation[3] = std::cos(kPi / 8);
    REQUIRE(t.set_transform(e, transform_matrix(tr)));
    t.update();
    const Aabb w = t.scene_bounds();
    REQUIRE(w.valid);
    CHECK(w.lo[0] == doctest::Approx(10 - std::sqrt(2.0)));
    CHECK(w.hi[0] == doctest::Approx(10 + std::sqrt(2.0)));
    CHECK(w.lo[2] == doctest::Approx(-1));
    // No producer bounds: a scan of the CPU view, skipping non-finite rows.
    const float pts[] = {0, 0, 0, 2, 4, -6, NAN, 0, 0, -1, 1, 1};
    const EntityId s = t.ensure("scan");
    REQUIRE(t.set_batch(s, points3(4, pts)));
    t.update();
    const Aabb sb = t.world_bounds(s);
    REQUIRE(sb.valid);
    CHECK(sb.lo[0] == -1);
    CHECK(sb.hi[1] == 4);
    CHECK(sb.lo[2] == -6);
    // Arrows cover their tips; boxes their half sizes.
    const float origins[] = {0, 0, 0};
    const float vectors[] = {0, 5, 0};
    Batch arrows;
    arrows.primitive = Primitive::Arrows;
    arrows.arrays.push_back(array(Role::Origins, 1, 12, VertexFormat::F32x3, 0, origins));
    arrows.arrays.push_back(array(Role::Vectors, 1, 12, VertexFormat::F32x3, 0, vectors));
    REQUIRE(t.set_batch(t.ensure("arrows"), arrows));
    t.update();
    CHECK(t.world_bounds(t.find("arrows")).hi[1] == 5);
    // A scene's bounds include its cameras' centres and leave out their pixel spaces.
    SceneTree c;
    const EntityId cam = c.ensure("cam");
    REQUIRE(c.set_transform(cam, translation({0, 0, -10})));
    REQUIRE(c.set_camera(cam, pinhole()));
    REQUIRE(c.set_batch(c.ensure("cam/marks"), points2(3)));
    c.update();
    const Aabb cb = c.scene_bounds();
    REQUIRE(cb.valid);
    CHECK(cb.lo[2] == -10);
    CHECK(cb.hi[2] == -10);
    // A hidden entity leaves the bounds.
    t.set_visible(t.find("arrows"), false);
    t.update();
    CHECK(t.scene_bounds().hi[1] == 4);
}

TEST_CASE("2-D scenes: transforms in the plane and images one scene unit per pixel") {
    SceneTree t(Space::k2D);
    const EntityId e = t.ensure("img");
    Transform2D tr;
    tr.translation[0] = 5;
    REQUIRE(t.set_transform(e, transform_matrix(tr)));
    Batch img;
    img.primitive = Primitive::Image;
    img.texture = caliper_gfx::Texture{9};
    img.image_width = 64;
    img.image_height = 32;
    REQUIRE(t.set_batch(e, img));
    t.update();
    const Aabb b = t.scene_bounds();
    REQUIRE(b.valid);
    CHECK(b.lo[0] == 5);
    CHECK(b.hi[0] == 69);
    CHECK(b.hi[1] == 32);
    CHECK(b.lo[2] == 0);
}
