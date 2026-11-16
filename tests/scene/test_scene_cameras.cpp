// The cameras: the orbit's azimuth convention for every up axis, the up-axis switch, the 2-D pan
// and zoom, looking through a pinhole, fitting, the starting camera and reset
// (src/host/scene/camera.h).
#include "scene/camera.h"
#include "scene_fixtures.h"

#include <doctest/doctest.h>

#include <cmath>

using namespace caliper_scene;
using namespace scene_fixtures;

namespace {

constexpr double kTol = 1e-6;
const Axis kAxes[6] = {Axis::PosX, Axis::NegX, Axis::PosY, Axis::NegY, Axis::PosZ, Axis::NegZ};

bool near(Vec3 a, Vec3 b, double tol = kTol) {
    return std::fabs(a.x - b.x) <= tol && std::fabs(a.y - b.y) <= tol && std::fabs(a.z - b.z) <= tol;
}
bool near(const Mat4& a, const Mat4& b, double tol = kTol) {
    for (int i = 0; i < 16; ++i)
        if (!(std::fabs(a.m[i] - b.m[i]) <= tol)) return false;
    return true;
}

// The design's statement written out per axis, independent of orbit_basis: u the up axis, a the
// first of +x, +y and +z perpendicular to it, b = a x u.
void stated_basis(Axis up, Vec3* u, Vec3* a, Vec3* b) {
    switch (up) {
        case Axis::PosY: *u = {0, 1, 0}; *a = {1, 0, 0}; *b = {0, 0, 1}; break;
        case Axis::NegY: *u = {0, -1, 0}; *a = {1, 0, 0}; *b = {0, 0, -1}; break;
        case Axis::PosZ: *u = {0, 0, 1}; *a = {1, 0, 0}; *b = {0, -1, 0}; break;
        case Axis::NegZ: *u = {0, 0, -1}; *a = {1, 0, 0}; *b = {0, 1, 0}; break;
        case Axis::PosX: *u = {1, 0, 0}; *a = {0, 1, 0}; *b = {0, 0, -1}; break;
        case Axis::NegX: *u = {-1, 0, 0}; *a = {0, 1, 0}; *b = {0, 0, 1}; break;
    }
}

Vec3 stated_eye(const Orbit& o) {
    Vec3 u, a, b;
    stated_basis(o.up, &u, &a, &b);
    const Vec3 e = (a * std::cos(o.azimuth) + b * std::sin(o.azimuth)) * std::cos(o.elevation) +
                   u * std::sin(o.elevation);
    return Vec3{o.target[0], o.target[1], o.target[2]} + e * o.distance;
}

double deg(double d) { return d * kPi / 180.0; }

// The NDC of a world point under proj * view.
Vec3 ndc(const CameraMatrices& m, Vec3 p) {
    double h[4];
    transform_h(m.proj * m.view, p, h);
    return {h[0] / h[3], h[1] / h[3], h[2] / h[3]};
}

}  // namespace

TEST_CASE("the orbit's view matrix is a look-at from the stated eye, for a grid of angles and every up axis") {
    int checked = 0;
    for (Axis up : kAxes)
        for (double az = -180; az <= 360; az += 37.5)
            for (double el = -89; el <= 89; el += 22.25) {
                Orbit o;
                o.up = up;
                o.target[0] = 1.5;
                o.target[1] = -2;
                o.target[2] = 0.25;
                o.distance = 7.5;
                o.azimuth = deg(az);
                o.elevation = deg(el);
                const Vec3 eye = stated_eye(o);
                CHECK(near(orbit_eye(o), eye));
                Vec3 u, a, b;
                stated_basis(up, &u, &a, &b);
                CHECK(near(orbit_view(o), look_at(eye, {1.5, -2, 0.25}, u)));
                ++checked;
            }
    CHECK(checked == 6 * 15 * 9);
}

TEST_CASE("the azimuth convention: +y up puts 0 on +x and 90 on +z; +z up puts 90 on -y") {
    Orbit o;
    o.distance = 1;
    o.elevation = 0;
    o.azimuth = 0;
    CHECK(near(orbit_eye(o), {1, 0, 0}));
    o.azimuth = deg(90);
    CHECK(near(orbit_eye(o), {0, 0, 1}));
    o.up = Axis::PosZ;
    o.azimuth = 0;
    CHECK(near(orbit_eye(o), {1, 0, 0}));
    o.azimuth = deg(90);
    CHECK(near(orbit_eye(o), {0, -1, 0}));
    // The points view's eye (live_views_draw.cpp, eye_of) for its defaults.
    Orbit pv;
    pv.distance = 3;
    const double ce = std::cos(kDefaultElevation), se = std::sin(kDefaultElevation);
    const double ca = std::cos(kDefaultAzimuth), sa = std::sin(kDefaultAzimuth);
    CHECK(near(orbit_eye(pv), {3 * ce * ca, 3 * se, 3 * ce * sa}));
    // Elevation lifts the eye along the up axis.
    o.elevation = deg(30);
    CHECK(dot(orbit_eye(o), axis_vector(Axis::PosZ)) == doctest::Approx(0.5));
}

TEST_CASE("the up-axis switch keeps target, distance and eye, rolls to the new axis, and round-trips") {
    for (Axis from : kAxes)
        for (Axis to : kAxes)
            for (double az : {-120.0, 10.0, 75.0, 200.0})
                for (double el : {-60.0, -5.0, 0.0, 33.0, 70.0}) {
                    Orbit o;
                    o.up = from;
                    o.target[0] = -3;
                    o.target[1] = 4;
                    o.target[2] = 2;
                    o.distance = 12;
                    o.azimuth = deg(az);
                    o.elevation = deg(el);
                    const Vec3 eye0 = orbit_eye(o);
                    const Mat4 view0 = orbit_view(o);
                    Orbit s = o;
                    switch_up(s, to);
                    const Vec3 dir = orbit_direction(o);
                    const double along = dot(dir, axis_vector(to));
                    if (std::fabs(along) > std::cos(deg(1.0))) {
                        // Within 1 degree of the new axis: held at 89 degrees.
                        CHECK(std::fabs(s.elevation) == doctest::Approx(kMaxElevation));
                        continue;
                    }
                    CHECK(near(orbit_eye(s), eye0));
                    CHECK(s.distance == o.distance);
                    CHECK(near(Vec3{s.target[0], s.target[1], s.target[2]}, {-3, 4, 2}));
                    // The view's up row is the new axis projected perpendicular to the line of sight.
                    const Mat4 v = orbit_view(s);
                    const Vec3 up_row{v.at(1, 0), v.at(1, 1), v.at(1, 2)};
                    const Vec3 f = dir * -1.0;
                    const Vec3 want = normalize(axis_vector(to) - f * dot(axis_vector(to), f));
                    CHECK(near(up_row, want));
                    // And back: the first view matrix again.
                    switch_up(s, from);
                    CHECK(near(orbit_view(s), view0));
                }
    // An eye on the new axis is held at 89 degrees.
    Orbit top;
    top.up = Axis::PosY;
    top.elevation = deg(0);
    top.azimuth = deg(90);   // the eye on +z
    switch_up(top, Axis::PosZ);
    CHECK(top.elevation == doctest::Approx(kMaxElevation));
}

TEST_CASE("pan and zoom: orthographic with y up or down, depth 0.5") {
    PanZoom p;
    p.center[0] = 10;
    p.center[1] = 20;
    p.half_height = 5;
    p.y_down = false;
    const Mat4 up = panzoom_projection(p, 2.0);
    double h[4];
    transform_h(up, {20, 25, 7}, h);   // the top-right corner: x + 2 * 5, y + 5
    CHECK(h[0] == doctest::Approx(1));
    CHECK(h[1] == doctest::Approx(1));
    CHECK(h[2] == doctest::Approx(0.5));
    CHECK(h[3] == doctest::Approx(1));
    p.y_down = true;
    transform_h(panzoom_projection(p, 2.0), {20, 25, 0}, h);   // y grows down: the bottom-right corner
    CHECK(h[0] == doctest::Approx(1));
    CHECK(h[1] == doctest::Approx(-1));
    SceneTree flat(Space::k2D);
    const CameraMatrices m = camera_matrices(ViewCamera(p), flat, 200, 100);
    REQUIRE(m.ok);
    CHECK(near(ndc(m, {0, 15, 0}), {-1, 1, 0.5}));
}

TEST_CASE("the pinhole projection maps the image corners to the clip corners, letterboxed") {
    Pinhole c = pinhole(640, 480, 500);
    c.cx = 310;
    c.cy = 250;
    const double zn = 0.1, zf = 50, pan[2] = {0, 0};
    // Same aspect: the corners map exactly.
    const Mat4 p = pinhole_projection(c, zn, zf, 640.0 / 480.0, 1.0, pan);
    auto project = [&](const Mat4& m, double u, double v, double z) {
        const Vec3 X{(u - c.cx) / c.fx * z, (v - c.cy) / c.fy * z, z};
        double h[4];
        transform_h(m, X, h);
        return Vec3{h[0] / h[3], h[1] / h[3], h[2] / h[3]};
    };
    CHECK(near(project(p, 0, 0, 3), {-1, 1, project(p, 0, 0, 3).z}));
    CHECK(near(project(p, 640, 0, 3), {1, 1, project(p, 640, 0, 3).z}));
    CHECK(near(project(p, 640, 480, 3), {1, -1, project(p, 640, 480, 3).z}));
    CHECK(near(project(p, 0, 480, 7), {-1, -1, project(p, 0, 480, 7).z}));
    CHECK(project(p, 100, 100, zn).z == doctest::Approx(0.0));
    CHECK(project(p, 100, 100, zf).z == doctest::Approx(1.0));
    // A wider view letterboxes x; a taller view y.
    const Mat4 wide = pinhole_projection(c, zn, zf, 2.0, 1.0, pan);
    CHECK(near(project(wide, 640, 0, 2), {(640.0 / 480.0) / 2.0, 1, project(wide, 640, 0, 2).z}));
    const Mat4 tall = pinhole_projection(c, zn, zf, 1.0, 1.0, pan);
    CHECK(near(project(tall, 640, 480, 2), {1, -0.75, project(tall, 640, 480, 2).z}));
    // Zoom and pan act in the view's normalised units.
    const double pan2[2] = {0.25, -0.5};
    const Mat4 zp = pinhole_projection(c, zn, zf, 640.0 / 480.0, 2.0, pan2);
    CHECK(near(project(zp, 0, 0, 2), {-2 + 0.25, 2 - 0.5, project(zp, 0, 0, 2).z}));
}

TEST_CASE("looking through a camera entity: its pose, the projection, the flat pixel space and the scissor") {
    SceneTree t;
    const EntityId cam = t.ensure("rig/cam");
    Transform tr;
    tr.translation[0] = 1;
    tr.translation[1] = -2.5;
    tr.translation[2] = 1.2;
    tr.rotation[0] = -std::sqrt(0.5);   // camera z along world +y, as in the worked example
    tr.rotation[3] = std::sqrt(0.5);
    REQUIRE(t.set_transform(cam, transform_matrix(tr)));
    Pinhole c = pinhole(640, 480, 525);
    REQUIRE(t.set_camera(cam, c));
    const EntityId img = t.ensure("rig/cam/rgb");
    Batch cloud = points3(1);
    cloud.bounds.add({-2, -2, 0});
    cloud.bounds.add({2, 2, 1});
    REQUIRE(t.set_batch(t.ensure("room"), cloud));
    t.update();
    ThroughPinhole through;
    through.camera = cam;
    const CameraMatrices m = camera_matrices(ViewCamera(through), t, 1280, 480);   // wider than 4:3
    REQUIRE(m.ok);
    CHECK(m.through == cam);
    CHECK(near(m.eye, {1, -2.5, 1.2}));
    // A world point straight ahead (+y) projects to the principal point, the view's centre.
    CHECK(near(ndc(m, {1, 1, 1.2}), {0, 0, ndc(m, {1, 1, 1.2}).z}));
    // A pixel-space point on the plane projects where the flat pixel space puts it.
    const double a = 640.0 / 480.0 / (1280.0 / 480.0);
    for (double u : {0.0, 100.0, 640.0})
        for (double v : {0.0, 240.0, 480.0}) {
            const Vec3 world = transform_point(t.world(img), {u, v, 0});
            double h[4];
            transform_h(m.pixel_to_clip, {u, v, 0}, h);
            const Vec3 flat{h[0] / h[3], h[1] / h[3], h[2] / h[3]};
            CHECK(near(Vec3{ndc(m, world).x, ndc(m, world).y, 0}, Vec3{flat.x, flat.y, 0}));
            CHECK(flat.x == doctest::Approx((2 * u / 640 - 1) * a));
            CHECK(flat.z == doctest::Approx(0.5));
        }
    // The scissor is the letterbox's inside: 640 of 1280 pixels, centred.
    CHECK(m.scissor[0] == 320);
    CHECK(m.scissor[1] == 0);
    CHECK(m.scissor[2] == 640);
    CHECK(m.scissor[3] == 480);
    // The bounds lie between the near and far planes.
    for (int k = 0; k < 8; ++k) {
        const Vec3 corner{(k & 1) ? 2.0 : -2.0, (k & 2) ? 2.0 : -2.0, (k & 4) ? 1.0 : 0.0};
        const double z = ndc(m, corner).z;
        CHECK(z >= 0.0);
        CHECK(z <= 1.0);
    }
    // No camera: not ok.
    through.camera = t.find("room");
    CHECK_FALSE(camera_matrices(ViewCamera(through), t, 100, 100).ok);
}

TEST_CASE("the orbit's projection: 45 degrees and the points view's near and far planes") {
    SceneTree t;
    Batch b = points3(1);
    b.bounds.add({-1, -2, -3});
    b.bounds.add({1, 2, 3});
    REQUIRE(t.set_batch(t.ensure("p"), b));
    t.update();
    ViewCamera v = Orbit{};
    fit(v, t.scene_bounds(), 1.5);
    const Orbit& o = std::get<Orbit>(v);
    const double r = std::sqrt(1.0 + 4.0 + 9.0);
    CHECK(o.distance == doctest::Approx(r / std::sin(std::min(kOrbitFovY / 2, std::atan(std::tan(kOrbitFovY / 2) * 1.5))) * 1.15));
    const CameraMatrices m = camera_matrices(v, t, 300, 200);
    const double d = o.distance;
    const Mat4 want = perspective_z01(kOrbitFovY, 1.5, std::max(d - 2 * r, d * 1e-3), d + 2 * r);
    CHECK(near(m.proj, want, 1e-9));
    // Every corner of the bounds is in the view and between the planes.
    for (int k = 0; k < 8; ++k) {
        const Vec3 corner{(k & 1) ? 1.0 : -1.0, (k & 2) ? 2.0 : -2.0, (k & 4) ? 3.0 : -3.0};
        const Vec3 n = ndc(m, corner);
        CHECK(std::fabs(n.x) <= 1.0);
        CHECK(std::fabs(n.y) <= 1.0);
        CHECK(n.z >= 0.0);
        CHECK(n.z <= 1.0);
    }
    double zn, zf;
    near_far(1.0, 10.0, &zn, &zf);   // the eye inside the bounds: the floor of d / 1000
    CHECK(zn == doctest::Approx(1e-3));
    CHECK(zf == doctest::Approx(21));
}

TEST_CASE("fits: the orbit frames the bounding sphere and keeps its angles; pan and zoom; a pinhole resets") {
    Aabb b;
    b.add({0, 0, 0});
    b.add({8, 2, 0});
    ViewCamera v = Orbit{};
    std::get<Orbit>(v).azimuth = 1.0;
    fit(v, b, 1.0);
    CHECK(std::get<Orbit>(v).target[0] == 4);
    CHECK(std::get<Orbit>(v).azimuth == 1.0);
    ViewCamera p = PanZoom{};
    fit(p, b, 2.0);
    CHECK(std::get<PanZoom>(p).center[0] == 4);
    CHECK(std::get<PanZoom>(p).center[1] == 1);
    CHECK(std::get<PanZoom>(p).half_height == doctest::Approx(std::max(1.0, 4.0 / 2.0) * 1.08));
    fit(p, b, 8.0);
    CHECK(std::get<PanZoom>(p).half_height == doctest::Approx(1.0 * 1.08));
    ThroughPinhole tp;
    tp.zoom = 3;
    tp.pan[0] = 0.5;
    ViewCamera t = tp;
    fit(t, b, 1.0);
    CHECK(std::get<ThroughPinhole>(t).zoom == 1.0);
    CHECK(std::get<ThroughPinhole>(t).pan[0] == 0.0);
    // No bounds: the box from -1 to 1, as the points view.
    ViewCamera e = Orbit{};
    fit(e, Aabb{}, 1.0);
    CHECK(std::get<Orbit>(e).distance == doctest::Approx(std::sqrt(3.0) / std::sin(kOrbitFovY / 2) * 1.15));
}

TEST_CASE("the starting camera: both forms, both up modes, keys filled from the fit, and reset") {
    SceneTree t(Space::k3D, Axis::PosZ, "m");
    Batch b = points3(1);
    b.bounds.add({-2, -2, 0});
    b.bounds.add({2, 2, 1});
    REQUIRE(t.set_batch(t.ensure("room"), b));
    const EntityId cam = t.ensure("rig/left");
    REQUIRE(t.set_camera(cam, pinhole()));
    t.update();
    // An empty dict is the fit at the default angles, read against the scene's up.
    StartingCamera none;
    StartResult r = starting_camera(none, t, 1.0);
    REQUIRE(std::holds_alternative<Orbit>(r.camera));
    Orbit o = std::get<Orbit>(r.camera);
    CHECK(o.up == Axis::PosZ);
    CHECK(o.azimuth == doctest::Approx(deg(54.4309905)).epsilon(1e-8));
    CHECK(o.elevation == doctest::Approx(deg(24.0642274)).epsilon(1e-8));
    ViewCamera fitted = Orbit{};
    std::get<Orbit>(fitted).up = Axis::PosZ;
    fit(fitted, t.scene_bounds(), 1.0);
    CHECK(o.distance == doctest::Approx(std::get<Orbit>(fitted).distance));
    // Given keys are used as given; the view matrix is the look-at from the stated eye.
    StartingCamera s;
    s.form = StartingCamera::Form::Orbit;
    s.target = Vec3{0, 0, 0.5};
    s.azimuth_deg = 30;
    s.elevation_deg = 25;
    r = starting_camera(s, t, 1.0);
    o = std::get<Orbit>(r.camera);
    CHECK(o.target[2] == 0.5);
    CHECK(o.distance == doctest::Approx(std::get<Orbit>(fitted).distance));
    Vec3 u, a, bb;
    stated_basis(Axis::PosZ, &u, &a, &bb);
    CHECK(near(orbit_view(o), look_at(stated_eye(o), {0, 0, 0.5}, u)));
    s.distance = 4;
    CHECK(std::get<Orbit>(starting_camera(s, t, 1.0).camera).distance == 4);
    // The entity form looks through the camera, and falls back to the fit without one.
    StartingCamera through;
    through.form = StartingCamera::Form::Entity;
    through.entity = "rig/left";
    r = starting_camera(through, t, 1.0);
    REQUIRE(std::holds_alternative<ThroughPinhole>(r.camera));
    CHECK(std::get<ThroughPinhole>(r.camera).camera == cam);
    through.entity = "room";
    r = starting_camera(through, t, 1.0);
    CHECK(r.fell_back);
    CHECK(std::holds_alternative<Orbit>(r.camera));
    // A 2-D scene opens on its fit; y down by default.
    SceneTree flat(Space::k2D);
    r = starting_camera(none, flat, 1.0);
    REQUIRE(std::holds_alternative<PanZoom>(r.camera));
    CHECK(std::get<PanZoom>(r.camera).y_down);
    CHECK_FALSE(r.complete);   // no bounds yet
    // Refusals.
    std::string err;
    StartingCamera bad = s;
    bad.elevation_deg = 89.5;
    CHECK_FALSE(validate_starting_camera(bad, Space::k3D, &err));
    bad = s;
    bad.distance = 0;
    CHECK_FALSE(validate_starting_camera(bad, Space::k3D, &err));
    bad = s;
    bad.azimuth_deg = NAN;
    CHECK_FALSE(validate_starting_camera(bad, Space::k3D, &err));
    bad = through;
    bad.entity = "a//b";
    CHECK_FALSE(validate_starting_camera(bad, Space::k3D, &err));
    CHECK_FALSE(validate_starting_camera(s, Space::k2D, &err));
    CHECK(validate_starting_camera(none, Space::k2D, &err));
    CHECK(validate_starting_camera(s, Space::k3D, &err));
}

TEST_CASE("a view's camera state: a moved camera survives a new starting camera; reset returns to it") {
    SceneTree t(Space::k3D, Axis::PosY, "m");
    ViewState vs;
    StartingCamera s;
    s.form = StartingCamera::Form::Orbit;
    s.azimuth_deg = 10;
    vs.set_starting(s, t, 1.0);
    // Keys left out are filled once the scene has bounds.
    Batch b = points3(1);
    b.bounds.add({10, 10, 10});
    b.bounds.add({12, 12, 12});
    REQUIRE(t.set_batch(t.ensure("p"), b));
    t.update();
    vs.update(t, 1.0);
    CHECK(std::get<Orbit>(vs.camera()).target[0] == 11);
    // An unmoved camera moves to a new starting camera at once.
    s.azimuth_deg = 50;
    vs.set_starting(s, t, 1.0);
    CHECK(std::get<Orbit>(vs.camera()).azimuth == doctest::Approx(deg(50)));
    // The person switches to Z up and orbits: the camera is theirs.
    switch_up(std::get<Orbit>(vs.camera()), Axis::PosZ);
    std::get<Orbit>(vs.camera()).azimuth += 0.3;
    vs.moved();
    s.azimuth_deg = 70;
    vs.set_starting(s, t, 1.0);
    CHECK(std::get<Orbit>(vs.camera()).up == Axis::PosZ);
    // Reset goes to the new starting camera and the scene's up.
    vs.reset(t, 1.0);
    CHECK(std::get<Orbit>(vs.camera()).up == Axis::PosY);
    CHECK(std::get<Orbit>(vs.camera()).azimuth == doctest::Approx(deg(70)));
    CHECK_FALSE(vs.is_moved());
}
