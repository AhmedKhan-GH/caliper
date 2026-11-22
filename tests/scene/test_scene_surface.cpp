// The scene tests of the surface primitive and the depth offset
// (design/instrument-interface.md, Draw Kinds and Depth Offset).
#include <doctest/doctest.h>

#include "scene/scene.h"

#include "scene/camera.h"
#include "scene/compile.h"
#include "fake_resources.h"
#include "scene_fixtures.h"

#include <algorithm>
#include <cmath>

using namespace caliper_scene;
using namespace scene_fixtures;

namespace {

Aabb box_of(double x, double y, double z, double cx = 0, double cy = 0, double cz = 0) {
    Aabb b;
    b.add({cx - x / 2, cy - y / 2, cz - z / 2});
    b.add({cx + x / 2, cy + y / 2, cz + z / 2});
    return b;
}

// The largest |x/w| and |y/w| over the eight corners of `b` seen by the orbit.
struct Extent {
    double x = 0, y = 0;                         // the largest |x| and |y| in normalised device units
    double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;   // the picture's extent
};
Extent corner_extent(const Orbit& o, const Aabb& b, double aspect) {
    const Mat4 vp = orbit_projection(o, b, aspect) * orbit_view(o);
    Extent e;
    for (int k = 0; k < 8; ++k) {
        const Vec3 p{(k & 1) ? b.hi[0] : b.lo[0], (k & 2) ? b.hi[1] : b.lo[1], (k & 4) ? b.hi[2] : b.lo[2]};
        double h[4];
        transform_h(vp, p, h);
        REQUIRE(h[3] > 0);
        e.x = std::max(e.x, std::fabs(h[0] / h[3]));
        e.y = std::max(e.y, std::fabs(h[1] / h[3]));
        e.x0 = std::min(e.x0, h[0] / h[3]);
        e.x1 = std::max(e.x1, h[0] / h[3]);
        e.y0 = std::min(e.y0, h[1] / h[3]);
        e.y1 = std::max(e.y1, h[1] / h[3]);
    }
    return e;
}

Orbit z_up_orbit() {
    Orbit o;
    o.up = Axis::PosZ;
    return o;
}

RenderTarget surface_target() {
    RenderTarget t;
    t.color = Texture{1};
    t.depth = Texture{2};
    return t;
}

struct Drawn {
    FakeResources res;
    SceneCompiler compiler;
    Frame frame;
    DrawList list{surface_target()};
    DrawMap map;
    CompileResult result;
    Mat4 mvp_of(size_t i) const {
        const DrawRecord& r = list.records()[i];
        return Mat4::from(params_of(list, r).mvp);
    }
    // The mvp of the draw that belongs to `e`.
    bool mvp_for(EntityId e, Mat4* out) const {
        for (size_t i = 0; i < list.records().size; ++i) {
            const DrawMapEntry* m = map.find(list.records()[i].pick_id);
            if (m && m->entity == e) {
                *out = mvp_of(i);
                return true;
            }
        }
        return false;
    }
};

void draw(Drawn& c, SceneTree& t, const ViewCamera& cam) {
    CompileOptions o;
    o.width = 400;
    o.height = 300;
    o.pixel_scale = 2.0f;
    t.update();
    c.compiler.prepare(t, c.res, c.frame);
    c.list.reset(surface_target());
    c.result = c.compiler.compile(t, cam, o, c.res, c.list, &c.map);
}

Batch unit_points(int8_t offset) {
    Batch b = points3(10);
    b.bounds.add({-1, -1, -1});
    b.bounds.add({1, 1, 1});
    b.depth_offset = offset;
    return b;
}

struct TwoPoints {
    SceneTree t;
    EntityId a, b;
    explicit TwoPoints(int8_t offset_b) {
        a = t.ensure("a");
        b = t.ensure("b");
        REQUIRE(t.set_batch(a, unit_points(0)));
        REQUIRE(t.set_batch(b, unit_points(offset_b)));
        REQUIRE(t.set_transform(b, translation({0.25, 0, 0})));
        t.update();
    }
};

}  // namespace

TEST_CASE("surface: the box fit keeps all eight corners inside the margin, touches it and centres the box") {
    const Aabb boxes[] = {box_of(2, 2, 2, 1, -2, 3), box_of(4, 1, 0.25)};
    for (const Aabb& box : boxes) {
        for (double aspect : {1.0, 2.0}) {
            ViewCamera v = z_up_orbit();
            const Orbit before = std::get<Orbit>(v);
            fit_box(v, box, aspect);
            const Orbit& o = std::get<Orbit>(v);
            CHECK(o.azimuth == before.azimuth);
            CHECK(o.elevation == before.elevation);
            CHECK(o.up == before.up);
            // The target stays within the box's bounding sphere, moved only to centre its picture.
            const Vec3 c = box.center();
            const Vec3 t{o.target[0], o.target[1], o.target[2]};
            CHECK(length(t - c) <= box.radius());
            const Extent e = corner_extent(o, box, aspect);
            CHECK(std::fabs(e.x0 + e.x1) < 2e-3);
            CHECK(std::fabs(e.y0 + e.y1) < 2e-3);
            const double limit = 1.0 / kBoxFitMargin;
            CHECK(e.x <= limit + 1e-9);
            CHECK(e.y <= limit + 1e-9);
            CHECK((std::fabs(e.x - limit) <= 1e-6 || std::fabs(e.y - limit) <= 1e-6));
        }
    }
}

TEST_CASE("surface: the box fit of a cube works for the other up axes too") {
    const Aabb box = box_of(2, 3, 1);
    for (Axis up : {Axis::PosX, Axis::PosY, Axis::PosZ}) {
        ViewCamera v = Orbit{};
        std::get<Orbit>(v).up = up;
        fit_box(v, box, 1.5);
        const Extent e = corner_extent(std::get<Orbit>(v), box, 1.5);
        CHECK(e.x <= 1.0 / kBoxFitMargin + 1e-9);
        CHECK(e.y <= 1.0 / kBoxFitMargin + 1e-9);
    }
}

TEST_CASE("surface: the box fit of a wide flat box is nearer than the sphere fit") {
    const Aabb box = box_of(4, 1, 0.25);
    ViewCamera boxed = z_up_orbit(), sphere = z_up_orbit();
    fit_box(boxed, box, 2.0);
    fit(sphere, box, 2.0);
    const Orbit& ob = std::get<Orbit>(boxed);
    const Orbit& os = std::get<Orbit>(sphere);
    CHECK(ob.distance < os.distance);
    // The share of the picture's width the projected box covers.
    auto width_share = [&](const Orbit& o) {
        const Mat4 vp = orbit_projection(o, box, 2.0) * orbit_view(o);
        double lo = 1e9, hi = -1e9;
        for (int k = 0; k < 8; ++k) {
            const Vec3 p{(k & 1) ? box.hi[0] : box.lo[0], (k & 2) ? box.hi[1] : box.lo[1],
                         (k & 4) ? box.hi[2] : box.lo[2]};
            double h[4];
            transform_h(vp, p, h);
            lo = std::min(lo, h[0] / h[3]);
            hi = std::max(hi, h[0] / h[3]);
        }
        return (hi - lo) / 2.0;
    };
    const double wb = width_share(ob), ws = width_share(os);
    CHECK(wb > ws);
    CHECK(wb <= 1.0 / kBoxFitMargin * 1.0 + 1e-9);
}

TEST_CASE("surface: aabb_contains takes tolerance, an invalid inner and an invalid outer") {
    const Aabb outer = box_of(2, 2, 2);   // extent 2
    CHECK(aabb_contains(outer, box_of(1, 1, 1), 0.0));
    CHECK(aabb_contains(outer, outer, 0.0));
    CHECK_FALSE(aabb_contains(outer, box_of(2.5, 1, 1), 0.0));
    // A 0.2 overshoot on one side is 10 percent of the larger extent.
    const Aabb over = box_of(2.2, 2, 2, 0.1);   // hi.x 1.2, lo.x -1.0
    CHECK_FALSE(aabb_contains(outer, over, 0.1 - 1e-3));
    CHECK(aabb_contains(outer, over, 0.1 + 1e-3));
    Aabb none;
    CHECK(aabb_contains(outer, none, 0.0));
    CHECK(aabb_contains(none, none, 0.0));
    CHECK_FALSE(aabb_contains(none, outer, 1.0));
}

TEST_CASE("surface: starting_camera with box_fit fills the target and distance from fit_box") {
    SceneTree t;
    t.set_up(Axis::PosZ);
    Batch b = points3(4);
    b.bounds.add({-2, -0.5, -0.125});
    b.bounds.add({2, 0.5, 0.125});
    REQUIRE(t.set_batch(t.ensure("flat"), b));
    t.update();
    const double aspect = 2.0;
    ViewCamera expected = Orbit{};
    std::get<Orbit>(expected).up = Axis::PosZ;
    fit_box(expected, t.scene_bounds(), aspect);
    const Orbit& eo = std::get<Orbit>(expected);

    const StartResult r = starting_camera(StartingCamera{}, t, aspect, true);
    REQUIRE(std::holds_alternative<Orbit>(r.camera));
    const Orbit& o = std::get<Orbit>(r.camera);
    CHECK(r.complete);
    CHECK(o.distance == doctest::Approx(eo.distance));
    for (int i = 0; i < 3; ++i) CHECK(o.target[i] == doctest::Approx(eo.target[i]));
    CHECK(o.up == Axis::PosZ);
    // The sphere fit is farther.
    const StartResult s = starting_camera(StartingCamera{}, t, aspect, false);
    CHECK(std::get<Orbit>(s.camera).distance > o.distance);

    // A given distance and angles are kept; the target still comes from the fit at those angles.
    StartingCamera given;
    given.form = StartingCamera::Form::Orbit;
    given.distance = 7.5;
    given.azimuth_deg = 30;
    const StartResult g = starting_camera(given, t, aspect, true);
    const Orbit& go = std::get<Orbit>(g.camera);
    CHECK(go.distance == 7.5);
    CHECK(go.azimuth == doctest::Approx(30.0 * kPi / 180.0));
    ViewCamera at30 = Orbit{};
    std::get<Orbit>(at30).up = Axis::PosZ;
    std::get<Orbit>(at30).azimuth = 30.0 * kPi / 180.0;
    fit_box(at30, t.scene_bounds(), aspect);
    for (int i = 0; i < 3; ++i) CHECK(go.target[i] == doctest::Approx(std::get<Orbit>(at30).target[i]));
}

TEST_CASE("surface: a depth offset subtracts k units of w from the z row of the batch's mvp") {
    TwoPoints plain(0), offset(2);
    const ViewCamera cam = [&] {
        ViewCamera v = Orbit{};
        std::get<Orbit>(v).up = plain.t.up();
        fit(v, plain.t.scene_bounds(), 400.0 / 300.0);
        return v;
    }();
    Drawn p, q;
    draw(p, plain.t, cam);
    draw(q, offset.t, cam);
    REQUIRE(p.result.ok);
    REQUIRE(q.result.ok);
    REQUIRE(p.list.records().size == 2);
    REQUIRE(q.list.records().size == 2);
    Mat4 pa, pb, qa, qb;
    REQUIRE(p.mvp_for(plain.a, &pa));
    REQUIRE(p.mvp_for(plain.b, &pb));
    REQUIRE(q.mvp_for(offset.a, &qa));
    REQUIRE(q.mvp_for(offset.b, &qb));
    for (int c = 0; c < 4; ++c) {
        // The draw without an offset is unchanged.
        for (int r = 0; r < 4; ++r) CHECK(qa.at(r, c) == pa.at(r, c));
        // The offset draw: rows x, y and w are as before; z is z - 2 u w.
        for (int r : {0, 1, 3}) CHECK(qb.at(r, c) == pb.at(r, c));
        CHECK(std::fabs(qb.at(2, c) - (pb.at(2, c) - 2.0 * kDepthOffsetUnit * pb.at(3, c))) <= 1e-6);  // f32 storage of the mvp
    }
    // The offset moves the draw toward the camera: smaller depth for a point in front.
    CHECK(qb.at(2, 3) < pb.at(2, 3));
}

TEST_CASE("surface: the depth offset moves depth by k units at every distance and its sign flips the direction") {
    SceneTree t;
    const EntityId e = t.ensure("n");
    REQUIRE(t.set_batch(e, unit_points(-3)));
    t.update();
    ViewCamera cam = Orbit{};
    std::get<Orbit>(cam).up = t.up();
    fit(cam, t.scene_bounds(), 400.0 / 300.0);
    Drawn d;
    draw(d, t, cam);
    REQUIRE(d.result.ok);
    SceneTree t0;
    const EntityId e0 = t0.ensure("n");
    REQUIRE(t0.set_batch(e0, unit_points(0)));
    Drawn d0;
    draw(d0, t0, cam);
    Mat4 m, m0;
    REQUIRE(d.mvp_for(e, &m));
    REQUIRE(d0.mvp_for(e0, &m0));
    for (int c = 0; c < 4; ++c)
        CHECK(std::fabs(m.at(2, c) - (m0.at(2, c) + 3.0 * kDepthOffsetUnit * m0.at(3, c))) <= 1e-6);  // f32 storage of the mvp
}

TEST_CASE("surface: a 2-D scene ignores the depth offset") {
    auto compile_flat = [](int8_t offset, Mat4* mvp) {
        SceneTree flat(Space::k2D);
        Batch p2 = points2(5);
        p2.bounds.add({0, 0, 0});
        p2.bounds.add({10, 5, 0});
        p2.depth_offset = offset;
        REQUIRE(flat.set_batch(flat.root(), p2));
        flat.update();
        const ViewCamera cam = starting_camera(StartingCamera{}, flat, 400.0 / 300.0).camera;
        Drawn d;
        draw(d, flat, cam);
        REQUIRE(d.result.ok);
        REQUIRE(d.list.records().size == 1);
        *mvp = d.mvp_of(0);
    };
    Mat4 a, b;
    compile_flat(0, &a);
    compile_flat(2, &b);
    for (int i = 0; i < 16; ++i) CHECK(a.m[i] == b.m[i]);
}
