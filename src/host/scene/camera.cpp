// camera.cpp: the orbit, pan and zoom, looking through a pinhole, fitting and the starting
// camera (camera.h).
#include "scene/camera.h"

#include <algorithm>
#include <cmath>

namespace caliper_scene {

namespace {

bool fail(std::string* error, const std::string& text) {
    if (error) *error = text;
    return false;
}

Vec3 target_of(const Orbit& o) { return {o.target[0], o.target[1], o.target[2]}; }

// The points view frames the box from -1 to 1 when it has no bounds.
Aabb or_unit(const Aabb& b) {
    if (b.valid) return b;
    Aabb u;
    u.add({-1, -1, -1});
    u.add({1, 1, 1});
    return u;
}

}  // namespace

// ---- The orbit ------------------------------------------------------------------------------------------

OrbitBasis orbit_basis(Axis up) {
    OrbitBasis r;
    r.u = axis_vector(up);
    const Vec3 candidates[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (const Vec3& c : candidates)
        if (dot(c, r.u) == 0) {
            r.a = c;
            break;
        }
    r.b = cross(r.a, r.u);
    return r;
}

Vec3 orbit_direction(const Orbit& o) {
    const OrbitBasis k = orbit_basis(o.up);
    const double ce = std::cos(o.elevation), se = std::sin(o.elevation);
    const double ca = std::cos(o.azimuth), sa = std::sin(o.azimuth);
    return (k.a * ca + k.b * sa) * ce + k.u * se;
}

Vec3 orbit_eye(const Orbit& o) { return target_of(o) + orbit_direction(o) * o.distance; }

Mat4 orbit_view(const Orbit& o) { return look_at(orbit_eye(o), target_of(o), axis_vector(o.up)); }

void switch_up(Orbit& o, Axis up) {
    const Vec3 e = orbit_direction(o);   // the eye stays where it is: only its angles change
    o.up = up;
    const OrbitBasis k = orbit_basis(up);
    double el = std::asin(std::clamp(dot(e, k.u), -1.0, 1.0));
    const double ea = dot(e, k.a), eb = dot(e, k.b);
    const double az = (ea == 0 && eb == 0) ? o.azimuth : std::atan2(eb, ea);
    if (el > kMaxElevation) el = kMaxElevation;
    if (el < -kMaxElevation) el = -kMaxElevation;
    o.azimuth = az;
    o.elevation = el;
}

void near_far(double depth, double r, double* zn, double* zf) {
    double n = std::max(depth - 2.0 * r, std::max(std::fabs(depth) * 1e-3, 1e-6));
    double f = depth + 2.0 * r;
    if (f <= n * 1.001) f = n * 2.0;
    *zn = n;
    *zf = f;
}

// ---- Projections ----------------------------------------------------------------------------------------

Mat4 orbit_projection(const Orbit& o, const Aabb& world, double aspect) {
    const Aabb b = or_unit(world);
    const double r = std::max(b.radius(), 1e-6);
    const Vec3 eye = orbit_eye(o);
    const Vec3 fwd = orbit_direction(o) * -1.0;
    const double depth = dot(b.center() - eye, fwd);
    double zn, zf;
    near_far(depth, r, &zn, &zf);
    return perspective_z01(kOrbitFovY, aspect > 0 ? aspect : 1.0, zn, zf);
}

Mat4 panzoom_projection(const PanZoom& p, double aspect) {
    if (!(aspect > 0)) aspect = 1.0;
    const double hh = std::max(p.half_height, 1e-12), hw = hh * aspect;
    const double sy = p.y_down ? -1.0 : 1.0;
    Mat4 m = Mat4::zero();
    m.m[0] = 1.0 / hw;
    m.m[5] = sy / hh;
    m.m[10] = 0.0;   // z ignored: everything at depth 0.5, drawn in list order
    m.m[12] = -p.center[0] / hw;
    m.m[13] = -sy * p.center[1] / hh;
    m.m[14] = 0.5;
    m.m[15] = 1.0;
    return m;
}

namespace {

// The letterbox scale of a W x H picture in a view of the given aspect.
void letterbox(const Pinhole& c, double view_aspect, double* sx, double* sy) {
    const double a = double(c.width) / double(c.height);
    *sx = 1.0;
    *sy = 1.0;
    if (view_aspect > a) *sx = a / view_aspect;
    else if (view_aspect < a) *sy = view_aspect / a;
}

}  // namespace

Mat4 pinhole_projection(const Pinhole& c, double zn, double zf, double view_aspect, double zoom,
                        const double pan[2]) {
    const double W = c.width, H = c.height;
    double sx, sy;
    letterbox(c, view_aspect, &sx, &sy);
    Mat4 m = Mat4::zero();
    m.at(0, 0) = 2 * c.fx / W * sx * zoom;
    m.at(0, 2) = (2 * c.cx / W - 1) * sx * zoom + pan[0];
    m.at(1, 1) = -2 * c.fy / H * sy * zoom;
    m.at(1, 2) = (1 - 2 * c.cy / H) * sy * zoom + pan[1];
    m.at(2, 2) = zf / (zf - zn);
    m.at(2, 3) = -zf * zn / (zf - zn);
    m.at(3, 2) = 1.0;
    return m;
}

CameraMatrices camera_matrices(const ViewCamera& cam, const SceneTree& tree, uint32_t width, uint32_t height) {
    CameraMatrices out;
    const double aspect = (width > 0 && height > 0) ? double(width) / double(height) : 1.0;
    if (const Orbit* o = std::get_if<Orbit>(&cam)) {
        out.view = orbit_view(*o);
        out.proj = orbit_projection(*o, tree.scene_bounds(), aspect);
        out.eye = orbit_eye(*o);
        out.perspective = true;
        out.ok = finite(out.view) && finite(out.proj);
        if (!out.ok) out.note = "the orbit is not finite";
        return out;
    }
    if (const PanZoom* p = std::get_if<PanZoom>(&cam)) {
        out.view = Mat4{};
        out.proj = panzoom_projection(*p, aspect);
        out.ok = finite(out.proj);
        if (!out.ok) out.note = "the pan and zoom is not finite";
        return out;
    }
    const ThroughPinhole& t = std::get<ThroughPinhole>(cam);
    const Pinhole* c = tree.camera(t.camera);
    if (!c || !tree.drawable(t.camera)) {
        out.note = "the entity holds no camera, or its pose is not finite";
        return out;
    }
    const Mat4 frame = tree.camera_frame(t.camera);
    if (!inverse(frame, &out.view)) {
        out.note = "the camera's pose is singular";
        return out;
    }
    const Aabb b = or_unit(tree.scene_bounds());
    const double depth = transform_point(out.view, b.center()).z;   // z forward in the camera frame
    double zn, zf;
    near_far(depth, std::max(b.radius(), 1e-6), &zn, &zf);
    out.proj = pinhole_projection(*c, zn, zf, aspect, t.zoom, t.pan);
    out.eye = transform_point(frame, {0, 0, 0});
    out.perspective = true;
    out.through = t.camera;
    // The pixel space onto the same picture, flat: (u, v) -> (2u / W - 1, 1 - 2v / H), z 0.5.
    double sx, sy;
    letterbox(*c, aspect, &sx, &sy);
    Mat4 px = Mat4::zero();
    px.at(0, 0) = 2.0 / c->width * sx * t.zoom;
    px.at(0, 3) = -sx * t.zoom + t.pan[0];
    px.at(1, 1) = -2.0 / c->height * sy * t.zoom;
    px.at(1, 3) = sy * t.zoom + t.pan[1];
    px.at(2, 3) = 0.5;
    px.at(3, 3) = 1.0;
    out.pixel_to_clip = px;
    // The letterbox's inside, in pixels, top-left origin.
    const double x0 = (-sx * t.zoom + t.pan[0] + 1) * 0.5 * width;
    const double x1 = (sx * t.zoom + t.pan[0] + 1) * 0.5 * width;
    const double y0 = (1 - (sy * t.zoom + t.pan[1])) * 0.5 * height;
    const double y1 = (1 - (-sy * t.zoom + t.pan[1])) * 0.5 * height;
    const double cx0 = std::clamp(std::floor(x0 + 0.5), 0.0, double(width));
    const double cx1 = std::clamp(std::floor(x1 + 0.5), 0.0, double(width));
    const double cy0 = std::clamp(std::floor(y0 + 0.5), 0.0, double(height));
    const double cy1 = std::clamp(std::floor(y1 + 0.5), 0.0, double(height));
    out.scissor[0] = uint16_t(cx0);
    out.scissor[1] = uint16_t(cy0);
    out.scissor[2] = uint16_t(std::max(0.0, cx1 - cx0));
    out.scissor[3] = uint16_t(std::max(0.0, cy1 - cy0));
    out.ok = finite(out.view) && finite(out.proj);
    return out;
}

// ---- Fitting ------------------------------------------------------------------------------------------------

void fit(ViewCamera& cam, const Aabb& world, double aspect) {
    if (!(aspect > 0)) aspect = 1.0;
    const Aabb b = or_unit(world);
    const Vec3 c = b.center();
    const Vec3 e = b.half_extent();
    if (Orbit* o = std::get_if<Orbit>(&cam)) {
        double r = length(e);
        if (!(r > 1e-12) || !std::isfinite(r)) r = 1.0;
        const double half_fovy = kOrbitFovY * 0.5;
        const double half_fovx = std::atan(std::tan(half_fovy) * aspect);
        o->target[0] = c.x;
        o->target[1] = c.y;
        o->target[2] = c.z;
        o->distance = r / std::sin(std::min(half_fovy, half_fovx)) * kDefaultFitMargin;
        return;
    }
    if (PanZoom* p = std::get_if<PanZoom>(&cam)) {
        double hh = std::max(std::fabs(e.y), std::fabs(e.x) / aspect);
        if (!(hh > 1e-12) || !std::isfinite(hh)) hh = 1.0;
        p->center[0] = c.x;
        p->center[1] = c.y;
        p->half_height = hh * 1.08;
        return;
    }
    ThroughPinhole& t = std::get<ThroughPinhole>(cam);
    t.zoom = 1.0;
    t.pan[0] = t.pan[1] = 0.0;
}

void fit_box(ViewCamera& cam, const Aabb& world, double aspect) {
    Orbit* o = std::get_if<Orbit>(&cam);
    if (!o) {
        fit(cam, world, aspect);
        return;
    }
    if (!(aspect > 0)) aspect = 1.0;
    const Aabb b = or_unit(world);
    const Vec3 c = b.center();
    double r = b.radius();
    if (!(r > 1e-12) || !std::isfinite(r)) r = 1.0;
    // The camera's frame: e from the target to the eye, and the picture's right and up.
    const Vec3 e = orbit_direction(*o);
    const Vec3 right = normalize(cross(e * -1.0, axis_vector(o->up)));
    const Vec3 up = cross(right, e * -1.0);
    const double ty = std::tan(kOrbitFovY * 0.5) / kBoxFitMargin;
    const double tx = ty * aspect;
    // A corner q (from the centre) lies in the view when |q.right| <= tx (d - q.e) and
    // |q.up| <= ty (d - q.e); the distance is the least d that holds for all eight.
    double d = 0, front = -r;
    for (int k = 0; k < 8; ++k) {
        const Vec3 p{(k & 1) ? b.hi[0] : b.lo[0], (k & 2) ? b.hi[1] : b.lo[1], (k & 4) ? b.hi[2] : b.lo[2]};
        const Vec3 q = p - c;
        const double z = dot(q, e);
        d = std::max(d, z + std::fabs(dot(q, right)) / tx);
        d = std::max(d, z + std::fabs(dot(q, up)) / ty);
        front = std::max(front, z);
    }
    // The eye stays outside the box, at least a tenth of the radius in front of its nearest corner.
    d = std::max(d, front + 0.1 * r);
    if (!(d > 0) || !std::isfinite(d)) d = r / std::sin(kOrbitFovY * 0.5) * kDefaultFitMargin;
    o->target[0] = c.x;
    o->target[1] = c.y;
    o->target[2] = c.z;
    o->distance = d;
}

bool aabb_contains(const Aabb& outer, const Aabb& inner, double tolerance) {
    if (!inner.valid) return true;
    if (!outer.valid) return false;
    double ext = 0;
    for (int i = 0; i < 3; ++i) ext = std::max(ext, outer.hi[i] - outer.lo[i]);
    const double t = tolerance * (ext > 0 ? ext : 1.0);
    for (int i = 0; i < 3; ++i)
        if (inner.lo[i] < outer.lo[i] - t || inner.hi[i] > outer.hi[i] + t) return false;
    return true;
}

// ---- The starting camera ----------------------------------------------------------------------------------------

bool validate_starting_camera(const StartingCamera& s, Space space, std::string* error) {
    if (s.form == StartingCamera::Form::Fit && !s.target && !s.distance && !s.azimuth_deg && !s.elevation_deg &&
        s.entity.empty())
        return true;
    if (space == Space::k2D) return fail(error, "a 2-D scene takes no view_camera; its fit frames it whole");
    if (s.form == StartingCamera::Form::Entity) {
        std::string why;
        if (s.entity.empty() || !valid_path(s.entity, &why))
            return fail(error, "view_camera names an entity path outside the grammar");
        return true;
    }
    if (s.target && !finite(*s.target)) return fail(error, "view_camera's target is not finite");
    if (s.distance && (!std::isfinite(*s.distance) || !(*s.distance > 0)))
        return fail(error, "view_camera's distance is finite and above 0");
    if (s.azimuth_deg && !std::isfinite(*s.azimuth_deg)) return fail(error, "view_camera's azimuth is not finite");
    if (s.elevation_deg && (!std::isfinite(*s.elevation_deg) || *s.elevation_deg < -89.0 || *s.elevation_deg > 89.0))
        return fail(error, "view_camera's elevation is from -89 to 89 degrees");
    return true;
}

StartResult starting_camera(const StartingCamera& s, const SceneTree& tree, double aspect, bool box_fit) {
    StartResult r;
    const Aabb bounds = tree.scene_bounds();
    if (tree.space() == Space::k2D) {
        PanZoom p;
        p.y_down = tree.up() != Axis::PosY;
        ViewCamera v = p;
        fit(v, bounds, aspect);
        r.camera = v;
        r.complete = bounds.valid;
        return r;
    }
    if (s.form == StartingCamera::Form::Entity) {
        const EntityId e = tree.find(s.entity);
        if (e && tree.camera(e)) {
            ThroughPinhole t;
            t.camera = e;
            r.camera = t;
            return r;
        }
        r.fell_back = true;
    }
    Orbit o;
    o.up = tree.up();
    if (s.form == StartingCamera::Form::Orbit) {
        if (s.azimuth_deg) o.azimuth = *s.azimuth_deg * kPi / 180.0;
        if (s.elevation_deg) o.elevation = *s.elevation_deg * kPi / 180.0;
    }
    ViewCamera v = o;
    if (box_fit)
        fit_box(v, bounds, aspect);   // the target and distance of the fit, at these angles
    else
        fit(v, bounds, aspect);
    Orbit& fo = std::get<Orbit>(v);
    const bool orbit_form = s.form == StartingCamera::Form::Orbit;
    if (orbit_form && s.target) {
        fo.target[0] = s.target->x;
        fo.target[1] = s.target->y;
        fo.target[2] = s.target->z;
    }
    if (orbit_form && s.distance) fo.distance = *s.distance;
    r.complete = bounds.valid || (orbit_form && s.target && s.distance);
    r.camera = v;
    return r;
}

// ---- ViewState ---------------------------------------------------------------------------------------------------

void ViewState::set_starting(const StartingCamera& s, const SceneTree& tree, double aspect) {
    start_ = s;
    if (!moved_) reset(tree, aspect);
}

void ViewState::reset(const SceneTree& tree, double aspect) {
    const StartResult r = starting_camera(start_, tree, aspect);
    camera_ = r.camera;
    fell_back_ = r.fell_back;
    pending_ = !r.complete;
    moved_ = false;
}

void ViewState::update(const SceneTree& tree, double aspect) {
    if (moved_ || !pending_) return;
    if (!tree.scene_bounds().valid) return;
    reset(tree, aspect);
}

}  // namespace caliper_scene
