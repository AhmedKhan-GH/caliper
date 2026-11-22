// camera.h: the cameras of the scene core (design/renderer-core.md, "Entity tree, cameras and
// bounds"; design/scenes.md, "The up-axis switch" and "The starting camera").
//
// Three camera models cover the views. The orbit turns around a target at a distance with an
// azimuth and an elevation about one of the six up axes; the eye is at
//     target + distance * (cos(el) * (cos(az) * a + sin(az) * b) + sin(el) * u),
// u the up axis, a the first of +x, +y and +z perpendicular to u, and b = a x u, which with
// +y up puts azimuth 0 on the +x side and 90 degrees on the +z side, the points view's
// convention. Its field of view is 45 degrees and its near and far planes follow the bounds as
// the points view computes them. Its default angles and the fit's margin are the points view's
// (camera_defaults.h). Pan and zoom is orthographic with y up or down. Looking
// through a pinhole projects with the intrinsics, letterboxed to the view. Clip space has x
// right, y up and depth in [0, 1] with near at 0; matrices are column-major.
#pragma once

#include "scene/camera_defaults.h"
#include "scene/scene.h"
#include "scene/scene_math.h"

#include <optional>
#include <string>
#include <variant>

namespace caliper_scene {

inline constexpr double kOrbitFovY = 45.0 * kPi / 180.0;
inline constexpr double kMaxElevation = 89.0 * kPi / 180.0;     // the bound the orbit keeps within

struct Orbit {
    double target[3] = {0, 0, 0};
    double distance = 3.0;
    double azimuth = kDefaultAzimuth;      // radians
    double elevation = kDefaultElevation;  // radians
    Axis up = Axis::PosY;
};
struct PanZoom {
    double center[2] = {0, 0};
    double half_height = 1.0;
    bool y_down = true;                    // pixel spaces; false for plots
};
struct ThroughPinhole {
    EntityId camera;
    double zoom = 1.0;                     // of the letterboxed picture
    double pan[2] = {0, 0};                // in normalised device units of the view
};
using ViewCamera = std::variant<Orbit, PanZoom, ThroughPinhole>;

// ---- The orbit ----------------------------------------------------------------------------------

struct OrbitBasis { Vec3 u, a, b; };
OrbitBasis orbit_basis(Axis up);
Vec3 orbit_direction(const Orbit& o);      // the unit vector e from the target to the eye
Vec3 orbit_eye(const Orbit& o);
Mat4 orbit_view(const Orbit& o);           // the look-at from the eye to the target, up u
// Switches the up axis keeping the target, the distance and the eye position: azimuth and
// elevation are computed again from the eye in the new basis, and an eye within 1 degree of the
// new axis is held at an elevation of 89 degrees.
void switch_up(Orbit& o, Axis up);

// The near and far planes of a view whose bounds' centre lies at depth `depth` along the line
// of sight, with bounding radius r: zn = max(depth - 2r, depth / 1000), zf = depth + 2r, as the
// points view computes them (with its floor of 1e-6 and its far plane of 2 zn when they meet).
void near_far(double depth, double r, double* zn, double* zf);

// ---- Matrices of a view -----------------------------------------------------------------------------

struct CameraMatrices {
    bool ok = false;
    Mat4 view, proj;                       // clip = proj * view * world
    Vec3 eye;                              // world position of the eye (orbit, pinhole)
    bool perspective = false;
    EntityId through;                      // the camera looked through, if any
    // Looking through a pinhole: the camera's pixel space straight onto the letterboxed picture,
    // for its pixel-space children, which are drawn flat over the projected 3-D content.
    Mat4 pixel_to_clip;
    uint16_t scissor[4] = {0, 0, 0xffff, 0xffff};   // the picture: the letterbox's inside, in pixels
    std::string note;                      // why ok is false
};
// The matrices for a view of width x height physical pixels. Near and far planes come from the
// scene's bounds after the tree's last update().
CameraMatrices camera_matrices(const ViewCamera& cam, const SceneTree& tree, uint32_t width, uint32_t height);

Mat4 orbit_projection(const Orbit& o, const Aabb& world, double aspect);
Mat4 panzoom_projection(const PanZoom& p, double aspect);
// The pinhole's projection in its camera frame (x right, y down, z forward):
//   clip.x = (2 fx / W) X + (2 cx / W - 1) Z,  clip.y = -(2 fy / H) Y + (1 - 2 cy / H) Z,
//   clip.z = f (Z - n) / (f - n),  clip.w = Z,
// then letterboxed to the view's aspect, zoomed and panned.
Mat4 pinhole_projection(const Pinhole& c, double zn, double zf, double view_aspect, double zoom,
                        const double pan[2]);

// ---- Fitting --------------------------------------------------------------------------------------------

// The orbit's target goes to the bounds' centre at the distance that frames the bounding sphere
// (keeping its angles); the pan and zoom centres the bounds and takes the half height from the
// larger extent for the aspect; looking through a pinhole returns to zoom 1 and no pan. Without
// valid bounds the fit frames the box from -1 to 1, as the points view does.
void fit(ViewCamera& cam, const Aabb& world, double aspect);

// ---- The starting camera and reset ------------------------------------------------------------------------

struct StartingCamera {
    enum class Form { Fit, Orbit, Entity } form = Form::Fit;   // Fit: none given, or an empty dict
    std::optional<Vec3> target;
    std::optional<double> distance;        // scene units, above 0
    std::optional<double> azimuth_deg;     // any finite number
    std::optional<double> elevation_deg;   // -89 to 89
    std::string entity;                    // Form::Entity: the path of a camera entity
};
// The refusals of scenes.md for a view_camera: not finite, a distance not above 0, an elevation
// outside -89 to 89 degrees, a path outside the grammar, any view_camera in a 2-D scene.
bool validate_starting_camera(const StartingCamera& s, Space space, std::string* error = nullptr);

struct StartResult {
    ViewCamera camera;
    bool complete = true;      // false: keys were left to fill from a fit the scene cannot make yet
    bool fell_back = false;    // the entity form named no camera; the view opens on the fit and says so
};
// The camera a view opens on: the orbit's keys given are used as given and the rest filled from
// the fit, the angles read against the scene's up; the entity form looks through that entity's
// pinhole; a 2-D scene opens on the fit of its pan and zoom.
StartResult starting_camera(const StartingCamera& s, const SceneTree& tree, double aspect);

// A view's camera state: where it starts, where it is, and whether the person moved it. A
// remembered camera wins until reset; a new starting camera moves an unmoved camera at once.
class ViewState {
public:
    void set_starting(const StartingCamera& s, const SceneTree& tree, double aspect);
    void reset(const SceneTree& tree, double aspect);   // to the starting camera and the scene's up
    void moved() { moved_ = true; pending_ = false; }    // the person orbited, panned, zoomed or fitted
    // Fills the keys left out once the scene has bounds, while the camera is unmoved.
    void update(const SceneTree& tree, double aspect);
    const ViewCamera& camera() const { return camera_; }
    ViewCamera& camera() { return camera_; }
    bool is_moved() const { return moved_; }
    bool fell_back() const { return fell_back_; }
    const StartingCamera& starting() const { return start_; }

private:
    StartingCamera start_;
    ViewCamera camera_ = Orbit{};
    bool moved_ = false, pending_ = true, fell_back_ = false;
};

}  // namespace caliper_scene
