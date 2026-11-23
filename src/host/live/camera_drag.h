// camera_drag.h: how a press on a scene's picture moves the camera (an orbit of the left button, a
// pan of the right or middle button) and when the press was a click instead (a pick, or a right
// click's menu).
//
// The camera follows the pointer from the first point of motion after the press, so a slow start
// never waits. The release decides the click: when the pointer never went more than
// kCameraClickTolerance points from the press during the whole press, the press was a click. The
// view then goes back, before the frame draws, to the state that the press keeps (its base): the
// camera and the flags that the press's motion changes, as they would be had the pointer not
// moved. A left click picks the item under the press point, and a right click opens the menu. A
// drag that went farther is a drag, also when it ends near its start.
//
// A change of the camera during a press that does not come from the press's motion stays: a fit,
// a zoom, an up switch or camera.set. The view's own changes go through camera_press_change(),
// which applies them to the base and puts the press's motion back on top. A change from elsewhere
// (camera.set, a reframe by new data) becomes the base when the next frame sees it.
//
// The tolerance is 2 points: the motion of a click on a mouse or a trackpad, and less than a person
// notices as a turn (0.02 radians). Drags that move no camera keep their own thresholds: ImGui's
// drag threshold (io.MouseDragThreshold) for a spec panel's box selection, a drag from the builder
// pane or a slider, and kClickSlop for the selections of the live views. The browser element
// (web/src/views/interact.ts) follows the same rule with the same tolerance in CSS pixels.
#pragma once

#include "scene/camera.h"

#include <algorithm>

namespace caliper_live_views {

inline constexpr float kCameraClickTolerance = 2.0f;   // points from the press

// Whether a press whose pointer went at most sqrt(max_distance_sqr) points from the press was a click.
inline bool camera_press_is_click(float max_distance_sqr) {
    return max_distance_sqr <= kCameraClickTolerance * kCameraClickTolerance;
}

// Whether two cameras are the same, field by field.
inline bool same_camera(const caliper_scene::ViewCamera& a, const caliper_scene::ViewCamera& b) {
    if (a.index() != b.index()) return false;
    if (const caliper_scene::Orbit* o = std::get_if<caliper_scene::Orbit>(&a)) {
        const caliper_scene::Orbit& p = std::get<caliper_scene::Orbit>(b);
        return o->target[0] == p.target[0] && o->target[1] == p.target[1] && o->target[2] == p.target[2] &&
               o->distance == p.distance && o->azimuth == p.azimuth && o->elevation == p.elevation && o->up == p.up;
    }
    if (const caliper_scene::PanZoom* z = std::get_if<caliper_scene::PanZoom>(&a)) {
        const caliper_scene::PanZoom& p = std::get<caliper_scene::PanZoom>(b);
        return z->center[0] == p.center[0] && z->center[1] == p.center[1] && z->half_height == p.half_height &&
               z->y_down == p.y_down;
    }
    const caliper_scene::ThroughPinhole& t = std::get<caliper_scene::ThroughPinhole>(a);
    const caliper_scene::ThroughPinhole& p = std::get<caliper_scene::ThroughPinhole>(b);
    return t.camera == p.camera && t.zoom == p.zoom && t.pan[0] == p.pan[0] && t.pan[1] == p.pan[1];
}

// One press on a camera's picture. `State` is what the press keeps: the view's camera and the
// flags that the press's motion changes (a scene view's moved()).
template <class State>
struct CameraPress {
    bool down = false;
    bool keep = false;          // the press keeps the view as it is at its release (an animation ran)
    float x0 = 0, y0 = 0;       // the press
    float bx = 0, by = 0;       // the pointer when the base was taken
    float lx = 0, ly = 0;       // the pointer at the last frame
    float max_d2 = 0;           // the largest squared distance of the pointer from the press
    State base{};               // what a click puts back
    State left{};               // what the last frame left
    bool click() const { return camera_press_is_click(max_d2); }
};

// The press drives a view through an adapter `V` with:
//   using State = ...;                       what a press keeps, and
//   static bool same(const State&, const State&);
//   State state() const;                     the view's camera and flags now
//   void restore(const State&);              puts them back and draws again
//   void move(float dx, float dy);           the press's motion: an orbit or a pan
template <class V>
void camera_press_begin(CameraPress<typename V::State>& p, const V& v, float x, float y) {
    p = CameraPress<typename V::State>{};
    p.down = true;
    p.x0 = p.bx = p.lx = x;
    p.y0 = p.by = p.ly = y;
    p.base = p.left = v.state();
}

// One frame of a press while its button is held and on the frame it goes up (`released`), with
// the pointer at (x, y) when `valid`. A change since the last frame that the press did not make
// becomes the base. The motion moves the camera from the first point; a release within the
// tolerance puts the base back. Returns whether the press ended as a click.
template <class V>
bool camera_press_frame(CameraPress<typename V::State>& p, V& v, bool valid, float x, float y, bool released) {
    if (!p.down) return false;
    if (!V::same(v.state(), p.left)) {
        p.base = v.state();
        p.bx = p.lx;
        p.by = p.ly;
    }
    if (valid) {
        const float ox = x - p.x0, oy = y - p.y0;
        p.max_d2 = std::max(p.max_d2, ox * ox + oy * oy);
        const float dx = x - p.lx, dy = y - p.ly;
        p.lx = x;
        p.ly = y;
        if (dx != 0.0f || dy != 0.0f) v.move(dx, dy);
    }
    bool clicked = false;
    if (released) {
        clicked = p.click();
        if (clicked && !p.keep) v.restore(p.base);
        p.down = false;
    }
    p.left = v.state();
    return clicked;
}

// A change of the view's own during a press (a fit, a zoom, an up switch): while the press can
// still be a click, the change goes to the base, and the press's motion since the base goes back on
// top, so that a click undoes only its own motion. Otherwise the change applies as it is.
template <class V, class F>
void camera_press_change(CameraPress<typename V::State>& p, V& v, F&& change) {
    if (!p.down || !p.click()) {
        change();
        if (p.down) p.left = v.state();
        return;
    }
    v.restore(p.base);
    change();
    p.base = v.state();
    const float dx = p.lx - p.bx, dy = p.ly - p.by;
    if (dx != 0.0f || dy != 0.0f) v.move(dx, dy);
    p.left = v.state();
}

}  // namespace caliper_live_views
