// camera_drag.h: how a press on a scene's picture moves the camera (an orbit of the left button, a
// pan of the right or middle button) and when the press was a click instead (a pick, or a right
// click's menu).
//
// The camera follows the pointer from the first point of motion after the press, so a slow start
// never waits. The release decides the click: when the pointer never went more than
// kCameraClickTolerance points from the press during the whole press, the press was a click. The
// camera then goes back to the camera saved at the press, exactly and before the frame draws, so the
// wobble of a click leaves nothing behind. A left click picks the item under the press point, and a
// right click opens the menu. A drag that went farther is a drag, also when it ends near its start.
// The tolerance is 2 points: the motion of a click on a mouse or a trackpad, and less than a person
// notices as a turn (0.02 radians). Drags that move no camera keep their own thresholds: ImGui's
// drag threshold (io.MouseDragThreshold) for a spec panel's box selection, a drag from the builder
// pane or a slider, and kClickSlop for the selections of the live views. The browser element
// (web/src/views/interact.ts) does not follow this rule yet: its camera drags start past its
// 3-pixel click slop.
#pragma once

#include "scene/camera.h"

namespace caliper_live_views {

inline constexpr float kCameraClickTolerance = 2.0f;   // points from the press

// Whether a press whose pointer went at most sqrt(max_distance_sqr) points from the press was a click.
inline bool camera_press_is_click(float max_distance_sqr) {
    return max_distance_sqr <= kCameraClickTolerance * kCameraClickTolerance;
}

// Whether two cameras are the same, field by field. The camera saved at a press comes back only
// while the camera is the one that the press's own motion left, so that a fit, a zoom or camera.set
// during the press stays.
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

}  // namespace caliper_live_views
