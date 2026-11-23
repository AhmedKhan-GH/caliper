// camera_drag.h: when a press on a scene's picture becomes a camera drag (an orbit, or a pan of the
// right or middle button) and when it stays a click (a pick, or a right click's panel menu).
//
// A press whose pointer stays less than kCameraDragStart points from the press is a click. On the
// first frame that the pointer is that far from the press, the drag starts, and it moves the camera
// by the whole way from the press, the motion inside the tolerance included. Thus a slow start
// neither waits nor jumps, and the camera's movement equals the pointer's. ImGui's own drag
// threshold (io.MouseDragThreshold) stays for every other drag: a box selection, a drag from the
// builder pane, a slider.
#pragma once

namespace caliper_live_views {

inline constexpr float kCameraDragStart = 2.0f;   // points from the press

// Whether the pointer at (dx, dy) points from the press makes the press a camera drag.
inline bool camera_drag_started(float dx, float dy) {
    return dx * dx + dy * dy >= kCameraDragStart * kCameraDragStart;
}

// The same for a squared distance from the press, such as ImGui's io.MouseDragMaxDistanceSqr.
inline bool camera_drag_started_sqr(float distance_sqr) {
    return distance_sqr >= kCameraDragStart * kCameraDragStart;
}

}  // namespace caliper_live_views
