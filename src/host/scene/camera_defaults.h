// camera_defaults.h: the default camera of the 3-D views, defined once for the live points view
// (live/live_views_state.h: kCamAz, kCamEl, kCamFit) and the scene core's orbit (camera.h), so
// that a scene opens exactly as caliper.points opens a cloud (design/scenes.md, "The starting
// camera"). The TypeScript points view writes the same values out (web/src/views/camera.ts,
// CAMERA). It includes nothing, so that any view can take it.
#pragma once

namespace caliper_scene {

// A three-quarter view: the azimuth and the elevation in radians (54.431 and 24.064 degrees), and
// the distance that fits the data's bounding sphere into the narrower field of view, times this
// margin, which leaves room for the points view's box labels.
inline constexpr double kDefaultAzimuth = 0.95;
inline constexpr double kDefaultElevation = 0.42;
inline constexpr double kDefaultFitMargin = 1.15;

}  // namespace caliper_scene
