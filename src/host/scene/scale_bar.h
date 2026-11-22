// scale_bar.h: the scale bar of a scene view (design/scenes.md, "The scale bar"): its length, its
// label and when it shows. The view layer draws it with ImGui over the picture, natively and in a
// browser alike (live/live_views_draw.cpp); this header is the one statement of the rule.
//
// The length is the longest of 1, 2 and 5 times a power of ten, in scene units, whose bar fits in
// a fifth of the picture's width, so the bar takes 8 % to 20 % of the width. In 3-D the scale is
// the orbit target's, the centre of the picture, and the label says so; in 2-D it is exact.
// Through a pinhole the bar hides, because the picture has no single depth to measure at.
#pragma once

#include "scene/camera.h"

#include <string>
#include <string_view>

namespace caliper_scene {

inline constexpr double kScaleBarMinPicture = 160.0;   // logical pixels: a narrower picture has no bar
inline constexpr double kScaleBarShare = 0.2;          // the bar fits in this share of the width
inline constexpr double kScaleBarMargin = 12.0;        // logical pixels in from the right and bottom
inline constexpr double kScaleBarThickness = 2.0;      // logical pixels
inline constexpr double kScaleBarTick = 6.0;           // the end ticks' height, logical pixels

// The longest of 1, 2 and 5 times 10^k scene units whose bar, length / units_per_px pixels long,
// fits in kScaleBarShare of width_px. 0 when either is not finite or not above 0.
double scale_bar_length(double units_per_px, double width_px);

// The label of a length: the number, then the unit as the scene gives it, and " at target" in
// 3-D. For m, s and g with the prefixes n, µ (or u), m, c and k, a number outside 0.01 to 1,000
// takes the prefix, in steps of 1,000, that brings it between 1 and 1,000, a prefixed unit being
// converted through its base ("5 mm", "2 km", "200 µs"); "u" is written "µ". Any other unit
// follows the number as given, in plain decimals from 0.001 to 999,999 and as "5e-7" outside
// them. With no unit ("", "arbitrary" or "a.u.") the number stands alone.
std::string scale_bar_label(double length, std::string_view units, bool at_target);

// Scene units per logical pixel of a picture height_px tall: at the orbit target in 3-D,
// 2 · distance · tan(fovy / 2) / height_px; exactly in 2-D, 2 · half_height / height_px; 0 through
// a pinhole.
double scale_bar_units_per_pixel(const ViewCamera& cam, double height_px);

struct ScaleBar {
    bool visible = false;
    double length = 0;     // scene units
    double pixels = 0;     // the bar's length in logical pixels
    std::string label;
};

// The bar of a picture of width_px x height_px logical pixels. Hidden through a pinhole, while
// the picture is narrower than kScaleBarMinPicture and before the scene has bounds.
ScaleBar scale_bar(const ViewCamera& cam, bool has_bounds, std::string_view units, double width_px,
                   double height_px);

}  // namespace caliper_scene
