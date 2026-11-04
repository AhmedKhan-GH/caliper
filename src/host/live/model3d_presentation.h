#pragma once

#include "live/model3d_geometry.h"

namespace caliper_live::model3d {

struct ViewportLayout { Rect scene, labels; };
// Labels occupy a separate column on the right. The scene rectangle is also
// the camera's projection and picking viewport.
ViewportLayout viewport_layout(Rect available, double label_width, double gap = 12);

// Bring the label column beside the projected stack without changing the
// camera viewport. Near/far-clipped or invalid frames retain the reserved
// column. The column may move left into otherwise unused scene space.
Rect column_beside_stack(const Scene& scene, const Camera& camera,
                         Rect scene_viewport, Rect reserved_column, double gap = 12);

struct LayerLabel {
    size_t group = 0;
    Rect bounds;
    double anchor_x = 0, anchor_y = 0;
};
// Labels remain aligned with their projected group. Crowded labels are omitted,
// preferring full-attention layers and the embedding/final norm. Returned
// rectangles are contained in the label column and never overlap.
std::vector<LayerLabel> label_layout(const Scene& scene, const Camera& camera,
                                   Rect scene_viewport, Rect label_column,
                                   double line_height, double gap = 3);

} // namespace caliper_live::model3d
