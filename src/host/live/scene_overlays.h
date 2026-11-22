#pragma once
// The overlays of a scene view (design/instrument-interface.md, Label
// Placement and Annotations and Markers): the item labels of the scene,
// placed without overlap, and the labels and markers of a scene panel's
// annotations. draw_scene_in_rect() (live_views_scene.h) calls
// draw_scene_overlays() once a frame, after the scene and its scale bar.
// Defined in scene_overlays.cpp.
#include <imgui.h>

#include <vector>

namespace caliper_scene {
struct LabelAnchor;  // scene/compile.h
}

namespace caliper_live_views {

struct SceneOverlayInput {
    ImDrawList* draw = nullptr;
    ImVec2 origin;                    // the view's top-left corner, in screen pixels
    float scale_x = 1, scale_y = 1;   // from the scene's physical pixels to screen pixels
    const std::vector<caliper_scene::LabelAnchor>* labels = nullptr;   // the scene's item labels
    ImU32 ink = 0, shade = 0;
};

void draw_scene_overlays(const SceneOverlayInput& input);

}  // namespace caliper_live_views
