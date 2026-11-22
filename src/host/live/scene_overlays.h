#pragma once
// The overlays of a scene view (design/instrument-interface.md, Label
// Placement and Annotations and Markers): the item labels of the scene,
// placed without overlap, and the labels and markers of a scene panel's
// annotations. draw_scene_in_rect() (live_views_scene.h) calls
// draw_scene_overlays() once a frame, after the scene and its scale bar.
// Defined in scene_overlays.cpp.
#include <imgui.h>

#include "spec_runtime/label_layout.h"   // LabelRect

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
    // One flag for each label: true for an annotation's text, which places
    // before the items' labels; null when none is an annotation.
    const std::vector<bool>* annotations = nullptr;
    // One flag for each label: true for a marker, whose ring is drawn at its
    // anchor over the picture, never hidden by the scene, and which no label
    // covers; null when there is none.
    const std::vector<bool>* rings = nullptr;
    ImU32 ink = 0, shade = 0;
    ImU32 accent = 0;   // the rings' colour; the ink when 0
    // Rectangles in screen pixels that the view's own texts take, such as the
    // selection count, the scale bar with its label and a stage's message:
    // no label covers them, as no label covers a ring. Null when there are none.
    const std::vector<caliper_host::LabelRect>* keep_clear = nullptr;
    // One priority for each label, lower placed first, in place of the default (an annotation's
    // index, else 1000 and its index); null for the default.
    const std::vector<int>* priorities = nullptr;
};

void draw_scene_overlays(const SceneOverlayInput& input);

}  // namespace caliper_live_views
