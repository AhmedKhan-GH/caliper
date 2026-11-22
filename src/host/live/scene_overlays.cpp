// The overlays of a scene view (scene_overlays.h). Today's behavior: each
// item label at its anchor, with a shadow, and no overlap avoidance; the
// placement of label_layout.h replaces it.
#include "live/scene_overlays.h"

#include "scene/compile.h"

namespace caliper_live_views {

void draw_scene_overlays(const SceneOverlayInput& in) {
    if (!in.draw || !in.labels) return;
    for (const caliper_scene::LabelAnchor& a : *in.labels) {
        const ImVec2 at(in.origin.x + a.x * in.scale_x + 4.0f,
                        in.origin.y + a.y * in.scale_y - ImGui::GetTextLineHeight() - 2.0f);
        in.draw->AddText(ImVec2(at.x + 1.0f, at.y + 1.0f), in.shade, a.text.c_str());
        in.draw->AddText(at, in.ink, a.text.c_str());
    }
}

}  // namespace caliper_live_views
