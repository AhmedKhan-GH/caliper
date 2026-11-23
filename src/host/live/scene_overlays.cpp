// The overlays of a scene view (scene_overlays.h, design/instrument-interface.md,
// Label Placement).
//
// The item labels of the scene go through the label placement
// (spec_runtime/label_layout.h) inside the view's clip rectangle, nearest
// first, so that no two overlap: each tries eight places around its anchor,
// then eight farther ones with a leader line. A label with no free place
// shows a dot at its anchor, and a hover over the dot shows its text. The
// placement runs again only when an anchor, a text or the view's size
// changes, because the last few placements are kept. The view's own texts,
// such as the selection count and the scale bar, are obstacles that no label
// covers.
#include "live/scene_overlays.h"
#include "ui/sternbild_theme.h"

#include "scene/compile.h"
#include "spec_runtime/label_layout.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

namespace caliper_live_views {

namespace {

using caliper_host::LabelBox;
using caliper_host::LabelRect;
using caliper_host::PlacedLabel;

bool same(const LabelBox& a, const LabelBox& b) {
    return a.anchor_x == b.anchor_x && a.anchor_y == b.anchor_y && a.width == b.width && a.height == b.height &&
           a.priority == b.priority && a.pinned == b.pinned;
}

// The placements of the last few inputs, so that a still view does not place
// its labels again each frame.
struct Placement {
    std::vector<LabelBox> boxes;
    LabelRect panel;
    std::vector<LabelRect> obstacles;
    std::vector<PlacedLabel> placed;
};

bool same_rect(const LabelRect& a, const LabelRect& b) {
    return a.x0 == b.x0 && a.y0 == b.y0 && a.x1 == b.x1 && a.y1 == b.y1;
}

const std::vector<PlacedLabel>& placed_for(const std::vector<LabelBox>& boxes, const LabelRect& panel,
                                           const std::vector<LabelRect>& obstacles) {
    static std::deque<Placement> kept;
    for (const Placement& p : kept) {
        if (p.boxes.size() != boxes.size() || !same_rect(p.panel, panel) || p.obstacles.size() != obstacles.size())
            continue;
        bool equal = true;
        for (std::size_t i = 0; i < boxes.size() && equal; ++i) equal = same(boxes[i], p.boxes[i]);
        for (std::size_t i = 0; i < obstacles.size() && equal; ++i) equal = same_rect(obstacles[i], p.obstacles[i]);
        if (equal) return p.placed;
    }
    if (kept.size() >= 8) kept.pop_back();
    kept.push_front({boxes, panel, obstacles, caliper_host::place_labels(boxes, panel, obstacles)});
    return kept.front().placed;
}

}  // namespace

void draw_scene_overlays(const SceneOverlayInput& in) {
    if (!in.draw || !in.labels || in.labels->empty()) return;
    const ImVec2 clip0 = in.draw->GetClipRectMin(), clip1 = in.draw->GetClipRectMax();
    const LabelRect panel{clip0.x, clip0.y, clip1.x, clip1.y};
    const float pad = 2.0f;
    std::vector<LabelBox> boxes;
    boxes.reserve(std::min(in.labels->size(), caliper_host::kMaxLabels));
    for (std::size_t i = 0; i < in.labels->size() && i < caliper_host::kMaxLabels; ++i) {
        const caliper_scene::LabelAnchor& a = (*in.labels)[i];
        const ImVec2 size = ImGui::CalcTextSize(a.text.c_str());
        LabelBox b;
        b.anchor_x = std::round(in.origin.x + a.x * in.scale_x);
        b.anchor_y = std::round(in.origin.y + a.y * in.scale_y);
        b.width = std::ceil(size.x) + 2 * pad;
        b.height = std::ceil(size.y) + 2 * pad;
        // An annotation's text first, then the item labels, nearest first.
        const bool first = in.annotations && i < in.annotations->size() && (*in.annotations)[i];
        b.priority = in.priorities && i < in.priorities->size() ? (*in.priorities)[i] : (first ? 0 : 1000) + (int)i;
        boxes.push_back(b);
    }
    // The markers' rings: drawn at their anchors over the picture, and kept free of labels.
    const float ring = std::max(5.0f, std::round(0.45f * ImGui::GetFontSize()));
    std::vector<LabelRect> rings;
    for (std::size_t i = 0; i < boxes.size(); ++i)
        if (in.rings && i < in.rings->size() && (*in.rings)[i])
            rings.push_back({boxes[i].anchor_x - ring - 1, boxes[i].anchor_y - ring - 1, boxes[i].anchor_x + ring + 1,
                             boxes[i].anchor_y + ring + 1});
    // The view's own texts are obstacles as the rings are, after them.
    std::vector<LabelRect> obstacles = rings;
    if (in.keep_clear) obstacles.insert(obstacles.end(), in.keep_clear->begin(), in.keep_clear->end());
    const std::vector<PlacedLabel>& placed = placed_for(boxes, panel, obstacles);
    const ImU32 accent = in.accent ? in.accent : in.ink;
    for (const LabelRect& r : rings) {
        const ImVec2 c((r.x0 + r.x1) * 0.5f, (r.y0 + r.y1) * 0.5f);
        in.draw->AddCircle(c, ring + 1.0f, in.shade, 0, 4.0f);
        in.draw->AddCircle(c, ring, accent, 0, 2.0f);
    }
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const ImU32 dot = (in.ink & 0x00FFFFFFu) | (IM_COL32_A_MASK & sternbild::col::shade(200));
    const ImU32 leader = (in.ink & 0x00FFFFFFu) | (IM_COL32_A_MASK & sternbild::col::shade(120));
    for (std::size_t i = 0; i < boxes.size() && i < placed.size(); ++i) {
        const LabelBox& b = boxes[i];
        const PlacedLabel& p = placed[i];
        const std::string& text = (*in.labels)[i].text;
        if (!p.shown) {
            in.draw->AddCircleFilled(ImVec2(b.anchor_x, b.anchor_y), 2.0f, dot);
            if (std::fabs(mouse.x - b.anchor_x) <= 5.0f && std::fabs(mouse.y - b.anchor_y) <= 5.0f &&
                mouse.x >= panel.x0 && mouse.x < panel.x1 && mouse.y >= panel.y0 && mouse.y < panel.y1)
                ImGui::SetTooltip("%s", text.c_str());
            continue;
        }
        if (p.leader) {
            const float cx = std::clamp(b.anchor_x, p.x, p.x + b.width), cy = std::clamp(b.anchor_y, p.y, p.y + b.height);
            in.draw->AddLine(ImVec2(b.anchor_x, b.anchor_y), ImVec2(cx, cy), leader, 1.0f);
        }
        const ImVec2 at(p.x + pad, p.y + pad);
        in.draw->AddText(ImVec2(at.x + 1.0f, at.y + 1.0f), in.shade, text.c_str());
        in.draw->AddText(at, in.ink, text.c_str());
    }
}

}  // namespace caliper_live_views
