#include "live/model3d_presentation.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace caliper_live::model3d {
namespace {

Rect valid(Rect rect) {
    if (!std::isfinite(rect.x)) rect.x = 0;
    if (!std::isfinite(rect.y)) rect.y = 0;
    if (!std::isfinite(rect.width) || rect.width < 0) rect.width = 0;
    if (!std::isfinite(rect.height) || rect.height < 0) rect.height = 0;
    return rect;
}

int priority(const Group& group) {
    if (group.layer_type == "full_attention" || group.layer_type == "full") return 0;
    return group.layer_index < 0 ? 1 : 2;
}

bool collides(Rect a, Rect b, double gap) {
    return a.y < b.y + b.height + gap && b.y < a.y + a.height + gap;
}

bool usable(Rect rect) {
    return std::isfinite(rect.x) && std::isfinite(rect.y) &&
           std::isfinite(rect.width) && std::isfinite(rect.height) &&
           rect.width > 0 && rect.height > 0;
}

} // namespace

ViewportLayout viewport_layout(Rect available, double label_width, double gap) {
    available = valid(available);
    if (!std::isfinite(label_width)) label_width = 0;
    if (!std::isfinite(gap)) gap = 0;
    label_width = std::clamp(label_width, 0.0, available.width * 0.35);
    gap = label_width > 0 ? std::clamp(gap, 0.0, available.width * 0.05) : 0;
    const double scene_width = std::max(0.0, available.width - label_width - gap);
    return {{available.x, available.y, scene_width, available.height},
            {available.x + scene_width + gap, available.y, label_width, available.height}};
}

Rect column_beside_stack(const Scene& scene, const Camera& camera,
                         Rect scene_viewport, Rect reserved_column, double gap) {
    if (scene.groups.empty() || !usable(scene_viewport) || !usable(reserved_column) ||
        reserved_column.x < scene_viewport.x) return reserved_column;
    if (!std::isfinite(gap) || gap < 0) gap = 0;
    const double aspect = scene_viewport.width / scene_viewport.height;
    double right = -std::numeric_limits<double>::infinity();
    for (const Group& group : scene.groups) {
        if (!usable(group.bounds) || !std::isfinite(group.z)) return reserved_column;
        for (double x : {group.bounds.x, group.bounds.x + group.bounds.width}) {
            for (double y : {group.bounds.y, group.bounds.y + group.bounds.height}) {
                const Projection corner = project({x, y, group.z}, camera, aspect, scene_viewport);
                // Offscreen X/Y is safe: the render target clips to the scene
                // rectangle. A near/far-plane crossing is not, since its
                // unprojected edge cannot bound a readable label column.
                if (!std::isfinite(corner.x) || !std::isfinite(corner.y) ||
                    !std::isfinite(corner.depth) || corner.depth <= 0 || corner.depth >= 1)
                    return reserved_column;
                right = std::max(right, corner.x);
            }
        }
    }
    Rect column = reserved_column;
    column.x = std::clamp(right + gap, scene_viewport.x, reserved_column.x);
    return column;
}

std::vector<LayerLabel> label_layout(const Scene& scene, const Camera& camera,
                                   Rect scene_viewport, Rect label_column,
                                   double line_height, double gap) {
    scene_viewport = valid(scene_viewport);
    label_column = valid(label_column);
    if (scene_viewport.width <= 0 || scene_viewport.height <= 0 ||
        label_column.width <= 0 || label_column.height <= 0 ||
        !std::isfinite(line_height) || line_height <= 0 || line_height > label_column.height)
        return {};
    if (!std::isfinite(gap) || gap < 0) gap = 0;

    std::vector<LayerLabel> candidates;
    candidates.reserve(scene.groups.size());
    const double aspect = scene_viewport.width / scene_viewport.height;
    for (size_t i = 0; i < scene.groups.size(); ++i) {
        const Group& group = scene.groups[i];
        const Vec3 anchor{group.bounds.x + group.bounds.width,
                          group.bounds.y + group.bounds.height * 0.5, group.z};
        const Projection projected = project(anchor, camera, aspect, scene_viewport);
        if (!projected.visible || !std::isfinite(projected.x) || !std::isfinite(projected.y)) continue;
        const double top = projected.y - line_height * 0.5;
        // Do not move a label away from its layer to make it fit. Omitting a
        // partly clipped row preserves both alignment and readable text.
        if (top < label_column.y || top + line_height > label_column.y + label_column.height) continue;
        candidates.push_back({i, {label_column.x, top, label_column.width, line_height},
                              projected.x, projected.y});
    }
    std::stable_sort(candidates.begin(), candidates.end(), [&](const LayerLabel& a, const LayerLabel& b) {
        const int pa = priority(scene.groups[a.group]), pb = priority(scene.groups[b.group]);
        if (pa != pb) return pa < pb;
        if (a.bounds.y != b.bounds.y) return a.bounds.y < b.bounds.y;
        return a.group < b.group;
    });

    std::vector<LayerLabel> result;
    result.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (std::none_of(result.begin(), result.end(), [&](const LayerLabel& existing) {
                return collides(candidate.bounds, existing.bounds, gap);
            })) result.push_back(candidate);
    }
    std::sort(result.begin(), result.end(), [](const LayerLabel& a, const LayerLabel& b) {
        if (a.bounds.y != b.bounds.y) return a.bounds.y < b.bounds.y;
        return a.group < b.group;
    });
    return result;
}

} // namespace caliper_live::model3d
