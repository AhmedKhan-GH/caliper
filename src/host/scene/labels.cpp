#include "scene/labels.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace caliper_scene {
std::vector<LabelAnchor> project_labels(const SceneTree& tree, const ViewCamera& camera, uint32_t width,
                                        uint32_t height) {
    std::vector<LabelAnchor> out;
    const CameraMatrices cm = camera_matrices(camera, tree, width, height);
    if (!cm.ok || width == 0 || height == 0) return out;
    for (uint32_t i = 0; i < tree.entity_count(); ++i) {
        const EntityId e = tree.entity_at(i);
        if (!tree.drawable(e)) continue;
        const EntityId pcam = tree.pixel_space_of(e);
        const bool overlay = pcam && pcam == cm.through;
        for (Primitive p : {Primitive::Points, Primitive::Boxes}) {
            const Batch* b = tree.batch(e, p);
            if (!b) continue;
            const BatchArray* pos = b->array(p == Primitive::Points ? Role::Positions : Role::Centers);
            const BatchArray* ids = b->array(Role::ClassIds);
            if (!pos || !pos->cpu) continue;
            if (b->labels.empty() && !(ids && ids->cpu)) continue;
            const uint64_t n = pos->count;
            const uint64_t step = n > kMaxLabelsPerBatch ? (n + kMaxLabelsPerBatch - 1) / kMaxLabelsPerBatch : 1;
            const Mat4 m = overlay ? cm.pixel_to_clip * tree.pixel_local(e) : cm.proj * cm.view * tree.world(e);
            const uint32_t comps = pos->format == caliper_gfx::VertexFormat::F32x3 ? 3 : 2;
            for (uint64_t k = 0; k < n; k += step) {
                std::string text;
                if (b->labels.size() == n) text = b->labels[k];
                else if (b->labels.size() == 1) text = b->labels[0];
                else if (ids && ids->cpu) {
                    int32_t id;
                    // One row stands for every point, whatever its stride (stride_of).
                    std::memcpy(&id, static_cast<const uint8_t*>(ids->cpu) + k * (ids->count == 1 ? 0 : ids->stride), 4);
                    text = tree.class_label(uint32_t(id));
                }
                if (text.empty()) continue;
                float f[3] = {0, 0, 0};
                std::memcpy(f, static_cast<const uint8_t*>(pos->cpu) + (pos->stride ? k * pos->stride : 0), comps * 4);
                const Vec3 local{f[0], f[1], f[2]};
                double h[4];
                transform_h(m, local, h);
                if (!(h[3] > 0)) continue;
                const double x = h[0] / h[3], y = h[1] / h[3], z = h[2] / h[3];
                if (std::fabs(x) > 1 || std::fabs(y) > 1 || z < 0 || z > 1) continue;
                LabelAnchor a;
                a.entity = e;
                a.primitive = p;
                a.item = k;
                a.x = float((x + 1) * 0.5 * width);
                a.y = float((1 - y) * 0.5 * height);
                a.depth = overlay ? 0.0f : float(length(transform_point(tree.world(e), local) - cm.eye));
                a.text = std::move(text);
                out.push_back(std::move(a));
            }
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const LabelAnchor& a, const LabelAnchor& b) { return a.depth < b.depth; });
    if (out.size() > kMaxLabelsDrawn) out.resize(kMaxLabelsDrawn);
    return out;
}


}
