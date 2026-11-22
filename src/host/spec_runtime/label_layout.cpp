// The placement of labels (label_layout.h, design/instrument-interface.md,
// Label Placement).
//
// The labels go in order of priority, and among equal priorities in the
// order of the input, so that the same input always gives the same result.
// The markers and every label placed so far are obstacles. A pinned label
// sits with its top-left corner at its anchor whatever it covers, and is an
// obstacle for the labels after it. Every other label tries eight positions
// around its anchor at a near distance, then the same eight at three times
// that distance with a leader line, and takes the first position that lies
// inside the panel and overlaps no obstacle. A label that finds none is not
// shown, and its caller shows only the anchor. Labels past kMaxLabels in the
// order are not shown either.
#include "spec_runtime/label_layout.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// The live views' library (libcaliper, and the view layer of the web build)
// holds this file, because the scene overlays place their labels with it; the
// spec runtime links that library.
namespace caliper_host {

namespace {

// Two rectangles overlap when their interiors meet; touching edges do not.
bool overlaps(const LabelRect& a, const LabelRect& b) {
    return a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
}

bool inside(const LabelRect& r, const LabelRect& panel) {
    return r.x0 >= panel.x0 && r.y0 >= panel.y0 && r.x1 <= panel.x1 && r.y1 <= panel.y1;
}

// The near distance between an anchor and its label: half the label's
// height, at least 3 pixels. At three times this distance a label above or
// below its anchor clears the near label of the same anchor.
float near_distance(const LabelBox& b) { return std::max(3.0f, std::ceil(0.5f * b.height)); }

// The top-left corner of candidate `k` (0 to 7) at distance `d`: right
// above, right below, left above, left below, above, below, right, left.
void candidate(const LabelBox& b, int k, float d, float* x, float* y) {
    const float ax = b.anchor_x, ay = b.anchor_y, w = b.width, h = b.height;
    switch (k) {
        case 0: *x = ax + d; *y = ay - d - h; return;
        case 1: *x = ax + d; *y = ay + d; return;
        case 2: *x = ax - d - w; *y = ay - d - h; return;
        case 3: *x = ax - d - w; *y = ay + d; return;
        case 4: *x = ax - std::round(0.5f * w); *y = ay - d - h; return;
        case 5: *x = ax - std::round(0.5f * w); *y = ay + d; return;
        case 6: *x = ax + d; *y = ay - std::round(0.5f * h); return;
        default: *x = ax - d - w; *y = ay - std::round(0.5f * h); return;
    }
}

}  // namespace

std::vector<PlacedLabel> place_labels(const std::vector<LabelBox>& boxes, const LabelRect& panel,
                                      const std::vector<LabelRect>& markers) {
    std::vector<PlacedLabel> out(boxes.size());
    std::vector<std::size_t> order(boxes.size());
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return boxes[a].priority < boxes[b].priority; });

    std::vector<LabelRect> taken(markers.begin(), markers.end());
    taken.reserve(markers.size() + std::min(boxes.size(), kMaxLabels));
    const auto free_at = [&](const LabelRect& r) {
        if (!inside(r, panel)) return false;
        for (const LabelRect& t : taken)
            if (overlaps(r, t)) return false;
        return true;
    };

    for (std::size_t n = 0; n < order.size(); ++n) {
        const LabelBox& b = boxes[order[n]];
        PlacedLabel& p = out[order[n]];
        p.x = b.anchor_x;
        p.y = b.anchor_y;
        if (n >= kMaxLabels || !std::isfinite(b.anchor_x) || !std::isfinite(b.anchor_y) || !std::isfinite(b.width) ||
            !std::isfinite(b.height) || b.width < 0 || b.height < 0)
            continue;
        if (b.pinned) {
            p.shown = true;
            taken.push_back({b.anchor_x, b.anchor_y, b.anchor_x + b.width, b.anchor_y + b.height});
            continue;
        }
        const float d = near_distance(b);
        for (int ring = 0; ring < 2 && !p.shown; ++ring) {
            for (int k = 0; k < 8; ++k) {
                float x = 0, y = 0;
                candidate(b, k, ring == 0 ? d : 3.0f * d, &x, &y);
                const LabelRect r{x, y, x + b.width, y + b.height};
                if (!free_at(r)) continue;
                p.x = x;
                p.y = y;
                p.shown = true;
                p.leader = ring == 1;
                taken.push_back(r);
                break;
            }
        }
    }
    return out;
}

}  // namespace caliper_host
