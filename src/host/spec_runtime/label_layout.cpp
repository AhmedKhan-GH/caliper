// The placement of labels (label_layout.h). Not built yet: each label sits
// at its anchor, shown, with no overlap check.
#include "spec_runtime/label_layout.h"

namespace caliper_host {

std::vector<PlacedLabel> place_labels(const std::vector<LabelBox>& boxes, const LabelRect& panel,
                                      const std::vector<LabelRect>& markers) {
    (void)panel;
    (void)markers;
    std::vector<PlacedLabel> out;
    out.reserve(boxes.size());
    for (const LabelBox& b : boxes) out.push_back({b.anchor_x, b.anchor_y, true, false});
    return out;
}

}  // namespace caliper_host
