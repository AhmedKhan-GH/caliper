#pragma once
// The placement of labels (design/instrument-interface.md, Label
// Placement): a pure function over rectangles, without ImGui, so that tests
// check it without a window and the browser can build it.
//
// Labels are placed in order of priority. For each, eight positions around
// the anchor are tried, then the same eight at three times the distance with
// a leader line; the first that overlaps no placed label and no marker and
// stays inside the panel wins. A label with no free position shows only its
// anchor. The same input always gives the same placement, and a panel
// places at most kMaxLabels labels. Defined in label_layout.cpp.
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_host {

inline constexpr std::size_t kMaxLabels = 512;

struct LabelRect {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};

// One label to place: its anchor, its size, its priority (lower first:
// picks and state, then annotations in spec order, then item labels nearest
// first) and whether it is pinned at its anchor.
struct LabelBox {
    float anchor_x = 0, anchor_y = 0;
    float width = 0, height = 0;
    int priority = 0;
    bool pinned = false;
};

struct PlacedLabel {
    float x = 0, y = 0;     // the top-left corner of the text
    bool shown = false;     // false: only the anchor shows, and a hover shows the text
    bool leader = false;    // placed at three times the distance, with a leader line
};

// One result for each box, in the order of `boxes`.
std::vector<PlacedLabel> place_labels(const std::vector<LabelBox>& boxes, const LabelRect& panel,
                                      const std::vector<LabelRect>& markers);

// The ellipsis that ends a cut text, U+2026 in UTF-8.
inline constexpr const char* kTextEllipsis = "\xE2\x80\xA6";

// A text cut to fit `width` (design/instrument-interface.md, Label
// Placement): `text` itself when it fits; else its longest prefix that ends
// on a whole UTF-8 character and on no space, followed by kTextEllipsis, that
// fits; else "" when not even the ellipsis fits. `width_of` measures a text,
// as ImGui's text size does; a longer prefix must never measure narrower.
// The search measures O(log n) prefixes, thus a text of 1,024 bytes costs
// about a dozen measures.
std::string fit_with_ellipsis(std::string_view text, float width,
                              const std::function<float(std::string_view)>& width_of);

}  // namespace caliper_host
