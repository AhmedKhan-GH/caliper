#pragma once
// The layout of a spec's panels as plain rectangles (design/spec-format.md,
// Layout), computed without ImGui so that tests check it without a window.
//
// The layout tree is a panel id, a split or tabs. A split divides its space
// along its axis: `horizontal` side by side, `vertical` top to bottom; with
// two children the first takes `ratio` (0.5 by default) of the space, with
// more they share it equally. Tabs show one child at a time under a strip of
// tab labels. Between the children of a split lies a gap of `gap` pixels, the
// 1 px rule of the house style, which the runtime draws by filling the area
// with the rule colour before the panels.
//
// Every panel has a minimum size that grows with the text scale, and a node's
// minimum is what its children need. A split gives each child at least its
// minimum while the space allows, and otherwise shares the space in
// proportion to the minimums, so the rectangles never leave the area. Every
// edge is a whole pixel. Header-only parts need nothing but the spec library.
#include <caliper_builder/spec/spec.h>

#include <map>
#include <string>
#include <vector>

namespace caliper_host {

struct SpecRect {
    float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;
    float right() const { return x + w; }
    float bottom() const { return y + h; }
};

struct SpecLayoutMetrics {
    float min_panel_w = 180.0f;   // a panel's minimum width
    float min_panel_h = 120.0f;   // and height
    float tab_bar_h = 24.0f;      // the strip of tab labels
    float gap = 1.0f;             // between the children of a split

    // The metrics at a text scale: the minimums and the tab strip grow with
    // it, from 12 by 8 lines of the 15 px UI face at scale 1; the gap stays
    // one physical rule.
    static SpecLayoutMetrics for_text_scale(float text_scale);
};

// A placed panel, with the path of its layout node: the indices of the
// children from the root, joined by '.', "" for the root.
struct SpecLayoutPanel {
    std::string id;
    std::string path;
    SpecRect rect;
};

// A tabs node: its strip, its labels in order and the tab shown.
struct SpecLayoutTabs {
    std::string path;
    SpecRect bar;
    std::vector<std::string> labels;
    int active = 0;
};

struct SpecLayout {
    std::vector<SpecLayoutPanel> panels;   // the panels shown, in tree order
    std::vector<SpecLayoutTabs> tabs;      // every tabs node on a shown path
    // The panel shown at `id`, or null.
    const SpecLayoutPanel* find(const std::string& id) const;
};

// The tab shown for each tabs node, by its path; a node that is not listed,
// or whose index is out of range, shows its first tab.
using SpecTabChoice = std::map<std::string, int>;

// The minimum size of a node.
SpecRect spec_layout_min_size(const caliper_builder::spec::LayoutNode& node, const SpecLayoutMetrics& m);

// The rectangles of the layout `root` in `area`. A tab's label is its
// panel's title (or id), or a nested node's title, else the title of the
// first panel inside it.
SpecLayout compute_spec_layout(const caliper_builder::spec::LayoutNode& root, const caliper_builder::spec::Spec& spec,
                               const SpecRect& area, const SpecLayoutMetrics& m, const SpecTabChoice& tabs = {});

// The shares of `length` for children with minimums `mins` along one axis,
// after the gaps: two children with `ratio`, more children equally, each at
// least its minimum while the length allows, else in proportion to the
// minimums. Whole pixels that sum to the length.
std::vector<float> spec_layout_shares(float length, const std::vector<float>& mins, double ratio);

// The text of a panel's title: its title, else its id.
std::string spec_panel_title(const caliper_builder::spec::Panel& panel);

}  // namespace caliper_host
