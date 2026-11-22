#pragma once
// The layout of a spec's panels as plain rectangles (design/spec-format.md,
// Layout), computed without ImGui so that tests check it without a window.
//
// The layout tree is a panel id, a split or tabs. A split divides its space
// along its axis: `horizontal` side by side, `vertical` top to bottom; with
// `shares` (spec version 1) each child takes its share of the space, else
// with two children the first takes `ratio` (0.5 by default) of the space,
// and with more they share it equally. Tabs show one child at a time under a strip of
// tab labels. Between the children of a split lies a gap of `gap` pixels, the
// 1 px rule of the house style, which the runtime draws by filling the area
// with the rule colour before the panels.
//
// Every panel has a minimum size that grows with the text scale, and a node's
// minimum is what its children need. A leaf's `min` (spec version 1) sets
// its minimum in characters and lines of the text scale. A split gives each child at least its
// minimum while the space allows, and otherwise shares the space in
// proportion to the minimums, so the rectangles never leave the area. Every
// edge is a whole pixel. Header-only parts need nothing but the spec library.
#include <caliper_builder/spec/spec.h>

#include <functional>
#include <map>
#include <optional>
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
    // Controls (spec version 1, design/instruments.md): a control leaf's
    // minimum width, and its height, one control row, which it takes in a
    // vertical split while the other children share the rest. The height is
    // `control_height` of the control when set (the runtime sets it to
    // spec_control_height()), else control_row_h.
    float min_control_w = 120.0f;
    float control_row_h = 30.0f;
    // A leaf's `min` of [columns, lines] (design/instrument-interface.md,
    // Layout Shares): the width of one character and the height of one line
    // of the UI face.
    float char_w = 8.0f;
    float line_h = 18.0f;
    std::function<float(const caliper_builder::spec::Control&)> control_height;

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
    bool tab = false;   // a tab of a tabs node, whose label carries its title
    bool control = false;   // the leaf names a control of the spec, not a panel
};

// A tabs node: its strip, its labels in order and the tab shown.
struct SpecLayoutTabs {
    std::string path;
    SpecRect bar;
    std::vector<std::string> labels;
    int active = 0;
};

// A split node on a shown path, for the builder's divider drags: its path,
// its rectangle, its axis, the gap between its first two children and the
// length their shares divide. Only a split of two children takes a ratio, so
// only those dividers can be dragged.
struct SpecLayoutSplit {
    std::string path;
    SpecRect rect;
    bool horizontal = true;   // the children side by side
    size_t children = 2;
    SpecRect divider;         // between the first and the second child
    float length = 0.0f;      // the rectangle's length along its axis, less the gaps
    double ratio = 0.5;       // the spec's ratio, or 0.5
};

struct SpecLayout {
    std::vector<SpecLayoutPanel> panels;   // the panels shown, in tree order
    std::vector<SpecLayoutTabs> tabs;      // every tabs node on a shown path
    std::vector<SpecLayoutSplit> splits;   // every split node on a shown path, in tree order
    // The panel shown at `id`, or null.
    const SpecLayoutPanel* find(const std::string& id) const;
};

// The tab shown for each tabs node, by its path; a node that is not listed,
// or whose index is out of range, shows its first tab.
using SpecTabChoice = std::map<std::string, int>;

// The minimum size of a node. With the spec, a leaf that names a control
// has the minimum of a control: min_control_w by its height.
SpecRect spec_layout_min_size(const caliper_builder::spec::LayoutNode& node, const SpecLayoutMetrics& m);
SpecRect spec_layout_min_size(const caliper_builder::spec::LayoutNode& node, const SpecLayoutMetrics& m,
                              const caliper_builder::spec::Spec* spec);
// The height a node takes in a vertical split when everything in it is a
// control: a control leaf's height, the tallest child of a horizontal split
// of such nodes, the children of a vertical one with their gaps. Empty when
// the node holds a panel or tabs, which share the rest of the height.
std::optional<float> spec_layout_fixed_height(const caliper_builder::spec::LayoutNode& node,
                                              const caliper_builder::spec::Spec& spec, const SpecLayoutMetrics& m);

// The rectangles of the layout `root` in `area`. A tab's label is its
// panel's title (spec_panel_title), or a nested node's title, else the title
// of the first panel inside it.
SpecLayout compute_spec_layout(const caliper_builder::spec::LayoutNode& root, const caliper_builder::spec::Spec& spec,
                               const SpecRect& area, const SpecLayoutMetrics& m, const SpecTabChoice& tabs = {});

// The shares of `length` for children with minimums `mins` along one axis,
// after the gaps: two children with `ratio`, more children equally, each at
// least its minimum while the length allows, else in proportion to the
// minimums. Whole pixels that sum to the length.
std::vector<float> spec_layout_shares(float length, const std::vector<float>& mins, double ratio);
// The same with the weights of a split's `shares`, one per child: each child
// takes its weight's part of the length, a child whose part falls below its
// minimum keeps the minimum, and the others share the rest again by their
// weights. Empty weights, or a count that differs from the minimums', give
// the function above.
std::vector<float> spec_layout_shares(float length, const std::vector<float>& mins, double ratio,
                                      const std::vector<double>& weights);

// The title a panel is shown under: its title, else the names of what it
// draws, never its bare id: the y columns of a series ("loss, accuracy"),
// "value by step" for points over two columns, the source's catalog name for
// a whole source ("run42_metrics"; a derived or SQL source by its spec id),
// the value of a text panel, else the primitive ("Text").
std::string spec_panel_title(const caliper_builder::spec::Spec& spec, const caliper_builder::spec::Panel& panel);

// The name a binding draws under: its column, or its source's name as
// spec_panel_title() names a whole source.
std::string spec_binding_name(const caliper_builder::spec::Spec& spec, const std::string& binding);

}  // namespace caliper_host
