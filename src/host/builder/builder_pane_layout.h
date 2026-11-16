#pragma once
// ============================================================================
// builder_pane_layout.h: the builder pane's regions and the canvas's drop
// targets as plain rectangles, computed without ImGui so that tests check
// them at any window size (docs/explanations/ui-principles.md, Checking a
// layout; design/builder-pane.md).
//
// The regions of a page in build mode, inside the work area the host's
// strips leave:
//
//   ┌──────────┬─────────────────────────────┬──────────────┐
//   │ Sources  │ canvas: the spec's panels   │ Properties   │
//   │          │                             │ (upper part) │
//   │          │                             ├──────────────┤
//   ├──────────┤                             │ chat (lower, │
//   │Primitives├─────────────────────────────┤ when given)  │
//   │          │ spec text (toggled)         │              │
//   └──────────┴─────────────────────────────┴──────────────┘
//
// Each column is a share of the width with a minimum that grows with the
// text scale; the canvas takes the rest and stays the largest region. On a
// small window the regions give way in this order: the text strip shrinks
// to its minimum and is then hidden, the right column and then the left
// column shrink to their minimums, then the right column is hidden, then
// the left. The canvas never goes below its minimum while a column is shown.
//
// Drop targets. While something is dragged over the canvas, the point under
// the pointer names one zone: the empty canvas; an edge of the whole layout
// (a thin band along the canvas's border, which places the dragged thing
// beside, above or below everything); an edge of a panel (its outer quarter,
// at most a few lines, which places it left of, right of, above or below that
// panel); the panel's centre (tabbed with it); or, for a source, one of the
// panel's slots, laid out as a column of rows in its centre.
// ============================================================================
#include "spec_runtime/spec_layout.h"

#include <string>
#include <vector>

namespace caliper_host {

struct BuilderPaneMetrics {
    float line = 15.0f;          // the UI face's size at this scale
    float rule = 1.0f;           // between regions: one physical rule
    float left_share = 0.17f;    // the left column's share of the width
    float right_share = 0.21f;   // the right column's
    float strip_share = 0.30f;   // the text strip's share of the centre's height
    float chat_share = 0.55f;    // the lower part's share of the right column, when shown
    float left_min = 210.0f;     // 14 lines
    float right_min = 270.0f;    // 18 lines
    float canvas_min_w = 360.0f; // 24 lines
    float canvas_min_h = 240.0f; // 16 lines
    float strip_min_h = 135.0f;  // 9 lines: a header and six lines of text
    float header = 24.0f;        // a region's header strip
    float row = 22.0f;           // a list row
    float pad = 8.0f;
    float primitives_h = 0.0f;   // the Primitives section: its header, seven rows and a pad
    float sources_min_h = 0.0f;  // the Sources section's least: header, filter and four rows
    float properties_min_h = 0.0f;
    float edge_band = 90.0f;     // a panel's edge zone, at most: six lines
    float root_band = 10.0f;     // the band along the canvas's border
    float slot_row = 30.0f;      // a slot's drop row
    float slot_w = 220.0f;       // its width, at most
    float divider_slop = 4.0f;   // a divider's grip on each side of its rule
    static BuilderPaneMetrics for_text_scale(float text_scale);
};

struct BuilderPaneLayoutInput {
    SpecRect area;               // the work area the host's strips leave
    bool build = true;           // build mode; without it the canvas takes the area
    bool text_open = false;      // the spec text strip
    bool chat = false;           // the right column's lower part (the side chat)
};

struct BuilderPaneLayout {
    bool left = false, right = false, strip = false, chat = false;   // shown
    SpecRect left_column, sources, primitives;
    SpecRect canvas;
    SpecRect strip_rect;
    SpecRect right_column, properties, chat_rect;
    std::vector<SpecRect> rules;          // 1 px rules between the regions
    std::vector<std::string> collapsed;   // what gave way, in order: "strip", "right", "left"
};

BuilderPaneLayout compute_builder_pane_layout(const BuilderPaneLayoutInput& in, const BuilderPaneMetrics& m);

// ---- drop targets -----------------------------------------------------------------------

enum class DropZone { none, empty, root, edge, centre, slot };
const char* drop_zone_name(DropZone z);

struct DropHit {
    DropZone zone = DropZone::none;
    std::string panel;   // edge, centre and slot: the panel under the point
    std::string where;   // root and edge: left, beside, above or below; centre: tabbed
    int slot = -1;       // slot: its index in the slot rows
};

// The zone under (x, y) in the canvas. `slots` is the number of slot rows a
// dragged source offers in a panel's centre (0 for a primitive or a panel),
// laid out under the panel's title strip of height `strip`. A point outside
// the canvas is DropZone::none.
DropHit hit_drop(const SpecLayout& layout, const SpecRect& canvas, float x, float y, int slots, float strip,
                 const BuilderPaneMetrics& m);

// The slot rows in a panel's rectangle: a column centred in it, one row per
// slot, cut to what fits under the panel's strip of height `strip`.
std::vector<SpecRect> drop_slot_rows(const SpecRect& panel, float strip, int slots, const BuilderPaneMetrics& m);

// The draggable divider under (x, y), a split of two children whose rule is
// within `divider_slop` of the point; null when there is none.
const SpecLayoutSplit* hit_divider(const SpecLayout& layout, float x, float y, const BuilderPaneMetrics& m);

// The ratio a divider dragged to (x, y) gives its split, snapped to
// hundredths from 0.05 to 0.95.
double divider_ratio(const SpecLayoutSplit& split, float x, float y);

// The panel whose title strip (its top `strip` pixels; a tab has none) holds
// (x, y), or "".
std::string hit_header(const SpecLayout& layout, float x, float y, float strip);

// The panel whose rectangle holds (x, y), or "".
std::string hit_panel(const SpecLayout& layout, float x, float y);

bool spec_rect_contains(const SpecRect& r, float x, float y);

}  // namespace caliper_host
