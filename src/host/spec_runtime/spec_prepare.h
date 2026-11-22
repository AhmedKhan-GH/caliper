#pragma once
// The data of one spec panel made ready to draw (design/spec-format.md, The
// primitives), from the resolved data the session gives
// (caliper_builder/session/resolve.h). Pure data: no ImGui, so tests check
// it without a window.
//
// prepare_panel() reads the panel's slots and options and fills the part of
// PreparedPanel that its primitive draws, or a state in place of the data:
// a short noun phrase ("No data", "Source offline") and a name in mono, the
// source or code, never a sentence. Nothing here throws on any input.
//
// Every size is bounded, and each bound that applies is recorded in
// `bounds`, as the live views bound theirs:
//   - a series line of more than series_raw_max values is cut to a min/max
//     envelope of series_buckets buckets (at most twice that many points),
//     which keeps its spikes, as a live series does above 8192 values;
//   - a heatmap or image wider or taller than max_texture_side is sampled at
//     a stride on each axis;
//   - a point cloud of more than max_points points is sampled at a stride;
//   - a table shows at most max_table_rows rows;
//   - a text value or table cell is cut at max_cell_chars characters.
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/spec/spec.h>

#include <cstdint>
#include <string>
#include <vector>

namespace caliper_host {

struct PrepareLimits {
    size_t series_raw_max = 8192;
    size_t series_buckets = 2048;
    int max_texture_side = 2048;
    size_t max_points = 100000;
    size_t max_table_rows = 1000;
    size_t max_cell_chars = 256;
};

enum class PanelStateKind {
    ready,        // the data is drawn
    no_spec,      // no spec, or no such panel
    offline,      // a source the app serves, not attached (resolve.unavailable)
    empty,        // the data has no values
    mismatch,     // the data does not fit the slot
    too_large,    // past a bound of the resolver
    query_error,  // a query failed or timed out
    missing,      // a source, column or file that is not there
    error,        // anything else
};

struct PanelState {
    PanelStateKind kind = PanelStateKind::ready;
    std::string label;    // a noun phrase: "No data", "Source offline"
    std::string detail;   // a name in mono: the source, the binding or the code
    bool ready() const { return kind == PanelStateKind::ready; }
};

// The state of a resolver error code (resolve.unavailable and the like).
PanelState panel_state_for_error(const std::string& code, const std::string& detail);

struct SeriesLine {
    std::string name;            // the binding's column, or its source
    std::vector<double> x, y;    // finite points, decimated when long
    size_t count = 0;            // values before decimation
    double last = 0.0;           // the last finite value
    bool has_last = false;
};

struct PreparedSeries {
    std::vector<SeriesLine> lines;
    std::string x_name;          // the x binding's column, or "step" for the index
    bool x_index = true;         // x is the index 0, 1, 2 ...
};

// A grid of colours, for a heatmap or an image: rows x cols RGBA8, row-major,
// top row first.
struct PreparedGrid {
    int rows = 0, cols = 0;          // the drawn size, after any stride
    int source_rows = 0, source_cols = 0;
    int stride_r = 1, stride_c = 1;
    std::vector<uint8_t> rgba;
    std::vector<float> values;       // a heatmap's values as drawn (for hover)
    float vmin = 0.0f, vmax = 1.0f;  // the colormap's ends
    int colormap = 0;                // SpecColormap
    int channels = 1;                // an image's channels: 1, 3 or 4
};

struct PreparedPoints {
    bool three_d = false;
    std::vector<float> x, y, z;      // sampled at `stride` past max_points
    std::vector<float> color;        // empty: one flat colour
    std::vector<float> size;         // empty: one size
    float cmin = 0.0f, cmax = 1.0f, smin = 0.0f, smax = 1.0f;
    int colormap = 0;
    size_t total = 0;                // points before sampling
    size_t stride = 1;
    std::string color_name, size_name;
    // Per point, when `color` or `size` is bound: the colour through the
    // colormap (r | g<<8 | b<<16 | a<<24), and a factor of the base point
    // size from 0.5 at `smin` to 2 at `smax`.
    std::vector<uint32_t> rgba;
    std::vector<float> scale;
    // Class labels: a colour binding of an integer type whose values are
    // whole numbers from 0 to 255 with at most 16 distinct ones is drawn as
    // classes, label L in the live views' categorical colour L mod 16, as
    // the live points view draws classes, with a legend of the labels in
    // `classes` in place of the colormap's scale.
    std::vector<int> classes;
};

// The colour of class label `label` (r | g<<8 | b<<16 | a<<24): the live
// views' categorical palette.
uint32_t spec_class_color(int label);

struct PreparedHistogram {
    std::vector<double> counts;
    double lo = 0.0, hi = 1.0;
    double total = 0.0;              // values counted
    size_t outside = 0;              // finite values outside the range
};

struct PreparedTable {
    std::vector<std::string> columns;
    std::vector<bool> numeric;       // right-aligned figures
    std::vector<std::vector<std::string>> cells;
    size_t total_rows = 0;           // rows the resolver gave
    bool truncated = false;          // cut here or by the resolver
};

struct PreparedText {
    std::string text;
    std::string value;
    bool has_value = false;
};

// The ids of the first four colormaps of the format, as the live views number theirs, plus
// greys. All sixteen names map to ids 0 to 15 (caliper_builder::spec::colormap_id).
enum SpecColormap { kCmapViridis = 0, kCmapMagma = 1, kCmapRdbu = 2, kCmapGreys = 3 };
// The id of a colormap name, or the fallback for an unknown name.
int spec_colormap(const std::string& name, int fallback);
// The 256-entry RGBA8 table of a colormap id (r | g<<8 | b<<16 | a<<24), from the
// graphics layer's builtin_lut; viridis for an unknown id.
const uint32_t* spec_colormap_lut(int colormap);

struct PreparedPanel {
    std::string id;
    std::string title;
    caliper_builder::spec::Primitive primitive = caliper_builder::spec::Primitive::series;
    PanelState state;
    std::string info;                 // the header's info string, in mono
    std::string x_label, y_label, z_label;
    bool x_log = false, y_log = false;
    bool has_x_range = false, has_y_range = false;
    double x_range[2] = {0, 1}, y_range[2] = {0, 1};

    PreparedSeries series;
    PreparedGrid grid;                // heatmap, image
    PreparedPoints points;
    PreparedHistogram histogram;
    PreparedTable table;
    PreparedText text;
    // A scene panel (spec version 1): the live scene it draws, as catalog.list
    // names it ("model" for live:model), and its options. The runtime draws
    // the scene itself; nothing is prepared from data.
    std::string scene;
    caliper_builder::spec::SceneOptions scene_options;

    std::vector<std::string> bounds;  // the bounds that applied, in plain words
};

// `spec` names the catalog id of a source in a state's detail.
PreparedPanel prepare_panel(const caliper_builder::spec::Spec& spec, const caliper_builder::spec::Panel& panel,
                            const caliper_builder::PanelData& data, const PrepareLimits& limits = {});

// The catalog id at the root of a source's chain of `from` ("feed:gpu"),
// "sql" for a query, or the id itself when the spec does not have it.
std::string spec_source_origin(const caliper_builder::spec::Spec& spec, const std::string& source_id);

// Four significant digits, as the live views' headers write numbers.
std::string spec_fmt4(double v);

}  // namespace caliper_host
