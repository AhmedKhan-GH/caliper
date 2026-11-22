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

#include "scene/camera.h"   // StartingCamera

#include <caliper_builder/session/values.h>   // PickFields
#include <caliper_builder/spec/annotation.h>
#include <caliper_builder/spec/entity.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace caliper_live_views {
class SceneView;
}

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
    // A bound text (the options `format` and `fields`): the panel draws it at
    // each frame from the values at the cursor, with the bound value as its
    // field `value` (null while the slot has none).
    bool bound = false;
    nlohmann::json value_json;
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

// ---- Scene entities (design/instrument-interface.md, Scene Entities and Data Box) ----------

// The data box of a scene panel's entities: the range of the data on each axis, x, y and z, and
// the box that the entities are drawn in, centred at the origin of the panel's scene. A wide flat
// range becomes a box whose sides differ at most four times (`box = "auto"`), or the proportions
// that the option `box` gives; the longest side is 2 scene units.
struct EntityBox {
    double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};   // data units
    double half[3] = {1, 1, 1};                    // the drawn box's half sizes, scene units
    // One axis from data units into the box, and back.
    double to_box(int axis, double v) const;
    double to_data(int axis, double b) const;
};

// What one entity draws: its arrays in the box's units, ready for one part of the scene view
// (points, lines as one strip or segments, or a surface as a mesh of the grid's cells), and
// what a pick on it needs to give the fields of the section Pick Fields.
struct EntityDraw {
    std::string entity;                       // the entity's path
    std::string path;                         // "spec/<path>" in the panel's scene
    caliper_builder::spec::DrawKind draw = caliper_builder::spec::DrawKind::points;
    caliper_builder::Error error;             // entity.shape, entity.length, entity.limit or the resolve error
    std::vector<float> positions;             // (N, 3), scene units
    std::vector<double> values;               // (N), data values for the colormap; empty for a flat colour
    std::vector<uint8_t> colors;              // (N, 4); empty when absent
    std::vector<float> radii;                 // points: (N), scene units; empty when absent
    std::vector<float> normals;               // surface: (V, 3)
    std::vector<int32_t> indices;             // surface: (T, 3), the cells whose four heights are numbers
    bool strip = false;                       // lines: one strip through the points, else segments
    int64_t rows = 0, cols = 0;               // surface: the grid
    std::vector<double> data[3];              // each item's x, y and z in data units
    std::vector<double> row_coord, col_coord; // surface: the x of each row, the y of each column
    std::vector<int64_t> steps;               // the step of each row (surface) or item (lines), after a window
    bool column_from_y = false, row_from_x = false;   // surface: the input's indices from the y and x slots
    int colormap = 0;
    double vmin = 0, vmax = 1;
    uint32_t flat_rgba = 0;                   // 0: the scene's default colour
    float size_px = 0, width_px = 2;
    bool square = false;
    int depth_offset = 0;
    bool visible = true;
    uint64_t generation = 0;
};

struct EntityScene {
    std::string up = "+z";                    // the panel's option up, else z up
    bool walls = true;                        // the option axes: the back walls of the box with a grid
    EntityBox box;
    std::vector<EntityDraw> draws;            // in the panel's entity order
    bool empty() const { return draws.empty(); }
};

// Builds the entities of a scene panel from their resolved arrays (PanelData::entities). An
// entity with an error, a wrong shape or a length that differs between its slots draws nothing
// and keeps its error. Literal options apply; a reference takes its default.
EntityScene build_entity_scene(const caliper_builder::spec::Panel& panel, const caliper_builder::PanelData& data);

// The pick fields of a hit on item `item` of the entity part at `path` ("spec/<entity path>"): a
// point or a line vertex, or a vertex of a surface's cell. Empty when `path` is no entity.
std::optional<caliper_builder::PickFields> entity_pick_fields(const EntityScene& scene, const std::string& path,
                                                              uint64_t item);

// The starting camera of a scene panel from its option view_camera: an orbit table, a saved
// view's name (the option `views`), or a camera entity's path. Angles are in degrees.
caliper_scene::StartingCamera spec_starting_camera(const caliper_builder::spec::Panel& panel);

// One annotation of a scene panel as the scene draws it (design/instrument-interface.md,
// Annotation Table): a marker or a label at a point in data units or at an entity's item, or a
// threshold or a band across the data box on one axis, with its text at the cursor.
struct SceneAnnotationMark {
    std::string id;
    caliper_builder::spec::AnnotationKind kind = caliper_builder::spec::AnnotationKind::label;
    double at[3] = {0, 0, 0};     // marker and label: the point in data units
    std::string entity;           // marker and label at an entity's item: its path, with `index`
    int64_t index = -1;
    int axis = 0;                 // threshold and band: 0 x, 1 y, 2 z
    double lo = 0, hi = 0;        // threshold: lo == hi
    std::string text;
    bool muted = false;
};

// Writes a scene panel's annotations into its scene view, under "spec-overlay/<kind>/<id>": a
// marker as the anchor of a ring that the scene's overlay draws on top (scene_overlays.h) and a
// label as a small dot, each with its text as the item's label, and a threshold or a band as the
// outline of its plane or its two planes across the data box. Each
// point maps through the scene's data box; a mark whose entity item does not exist is left out.
// `drawn` holds the paths written by the last call, and the paths it no longer writes are
// cleared. The scene's labels place these texts before the items' own (scene_overlays.h).
void apply_annotation_marks(caliper_live_views::SceneView& view, const EntityScene& scene,
                            const std::vector<SceneAnnotationMark>& marks, std::vector<std::string>* drawn);

class SpecScene;
// The entities that a SpecScene (spec_entities.h) built last.
const EntityScene& spec_scene_content(const SpecScene& scene);

// Writes the entities into a scene view that the panel owns: the scene's up, one part for each
// entity that draws (points, lines, or a surface as a mesh) with its colormap, range, size, width
// and depth offset, and a clear for each entity that no longer draws. `drawn` holds the paths
// written by the last call. The scene's bounds are the data box, so that the fit frames it. With
// walls, the three walls of the box away from the starting camera, with grid lines at round
// numbers of the data, go to "spec-box/walls".
void apply_entity_scene(caliper_live_views::SceneView& view, const EntityScene& scene,
                        std::vector<std::string>* drawn);

}  // namespace caliper_host
