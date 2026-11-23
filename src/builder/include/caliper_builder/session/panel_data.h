#pragma once
// The data of one panel as the browser's spec page draws it (the command
// panel.data; design/spec-format.md, The browser). The app resolves a panel
// (resolve.h), so the transforms run once, where the data lives, and this
// module makes the resolved data compact and bounded with the bounds of the
// native spec runtime (src/host/spec_runtime/spec_prepare.h), so that a page
// never receives more than a panel can show:
//   - series: one line per y binding, or per row of a rank-2 y tensor (at
//     most 64), as finite (x, y) points without the points a log axis cannot
//     show; a line of more than 8,192 points becomes the min/max envelope of
//     2,048 buckets, as the native runtime and the live series decimate;
//   - heatmap and image: the values sampled at a stride so that no side is
//     longer than 2,048 (an image keeps its aspect with one stride);
//   - points: at most 100,000 points, sampled at a stride, without the points
//     whose position is not finite;
//   - histogram: the counts of its bins, with the native runtime's range and
//     bin rule, so the values themselves never travel;
//   - table: at most 1,000 rows, a text cell cut at 256 characters;
//   - text: the one value it shows.
// Arrays of numbers travel as {"dtype": "f32" | "f64", "shape": [...],
// "base64": "..."}, the little-endian bytes in base64; tables, cells and
// counts as JSON. The page names lines and axes, maps colours, infers class
// labels and formats figures with the same rules as the native runtime.
//
// The version of a panel's data is a short hash of the spec's canonical
// JSON, the catalog's revision and the panel's generation (resolve.h), so a
// page that sends the version it has gets "unchanged" without a resolve
// until one of the three changes. Nothing here throws.
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/spec/spec.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace caliper_builder {

// The native spec runtime's bounds (PrepareLimits in spec_prepare.h).
struct PanelDataLimits {
    std::size_t series_raw_max = 8192;   // a line longer than this is decimated
    std::size_t series_buckets = 2048;   // into this many buckets
    std::size_t series_rows = 64;        // lines of one rank-2 y tensor
    std::int64_t max_texture_side = 2048;
    std::size_t max_points = 100000;
    std::size_t max_table_rows = 1000;
    std::size_t max_cell_chars = 256;
};

// The version of the data of panel `panel_id` under `spec`, from the
// catalog's revision and the panel's generation: 16 hex digits.
std::string panel_data_version(const spec::Spec& spec, const std::string& panel_id, std::uint64_t catalog_revision,
                               std::uint64_t generation);

// The result of panel.data for resolved data: {"panel", "primitive",
// "version", "unchanged": false, "error", "state", "bounds", "data"}.
// `error` is the resolver's error, `state` a state of the data
// ({"kind": "empty" | "mismatch", "detail"}), and `data` the primitive's
// data, null with either.
nlohmann::json panel_data_result(const spec::Spec& spec, const PanelData& data, std::uint64_t catalog_revision,
                                 const PanelDataLimits& limits = {});

// The entities of a scene panel (design/instrument-interface.md, Scene
// Entities): panel_data_result() calls it for a scene panel with entities
// and it adds their arrays and states to the panel's `data` object.
// Defined in panel_data_scene.cpp.
void encode_scene_entities(const spec::Panel& panel, const PanelData& data, const PanelDataLimits& limits,
                           nlohmann::json* out);

// The result of panel.data when the caller has `version` already.
nlohmann::json panel_data_unchanged(const spec::Spec& spec, const std::string& panel_id, const std::string& version);

// ---- The viewer's programs and controls (panel_data_viewer.cpp) ----------
//
// The browser's spec page draws the instances of pieces on a scene panel with
// its WebAssembly views, and shows each control of the layout with its
// state, from panel.data alone: the viewer token opens no stream and runs no
// command that changes anything (design/command-layer.md, Origins).

class PieceLibrary;
class ValueStore;
class StepHistory;
class RunControl;

// What the viewer's answers read. `values` and `history` are the session's
// for its open spec, null for a saved spec that is not open.
struct ViewerInputs {
    const PieceLibrary* library = nullptr;
    std::string workspace;
    const ValueStore* values = nullptr;
    const StepHistory* history = nullptr;
};

// At most this many programs of one panel, and this many bytes of resources
// in one answer, within the resource history budget.
inline constexpr std::size_t kViewerProgramsMax = 64;
inline constexpr std::uint64_t kViewerResourceBytesMax = 256ull << 20;

// The instances of pieces whose draw half targets the scene panel
// `panel_id`, in the spec's order: each {"id", "title", "piece": {name,
// version, from}, "state", "error", "instance", "params", "resources"}. The
// state is "ready" with `instance`, the instance form of PROGRAM set as the
// app sends it to a script ({op "set", id, piece {name, version, hash},
// manifest, bind, params, panel, segments, common, derive, draw}) with the
// params' values of now, and the newest kept version of each published
// resource at the session's cursor ({resource, step, count, stride, width,
// height, base64}). A program of a foreign spec whose hash nobody reviewed is
// "review", and a missing piece, another hash than the pin's or imports that
// do not resolve "refused", each with `error` and without files. The piece
// comes from the library, this workspace or the spec by the pin's name and
// version only. `digest` gets 16 hex digits that change when anything the
// page feeds its module changes.
nlohmann::json viewer_programs(const spec::Spec& spec, const std::string& panel_id, const ViewerInputs& in,
                               std::string* digest);

// The result of panel.data for a control `control` of the layout: primitive
// "control" and `data` {"id", "kind", "title", "bind", "value", "text",
// "enabled", "why", and "min", "max", "choices" when the control has them}:
// the state the app's control shows (a toggle of intervene:<point> is on
// while the run has an ablation of the point that has not ended, with the
// app's text), `enabled` false and `why` the reason a viewer cannot change
// it. `run` is the session's run control, null without one.
nlohmann::json control_data_result(const spec::Spec& spec, const spec::Control& control, const ViewerInputs& in,
                                   RunControl* run, const std::string& version);

}  // namespace caliper_builder
