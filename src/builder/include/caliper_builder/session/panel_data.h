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

}  // namespace caliper_builder
