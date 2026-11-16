#pragma once
// The resolved data of a panel: for each bound slot, the data the spec
// runtime will draw (design/builder-session.md, Resolving a panel's data).
// It is the data path of the future spec runtime and of panel.render.
//
// resolve_panel() walks each binding of one panel of a spec:
// - The binding names a spec source; a chain of `from` is followed to its
//   root, a `catalog` or `sql` source (at most limits::kSources hops, so a
//   cycle stops).
// - A tensor root (a `tensor:` catalog source) is mapped from its file through
//   the loader store, which holds the data of every `table:` and `tensor:`
//   source in this phase, and then the transforms of the root and of each derived
//   source down the chain are applied in order with the loaders library's
//   tensor_ops. The size of every intermediate result is computed before it is
//   made and refused with resolve.too_large past `max_tensor_bytes`.
// - A tabular root (a `table:` catalog source, or a `sql` source) is read once
//   per call through the store's bounded read-only query: a table selects the
//   columns the panel needs, or all of them for a whole-source binding; a
//   `sql` source runs as written. The rows are cut at `max_rows`, and at
//   `max_cells` cells in all; `truncated` says so.
// - A column binding gives one column; a whole tabular binding gives a table
//   (for table.rows, the columns of options.columns, in that order); a tensor
//   binding gives the tensor.
// Sources the app serves (`live:`, `feed:`, `metric:`) answer
// resolve.unavailable until the wiring adds their data.
//
// Errors: panel.not_found, resolve.unknown_source, resolve.cycle,
// resolve.unknown_column, resolve.not_tabular, resolve.not_a_tensor,
// resolve.unavailable, resolve.too_large, resolve.query_failed,
// resolve.timeout and the transform.* and loader.* codes of the libraries
// below. Nothing here throws.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/query.h>
#include <caliper_builder/loaders/store.h>
#include <caliper_builder/loaders/tensor.h>
#include <caliper_builder/spec/spec.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {

struct ResolveLimits {
    std::uint64_t max_rows = 100000;                       // rows read from one tabular source
    std::uint64_t max_cells = std::uint64_t(1) << 22;      // rows times columns of one tabular source
    std::uint64_t max_tensor_bytes = std::uint64_t(256) << 20;  // one tensor made by a transform
    std::uint32_t query_timeout_ms = 10000;
};

struct SlotData {
    enum class Kind { tensor, column, table };

    std::string slot;        // "values", "y"
    std::size_t index = 0;   // the position in a list slot (series.y), else 0
    std::string binding;     // "metrics.loss"
    std::string source;      // the spec source the binding names
    std::string column;      // the column of a column binding; empty otherwise
    Kind kind = Kind::tensor;
    TensorData tensor;                  // Kind::tensor
    std::vector<ColumnInfo> columns;    // Kind::column: the one column; Kind::table: every column given
    nlohmann::json values = nlohmann::json::array();  // Kind::column: the cells; Kind::table: rows of cells
    bool truncated = false;             // the rows were cut at a limit
};

const char* slot_kind_name(SlotData::Kind kind);

struct PanelData {
    Error error;
    std::string panel;
    spec::Primitive primitive = spec::Primitive::series;
    std::vector<SlotData> slots;  // in slot name order, list slots in their order
};

PanelData resolve_panel(const spec::Spec& spec, std::string_view panel_id, LoaderStore& store,
                        const ResolveLimits& limits = {});

// A summary of resolved data for a test or a log: each slot's kind, shape or
// row count, and up to `max_values` of its values.
nlohmann::json panel_data_summary(const PanelData& data, std::size_t max_values = 16);

}  // namespace caliper_builder
