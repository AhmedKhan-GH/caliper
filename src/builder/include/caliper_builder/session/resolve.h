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
// - A root that the app serves (a `live:`, `feed:` or `metric:` catalog
//   source) is read through the CatalogDataSource registered for its kind in
//   DataSources (below): a live view gives its newest snapshot as a tensor,
//   on which the transforms run as on any tensor; a feed channel and a metric
//   give columns. Without a registered source such a root answers
//   resolve.unavailable.
//
// Refresh. Every served source has a generation that grows whenever its data
// changes. PanelData::generation and panel_generation() mix the generations
// of a panel's served roots, so a caller that drew a panel learns whether its
// data changed by comparing two numbers, without reading any data. The result
// of a chain of transforms over a served tensor is cached in DataSources by
// the chain and the generation it was computed from, so the transforms run at
// most once per new snapshot and chain, however many panels or frames ask.
//
// Errors: panel.not_found, resolve.unknown_source, resolve.cycle,
// resolve.unknown_column, resolve.not_tabular, resolve.not_a_tensor,
// resolve.unavailable, resolve.too_large, resolve.query_failed,
// resolve.timeout and the transform.* and loader.* codes of the libraries
// below. Nothing here throws.
#include <caliper_builder/catalog/catalog_entry.h>
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/query.h>
#include <caliper_builder/loaders/store.h>
#include <caliper_builder/loaders/tensor.h>
#include <caliper_builder/session/chain_hooks.h>
#include <caliper_builder/spec/spec.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <mutex>
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
    std::uint64_t generation = 0;       // of the served source it was read from; 0 for a loader output
    // After a `window` step: the step of each row of `tensor` (W); empty otherwise.
    TensorData steps;
    // While the cursor is held (design/instrument-interface.md, Latest at the
    // Cursor): the kept step whose value the slot shows; empty while the
    // cursor is live or when the source keeps no steps.
    std::optional<std::int64_t> step;
    // The cursor is held but the source keeps no step at or before it, so the
    // slot shows the newest value; the panel's info line says "Live only".
    bool live_only = false;
    // After an operator (`surface`, `path`): its named geometry arrays;
    // empty otherwise (design/instrument-interface.md, Operators).
    GeometryArrays geometry;
};

const char* slot_kind_name(SlotData::Kind kind);

// The resolved arrays of one entity of a scene panel
// (design/instrument-interface.md, Array Resolution).
struct EntityData {
    std::string path;
    Error error;                    // entity.shape, entity.length or a resolve error
    std::vector<SlotData> slots;    // one per bound slot; a source id gives its geometry arrays
    std::uint64_t generation = 0;
};

struct PanelData {
    Error error;
    std::string panel;
    spec::Primitive primitive = spec::Primitive::series;
    std::vector<SlotData> slots;  // in slot name order, list slots in their order
    std::vector<EntityData> entities;  // scene panels: one per entity, in spec order
    // The generations of the served roots the panel binds, mixed in binding
    // order, with the cursor and the versions of the references it reads;
    // 0 when it reads neither. Equal to panel_generation() for as long as
    // none of those sources and references changes.
    std::uint64_t generation = 0;
};

// ---- Sources that the app serves ------------------------------------------

// The bounds a CatalogDataSource keeps to when it gives data.
struct ServedRequest {
    std::uint64_t max_rows = 100000;                       // rows of a tabular source
    std::uint64_t max_cells = std::uint64_t(1) << 22;      // rows times columns
    std::uint64_t max_bytes = std::uint64_t(256) << 20;    // bytes of a tensor
};

// The current data of one served source.
struct ServedData {
    Error error;                     // resolve.unavailable when the source is gone or has no data yet
    std::uint64_t generation = 0;    // of the data given
    bool tabular = false;
    TensorData tensor;                                // a tensor-shaped source: owned, never changed after
    std::vector<ColumnInfo> columns;                  // a tabular source, with DuckDB type names
    nlohmann::json rows = nlohmann::json::array();    // rows of cells in column order; nan, inf and -inf as strings
    bool truncated = false;                           // the source had more rows than the request allowed
};

// The data of the sources of one kind that the app serves (live views, feed
// channels or metrics), by catalog id. The app implements one per kind and
// registers it with DataSources::set(). Both functions may be called from any
// thread, at the same time, and must not throw.
class CatalogDataSource {
public:
    virtual ~CatalogDataSource() = default;
    // The generation of the source's current data: it grows whenever the data
    // changes, and is 0 for an unknown source or one with no data yet. It must
    // be cheap and never wait long, because the spec runtime may ask it once
    // per panel and frame.
    virtual std::uint64_t generation(const std::string& id) = 0;
    // The current data, within the request's bounds, with its generation. A
    // tensor is a copy that the source never changes afterwards.
    virtual ServedData read(const std::string& id, const ServedRequest& request) = 0;
};

// The registered data sources, one per served kind, and the cache of
// transformed tensors. The session owns one; the app registers its sources
// with set(). Thread-safe. No function here throws.
class DataSources {
public:
    struct Options {
        std::size_t max_cache_entries = 64;                       // transformed tensors kept
        std::uint64_t max_cache_bytes = std::uint64_t(256) << 20;  // their bytes in all
    };
    DataSources();
    explicit DataSources(Options options);
    DataSources(const DataSources&) = delete;
    DataSources& operator=(const DataSources&) = delete;

    // Registers the source of `kind` (live, feed, metric or run), or removes it for
    // null; false for table and tensor. The cache is emptied, since the new
    // source's generations are not the old one's.
    bool set(CatalogKind kind, std::shared_ptr<CatalogDataSource> source);
    std::shared_ptr<CatalogDataSource> get(CatalogKind kind) const;
    // The generation of a served catalog id from its source; 0 when no source
    // is registered for its kind or the id is not served.
    std::uint64_t generation(std::string_view catalog_id) const;

    struct CacheStats {
        std::uint64_t computed = 0;  // chains whose transforms ran
        std::uint64_t reused = 0;    // chains answered from the cache
        std::size_t entries = 0;
        std::uint64_t bytes = 0;
    };
    CacheStats cache_stats() const;

    // The cache that resolve_panel() uses: the result of a chain of transforms
    // (named by `key`) computed from one generation of its served root. A
    // tensor larger than max_cache_bytes is not kept; the least recently used
    // entries go first.
    bool find_cached(const std::string& key, std::uint64_t generation, TensorData* out);
    void store_cached(const std::string& key, std::uint64_t generation, const TensorData& tensor);

private:
    struct Cached {
        std::uint64_t generation = 0;
        TensorData tensor;
        std::list<std::string>::iterator use;
    };
    Options options_;
    mutable std::mutex mutex_;
    std::shared_ptr<CatalogDataSource> live_, feed_, metric_, run_;
    std::map<std::string, Cached> cache_;
    std::list<std::string> uses_;  // keys, most recently used first
    std::uint64_t cache_bytes_ = 0;
    std::uint64_t computed_ = 0;
    std::uint64_t reused_ = 0;
};

// `sources` may be null, and then every served root answers
// resolve.unavailable.
// `chain` gives the references their values and the cursor its kept steps
// (chain_hooks.h); null resolves every reference to its default and reads
// the newest values.
PanelData resolve_panel(const spec::Spec& spec, std::string_view panel_id, LoaderStore& store,
                        const ResolveLimits& limits = {}, DataSources* sources = nullptr,
                        const ChainContext* chain = nullptr);

// The generation that resolve_panel() would give panel `panel_id` now, from
// the sources' generation() alone: a few map lookups and one cheap call per
// served root, with no data read. 0 when the panel binds no served source or
// does not exist. The spec runtime calls it once per frame for a panel it
// shows, and resolves the panel again only when it differs from the
// generation of the data it drew.
//
// `chain`, when given, is the one resolve_panel() takes: while the cursor is
// held, a panel that reads kept steps (a source that keeps steps, or a
// window) mixes in the cursor's step, so that a move of the cursor changes
// the generation. Without it the cursor is taken as live.
//
// With a value store in `chain`, the versions of every reference that the
// panel reads (its slots, its options, its entities' slots and options and
// the steps of its chains) are mixed in too, so that a changed control,
// state or pick changes the generation also while no source changes, and a
// panel that reads references but no served source has a generation too.
std::uint64_t panel_generation(const spec::Spec& spec, std::string_view panel_id, const DataSources& sources,
                               const ChainContext* chain = nullptr);

// The panel with the references of its options and of its entities' options
// applied (design/instrument-interface.md, Reference Places): `range`,
// `x_range`, `y_range` and `colormap` take the value that their reference
// reads now, or keep the panel's own default while it reads none; the table
// of an entity option's reference keeps its form and takes the value as its
// `default`, which the scene reads. A null `chain` or one without a value
// store gives each reference its default.
spec::Panel panel_with_values(const spec::Panel& panel, const ChainContext* chain);

// ---- Kept steps (history.h) -------------------------------------------------

// The steps that the app keeps for `source` before the budget
// (design/instrument-interface.md, Step History): its `history`, else 64
// (kDefaultLiveDepth) for a source whose root is a live tensor and that has
// no `from`, else 0. A chain that holds a window keeps none, and nor does a
// live root whose first step is an operator. Defined in history.cpp.
std::int64_t history_depth(const spec::Spec& spec, const spec::Source& source);

// The history point of `source`: how many steps of its chain, counted from
// the root's first step down to the source's last, the kept value has
// applied. It is the longest run of shipped steps (slice, reduce, normalise)
// from the start that read no reference. Defined in history.cpp.
std::size_t history_point(const spec::Spec& spec, const spec::Source& source);

// The catalog ids of the live views whose values `history` keeps for the
// spec it was last configured with, so that the live catalog copies each of
// their snapshots. Any thread. Defined in history.cpp.
std::vector<std::string> kept_catalog_ids(const StepHistory& history);

// A summary of resolved data for a test or a log: each slot's kind, shape or
// row count, and up to `max_values` of its values.
nlohmann::json panel_data_summary(const PanelData& data, std::size_t max_values = 16);

}  // namespace caliper_builder
