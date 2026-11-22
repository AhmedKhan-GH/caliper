#pragma once
// CatalogEntry: one source of the catalog, which a spec can bind by its id
// `<kind>:<name>` (design/catalog.md). An entry is a plain value. Its JSON
// form always holds all fourteen keys of the contracts, with null where a value
// does not apply or is not known.
//
// catalog_entry_from_json() is strict: a missing key, an unknown key, a value
// of the wrong type or a value out of bounds gives the error
// "catalog.invalid_entry", with `where` naming the key ("columns[3].type").
// validate_catalog_entry() runs the same checks on a struct, so the entries a
// provider builds in C++ are held to the same rules. parse_catalog_id() splits
// an id at its first colon and checks both parts ("catalog.invalid_id").
// Strings must be valid UTF-8, so that a JSON dump of an entry never throws.
// No function here throws.
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {

// The six kinds of source. table, feed and metric are tabular (columns and a
// row count); tensor, live and run are tensor-shaped (a dtype and a shape). A
// run source is a series of a managed run of the model runner,
// run:<run id>/<view> (design/hosted-runs.md).
enum class CatalogKind { table, tensor, live, feed, metric, run };

const char* catalog_kind_name(CatalogKind kind);
std::optional<CatalogKind> catalog_kind_from_name(std::string_view name);
bool catalog_kind_is_tabular(CatalogKind kind);

// The element types of a tensor-shaped source, named as in the contracts.
enum class CatalogDtype { u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, boolean };

const char* catalog_dtype_name(CatalogDtype dtype);  // "bool" for boolean
std::optional<CatalogDtype> catalog_dtype_from_name(std::string_view name);
std::size_t catalog_dtype_size(CatalogDtype dtype);  // in bytes

// The bounds of an entry. Sizes of strings are in bytes of UTF-8.
struct CatalogLimits {
    static constexpr std::size_t max_name = 256;           // a name and a title
    static constexpr std::size_t max_origin_value = 1024;  // a string of `origin`
    static constexpr std::size_t max_columns = 4096;
    static constexpr std::size_t max_column_name = 256;
    static constexpr std::size_t max_column_type = 1024;
    static constexpr std::size_t max_rank = 8;
    static constexpr std::int64_t max_elements = std::int64_t(1) << 62;  // product of a shape
    static constexpr std::size_t max_warnings = 16;
    static constexpr std::size_t max_warning = 1024;
    static constexpr double max_rate_hz = 1e9;
};

struct CatalogColumn {
    std::string name;
    std::string type;  // a DuckDB type name, such as "BIGINT" or "VARCHAR"
};

// Where a source comes from. The keys are a closed set (catalog.md, Origins);
// a key that is not set is left out of the JSON form.
struct CatalogOrigin {
    std::optional<std::string> loader;      // a loader's id
    std::optional<std::string> file;        // relative to the workspace, or a dropped file
    std::optional<std::string> output;      // the loader output's name
    std::optional<std::string> tensor;      // the tensor's name inside the file
    std::optional<std::string> database;    // "workspace"
    std::optional<std::string> schema;      // a DuckDB schema
    std::optional<std::string> table;       // a DuckDB table or view
    std::optional<std::string> session;     // the attached live session
    std::optional<std::string> view;        // a live view's name
    std::optional<std::string> view_type;   // heatmap, points, histogram, series
    std::optional<std::string> service;     // "feed" or "metrics"
    std::optional<std::string> channel;     // a feed channel's id
    std::optional<std::string> units;       // a feed channel's units
    std::optional<std::string> experiment;  // a metrics experiment
    std::optional<std::int64_t> run;        // a metrics run's id
    std::optional<std::string> run_name;
    std::optional<std::string> tag;         // a metrics tag
};

struct CatalogEntry {
    std::string id;  // "<kind>:<name>"
    CatalogKind kind = CatalogKind::table;
    std::string name;
    std::string title;
    CatalogOrigin origin;
    std::optional<std::vector<CatalogColumn>> columns;  // tabular kinds only
    std::optional<std::uint64_t> rows;                  // tabular kinds; may stay unknown
    std::optional<CatalogDtype> dtype;                  // tensor-shaped kinds only
    std::optional<std::vector<std::int64_t>> shape;     // tensor-shaped kinds only
    std::optional<double> rate_hz;
    std::optional<std::uint64_t> bytes;
    bool wire = false;
    std::vector<std::string> warnings;
    std::optional<std::string> updated_at;  // UTC ISO 8601, "2026-11-15T12:00:00Z"
};

// "<kind>:<name>".
std::string make_catalog_id(CatalogKind kind, std::string_view name);

struct CatalogIdParse {
    CatalogKind kind = CatalogKind::table;
    std::string name;
    Error error;
};
CatalogIdParse parse_catalog_id(std::string_view id);

// Checks a name or a title: 1 to 256 bytes of valid UTF-8, no control
// characters, no space at either end. An error names `where`.
Error check_catalog_name(std::string_view text, std::string_view where);

// True for "YYYY-MM-DDTHH:MM:SS[.fraction]Z" with fields in range.
bool is_utc_timestamp(std::string_view text);

// Checks every rule of catalog.md on a struct; the first fault is returned.
Error validate_catalog_entry(const CatalogEntry& entry);

nlohmann::json catalog_column_to_json(const CatalogColumn& column);
nlohmann::json catalog_origin_to_json(const CatalogOrigin& origin);
nlohmann::json catalog_entry_to_json(const CatalogEntry& entry);

struct CatalogEntryParse {
    CatalogEntry entry;
    Error error;
};
CatalogEntryParse catalog_entry_from_json(const nlohmann::json& value);

}  // namespace caliper_builder
