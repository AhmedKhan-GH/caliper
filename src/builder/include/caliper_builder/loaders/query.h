#pragma once
// The result of a read-only query of the workspace database, and the column
// description that a table source carries in the catalog.
//
// Rows are JSON arrays in column order. Numbers, strings, booleans and null
// map directly; a non-finite float is the string "NaN", "Infinity" or
// "-Infinity"; every other DuckDB type (dates, decimals that do not fit a
// double, lists, structs, blobs, 128-bit integers) is its DuckDB text form.
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace caliper_builder {

struct ColumnInfo {
    std::string name;
    std::string type;  // the DuckDB type name, such as "BIGINT" or "VARCHAR"

    nlohmann::json to_json() const { return {{"name", name}, {"type", type}}; }
};

struct QueryResult {
    Error error;
    std::vector<ColumnInfo> columns;
    nlohmann::json rows = nlohmann::json::array();
    bool truncated = false;  // more rows existed than the row limit allowed
    double elapsed_ms = 0;

    // {"columns": [...], "rows": [[...]], "truncated": bool, "elapsed_ms": x}
    nlohmann::json to_json() const {
        nlohmann::json cols = nlohmann::json::array();
        for (const ColumnInfo& c : columns) cols.push_back(c.to_json());
        return {{"columns", cols}, {"rows", rows}, {"truncated", truncated}, {"elapsed_ms", elapsed_ms}};
    }
};

// The bounds that query() clamps its arguments to.
constexpr uint64_t kMaxQueryRows = 100000;
constexpr uint32_t kMaxQueryTimeoutMs = 60000;

}  // namespace caliper_builder
