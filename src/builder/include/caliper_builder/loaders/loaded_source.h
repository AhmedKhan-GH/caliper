#pragma once
// LoadedSource: one output of a loader run, a `table` or `tensor` source of the
// catalog. Its catalog fields are exactly those of the catalog entry of the
// sprint 3 contracts (section 3), and to_json() gives that entry's JSON form:
//
//   {"id": "tensor:attn", "kind": "tensor", "name": "attn", "title": "attn",
//    "origin": {"loader": "run42-weights", "file": "model.safetensors"},
//    "columns": null, "rows": null, "dtype": "f32", "shape": [12, 64],
//    "rate_hz": null, "bytes": 3072, "wire": true, "warnings": [],
//    "updated_at": "2026-11-15T12:00:00Z"}
//
// The storage fields after them say where the data lives; they are kept in
// caliper.sources and are not part of the catalog entry. The catalog module's
// CatalogProvider for loader outputs is a thin adapter over this struct.
#include <caliper_builder/loaders/query.h>
#include <caliper_builder/loaders/tensor.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace caliper_builder {

struct LoadedSource {
    // The catalog entry.
    std::string id;     // "<kind>:<name>"
    std::string kind;   // "table" or "tensor"
    std::string name;
    std::string title;
    std::string loader;  // origin.loader
    std::string file;    // origin.file: the recipe's input.path for a table, the tensor's file otherwise
    std::optional<std::vector<ColumnInfo>> columns;
    std::optional<int64_t> rows;
    std::optional<DType> dtype;
    std::optional<std::vector<int64_t>> shape;
    std::optional<double> rate_hz;
    std::optional<int64_t> bytes;
    bool wire = true;
    std::vector<std::string> warnings;
    std::string updated_at;

    // Where the data lives.
    std::string table_mode;  // "view" or "table" for a table source
    std::string tensor;      // the tensor's name in its file
    std::string data_file;   // the file that holds a tensor's bytes (canonical)
    int64_t data_offset = 0;
    bool derived = false;    // data_file is a derived tensor under <workspace>/tensors/

    // The catalog entry's JSON form.
    nlohmann::json to_json() const;
    // The storage fields: {"table_mode", "tensor", "data_file", "data_offset", "derived"}.
    nlohmann::json storage_json() const;
};

}  // namespace caliper_builder
