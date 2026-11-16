#pragma once
// The tabular readers (csv, parquet, json, jsonl, and the arrow refusal): the
// relation that reads a list of files through DuckDB, and the view or table
// that one loader output makes from it. Internal to the loaders module.
//
// The files are canonical paths that already passed confinement; DuckDB's own
// allowed paths are the second line. An output's `sql` must be exactly one
// SELECT over the relation `input`; it is embedded as
//   CREATE VIEW "name" AS WITH input AS (<relation>) SELECT * FROM (<sql>)
// and the whole statement is parsed again and must be one CREATE.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/query.h>
#include <caliper_builder/loaders/readers.h>

#include "db.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace caliper_builder {
namespace detail {

struct TabularRead {
    ReaderKind reader = ReaderKind::csv;
    std::vector<std::string> files;
    nlohmann::json options = nlohmann::json::object();  // validated reader options
};

struct TableOutput {
    std::string name;
    bool copy = false;  // mode = "table"
    std::optional<std::string> sql;
};

struct TableMade {
    std::vector<ColumnInfo> columns;
    int64_t rows = 0;
};

// `SELECT * FROM read_x([...], options)` for the files.
Error tabular_relation(Db& db, const TabularRead& read, std::string* sql);

// Creates the output's view or table (dropping any relation of that name, which
// the caller owns) and reads back its columns and row count.
Error make_table_output(Db& db, const TabularRead& read, const TableOutput& output, TableMade* made);

}  // namespace detail
}  // namespace caliper_builder
