#pragma once
// Db: the workspace database (workspace.duckdb, or an in-memory database in
// tests) opened with the safety settings of docs/plans/loaders.md ("What the
// database may read"). It hides DuckDB behind a small interface so that only
// db.cpp includes duckdb.hpp; every DuckDB call is wrapped so that no
// exception leaves it.
//
// Two kinds of SQL reach it. Trusted SQL, which this library builds with
// quoted identifiers and literals, goes through execute() and the helpers.
// Untrusted SQL (a loader's `sql`, a query) must pass check_select() first,
// and query() runs it read-only inside a transaction that is rolled back,
// with a row limit and a time limit. Not thread-safe: the caller serialises.
// Internal to the loaders module.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/query.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caliper_builder {
namespace detail {

struct DbOptions {
    std::vector<std::string> allowed_directories;  // canonical paths; a separator is appended
    std::vector<std::string> allowed_files;        // canonical paths
    std::string temp_directory;                    // empty: DuckDB's default
    int threads = 2;
    std::string memory_limit = "1GB";
};

class Db {
public:
    // Opens `path` (empty for an in-memory database) with the safety settings.
    static Error open(const std::string& path, const DbOptions& options, std::unique_ptr<Db>* out);
    ~Db();

    // Runs trusted SQL from this library.
    Error execute(const std::string& sql);
    // Runs trusted SQL and returns its rows as JSON arrays.
    Error rows(const std::string& sql, nlohmann::json* out);
    // The first column of the first row of trusted SQL, as an integer.
    Error scalar(const std::string& sql, int64_t* out);
    // The columns of a relation, read with `SELECT * FROM (<sql>) LIMIT 0`.
    Error describe(const std::string& relation_sql, std::vector<ColumnInfo>* out);

    // Exactly one SELECT statement, checked with DuckDB's parser.
    Error check_select(const std::string& sql);
    // Exactly one CREATE statement.
    Error check_create(const std::string& sql);
    // Read-only: one SELECT, rolled back, at most row_limit rows, interrupted after timeout_ms.
    QueryResult query(const std::string& sql, uint64_t row_limit, uint32_t timeout_ms);

    Error begin();
    Error commit();
    Error rollback();

    // Whether a table function exists, such as read_json in a DuckDB without its JSON extension.
    bool has_table_function(const std::string& name);
    // "VIEW", "BASE TABLE" or "" for a relation in the main schema.
    std::string relation_type(const std::string& name);
    // Drops a view or table of the main schema if it exists.
    Error drop_relation(const std::string& name);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// A SQL string literal and a quoted identifier. Both refuse NUL; the literal
// refuses control characters other than a tab.
std::string sql_literal(const std::string& s);
std::string sql_identifier(const std::string& s);
bool sql_text_ok(const std::string& s, bool allow_tab);

}  // namespace detail
}  // namespace caliper_builder
