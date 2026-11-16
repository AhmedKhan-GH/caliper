#pragma once
// Writes a Parquet file through DuckDB, for the tests that import one. Kept
// apart from helpers.h because duckdb.hpp is a large header.
#include <doctest/doctest.h>
#include <duckdb.hpp>

#include <filesystem>
#include <string>

namespace session_test {

namespace fs = std::filesystem;

// Writes a Parquet file through DuckDB's COPY, from an in-memory database.
inline void write_parquet(const std::string& path, const std::string& select) {
    fs::create_directories(fs::path(path).parent_path());
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    std::string quoted = "'";
    for (char c : path) quoted += c == '\'' ? std::string("''") : std::string(1, c);
    quoted += "'";
    auto r = con.Query("COPY (" + select + ") TO " + quoted + " (FORMAT parquet)");
    REQUIRE_MESSAGE(!r->HasError(), r->GetError());
}

}  // namespace session_test
