// Tests of the tabular readers through DuckDB and of the database's safety
// settings: what SQL may read, the single-SELECT check, and the query helper's
// row and time limits. Every fixture is written by the test, Parquet through
// DuckDB's own COPY.
#include <doctest/doctest.h>

#include "../db.h"
#include "../tabular.h"
#include "test_util.h"

#include <filesystem>

using namespace caliper_builder;
using namespace caliper_builder::detail;
using namespace loaders_test;
using nlohmann::json;

namespace {

struct Fixture {
    std::string root;  // holds ws/, ws2/ and outside/
    std::string ws;
    std::unique_ptr<Db> db;

    // With `grant`, the file outside/granted.csv is an allowed path.
    explicit Fixture(const std::string& name, bool grant = false) {
        root = fresh_dir(name);
        ws = root + "/ws";
        std::filesystem::create_directories(ws);
        std::filesystem::create_directories(root + "/ws2");
        std::filesystem::create_directories(root + "/outside");
        write_text(ws + "/run.csv", "step,loss\n0,1.5\n1,1.25\n2,1.0\n");
        write_text(ws + "/more.csv", "step,loss,lr\n3,0.75,0.1\n");
        write_text(ws + "/run.tsv", "a\tb\n1\tx\n2\ty\n");
        write_text(ws + "/semi.csv", "# comment\nk;v\n1;NA\n2;5\n");
        write_text(ws + "/rows.json", R"([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])");
        write_text(ws + "/rows.jsonl", "{\"id\": 1, \"tag\": \"x\"}\n{\"id\": 2, \"tag\": \"y\"}\n{\"id\": 3, \"tag\": null}\n");
        write_text(root + "/ws2/secret.csv", "s\nhidden\n");
        write_text(root + "/outside/secret.csv", "s\nhidden\n");
        write_text(root + "/outside/granted.csv", "g\n1\n");
        DbOptions options;
        options.allowed_directories = {ws};
        if (grant) options.allowed_files = {root + "/outside/granted.csv"};
        options.temp_directory = ws + "/workspace.duckdb.tmp";
        Error e = Db::open(ws + "/workspace.duckdb", options, &db);
        REQUIRE_MESSAGE(e.ok(), e.message);
        REQUIRE(db->execute("COPY (SELECT range AS step, range * 0.5 AS loss FROM range(5)) TO " +
                            sql_literal(ws + "/m.parquet") + " (FORMAT parquet)")
                    .ok());
        REQUIRE(db->execute("COPY (SELECT range AS step, 'b' AS extra FROM range(5, 7)) TO " +
                            sql_literal(ws + "/m2.parquet") + " (FORMAT parquet)")
                    .ok());
    }

    TableMade make(ReaderKind reader, std::vector<std::string> files, const std::string& name,
                   std::optional<std::string> sql = {}, json options = json::object(), bool copy = false,
                   Error* error = nullptr) {
        TabularRead read{reader, std::move(files), std::move(options)};
        TableMade made;
        Error e = make_table_output(*db, read, TableOutput{name, copy, std::move(sql)}, &made);
        if (error) *error = e;
        else REQUIRE_MESSAGE(e.ok(), e.code << ": " << e.message);
        return made;
    }

    json all(const std::string& sql) {
        QueryResult r = db->query(sql, 1000, 5000);
        REQUIRE_MESSAGE(r.error.ok(), r.error.code << ": " << r.error.message);
        return r.rows;
    }
};

}  // namespace

TEST_CASE("csv views, tables, SQL and options") {
    Fixture f("tab_csv");
    TableMade m = f.make(ReaderKind::csv, {f.ws + "/run.csv"}, "run");
    REQUIRE(m.columns.size() == 2);
    CHECK(m.columns[0].name == "step");
    CHECK(m.columns[0].type == "BIGINT");
    CHECK(m.columns[1].type == "DOUBLE");
    CHECK(m.rows == 3);
    CHECK(f.db->relation_type("run") == "VIEW");
    CHECK(f.all("SELECT sum(loss) FROM run") == json::parse("[[3.75]]"));

    m = f.make(ReaderKind::csv, {f.ws + "/run.csv"}, "late", std::string("SELECT step, loss FROM input WHERE step > 0"));
    CHECK(m.rows == 2);
    m = f.make(ReaderKind::csv, {f.ws + "/run.csv"}, "late2",
               std::string("  WITH t AS (SELECT * FROM input) SELECT max(step) AS s FROM t;  -- the last step\n"));
    CHECK(m.rows == 1);
    CHECK(m.columns[0].name == "s");
    m = f.make(ReaderKind::csv, {f.ws + "/run.csv"}, "late3",
               std::string("SELECT 'a;b' AS \"x;y\", $$c;d$$ AS z /* ; */ FROM input -- ;\n ;;"));
    CHECK(m.rows == 3);
    CHECK(f.all("SELECT \"x;y\", z FROM late3 LIMIT 1") == json::parse(R"([["a;b", "c;d"]])"));

    m = f.make(ReaderKind::csv, {f.ws + "/run.tsv"}, "tsv");
    CHECK(m.columns.size() == 2);
    CHECK(m.columns[1].type == "VARCHAR");

    m = f.make(ReaderKind::csv, {f.ws + "/semi.csv"}, "semis", {},
               {{"delimiter", ";"}, {"skip", 1}, {"null", "NA"}, {"header", true}});
    CHECK(m.columns[0].name == "k");
    CHECK(f.all("SELECT count(*) FROM semis WHERE v IS NULL") == json::parse("[[1]]"));

    m = f.make(ReaderKind::csv, {f.ws + "/run.csv", f.ws + "/more.csv"}, "both", {}, {{"union_by_name", true}});
    CHECK(m.rows == 4);
    CHECK(m.columns.size() == 3);

    // A table copies the rows; a view reads the file on each query.
    m = f.make(ReaderKind::csv, {f.ws + "/more.csv"}, "copied", {}, json::object(), true);
    CHECK(f.db->relation_type("copied") == "BASE TABLE");
    m = f.make(ReaderKind::csv, {f.ws + "/more.csv"}, "viewed");
    std::filesystem::remove(f.ws + "/more.csv");
    CHECK(f.all("SELECT count(*) FROM copied") == json::parse("[[1]]"));
    CHECK_FALSE(f.db->query("SELECT count(*) FROM viewed", 10, 5000).error.ok());

    // Making an output again replaces it, and a view can become a table.
    m = f.make(ReaderKind::csv, {f.ws + "/run.csv"}, "run", {}, json::object(), true);
    CHECK(f.db->relation_type("run") == "BASE TABLE");
}

TEST_CASE("parquet, json and jsonl") {
    Fixture f("tab_formats");
    TableMade m = f.make(ReaderKind::parquet, {f.ws + "/m.parquet"}, "metrics");
    CHECK(m.rows == 5);
    CHECK(m.columns[0].name == "step");
    CHECK(f.all("SELECT loss FROM metrics WHERE step = 4") == json::parse("[[2.0]]"));
    m = f.make(ReaderKind::parquet, {f.ws + "/m.parquet", f.ws + "/m2.parquet"}, "both", {}, {{"union_by_name", true}});
    CHECK(m.rows == 7);
    CHECK(m.columns.size() == 3);

    m = f.make(ReaderKind::json, {f.ws + "/rows.json"}, "rows_json");
    CHECK(m.rows == 2);
    CHECK(m.columns[1].name == "name");
    m = f.make(ReaderKind::json, {f.ws + "/rows.json"}, "rows_array", {}, {{"format", "array"}});
    CHECK(m.rows == 2);
    m = f.make(ReaderKind::jsonl, {f.ws + "/rows.jsonl"}, "rows_jsonl", std::string("SELECT * FROM input WHERE tag IS NOT NULL"));
    CHECK(m.rows == 2);

    Error e;
    f.make(ReaderKind::arrow, {f.ws + "/m.parquet"}, "arrow", {}, json::object(), false, &e);
    CHECK(e.code == "loader.arrow_not_available");
    f.make(ReaderKind::npy, {f.ws + "/m.parquet"}, "npy", {}, json::object(), false, &e);
    CHECK(e.code == "loader.bad_value");
    f.make(ReaderKind::csv, {}, "none", {}, json::object(), false, &e);
    CHECK(e.code == "loader.no_match");
    f.make(ReaderKind::csv, {f.ws + "/run.csv"}, "opt", {}, {{"delim", ","}}, false, &e);
    CHECK(e.code == "loader.unknown_key");
    CHECK(e.where == "reader.delim");
    f.make(ReaderKind::csv, {f.ws + "/run.csv"}, "opt", {}, {{"delimiter", ",,"}}, false, &e);
    CHECK(e.code == "loader.bad_value");
}

TEST_CASE("an output's SQL is exactly one SELECT") {
    Fixture f("tab_sql");
    Error e;
    for (const char* sql : {"DELETE FROM input", "SELECT 1; SELECT 2", "COPY input TO 'x.csv'", "ATTACH ':memory:' AS m",
                            "CREATE TABLE t AS SELECT 1", "SET threads = 1", "EXPLAIN SELECT 1",
                            "INSTALL httpfs", "LOAD httpfs", "", "   "}) {
        CAPTURE(sql);
        f.make(ReaderKind::csv, {f.ws + "/run.csv"}, "x", std::string(sql), json::object(), false, &e);
        CHECK(e.code == "loader.sql_not_select");
        CHECK(e.where == "sql");
    }
    for (const char* sql : {"SELEC 1", "SELECT * FROM input) t, (SELECT 1", "SELECT 1 /* open"}) {
        CAPTURE(sql);
        f.make(ReaderKind::csv, {f.ws + "/run.csv"}, "x", std::string(sql), json::object(), false, &e);
        CHECK(e.code == "loader.sql");
    }
    // A query that names a column the input lacks fails in DuckDB's binder.
    f.make(ReaderKind::csv, {f.ws + "/run.csv"}, "x", std::string("SELECT nope FROM input"), json::object(), false, &e);
    CHECK(e.code == "loader.sql");
    // A name with a quote is quoted, not injected.
    TableMade m = f.make(ReaderKind::csv, {f.ws + "/run.csv"}, "a\"b");
    CHECK(m.rows == 3);
}

TEST_CASE("SQL reads only the workspace and the granted files") {
    Fixture f("tab_paths");
    const std::string outside = f.root + "/outside/secret.csv";
    Error e;
    f.make(ReaderKind::csv, {outside}, "x", {}, json::object(), false, &e);
    CHECK(e.code == "loader.path_not_allowed");
    for (const std::string& path : {outside, f.root + "/ws2/secret.csv", f.ws + "/../outside/secret.csv",
                                    std::string("file://") + outside}) {
        CAPTURE(path);
        for (const std::string& fn : {"read_csv", "read_text", "read_blob", "glob", "sniff_csv", "parquet_metadata"}) {
            QueryResult r = f.db->query("SELECT * FROM " + fn + "(" + sql_literal(path) + ")", 10, 5000);
            CHECK_MESSAGE(r.error.code == "loader.path_not_allowed", fn << ": " << r.error.message);
        }
    }
    // Through the output's SQL as well.
    f.make(ReaderKind::csv, {f.ws + "/run.csv"}, "x",
           std::string("SELECT * FROM read_csv(" + sql_literal(outside) + ")"), json::object(), false, &e);
    CHECK(e.code == "loader.path_not_allowed");
    QueryResult r;
#ifndef _WIN32
    // A symbolic link inside the workspace that points outside it.
    std::filesystem::create_symlink(outside, f.ws + "/link.csv");
    r = f.db->query("SELECT * FROM read_csv(" + sql_literal(f.ws + "/link.csv") + ")", 10, 5000);
    CHECK(r.error.code == "loader.path_not_allowed");
#endif
    // A granted file outside the workspace is readable, and only that file.
    Fixture g("tab_paths_granted", true);
    r = g.db->query("SELECT * FROM read_csv(" + sql_literal(g.root + "/outside/granted.csv") + ")", 10, 5000);
    CHECK_MESSAGE(r.error.ok(), r.error.message);
    r = g.db->query("SELECT * FROM read_csv(" + sql_literal(g.root + "/outside/secret.csv") + ")", 10, 5000);
    CHECK(r.error.code == "loader.path_not_allowed");
}

TEST_CASE("the configuration is locked and extensions cannot be installed") {
    Fixture f("tab_lock");
    for (const char* sql : {"SET enable_external_access = true", "SET allowed_directories = ['/']",
                            "SET allowed_paths = ['/']", "SET threads = 8", "SET autoinstall_known_extensions = true",
                            "SET lock_configuration = false", "INSTALL httpfs", "LOAD httpfs"}) {
        CAPTURE(sql);
        CHECK_FALSE(f.db->execute(sql).ok());
    }
    CHECK(f.all("SELECT current_setting('threads')") == json::parse("[[2]]"));
    CHECK(f.all("SELECT current_setting('enable_external_access')") == json::parse("[[false]]"));
    CHECK(f.all("SELECT current_setting('lock_configuration')") == json::parse("[[true]]"));
    // The query() table function takes only a SELECT.
    QueryResult r = f.db->query("SELECT * FROM query('ATTACH '':memory:'' AS q')", 10, 5000);
    CHECK_FALSE(r.error.ok());
    CHECK(f.all("SELECT * FROM query('SELECT 42')") == json::parse("[[42]]"));
}

TEST_CASE("the query helper is read-only and bounded") {
    Fixture f("tab_query");
    REQUIRE(f.db->execute("CREATE TABLE t AS SELECT range AS x FROM range(10)").ok());
    for (const char* sql : {"CREATE TABLE u(x INT)", "INSERT INTO t VALUES (1)", "DELETE FROM t", "DROP TABLE t",
                            "ATTACH ':memory:' AS m", "COPY t TO 'x.csv'", "SET threads = 1", "CALL pragma_version()",
                            "SELECT 1; DROP TABLE t", "BEGIN", "EXPORT DATABASE 'x'", "CHECKPOINT", "VACUUM",
                            "PRAGMA enable_profiling", "EXPLAIN SELECT 1"}) {
        CAPTURE(sql);
        CHECK(f.db->query(sql, 10, 5000).error.code == "loader.sql_not_select");
    }
    CHECK(f.all("SELECT count(*) FROM t") == json::parse("[[10]]"));
    // DuckDB parses the table-valued pragmas as SELECTs over pragma functions, which only read.
    CHECK(f.db->query("PRAGMA database_list", 10, 5000).error.ok());

    QueryResult r = f.db->query("SELECT x FROM t ORDER BY x", 3, 5000);
    REQUIRE(r.error.ok());
    CHECK(r.rows == json::parse("[[0], [1], [2]]"));
    CHECK(r.truncated);
    CHECK(r.columns[0].type == "BIGINT");
    r = f.db->query("SELECT x FROM t", 10, 5000);
    CHECK_FALSE(r.truncated);
    r = f.db->query("FROM range(250000)", 1000000, 5000);  // the limit is clamped
    CHECK(r.rows.size() == kMaxQueryRows);
    CHECK(r.truncated);

    r = f.db->query("SELECT count(*) FROM range(10000000000) a", 10, 100);
    CHECK(r.error.code == "query.timeout");
    CHECK(r.elapsed_ms < 5000);
    CHECK(f.all("SELECT 1") == json::parse("[[1]]"));  // the connection still works

    r = f.db->query(
        "SELECT 1::TINYINT, 18446744073709551615::UBIGINT, 1.5::FLOAT, 'inf'::DOUBLE, NULL, true, 'text', "
        "DATE '2026-11-15', [1, 2], 2.25::DECIMAL(5, 2), 170141183460469231731687303715884105727::HUGEINT",
        10, 5000);
    REQUIRE_MESSAGE(r.error.ok(), r.error.message);
    const json row = r.rows[0];
    CHECK(row[0] == 1);
    CHECK(row[1] == 18446744073709551615ULL);
    CHECK(row[2] == 1.5);
    CHECK(row[3] == "Infinity");
    CHECK(row[4].is_null());
    CHECK(row[5] == true);
    CHECK(row[6] == "text");
    CHECK(row[7] == "2026-11-15");
    CHECK(row[8] == "[1, 2]");
    CHECK(row[9] == 2.25);
    CHECK(row[10] == "170141183460469231731687303715884105727");
    CHECK(r.to_json()["columns"].size() == 11);
}
