// Db: DuckDB opened with the safety settings, behind a small interface. This is
// the one file of the module that includes duckdb.hpp.
#include "db.h"

#include <duckdb.hpp>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <thread>

namespace caliper_builder {
namespace detail {

using json = nlohmann::json;

struct Db::Impl {
    std::unique_ptr<duckdb::DuckDB> db;
    std::unique_ptr<duckdb::Connection> con;
};

namespace {

std::string message_of(const std::exception& e) {
    try {
        return duckdb::ErrorData(e).Message();
    } catch (...) {
        return e.what();
    }
}

Error error_of(duckdb::ExceptionType type, const std::string& message) {
    if (type == duckdb::ExceptionType::PERMISSION) {
        return Error("loader.path_not_allowed", "The database refused to read a path: " + message);
    }
    if (type == duckdb::ExceptionType::INTERRUPT) return Error("query.timeout", "The query was interrupted.");
    return Error("loader.sql", message);
}

Error error_of(duckdb::QueryResult& r) { return error_of(r.GetErrorType(), r.GetError()); }

Error error_of(const std::exception& e) {
    try {
        duckdb::ErrorData data(e);
        return error_of(data.Type(), data.Message());
    } catch (...) {
        return Error("loader.sql", e.what());
    }
}

json float_json(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    return v;
}

json value_json(const duckdb::Value& v) {
    if (v.IsNull()) return nullptr;
    using duckdb::LogicalTypeId;
    switch (v.type().id()) {
        case LogicalTypeId::BOOLEAN: return v.GetValue<bool>();
        case LogicalTypeId::TINYINT:
        case LogicalTypeId::SMALLINT:
        case LogicalTypeId::INTEGER:
        case LogicalTypeId::BIGINT: return v.GetValue<int64_t>();
        case LogicalTypeId::UTINYINT:
        case LogicalTypeId::USMALLINT:
        case LogicalTypeId::UINTEGER:
        case LogicalTypeId::UBIGINT: return v.GetValue<uint64_t>();
        case LogicalTypeId::FLOAT:
        case LogicalTypeId::DOUBLE: return float_json(v.GetValue<double>());
        case LogicalTypeId::DECIMAL: {
            const double d = v.GetValue<double>();
            return std::isfinite(d) ? json(d) : json(v.ToString());
        }
        case LogicalTypeId::VARCHAR: return v.GetValue<std::string>();
        case LogicalTypeId::HUGEINT:
        case LogicalTypeId::UHUGEINT: {
            // A 128-bit integer is a JSON number when it fits 64 bits (sum() of BIGINT is one), text otherwise.
            const std::string text = v.ToString();
            errno = 0;
            char* end = nullptr;
            if (!text.empty() && text[0] == '-') {
                const long long n = std::strtoll(text.c_str(), &end, 10);
                if (errno == 0 && end && *end == '\0') return static_cast<int64_t>(n);
            } else {
                const unsigned long long n = std::strtoull(text.c_str(), &end, 10);
                if (errno == 0 && end && *end == '\0') return static_cast<uint64_t>(n);
            }
            return text;
        }
        default: return v.ToString();
    }
}

}  // namespace

Error Db::open(const std::string& path, const DbOptions& options, std::unique_ptr<Db>* out) {
    std::unique_ptr<Db> db(new Db());
    db->impl_ = std::make_unique<Impl>();
    try {
        duckdb::DBConfig config;
        // temp_directory must be set before external access is disabled, which forbids changing it.
        if (!options.temp_directory.empty()) {
            config.SetOptionByName("temp_directory", duckdb::Value(options.temp_directory));
        }
        config.SetOptionByName("threads", duckdb::Value::BIGINT(options.threads));
        config.SetOptionByName("memory_limit", duckdb::Value(options.memory_limit));
        config.SetOptionByName("autoinstall_known_extensions", duckdb::Value::BOOLEAN(false));
        config.SetOptionByName("autoload_known_extensions", duckdb::Value::BOOLEAN(false));
        config.SetOptionByName("allow_community_extensions", duckdb::Value::BOOLEAN(false));
        config.SetOptionByName("allow_persistent_secrets", duckdb::Value::BOOLEAN(false));
        config.SetOptionByName("enable_external_access", duckdb::Value::BOOLEAN(false));
        const char sep = static_cast<char>(std::filesystem::path::preferred_separator);
        for (std::string dir : options.allowed_directories) {
            if (dir.empty()) continue;
            if (dir.back() != sep) dir += sep;
            config.options.allowed_directories.insert(dir);
        }
        for (const std::string& file : options.allowed_files) config.options.allowed_paths.insert(file);
        config.SetOptionByName("lock_configuration", duckdb::Value::BOOLEAN(true));
        db->impl_->db = std::make_unique<duckdb::DuckDB>(path.empty() ? nullptr : path.c_str(), &config);
        db->impl_->con = std::make_unique<duckdb::Connection>(*db->impl_->db);
    } catch (const std::exception& e) {
        return Error("loader.database", "The workspace database could not be opened: " + message_of(e), path);
    }
    *out = std::move(db);
    return {};
}

Db::~Db() {
    if (!impl_) return;
    try {
        impl_->con.reset();
        impl_->db.reset();
    } catch (...) {
    }
}

Error Db::execute(const std::string& sql) {
    try {
        auto r = impl_->con->Query(sql);
        if (r->HasError()) return error_of(*r);
    } catch (const std::exception& e) {
        return error_of(e);
    }
    return {};
}

Error Db::rows(const std::string& sql, json* out) {
    *out = json::array();
    try {
        auto r = impl_->con->Query(sql);
        if (r->HasError()) return error_of(*r);
        for (duckdb::idx_t row = 0; row < r->RowCount(); ++row) {
            json values = json::array();
            for (duckdb::idx_t col = 0; col < r->ColumnCount(); ++col) values.push_back(value_json(r->GetValue(col, row)));
            out->push_back(std::move(values));
        }
    } catch (const std::exception& e) {
        return error_of(e);
    }
    return {};
}

Error Db::scalar(const std::string& sql, int64_t* out) {
    json r;
    if (Error e = rows(sql, &r)) return e;
    if (r.empty() || r[0].empty() || !r[0][0].is_number_integer()) {
        return Error("loader.sql", "The query did not return an integer.");
    }
    *out = r[0][0].get<int64_t>();
    return {};
}

Error Db::describe(const std::string& relation_sql, std::vector<ColumnInfo>* out) {
    out->clear();
    try {
        auto r = impl_->con->Query("SELECT * FROM (" + relation_sql + ") LIMIT 0");
        if (r->HasError()) return error_of(*r);
        for (duckdb::idx_t i = 0; i < r->ColumnCount(); ++i) out->push_back({r->names[i], r->types[i].ToString()});
    } catch (const std::exception& e) {
        return error_of(e);
    }
    return {};
}

namespace {

Error check_one(duckdb::Connection& con, const std::string& sql, duckdb::StatementType type, const char* what) {
    try {
        auto statements = con.ExtractStatements(sql);
        if (statements.size() != 1 || statements[0]->type != type) {
            return Error("loader.sql_not_select", std::string("The SQL must be exactly one ") + what + " statement.");
        }
    } catch (const std::exception& e) {
        return Error("loader.sql", "The SQL does not parse: " + message_of(e));
    }
    return {};
}

}  // namespace

Error Db::check_select(const std::string& sql) {
    return check_one(*impl_->con, sql, duckdb::StatementType::SELECT_STATEMENT, "SELECT");
}

Error Db::check_create(const std::string& sql) {
    return check_one(*impl_->con, sql, duckdb::StatementType::CREATE_STATEMENT, "CREATE");
}

QueryResult Db::query(const std::string& sql, uint64_t row_limit, uint32_t timeout_ms) {
    QueryResult result;
    const auto start = std::chrono::steady_clock::now();
    if (row_limit < 1) row_limit = 1;
    if (row_limit > kMaxQueryRows) row_limit = kMaxQueryRows;
    if (timeout_ms < 1) timeout_ms = 1;
    if (timeout_ms > kMaxQueryTimeoutMs) timeout_ms = kMaxQueryTimeoutMs;
    duckdb::Connection& con = *impl_->con;
    std::vector<duckdb::unique_ptr<duckdb::SQLStatement>> statements;
    try {
        statements = con.ExtractStatements(sql);
    } catch (const std::exception& e) {
        result.error = Error("loader.sql", "The SQL does not parse: " + message_of(e));
        return result;
    }
    if (statements.size() != 1 || statements[0]->type != duckdb::StatementType::SELECT_STATEMENT) {
        result.error = Error("loader.sql_not_select", "The SQL must be exactly one SELECT statement.");
        return result;
    }
    try {
        con.BeginTransaction();
    } catch (const std::exception& e) {
        result.error = error_of(e);
        return result;
    }
    // The watchdog interrupts the query when the time limit passes; it never fires after `done`.
    std::mutex m;
    std::condition_variable cv;
    bool done = false, fired = false;
    std::thread watchdog([&] {
        std::unique_lock<std::mutex> lock(m);
        if (!cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return done; })) {
            fired = true;
            con.Interrupt();
        }
    });
    try {
        auto r = con.SendQuery(std::move(statements[0]));
        if (r->HasError()) {
            result.error = error_of(*r);
        } else {
            for (duckdb::idx_t i = 0; i < r->ColumnCount(); ++i) {
                result.columns.push_back({r->names[i], r->types[i].ToString()});
            }
            uint64_t count = 0;
            bool full = false;
            while (!full) {
                auto chunk = r->Fetch();
                if (!chunk || chunk->size() == 0) {
                    if (r->HasError()) result.error = error_of(*r);
                    break;
                }
                for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
                    if (count == row_limit) {
                        result.truncated = true;
                        full = true;
                        break;
                    }
                    json values = json::array();
                    for (duckdb::idx_t col = 0; col < chunk->ColumnCount(); ++col) {
                        values.push_back(value_json(chunk->GetValue(col, row)));
                    }
                    result.rows.push_back(std::move(values));
                    ++count;
                }
            }
        }
    } catch (const std::exception& e) {
        result.error = error_of(e);
    }
    {
        std::lock_guard<std::mutex> lock(m);
        done = true;
    }
    cv.notify_all();
    watchdog.join();
    try {
        con.Rollback();
    } catch (...) {
    }
    if (result.error && fired) {
        result.error = Error("query.timeout", "The query ran past its time limit of " + std::to_string(timeout_ms) + " ms.");
    }
    if (result.error) {
        result.rows = json::array();
        result.columns.clear();
        result.truncated = false;
    }
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

Error Db::begin() { return execute("BEGIN TRANSACTION"); }
Error Db::commit() { return execute("COMMIT"); }
Error Db::rollback() { return execute("ROLLBACK"); }

bool Db::has_table_function(const std::string& name) {
    int64_t n = 0;
    if (scalar("SELECT count(*) FROM duckdb_functions() WHERE function_type = 'table' AND function_name = " +
                   sql_literal(name),
               &n)) {
        return false;
    }
    return n > 0;
}

std::string Db::relation_type(const std::string& name) {
    json r;
    if (rows("SELECT table_type FROM information_schema.tables WHERE table_schema = 'main' AND table_name = " +
                 sql_literal(name),
             &r) ||
        r.empty() || !r[0][0].is_string()) {
        return "";
    }
    return r[0][0].get<std::string>();
}

Error Db::drop_relation(const std::string& name) {
    const std::string type = relation_type(name);
    if (type == "VIEW") return execute("DROP VIEW " + sql_identifier(name));
    if (!type.empty()) return execute("DROP TABLE " + sql_identifier(name));
    return {};
}

bool Db::needs_quotes(const std::string& name) {
    int64_t n = 0;
    if (scalar("SELECT count(*) FROM duckdb_keywords() WHERE keyword_name = " + sql_literal(name) +
                   " AND keyword_category <> 'unreserved'",
               &n)) {
        return false;
    }
    return n > 0;
}

bool sql_text_ok(const std::string& s, bool allow_tab) {
    for (unsigned char c : s) {
        if (c == 0) return false;
        if (c < 0x20 && !(allow_tab && c == '\t')) return false;
    }
    return true;
}

std::string sql_literal(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "''";
        else out += c;
    }
    return out + "'";
}

std::string sql_identifier(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    return out + "\"";
}

}  // namespace detail
}  // namespace caliper_builder
