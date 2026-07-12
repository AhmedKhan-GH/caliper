#include "data_store.h"
#include "duckdb_arrow_stream.h"   // shared SQL→Arrow producer (also metrics.v1_1)

#include <duckdb.hpp>

#include <cctype>
#include <mutex>

namespace caliper_host {
namespace {

// Each thread sees the error of ITS last failing DataStore call (the data.v1
// last_error contract); a shared string would let one thread's failure
// overwrite another's mid-read.
thread_local std::string t_last_error;

void set_error(const std::string& msg) { t_last_error = msg; }

bool is_identifier(const std::string& s) {
    if (s.empty()) return false;
    if (!std::isalpha(static_cast<unsigned char>(s[0])) && s[0] != '_')
        return false;
    for (char c : s)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
            return false;
    return true;
}

std::string sql_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        out += c;
        if (c == '\'') out += '\'';  // '' escapes ' in SQL
    }
    out += "'";
    return out;
}

}  // namespace

struct DataStore::Impl {
    std::unique_ptr<duckdb::DuckDB>     db;
    std::unique_ptr<duckdb::Connection> con;
    std::mutex mu;
    bool opened = false;
};

DataStore::DataStore() : impl_(std::make_unique<Impl>()) {}
DataStore::~DataStore() = default;

bool DataStore::open(const std::string& path) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    try {
        const char* target = (path == ":memory:") ? nullptr : path.c_str();
        impl_->db  = std::make_unique<duckdb::DuckDB>(target);
        impl_->con = std::make_unique<duckdb::Connection>(*impl_->db);
        impl_->opened = true;
        return true;
    } catch (const std::exception& e) {
        set_error(e.what());
        return false;
    }
}

void DataStore::close() {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->con.reset();
    impl_->db.reset();
    impl_->opened = false;
}

bool DataStore::exec(const std::string& sql) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->opened) { set_error("data store is not open"); return false; }
    auto r = impl_->con->Query(sql);
    if (!r || r->HasError()) {
        set_error(r ? r->GetError() : "query failed");
        return false;
    }
    return true;
}

bool DataStore::query(const std::string& sql, ArrowArrayStream* out) {
    if (!out) { set_error("null output stream"); return false; }
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->opened) { set_error("data store is not open"); return false; }

    auto result = impl_->con->Query(sql);   // materializes fully
    if (!result || result->HasError()) {
        set_error(result ? result->GetError() : "query failed");
        return false;
    }

    arrow_stream::fill(
        duckdb::unique_ptr<duckdb::MaterializedQueryResult>(
            static_cast<duckdb::MaterializedQueryResult*>(result.release())),
        impl_->con->context->GetClientProperties(), out);
    return true;
}

bool DataStore::register_dataset(const std::string& name,
                                 const std::string& uri) {
    if (!is_identifier(name)) {
        set_error("dataset name must be an identifier: " + name);
        return false;
    }
    // Views over the readers keep registration cheap and the source fresh.
    std::string select;
    auto ends_with = [&](const char* suf) {
        std::string s(suf);
        return uri.size() >= s.size() &&
               uri.compare(uri.size() - s.size(), s.size(), s) == 0;
    };
    if (ends_with(".parquet"))
        select = "SELECT * FROM read_parquet(" + sql_quote(uri) + ")";
    else if (ends_with(".csv"))
        select = "SELECT * FROM read_csv_auto(" + sql_quote(uri) + ")";
    else if (is_identifier(uri))
        select = "SELECT * FROM " + uri;   // an existing table/view
    else {
        set_error("unsupported dataset uri: " + uri);
        return false;
    }
    return exec("CREATE OR REPLACE VIEW " + name + " AS " + select);
}

bool DataStore::open_dataset(const std::string& name, ArrowArrayStream* out) {
    if (!is_identifier(name)) {
        set_error("dataset name must be an identifier: " + name);
        return false;
    }
    return query("SELECT * FROM " + name, out);
}

std::string DataStore::last_error() const { return t_last_error; }

}  // namespace caliper_host
