#include "metrics_store.h"

#include <duckdb.hpp>

#include <mutex>
#include <unordered_set>

namespace caliper_host {

// All state (connection + prepared statements + mutex) lives here so no DuckDB
// type escapes into the header. One connection, one mutex — every public method
// takes the lock, so calls from job threads serialize (metrics.v1 contract).
struct MetricsStore::Impl {
    std::unique_ptr<duckdb::DuckDB>     db;
    std::unique_ptr<duckdb::Connection> con;

    // Prepared once at open; re-bound under the mutex on every write.
    std::unique_ptr<duckdb::PreparedStatement> ins_run;
    std::unique_ptr<duckdb::PreparedStatement> ins_scalar;
    std::unique_ptr<duckdb::PreparedStatement> ins_histogram;
    std::unique_ptr<duckdb::PreparedStatement> ins_image;
    std::unique_ptr<duckdb::PreparedStatement> upd_end_run;
    std::unique_ptr<duckdb::PreparedStatement> upd_hparams;

    std::mutex mu;
    uint64_t   next_id = 1;  // monotonic id source; seeded MAX(id)+1 on open

    // Known run ids, kept in sync under mu. Writers probe this instead of a SQL
    // COUNT so a hot training loop (a scalar per step) stays inert-for-unknown
    // without a round-trip per call. Seeded from disk in open().
    std::unordered_set<uint64_t> known_runs;

    // Caller must hold mu.
    bool run_exists(uint64_t run) const {
        return known_runs.count(run) != 0;
    }
};

MetricsStore::MetricsStore() : impl_(std::make_unique<Impl>()) {}
MetricsStore::~MetricsStore() = default;

bool MetricsStore::open(const std::string& path) {
    std::lock_guard<std::mutex> lk(impl_->mu);

    const char* target = (path == ":memory:") ? nullptr : path.c_str();
    // DuckDB throws (e.g. IOException when another Caliper instance holds the
    // file lock) — a failed open must degrade to a no-op service, never
    // terminate the host. Same guard as ArtifactStore/DataStore.
    try {
        impl_->db  = std::make_unique<duckdb::DuckDB>(target);
        impl_->con = std::make_unique<duckdb::Connection>(*impl_->db);
    } catch (const std::exception&) {
        impl_->db.reset();
        impl_->con.reset();
        return false;
    }

    auto run_ddl = [&](const char* sql) {
        auto r = impl_->con->Query(sql);
        return r && !r->HasError();
    };

    bool ok = true;
    ok &= run_ddl("CREATE TABLE IF NOT EXISTS runs ("
                  "id BIGINT PRIMARY KEY, experiment VARCHAR, name VARCHAR, "
                  "done BOOLEAN, hparams VARCHAR)");
    ok &= run_ddl("CREATE TABLE IF NOT EXISTS scalars ("
                  "run BIGINT, tag VARCHAR, step BIGINT, value DOUBLE)");
    ok &= run_ddl("CREATE TABLE IF NOT EXISTS histograms ("
                  "run BIGINT, tag VARCHAR, step BIGINT, count BIGINT, data BLOB)");
    ok &= run_ddl("CREATE TABLE IF NOT EXISTS images ("
                  "run BIGINT, tag VARCHAR, step BIGINT, w INT, h INT, c INT, data BLOB)");
    if (!ok) return false;

    // Seed the in-memory run set and the id counter from any persisted rows so
    // a reopened on-disk store recognizes existing runs and never re-issues an
    // id (next_id = MAX(id)+1).
    {
        auto r = impl_->con->Query("SELECT id FROM runs");
        if (r && !r->HasError()) {
            for (auto chunk = r->Fetch(); chunk; chunk = r->Fetch()) {
                for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
                    auto id = static_cast<uint64_t>(
                        chunk->GetValue(0, row).GetValue<int64_t>());
                    impl_->known_runs.insert(id);
                    if (id + 1 > impl_->next_id) impl_->next_id = id + 1;
                }
            }
        }
    }

    auto& con = *impl_->con;
    impl_->ins_run       = con.Prepare("INSERT INTO runs VALUES (?, ?, ?, ?, ?)");
    impl_->ins_scalar    = con.Prepare("INSERT INTO scalars VALUES (?, ?, ?, ?)");
    impl_->ins_histogram = con.Prepare("INSERT INTO histograms VALUES (?, ?, ?, ?, ?)");
    impl_->ins_image     = con.Prepare("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?)");
    impl_->upd_end_run   = con.Prepare("UPDATE runs SET done = TRUE WHERE id = ?");
    impl_->upd_hparams   = con.Prepare("UPDATE runs SET hparams = ? WHERE id = ?");

    return impl_->ins_run && impl_->ins_scalar && impl_->ins_histogram &&
           impl_->ins_image && impl_->upd_end_run && impl_->upd_hparams;
}

uint64_t MetricsStore::begin_run(const std::string& experiment,
                                 const std::string& name) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->con) return 0;

    uint64_t id = impl_->next_id;
    duckdb::vector<duckdb::Value> args;
    args.push_back(duckdb::Value::BIGINT(static_cast<int64_t>(id)));
    args.push_back(duckdb::Value(experiment));   // VARCHAR, verbatim
    args.push_back(duckdb::Value(name));         // VARCHAR, verbatim
    args.push_back(duckdb::Value::BOOLEAN(false));
    args.push_back(duckdb::Value(duckdb::LogicalType::VARCHAR));  // NULL hparams
    auto res = impl_->ins_run->Execute(args, /*allow_stream_result=*/true);
    if (!res || res->HasError()) return 0;
    impl_->known_runs.insert(id);
    ++impl_->next_id;
    return id;
}

void MetricsStore::end_run(uint64_t run) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->con) return;
    impl_->upd_end_run->Execute(static_cast<int64_t>(run));  // no-op if id absent
}

void MetricsStore::scalar(uint64_t run, const std::string& tag,
                          int64_t step, double value) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->con || !impl_->run_exists(run)) return;
    duckdb::vector<duckdb::Value> args;
    args.push_back(duckdb::Value::BIGINT(static_cast<int64_t>(run)));
    args.push_back(duckdb::Value(tag));          // VARCHAR, verbatim
    args.push_back(duckdb::Value::BIGINT(step));
    args.push_back(duckdb::Value::DOUBLE(value));
    impl_->ins_scalar->Execute(args, /*allow_stream_result=*/true);
}

void MetricsStore::histogram(uint64_t run, const std::string& tag, int64_t step,
                             const float* values, int64_t count) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->con || !impl_->run_exists(run) || count < 0) return;
    if (count > 0 && !values) return;

    duckdb::vector<duckdb::Value> args;
    args.push_back(duckdb::Value::BIGINT(static_cast<int64_t>(run)));
    args.push_back(duckdb::Value(tag));
    args.push_back(duckdb::Value::BIGINT(step));
    args.push_back(duckdb::Value::BIGINT(count));
    args.push_back(duckdb::Value::BLOB(
        reinterpret_cast<duckdb::const_data_ptr_t>(values),
        static_cast<duckdb::idx_t>(count) * sizeof(float)));
    impl_->ins_histogram->Execute(args, /*allow_stream_result=*/true);
}

void MetricsStore::image(uint64_t run, const std::string& tag, int64_t step,
                         const void* bytes, int32_t w, int32_t h, int32_t c) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->con || !impl_->run_exists(run)) return;
    if (w < 0 || h < 0 || c < 0) return;

    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h) *
                     static_cast<size_t>(c);
    if (n > 0 && !bytes) return;

    duckdb::vector<duckdb::Value> args;
    args.push_back(duckdb::Value::BIGINT(static_cast<int64_t>(run)));
    args.push_back(duckdb::Value(tag));
    args.push_back(duckdb::Value::BIGINT(step));
    args.push_back(duckdb::Value::INTEGER(w));
    args.push_back(duckdb::Value::INTEGER(h));
    args.push_back(duckdb::Value::INTEGER(c));
    args.push_back(duckdb::Value::BLOB(
        reinterpret_cast<duckdb::const_data_ptr_t>(bytes), static_cast<duckdb::idx_t>(n)));
    impl_->ins_image->Execute(args, /*allow_stream_result=*/true);
}

void MetricsStore::hparams_json(uint64_t run, const std::string& json_utf8) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->con) return;
    duckdb::vector<duckdb::Value> args;
    args.push_back(duckdb::Value(json_utf8));            // VARCHAR, verbatim
    args.push_back(duckdb::Value::BIGINT(static_cast<int64_t>(run)));
    impl_->upd_hparams->Execute(args, /*allow_stream_result=*/true);  // no-op if absent
}

std::vector<RunInfo> MetricsStore::runs() {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::vector<RunInfo> out;
    if (!impl_->con) return out;

    auto res = impl_->con->Query(
        "SELECT id, experiment, name, done, hparams FROM runs ORDER BY id");
    if (!res || res->HasError()) return out;

    for (auto chunk = res->Fetch(); chunk; chunk = res->Fetch()) {
        for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
            RunInfo r;
            r.id         = static_cast<uint64_t>(chunk->GetValue(0, row).GetValue<int64_t>());
            r.experiment = duckdb::StringValue::Get(chunk->GetValue(1, row));
            r.name       = duckdb::StringValue::Get(chunk->GetValue(2, row));
            r.done       = chunk->GetValue(3, row).GetValue<bool>();
            auto hp      = chunk->GetValue(4, row);
            r.hparams    = hp.IsNull() ? std::string() : duckdb::StringValue::Get(hp);
            out.push_back(std::move(r));
        }
    }
    return out;
}

std::vector<std::string> MetricsStore::scalar_tags(uint64_t run) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::vector<std::string> out;
    if (!impl_->con) return out;

    auto stmt = impl_->con->Prepare(
        "SELECT DISTINCT tag FROM scalars WHERE run = ?");
    if (!stmt || stmt->HasError()) return out;
    auto res = stmt->Execute(static_cast<int64_t>(run));
    if (!res || res->HasError()) return out;

    for (auto chunk = res->Fetch(); chunk; chunk = res->Fetch()) {
        for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
            out.push_back(duckdb::StringValue::Get(chunk->GetValue(0, row)));
        }
    }
    return out;
}

std::vector<std::pair<int64_t, double>> MetricsStore::scalars(
    uint64_t run, const std::string& tag) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::vector<std::pair<int64_t, double>> out;
    if (!impl_->con) return out;

    // ORDER BY step is the §16 contract: shuffled-order writes read back
    // strictly step-ascending.
    auto stmt = impl_->con->Prepare(
        "SELECT step, value FROM scalars WHERE run = ? AND tag = ? ORDER BY step");
    if (!stmt || stmt->HasError()) return out;
    duckdb::vector<duckdb::Value> args;
    args.push_back(duckdb::Value::BIGINT(static_cast<int64_t>(run)));
    args.push_back(duckdb::Value(tag));
    auto res = stmt->Execute(args, /*allow_stream_result=*/true);
    if (!res || res->HasError()) return out;

    for (auto chunk = res->Fetch(); chunk; chunk = res->Fetch()) {
        for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
            out.emplace_back(chunk->GetValue(0, row).GetValue<int64_t>(),
                             chunk->GetValue(1, row).GetValue<double>());
        }
    }
    return out;
}

std::vector<HistogramInfo> MetricsStore::histograms(uint64_t run,
                                                    const std::string& tag) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::vector<HistogramInfo> out;
    if (!impl_->con) return out;

    auto stmt = impl_->con->Prepare(
        "SELECT step, count, octet_length(data) FROM histograms "
        "WHERE run = ? AND tag = ? ORDER BY step");
    if (!stmt || stmt->HasError()) return out;
    duckdb::vector<duckdb::Value> args;
    args.push_back(duckdb::Value::BIGINT(static_cast<int64_t>(run)));
    args.push_back(duckdb::Value(tag));
    auto res = stmt->Execute(args, /*allow_stream_result=*/true);
    if (!res || res->HasError()) return out;

    for (auto chunk = res->Fetch(); chunk; chunk = res->Fetch()) {
        for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
            HistogramInfo h;
            h.step        = chunk->GetValue(0, row).GetValue<int64_t>();
            h.count       = chunk->GetValue(1, row).GetValue<int64_t>();
            h.byte_length = chunk->GetValue(2, row).GetValue<int64_t>();
            out.push_back(h);
        }
    }
    return out;
}

}  // namespace caliper_host
