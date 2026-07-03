#pragma once
// caliper_host::DataStore — SQL over the host's embedded DuckDB, results out
// as Arrow C streams (backs caliper.data.v1, PLATFORM.md §7.7).
//
// Stream model: query()/open_dataset() fully materialize the result, then
// hand back a self-contained ArrowArrayStream (the producer state owns the
// materialized rows). Draining a stream therefore never touches the
// connection — a job thread can drain while another thread queries.
//
// Threading: open/query/register/open_dataset serialize on one internal
// mutex (the artifacts/metrics model). last_error() is thread-local: each
// thread sees the error of ITS last failing call.
//
// No DuckDB type appears in this header (pimpl); ArrowArrayStream comes from
// the vendored spec header, which is the ABI's own tabular type.
#include <caliper/arrow_c.h>

#include <memory>
#include <string>

namespace caliper_host {

class DataStore {
public:
    DataStore();
    ~DataStore();

    DataStore(const DataStore&) = delete;
    DataStore& operator=(const DataStore&) = delete;

    // Open a database at `path` (or ":memory:"). False on failure; all other
    // methods are inert (false + last_error) on an unopened store.
    bool open(const std::string& path);

    // Deterministic teardown before main returns (see MetricsStore::close()
    // — static-destructor DuckDB teardown aborts). Idempotent. Streams
    // already handed out stay valid: they own materialized results, not the
    // connection.
    void close();

    // Run a statement for its side effects (DDL/INSERT); no result stream.
    bool exec(const std::string& sql);

    // --- mirrors the caliper.data.v1 vtable ---

    // Run SQL; on success fill *out with a live stream (caller drains and
    // releases). On failure *out is untouched and last_error() explains.
    bool query(const std::string& sql, ArrowArrayStream* out);

    // Name a dataset: uri may be a .parquet/.csv path or an existing table
    // name. Names must be identifiers ([A-Za-z_][A-Za-z0-9_]*); anything
    // else is rejected (names are spliced into SQL as identifiers).
    bool register_dataset(const std::string& name, const std::string& uri);

    bool open_dataset(const std::string& name, ArrowArrayStream* out);

    // The error of this thread's last failing call ("" if none yet).
    std::string last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_host
