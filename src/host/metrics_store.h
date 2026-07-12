#pragma once
// caliper_host::MetricsStore — the DuckDB-backed run/tag/step store behind
// caliper.metrics.v1 (PLATFORM.md §11: DuckDB is a host implementation detail).
//
// TensorBoard vocabulary: an experiment groups runs; each run carries tagged
// series indexed by step (scalars, histograms, images) plus a hparams JSON blob.
//
// Threading: every method is callable from applet job threads. The store owns
// one DuckDB connection guarded by one mutex — calls serialize internally
// (the metrics.v1 header promises this). This is the sanctioned v1 model.
//
// No DuckDB type appears in this header: the connection lives behind a pimpl,
// so consumers of caliper_host_lib never pull in <duckdb.hpp>.
//
// ArrowArrayStream (the ABI's own tabular type, from the vendored spec header)
// is the only non-primitive type here — it backs the caliper.metrics.v1_1 read
// surface (query()), which streams metric rows out exactly as caliper.data.v1
// does, but against THIS store's live connection.
#include <caliper/arrow_c.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace caliper_host {

struct RunInfo {
    uint64_t    id;
    std::string experiment;
    std::string name;
    bool        done;
    std::string hparams;  // JSON, empty if never set
};

// One persisted histogram record (the raw float blob is not returned — v1
// readers only need its shape; the blob's survival is proven by byte_length).
struct HistogramInfo {
    int64_t step;
    int64_t count;        // number of float values written
    int64_t byte_length;  // size of the stored BLOB in bytes
};

class MetricsStore {
public:
    MetricsStore();
    ~MetricsStore();

    MetricsStore(const MetricsStore&) = delete;
    MetricsStore& operator=(const MetricsStore&) = delete;

    // Open a database at `path` (or ":memory:") and create the schema.
    // Returns false if the database could not be opened.
    bool open(const std::string& path);

    // Deterministic teardown: drop the connection + database now. Process-
    // lifetime stores MUST be closed before main returns — destroying a
    // DuckDB instance from a static destructor races DuckDB's own globals
    // (undefined cross-TU order) and aborts in malloc. Idempotent.
    void close();

    // --- writers (mirror the caliper.metrics.v1 vtable) ---

    // Begin a run; returns a unique nonzero id, or 0 on error. Ids are unique
    // across the lifetime of an open store (see .cpp: MAX(id)+1 on open, then
    // monotonic in-memory counter — reopen resumes above the persisted max).
    uint64_t begin_run(const std::string& experiment, const std::string& name);
    void     end_run(uint64_t run);  // flips `done`; inert for unknown runs
    void     scalar(uint64_t run, const std::string& tag, int64_t step, double value);
    void     histogram(uint64_t run, const std::string& tag, int64_t step,
                       const float* values, int64_t count);
    // hwc_u8 image bytes (row-major H*W*C); inert for unknown runs.
    void     image(uint64_t run, const std::string& tag, int64_t step,
                  const void* bytes, int32_t w, int32_t h, int32_t c);
    void     hparams_json(uint64_t run, const std::string& json_utf8);

    // --- history management (host dashboard only; not part of metrics.v1) ---

    // Remove one run and all its series. The id is never reissued (artifacts
    // lineage keys on run ids), and future writes to it stay inert.
    void delete_run(uint64_t run);
    // Remove ALL runs/series. Ids keep counting upward — see delete_run.
    void clear_all();

    // --- readers ---

    std::vector<RunInfo>     runs();
    std::vector<std::string> scalar_tags(uint64_t run);
    // Step-ascending (ORDER BY step) list of (step, value) pairs.
    std::vector<std::pair<int64_t, double>> scalars(uint64_t run, const std::string& tag);
    std::vector<HistogramInfo> histograms(uint64_t run, const std::string& tag);

    // --- read surface across the ABI (backs caliper.metrics.v1_1) ---

    // Run SQL against THIS store's live connection; on success fill *out with a
    // live Arrow C stream (caller drains and releases). Serialized on the same
    // mutex as every write, so a host UI thread may query while an applet worker
    // streams scalars in — reads are immediately consistent with those writes
    // (unlike a separate read-only DuckDB instance, which only sees checkpointed
    // state). READ-ONLY ENFORCED: the SQL is parsed (not executed) and refused
    // unless it is exactly one SELECT statement — the connection is the live
    // WRITER, so an INSERT/DROP slipping through would corrupt the store. On
    // refusal or failure *out is untouched and last_error() explains.
    bool query(const std::string& sql, ArrowArrayStream* out);

    // The error of this thread's last failing query() ("" if none yet).
    std::string last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_host
