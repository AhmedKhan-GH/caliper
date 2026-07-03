#pragma once
// caliper_host::ArtifactStore — the content-addressed blob store behind
// caliper.artifacts.v1 (PLATFORM.md §7.8): checkpoints keyed by sha256,
// deduplicated on disk, lineage-linked to the metrics run that produced them.
//
// Layout: blobs live as files at <root>/artifacts/<64-hex-digest>; the index
// (digest, name, run, len, ts) lives in DuckDB at <root>/artifacts.duckdb.
//
// Threading: every method is callable from applet job threads. One DuckDB
// connection guarded by one mutex — calls serialize internally (the
// artifacts.v1 header promises this). Same sanctioned model as MetricsStore.
//
// No DuckDB type appears in this header (pimpl), so consumers of
// caliper_host_lib never pull in <duckdb.hpp>.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caliper_host {

class ArtifactStore {
public:
    ArtifactStore();
    ~ArtifactStore();

    ArtifactStore(const ArtifactStore&) = delete;
    ArtifactStore& operator=(const ArtifactStore&) = delete;

    // Open (or create) the store rooted at `root_dir`: creates
    // <root>/artifacts/ and the index DB. Returns false on failure; all
    // other methods are inert on an unopened store.
    bool open(const std::string& root_dir);

    // Deterministic teardown before main returns (see MetricsStore::close()
    // — static-destructor DuckDB teardown aborts). Idempotent.
    void close();

    // --- mirrors the caliper.artifacts.v1 vtable ---

    // Store bytes under their sha256. Identical bytes dedup to one file;
    // every call upserts an index row (name, run, ts), so a reused name
    // resolves to its newest digest. run=0 means unlinked. Writes 64 hex
    // chars + NUL into out_digest[65]. Returns false on error.
    bool put(const std::string& name, const void* bytes, uint64_t len,
             uint64_t run, char out_digest[65]);

    // Resolve a digest OR a name (name -> newest row's digest) to the blob's
    // absolute path. Empty string if unknown.
    std::string path_of(const std::string& digest_or_name);

    bool exists(const std::string& digest_or_name);

    // Run lineage: digests recorded against `run` (any order).
    std::vector<std::string> by_run(uint64_t run);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_host
