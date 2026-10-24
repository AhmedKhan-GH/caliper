#pragma once
// Explicit DuckDB resource limits for the host's three stores (PLATFORM.md
// §11). DuckDB's own defaults are ALL hardware threads and 80% of physical RAM
// — per instance. Three instances inside a UI host that also runs applet job
// threads and libtorch is 3x oversubscription and no memory budget at all.
// The defaults below are a floor against runaway, not a tuning:
//   metrics    2 threads / 512MB   append-mostly, small result sets
//   artifacts  1 thread  / 256MB   a digest index
//   data       4 threads / 2GB     user datasets — the only store that scans
// Override per machine: CALIPER_DUCKDB_THREADS and CALIPER_DUCKDB_MEMORY_LIMIT
// (DuckDB's size string, e.g. "4GB") apply to all three; the per-store
// CALIPER_DUCKDB_<METRICS|DATA|ARTIFACTS>_THREADS / _MEMORY_LIMIT win over
// those. An unparsable value falls back to the default with one stderr line.
//
// Header-only, included by the three store .cpp files ONLY: no DuckDB type
// may reach a public header (the pimpl seam, CMakeLists.txt "DuckDB is a host
// implementation detail").
#include <duckdb.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace caliper_host {

struct DuckDBLimits {
    int64_t     threads;
    std::string memory_limit;   // DuckDB size string ("512MB", "2GB", ...)
};

inline constexpr DuckDBLimits kMetricsDuckDBLimits   {2, "512MB"};
inline constexpr DuckDBLimits kArtifactsDuckDBLimits {1, "256MB"};
inline constexpr DuckDBLimits kDataDuckDBLimits      {4, "2GB"};

namespace duckdb_config_detail {

inline const char* env_first(const std::string& a, const std::string& b) {
    if (const char* v = std::getenv(a.c_str()); v && *v) return v;
    if (const char* v = std::getenv(b.c_str()); v && *v) return v;
    return nullptr;
}

inline bool parse_threads(const char* s, int64_t& out) {
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0' || v < 1 || v > 1024) return false;
    out = v;
    return true;
}

} // namespace duckdb_config_detail

// The limits one store opens with: its defaults, overridden by the environment.
// `store` is the upper-case name used in the per-store variables.
inline DuckDBLimits duckdb_limits_for(const char* store, DuckDBLimits defaults) {
    using namespace duckdb_config_detail;
    DuckDBLimits lim = defaults;
    const std::string per = std::string("CALIPER_DUCKDB_") + store + "_";
    if (const char* t = env_first(per + "THREADS", "CALIPER_DUCKDB_THREADS")) {
        if (!parse_threads(t, lim.threads))
            std::fprintf(stderr,
                         "[duckdb] %s: ignoring threads override \"%s\" "
                         "(want an integer 1..1024); using %lld\n",
                         store, t, static_cast<long long>(lim.threads));
    }
    if (const char* m = env_first(per + "MEMORY_LIMIT",
                                  "CALIPER_DUCKDB_MEMORY_LIMIT"))
        lim.memory_limit = m;   // validated by DuckDB in duckdb_apply_limits
    return lim;
}

// Put the limits on a DBConfig about to open a store. A memory string DuckDB
// rejects falls back to the default (one stderr line) rather than failing the
// open — a misspelt override must not take a service down.
inline void duckdb_apply_limits(duckdb::DBConfig& cfg, const char* store,
                                DuckDBLimits defaults) {
    const DuckDBLimits lim = duckdb_limits_for(store, defaults);
    cfg.SetOptionByName("threads", duckdb::Value::BIGINT(lim.threads));
    try {
        cfg.SetOptionByName("memory_limit", duckdb::Value(lim.memory_limit));
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "[duckdb] %s: ignoring memory_limit override \"%s\" (%s); "
                     "using %s\n",
                     store, lim.memory_limit.c_str(), e.what(),
                     defaults.memory_limit.c_str());
        cfg.SetOptionByName("memory_limit", duckdb::Value(defaults.memory_limit));
    }
}

} // namespace caliper_host
