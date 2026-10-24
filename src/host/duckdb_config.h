#pragma once
// Explicit DuckDB resource limits for the host's three stores (PLATFORM.md
// §11), applied to the DBConfig each store opens with. The numbers — a rule
// over this machine's RAM and cores — and the environment overrides live in
// duckdb_sizing.h, which carries no DuckDB type.
//
// Header-only, included by the three store .cpp files ONLY: no DuckDB type
// may reach a public header (the pimpl seam, CMakeLists.txt "DuckDB is a host
// implementation detail").
#include "duckdb_sizing.h"

#include <duckdb.hpp>

#include <cstdio>
#include <string>

namespace caliper_host {

// Put the limits on a DBConfig about to open a store: `sizing` (one of the
// k*DuckDBLimits rules) sized from this machine, then the environment. A
// memory string DuckDB rejects falls back to the computed default (one stderr
// line) rather than failing the open — a misspelt override must not take a
// service down.
inline void duckdb_apply_limits(duckdb::DBConfig& cfg, const char* store,
                                const DuckDBSizing& sizing) {
    const DuckDBLimits defaults =
        duckdb_default_limits(sizing, machine_resources());
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
