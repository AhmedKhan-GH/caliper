#pragma once
// How large the host's three DuckDB stores may grow (PLATFORM.md §11), as a
// rule over the machine rather than fixed numbers. DuckDB's own defaults are
// ALL hardware threads and 80% of physical RAM — per instance; three of those
// inside a UI host that also runs applet job threads and libtorch is 3x
// oversubscription and no memory budget at all. Memory is a share of physical
// RAM held between a floor and a ceiling; threads, a share of the cores:
//
//              memory                         threads
//   metrics     2% of RAM in [512MB, 4GB]     cores/4  in [1, 4]
//   artifacts   1% of RAM in [256MB, 2GB]     cores/16 in [1, 2]
//   data       25% of RAM in [2GB, 32GB]      cores/2  in [2, 16]
//
// MB/GB are DuckDB's decimal units. The floors are the fixed limits this rule
// replaced, so a small machine (or one whose RAM can't be read) opens exactly
// as before, and an 8-core machine gets the same 2/1/4 threads; a 64GB Mac
// gets a 17GB data store instead of 2GB.
//
// Overrides: CALIPER_DUCKDB_THREADS and CALIPER_DUCKDB_MEMORY_LIMIT (DuckDB's
// size string, e.g. "4GB") apply to all three; the per-store
// CALIPER_DUCKDB_<METRICS|DATA|ARTIFACTS>_THREADS / _MEMORY_LIMIT win over
// those. An unparsable value falls back to the computed default with one
// stderr line.
//
// DuckDB-free on purpose: duckdb_config.h applies the result to a DBConfig,
// and the tests pin the rule here without DuckDB on their include path.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(_WIN32)
// Declare, don't include (the vulkan_renderer.cpp discipline): <windows.h>
// here would leak its macros into the three store TUs after <duckdb.hpp>.
// Layout of MEMORYSTATUSEX, unchanged since Windows 2000. A TU that also
// includes <windows.h> gets a compile error for the second declaration —
// loud, not wrong. Written for Windows, not yet built or run there.
struct CaliperMemoryStatusEx {
    unsigned long dwLength;
    unsigned long dwMemoryLoad;
    unsigned long long ullTotalPhys;
    unsigned long long ullAvailPhys;
    unsigned long long ullTotalPageFile;
    unsigned long long ullAvailPageFile;
    unsigned long long ullTotalVirtual;
    unsigned long long ullAvailVirtual;
    unsigned long long ullAvailExtendedVirtual;
};
extern "C" __declspec(dllimport) int __stdcall GlobalMemoryStatusEx(
    CaliperMemoryStatusEx* lpBuffer);
#else
#include <unistd.h>
#endif

namespace caliper_host {

struct DuckDBLimits {
    int64_t     threads;
    std::string memory_limit;   // DuckDB size string ("512MB", "17179MB", ...)
};

// One store's sizing rule (see the table above).
struct DuckDBSizing {
    double  ram_share;           // fraction of physical RAM
    int64_t memory_floor_mb;     // DuckDB MB (10^6 bytes)
    int64_t memory_ceiling_mb;
    int64_t cores_per_thread;    // one DuckDB thread per this many cores
    int64_t threads_floor;
    int64_t threads_ceiling;
};

inline constexpr DuckDBSizing kMetricsDuckDBLimits   {0.02,  512,  4000,  4, 1,  4};
inline constexpr DuckDBSizing kArtifactsDuckDBLimits {0.01,  256,  2000, 16, 1,  2};
inline constexpr DuckDBSizing kDataDuckDBLimits      {0.25, 2000, 32000,  2, 2, 16};

// What the rule is computed from; 0 = could not be read (the floors apply).
struct MachineResources {
    uint64_t ram_bytes = 0;
    unsigned cores = 0;
};

inline uint64_t physical_ram_bytes() {
#if defined(__APPLE__)
    uint64_t bytes = 0;
    size_t len = sizeof bytes;
    if (sysctlbyname("hw.memsize", &bytes, &len, nullptr, 0) != 0) return 0;
    return bytes;
#elif defined(_WIN32)
    CaliperMemoryStatusEx ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? ms.ullTotalPhys : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page  = sysconf(_SC_PAGE_SIZE);
    return (pages > 0 && page > 0)
               ? static_cast<uint64_t>(pages) * static_cast<uint64_t>(page)
               : 0;
#endif
}

inline MachineResources machine_resources() {
    return {physical_ram_bytes(), std::thread::hardware_concurrency()};
}

// The limits a store opens with on machine `m`, before any override.
inline DuckDBLimits duckdb_default_limits(const DuckDBSizing& s,
                                          const MachineResources& m) {
    const auto share_mb = static_cast<int64_t>(
        static_cast<double>(m.ram_bytes) * s.ram_share / 1e6);
    const int64_t mb =
        std::clamp(share_mb, s.memory_floor_mb, s.memory_ceiling_mb);
    const int64_t threads =
        std::clamp(static_cast<int64_t>(m.cores) / s.cores_per_thread,
                   s.threads_floor, s.threads_ceiling);
    return {threads, std::to_string(mb) + "MB"};
}

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

} // namespace caliper_host
