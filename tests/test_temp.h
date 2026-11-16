#pragma once
// A path of its own under the system's temporary folder for one test run:
// <name>-<pid>-<random>. The temporary folder is shared by every process of the
// user, so a fixed name is shared too: two runs of a suite at the same time
// (two build folders, two checkouts, ctest -j) would delete and overwrite one
// another's files there. Nothing is created; the caller makes and removes it.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#ifdef _WIN32
#include <process.h>   // _getpid
#else
#include <unistd.h>    // getpid
#endif

namespace caliper_test {

inline std::filesystem::path unique_temp_path(const std::string& name) {
#ifdef _WIN32
    const long pid = (long)_getpid();
#else
    const long pid = (long)getpid();
#endif
    std::random_device rd;
    const uint64_t r = (uint64_t)rd() << 32 | (uint64_t)rd();
    char suffix[48];
    std::snprintf(suffix, sizeof suffix, "-%ld-%016llx", pid, (unsigned long long)r);
    return std::filesystem::temp_directory_path() / (name + suffix);
}

}  // namespace caliper_test
