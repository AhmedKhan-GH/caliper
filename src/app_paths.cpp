#include "app_paths.h"

#include <cstdlib>
#include <filesystem>
#include <system_error>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <shlobj.h>
#endif

namespace fs = std::filesystem;

namespace caliper {

namespace {

fs::path compute_app_data_dir() {
#ifdef __APPLE__
    const char* home = std::getenv("HOME");
    fs::path base = fs::path(home ? home : "/tmp")
                  / "Library" / "Application Support" / "Caliper";
#elif defined(_WIN32)
    fs::path base;
    PWSTR appdata = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0,
                                       nullptr, &appdata))) {
        base = fs::path(appdata) / L"Caliper";
        CoTaskMemFree(appdata);
    } else {
        base = fs::current_path() / "caliper-data";
    }
#else
    fs::path base;
    const char* xdg = std::getenv("XDG_DATA_HOME");
    if (xdg && *xdg) {
        base = fs::path(xdg) / "caliper";
    } else {
        const char* home = std::getenv("HOME");
        base = fs::path(home ? home : "/tmp") / ".local" / "share" / "caliper";
    }
#endif

    std::error_code ec;
    fs::create_directories(base, ec);
    // Even if create_directories fails, return the path; callers can fail
    // gracefully when their own file ops error out.
    return base;
}

} // anonymous namespace

const std::string& app_data_dir() {
    // Computed once and cached. The directory is created on first call.
    static const std::string cached = compute_app_data_dir().string();
    return cached;
}

std::string app_data_path(const std::string& filename) {
    return (fs::path(app_data_dir()) / filename).string();
}

} // namespace caliper
