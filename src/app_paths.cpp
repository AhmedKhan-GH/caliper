#include "app_paths.h"

#include <cstdlib>
#include <filesystem>
#include <system_error>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <shlobj.h>
#elif defined(__APPLE__)
#  include <mach-o/dyld.h>
#  include <cstdint>
#endif

namespace fs = std::filesystem;

namespace caliper {

namespace {

// The OS default app-data folder. Computes the path only; creates nothing.
fs::path os_app_data_dir() {
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
    return base;
}

// CALIPER_DATA_DIR when it is set and not empty, else the OS default; read
// once, at the first lookup. Creates nothing.
const std::string& default_app_data_dir() {
    static const std::string dir = [] {
        if (const char* env = std::getenv("CALIPER_DATA_DIR"); env && *env)
            return fs::path(env).string();
        return os_app_data_dir().string();
    }();
    return dir;
}

} // anonymous namespace

// Embed v1.1 override root. Empty => no override (the OS default is used). The
// caliper exe never sets this, so its default path resolution is unchanged.
namespace {
std::string g_override_root;   // "" = unset
}

void set_app_data_dir_override(const std::string& dir) {
    if (dir.empty()) { g_override_root.clear(); return; }
    std::error_code ec;
    fs::create_directories(dir, ec);   // best-effort, mirrors compute_app_data_dir
    g_override_root = fs::path(dir).string();
}

const std::string& app_data_dir() {
    if (!g_override_root.empty()) return g_override_root;
    // Created once, at the first call. Even if that fails the path is
    // returned; callers fail gracefully when their own file operations do.
    static const bool created = [] {
        std::error_code ec;
        fs::create_directories(default_app_data_dir(), ec);
        return true;
    }();
    (void)created;
    return default_app_data_dir();
}

std::string app_data_location() {
    return g_override_root.empty() ? default_app_data_dir() : g_override_root;
}

std::string app_data_path(const std::string& filename) {
    return (fs::path(app_data_dir()) / filename).string();
}

std::string executable_dir() {
#ifdef __APPLE__
    char buf[4096];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) != 0) return {};
    return fs::path(buf).parent_path().string();
#elif defined(_WIN32)
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return {};
    return fs::path(buf).parent_path().string();
#else
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    return ec ? std::string() : exe.parent_path().string();
#endif
}

std::string shipped_applets_dir() {
    const std::string dir = executable_dir();
    return dir.empty() ? std::string() : (fs::path(dir) / "applets").string();
}

std::string user_applets_dir() {
    if (const char* dir = std::getenv("CALIPER_USER_APPLETS_DIR"); dir && *dir)
        return dir;
    return (fs::path(app_data_location()) / "applets").string();
}

} // namespace caliper
