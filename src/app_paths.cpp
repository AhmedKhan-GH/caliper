#include "app_paths.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <shlobj.h>
#elif defined(__APPLE__)
#  include <crt_externs.h>
#  include <mach-o/dyld.h>
#  include <cstdint>
#else
#  include <unistd.h>
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

// The process's environment, as `name=value` strings. A dynamic library on
// macOS (the Python extension links this file) has no `environ` of its own.
char** environment() {
#ifdef _WIN32
    return _environ;
#elif defined(__APPLE__)
    return *_NSGetEnviron();
#else
    return environ;
#endif
}

// The guard's marker (app_data_guard()); a string literal, set before threads.
std::atomic<const char*> g_guard_what{nullptr};

// Why the environment makes this a test or hidden run, "" when it does not:
// CALIPER_HIDDEN=1, or the first CALIPER_TEST_* variable that is not empty.
std::string guard_reason_from_environment() {
    if (const char* h = std::getenv("CALIPER_HIDDEN"); h && std::strcmp(h, "1") == 0)
        return "the hidden app (CALIPER_HIDDEN=1)";
    static const char kPrefix[] = "CALIPER_TEST_";
    for (char** e = environment(); e && *e; ++e) {
        const char* kv = *e;
        if (std::strncmp(kv, kPrefix, sizeof(kPrefix) - 1) != 0) continue;
        const char* eq = std::strchr(kv, '=');
        if (!eq || eq[1] == '\0') continue;
        return "a test run (" + std::string(kv, eq) + " is set)";
    }
    return {};
}

// The default folder and whether it was given explicitly (CALIPER_DATA_DIR),
// read once, at the first lookup. Creates nothing.
struct DefaultDir {
    std::string path;
    bool given = false;
};

const DefaultDir& resolved_default() {
    static const DefaultDir dir = [] {
        if (const char* env = std::getenv("CALIPER_DATA_DIR"); env && *env)
            return DefaultDir{fs::path(env).string(), true};
        return DefaultDir{os_app_data_dir().string(), false};
    }();
    return dir;
}

// CALIPER_DATA_DIR when it is set and not empty, else the OS default, which a
// test or hidden run refuses (THE GUARD, app_paths.h): it says why on standard
// error and ends the process with exit code 1, before anything is created.
const std::string& default_app_data_dir() {
    const DefaultDir& dir = resolved_default();
    if (!dir.given) {
        const std::string why = app_data_guard_reason();
        if (!why.empty()) {
            std::fprintf(stderr,
                         "caliper: %s has no data folder of its own and would use your "
                         "app-data folder %s; set CALIPER_DATA_DIR to a folder of its own\n",
                         why.c_str(), dir.path.c_str());
            std::fflush(stdout);
            std::fflush(stderr);
            // No static destructors: other threads may be running.
            std::_Exit(1);
        }
    }
    return dir.path;
}

} // anonymous namespace

void app_data_guard(const char* what) {
    g_guard_what.store(what && *what ? what : "a test or hidden run");
}

std::string app_data_guard_reason() {
    if (const char* what = g_guard_what.load()) return what;
    static const std::string from_env = guard_reason_from_environment();
    return from_env;
}

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

std::string default_workspace_dir() {
    return (fs::path(app_data_location()) / "workspace").string();
}

} // namespace caliper
