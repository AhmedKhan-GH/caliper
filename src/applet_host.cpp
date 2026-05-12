#include "applet_host.h"

#include <iostream>
#include <filesystem>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

namespace fs = std::filesystem;

namespace {

void* lib_open(const char* path) {
#ifdef _WIN32
    return (void*)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

void lib_close(void* h) {
#ifdef _WIN32
    FreeLibrary((HMODULE)h);
#else
    dlclose(h);
#endif
}

void* lib_sym(void* h, const char* name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)h, name);
#else
    return dlsym(h, name);
#endif
}

const char* lib_error() {
#ifdef _WIN32
    static thread_local char buf[256];
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(),
                   0, buf, sizeof(buf), nullptr);
    return buf;
#else
    return dlerror();
#endif
}

#ifdef _WIN32
  const char* kLibExt = ".dll";
#elif __APPLE__
  const char* kLibExt = ".dylib";
#else
  const char* kLibExt = ".so";
#endif

} // namespace

void AppletHost::scan(const std::string& dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return;

    for (auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext != kLibExt) continue;

        std::string path = entry.path().string();
        void* h = lib_open(path.c_str());
        if (!h) {
            std::cerr << "[applet] Failed to load " << path
                      << ": " << lib_error() << std::endl;
            continue;
        }

        auto fn_info = (PFN_applet_info)lib_sym(h, "applet_info");
        if (!fn_info) {
            std::cerr << "[applet] " << path
                      << " missing applet_info()" << std::endl;
            lib_close(h);
            continue;
        }

        CaliperAppletInfo info = fn_info();
        if (info.abi != CALIPER_APPLET_ABI) {
            std::cerr << "[applet] " << path
                      << " ABI mismatch: got " << info.abi
                      << ", want " << CALIPER_APPLET_ABI << std::endl;
            lib_close(h);
            continue;
        }

        LoadedApplet a;
        a.path          = path;
        a.handle        = h;
        a.fn_info       = fn_info;
        a.fn_create     = (PFN_applet_create)lib_sym(h, "applet_create");
        a.fn_destroy    = (PFN_applet_destroy)lib_sym(h, "applet_destroy");
        a.fn_initialize = (PFN_applet_initialize)lib_sym(h, "applet_initialize");
        a.fn_draw_ui    = (PFN_applet_draw_ui)lib_sym(h, "applet_draw_ui");
        a.fn_cleanup    = (PFN_applet_cleanup)lib_sym(h, "applet_cleanup");
        a.info          = info;

        if (!a.fn_create || !a.fn_destroy || !a.fn_initialize ||
            !a.fn_draw_ui || !a.fn_cleanup) {
            std::cerr << "[applet] " << path
                      << " missing required exports" << std::endl;
            lib_close(h);
            continue;
        }

        std::cout << "[applet] Loaded: " << info.name
                  << " v" << info.version << std::endl;
        applets_.push_back(std::move(a));
    }
}

bool AppletHost::launch(int idx, const CaliperHostContext& host) {
    if (idx < 0 || idx >= (int)applets_.size()) return false;
    auto& a = applets_[idx];

    if (a.instance) teardown(idx);

    a.instance = a.fn_create();
    if (!a.instance) return false;

    if (!a.fn_initialize(a.instance, &host)) {
        a.fn_destroy(a.instance);
        a.instance = nullptr;
        return false;
    }

    a.initialized = true;
    return true;
}

void AppletHost::draw(int idx, int w, int h) {
    if (idx < 0 || idx >= (int)applets_.size()) return;
    auto& a = applets_[idx];
    if (!a.instance || !a.initialized) return;
    a.fn_draw_ui(a.instance, w, h);
}

void AppletHost::teardown(int idx) {
    if (idx < 0 || idx >= (int)applets_.size()) return;
    auto& a = applets_[idx];
    if (!a.instance) return;

    if (a.initialized) {
        a.fn_cleanup(a.instance);
        a.initialized = false;
    }
    a.fn_destroy(a.instance);
    a.instance = nullptr;
}

void AppletHost::close_all() {
    for (int i = 0; i < (int)applets_.size(); i++)
        teardown(i);
    for (auto& a : applets_) {
        if (a.handle) lib_close(a.handle);
    }
    applets_.clear();
}
