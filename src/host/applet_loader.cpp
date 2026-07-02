#include "applet_loader.h"
#include "crash_guard.h"
#include <algorithm>
#include <filesystem>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

namespace fs = std::filesystem;

namespace caliper_host {
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
std::string lib_error() {
#ifdef _WIN32
    char buf[256] = {};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(),
                   0, buf, sizeof(buf), nullptr);
    return buf;
#else
    const char* e = dlerror();
    return e ? e : "unknown dlopen error";
#endif
}

#ifdef _WIN32
  constexpr const char* kExt = ".dll";
  constexpr const char* kPrefix = "";
#elif __APPLE__
  constexpr const char* kExt = ".dylib";
  constexpr const char* kPrefix = "lib";
#else
  constexpr const char* kExt = ".so";
  constexpr const char* kPrefix = "lib";
#endif

constexpr const char* kManifestSuffix = ".caliper.toml";

// <stem>.caliper.toml -> sibling (lib)?<stem>.<ext>, or "".
std::string find_binary(const fs::path& dir, const std::string& stem) {
    for (const std::string& name : {stem + kExt, kPrefix + stem + kExt}) {
        std::error_code ec;
        if (fs::is_regular_file(dir / name, ec)) return (dir / name).string();
    }
    return {};
}

} // namespace

AppletLoader::AppletLoader(HostCaps caps, std::string data_root)
    : caps_(std::move(caps)), data_root_(std::move(data_root)) {}

int AppletLoader::scan(const std::string& dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return 0;

    int added = 0;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        const std::string fname = e.path().filename().string();
        if (fname.size() <= std::string(kManifestSuffix).size() ||
            fname.substr(fname.size() - std::string(kManifestSuffix).size())
                != kManifestSuffix)
            continue;

        AppletEntry entry;
        auto parsed = parse_manifest_file(e.path().string());
        if (!parsed.ok) {
            entry.manifest.name = fname;
            entry.status = AppletStatus::Failed;
            entry.status_text = parsed.error;
            entries_.push_back(std::move(entry));
            added++;
            continue;
        }
        entry.manifest = std::move(parsed.manifest);
        entry.data_dir = data_root_ + "/" + entry.manifest.id;

        const std::string stem =
            fname.substr(0, fname.size() - std::string(kManifestSuffix).size());
        entry.dylib_path = find_binary(e.path().parent_path(), stem);
        if (entry.dylib_path.empty()) {
            entry.status = AppletStatus::Failed;
            entry.status_text = "applet binary not found next to " + fname;
        } else if (auto n = negotiate(entry.manifest, caps_); !n.ok) {
            entry.status = AppletStatus::Refused;
            entry.status_text = n.reason;
        } else {
            entry.status = AppletStatus::Ready;
        }
        entries_.push_back(std::move(entry));
        added++;
    }
    std::sort(entries_.begin(), entries_.end(),
              [](const AppletEntry& a, const AppletEntry& b) {
                  return a.manifest.name < b.manifest.name;
              });
    // Rescanning reallocates host_blocks_, which active applets hold pointers
    // into — scan() must only run before any launch (true for both the app
    // and the tests; enforce it if a rescan feature ever appears).
    host_blocks_.assign(entries_.size(), CaliperHost{});
    return added;
}

bool AppletLoader::launch(int idx, CaliperHost host_proto) {
    if (idx < 0 || idx >= count()) return false;
    AppletEntry& a = entries_[idx];
    if (a.status == AppletStatus::Active) teardown(idx);
    if (a.status != AppletStatus::Ready) return false;

    auto fail = [&](std::string why) {
        a.status = AppletStatus::Failed;
        a.status_text = std::move(why);
        return false;
    };

    if (!a.handle) {
        a.handle = lib_open(a.dylib_path.c_str());
        if (!a.handle) return fail("load failed: " + lib_error());
        auto get_desc = (const CaliperAppletDescriptor* (*)(void))
            lib_sym(a.handle, CALIPER_DESCRIPTOR_SYMBOL);
        if (!get_desc)
            return fail(std::string("missing export ") + CALIPER_DESCRIPTOR_SYMBOL);
        a.desc = get_desc();
    }

    // Descriptor sanity: the binary must agree with its manifest (§14).
    const auto* d = a.desc;
    if (!d || d->struct_size < sizeof(CaliperAppletDescriptor))
        return fail("descriptor missing or truncated");
    if (d->abi_epoch != caps_.abi_epoch)
        return fail("descriptor ABI epoch disagrees with manifest");
    if (!d->id || a.manifest.id != d->id)
        return fail("descriptor id disagrees with manifest");
    if (!d->version || a.manifest.version != d->version)
        return fail("descriptor version disagrees with manifest");
    if (!d->api.create || !d->api.destroy || !d->api.initialize ||
        !d->api.frame || !d->api.cleanup)
        return fail("descriptor function table incomplete");

    std::error_code ec;
    fs::create_directories(a.data_dir, ec);
    host_blocks_[idx] = host_proto;
    host_blocks_[idx].applet_data_dir = a.data_dir.c_str();

    void* instance = nullptr;
    auto cr = guarded_call([&] { instance = d->api.create(); });
    if (!cr.ok) { a.status = AppletStatus::Quarantined;
                  a.status_text = "crashed in create(): " + cr.fault; return false; }
    if (!instance) return fail("create() returned null");

    bool init_ok = false;
    auto ir = guarded_call([&] {
        init_ok = d->api.initialize(instance, &host_blocks_[idx]);
    });
    if (!ir.ok) { a.status = AppletStatus::Quarantined;
                  a.status_text = "crashed in initialize(): " + ir.fault; return false; }
    if (!init_ok) {
        guarded_call([&] { d->api.destroy(instance); });
        return fail("initialize() returned false");
    }

    a.instance = instance;
    a.status = AppletStatus::Active;
    a.status_text.clear();
    return true;
}

bool AppletLoader::frame(int idx, const CaliperFrameInfo& info) {
    if (idx < 0 || idx >= count()) return false;
    AppletEntry& a = entries_[idx];
    if (a.status != AppletStatus::Active || !a.instance) return false;

    auto r = guarded_call([&] { a.desc->api.frame(a.instance, &info); });
    if (!r.ok) {
        // Memory is suspect after a fault: abandon the instance, never call
        // cleanup/destroy/dlclose on it (§15 honesty).
        a.status = AppletStatus::Quarantined;
        a.status_text = "crashed in frame(): " + r.fault;
        a.instance = nullptr;
        return false;
    }
    return true;
}

void AppletLoader::teardown(int idx) {
    if (idx < 0 || idx >= count()) return;
    AppletEntry& a = entries_[idx];
    if (a.status != AppletStatus::Active || !a.instance) return;

    auto cl = guarded_call([&] { a.desc->api.cleanup(a.instance); });
    auto de = cl.ok
        ? guarded_call([&] { a.desc->api.destroy(a.instance); })
        : cl;
    a.instance = nullptr;
    if (!cl.ok || !de.ok) {
        a.status = AppletStatus::Quarantined;
        a.status_text = "crashed during teardown: " + (cl.ok ? de : cl).fault;
        return;
    }
    a.status = AppletStatus::Ready;
}

void AppletLoader::close_all() {
    for (int i = 0; i < count(); i++) teardown(i);
    for (auto& a : entries_) {
        // Quarantined dylibs are left mapped: running static destructors in a
        // corrupted image is worse than a small leak at shutdown.
        if (a.handle && a.status != AppletStatus::Quarantined) lib_close(a.handle);
        a.handle = nullptr;
        a.desc = nullptr;
    }
    entries_.clear();
    host_blocks_.clear();
}

} // namespace caliper_host
