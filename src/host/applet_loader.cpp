#include "applet_loader.h"
#include "crash_guard.h"
#include "job_system.h"
#include <toml++/toml.hpp>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX   // windows.h's min and max macros break std::min and std::max
  #endif
  #include <windows.h>
#else
  #include <dlfcn.h>
  #include <cerrno>
  #include <fcntl.h>
  #include <sys/file.h>
  #include <unistd.h>
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

long process_id() {
#ifdef _WIN32
    return (long)GetCurrentProcessId();
#else
    return (long)getpid();
#endif
}

// An exclusive lock on <data_dir>/.caliper-data.lock for the stamp's
// read-modify-write, held until destruction. The OS drops it when the
// process ends, so a crash never leaves the folder locked. The lock file
// stays in the folder: removing it while another process waits on it would
// let a third one take a second lock.
class StampLock {
public:
    explicit StampLock(const fs::path& dir) {
        const std::string path = (dir / ".caliper-data.lock").string();
#ifdef _WIN32
        h_ = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h_ != INVALID_HANDLE_VALUE) {
            OVERLAPPED ov{};
            locked_ = LockFileEx(h_, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &ov) != 0;
        }
#else
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd_ >= 0) {
            int rc;
            do rc = ::flock(fd_, LOCK_EX); while (rc != 0 && errno == EINTR);
            locked_ = rc == 0;
        }
#endif
    }
    ~StampLock() {
#ifdef _WIN32
        if (h_ != INVALID_HANDLE_VALUE) {
            if (locked_) {
                OVERLAPPED ov{};
                UnlockFileEx(h_, 0, 1, 0, &ov);
            }
            CloseHandle(h_);
        }
#else
        if (fd_ >= 0) {
            if (locked_) ::flock(fd_, LOCK_UN);
            ::close(fd_);
        }
#endif
    }
    StampLock(const StampLock&) = delete;
    StampLock& operator=(const StampLock&) = delete;
    bool locked() const { return locked_; }

private:
#ifdef _WIN32
    HANDLE h_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
    bool locked_ = false;
};

} // namespace

uint32_t read_data_schema_stamp(const std::string& data_dir) {
    const fs::path stamp = fs::path(data_dir) / kDataStampName;
    std::error_code ec;
    if (!fs::is_regular_file(stamp, ec)) return 0;
    try {
        const toml::table t = toml::parse_file(stamp.string());
        const auto* n = t["schema"].as_integer();
        if (!n || n->get() < 1 || n->get() > INT32_MAX) return 0;
        return (uint32_t)n->get();
    } catch (...) {
        return 0;
    }
}

bool raise_data_schema_stamp(const std::string& data_dir, uint32_t schema) {
    if (schema == 0) return true;
    if (read_data_schema_stamp(data_dir) >= schema) return true;
    std::error_code ec;
    fs::create_directories(data_dir, ec);
    const fs::path stamp = fs::path(data_dir) / kDataStampName;
    // Two processes (or threads) raising the stamp at once would each read
    // the old value and the later rename would win, possibly with the lower
    // schema. The read-modify-write therefore runs under a lock on the
    // folder. Each write also goes through a temporary file of its own
    // (process id and a counter), and the stamp is read back after the
    // rename, in case the lock is not honoured, as on some network file
    // systems.
    static std::mutex in_process;
    std::lock_guard<std::mutex> guard(in_process);
    StampLock lock(data_dir);
    static std::atomic<unsigned> counter{0};
    for (int attempt = 0; attempt < 8; attempt++) {
        if (read_data_schema_stamp(data_dir) >= schema) return true;
        const fs::path tmp =
            fs::path(data_dir) / (std::string(kDataStampName) + "." +
                                  std::to_string(process_id()) + "." +
                                  std::to_string(counter.fetch_add(1)) + ".tmp");
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) return false;
            out << "# Written by the Caliper host. The highest data schema that has\n"
                   "# opened this folder; an applet with a lower schema is refused.\n"
                   "schema = " << schema << "\n";
            out.flush();
            if (!out) {
                out.close();
                fs::remove(tmp, ec);
                return false;
            }
        }
        fs::rename(tmp, stamp, ec);   // replaces the old stamp in one step
        if (ec) {
            fs::remove(tmp, ec);
            return false;
        }
    }
    return read_data_schema_stamp(data_dir) >= schema;
}

std::vector<std::string> split_applet_path(const std::string& value) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= value.size()) {
        size_t end = value.find(kAppletPathSeparator, start);
        if (end == std::string::npos) end = value.size();
        if (end > start) out.push_back(value.substr(start, end - start));
        start = end + 1;
    }
    return out;
}

namespace {
bool is_manifest_name(const std::string& fname) {
    const std::string suffix = kManifestSuffix;
    return fname.size() > suffix.size() &&
           fname.compare(fname.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// True when `dir` directly holds a regular file named <stem>.caliper.toml.
bool holds_manifest(const fs::path& dir) {
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
        if (is_manifest_name(it->path().filename().string()) &&
            it->is_regular_file(ec))
            return true;
    return false;
}
} // namespace

std::string resolve_applet_folder(const std::string& dir) {
    std::error_code ec;
    if (dir.empty() || !fs::is_directory(dir, ec) || holds_manifest(dir)) return dir;
    const fs::path sub = fs::path(dir) / "applets";
    if (fs::is_directory(sub, ec) && holds_manifest(sub)) return sub.string();
    return dir;
}

std::vector<std::string> applet_search_dirs(
    const std::vector<std::string>& command_line_dirs, const char* applet_path_env,
    const std::string& user_dir, const std::string& shipped_dir) {
    std::vector<std::string> wanted = command_line_dirs;
    if (applet_path_env)
        for (auto& d : split_applet_path(applet_path_env)) wanted.push_back(d);
    const size_t development = wanted.size();
    wanted.push_back(user_dir);
    wanted.push_back(shipped_dir);

    std::vector<std::string> out;
    std::vector<fs::path> seen;
    for (size_t i = 0; i < wanted.size(); i++) {
        const std::string& d = wanted[i];
        if (d.empty()) continue;
        std::error_code ec;
        fs::path p = fs::absolute(fs::path(d), ec);
        if (ec) p = fs::path(d);
        p = p.lexically_normal();
        // "dir/" and "dir" are one folder; so are two spellings of a link.
        if (p.has_relative_path() && p.filename().empty()) p = p.parent_path();
        if (i < development) p = fs::path(resolve_applet_folder(p.string()));
        fs::path key = fs::weakly_canonical(p, ec);
        if (ec) key = p;
        if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;
        seen.push_back(key);
        out.push_back(p.string());
    }
    return out;
}

const char* applet_status_word(AppletStatus s) {
    switch (s) {
        case AppletStatus::Ready:       return "READY";
        case AppletStatus::Active:      return "ACTIVE";
        case AppletStatus::Refused:     return "REFUSED";
        case AppletStatus::Failed:      return "FAILED";
        case AppletStatus::Quarantined: return "QUARANTINED";
        case AppletStatus::Shadowed:    return "SHADOWED";
    }
    return "UNKNOWN";
}

std::string shadowed_reason(const std::string& winner_version,
                            const std::string& winner_folder) {
    return "Version " + winner_version + " in " + winner_folder +
           " is used instead.";
}

AppletLoader::AppletLoader(HostCaps caps, std::string data_root)
    : caps_(std::move(caps)), data_root_(std::move(data_root)) {}

int AppletLoader::scan(const std::vector<std::string>& dirs) {
    int added = 0;
    for (const std::string& d : dirs) added += scan(d);
    return added;
}

int AppletLoader::scan(const std::string& dir) {
    const int rank = next_rank_++;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return 0;

    int added = 0;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        const std::string fname = e.path().filename().string();
        if (!is_manifest_name(fname)) continue;

        AppletEntry entry;
        entry.folder = e.path().parent_path().string();
        entry.manifest_path = e.path().string();
        entry.search_rank = rank;
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
        // The id becomes a path under the data root. The parser already
        // refuses ids outside [a-z0-9.-]; this re-check is the boundary's own
        // guarantee, independent of where the manifest came from.
        if (!is_valid_applet_id(entry.manifest.id)) {
            entry.manifest.name = fname;
            entry.status = AppletStatus::Failed;
            entry.status_text = "applet id is not path-safe: " + entry.manifest.id;
            entries_.push_back(std::move(entry));
            added++;
            continue;
        }
        entry.data_dir = data_root_ + "/" + entry.manifest.id;

        const std::string stem =
            fname.substr(0, fname.size() - std::string(kManifestSuffix).size());
        entry.dylib_path = find_binary(e.path().parent_path(), stem);
        const uint32_t stored = entry.manifest.data_schema
                                    ? read_data_schema_stamp(entry.data_dir)
                                    : 0;
        if (entry.dylib_path.empty()) {
            entry.status = AppletStatus::Failed;
            entry.status_text = "applet binary not found next to " + fname;
        } else if (auto n = negotiate(entry.manifest, caps_, stored); !n.ok) {
            entry.status = AppletStatus::Refused;
            entry.status_text = n.reason;
        } else {
            entry.status = AppletStatus::Ready;
        }
        entries_.push_back(std::move(entry));
        added++;
    }
    resolve_duplicates();
    // By name; copies of one applet newest first, then in search order.
    std::sort(entries_.begin(), entries_.end(),
              [](const AppletEntry& a, const AppletEntry& b) {
                  if (a.manifest.name != b.manifest.name)
                      return a.manifest.name < b.manifest.name;
                  if (int c = semver_cmp(a.manifest.version, b.manifest.version))
                      return c > 0;
                  return a.search_rank < b.search_rank;
              });
    // Rescanning reallocates host_blocks_, which active applets hold pointers
    // into — scan() must only run before any launch (true for both the app
    // and the tests; enforce it if a rescan feature ever appears).
    host_blocks_.assign(entries_.size(), CaliperHost{});
    return added;
}

// Recomputed from scratch after every scan: a Shadowed copy passed every
// check, so it is a candidate again until the comparison says otherwise.
void AppletLoader::resolve_duplicates() {
    for (auto& a : entries_)
        if (a.status == AppletStatus::Shadowed) {
            a.status = AppletStatus::Ready;
            a.status_text.clear();
        }
    // The chosen copy of each id: the highest version among the Ready
    // copies; among equal versions the one found first (then, within one
    // folder, the first library path, so the choice never depends on the
    // order the folder lists its files in).
    auto better = [](const AppletEntry& a, const AppletEntry& b) {
        if (int c = semver_cmp(a.manifest.version, b.manifest.version)) return c > 0;
        if (a.search_rank != b.search_rank) return a.search_rank < b.search_rank;
        return a.dylib_path < b.dylib_path;
    };
    std::map<std::string, const AppletEntry*> chosen;
    for (const auto& a : entries_) {
        if (a.status != AppletStatus::Ready) continue;
        auto [it, inserted] = chosen.emplace(a.manifest.id, &a);
        if (!inserted && better(a, *it->second)) it->second = &a;
    }
    for (auto& a : entries_) {
        if (a.status != AppletStatus::Ready) continue;
        const AppletEntry* best = chosen[a.manifest.id];
        if (best == &a) continue;
        a.status = AppletStatus::Shadowed;
        a.status_text = shadowed_reason(best->manifest.version, best->folder);
    }
}

int AppletLoader::find(const std::string& id) const {
    int fallback = -1;
    for (int i = 0; i < count(); i++) {
        const AppletEntry& a = entries_[i];
        if (a.manifest.id != id) continue;
        if (a.status == AppletStatus::Ready || a.status == AppletStatus::Active)
            return i;
        if (fallback < 0 && a.status != AppletStatus::Shadowed) fallback = i;
    }
    return fallback;
}

bool AppletLoader::open_and_check(AppletEntry& a) {
    auto fail = [&](std::string why) {
        a.status = AppletStatus::Failed;
        a.status_text = std::move(why);
        return false;
    };

    // The stamp may have been raised since scan() (another copy of the
    // applet, or another process using the same data folder): checked again
    // before the library is opened.
    if (a.manifest.data_schema != 0) {
        const uint32_t stored = read_data_schema_stamp(a.data_dir);
        if (a.manifest.data_schema < stored) {
            a.status = AppletStatus::Refused;
            a.status_text = data_schema_reason(stored, a.manifest.data_schema);
            return false;
        }
    }

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

    // The descriptor's required_services is the binary's own declaration —
    // the source of truth. negotiate() checked the MANIFEST's list before
    // dlopen (the fast path); a manifest that under-declares must not let an
    // applet run against a host missing a service it will call. Same card
    // text as the manifest refusal. The image stays mapped like every other
    // post-dlopen failure (close_all never dlcloses).
    if (d->required_services)
        for (const char* const* s = d->required_services; *s; ++s)
            if (!caps_.services.count(*s)) {
                a.status = AppletStatus::Refused;
                a.status_text = missing_service_reason(*s);
                return false;
            }
    return true;
}

bool AppletLoader::launch(int idx, CaliperHost host_proto) {
    if (idx < 0 || idx >= count()) return false;
    AppletEntry& a = entries_[idx];
    if (a.status == AppletStatus::Active) teardown(idx);
    if (a.status != AppletStatus::Ready) return false;
    if (!open_and_check(a)) return false;
    const auto* d = a.desc;

    auto fail = [&](std::string why) {
        a.status = AppletStatus::Failed;
        a.status_text = std::move(why);
        return false;
    };

    std::error_code ec;
    fs::create_directories(a.data_dir, ec);
    // The stamp is raised before any applet code can touch the folder: the
    // applet migrates its data in initialize(), so data that a failed
    // initialize() left half-migrated is still marked.
    if (!raise_data_schema_stamp(a.data_dir, a.manifest.data_schema))
        return fail("could not record the data schema in " +
                    (fs::path(a.data_dir) / kDataStampName).string());
    host_blocks_[idx] = host_proto;
    host_blocks_[idx].applet_data_dir = a.data_dir.c_str();

    // Jobs submitted from here on (initialize() may start one) are this
    // instance's: a fault in one quarantines it (quarantine_if_job_faulted).
    a.jobs = std::make_shared<JobOwner>();
    JobOwnerScope jobs_scope(a.jobs);

    void* instance = nullptr;
    auto cr = guarded_call([&] { instance = d->api.create(); });
    if (!cr.ok) {
        quarantine(a, "crashed in create(): " + cr.fault, cr.kind, nullptr, false);
        return false;
    }
    if (!instance) return fail("create() returned null");

    bool init_ok = false;
    auto ir = guarded_call([&] {
        init_ok = d->api.initialize(instance, &host_blocks_[idx]);
    });
    if (!ir.ok) {
        quarantine(a, "crashed in initialize(): " + ir.fault, ir.kind, instance,
                   false);
        return false;
    }
    if (!init_ok) {
        guarded_call([&] { d->api.destroy(instance); });
        return fail("initialize() returned false");
    }

    a.instance = instance;
    a.status = AppletStatus::Active;
    a.status_text.clear();
    return true;
}

bool AppletLoader::probe(int idx) {
    if (idx < 0 || idx >= count()) return false;
    AppletEntry& a = entries_[idx];
    if (a.status != AppletStatus::Ready && a.status != AppletStatus::Shadowed)
        return false;
    if (!open_and_check(a)) return false;
    const auto* d = a.desc;
    a.jobs = std::make_shared<JobOwner>();
    JobOwnerScope jobs_scope(a.jobs);
    void* instance = nullptr;
    auto cr = guarded_call([&] { instance = d->api.create(); });
    if (!cr.ok) {
        quarantine(a, "crashed in create(): " + cr.fault, cr.kind, nullptr, false);
        return false;
    }
    if (!instance) {
        a.status = AppletStatus::Failed;
        a.status_text = "create() returned null";
        return false;
    }
    auto de = guarded_call([&] { d->api.destroy(instance); });
    if (!de.ok) {
        quarantine(a, "crashed during teardown: " + de.fault, de.kind, nullptr, false);
        return false;
    }
    return true;
}

void AppletLoader::quarantine(AppletEntry& a, const std::string& why,
                              FaultKind kind, void* instance, bool initialized) {
    a.status = AppletStatus::Quarantined;
    a.status_text = why;
    a.fault_kind = kind;
    a.instance = nullptr;   // never called again, whatever the kind
    const bool keep = kind == FaultKind::Exception && instance != nullptr;
    a.stale_instance = keep ? instance : nullptr;
    a.stale_initialized = keep && initialized;
}

// A guarded job the instance submitted faulted on its worker. The job ran its
// code on the instance's state, so the instance is quarantined exactly as for
// the same fault in frame(): a signal abandons it (never called again, no
// cleanup/destroy, §15), an exception keeps it for restart(). Checked here, on
// the loader's thread, before each call into the instance; the worker only
// flagged it (JobOwner). The job's GuardResult stayed on its worker and only
// its text reached the owner, so the kind is read back from the text
// (fault_kind_of, which errs toward Signal).
bool AppletLoader::quarantine_if_job_faulted(AppletEntry& a) {
    if (!a.jobs || !a.jobs->faulted()) return false;
    const std::string fault = a.jobs->fault();
    quarantine(a, "crashed in " + fault, fault_kind_of(fault), a.instance, true);
    return true;
}

bool AppletLoader::frame(int idx, const CaliperFrameInfo& info) {
    if (idx < 0 || idx >= count()) return false;
    AppletEntry& a = entries_[idx];
    if (a.status != AppletStatus::Active || !a.instance) return false;
    if (quarantine_if_job_faulted(a)) return false;

    JobOwnerScope jobs_scope(a.jobs);
    auto r = guarded_call([&] { a.desc->api.frame(a.instance, &info); });
    if (!r.ok) {
        // Memory is suspect after a signal: abandon the instance, never call
        // cleanup/destroy/dlclose on it (§15 honesty). An exception unwound
        // cleanly, so its instance is kept, uncalled, for restart().
        quarantine(a, "crashed in frame(): " + r.fault, r.kind, a.instance, true);
        return false;
    }
    return true;
}

void AppletLoader::teardown(int idx) {
    if (idx < 0 || idx >= count()) return;
    AppletEntry& a = entries_[idx];
    if (a.status != AppletStatus::Active || !a.instance) return;
    if (quarantine_if_job_faulted(a)) return;

    JobOwnerScope jobs_scope(a.jobs);
    void* const instance = a.instance;
    auto cl = guarded_call([&] { a.desc->api.cleanup(instance); });
    if (!cl.ok) {
        // destroy() is still owed after a throw from cleanup(): restart()
        // pays it. A signal abandons the instance as usual.
        quarantine(a, "crashed during teardown: " + cl.fault, cl.kind, instance,
                   false);
        return;
    }
    auto de = guarded_call([&] { a.desc->api.destroy(instance); });
    a.instance = nullptr;
    if (!de.ok) {
        quarantine(a, "crashed during teardown: " + de.fault, de.kind, nullptr,
                   false);
        return;
    }
    a.status = AppletStatus::Ready;
}

bool AppletLoader::restartable(int idx) const {
    if (idx < 0 || idx >= count()) return false;
    const AppletEntry& a = entries_[idx];
    return a.status == AppletStatus::Quarantined &&
           a.fault_kind == FaultKind::Exception;
}

bool AppletLoader::restart(int idx, CaliperHost host_proto) {
    if (!restartable(idx)) return false;
    AppletEntry& a = entries_[idx];

    // Unload: release the old instance as teardown() would have, had the
    // exception not stopped it — cleanup() only if initialize() had
    // succeeded (as launch() does for an initialize() that returns false),
    // then destroy(), both guarded. Its jobs were joined when the host went
    // Home after the quarantine (main.cpp).
    if (void* const old = a.stale_instance) {
        const bool initialized = a.stale_initialized;
        a.stale_instance = nullptr;
        a.stale_initialized = false;
        JobOwnerScope jobs_scope(a.jobs);
        GuardResult r;
        bool cleaned = !initialized;
        if (initialized) {
            r = guarded_call([&] { a.desc->api.cleanup(old); });
            cleaned = r.ok;
        }
        if (cleaned) r = guarded_call([&] { a.desc->api.destroy(old); });
        if (!r.ok) {
            // Quarantined again with the new reason. After a throw from
            // cleanup() the old instance still awaits destroy(), so a later
            // restart() tries that; after destroy() nothing is left.
            quarantine(a, "crashed while restarting: " + r.fault, r.kind,
                       cleaned ? nullptr : old, false);
            return false;
        }
    }

    a.status = AppletStatus::Ready;
    a.status_text.clear();
    a.fault_kind = FaultKind::None;
    return launch(idx, host_proto);
}

void AppletLoader::close_all() {
    for (int i = 0; i < count(); i++) teardown(i);
    for (auto& a : entries_) {
        // ALL dylibs are deliberately left mapped at shutdown — never
        // dlclosed. close_all() only runs when the process is exiting, where
        // unloading buys nothing (the OS reclaims every page anyway) and
        // risks everything: dlclose runs the image's static destructors, and
        // heavyweight applet deps (libtorch above all) are not unload-safe —
        // their global thread pools and allocator state can abort in malloc
        // at teardown. Same policy every mature plugin host converges on.
        // (Quarantined images were already exempt: running destructors in a
        // corrupted image is worse still. A stale instance kept for restart()
        // is left with its image.)
        a.handle = nullptr;
        a.desc = nullptr;
    }
    entries_.clear();
    host_blocks_.clear();
}

} // namespace caliper_host
