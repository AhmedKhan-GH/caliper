#pragma once
#include "applet_manifest.h"
#include "crash_guard.h"
#include "negotiation.h"
#include <caliper/abi.h>
#include <memory>
#include <string>
#include <vector>

namespace caliper_host {

class JobOwner;   // job_system.h

enum class AppletStatus {
    Ready,        // negotiated, will dlopen on launch
    Refused,      // pre-dlopen negotiation refusal (friendly reason)
    Failed,       // broken: parse error, missing binary, bad descriptor, init false
    Active,       // instance running
    Quarantined,  // faulted; not called again unless restart()ed (§15)
    Shadowed,     // passed every pre-dlopen check, but another copy of the
                  // same id is chosen (higher version, or found first)
};

struct AppletEntry {
    AppletManifest manifest;
    std::string dylib_path;    // "" when binary missing
    std::string data_dir;      // per-applet sandbox; storage for the ABI pointer
    std::string folder;        // the folder the manifest was found in
    std::string manifest_path; // <folder>/<stem>.caliper.toml
    int search_rank = 0;       // that folder's position in the search order
    AppletStatus status = AppletStatus::Failed;
    std::string status_text;   // card text for Refused/Failed/Quarantined/Shadowed

    void* handle = nullptr;
    const CaliperAppletDescriptor* desc = nullptr;
    void* instance = nullptr;
    // Owner of the guarded jobs this instance submits (fresh per launch): a
    // fault in one of them quarantines the instance like a fault in frame().
    std::shared_ptr<JobOwner> jobs;

    // How a Quarantined entry faulted (None otherwise). Exception: restart()
    // may unload and relaunch it in this process. Signal: its memory may be
    // corrupt, so it stays quarantined until the host restarts.
    FaultKind fault_kind = FaultKind::None;
    // After an Exception quarantine the instance is still a sound object (the
    // stack unwound and its destructors ran), so it is kept here, never
    // called, for restart() to clean up and destroy before making a new one.
    // Null after a Signal (abandoned: no cleanup/destroy, §15) or when the
    // fault left nothing to release.
    void* stale_instance = nullptr;
    bool stale_initialized = false;   // initialize() had succeeded: cleanup first
};

// The data-schema stamp (docs/explanations/compatibility.md): a small TOML
// file the host keeps in each applet's data folder, recording the highest
// [data].schema that has opened the folder. The applet never writes it.
inline constexpr const char* kDataStampName = ".caliper-data.toml";
// The schema recorded in <data_dir>/.caliper-data.toml; 0 when there is no
// stamp or it cannot be read (a damaged stamp is ignored and rewritten by the
// next launch that declares a schema).
uint32_t read_data_schema_stamp(const std::string& data_dir);
// Records `schema` when it is higher than the recorded one, through a
// temporary file renamed over the stamp, so a reader never sees half a file.
// True when the stamp now holds at least `schema`.
bool raise_data_schema_stamp(const std::string& data_dir, uint32_t schema);

// The folders the host searches for applets, in order
// (docs/explanations/compatibility.md, "Several copies of one applet"): the
// development paths, first those given on the command line and then those in
// CALIPER_APPLET_PATH, each list in its own order; then the user applets
// folder; then the applets shipped with the app. Relative paths are made
// absolute against the current directory, each development folder is
// resolved by resolve_applet_folder(), empty entries are dropped, and a
// folder listed twice is searched once, at its first position.
std::vector<std::string> applet_search_dirs(
    const std::vector<std::string>& command_line_dirs,
    const char* applet_path_env,          // CALIPER_APPLET_PATH, or nullptr
    const std::string& user_dir,          // "" to leave out
    const std::string& shipped_dir);      // "" to leave out
// The folder the host scans for a development folder `dir`: `dir` itself
// when it holds a manifest (<stem>.caliper.toml), otherwise its applets/
// subfolder when that holds one, which is where caliper_add_applet() puts
// an applet's build output, so an applet's build folder can be named
// directly. Otherwise, a folder that does not exist included, `dir` as given.
std::string resolve_applet_folder(const std::string& dir);
// CALIPER_APPLET_PATH split into folders: ';' separates them on Windows and
// ':' elsewhere. Empty entries are dropped.
std::vector<std::string> split_applet_path(const std::string& value);
inline constexpr char kAppletPathSeparator =
#ifdef _WIN32
    ';';
#else
    ':';
#endif

// The status in capitals, as the landing screen and the command line write it.
const char* applet_status_word(AppletStatus s);

// Card text of a copy that another copy of the same id shadows.
std::string shadowed_reason(const std::string& winner_version,
                            const std::string& winner_folder);

// Manifest-first loader (docs/explanations/compatibility.md): scan() never dlopens; launch()
// performs dlopen -> descriptor sanity -> guarded create/initialize.
//
// Several copies of one id: after every scan, the copy that stays Ready is
// the highest version that passed every pre-dlopen check, and among equal
// versions the one found first in the search order. The other copies that
// passed are Shadowed, with a text naming the chosen copy's version and
// folder; refused and failed copies keep their own status and reason.
// Choosing never runs applet code. A chosen copy that later fails at launch
// is not replaced by the next one until the host scans again.
class AppletLoader {
public:
    AppletLoader(HostCaps caps, std::string data_root);
    ~AppletLoader() { close_all(); }

    // Adds the applets in `dir` at the next position of the search order.
    // Returns the entries added.
    int scan(const std::string& dir);
    // Scans each folder in order (applet_search_dirs gives the host's order).
    int scan(const std::vector<std::string>& dirs);
    int count() const { return (int)entries_.size(); }
    // The entry to launch for `id`: the chosen copy when one is Ready (or
    // running), otherwise the first copy that is not Shadowed; -1 when the
    // id is unknown.
    int find(const std::string& id) const;
    AppletEntry&       at(int i)       { return entries_[i]; }
    const AppletEntry& at(int i) const { return entries_[i]; }

    // host_proto: filled CaliperHost except applet_data_dir, which the loader
    // points at this entry's sandbox dir before initialize().
    bool launch(int idx, CaliperHost host_proto);
    // The headless acceptance check of a Ready or Shadowed entry: dlopen,
    // descriptor sanity, the binary's required services, then a guarded
    // create() and destroy(); initialize() never runs, so the data folder and
    // its stamp are untouched. True when all of it passed. On failure the
    // entry takes the status and text that launch() would give it.
    bool probe(int idx);
    bool frame(int idx, const CaliperFrameInfo& info);  // false => just quarantined
    void teardown(int idx);
    void close_all();

    // A Quarantined entry whose fault was a C++ exception (fault_kind). Never
    // one that faulted by a signal: a fresh instance in a process whose
    // memory the old one may have corrupted would only inherit the damage.
    bool restartable(int idx) const;
    // Unloads a restartable entry's stale instance (guarded cleanup() if
    // initialize() had succeeded, then destroy(); the image stays mapped, as
    // it always does, see close_all) and launches a fresh one. Returns false,
    // changing nothing, when the entry is not restartable; false with the
    // entry Quarantined again when releasing the old instance faults; else
    // whatever launch() returns.
    bool restart(int idx, CaliperHost host_proto);

private:
    void resolve_duplicates();
    // dlopen (once) and the checks between it and create(); false with the
    // entry Failed or Refused and its reason.
    bool open_and_check(AppletEntry& a);
    bool quarantine_if_job_faulted(AppletEntry& a);
    // `instance` faulted in `where`: Quarantined with the guard's reason and
    // kind. An Exception keeps the instance for restart(); a Signal abandons
    // it. `initialized`: initialize() had returned true for it.
    static void quarantine(AppletEntry& a, const std::string& why, FaultKind kind,
                           void* instance, bool initialized);

    HostCaps caps_;
    std::string data_root_;
    std::vector<AppletEntry> entries_;
    std::vector<CaliperHost> host_blocks_;  // stable storage per entry
    int next_rank_ = 0;                      // search position of the next scan
};

} // namespace caliper_host
