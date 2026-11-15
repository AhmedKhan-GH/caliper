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
};

struct AppletEntry {
    AppletManifest manifest;
    std::string dylib_path;    // "" when binary missing
    std::string data_dir;      // per-applet sandbox; storage for the ABI pointer
    AppletStatus status = AppletStatus::Failed;
    std::string status_text;   // card text for Refused/Failed/Quarantined

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

// Manifest-first loader (docs/explanations/compatibility.md): scan() never dlopens; launch()
// performs dlopen -> descriptor sanity -> guarded create/initialize.
class AppletLoader {
public:
    AppletLoader(HostCaps caps, std::string data_root);
    ~AppletLoader() { close_all(); }

    int scan(const std::string& dir);            // returns entries added
    int count() const { return (int)entries_.size(); }
    AppletEntry&       at(int i)       { return entries_[i]; }
    const AppletEntry& at(int i) const { return entries_[i]; }

    // host_proto: filled CaliperHost except applet_data_dir, which the loader
    // points at this entry's sandbox dir before initialize().
    bool launch(int idx, CaliperHost host_proto);
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
};

} // namespace caliper_host
