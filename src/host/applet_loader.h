#pragma once
#include "applet_manifest.h"
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
    Quarantined,  // faulted; never called again this session (§15)
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
};

// Manifest-first loader (PLATFORM.md §14): scan() never dlopens; launch()
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

private:
    bool quarantine_if_job_faulted(AppletEntry& a);

    HostCaps caps_;
    std::string data_root_;
    std::vector<AppletEntry> entries_;
    std::vector<CaliperHost> host_blocks_;  // stable storage per entry
};

} // namespace caliper_host
