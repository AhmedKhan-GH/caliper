#pragma once
#include "instrument_manifest.h"
#include "crash_guard.h"
#include "negotiation.h"
#include <caliper/abi.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace caliper_host {

class JobOwner;   // job_system.h

enum class InstrumentStatus {
    Ready,        // negotiated, will dlopen on launch
    Refused,      // pre-dlopen negotiation refusal (friendly reason)
    Failed,       // broken: parse error, missing binary, bad descriptor, init false
    Active,       // instance running
    Quarantined,  // faulted; not called again unless restart()ed (§15)
    Shadowed,     // passed every pre-dlopen check, but another copy of the
                  // same id is chosen (higher version, or found first)
};

struct InstrumentEntry {
    InstrumentManifest manifest;   // manifest.kind: Library or Spec
    std::string dylib_path;        // "" when binary missing; always "" for kind = "spec"
    // Spec file of a kind = "spec" instrument, <folder>/<manifest.spec>, once that
    // file passed the listing checks (spec_listing_checks); "" otherwise and for
    // a library instrument. The app reads it with read_spec_file (instrument_spec.h).
    std::string spec_path;
    std::string data_dir;      // per-instrument sandbox; storage for the ABI pointer ("" for kind = "spec")
    std::string folder;        // the folder the manifest was found in
    std::string manifest_path; // <folder>/<stem>.caliper.toml
    int search_rank = 0;       // that folder's position in the search order
    InstrumentStatus status = InstrumentStatus::Failed;
    // Card text for Refused/Failed/Quarantined/Shadowed; on a kind = "spec" instrument
    // that stays Ready after launch(), why no page opened (kSpecPageUnavailable).
    std::string status_text;

    void* handle = nullptr;
    const CaliperInstrumentDescriptor* desc = nullptr;
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

// The data-schema stamp (docs/development/compatibility-and-epochs.md): a small TOML
// file the host keeps in each instrument's data folder, recording the highest
// [data].schema that has opened the folder. The instrument never writes it.
inline constexpr const char* kDataStampName = ".caliper-data.toml";
// The schema recorded in <data_dir>/.caliper-data.toml; 0 when there is no
// stamp or it cannot be read (a damaged stamp is ignored and rewritten by the
// next launch that declares a schema).
uint32_t read_data_schema_stamp(const std::string& data_dir);
// Records `schema` when it is higher than the recorded one, through a
// temporary file renamed over the stamp, so a reader never sees half a file.
// True when the stamp now holds at least `schema`.
bool raise_data_schema_stamp(const std::string& data_dir, uint32_t schema);

// The folders the host searches for instruments, in order
// (docs/development/compatibility-and-epochs.md, "Several copies of one instrument"): the
// development paths, first those given on the command line and then those in
// CALIPER_INSTRUMENT_PATH, each list in its own order; then the open workspace's
// instruments/ folder, when a workspace is given; then the user instruments folder;
// then the instruments shipped with the app. Relative paths are made absolute
// against the current directory, each development folder is resolved by
// resolve_instrument_folder(), empty entries are dropped, and a folder listed
// twice is searched once, at its first position. A loader that scans the
// workspace's folder must be told the workspace (InstrumentLoader::set_workspace).
std::vector<std::string> instrument_search_dirs(
    const std::vector<std::string>& command_line_dirs,
    const char* instrument_path_env,         // CALIPER_INSTRUMENT_PATH, or nullptr
    const std::string& user_dir,             // "" to leave out
    const std::string& shipped_dir,          // "" to leave out
    const std::string& workspace_dir = "");  // the workspace's root; "" for none
// The folder of a workspace that holds its spec-based instruments: <workspace>/instruments
// (design/builder-session.md); "" for "".
std::string workspace_instruments_dir(const std::string& workspace_dir);
// The folder the host scans for a development folder `dir`: `dir` itself
// when it holds a manifest (<stem>.caliper.toml), otherwise its instruments/
// subfolder when that holds one, which is where caliper_add_instrument() puts
// an instrument's build output, so an instrument's build folder can be named
// directly. Otherwise, a folder that does not exist included, `dir` as given.
std::string resolve_instrument_folder(const std::string& dir);
// CALIPER_INSTRUMENT_PATH split into folders: ';' separates them on Windows and
// ':' elsewhere. Empty entries are dropped.
std::vector<std::string> split_instrument_path(const std::string& value);
inline constexpr char kInstrumentPathSeparator =
#ifdef _WIN32
    ';';
#else
    ':';
#endif

// The status in capitals, as the landing screen and the command line write it.
const char* instrument_status_word(InstrumentStatus s);

// Card text of a copy that another copy of the same id shadows.
std::string shadowed_reason(const std::string& winner_version,
                            const std::string& winner_folder);

// ---- spec-based instruments -----------------------------------------------
// An instrument written as a spec (kind = "spec") is a folder with a manifest
// and a spec and no library: data, not code. The loader lists it like a library
// instrument, with the same statuses and the same choice among copies of one id,
// but never maps or calls anything for it, so the crash guard and the watchdog
// have nothing to guard. Listing checks only what is cheap (spec_listing_checks);
// check_spec() validates the spec fully with the builder's spec library,
// which a build without CALIPER_WITH_BUILDER does not have: there every
// kind = "spec" instrument is Refused with no_spec_runtime_reason().

// Where kind = "spec" instruments are looked for, besides a manifest directly in a
// searched folder: a manifest of kind "spec" in a subfolder of a searched
// folder (<dir>/<name>/<stem>.caliper.toml), as a workspace keeps them. A
// subfolder that is a symbolic link or whose name starts with '.' is skipped,
// and at most this many subfolders of one folder, and this many manifests in
// them, are looked at.
inline constexpr std::size_t kMaxSpecSubfolders = 4096;
// The largest manifest read from a subfolder or from the workspace's folder.
inline constexpr std::uintmax_t kMaxSpecManifestBytes = 64 * 1024;
// The largest spec file: the spec format's limit of 512 KiB (spec.too_large).
inline constexpr std::uintmax_t kMaxSpecFileBytes = 512 * 1024;

// True when this build validates kind = "spec" instruments (it was built with the
// builder's spec library); false when every kind = "spec" instrument is Refused.
bool spec_runtime_available();
// The oldest and newest spec format versions this build reads (caliper_spec);
// -1 without the spec runtime.
int64_t spec_min_format_version();
int64_t spec_format_version();

// Refusal and failure texts of kind = "spec" instruments (docs/reference/refusal-messages.md).
std::string no_spec_runtime_reason();
std::string spec_version_reason(int64_t spec_version, int64_t host_version);
std::string workspace_library_reason();
// The text a Ready kind = "spec" instrument keeps after launch(): instruments
// written as specs open as a page, which the loader does not draw. The app opens
// the page itself.
inline constexpr const char* kSpecPageUnavailable =
    "An instrument written as a spec opens as a page, which this version of Caliper does not draw yet.";

// Why the file at `spec_path` cannot be an instrument's spec, or "" when it
// can: it exists, it is a regular file and not a symbolic link (wherever it
// points), and it holds at most kMaxSpecFileBytes. `manifest_file` names the
// manifest in the text.
std::string spec_file_problem(const std::string& spec_path, const std::string& manifest_file);

// The caliper_spec version of a spec text, read from its first lines without
// parsing the TOML: the line `caliper_spec = <integer>` before the first
// table header. False with `why` when there is no such line or its value is
// not a decimal integer.
bool read_spec_version_line(std::istream& in, int64_t& version, std::string& why);

// One diagnostic of the spec library, in a form that needs no builder header.
struct SpecDiagnostic {
    bool error = true;     // false: a warning
    std::string code;      // e.g. spec.unknown_source
    std::string message;   // one plain sentence
    std::string path;      // e.g. panel[0].bind.y; "" for the whole document
    int line = 0;          // 1-based, in the spec file; 0 when not known
    int column = 0;
};
// "<file>:<line>:<column>: <code>: <message>", with "error " or "warning "
// before the code when `severity`; without a line, "<file>: ..." and the
// document path after the message.
std::string format_spec_diagnostic(const std::string& file, const SpecDiagnostic& d,
                                   bool severity);

// Manifest-first loader (docs/development/compatibility-and-epochs.md): scan() never dlopens; launch()
// performs dlopen -> descriptor sanity -> guarded create/initialize.
//
// Several copies of one id: after every scan, the copy that stays Ready is
// the highest version that passed every pre-dlopen check, and among equal
// versions the one found first in the search order. The other copies that
// passed are Shadowed, with a text naming the chosen copy's version and
// folder; refused and failed copies keep their own status and reason.
// Choosing never runs instrument code. A chosen copy that later fails at launch
// is not replaced by the next one until the host scans again.
class InstrumentLoader {
public:
    InstrumentLoader(HostCaps caps, std::string data_root);
    ~InstrumentLoader() { close_all(); }

    // The open workspace, before scanning (design/builder-session.md). Its
    // instruments/ folder (workspace_instruments_dir) is scanned as a search folder
    // like any other, at the position instrument_search_dirs gives it, but holds
    // spec-based instruments only: a workspace is data, so a library instrument's
    // manifest there is listed as Refused (workspace_library_reason) and its library
    // is never opened, and every manifest in it is read with the bounds of a
    // subfolder's. "" for no workspace.
    void set_workspace(const std::string& workspace_dir);
    const std::string& workspace_dir() const { return workspace_dir_; }

    // Adds the instruments in `dir` at the next position of the search order.
    // Returns the entries added.
    int scan(const std::string& dir);
    // Scans each folder in order (instrument_search_dirs gives the host's order).
    int scan(const std::vector<std::string>& dirs);
    // Lists the workspace's instruments/ folder again, so that a spec-based instrument
    // saved since the last scan (instrument.save) appears and one deleted goes: the
    // entries found there are replaced by what the folder holds now, at the
    // folder's position in the search order, or at the next position when
    // it was not scanned before. Entries move, so indices taken before are
    // stale. Like scan(), it reallocates what a running library instrument
    // holds, so it does nothing and returns -1 while one is Active; a kind = "spec"
    // instrument marked Active by the app is listed again as Ready. Returns the
    // entries listed from the folder.
    int rescan_workspace();
    int count() const { return (int)entries_.size(); }
    // The entry to launch for `id`: the chosen copy when one is Ready (or
    // running), otherwise the first copy that is not Shadowed; -1 when the
    // id is unknown.
    int find(const std::string& id) const;
    InstrumentEntry&       at(int i)       { return entries_[i]; }
    const InstrumentEntry& at(int i) const { return entries_[i]; }
    // The entry was found in the open workspace's instruments/ folder: one of the
    // person's dashboards when it is a kind = "spec" instrument.
    bool in_workspace(int i) const {
        return workspace_rank_ >= 0 && i >= 0 && i < (int)entries_.size() && entries_[i].search_rank == workspace_rank_;
    }

    // host_proto: filled CaliperHost except instrument_data_dir, which the loader
    // points at this entry's sandbox dir before initialize().
    // A kind = "spec" instrument is never loaded: launch() validates its spec
    // (check_spec) and returns false, leaving it Ready with kSpecPageUnavailable
    // as its text when the spec is valid, or Failed with the first error.
    bool launch(int idx, CaliperHost host_proto);
    // Full validation of a kind = "spec" instrument that passed the listing checks
    // (Ready or Shadowed): its spec is read again, parsed and validated with the
    // spec library (read_spec_file in instrument_spec.h). On an error the entry
    // becomes Failed with the first error, formatted by
    // format_spec_diagnostic, as its text. `diagnostics` receives every error
    // and warning, located in the file. False, changing nothing, for a
    // library instrument, an entry that did not pass, or a build without the
    // spec runtime.
    bool check_spec(int idx, std::vector<SpecDiagnostic>* diagnostics = nullptr);
    // The headless acceptance check of a Ready or Shadowed entry: dlopen,
    // descriptor sanity, the binary's required services, then a guarded
    // create() and destroy(); initialize() never runs, so the data folder and
    // its stamp are untouched. True when all of it passed. On failure the
    // entry takes the status and text that launch() would give it. For a
    // kind = "spec" instrument, check_spec().
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
    // Adds the entry of one manifest found in a searched folder at `rank`.
    // `in_subfolder`: found in a subfolder, where only kind = "spec" instruments
    // are taken; `in_workspace`: found under the workspace's instruments/ folder.
    // Returns the entries added (0 or 1).
    int add_manifest(const std::filesystem::path& manifest, int rank, bool in_subfolder,
                     bool in_workspace);
    // The checks of a kind = "spec" instrument made at listing time, on an entry
    // whose manifest parsed: the spec file, then min_host, the spec runtime and
    // the spec's format version. Sets status, status_text and spec_path.
    void spec_listing_checks(InstrumentEntry& a) const;
    bool is_workspace_folder(const std::string& dir) const;
    void resolve_duplicates();
    // dlopen (once) and the checks between it and create(); false with the
    // entry Failed or Refused and its reason.
    bool open_and_check(InstrumentEntry& a);
    bool quarantine_if_job_faulted(InstrumentEntry& a);
    // `instance` faulted in `where`: Quarantined with the guard's reason and
    // kind. An Exception keeps the instance for restart(); a Signal abandons
    // it. `initialized`: initialize() had returned true for it.
    static void quarantine(InstrumentEntry& a, const std::string& why, FaultKind kind,
                           void* instance, bool initialized);

    HostCaps caps_;
    std::string data_root_;
    std::string workspace_dir_;           // set_workspace(); "" for none
    std::string workspace_key_;           // its instruments/ folder, made comparable
    std::vector<InstrumentEntry> entries_;
    std::vector<CaliperHost> host_blocks_;  // stable storage per entry
    int next_rank_ = 0;                      // search position of the next scan
    int workspace_rank_ = -1;                // the workspace folder's position, once scanned
    int scan_at(const std::string& dir, int rank);
};

} // namespace caliper_host
