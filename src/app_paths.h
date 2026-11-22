#pragma once

#include <string>

// ============================================================================
// Cross-platform app-data directory lookup.
//
// Returns the canonical user-data directory for Caliper on the current OS,
// creating it on first use. Use this instead of the working directory for
// settings, caches, and small persistent state files.
//
//   macOS   : ~/Library/Application Support/Caliper/
//   Linux   : $XDG_DATA_HOME/caliper/  (fallback: ~/.local/share/caliper/)
//   Windows : %APPDATA%/Caliper/
//
// CALIPER_DATA_DIR, when set and not empty, replaces that folder for the whole
// process (it is read once, at the first lookup), and an embedder's
// set_app_data_dir_override() replaces both.
//
// THE GUARD. A test or hidden run never uses the person's own folder: in such
// a run, a lookup that would resolve to the OS folder above, because neither
// CALIPER_DATA_DIR nor an override names another, prints one line on standard
// error and ends the process with exit code 1 before anything is created or
// opened there. A run is a test or hidden run when
//   - app_data_guard() was called: the app with --hidden (main.cpp) and the
//     test binaries (tests/test_run_marker.cpp);
//   - CALIPER_HIDDEN is 1, or any CALIPER_TEST_* variable is set and not
//     empty: the input driver, the tests' hooks, and CALIPER_TEST_RUN=1, which
//     ctest sets for every entry (CaliperOfflineTests.cmake).
// A data folder given explicitly, even the OS folder itself, is used as given.
// A visible app, `caliper shot` (which keeps its stores in a temporary folder
// unless CALIPER_DATA_DIR is set) and an embedder's core are not guarded runs
// by themselves.
// ============================================================================

namespace caliper {

// Absolute path to the Caliper app-data directory. Always trailing-slash free.
// The directory is created if missing.
const std::string& app_data_dir();

// Convenience: returns app_data_dir() / filename.
std::string app_data_path(const std::string& filename);

// The app-data folder app_data_dir() returns, without creating it: for code
// that only reads, such as the read-only subcommands of the command line.
std::string app_data_location();

// Override the app-data root (embed v1.1: CaliperCoreDesc.data_dir routing).
// While set, app_data_dir()/app_data_path() resolve UNDER `dir` (created if
// missing) instead of the OS default, so an embedder's metrics/artifacts/
// applet-data land under its own per-project root. Pass "" to restore the
// default (CALIPER_DATA_DIR, else the OS folder). NOT used by the caliper exe
// (it never overrides). Set once on the create thread before
// the stores open; not intended for concurrent mutation.
void set_app_data_dir_override(const std::string& dir);

// Marks this process as a test or hidden run (THE GUARD above). `what` names
// it in the refusal and must outlive the process (a string literal). Call it
// before the first lookup; the hidden app calls it as it starts.
void app_data_guard(const char* what);

// Why this process is a test or hidden run, "" when it is not.
std::string app_data_guard_reason();

// The folder that holds the running executable, "" when the OS cannot say.
std::string executable_dir();

// The folder that holds the applets shipped with the app: applets/ beside
// the executable; "" when the executable's folder is unknown.
std::string shipped_applets_dir();

// The user applets folder: CALIPER_USER_APPLETS_DIR when it is set and not
// empty, otherwise applets/ under the app-data folder. Creates nothing.
std::string user_applets_dir();

// The builder's default workspace: workspace/ under the app-data folder, so
// CALIPER_DATA_DIR moves it too. --workspace and CALIPER_WORKSPACE choose
// another (host_cli.h, workspace_choice). Creates nothing.
std::string default_workspace_dir();

} // namespace caliper
