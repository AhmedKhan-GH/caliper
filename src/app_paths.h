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

// The folder that holds the running executable, "" when the OS cannot say.
std::string executable_dir();

// The folder that holds the applets shipped with the app: applets/ beside
// the executable; "" when the executable's folder is unknown.
std::string shipped_applets_dir();

// The user applets folder: CALIPER_USER_APPLETS_DIR when it is set and not
// empty, otherwise applets/ under the app-data folder. Creates nothing.
std::string user_applets_dir();

} // namespace caliper
