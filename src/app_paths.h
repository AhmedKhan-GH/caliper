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
// ============================================================================

namespace caliper {

// Absolute path to the Caliper app-data directory. Always trailing-slash free.
// The directory is created if missing.
const std::string& app_data_dir();

// Convenience: returns app_data_dir() / filename.
std::string app_data_path(const std::string& filename);

// Override the app-data root (embed v1.1: CaliperCoreDesc.data_dir routing).
// While set, app_data_dir()/app_data_path() resolve UNDER `dir` (created if
// missing) instead of the OS default, so an embedder's metrics/artifacts/
// applet-data land under its own per-project root. Pass "" to restore the OS
// default. NOT used by the caliper exe (it never overrides), so the default
// behavior is byte-for-byte unchanged. Set once on the create thread before
// the stores open; not intended for concurrent mutation.
void set_app_data_dir_override(const std::string& dir);

} // namespace caliper
