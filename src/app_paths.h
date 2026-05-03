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

} // namespace caliper
