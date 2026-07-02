#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace caliper_host {

// Parsed caliper.toml (PLATFORM.md §10.3). Unknown keys/tables are ignored
// for forward compatibility; missing required fields are errors.
struct AppletManifest {
    std::string id;        // reverse-DNS
    std::string name;
    std::string version;   // strict x.y.z
    std::string summary;
    std::string tag;
    uint32_t    abi_epoch = 0;
    std::string min_host;  // "" = no floor; else strict x.y.z
    std::vector<std::string> required_services;
    std::vector<std::string> optional_services;
};

struct ManifestResult {
    bool ok = false;
    AppletManifest manifest;
    std::string error;     // human-readable, shown on the failure card
};

ManifestResult parse_manifest_text(const std::string& toml_text);
ManifestResult parse_manifest_file(const std::string& path);
bool is_valid_semver(const std::string& v);

} // namespace caliper_host
