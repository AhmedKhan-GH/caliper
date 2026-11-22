#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace caliper_host {

// What an applet is ([applet].kind): a compiled library the host opens and
// calls, or a spec applet, a folder of data (a manifest and a spec, no
// library) that the app draws as a page and never executes.
enum class AppletKind { Library, Spec };

// Parsed caliper.toml (docs/applets/manifest.md). Unknown keys/tables are ignored
// for forward compatibility; missing required fields are errors. A spec
// applet (kind = "spec") has no [compat].abi_epoch, and the keys that only a
// library can honour ([compat].abi_epoch and torch, [build], [services],
// [data]) are errors in its manifest.
struct AppletManifest {
    std::string id;        // reverse-DNS; see is_valid_applet_id
    std::string name;
    std::string version;   // strict x.y.z
    std::string summary;
    std::string tag;
    uint32_t    abi_epoch = 0;
    std::string min_host;  // "" = no floor; else strict x.y.z
    std::string torch;     // [compat].torch: exact libtorch version; "" = none
    std::vector<std::string> required_services;
    std::vector<std::string> optional_services;
    std::string sdk;       // [build].sdk: SDK version that built it; a record only
    std::string ui_stack;  // [build].ui_stack: UI-stack hash; "" = not checked
    uint32_t    data_schema = 0;  // [data].schema; 0 = the key is absent
    AppletKind  kind = AppletKind::Library;  // [applet].kind; Library when absent
    std::string spec;      // [applet].spec: a spec applet's spec file, a plain file name
};

struct ManifestResult {
    bool ok = false;
    // When !ok, only manifest.kind is filled: the kind the text declares (or
    // Library), so that a broken spec applet is still listed as one.
    AppletManifest manifest;
    std::string error;     // human-readable, shown on the failure card
};

ManifestResult parse_manifest_text(const std::string& toml_text);
ManifestResult parse_manifest_file(const std::string& path);
bool is_valid_semver(const std::string& v);
// Lowercase reverse-DNS, [a-z0-9.-] only, non-empty dot-separated segments,
// at most 128 chars. The id doubles as the applet's data folder name, so this
// is what keeps a crafted manifest ("../../x") inside the data root.
bool is_valid_applet_id(const std::string& id);
// A file name with no folder part: not empty, not "." or "..", at most 255
// bytes, and no '/', '\\', ':' or control character. applet.spec must be one,
// so a spec applet's spec is always a file directly in its folder.
bool is_plain_file_name(const std::string& name);
// "library" or "spec", as the manifest writes the kind.
const char* applet_kind_word(AppletKind kind);

} // namespace caliper_host
