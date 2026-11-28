#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace caliper_host {

// How an instrument is built ([instrument].kind): a shared library the host opens and
// calls, or a spec, a folder of data (a manifest and a spec, no
// library) that the app draws as a page and never executes.
enum class InstrumentKind { Library, Spec };

// Parsed caliper.toml (docs/instruments/manifest.md). Unknown keys/tables are ignored
// for forward compatibility; missing required fields are errors. An instrument written as a
// spec (kind = "spec") has no [compat].abi_epoch, and the keys that only a
// library can honour ([compat].abi_epoch and torch, [build], [services],
// [data]) are errors in its manifest.
struct InstrumentManifest {
    std::string id;        // reverse-DNS; see is_valid_instrument_id
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
    InstrumentKind kind = InstrumentKind::Library;  // [instrument].kind; Library when absent
    std::string spec;      // [instrument].spec: spec file of a kind = "spec" instrument, a plain file name
};

struct ManifestResult {
    bool ok = false;
    // When !ok, only manifest.kind is filled: the kind the text declares (or
    // Library), so that a broken kind = "spec" instrument is still listed with its kind.
    InstrumentManifest manifest;
    std::string error;     // human-readable, shown on the failure card
};

ManifestResult parse_manifest_text(const std::string& toml_text);
ManifestResult parse_manifest_file(const std::string& path);
bool is_valid_semver(const std::string& v);
// Lowercase reverse-DNS, [a-z0-9.-] only, non-empty dot-separated segments,
// at most 128 chars. The id doubles as the instrument's data folder name, so this
// is what keeps a crafted manifest ("../../x") inside the data root.
bool is_valid_instrument_id(const std::string& id);
// A file name with no folder part: not empty, not "." or "..", at most 255
// bytes, and no '/', '\\', ':' or control character. instrument.spec must be one,
// so a spec file is always a file directly in the instrument's folder.
bool is_plain_file_name(const std::string& name);
// "library" or "spec", as the manifest writes the kind.
const char* instrument_kind_word(InstrumentKind kind);

} // namespace caliper_host
