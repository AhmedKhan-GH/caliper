#pragma once
// The files of a saved instrument written as a spec (design/spec-format.md,
// The manifest of an instrument): instruments/<id>/<id>.caliper.toml and
// <id>.spec.toml. Not a public header.
#include <caliper_builder/error.h>

#include <cstddef>
#include <optional>
#include <string>

namespace caliper_builder::session_detail {

constexpr std::size_t kMaxManifestBytes = 64 * 1024;

struct ManifestInfo {
    Error error;
    std::string id;       // instrument.id
    std::string name;     // instrument.name
    std::string version;  // instrument.version
    std::string kind;     // instrument.kind; "library" when absent
    std::string spec;     // instrument.spec
};

// Reads the [instrument] table of a manifest. Other tables are not checked here.
ManifestInfo read_manifest(const std::string& path);

// The manifest of an instrument written as a spec, in the key order of the spec format page.
std::string manifest_text(const std::string& manifest_id, const std::string& name, const std::string& version,
                          const std::optional<std::string>& summary, const std::string& spec_file);

// "local.<id>" with each `_` replaced by `-`, as the manifest's grammar needs.
std::string manifest_id_for(const std::string& workspace_id);

// A file name only: no `/`, `\`, `..` segment or control character.
bool plain_file_name(const std::string& name);

// A TOML basic string, quoted and escaped.
std::string toml_quote(const std::string& text);

}  // namespace caliper_builder::session_detail
