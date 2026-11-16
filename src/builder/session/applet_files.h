#pragma once
// The files of a saved spec applet (docs/plans/spec-format.md, The manifest
// of a spec applet): applets/<id>/<id>.caliper.toml and <id>.spec.toml. Not a
// public header.
#include <caliper_builder/error.h>

#include <cstddef>
#include <optional>
#include <string>

namespace caliper_builder::session_detail {

constexpr std::size_t kMaxManifestBytes = 64 * 1024;

struct ManifestInfo {
    Error error;
    std::string id;       // applet.id
    std::string name;     // applet.name
    std::string version;  // applet.version
    std::string kind;     // applet.kind; "library" when absent
    std::string spec;     // applet.spec
};

// Reads the [applet] table of a manifest. Other tables are not checked here.
ManifestInfo read_manifest(const std::string& path);

// The manifest of a spec applet, in the key order of the spec format page.
std::string manifest_text(const std::string& manifest_id, const std::string& name, const std::string& version,
                          const std::optional<std::string>& summary, const std::string& spec_file);

// "local.<id>" with each `_` replaced by `-`, as the manifest's grammar needs.
std::string manifest_id_for(const std::string& workspace_id);

// A file name only: no `/`, `\`, `..` segment or control character.
bool plain_file_name(const std::string& name);

// A TOML basic string, quoted and escaped.
std::string toml_quote(const std::string& text);

}  // namespace caliper_builder::session_detail
