#pragma once
// Spec-based instruments on disk for the loader and command-line tests: a folder
// with a manifest of kind "spec" and its spec file, and no library
// (docs/instruments/manifest.md).
#include <filesystem>
#include <fstream>
#include <string>

namespace caliper_test {

// A valid spec: an instrument with no panels.
inline const char* kValidSpec = "caliper_spec = 2\n\n[instrument]\nid = \"demo\"\n";

// A spec that parses but fails validation: its panel binds a source the spec
// does not have, at line 9.
inline const char* kInvalidSpec =
    "caliper_spec = 2\n"
    "\n"
    "[instrument]\n"
    "id = \"broken\"\n"
    "\n"
    "[[panel]]\n"
    "id = \"curve\"\n"
    "primitive = \"series\"\n"
    "bind = { y = \"nope.loss\" }\n";

// A valid spec with a warning: a source that no panel binds, at line 6.
inline const char* kWarningSpec =
    "caliper_spec = 2\n"
    "\n"
    "[instrument]\n"
    "id = \"warned\"\n"
    "\n"
    "[[source]]\n"
    "id = \"metrics\"\n"
    "catalog = \"table:metrics\"\n";

// <dir>/<stem>.caliper.toml naming <stem>.spec.toml, and that spec holding
// `spec_text` (none when `spec_text` is null). `extra` is appended to the
// manifest's [instrument] table or after it.
inline void write_spec_instrument(const std::filesystem::path& dir, const std::string& stem,
                                  const std::string& id, const std::string& version,
                                  const char* spec_text = kValidSpec,
                                  const std::string& extra = "") {
    std::filesystem::create_directories(dir);
    std::ofstream(dir / (stem + ".caliper.toml"))
        << "[instrument]\nid = \"" << id << "\"\nname = \"Example spec\"\nversion = \"" << version
        << "\"\nkind = \"spec\"\nspec = \"" << stem << ".spec.toml\"\n" << extra;
    if (spec_text) std::ofstream(dir / (stem + ".spec.toml"), std::ios::binary) << spec_text;
}

} // namespace caliper_test
