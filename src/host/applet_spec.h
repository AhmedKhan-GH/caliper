#pragma once
// The spec of a spec applet, read and checked with the builder's spec
// library (design/spec-format.md). Only a build with CALIPER_WITH_BUILDER
// has this file's definitions: applet_spec.cpp joins caliper_host_lib with
// the spec library then, and applet_loader.h is the part of the loader that
// builds without it. The app opens a spec applet's page from an entry's
// spec_path with read_spec_file().
#include "applet_loader.h"
#include <caliper_builder/spec/spec.h>
#include <caliper_builder/spec/validate.h>
#include <cstdint>
#include <string>
#include <vector>

namespace caliper_host {

struct SpecFile {
    bool ok = false;        // read, parsed and valid; warnings do not count
    // When !ok: the first error, as format_spec_diagnostic writes it without
    // the severity, or why the file could not be read. The loader shows it
    // as the entry's status text.
    std::string reason;
    caliper_builder::spec::Spec spec;          // the parsed spec, when ok
    // Every error and warning of the spec library, in its order, with the
    // line and column of each in the file when known.
    std::vector<SpecDiagnostic> diagnostics;
    // What validation learned of each source, when the spec parsed.
    std::vector<caliper_builder::spec::SourceInfo> sources;
};

// Reads the spec file at `spec_path` after the checks of the listing again
// (spec_file_problem: a regular file, not a symbolic link, at most
// kMaxSpecFileBytes), parses it with caliper_builder::spec::parse_toml and
// validates it with validate() and `resolver`, locating every diagnostic.
// The loader validates without a resolver, which checks every rule that needs
// only each source's kind.
SpecFile read_spec_file(const std::string& spec_path,
                        const caliper_builder::spec::SourceResolver& resolver = {});

// The spec format version the spec library reads (caliper_spec).
int64_t spec_runtime_format_version();

} // namespace caliper_host
