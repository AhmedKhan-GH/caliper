#include "instrument_spec.h"
#include <caliper_builder/spec/format.h>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace fs = std::filesystem;
namespace spec = caliper_builder::spec;

namespace caliper_host {

static_assert(kMaxSpecFileBytes == spec::limits::kTextBytes,
              "the loader reads at most what the spec library parses");

int64_t spec_runtime_min_format_version() { return spec::kFormatVersion; }
int64_t spec_runtime_format_version() { return spec::kLatestFormatVersion; }

namespace {
SpecDiagnostic to_host(const spec::Diagnostic& d) {
    SpecDiagnostic out;
    out.error = d.is_error();
    out.code = d.code;
    out.message = d.message;
    out.path = d.path;
    out.line = d.line;
    out.column = d.column;
    return out;
}
} // namespace

SpecFile read_spec_file(const std::string& spec_path, const spec::SourceResolver& resolver) {
    SpecFile out;
    const fs::path p(spec_path);
    const std::string name = p.filename().string();
    // The file may have changed since the listing: the same checks again.
    if (std::string why = spec_file_problem(spec_path, "its manifest"); !why.empty()) {
        out.reason = why;
        return out;
    }
    std::ifstream in(p, std::ios::binary);
    std::string text;
    if (in) {
        text.resize(kMaxSpecFileBytes + 1);
        in.read(&text[0], (std::streamsize)text.size());
        text.resize((size_t)in.gcount());
    }
    if (!in && !in.eof()) {
        out.reason = "spec file " + name + " could not be read";
        return out;
    }
    if (text.size() > kMaxSpecFileBytes) {
        out.reason = "spec file " + name + " is larger than 512 KiB";
        return out;
    }

    spec::ParseResult parsed = spec::parse_toml(text);
    std::vector<spec::Diagnostic> found;
    if (parsed.ok()) {
        spec::ValidationReport report = spec::validate(parsed.spec, resolver);
        spec::locate(report.diagnostics, parsed.positions);
        found = std::move(report.diagnostics);
        out.sources = std::move(report.sources);
    } else {
        found = std::move(parsed.diagnostics);
    }
    for (const spec::Diagnostic& d : found) {
        out.diagnostics.push_back(to_host(d));
        if (out.reason.empty() && d.is_error())
            out.reason = format_spec_diagnostic(name, out.diagnostics.back(), false);
    }
    out.ok = out.reason.empty();
    if (out.ok) out.spec = std::move(parsed.spec);
    return out;
}

} // namespace caliper_host
