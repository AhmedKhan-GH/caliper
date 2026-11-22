#pragma once
// Validation of a spec (design/spec-format.md#validation).
//
// validate() checks a Spec however it was made: ids, lengths, counts and
// ranges; the references between sources, panels and the layout; cycles in
// `from`; the slots of each primitive; and, when the optional SourceResolver
// knows them, the columns, dtypes and shapes of the catalog sources, which it
// propagates through `from` and the tensor transforms. It reports errors and
// warnings with stable codes and document paths, and what it learned of each
// source. A spec is valid when it parses and validate() reports no error.
//
// The resolver is a callback, so the spec module does not depend on the catalog
// module. It may be called from validate() once per catalog source; an
// exception it throws is caught and reported as spec.internal. Nothing here
// throws.
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <caliper_builder/spec/format.h>
#include <caliper_builder/spec/spec.h>

namespace caliper_builder::spec {

struct ResolvedColumn {
    std::string name;
    std::string type;  // a DuckDB type name, such as "BIGINT" or "VARCHAR"
};

// What the catalog knows of one catalog id.
struct ResolvedSource {
    std::string kind;  // "table", "tensor", "live", "feed" or "metric"
    std::optional<std::vector<ResolvedColumn>> columns;  // tabular kinds
    std::string dtype;  // tensor kinds: u8 i8 u16 i16 u32 i32 u64 i64 f16 f32 f64 bool; empty if not known
    std::optional<std::vector<std::int64_t>> shape;  // tensor kinds; -1 for a size that is not fixed
    bool wire = false;  // the source has a wire form, so it can run in the browser
};

// Returns what the catalog knows of an id, or nothing when it does not know it.
using SourceResolver = std::function<std::optional<ResolvedSource>(const std::string& catalog_id)>;

enum class DataKind { unknown, tabular, tensor };

// What validation learned of one source of the spec, after its transforms.
struct SourceInfo {
    std::string id;
    DataKind kind = DataKind::unknown;
    std::optional<std::vector<ResolvedColumn>> columns;  // when known
    std::string dtype;                                   // empty when not known
    std::optional<std::vector<std::int64_t>> shape;      // when the rank is known; -1 for unknown sizes
    std::optional<bool> wire;                            // of its catalog source, when known
};

struct ValidationReport {
    std::vector<Diagnostic> diagnostics;  // errors and warnings, at most limits::kDiagnostics
    std::vector<SourceInfo> sources;      // one per source of the spec, in order
    bool ok() const { return !has_errors(diagnostics); }
};

ValidationReport validate(const Spec& spec, const SourceResolver& resolver = {});

// Parses a TOML text and, when it parses, validates it, with every diagnostic
// located in the text.
ValidationReport check_toml(std::string_view text, const SourceResolver& resolver = {});

const char* to_string(DataKind kind);
nlohmann::json to_json(const SourceInfo& info);

}  // namespace caliper_builder::spec
