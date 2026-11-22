#pragma once
// Shared helpers of the spec tests: fixtures, diagnostic lookups and a
// resolver that knows the catalog sources the fixtures name.
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <caliper_builder/spec/format.h>
#include <caliper_builder/spec/spec.h>
#include <caliper_builder/spec/validate.h>

namespace spec_test {

namespace cs = caliper_builder::spec;

inline std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

inline std::string fixture(const std::string& name) { return read_file(std::string(CALIPER_SPEC_FIXTURES) + "/" + name); }

inline const std::vector<std::string>& valid_fixtures() {
    static const std::vector<std::string> v = {"attention-entropy.spec.toml", "dashboard.spec.toml",
                                               "every-primitive.spec.toml", "minimal.spec.toml"};
    return v;
}

inline std::set<std::string> codes(const std::vector<cs::Diagnostic>& ds) {
    std::set<std::string> s;
    for (const auto& d : ds) s.insert(d.code);
    return s;
}

inline const cs::Diagnostic* find(const std::vector<cs::Diagnostic>& ds, const std::string& code) {
    for (const auto& d : ds)
        if (d.code == code) return &d;
    return nullptr;
}

inline bool has(const std::vector<cs::Diagnostic>& ds, const std::string& code, const std::string& path = "") {
    for (const auto& d : ds)
        if (d.code == code && (path.empty() || d.path == path)) return true;
    return false;
}

inline std::string dump(const std::vector<cs::Diagnostic>& ds) {
    std::string s;
    for (const auto& d : ds) s += d.code + " at " + d.path + " (" + std::to_string(d.line) + ":" +
                                  std::to_string(d.column) + "): " + d.message + "\n";
    return s;
}

// The catalog the fixtures use.
inline cs::SourceResolver fixture_resolver() {
    return [](const std::string& id) -> std::optional<cs::ResolvedSource> {
        cs::ResolvedSource r;
        if (id == "live:attn.weights") {
            r.kind = "live";
            r.dtype = "f32";
            r.shape = std::vector<std::int64_t>{12, 12, 64, 64};
            r.wire = true;
        } else if (id == "table:run42_metrics") {
            r.kind = "table";
            r.columns = std::vector<cs::ResolvedColumn>{{"step", "BIGINT"}, {"loss", "DOUBLE"}, {"accuracy", "FLOAT"},
                                                        {"note", "VARCHAR"}, {"time", "TIMESTAMP"}};
            r.wire = true;
        } else if (id == "tensor:attn_weights") {
            r.kind = "tensor";
            r.dtype = "f16";
            r.shape = std::vector<std::int64_t>{12, 12, 64, 64};
            r.wire = true;
        } else if (id == "tensor:acts") {
            r.kind = "tensor";
            r.dtype = "f64";
            r.shape = std::vector<std::int64_t>{32, 128, 768};
            r.wire = true;
        } else if (id == "tensor:sample_png") {
            r.kind = "tensor";
            r.dtype = "u8";
            r.shape = std::vector<std::int64_t>{256, 256, 3};
            r.wire = true;
        } else if (id == "tensor:embedding") {
            r.kind = "tensor";
            r.dtype = "f32";
            r.shape = std::vector<std::int64_t>{1000, 3};
            r.wire = true;
        } else if (id == "tensor:embedding_labels") {
            r.kind = "tensor";
            r.dtype = "i64";
            r.shape = std::vector<std::int64_t>{1000};
            r.wire = true;
        } else if (id == "feed:gpu") {
            r.kind = "feed";
            r.columns = std::vector<cs::ResolvedColumn>{{"ts", "TIMESTAMP"}, {"util", "DOUBLE"}};
            r.wire = true;
        } else if (id == "metric:run42/train/loss") {
            r.kind = "metric";
            r.columns = std::vector<cs::ResolvedColumn>{{"step", "BIGINT"}, {"value", "DOUBLE"}};
            r.wire = true;
        } else {
            return std::nullopt;
        }
        return r;
    };
}

}  // namespace spec_test
