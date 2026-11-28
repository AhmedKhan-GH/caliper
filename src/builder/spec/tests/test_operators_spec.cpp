// The spec tests of the operators surface and path, params and the tensor output
// (design/instrument-interface.md, Operators).
#include <doctest/doctest.h>

#include <caliper_builder/spec/format.h>
#include <caliper_builder/spec/validate.h>

#include "helpers.h"

using namespace spec_test;

namespace {

const char* kHead = "caliper_spec = 2\n[instrument]\nid = \"o\"\n";

cs::ValidationReport check(const std::string& body) { return cs::check_toml(std::string(kHead) + body, fixture_resolver()); }

// A (64, 64) slice of the live attention weights, then `steps`.
std::string live_source(const std::string& steps) {
    return "[[source]]\nid = \"terrain\"\ncatalog = \"live:attn.weights\"\n"
           "transform = [{ op = \"slice\", axis = 0, index = 0 }, { op = \"slice\", axis = 0, index = 0 }" +
           (steps.empty() ? std::string() : ", " + steps) + "]\n";
}

const char* kSurfaceProgram = R"(
[[program]]
id = "rows"
wgsl = """
struct Params { level: f32, count: u32 }
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.count) { out[id.x] = w[id.x] * params.level; }
}
"""
inputs = [{ var = "w", source = "live:layers.0.weight" }]
output = { kind = "tensor", shape = [16] }
)";

}  // namespace

TEST_CASE("operators: surface and path are read, built and keep their result out of later steps") {
    cs::ValidationReport r = check(live_source("{ op = \"surface\", max_cells = 1024, pool = \"max\" }"));
    INFO(dump(r.diagnostics));
    CHECK(r.ok());
    CHECK_FALSE(has(r.diagnostics, "spec.not_built"));
    REQUIRE(r.sources.size() == 1);
    CHECK_FALSE(r.sources[0].shape);   // geometry, not one tensor

    r = check("[[source]]\nid = \"trail\"\ncatalog = \"tensor:embedding\"\n"
              "transform = [{ op = \"path\", columns = [2, 0], max_points = 100 }]\n");
    INFO(dump(r.diagnostics));
    CHECK(r.ok());

    // A step after an operator, in the same source or in a source from it.
    r = check(live_source("{ op = \"surface\" }, { op = \"normalise\", mode = \"minmax\" }"));
    CHECK(has(r.diagnostics, "spec.bad_transform", "source[0].transform[3]"));
    r = check(live_source("{ op = \"surface\" }") +
              "[[source]]\nid = \"after\"\nfrom = \"terrain\"\ntransform = [{ op = \"reduce\", fn = \"mean\", axis = 0 }]\n");
    CHECK(has(r.diagnostics, "spec.bad_transform", "source[1].transform[0]"));
}

TEST_CASE("operators: parameters out of range, keys of another op and a wrong rank are refused") {
    cs::ValidationReport r = check(live_source("{ op = \"surface\", max_cells = 300000 }"));
    CHECK(has(r.diagnostics, "spec.out_of_range", "source[0].transform[2].max_cells"));
    r = check(live_source("{ op = \"surface\", pool = \"median\" }"));
    CHECK(has(r.diagnostics, "spec.bad_value", "source[0].transform[2].pool"));
    r = check(live_source("{ op = \"surface\", max_points = 10 }"));
    INFO(dump(r.diagnostics));
    CHECK(has(r.diagnostics, "spec.unknown_key", "source[0].transform[2].max_points"));
    r = check(live_source("{ op = \"path\", columns = [0] }"));
    CHECK(has(r.diagnostics, "spec.bad_value", "source[0].transform[2].columns"));
    r = check(live_source("{ op = \"path\", max_points = 1 }"));
    CHECK(has(r.diagnostics, "spec.out_of_range", "source[0].transform[2].max_points"));
    r = check(live_source("{ op = \"path\", pool = \"max\" }"));
    CHECK(has(r.diagnostics, "spec.unknown_key", "source[0].transform[2].pool"));

    // The catalog gives (12, 12, 64, 64): a surface of rank 4, and a path past the last column.
    r = check("[[source]]\nid = \"w\"\ncatalog = \"live:attn.weights\"\ntransform = [{ op = \"surface\" }]\n");
    CHECK(has(r.diagnostics, "spec.operator_shape", "source[0].transform[0]"));
    const cs::Diagnostic* d = find(r.diagnostics, "spec.operator_shape");
    REQUIRE(d);
    CHECK(d->line > 0);
    r = check("[[source]]\nid = \"trail\"\ncatalog = \"tensor:embedding\"\n"
              "transform = [{ op = \"path\", columns = [0, 5] }]\n");
    CHECK(has(r.diagnostics, "spec.operator_shape", "source[0].transform[0].columns[1]"));
}

TEST_CASE("operators: params and the tensor output of a program") {
    cs::ValidationReport r = check(std::string(kSurfaceProgram) + "params = { level = 0.5, count = 16 }\n");
    INFO(dump(r.diagnostics));
    CHECK(r.ok());
    CHECK_FALSE(has(r.diagnostics, "spec.not_built"));

    // A name the struct lacks, and a struct field that params lacks.
    r = check(std::string(kSurfaceProgram) + "params = { level = 0.5, scale = 2 }\n");
    CHECK(has(r.diagnostics, "program.params_mismatch"));
    r = check(std::string(kSurfaceProgram));
    CHECK(has(r.diagnostics, "program.params_mismatch"));

    // A tensor output past 1,000,000 elements, and with keys of another kind.
    std::string big = kSurfaceProgram;
    big.replace(big.find("shape = [16]"), 12, "shape = [1001, 1000]");
    r = check(big + "params = { level = 0.5, count = 16 }\n");
    CHECK(has(r.diagnostics, "program.over_limits", "program[0].output.shape"));
}
