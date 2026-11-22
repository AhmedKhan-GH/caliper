// Tests of the sandbox of GPU programs and stages (sandbox.h,
// design/gpu-programs.md, Sandbox). The value types and the stage contract
// are fixed; the checks themselves get their cases with their bodies.
#include <caliper_builder/programs/sandbox.h>

#include <doctest/doctest.h>

namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;

TEST_CASE("the program value types name their kinds and fixed bindings") {
    CHECK(cs::output_kind_from_name(cs::output_kind_name(cs::OutputKind::pixels)) == cs::OutputKind::pixels);
    CHECK(cs::geometry_part_from_name("lines") == cs::GeometryPart::lines);
    CHECK_FALSE(cs::geometry_part_from_name("mesh"));
    CHECK(cs::geometry_array_from_name("radii") == cs::GeometryArray::radii);
    CHECK(cs::geometry_array_words(cs::GeometryArray::positions) == 3);
    CHECK(cs::stage_kind_from_name("vertex") == cs::StageKind::vertex);
    CHECK_FALSE(cs::stage_kind_from_name("compute"));
    CHECK(cs::default_entity("rows") == "programs/rows");
    CHECK(cs::program_view_name("rows") == "program.rows");
    cs::Stage s;
    s.panel = "world";
    s.entity = "programs/rows";
    CHECK(cs::stage_id(s) == "world/programs/rows/material");
    CHECK(cs::stage_target(s) == "stage:world/programs/rows/material");
    cs::ProgramLimits limits;
    CHECK(limits.time_ms_or_default() == cs::program_limits::kTimeMsDefault);
    CHECK(limits.memory_bytes_or_default() == (std::uint64_t(64) << 20));
    CHECK(cs::Provenance{}.empty());
}

TEST_CASE("the stage prelude declares the three structs of the contract on its stated lines") {
    const std::string_view p = cp::stage_prelude();
    std::uint32_t lines = 0;
    for (char c : p) lines += c == '\n';
    CHECK(lines == cp::stage_prelude_lines());
    for (const char* name : {"struct MaterialIn", "struct VertexIn", "struct VertexOut"})
        CHECK(p.find(name) != std::string_view::npos);
}

TEST_CASE("without the compiler library the compile answers program.compiler_unavailable") {
    if (cp::compiler_available()) return;
    CHECK(cp::compile_and_reflect("fn f() {}").verdict.code == cp::kProgramCompilerUnavailable);
}
