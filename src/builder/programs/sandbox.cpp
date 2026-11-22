// The sandbox of GPU programs and stages (design/gpu-programs.md, Sandbox).
// The interface fixes the checks and their order; their bodies come next and
// answer command.not_implemented until then.
#include <caliper_builder/programs/sandbox.h>

#if defined(CALIPER_BUILDER_PROGRAMS_HAVE_WGSL)
#include <caliper/wgsl.h>
#endif

namespace caliper_builder::programs {

namespace {

Verdict not_implemented(const char* what) {
    Verdict v;
    v.code = "command.not_implemented";
    v.message = std::string(what) + " is not built yet (design/gpu-programs.md, Sandbox).";
    return v;
}

}  // namespace

bool compiler_available() {
#if defined(CALIPER_BUILDER_PROGRAMS_HAVE_WGSL)
    return true;
#else
    return false;
#endif
}

Verdict check_declaration(const spec::Program&) { return not_implemented("The check of a program's declaration"); }

Verdict check_stage_declaration(const spec::Stage&) { return not_implemented("The check of a stage's declaration"); }

Verdict check_provenance(const spec::Provenance&, std::string_view, std::string_view) {
    return not_implemented("The check of a spec's provenance");
}

Verdict check_source(std::string_view, std::size_t) { return not_implemented("The check of a WGSL source"); }

Compiled compile_and_reflect(std::string_view) {
    Compiled out;
    if (!compiler_available()) {
        out.verdict.code = kProgramCompilerUnavailable;
        out.verdict.message = "This build has no WGSL compiler, so the source cannot be checked here.";
        return out;
    }
    out.verdict = not_implemented("The compile and reflection of a WGSL source");
    return out;
}

Verdict check_program(const spec::Program&, const Reflection&) {
    return not_implemented("The check of a program's bindings");
}

Verdict check_stage(const spec::Stage&, const Reflection&) { return not_implemented("The check of a stage's bindings"); }

std::string sha256_hex(std::string_view) { return std::string(); }

std::array<std::uint32_t, 3> dispatch_size(const spec::Program&, const Reflection&) { return {0, 0, 0}; }

std::uint64_t output_bytes(const spec::Program&) { return 0; }

}  // namespace caliper_builder::programs
