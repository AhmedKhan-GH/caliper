#pragma once
// caliper_builder::programs: the sandbox of GPU programs
// (design/gpu-programs.md, Sandbox). A declaration and the compiler's
// reflection go in, and a verdict comes out.
//
// The same checks run in two places: in the editor before a program
// edit (the session's program commands), and in the producer runtime in the
// script before a program runs.
// They run in this order, and the first failure stops the program:
//
//   1. check_declaration(): ids, counts, sizes
//      and limits inside their maximums (spec/program.h, program_limits)
//   2. check_provenance(): a source of a foreign spec has its hash in
//      provenance.reviewed, else the program waits for a review
//   3. check_source(): the size limit, and no enable, requires or diagnostic
//      directive
//   4. compile_and_reflect(): caliper-wgsl parses and validates the source
//      and reflects its bindings and entry points
//   5. and 6. check_program(): the entry point, the
//      workgroup size, and each binding against the declaration; an input
//      must be a storage buffer with read access only
//   7. to 9. the host's own: the read-only import of each input, the memory
//      budget, and the frame validation of the graphics layer
//
// The library caliper_builder_programs has no dependency beyond the standard
// library and the header-only value types of spec/program.h. Without the
// caliper-wgsl library (an imported target that a build may lack),
// compiler_available() is false and compile_and_reflect() answers
// program.compiler_unavailable; the editor then applies an edit with the
// warning program.not_checked, and the hosts check again before any backend
// sees the program. The functions are defined in src/builder/programs/.
// Nothing here throws. Every function is safe from any thread.
#include <caliper_builder/programs/reflection.h>
#include <caliper_builder/spec/program.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace caliper_builder::programs {

// ---- codes (design/gpu-programs.md, Command Errors and Host Errors)
// The editor's codes.
inline constexpr const char* kProgramExists = "program.exists";
inline constexpr const char* kProgramNotFound = "program.not_found";
inline constexpr const char* kProgramBadTarget = "program.bad_target";
inline constexpr const char* kProgramTooLarge = "program.too_large";
inline constexpr const char* kProgramDirectiveNotAllowed = "program.directive_not_allowed";
inline constexpr const char* kProgramCompileFailed = "program.compile_failed";
inline constexpr const char* kProgramNoEntry = "program.no_entry";
inline constexpr const char* kProgramBindingUndeclared = "program.binding_undeclared";
inline constexpr const char* kProgramWritesInput = "program.writes_input";
inline constexpr const char* kProgramOutputMismatch = "program.output_mismatch";
inline constexpr const char* kProgramOverLimits = "program.over_limits";
inline constexpr const char* kProgramCompilerUnavailable = "program.compiler_unavailable";
inline constexpr const char* kProgramNotChecked = "program.not_checked";   // a warning
// The struct Params of a program differs from the names of its `params` (design/instrument-interface.md, Program
// Parameters).
inline constexpr const char* kProgramParamsMismatch = "program.params_mismatch";
// The hosts' codes.
inline constexpr const char* kProgramNeedsReview = "program.needs_review";   // state review, not an error event
inline constexpr const char* kProgramUnknownInput = "program.unknown_input";
inline constexpr const char* kProgramImportRefused = "program.import_refused";
inline constexpr const char* kProgramOverMemory = "program.over_memory";
inline constexpr const char* kProgramLayerRefused = "program.layer_refused";
inline constexpr const char* kProgramOverTime = "program.over_time";
inline constexpr const char* kProgramOverBudget = "program.over_budget";   // the script stopped every program
inline constexpr const char* kProgramTimeout = "program.timeout";
inline constexpr const char* kProgramDeviceLost = "program.device_lost";
inline constexpr const char* kProgramUnavailable = "program.unavailable";

// ---- reflection, verdict and compile result: programs/reflection.h

// Whether this build has the caliper-wgsl library.
bool compiler_available();

// 1. The declaration of a program: the id, the input count and names, the
// output's kind, target, part, arrays (positions among them, radii only for
// points), max_count, size and format, the limits and the workgroup counts.
// The editor also checks the target panel against the spec; this check does
// not see the spec.
Verdict check_declaration(const spec::Program& program);

// 2. Whether a program may run under `provenance`: ok for a spec
// that is not foreign, or when `target` (spec::program_target()) is reviewed with this hash; else program.needs_review.
Verdict check_provenance(const spec::Provenance& provenance, std::string_view target, std::string_view sha256);

// 3. The source before the compiler: at most `max_bytes`
// (program_limits::kProgramSourceBytes), and no
// directive (enable, requires or diagnostic) at the top of the module.
Verdict check_source(std::string_view wgsl, std::size_t max_bytes);

// 4. Parses and validates the source with caliper-wgsl and reflects it. A
// refusal gives program.compile_failed with the compiler's one-line message,
// line and column; without the library, program.compiler_unavailable.
Compiled compile_and_reflect(std::string_view wgsl);
// The same for the module that keeps only the entry point `entry`
// (caliper_wgsl_options.entry_point): its bindings are those that the entry
// point uses, as a host binds them for a program of that entry point alone.
// A source without that entry point gives program.compile_failed.
Compiled compile_and_reflect_entry(std::string_view wgsl, std::string_view entry);

// 5. and 6. A program's reflection against its declaration: one compute entry
// point (or the named one) with at most kWorkgroupInvocations invocations;
// each binding is an input (storage, read access only: else
// program.writes_input), an output array of the declaration or `count`
// (storage, read_write), `pixels` for a pixels output, or the uniform
// `caliper` of kFrameUniformBytes; any other binding is
// program.binding_undeclared, and a missing or mismatched output is
// program.output_mismatch.
Verdict check_program(const spec::Program& program, const Reflection& reflection);

// ---- helpers the hosts share
// The SHA-256 of a source, in lowercase hex: provenance, status and the
// PROGRAM message carry it.
std::string sha256_hex(std::string_view text);
// The workgroup counts of a program's dispatch: its `workgroups`, else
// ceil(max_count / x) for geometry, ceil(width / x), ceil(height / y) for
// pixels and ceil(elements / x) for a tensor, with the entry's workgroup size
// from the reflection.
std::array<std::uint32_t, 3> dispatch_size(const spec::Program& program, const Reflection& reflection);
// The bytes of a program's output buffers: each array for max_count items and
// the counter, width * height * 4 for pixels, or 4 for each element of a
// tensor output.
std::uint64_t output_bytes(const spec::Program& program);
// The elements of a tensor output: the product of its shape; 0 for another
// kind or an empty shape.
std::uint64_t tensor_elements(const spec::ProgramOutput& output);

// ---- params (design/instrument-interface.md, Program Parameters)
// The WGSL name of a program's tensor output: var<storage, read_write> out: array<f32>.
inline constexpr const char* kTensorVar = "out";
// The scalar types of a field of the struct Params.
enum class ParamType : std::uint32_t { f32 = 0, i32 = 1, u32 = 2 };
// One field of the struct Params, in struct order. Each field takes 4 bytes,
// at offset 4 times its index.
struct ParamField {
    std::string name;
    ParamType type = ParamType::f32;
};
// The fields of `struct Params` in `wgsl`, in order. False when the source
// has no such struct; a field of another type than f32, i32 or u32, or with
// an attribute, gives program.params_mismatch in `why`.
bool params_fields(std::string_view wgsl, std::vector<ParamField>* fields, Verdict* why);
// A program's params against its source: the names of `params` and the
// fields of its struct Params are the same set, and the source declares the
// struct exactly when the program has params; else program.params_mismatch.
Verdict check_params(const spec::Program& program);
// Whether `value` is inside the declared range of a field of struct Params:
// a finite number of float range for f32, and after rounding a whole number
// of the type's range for i32 and u32. A value outside it is refused, not
// clamped, when a reference changes a program's params.
bool param_in_range(ParamType type, double value);
// The bytes of the uniform `params`: the value of each field by name,
// converted to its type (i32 and u32 rounded and clamped), 4 bytes each,
// padded to a multiple of 16 bytes; a field without a value is 0. Empty when
// there are no fields.
std::vector<std::uint8_t> pack_params(const std::vector<ParamField>& fields,
                                      const std::vector<std::pair<std::string, double>>& values);
}  // namespace caliper_builder::programs
