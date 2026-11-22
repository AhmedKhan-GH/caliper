#pragma once
// The editor's sandbox checks of a program's or a stage's source
// (design/gpu-programs.md, Check Order, steps 3 to 6), which the program and
// stage edits (program_edit.cpp) and stage.check (program_commands.cpp)
// share. Private to the session library. Defined in program_edit.cpp.
//
// A stage compiles twice: with the stage prelude before it, for the
// reflection that check_stage() reads, and again with a call of its function
// after it, so that a missing function or a wrong signature gives a message.
// Each line and column is in the given source. Without the compiler library
// the source checks stop after check_source(), and `checked` is false.
#include <caliper_builder/programs/sandbox.h>
#include <caliper_builder/spec/program.h>

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <string_view>

namespace caliper_builder::program_checks {

struct Outcome {
    programs::Verdict error;           // the first failure, or ok
    programs::Reflection reflection;   // of the source alone (a stage: with the prelude)
    bool compiled = false;             // the compiler took the source
    bool checked = false;              // every check ran; false without the compiler
};

// A program's source against its declaration.
Outcome check_program_source(const spec::Program& program);
// A stage's source: its size, no directive, it compiles with the prelude, no
// binding, no entry point, and its function has the contract's signature.
Outcome check_stage_source(const spec::Stage& stage);
// A compute source without a declaration (stage.check with kind compute): its
// size, no directive, it compiles, and it has the compute entry point `entry`
// (or one compute entry point) with at most 256 invocations in a workgroup.
Outcome check_compute_source(std::string_view wgsl, const std::optional<std::string>& entry);

// A message of the command table's compiler_message form.
nlohmann::json message_json(std::string_view severity, const programs::Verdict& verdict);
// The warning of an edit that the build could not check (program.not_checked).
programs::Verdict not_checked_warning();

}  // namespace caliper_builder::program_checks
