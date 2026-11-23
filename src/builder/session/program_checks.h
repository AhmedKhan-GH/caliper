#pragma once
// The editor's sandbox checks of a program's source
// (design/gpu-programs.md, Check Order, steps 3 to 6), which the program
// edits (program_edit.cpp) use. Private to the session library. Defined in
// program_edit.cpp.
//
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
    programs::Reflection reflection;   // of the source alone
    bool compiled = false;             // the compiler took the source
    bool checked = false;              // every check ran; false without the compiler
};

// A program's source against its declaration.
Outcome check_program_source(const spec::Program& program);

// A message of the command table's compiler_message form.
nlohmann::json message_json(std::string_view severity, const programs::Verdict& verdict);
// The warning of an edit that the build could not check (program.not_checked).
programs::Verdict not_checked_warning();

}  // namespace caliper_builder::program_checks
