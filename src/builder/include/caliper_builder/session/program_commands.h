#pragma once
// The stage and program commands (design/gpu-programs.md, Commands).
//
// stage.set, stage.remove, program.create, program.patch and program.remove
// are edits of the open spec: apply_spec_edit() hands them to
// apply_program_edit(), which follows the rules of the panel edits (a dry
// run, the validator's codes, the result fields of the command table) and
// runs the sandbox checks of the editor first (programs/sandbox.h). A panel
// removal also removes the stages of that panel in the same step.
// stage.check and program.status read: program_command() answers them.
#include <caliper_builder/commands/registry.h>
#include <caliper_builder/session/program_host.h>
#include <caliper_builder/session/spec_edit.h>

#include <nlohmann/json.hpp>

#include <string_view>

namespace caliper_builder {

// stage.set, stage.remove, program.create, program.patch and program.remove,
// by name. Defined in program_edit.cpp.
SpecEdit apply_program_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                            const spec::SourceResolver& resolver);

// stage.check and program.status, by the invocation's name. `spec` is the
// open spec (program.status lists its programs and stages, with host none
// for an item that no host reports); `programs` and `stages` are the
// session's hosts, or null. Defined in program_commands.cpp.
CommandOutput program_command(const CommandContext& context, const spec::Spec& spec, const ProgramHost* programs,
                              const StageHost* stages);

}  // namespace caliper_builder
