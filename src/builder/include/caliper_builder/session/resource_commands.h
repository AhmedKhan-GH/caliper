#pragma once
// program.read of design/programmable-graphics.md, Program Commands: the
// elements of a published resource of a program at a step, from the kept
// versions of the step history. Not built yet: it answers program.not_built.
#include <caliper_builder/commands/registry.h>
#include <caliper_builder/spec/spec.h>

namespace caliper_builder {

class StepHistory;

// Answers program.read. `spec` is a copy of the open spec; `history` is the
// session's step history. Defined in resource_commands.cpp.
CommandOutput resource_command(const CommandContext& context, const spec::Spec& spec, const StepHistory& history);

}  // namespace caliper_builder
