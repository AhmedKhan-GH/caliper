// program.read (resource_commands.h, design/programmable-graphics.md).
// Not built yet: it answers program.not_built.
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/resource_commands.h>

namespace caliper_builder {

CommandOutput resource_command(const CommandContext& context, const spec::Spec&, const StepHistory&) {
    return CommandOutput::fail(Error(programs::kProgramNotBuilt, context.invocation.name + " is not built yet."));
}

}  // namespace caliper_builder
