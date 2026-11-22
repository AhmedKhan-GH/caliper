// The piece commands (piece_commands.h, design/programmable-graphics.md).
// Not built yet: each one answers program.not_built.
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/session/piece_commands.h>

namespace caliper_builder {

CommandOutput piece_command(const CommandContext& context, const spec::Spec&, const std::string&) {
    return CommandOutput::fail(Error(programs::kProgramNotBuilt, context.invocation.name + " is not built yet."));
}

}  // namespace caliper_builder
