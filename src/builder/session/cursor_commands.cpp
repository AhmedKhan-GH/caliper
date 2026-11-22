// The cursor commands
// (design/instrument-interface.md, Commands). Not built yet: each command
// answers command.not_implemented.
#include <caliper_builder/session/instrument_commands.h>

namespace caliper_builder {

CommandOutput cursor_command(const CommandContext& ctx, InstrumentContext& ic) {
    (void)ic;
    return CommandOutput::fail(not_implemented(ctx.invocation.name));
}

}  // namespace caliper_builder
