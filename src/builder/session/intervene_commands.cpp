// The intervention commands (design/gpu-programs.md, Intervention Commands).
// The interface fixes their names, their arguments, their results and the
// RunControl functions they call; their bodies come next and answer
// command.not_implemented until then.
#include <caliper_builder/session/intervene_commands.h>

#include <string>

namespace caliper_builder {

CommandOutput intervene_command(const CommandContext& context, RunControl*) {
    return CommandOutput::fail(Error("command.not_implemented",
                                     context.invocation.name + " is not built yet (design/gpu-programs.md)."));
}

}  // namespace caliper_builder
