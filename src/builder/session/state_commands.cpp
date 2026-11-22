// The state commands and control.set
// (design/instrument-interface.md, Commands). Not built yet: each command
// answers command.not_implemented.
#include <caliper_builder/session/instrument_commands.h>

namespace caliper_builder {

SpecEdit apply_state_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                         const spec::SourceResolver& resolver) {
    (void)spec;
    (void)args;
    (void)resolver;
    SpecEdit out;
    out.error = not_implemented(name);
    return out;
}

CommandOutput state_command(const CommandContext& ctx, InstrumentContext& ic) {
    (void)ic;
    return CommandOutput::fail(not_implemented(ctx.invocation.name));
}

}  // namespace caliper_builder
