// The run commands (run_control.h, design/instruments.md, Run Control).
//
// Stubs of milestone M1's interface commit: the four commands answer
// command.not_implemented until the track "controls" fills this file.
#include <caliper_builder/session/run_control.h>

namespace caliper_builder {

nlohmann::json run_state_json(const RunState& state) {
    (void)state;
    return nlohmann::json::object();   // stub: track controls
}

CommandOutput run_command(const CommandContext& context, RunControl* control) {
    (void)control;
    return CommandOutput::fail(Error("command.not_implemented",
                                     context.invocation.name + " is part of milestone M1 and is not built yet.",
                                     context.invocation.name));
}

}  // namespace caliper_builder
