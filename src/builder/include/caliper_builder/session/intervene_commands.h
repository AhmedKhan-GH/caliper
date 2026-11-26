#pragma once
// The intervention commands (design/gpu-programs.md, Intervention Commands):
// intervene.points and intervene.list read, intervene.set and intervene.clear
// act on the attached script at its next step, with the effect `run`.
#include <caliper_builder/commands/registry.h>
#include <caliper_builder/session/run_control.h>

namespace caliper_builder {

// The handler of the four commands, by the invocation's name. `control` is
// the session's run control, or null, which answers command.not_available. A
// run control whose script did not announce `interventions` answers
// intervene.not_supported. Defined in intervene_commands.cpp.
Error check_intervention_spec(const nlohmann::json& spec, const std::string& where);

CommandOutput interpretation_command(const CommandContext& context, RunControl* control);

CommandOutput intervene_command(const CommandContext& context, RunControl* control);

}  // namespace caliper_builder
