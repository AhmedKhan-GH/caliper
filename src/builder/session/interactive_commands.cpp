// The shared parts of the commands of design/instrument-interface.md
// (interactive_commands.h).
#include <caliper_builder/session/interactive_commands.h>

#include <caliper_builder/session/session.h>

namespace caliper_builder {

Error not_implemented(std::string_view name) {
    return Error("command.not_implemented", std::string(name) + " is part of the command table but is not built yet.",
                 std::string(name));
}

CommandOutput SessionHost::camera_command(const std::string& name, const nlohmann::json& args) {
    (void)args;
    return CommandOutput::fail(not_implemented(name));
}

}  // namespace caliper_builder
