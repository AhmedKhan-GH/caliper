// The edits of the spec's controls (spec_edit.h, design/instruments.md,
// Control Commands).
//
// A stub of milestone M1's interface commit: control.create, control.patch
// and control.remove answer command.not_implemented until the track
// "controls" fills this file.
#include <caliper_builder/session/spec_edit.h>

namespace caliper_builder {

SpecEdit apply_control_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                            const spec::SourceResolver& resolver) {
    (void)args;
    (void)resolver;
    SpecEdit out;
    out.spec = spec;
    out.error = Error("command.not_implemented", std::string(name) + " is part of milestone M1 and is not built yet.",
                      std::string(name));
    return out;
}

}  // namespace caliper_builder
