// The annotation commands
// (design/instrument-interface.md, Commands). Not built yet: each command
// answers command.not_implemented.
#include <caliper_builder/session/instrument_commands.h>

namespace caliper_builder {

SpecEdit apply_annotation_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                              const spec::SourceResolver& resolver) {
    (void)spec;
    (void)args;
    (void)resolver;
    SpecEdit out;
    out.error = not_implemented(name);
    return out;
}

}  // namespace caliper_builder
