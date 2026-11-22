// The stage and program edits of the open spec (design/gpu-programs.md, Stage
// Commands and Program Commands): stage.set, stage.remove, program.create,
// program.patch and program.remove. The interface fixes their names, their
// arguments and their results in the command table; their bodies come next
// and answer command.not_implemented until then.
#include <caliper_builder/session/program_commands.h>

#include <string>

namespace caliper_builder {

SpecEdit apply_program_edit(const spec::Spec&, std::string_view name, const nlohmann::json&,
                            const spec::SourceResolver&) {
    SpecEdit out;
    out.error = Error("command.not_implemented",
                      std::string(name) + " is not built yet (design/gpu-programs.md); the spec is unchanged.");
    return out;
}

}  // namespace caliper_builder
