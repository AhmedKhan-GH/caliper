// The operator steps of a source chain in the app (chain_hooks.h). Not
// built yet: the validator refuses the operators (spec.not_built), and the
// hook answers resolve.not_built.
#include <caliper_builder/session/chain_hooks.h>

namespace caliper_builder {

Error apply_operator(const spec::Transform& step, const TensorData& input, const TensorData* row_steps,
                     GeometryArrays* out) {
    (void)input;
    (void)row_steps;
    (void)out;
    return Error("resolve.not_built", std::string("The operator ") + spec::to_string(step.op) + " is not built yet.");
}

bool operator_runs_in_script(const spec::Spec& spec, const spec::Source& source, std::size_t index) {
    (void)spec;
    (void)source;
    (void)index;
    return false;
}

}  // namespace caliper_builder
