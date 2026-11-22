// The step `window` over kept steps (chain_hooks.h). Not built yet: the
// validator refuses it (spec.not_built), and the hook answers
// resolve.not_built.
#include <caliper_builder/session/chain_hooks.h>

namespace caliper_builder {

Error apply_window(const spec::Transform& step, const spec::Source& source, const ChainContext& ctx, TensorData* out,
                   TensorData* steps) {
    (void)step;
    (void)ctx;
    (void)out;
    (void)steps;
    return Error("resolve.not_built", "The step window is not built yet.", "source " + source.id);
}

}  // namespace caliper_builder
