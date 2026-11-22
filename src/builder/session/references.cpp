// The chain hooks of binding references (chain_hooks.h). Not built yet: the
// validator refuses every reference (spec.not_built), so no step that
// reaches these hooks holds one.
#include <caliper_builder/session/chain_hooks.h>

namespace caliper_builder {

Error substitute_references(const spec::Transform& step, const ChainContext& ctx, spec::Transform* out) {
    (void)ctx;
    *out = step;
    return {};
}

std::string references_key(const std::vector<const spec::Source*>& chain, const ChainContext& ctx) {
    (void)chain;
    (void)ctx;
    return {};
}

}  // namespace caliper_builder
