// The validator's checks of state variables and the option `select`
// (design/instrument-interface.md). Until the feature is built, the check
// reports each use of its keys with spec.not_built (validate_additions.h).
#include "validate_additions.h"

namespace caliper_builder::spec::detail {

void validate_states(const AdditionsCheck& c) {
    if (!c.spec.states.empty()) not_built(c.sink, "state variables", "state");
    for (std::size_t i = 0; i < c.spec.panels.size(); ++i)
        if (c.spec.panels[i].options.select)
            not_built(c.sink, "the option select", index_path("panel", i) + ".options.select");
}

}  // namespace caliper_builder::spec::detail
