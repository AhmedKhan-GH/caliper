// The validator's checks of bound texts in text panels and readouts
// (design/instrument-interface.md). Until the feature is built, the check
// reports each use of its keys with spec.not_built (validate_additions.h).
#include "validate_additions.h"

namespace caliper_builder::spec::detail {

void validate_bound_text(const AdditionsCheck& c) {
    for (std::size_t i = 0; i < c.spec.panels.size(); ++i)
        if (c.spec.panels[i].options.bound_text)
            not_built(c.sink, "the option format", index_path("panel", i) + ".options.format");
    for (std::size_t i = 0; i < c.spec.controls.size(); ++i)
        if (c.spec.controls[i].options.bound_text)
            not_built(c.sink, "the option format", index_path("control", i) + ".options.format");
}

}  // namespace caliper_builder::spec::detail
