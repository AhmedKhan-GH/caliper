// The validator's checks of annotations, legends and scale bars
// (design/instrument-interface.md). Until the feature is built, the check
// reports each use of its keys with spec.not_built (validate_additions.h).
#include "validate_additions.h"

namespace caliper_builder::spec::detail {

void validate_annotations(const AdditionsCheck& c) {
    for (std::size_t i = 0; i < c.spec.panels.size(); ++i) {
        const Panel& p = c.spec.panels[i];
        const std::string path = index_path("panel", i);
        if (!p.annotations.empty()) not_built(c.sink, "annotations", path + ".annotation");
        if (!p.options.legend.is_null()) not_built(c.sink, "the option legend", path + ".options.legend");
        if (p.primitive != Primitive::scene && p.options.scene.scale_bar)
            not_built(c.sink, "the option scale_bar outside a scene", path + ".options.scale_bar");
    }
}

}  // namespace caliper_builder::spec::detail
