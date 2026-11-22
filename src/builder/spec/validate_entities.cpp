// The validator's checks of the entities of a scene panel and of the scene
// options `box` and `views`
// (design/instrument-interface.md). Until the feature is built, the check
// reports each use of its keys with spec.not_built (validate_additions.h).
#include "validate_additions.h"

namespace caliper_builder::spec::detail {

void validate_entities(const AdditionsCheck& c) {
    for (std::size_t i = 0; i < c.spec.panels.size(); ++i) {
        const Panel& p = c.spec.panels[i];
        const std::string path = index_path("panel", i);
        if (!p.entities.empty()) not_built(c.sink, "entities", path + ".entity");
        if (!p.options.box.is_null()) not_built(c.sink, "the option box", path + ".options.box");
        if (!p.options.views.empty()) not_built(c.sink, "the option views", path + ".options.views");
    }
}

}  // namespace caliper_builder::spec::detail
