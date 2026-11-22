// The validator's checks of binding references
// (design/instrument-interface.md). Until the feature is built, the check
// reports each use of its keys with spec.not_built (validate_additions.h).
#include "validate_additions.h"

namespace caliper_builder::spec::detail {

namespace {

void refuse_map(const AdditionsCheck& c, const std::map<std::string, Reference>& refs, const std::string& path) {
    for (const auto& [key, ref] : refs) {
        (void)ref;
        not_built(c.sink, "references", path + "." + key);
    }
}

void refuse_params(const AdditionsCheck& c, const Params& params, const std::string& path) {
    for (const auto& [name, value] : params)
        if (is_reference(value)) not_built(c.sink, "references", path + ".params." + name);
}

}  // namespace

void validate_references(const AdditionsCheck& c) {
    for (std::size_t i = 0; i < c.spec.sources.size(); ++i) {
        const std::string path = index_path("source", i);
        const auto& transforms = c.spec.sources[i].transforms;
        for (std::size_t j = 0; j < transforms.size(); ++j)
            refuse_map(c, transforms[j].references, index_path(path + ".transform", j));
    }
    for (std::size_t i = 0; i < c.spec.panels.size(); ++i) {
        const Panel& p = c.spec.panels[i];
        const std::string path = index_path("panel", i);
        refuse_map(c, p.options.references, path + ".options");
        for (const auto& [slot, value] : p.bind)
            if (value.reference) not_built(c.sink, "references", path + ".bind." + slot);
    }
    for (std::size_t i = 0; i < c.spec.controls.size(); ++i)
        refuse_map(c, c.spec.controls[i].options.references, index_path("control", i) + ".options");
    for (std::size_t i = 0; i < c.spec.programs.size(); ++i)
        refuse_params(c, c.spec.programs[i].params, index_path("program", i));
    for (std::size_t i = 0; i < c.spec.stages.size(); ++i)
        refuse_params(c, c.spec.stages[i].params, index_path("stage", i));
}

}  // namespace caliper_builder::spec::detail
