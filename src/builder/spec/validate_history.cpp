// The validator's checks of kept steps: a source's `history`, the `window`
// step and the transport control
// (design/instrument-interface.md). Until the feature is built, the check
// reports each use of its keys with spec.not_built (validate_additions.h).
#include "validate_additions.h"

namespace caliper_builder::spec::detail {

void validate_history(const AdditionsCheck& c) {
    for (std::size_t i = 0; i < c.spec.sources.size(); ++i) {
        const Source& s = c.spec.sources[i];
        const std::string path = index_path("source", i);
        if (s.history) not_built(c.sink, "the key history", path + ".history");
        for (std::size_t j = 0; j < s.transforms.size(); ++j)
            if (s.transforms[j].op == TransformOp::window)
                not_built(c.sink, "the step window", index_path(path + ".transform", j));
    }
    for (std::size_t i = 0; i < c.spec.controls.size(); ++i) {
        const ControlOptions& o = c.spec.controls[i].options;
        const std::string path = index_path("control", i) + ".options";
        if (o.speed) not_built(c.sink, "the option speed", path + ".speed");
        if (o.keys) not_built(c.sink, "the option keys", path + ".keys");
    }
}

bool apply_window_shape(const Transform& t, SourceInfo& info, const std::string& path, Sink& sink) {
    (void)t;
    (void)path;
    (void)sink;
    info.shape.reset();
    return true;
}

}  // namespace caliper_builder::spec::detail
