// The validator's checks of the operators `surface` and `path`, of `params`
// and of the `tensor` output
// (design/instrument-interface.md). Until the feature is built, the check
// reports each use of its keys with spec.not_built (validate_additions.h).
#include "validate_additions.h"

namespace caliper_builder::spec::detail {

void validate_operators(const AdditionsCheck& c) {
    for (std::size_t i = 0; i < c.spec.sources.size(); ++i) {
        const Source& s = c.spec.sources[i];
        for (std::size_t j = 0; j < s.transforms.size(); ++j)
            if (s.transforms[j].op == TransformOp::surface || s.transforms[j].op == TransformOp::path)
                not_built(c.sink, std::string("the operator ") + to_string(s.transforms[j].op),
                          index_path(index_path("source", i) + ".transform", j));
    }
    for (std::size_t i = 0; i < c.spec.programs.size(); ++i) {
        const Program& p = c.spec.programs[i];
        const std::string path = index_path("program", i);
        if (!p.params.empty()) not_built(c.sink, "the key params", path + ".params");
        if (p.output.kind == OutputKind::tensor) not_built(c.sink, "the output kind tensor", path + ".output.kind");
    }
    for (std::size_t i = 0; i < c.spec.stages.size(); ++i)
        if (!c.spec.stages[i].params.empty())
            not_built(c.sink, "the key params", index_path("stage", i) + ".params");
}

bool apply_operator_shape(const Transform& t, SourceInfo& info, const std::string& path, Sink& sink) {
    (void)t;
    (void)path;
    (void)sink;
    info.shape.reset();
    return true;
}

}  // namespace caliper_builder::spec::detail
