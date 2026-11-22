// The validator's checks of layout shares and leaf minimums
// (design/instrument-interface.md). Until the feature is built, the check
// reports each use of its keys with spec.not_built (validate_additions.h).
#include "validate_additions.h"

namespace caliper_builder::spec::detail {

namespace {

void walk(const AdditionsCheck& c, const LayoutNode& node, const std::string& path) {
    if (!node.shares.empty()) not_built(c.sink, "the key shares", path + ".shares");
    if (node.min) not_built(c.sink, "the key min", path + ".min");
    for (std::size_t i = 0; i < node.children.size(); ++i) walk(c, node.children[i], index_path(path + ".children", i));
}

}  // namespace

void validate_layout(const AdditionsCheck& c) {
    if (c.spec.layout) walk(c, *c.spec.layout, "layout");
}

}  // namespace caliper_builder::spec::detail
