// The validator's checks of layout shares and leaf minimums
// (design/instrument-interface.md, Layout Shares).
//
// A split's `shares` hold one positive number for each child and replace
// `ratio`, so a split with both is spec.bad_layout, as are shares on tabs or
// shares of the wrong count. A leaf's `min` is [columns, lines] of the text
// scale, each from 4 to 400.
#include "validate_additions.h"

#include <cmath>

namespace caliper_builder::spec::detail {

namespace {

constexpr std::int64_t kMinCells = 4;
constexpr std::int64_t kMaxCells = 400;

void walk(const AdditionsCheck& c, const LayoutNode& node, const std::string& path, int depth) {
    if (depth > 64) return;   // the format allows 8 levels; a guard, not a rule
    if (!node.shares.empty()) {
        const std::string sp = path + ".shares";
        if (node.kind != LayoutNode::Kind::split) {
            c.sink.error("spec.bad_layout", path + " is not a split, so it takes no shares.", sp);
        } else if (node.ratio) {
            c.sink.error("spec.bad_layout", path + " has both ratio and shares; shares replace the ratio.", sp);
        } else if (node.shares.size() != node.children.size()) {
            c.sink.error("spec.bad_layout",
                         sp + " has " + std::to_string(node.shares.size()) + " shares for " +
                             std::to_string(node.children.size()) + " children; a split has one share per child.",
                         sp);
        }
        for (std::size_t i = 0; i < node.shares.size(); ++i) {
            const double s = node.shares[i];
            if (!std::isfinite(s) || s <= 0.0 || s > 1e6)
                c.sink.error("spec.out_of_range", index_path(sp, i) + " must be a positive number, at most 1,000,000.",
                             index_path(sp, i));
        }
    }
    if (node.min) {
        const std::string mp = path + ".min";
        if (node.kind != LayoutNode::Kind::panel) {
            c.sink.error("spec.bad_layout", path + " is not a leaf, so it takes no min.", mp);
        } else {
            const auto& m = *node.min;
            for (std::size_t i = 0; i < 2; ++i)
                if (m[i] < kMinCells || m[i] > kMaxCells)
                    c.sink.error("spec.out_of_range",
                                 index_path(mp, i) + (i == 0 ? " (columns)" : " (lines)") + " must be from 4 to 400.",
                                 index_path(mp, i));
        }
    }
    for (std::size_t i = 0; i < node.children.size(); ++i)
        walk(c, node.children[i], index_path(path + ".children", i), depth + 1);
}

}  // namespace

void validate_layout(const AdditionsCheck& c) {
    if (c.spec.layout) walk(c, *c.spec.layout, "layout", 0);
}

}  // namespace caliper_builder::spec::detail
