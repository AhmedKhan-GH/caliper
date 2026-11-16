#pragma once
// unified_line_diff(): a unified diff of two texts, line by line, as `diff -u`
// prints it, for showing a spec edit to a person before or after it applies
// (design/builder-session.md, Diffs).
//
// The diff starts with "--- <before label>" and "+++ <after label>", then
// one hunk per group of changed lines with `context` unchanged lines around
// it; hunks closer than twice the context merge. A text that does not end with
// a line feed gets the marker "\ No newline at end of file" after its last
// line. Identical texts give an empty string.
//
// The work is bounded: the lines both texts share at the start and at the end
// are set aside first, and the rest is compared with a longest common
// subsequence only when the product of its line counts is at most 2^22;
// beyond that the middle is shown as removed and added as a whole, which is
// still a correct diff, only a longer one. No function here throws.
#include <cstddef>
#include <string>
#include <string_view>

namespace caliper_builder {

std::string unified_line_diff(std::string_view before, std::string_view after,
                              std::string_view before_label = "before", std::string_view after_label = "after",
                              std::size_t context = 3);

}  // namespace caliper_builder
