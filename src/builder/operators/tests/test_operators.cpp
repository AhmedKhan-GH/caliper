// The tests of the built-in operators (caliper_builder/operators/operators.h):
// the CPU references against fixed grids, the WGSL sources and the program
// declaration of a placed operator.
#include <doctest/doctest.h>

#include <caliper_builder/operators/operators.h>

namespace ops = caliper_builder::operators;

TEST_CASE("operators: both WGSL sources are compiled in") {
    CHECK_FALSE(ops::builtin_wgsl("surface").empty());
    CHECK_FALSE(ops::builtin_wgsl("path").empty());
    CHECK(ops::builtin_wgsl("other").empty());
}
