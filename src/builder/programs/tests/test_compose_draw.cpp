// Tests of the composer of a piece's draw half (programs/pieces.h,
// design/programmable-graphics.md, Generated Header). Until the composer is
// built, it answers program.not_built.
#include <caliper_builder/programs/pieces.h>

#include <doctest/doctest.h>

namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;

TEST_CASE("the composer of the draw half answers program.not_built") {
    cs::Piece piece;
    piece.name = "attention_arcs";
    const cp::Resolved resolved;
    const cp::Composed c = cp::compose_draw(piece, resolved, 0);
    CHECK(c.verdict.code == cp::kProgramNotBuilt);
    CHECK(c.text.empty());
}
