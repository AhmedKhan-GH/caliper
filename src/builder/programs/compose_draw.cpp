// The composer of a piece's draw half: the prelude and the generated
// header of each pass (programs/pieces.h, Generated Header).
// Not built yet: each function answers program.not_built until the work of
// design/programmable-graphics.md fills it. This file needs only the
// standard library and spec/piece.h, because the browser module compiles
// it too.
#include <caliper_builder/programs/pieces.h>

namespace caliper_builder::programs {

namespace {

Verdict not_built(const char* what) {
    Verdict v;
    v.code = kProgramNotBuilt;
    v.message = std::string(what) + " of pieces is not built yet.";
    return v;
}

}  // namespace

Composed compose_draw(const spec::Piece&, const Resolved&, std::size_t) {
    Composed c;
    c.verdict = not_built("The composer of the draw half");
    return c;
}

}  // namespace caliper_builder::programs
