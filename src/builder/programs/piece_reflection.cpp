// The checks of a compiled pass of either half against the compiler's
// reflection (programs/pieces.h, Check Order 7).
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

Verdict check_piece_reflection(const spec::Piece&, Half, std::size_t, const Reflection&) {
    return not_built("The reflection check");
}

}  // namespace caliper_builder::programs
