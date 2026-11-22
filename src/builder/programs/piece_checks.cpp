// Layouts and the checks of a piece's manifest, files and bindings
// (programs/pieces.h, Check Order 1, 2 and 4).
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

Verdict element_layout(const spec::Layout&, ElementLayout* out) {
    if (out) *out = ElementLayout{};
    return not_built("The layout of a resource");
}

Verdict check_piece_manifest(const spec::Piece&) { return not_built("The manifest check"); }

Verdict check_piece_source(std::string_view, std::size_t) { return not_built("The source check"); }

Verdict check_web_limits(const spec::Piece&) { return not_built("The check of the WebGPU limits"); }

Verdict check_tensor_binding(const spec::PieceInput&, std::string_view, std::uint32_t) {
    return not_built("The binding check");
}

Verdict check_buffer_binding(const spec::PieceInput&, const spec::Layout&) { return not_built("The binding check"); }

}  // namespace caliper_builder::programs
