// Names, files, imports, hash, composition and checks of pieces
// (programs/pieces.h). Only the embedded texts are built, in the generated
// caliper_programs_text.cpp; every function here answers program.not_built
// until the work of design/programmable-graphics.md fills it. This file needs
// only the standard library and spec/piece.h, because the browser module
// compiles it too.
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

Verdict check_piece_name(std::string_view, std::uint32_t) { return not_built("The name check"); }

Verdict check_piece_file(const spec::Piece&, std::string_view) { return not_built("The file check"); }

Resolved resolve_imports(const spec::Piece&) {
    Resolved r;
    r.verdict = not_built("The resolution of imports");
    return r;
}

std::string canonical_manifest(const spec::Piece&) { return std::string(); }

std::string piece_hash(const spec::Piece&) { return std::string(); }

Verdict element_layout(const spec::Layout&, ElementLayout* out) {
    if (out) *out = ElementLayout{};
    return not_built("The layout of a resource");
}

SourcePlace source_place(const Composed&, std::uint32_t line, std::uint32_t column) {
    return SourcePlace{"header", line, column};
}

Composed compose_derive(const spec::Piece&, const Resolved&) {
    Composed c;
    c.verdict = not_built("The composer of the derive half");
    return c;
}

Composed compose_draw(const spec::Piece&, const Resolved&, std::size_t) {
    Composed c;
    c.verdict = not_built("The composer of the draw half");
    return c;
}

Verdict check_piece_manifest(const spec::Piece&) { return not_built("The manifest check"); }

Verdict check_piece_source(std::string_view, std::size_t) { return not_built("The source check"); }

Verdict check_piece_reflection(const spec::Piece&, Half, std::size_t, const Reflection&) {
    return not_built("The reflection check");
}

Verdict check_web_limits(const spec::Piece&) { return not_built("The check of the WebGPU limits"); }

Verdict check_tensor_binding(const spec::PieceInput&, std::string_view, std::uint32_t) {
    return not_built("The binding check");
}

Verdict check_buffer_binding(const spec::PieceInput&, const spec::Layout&) { return not_built("The binding check"); }

}  // namespace caliper_builder::programs
