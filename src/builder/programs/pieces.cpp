// Names, paths, imports, the canonical manifest, the hash and the places
// of composed lines of pieces (programs/pieces.h).
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

Verdict check_piece_name(std::string_view, std::uint32_t) { return not_built("The name check"); }

Verdict check_piece_file(const spec::Piece&, std::string_view) { return not_built("The file check"); }

Resolved resolve_imports(const spec::Piece&) {
    Resolved r;
    r.verdict = not_built("The resolution of imports");
    return r;
}

std::string canonical_manifest(const spec::Piece&) { return std::string(); }

std::string piece_hash(const spec::Piece&) { return std::string(); }

SourcePlace source_place(const Composed&, std::uint32_t line, std::uint32_t column) {
    return SourcePlace{"header", line, column};
}

}  // namespace caliper_builder::programs
