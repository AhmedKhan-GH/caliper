// The checks of pieces for the draw tests (draw/draw_runtime.h, DrawChecks): the functions of
// programs/pieces.h, with the stand-ins of example_pieces.h where this tree's answer
// program.not_built, so that the render graph is tested before and after the checks of pieces.
#pragma once

#include "draw/draw_runtime.h"

#include "example_pieces.h"

namespace caliper_test_pieces {

inline caliper_draw::DrawChecks draw_checks() {
    caliper_draw::DrawChecks c = caliper_draw::DrawChecks::pieces();
    const cs::Piece p = ring_arcs();
    if (cp::check_piece_manifest(p).code == cp::kProgramNotBuilt) c.manifest = [](const cs::Piece&) { return cp::Verdict{}; };
    if (cp::check_piece_source("", 1).code == cp::kProgramNotBuilt)
        c.source = [](std::string_view, std::size_t) { return cp::Verdict{}; };
    if (cp::resolve_imports(p).verdict.code == cp::kProgramNotBuilt) c.resolve = resolve;
    cp::ElementLayout e;
    if (cp::element_layout(element("f32"), &e).code == cp::kProgramNotBuilt) c.layout = layout;
    if (cp::piece_hash(p).empty()) c.hash = hash;
    return c;
}

// Packs params in the manifest's order, as the host packs Params.
inline std::vector<std::uint8_t> pack(const std::vector<double>& values, const cs::Piece& piece) {
    std::vector<std::uint8_t> out(piece.params.size() * 4, 0);
    for (std::size_t i = 0; i < piece.params.size() && i < values.size(); ++i) {
        std::uint32_t bits = 0;
        if (piece.params[i].type == cs::ParamType::f32) {
            const float f = float(values[i]);
            std::memcpy(&bits, &f, 4);
        } else {
            bits = std::uint32_t(std::int32_t(values[i]));
        }
        std::memcpy(out.data() + 4 * i, &bits, 4);
    }
    return out;
}

}  // namespace caliper_test_pieces
