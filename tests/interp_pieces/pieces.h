#pragma once
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/spec/spec.h>
#include <stdexcept>
#include <string>

namespace interp_pieces {
inline caliper_builder::spec::Piece load(const std::string& name) {
    namespace cp = caliper_builder::programs;
    namespace cs = caliper_builder::spec;
    const std::string prefix = "pieces/" + name + "/1/";
    cs::Piece piece;
    for (const auto& file : cp::library_files()) {
        if (file.path != prefix + "piece.toml") continue;
        const auto parsed = cs::piece_from_toml(file.text);
        if (!parsed.ok()) throw std::runtime_error("Invalid manifest: " + name);
        piece = parsed.piece;
    }
    if (piece.name.empty()) throw std::runtime_error("Missing piece: " + name);
    for (const auto& file : cp::library_files())
        if (file.path.substr(0, prefix.size()) == prefix && file.path != prefix + "piece.toml")
            piece.files[std::string(file.path.substr(prefix.size()))] = file.text;
    piece.hash = cp::piece_hash(piece);
    return piece;
}
}
