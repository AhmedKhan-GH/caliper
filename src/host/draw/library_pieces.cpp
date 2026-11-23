// library_pieces.cpp: the library pieces of version 1 (library_pieces.h).
#include "draw/library_pieces.h"

#include <caliper_builder/programs/pieces.h>

#include <map>
#include <utility>

namespace caliper_draw {

namespace cs = caliper_builder::spec;
namespace cp = caliper_builder::programs;

const cs::Piece* library_piece(const std::string& name) {
    static const std::map<std::string, cs::Piece> pieces = [] {
        std::map<std::string, std::map<std::string, std::string>> folders;
        for (const cp::LibraryFile& f : cp::library_files()) {
            const std::string path(f.path);
            if (path.rfind("pieces/", 0) != 0) continue;
            const std::size_t cut = path.rfind('/');
            folders[path.substr(0, cut)][path.substr(cut + 1)] = std::string(f.text);
        }
        std::map<std::string, cs::Piece> out;
        for (auto& [folder, texts] : folders) {
            auto manifest = texts.find("piece.toml");
            if (manifest == texts.end()) continue;
            cs::PieceParse parsed = cs::piece_from_toml(manifest->second);
            if (!parsed.ok() || parsed.piece.version != 1) continue;
            cs::Piece p = std::move(parsed.piece);
            if ("pieces/" + p.name + "/1" != folder) continue;
            for (const char* file : cs::kPieceFiles)
                if (auto it = texts.find(file); it != texts.end()) p.files[file] = it->second;
            p.hash = cp::piece_hash(p);
            out.emplace(p.name, std::move(p));
        }
        return out;
    }();
    auto it = pieces.find(name);
    return it == pieces.end() ? nullptr : &it->second;
}

}  // namespace caliper_draw
