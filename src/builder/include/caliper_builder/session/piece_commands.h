#pragma once
// The sources of pieces and the piece commands of
// design/programmable-graphics.md (Piece Sources, Versions and Pins, Piece
// Commands): piece.check, piece.list, piece.show, piece.save, piece.copy,
// piece.update and piece.remove.
//
// A piece comes from one of three sources. The built-in library ships
// read-only with Caliper (programs::library_files()). The workspace holds
// <workspace>/pieces/<name>/<version>/, which only these functions write: a
// name and a version are checked before any path is built, a link anywhere
// in the pieces folder is refused, and a file is opened only after its real
// path is inside <workspace>/pieces/ (piece.bad_path). The open spec holds
// the inline pieces in its array `piece`. A saved version never changes.
//
// The editor, the app's program host and the commands resolve a program's
// pin with resolve_pin(): piece.missing when the source lacks the piece,
// piece.hash_mismatch when its hash differs from the pinned one. Nothing
// here throws, and every function is safe from any thread.
#include <caliper_builder/commands/registry.h>
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/spec/spec.h>
#include <caliper_builder/spec/validate.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {

// The pieces of the built-in library, read once from the embedded files of
// programs::library_files(). A test builds one from its own files.
class PieceLibrary {
public:
    explicit PieceLibrary(const std::vector<programs::LibraryFile>& files);
    // The library that this build embeds.
    static const PieceLibrary& builtin();
    // Every piece, by name and then version.
    const std::vector<spec::Piece>& pieces() const { return pieces_; }
    // The piece of that name at `version`, or its newest version; null when none.
    const spec::Piece* find(std::string_view name, std::optional<std::uint32_t> version) const;
    // The embedded files, for caliper-mcp's resources: path below library/ and text.
    const std::vector<programs::LibraryFile>& files() const { return files_; }

private:
    std::vector<programs::LibraryFile> files_;
    std::vector<spec::Piece> pieces_;
};

// A piece that a source holds.
struct FoundPiece {
    Error error;                                  // empty code: found
    spec::Piece piece;                            // with its hash computed
    spec::PieceFrom from = spec::PieceFrom::library;
    std::string path;                             // "pieces/<name>/<version>" in the workspace,
                                                  // "caliper://library/pieces/<name>/<version>/" in the
                                                  // library, empty for an inline piece
};

// The piece `name` of `from` at `version`, or its newest version when
// version is absent: piece.bad_name, piece.bad_path, piece.bad_manifest or
// piece.not_found.
FoundPiece find_piece(const PieceLibrary& library, const std::string& workspace, const spec::Spec& spec,
                      spec::PieceFrom from, std::string_view name, std::optional<std::uint32_t> version);

// The piece that a pin names. piece.missing when its source lacks it,
// piece.hash_mismatch when the pin has a hash and the piece's differs, and
// for an inline piece also when the piece's own `hash` differs from its
// text. A pin without a hash takes any hash of its name and version.
FoundPiece resolve_pin(const PieceLibrary& library, const std::string& workspace, const spec::Spec& spec,
                       const spec::PieceRef& pin);

// Checks an instance against its piece and the catalog: each input bound
// once with the form of its kind and, when the catalog knows the tensor, its
// dtype and rank (piece.bad_binding), a buffer input bound to a published
// buffer of another program's piece with its layout, each parameter of the
// program a parameter of the piece (program.params_mismatch), and a panel
// when the piece draws (spec.missing_key).
Error check_instance(const PieceLibrary& library, const std::string& workspace, const spec::Spec& spec,
                     const spec::Program& program, const spec::Piece& piece, const spec::SourceResolver& resolver);

// Completes program.create and program.patch of an instance, after the edit
// made `after`: resolves the pin of the program `id` (piece.missing,
// piece.hash_mismatch), writes the piece's hash into a pin without one, and
// runs check_instance(). `args` gets the hash too, so that the journal
// replays the same pin. A program of the one-pass form passes unchanged.
Error complete_instance(const PieceLibrary& library, const std::string& workspace, spec::Spec& after,
                        const std::string& id, const spec::SourceResolver& resolver, nlohmann::json* args);

// The checks of piece.check on one piece in its JSON form: the codec, the
// name and the manifest, the files, the imports and the hash, and each pass
// composed and compiled when the composer and the compiler are in this
// build. The result has the keys of piece.check; `piece` gets the decoded
// piece. A composer that is not built yet gives a warning and no pass.
nlohmann::json check_piece_json(const nlohmann::json& piece_json, bool headers, spec::Piece* piece);

// The workspace edit of piece.save: the inline piece `name` at `version`
// (its newest when absent) checked and written to the workspace, removed
// from the spec's array `piece`, and each program that pinned it pinned to
// the workspace copy with its hash. `after` is the spec to install as one
// undo step; with `dry`, nothing is written.
struct PieceSave {
    Error error;
    spec::Spec after;
    nlohmann::json result = nlohmann::json::object();
};
PieceSave save_inline_piece(const std::string& workspace, const spec::Spec& spec, const nlohmann::json& args,
                            bool dry);

// Answers one piece command by the invocation's name. `spec` is a copy of the
// open spec, or empty without one; `workspace` is the workspace folder.
// piece.save is answered by the session, which installs the spec of
// save_inline_piece() as one undo step. Defined in piece_commands.cpp.
CommandOutput piece_command(const CommandContext& context, const spec::Spec& spec, const std::string& workspace);
CommandOutput piece_command(const CommandContext& context, const spec::Spec& spec, const std::string& workspace,
                            const PieceLibrary& library);

}  // namespace caliper_builder
