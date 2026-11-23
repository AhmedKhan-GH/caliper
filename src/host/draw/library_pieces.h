#pragma once
// caliper_draw: the pieces of the built-in library that the host draws with
// (design/programmable-graphics.md, Library Pieces).
//
// The scene view (draw/scene_pieces.h) and the live points view (live/live_points.h) draw their
// shapes with the library pieces of version 1. This reads each one once from the files that the
// build embeds (programs/pieces.h, library_files()) with the spec library's manifest reader, puts
// its files beside the manifest and computes its hash, as the editor does for a library pin.
#include <caliper_builder/spec/spec.h>

#include <string>

namespace caliper_draw {

// The library piece `name` of version 1, with its files and hash; null when the build has none or
// its manifest does not parse. The pieces are read on the first call and live as long as the process.
const caliper_builder::spec::Piece* library_piece(const std::string& name);

}  // namespace caliper_draw
