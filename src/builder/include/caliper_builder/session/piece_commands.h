#pragma once
// The piece commands of design/programmable-graphics.md, Piece Commands:
// piece.check, piece.list, piece.show, piece.save, piece.copy, piece.update
// and piece.remove. They read the built-in library, write the workspace's
// pieces/ folder inside its path rules, and piece.save also edits the open
// spec. Not built yet: each one answers program.not_built.
#include <caliper_builder/commands/registry.h>
#include <caliper_builder/spec/spec.h>

#include <string>

namespace caliper_builder {

// Answers one piece command by the invocation's name. `spec` is a copy of the
// open spec, or empty without one; `workspace` is the workspace folder.
// Defined in piece_commands.cpp.
CommandOutput piece_command(const CommandContext& context, const spec::Spec& spec, const std::string& workspace);

}  // namespace caliper_builder
