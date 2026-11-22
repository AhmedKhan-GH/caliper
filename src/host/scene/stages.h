#pragma once
// The viewer's stages in the scene compile path (design/gpu-programs.md,
// Viewer Stage Interface and Stage Declaration).
//
// A stage replaces the material function or the vertex function of the draw
// of one entity. The spec runtime compiles each stage of a scene panel: it
// runs the sandbox checks (caliper_builder/programs/sandbox.h), composes the
// WGSL of the entity's built-in program with stage_prelude() of the sandbox,
// the stage's source and stage_entries() below, translates it for the
// backend with caliper-wgsl, makes the program through the graphics layer,
// and gives it to the compiler as a StageProgram. The compiler asks a
// StageLookup for each entity's program while it compiles a draw list and a
// pick list, and uses it in place of the built-in program of the primitive,
// in both passes, thus a hooked entity still picks and selects.
//
// A primitive takes stages only where stage_entries() gives entry points; the
// first version gives them for points and lines (the parts of a geometry
// output). Threads: the frame thread, as the compiler. Nothing here throws.
#include "gfx/gfx.h"
#include "scene/scene.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace caliper_scene {

// The kinds of a stage, as spec::StageKind.
enum class StageKind : uint8_t { Material = 0, Vertex = 1 };

// The program of one entity with its stages in place. The spec runtime owns
// the program; it stays valid until the runtime gives a new StageProgram for
// the entity or drops it, after the frames that used it completed.
struct StageProgram {
    caliper_gfx::Program program;   // made through the layer from the composed shader
    std::string vs;                 // the vertex entry point in it
    std::string fs;                 // the fragment entry point of the draw pass
    std::string fs_pick;            // the fragment entry point of the pick pass
    bool material = false;          // the program holds a material stage
    bool vertex = false;            // the program holds a vertex stage
    uint64_t source_hash = 0;       // of the composed source, for the runtime's cache
};

// The stages of a scene panel, which the compiler asks while it compiles.
class StageLookup {
public:
    virtual ~StageLookup() = default;
    // The stage program of the entity at `path` for this primitive, or null to
    // keep the built-in program.
    virtual const StageProgram* find(std::string_view path, Primitive primitive) const = 0;
};

// The WGSL entry points that the viewer puts after a stage's source for one
// primitive: a vertex entry point that calls vertex() when `vertex` is set,
// and fragment entry points of the draw and the pick pass that call
// material() when `material` is set, over the bindings and the vertex inputs
// of that primitive's built-in program. Empty for a primitive that takes no
// stage. Defined in stages.cpp.
std::string stage_entries(Primitive primitive, bool material, bool vertex);

// The entry point names that stage_entries() gives.
inline constexpr const char* kStageVertexEntry = "vs_stage";
inline constexpr const char* kStageFragmentEntry = "fs_stage";
inline constexpr const char* kStagePickEntry = "fs_stage_pick";

}  // namespace caliper_scene
