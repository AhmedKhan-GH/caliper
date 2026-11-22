// The viewer's stage entry points (stages.h, design/gpu-programs.md). The
// interface is fixed; the entry points come next, and until then no
// primitive takes a stage.
#include "scene/stages.h"

namespace caliper_scene {

std::string stage_entries(Primitive, bool, bool) { return std::string(); }

}  // namespace caliper_scene
