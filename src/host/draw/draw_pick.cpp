// The item map of a DrawRuntime (draw_runtime.h, design/programmable-graphics.md,
// Picking): the fields of a picked element from the CPU copy of the version
// that the frame showed. Not built yet: nothing is picked.
#include "draw/draw_runtime.h"
#include "draw/draw_runtime_impl.h"

namespace caliper_draw {

std::optional<PickedItem> DrawRuntime::pick(std::uint32_t) const { return std::nullopt; }

}  // namespace caliper_draw
