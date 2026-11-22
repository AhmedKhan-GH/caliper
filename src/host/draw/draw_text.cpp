// The labels of a DrawRuntime (draw_runtime.h, design/programmable-graphics.md,
// Text): one label at each valid anchor of the version that the frame showed.
// Not built yet: there are none.
#include "draw/draw_runtime.h"
#include "draw/draw_runtime_impl.h"

namespace caliper_draw {

std::vector<Label> DrawRuntime::labels() const { return {}; }

}  // namespace caliper_draw
