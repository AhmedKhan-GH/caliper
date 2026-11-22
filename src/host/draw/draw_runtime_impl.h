#pragma once
// The state of a DrawRuntime (draw_runtime.h), which draw_runtime.cpp (the
// render graph), draw_pick.cpp (the item map) and draw_text.cpp (the labels)
// share. Internal to src/host/draw.
#include "draw/draw_runtime.h"

namespace caliper_draw {

struct DrawRuntime::Impl {
    caliper_gfx::Backend* gfx = nullptr;
    bool webgpu = false;
    DrawStatus status;
};

}  // namespace caliper_draw
