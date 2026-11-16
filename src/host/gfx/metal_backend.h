// metal_backend.h: the Metal backend of the graphics layer (gfx.h). Apple only.
#pragma once

#include "gfx/gfx.h"

#include <memory>

namespace caliper_gfx {

// The layer on an existing id<MTLDevice> and id<MTLCommandQueue>, passed as bridged
// pointers: the renderer's own (HostRenderer::native_device and native_queue), so the layer
// creates no second device and its submits are ordered with the renderer's command buffers by
// commit order. nullptr when either is null.
std::unique_ptr<Backend> make_metal_backend(void* mtl_device, void* mtl_command_queue);

}  // namespace caliper_gfx
