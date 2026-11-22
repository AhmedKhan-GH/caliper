// metal_backend.h: the Metal backend of the graphics layer (gfx.h). Apple only.
#pragma once

#include "gfx/gfx.h"

#include <cstddef>
#include <memory>

namespace caliper_gfx {

// The layer on an existing id<MTLDevice> and id<MTLCommandQueue>, passed as bridged
// pointers: the renderer's own (HostRenderer::native_device and native_queue), so the layer
// creates no second device and its submits are ordered with the renderer's command buffers by
// commit order. nullptr when either is null.
std::unique_ptr<Backend> make_metal_backend(void* mtl_device, void* mtl_command_queue);

// Destroys a backend without a wait for the GPU. While its last submit still runs, as a pass that
// does not end, the Metal objects of a Metal backend wait in a reaper until the driver ends that
// work. A backend destroyed otherwise waits for its last submit, which the renderer's teardown
// needs. metal_backends_reaping() is the number of such backends whose objects still wait.
void destroy_metal_backend_without_wait(std::unique_ptr<Backend> backend);
size_t metal_backends_reaping();

// The adapter of migration step 8 (design/renderer-core.md, "Migration"), until step 13 deletes the
// geometry service: the Metal renderer's old geometry code draws from the buffers the tensor bridge
// imported through the layer and into and from the layer's textures, so there is one allocation
// table. The id<MTLBuffer> of a live buffer and the id<MTLTexture> of a live texture of a Metal
// backend, unretained; nullptr for a released or unknown handle or another backend. A command
// buffer that binds one retains it, so the layer's deferred release cannot free it under a draw.
void* metal_buffer(Backend& b, Buffer buffer);
void* metal_texture(Backend& b, Texture texture);

}  // namespace caliper_gfx
