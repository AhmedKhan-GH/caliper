// metal_backend.h: the Metal backend of the graphics layer (gfx.h). Apple only.
#pragma once

#include "gfx/gfx.h"

#include <cstddef>
#include <memory>
#include <string>

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

// Compiles MSL text for an id<MTLDevice> (a bridged pointer) ahead of create_program, with the
// compute pipeline of each named entry point, on any thread. A later create_program of the same
// text and fast_math on a Metal backend of that device takes the library, and its compute
// pipelines take those pipelines, instead of compiling again. The store keeps the last 8 such
// texts. False with *error when the compiler refuses the text.
bool metal_precompile(void* mtl_device, const char* msl, const char* const* compute_entries, size_t entry_count,
                      bool fast_math, std::string* error);

// The id<MTLBuffer> of a live buffer of a Metal backend, unretained; nullptr for a released or
// unknown handle or another backend. The producer hands it to the code that reads the buffer. A
// command buffer that binds it retains it, so the layer's deferred release cannot free it under a
// read.
void* metal_buffer(Backend& b, Buffer buffer);

}  // namespace caliper_gfx
