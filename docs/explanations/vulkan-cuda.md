# Vulkan and CUDA on Windows

On Windows with an NVIDIA GPU, Caliper renders with Vulkan and shares GPU memory with CUDA, so a torch tensor reaches
the screen without passing through host memory. This page explains how the two APIs are paired, which path an update
takes, how the paths are synchronized, and what each costs. The renderer seam is described in
[Rendering](rendering.md), and the general idea in [Zero-copy tensor visualization](zero-copy.md).

The backend is `src/host/renderer/vulkan_renderer.cpp`, a Windows-only translation unit. The host loads the CUDA driver
API from `nvcuda.dll` at run time (`src/host/cuda_driver.h`) and uses the Vulkan loader that ships with the GPU
driver, so neither the CUDA toolkit nor the Vulkan SDK is needed to build or run it. A machine without an NVIDIA GPU
builds and runs the same binary; the Vulkan backend then renders without interop, and the CPU-staged paths apply.

## Pairing the devices

Vulkan and CUDA each enumerate their own devices, and a tensor is only reachable by Vulkan when both name the same
physical GPU. The backend picks a Vulkan device by preferring a discrete GPU that exports external memory. It then
walks the CUDA devices and chooses the one whose device UUID equals the Vulkan device's
(`VkPhysicalDeviceIDProperties::deviceUUID` against `cuDeviceGetUuid`). A match makes that pair the interop pair for
the session, and the CUDA side retains the device's primary context, the same context torch uses.

With no match, interop is switched off rather than falling back to device 0, because pairing with the wrong GPU
would corrupt images. This covers hybrid laptops where Vulkan picks the integrated GPU, machines without CUDA, and
software Vulkan drivers. Vulkan still renders, and the bridge CPU-stages tensors.

## Which path an update takes

Interop has two parts that work independently: the memory that is shared between the APIs, and how the work is
ordered between them. The update ladder is described first in [Rendering](rendering.md#the-update-ladder-on-windows).
This section gives the detail.

### The direct image path

The first rung gives CUDA the Vulkan image to write. Each
texture's `VkImage` is allocated as an exportable dedicated allocation, imported into CUDA once, mapped with
`cuExternalMemoryGetMappedMipmappedArray` and opened as a surface object. Each update launches one kernel on the
producer's stream that reads the tensor in place and writes the image. No Vulkan pass runs, and the tensor is not
copied.

- **The kernels ship as PTX.** `src/host/renderer/kernels/cmap_direct.ptx` is checked in, embedded into the binary at
  build time and loaded with `cuModuleLoadData`. Only the driver API is used, so no build needs the toolkit. The
  colormap exists as a third implementation, beside the GLSL and Metal ones, and is held byte-exact to
  `map_f32_to_rgba8` by the gfx tests. The kernels are compiled without fused multiply-add and without fast math,
  because either would change rounding.
- **What it accepts.** An f32 tensor of shape `(H, W)` with a colormap, 4-byte aligned, or a u8 tensor of shape
  `(H, W, 4)`, with a shape equal to the texture's. The image must be a format CUDA arrays support, and RGBA8 is.
  Anything else takes the buffer path. The caller has already checked the tensor's extent against its allocation, so a
  bad read cannot poison torch's context.
- **A second image per texture.** After a frame has sampled a texture, the next update writes a second image and the
  two are swapped, with the `ImTextureID` the applet holds kept stable. With a single image the producer's stream would
  have to wait for the frame that is sampling it, and the frame sits waiting for its swapchain image. A texture that
  has not yet been drawn is written in place and never allocates the second image.
- **The offscreen canvas stays on the buffer path**, because it composites without the frame's timeline waits.

`last_device_path()` reports `direct-cmap` or `direct-blit` for this rung. If the exportable image cannot be created,
or the CUDA import fails, the backend logs the reason once and the texture uses the buffer path.

### The buffer path

The fallback rung is the design that came first. It is interop at the level of buffers,
and the direct path supersedes it for tensor updates. Each texture has a shared `VkBuffer`, created with
`VkExportMemoryAllocateInfo` and an opaque Win32 handle, and imported into CUDA with `cuImportExternalMemory`, so one
allocation is visible as both a `VkBuffer` and a CUDA pointer. An update:

1. checks the tensor's byte extent against its CUDA allocation with `cuMemGetAddressRange`, and refuses on a miss;
2. copies the tensor into the shared buffer with `cuMemcpyDtoDAsync`, one copy inside VRAM;
3. runs a Vulkan pass that turns the buffer into the image: a compute shader that implements the colormap index rule
   for f32, or `vkCmdCopyBufferToImage` for u8 with four channels.

The paths are reported as `compute` and `blit`. Buffer-level interop never lets CUDA touch an optimally tiled image,
which avoids the tiling and alignment problems of sharing images. The direct path accepts those problems in exchange
for removing the copy and the pass.

### CPU staging

When no device path applies, the bridge converts on the CPU and uploads. This is also the path for CPU tensors.

### `alloc_shared` and imported allocations

`alloc_shared` on Vulkan returns a device-backed tensor: `HostRenderer::alloc_device_shared` makes the texture's shared
buffer the tensor's memory and returns its CUDA pointer. The applet's kernels write the texture's memory directly, and
the update runs the buffer-to-image pass with no copy. An optimally tiled image cannot alias linear buffer memory the
way a unified-memory buffer can back a Metal texture, so zero-copy here means zero copies of the data, not zero GPU
work. Without interop (CUDA unpaired) `alloc_shared` falls back to a CPU vector.

An [imported allocation](zero-copy.md#imported-allocations) is a block the applet exported with
`cuMemCreate` and a shareable handle. `import_allocation` imports it into Vulkan with `VkImportMemoryWin32HandleInfoKHR`
after duplicating the handle, and unwinds fully on failure. `update_texture_from_alloc` then reads it at a byte offset
with no copy: a descriptor offset for the f32 compute pass, a `bufferOffset` for the u8 blit, reported as
`compute-imported` and `blit-imported`. The f32 offset has to be a multiple of the device's
`minStorageBufferOffsetAlignment`, and the u8 offset a multiple of four; torch sub-allocations are 512-byte aligned, so
real offsets pass. Geometry draws from the same imports.

### Why torch's caching allocator cannot be imported

Importing memory into Vulkan needs an allocation made exportable, with a Win32 handle. `cudaMalloc` memory, which is
what torch's caching allocator hands out, has no handle. That is why Vulkan exports and CUDA imports in this backend,
and why the buffer path had one in-VRAM copy as its floor for an arbitrary tensor. The direct path removes the floor
by taking a different route: it never makes the source tensor visible to Vulkan, and only the output image crosses.
Making every torch tensor exportable, through a `cuMemCreate` pool behind all allocations, was considered and not
built, because the direct path makes it unnecessary. `ExportablePool` remains the opt-in route for tensors an applet
allocates on purpose.

### DLPack

DLPack is a C descriptor of a tensor: pointer, dtype, shape, strides, device, and a deleter for ownership. In Python,
`__dlpack__(stream=...)` also lets a consumer name the CUDA stream the producer should order its work on.
`CaliperTensor` follows the same shape, and `CaliperTensor.stream` does the job of the stream argument. DLPack does not
make caching-allocator memory importable into Vulkan. Its value here is framework coverage at the edges: jax, cupy and
mlx can reach the bridge through the same descriptor.

## Synchronization

The producer's CUDA work and Vulkan's work run on different queues, so the backend orders them on the GPU instead of
waiting on the CPU.

### Timeline semaphores

Each texture has a shared timeline semaphore, created in Vulkan with `VK_KHR_timeline_semaphore`, exported as an opaque
Win32 handle and imported into CUDA with `cuImportExternalSemaphore`. On the buffer path, CUDA's stream-ordered copy
signals `base+1`, the texture's Vulkan pass waits for it and signals `base+2`, and the frame's submission waits for
`base+2` before ImGui samples the texture. On the direct path, the kernel is ordered on the producer's stream, and
CUDA signals the value the frame submission waits for. A second timeline that each frame's submission signals, imported
into CUDA, lets a kernel that would overwrite an image wait for the last frame that sampled it.

The one remaining host wait on the buffer path is `retire()`. When the same texture is updated again before its
previous update finished, the frame thread waits, because the command buffer is re-recorded and the shared buffer is
overwritten. The direct path has no such wait.

When the producer hands over a non-NULL stream, which means the host honors the stream-ordered handoff, the copy and
the signal ride that stream and the adapter's full-device drain is skipped.

### The synchronous fallback

A device that cannot export timeline semaphores (the `timelineSemaphore` feature and the Win32 external-semaphore
extension are both required) cannot pipeline. The texture then uses the synchronous model: copy, `cuCtxSynchronize`,
and a fence-waited Vulkan submission. Correctness never depends on the optimization, and the host reports that it
does not honor stream-ordered handoff there, so adapters keep draining. The direct path needs the pipelined objects and is unavailable
in this mode.

### Build-time SPIR-V

The Vulkan shaders are compiled to SPIR-V at build time by an in-tree glslang and embedded in the binary as byte arrays. Colormaps are fixed by the byte-exact contract, so run-time code
generation would buy nothing. The build fetches the Vulkan headers and volk, so no SDK needs to be installed. All
shaders go through one CMake function, `caliper_compile_shader`, because the generated headers have to stay bit-identical
for the byte-exact guarantee to hold.

## Reading the log and the switches

The backend prints one-time lines to stderr, each tagged `[vulkan] device path:`, so a run shows which path is live.

| Line | Meaning |
|---|---|
| `sync mode: pipelined (shared timeline semaphores)` | Interop is paired and timeline semaphores work. Printed at startup |
| `sync mode: synchronous (timeline semaphores unavailable)` | Interop is paired without pipelining |
| `CUDA interop OK — zero-copy (kernel reads the tensor, writes the image)` | The first update took the direct path |
| `CUDA interop OK — GPU-resident (no CPU staging)` | The first update took the buffer path, with one in-VRAM copy |
| `CUDA interop OK — zero-copy (alloc_shared in place)` | An `alloc_shared` tensor was updated in place |
| `import path OK — zero-copy (imported allocation in place)` | An update read an imported allocation |
| `geometry path OK — primitives drawn from imported allocations in place` | Geometry drew from imported memory |
| `direct image: ... using the buffer path` | The direct path failed to set up for a texture |
| `external-memory interop unavailable`, `tensor not CUDA-resident` | The update falls back to CPU staging |

Each message prints once per process. Two environment variables change the backend's behavior:

- `CALIPER_VULKAN_NO_DIRECT` (any value) disables the direct path, so every update takes the buffer path. It selects the
  buffer path in the same binary, which makes comparisons straightforward.
- `CALIPER_VULKAN_STATS` (any value) records frame times and update stalls and prints a summary at exit, in the form
  `[vulkan] stats: N frames, ms p50 .. p99 .. max ..; updates direct .. buffer ..; kernel waits on a frame ..; retire
  blocks .. (.. ms total)`.

There is no fallback renderer: when Vulkan cannot start, the app refuses with a message that names the reason
([No fallback renderer](rendering.md#no-fallback-renderer)). See [Verify on Windows hardware](../guides/verify-on-windows.md)
for the full procedure.

## Measured costs

These measurements come from an RTX 500 Ada laptop GPU, driver 596.47, in a Release build. They are one machine and
small samples, and they describe this hardware rather than a general rule.

**Training with the direct path and the buffer path.** embed_scope trained MNIST for 20 seconds per run on torch's real
stream, with three alternating runs per path.

| | Buffer path | Direct path |
|---|---|---|
| Training steps | 1039, 1093, 968 | 1005, 1003, 1017 |
| Frame thread blocked in `retire()` | 2.5 to 3.5 s | 0 |
| Frame time p99 | 19.8 to 22.5 ms | 12.8 to 19.3 ms |
| Frame time p50 | 8.2 to 8.4 ms | 8.3 ms, 8.3 ms, 10.0 ms |

Training speed is level within run-to-run noise. Typical frame time is pinned to the display's vsync, so it does not
move. The gain is in how long the frame thread blocks and in the worst frames.

**The second image is required.** With a single image, about 97 percent of the kernels waited on torch's stream for the
frame that was sampling that image. In 30-second runs the direct path reached 880 to 897 training steps while the
buffer path finished all 1407, so a single image cost training at least 37 percent. The test suite passed with that
version; only a training run showed it. With two images the steps recover, although kernels still wait on a frame for
5 to 84 percent of updates depending on the run.

**One update, by size.** "Call" is the frame thread's time, and "done" adds the wait for the GPU. One run, from the
gfx test `Vulkan+CUDA: timing — update cost by size`.

| Size | Buffer call / done | Direct call / done |
|---|---|---|
| 256 by 256 | 57 / 265 µs | 30 / 65 µs |
| 1024 by 1024 | 52 / 312 µs | 24 / 137 µs |
| 2048 by 2048 | 70 / 730 µs | 38 / 419 µs |
| 4096 by 4096 | 100 / 1971 µs | 51 / 1391 µs |

**Creating and releasing a texture** at applet sizes costs about 1 ms together. It was 3 to 4 ms before the direct path,
because a texture that is only updated before its first frame needs neither a shared buffer nor a second image.
Release still waits for the queue to go idle.

**The handoff itself.** Eliding the adapter's drain through stream ordering changed training steps per second by about
zero on this hardware, within laptop thermal variance of 15 to 20 percent, because the training loop's own per-step
`loss.item()` already synchronizes. The verified gain is that the frame thread stops stalling, and that ordering is
correct, as [Zero-copy tensor visualization](zero-copy.md#what-the-stream-ordered-rung-buys) explains.
