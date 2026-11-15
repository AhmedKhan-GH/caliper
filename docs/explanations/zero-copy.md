# Zero-copy tensor visualization

This page explains how a tensor goes from a training step to pixels without a CPU round trip, on each platform. It
covers where the bytes live, when it is safe to read them, how many times they are copied on each path, and what the
fallback costs. The renderer seam it relies on is described in [Rendering](rendering.md), and the Windows
interop machinery in [Vulkan and CUDA on Windows](vulkan-cuda.md).

## What the words mean

Two phrases are used, and they are kept apart ([D33](../decisions/d033-honest-fallbacks.md)).

- **Zero-copy** means the tensor's data is copied nowhere: not into host memory and not to a second place in GPU
  memory. A GPU pass may still convert it (a colormap, a buffer-to-texture write), because converting is work on the
  data rather than a copy of it.
- **GPU-resident, no CPU staging** means the data never reaches host memory but is copied at least once inside the GPU.

Status lines in applets and in the host's log use the same words. An applet claims "zero-copy" only when the zero-copy
path actually ran that frame.

## How a tensor reaches a texture

An applet never talks to a graphics API. It fills a `CaliperTensor`, a plain C struct whose `device` field says where
the bytes live, and hands it to the [`tensor_bridge.v1`](../references/service-tensor-bridge-v1.md) service. The
backend behind the host turns it into a texture in whatever way that platform does best, and returns an
`ImTextureID` for `ImGui::Image`. The contract says nothing about Metal or Vulkan, which is why `CALIPER_DEV_CUDA`
was in the device enum before the Windows backend existed.

### Apple Silicon

Apple Silicon has unified memory. The CPU and GPU share one pool of physical RAM, so the storage of a PyTorch MPS
tensor is already an `MTLBuffer`, the object Metal renders from. There is no separate GPU memory to copy out of. The
handoff is a pointer cast plus the rules that make the cast safe:

1. The worker's training kernels write the weight buffer.
2. The adapter checks that the tensor is contiguous and has `storage_offset() == 0`. The ABI has no offset channel, so
   a view would address the wrong texels, and the adapter rejects views instead of guessing. The fresh result of a GPU
   operation always qualifies.
3. The bridge casts the pointer to `id<MTLBuffer>`. No bytes move.
4. A GPU pass reads the buffer in place and writes the `MTLTexture`: a compute kernel with the colormap table for f32
   data, a blit for u8.
5. The frame thread draws the texture with `ImGui::Image`, and the pixels appear that frame.

### Windows with an NVIDIA GPU

A discrete GPU has no unified memory. VRAM and system RAM are separate and joined by PCIe, so zero-copy here means
keeping the data in VRAM and letting the compute API and the graphics API reach the same memory instead of bouncing
through system RAM.

Torch's caching allocator does not produce exportable allocations, so Vulkan cannot import a torch tensor. The
Windows backend instead lets a CUDA kernel in torch's own context write the Vulkan image. The context is the device's
primary CUDA context, so a kernel launched by Caliper can read any torch tensor through its device pointer. Only the
output image has to be visible to both APIs, and Caliper allocates that itself, as exportable memory. The full ladder
of paths, the synchronization with semaphores and the measured costs are in
[Vulkan and CUDA on Windows](vulkan-cuda.md).

The host stays free of the CUDA toolkit. It loads the CUDA driver API from `nvcuda.dll` at run time, and the Vulkan
loader comes from the GPU driver.

### The OpenGL fallback

On the OpenGL backend, or for any CPU tensor, the bridge stages through host memory. It colormaps on the CPU into an
RGBA scratch buffer and uploads once with `glTexSubImage2D`. The ABI, the textures and the tests are the same, and
only the copy that the other paths avoid is there. Windows also falls back to this path when Vulkan initialization or
interop fails at run time. The applet code and the on-screen result are identical, and the cost is the copy. The worst
case is a working slow path, never a broken one.

## Data crossings per path

| Path | Host copies | Copies inside the GPU | What does the work |
|---|---|---|---|
| Metal, any MPS tensor | 0 | 0 | One compute pass (f32 with a colormap) or blit (u8) reads the buffer in place |
| Metal, imported allocation | 0 | 0 | The pass reads the imported `MTLBuffer` at a byte offset |
| Vulkan with CUDA, any CUDA tensor, direct path | 0 | 0 | A CUDA kernel reads the tensor in place and writes the image |
| Vulkan with CUDA, any CUDA tensor, buffer path | 0 | 1 | A copy into the shared buffer, then a Vulkan pass. Used when the direct path refuses the tensor and for the offscreen canvas |
| Vulkan with CUDA, `alloc_shared` | 0 | 0 | The applet's kernels write the texture's own buffer; a Vulkan pass converts it |
| Vulkan with CUDA, tensor in an exportable pool | 0 | 0 | The bridge imports the pool block once; the pass reads it at a byte offset |
| Metal and Vulkan with CUDA, imported geometry (`geometry.v1` to `v1_3`) | 0 | 0 | Indexed triangles, lines and points are vertex-pulled in place from imported buffers |
| OpenGL, or any CPU tensor | 1 (staging) | 0 | The bridge converts on the CPU and uploads |

`alloc_shared` differs by platform. On Vulkan it returns a device-backed tensor, so a CUDA kernel can write the
texture's memory in place. On Metal it returns a CPU vector in unified memory: that is zero-copy for CPU writers,
while a device writer goes through the normal update path.

## The sync ladder

The data path says where the bytes live. The sync ladder says when it is safe to read them. There are two rungs,
negotiated at run time. The applet asks the bridge what the host honors, and the adapter does the right thing:

```cpp
uint32_t caps = bridge.caps();                          // caliper.tensor_bridge.v1_1
auto ct = caliper::adapters::stream_to_tensor(t, caps);
```

- **Drain** (`stream == NULL`). The adapter synchronizes the whole producer device before the handoff, with
  `torch::mps::synchronize()` or `torch::cuda::synchronize()`, so the renderer can read the tensor immediately. This is
  always correct and costs a full-device barrier. It is what `synced_to_tensor` does, and what `stream_to_tensor`
  falls back to when capability bit 0 is absent.
- **Stream-ordered** (capability bit 0, `CALIPER_BRIDGE_CAP_STREAM_ORDERED`). There is no drain. The adapter puts the
  producer's queue or stream in `CaliperTensor.stream` (an `MTLCommandQueue*` on MPS, a `CUstream` on CUDA), and the
  renderer orders its work on the GPU after the producer's queued work. Metal uses a per-texture `MTLSharedEvent`.
  Vulkan with CUDA uses a shared timeline semaphore that rides the producer's stream. The CPU never waits on the hot
  path ([D24](../decisions/d024-stream-ordered-handoff-through-caps.md)).

Metal always grants the stream-ordered rung. Vulkan grants it when the device exports timeline semaphores; where it
does not, the handoff stays synchronous and adapters keep draining. OpenGL does not grant it.

Three findings shape the adapter:

1. **Torch's public MPS stream calls are not serialized.** In the libtorch this repository builds against,
   `deviceSynchronize` and `commitStream` are straight-line Objective-C message sends with no dispatch. Calling the
   handoff while the training thread encodes kernels corrupted command-buffer state and aborted or crashed the
   process in three different ways. Both adapter rungs therefore run their MPS work as one block on torch's own
   stream dispatch queue, and any later adapter code that touches MPS has to do the same.
2. **CUDA needs no such serialization.** Driver calls are thread-safe by API contract. That was the same kind of
   assumption that failed on MPS, so a stress test pins it: 500 handoffs on a pool stream against a concurrently
   training thread, in `tests/test_torch_adapter.cpp`.
3. **A NULL stream from a CUDA producer can still be an honored handoff.** `CUDAStream::stream()` returns `nullptr`
   for the legacy default stream. The renderer's NULL rung enqueues on that same default stream, so ordering holds
   and the drain stays elided. A test that wants to prove the stream channel is live has to pin a non-default pool
   stream.

### Geometry stays on the drain rung

The `geometry.v1` calls (`draw_points`, `draw_primitives`) address device memory as an allocation id and a byte
offset. The ABI has no `CaliperTensor` and so no `stream` field, so a stream-ordered handoff is impossible there. It is
also not needed. Correctness rests on two invariants that the applet owns, and both are required:

1. **The drain, before publishing.** Every applet worker drains its device before it flips the ready slot under the
   publish mutex. By the time the frame thread learns which slot to draw, the producer's writes are retired, so
   there is no in-flight producer work for a semaphore to order against. The renderer only has to make completed
   writes visible to the vertex stage: a memory barrier in a fenced submission on Vulkan, command-queue commit order
   on Metal.
2. **The triple buffer, for slot stability.** The worker writes the slot that is neither the ready slot nor the
   display slot, so it never rewrites the slot the frame thread is reading in place.

The triple buffer alone would let the frame read a slot whose producer writes are still in flight. The drain alone
would let the worker overwrite a slot while it is being read. Together they make the in-place path safe without the
stream-ordered gate that the texture path uses. The rule is part of the contract in the
[geometry reference](../references/service-geometry-v1.md), so a new worker-to-frame publish path inherits it.

### What the stream-ordered rung buys

Eliding the drain did not change training speed. On an RTX 500 Ada, embed_scope's steps per second differed by about
zero between the drained and the stream-ordered handoff, because the training loop's own per-step `loss.item()` call
already synchronizes. The gain is that the frame thread stops stalling on the training queue, and that the ordering is
correct by construction.

The biggest frame-thread stall that was measured had nothing to do with tensors. A database table rebuilt on the UI
thread froze the frame, which is why the host forbids I/O there ([D39](../decisions/d039-no-io-on-the-frame-thread.md)).
Two habits keep texture work cheap: create a texture once and call `update_texture` afterwards, because a create costs
about a millisecond where an update costs a fraction of one.

## Imported allocations

A texture update normally reads a tensor wherever it lies. Imported allocations go one step further: the applet
exports a block of device memory once, the host imports it once, and from then on both the texture path and the
geometry path address the memory as an allocation id plus a byte offset. This is
[`caliper.tensor_bridge.v1_2`](../references/service-tensor-bridge-v1.md), and it is opt-in per allocation
([D34](../decisions/d034-opt-in-imported-allocations.md)).

**Windows.** An applet allocates its torch CUDA tensors from `caliper::adapters::ExportablePool`, a torch `MemPool`
whose blocks are created with `cuMemCreate` and a shareable handle. The host imports each block into Vulkan on first
use, and `update_texture_from_alloc` runs the colormap or blit pass directly on the imported buffer at the tensor's
offset. Memory that was born in the pool needs no copy. Memory that was born unshareable, such as torch's default
caching allocator, takes the ordinary path of the update ladder.

**Metal.** Handle kind `CALIPER_ALLOC_HANDLE_MTLBUFFER` carries an in-process `id<MTLBuffer>`. There is no OS handle
and no driver export, because unified memory means the pointer already is the resource. The renderer keeps a strong
reference as the host's copy of the handle, checks that the buffer lives on the renderer's own device, and declines a
buffer that is too short. `ExportablePool` on MPS makes any tensor eligible without an allocator change: its
`to_bridge` imports the tensor's own storage buffer. The tensor must stay alive until the pool is destroyed or the
import is released, because torch's MPS allocator may otherwise hand the same buffer to an unrelated tensor while the
host still draws from it.

Every miss returns `false` and the caller falls back to the copying path: no capability bit, an import the host
declined, a misaligned offset, a window that runs past the allocation, a released allocation. A failed import is
never a crash or a wrong image. The import memory-stability contract applies: the bytes must not be rewritten until
the next update of the same texture.

## Derived graphics

The texture route is the narrowest use of this pipeline. An applet can compute anything from the live training state,
on the GPU, mid-training, and turn the result into the graphic that carries the insight. Image-like results (heatmaps,
feature maps, attention grids) go through the bridge as textures. Coordinates go through ImPlot and ImPlot3D as small
arrays, which cost kilobytes and need no zero-copy path. Tokens and labels go through styled text. The routes compose
in one panel, and the zero-copy path matters for the dense outputs.
