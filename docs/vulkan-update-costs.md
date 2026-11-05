# Vulkan + CUDA Update Costs

| | |
|---|---|
| **Status** | Proposal. Nothing here is measured on the frame yet; §3 lists what is. |
| **Date** | 2026-11-04 |
| **Owner** | Ahmed Khan |
| **Scope** | Why the Windows backend does more work per tensor update than Metal, what DLPack does and does not change, and an ordered plan to close the gap. |
| **Parent** | `docs/vulkan-cuda-backend.md` (D19–D22), `docs/metal-pipelining.md` (D23–D24), `ZEROCOPY.md` |

The Windows backend is noticeably slower than Metal, and `vulkan_renderer.cpp` is much larger than `metal_renderer.mm`.
The extra code is mostly the explicit API surface Vulkan asks for. That includes swapchain, descriptors,
image layouts and exportable memory, plus the CUDA driver loader. Some of it runs on every update, and §1 lists
that part. Line numbers are against `main` at `9abcd47`.

---

## 1. What each backend does per tensor update

| Step | Metal (`metal_renderer.mm`) | Vulkan + CUDA (`vulkan_renderer.cpp`) |
|---|---|---|
| Reach the tensor's memory | `(__bridge id<MTLBuffer>)t.data`, a pointer cast (`tex_update_from_device`) | `cuMemcpyDtoDAsync` into the exportable shared buffer (`:3424-3426`), a full copy inside VRAM; skipped only for `alloc_shared` tensors |
| Order after the producer | One `MTLSharedEvent`: signal on torch's queue, wait on Caliper's (`order_after_producer`) | CUDA signals timeline `base+1` on torch's stream (`:3428-3431`), the pass waits `base+1` and signals `base+2` (`:3445-3460`), the frame waits `base+2` (`:2276`) |
| Command buffer | A new one per update, retained until it retires | One per texture, re-recorded each update (`:3436-3443`), so `retire()` (`:3389`) **blocks the frame thread** until the previous update of that texture has finished |
| LUT and descriptors | LUT buffer made per update | LUT copied to a host-visible buffer and the descriptor set rewritten per update (`:3397-3404`) |
| The pass | `cmap_f32` reads the tensor in place, writes the texture | `colormap.comp` reads the shared buffer, writes the `VkImage` |

Per update of an `H×W` f32 tensor, the copy adds `8·H·W` bytes of VRAM traffic (read plus write) on top of the
pass. For 4096 × 4096 that is 128 MiB, about 1.3 ms at 100 GB/s. That is arithmetic, not a measurement.

`retire()` blocks only when the same texture is updated again before its previous update finished. The previous
update waits for every kernel torch queued before it, so the block is as long as the training stream is busy.
An applet that updates on every frame while training runs pays it. `gpt_scope` updates only when a new probe
lands (`need` at `gpt_scope.cpp:1307`), so it rarely does.

## 2. Differences outside the update

| | Metal | Vulkan |
|---|---|---|
| Frames in flight | `CAMetalLayer` default: 3 drawables | FIFO with `kMinImageCount = 2` (`:263`, `:1808`). The frame thread blocks in `vkAcquireNextImageKHR` or `vkWaitForFences` (`:2245-2252`) after one late frame |
| Creating a texture | cheap | about 1.4 ms vs about 0.27 ms for an update (`ZEROCOPY.md`): exportable allocation plus CUDA import |
| Releasing a texture | cheap | `vkQueueWaitIdle` (`:661`) |
| Presentation | the display engine scans out the drawable | On a hybrid laptop such as the RTX 500 Ada test machine, a window presented through the integrated GPU costs a cross-adapter copy every frame. This is a hypothesis to check (§5, W0) |
| Interop failure | n/a | Falls back to the OpenGL path, which stages through system RAM. The startup log says which path is live: `[vulkan] device path: …` (`:1803`) |

## 3. What has been measured

- **Training throughput:** steps per second with the drain elided vs drained differs by about 0. Laptop thermal throttling (±15–20%) and the training loop's own `loss.item()` sync dominate (`docs/m2a-windows-verification.md` §5).
- **Texture create vs update:** about 1.4 ms vs about 0.27 ms on Vulkan + CUDA (`ZEROCOPY.md`).
- **Not measured:** frame time on Windows. Nobody has split the slowdown into the costs in §1 and §2.

## 4. Where DLPack fits

DLPack is a C descriptor: pointer, dtype, shape, strides, device, and a deleter for ownership. In Python,
`__dlpack__(stream=…)` additionally lets the consumer name a CUDA stream for the producer to order its work on.

- `CaliperTensor` already follows it (`PLATFORM.md` D3), and `CaliperTensor.stream` does the job of the stream argument.
- The Python spike already takes MPS tensors through `__dlpack__` (`SPIKE.md`).

What DLPack cannot do is make memory from torch's caching allocator importable into Vulkan. Import needs an
allocation made exportable, with a Win32 or fd handle, and `cudaMalloc` memory has none. That is why
`vulkan-cuda-backend.md` §3.5 sets one in-VRAM copy as the floor for arbitrary tensors. DLPack's value here is
framework coverage at the edges (jax, cupy, mlx through the same descriptor), not the cost of the update.

## 5. Plan

Each step can ship on its own and keeps the fallbacks that exist today.

**W0: Measure on the RTX 500 Ada.**
- CPU timers around `update_pipelined`, with a count and total of `retire()` waits over 0.1 ms.
- `vkCmdWriteTimestamp` around the colormap pass and the frame's draw.
- `cuEventRecord` around the copy.
- A frame-time histogram.
- Confirm the log says `CUDA interop OK`.
- Rerun with the app forced onto the NVIDIA GPU (Windows graphics settings, *High performance*) to test the hybrid presentation hypothesis.

Exit: a table of where frame time goes, which sets the order of what follows.

**W1: Three swapchain images.** Set `kMinImageCount = 3`, keep FIFO. This matches Metal's three drawables and lets one
late frame pass without blocking the frame thread. One constant; the gfx suite covers the swapchain.

**W2: CUDA writes the image directly.** The proposal of this document.

Caliper's CUDA context is the device's primary context (`cuDevicePrimaryCtxRetain`, `:2605`), the same one torch
uses. So a CUDA kernel launched by Caliper can read any torch tensor through its own device pointer. The only memory
that has to cross APIs is the output image, which Caliper allocates itself and can make exportable.

1. Allocate each texture's `VkImage` as an exportable dedicated allocation, and import it into CUDA once at create time.
2. Map it with `cuExternalMemoryGetMappedMipmappedArray`, and write it through a surface object.
3. Ship the colormap and the u8 blit as PTX, loaded with `cuModuleLoadData`. That is driver API only, so the host still links no toolkit (D11).
4. Each update launches one kernel on the producer's stream, after `cuWaitExternalSemaphoresAsync` on an *image free* value that the last frame sampling the image signalled. Then it signals a value that the frame submit waits on.
5. Two images per texture with the `ImTextureID` swapped, instead of the wait in step 4, if W0 shows that wait matters.

This removes:
- the in-VRAM copy, for every tensor and not only `alloc_shared` ones;
- the Vulkan compute pass;
- one semaphore hop;
- the per-update re-record, descriptor rewrite and LUT copy;
- `retire()`.

The update becomes the Metal shape: one pass that reads the tensor in place.

What it costs:
- **It reverses D19** (interop at buffer level only), so it needs a decision.
- **The colormap gets a third copy** (MSL, GLSL, PTX). It must stay byte-exact against `map_f32_to_rgba8`, and the existing gfx byte-exact tests are the gate.
- **The format is constrained:** the image must be a format CUDA arrays support (RGBA8 UNORM is).
- **There are fallbacks:** the current buffer path stays as the rung below when the image import fails, and the synchronous path stays below that.

**W3: Cheaper create and release.** Hand out textures from a small pool of exportable images by size class, and
replace `vkQueueWaitIdle` in `tex_release` with the deferred-destruction queue `vulkan-cuda-backend.md` §3.2
originally specified.

**Not planned:** a torch `MemPool` backed by a `cuMemCreate` allocator with Win32 handles, so every tensor becomes
importable into Vulkan. W2 makes it unnecessary: once CUDA does the pass, the source tensor never has to be visible
to Vulkan.

## 6. Decision for ratification

| # | Decision | Rationale / trade accepted |
|---|---|---|
| D25 | **The update pass runs in CUDA and writes the Vulkan image**, superseding D19 for the tensor-update path. The buffer-level path stays as the fallback | Removes the copy and the cross-API pass for every tensor, matching Metal. Costs a third colormap implementation held byte-exact by the gfx tests, and an image-format constraint |
