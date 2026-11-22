# caliper.tensor_bridge.v1, v1_1 and v1_2

Service ids `caliper.tensor_bridge.v1`, `caliper.tensor_bridge.v1_1` and `caliper.tensor_bridge.v1_2`. The bridge turns
a `CaliperTensor` into a live texture this frame, GPU-resident on the native backends. v1_1 adds stream-ordered
handoff, and v1_2 adds imported device allocations. This page embeds each header verbatim, and the docs build fails
if a file moves.

```c
--8<-- "sdk/include/caliper/services/tensor_bridge_v1.h"
```

```c
--8<-- "sdk/include/caliper/services/tensor_bridge_v1_1.h"
```

```c
--8<-- "sdk/include/caliper/services/tensor_bridge_v1_2.h"
```

## Semantics

On the native backends the tensor's device memory becomes the texture with no CPU round trip. The ABI never names a
graphics API, so the renderer stays swappable (see [Rendering](../rendering.md)). What the paths cost is in
[Zero-copy tensor visualization](../../python/zero-copy-tensors.md).

### Acceptance rules

The host validates every tensor before it becomes a texture. A violation returns `0` or `false` and emits one
`caliper.log.v1` line. The bridge never misinterprets bytes into a wrong texture. The shipped gate:

- **2-D `(H,W)` `f32`** goes to `texture_from_tensor_mapped`, colormapped through a built-in table, scaling
  `[vmin,vmax]` to `[0,1]`;
- **3-D `(H,W,C≤4)` `u8`** goes to `texture_from_tensor`, direct RGBA;
- the tensor is **contiguous** (row-major, no gaps);
- its device is **CPU or the active backend's device** (Metal on macOS, CUDA on Windows).

The applet-side [torch adapter](../adapters.md) enforces the mirror of these rules before the tensor reaches the host,
and rejects rather than silently copying.

### Frame thread only

Every bridge entry point is frame-thread-only. Unlike [`caliper.metrics.v1`](caliper-metrics-v1.md), which a job thread
may call, the bridge touches renderer state and must be called from the applet's `frame()`. In the usual pattern the
training worker snapshots weights under a mutex and never calls the bridge, and the frame thread reads that snapshot
and does every upload. Textures are owned by the frame thread and released on it, after the job wait and before
renderer teardown.

### Per-backend behavior

The applet code is identical on every backend. Where the staging happens differs, and the device-path string that the
renderer reports says which path ran.

| Backend | Path | Staging |
|---|---|---|
| Metal (macOS default) | A device compute pass (f32 with a colormap) or a blit (u8) reads the MPS `MTLBuffer` in place | None |
| Vulkan with CUDA (Windows default) | First, a CUDA kernel that reads the tensor and writes the Vulkan image (`direct-cmap`, `direct-blit`). Then, a copy into a shared buffer and a Vulkan pass (`compute`, `blit`). Then, CPU staging | None on the first two rungs; no in-VRAM copy on the direct rung |
| OpenGL (`CALIPER_RENDERER=gl`) | The bridge converts on the CPU and uploads | The bridge stages; the applet never touches a pixel. `tex_update_from_device` always returns `false` |

Each path is tested by uploading known tensors, reading the texture back and comparing byte for byte against the CPU
reference. The `caliper_gfx_tests` suite has an environment per backend: OpenGL, Metal, Vulkan (CPU-staged rows) and
Vulkan with CUDA (device rows). The `compute` path is exact against `map_f32_to_rgba8` at ragged sizes such as 4 by 4,
5 by 3 and 17 by 9, and the `blit` path against `expand_u8_to_rgba8`. The Windows ladder is described in
[Vulkan and CUDA on Windows](../../explanations/vulkan-cuda.md).

!!! note "One tensor cannot be both"
    A single tensor is either zero-copy on Metal or accepted on OpenGL, not both. The OpenGL bridge's active device is
    CPU, so it rejects an MPS tensor as a foreign device. The exemplar hands the training-device tensor first. If the
    create returns `0`, as on a non-Metal renderer, it relocates the tensor to CPU and the bridge stages it. The
    applet does no pixel work on either path. The bridge's own accept or reject drives the choice.

### Lifecycle: create once, update after

- `texture_from_tensor` and `texture_from_tensor_mapped` create a texture and return a `CaliperTextureId`.
- `update_texture` re-uploads into an existing texture of the same shape and dtype. Create once on the first
  snapshot and call `update_texture` after that. Do not recreate a texture per frame.
- `release_texture` frees it.

Creating a texture costs far more than updating one. On the Windows backend, creating and releasing a texture at
applet sizes costs about 1 ms together, and an update costs from about 30 microseconds to 1.4 ms depending on the texture size
([measured costs](../../explanations/vulkan-cuda.md#measured-costs)).

!!! warning "The range is pinned in v1"
    `update_texture` has no colormap-range channel. The `[vmin,vmax]` of a colormapped texture is fixed at creation.
    Pin a symmetric RdBu range at the first snapshot. Learned filters still visibly sharpen, because the structure
    changes and not only the scale, and the UI should say so. Recreating the texture per snapshot to re-range would
    violate create-once and is deliberately not done.

### `alloc_shared`

`alloc_shared` allocates tensor memory that is the texture's backing store. The applet wraps `out_tensor->data` (for
example with `torch::from_blob`) and writes into it, and the texture sees the result after at most a layout
transition. Free it with `free_shared`. What it returns depends on the backend:

- **Vulkan with CUDA** returns a device-backed tensor on `CALIPER_DEV_CUDA`. The applet's kernels write the texture's
  memory in place, and the update runs the buffer-to-image pass with no copy.
- **Metal** returns a CPU vector in unified memory. That is zero-copy for CPU writers. A device writer still goes
  through `update_texture`. The Metal `alloc_shared` is not device-backed.
- **OpenGL**, and Vulkan without paired interop, return a CPU vector.

### `CaliperTextureId` lifetime

`CaliperTextureId` is a `uint64_t`, opaque to applets: `0` is invalid, and it is compare-only. Never interpret or
dereference it, because its representation is internal to the backend. Its value can be cast directly to `ImTextureID`
for `ImGui::Image`, because the host hands out the ImGui-compatible handle for each backend: the texture name on
OpenGL, the `id<MTLTexture>` pointer on Metal and the descriptor set on Vulkan. Ids from `texture_from_tensor*` are
freed with `release_texture`, and ids from `alloc_shared` with `free_shared`.

### Sync model

v1 is stream-free: `CaliperTensor.stream` is `NULL`. Correctness on device textures comes from the applet draining
the device once at the handoff, with `torch::mps::synchronize()` or `torch::cuda::synchronize()` through the adapter's
`synced_to_tensor`, and then calling `update_texture`. That drain is a full-device barrier. v1_1 replaces it with
stream ordering where the host honors it.

On a host that vends `caliper.tensor_bridge.v1_1`, the `caps()` query says what is honored. Metal honors
stream-ordered handoff, and Vulkan honors it when the device exports timeline semaphores. Where it is honored the adapter skips the drain
and puts the producer's stream in `CaliperTensor.stream`. Otherwise v1's drain applies. The adapter side and the
reasons are in [Adapters](../adapters.md) and
[Zero-copy tensor visualization](../../python/zero-copy-tensors.md#the-sync-ladder).

## Additive revisions: v1_1 and v1_2

Two additive revisions extend the bridge under one discipline, the [growth rules](../../explanations/compatibility.md#growth-rules).
The struct is prefix-identical, with the same members in the same order and with the same semantics. New members are
appended, and each revision has one new capability bit. There is no ABI epoch bump, and the earlier headers, tables
and ids stay frozen. An applet negotiates by asking the host for each id and calling `caps()`. A missing id or an
unset bit means "fall back to the previous contract", never a crash.

### `caliper.tensor_bridge.v1_1`: stream-ordered handoff

v1_1 adds one query, `caps()`, to the six v1 operations. Bit 0, `CALIPER_BRIDGE_CAP_STREAM_ORDERED`, set means the host
honors a non-NULL `CaliperTensor.stream`. The device update is ordered on the producer's stream or queue (`CUstream`
on CUDA, `MTLCommandQueue*` on Metal), so the adapter may skip its full device drain. Hosts that do not vend this id,
or leave the bit unset, keep the v1 contract: the adapter drains and `stream` stays `NULL`.

### `caliper.tensor_bridge.v1_2`: imported device allocations

v1_2 adds three entry points to the seven v1_1 members. Bit 1, `CALIPER_BRIDGE_CAP_IMPORT_ALLOC`, set means the host
can import an applet-exported device allocation and run device texture updates directly from it, with no copy of the
tensor data.

- `import_allocation(os_handle, size_bytes, handle_type)` hands the host an OS shareable handle (from
  `cuMemExportToShareableHandle`) or, on Apple, `CALIPER_ALLOC_HANDLE_MTLBUFFER`, an in-process `id<MTLBuffer>` that
  the host retains. It returns a `CaliperAllocId`, or `0` when the host cannot import, so the applet stays on the
  earlier path. The host duplicates the handle, and the applet keeps ownership of its copy.
- `release_allocation(alloc)` frees the import.
- `update_texture_from_alloc(tex, alloc, offset_bytes, desc)` updates a texture, created first through
  `texture_from_tensor*`, from bytes inside the imported allocation at `offset_bytes`. `desc` carries
  shape, dtype, strides and stream, and `desc->data` is ignored. The acceptance gates are those of `update_texture`,
  and `false` tells the caller to fall back. Under the memory-stability contract the pass reads the imported bytes in
  place, so `[offset_bytes, offset_bytes + extent)` must not be rewritten until the next update of the same texture.

`CaliperAllocId` is an opaque `uint64_t`: `0` is invalid, and it is compare-only. It is the import machinery that
[`caliper.geometry.v1` and its revisions](caliper-geometry-v1.md) draw from. Geometry sources are pairs of
`(CaliperAllocId, byte offset)` into these same imports, and they reuse the caches, gates and lifecycle of v1_2. The
applet side of an import is [`ExportablePool`](../torch-per-platform.md#opting-in-to-exportablepool).
