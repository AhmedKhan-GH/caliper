# caliper.tensor_bridge.v1, v1_1 and v1_2

The service ids are `caliper.tensor_bridge.v1`, `caliper.tensor_bridge.v1_1` and `caliper.tensor_bridge.v1_2`. The bridge
turns a `CaliperTensor` into a live texture in the current frame, and the texture stays GPU-resident on the native
backends. Version v1_1 adds stream-ordered handoff, and v1_2 adds imported device allocations. This page embeds each
header verbatim, and the docs build fails if a header file moves.

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
graphics API, so the renderer stays swappable (see [Rendering](../rendering.md)). The cost of each path is in
[Zero-copy tensor visualization](../../python/zero-copy-tensors.md).

### Acceptance Rules

The host validates every tensor before it becomes a texture. A violation returns `0` or `false` and emits one
`caliper.log.v1` line. The bridge never misinterprets bytes into a wrong texture. The shipped gate accepts two input forms:

- **2-D `(H,W)` `f32`** goes to `texture_from_tensor_mapped`, colormapped through a built-in table that scales
  `[vmin,vmax]` to `[0,1]`.
- **3-D `(H,W,C≤4)` `u8`** goes to `texture_from_tensor` as direct RGBA.

The tensor must also be **contiguous** (row-major, no gaps). Its device must be CPU or the active backend's device, which
is Metal on macOS and CUDA on Windows. The applet-side [torch adapter](../adapters.md) enforces the mirror of these rules
before the tensor reaches the host. It rejects a bad tensor and never copies it silently.

### Frame Thread Only

Every bridge entry point is frame-thread-only. Unlike [`caliper.metrics.v1`](caliper-metrics-v1.md), which a job thread
may call, the bridge touches renderer state and must be called from the applet's `frame()`. In the usual pattern the
training worker snapshots weights under a mutex and never calls the bridge. The frame thread reads that snapshot and
does every upload. The frame thread owns the textures and releases them after the job wait and before renderer teardown.

### Per-Backend Behavior

The applet code is identical on every backend. The place where staging happens differs, and the device-path string that
the renderer reports says which path ran.

| Backend | Path | Staging |
|---|---|---|
| Metal (macOS default) | A device compute pass (f32 with a colormap) or a blit (u8) reads the MPS `MTLBuffer` in place | None |
| Vulkan with CUDA (Windows default) | Tried in order: a CUDA kernel reads the tensor and writes the Vulkan image (`direct-cmap`, `direct-blit`), then a copy into a shared buffer and a Vulkan pass (`compute`, `blit`), then CPU staging | None on the first two rungs, and no in-VRAM copy on the direct rung |
| OpenGL (`CALIPER_RENDERER=gl`) | The bridge converts on the CPU and uploads | The bridge stages, and the applet never touches a pixel. `tex_update_from_device` always returns `false` |

The Windows ladder is described in [Vulkan and CUDA on Windows](../../development/vulkan-and-cuda.md).

!!! note "One tensor cannot be both"
    A single tensor is either zero-copy on Metal or accepted on OpenGL, not both. The OpenGL bridge's active device is
    CPU, so it rejects an MPS tensor as a foreign device. The exemplar hands over the training-device tensor first. If
    the create returns `0`, as on a non-Metal renderer, it relocates the tensor to CPU and the bridge stages it. The
    applet does no pixel work on either path, because the bridge's accept or reject drives the choice.

### Texture Lifecycle

- `texture_from_tensor` and `texture_from_tensor_mapped` create a texture and return a `CaliperTextureId`.
- `update_texture` re-uploads into an existing texture of the same shape and dtype.
- `release_texture` frees the texture.

Create the texture once on the first snapshot and call `update_texture` after that. Do not recreate a texture per
frame, because creating a texture costs far more than updating one.

!!! warning "The range is pinned in v1"
    `update_texture` has no colormap-range channel, so the `[vmin,vmax]` of a colormapped texture is fixed at creation.
    Pin a symmetric RdBu range at the first snapshot. Learned filters still visibly sharpen, because the structure
    changes and not only the scale, and the UI should say so. The bridge does not recreate the texture per snapshot to
    re-range, because that would break create-once.

### `alloc_shared`

`alloc_shared` allocates tensor memory that is the texture's backing store. The applet wraps `out_tensor->data` (for
example with `torch::from_blob`) and writes into it. The texture sees the result after at most a layout transition.
Free the memory with `free_shared`. The result depends on the backend:

- **Vulkan with CUDA** returns a device-backed tensor on `CALIPER_DEV_CUDA`. The applet's kernels write the texture's
  memory in place, and the update runs the buffer-to-image pass with no copy.
- **Metal** returns a CPU vector in unified memory. That is zero-copy for CPU writers, but a device writer still goes
  through `update_texture`.
- **OpenGL**, and Vulkan without paired interop, return a CPU vector.

### `CaliperTextureId` Lifetime

`CaliperTextureId` is a `uint64_t`, opaque to applets. The value `0` is invalid, and the id is compare-only. Never
interpret or dereference it, because its representation is internal to the backend. Cast its value directly to
`ImTextureID` for `ImGui::Image`, because the host hands out the ImGui-compatible handle for each backend. That handle
is the texture name on OpenGL, the `id<MTLTexture>` pointer on Metal and the descriptor set on Vulkan. Free ids from
`texture_from_tensor*` with `release_texture`, and ids from `alloc_shared` with `free_shared`.

### Sync Model

v1 is stream-free, so `CaliperTensor.stream` is `NULL`. Correctness on device textures comes from the applet draining
the device once at the handoff. The applet calls `torch::mps::synchronize()` or `torch::cuda::synchronize()` through the
adapter's `synced_to_tensor`, and then calls `update_texture`. That drain is a full-device barrier, and v1_1 replaces it
with stream ordering where the host honors it.

On a host that vends `caliper.tensor_bridge.v1_1`, the `caps()` query says what the host honors. Metal honors
stream-ordered handoff, and Vulkan honors it when the device exports timeline semaphores. Where the host honors it, the
adapter skips the drain and puts the producer's stream in `CaliperTensor.stream`. Otherwise the v1 drain applies. The
adapter side and the reasons are in [Adapters](../adapters.md) and
[Zero-copy tensor visualization](../../python/zero-copy-tensors.md#sync-ladder).

## Additive Revisions

Both revisions extend the bridge under the [growth rules](../../development/compatibility-and-epochs.md#growth-rules).
Each struct keeps the same members in the same order with the same semantics, and new members go at the end. Each
revision adds one capability bit. There is no ABI epoch bump, and the earlier headers, tables and ids stay frozen. An
applet asks the host for each id and calls `caps()`. A missing id or an unset bit means the applet falls back to the
previous contract, never a crash.

### `caliper.tensor_bridge.v1_1` Handoff

v1_1 adds one query, `caps()`, to the six v1 operations. Bit 0, `CALIPER_BRIDGE_CAP_STREAM_ORDERED`, means the host
honors a non-NULL `CaliperTensor.stream`. The device update is then ordered on the producer's stream or queue (`CUstream`
on CUDA, `MTLCommandQueue*` on Metal), so the adapter may skip its full device drain. If the host does not vend this id
or leaves the bit unset, the v1 contract holds. The adapter drains, and `stream` stays `NULL`.

### `caliper.tensor_bridge.v1_2` Imported Allocations

v1_2 adds three entry points to the seven v1_1 members. Bit 1, `CALIPER_BRIDGE_CAP_IMPORT_ALLOC`, means the host can
import an applet-exported device allocation and run device texture updates directly from it, with no copy of the
tensor data.

- `import_allocation(os_handle, size_bytes, handle_type)` hands the host an OS shareable handle (from
  `cuMemExportToShareableHandle`) or, on Apple, `CALIPER_ALLOC_HANDLE_MTLBUFFER`, an in-process `id<MTLBuffer>` that
  the host retains. It returns a `CaliperAllocId`, or `0` when the host cannot import, so the applet stays on the
  earlier path. The host duplicates the handle, and the applet keeps ownership of its copy.
- `release_allocation(alloc)` frees the import.
- `update_texture_from_alloc(tex, alloc, offset_bytes, desc)` updates a texture from bytes inside the imported
  allocation at `offset_bytes`. The texture must exist already through `texture_from_tensor*`. `desc` carries shape,
  dtype, strides and stream, and the host ignores `desc->data`. The acceptance gates are those of `update_texture`, and
  `false` tells the caller to fall back. The pass reads the imported bytes in place, under the memory-stability
  contract. Thus `[offset_bytes, offset_bytes + extent)` must not change until the next update of the same texture.

`CaliperAllocId` is an opaque `uint64_t`. The value `0` is invalid, and the id is compare-only. It is the import
machinery that [`caliper.geometry.v1` and its revisions](caliper-geometry-v1.md) draw from. Geometry sources are pairs
of `(CaliperAllocId, byte offset)` into these same imports, and they reuse the caches, gates and lifecycle of v1_2. The
applet side of an import is [`ExportablePool`](../torch-per-platform.md#exportablepool-opt-in).
