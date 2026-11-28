# Adapters

The adapter headers are instrument-side helpers. They are header-only, and the host never includes them. They cover the torch adapter, the allocation registry, the exportable pool, the OBJ loader and the orbit camera.

## `torch.hpp`

`caliper/adapters/torch.hpp` is a header-only bridge from a `torch::Tensor` to the frozen [`CaliperTensor`](tensor-and-arrow-headers.md#tensor-header). It compiles against the libtorch of the instrument, so torch stays out of the ABI. The host links no torch. It hands the `CaliperTensor` to [`caliper.tensor_bridge.v1`](services/caliper-tensor-bridge-v1.md).

The adapter has three entry points:

| Entry point | Sync behavior | Use when |
|---|---|---|
| `to_tensor(t)` | none | CPU tensors, or you own the ordering yourself |
| `synced_to_tensor(t)` | full device drain before handoff | device tensors on a v1 host (always correct, costs a barrier) |
| `stream_to_tensor(t, caps)` | drain or stream-ordered handoff, negotiated | device tensors. Pass `bridge.caps()` and get the best rung the host honors |

### Copy Rejection

The adapter never silently copies. Each entry point returns `std::optional<CaliperTensor>`. On success, the value aliases the memory of the tensor with zero copies. When the tensor cannot be a v1 `CaliperTensor`, the entry point returns `std::nullopt` and copies nothing. The adapter rejects a tensor for these reasons:

- An unsupported dtype. Only `f32`, `f16`, `bf16`, `i64`, `i32` and `u8` map. It rejects `f64`, `bool`, complex, quantized and fp8.
- `ndim > 8`, because the frozen shape and strides arrays hold 8 entries.
- A non-contiguous layout, because the bridge assumes row-major order with no gaps.
- An MPS view with a nonzero storage offset (see below).
- A device other than CPU, MPS or CUDA.

The caller repairs the tensor in instrument code, for example with an explicit `.contiguous()` before the next call. Thus the cost of a copy is visible where the instrument pays it.

### MPS Offset and Contiguity

An MPS tensor must also have `storage_offset() == 0`. The frozen `CaliperTensor` has no storage-offset channel, and the bridge casts `storage().mutable_data()` straight to an `id<MTLBuffer>`. A view with a nonzero offset would start the buffer at the wrong address and silently address the wrong texels. The adapter rejects such a view. The caller then clones it with `.clone()`, or with `.contiguous()` of a materialized tensor, so the buffer starts at offset 0.

A per-kernel `select()` slice of an `(8,1,3,3)` weight has this problem. The worker takes an owned clone of each `(3,3)` filter, which also decouples the snapshot from the live weight.

### Windows CUDA Tensors

A CUDA tensor maps to `CALIPER_DEV_CUDA`, with `data` set to the device pointer. The Vulkan backend imports it through external-memory interop (see [Zero-copy](../python/zero-copy-tensors.md)). The adapter compiles the CUDA branch under this guard:

```cpp
#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>) && \
    __has_include(<cuda_runtime_api.h>)
```

The check of the c10 header alone is not enough, because the mac libtorch and the Windows CPU libtorch ship the c10/cuda headers without the CUDA toolkit headers.

### `synced_to_tensor` Cost

`synced_to_tensor` is `to_tensor` preceded by a device drain. Thus the texture that the host uploads this frame reflects every kernel that the instrument has enqueued.

`torch::mps::synchronize()` and `torch::cuda::synchronize()` are full device barriers. They block the calling CPU thread until the device drains. Pay this cost once at the handoff and not at each frame. CPU tensors need no sync, so the adapter skips the drain for them.

### `stream_to_tensor` Handoff

```cpp
const uint32_t caps = bridge.caps();          // tensor_bridge.v1_1, additive
auto ct = caliper::adapters::stream_to_tensor(t, caps);
```

When the host grants `CALIPER_BRIDGE_CAP_STREAM_ORDERED` (caps bit 0), the adapter skips the drain. It publishes the queue of the producer in `CaliperTensor.stream` instead. This value is an `MTLCommandQueue*` on MPS and a `cudaStream_t` on CUDA. The renderer orders its copy on the GPU after the queued work of the producer, so no CPU thread waits. Without the caps bit, `stream_to_tensor(t, 0)` is the same as `synced_to_tensor`, with byte-identical v1 behavior.

On CUDA, a NULL `stream` can be an honored handoff. The default stream of torch has the handle `nullptr` (the legacy default stream), and the NULL rung of the renderer uses that same stream. Only a producer on a non-default stream carries a non-NULL handle. Do not write assertions that assume otherwise.

### MPS Serialization

The public MPS stream calls of torch are not internally serialized. Command buffers corrupt when the frame thread drains while the training thread encodes. Thus the adapter runs the whole MPS portion of both rungs, the v1 drain and the stream handoff, as one block on the own stream dispatch queue of torch. This has two results for instrument authors:

1. Any thread can call the adapter safely. But any raw MPS or Metal call that you add must also use that dispatch queue.
2. CUDA has no such rule, because driver calls are thread-safe by contract.

## `alloc_registry.hpp`

`AllocRegistry` maps a raw device pointer back to the exportable allocation that contains it, so an adapter can hand the bridge an `(alloc, offset)` pair instead of a bare pointer. It is plain C++ with no CUDA dependency and works on every platform.

- `add(base, size, os_handle)` records an allocation by its base address, and `remove(base)` forgets it.
- `find(p, extent_bytes)` returns a `Hit` with the operating-system handle, the allocation size, the byte `offset` of `p` inside it and the `base`. It returns nothing when `p` lies outside every recorded allocation or when fewer than `extent_bytes` remain after `p`.

Every call takes one mutex, so any thread may use a registry. `ExportablePool` owns one.

## `exportable_pool.hpp`

`ExportablePool` gives an instrument tensors that the bridge can import, so a texture updates with zero copies of the data. It is instrument-side only, because the [host links no ML framework](../introduction/architecture.md#design-decisions). Every failure degrades to "not pool-backed", and the caller falls back to the `stream_to_tensor` and `update_texture` path.

The class has the same public surface on every platform, and three implementations behind it:

- **CUDA** (not macOS, with the c10 and CUDA runtime headers present). The pool is a `CUDAPluggableAllocator` in a `c10::cuda::MemPool`. It creates each block with a shareable handle type and registers the block in the `AllocRegistry`. It needs no CUDA toolkit link, and torch owns the CUDA context.
- **MPS** (macOS). Unified memory means that the storage of every MPS tensor already is an `id<MTLBuffer>`, so `use()` does nothing and `to_bridge` imports the storage buffer of the tensor. A tensor passed to `to_bridge` must stay alive until the pool is destroyed or the import is released. Otherwise the MPS caching allocator could hand the same buffer to an unrelated tensor while the host still renders from it. The `storage_offset` of a tensor travels in the returned `offset`.
- **Neither.** Instrument code still compiles, but `ok()` is `false`, `use()` does nothing and `to_bridge` declines.

The members an instrument uses:

| Member | Meaning |
|---|---|
| `ExportablePool(device_index)` | Create the pool. `ok()` reports whether it is backed. |
| `use()` | Returns a `Scope`. Tensors allocated while the `Scope` is alive land in the pool. The `Scope` cannot be copied or moved, so the begin and end calls balance exactly once. |
| `to_bridge(bridge, tensor)` | Find the tensor's block, import it into the bridge once (cached by base address, and a refused import is cached too), and return a `BridgeRef` of `{alloc, offset}`. Returns nothing when the tensor is not in the pool or the host declined. |
| `registry()` | The pool's `AllocRegistry`. |

The returned `BridgeRef` goes to the `*_from_alloc` calls of [`caliper.tensor_bridge.v1_2`](services/caliper-tensor-bridge-v1.md). Release and import calls belong to the frame thread, and the destructor of the pool releases the imports it made. Torch may free a block from any thread. Thus the pool queues the release of the import of that block and drains the queue on the frame thread, in `to_bridge` and in the destructor.

An ML instrument allocates its drawn GPU buffers this way. The platform differences are in [Torch per Platform](torch-per-platform.md).

## `obj.hpp`

`caliper::obj` is a small Wavefront OBJ loader, with no dependency beyond the standard library.

- `load(std::istream&, Mesh&, std::string* error)` and `load_file(path, Mesh&, std::string* error)` return `true` on success. On failure they return `false`, leave the output empty and write a message of the form `OBJ line N: reason`. They reject malformed positions, texture coordinates, normals and face indices, out-of-range or relative indices, faces with fewer than three vertices and a file with no faces.
- `Mesh` holds `positions` (vertex count by 3), `normals` (by 3), `uvs` (by 2) and triangulated `indices`, plus `has_normals` and `has_uvs`. The loader deduplicates vertices on their position, texture and normal index triple, so a mesh with hard edges or seams has more vertices than the file has positions. A missing normal or UV reads as zero, and the flags say whether the whole mesh had them. The loader fan-triangulates polygons.

The loader does only OBJ. glTF and other asset formats are not supported.

## `orbit_camera.hpp`

Header-only orbit-camera math in `caliper::adapters`, so that drawing and picking use one eye computation and a pick ray lands where the draw put the pixel.

- `V3` with `+`, `-`, scalar `*`, `dot`, `cross` and `normalized`. A zero-length vector normalizes to world up, so no call produces NaN.
- `look_at(eye, at, up, m)` writes a right-handed, column-major view matrix.
- `perspective(fovy, aspect, near, far, m)` writes a Vulkan-style projection with depth in `[0, 1]`. It does not flip Y, because the negative-viewport convention of the backend supplies the up direction.
- `orbit_eye(azimuth, elevation, distance, target)` returns the eye position. The azimuth sweeps the horizontal ring, the elevation lifts toward world up and the distance is the radius.
- `cursor_ray(eye, target, fov_deg, aspect, ndc_x, ndc_y)` returns the normalized world-space ray through a point in normalized device coordinates, with +y up.
- `kOrbitPi` is the constant the camera code uses.

## `torch.hpp` Source

```cpp
--8<-- "sdk/include/caliper/adapters/torch.hpp"
```
