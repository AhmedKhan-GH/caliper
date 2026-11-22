# Adapters

The headers under `sdk/include/caliper/adapters/` are applet-side helpers. They
are header-only, and the host never includes them. The first section covers the
torch adapter. The later sections cover the allocation registry, the exportable
pool, the OBJ loader and the orbit camera.

## torch.hpp

`caliper/adapters/torch.hpp` is a **header-only** bridge from a `torch::Tensor`
to the frozen [`CaliperTensor`](tensor.md) It compiles
against the **applet's** libtorch and is **never** included by the host: torch
stays out of the ABI. The host links no torch — it consumes only the
`CaliperTensor` this adapter produces, then hands it to
[`caliper.tensor_bridge.v1`](service-tensor-bridge-v1.md).

Three entry points, one ladder:

| Entry point | Sync behavior | Use when |
|---|---|---|
| `to_tensor(t)` | none | CPU tensors, or you own the ordering yourself |
| `synced_to_tensor(t)` | full device drain before handoff | device tensors on a v1 host (always correct, costs a barrier) |
| `stream_to_tensor(t, caps)` | drain **or** stream-ordered handoff, negotiated | device tensors; pass `bridge.caps()` and get the best rung the host honors |

### Reject, never copy

The adapter's central contract: it **never silently copies**. Each entry point
returns `std::optional<CaliperTensor>` that aliases the tensor's memory
(zero-copy) on success, and `std::nullopt` — copying nothing — when the tensor
cannot be represented as a v1 `CaliperTensor`. Rejection reasons:

- an **unsupported dtype** (only `f32, f16, bf16, i64, i32, u8` map; `f64`,
  `bool`, complex, quantized, fp8 are rejected);
- **`ndim > 8`** (the frozen shape/strides arrays are `[8]`);
- **non-contiguous** (the bridge assumes row-major with no gaps);
- an **MPS view with a nonzero storage offset** (see below);
- a **device other than CPU, MPS, or CUDA**.

The caller repairs the tensor in **applet code** — an explicit `.contiguous()`
before calling again — so the cost of any copy is *visible where it is paid*, not
hidden inside the bridge. This is the teaching point the exemplar exists to make.

### The MPS offset-0 / contiguous rule

An MPS tensor must additionally have `storage_offset() == 0`. The reason is
structural, not conservative: the frozen `CaliperTensor` has **no
storage-offset channel**, and the bridge casts `storage().mutable_data()`
straight to an `id<MTLBuffer>`. A view with a nonzero offset would leave the
buffer starting at the wrong address and **silently address the wrong texels**.
So the adapter rejects it, and the caller clones the view (`.clone()`, or
`.contiguous()` of a materialized tensor) so the buffer starts at offset 0.

A per-kernel `select()` slice of an `(8,1,3,3)` weight hits exactly this: it
carries a nonzero offset, so the worker takes an **owned clone** of each
`(3,3)` filter — which also decouples the snapshot from the still-mutating live
weight.

### CUDA tensors (Windows)

A CUDA tensor maps to `CALIPER_DEV_CUDA` with `data` = the device pointer; the
Vulkan backend imports it via external-memory interop (see [Zero-copy](../explanations/zero-copy.md)). The
CUDA branch is compiled under

```cpp
#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>) && \
    __has_include(<cuda_runtime_api.h>)
```

— `__has_include` of the c10 header alone is not enough because **mac libtorch
and the Windows CPU libtorch ship the c10/cuda headers without the CUDA toolkit
headers** they include. If this
guard is ever wrong in the compiled-out direction, everything stays green
while the drain silently returns; the torch test suite carries a tripwire
case that fails loudly instead (`stream` must round-trip a non-default pool
stream).

### `synced_to_tensor` — cost, honestly

`synced_to_tensor` is `to_tensor` preceded by a device drain, so the texture the
host uploads **this frame** reflects every kernel the applet has enqueued
(sync-then-update — the v1 correctness story for device textures). The honest
cost: `torch::mps::synchronize()` / `torch::cuda::synchronize()` is a **full
device barrier** — it blocks the calling CPU thread until the device drains.
Pay it **once at the handoff, not per frame**. CPU tensors need no sync, so the
call is skipped for them.

### `stream_to_tensor` — the negotiated handoff (bridge v1.1)

```cpp
const uint32_t caps = bridge.caps();          // tensor_bridge.v1_1, additive
auto ct = caliper::adapters::stream_to_tensor(t, caps);
```

When the host grants `CALIPER_BRIDGE_CAP_STREAM_ORDERED` (caps bit 0), the
adapter **skips the drain** and instead publishes the producer's queue in
`CaliperTensor.stream` — an `MTLCommandQueue*` on MPS, a `cudaStream_t` on
CUDA. The renderer GPU-orders its copy after the producer's queued work
(per-texture `MTLSharedEvent` on Metal; a shared timeline semaphore riding the
producer stream on Vulkan+CUDA), so no CPU thread waits. Without the caps bit,
`stream_to_tensor(t, 0)` **is** `synced_to_tensor` — byte-identical v1
behavior, which is also the negotiation pin the tests hold.

**CUDA nuance — NULL can be an honored handoff:** torch's *default* stream
handle is literally `nullptr` (legacy default stream), so a `t.stream == NULL`
from a CUDA producer still orders correctly — the renderer's NULL rung uses
that same default stream. Only a producer on a non-default stream carries a
non-NULL handle. Don't write assertions that assume otherwise.

### Thread safety — the MPS serialization rule

**None of torch's public MPS stream calls are internally serialized** (proven
by disassembly; the crashes were real: command-buffer corruption when the
frame thread drained or committed while the training thread encoded). The
adapter therefore runs the *entire* MPS portion of both rungs — the v1 drain
(`8b0a010`) and the stream handoff (`545a2f7`) — **as one block on torch's own
stream dispatch queue**. Two consequences for applet authors:

1. Calling the adapter from any thread is safe *on the adapter's own
   operations* — but the rule extends to you: any additional raw MPS/Metal
   calls you make must also ride that dispatch queue.
2. CUDA has no analogous rule (driver calls are thread-safe by contract), and
   this is verified empirically by a concurrency stress test rather than
   assumed — the MPS lesson was exactly that "should be safe" isn't evidence.

In the exemplar the training **worker** performs the handoff when it
snapshots, so the **frame thread** never syncs during upload.

## alloc_registry.hpp

`AllocRegistry` maps a raw device pointer back to the exportable allocation that
contains it, so an adapter can hand the bridge an `(alloc, offset)` pair instead
of a bare pointer. It is plain C++ with no CUDA dependency, and it is unit-tested
on every platform.

- `add(base, size, os_handle)` records an allocation by its base address, and
  `remove(base)` forgets it.
- `find(p, extent_bytes)` returns a `Hit` with the operating-system handle, the
  allocation size, the byte `offset` of `p` inside it and the `base`. It returns
  nothing when `p` lies outside every recorded allocation or when fewer than
  `extent_bytes` remain after `p`.

Every call takes one mutex, so any thread may use a registry. `ExportablePool`
owns one and fills it as blocks are created.

## exportable_pool.hpp

`ExportablePool` gives an applet tensors that the bridge can import, so a texture
or a geometry buffer updates with zero copies of the data. It is applet-side only,
because the [host links no ML framework](../introduction/architecture.md#platform-rules). Every failure degrades to
"not pool-backed", and the caller falls back to the `stream_to_tensor` and
`update_texture` path.

The class has the same public surface on every platform, and three
implementations behind it:

- **CUDA** (not macOS, with the c10 and CUDA runtime headers present). The header
  carries a small CUDA driver virtual-memory loader that resolves only the nine
  `cuMem*` symbols the pool needs from `nvcuda.dll` or `libcuda.so.1`, with no
  toolkit link and no `cuInit`, because torch owns the context. The pool is a
  `CUDAPluggableAllocator` whose allocation function calls `cuMemCreate` with a
  shareable handle type and registers each block in the `AllocRegistry`, wrapped
  in a `c10::cuda::MemPool`.
- **MPS** (macOS). Unified memory means every MPS tensor's storage already is an
  `id<MTLBuffer>`, so `use()` does nothing and `to_bridge` imports the tensor's
  own storage buffer. A tensor passed to `to_bridge` must stay alive until the
  pool is destroyed or the import is released, because the MPS caching allocator
  could otherwise hand the same buffer to an unrelated tensor while the host
  still renders from it. A tensor's `storage_offset` travels in the returned
  `offset`.
- **Neither.** The class exists so applet code compiles, but `ok()` is `false`,
  `use()` does nothing and `to_bridge` declines.

The members an applet uses:

| Member | Meaning |
|---|---|
| `ExportablePool(device_index)` | Create the pool. `ok()` reports whether it is backed. |
| `use()` | Returns a `Scope`. Tensors allocated while the `Scope` is alive land in the pool. The `Scope` cannot be copied or moved, so the begin and end calls balance exactly once. |
| `to_bridge(bridge, tensor)` | Find the tensor's block, import it into the bridge once (cached by base address, and a refused import is cached too), and return a `BridgeRef` of `{alloc, offset}`. Returns nothing when the tensor is not in the pool or the host declined. |
| `registry()` | The pool's `AllocRegistry`. |

The returned `BridgeRef` goes to the `*_from_alloc` calls of
[`caliper.tensor_bridge.v1_2`](service-tensor-bridge-v1.md). Release and import calls belong to the frame thread, and the pool's
destructor releases the imports it made. Torch may free a block from any
thread, so the pool queues the release of that block's import and drains the
queue on the frame thread, in `to_bridge` and in the destructor. FieldScope, FlowScope, GPTScope, SculptScope,
MeshScope, InstanceScope and TwinScope all allocate their drawn buffers this way (see
[Applets](applets.md)). The platform differences are in
[Torch on each platform](torch-on-each-platform.md).

## obj.hpp

`caliper::obj` is a small Wavefront OBJ loader, with no dependency beyond the
standard library.

- `load(std::istream&, Mesh&, std::string* error)` and
  `load_file(path, Mesh&, std::string* error)` return `true` on success. On
  failure they return `false`, leave the output empty and write a message of the
  form `OBJ line N: reason`. Malformed positions, texture coordinates, normals and
  face indices, out-of-range or relative indices, faces with fewer than three
  vertices and a file with no faces are all rejected.
- `Mesh` holds `positions` (vertex count by 3), `normals` (by 3), `uvs` (by 2) and
  triangulated `indices`, plus `has_normals` and `has_uvs`. Vertices are
  deduplicated on their position, texture and normal index triple, so a mesh with
  hard edges or seams has more vertices than the file has positions. A missing
  normal or UV reads as zero, and the flags say whether the whole mesh had them.
  Polygons are fan-triangulated.

The loader does only OBJ. glTF and other asset formats are not supported.
TwinScope loads its housing with it.

## orbit_camera.hpp

Header-only orbit-camera math in `caliper::adapters`, so that drawing and picking
use one eye computation and a pick ray lands where the draw put the pixel.

- `V3` with `+`, `-`, scalar `*`, `dot`, `cross` and `normalized`. A zero-length
  vector normalizes to world up, so no call produces NaN.
- `look_at(eye, at, up, m)` writes a right-handed, column-major view matrix.
- `perspective(fovy, aspect, near, far, m)` writes a Vulkan-style projection with
  depth in `[0, 1]`. It does not flip Y, because the backend's negative-viewport
  convention supplies the up direction.
- `orbit_eye(azimuth, elevation, distance, target)` returns the eye position:
  azimuth sweeps the horizontal ring, elevation lifts toward world up and
  distance is the radius.
- `cursor_ray(eye, target, fov_deg, aspect, ndc_x, ndc_y)` returns the normalized
  world-space ray through a point in normalized device coordinates, with +y up.
- `kOrbitPi` is the constant the camera code uses.

TwinScope uses it for both its camera and its surface picking.

## Full source of torch.hpp

```cpp
--8<-- "sdk/include/caliper/adapters/torch.hpp"
```
