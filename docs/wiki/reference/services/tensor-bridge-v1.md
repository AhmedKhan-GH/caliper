# caliper.tensor_bridge.v1

Service id `caliper.tensor_bridge.v1` — the framework's USP, productized: a
`CaliperTensor` becomes a live texture this frame, GPU-resident on the native
backends (PLATFORM.md §7.4). This page embeds the header verbatim; the docs
build fails if the file moves.

```c
--8<-- "sdk/include/caliper/services/tensor_bridge_v1.h"
```

## Semantics

The bridge turns a `CaliperTensor` into a live texture **this frame**. It is the
framework's reason to exist: on the native backends the tensor's device memory
*becomes* the texture with no CPU round-trip; the ABI never names a graphics API,
so the renderer stays swappable forever (see [Rendering](../../explanation/rendering.md)).

### Acceptance rules (the v1 gate)

The host validates every tensor before it becomes a texture. A violation returns
`0`/`false` and emits one `caliper.log.v1` line — the bridge **never**
misinterprets bytes into a wrong texture. Faithful to the shipped gate:

- **2-D `(H,W)` `f32`** → `texture_from_tensor_mapped` (colormapped through a
  built-in LUT, scaling `[vmin,vmax]` → `[0,1]`);
- **3-D `(H,W,C≤4)` `u8`** → `texture_from_tensor` (direct RGBA/…);
- **contiguous** (row-major, no gaps);
- **device CPU or the active backend's device** (Metal on macOS, CUDA on Windows).

The applet-side [torch adapter](../../reference/adapters.md) enforces the mirror
of these rules before the tensor ever reaches the host — rejecting rather than
silently copying.

### UI-thread-only

Every bridge entry point is **frame-thread-only**. Unlike
[`caliper.metrics.v1`](metrics-v1.md) (callable from a job thread), the bridge
touches renderer state and must be called only from the applet's `frame()`. In
the usual pattern the training worker *snapshots* weights under a mutex and
never calls the bridge; the frame thread reads that snapshot and does every
upload. Textures are therefore frame-thread-owned and released on the frame
thread (after the job wait, before renderer teardown).

### Per-backend behavior

The applet code is **identical** on both backends. Where the staging happens
differs, and the honest device-path string tells you which ran:

| Backend | Path | Staging | Status |
|---------|------|---------|--------|
| **Metal** (`CALIPER_RENDERER=metal`) | device **compute** (f32 + LUT) / **blit** (u8 HWC) | **zero CPU staging** — the MPS `MTLBuffer` is colormapped on-GPU | §16-verified pixel-exact vs a CPU reference (C5) |
| **Vulkan** (Windows default) | device **compute** / **blit** through Vulkan-CUDA external memory | **zero CPU staging** — one device-to-device copy inside VRAM at most | Hardware self-test verified byte-exact; gfx-harness CI wiring remains pending |
| **GL** (`CALIPER_RENDERER=gl`) | CPU-staged upload | the *bridge* stages; the applet never touches a pixel | §16-verified pixel-exact; `tex_update_from_device` always returns `false` (frozen fallback) |

The **§16 contract** is tested by uploading known tensors, reading the texture
back, and comparing byte-for-byte. Metal and GL run through
`caliper_gfx_tests`; Vulkan has the equivalent hardware self-test, with its
integration into the gfx-harness CI matrix still pending. Metal's `compute`
path is exact vs the CPU `map_f32_to_rgba8` reference at ragged sizes (4×4, 5×3,
17×9); the `blit` path is exact vs `expand_u8_to_rgba8`.

!!! note "One tensor cannot be both"
    A single tensor is either zero-copy on Metal *or* accepted on GL, not both:
    the GL bridge's active device is CPU and rejects an MPS tensor as a foreign
    device. The exemplar hands the training-device tensor first; if the create
    returns `0` (non-Metal renderer) it relocates the tensor to CPU and the
    bridge stages it. The applet does no pixel work on either path (§6c) — the
    bridge's own accept/reject drives the choice.

### Lifecycle: create once, update after

- `texture_from_tensor` / `texture_from_tensor_mapped` **create** a texture and
  return a `CaliperTextureId`.
- `update_texture` **re-uploads** into an existing texture of the **same
  shape/dtype**. Create once on the first snapshot, `update_texture` thereafter —
  do not recreate per frame.
- `release_texture` frees it.

!!! warning "Pinned range in v1 (frozen)"
    `update_texture` has **no colormap-range channel** — it is frozen. The
    `[vmin,vmax]` of a colormapped texture is fixed at creation. Pin a
    symmetric RdBu range at the first snapshot; learned filters still visibly
    sharpen (structure changes, not just scale), and the UI should say so.
    Recreating the texture per snapshot to re-range would violate create-once and
    is deliberately not done.

### `alloc_shared` — v1 honesty

`alloc_shared` allocates tensor memory that **is** the texture's backing store:
the applet wraps `out_tensor->data` (e.g. `torch::from_blob`) and writes into it,
and the texture sees the result after at most a layout transition. In v1 it
returns a **unified-memory CPU-device tensor** — literal zero-copy for **CPU
writers**; **device writers** must still stage through `update_texture`. Free it
with `free_shared`.

### `CaliperTextureId` lifetime

`CaliperTextureId` is a `uint64_t`, opaque to applets (`0` = invalid;
compare-only — never interpret or dereference it, its representation is
backend-internal). Its value is directly castable to `ImTextureID` for
`ImGui::Image`: the host vends the ImGui-compatible handle per backend (the GL
texture name on GL, the `id<MTLTexture>` pointer on Metal), so the cast Just
Works on both — binding an integer table id here is what crashed
`ImGui_ImplMetal` on the first `Image`. The renderer stays swappable because the
value is the host's business, never the applet's (§5.4). Ids from
`texture_from_tensor*` are freed with `release_texture`; ids from `alloc_shared`
with `free_shared`.

### Sync model

The ABI is stream-free in v1 (`stream == NULL`). Correctness on device textures
comes from the applet draining the device **once** at the handoff
(`torch::mps::synchronize()` via the adapter's `synced_to_tensor`) — sync-then-
update — not from a stream channel. See the [adapter reference](../../reference/adapters.md)
for the cost of that barrier and why you pay it once, not per frame.

## Additive revisions: v1_1 and v1_2

Two **additive** revisions extend the bridge under the same discipline (D24): the
struct is **prefix-identical** — the same members in the same order, same
semantics — with new members appended and one new caps bit each. No ABI epoch
bump; the v1 header, table, and id are untouched and frozen. An applet negotiates
by asking the host for each id and calling `caps()`; a missing id or unset bit
means "fall back to the previous contract", never a crash.

### `caliper.tensor_bridge.v1_1` — stream-ordered handoff

Adds one query, `caps()`, over the v1-identical six operations. Bit 0,
`CALIPER_BRIDGE_CAP_STREAM_ORDERED`, set means the host honours a non-NULL
`CaliperTensor.stream`: the device update is ordered on the **producer's
stream/queue** (`CUstream` on CUDA, `MTLCommandQueue*` on Metal), so the adapter
may **skip its full device drain**. Hosts that don't vend this id — or leave the
bit unset — keep the v1 contract: the adapter drains and `stream` stays NULL.

### `caliper.tensor_bridge.v1_2` — imported device allocations

Adds three entry points over v1_1's seven members. Bit 1,
`CALIPER_BRIDGE_CAP_IMPORT_ALLOC`, set means the host can **import an
applet-exported device allocation** and run device texture updates directly *from*
it — zero copies of the tensor data:

- `import_allocation(os_handle, size_bytes, handle_type)` hands the host an OS
  shareable handle (from `cuMemExportToShareableHandle`) — or, on Apple,
  `CALIPER_ALLOC_HANDLE_MTLBUFFER`, an in-process `id<MTLBuffer>` the host retains —
  and returns a `CaliperAllocId` (`0` when the host cannot import, so the applet
  stays on the v1 D2D-copy path). The host dups the handle; the applet keeps
  ownership of its copy.
- `release_allocation(alloc)` frees the import.
- `update_texture_from_alloc(tex, alloc, offset_bytes, desc)` updates a texture
  (created first via `texture_from_tensor*`) from bytes living **inside** the
  imported allocation at `offset_bytes`; `desc` carries shape/dtype/strides/stream
  and `desc->data` is ignored. Same acceptance gates as `update_texture`; `false`
  → caller falls back. The **memory-stability contract**: the pass reads the
  imported bytes in place, so `[offset_bytes, offset_bytes + extent)` must not be
  rewritten until the next update of the same texture.

`CaliperAllocId` is an opaque `uint64_t` (`0` = invalid; compare-only). This is the
import machinery that [`caliper.geometry.v1` / v1_1](geometry-v1.md) draws
geometry from: geometry sources are `(CaliperAllocId, byte offset)` pairs into
these same imports, reusing v1.2's caches, gates, and lifecycle wholesale.
