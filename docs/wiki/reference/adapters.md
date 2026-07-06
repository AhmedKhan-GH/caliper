# Adapters (torch → CaliperTensor)

`caliper/adapters/torch.hpp` is a **header-only** bridge from a `torch::Tensor`
to the frozen [`CaliperTensor`](tensor.md) (PLATFORM.md §7.2). It compiles
against the **applet's** libtorch and is **never** included by the host: torch
stays out of the ABI (D11). The host links no torch — it consumes only the
`CaliperTensor` this adapter produces, then hands it to
[`caliper.tensor_bridge.v1`](services/tensor-bridge-v1.md).

Three entry points, one ladder:

| Entry point | Sync behavior | Use when |
|---|---|---|
| `to_tensor(t)` | none | CPU tensors, or you own the ordering yourself |
| `synced_to_tensor(t)` | full device drain before handoff | device tensors on a v1 host (always correct, costs a barrier) |
| `stream_to_tensor(t, caps)` | drain **or** stream-ordered handoff, negotiated | device tensors; pass `bridge.caps()` and get the best rung the host honors |

## Reject, never copy

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

## The MPS offset-0 / contiguous rule

An MPS tensor must additionally have `storage_offset() == 0`. The reason is
structural, not conservative: the frozen `CaliperTensor` has **no
storage-offset channel**, and the bridge casts `storage().mutable_data()`
straight to an `id<MTLBuffer>`. A view with a nonzero offset would leave the
buffer starting at the wrong address and **silently address the wrong texels**.
So the adapter rejects it, and the caller clones the view (`.clone()`, or
`.contiguous()` of a materialized tensor) so the buffer starts at offset 0.

MLScope hits exactly this: a per-kernel `select()` slice of the `(8,1,3,3)`
weight carries a nonzero offset, so the worker takes an **owned clone** of each
`(3,3)` filter — which also decouples the snapshot from the still-mutating live
weight.

## CUDA tensors (Windows)

A CUDA tensor maps to `CALIPER_DEV_CUDA` with `data` = the device pointer; the
Vulkan backend imports it via external-memory interop (see `ZEROCOPY.md`). The
CUDA branch is compiled under

```cpp
#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>)
```

— `__has_include` alone is not enough because **mac libtorch ships the
c10/cuda headers without the CUDA toolkit headers** they include. If this
guard is ever wrong in the compiled-out direction, everything stays green
while the drain silently returns; the torch test suite carries a tripwire
case that fails loudly instead (`stream` must round-trip a non-default pool
stream).

## `synced_to_tensor` — cost, honestly

`synced_to_tensor` is `to_tensor` preceded by a device drain, so the texture the
host uploads **this frame** reflects every kernel the applet has enqueued
(sync-then-update — the v1 correctness story for device textures). The honest
cost: `torch::mps::synchronize()` / `torch::cuda::synchronize()` is a **full
device barrier** — it blocks the calling CPU thread until the device drains.
Pay it **once at the handoff, not per frame**. CPU tensors need no sync, so the
call is skipped for them.

## `stream_to_tensor` — the negotiated handoff (bridge v1.1)

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

## Thread safety — the MPS serialization rule

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

## Full source

```cpp
--8<-- "sdk/include/caliper/adapters/torch.hpp"
```
