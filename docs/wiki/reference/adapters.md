# Adapters (torch → CaliperTensor)

`caliper/adapters/torch.hpp` is a **header-only** bridge from a `torch::Tensor`
to the frozen [`CaliperTensor`](tensor.md) (PLATFORM.md §7.2). It compiles
against the **applet's** libtorch and is **never** included by the host: torch
stays out of the ABI (D11). The host links no torch — it consumes only the
`CaliperTensor` this adapter produces, then hands it to
[`caliper.tensor_bridge.v1`](services/tensor-bridge-v1.md).

## Reject, never copy

The adapter's central contract: it **never silently copies**. `to_tensor`
returns `std::optional<CaliperTensor>` that aliases the tensor's memory
(zero-copy) on success, and `std::nullopt` — copying nothing — when the tensor
cannot be represented as a v1 `CaliperTensor`. Rejection reasons:

- an **unsupported dtype** (only `f32, f16, bf16, i64, i32, u8` map; `f64`,
  `bool`, complex, quantized, fp8 are rejected);
- **`ndim > 8`** (the frozen shape/strides arrays are `[8]`);
- **non-contiguous** (the bridge assumes row-major with no gaps);
- an **MPS view with a nonzero storage offset** (see below);
- a **device other than CPU or MPS**.

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

## `synced_to_tensor` — cost, honestly

`synced_to_tensor` is `to_tensor` preceded by a device drain, so the texture the
host uploads **this frame** reflects every kernel the applet has enqueued
(sync-then-update — the v1 correctness story for device textures, since the ABI
has no stream channel). The honest cost: `torch::mps::synchronize()` is a **full
device barrier** — it blocks the calling CPU thread until every MPS stream drains.
Pay it **once at the handoff, not per frame**. CPU tensors need no sync, so the
call is skipped for them.

In the exemplar the training **worker** pays this barrier once when it snapshots,
so the **frame thread** never syncs during upload.

## Full source

```cpp
--8<-- "sdk/include/caliper/adapters/torch.hpp"
```
