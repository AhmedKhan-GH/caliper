# caliper/tensor.h

The tensor interchange type (PLATFORM.md §7.2) — DLPack-aligned, frozen once shipped. This page embeds the header verbatim; the docs build fails if the file moves.

`CaliperTensor` is a **type, not a service** — a plain struct that describes a
block of memory (dtype, shape, strides in elements, device, and an optional
stream) the way DLPack does, so a torch, numpy, or mlx tensor is a cast away
rather than a copy. It is passed *into* services, not vended by one. Today its
sole consumer is [`caliper.metrics.v1`](services/metrics-v1.md), whose `image()`
takes a CPU-resident, contiguous, HWC `u8` `CaliperTensor`; the bigger use — GPU-
resident tensors crossing the ABI with no CPU staging — arrives with
`caliper.tensor_bridge.v1` in Phase 2C, which is why the layout is frozen now.

```c
--8<-- "sdk/include/caliper/tensor.h"
```
