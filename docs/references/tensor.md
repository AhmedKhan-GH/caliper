# caliper/tensor.h

The tensor interchange type, DLPack-aligned and frozen. This page embeds the header verbatim; the docs build fails if the file moves.

`CaliperTensor` is a **type, not a service**: a plain struct that describes a
block of memory (dtype, shape, strides in elements, device, and an optional
stream) the way DLPack does, so a torch, numpy, or mlx tensor is a cast away
rather than a copy. It is passed *into* services, not vended by one. The
services that take it are [`caliper.metrics.v1`](service-metrics-v1.md), whose
`image()` accepts a CPU-resident, contiguous, HWC `u8` tensor,
[`caliper.tensor_bridge.v1`](service-tensor-bridge-v1.md) and its revisions
`v1_1` and `v1_2`, which turn a tensor on any supported device into a texture,
and [`caliper.geometry.v1`](service-geometry-v1.md) through `v1_3`, which read
tensors as point and primitive data. The layout is frozen so that GPU-resident
tensors can cross the ABI with no CPU staging.

```c
--8<-- "sdk/include/caliper/tensor.h"
```
