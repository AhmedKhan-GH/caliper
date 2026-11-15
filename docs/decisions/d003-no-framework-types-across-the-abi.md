# D3. Torch and DuckDB types never cross the ABI

| Date | Status |
|---|---|
| 2026-06-10 | Accepted |

## Decision

No PyTorch, DuckDB or other framework type appears in an SDK header. Tensors cross the ABI as `CaliperTensor`, a struct aligned with DLPack, and tabular data crosses as Arrow C streams.

## Reason

Plain C structs survive version skew between the host and an applet, and converting at the edge is cheap. The trade is that each framework needs a small adapter on the applet side.

## Notes

The code implements this decision: `tensor.h` and `arrow_c.h` are the interchange headers, and the SDK headers contain no torch or DuckDB types (the metrics store header hides its implementation behind an opaque pointer).
