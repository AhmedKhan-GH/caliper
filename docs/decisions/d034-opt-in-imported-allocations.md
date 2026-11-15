# D34. Imported allocations are opt-in per allocation

| Date | Status |
|---|---|
| 2026-07-07 | Accepted |

## Decision

An applet opts in for each allocation by creating its tensors from the exportable pool. The host imports each block once, and every update then runs on the imported buffer at the tensor's byte offset. Memory that is not born in the pool keeps the path that copies once on the device.

## Reason

The default allocator of torch cannot export its memory, so zero-copy can only apply to memory that was allocated to be shared. Every miss returns `false` and the caller falls back, so an import that fails is never a crash or a wrong image.

## Notes

On Metal the import needs no operating-system handle: unified memory means the buffer pointer already is the resource, so the host keeps an in-process reference to the buffer. See [D14](d014-bridge-allocates-shared-tensors.md) and [D33](d033-honest-fallbacks.md).
