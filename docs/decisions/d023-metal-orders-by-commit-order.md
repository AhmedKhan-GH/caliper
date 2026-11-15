# D23. Metal orders work by single-queue commit order

| Date | Status |
|---|---|
| 2026-07-05 | Accepted |

## Decision

On Metal, drawing is ordered by the commit order of one command queue, not by synchronization objects. A `MTLSharedEvent` is used only where two queues really must be ordered, namely the producer queue of the stream-ordered handoff. Test readback waits on the renderer's own queue.

## Reason

Metal's queue semantics and its retention of resources by command buffers give what Vulkan needed explicit machinery for. Adding the same machinery anyway would copy the Windows design without a reason.

## Notes

The Vulkan counterpart is [D25](d025-cuda-writes-the-vulkan-image.md) and the stream handoff is [D24](d024-stream-ordered-handoff-through-caps.md).
