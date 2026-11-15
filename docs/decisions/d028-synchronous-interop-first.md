# D28. Synchronous interop first, semaphores second

| Date | Status |
|---|---|
| 2026-07-05 | Accepted |

## Decision

The CUDA to Vulkan handoff was first built synchronously, with a stream synchronize and fenced submits. Semaphore pipelining came second, using a shared timeline semaphore for each texture. The synchronous path stays as the fallback where timeline semaphores are not available.

## Reason

Correctness came first: the pipelined path was added only after the synchronous one was verified byte-exact, and it degrades to the synchronous one when the device cannot export timeline semaphores.

## Notes

Numbered D21 in the original Vulkan and CUDA design; renumbered because D19 to D22 are the applet decisions.
