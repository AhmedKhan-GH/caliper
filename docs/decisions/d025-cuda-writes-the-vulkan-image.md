# D25. CUDA writes the Vulkan image

| Date | Status |
|---|---|
| 2026-11-05 | Accepted |

## Decision

On Windows the update pass runs in CUDA and writes the Vulkan image directly. This supersedes [D26](d026-vulkan-cuda-interop-at-buffer-level.md) for tensor updates, and the buffer-level path stays as the fallback.

## Reason

It removes the copy and the cross-API pass for every tensor, which matches what Metal does. The costs are a third colormap implementation that the graphics tests must hold byte-exact, a constraint on the image format, and a second image for a texture that is updated after it has been drawn.
