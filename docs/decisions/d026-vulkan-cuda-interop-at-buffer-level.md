# D26. Interop at buffer level

| Date | Status |
|---|---|
| 2026-07-05 | Superseded |

## Decision

CUDA never touches Vulkan image memory. It writes an interop buffer, and the conversion from buffer to image happens on the Vulkan timeline.

## Reason

This sidesteps the tiling and alignment problems of sharing image memory, at the price of one on-GPU pass per update, which has the same shape as the Metal compute path.

## Notes

Numbered D19 in the original Vulkan and CUDA design; renumbered because D19 to D22 are the applet decisions. Superseded for tensor updates by [D25](d025-cuda-writes-the-vulkan-image.md); the buffer-level path remains the fallback.
