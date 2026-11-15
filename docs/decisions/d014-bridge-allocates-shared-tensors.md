# D14. The bridge allocates shared tensors

| Date | Status |
|---|---|
| 2026-06-10 | Accepted |

## Decision

The tensor bridge can allocate texture-backed shared tensors (`alloc_shared`), not only mirror tensors that already exist.

## Reason

An applet that writes into memory the texture reads directly moves nothing per update, so live weights and saliency maps become genuinely zero-copy. Applets adopt it with a single `torch::from_blob` call.

## Notes

The code implements this decision: `alloc_shared` is in `tensor_bridge_v1.h`.
