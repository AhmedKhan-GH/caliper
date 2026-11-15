# D27. UUID-matched device pairing

| Date | Status |
|---|---|
| 2026-07-05 | Accepted |

## Decision

The Vulkan device that renders is the one whose UUID matches a CUDA device. When no device matches, interop is switched off instead of raising an error, and the bridge stages tensors through the CPU.

## Reason

Hybrid-GPU laptops and machines with only a software Vulkan device keep working through the staged path, and the mistake of pairing the wrong GPU cannot occur.

## Notes

Numbered D20 in the original Vulkan and CUDA design; renumbered because D19 to D22 are the applet decisions.
