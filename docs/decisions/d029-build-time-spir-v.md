# D29. Build-time embedded SPIR-V

| Date | Status |
|---|---|
| 2026-07-05 | Accepted |

## Decision

Vulkan shaders are compiled to SPIR-V at build time and embedded in the host. They are not compiled when the program runs.

## Reason

The determinism contract fixes the shaders for good, so runtime code generation buys nothing and would add a shader-compiler dependency.

## Notes

Numbered D22 in the original Vulkan and CUDA design; renumbered because D19 to D22 are the applet decisions.
