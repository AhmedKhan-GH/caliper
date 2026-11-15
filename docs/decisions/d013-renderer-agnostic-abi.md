# D13. Renderer-agnostic ABI, Metal and Vulkan primary, OpenGL fallback

| Date | Status |
|---|---|
| 2026-06-10 | Accepted |

## Decision

From epoch 2 the ABI does not depend on a graphics API. The primary backends are Metal on macOS and Vulkan on Windows, with OpenGL 3.3 kept as a frozen fallback. Textures cross the ABI as the opaque handle `CaliperTextureId`.

## Reason

Showing tensors without a copy needs pixels that live on the GPU. OpenGL is deprecated on macOS and cannot reach MPS memory, so every Mac texture took a CPU round trip. The decision was made while no outside applets existed, so the renderer stays internal to the host and changing it never bumps the epoch or forces applets to rebuild.

## Notes

The code implements the backend half: Metal is the macOS default, Vulkan is the Windows default, and `CALIPER_RENDERER=gl` selects the fallback. The other half is unbuilt: GLEW is still the OpenGL loader, and replacing it with a GLAD 3.3-core loader is an open question.
