# Explanations

Explanations describe how Caliper works and why it is built that way. They are for reading away from the keyboard.

| Page | Subject |
|---|---|
| [Architecture](architecture.md) | The host, the SDK, the applets and the frame loop |
| [Rendering](rendering.md) | The renderer seam, the Metal, Vulkan and OpenGL backends, and the geometry path |
| [Zero-copy tensor visualization](zero-copy.md) | How a tensor reaches the screen without a CPU round trip on each platform, and how the handoff is synchronized |
| [Vulkan and CUDA on Windows](vulkan-cuda.md) | Device pairing, the direct image path, pipelining and the measured costs on Windows |
| [Live streaming](live-streaming.md) | How a Python process streams tensors to the app through shared memory |
| [Compatibility and epochs](compatibility.md) | How the ABI and the services grow without breaking applets |
| [Trust model](trust-model.md) | What the host checks before loading an applet, and how it contains a fault |
| [Testing](testing.md) | The test layers, the fixture host and the pixel-exact graphics suite |
| [UI principles](ui-principles.md) | The rules for text, layout and behaviour in the host and in every applet |
