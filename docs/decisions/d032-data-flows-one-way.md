# D32. Data flows one way, tensors to pixels

| Date | Status |
|---|---|
| 2026-07-09 | Accepted |

## Decision

Data flows from tensors to pixels to the UI and never back. There is no render-to-tensor readback. Applets cannot supply shaders; appearance comes from a fixed menu of shading modes.

## Reason

Showing the state of a model is the purpose of the instrument, and training on pixels is not. Applet-supplied shaders would break the boundary between applet and host and weaken its security model. A fixed menu keeps every image reproducible against the CPU reference.

## Notes

Whether applet-supplied shaders stay excluded once agent-built panels exist is an open question in the [decision log](index.md#open-questions).
