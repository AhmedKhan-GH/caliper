# D36. libcaliper owns the applet canvas

| Date | Status |
|---|---|
| 2026-07-11 | Accepted |

## Decision

When another program embeds libcaliper, libcaliper renders the applet canvas from start to finish. The embedder supplies a native child view for each canvas, and libcaliper draws into it through Metal or Vulkan, with one ImGui context per canvas. The embedder draws its own interface around the canvases.

## Reason

A toolkit such as wxWidgets paints through its own drawing backends and cannot paint the pixels that applets produce on the GPU. Keeping the canvas in one place also keeps the applet contract the same in every host. As [D5](d005-runtime-packs.md) requires, the one libtorch of the process belongs to the embedder.

## Notes

Whether a second native host that embeds libcaliper is still needed once applets live outside the core is an open question in the [decision log](index.md#open-questions).
