# Web Geometry

This page plans what remains of drawing scenes in a browser: pipelines created without blocking, encoded frames decoded by the browser, the scene budgets with the refusal line and the rebuild after a lost device, the parity tests not yet written, and what a browser without WebGPU shows instead of a scene, a text log and a bar chart.

The rest is built, and the docs describe it. The web viewer's WebAssembly views draw every scene primitive through the WebGPU backend of the graphics layer, from the one shader source, with the scene core's picking, hover and lasso. Rendering (`docs/applets/rendering.md`), WebGPU Backend, and Graphics Layer (`docs/applets/graphics-layer.md`) describe the backend, Shader Authoring (`docs/applets/shader-authoring.md`) the shader sources, Scene Rendering (`docs/applets/scene-rendering.md`) the scene draws, Testing (`docs/development/testing.md`) the scene rows in headless Chromium, and Browser Elements (`docs/web/browser-elements.md`), Renderer Selection, the notice that a browser without WebGPU shows today in place of a scene.

The zero-copy backends are the product path, Metal on macOS, Vulkan with CUDA on Windows and Linux, and WebGPU in the browser, so the WebGPU backend serves only the browser and no native fallback through Dawn is planned.

## Design

The scene draws, bind groups, colour, depth, entity transforms, picking and uploads of the original design are built as Scene Rendering (`docs/applets/scene-rendering.md`) describes them; these parts remain.

### Pipelines and Their Cache

Pipelines are cached under the hash of their descriptor, as built, but they are created synchronously today. Each pipeline is to be made with `createRenderPipelineAsync` the first time a scene needs it, and compiling then never blocks the browser's main thread. A frame whose pipelines are not ready yet is not drawn and the previous pixels stay, which is the geometry service's rule of one atomic frame.

### Encoded Frames

An encoded frame, JPEG or PNG, is decoded by the browser with `createImageBitmap`, copied into an `rgba8unorm` texture with `copyExternalImageToTexture`, and drawn by `fs_image_rgba`, so the module carries no image decoder. The scene view answers today that encoded images are not drawn yet.

### Uploads and Reuse

The buffers of a batch, their reuse and the spare pool are built (Rendering (`docs/applets/rendering.md`), WebGPU Backend), and the heap copy serves hover values today, while the rebuild after a lost device and the fallback are planned. The heap keeps the latest bytes of every batch, for hover values, for rebuilding the GPU state after a lost device and for the Canvas 2D fallback, so a scene's bytes are held twice, once in the heap and once on the GPU.

### Budgets and Level of Detail

A browser always receives a scene over the network, so the data view budgets of [Scenes](scenes.md#scene-data-views) bound what it holds, and the backend adds a bound on the whole:

| Budget | Value | Reason |
|---|---|---|
| Scene bytes in one mount | 256 MiB | Half the module's 512 MB heap cap, leaving room for a 64 MiB frame being decoded and for the other views |
| One array | 256 MiB | WebGPU's default buffer size |
| Points in a batch | 1,000,000 at most, 100,000 by default | The data view's `max_points` |
| Boxes, arrows and segments in a batch | 1,048,576 at most, 100,000 by default | The data view's `max_items` |
| Mesh vertices in a batch | 4,194,304 at most, 1,000,000 by default | The data view's `max_vertices` |
| Image side | 8,192 pixels at most, 1,024 by default; 4,096 on a compatibility device | The data view's `max_extent` and WebGPU's texture limit |
| Draws in a frame | 4,096 | A 1 MiB set of parameter slots, enough for 2,048 batches with their frustums |

Level of detail is the producer's work: it samples a cloud of 20 million points on its device before anything travels, and the [Wire Protocol 3.2 Draft](wire-protocol-3-2.md) carries the budgets. The viewer draws what arrives in full and does not decimate further. A scene over the byte budget is not drawn whole: the batches that do not fit are left out, and the scene view says so in a line under its header, for example "world/cloud is not drawn: the scene would exceed the browser's 256 MiB."

## Parity with the Native Renderers

The parity rules are built: `tests/scene/draw_reference.h` gives each pixel the identity of its last writer and a tolerance of 0 levels for lookups, 1 for blended, bilinear and line-interpolated pixels and 2 for lit faces, and lets an edge pixel take the colour of either side or of the far side that the top write hid. The compile checks, the scene rows on both device levels and the colormap rows run in headless Chromium, as Testing (`docs/development/testing.md`) describes.

These rows remain to be written:

| Test | What it checks |
|---|---|
| Colormap through a depth image | Renderer core's colormap inputs through a depth image in the browser; every pixel exact |
| Near plane | Segments, arrows and boxes that cross the near plane keep exactly their visible parts |
| Index clamp | A mesh with indices past its last vertex draws as the native clamp does |
| Picking | The pick pass returns the expected entity and item at chosen pixels, through depth and line widths |
| Reuse | 300 updates of one entity leave the buffer count unchanged; an entity transform uploads only parameter slots |
| Budget | A scene at the byte budget draws; one over it leaves out batches and shows its line |
| Frame time | A scene of 1,000,000 points, 10,000 boxes and 100,000 segments draws at 60 frames per second on the Mac's GPU |
| Lost device | After `device.destroy()` the scene is drawn again from the heap copies |

## Fallback Without WebGPU

The element's own TypeScript views draw only with Canvas 2D, as Browser Elements (`docs/web/browser-elements.md`), Renderer Selection, describes. Today a browser without WebGPU shows a notice in place of a scene and draws none of it.

Where WebGPU is absent, refused or lost, the element draws a scene with its TypeScript views in Canvas 2D, and it draws only the scene's points. It walks the entity tree, composes each entity's world matrix from the entity transforms in TypeScript, transforms each points batch on the CPU and draws all of them as the existing points view draws one cloud: round sprites per colour, far points first in 3-D, with the same orbit camera, hover and lasso. Colours follow the same rules, with colormap indices computed in the reference's f32 arithmetic so that they match. The fallback draws at most 100,000 points, which is the points data view's default budget, and samples a larger scene with a stride.

Everything else is not drawn, and the view says so. A notice appears as the first line of the scene view's body, under its header, in the faint text style, while the scene holds primitives the fallback cannot draw. Its wording names only the kinds that the scene holds, in this order: lines, arrows, boxes, meshes and images.

> This view shows only the scene's points, because this browser has no WebGPU. The scene's lines, boxes and images (9 entities) are not drawn.

When the points were sampled, a second sentence follows: "It draws 100,000 of its 1,250,000 points." The notice's tooltip reads: "Lines, boxes, meshes and images in 3-D need WebGPU. Open this page in a browser with WebGPU enabled to see the whole scene." The notice reads the same in a 2-D scene, where images are not drawn either.

The two other new kinds are simpler in the fallback. The text log is plain DOM: a scrolling list of lines with their stamps, inside an element with the `log` role, that follows the newest line unless the user scrolls back, keeps its newest 10,000 lines and lets text be selected, searched and read by a screen reader. The bar chart is Canvas 2D, drawn with the axes of the existing plot code as the histogram view draws its bins.

The fallback also draws on its canvas the scale bar that Scene Rendering (`docs/applets/scene-rendering.md`) describes, by the same rule and from the same shared table of tests as the view layer: measured at the orbit target in 3-D, exact in 2-D, hidden with the same display option. With WebGPU the bar is not a draw of this backend; the view layer draws it with ImGui over the picture, as natively. Its capture rows run in headless Chromium on both paths.

## Work Items

| Work item | Size in builder days | Depends on |
|---|---|---|
| Encoded frames through the browser's decoder | The rest of an item of 2 | Wire protocol 3.2's scene messages |
| Budgets, the refusal line and rebuilding after a lost device | 2 | The data view budgets of Scenes and Wire protocol 3.2 |
| The parity rows still to write: near plane, index clamp, picking, reuse, budget, frame time, lost device and the colormap through a depth image | The rest of an item of 4 | The scene rows in headless Chromium |
| Asynchronous pipeline creation | The rest of an item of 4 | The WebGPU backend of the graphics layer |
| The scene fallback in TypeScript: entity transforms, points only, the notice | 2 | The scene messages in the TypeScript codec |
| The text log in DOM and the bar chart in Canvas 2D | 3 | Kinds 7 and 8 confirmed by the owners of the protocol |

The three whole items come to 7 builder days, and the three remainders finish items of 2, 4 and 4 days whose other parts are built. The browser runtime of the spec format shows a `scene` panel through the same scene view and needs nothing more from this page.

## Limits

- **No zero-copy in a browser.** Every array is copied twice in a browser, from the network into the heap and from the heap into the GPU process.
- **Memory.** A scene is held in the heap and on the GPU, within 256 MiB, under a heap that cannot pass 4 GiB.
- **Applets.** An applet's scene reaches a browser only through the hub, copied, and only where it is wire-capable ([Scenes](scenes.md#wire-capable-scenes)).
- **Performance.** No frame time in this page was measured; the frame-time test sets the target.

## Decisions

- **Encoded frames decoded by the browser.** The module stays free of an image decoder.
- **Level of detail at the producer only.** Sampling where the data lives keeps both copies small.
- **The scale bar in the fallback repeats the view layer's rule.** The fallback draws the bar from the same small rule and table of tests instead of adding a draw.

## Open Decisions

- **The text log with WebGPU present.** It can stay in the view layer or always be the DOM view. Recommendation: always the DOM view, because text in the DOM can be selected, searched and read aloud.
- **Decimation while the camera moves.** It would keep orbiting smooth over the budget at the cost of a second level of detail. Recommendation: not now; revisit if the frame-time test fails.
