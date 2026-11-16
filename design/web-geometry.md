# Web geometry

This page plans how the web viewer draws scenes in a browser: points, lines, arrows, boxes, meshes, images and camera frustums in one 3-D or 2-D space. The web viewer is Caliper's view layer compiled to WebAssembly and drawn through WebGPU, as [Streaming and embedding](streaming-and-embedding.md#the-web-viewer) describes. The drawing is done by the WebGPU backend of the graphics layer that [Renderer core](renderer-core.md) specifies, fed by the scene core that [Scenes](scenes.md) compiles through. This page says how that backend realises each draw of a scene within WebGPU's limits, gives its shaders in WGSL, defines what parity with the native renderers means in a browser and how it is tested, and says what a browser without WebGPU shows instead. The same backend later serves the native fallback through Dawn, and the page notes what differs there.

## What is built and what is planned

Nothing on this page is built. It rests on these built pieces, which are on the `sprint2` branch and not yet on `main`:

- The web viewer module in `web/wasm/`: the live views, Dear ImGui and ImPlot compiled with Emscripten, the wire protocol client, and the WebGPU backend `src/host/renderer/webgpu_renderer.cpp` behind `HostRenderer`. The web viewer spike, a plan page on the `sprint2` branch, reports how this path was proved.
- In that backend, the heatmap colormap in WGSL. It matches the CPU reference `map_f32_to_rgba8` byte for byte on every adapter tested, because it compares each value with thresholds that the CPU derives from the reference instead of recomputing the reference's division.
- In that backend, the points of the geometry path and nothing more. A points payload is copied from WebAssembly memory into a GPU buffer (`import_host_allocation`), and each point is an instanced quad whose vertex shader pulls its position, value and colour table from three storage buffers. Lines, triangles, indices, normals, UVs, textures and instance matrices are refused.
- In the TypeScript package `web/src`, the views written for the browser: a WebGPU points renderer (`render/points-gpu.ts`) that already draws instanced quads from instance-step vertex buffers, and the Canvas 2D views that draw where WebGPU is absent, among them the points view with its orbit camera (`views/points.ts`, `views/camera.ts`).

Natively, Metal and Vulkan draw the whole geometry service, `caliper.geometry.v1` to `v1_3` (`docs/references/service-geometry-v1.md`): points, lines, triangles, textures and instanced poses, which `geom.vert` pulls from eight storage buffers (Rendering (`docs/explanations/rendering.md`)).

This page adds the WebGPU realisation of every draw a scene needs: pipelines and their WGSL, bind groups and the parameter block, colour, depth and draw order, picking, uploads, budgets and the parity tests. It also specifies the scene fallback in Canvas 2D and what the text log and bar chart views are there.

## The backend in the renderer core

The scene core turns a scene's entities into a draw list, the graphics layer takes that draw list for one target, and each backend implements the layer once. The WebGPU backend runs in the browser on the browser's WebGPU and, later, natively on Dawn. It never sees entities or entity paths. It sees draw records, each with a pipeline, its bind groups and dynamic offsets, up to six vertex bindings, an optional index buffer and a pick identity. The scene's draws are these:

| Draw | Made from | Entry points | Streams | Draw call |
|---|---|---|---|---|
| points | points | `vs_points`, `fs_points` | position, colour source, radius, per instance | 6 vertices, one instance per point |
| depth points | a depth image drawn as a point cloud | `vs_depth_points`, `fs_points` | none; the depth texture | 6 vertices, one instance per pixel |
| segments | lines in any mode, index-pair edges after the edge gather, camera frustums | `vs_lines`, `fs_color` | both ends, colour sources at both ends, break mask | 6 vertices, one instance per segment or edge |
| arrows | arrows | `vs_lines` with `line_mode` 1 | origin, vector, colour source | 9 vertices, one instance per arrow |
| box edges | boxes | `vs_box_wire`, `fs_color` | center, half sizes, rotation, colour source | 72 vertices in 3-D, 24 in 2-D, one instance per box |
| box faces | boxes with `fill` | `vs_box_solid`, `fs_color` | as box edges | 36 vertices in 3-D, 6 in 2-D, one instance per box |
| mesh | meshes | `vs_mesh` with `fs_color`, `fs_mesh_flat` or `fs_mesh_tex` | position, normal, colour source, UV per vertex; pose and value per instance | indexed, one instance unless posed |
| image | colour, depth and segmentation images | `vs_image` with `fs_image_rgba`, `fs_image_value` or `fs_image_class` | none; the image texture | 6 vertices |

Every draw also has a pick variant, whose fragment stage writes the pick identity instead of a colour (see [Picking](#picking)), and indexed meshes and index-pair edges depend on one compute module, the index clamp and its edge gather.

The geometry service is not one of these draws. It keeps its vertex-pulling program, which reads the storage buffers of group 3, and its one-pixel hardware lines, because its byte-exact rows and its index-clamp contract depend on them; only its points become instanced quads ([Renderer core](renderer-core.md)). In a browser it therefore runs on a default device and not on a compatibility device, where its capability bits stay unset as they do on OpenGL today.

## WebGPU limits that shape the design

| Limit | Default | Compatibility mode | What it means here |
|---|---|---|---|
| Storage buffers per shader stage | 8 | 8 | The native `geom.vert` binds exactly eight read-only storage buffers in its vertex stage |
| Storage buffers in the vertex stage | 8 | 0 | A compatibility device refuses any. Chromium refused the native layout with "number of storage buffers used in vertex stage (8) exceeds maxStorageBuffersInVertexStage (0)" |
| Point size | one pixel | one pixel | Point rasterization selects one fragment, in the pixel that contains the point (section 23.2.5.1), and no size can be set. Points are instanced quads |
| Line width | about one pixel | about one pixel | Line rasterization is left to the implementation, for example "a 1px-width rectangle" or Bresenham's algorithm (section 23.2.5.2), and no width can be set. Scene lines are instanced quads |
| Immediate data | 64 bytes | 64 bytes | The spike found no push constants. The specification now has immediate data (section 14.2, `setImmediates`), and Chromium 158 offers it with the WGSL language feature `immediate_address_space`. 64 bytes cannot hold the 192-byte parameter block, and the renderer core uses none in this sprint |
| Uniform buffers per stage | 12 | 12 | Three are used |
| Uniform binding size | 65,536 bytes | 16,384 bytes | The largest uniform binding is the 2,080-byte colormap block |
| Dynamic uniform buffers per pipeline layout | 8 | 8 | Two are used, the parameter block and the colormap block |
| Uniform offset alignment | 256 bytes | 256 bytes | Each 192-byte parameter block takes a 256-byte slot, and each colormap block 2,304 bytes |
| Bind groups | 4 | 4 | Scene draws use three; group 3 belongs to the geometry service |
| Bind groups plus vertex buffers | 24 | 24 | At most nine are used |
| Vertex buffers | 8 | 8 | A mesh uses six |
| Vertex attributes | 16 | 16 | A mesh uses nine |
| Vertex buffer stride | 2,048 bytes | 2,048 bytes | Strides here are 0 to 64 bytes |
| Inter-stage variables | 16 | 15 | Four are used: colour, UV, the pick identity and the entity-space position |
| Buffer size | 268,435,456 bytes (256 MiB) | the same | One array of one batch must fit one buffer |
| Storage binding size | 134,217,728 bytes (128 MiB) | the same | Binds only the geometry service's sources and the index clamp's input |
| 2-D texture side | 8,192 | 4,096 | The largest image side the viewer accepts |
| Colour attachments | 8 | 4 | One is used per pass |
| Flat interpolation | any sampling | `either` only | Chromium refused a flat output with the default sampling on a compatibility device: "The interpolation sampling (First) ... is not supported in compatibility mode". The pick identity is `@interpolate(flat, either)` |
| WebAssembly heap | 4 GiB | | The browser's WebAssembly is 32-bit. The module starts its heap at 16 MB and caps it at 512 MB today |
| Copy into the GPU process | | | Every `writeBuffer` copies bytes from the heap into the browser's GPU process. In the spike a 2048 × 2048 heatmap sent 16 MB a frame, and the GPU time rose to 70 ms |

Compatibility mode is the specification's lower feature level, for devices on older graphics APIs. A page gets it by requesting the adapter with `featureLevel: "compatibility"` and not requesting the feature `core-features-and-limits`. Every scene draw fits it, so that one backend serves both levels.

The numeric limits, point and line rasterization and immediate data were read in the WebGPU specification, the W3C Candidate Recommendation Draft of 15 September 2026 (section 3.6.2 for the limits). Chromium 158's headless shell reported the same values as device limits, on its SwiftShader adapter and on the Mac's GPU through Metal, and it reproduced both compatibility refusals quoted above. The Metal adapter offers more on request, such as 10 vertex-stage storage buffers and 4,294,967,292-byte buffers, and the design requests none of it. The 4 GiB ceiling, the 512 MB heap cap and the cost of the copy into the GPU process are taken from the spike and the module's README, and were not measured again.

## The design

### Vertex and instance-step buffers

Scene draws feed their vertex shaders from vertex buffers and instance-step buffers, not by pulling from storage buffers as `geom.vert` does. There are four reasons:

- **Compatibility mode allows no storage buffer in the vertex stage.** Packing the eight sources into one or two storage buffers would keep vertex pulling, but it still needs one, and requesting a higher limit is not portable.
- **An offset reads an array again without a copy.** A segment's two ends are the positions buffer bound twice, at offsets 0 and 12 with a stride of 24 bytes for pairs or 12 bytes for strips. Per-vertex colours of lines bind the same way.
- **The input assembler converts formats.** The positions, centers and half sizes of a 2-D scene bind as `float32x2` and arrive in a `vec3<f32>` with z = 0, as the specification fills missing components from (0, 0, 0, 1). A one-byte break mask binds as `uint8`.
- **A stride of 0 makes a stream constant.** One colour for a batch, one radius, a mesh without normals, a box without rotations or a mesh without instance poses binds a 16-byte constant buffer at stride 0. One shader then serves present and absent streams, and pipelines differ only in their vertex layouts.

Each item's colour arrives in one 4-byte stream read as raw `uint32` bits, which `color_mode` interprets as an f32 value, a packed RGBA colour or a class id. Reading bits keeps NaN values intact, as the native shaders do by binding their attributes as `uint[]`, and the colormap decides on the bits.

The cost is twofold. A hardware index fetch cannot clamp an index, so a compute pass clamps a mesh's indices once per batch update ([Meshes](#meshes)). And a pipeline is needed per vertex layout, which the cache absorbs.

### Bind groups and the parameter block

The layout is the one of [Renderer core](renderer-core.md):

| Group | Binding | Holds | Stages | Changes |
|---|---|---|---|---|
| 0 | 0 | `Frame`, 48 bytes: the viewport, the projection's scale, the pick window, the class count and the default class colour | vertex, fragment | once per target and frame |
| 0 | 1 | The class table, an `rgba8unorm` texture of 256 × 256 texels, one per class id from 0 to 65,535 | vertex, fragment | when the scene's class table changes |
| 1 | 0 | `PrimParams`, 192 bytes in a 256-byte slot, dynamic offset | vertex, fragment | per draw |
| 1 | 1 | `Cmap`, 2,080 bytes: the range, the flip, 258 keys and the 256-entry table, dynamic offset | vertex, fragment | per colormap and range |
| 2 | 0 to 3 | The image of a textured draw and its sampler | fragment; vertex as well for depth points | per image |
| 3 | | The geometry service's storage buffers, for its vertex pulling only | vertex | never bound by scene draws |

`PrimParams` keeps the native block's size and the offset of every field whose meaning survives: the combined transform, the normal matrix, `vertex_count`, the colour and shade modes, the flat colour, the range, `size_px`, `use_instance` and `use_instance_attr`. The element bases of vertex pulling have no meaning when buffers are bound at offsets, so their slots carry what the quad pipelines need: the pick identity, the radius and shape of points, the line mode and the image mode. One CPU function fills the block for every backend, so the matrices and modes that the shaders read are the same bytes everywhere. Renderer core's shader generator owns the final layout and emits its C++ struct. A frame's blocks live in the frame's uniform arena, written once a frame for the slots that changed.

The class table texture holds every id: the scene core writes the colours of the table's entries and, for each id without an entry, colour id mod 16 of the fixed palette, the rule of [Scenes](scenes.md). Only an id outside 0 to 65,535 takes the frame's default colour.

### Pipelines and their cache

A pipeline is cached under the hash of its descriptor: the draw, its vertex layout (two or three position components, and which streams have stride 0), its blend mode, its depth test and write, and its target, colour or pick. Pipelines are made with `createRenderPipelineAsync` the first time a scene needs them, so compiling never blocks the browser's main thread. A frame whose pipelines are not ready yet is not drawn and the previous pixels stay, which is the geometry service's rule of one atomic frame. A 3-D scene with every kind of primitive needs about twenty pipelines, and as many pick variants.

### Points

Each point is a quad of two triangles around its projected centre. Its size is a pixel size, as natively, or a radius in scene units: the radius times `radius_scale` (the entity's world scale) times the projection's scale gives the half size in clip units, which is exact for a perspective or orthographic projection without shear. A point is never smaller than one pixel. Scene points are discs, cut in the fragment stage with a hard edge, while the geometry service's points stay the native squares. A point with a NaN coordinate is moved outside the clip volume.

A depth image under a camera is also drawn as a point cloud without a point buffer: instance i reads pixel i of the depth texture, and the scene core folds the inverse intrinsics into the transform, so that pixel (x, y) at depth d enters it as ((x + 0.5)·d, (y + 0.5)·d, d), which is the back-projection of [Scenes](scenes.md). Zero, negative, infinite and NaN depths are skipped, decided on the bits.

### Lines and arrows

A segment is one instance, a quad of the batch's width in pixels with each end extended by half the width. Strips bind a one-byte break mask per vertex, which the producer computes on its device from `strip_lengths`, and instance k is hidden where byte k marks the last vertex of a strip; per-strip colours arrive expanded per vertex, also from the device. There are no mitre or round joins. The extended ends close the joins of thin strips, and an alpha-blended wide strip shows darker overlaps where segments meet. The shader clips each segment to the near plane before expanding it, so a segment that passes behind the camera keeps its visible part.

[Index-pair edges](scenes.md#index-pair-edges) reach the same draw through the edge gather, a second entry point of the index clamp's module ([The index clamp](#the-index-clamp)). Once per batch update it clamps each pair to the last vertex and copies the positions of its two ends, and their per-vertex colour sources, into two streams the layer owns, which `vs_lines` reads as it reads segments: positions with a stride of 24 bytes and offsets 0 and 12 in 3-D (16, 0 and 8 in 2-D), colour sources with a stride of 8 and offsets 0 and 4. Per-edge colours and values bind their (E,) stream twice, at offset 0 with a stride of 4, and the break mask is the stride-0 constant. The line shader therefore needs no new variant. The gather uses storage buffers in the compute stage only, five of them, which compatibility mode allows, and costs 24 to 32 bytes per edge in 3-D. The pick item is the instance index, which is the edge. The fallback draws no lines, so its notice is unaffected. The gather is sized in [Scenes](scenes.md).

An arrow is one instance of nine vertices: a shaft quad and a flat triangular head in screen space, a fifth of the arrow's length on screen and at least three widths wide. Lines, arrows and box edges keep their width in pixels at every zoom.

### Boxes

A box is an instance of the unit cube, or of the unit square in 2-D. Its center, half sizes and rotation quaternion are instance attributes at their offsets in the batch's arrays, and the vertex stage turns them into the column-major 4 × 4 pose that a `caliper.geometry.v1_3` instance pose holds: translation, rotation, then the half sizes as scale. The pose is applied first, as `mvp · (pose · v)`, and never folded into `mvp`. Edges are 12 segment quads, 4 in 2-D. With `fill`, the faces are 12 triangles, 2 in 2-D, drawn translucent with alpha blending and depth test only. A face's normal is an axis vector, so the pose maps it onto a vector parallel to the true normal under any rotation and positive scale, and the native normal chain lights a face exactly when lighting is asked for; boxes need none of the rigidity check that general instanced meshes need under Lambert lighting. Labels are text that the view layer draws over the image at anchors that the scene core projects.

### Meshes

A mesh binds its positions, normals, colour source and UVs as vertex buffers and its indices as a `uint32` index buffer for `drawIndexed`. With normals, lighting is the native per-vertex headlight, with the normal matrix computed on the CPU. Without normals, faces are flat: the fragment stage lights each face from the cross product of the derivatives of its entity-space position, on both sides. A textured mesh samples its image with the fixed sampler of `caliper.geometry.v1_2`: linear, clamped to the edge, without mipmaps. A mesh with instance poses takes them and its per-instance values as the geometry service's v1.3 records do; without them, an identity pose is bound at stride 0.

Index values come from the producer and cannot be checked without reading them. A hardware index fetch cannot clamp, and WebGPU either discards a draw with an index out of range or reads it as an invalid memory reference, so a small compute pass copies a batch's indices into an index buffer that the layer owns, each clamped to the last vertex, once per batch update. This keeps the geometry service's index-clamp contract for scenes.

### Images and camera frustums

An image is one quad: on its camera's image plane at the camera's plane distance in 3-D, or at its entity's origin, one scene unit per pixel, in 2-D. The scene core folds the map from the unit square to that place into the transform, so the quad needs no vertex buffer. Each image's texture is filled from its buffer by the layer's `fill_texture_from_buffer`, one GPU pass that also absorbs row alignment and, as this page asks of it, expands three channels to four and scales u16 depth.

| Image | Texture | Drawn by |
|---|---|---|
| Colour, u8 (H, W, 3) or (H, W, 4) | `rgba8unorm`; the fill expands three channels to four, as WebGPU has no three-channel format | `fs_image_rgba`, linear sampling, at its opacity |
| Grey, u8 (H, W) | `r8unorm` | `fs_image_rgba` with `image_mode` 1 |
| Depth, f32 or u16 | `r32uint` holding the f32 bits in scene units; u16 is scaled when the texture is filled | `fs_image_value`, the colormap per pixel, nearest; and depth points |
| Segmentation | `r32uint` holding the i32 class ids, which the producer casts on its device | `fs_image_class`, the class table per pixel, nearest, at its opacity |
| Encoded frame, JPEG or PNG | `rgba8unorm`, decoded by the browser with `createImageBitmap` and copied with `copyExternalImageToTexture` | `fs_image_rgba` |

Depth and segmentation textures hold bits rather than floats so that NaN and infinite depths reach the colormap's integer decision unchanged. A camera's frustum is eight segments, four from its centre to the corners of its image plane and four around the plane, in the camera's colour.

### Colour

A colormapped value is decided by integer comparisons against unsigned keys that the CPU derives from the reference, with the function and the 2,080-byte block of [Renderer core](renderer-core.md), copied unchanged into the WGSL below. The shader tests NaN on the bits, maps the bits to an ordered key, guesses the level with the usual formula and accepts the guess only when two integer comparisons confirm it, with a binary search as the last resort. The guess never decides, so no adapter's float arithmetic can move an index. The CPU computes the keys for each colormap and range, about 8,000 evaluations of the reference, and caches the block under the bits of the range. The same function runs in the vertex stage for points, arrows and mesh vertices and in the fragment stage for depth images. It replaces the native formula that the spike's points shader still uses and the f32 thresholds of the spike's heatmap shader, because WGSL lets an implementation assume that no infinity reaches a float comparison.

Class ids look up the class table. Flat and per-item RGBA colours are unpacked and divided by 255 as natively, and the conversion of the output to 8 bits rounds them back exactly.

### Depth, faces, blending and order

The state is the geometry service's fixed state: a `depth32float` buffer cleared to 1, the `less-equal` compare when testing (`always` when only writing), no culling and therefore two-sided faces, and the three blend modes with the native factors. WebGPU's clip space has +Y up and z in [0, 1], the convention of the geometry service and of Renderer core, so the scene core's projection matrices need no flip. A 3-D scene is right-handed with +Y up by default. The backend draws in draw-list order, and the scene core orders that list: opaque draws first, then translucent ones in entity order, unsorted, as the geometry service documents. A 2-D scene draws without depth in draw-list order.

### Entity transforms

The scene core computes each entity's world matrix as the product of the entity transforms down its path, in double precision. For each draw it multiplies projection, view and world relative to the camera, rounds the product to f32 once, and writes it to the draw's slot, together with the normal matrix for lit draws. An image's plane map and a depth cloud's inverse intrinsics are folded in the same way, while box and instance poses stay per instance. A changed entity transform therefore rewrites 256 bytes for each draw in that entity's subtree and uploads no geometry.

### Picking

On a pointer event the scene core compiles the pick variants of the visible draws into a pick pass: a 16 × 16 `RG32Uint` target around the pointer, reached by zooming clip space through the frame's pick window, with its own depth buffer. Every pickable draw writes its pick identity and the item, the instance index, or a vertex of the triangle for a mesh; an image writes no item, and the CPU finds its pixel from the quad. The window comes back through an asynchronous readback a frame or two later, and a result that arrives after the draw list changed is dropped. A hover tooltip then reads the item's values from the heap copy of its batch.

Ray tests on the CPU were rejected for hover and click. The pick pass is exact for every primitive with the pixels the user sees, including line widths, point discs, box faces and depth order; its CPU cost does not grow with the item count; and ray tests would have to repeat the shaders' screen-space expansion of lines and points. Lasso and rectangle selections stay on the CPU, over projected positions, because they need every item inside a polygon and not the front one.

### Uploads and reuse

Each batch, one primitive of one entity, owns a GPU buffer per array, with a capacity rounded up to a power of two and at least 4 KiB. A batch replaced by one of the same or a smaller size is written into its buffers with `writeBuffer`, which the queue orders after the draws already submitted. A larger batch gets new buffers, and the old ones join a pool of spare buffers, at most 64 MiB, as the spike's renderer reuses buffers because creating one per snapshot backed its queue up by seconds. A batch that did not change uploads nothing, and an entity transform uploads only parameter slots.

Everything is uploaded as it arrived, because the producer has already cast dtypes, computed break masks and expanded per-strip colours on its device, and the GPU builds box poses and clamps indices. Encoded frames are decoded by the browser, so the module carries no image decoder. The heap keeps the latest bytes of every batch, for hover values, for rebuilding the GPU state after a lost device and for the Canvas 2D fallback, so a scene's bytes are held twice, once in the heap and once on the GPU.

### Budgets and level of detail

A browser always receives a scene over the network, so the data view budgets of [Scenes](scenes.md) bound what it holds, and the backend adds a bound on the whole:

| Budget | Value | Reason |
|---|---|---|
| Scene bytes in one mount | 256 MiB | Half the module's 512 MB heap cap, leaving room for a 64 MiB frame being decoded and for the other views |
| One array | 256 MiB | WebGPU's default buffer size |
| Points in a batch | 1,000,000 at most, 100,000 by default | The data view's `max_points` |
| Boxes, arrows and segments in a batch | 1,048,576 at most, 100,000 by default | The data view's `max_items` |
| Mesh vertices in a batch | 4,194,304 at most, 1,000,000 by default | The data view's `max_vertices` |
| Image side | 8,192 pixels at most, 1,024 by default; 4,096 on a compatibility device | The data view's `max_extent` and WebGPU's texture limit |
| Draws in a frame | 4,096 | A 1 MiB set of parameter slots, enough for 2,048 batches with their frustums |

Level of detail is the producer's work: it samples a cloud of 20 million points on its device before anything travels, and [Wire protocol 3.2](wire-protocol-3-2.md) carries the budgets. The viewer draws what arrives in full and does not decimate further. A scene over the byte budget is not drawn whole: the batches that do not fit are left out, and the scene view says so in a line under its header, for example "world/cloud is not drawn: the scene would exceed the browser's 256 MiB."

### The native fallback through Dawn

The same backend compiles natively against Dawn, which offers the same `webgpu.h`, as Renderer core plans for machines without Metal or Vulkan. These things differ there:

- Dawn cannot wrap an `MTLBuffer` or import Vulkan external memory as a buffer, so the fallback always copies, as the browser does, and its info line says so.
- There is no 4 GiB heap and no GPU process, so a copy is a staging write within the process.
- Native code may block on the queue, where the browser allows only callbacks.
- The surface is the app's window, presented explicitly, where the browser presents the canvas when the frame returns to its event loop.
- Native adapters offer higher limits. The backend keeps to the defaults on both, so one path is tested, and its fit to compatibility mode lets Dawn's backends for older graphics APIs replace the OpenGL fallback for scenes; the geometry service stays unavailable there.

## Shader sources

The WGSL below is one module in seven blocks, concatenated in order, and a separate module for the index clamp. It holds the vertex and fragment stages of the scene draws, their pick variants and the parameter structs. It is a design draft and is built into nothing. It is written by hand because that is faster than waiting for the generator of Renderer core, and it becomes part of the one shader source when that lands; the colormap function and its block are Renderer core's, unchanged.

The blocks were validated as they stand on this page. A script extracted them and compiled both modules in Chromium 158's headless shell, driven by Playwright 1.65 and never shown on screen, on the SwiftShader adapter and on the Mac's GPU through Metal, each with a default device and a compatibility device. `getCompilationInfo` reported no message in any of the four runs. All 21 pipelines that cover the entry points, blend modes and targets above, the pick variants and the compute clamp included, were created with explicit layouts and no validation error. A 64 × 64 smoke scene then drew every kind and was read back. 28 colormapped points, whose values sit on both sides of key edges of an increasing and a decreasing range, with NaN and both infinities among them, matched the CPU reference byte for byte, and so did flat, RGBA, class and depth-cloud colours. A 3-pixel line, an arrow, a wire box, a box rotated by a quaternion, a mesh without normals whose out-of-range index the compute pass clamped, a class image with an id outside the table, and the pick target's pairs of entity and item were where they belong. All 21 checks passed in all four runs. No `naga` or `tint` command was available, so no other tool was used. The script lives outside the repository; a work item below makes it a test. The edge gather of the index clamp's module was added afterwards and checked the same way, in the same four runs: both modules compiled with no message, the gather's compute pipeline and the two pipelines that draw edges with per-edge colour streams were created, 24 pipelines in all, and nine more checks passed, so 30 of 30. The gathered positions and colour sources read back exactly, with an index past the last vertex clamped to it; per-edge values through the colormap, one at a key edge, one just below another and one NaN, matched the CPU reference byte for byte at the edges' centres; and the pick target returned each edge's index.

### Declarations

```wgsl
// Group 0: one per target and frame (48 bytes).
struct Frame {
  viewport: vec2<f32>,    //  0  the target's size in physical pixels
  proj_scale: vec2<f32>,  //  8  proj[0][0], proj[1][1]
  ndc_scale: vec2<f32>,   // 16  (1, 1), or the zoom of the pick window
  ndc_offset: vec2<f32>,  // 24  (0, 0), or the offset of the pick window
  class_count: u32,       // 32  ids the class table covers (65,536)
  class_default: u32,     // 36  packed RGBA for an id outside them
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var class_table: texture_2d<f32>;   // rgba8unorm, 256 x 256: ids 0 to 65,535

// Group 1: the draw's 256-byte slot and its colormap block, both at dynamic offsets.
struct PrimParams {        // 192 bytes, the native size
  mvp: mat4x4<f32>,        //   0  as native
  nmat0: vec4<f32>,        //  64  as native
  nmat1: vec4<f32>,        //  80  as native
  nmat2: vec4<f32>,        //  96  as native
  pick_id: u32,            // 112  native pos_base
  item_base: u32,          // 116  native idx_base
  point_shape: u32,        // 120  native nrm_base: 0 square, 1 disc
  radius_mode: u32,        // 124  native attr_base: 0 size_px, 1 radius in scene units
  unused0: u32,            // 128  native use_index; the clamp happens before the draw
  vertex_count: u32,       // 132  as native
  color_mode: u32,         // 136  as native, and 4 for a class id
  shade_mode: u32,         // 140  as native
  flat_rgba: u32,          // 144  as native
  vmin: f32,               // 148  as native; the colormap reads the range of its block
  vmax: f32,               // 152  as native
  size_px: f32,            // 156  as native for points; also the width of lines
  radius_scale: f32,       // 160  native uv_base
  use_instance: u32,       // 164  as native
  line_mode: u32,          // 168  native inst_base: 0 segments, 1 arrows
  use_instance_attr: u32,  // 172  as native
  image_mode: u32,         // 176  native inst_attr_base: 0 RGBA, 1 grey
  pad0: u32,
  pad1: u32,
  pad2: u32,
};
@group(1) @binding(0) var<uniform> p: PrimParams;

// The colormap block of Renderer core, 2,080 bytes. keys: 258 entries; keys[0] = 0
// (always passed), keys[256] = keys[257] = 0xffffffff (never passed).
struct Cmap { vmin: f32, vmax: f32, flip: u32, pad: u32, keys: array<vec4<u32>, 65>, lut: array<vec4<u32>, 64> };
@group(1) @binding(1) var<uniform> m: Cmap;

// Group 2: the image of a textured draw, by kind. Group 3 holds the geometry
// service's storage buffers and is never bound by these pipelines.
@group(2) @binding(0) var img_rgba: texture_2d<f32>;    // rgba8unorm or r8unorm
@group(2) @binding(1) var img_sampler: sampler;         // linear, clamp to edge
@group(2) @binding(2) var img_class: texture_2d<u32>;   // r32uint: i32 class ids as bits
@group(2) @binding(3) var img_value: texture_2d<u32>;   // r32uint: f32 depths or values as bits

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) @interpolate(flat, either) pick: vec2<u32>,
  @location(3) obj: vec3<f32>,   // the entity-space position, for flat mesh faces
};
```

### Helpers and the colormap

```wgsl
fn unpack_rgba(c: u32) -> vec4<f32> {
  return vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu),
                   f32((c >> 16u) & 0xffu), f32((c >> 24u) & 0xffu)) / 255.0;
}

fn is_nan_bits(v: f32) -> bool {
  let b = bitcast<u32>(v);
  return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

fn any_nan3(v: vec3<f32>) -> bool {
  return is_nan_bits(v.x) || is_nan_bits(v.y) || is_nan_bits(v.z);
}

// A position the clipper removes (z > w).
fn hidden() -> vec4<f32> { return vec4<f32>(0.0, 0.0, 2.0, 1.0); }

// The colormap of Renderer core: the index is decided by integer comparisons
// against keys the CPU derives from the reference; the float guess only picks
// which comparisons run.
fn key_at(k: u32) -> u32 { return m.keys[k >> 2u][k & 3u]; }
fn level_ok(key: u32, k: u32) -> bool { return key >= key_at(k) && key < key_at(k + 1u); }

fn colormap_index(bits: u32) -> u32 {
    if ((bits & 0x7fffffffu) > 0x7f800000u) { return 0u; }                    // NaN, from the bits
    let key = select(bits | 0x80000000u, ~bits, (bits >> 31u) != 0u) ^ m.flip;
    let v = bitcast<f32>(bits);
    let g = min(u32(clamp((v - m.vmin) / (m.vmax - m.vmin), 0.0, 1.0) * 255.0 + 0.5), 255u);
    if (level_ok(key, g)) { return g; }                                       // the usual case
    if (g > 0u && level_ok(key, g - 1u)) { return g - 1u; }
    if (g < 255u && level_ok(key, g + 1u)) { return g + 1u; }
    var lo = 0u; var hi = 256u;                                               // keys[lo] <= key < keys[hi]
    loop { if (hi - lo <= 1u) { break; } let mid = (lo + hi) / 2u;
           if (key >= key_at(mid)) { lo = mid; } else { hi = mid; } }
    return lo;
}

fn cmap_color(bits: u32) -> vec4<f32> {
  let k = colormap_index(bits);
  return unpack_rgba(m.lut[k >> 2u][k & 3u]);
}

fn class_color(id: u32) -> vec4<f32> {
  if (id >= frame.class_count) { return unpack_rgba(frame.class_default); }
  return textureLoad(class_table, vec2<u32>(id & 255u, id >> 8u), 0);
}

// One item's colour from its 4-byte source, read as bits so that NaN survives.
fn item_color(src: u32) -> vec4<f32> {
  switch (p.color_mode) {
    case 1u: { return cmap_color(src); }
    case 2u: { return unpack_rgba(src); }
    case 3u: { return vec4<f32>(1.0); }
    case 4u: { return class_color(src); }
    default: { return unpack_rgba(p.flat_rgba); }
  }
}

// The native instance normal chain, in the native order of operations.
fn inst_normal(q: mat4x4<f32>, n: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(q[0].x * n.x + q[1].x * n.y + q[2].x * n.z,
                   q[0].y * n.x + q[1].y * n.y + q[2].y * n.z,
                   q[0].z * n.x + q[1].z * n.y + q[2].z * n.z);
}

// The native Lambert headlight: ambient 0.30, alpha untouched.
fn view_normal(n: vec3<f32>) -> vec3<f32> {
  return normalize(n.x * p.nmat0.xyz + n.y * p.nmat1.xyz + n.z * p.nmat2.xyz);
}

fn lambert(c: vec4<f32>, n: vec3<f32>) -> vec4<f32> {
  let lit = 0.30 + 0.70 * max(dot(view_normal(n), vec3<f32>(0.0, 0.0, 1.0)), 0.0);
  return vec4<f32>(c.rgb * lit, c.a);
}

fn to_target(c: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(c.xy * frame.ndc_scale + frame.ndc_offset * c.w, c.z, c.w);
}

// Corner c of a quad of two triangles, as (end 0 or 1, side -1 or 1).
fn quad_corner(c: u32) -> vec2<f32> {
  var t = array<vec2<f32>, 6>(vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
                              vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  return t[c];
}

// Clips a segment to the near plane (clip z >= 0); false when all of it is behind.
fn clip_near(a: ptr<function, vec4<f32>>, b: ptr<function, vec4<f32>>) -> bool {
  let za = (*a).z;
  let zb = (*b).z;
  if (za < 0.0 && zb < 0.0) { return false; }
  if (za < 0.0) { *a = mix(*a, *b, za / (za - zb)); }
  if (zb < 0.0) { *b = mix(*b, *a, zb / (zb - za)); }
  return true;
}

// Corner c of segment (a, b) in clip space, as a quad width_px wide whose ends
// are extended by half the width (square caps).
fn segment_corner(a_in: vec4<f32>, b_in: vec4<f32>, c: u32, width_px: f32) -> vec4<f32> {
  var a = a_in;
  var b = b_in;
  if (!clip_near(&a, &b)) { return hidden(); }
  let half_vp = 0.5 * frame.viewport;
  let sa = a.xy / a.w * half_vp;
  let sb = b.xy / b.w * half_vp;
  let len = length(sb - sa);
  let dir = select(vec2<f32>(1.0, 0.0), (sb - sa) / len, len > 1e-6);
  let q = quad_corner(c);
  let w = max(width_px, 1.0) * 0.5;
  let off = vec2<f32>(-dir.y, dir.x) * (q.y * w) + dir * select(-w, w, q.x > 0.5);
  let e = select(a, b, q.x > 0.5);
  return to_target(vec4<f32>(e.xy + off / half_vp * e.w, e.z, e.w));
}
```

### The point shaders

```wgsl
struct PointIn {
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
  @location(0) pos: vec3<f32>,   // float32x3, or float32x2 in 2-D (z reads 0)
  @location(1) color_src: u32,   // stride 4, or stride 0 for one colour
  @location(2) radius: f32,      // stride 4, or stride 0 for one radius
};

@vertex
fn vs_points(in: PointIn) -> VOut {
  var o: VOut;
  let q = quad_corner(in.vid);
  let corner = vec2<f32>(q.x * 2.0 - 1.0, q.y);
  o.uv = corner;
  o.pick = vec2<u32>(p.pick_id, p.item_base + in.iid);
  o.color = item_color(in.color_src);
  if (any_nan3(in.pos)) { o.pos = hidden(); return o; }
  let c = p.mvp * vec4<f32>(in.pos, 1.0);
  // Half the size in clip units: size_px as natively, or a radius in scene
  // units projected with the frame's scale, never below half a pixel.
  var k = (p.size_px / frame.viewport) * c.w;
  if (p.radius_mode == 1u) {
    k = max(in.radius * p.radius_scale * frame.proj_scale, c.w / frame.viewport);
  }
  o.pos = to_target(vec4<f32>(c.xy + corner * k, c.z, c.w));
  return o;
}

// A depth image under a camera as a point cloud: instance i is pixel i. The
// scene core folds the inverse intrinsics into mvp, so the pixel (x, y) at
// depth d is (x * d, y * d, d) before it.
@vertex
fn vs_depth_points(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VOut {
  var o: VOut;
  let size = textureDimensions(img_value);
  let px = vec2<u32>(iid % size.x, iid / size.x);
  let bits = textureLoad(img_value, px, 0).r;
  let q = quad_corner(vid);
  let corner = vec2<f32>(q.x * 2.0 - 1.0, q.y);
  o.uv = corner;
  o.pick = vec2<u32>(p.pick_id, p.item_base + iid);
  o.color = select(unpack_rgba(p.flat_rgba), cmap_color(bits), p.color_mode == 1u);
  // Skip zero, negative, infinite and NaN depths, decided on the bits.
  if ((bits >> 31u) != 0u || bits == 0u || (bits & 0x7fffffffu) >= 0x7f800000u) { o.pos = hidden(); return o; }
  let d = bitcast<f32>(bits);
  let xy = (vec2<f32>(px) + 0.5) * d;
  let c = p.mvp * vec4<f32>(xy, d, 1.0);
  o.pos = to_target(vec4<f32>(c.xy + corner * (p.size_px / frame.viewport) * c.w, c.z, c.w));
  return o;
}

@fragment
fn fs_points(in: VOut) -> @location(0) vec4<f32> {
  if (p.point_shape == 1u && dot(in.uv, in.uv) > 1.0) { discard; }
  return in.color;
}

@fragment
fn fs_points_pick(in: VOut) -> @location(0) vec2<u32> {
  if (p.point_shape == 1u && dot(in.uv, in.uv) > 1.0) { discard; }
  return in.pick;
}
```

### The line and arrow shaders

```wgsl
struct LineIn {
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
  @location(0) a: vec3<f32>,     // segment start, or arrow origin
  @location(1) b: vec3<f32>,     // segment end, or arrow vector
  @location(2) color_a: u32,     // colour source at a, or of the whole arrow
  @location(3) color_b: u32,     // colour source at b
  @location(4) keep: u32,        // uint8 break mask: 0 at the last vertex of a strip
};

@vertex
fn vs_lines(in: LineIn) -> VOut {
  var o: VOut;
  o.pick = vec2<u32>(p.pick_id, p.item_base + in.iid);
  let tip = select(in.b, in.a + in.b, p.line_mode == 1u);
  let at_b = quad_corner(in.vid % 6u).x > 0.5 && p.line_mode == 0u;
  o.color = item_color(select(in.color_a, in.color_b, at_b));
  if (in.keep == 0u || any_nan3(in.a) || any_nan3(tip)) { o.pos = hidden(); return o; }
  var a = p.mvp * vec4<f32>(in.a, 1.0);
  var b = p.mvp * vec4<f32>(tip, 1.0);
  if (p.line_mode == 0u) {
    o.pos = segment_corner(a, b, in.vid, p.size_px);
    return o;
  }
  // An arrow: vertices 0 to 5 are the shaft, 6 to 8 the head, a flat triangle in
  // screen space a fifth of the arrow's length, at least three widths wide.
  if (!clip_near(&a, &b)) { o.pos = hidden(); return o; }
  let half_vp = 0.5 * frame.viewport;
  let sa = a.xy / a.w * half_vp;
  let sb = b.xy / b.w * half_vp;
  let len = length(sb - sa);
  let dir = select(vec2<f32>(1.0, 0.0), (sb - sa) / len, len > 1e-6);
  let w = max(p.size_px, 1.0);
  let head = 0.2 * len;
  let base = sb - dir * head;
  if (in.vid < 6u) {
    o.pos = segment_corner(a, vec4<f32>(base / half_vp * b.w, b.z, b.w), in.vid, w);
    return o;
  }
  let side = vec2<f32>(-dir.y, dir.x) * max(1.5 * w, 0.4 * head);
  var s = sb;
  if (in.vid == 7u) { s = base + side; }
  if (in.vid == 8u) { s = base - side; }
  o.pos = to_target(vec4<f32>(s / half_vp * b.w, b.z, b.w));
  return o;
}

@fragment
fn fs_color(in: VOut) -> @location(0) vec4<f32> { return in.color; }

@fragment
fn fs_pick(in: VOut) -> @location(0) vec2<u32> { return in.pick; }
```

### The box shaders

```wgsl
struct BoxIn {
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
  @location(0) center: vec3<f32>,     // float32x3, or float32x2 in 2-D (z reads 0)
  @location(1) half_size: vec3<f32>,  // the same
  @location(2) rotation: vec4<f32>,   // a quaternion (x, y, z, w); stride 0 (identity) when absent
  @location(3) color_src: u32,
};

// The column-major 4 x 4 pose that a caliper.geometry.v1_3 instance holds:
// translation, rotation, then the half sizes as scale.
fn box_pose(in: BoxIn) -> mat4x4<f32> {
  var q = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  let n2 = dot(in.rotation, in.rotation);
  if (n2 > 0.0) { q = in.rotation * inverseSqrt(n2); }
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  let c0 = vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y));
  let c1 = vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x));
  let c2 = vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
  let h = in.half_size;
  return mat4x4<f32>(vec4<f32>(c0 * h.x, 0.0), vec4<f32>(c1 * h.y, 0.0),
                     vec4<f32>(c2 * h.z, 0.0), vec4<f32>(in.center, 1.0));
}

fn on_axis(axis: u32, along: f32, s: f32, t: f32) -> vec3<f32> {
  if (axis == 0u) { return vec3<f32>(along, s, t); }
  if (axis == 1u) { return vec3<f32>(s, along, t); }
  return vec3<f32>(s, t, along);
}

// Wire boxes: 12 edges of the unit cube as screen-space quads, 72 vertices; the
// first 24 are the square at z = -1, which is all a 2-D box draws.
@vertex
fn vs_box_wire(in: BoxIn) -> VOut {
  var o: VOut;
  o.pick = vec2<u32>(p.pick_id, p.item_base + in.iid);
  o.color = item_color(in.color_src);
  let e = in.vid / 6u;
  var axis = (e % 4u) / 2u;
  var s = select(-1.0, 1.0, (e & 1u) != 0u);
  var t = select(-1.0, 1.0, e >= 4u);
  if (e >= 8u) { axis = 2u; t = select(-1.0, 1.0, (e & 2u) != 0u); }
  let pose = box_pose(in);
  let a = p.mvp * (pose * vec4<f32>(on_axis(axis, -1.0, s, t), 1.0));
  let b = p.mvp * (pose * vec4<f32>(on_axis(axis, 1.0, s, t), 1.0));
  o.pos = segment_corner(a, b, in.vid % 6u, p.size_px);
  return o;
}

// Filled boxes: 6 faces of two triangles, 36 vertices; the first 6 are the face
// at z = +1, which is all a 2-D box draws. Each normal is an axis vector, so the
// pose maps it onto a vector parallel to the true normal under any rotation and
// positive scale, and the native normal chain lights a box exactly.
@vertex
fn vs_box_solid(in: BoxIn) -> VOut {
  var o: VOut;
  o.pick = vec2<u32>(p.pick_id, p.item_base + in.iid);
  let face = in.vid / 6u;
  var order = array<u32, 3>(2u, 0u, 1u);
  let axis = order[face / 2u];
  let sgn = select(1.0, -1.0, (face & 1u) != 0u);
  let q = quad_corner(in.vid % 6u);
  let pose = box_pose(in);
  o.pos = to_target(p.mvp * (pose * vec4<f32>(on_axis(axis, sgn, q.x * 2.0 - 1.0, q.y), 1.0)));
  var c = item_color(in.color_src);
  if (p.shade_mode == 1u) { c = lambert(c, inst_normal(pose, on_axis(axis, sgn, 0.0, 0.0))); }
  o.color = c;
  return o;
}
```

### The mesh shaders

```wgsl
struct MeshIn {
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,  // stride 0 when the mesh has no normals
  @location(2) color_src: u32,     // RGBA or value bits per vertex; stride 0 when one
  @location(3) uv: vec2<f32>,      // stride 0 when untextured
  @location(4) m0: vec4<f32>,      // the v1.3 instance pose; stride 0 (identity) when absent
  @location(5) m1: vec4<f32>,
  @location(6) m2: vec4<f32>,
  @location(7) m3: vec4<f32>,
  @location(8) inst_src: u32,      // the v1.3 per-instance value bits; stride 0 when absent
};

@vertex
fn vs_mesh(in: MeshIn) -> VOut {
  var o: VOut;
  o.pick = vec2<u32>(p.pick_id, p.item_base + in.vid);
  o.uv = in.uv;
  o.obj = in.pos;
  let inst = p.use_instance != 0u;
  let pose = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
  if (any_nan3(in.pos)) {
    o.pos = hidden();
  } else if (inst) {
    o.pos = to_target(p.mvp * (pose * vec4<f32>(in.pos, 1.0)));   // never fold mvp * pose
    o.obj = (pose * vec4<f32>(in.pos, 1.0)).xyz;
  } else {
    o.pos = to_target(p.mvp * vec4<f32>(in.pos, 1.0));
  }
  var c = item_color(in.color_src);
  if (inst && p.use_instance_attr != 0u) { c = cmap_color(in.inst_src); }
  if (p.shade_mode == 1u) {
    var n = normalize(in.normal);
    if (inst) { n = inst_normal(pose, n); }
    c = lambert(c, n);
  }
  o.color = c;
  return o;
}

// A mesh without normals: flat faces, each lit by the headlight from the normal
// that the derivatives of its entity-space position give, on both of its sides.
@fragment
fn fs_mesh_flat(in: VOut) -> @location(0) vec4<f32> {
  let n = view_normal(cross(dpdx(in.obj), dpdy(in.obj)));
  let lit = 0.30 + 0.70 * abs(n.z);
  return vec4<f32>(in.color.rgb * lit, in.color.a);
}

// The native textured fragment: the sampled colour times the vertex colour's RGB.
@fragment
fn fs_mesh_tex(in: VOut) -> @location(0) vec4<f32> {
  var s = textureSample(img_rgba, img_sampler, in.uv);
  if (p.image_mode == 1u) { s = vec4<f32>(s.rrr, 1.0); }
  return vec4<f32>(s.rgb * in.color.rgb, s.a);
}
```

### The image shaders

```wgsl
// One quad on a camera's image plane, or at the entity's origin in 2-D. mvp maps
// the unit square onto the image (the scene core folds that map into it), so no
// vertex buffer is bound. flat_rgba is a tint whose alpha is the opacity.
@vertex
fn vs_image(@builtin(vertex_index) vid: u32) -> VOut {
  var o: VOut;
  let q = quad_corner(vid);
  o.uv = vec2<f32>(q.x, q.y * 0.5 + 0.5);
  o.color = unpack_rgba(p.flat_rgba);
  o.pick = vec2<u32>(p.pick_id, 0u);
  o.pos = to_target(p.mvp * vec4<f32>(o.uv, 0.0, 1.0));
  return o;
}

@fragment
fn fs_image_rgba(in: VOut) -> @location(0) vec4<f32> {
  var s = textureSample(img_rgba, img_sampler, in.uv);
  if (p.image_mode == 1u) { s = vec4<f32>(s.rrr, 1.0); }
  return s * in.color;
}

fn texel_of(uv: vec2<f32>, size: vec2<u32>) -> vec2<u32> {
  return min(vec2<u32>(uv * vec2<f32>(size)), size - vec2<u32>(1u));
}

// A segmentation image: each pixel's class colour, nearest, at the opacity.
@fragment
fn fs_image_class(in: VOut) -> @location(0) vec4<f32> {
  let c = class_color(textureLoad(img_class, texel_of(in.uv, textureDimensions(img_class)), 0).r);
  return vec4<f32>(c.rgb, c.a * in.color.a);
}

// A depth or value image: each pixel through the colormap, nearest.
@fragment
fn fs_image_value(in: VOut) -> @location(0) vec4<f32> {
  let c = cmap_color(textureLoad(img_value, texel_of(in.uv, textureDimensions(img_value)), 0).r);
  return vec4<f32>(c.rgb, c.a * in.color.a);
}
```

### The index clamp

```wgsl
// A separate module: the index clamp, once per batch update. It copies the
// producer's indices into an index buffer the layer owns, each clamped to the
// last vertex, the geometry service's index-clamp contract.
struct ClampParams { count: u32, last: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> cp: ClampParams;
@group(0) @binding(1) var<storage, read> idx_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> idx_out: array<u32>;

@compute @workgroup_size(64)
fn cs_clamp_indices(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < cp.count) { idx_out[id.x] = min(idx_in[id.x], cp.last); }
}

// The edge gather, in the same module, once per update of a lines batch with
// index pairs. Each pair is clamped to the last vertex, and the positions and
// per-vertex colour sources of its two ends are copied into the streams that
// vs_lines reads, so that edge k is drawn as segment k. A grid-stride loop keeps
// the dispatch within 65,535 workgroups for any edge count.
struct GatherParams { count: u32, last: u32, dims: u32, per_vertex: u32 };
@group(0) @binding(3) var<uniform> gp: GatherParams;
@group(0) @binding(4) var<storage, read> pairs: array<u32>;           // (E, 2): i32 indices as bits
@group(0) @binding(5) var<storage, read> vert_pos: array<f32>;        // (V, dims)
@group(0) @binding(6) var<storage, read> vert_src: array<u32>;        // (V,) colour sources, read when per_vertex is 1
@group(0) @binding(7) var<storage, read_write> seg_pos: array<f32>;   // (E, 2, dims): ends at byte offsets 0 and 4 * dims
@group(0) @binding(8) var<storage, read_write> seg_src: array<u32>;   // (E, 2): ends at byte offsets 0 and 4

@compute @workgroup_size(64)
fn cs_gather_edges(@builtin(global_invocation_id) id: vec3<u32>,
                   @builtin(num_workgroups) groups: vec3<u32>) {
  for (var e = id.x; e < gp.count; e += groups.x * 64u) {
    for (var k = 0u; k < 2u; k++) {
      let v = min(pairs[2u * e + k], gp.last);
      for (var c = 0u; c < gp.dims; c++) {
        seg_pos[(2u * e + k) * gp.dims + c] = vert_pos[v * gp.dims + c];
      }
      if (gp.per_vertex == 1u) { seg_src[2u * e + k] = vert_src[v]; }
    }
  }
}
```

## Parity with the native renderers

Natively, every backend is byte-exact against one shared CPU reference. In a browser that holds for colour and not for every pixel, because WGSL lets an implementation round the vertex transform and the lighting differently from the CPU. Parity in the browser therefore means three things, checked against the same CPU reference images that the native rows use:

- **Exact colour.** Away from edges, a pixel's colour equals the reference byte for byte when it comes from a lookup: a flat colour, a per-item RGBA colour, a class colour, a colormapped value, or an image texel looked up by nearest.
- **Edges.** A pixel whose centre lies within one pixel of a primitive's edge in the reference, the edges of point discs included, may take the colour of either side.
- **Tolerances.** Lit pixels, by the headlight or by flat faces, may differ by 2 levels in each channel, as the native instanced Lambert rows allow. Linearly sampled image pixels and alpha-blended or additive pixels may differ by 1 level.

The tests run in headless Chromium through Playwright, on SwiftShader in CI and on the Mac's GPU locally, and read the target back through the layer's asynchronous readback:

| Test | What it checks |
|---|---|
| Compile and pipelines | Both modules compile and every pipeline is created on a default and on a compatibility device |
| Scene rows | One fixture per primitive, from the native rows, rendered at 256 × 256 and compared with the reference under the three rules above |
| Colormap | Renderer core's colormap inputs, every key edge and its neighbours, ±0, ±∞, the NaNs and random bit patterns, through points and a depth image; every pixel exact |
| Near plane | Segments, arrows and boxes that cross the near plane keep exactly their visible parts |
| Index clamp | A mesh with indices past its last vertex draws as the native clamp does |
| Picking | The pick pass returns the expected entity and item at chosen pixels, through depth and line widths |
| Reuse | 300 updates of one entity leave the buffer count unchanged; an entity transform uploads only parameter slots |
| Budget | A scene at the byte budget draws; one over it leaves out batches and shows its line |
| Frame time | A scene of 1,000,000 points, 10,000 boxes and 100,000 segments draws at 60 frames per second on the Mac's GPU |
| Lost device | After `device.destroy()` the scene is drawn again from the heap copies |

## The fallback without WebGPU

Where WebGPU is absent, refused or lost, the element draws a scene with its TypeScript views in Canvas 2D, and it draws only the scene's points. It walks the entity tree, composes each entity's world matrix from the entity transforms in TypeScript, transforms each points batch on the CPU and draws all of them as the existing points view draws one cloud: round sprites per colour, far points first in 3-D, with the same orbit camera, hover and lasso. Colours follow the same rules, with colormap indices computed in the reference's f32 arithmetic so that they match. The fallback draws at most 100,000 points, which is the points data view's default budget, and samples a larger scene with a stride.

Everything else is not drawn, and the view says so. A notice appears as the first line of the scene view's body, under its header, in the faint text style, while the scene holds primitives the fallback cannot draw. Its wording names only the kinds that the scene holds, in this order: lines, arrows, boxes, meshes and images.

> This view shows only the scene's points, because this browser has no WebGPU. The scene's lines, boxes and images (9 entities) are not drawn.

When the points were sampled, a second sentence follows: "It draws 100,000 of its 1,250,000 points." The notice's tooltip reads: "Lines, boxes, meshes and images in 3-D need WebGPU. Open this page in a browser with WebGPU enabled to see the whole scene." The notice reads the same in a 2-D scene, where images are not drawn either.

The two other new kinds are simpler in the fallback. The text log is plain DOM: a scrolling list of lines with their stamps, inside an element with the `log` role, that follows the newest line unless the user scrolls back, keeps its newest 10,000 lines and lets text be selected, searched and read by a screen reader. The bar chart is Canvas 2D, drawn with the axes of the existing plot code as the histogram view draws its bins.

The fallback also draws the [scale bar](scenes.md#the-scale-bar) on its canvas, by the same rule and from the same shared table of tests as the view layer: measured at the orbit target in 3-D, exact in 2-D, hidden with the same display option. With WebGPU the bar is not a draw of this backend; the view layer draws it with ImGui over the picture, as natively. Its capture rows run in headless Chromium on both paths, and it is sized in [Scenes](scenes.md).

## Work items

| Work item | Size in builder days | Depends on |
|---|---|---|
| The scene draws on the WebGPU backend of the graphics layer: vertex layouts with stride 0, the pipeline descriptors and their pick variants, the frame's uniform arena, buffers with capacity and a spare pool | 4 | The graphics layer and its WebGPU backend in Renderer core; the `web/wasm` module of sprint 2 |
| The scene shaders in WGSL by hand, from this page, then moved into the one shader source | 3, and 1 for the move | The shader generator of Renderer core |
| The index clamp pass, the texture fills for the image kinds with their channel expansion and depth scaling, encoded frames through the browser's decoder | 2 | `fill_texture_from_buffer` of Renderer core; wire protocol 3.2's scene messages |
| The class table texture with the palette rule, and the colormap blocks of Renderer core cached by range | 1 | The colormap keys of Renderer core |
| Picking: the pick pass, its readback, hover values from the heap copies | 2 | The scene core's pick pass |
| Budgets, the refusal line and rebuilding after a lost device | 2 | The data view budgets of Scenes and Wire protocol 3.2 |
| Parity tests in headless Chromium, with the compatibility device in CI | 4 | CPU reference rows for scenes from Renderer core |
| The scene fallback in TypeScript: entity transforms, points only, the notice | 2 | The scene messages in the TypeScript codec |
| The text log in DOM and the bar chart in Canvas 2D | 3 | Kinds 7 and 8 confirmed by sprint 2's kind table |

The total is 24 builder days. The native fallback through Dawn is sized in [Renderer core](renderer-core.md). The browser runtime of sprint 3's spec format shows a `scene` panel through the same scene view and needs nothing more from this page.

## Limits

- **No zero-copy in a browser or on Dawn.** Every array is copied twice in a browser, from the network into the heap and from the heap into the GPU process. Import never holds there or on the Dawn fallback.
- **Memory.** A scene is held in the heap and on the GPU, within 256 MiB, under a heap that cannot pass 4 GiB.
- **Pixels.** Colours are exact, but edges and lit surfaces follow the tolerances above, not byte equality.
- **Lines.** Joins are not mitred, and wide alpha-blended strips darken where segments overlap.
- **Transparency.** Translucent draws are not sorted.
- **Picking.** A hover answer arrives one or two frames late.
- **The geometry service.** It needs a default device, so a compatibility device shows scenes but not the service's applets.
- **The index clamp of large meshes.** The clamp pass as drafted dispatches one row of workgroups of 64 indices, which passes WebGPU's limit of 65,535 workgroups in one dimension above 4,194,240 indices, about 1.4 million triangles. The built pass needs the grid-stride loop that the edge gather already has, before a mesh at the limit of [Scenes](scenes.md) can be drawn.
- **Browsers.** Only Chromium was used for checks. Safari and Firefox are untested, as in the spike.
- **Performance.** No frame time in this page was measured; the frame-time test sets the target.

## Decisions

- **Instance-step and vertex buffers for scene draws, no storage buffers in their vertex stage.** Compatibility mode allows none, offsets read arrays without copies, and stride 0 serves absent streams.
- **The native 192-byte parameter block in 256-byte uniform slots with dynamic offsets, and no immediate data in this sprint.** One CPU function fills the block for every backend, 64 bytes of immediate data cannot hold it, and Renderer core uses none yet.
- **One 4-byte colour source per item, read as bits.** One stream serves values, RGBA and class ids, and NaN values survive to the colormap.
- **Renderer core's colormap, decided by integer keys, in every stage.** No adapter's float arithmetic can then move an index.
- **Depth and segmentation textures hold bits.** The integer decision then sees NaN and infinite depths as they are.
- **Lines as screen-space quads with extended ends and no joins; arrow heads flat in screen space.** Both keep their width in pixels and fit one instanced draw.
- **Index-pair edges gathered into the segment streams, decided by the user.** A compute gather keeps `vs_lines` unchanged and the vertex stage free of storage buffers, at 24 to 32 bytes per edge, where fetching both ends by index in the vertex stage would need a new variant and a positions texture per batch.
- **Box poses built in the vertex stage from center, half sizes and rotation.** The arrays bind as they arrive, and the pose is the matrix of a `caliper.geometry.v1_3` instance.
- **Flat faces from derivatives for meshes without normals.** Scenes asks for flat faces, and derivatives give them without duplicating vertices.
- **The scene core's pick pass, 16 × 16 pixels.** It is exact for every primitive at a CPU cost independent of the item count.
- **Mesh indices clamped by a compute pass.** WebGPU does not define an out-of-range index the way the native clamp does.
- **Encoded frames decoded by the browser.** The module stays free of an image decoder.
- **`@interpolate(flat, either)` for the pick identity.** Compatibility mode refuses the default sampling.
- **Level of detail at the producer only.** Sampling where the data lives keeps both copies small.
- **Default WebGPU limits only.** One path then runs on every adapter and at both feature levels.
- **The scale bar as overlay lines and text, decided by the user.** The view layer draws it, so the native view and the WebGPU view show the same bar, and the fallback repeats one small rule instead of adding a draw.

## Open decisions

- **The edge of scene points.** Antialiased disc edges need alpha blending and give up depth writes for points; this page and [Scenes](scenes.md) draw hard-edged discs, which keep depth and parity simple. Recommendation: hard edges, with antialiasing only for points drawn without depth.
- **Arrow heads in 3-D.** A cone would be lit and sized in scene units; this page and [Scenes](scenes.md) draw a flat triangle in screen space, which matches the shaft's pixel width and needs no lighting. Recommendation: the flat head, with a cone as a later option.
- **The text log with WebGPU present.** It can stay in the view layer or always be the DOM view. Recommendation: always the DOM view, because text in the DOM can be selected, searched and read aloud.
- **Decimation while the camera moves.** It would keep orbiting smooth over the budget at the cost of a second level of detail. Recommendation: not now; revisit if the frame-time test fails.
- **The heap cap.** 512 MB holds a 256 MiB scene; 2 GB would hold larger ones and costs nothing until used. Recommendation: keep 512 MB until a demo needs more.
- **Requesting higher limits.** Larger buffers would allow arrays above 256 MiB on most GPUs. Recommendation: no, the budgets keep arrays far smaller.
