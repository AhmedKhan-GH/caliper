# Rendering

Caliper draws tensors that already live on the GPU, so its renderer has to speak the graphics API the tensors are
reachable from. OpenGL cannot do that on a Mac: it is deprecated there, capped at version 4.1, and has no path to
the memory that PyTorch's MPS backend allocates. Every Mac tensor would take a round trip through the CPU on its way
to becoming a texture, in the hottest loop the framework owns.

The answer is to keep graphics out of the contract entirely. This page describes the seam that does it, the
backends behind it, how each one updates a texture, and how the geometry path draws 3-D data. How a tensor reaches
pixels without a CPU round trip, and what each path costs, is in [Zero-copy tensor visualization](zero-copy.md).

## Why the ABI never names a graphics API

Applets render only through ImGui and ImPlot. A tensor crosses to the renderer as a [`CaliperTensor`](../references/tensor.md),
and what comes back is an opaque `CaliperTextureId`. The host keeps a table from that id to the backend's own handle,
so the id is never a raw OpenGL, Metal or Vulkan object. Because the contract hides the backend, swapping or adding
one needs no epoch bump and no applet rebuild.

The seam is `HostRenderer` (`src/host/renderer/host_renderer.h`), a host-internal interface the ABI never sees. Each
backend is its own translation unit, and the host picks one at startup.

## Backends

| Backend | Selected | Update path | Notes |
|---|---|---|---|
| Metal | Default on macOS | A compute pass (f32 with a colormap) or a blit (u8) reads the MPS `MTLBuffer` in place | No CPU staging. Apple-only translation unit |
| Vulkan with CUDA interop | Default on Windows | A CUDA kernel writes the Vulkan image directly; see [the update ladder](#the-update-ladder-on-windows) | Details in [Vulkan and CUDA on Windows](vulkan-cuda.md). Windows-only translation unit |
| OpenGL 3.3 core | `CALIPER_RENDERER=gl`, or the only backend on other platforms | The bridge colormaps on the CPU and uploads the result | The fallback, loaded through GLEW. `tex_update_from_device` always returns `false` |
| WebGPU | The browser views (`web/wasm`) | A WGSL compute pass colormaps the host-memory snapshot into the texture | Browser only; see [the WebGPU backend](#the-webgpu-backend) |

`core_select_renderer()` in `src/host/core_lifecycle.cpp` picks the default: Metal on macOS and Vulkan on Windows.
`CALIPER_RENDERER=gl` selects OpenGL on either. If the native backend fails to initialize (no driver, a remote
desktop session), `src/main.cpp` recreates the window with OpenGL. Every backend produces the same pixels
for the same tensor: the gfx test suite runs one matrix per backend and compares each readback byte for byte against
the CPU reference conversion (`map_f32_to_rgba8` and `expand_u8_to_rgba8`). The suite has an environment for OpenGL,
for Metal, for Vulkan (the CPU-staged rows, which run on any Vulkan driver) and for Vulkan with CUDA (the device
rows, which run when the paired CUDA device is the only one in the machine).

## The WebGPU backend

The browser views are the native view layer compiled to WebAssembly, and `src/host/renderer/webgpu_renderer.cpp`
draws them behind the same `HostRenderer` seam, on the browser's own WebGPU through the standard `webgpu.h`. It
covers what the live views use:

- **The frame.** The renderer draws on the canvas's WebGPU surface and records ImGui's draw data through
  `imgui_impl_wgpu`, with nearest sampling for magnified heatmap cells.
- **Textures.** RGBA8 textures for images and heatmaps. Every snapshot arrives as host memory, so the bridge asks the
  renderer to colormap it through `tex_update_mapped_from_host`, an additive hook that native backends leave at its
  default `false`. A WGSL compute pass then gives exactly the bytes of the CPU reference `map_f32_to_rgba8`: the CPU
  derives, from that function itself, the value at which each of the 255 colour steps begins, and the shader only
  compares values with those thresholds, so the GPU's rounding cannot change a colour.
- **Points.** A points payload is copied into a GPU buffer through `HostRenderer::import_host_allocation`, and each
  point is drawn as an instanced quad of the native point size, with the native colour rule and depth test, because
  WebGPU draws a point as one pixel. Released buffers are reused for the next snapshot of the same size.
- **Lifetimes.** A released texture is destroyed at once, and its view is released at the end of the frame, after
  ImGui's backend has dropped the bind groups it made while recording that frame.

It refuses the rest of the interface: device tensors and imported GPU memory (a browser has neither), geometry other
than points, and the embed canvas. Its two shaders are written by hand in WGSL, because WebGPU has no push constants,
draws points one pixel wide and limits the storage buffers of a vertex stage, which changes the native shaders' design
and not only their syntax. `web/wasm/README.md` gives the module's build, its size and its checks. The same renderer
is planned as the native fallback through Dawn, chosen only when the Metal or Vulkan renderer is unavailable, and
the OpenGL fallback is to be retired once it covers that case.

The host's own screens carry no raw OpenGL either. The landing screen, the launch flow and the dashboards are drawn
with ImGui, so they look the same on every backend.

## The pixel-space contract

Applets never issue graphics calls. They describe what to draw in ImGui terms and hand tensors to the bridge, and the
host owns the window, the contexts and every GPU call. A tensor becomes a texture through
[`caliper.tensor_bridge.v1`](../references/service-tensor-bridge-v1.md). That service is the one place where the backend
difference lives: device compute or blit on Metal and Vulkan, CPU staging on OpenGL. The applet runs the same code on
every backend.

## The update ladder on Windows

The Vulkan backend tries three rungs for each device-resident CUDA tensor and takes the first that accepts it. The
string that `last_device_path()` reports says which one ran.

1. **Direct** (`direct-cmap` for f32 with a colormap, `direct-blit` for u8). A CUDA kernel, launched on the producer's
   stream, reads the tensor where it lies and writes the Vulkan image through a surface object. There is no copy and
   no Vulkan compute pass.
2. **Buffer** (`compute` or `blit`). The tensor is copied inside VRAM into a buffer shared with Vulkan, and a Vulkan
   pass turns that buffer into the image. This rung takes over when the direct path is unavailable or refuses the
   tensor, and for the offscreen canvas.
3. **CPU staging.** The bridge converts on the CPU and uploads. This is the last rung, and the only one for a CPU
   tensor.

Each rung keeps the output byte-identical to the CPU reference. When a rung is unavailable the next one runs, and the
startup log says so, as the [fallback rule](architecture.md#platform-rules) requires.

## Data flows one way

Data moves from tensors to pixels to ImGui, and no further, as the [platform rules](architecture.md#platform-rules)
require. There is no render-to-tensor entry point, no readback on any hot path, and no way for an applet to supply its
own shader. The
backends' `debug_readback_rgba8` exists for the test harness only. Training consumes the same tensors the renderer
reads and never the renderer's output.

Several things look as though they need a new host capability and do not:

- **Static assets** such as CAD, OBJ or glTF shells. The applet loads the file and fills vertex and index tensors; the
  host never learns about file formats.
- **Picking, hover and selection.** The applet casts rays against its own tensors. Interaction is math the applet owns.
- **Annotation and heads-up displays.** ImGui composites over the view image.
- **Training on the visualized data.** Training reads the tensors the renderer reads.

## The geometry path

Textures show tensors as images. The geometry path draws a tensor as 3-D geometry. The applet writes vertices,
indices, normals and attributes into device memory, imports them into the host once through
[`caliper.tensor_bridge.v1_2`](../references/service-tensor-bridge-v1.md), and calls
[`caliper.geometry.v1` and its revisions](../references/service-geometry-v1.md) each frame. The host draws the geometry
in place into an offscreen view, and the view is an ordinary `CaliperTextureId` that the applet shows with
`ImGui::Image`.

- **Offscreen views.** A view is a texture the host renders into. It lives in the same id table as bridge textures, so
  `ImGui::Image` draws it like any other texture.
- **Depth views.** A view created with the depth flag carries a D32 float depth buffer that is private to the renderer.
  Setting depth state against a view without one is refused, never ignored.
- **Vertex pulling.** There is no vertex-input state. The vertex shader reads positions, indices, normals and
  attributes straight out of the imported buffers by vertex index, and an indexed draw is issued as a non-indexed draw
  of `index_count` vertices. One mechanism serves every topology on both backends.
- **One atomic, gated frame.** `draw_primitives` renders a whole frame from an array of draw descriptors. Every draw
  is checked before anything is encoded, and one failed check refuses the whole frame with the previous pixels
  intact. A begin and end API could fail after the clear, which would break that guarantee.
- **Growth by records.** Textures on meshes and instanced transforms were added by growing the draw record with
  appended fields, not by adding entry points. A revision is a new service id whose table and record begin with the
  previous revision's bytes, gated by a new capability bit, and a host draws a zero tail exactly as the earlier
  revision did (see the [growth rules](compatibility.md#growth-rules)). Both backends stay byte-exact against one shared
  CPU reference.

The Vulkan backend keeps a pipeline cache keyed by topology, blend mode and depth state, a descriptor set per draw,
and a ring of per-draw parameter blocks. The Metal backend keeps the equivalent pipeline and depth-stencil caches.
Metal compiles its shaders at run time, and Vulkan embeds SPIR-V built at compile time
([Build-time SPIR-V](vulkan-cuda.md#build-time-spir-v)).

## The scene core

The renderer core's scene core is in `src/host/scene/`, in the namespace `caliper_scene` and the library
`caliper_scene`. It is the one camera and scene model that the views are to share, and it compiles a scene into a draw
list of the renderer core's graphics layer (`src/host/gfx/`). It is plain C++17 that never calls a GPU. Nothing uses
it yet: no view, no service and not the browser draws through it, and the points live view keeps its own camera.

It holds these parts:

- **The scene store.** A scene has a space (3-D or 2-D), an up axis and a unit note. Its entities are named by paths
  of 1 to 16 segments, each of the characters `A-Z a-z 0-9 _ . -`, at most 255 bytes in all, and a scene holds at most
  1,024 entities with the root among them. They are stored as arrays with each parent before its children. An entity
  holds an entity transform, a pinhole camera, and one batch of each primitive type. A batch describes arrays in
  graphics-layer buffers by role (the buffer, byte offset, count, stride and format) and never copies their data.
  Every part has a stamped layer and a static layer: a clear removes the stamped parts, and a static clear removes both.
- **World matrices.** One forward pass over the changed subtrees composes each entity's world matrix in double
  precision. An entity whose world pose is not finite is hidden with its subtree. The children of a camera entity live
  in the camera's pixel space, on its image plane.
- **Cameras.** The orbit turns about any of the six axes with a 45° field of view, and its near and far planes follow
  the bounds as the points view computes them. Azimuth 0 lies along the first of +x, +y and +z that is perpendicular to
  the up axis. Switching the up axis keeps the target, the distance and the eye position. The 2-D pan and zoom has y up
  or y down. Looking through a pinhole projects with its intrinsics, letterboxed to the view. The starting camera of a
  scene and reset are part of a view's camera state. Clip space has x right, y up and depth from 0 at the near plane to
  1, and matrices are column-major.
- **Bounds and fitting.** A batch's bounds come from the producer, or else from a scan of its CPU-visible array. They
  are moved to world space by their eight corners. Fitting frames the bounding sphere with the orbit, or the larger
  extent with the pan and zoom.
- **Compiling.** For the visible entities, the compiler chooses each batch's program from the generated program table
  and binds its arrays as vertex or instance-step buffers at their offsets. An absent array reads a constant at stride
  0. The compiler writes the frame block, each draw's parameter block and the colormap blocks into the draw list's arena,
  and appends the draws: opaque ones first, then translucent ones, each in the byte order of the entity paths. When the
  view looks through a camera, that camera's pixel space is drawn flat over the picture last. A separate step records
  the index clamp of meshes and the edge gather of index-pair edges as compute dispatches. Every primitive compiles:
  points, lines as segments, strips and index-pair edges, arrows, boxes with their faces, meshes, images in 2-D or on a
  camera's image plane with their depth clouds, segmentation, camera frustums and entity axes. Labels are not draws,
  and the scene core gives their projected anchors instead.
- **Picking.** The pick pass compiles the same draws with their pick entries into a 16 x 16 `RG32Uint` window around
  the pointer, which it reaches by zooming clip space. Decoding a read-back window gives the entity, the primitive and
  the item under the pointer, the world position from the depth, and for an image the pixel.

The scene core asks a `Resources` object for the handles its records name: pipelines of the generated programs, bind
groups, the buffer of its constant streams, the buffers it owns for clamped indices and gathered edges, and the
colormap blocks. `BackendResources` provides them over a backend of the graphics layer and is built when the layer's
core is part of the build.

Its tests are `caliper_scene_tests` (`tests/scene/`). They run headless with no GPU and check the matrices, every
camera, the fits, compiled draw lists read as data, clears and static items, the limits, and picks decoded from
synthetic ID texels. Where the layer's core is part of the build, they also submit every fixture scene to the layer's
recording backend, whose validation must accept it. They build and run with any preset, for example
`cmake --build cmake-build-notorch --target caliper_scene_tests` and then `ctest -R caliper_scene_tests` from the build
folder. The graphics suite's row of a one-entity scene, drawn on Metal and read back byte for byte, has its fixture
and CPU reference in `tests/scene/scene_row_fixture.h`, and `caliper_scene_tests` checks the compiled draw against that
reference on the CPU.

## Why Metal pipelines by queue commit order

The Metal renderer submits tensor operations and the frame's drawing to one `MTLCommandQueue`, and Metal starts the
command buffers on one queue in the order they were committed. That gives most of what the Vulkan backend needs
explicit machinery for:

- **Draw after update.** The tensor operation is committed during the applet's `frame()`, before the frame's own
  command buffer, so the frame samples finished texels with no added synchronization.
- **Resource lifetime.** A command buffer retains the resources it references until it completes, so a buffer or a
  texture released mid-flight stays alive until the GPU is done with it. The renderer never encodes unretained.
- **Rapid re-update.** Two updates to one texture are two command buffers on one queue, serialized by commit order.
  Each update allocates a fresh command buffer, so there is nothing to re-record.

Metal therefore keeps no per-texture sync objects, no back-pressure wait and no list of pending waits. An
`MTLSharedEvent` appears only where a second queue exists, which is the producer's queue when a stream-ordered handoff
is in use. The remaining CPU waits are in readbacks, the offscreen canvas, the geometry rigidity check and a rare fallback, not in the update path.

The larger stall was never in the renderer. Draining the whole device at the handoff was the dominant cost, and that
is what the stream-ordered handoff removes; [Zero-copy tensor visualization](zero-copy.md#the-sync-ladder) covers it.

## The graphics layer

The renderer core's graphics layer is in `src/host/gfx/`, in the namespace `caliper_gfx`. It is an internal C++17
layer shaped to WebGPU's model and default limits, and nothing of it crosses the ABI. No view or service draws through
it yet: the renderers above keep their own paths, and their clients are to move onto the layer one at a time.

The layer offers these pieces:

- **Resources.** Buffers, textures, samplers, programs, render and compute pipelines and bind groups, named by typed
  64-bit handles. A released handle is refused rather than reaching another resource, and the object behind it is
  destroyed only after the last submit that used it has completed.
- **Draw lists.** A draw list holds the plain records of the draws for one target. Each record names a pipeline, up
  to four bind groups and four dynamic offsets, up to eight vertex buffers at any 4-byte-aligned offset (read per
  vertex, per instance or as a constant), an optional buffer of u32 indices, the counts and a pick id. A target has a
  colour attachment, an optional `RG32Uint` ID attachment and an optional `Depth32Float` depth attachment.
- **The uniform arena.** `DrawList::uniform` copies a parameter block into a 256-byte-aligned slice of the list's
  arena and returns its dynamic offset. The backend uploads each arena once per submit.
- **Frames.** A frame is an ordered list of producer waits, compute dispatches, buffer-to-texture copies and render
  passes. A submit returns a serial, and completion is polled. A blocking wait exists only where the backend allows
  one.
- **Validation.** Every descriptor and every frame is checked against the backend's limits before anything is
  encoded. One bad draw refuses the whole submit and leaves its targets untouched, and `last_error()` says why.
- **Readback.** A texture region is read back asynchronously.
- **Import.** External memory becomes a layer buffer or texture, and the result reports the route: in place, copied
  on the GPU, copied from the host, or refused. `take_stats()` counts the imports, copies and bytes copied, so an info
  line can say what ran.

Two backends exist. The Metal backend runs on the Metal renderer's own device and queue (`HostRenderer::native_device`
and `native_queue`), so the layer creates no second device, and its work is ordered with the renderer's by commit
order. It imports an `MTLBuffer` of that device, and page-aligned host memory, in place, and it copies other host
memory. It runs the programs of the shader generator ([Write a shader](../guides/write-a-shader.md)), and it binds the
buffer of storage sizes that their MSL reads. It keeps Metal's fast paths:

- bind groups are resolved once into Metal objects;
- dynamic offsets move with `setVertexBufferOffset`;
- state that repeats from one draw to the next is not set again;
- command buffers do not retain their resources, because the layer defers every release.

The recording backend has no GPU. It records each frame and each resource operation, and it keeps buffer and texture
contents in host memory for headless tests. A WebGPU backend for the browser and a Vulkan backend are planned.

The layer's tests run in two suites:

- `caliper_tests --test-case="gfx layer*"` runs the core on the recording backend. It covers the handles, the
  limits, the arena, deferred release, readback, the routes of import, and the refusal of bad frames.
- `caliper_gfx_tests --test-case="gfx/Metal layer*"` runs the layer's rows of the graphics suite on Metal. One row
  draws a list of quads with an indexed draw, depth, additive blending and the ID target. Another draws the
  generated points program from an imported `MTLBuffer`, and checks that the route is reported as in place, with
  zero bytes copied. Each target is read back through the layer and compared byte for byte with a CPU reference.
  Further rows run the generated index clamp as a dispatch and make every generated program's pipelines.
  `CALIPER_GFX_CAPTURE_DIR` writes PNG captures of the rows, and `CALIPER_GFX_LAYER_BENCH=1` runs a row that times
  the CPU encoding of 1,000 draws a frame on both backends.
