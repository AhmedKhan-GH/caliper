# Renderer core

This page plans the renderer core, the first work item of sprint 4: a graphics layer that each backend implements once, one shader source translated for every backend, and a scene core that every view draws through. It is designed before [scenes](scenes.md) and [web geometry](web-geometry.md), so that scenes, the browser's geometry and the native WebGPU fallback through Dawn are each built once instead of once per backend. It records what the renderers look like today, specifies the layer's interface, chooses the shader language, defines the camera and scene conventions every backend shares, expresses zero-copy as a backend capability with a measurement that proves the fast path keeps its speed, weighs adopting an existing library, and orders the migration, which, as the user decided, keeps no old path, retires OpenGL early, leaves each graphics service with one version and moves applets from the geometry service to a scene service.

## What is built and what is planned

Nothing on this page is built. It rests on these built pieces:

- **The renderer seam**, `HostRenderer` in `src/host/renderer/host_renderer.h`, with its Metal, Vulkan and OpenGL backends on `main` (Rendering (`docs/explanations/rendering.md`)) and the browser's WebGPU backend on the sprint 2 branch (`src/host/renderer/webgpu_renderer.cpp`).
- **The zero-copy paths** of Zero-copy tensor visualization (`docs/explanations/zero-copy.md`) and Vulkan and CUDA on Windows (`docs/explanations/vulkan-cuda.md`): in-place reads of MPS `MTLBuffer`s, imported allocations, the CUDA direct path and the sync ladder.
- **The geometry service**, `caliper.geometry.v1` to `v1_3` (`docs/references/service-geometry-v1.md`), and the tensor bridge (`docs/references/service-tensor-bridge-v1.md`), whose revisions step 13 of the migration collapses into one version, while the geometry service is deleted once the scene service has replaced it, as the user decided ([One version of each graphics service](#one-version-of-each-graphics-service)).
- **The live views** (heatmap, image, points, series, histogram) and the shared-memory session of Live streaming (`docs/explanations/live-streaming.md`).
- **The pixel-exact graphics suite** (Testing (`docs/explanations/testing.md`)), which compares every backend's readback byte for byte with one CPU reference, and the benchmarks in `python/bench/`.
- **The web viewer spike** of the sprint 2 branch (`docs/plans/web-viewer-spike.md` there), which found the WebGPU constraints this design is shaped to.

## The problem

The facts below were read on `main` and on the sprint 2 branch. Where they differ from the problem as first reported, the difference is stated.

### Shader copies

| Shader | Metal (MSL strings in `metal_renderer.mm`) | Vulkan (GLSL in `src/host/renderer/shaders/`, SPIR-V at build time) | CUDA (`kernels/cmap_direct.cu`) | WebGPU, sprint 2 (WGSL strings in `webgpu_renderer.cpp`) |
|---|---|---|---|---|
| Colormap, f32 to RGBA8 | `cmap_f32`, 31 lines | `colormap.comp`, 43 lines, push constants | `caliper_cmap_f32`, 24 lines | `cmap_f32`, 37 lines, CPU-derived thresholds |
| u8 image copy | An API blit | An API copy | `caliper_blit_rgba8`, 12 lines | An upload |
| Points | `points_vs` and `points_fs`, 47 lines, `[[point_size]]` | `points.vert` and `points.frag`, 52 lines, push constants, `gl_PointSize` | None | `vs` and `fs`, 47 lines, instanced quads |
| Geometry | `geom_vs`, `geom_vs_point`, `geom_fs`, `geom_tex_fs`, 190 lines | `geom.vert`, `geom.frag`, `geom_tex.frag`, 184 lines, `gl_PointSize` | None | None |

The colormap exists four times by hand and the points shader three times. The geometry shaders exist twice, because the browser draws points only. The OpenGL backend has no shaders and no device path. The PTX (153 lines) is generated from the CUDA source and checked in, so it is not a separate hand copy. Push constants appear in two of the six GLSL shaders (`colormap.comp` and `points.vert`); Metal passes the same parameters as inline bytes. A hardware point size appears in four entry points. The colormap's index rule, `idx = clamp((v - vmin) / (vmax - vmin), 0, 1) * 255 + 0.5`, is written eleven times in shader code across MSL, GLSL, CUDA and WGSL, counting the TypeScript fallback's `lut_index`, beside the CPU reference `map_f32_to_rgba8`. The parameter block of the geometry shaders exists in four hand-synchronised copies, as a comment in `geom.vert` warns.

### The host renderer's methods

`HostRenderer` has 42 virtual methods on `main` and 47 on the sprint 2 branch. The spike added `tex_update_mapped_from_host` and `import_host_allocation`, and window capture added three.

| Group | Methods | Count |
|---|---|---|
| Window and frame | `init`, `new_frame`, `render`, `shutdown`, `name`, `window_hints` | 6 |
| Embed canvas | `canvas_supported`, `canvas_init`, `canvas_resize`, `canvas_new_frame`, `canvas_render`, `canvas_read_pixels`, `canvas_shutdown` | 7 |
| Window capture (sprint 2) | `window_capture_supported`, `window_capture_request`, `window_capture_poll` | 3 |
| RGBA8 textures | `tex_create_rgba8`, `tex_upload_rgba8`, `tex_release`, `tex_imtexture_id` | 4 |
| Texture updates, one per source | `tex_update_from_device`, `tex_update_from_imported`, `tex_update_mapped_from_host` (sprint 2), `last_device_path` | 4 |
| Completion, one family per kind of work | `device_update_wait`, `device_update_retain_last`, `device_update_is_done`, `device_update_release`, `geom_draw_retain_last`, `honors_stream_ordered_handoff` | 6 |
| Device and interop | `native_device`, `interop_device`, `alloc_device_shared` | 3 |
| Import, one per source | `supports_external_import`, `import_external_allocation`, `import_host_allocation` (sprint 2), `release_external_allocation` | 4 |
| Geometry, one per ABI revision | `supports_geometry`, `supports_geometry_primitives`, `supports_geometry_textured`, `supports_geometry_instanced`, `geom_create_view`, `geom_create_view_ex`, `geom_draw_points`, `geom_draw_primitives` | 8 |
| Tests only | `debug_readback_rgba8`, `debug_force_buffer_path` | 2 |

Eight methods are capability booleans. Each feature added an entry point shaped to that feature rather than a general resource, so the browser's points needed `import_host_allocation` and its heatmap needed `tex_update_mapped_from_host`.

### Platform details in the views

- `bind_buffer` in `src/host/live/live_views.cpp` accepts a `CALIPER_DEV_METAL` tensor and imports it as `CALIPER_ALLOC_HANDLE_MTLBUFFER`, tries the browser's host import, and otherwise stages a CPU payload into a new `MTLBuffer` made on `native_device()`, refusing on any other renderer with "CPU payload needs a Metal renderer". Its helpers live in `live_views_metal.mm`.
- The points view labels its path from the tensor's device (`imported = device == CALIPER_DEV_METAL`), not from what the renderer did.
- The embed v1.2 views (`embed_core.cpp` on `main`, `embed_views.cpp` on sprint 2) have a Metal branch for device tensors that never falls back to a copy.
- `live_core.cpp` needs `native_device()` to open a shared-memory or network source, because the slots become `MTLBuffer`s, and the public `include/caliper/live.h` carries an `mtl_buffer` field.
- `HostGeomDraw` carries a CPU pointer to a 256-entry colour table that Metal binds as inline bytes.
- `main.cpp` compares `renderer_->name()` with `"gl"`.

### Drawing worlds

Three drawing worlds share no camera or scene model natively, and the browser has a fourth:

1. **ImGui and ImPlot**: the series and histogram live views, the host's screens, and ImPlot3D in five applets (`embed_scope`, `field_scope`, `flow_scope`, `gpt_scope`, `sculpt_scope`), which projects on the CPU with its own camera.
2. **The geometry service**: offscreen views drawn from imported allocations with camera matrices the applet owns. The points live view calls the same renderer entry points directly, with its own orbit and 2-D camera in `live_views_state.h`.
3. **Texture views**: the heatmap and image live views and the embed v1.2 views, bridge textures shown with `ImGui::Image`.
4. **The browser's TypeScript views**, the fallback where WebGPU is absent, with their own camera (`web/src/views/camera.ts`) and their own WGSL for heatmaps and points. Their WebGPU renderers are removed in step 9 of the migration, as the user decided, and their Canvas 2D drawing stays as the fallback ([The TypeScript views in the browser](#the-typescript-views-in-the-browser)).

### Backends

| Backend | Lines | Where | Device path | Geometry |
|---|---|---|---|---|
| Metal | 1,815 on `main`, 1,908 on sprint 2 | macOS, default | Yes | `v1` to `v1_3` |
| Vulkan with CUDA | 4,143 | Windows only, default | Yes, with the CUDA direct path | `v1` to `v1_3` |
| OpenGL 3.3 through GLEW | 138 | Fallback on both | No | No |
| WebGPU | 872 and a 181-line header, sprint 2 | Browser only | Host bytes only | Points |

Dawn is a fifth role rather than a fifth implementation: the WebGPU backend is written against the standard `webgpu.h`, which Dawn implements natively. Without a shared layer, scenes would still have to be written three times (Metal, Vulkan, WebGPU), and the shaders four.

## The graphics layer

### Its shape

The graphics layer is a thin internal C++ layer, namespace `caliper_gfx` in `src/host/gfx/`, that offers buffers, textures, samplers, pipelines, bind groups and draw lists. Nothing of it crosses the ABI. It is shaped to WebGPU's model and default limits, so that the browser backend is a direct mapping and the native backends map down to it without loss:

- **Bind groups**, at most four per pipeline: group 0 the per-frame block and the class table, group 1 the per-draw block and the colormap table at dynamic offsets, group 2 images and samplers, group 3 storage buffers for vertex pulling only. Groups 0 to 2 are the layout [Web geometry](web-geometry.md) validated in a browser.
- **Uniform buffers with dynamic offsets** in place of push constants. Each frame owns a uniform arena; a draw's parameters are a 256-byte-aligned slice of it.
- **No point size and no wide lines.** New programs draw points and wide lines as instanced quads expanded in the vertex stage. Hardware point lists and line lists remain in the interface, one pixel wide, because WebGPU has them; until step 13 the geometry service's byte-exact rows depend on them, and after it no client uses them.
- **Vertex and instance-step buffers** feed per-item data: an attribute can sit at any 4-byte-aligned offset and any stride up to 2,048 bytes, so an imported slot or a column of a wider array is bound without a copy and without the 256-byte alignment that storage bindings need.
- **Storage buffers in the vertex stage** are an optional capability, used only by the geometry service's vertex pulling. WebGPU's compatibility mode allows none there. Step 13 removes the capability and group 3 with the geometry service, since nothing else needs them.
- **Asynchronous completion everywhere.** Work is submitted and returns a serial; completion is polled. A blocking wait exists only where the backend allows it, never in a browser.

### Handles and descriptors

```cpp
namespace caliper_gfx {

// A handle is a slot and a generation in 64 bits. 0 is none, and an id is never reused,
// so a use after release is refused instead of reaching another resource.
template <class Tag> struct Handle { uint64_t id = 0; explicit operator bool() const { return id != 0; } };
using Buffer = Handle<struct BufferTag>;  using Texture = Handle<struct TextureTag>;  using Sampler = Handle<struct SamplerTag>;
using Program = Handle<struct ProgramTag>;  using Pipeline = Handle<struct PipelineTag>;
using BindGroup = Handle<struct BindGroupTag>;  using Readback = Handle<struct ReadbackTag>;
using Serial = uint64_t;   // one per submit, increasing

enum BufferUsage : uint32_t { kVertex = 1, kIndex = 2, kUniform = 4, kStorage = 8, kCopySrc = 16, kCopyDst = 32 };
enum TextureUsage : uint32_t { kSampled = 1, kStorageTex = 2, kRenderTarget = 4, kCopySrcTex = 8, kCopyDstTex = 16 };
enum class Format : uint32_t { RGBA8Unorm, BGRA8Unorm, R8Unorm, R16Uint, R32Uint, RG32Uint, R32Float, Depth32Float };

struct BufferDesc  { uint64_t size; uint32_t usage; bool host_visible = false; const char* label = nullptr; };
struct TextureDesc { uint32_t width, height; Format format; uint32_t usage; const char* label = nullptr; };
struct SamplerDesc { bool linear = true; };   // clamp to edge, no mipmaps, as the geometry service fixes

enum class VertexFormat : uint32_t { F32, F32x2, F32x3, F32x4, Unorm8x4, U8, U32, S32, U32x2 };   // stride 0: constant
enum class Step : uint32_t { Vertex, Instance };
struct VertexAttribute { uint32_t location; VertexFormat format; uint32_t offset; };
struct VertexLayout { uint32_t stride; Step step; uint8_t attribute_count; VertexAttribute attributes[4]; };

enum class Topology : uint32_t { TriangleList, TriangleStrip, LineList, LineStrip, PointList };
enum class Blend : uint32_t { Opaque, Alpha, Additive };   // the geometry service's three equations
struct RenderPipelineDesc {
    Program program; const char* vertex_entry; const char* fragment_entry;
    uint8_t vertex_layout_count; VertexLayout vertex_layouts[8];
    Topology topology;
    Format color_format; Blend blend;
    bool id_target = false;                  // a second attachment, RG32Uint: pick id and item
    bool depth = false, depth_test = false, depth_write = false;   // D32 float, less-or-equal
    uint8_t constant_count = 0; struct { uint16_t id; float value; } constants[8];  // WGSL override constants
};
struct ComputePipelineDesc { Program program; const char* entry; };

struct BindEntry { uint32_t binding; Buffer buffer; uint64_t offset = 0, size = 0; Texture texture; Sampler sampler; };
struct BindGroupDesc { Pipeline pipeline; uint32_t group; uint8_t entry_count; BindEntry entries[8]; };
}
```

Pipelines take their bind group layouts from the program, which the shader generator describes (see [One shader source](#one-shader-source)), so there is no separate layout object to keep in step with the shader. The layer caches pipelines by a hash of the descriptor, as both native backends do today.

### Capabilities and limits

```cpp
struct Limits {   // WebGPU's defaults are the floor; a native backend reports its own, larger values
    uint32_t max_texture_dimension_2d = 8192, max_bind_groups = 4, max_dynamic_uniform_buffers = 8;
    uint32_t max_storage_buffers_per_stage = 8, max_storage_buffers_in_vertex_stage = 8;  // 0 in compatibility mode
    uint32_t max_uniform_buffer_binding_size = 65536;
    uint64_t max_storage_buffer_binding_size = 134217728, max_buffer_size = 268435456;
    uint32_t min_uniform_buffer_offset_alignment = 256, min_storage_buffer_offset_alignment = 256;
    uint32_t max_vertex_buffers = 8, max_vertex_attributes = 16, max_vertex_stride = 2048, max_color_attachments = 8;
};

enum Cap : uint64_t {
    kCapVertexStorage      = 1ull << 0,   // storage buffers readable in the vertex stage
    kCapPersistentMap      = 1ull << 1,   // host-visible buffers stay mapped
    kCapBlockingWait       = 1ull << 2,   // wait(serial) may block; never in a browser
    kCapTimestamps         = 1ull << 3,   // GPU time per pass
    kCapImportMetalBuffer  = 1ull << 8,   // an id<MTLBuffer> of this device, in place
    kCapImportHostMemory   = 1ull << 9,   // page-aligned host memory wrapped in place
    kCapImportWin32Memory  = 1ull << 10,  // an exportable CUDA allocation, through Vulkan external memory
    kCapImportMetalTexture = 1ull << 11,
    kCapCudaWritesTexture  = 1ull << 12,  // a CUDA kernel writes a layer texture (the direct path)
    kCapWaitMetalEvent     = 1ull << 13,  // reads ordered after an MTLSharedEvent value
    kCapWaitCudaStream     = 1ull << 14,  // reads ordered after a CUDA stream, by a shared timeline semaphore
};

struct Caps { uint64_t bits; Limits limits;
               const char* backend; /* metal, vulkan, webgpu (browser), dawn */ const char* adapter; };
```

The capability bits replace the renderer's five `supports_*` booleans, `honors_stream_ordered_handoff` and `interop_device()`; the import bits are the subject of [Zero-copy as a backend capability](#zero-copy-as-a-backend-capability).

### The draw list

A draw list is the backend-neutral list of draws a view hands to the layer for one target. Its records are plain data, so a view can build one without touching a backend, and the headless tests can inspect one.

```cpp
struct VertexBinding { Buffer buffer; uint64_t offset; };     // offset a multiple of 4

struct DrawRecord {
    Pipeline  pipeline;
    BindGroup groups[4];                 // 0 frame, 1 draw (dynamic), 2 images, 3 storage (vertex pulling only)
    uint32_t  dynamic_offsets[4];        // the frame block, the draw block and the colormap table, in the arena
    VertexBinding vertex[8];             // WebGPU's limit of vertex buffers
    Buffer    index_buffer; uint64_t index_offset = 0;   // u32 indices; none for a non-indexed draw
    uint32_t  count;                     // vertices, or indices, per instance
    uint32_t  instance_count = 1, first_vertex = 0, first_instance = 0;
    uint32_t  pick_id = 0;               // written to the ID target with the item; 0 is not pickable
    uint16_t  scissor[4] = {0, 0, 0xffff, 0xffff};
};

struct RenderTarget { Texture color; Texture id; Texture depth; uint32_t clear_rgba; bool clear = true; };

class DrawList {
public:
    explicit DrawList(const RenderTarget& target);
    uint32_t uniform(const void* data, uint32_t size);   // copies into the arena; returns the dynamic offset
    void draw(const DrawRecord& record);
    const RenderTarget& target() const;  std::span<const DrawRecord> records() const;
};
```

The header as built, `src/host/gfx/gfx.h`, settles what this sketch left open, and where the two differ the header is the reference:

- **A program is created from a `ProgramDesc`,** which holds its entry points, its bindings and its WGSL, MSL and SPIR-V sources. The shader generator emits a table of them, and a program's id is its index in that table.
- **A bind group is made for a program and a group index, not for a pipeline.** It is valid with every pipeline of that program and with any program that declares the same group layout, so group 0 is shared.
- **The Metal binding map is part of the program's description.** Each binding names its Metal index, the same in every stage, and vertex buffer slot i takes a Metal buffer index above those of the bindings.
- **A draw has four dynamic offsets and eight vertex buffers.** The frame block of group 0 lives in the uniform arena beside the draw block and the colormap block, and eight is WebGPU's limit of vertex buffers.
- **The uniform arena is a pseudo-buffer.** A dynamic uniform binding that names it binds the arena of the pass it is used in, so a bind group never changes from frame to frame.
- **Validation lives in the layer's core, once.** A backend implements protected hooks; a refused call returns a null handle, false or serial 0 and leaves its reason to be read. A whole frame is validated before anything is encoded, and one bad draw refuses the submit.
- **Handles carry their kind and a generation and are never reused,** and pipelines are cached by descriptor and counted by reference.
- **The layer compiles as C++17,** the repository's standard, and is used from one thread.

### The backend interface

```cpp
enum class Route : uint32_t { InPlace, CopiedOnGpu, CopiedFromHost, Refused };

struct Stats {   // per submit; the frame's sum feeds the info line and the benchmarks
    uint32_t imports_in_place, imports_copied, copies, draws, dispatches;
    uint64_t bytes_copied_from_host, bytes_copied_on_gpu;
    double encode_cpu_ms, gpu_ms;     // gpu_ms < 0 without kCapTimestamps
};

struct TensorSource { Buffer buffer; uint64_t offset; uint32_t w, h, channels; CaliperDType dtype;
                      int64_t stride_x, stride_y; const CaliperTensor* tensor; ExternalSync sync; };
struct TextureMapping { const uint32_t* lut256; float vmin, vmax; };   // ignored for u8 data
struct UpdateResult { bool ok; Route route; const char* path; Serial serial; };

class Frame {   // the passes of one submit, in order; Vulkan derives its barriers from them
public:
    void wait(const ExternalSync& sync);             // honour a producer's event before the next pass
    void dispatch(Pipeline p, std::span<const BindGroup> groups, std::span<const uint32_t> offsets,
                  uint32_t x, uint32_t y, uint32_t z, std::span<const Texture> writes);
    void copy_buffer_to_texture(Buffer src, uint64_t offset, uint32_t row_pitch, Texture dst);
    void render(DrawList&& list);
};

class Backend {   // implemented once per backend: Metal, Vulkan, WebGPU (browser and Dawn)
public:
    virtual ~Backend() = default;
    virtual const Caps& caps() const = 0;

    virtual Buffer    create_buffer(const BufferDesc&) = 0;
    virtual Texture   create_texture(const TextureDesc&) = 0;
    virtual Sampler   create_sampler(const SamplerDesc&) = 0;
    virtual Program   create_program(const ProgramDesc&) = 0;      // an entry of the generated table
    virtual Pipeline  create_render_pipeline(const RenderPipelineDesc&) = 0;   // cached by descriptor
    virtual Pipeline  create_compute_pipeline(const ComputePipelineDesc&) = 0;
    virtual BindGroup create_bind_group(const BindGroupDesc&) = 0;
    virtual void      release(uint64_t handle) = 0;   // deferred until the last submit that used it completed

    virtual void  write_buffer(Buffer, uint64_t offset, const void* data, uint64_t size) = 0;
    virtual void  write_texture(Texture, const void* data, uint32_t row_pitch) = 0;
    virtual void* mapped(Buffer) = 0;                 // with kCapPersistentMap, else nullptr

    virtual ImportResult import_buffer(const ExternalBuffer&) = 0;
    virtual ImportResult import_texture(const ExternalTexture&) = 0;

    virtual Serial submit(Frame&& frame) = 0;
    virtual bool   completed(Serial) const = 0;        // never blocks
    virtual bool   wait(Serial) = 0;                   // tests and WAIT_GPU; false without kCapBlockingWait
    virtual Serial last_use(uint64_t handle) const = 0;

    virtual Readback read_texture(Texture, uint32_t x, uint32_t y, uint32_t w, uint32_t h) = 0;
    virtual bool     take_readback(Readback, std::vector<uint8_t>* out) = 0;   // false until ready

    virtual uint64_t imgui_texture_id(Texture) = 0;    // what ImGui::Image takes on this backend
    virtual Stats    take_stats() = 0;

    // Operations with a portable default built from the calls above (src/host/gfx/ops.cpp).
    // A backend overrides one only for a faster private path with the same result.
    virtual UpdateResult update_texture_from_tensor(Texture dst, const TensorSource& src, const TextureMapping& m);
    virtual UpdateResult fill_texture_from_buffer(Texture dst, Buffer src, uint64_t offset, uint32_t row_pitch);
};
```

The interface has 24 methods, implemented once per backend, against 47 special-purpose ones today. `update_texture_from_tensor` imports or copies the source, then runs the colormap program for f32 data or a copy for u8 data, and returns the route and the path label the views already show (`compute-imported`, `blit`, `cpu-staged` and the others). Its default covers Metal, Vulkan's buffer path and WebGPU; the Vulkan backend overrides it with the CUDA direct kernel, which writes the layer's texture on the producer's stream. `fill_texture_from_buffer` fills a texture from a buffer region by one GPU pass that also absorbs row pitch, as scenes need for images, depth and segmentation. The window, the swapchain, the embed canvas and window capture stay in a slimmer `HostRenderer`, which owns one `Backend` and the ImGui backend.

### Frames, and why there is no frame graph

A frame is an ordered list of passes: waits on producers, colormap dispatches, one render pass per view that changed, and the ImGui pass, which the ImGui backend records. Each pass declares what it writes, and the Vulkan backend derives its barriers from that; Metal and WebGPU order work on one queue by themselves. A frame graph, with transient resources, aliasing and pass culling, does not earn its place: a frame has a handful of passes, every view target persists across frames because views redraw only when their data, camera or size changed, and nothing runs on a second queue. It would earn its place with post-processing chains, such as selection outlines, ambient occlusion or order-independent transparency.

### How Metal and Vulkan keep their speed

| Concept | Metal | Vulkan | WebGPU |
|---|---|---|---|
| Buffer, texture, sampler | `MTLBuffer`, `MTLTexture`, `MTLSamplerState`, one to one | `VkBuffer` and memory, `VkImage` and view, `VkSampler`, one to one | One to one |
| Pipeline | `MTLRenderPipelineState` and depth state, cached as today | `VkPipeline`, cached as today | One to one |
| Bind group | No object: the entries are bound with `setVertexBuffer` and `setFragmentTexture` at draw time | A descriptor set, one to one | One to one |
| Per-draw uniform | A slice of a shared ring buffer by `setVertexBufferOffset`, or `setVertexBytes` under 4 KB, which is Metal's fastest path and what the backend uses today | A dynamic uniform descriptor offset, about as cheap as a push constant | A dynamic offset |
| Import | Retains the `MTLBuffer`; `newBufferWithBytesNoCopy` over a shared-memory slot | `VkImportMemoryWin32HandleInfoKHR` into a `VkBuffer`, as today | Not available: a copy |
| Ordering | Queue commit order, `MTLSharedEvent` for the stream-ordered handoff, as today | Barriers from the pass list; timeline semaphores shared with CUDA, as today | Queue order |
| Draw list | Encoded record by record into one render encoder | Recorded record by record into one command buffer | One pass encoder |

The draw list adds one loop over plain records per view, a few nanoseconds a draw. The device paths that make Caliper fast stay private paths behind the same interface: the Metal compute pass reads the producer's `MTLBuffer` in place exactly as now, and the Vulkan backend keeps the CUDA direct path, its second image per texture, and the buffer path below it.

### ImGui and ImPlot

ImGui, ImPlot and ImPlot3D keep drawing through ImGui's own backends (`imgui_impl_metal`, `imgui_impl_vulkan`, `imgui_impl_wgpu`), which the slimmer `HostRenderer` initialises on the layer's device and queue. The layer supplies the textures they show: `imgui_texture_id` returns the `MTLTexture`, the Vulkan descriptor set, or the WebGPU texture view, so `ImGui::Image` and ImPlot images draw layer textures unchanged. The frame's layer passes are submitted before the ImGui pass on the same queue. A released texture is destroyed only after the last ImGui frame that drew it completed, and the WebGPU backend tells `imgui_impl_wgpu` to forget its cached bind group, the stale-bind-group finding of the spike.

### What is rebuilt on top

| Client | Rebuilt as |
|---|---|
| Tensor bridge (one version, `caliper.tensor_bridge.v1`, from step 13) | Bridge textures are layer textures; `update_texture` calls `update_texture_from_tensor`; `import_allocation` calls `import_buffer`, and the allocation table, which the geometry service shares, belongs to the layer from step 8; `alloc_shared` on Vulkan becomes a backend-private path behind `kCapCudaWritesTexture` at Windows parity |
| Embed views and the heatmap and image live views | The bridge as above; the path label comes from the returned route, never from the tensor's device. From step 13 the typed live views are the embedding library's one kind of view, and the v1.2 tensor views are folded into them |
| Points live view | A scene of one entity in the scene core; `bind_buffer` becomes one `import_buffer` call, and `live_views_metal.mm` goes |
| Series and histogram live views | Unchanged: ImPlot |
| Geometry service | Not rebuilt. It keeps running on the Metal renderer, reading the layer's imported buffers through an adapter from step 8, until its last applets move to the scene service, and step 13 deletes it, as the user decided |
| Scene service (`caliper.scene.v1`, from step 11) | The applet face of the scene core: an applet's records become entities and batches whose arrays are the bridge's allocations imported through `import_buffer`, drawn by the same scene view as a script's scene ([The scene service for applets](scenes.md#the-scene-service-for-applets)) |
| Export service (one version, `caliper.export.v1`, from step 13) | Draws a scene through the scene core into a layer texture and reads it back with `read_texture` |
| Scenes | The scene core compiling entities into draw lists ([Scenes](scenes.md)) |
| The browser | The WebGPU backend of the layer running the same scene core and live views ([Web geometry](web-geometry.md)). The TypeScript WebGPU views are removed, and the TypeScript Canvas 2D views stay as the fallback |
| `HostRenderer` | The window, the frame loop, the embed canvas, window capture and ImGui setup, owning one `Backend` ([The renderer seam at its final shape](#the-renderer-seam-at-its-final-shape)) |

## One shader source

Each shader is written once and translated to MSL, SPIR-V and WGSL. The choice is between Slang, WGSL translated by Tint or by Naga, and keeping hand-written copies.

### The options

| | Slang | WGSL with Naga | WGSL with Tint | Hand-written copies |
|---|---|---|---|---|
| Source language | Slang, close to HLSL, with modules and generics | WGSL, the browser's own language | WGSL | MSL, GLSL, WGSL and CUDA |
| MSL | Yes, marked experimental in its own target table | Yes, a primary back end | Yes | By hand |
| SPIR-V | Yes, supported | Yes, a primary back end | Yes | glslang at build time, as today |
| WGSL | Yes, marked experimental, "still work in-progress" | The source itself | The source itself | By hand |
| CUDA kernels | Emits CUDA C++ and PTX through NVRTC, with its own kernel parameter layout | No | No | By hand, as today |
| Tool weight | A prebuilt release, about 60 MB compressed for macOS arm64 (175 MB of libraries unpacked) | `cargo install naga-cli` (a Rust toolchain); no prebuilt binaries | Built from Dawn's source tree; Dawn's nightly archives may include it (unverified) | None beyond today's glslang |
| Licence | Apache-2.0 with the LLVM exception | MIT or Apache-2.0 | BSD-3-Clause | Not applicable |
| Translation run | At developer time, generated sources checked in | The same | The same | Not applicable |
| Reflection for C++ structs and bindings | JSON reflection from `slangc` | From the WGSL declarations, by the generator | Tint's inspector (unverified from its command line) | By hand, four copies of one struct |
| Debugging | Generated MSL and WGSL are readable but verbose; matrices are rebuilt from vector arrays | The browser runs the authored source; generated MSL keeps the WGSL names | As for Naga | Native source on every API |
| What WebGPU lacks | Point size fails to compile for WGSL and compiles for MSL and SPIR-V, so targets can diverge unless all are built; push constants become `var<immediate>`, a WGSL extension in the specification's draft; a read-write texture becomes a read-write storage texture, invalid for RGBA8 in core WebGPU | The language has no point size, no push constants and WebGPU's access modes, so a shader that compiles is portable by construction | The same | Each copy works around them by hand |

### A trial translation

Slang's release for macOS arm64 (version 2026.22, from its own release page) translated three small shaders in a scratch folder. A points shader with push constants and `SV_PointSize` compiled to MSL (87 lines, `[[point_size]]`) and SPIR-V and was refused for WGSL with "system value semantic 'sv_pointsize' is not supported". Without the point size its push constants became `requires immediate_address_space; var<immediate>`. An instanced-quad shader shaped like the layer compiled to all three targets (54 lines of WGSL, 103 of MSL). The integer colormap below compiled to all four targets only after its output texture was changed from `RWTexture2D`, which became a read-write `rgba8unorm` storage texture, to a write-only `WTexture2D` with `Store`; its CUDA output reads its parameters from a global block, unlike today's launch signature. Each of the three needed WebGPU knowledge that the language does not enforce. Naga and Tint were not run: Naga has no prebuilt release, and Tint is built from Dawn.

### Exact colormap output

Natively the colormap matches the CPU reference byte for byte because each hand copy repeats the reference's float operations, CUDA with fused multiply-add and fast math switched off. WGSL allows a division 2.5 ULP from exact and lets an implementation assume that no NaN or infinity occurs, and the spike saw 2 to 18 texels one level off at the two largest sizes on one adapter. A translator cannot promise the reference's float arithmetic either. The technique therefore decides the index with integers only, against keys the CPU derives from the reference itself:

1. The reference is monotone in the input, because every IEEE operation in it is. Mapping a float's bits to an unsigned key that orders floats numerically (`bits | 0x80000000` for a positive float, `~bits` for a negative one) makes it monotone in the key.
2. For each mapping (vmin, vmax), the CPU finds by binary search over keys, for every level k from 1 to 255, the first key whose reference index is at least k, and a sentinel `0xffffffff` when no float reaches k (a range with an infinite end; the spike's derivation assumed that +∞ always reaches the top level). A decreasing range flips every key, so the shader only ever tests `key >= keys[k]`. The table costs about 8,000 reference evaluations, under a tenth of a millisecond, and is cached per mapping.
3. The shader tests NaN on the bits, guesses the level with the usual float formula, and accepts the guess only if two integer comparisons confirm it, trying the two neighbouring levels next and a binary search of eight integer steps last. The float guess only chooses which comparisons run; the decision is always made by integer comparisons, which every target performs exactly. Keys replace the f32 thresholds of the spike and of the draft in [Web geometry](web-geometry.md), because a float comparison against an infinite threshold is among what WGSL lets an implementation assume away.
4. The colour is the 32-bit table entry. The only float operations left are the conversion of a byte to [0, 1] and back at the store, whose error is several orders of magnitude below the half level that would change a byte.

```wgsl
// keys: 258 entries; keys[0] = 0 (always passed), keys[256] = keys[257] = 0xffffffff (never passed).
struct Cmap { vmin: f32, vmax: f32, flip: u32, pad: u32, keys: array<vec4<u32>, 65>, lut: array<vec4<u32>, 64> };
@group(1) @binding(1) var<uniform> m: Cmap;

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
```

The same function serves the compute colormap and the vertex-stage colormap of points, lines and meshes. The CUDA kernel implements the same steps from a C++ header shared with the CPU, marked `__host__ __device__`, so the algorithm has two sources, WGSL and C++, instead of eleven copies.

The verification suite proves it per backend. A headless test runs the C++ implementation against `map_f32_to_rgba8` for all 2<sup>32</sup> inputs on a set of ranges (under a slow label) and on boundary sets in the fast suite. New rows of the graphics suite run, on every backend and every path (compute from a layer buffer, from an imported buffer, the CUDA direct kernel, and the vertex-stage colormap of a points draw), for ranges that increase, decrease, are degenerate, have an infinite end, span subnormals or reach the largest floats: each threshold key and its two neighbours, ±0, ±∞, quiet, signalling and negative NaNs, and 65,536 random bit patterns, each compared byte for byte with the reference. The browser harness of the sprint 2 branch runs the same inputs on SwiftShader and on a hardware adapter. If the native fast path were measurably slower with the integer decision, the native backends could keep the float formula whose exactness the rows prove today, behind the same rows; the measurement below decides.

### Recommendation

Write every shader once in WGSL, translate it to MSL and SPIR-V with Naga (pinned) at developer time, and check the generated sources in, with a CI job that regenerates them and fails on a difference. The browser and Dawn take the WGSL directly. WGSL enforces the layer's model, so a shader that compiles is portable by construction; Naga's WGSL front end and its MSL and SPIR-V back ends are its primary paths; and the browser, the target least under Caliper's control, runs the shader exactly as written. A generator (`tools/shaders/`) concatenates shared snippets, emits each program's bind group description and a C++ struct with `static_assert`ed offsets for every uniform block, and replaces the four hand-synchronised copies of the geometry parameters. Builds need no shader tool, and the Windows build drops its build-time glslang. Two points are unverified and are checked in the step that introduces the generator: whether Naga's command line takes an explicit Metal binding map (otherwise a small driver on the `naga` crate passes one), and how its Metal output sizes runtime-sized arrays when bounds checks are off. The CUDA kernels stay hand-written. Slang is the alternative if the programs grow to need generics and modules.

The user has decided this, with one addition. A shader may have a hand-written native version on one platform when that platform needs it, for example an MSL version that keeps a Metal-only fast path. The shader's entry in the generator's list then names that platform as hand-written, the generator skips that output and leaves the hand-written file alone, and the CI regeneration job checks only the generated outputs. The hand-written version takes the same bind group description and uniform block structs that the generator emits, and the byte-exact rows of the graphics suite cover it exactly as they cover a generated one, so it cannot drift from the WGSL source unnoticed. Because the generated MSL and SPIR-V are committed, building and shipping Caliper needs no Rust and no shader tool; only a developer who changes a shader, and the CI job, run Naga.

## The scene core

The scene core is the one camera and scene model that every view uses: the points live view, the scene view, applets' scenes through the scene service, and the browser. Its content, the entities, primitives and arrays, is fixed by the contracts and elaborated in [Scenes](scenes.md); this section defines the core they compile through.

### Conventions

| Convention | Value, for every backend |
|---|---|
| World | Right-handed. Default up +y in 3-D, which the points view and the geometry service already assume; a scene's up axis rotates the orbit camera, never the data, and every 3-D view can switch between Y up and Z up ([Scenes](scenes.md#the-up-axis-switch)) |
| View space | Right-handed, the camera looks along −z with +y up and +x right (the points view's look-at) |
| Clip space | x right, y up, depth z in [0, 1] with near at 0, as WebGPU, Metal and the geometry service define it. The Vulkan backend flips y with a negative viewport height, as it does today; no view or shader flips anything |
| Depth | D32 float, cleared to 1.0, compare less-or-equal; not reversed |
| Matrices | Column-major, `clip = P · V · M · p`; composed in double on the CPU and handed to the GPU relative to the camera, so large coordinates keep precision |
| Framebuffer and textures | Origin at the top-left, y down; a pixel's centre is at (i + 0.5, j + 0.5) |
| Pinhole | The camera frame is x right, y down, z forward unless the scene says otherwise; pixel coordinates as in the row above |

### Entity tree, cameras and bounds

The tree is stored as arrays indexed by entity, with each parent before its children, so one forward pass recomputes the world matrices of the dirty subtrees: `world[i] = world[parent[i]] · local[i]`. Three camera models cover the views:

- **Orbit**: target, distance, azimuth, elevation and up axis, which the view can switch while keeping the target, the distance and the eye position. The eye is at target + distance · (cos(elevation) · (cos(azimuth) · a + sin(azimuth) · b) + sin(elevation) · u), where u is the up axis, a is the first of +x, +y and +z perpendicular to u, and b = a × u, which is the points view's convention with +y up and the one [the starting camera](scenes.md#the-starting-camera) of a scene uses. It has a 45° field of view, and near and far planes from the bounds, as the points view computes them today (`zn = max(d − 2r, d / 1000)`, `zf = d + 2r`).
- **Pan and zoom**: a centre and a half height, orthographic, with y up for plots or y down for pixel spaces.
- **Through a pinhole**: the camera entity's world pose and a projection built from the intrinsics. In the camera frame (x right, y down, z forward), `clip.x = (2 fx / W) X + (2 cx / W − 1) Z`, `clip.y = −(2 fy / H) Y + (1 − 2 cy / H) Z`, `clip.z = f (Z − n) / (f − n)` and `clip.w = Z`, letterboxed to the view's aspect.

Bounds come from the per-axis bounds that the producer computes on its device and sends with each batch, as the points view uses its stats today; failing that, from a sampled scan of a CPU-visible array. A batch's bounds are transformed to world space by their eight corners. Fitting places the orbit target at the bounds' centre at a distance that frames the bounding sphere, and sets the pan-and-zoom half height from the larger of the two extents.

### Picking

On a pointer event the scene core compiles the pick variants of the visible draws into a pick pass: a 16 × 16 `RG32Uint` target around the pointer, reached by zooming clip space, with its own depth buffer, as [Web geometry](web-geometry.md) designs it. Every pickable draw writes its `pick_id` and the item, the instance index or a vertex of the triangle for a mesh. `read_texture` returns the window asynchronously, a frame or two later, and a result that arrives after the draw list changed is dropped. The pass costs nothing while the pointer is still; the layer also accepts an ID attachment written beside the colour (`id_target`), which [Scenes](scenes.md) describes, for a view that picks every frame. The pick gives the entity, the primitive and the item exactly, for every primitive, with no CPU copy of the geometry, and the depth at the pixel gives the world position. Lasso and rectangle selections project CPU-visible arrays on the CPU, as the points view does today.

### Compiling primitives into a draw list

The core walks the visible entities, picks each batch's program and variant from its primitive and its optional arrays, binds the arrays as instance-step vertex buffers at their offsets in imported or copied buffers, writes the per-draw uniform block (model-view-projection relative to the camera, normal matrix, viewport, sizes, flat colour), binds the cached mapping block of its colormap or class table, and appends draws: opaque first, then translucent in entity order. Labels are ImGui text at anchors projected on the CPU. Indices from a producer are clamped by a small compute pass into a layer-owned index buffer once per batch update, because a hardware index fetch cannot clamp, and the geometry service's index-clamp contract promises a defined picture. The same pass has a second entry, the edge gather, which clamps the index pairs of a lines batch and writes each edge's two ends into the stream that the segment draw reads ([Scenes](scenes.md#index-pair-edges)). The primitive-to-draw table is in [Scenes](scenes.md).

### Interface outline

```cpp
namespace caliper_scene {   // Primitive, Batch, Pinhole and Axis follow the scene model of the contracts
struct EntityId { uint32_t index, generation; };
struct Aabb { double lo[3], hi[3]; bool valid; };
struct BatchArray {                  // one array of a batch
    caliper_gfx::Buffer buffer; uint64_t offset; uint32_t stride; caliper_gfx::VertexFormat format;
    uint64_t count; const void* cpu; // cpu: a CPU view for hover, lasso and bounds, or nullptr
};

class SceneTree {
public:
    EntityId ensure(std::string_view path);                  // creates missing parents
    void set_transform(EntityId, const double m[16]);
    void set_camera(EntityId, const Pinhole&);
    void set_batch(EntityId, Primitive, const Batch&);       // replaces that primitive's batch
    void clear(EntityId, bool subtree);
    void update();                                           // world matrices and bounds of dirty subtrees
    Aabb world_bounds(EntityId subtree) const;
};

struct Orbit   { double target[3], distance, azimuth, elevation; Axis up; };
struct PanZoom { double center[2], half_height; bool y_down; };
struct ThroughPinhole { EntityId camera; double zoom, pan[2]; };
using ViewCamera = std::variant<Orbit, PanZoom, ThroughPinhole>;

void fit(ViewCamera&, const Aabb& world, float aspect);
struct CompileOptions { uint32_t width, height; float pixel_scale; bool id_target; uint32_t clear_rgba; };
struct DrawMap;                                              // pick_id to entity and primitive
void compile(const SceneTree&, const ViewCamera&, const CompileOptions&, caliper_gfx::DrawList& out, DrawMap* map);

class Picker {   // over the layer's asynchronous readback; Hit holds entity, primitive, item, depth, world position
public: void request(float x, float y, int radius_px); std::optional<Hit> poll(const DrawMap&); };
}
```

### The points view and ImPlot3D

The points live view becomes the one-entity case: its payload is a batch, its `Camera` struct becomes `Orbit` or `PanZoom`, its fit becomes `fit`, and its CPU hover stays as the fast hover where the payload is CPU-visible, with the pick pass as the general pick. Its pixels must not change: a gfx row compares the new path's output with the reference that step 7 records from the old path before replacing it ([The rule of every step](#the-rule-of-every-step)). ImPlot3D stays as it is. It draws through ImGui's draw lists with its own camera, it serves applets' small plots well, and it costs nothing on the GPU path; an applet whose 3-D picture is a scene rather than a small chart uses the scene service instead.

## Zero-copy as a backend capability

### Import

Import is how zero-copy is expressed from now on: the layer wraps an external GPU buffer or texture as a layer buffer or texture with no copy, and where a backend cannot, it copies and says so.

```cpp
enum class ExternalKind : uint32_t { MetalBuffer, HostMemory, Win32Memory, MetalTexture };

struct ExternalSync {
    enum Kind : uint32_t { Drained, MetalEvent, CudaStream } kind = Drained;
    void* object = nullptr;   // id<MTLSharedEvent>, or a CUstream
    uint64_t value = 0;       // the event value the producer signals
};

struct ExternalBuffer {
    ExternalKind kind;
    void* handle;             // id<MTLBuffer>, a page-aligned host pointer, or a Win32 HANDLE (duplicated)
    uint64_t size;
    uint32_t usage;           // kVertex | kIndex | kStorage | kCopySrc
    bool allow_copy = true;   // false: refuse rather than copy
    const char* label = nullptr;
};
struct ExternalTexture { ExternalKind kind; void* handle; uint32_t width, height; Format format; uint32_t usage; };

struct ImportResult {
    Buffer buffer; Texture texture; Route route;   // the one that was imported
    uint64_t bytes_copied;    // 0 in place
    const char* path;         // the label the view's info line shows
    std::string reason;       // why it copied or refused
};
```

### Lifetime and synchronisation

- **Ownership.** An import holds the host's reference: Metal retains the `MTLBuffer`, Vulkan duplicates the handle and owns the imported memory. The producer keeps the memory alive until the import is released, as the `ExportablePool` rule already requires.
- **Release.** `release` is deferred until the last submit that read the buffer completed. `last_use` and `completed` replace the three handle families of today (`device_update_retain_last`, `geom_draw_retain_last` and their release), so a live view re-claims a shared-memory slot only once the frame that last read it retired, without blocking.
- **Stability.** The bytes must not be rewritten until that submit completed: the memory-stability contract of imported allocations, of the geometry service and of the scene service that replaces it, unchanged.
- **Ordering.** `Drained` means the producer finished writing before the handoff, the geometry service's drain and triple buffer and the session's published slot; the layer makes the writes visible and nothing more. `MetalEvent` and `CudaStream` order the reads after the producer's queue on the GPU, the stream-ordered rung of the sync ladder (`docs/explanations/zero-copy.md`). No CPU wait is ever on the hot path.
- **Direction.** Data flows one way. The layer never writes imported memory; the one inverse case, a CUDA kernel writing a layer texture on the direct path, is `kCapCudaWritesTexture`.

### Routes per backend

| Source | Metal | Vulkan with CUDA | WebGPU in a browser | Dawn (native fallback) |
|---|---|---|---|---|
| MPS tensor (`MTLBuffer`) | In place | Not applicable | Not applicable | Copied: Dawn has no `MTLBuffer` import |
| Shared-memory slot of the live session | In place, wrapped with `newBufferWithBytesNoCopy` as today | In place once the Windows session puts slots in exportable CUDA memory ([Windows parity](windows-parity.md)) | Not applicable | Copied (Dawn's host-pointer import is unverified for this) |
| CUDA tensor in the exportable pool | Not applicable | In place | Not applicable | Copied |
| CUDA tensor from torch's caching allocator | Not applicable | Textures: the direct kernel, no copy. Buffers: one copy inside the GPU into an exportable buffer | Not applicable | Copied |
| Host bytes: a CPU tensor or a network snapshot | One upload | One upload | One upload | One upload |

The route and its label come back from every import and update, and the view's info line shows them, so a browser or a Dawn fallback never claims zero-copy. A WebGPU or Dawn buffer is capped at 256 MiB and a storage binding at 128 MiB by default; an import above the backend's limits is refused with the reason.

### Keeping the fast path as fast

The zero-copy paths must stay exactly as fast through the migration. Each migration step is measured before and after, on the same machine, with these existing benchmarks and counters:

- `python/bench/bench_live.py table --size 256|1024|2048 --repeat 3` on MPS: steps per second against the commit-matched control, the cost of `caliper.show` (median and p95 microseconds), step-to-screen latency (median and p95), the viewer's frames per second and its path label, such as `W:compute-imported`.
- `python/bench/bench_show.py --mode window|attach --device mps`: the same in the in-process window.
- The graphics suite's timing rows, `gfx/Vulkan+CUDA: timing — update cost by size` (call and completion microseconds, frame mean and maximum, by path and size from 256 to 4,096) and `create+release cycle vs update`.
- `CALIPER_VULKAN_STATS=1`: frame-time p50, p99 and maximum, direct and buffer update counts, kernel waits and retire blocks.
- `caliper-viewer --stats --latency-log`, and the WebGPU backend's counters in the browser harness (GPU milliseconds, dispatches, live textures and allocations).

The first step adds what is missing: the frame-time statistics on Metal that Vulkan has, copy counters in today's paths so that "copies per frame" can be compared before any code moves, a points row in `bench_live.py` (100,000, 1,000,000 and 4,000,000 points), a gfx timing row for geometry draws by count, and GPU time from `MTLCommandBuffer` start and end times and Vulkan timestamp queries.

| Metric | Acceptance after a step |
|---|---|
| Copies per frame and bytes copied on a zero-copy route | Exactly 0, as before |
| Path labels | Identical |
| Byte-exact graphics rows | All green, Lambert rows within ±2 LSB as today |
| Steps per second, as a share of the control | Within 2 points (the two repeats of the 1,024 sampler row differ by 1.1 points today) |
| `caliper.show` cost, median | Within 5 % or 2 µs |
| Step-to-screen latency | Median within 1 ms, p95 within 2 ms |
| Frame time | p50 within 3 %, p99 within 5 % |
| GPU time of an update and of a view's draw | Within 5 % or 20 µs |
| Update call time in the timing rows | Within 5 % or 2 µs |
| The layer's CPU encode time per frame | At most 0.05 ms more |

The before build is the step's first commit, which records the step's references and changes no drawing code ([The rule of every step](#the-rule-of-every-step)), and the after build is its last commit. Both are built in Release and run on the reference Mac, on mains power, with the same torch, interleaved three times each; the benchmarks already record the load average. The after numbers must also meet the thresholds against step 0's baseline, so that losses within the thresholds cannot add up across steps, and a zero-copy route keeps exactly 0 copies throughout, because the user has kept zero-copy as a hard requirement. Results are committed under `python/bench/results/` with the step's name. A step that misses a threshold is fixed, or reverted with `git revert` of its series, before the next one starts. The Vulkan step at Windows parity compares with a build of the last commit before step 8, which still has the Vulkan renderer's paths, run interleaved with its own build on the same Windows machine.

## The alternatives

The deciding criterion is whether zero-copy import survives on Metal and on Vulkan with CUDA.

| Library | Licence | Weight | Backends | Wraps an `MTLBuffer` | Imports Vulkan external memory as a buffer | ImGui | Browser |
|---|---|---|---|---|---|---|---|
| Dawn | BSD-3-Clause | Large C++ build (abseil, SPIRV-Tools, Tint and others); unsigned nightly binaries | Metal, Vulkan, D3D12, D3D11, OpenGL ES, SwiftShader | No. `SharedBufferMemory` takes a D3D12 resource, a Windows handle or a host pointer only; `MetalBackend.h` offers no wrap | No. `WrapVulkanImage` wraps images, not buffers; `SharedFence` does take an `MTLSharedEvent` and Vulkan semaphores | `imgui_impl_wgpu` | The same API in the browser through emdawnwebgpu |
| wgpu-native | MIT or Apache-2.0 | Prebuilt libraries (v29.0.1.1) | Metal, Vulkan, D3D12, OpenGL | No in the C API, which has only getters such as `wgpuDeviceGetNativeMetalDevice`. The Rust `wgpu-hal` has `metal::Device::buffer_from_raw`, out of reach from C++ | No. `wgpu-hal`'s Vulkan side wraps textures (`texture_from_raw`), not buffers | `imgui_impl_wgpu` | Not through wgpu-native |
| bgfx | BSD-2-Clause | Source build with its own shader compiler, `shaderc` | Metal, Vulkan, D3D, OpenGL, WebGPU through Dawn | No. Native pointers only for textures, through the `_external` argument of `createTexture2D`; no buffer equivalent | No | Its examples' ImGui renderer | WebGL 2 |
| sokol_gfx | zlib | One header | Metal, D3D11, OpenGL, WebGPU; Vulkan "experimental" | Yes, `sg_buffer_desc.mtl_buffer` | No: no Vulkan injection fields | `sokol_imgui.h` | WebGL 2 and WebGPU |
| Diligent Engine | Apache-2.0, but its Metal backend is "available under commercial license" | Large; C++20 | D3D11, D3D12, Vulkan, Metal (commercial), OpenGL, WebGPU through Dawn | `IRenderDeviceMtl::CreateBufferFromMtlResource`, in the commercial backend only | Wraps a `VkBuffer` with `IRenderDeviceVk::CreateBufferFromVulkanResource` and a timeline semaphore with `CreateFenceFromVulkanResource` | `ImGuiImplDiligent` | WebGPU and WebGL |

No library keeps both zero-copy paths under a licence Caliper can ship: Dawn and wgpu-native lose both, bgfx loses both, sokol_gfx keeps Metal but has no stable Vulkan with injection, and Diligent keeps both only with a commercial Metal backend. The recommendation, which the user has accepted, is to build the thin layer with Metal, Vulkan and WebGPU backends, and to use Dawn as the WebGPU implementation where copies are acceptable and reported: in the browser through emdawnwebgpu, and natively as the fallback for machines without Metal or Vulkan, which the OpenGL path served until the migration retires it. This would change if Dawn gained `SharedBufferMemory` for an `MTLBuffer` and for Vulkan external memory, because its `SharedFence` already covers the synchronisation; then one WebGPU backend on Dawn could serve every role, and Caliper would maintain one backend instead of three. Dawn's interop is worth re-reading at each sprint.

## Migration

The user has decided the order: the minimum cut first, so that scenes can start, and then every remaining step in full. No step is optional. The user has also decided that the migration keeps no backwards compatibility, because nothing is deployed or commercialised yet: no step keeps an old path; `HostRenderer` shrinks to the window, the frame loop and ImGui setup; each graphics service keeps one version, under one bump of the ABI epoch; the GLEW OpenGL fallback is retired early; the hand-written shader copies and the platform code in the views go as each client moves; the TypeScript WebGPU views go while the Canvas 2D fallback stays; and zero-copy stays a hard requirement. [Decisions](#decisions) gives each of these with its reason.

### The rule of every step

Each step is an increment that can be seen and checked on its own:

- it lands as its own series of commits, each of which builds and passes its tests;
- it replaces what it moves and deletes the old code in the same series, together with the `HostRenderer` methods and the tests that only the old code used, so no old path stays behind as a fallback or a switch, and a step is reverted with `git revert` of its series;
- it has measurements before and after against the thresholds of [Keeping the fast path as fast](#keeping-the-fast-path-as-fast), and the byte-exact pixel checks of the graphics suite;
- it has offscreen captures of what it changed, taken without a window, for the user to look at;
- the next step does not start until the current one passes all of these.

With no old path left at run time, a row can no longer draw one payload through the old and the new path and compare them, so the comparison moves in time. The first commit of a step changes no drawing code. It runs the rows, captures and benchmarks of what the step will change on the code as it stands, and commits their outputs: the readback bytes become fixtures that the tests read, as hashes, with the images kept beside the step's numbers under `python/bench/results/<step>/`, and the numbers are the step's "before". The commits that replace the client must then reproduce every fixture byte for byte where the step promises unchanged pixels, and meet the thresholds against the "before" numbers and against step 0's baseline. The fixtures stay in the suite as references, so a later step that moves the same pixels is caught too. Where a CPU reference row already defines the pixels, as for the colormap and the geometry rows, that row is the check and no fixture is needed. The browser harness already writes screenshots, which step 4 keeps as its references; step 7 adds the same reference mode to the graphics suite, and the later steps reuse it.

### The steps

| Step | What changes and what is deleted | What proves it | Builder days |
|---|---|---|---|
| 0. Counters and baseline | Copy counters and Metal frame-time statistics in today's renderers, the points benchmark row, the geometry timing row; a recorded baseline. Nothing is deleted | The counters match the known routes (an MPS heatmap shows 0 copies) | 2 |
| 1. Layer and Metal backend | `src/host/gfx/`: the interface, validation, the uniform arena, frames, serials, readback; the Metal backend on the renderer's device and queue; a recording backend for headless tests. Nothing uses it yet, so nothing is deleted | Headless tests of validation and recorded draw lists; a gfx row draws a quad through the layer and reads it back byte-exact | 6 |
| 2. Shader generator | WGSL sources, the generator, pinned Naga, generated MSL and SPIR-V checked in, the CI check; the scene programs | The generated sources rebuild identically in CI; the programs compile on Metal and in a browser | 3 |
| 3. Scene core | The tree, world matrices, cameras, bounds and fitting, compile, the ID target and picks | Headless tests of matrices, fits and compiled draw lists; a gfx row of a one-entity scene | 6 |
| 4. WebGPU backend of the layer, browser | The sprint 2 backend's resources and draws move behind the layer, with WGSL from the one source. The WebGPU renderer's own resource and draw code and its WGSL strings are deleted in the same series; its texture and points entry points remain only as thin calls into the layer until steps 7 and 8 delete them | The browser harness: screenshots compared with those taken at the step's first commit, and the colormap rows on SwiftShader and a hardware adapter | 5 |
| Minimum cut | Steps 0 to 4 | | 22 |
| 5. Retire OpenGL | `gl_renderer.cpp`, GLEW (its submodule, its source generation in `init-submodules.sh`, its CMake wrapper, the prebuilt download on Windows and its CI cache key), ImGui's OpenGL backend and `find_package(OpenGL)`, the OpenGL environment of the graphics suite and its rows, `CALIPER_RENDERER=gl`, the fallback code in `main.cpp`, `core_lifecycle.cpp` and the embedding core, and the comparison of the renderer's name with `"gl"` go. Where neither Metal nor Vulkan starts, the app, `caliper shot` and the embedding core refuse with a message ([No fallback renderer until Dawn](#no-fallback-renderer-until-dawn)) | Both lanes build with no GLEW and no OpenGL; a test forces the renderer's start to fail and checks the message and the exit code; the whole suite; the Metal numbers unchanged | 2 |
| 6. Colormap by keys | The key derivation in the shared C++ header, the exhaustive CPU test, the integer decision in the browser's WGSL with its boundary rows, and the boundary rows on Metal through the generated program, from a layer buffer and from an imported buffer. No hand-written copy is edited, because each is deleted by the step that moves its client: the Metal compute colormap in step 8, the GLSL `colormap.comp` and the CUDA kernel with the Vulkan renderer's paths in step 8, and the TypeScript WGSL in step 9 | The exhaustive CPU test; the boundary rows in a browser and on Metal | 2 |
| 7. Points live view on the scene core | The points live view becomes a scene of one entity; `bind_buffer` becomes one `import_buffer` call. Deleted: `live_views_metal.mm`, the staging of CPU payloads into Metal buffers and its refusal on other renderers, the path label taken from the tensor's device, `geom_draw_retain_last`, and in the browser `import_host_allocation`. The step's first commit adds the graphics suite's reference mode | `test_live_views`; the points row against the reference recorded from the old path; the points benchmark within thresholds; the browser harness's points screenshots against those from before | 3 |
| 8. Texture path on Metal, and the Vulkan renderer reduced | The tensor bridge, the embed views and the heatmap and image live views go through `update_texture_from_tensor`, which runs the generated colormap by keys; the bridge's allocation table moves to `import_buffer`; the path label comes from the returned route. Deleted: the texture, per-source update, completion, interop and import methods of `HostRenderer` in every backend, `cmap_f32` and the blit code of the Metal renderer, the embed views' Metal branch and their check of the renderer's device kind, and the WebGPU renderer's texture calls of step 4. Until step 13, which deletes the geometry service, its old code on the Metal renderer reads the layer's imported buffers through an adapter, so there is one allocation table. On Windows the Vulkan renderer's feature paths go ([The Vulkan renderer until Windows parity](#the-vulkan-renderer-until-windows-parity)) | Every gfx/Metal texture row; the boundary rows through the bridge; `bench_live` and `bench_show` within thresholds, which also measure the native cost of the integer colormap; identical path labels; the bridge refusing on a missing backend, in both lanes; the Windows lane green | 4 |
| 9. TypeScript WebGPU views removed | `web/src/render/gpu.ts`, `heatmap-gpu.ts` and `points-gpu.ts` go, with the WebGPU branches of `views/heatmap.ts` and `views/points.ts` and the element's request for a device for its own views ([The TypeScript views in the browser](#the-typescript-views-in-the-browser)) | The browser tests in all three projects; the Canvas 2D captures of heatmap, image and points against those from before; the TypeScript colormap against the C++ cases | 1.5 |
| 10. Scene model additions | Additive blending for points and lines; instanced meshes with instance poses and instance colours or values, their normals through the cofactor of each pose; and a constant depth offset for lines, arrows and box edges toward the camera ([Scenes](scenes.md#primitives)). Each lands in the scene core and its programs, Caliper LIB, the three codecs of protocol 3.2 and the browser backend, so that one scene model stays | Headless tests of the compiled draws; capture rows natively and in headless Chromium under the parity rules; an instanced mesh with a sheared pose lit like its CPU reference within 2 levels; a wireframe over a mesh drawn whole | 3 |
| 11. The scene service, and the particle applets on it | `caliper.scene.v1` ([The scene service for applets](scenes.md#the-scene-service-for-applets)): the table and its records; the host's implementation over the scene core, with arrays imported through the tensor bridge's allocation table; the refusals; the tokens; the view item with its remembered state; the events; the sugar and the torch adapter; its reference page and tests. It is a new service id beside the geometry service, so no epoch changes yet. `field_scope`, `flow_scope` and `sculpt_scope` move onto it, and their `draw_points` calls, their own cameras and their ImPlot3D fallbacks are deleted ([The applets on the scene service](#the-applets-on-the-scene-service)) | The service's headless tests on the recording backend, in both lanes; its capture rows on Metal; each applet's captures before and after and its tests, with 0 copies per frame on its zero-copy route | 12.5 |
| 12. Three more applets on the scene service | `embed_scope`, `gpt_scope` and `instance_scope` move onto the scene service; their ImPlot3D plots, `draw_points` and instanced geometry calls, CPU picking, cameras and fallback ladders are deleted | Each applet's captures before and after and its tests | 4 |
| 13. The last applets, one version of each graphics service, the geometry service deleted, and the one epoch bump | `mesh_scope` and `twin_scope` move onto the scene service, and the export service draws scenes. `caliper.geometry.*` then has no client and is deleted: its headers, its reference page, the host's implementation, its MSL shaders and the four copies of its parameter block, the geometry methods and capability booleans of `HostRenderer`, `HostGeomDraw`, the adapter of step 8, its rows of the graphics suite, its caps bits and its sugar, and with it the graphics layer's `kCapVertexStorage` and bind group 3, which nothing else uses. The tensor bridge and the export service each become one table, the embed v1.2 tensor views are folded into the typed live views, the slot record of `include/caliper/live.h` loses its Metal buffer, the ABI epoch goes from 2 to 3 with the simplified check, the SDK to 0.3.0 and `caliper-embed` to 2.0.0, and every manifest in this repository is updated ([One version of each graphics service](#one-version-of-each-graphics-service)). Deleted as well: the revision headers and tables, the sugar's widening of records, the Python package's switch between two view routes, and `native_device()` | The two applets' captures before and after and their tests; the export rows on scenes; the ABI tests pin the new layouts; every applet in this repository builds against the installed SDK in both lanes, loads, and its page matches its capture from the end of its port, or from before this step for an applet without 3-D; an applet built for epoch 2 is refused before its library opens; the embedding, live and Python tests; no mention of `caliper.geometry` left in the tree | 11 |
| 14. `HostRenderer` at its final shape | What is left of the special-purpose seam goes ([The renderer seam at its final shape](#the-renderer-seam-at-its-final-shape)): `canvas_supported`, `window_capture_supported`, `debug_readback_rgba8`, and step 0's copy counters, which the layer's `Stats` now carries. Rendering (`docs/explanations/rendering.md`), the zero-copy pages, Compatibility and epochs (`docs/explanations/compatibility.md`) and the [roadmap](roadmap.md#renderers) are updated | The whole suite in both lanes; the docs build | 1.5 |
| 15. Dawn fallback | Dawn as an optional component built from pinned source; the WebGPU backend compiled against it; the host takes Metal or Vulkan, then Dawn, and `CALIPER_RENDERER=dawn` forces Dawn. A machine with neither Metal nor Vulkan now starts on Dawn instead of refusing. The Windows lane builds Dawn and runs the layer's rows and the app's tests on SwiftShader | The staged rows on Dawn over Metal and over SwiftShader, in both lanes; the host's screens draw; the info lines say that Dawn copies | 5 |
| 16. An applet outside this repository, ported | By the user's decision, one applet that lives outside this repository, with 3-D curves, a mesh, measurement marks and a model view, is ported onto the scene service and the single-version services, built against the SDK installed from the build branch, with its own tests run and offscreen captures taken before and after. The applet adapts to the API: where it draws something the API does not offer, the applet changes, through another primitive, a computation of its own or a simpler picture. Caliper gains a feature from the port only where it is a proper general feature in its own right, recorded with its reason as an item of its own. Other applets outside this repository remain the user's to decide. The step needs steps 13 and 14, not Dawn, so it may run beside step 15 | The applet builds against the new SDK and passes its tests, and its captures after the port have been compared with those before, with every difference listed and explained | 3 |
| In this sprint | Steps 0 to 16 | | 74.5 |
| 17. Vulkan backend, with [Windows parity](windows-parity.md) | The Vulkan backend of the layer, with the CUDA direct path, the buffer path and the timeline semaphores as private paths, written from the code that step 8 deleted, which the repository's history keeps; the tensor bridge, the scene service and export on it; the Windows refusals of step 8 go. The CUDA kernel by keys is a row of its own in Windows parity | Every gfx/Vulkan and Vulkan+CUDA row, written against the layer, on Windows hardware; the timing rows and `CALIPER_VULKAN_STATS` within thresholds against the last commit before step 8 | 8 |
| Total | Steps 0 to 17 | | 82.5 |

The total is 82.5 builder days, and 74.5 of them fall in this sprint. Against the plan before the user's decision on backwards compatibility, 54 and 46, deleting the dual paths saved 2.5 days, deleting old code inside steps 8 and 9 of that plan added 1.5, and one version of each graphics service and the removal of the TypeScript WebGPU views added 9.5. Against the plan before the user's decision on the scene service, 62.5 and 54.5, the scene service adds 17 days: the geometry service is no longer moved onto the layer (5.5) or given one version (2), and three applets no longer rename its records (1), while the scene model's additions (3), the service (10), the rewrites of eight applets (10), export drawing scenes (1 more than collapsing it) and the deletion of the geometry service (1.5) are new. The port of the applet outside this repository adds 3. The net change is 20 days more.

The minimum cut is what scenes and web geometry need before they start: the layer with its Metal backend, the generator with the scene programs, the scene core, and the browser's WebGPU backend, 22 builder days. Steps 1 and 3 can run in parallel once the interface header is agreed, and step 4 in parallel with step 3, so the cut takes about two weeks with two builders. Steps 4, 7 and 8 depend on sprint 2's WebGPU backend and live views being merged. Nothing depends on sprint 3.

### The order of the steps

1. **Steps 0 to 4 first,** as the user decided, because scenes and web geometry start on them.
2. **Step 5, OpenGL, next.** It depends on nothing, and retiring it before any client moves means that no later step has to keep OpenGL's CPU-staged route, its test environment or its name check working. This is the early retirement that the user asked for. A builder who is free while the cut or the loss landscape is under way may land it then.
3. **Step 6 before the clients.** The native rows of the generated colormap prove it once, before points, textures and scenes depend on it; its browser half comes earlier still, in the loss landscape, whose colours need it.
4. **Step 7, points, before step 8, textures.** The points view is macOS and browser code only, so it moves without touching Windows; it is the first live view on the scene core, which proves the scene core early; and it shares no allocation table with anything.
5. **Step 8 moves the bridge's allocation table to the layer.** The geometry service draws from the same allocations, so until step 13 its old code on the Metal renderer reads the layer's imported buffers through an adapter; it is never moved onto the layer, because it is deleted once its applets have moved. On Windows both the bridge and the geometry service lose their renderer at this step, which is why the Vulkan renderer is reduced once, here.
6. **Step 9, the TypeScript WebGPU views,** could follow step 4 at once. It is placed with the browser's scene work of [Sprint 4](sprint-4.md), so that the element's own views change together with the scene fallback in TypeScript.
7. **Step 10 before the service.** The applets need additive points and instanced meshes, and adding them to the scene model first, for Caliper LIB and the browser too, keeps one model instead of an applet dialect.
8. **Step 11 after the native scene view and its picking.** The service is the applet face of the scene view, which the app draws from increment 2 of [Sprint 4](sprint-4.md) and which picks from increment 5. The three particle applets move first, because each draws one points batch and they prove the in-place route.
9. **Step 12, three applets that use labels, classes, picking and instancing,** while the geometry service still serves the last two.
10. **Step 13, where the service ids change.** The last two applets export, so the export service moves with them; the geometry service is deleted in the step that moves its last client; and the tensor bridge and the export service lose their revisions there too, so the epoch changes once, where the service ids change. Adding the scene service's id in step 11 needed no epoch change, because a new id breaks nothing.
11. **Step 14 after step 13,** because what is left of the seam goes only once no client calls it.
12. **Step 15, Dawn, late in the sprint,** because it needs the WebGPU backend to draw every scene primitive, and with OpenGL already gone it adds a fallback rather than replacing one. It stays in the plan, later than the retirement of OpenGL, as the user decided.
13. **Step 16 last.** It needs the final services and the final SDK, and it is done when an applet built elsewhere runs on them.
14. **Step 17 with Windows parity,** as the user decided.

### One version of each graphics service

The user decided that each graphics service keeps one version, that the ABI epoch is bumped once, and that every applet in this repository is updated in the same changes. Step 13 does it, after steps 11 and 12 have moved most applets onto the scene service. The decision covers graphics only: protocol compatibility (producers of protocol 3.0 and the `/v0/stream` alias) and the revisions of the UI service, like those of the other services that do not draw, stay as they are.

| Service | Today | The one version |
|---|---|---|
| Tensor bridge | `caliper.tensor_bridge.v1`, `v1_1` and `v1_2`, three tables, each a prefix of the next | `caliper.tensor_bridge.v1` with today's `v1_2` table: the six texture operations, `caps`, `import_allocation`, `release_allocation` and `update_texture_from_alloc`. Both caps bits stay, because a backend can truly lack each: stream-ordered handoff, which Dawn and the browser cannot honour, and import, which Dawn, the browser and Vulkan without a paired CUDA device cannot do |
| Geometry | `caliper.geometry.v1` to `v1_3`, four tables, and draw records of 192, 216 and 256 bytes | None. It is deleted in step 13, when its last applets have moved, and applets draw in 3-D through the scene service, as the user decided |
| Scene | None | `caliper.scene.v1`, one table from its first version, added in step 11 ([The scene service for applets](scenes.md#the-scene-service-for-applets)) |
| Export | `caliper.export.v1` and `v1_1` | `caliper.export.v1` with today's `v1_1` table, which adds `last_error`, drawing a scene instead of geometry records ([Export](scenes.md#export)) |
| Embedding views | The v1.2 tensor views (`caliper_core_view_tensor` and the four calls beside it) and the v1.3 typed live views, whose heatmap and image draw through a v1.2 view of the same name | The typed live views alone. The heatmap and image views own a layer texture; the five v1.2 calls go, and `caliper_core_colormap_lut` stays. The slot record of `include/caliper/live.h` hands the slot's host address instead of a Metal buffer, and the views import it as host memory, which the Metal backend wraps in place as today. `CALIPER_EMBED_API_VERSION` becomes 2 with minor version 0, and the package `caliper-embed` becomes 2.0.0 |

**The ids.** The one version of each service keeps the plain `v1` id. The epoch bump already refuses every applet built for the old tables before it can ask for an id, so a new major id would protect nothing, and the applets that list only `v1` ids keep their service lists.

**Growth after this sprint.** Within epoch 3 a graphics service grows by appending to its one table, which `struct_size` guards, and its records grow by appending, which each record's own size guards; no new id and no caps bit mark the growth. An incompatible change is an epoch change. The other services keep the growth rules of Compatibility and epochs (`docs/explanations/compatibility.md`), which step 13 rewrites for the graphics services. Whether to return to revision ids after a first release is an [open decision](#open-decisions).

**The epoch bump.** Step 13 bumps the ABI epoch from 2 to 3 once, where the service ids change: the geometry ids and the revision ids of the tensor bridge and the export service disappear there. The commit that changes the tables also changes every manifest in the repository, so each commit of the series builds. Any other epoch change that the sprint might need, such as a change of the UI-stack pin, lands in the same commit or waits for the next epoch. The SDK becomes 0.3.0, as Compatibility and epochs allows before 1.0.

**The simplified epoch check.** The check stays, and it becomes simpler. One number, `CALIPER_ABI_EPOCH` in `abi.h`, is read everywhere: by the host's capabilities, by one prototype of `CaliperHost` that the app and the embedding core share (each writes the literal 2 today, in `src/host/ui/applet_page.cpp` and `src/host/embed/embed_core.cpp`), by the SDK's configure check, by the template of `caliper new` and by the test fixtures. The host compares it once before it opens a library, with a refusal that names both epochs and says to rebuild against the SDK of the host's epoch, and after opening the descriptor must agree with its manifest, as today. For the graphics services nothing else is negotiated: there are no revision ids, no caps bit per revision and no widening of records in the sugar.

**Outside this repository.** No other repository is touched by steps 0 to 15. Applets that live outside this repository build against the SDK, and each stops loading at step 13, refused before its library opens with the epoch message; one that uses the geometry service also stops building against the SDK 0.3.0 until it is rewritten onto the scene service. An embedding host outside this repository that calls the v1.2 views stops compiling against `caliper-embed` 2.0.0 until it moves to the typed live views. Step 16 ports one such applet, by the user's decision; whether and when to update the others is the user's decision. The `inspector` branch of this repository is in the same position when it next takes sprint 4.

### The applets on the scene service

Every applet in this repository that draws in 3-D moves onto the scene service, decided on the API's merits: each picture becomes what the scene model describes, and where an applet drew something no general scene model has, the applet changes rather than the model. No ImPlot3D plot in this repository stays, because each is a 3-D point cloud or a fallback for a missing geometry path, not a small chart with axes; ImPlot3D stays in the UI stack for applets whose 3-D plot is such a chart.

| Applet | What it draws today, and through what | What it becomes | Why | Step | Builder days |
|---|---|---|---|---|---|
| `field_scope` | A particle cloud through the geometry service's `draw_points`: additive, speed through magma, 1.5 pixels, with its own orbit camera; an ImPlot3D scatter as its CPU fallback | One additive points batch with values through magma at a pixel size, from its pool's slots, and a starting camera; the camera code and the fallback go | A particle cloud is a scene. The fallback's cases are covered: a CPU tensor is an array the service copies, and a host with no backend refuses with its reason | 11 | 1 |
| `flow_scope` | The same | The same | The same | 11 | 0.75 |
| `sculpt_scope` | The same, coloured by motion | The same | The same | 11 | 0.75 |
| `embed_scope` | ImPlot3D: ten class clouds, centroids as diamonds, the live batch as white sparks, fixed axes, and a hover found by projecting every point on the CPU each frame | Points with class ids and a class table, the centroids as larger points with labels, the live batch as white points, a starting camera in place of the fixed axes, and the hover from the pick pass | A 3-D embedding is a scene, and the per-frame CPU projection goes. Diamonds become discs, and the axes' titles go | 12 | 1.5 |
| `gpt_scope` | ThoughtSpace's residual stations through `draw_points` in three colour modes, with an ImPlot3D fallback; the token embedding as ImPlot3D text glyphs at 3-D positions | Two scenes: the stations as additive points through the mode's colormap, and the embedding as points with one label per token | Both are 3-D clouds; a hovered station or token comes back as an event, and labels are drawn by the view, nearest first, 512 at a time | 12 | 1.5 |
| `instance_scope` | Instanced gems through geometry v1.3: lit, opaque, rigid poses, a per-instance tint through magma; a one-gem ladder without instancing | One instanced mesh with instance poses and instance values through magma | Instanced meshes are now part of the scene model; the rigidity gate and the ladder go | 12 | 1 |
| `mesh_scope` | A lit surface coloured by error, a translucent white wireframe, the minibatch as additive amber points, the three draws exported; an ImPlot heatmap as its fallback | A mesh with normals and values, the wireframe as index-pair edges one pixel wide and translucent, the minibatch as additive points, export of the scene; the fallback goes | Each draw is a scene primitive; the wireframe wins over the surface by the edges' depth offset | 13 | 1.5 |
| `twin_scope` | Two textured meshes side by side, each sampling a bridge texture of its field, a toggle to per-vertex colours, wireframes, export | Two mesh entities placed by entity transforms, each textured from an RGBA image that the applet colormaps on its device or coloured by per-vertex values, edges as index-pair edges, export of the scene | A texture is an image in the scene model, so the field is colormapped by the applet, one gather in torch, instead of by the bridge | 13 | 2 |
| `pulse_scope`, `examples/hello`, `examples/sine_scope` | No 3-D | Unchanged, apart from the manifest's epoch | | 13 | |

The archived applets in `applets/legacy/` and `applets/legacy-dev/` draw no 3-D; only their manifests change.

**The acceptance of each port.** Each port takes `caliper shot` captures of the applet's page before and after, offscreen. The pictures are not byte-identical, because the drawing path changes on purpose: hardware points become discs, hardware lines become quads, and the camera's defaults move to the shared ones. The user compares them by eye with every difference listed and explained in the port's report. The applet's own tests pass. Its info line reports the route that ran, and copies per frame on its zero-copy route stay exactly 0 against step 0's baseline. Its frame time and GPU time are measured before and after and reported, but the thresholds of [Keeping the fast path as fast](#keeping-the-fast-path-as-fast) do not gate a port, because a port changes what is drawn; in particular, discs of six vertices cost more GPU time than hardware points for the largest particle clouds, and the report says by how much.

### The renderer seam at its final shape

`HostRenderer` keeps the window, the frame loop and ImGui setup, where the window includes the embed canvas, which is a host's view or an offscreen target, and window capture, which reads the frame that the swapchain presents. It owns one `Backend` of the layer and returns it from `gfx()`, or none where no backend exists. Each step deletes the methods it made unused, in every backend, and step 14 removes what is left.

| Group of the inventory | Methods | What happens | Step |
|---|---|---|---|
| Window and frame | 6 | Stays; `name()` is a label for the log and the export sidecar, never compared | |
| Embed canvas | 7 | Stays, with `canvas_read_pixels`, which `caliper shot` uses; `canvas_supported` goes, and `canvas_init` refuses with its reason instead | 14 |
| Window capture | 3 | Stays; `window_capture_supported` goes, and a request returns false with its reason instead | 14 |
| RGBA8 textures | 4 | Go, for `create_texture`, `write_texture`, `release` and `imgui_texture_id` | 8 |
| Texture updates, one per source | 4 | Go, for `update_texture_from_tensor` and `fill_texture_from_buffer`, which return the route and the path label | 8 |
| Completion, one family per kind of work | 6 | Go, for serials: `submit`, `completed`, `wait` and `last_use` | 7 and 8 |
| Device and interop | 3 | Go: `interop_device` for the caps bits, `native_device` for `import_buffer`, and `alloc_device_shared` for the Vulkan backend's private path behind `kCapCudaWritesTexture` at Windows parity | 8 and 13 |
| Import, one per source | 4 | Go, for `import_buffer`, `import_texture` and `release` | 7 and 8 |
| Geometry, one per ABI revision | 8 | Go with the geometry service, whose applets draw through the scene service | 13 |
| Tests only | 2 | Go: `debug_readback_rgba8` for `read_texture`, and `debug_force_buffer_path` with the Vulkan renderer's buffer path, to return as a test hook of the Vulkan backend | 8 and 14 |
| Measurement, from step 0 | 3 | The copy counters go to the layer's `Stats`; `frame_stats` stays for the frame loop's times and takes GPU times from `Stats` | 14 |

The seam ends with 16 methods: six for the window and the frame, six for the canvas, two for window capture, `gfx()` and `frame_stats`. The Metal, Vulkan and WebGPU renderers implement them, and Dawn's later.

### The Vulkan renderer until Windows parity

The Vulkan backend of the layer comes with Windows parity, as the user decided, and no step keeps an old path. Together these leave the Vulkan renderer with no client from step 8 on: the tensor bridge then draws through the layer, the geometry service and later the scene service draw from the bridge's allocations, and Windows has no backend of the layer until step 17. This plan therefore reduces the Vulkan renderer in step 8, and states what Windows loses until step 17. The user has accepted that gap: Windows and Vulkan are deferred until the user moves to the Windows machine.

- **What the Vulkan renderer keeps.** The window, the swapchain and the frame loop; ImGui, ImPlot and ImPlot3D through `imgui_impl_vulkan`; the embed canvas in a window and offscreen, with the offscreen readback that `caliper shot` uses; and its frame-time statistics. It implements the shrunk `HostRenderer` and nothing more, and its `gfx()` returns no backend.
- **How it keeps compiling.** The bridge, the services and the views are shared code written against the layer. On Windows they find no backend, and nothing in them names the Vulkan renderer, so they compile there unchanged. The Vulkan renderer's target loses the GLSL shaders, the build-time glslang and the generated SPIR-V and PTX headers. `cuda_driver.cpp` stays, because the device query and the SDK's exportable pool use it.
- **What Windows refuses meanwhile.** Every tensor bridge texture: the caps read 0, and creating a texture returns 0 with a log line saying that Windows has no graphics backend until the Vulkan backend of the graphics layer lands. Every geometry view, and from step 11 every applet scene, every PNG export and every embedding view is refused the same way: `caliper.scene.v1` reports no `CALIPER_SCENE_CAP_DRAW`. Applets load, draw their ImGui and ImPlot pages and take their CPU fallbacks, as the fallback rule asks of them, and `caliper shot` and the embed canvas work. The live views and the shared-memory session are macOS-only already.
- **What is unbuilt meanwhile.** The Vulkan renderer's texture, import, CUDA direct, buffer and geometry paths, its six GLSL shaders, the build-time glslang, the direct kernel and its PTX, and the gfx/Vulkan, gfx/Vulkan+CUDA and Vulkan geometry rows. The repository's history keeps them for step 17, except the geometry paths, which no step brings back.
- **What Windows gains.** The headless tests of the layer, the scene core and the services' gates run in the Windows lane on the recording backend, which no Windows test covered before. From step 15 a Windows build with Dawn starts on Dawn, because the reduced Vulkan renderer has no backend of the layer, and draws bridge textures, embedding views, scenes and applet scenes with a copy that the info lines report; a build without Dawn keeps the reduced Vulkan renderer. Dawn on Direct3D is verified with Windows parity.

### No fallback renderer until Dawn

From step 5 until step 15 Caliper has no fallback renderer, and a fallback that is absent says so:

- **The app** tries Metal on macOS or Vulkan on Windows. When that renderer cannot start, the app prints a message that names the renderer it tried and why it failed, and says that no fallback exists yet and that the WebGPU fallback through Dawn is planned, then exits with 1.
- **`caliper shot`** prints its existing `caliper: shot cannot draw offscreen: ...` with the same reason and exits with 1.
- **The embedding core** refuses to attach a canvas, with the same reason in `caliper_core_last_error`.
- **Who loses the app until Dawn.** Virtual machines and remote sessions without a working GPU, machines with a broken driver, and Linux, which is not supported and now has no renderer at all.
- **What CI loses.** The macOS lane is unchanged: it runs on Metal, or skips where the runner has none. The Windows runner has no GPU, so the Vulkan rows already skip there; whether an OpenGL 3.3 context came up on it has not been checked from here, and if it did, the OpenGL rows ran there and are now gone. From step 5 until step 15 the Windows lane's tests that need a renderer skip: `caliper_gfx_tests`, the drawing cases of `caliper_export_tests` and of `caliper_embed_tests`, `caliper_cli_shot_temp` and `caliper_cli_shot_data_dir`, and the offscreen run inside `caliper_embed_install`. Still running there are `caliper_tests`, with the layer's and the scene service's headless tests on the recording backend, the loader, manifest, negotiation, search, store and job tests and the codec cases; the command-line tests and `caliper_cli_new_builds`; `caliper_sdk_install` and the build half of `caliper_embed_install`; `caliper_torch_tests` and the applets' model tests on the CPU; `caliper_scene_tests`; the builds of `hello`, `sine_scope` and the scopes against the installed SDK; and the check that the host names no applet. Step 15 brings the skipped tests back on Dawn's SwiftShader.

### The TypeScript views in the browser

Step 9 removes the TypeScript WebGPU views, which duplicate the WebAssembly views, and keeps the Canvas 2D fallback, whose removal is a separate decision that the user has not made.

- **What goes.** `web/src/render/gpu.ts`, the shared device with its colour-table texture and the WGSL colormap; `web/src/render/heatmap-gpu.ts`; `web/src/render/points-gpu.ts`; the WebGPU branches of `web/src/views/heatmap.ts` and `web/src/views/points.ts`; the device in the views' host; and the element's request for a device, its list of kinds that need one and its handling of a lost device for its own views. The element keeps its check for a WebGPU adapter, which chooses the WebAssembly views.
- **What the fallback still draws.** Heatmaps and images (`render/heatmap-2d.ts`), point clouds with their orbit camera, hover and lasso (`views/points.ts`), series and histograms, which were always Canvas 2D, and, once [Web geometry](web-geometry.md#the-fallback-without-webgpu) is built, a scene's points with its notice, the text log in DOM and the bar chart. The TypeScript colormap of Canvas 2D, `render/colormap.ts`, stays, checked against the C++ cases.
- **Who draws what.** With WebGPU, the WebAssembly views draw every kind that their module lists, and they are the only WebGPU drawing in a browser. The element's own views draw with Canvas 2D everything else: a stream given through the `source` property, several series in one plate, a view that the module failed to mount, `renderer="ts"`, and every view in a browser without WebGPU. Their info line says `canvas2d`.
- **The tests.** The projects `webgpu` and `swiftshader` keep running the WebAssembly views, and the element's own views draw Canvas 2D in all three projects.

## Risks

- **A slower fast path.** The layer could add encoding cost or lose Metal's inline constants. The thresholds above catch it, against the commit before each step and against step 0's baseline, and the native backends keep their private paths behind the interface.
- **Pixels that move.** Shaders translated instead of hand-written could change float operation order. The rows of every step that promises unchanged pixels catch it, and a changed row stops the step. The applet ports change their pictures on purpose, and their differences are listed instead.
- **No switch back.** A regression found after a step has landed can no longer be switched off at run time. It is fixed forward, or the step's series is reverted with git, which may also need the later steps that build on it reverted. The fixtures recorded at each step's first commit inherit any defect of the old path; where a fixture and a CPU reference row disagree, the reference row decides.
- **A weaker graphics contract for applets.** The geometry service's rows were byte-exact against a CPU reference for every pixel; the scene service's rows follow the parity rules, with tolerances at edges and on lit and blended pixels ([From the geometry service](scenes.md#from-the-geometry-service)). The user accepted this in choosing the scene service.
- **Particle clouds cost more GPU time.** Discs of six vertices replace hardware points, so the largest clouds of `field_scope`, `flow_scope` and `sculpt_scope` take more GPU time per frame. Each port reports the numbers; the in-place route and its zero copies are unchanged.
- **The scene service freezes the scene model into the applet ABI.** Within epoch 3 the model can only grow by appending, and a change that cannot append costs an epoch bump and a rebuild of every applet.
- **Applets outside the repository.** From step 13 every applet built against the SDK 0.2.0 is refused before its library opens, one that draws through the geometry service no longer builds against the SDK 0.3.0 until it is rewritten, and an embedding host that calls the v1.2 views no longer compiles against `caliper-embed` 2.0.0. Steps 0 to 15 touch none of them, and step 16 ports one, by the user's decision.
- **Windows.** With the Vulkan backend at Windows parity and no old path kept, Windows refuses bridge textures, geometry views, applet scenes, exports and embedding views from step 8 until step 17, or draws them through Dawn with a copy from step 15 ([The Vulkan renderer until Windows parity](#the-vulkan-renderer-until-windows-parity)). The user has accepted that gap. The code that step 8 deletes comes back from history at step 17, where it is ported to the layer rather than moved, and the Windows numbers are then compared with a build of the last commit before step 8.
- **No fallback renderer until Dawn.** From step 5 until step 15 a machine where neither Metal nor Vulkan starts cannot run the app, and the Windows lane's tests that need a renderer skip ([No fallback renderer until Dawn](#no-fallback-renderer-until-dawn)).
- **The integer colormap's cost.** Two more table loads per texel are expected to vanish under a memory-bound pass; step 8 measures it natively, where the keys replace the float formula.
- **Large steps.** Step 11, the scene service with three applets, takes 12.5 builder days, and step 13 changes the ABI, the SDK, the embedding package, the Python package and two applets and deletes the geometry service, in 11. Each lands as a series whose commits all build: in step 13 the commit that bumps the epoch also changes every manifest, and the captures of every applet show that nothing else moved.
- **Translator details.** Naga's handling of Metal binding indices and runtime-sized arrays is unverified from its command line; a small driver on the crate is the fallback.
- **Dawn's weight and pace.** Its build is large and its nightly releases carry no guarantee; Caliper pins a source revision and caches the build.
- **Late picks.** A pick answers a frame or two late, where the points view's CPU hover answers at once, so the CPU hover stays where the data is CPU-visible.

## Limits

- Zero-copy holds only on the same machine and only where the backend imports: Metal for `MTLBuffer`s and shared-memory slots, Vulkan with CUDA for exportable allocations. The browser always copies, and so does the Dawn fallback. A tensor from torch's default CUDA allocator reaches a texture with no copy through the direct kernel, but a buffer use, such as a scene's points, copies it once inside the GPU.
- Until step 13 the geometry service needs storage buffers in the vertex stage; once it is deleted, nothing does, so every draw runs on a WebGPU device in compatibility mode. In the browser and on Dawn, WebGPU's default limits bound what is drawn: 256 MiB buffers, 128 MiB storage bindings and 8,192-pixel textures.
- Views are single-sample. Antialiasing of point discs is done in the fragment stage.
- There is no frame graph, no second queue and no render-to-tensor path.
- From step 5 until step 15 there is no fallback renderer: where neither Metal nor Vulkan starts, the app, `caliper shot` and the embed canvas refuse. Off macOS and Windows there is no renderer at all.
- Windows draws no bridge textures, geometry views, applet scenes, exports or embedding views from step 8 until step 17, except through Dawn with a copy from step 15, a gap that the user has accepted.
- From step 9 the element's own views in a browser draw only with Canvas 2D, so what they draw, including large point clouds given through the `source` property, is drawn at Canvas 2D's speed and budget.
- Applets lose what the scene model does not describe: triangle strips, unlit triangles, per-draw depth and blend state, camera matrices of their own and the rigidity gate ([From the geometry service](scenes.md#from-the-geometry-service)).
- The migration updates the applets and embedding hosts in this repository, and, by the user's decision, one applet outside it.

## Changes to the shared contracts

- **Every line is an expanded quad.** The contracts say lines are expanded quads everywhere. That now holds without exception: the geometry service's one-pixel hardware lines go with it in step 13.
- **CUDA import covers exportable memory only.** The contracts name "CUDA memory through Vulkan external memory". Only allocations made exportable (the `ExportablePool` and future session slots) can be imported; memory from torch's caching allocator has no handle and reaches the layer by the direct kernel or one copy inside the GPU.
- **The scene model gains additive blending and instanced meshes** for the applets' sake, and they are part of the model for Caliper LIB and the browser too ([Scenes](scenes.md#primitives)).

## Decisions

- **The minimum cut first, then the rest of the migration in full.** Decided by the user, who wants to see incremental progress and wants nothing of the full migration dropped. Steps 0 to 4 come first, and scenes start on them. Every remaining step then follows as planned work, not optional work: the retirement of OpenGL (step 5), the colormap and each existing view moved onto the layer (steps 6 to 8), the removal of the TypeScript WebGPU views (step 9), the scene model's additions, the scene service and the applets on it (steps 10 to 12), one version of each graphics service with the geometry service deleted (step 13), the seam at its final shape (step 14), the Dawn fallback (step 15) and the port of one applet outside this repository (step 16). Step 17, the Vulkan backend, stays with Windows parity. Each step is an increment a person can see, under the rule in [Migration](#migration).
- **No backwards compatibility in the graphics change.** Decided by the user, because nothing is deployed or commercialised yet, so nothing outside the repository has to keep working. The seven decisions below follow from it.
- **No dual paths.** Decided by the user. Each step replaces the old code and deletes it in the same series, and a revert is `git revert`. One path at a time is maintained and tested, and the measurements before and after and the byte-exact checks still gate every step.
- **`HostRenderer` shrinks to the window, the frame loop and ImGui setup.** Decided by the user. Its feature-shaped drawing, texture, import and geometry methods, its capability booleans and its debug readbacks are what the layer generalises; keeping them would keep a second interface to every backend.
- **One version of each graphics service, and one epoch bump.** Decided by the user. Revision ladders, caps bits per revision and the sugar's widening of records exist only to keep older applets working, and no such applet is deployed. Every applet in this repository is updated in the same series; applets outside it are not touched, and whether to update them is the user's decision.
- **OpenGL retired early, before Dawn.** Decided by the user. The GLEW path has no device path and no geometry, and every later step would otherwise have to keep its CPU-staged route working. Until Dawn lands, the absent fallback says so by refusing to start.
- **The hand-written shader copies and the platform code in the views deleted as each client moves.** Decided by the user. The generated programs and `import_buffer` replace them, so no copy is left to drift and no view branches on a platform, a device kind or a renderer's name.
- **The TypeScript WebGPU views removed, the Canvas 2D fallback kept.** Decided by the user. The WebGPU views duplicate the WebAssembly views with a third copy of the heatmap and points shaders and of the colormap. Dropping Canvas 2D too is a separate decision that the user has not made.
- **Zero-copy unchanged, and a hard requirement.** Decided by the user. In-place import through `import_buffer` is measured in every step against the commit before it and against step 0's baseline.
- **Protocol compatibility and the UI service's revisions are not part of this.** Decided by the user. Producers of protocol 3.0, the `/v0/stream` alias and the revisions of `caliper.ui` and of the other services that do not draw stay as they are.
- **OpenGL as the first step after the cut.** It depends on nothing, and every later step then has one backend fewer to keep working.
- **The epoch bump in step 13, where the service ids change.** The geometry ids and the revision ids of the tensor bridge and the export service disappear there, so the epoch changes once; the scene service's id was added in step 11 without one, because a new id breaks nothing.
- **The one version keeps the plain `v1` id.** The epoch bump already refuses every applet built for the old tables, so a new major id would protect nothing.
- **Export draws scenes.** Its tables carried the geometry record, so they change with the geometry service anyway, and drawing a scene keeps one drawing path.
- **The typed live views are the one embedding view.** They already draw heatmaps and images through the v1.2 views by name, so folding the v1.2 views into them removes a layer and the Python package's switch between two routes.
- **A scene service for applets, replacing the geometry service.** Decided by the user. Applets get the scene model, the shared camera, picking with events, the overlays, export and, where the scene is wire-capable, recordings and the browser, from one service instead of a fixed menu of draws with cameras of their own, and Caliper keeps one drawing world. It costs about 25.5 builder days of migration in place of about 8.5 for moving the geometry service onto the layer and giving it one version, rewrites eight applets, and changes the graphics contract for applets from byte-exact rows to the parity rules.
- **The applets adapt to the API, not the API to the applets.** Decided by the user. The scene service is designed as a general API; a capability of the geometry service that no general scene model has is dropped and the applet changes, and the port of the applet outside this repository adds a feature to Caliper only where it is general in its own right.
- **The particle applets first, the exporting applets last.** The particle applets each draw one points batch and prove the in-place route; the exporting applets move with the export service, in the step that deletes the geometry service.
- **One applet outside this repository ported, as the last step.** Decided by the user. It shows that an applet built elsewhere runs on the final services; the other applets outside this repository remain the user's to decide.
- **Build a thin layer with three backends: Metal, Vulkan and WebGPU.** Decided by the user. No existing library keeps both zero-copy paths under a licence Caliper can ship. The WebGPU backend serves the browser and, through Dawn, the native fallback.
- **The Vulkan backend comes with Windows parity, not in this sprint.** Decided by the user. It needs Windows hardware with CUDA, and Caliper LIB's sessions are macOS-only today. [Windows parity](windows-parity.md) carries the step. With no old path kept, Windows loses its GPU drawing from step 8 until then.
- **The gap on Windows is accepted.** Decided by the user: Windows and Vulkan are deferred until the user moves to the Windows machine, and the gap is accepted. From step 8, Windows draws its ImGui pages and refuses bridge textures, geometry views, applet scenes, exports and embedding views with a reason, until the Vulkan backend of the layer lands with [Windows parity](windows-parity.md).
- **Shape it to WebGPU.** The browser is the most constrained target, so a WebGPU-shaped interface makes that backend direct and costs the native backends nothing they need.
- **Dawn as the WebGPU implementation, not as the native layer.** It serves the browser and the fallback, where copies are acceptable and reported.
- **Instance-step vertex buffers for per-item data, and no vertex storage once the geometry service is gone.** Vertex buffers take any 4-byte offset and any stride, so imported slots and array columns bind without a copy, an alignment rule or a bind group per draw. Vertex pulling served only the geometry service, so `kCapVertexStorage` and bind group 3 go with it in step 13.
- **No frame graph.** A frame has few passes and no transient resources.
- **Serials replace the completion handles, and the route is returned, never inferred.** One mechanism serves every kind of work, and every info line reports what ran.
- **WGSL as the one source, Naga (pinned) as the translator, generated MSL and SPIR-V committed and checked by a CI regeneration job.** Decided by the user. The language enforces the layer's model, the paths used are Naga's primary ones, and building or shipping needs no Rust and no shader tool.
- **A shader may have a hand-written native version on one platform when that platform needs it.** Decided by the user. The generator skips that output, and the byte-exact tests still cover it.
- **CUDA kernels stay hand-written** and share the colormap's C++ header with the CPU. Decided by the user.
- **Integer keys decide the colormap.** Exactness then depends on no compiler's float arithmetic.
- **One statement of the conventions.** Right-handed, +y up, clip z in [0, 1], column-major, origin at the top-left; the Vulkan backend alone flips.
- **No immediate data in this sprint.** WebGPU's draft gives 64 bytes and browsers have only begun to offer it; dynamic offsets already serve every backend, and immediates can later carry a draw's slot index without changing the draw list.
- **ImGui keeps its backends.** The layer supplies the textures and the device, and orders its passes before the ImGui pass.

## Open decisions

- **How a graphics service changes after this sprint.** Options: (a) append to its one table and record within an epoch, and bump the epoch for an incompatible change, as [One version of each graphics service](#one-version-of-each-graphics-service) plans; (b) return to revision ids with a caps bit per revision. Recommendation: (a) until a first release reaches users outside this repository, then decide again.
- **Applets and embedding hosts outside this repository, apart from the one that step 16 ports.** Options: update each in its own repository once step 13 lands, or leave each until it is next worked on. Recommendation: update each when step 13 lands, since an applet without 3-D needs only its manifest and its service ids, while one that used the geometry service needs the rewrite that the applets here get. This plan touches none of them.
- **The colormap's decision on native paths.** Options: integer keys everywhere, or the float formula natively where the rows prove it, as a hand-written MSL version of the colormap program that the generator skips, with keys in the browser and on Dawn. Recommendation: keys everywhere, unless step 8's measurement shows a cost above the thresholds.
- **How Dawn is obtained.** Options: built from a pinned source revision, or the nightly binaries. Recommendation: pinned source, cached in CI, because the nightly binaries are unsigned and unguaranteed.
