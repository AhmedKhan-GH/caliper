# Renderer core

This page plans the renderer core, the first work item of sprint 4: a graphics layer that each backend implements once, one shader source translated for every backend, and a scene core that every view draws through. It is designed before [scenes](scenes.md) and [web geometry](web-geometry.md), so that scenes, the browser's geometry and the native WebGPU fallback through Dawn are each built once instead of once per backend. It records what the renderers look like today, specifies the layer's interface, chooses the shader language, defines the camera and scene conventions every backend shares, expresses zero-copy as a backend capability with a measurement that proves the fast path keeps its speed, weighs adopting an existing library, and orders the migration.

## What is built and what is planned

Nothing on this page is built. It rests on these built pieces:

- **The renderer seam**, `HostRenderer` in `src/host/renderer/host_renderer.h`, with its Metal, Vulkan and OpenGL backends on `main` (Rendering (`docs/explanations/rendering.md`)) and the browser's WebGPU backend on the sprint 2 branch (`src/host/renderer/webgpu_renderer.cpp`).
- **The zero-copy paths** of Zero-copy tensor visualization (`docs/explanations/zero-copy.md`) and Vulkan and CUDA on Windows (`docs/explanations/vulkan-cuda.md`): in-place reads of MPS `MTLBuffer`s, imported allocations, the CUDA direct path and the sync ladder.
- **The geometry service**, `caliper.geometry.v1` to `v1_3` (`docs/references/service-geometry-v1.md`), and the tensor bridge (`docs/references/service-tensor-bridge-v1.md`), whose public ABIs this plan does not change.
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
4. **The browser's TypeScript views**, the fallback where WebGPU is absent, with their own camera (`web/src/views/camera.ts`) and their own WGSL for heatmaps and points. They stay as they are.

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
- **No point size and no wide lines.** New programs draw points and wide lines as instanced quads expanded in the vertex stage. Hardware point lists and line lists remain, one pixel wide, because WebGPU has them and the geometry service's byte-exact rows depend on them.
- **Vertex and instance-step buffers** feed per-item data: an attribute can sit at any 4-byte-aligned offset and any stride up to 2,048 bytes, so an imported slot or a column of a wider array is bound without a copy and without the 256-byte alignment that storage bindings need.
- **Storage buffers in the vertex stage** are an optional capability, used only by the geometry service's vertex pulling. WebGPU's compatibility mode allows none there.
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
    uint32_t  dynamic_offsets[2];        // the draw block and the colormap table, in the frame's arena
    VertexBinding vertex[6];
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
    virtual Program   create_program(uint32_t program_id) = 0;     // from the generated table
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
| Tensor bridge (`caliper.tensor_bridge.v1` to `v1_2`, ABI unchanged) | Bridge textures are layer textures; `update_texture` calls `update_texture_from_tensor`; `import_allocation` calls `import_buffer`; `alloc_shared` on Vulkan stays a backend-private path behind `kCapCudaWritesTexture` |
| Embed v1.2 views and the heatmap and image live views | The bridge as above; the path label comes from the returned route, never from the tensor's device |
| Points live view | A scene of one entity in the scene core; `bind_buffer` becomes one `import_buffer` call, and `live_views_metal.mm` goes |
| Series and histogram live views | Unchanged: ImPlot |
| Geometry service (`caliper.geometry.v1` to `v1_3`, ABI unchanged) | Another client of the layer. Allocation ids map to layer buffers. `draw_primitives` keeps its gates, then compiles the draw array into a draw list of one geometry program that pulls vertices from storage buffers (`kCapVertexStorage`), which keeps the index-clamp contract. Points become instanced quads; lines stay one-pixel hardware lines. On a backend without vertex storage the caps bits stay unset, as on OpenGL today |
| Scenes | The scene core compiling entities into draw lists ([Scenes](scenes.md)) |
| The browser | The WebGPU backend of the layer running the same scene core and live views ([Web geometry](web-geometry.md)); the TypeScript fallback views are unchanged |

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

The same function serves the compute colormap and the vertex-stage colormap of points, meshes and the geometry service's `COLORMAP` mode. The CUDA kernel implements the same steps from a C++ header shared with the CPU, marked `__host__ __device__`, so the algorithm has two sources, WGSL and C++, instead of eleven copies.

The verification suite proves it per backend. A headless test runs the C++ implementation against `map_f32_to_rgba8` for all 2<sup>32</sup> inputs on a set of ranges (under a slow label) and on boundary sets in the fast suite. New rows of the graphics suite run, on every backend and every path (compute from a layer buffer, from an imported buffer, the CUDA direct kernel, and the vertex-stage colormap of a points draw), for ranges that increase, decrease, are degenerate, have an infinite end, span subnormals or reach the largest floats: each threshold key and its two neighbours, ±0, ±∞, quiet, signalling and negative NaNs, and 65,536 random bit patterns, each compared byte for byte with the reference. The browser harness of the sprint 2 branch runs the same inputs on SwiftShader and on a hardware adapter. If the native fast path were measurably slower with the integer decision, the native backends could keep the float formula whose exactness the rows prove today, behind the same rows; the measurement below decides.

### Recommendation

Write every shader once in WGSL, translate it to MSL and SPIR-V with Naga (pinned) at developer time, and check the generated sources in, with a CI job that regenerates them and fails on a difference. The browser and Dawn take the WGSL directly. WGSL enforces the layer's model, so a shader that compiles is portable by construction; Naga's WGSL front end and its MSL and SPIR-V back ends are its primary paths; and the browser, the target least under Caliper's control, runs the shader exactly as written. A generator (`tools/shaders/`) concatenates shared snippets, emits each program's bind group description and a C++ struct with `static_assert`ed offsets for every uniform block, and replaces the four hand-synchronised copies of the geometry parameters. Builds need no shader tool, and the Windows build drops its build-time glslang. Two points are unverified and are checked in the step that introduces the generator: whether Naga's command line takes an explicit Metal binding map (otherwise a small driver on the `naga` crate passes one), and how its Metal output sizes runtime-sized arrays when bounds checks are off. The CUDA kernels stay hand-written. Slang is the alternative if the programs grow to need generics and modules.

The user has decided this, with one addition. A shader may have a hand-written native version on one platform when that platform needs it, for example an MSL version that keeps a Metal-only fast path. The shader's entry in the generator's list then names that platform as hand-written, the generator skips that output and leaves the hand-written file alone, and the CI regeneration job checks only the generated outputs. The hand-written version takes the same bind group description and uniform block structs that the generator emits, and the byte-exact rows of the graphics suite cover it exactly as they cover a generated one, so it cannot drift from the WGSL source unnoticed. Because the generated MSL and SPIR-V are committed, building and shipping Caliper needs no Rust and no shader tool; only a developer who changes a shader, and the CI job, run Naga.

## The scene core

The scene core is the one camera and scene model that every view uses: the points live view, the geometry service's views where an applet wants it, the scene view and the browser. Its content, the entities, primitives and arrays, is fixed by the contracts and elaborated in [Scenes](scenes.md); this section defines the core they compile through.

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

The points live view becomes the one-entity case: its payload is a batch, its `Camera` struct becomes `Orbit` or `PanZoom`, its fit becomes `fit`, and its CPU hover stays as the fast hover where the payload is CPU-visible, with the pick pass as the general pick. Its pixels must not change: a gfx row draws the same payload through the old and the new path and compares them. ImPlot3D stays as it is. It draws through ImGui's draw lists with its own camera, it serves applets' small plots well, and it costs nothing on the GPU path; an applet that wants the scene core's camera uses the geometry service instead.

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
    bool allow_copy = true;   // false: refuse rather than copy (geometry service applets ask for this)
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
- **Stability.** The bytes must not be rewritten until that submit completed: the memory-stability contract of the geometry service and of imported allocations, unchanged.
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

The before and after builds come from adjacent commits, in Release, on the reference Mac (and the Windows machine for Vulkan steps), on mains power, with the same torch, run interleaved three times each; the benchmarks already record the load average. Results are committed under `python/bench/results/` with the step's name. A step that misses a threshold is fixed or reverted before the next one starts.

## The alternatives

The deciding criterion is whether zero-copy import survives on Metal and on Vulkan with CUDA.

| Library | Licence | Weight | Backends | Wraps an `MTLBuffer` | Imports Vulkan external memory as a buffer | ImGui | Browser |
|---|---|---|---|---|---|---|---|
| Dawn | BSD-3-Clause | Large C++ build (abseil, SPIRV-Tools, Tint and others); unsigned nightly binaries | Metal, Vulkan, D3D12, D3D11, OpenGL ES, SwiftShader | No. `SharedBufferMemory` takes a D3D12 resource, a Windows handle or a host pointer only; `MetalBackend.h` offers no wrap | No. `WrapVulkanImage` wraps images, not buffers; `SharedFence` does take an `MTLSharedEvent` and Vulkan semaphores | `imgui_impl_wgpu` | The same API in the browser through emdawnwebgpu |
| wgpu-native | MIT or Apache-2.0 | Prebuilt libraries (v29.0.1.1) | Metal, Vulkan, D3D12, OpenGL | No in the C API, which has only getters such as `wgpuDeviceGetNativeMetalDevice`. The Rust `wgpu-hal` has `metal::Device::buffer_from_raw`, out of reach from C++ | No. `wgpu-hal`'s Vulkan side wraps textures (`texture_from_raw`), not buffers | `imgui_impl_wgpu` | Not through wgpu-native |
| bgfx | BSD-2-Clause | Source build with its own shader compiler, `shaderc` | Metal, Vulkan, D3D, OpenGL, WebGPU through Dawn | No. Native pointers only for textures, through the `_external` argument of `createTexture2D`; no buffer equivalent | No | Its examples' ImGui renderer | WebGL 2 |
| sokol_gfx | zlib | One header | Metal, D3D11, OpenGL, WebGPU; Vulkan "experimental" | Yes, `sg_buffer_desc.mtl_buffer` | No: no Vulkan injection fields | `sokol_imgui.h` | WebGL 2 and WebGPU |
| Diligent Engine | Apache-2.0, but its Metal backend is "available under commercial license" | Large; C++20 | D3D11, D3D12, Vulkan, Metal (commercial), OpenGL, WebGPU through Dawn | `IRenderDeviceMtl::CreateBufferFromMtlResource`, in the commercial backend only | Wraps a `VkBuffer` with `IRenderDeviceVk::CreateBufferFromVulkanResource` and a timeline semaphore with `CreateFenceFromVulkanResource` | `ImGuiImplDiligent` | WebGPU and WebGL |

No library keeps both zero-copy paths under a licence Caliper can ship: Dawn and wgpu-native lose both, bgfx loses both, sokol_gfx keeps Metal but has no stable Vulkan with injection, and Diligent keeps both only with a commercial Metal backend. The recommendation, which the user has accepted, is to build the thin layer with Metal, Vulkan and WebGPU backends, and to use Dawn as the WebGPU implementation where copies are acceptable and reported: in the browser through emdawnwebgpu, and natively as the fallback that replaces OpenGL. This would change if Dawn gained `SharedBufferMemory` for an `MTLBuffer` and for Vulkan external memory, because its `SharedFence` already covers the synchronisation; then one WebGPU backend on Dawn could serve every role, and Caliper would maintain one backend instead of three. Dawn's interop is worth re-reading at each sprint.

## Migration

The user has decided the order: the minimum cut first, so that scenes can start, and then every remaining step in full. No step is optional, and each one is an increment that can be seen and checked on its own:

- it lands as its own series of commits;
- it has measurements before and after against the thresholds of [Keeping the fast path as fast](#keeping-the-fast-path-as-fast), and the byte-exact pixel checks of the graphics suite;
- it has offscreen captures of what it changed, taken without a window, for the user to look at;
- the next step does not start until the current one passes all of these.

The renderers and views move behind the layer one at a time, with the byte-exact graphics suite green after every step and the measurement above passing. Until a step lands, the old path stays in place and the step is reverted by switching its client back.

| Step | What changes | What proves it | Revert | Builder days |
|---|---|---|---|---|
| 0. Counters and baseline | Copy counters and Metal frame-time statistics in today's renderers, the points benchmark row, the geometry timing row; a recorded baseline | The counters match the known routes (an MPS heatmap shows 0 copies) | Additions only | 2 |
| 1. Layer and Metal backend | `src/host/gfx/`: the interface, validation, the uniform arena, frames, serials, readback; the Metal backend on the renderer's device and queue; a recording backend for headless tests | Headless tests of validation and recorded draw lists; a gfx row draws a quad through the layer and reads it back byte-exact | Nothing uses it yet | 6 |
| 2. Shader generator | WGSL sources, the generator, pinned Naga, generated MSL and SPIR-V checked in, the CI check; the scene programs | The generated sources rebuild identically in CI; the programs compile on Metal and in a browser | Additive | 3 |
| 3. Scene core | The tree, world matrices, cameras, bounds and fitting, compile, the ID target and picks | Headless tests of matrices, fits and compiled draw lists; a gfx row of a one-entity scene | Nothing uses it yet | 6 |
| 4. WebGPU backend of the layer, browser | The sprint 2 backend's resources and draws move behind the layer; WGSL from the one source | The browser harness: screenshots, the colormap rows on SwiftShader and a hardware adapter | Its old entry points stay one step | 5 |
| Minimum cut | Steps 0 to 4 | | | 22 |
| 5. Points live view on the scene core | `bind_buffer` becomes `import_buffer`; `live_views_metal.mm` goes; the same in the browser | `test_live_views`; the old-against-new pixel row; the points benchmark | Switch back to the renderer calls | 3 |
| 6. Colormap by keys | The integer decision in every path, the shared C++ header, the CUDA kernel's source and PTX | The exhaustive CPU test; the boundary rows on Metal, in the browser, and on Vulkan with CUDA | Keep the float formula natively (an open decision) | 3 |
| 7. Texture path on Metal | The bridge, the embed views and the heatmap and image live views through `update_texture_from_tensor` | Every gfx/Metal texture row; `bench_live` and `bench_show` within thresholds | Switch the bridge back | 3 |
| 8. Geometry service on Metal | One geometry program; points as quads; one-pixel lines | All gfx/metal geometry rows unchanged | Switch the service back | 5 |
| 9. Dawn fallback | Dawn as an optional component built from pinned source; the WebGPU backend compiled against it; Metal or Vulkan, then Dawn | The staged rows on Dawn over Metal, D3D12 and SwiftShader; the host's screens draw | Remove the option | 5 |
| 10. Vulkan backend, with [Windows parity](windows-parity.md) | The Vulkan backend of the layer, with the CUDA direct path, the buffer path and semaphores as private paths; texture path, geometry service and points on it | Every gfx/Vulkan and Vulkan+CUDA row on Windows hardware; the timing rows and `CALIPER_VULKAN_STATS` within thresholds | Switch the clients back | 8 |
| 11. Retire OpenGL | `gl_renderer.cpp`, GLEW and the GL test environment go; the window falls back to Dawn | The rows the GL environment runs today pass on Dawn; CI runs them headless on SwiftShader | Restore from history | 2 |
| 12. Clean-up | `HostRenderer` shrinks to window, canvas and capture; the hand-written MSL and the duplicated structs go, and the hand-written GLSL goes with the Vulkan step of [Windows parity](windows-parity.md), since the Vulkan renderer needs it until then; Rendering (`docs/explanations/rendering.md`), the zero-copy pages and the [roadmap](roadmap.md#renderers) are updated | The whole suite; the docs build | Not needed | 3 |
| Total | Steps 0 to 12. Step 10, 8 days, is done with Windows parity, which leaves 46 in this sprint | | | 54 |

The minimum cut is what scenes and web geometry need before they start: the layer with its Metal backend, the generator with the scene programs, the scene core, and the browser's WebGPU backend, 22 builder days. Steps 1 and 3 can run in parallel once the interface header is agreed, and step 4 in parallel with step 3, so the cut takes about two weeks with two builders. Steps 5 to 8 follow during or after scenes, on macOS. OpenGL is retired at step 11, once Dawn covers its cases on macOS and Windows: no driver, a remote desktop, a virtual machine and headless CI. The Vulkan step needs Windows hardware with CUDA and is done with [Windows parity](windows-parity.md), not in this sprint, as the user decided; until then scenes on Windows draw through Dawn, copying and saying so. Steps 4 and 5 depend on sprint 2's WebGPU backend and live views being merged. Nothing depends on sprint 3.

## Risks

- **A slower fast path.** The layer could add encoding cost or lose Metal's inline constants. The thresholds above catch it, and the native backends keep their private paths behind the interface.
- **Pixels that move.** Points drawn as quads instead of hardware points, and shaders translated instead of hand-written, could change float operation order or edge coverage. The existing rows place points on pixel centres, where quads and point sprites cover the same pixels; every geometry row has to stay green, and a changed row stops the step.
- **The integer colormap's cost.** Two more table loads per texel are expected to vanish under a memory-bound pass, but it is measured before step 6 lands.
- **Translator details.** Naga's handling of Metal binding indices and runtime-sized arrays is unverified from its command line; a small driver on the crate is the fallback.
- **Dawn's weight and pace.** Its build is large and its nightly releases carry no guarantee; Caliper pins a source revision and caches the build.
- **Two paths during the migration.** Until step 12 the old renderer paths and the layer coexist and both need maintenance; the order keeps that window short.
- **Late picks.** A pick answers a frame or two late, where the points view's CPU hover answers at once, so the CPU hover stays where the data is CPU-visible.

## Limits

- Zero-copy holds only on the same machine and only where the backend imports: Metal for `MTLBuffer`s and shared-memory slots, Vulkan with CUDA for exportable allocations. The browser always copies, and so does the Dawn fallback. A tensor from torch's default CUDA allocator reaches a texture with no copy through the direct kernel, but a buffer use, such as a scene's points, copies it once inside the GPU.
- The geometry service needs storage buffers in the vertex stage, so it is unavailable on a WebGPU device in compatibility mode, as it is on OpenGL today. In the browser and on Dawn, WebGPU's default limits bound what is drawn: 256 MiB buffers, 128 MiB storage bindings and 8,192-pixel textures.
- Views are single-sample, as the geometry service fixes. Antialiasing of point discs is done in the fragment stage.
- There is no frame graph, no second queue and no render-to-tensor path.

## Changes to the shared contracts

- **One-pixel lines stay hardware lines in the geometry service.** The contracts say lines are expanded quads everywhere. Expanded quads apply to every line with a width; the geometry service's one-pixel lines stay hardware line lists, which WebGPU offers, because its byte-exact rows (for example the crossing one-pixel lines with masked endpoints) depend on the hardware rule.
- **CUDA import covers exportable memory only.** The contracts name "CUDA memory through Vulkan external memory". Only allocations made exportable (the `ExportablePool` and future session slots) can be imported; memory from torch's caching allocator has no handle and reaches the layer by the direct kernel or one copy inside the GPU.

## Decisions

- **The minimum cut first, then the rest of the migration in full.** Decided by the user, who wants to see incremental progress and wants nothing of the full migration dropped. Steps 0 to 4 come first, and scenes start on them. Every remaining step then follows as planned work, not optional work: each existing view moved onto the layer (steps 5 to 8), the Dawn fallback (step 9), the retirement of the OpenGL path once Dawn covers its cases (step 11) and the clean-up (step 12). Step 10, the Vulkan backend, stays with Windows parity. Each step is an increment a person can see, under the rule in [Migration](#migration).
- **Build a thin layer with three backends: Metal, Vulkan and WebGPU.** Decided by the user. No existing library keeps both zero-copy paths under a licence Caliper can ship. The WebGPU backend serves the browser and, through Dawn, the native fallback.
- **The Vulkan backend comes with Windows parity, not in this sprint.** Decided by the user. It needs Windows hardware with CUDA, and Caliper LIB's sessions are macOS-only today. [Windows parity](windows-parity.md) carries the step.
- **Shape it to WebGPU.** The browser is the most constrained target, so a WebGPU-shaped interface makes that backend direct and costs the native backends nothing they need.
- **Dawn as the WebGPU implementation, not as the native layer.** It serves the browser and the fallback, where copies are acceptable and reported.
- **Instance-step vertex buffers for per-item data, vertex storage only for the geometry service.** Vertex buffers take any 4-byte offset and any stride, so imported slots and array columns bind without a copy, an alignment rule or a bind group per draw; vertex pulling keeps the geometry service's index-clamp contract and byte-exact rows.
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

- **The colormap's decision on native paths.** Options: integer keys everywhere, or the float formula natively where the rows prove it and keys in the browser and Dawn. Recommendation: keys everywhere, unless step 6's measurement shows a cost above the thresholds.
- **How Dawn is obtained.** Options: built from a pinned source revision, or the nightly binaries. Recommendation: pinned source, cached in CI, because the nightly binaries are unsigned and unguaranteed.
