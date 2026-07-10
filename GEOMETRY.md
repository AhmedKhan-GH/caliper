# GEOMETRY.md — caliper.geometry.v1_1: general primitives from zero-copy tensors

**Status: SHIPPED on BOTH platforms (macOS Metal + Windows Vulkan).**
As of 2026-07-09: Phase A (ABI + vend + SDK + host tests) green; Phase C
(Metal) complete — the point/non-point vertex-fn split fixed the
`[[point_size]]` pipeline-class bug and the full §9.2 matrix is implemented
and green on live Metal (13 byte-exact/tolerance rows, 173 assertions,
including the 22-case §2.3 gate battery and stride forward-compat); Phase D
shipped — `mesh_scope` is the §9.3 learned-surface exemplar per
`docs/superpowers/specs/2026-07-09-meshscope-learned-surface-design.md`,
run-verified both ways ("first zero-copy frame drawn (imported geometry, 3
draws)" on Metal+MPS; honest "fallback (no geometry.v1_1 backend)" under
`CALIPER_RENDERER=gl`). Phase B (Vulkan, per
`docs/superpowers/specs/2026-07-09-geometry-v1_1-vulkan-phase-b-design.md`)
complete on the Windows box (RTX 500 Ada): GLSL twins of the Metal geom
shader (std140 params byte-matched to `PrimParams`), depth views, pipeline
cache, per-frame descriptors + dynamic-UBO params ring; ALL 13 Metal §9.2
rows mirrored against the same CPU references and green on real NVIDIA
hardware — every drawing row byte-exact first try — plus a portable
no-CUDA gate-refusal row set; `mesh_scope` run-proven on Vulkan+CUDA (same
provenance line, zero-copy) and honestly falling back under
`CALIPER_RENDERER=gl`. v1 `points-imported` rows untouched and green.

This document is the end-to-end design for
extending the imported-geometry service from "instanced points only" (v1, shipped)
to general graphics primitives — indexed triangles, lines, strips, depth testing,
blend modes, and a fixed shading menu — all vertex-pulled **in place** from
applet-exported device allocations. Zero copies of geometry data, on both
platforms: Vulkan + CUDA (Windows) and Metal + MPS (macOS).

It is written to be implemented by subagents in independent phases (§10) under
the standing execution protocol: verification by artifacts only (commits exist,
files on disk, suites run), never by report text.

Governing docs: `PLATFORM.md` (ABI discipline), `ZEROCOPY.md` (import machinery),
`docs/metal-pipelining.md` (sync), `sdk/include/caliper/services/geometry_v1.h`
(the frozen v1 this revision extends).

---

## 1. Scope and philosophy

### 1.1 What this is

One new **additive** service revision, `caliper.geometry.v1_1`, following the
exact discipline of `tensor_bridge.v1 → v1_1 → v1_2`: prefix-identical struct,
new members appended, new caps bit, no ABI epoch bump, v1 untouched and frozen.

The applet's model stays: *"my geometry is tensors; the host draws them."*
An applet writes vertices/indices/normals/attributes into device memory
(torch CUDA tensors on Windows, MTLBuffers from the MPS pool on macOS), exports
them once through bridge-v1.2 `import_allocation`, and each frame calls one
atomic `draw_primitives` with an array of draw descriptors. The vertex stage
reads simulation memory in place, per frame.

**"Render anything" here means any geometry.** Any shape expressible as
points/lines/triangles — surfaces, wireframes, glyphs, marching-cubes output,
vector art — with appearance chosen from a fixed menu (§4.4). It does NOT mean
applet-supplied shaders; see non-goals.

**Purpose, and a direction-of-flow invariant (decided 2026-07-07):** this
service exists for **real-time visualization OF machine learning and
simulation state** — the Training-Lab/flow_scope/EmbedScope loop: model and
sim write device tensors, the host draws them in place, the human watches
live. It is NOT for machine learning *on* the rendered pixels. Data flows one
way — tensors → pixels → ImGui — and that is a platform invariant, not a
v1_1 gap: there is no render-to-tensor entry point, no readback on any hot
path, and `debug_readback_rgba8` stays test-harness-only. Training consumes
the same tensors the renderer reads, never the renderer's output.

### 1.2 Non-goals (explicit, decided)

| Rejected / deferred | Why |
|---|---|
| Applet-supplied shaders (SPIR-V/MSL upload) | Breaks the seam ("applets only see ImGui + bridge ids"), large security surface. Rejected, not deferred. |
| Render-to-tensor / pixel-based training (visual RL, differentiable rendering, synthetic image data, sensor sim) | **Rejected, not deferred** (§1.1 invariant, 2026-07-07): the goal is visualization *of* ML state, not ML *on* the visualization. Readback would drag a reverse sync contract into every frame; purpose-built sims (Isaac/MuJoCo/Genesis) own that loop if it ever matters. |
| Textures sampled on meshes | Deferred to a future v1_2 (would sample a bridge `CaliperTextureId`); keeps this revision bounded. |
| MSAA, mipmaps on views | Deferred. Views stay single-sample RGBA8. |
| Transparency sorting | Applet orders its draw array; unsorted alpha artifacts are documented, not fixed. |
| Line width > 1 px | `wideLines` is optional in Vulkan; portable thick lines are triangles. `size_px` applies to points only. |
| Back-face culling | v1_1 is two-sided (`CULL_NONE`) everywhere. Deterministic, no winding footguns. Future flag if needed. |
| Per-pixel (Phong) lighting | Lighting is evaluated in the vertex stage (Gouraud, §4.4). Cheaper, matches the trivial-FS structure, easier parity. |

### 1.3 Degradation ladder (unchanged, load-bearing)

Absent caps bit → every new entry point is inert and the applet keeps its CPU
fallback. Any gate violation → return `0`/`false`, emit one `caliper.log.v1`
line, **view pixels untouched** — never a wrong image. All gates run before
anything is encoded (on Metal this means before the render encoder exists,
because creating the encoder performs the clear — same rule as v1).

---

## 2. ABI: `sdk/include/caliper/services/geometry_v1_1.h` (new file)

New header includes `geometry_v1.h`. The struct is prefix-identical to
`CaliperGeometryV1` (same first five members: `struct_size`, `caps`,
`create_view`, `release_view`, `draw_points`) with three appended members.

```c
#define CALIPER_GEOMETRY_V1_1 "caliper.geometry.v1_1"

/* caps() bit 1: create_view_ex / draw_primitives are live. Implies nothing
 * about bit 0; hosts set both when the primitives path exists. */
#define CALIPER_GEOM_CAP_PRIMITIVES (1u << 1)

/* create_view_ex flags */
#define CALIPER_GEOM_VIEW_DEPTH (1u << 0)   /* attach a D32-float depth buffer */

/* CaliperGeomDraw.topology */
#define CALIPER_GEOM_TOPO_POINTS         0u
#define CALIPER_GEOM_TOPO_LINES          1u
#define CALIPER_GEOM_TOPO_LINE_STRIP     2u
#define CALIPER_GEOM_TOPO_TRIANGLES      3u
#define CALIPER_GEOM_TOPO_TRIANGLE_STRIP 4u

/* CaliperGeomDraw.color_mode */
#define CALIPER_GEOM_COLOR_FLAT        0u  /* flat_rgba for every vertex        */
#define CALIPER_GEOM_COLOR_COLORMAP    1u  /* (N,) f32 attr through LUT (§4.3)  */
#define CALIPER_GEOM_COLOR_VERTEX_RGBA 2u  /* (N,) u32 packed rgba8 per vertex  */

/* CaliperGeomDraw.shade_mode */
#define CALIPER_GEOM_SHADE_UNLIT   0u
#define CALIPER_GEOM_SHADE_LAMBERT 1u      /* requires normal_alloc != 0 (§4.4) */

/* CaliperGeomDraw.blend_mode */
#define CALIPER_GEOM_BLEND_OPAQUE   0u     /* blending disabled                 */
#define CALIPER_GEOM_BLEND_ALPHA    1u     /* SRC_ALPHA / ONE_MINUS_SRC_ALPHA   */
#define CALIPER_GEOM_BLEND_ADDITIVE 2u     /* ONE / ONE, both channels (v1 look)*/

/* CaliperGeomDraw.depth_flags */
#define CALIPER_GEOM_DEPTH_TEST  (1u << 0)
#define CALIPER_GEOM_DEPTH_WRITE (1u << 1)
```

### 2.1 The draw descriptor

One fixed-layout C struct per draw call. 8-byte fields first (no packing
surprises), `static_assert(sizeof == 192)` in every consumer.

```c
typedef struct CaliperGeomDraw {
    /* ---- geometry sources: (alloc, byte offset) into bridge-v1.2 imports.
     * pos is (vertex_count, 3) f32, contiguous. index (when index_alloc != 0)
     * is (index_count,) u32. normal is (vertex_count, 3) f32. attr meaning
     * depends on color_mode: COLORMAP -> (vertex_count,) f32; VERTEX_RGBA ->
     * (vertex_count,) u32 packed r|g<<8|b<<16|a<<24. All offsets 4-byte
     * aligned. alloc id 0 = source absent. ---- */
    CaliperAllocId pos_alloc;    uint64_t pos_offset;
    uint64_t       vertex_count;
    CaliperAllocId index_alloc;  uint64_t index_offset;
    uint64_t       index_count;              /* consumed when indexed */
    CaliperAllocId normal_alloc; uint64_t normal_offset;
    CaliperAllocId attr_alloc;   uint64_t attr_offset;

    /* ---- state ---- */
    uint32_t topology;     /* CALIPER_GEOM_TOPO_*  */
    uint32_t color_mode;   /* CALIPER_GEOM_COLOR_* */
    uint32_t shade_mode;   /* CALIPER_GEOM_SHADE_* */
    uint32_t blend_mode;   /* CALIPER_GEOM_BLEND_* */
    uint32_t depth_flags;  /* CALIPER_GEOM_DEPTH_* bits */
    uint32_t flat_rgba;    /* packed LE, used when COLOR_FLAT */
    int32_t  colormap;     /* bridge LUT id, used when COLOR_COLORMAP */
    float    vmin, vmax;   /* colormap range, same rule as v1 */
    float    size_px;      /* point size; ignored for other topologies */

    /* ---- per-draw model transform, column-major. mvp = proj*view*model is
     * premultiplied host-side; for LAMBERT the host also premultiplies the
     * view-space normal matrix (§4.4). Identity = pass world coords. ---- */
    float    model[16];

    uint32_t reserved[2];  /* must be zero; gated (nonzero = refuse) */
} CaliperGeomDraw;
```

### 2.2 New service members (appended after `draw_points`)

```c
    /* v1_1 additions ----------------------------------------------------- */

    /* Like create_view, plus flags. CALIPER_GEOM_VIEW_DEPTH attaches a
     * D32-float depth buffer (renderer-internal; never sampled, never an id).
     * Same texture table, released via release_view. 0 on failure. */
    CaliperTextureId (*create_view_ex)(uint32_t width, uint32_t height,
                                       uint32_t flags);

    /* Render ONE frame of `view`, atomically: gate EVERY draw first, then
     * clear color to clear_rgba (and depth to 1.0 when the view has depth),
     * then encode draws[0..draw_count) in array order into a single pass.
     * draw_stride = the applet's compiled sizeof(CaliperGeomDraw), so the
     * struct can grow additively (host reads min(stride, its own sizeof);
     * stride < the host's known minimum -> refuse).
     * draw_count == 0 is a pure clear.
     * false = NOTHING drawn or cleared, prior pixels intact, one log line.
     * Memory-stability contract identical to draw_points: every addressed
     * byte range is read IN PLACE and must not be rewritten until this
     * view's next draw. */
    bool (*draw_primitives)(CaliperTextureId view,
                            const CaliperGeomCamera* cam,
                            const CaliperGeomDraw* draws, uint32_t draw_count,
                            uint32_t draw_stride,
                            uint32_t clear_rgba);

    /* Reserved for v1_2 (textures-on-meshes). NULL in v1_1. */
    void (*reserved0)(void);
} CaliperGeometryV1_1;
```

`CaliperGeomCamera` is reused unchanged. **Clip-space convention (document in
the header):** GL-style NDC, +y up, clip z in **[0, 1]** (Vulkan/Metal/D3D
convention, not GL's [-1,1]). Applets own all camera math; `flow_scope`
already targets this. Vulkan lands +y-up via the negative-viewport trick;
Metal via a positive-height viewport — exactly as v1 does today.

### 2.3 Gates (all before any encoding; any failure refuses the whole frame)

Per frame: live view; `cam != NULL`; `draw_stride >= host's minimum known size`.
Per draw (skipped entirely when `draw_count == 0`):

1. `topology`, `color_mode`, `shade_mode`, `blend_mode` in range; `reserved` zero.
2. `pos_alloc` live; `pos_offset % 4 == 0`; `vertex_count > 0`;
   overflow-safe: `vertex_count <= U64MAX/12` and
   `pos_offset + vertex_count*12 <= alloc size`.
3. Indexed (`index_alloc != 0`): alloc live, `index_offset % 4 == 0`,
   `index_count > 0`, `index_count*4` bounds-checked. Non-indexed:
   `index_offset`/`index_count` ignored.
4. `LINE_*` topologies need ≥2 consumed vertices, `TRIANGLE*` need ≥3
   (consumed = `index_count` when indexed else `vertex_count`).
5. `shade_mode == LAMBERT` requires `normal_alloc != 0`; when present:
   alloc live, `normal_offset % 4 == 0`, `vertex_count*12` bounds-checked.
6. `color_mode != FLAT` requires `attr_alloc != 0`, live, aligned,
   `vertex_count*4` bounds-checked, and (COLORMAP) a resolvable `colormap` id.
7. Any `depth_flags` bit set on a view created **without**
   `CALIPER_GEOM_VIEW_DEPTH` → refuse (never silently ignore).

**Index *values* cannot be gated host-side** (they live in device memory).
Both backends' vertex shaders clamp: `vi = min(index[i], vertex_count - 1)`.
Deterministic, portable, kills the OOB-read UB. Out-of-range indices produce a
wrong-looking-but-defined image; documented as the applet's contract.

---

## 3. Host-internal seam: `src/host/renderer/host_renderer.h`

Two new virtuals with inert defaults (same pattern as `geom_draw_points`).
The TensorBridge layer (not the renderer) resolves `colormap` ids to LUT
pointers and applet-facing alloc ids to renderer alloc ids, then passes a
host-internal descriptor:

```cpp
// Host-side resolved draw: alloc ids are RENDERER ids, lut256 is resolved
// (null unless COLOR_COLORMAP). Field meanings identical to CaliperGeomDraw.
struct HostGeomDraw {
    uint64_t pos_alloc, pos_offset, vertex_count;
    uint64_t index_alloc, index_offset, index_count;
    uint64_t normal_alloc, normal_offset;
    uint64_t attr_alloc, attr_offset;
    uint32_t topology, color_mode, shade_mode, blend_mode, depth_flags;
    uint32_t flat_rgba;
    const uint32_t* lut256;      // resolved; null when not COLORMAP
    float vmin, vmax, size_px;
    float model[16];
};

virtual uint64_t geom_create_view_ex(int w, int h, uint32_t flags) { return 0; }
virtual bool geom_draw_primitives(uint64_t view_tex,
                                  const float* view16, const float* proj16,
                                  const HostGeomDraw* draws, uint32_t count,
                                  uint32_t clear_rgba) { return false; }
```

ABI-level gates (§2.3 items 1, 3-ignored-fields, 6-colormap-resolution, stride,
reserved-zero) run in the TensorBridge layer; renderer backends re-gate
liveness/alignment/bounds against their own tables (defense in depth, exactly
as `geom_draw_points` does today).

`supports_geometry()` continues to gate caps bit 0; a new
`supports_geometry_primitives()` (default `false`) gates caps bit 1, so a
backend can ship points before primitives.

---

## 4. Rendering semantics (both backends, byte-parity where stated)

### 4.1 The frame

`draw_primitives` = ONE render pass on the view:
clear color (+ depth = 1.0), then each draw in array order. Later draws paint
over earlier ones subject to their own depth/blend state. A wireframe-over-mesh
overlay is therefore: draw 0 = triangles (`DEPTH_TEST|DEPTH_WRITE`), draw 1 =
lines (`DEPTH_TEST` only) — the `LESS_OR_EQUAL` compare (fixed, §4.2) lets
coplanar edges win.

### 4.2 Fixed-function state

| State | Value | Notes |
|---|---|---|
| Depth compare | `LESS_OR_EQUAL` (when testing) | fixed; enables coplanar overlays |
| Depth clear | `1.0` | |
| Depth format | D32 float | Vulkan `VK_FORMAT_D32_SFLOAT`, Metal `MTLPixelFormatDepth32Float` |
| Cull | none (two-sided) | §1.2 |
| Line width | 1.0 fixed | |
| Point size | `size_px` clamped to device limits (existing v1 clamps) | |
| OPAQUE | blend disabled | |
| ALPHA | color `SRC_ALPHA/ONE_MINUS_SRC_ALPHA ADD`; alpha `ONE/ONE_MINUS_SRC_ALPHA ADD` | |
| ADDITIVE | `ONE/ONE ADD` both channels | byte-identical to v1 points |

### 4.3 Vertex pulling — uniform mechanism, no vertex-input state

Everything is pulled by `gl_VertexIndex` / `vertex_id` from whole-bound
buffers at element bases (`byte offset / 4`), exactly like `points.vert`:

```glsl
uint i  = uint(gl_VertexIndex);
uint vi = (use_index != 0u) ? min(idx[idx_base + i], uint(vertex_count) - 1u) : i;
vec3 p  = vec3(pos[pos_base + 3u*vi], pos[pos_base + 3u*vi + 1u], pos[pos_base + 3u*vi + 2u]);
```

Indexed draws are issued as **non-indexed** draws of `index_count` vertices —
no `vkCmdBindIndexBuffer`, no Metal indexed draw, no index-format zoo. One
mechanism for every topology on both backends.

Color, computed in the vertex stage and interpolated (FS stays
`return in.color`):
- `FLAT`: unpack `flat_rgba` (LE bytes / 255) — same unpacking as v1 clear.
- `COLORMAP`: **byte-identical** to v1 / `map_f32_to_rgba8`:
  `idx = uint(clamp((v-vmin)/(vmax-vmin),0,1)*255.0 + 0.5)`; NaN → 0;
  degenerate range → 0; LUT is the shared 256-entry table.
- `VERTEX_RGBA`: per-vertex packed u32, unpack `/255`.

### 4.4 Shading menu

- `UNLIT`: color as computed above.
- `LAMBERT`: single **headlight** (view-space light dir = `(0,0,1)`), Gouraud:
  ```
  n_vs = normalize(nmat * n);                 // nmat: host-computed 3x3
  lit  = 0.30 + 0.70 * max(dot(n_vs, vec3(0,0,1)), 0.0);
  color.rgb *= lit;                           // alpha untouched
  ```
  `nmat = transpose(inverse(upper3x3(view * model)))`, computed **on the CPU**
  per draw (double precision, then truncate) — no shader inverses. Ambient
  0.30 is a spec constant, not a parameter.

Flat/faceted shading = applet duplicates vertices with face normals. Applet
choice, zero host work.

### 4.5 Per-draw parameter block (host-internal reference layout)

Per draw the host uploads (not ABI, but both backends should match for parity
review ease):

```
float4x4 mvp;        // proj * view * model, premultiplied host-side
float4   nmat0, nmat1, nmat2;   // columns of the 3x3 normal matrix (xyz used)
uint     pos_base, idx_base, nrm_base, attr_base;   // element bases (offset/4)
uint     use_index, vertex_count_u32;   // clamp operand (§2.3)
uint     color_mode, shade_mode;
uint     flat_rgba;
float    vmin, vmax, size_px;
```
(pad to 16-byte multiple; ~160 B. This exceeds Vulkan's 128-byte push budget →
delivery per §5.3/§6.3.)

Shader permutations: implementers MAY use one über-shader with uniform
branches (the v1 `use_attr` pattern scaled up) or specialization; the
*observable* semantics above are what parity tests pin.

---

## 5. Windows backend: `src/host/renderer/vulkan_renderer.cpp`

Reuses wholesale: instance/device + UUID-paired CUDA gate, bridge-v1.2 import
(`OPAQUE_WIN32` from `cuMemExportToShareableHandle`), the imported-buffer
memory-barrier discipline of `geom_draw_points` (barrier all referenced
imported buffers `ALL_COMMANDS → VERTEX_SHADER` before the pass), the
pipelined timeline-semaphore ordering (D24/M2a) when live, `dev_note`
diagnostics, and the build-time SPIR-V embedding pattern
(`glslang -V --vn kGeomVertSpv` → `geom_vert_spv.h`, same as `points_vert_spv.h`).

### 5.1 Views with depth

`geom_create_view_ex(w, h, DEPTH)`: the existing color image/view/descset
(unchanged, same texture table) **plus** a per-view `VK_FORMAT_D32_SFLOAT`
image + view (`DEPTH_STENCIL_ATTACHMENT` usage, device-local, never sampled).
A second render pass object `geom_pass_depth_` (color: `loadOp CLEAR`,
`finalLayout SHADER_READ_ONLY_OPTIMAL` — same as `geom_pass_`; depth:
`loadOp CLEAR`, `storeOp DONT_CARE`, final `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`).
Per-view framebuffer binds color(+depth). Track `has_depth` in `Tex`.
`tex_release` destroys the depth image with the view.

### 5.2 Pipeline cache

Lazy map `geom_pipelines_: (topology, blend_mode, depth_flags, has_depth_pass) →
VkPipeline`, created on first use from the single geom vert/frag pair —
worst case dozens, in practice a handful. All share one new pipeline layout
(§5.3). Existing v1 `geom_pipeline_` and its shaders remain untouched
(frozen path; `draw_points` keeps using them).

`ia.topology` maps 1:1 (`POINT_LIST`, `LINE_LIST`, `LINE_STRIP`,
`TRIANGLE_LIST`, `TRIANGLE_STRIP`; `primitiveRestartEnable = VK_FALSE`).
Negative-viewport y-flip exactly as v1. Dynamic state: viewport + scissor.

### 5.3 Descriptors & params per draw

v1 binds one static set (pos/attr/LUT). v1_1 draws reference arbitrary buffer
combinations, so:

- **Per-frame descriptor pool** (reset at each `draw_primitives`), one set per
  draw. Layout: bindings 0-3 = SSBOs pos/idx/nrm/attr (absent source → bind
  `pos` as harmless placeholder, shader never reads it — the v1 trick), 
  binding 4 = LUT (UBO or SSBO, as v1), binding 5 = params
  `UNIFORM_BUFFER_DYNAMIC`.
- **Params ring**: one HOST_VISIBLE|COHERENT buffer, 256-byte-aligned slot per
  draw (`minUniformBufferOffsetAlignment` ≤ 256 everywhere), grown ×2 on
  demand, written once per frame, bound via dynamic offset.

LUT contents: the bridge already owns LUT upload for v1 (device LUT buffer per
draw call); reuse that machinery — one LUT slot per distinct colormap used in
the frame, uploaded before the pass.

### 5.4 Encoding

All gates → barriers over the union of referenced imported buffers → begin
`geom_pass[_depth_]` with clear values → for each draw: bind pipeline from
cache, bind its set + dynamic offset, `vkCmdDraw(consumed_count, 1, 0, 0)` →
end pass → `last_device_path_ = "primitives-imported"` + one `dev_note`.
Submission/sync identical to v1 `geom_draw_points`.

---

## 6. macOS backend: `src/host/renderer/metal_renderer.mm`

Reuses wholesale: `CALIPER_ALLOC_HANDLE_MTLBUFFER` retain-import, unified
memory, runtime-compiled MSL (extend the `kPointsShaderSrc` pattern with a new
`kGeomShaderSrc`), commit-order sync on `queue_` (producer MPS writes
CPU-drained before publish — the flow_scope contract), `debug_readback_rgba8`.

### 6.1 Views with depth

`geom_create_view_ex`: color texture exactly as today
(`RGBA8Unorm`, `Shared`, `ShaderRead|RenderTarget`, cleared at create) plus,
when flagged, a `Depth32Float` texture, `MTLStorageModePrivate`
(`Memoryless` on Apple silicon is a permitted optimization since
`storeAction = DontCare`), usage `RenderTarget`. Track alongside the color
texture; release together.

### 6.2 Pipeline & depth-stencil caches

- `MTLRenderPipelineState` cache keyed by (topology **class**
  point/line/triangle, blend_mode, has_depth_attachment) — Metal binds
  concrete topology at `drawPrimitives:`, so strips share the class pipeline.
  Pixel formats: color RGBA8Unorm; `depthAttachmentPixelFormat` set iff the
  view has depth.
- `MTLDepthStencilState` cache: 4 combinations of (test, write); compare
  `LessEqual` when testing, else `Always`.

### 6.3 Encoding

All gates **before the encoder exists** (encoder creation performs the clear —
v1's hard-won rule). Render-pass descriptor: color clear from `clear_rgba`;
depth attachment (when present) clear 1.0, `DontCare` store. Per draw:
`setRenderPipelineState`, `setDepthStencilState`, `setVertexBuffer` 0-3
(pos/idx/nrm/attr; absent → pos placeholder), `setVertexBytes` LUT (1 KB) and
params (§4.5, well under the 4 KB cap), then
`drawPrimitives:vertexStart:0 vertexCount:consumed`.
Positive-height viewport (no y-flip). Point-size clamp 511 stays.
`last_device_path_ = "primitives-imported"`.

---

## 7. Service vending: `src/host/host_services.cpp` + TensorBridge

Same object, one more table (the v1 pattern at `host_services.cpp:282-310`):

- `TensorBridge` grows `geom_create_view_ex` / `geom_draw_primitives` that
  resolve applet alloc ids → renderer ids, resolve `colormap` → `lut256`,
  run ABI-level gates, build `HostGeomDraw[]` (stack array or small vector),
  and forward. Caps: `geom_caps()` sets bit 1 iff
  `renderer->supports_geometry_primitives()`.
- `kGeom1_1`: prefix-identical initializer (same five function pointers as
  `kGeom1`) + the three new members. `reserved0 = NULL`.
- Register `CALIPER_GEOMETRY_V1_1` in `kIds`; `get_service` returns
  `&kGeom1_1` for it. Null-bridge (headless) → 0/no-op/false, as v1.
- Every refusal path emits one `caliper.log.v1` line naming the gate
  (e.g. `"geometry: draw 3 refused — index bounds"`).

---

## 8. SDK: `sdk/include/caliper/caliper.hpp`

Extend `caliper::Geometry` (keep v1 methods untouched):

```cpp
bool     has_primitives() const;   // caps bit 1
TextureId create_view_ex(uint32_t w, uint32_t h, uint32_t flags);
bool     draw_primitives(TextureId view, const CaliperGeomCamera& cam,
                         const CaliperGeomDraw* draws, uint32_t count,
                         uint32_t clear_rgba);   // passes sizeof(CaliperGeomDraw) as stride
```

Plus a header-only helper `caliper::geom_draw_defaults()` returning a
zero-initialized `CaliperGeomDraw` with `model` = identity — the one footgun
(all-zero model matrix renders nothing) removed at the source.

Torch layout notes (document in the header comment): positions
`.contiguous()` f32 `(N,3)`; indices as **int32** tensors with non-negative
values (torch has no uint32; the bit pattern is what the shader reads);
normals f32 `(N,3)`; packed colors as int32 bit patterns.

---

## 9. Tests

### 9.1 ABI: `tests/test_abi.cpp`

- `CALIPER_GEOMETRY_V1_1` registered; struct prefix-identical to v1
  (offset-of checks on the first five members); `sizeof(CaliperGeomDraw) == 192`;
  headless host: caps 0, every new entry 0/false.

### 9.2 gfx harness: `tests/gfx/gfx_main.cpp` — Vulkan rows + byte-exact-mirror Metal rows

Follow the existing pattern exactly (CPU-computed reference images, readback
compare, `geom_ref`-style helpers). Geometry for byte-exact rows must be
**pixel-center-unambiguous** (edges land between pixel centers) so rasterizer
tie-breaking never enters.

| Row | Pins | Compare |
|---|---|---|
| Unindexed triangle, FLAT, OPAQUE | topology + flat color + coverage | byte-exact |
| Indexed quad (4 verts, 6 idx), COLORMAP extremes at nonzero offsets | index pulling + LUT rule + offsets | byte-exact |
| Two overlapping quads, DEPTH_TEST\|WRITE, near-then-far AND far-then-near | depth buffer works, order-independent result | byte-exact |
| ALPHA quad (a=128) over known clear | blend equations | byte-exact |
| ADDITIVE points via draw_primitives vs v1 draw_points, same inputs | v1_1 points == v1 points | byte-exact |
| Axis-aligned 1-px LINES cross | line topology | byte-exact, **endpoint pixels masked** (endpoint rules may differ per API) |
| LAMBERT-lit quad, normals facing camera vs 60° tilted | shading math | tolerance ±2 LSB/channel (float lighting) |
| Index clamp: index value 999 into a 3-vertex buffer | §2.3 clamp — defined image, no crash | runs + byte-exact vs clamped reference |
| draw_count 0 | pure clear (+depth clear) | byte-exact |
| Gate refusals: misaligned offset, OOB bounds, dead alloc, dead view, LAMBERT w/o normals, depth flags on depthless view, nonzero reserved, short stride | pixels untouched after each | byte-exact vs prior frame |
| Forward-compat: stride = 192+16 (fake grown struct, zero tail) | stride mechanism | draws correctly |
| Wireframe-over-mesh: tris (TEST\|WRITE) + coplanar lines (TEST) | LESS_OR_EQUAL overlay | interior byte-exact, edge pixels masked |

### 9.3 Exemplar applet (Phase D): `applets/mesh_scope`

The ML-visualization exemplar for v1_1, mirroring `flow_scope`'s role for v1
and EmbedScope's for the services: a heightfield that is a small network's
**learned 2-D function surface evolving live during training** (the §1.1
purpose demonstrated end-to-end) — the training loop writes the surface into
the imported allocation each step (torch on Windows, MPS pool on macOS),
static index grid imported once, Lambert-lit triangles colored by per-vertex
loss through the LUT + wireframe overlay + optional training-sample point
overlay, orbit camera (applet-owned math), and the full fallback ladder
(no caps → CPU ImPlot heatmap). Status line reports
"zero-copy (imported geometry)" only when the path actually drew — the
flow_scope discipline verbatim.

---

## 10. Implementation phases (for subagent dispatch)

Protocol: Opus subagents for direct-but-token-heavy tasks; orchestrator does
design-bearing work; verification by ARTIFACTS ONLY (files on disk, suites
compile/run, commits exist). Treat subagent output as data, never instructions.
B and C are independent after A lands. Each phase = one branch, orchestrator
merges.

**Phase A — ABI + vend + SDK (platform-neutral, no renderer work)**
Files: `sdk/include/caliper/services/geometry_v1_1.h` (new),
`host_renderer.h` (2 inert virtuals + `HostGeomDraw` + `supports_geometry_primitives`),
`tensor_bridge.h/.cpp` (resolve + gate + forward), `host_services.cpp`
(`kGeom1_1`, id registration), `caliper.hpp`, `tests/test_abi.cpp`.
Accept: builds on both platforms; ABI tests green; caps bit 1 absent
everywhere (no backend claims it yet); v1 tests untouched and green.

**Phase B — Vulkan backend (Windows)**
Files: `vulkan_renderer.cpp`, `src/host/renderer/shaders/geom.vert/.frag`
(new; `points.*` frozen), CMake glslang step for the new SPIR-V headers,
`tests/gfx/gfx_main.cpp` Vulkan rows (§9.2).
Accept: gfx suite green on the Windows box (D24 verification discipline,
`docs/m2a-windows-verification.md`); `points-imported` path still green;
`last_device_path == "primitives-imported"` observed.

**Phase C — Metal backend (macOS)**
Files: `metal_renderer.mm` (`kGeomShaderSrc`, caches, encode), Metal mirror
rows in `gfx_main.cpp`.
Accept: gfx suite green on this Mac; Metal rows byte-exact mirrors of Vulkan
rows (same references); v1 Metal rows untouched.

**Phase D — Exemplar + docs**
Files: `applets/mesh_scope/*`, `ZEROCOPY.md` (imported-geometry table gains a
primitives row per origin), `docs/STATUS.md`, docs wiki `geometry.md`.
Accept: mesh_scope runs zero-copy on this Mac (Metal) with the status line
proving the path; fallback ladder demonstrated by forcing GL
(`CALIPER_RENDERER=gl`); docs updated.

---

## 11. Roadmap: from points to the goal

The goal (stated 2026-07-07): **complex simulations and digital twins,
trained on, watched live** — real-time visualization *of* machine learning,
never machine learning *on* the visualization (§1.1 invariant).

Status labels are honest: SHIPPED = merged and verified; **SPEC'D = this
document**; DIRECTIONAL = intent, not design — each needs its own spec pass
before any implementation.

| Stage | Capability | Status | What it unlocks toward the goal |
|---|---|---|---|
| R0 | `geometry.v1` instanced points | SHIPPED (both platforms) | particle sims, embeddings — flow_scope, EmbedScope |
| R1 | `geometry.v1_1` — this spec: indexed tris/lines/strips, depth, blend, fixed shading | **SPEC'D**, phases §10 A–D | any *geometry*: deforming meshes, learned surfaces, twin structure; `mesh_scope` proves live-training viz |
| R2 | `geometry.v1_2` — textures on meshes (sample a bridge `CaliperTextureId`; `reserved0` slot) | DIRECTIONAL | field data draped on geometry: heatmap-on-terrain, activation-on-surface — the twin's *state painted on its shape* |
| R3 | Instanced transforms from an imported alloc (one mesh × (N,16) f32 model matrices) | DIRECTIONAL | many-part twins at scale: fleets, swarms, articulated repeats — N objects, one draw, still zero-copy |
| R4 | Host-neutral service layer / second host (Compass, Phase 6 `libcaliper` per PLATFORM.md) | DIRECTIONAL | twins embedded outside the Caliper shell |

Explicitly NOT on this roadmap, at any stage (§1.2): render-to-tensor,
applet shaders, photoreal/PBR. If a stage seems to need one of these, the
stage is mis-scoped — redesign it, don't relitigate the invariant.

Things the goal needs that require **no new host capability** — do not grow
the ABI for them:
- **Static assets** (CAD/OBJ/glTF twin shells): applet-side loaders fill
  vertex/index tensors; the host never learns about file formats.
- **Picking/hover/selection**: applet-side ray casting against its own
  tensors (the flow_scope pattern) — interaction is math the applet owns.
- **Annotation/HUD**: ImGui composites over the view image for free.
- **Training-side anything**: training consumes the same tensors the
  renderer reads; the Training-Lab pattern already generalizes.

Sequencing: R1 phases A–D first (§10). R2/R3 are additive revisions in either
order, each gated on a demonstrated applet need — not built on spec. R4 is
the platform's call (PLATFORM.md Phases 3–6), not geometry's.

## 12. Resolved design questions (do not relitigate in implementation)

| Question | Decision | Why |
|---|---|---|
| New service vs v2 rewrite | Additive `v1_1`, prefix-identical | tensor_bridge discipline; v1 is IMMUTABLE and stays |
| Multi-draw API shape | One atomic call with a descriptor array | preserves gate-everything-then-encode; a stateful begin/end can fail after the clear, violating pixels-untouched |
| Index buffers | Vertex-pulled u32 + shader clamp | one mechanism both backends; index values are ungateable host-side; clamp kills UB deterministically |
| Struct evolution | `draw_stride` parameter | additive growth without a new entry point |
| Params delivery | Vulkan dynamic-UBO ring / Metal setVertexBytes | Lambert params exceed the 128-B push budget |
| Lighting | Fixed headlight Lambert, Gouraud, ambient 0.30 | fixed menu > applet shaders; VS-heavy/trivial-FS matches v1 structure |
| Culling | None | two-sided; no winding footguns; deterministic |
| Depth on plain v1 views | Refused, not ignored | degradation ladder: never a silently-wrong image |
| Custom shaders | Rejected outright | breaks the applet seam and its security model |
| Render-to-tensor readback | Rejected outright — one-directional flow is an invariant (§1.1) | state-based training viz is the product; pixel-based training is explicitly not; do not add readback entry points "while in there" |
