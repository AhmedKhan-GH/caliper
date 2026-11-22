# caliper.geometry.v1 to v1_3

The geometry services draw imported 3-D data into an offscreen view. An applet writes vertices, indices, normals, and attributes into device memory. It exports them once through [`caliper.tensor_bridge.v1_2`](caliper-tensor-bridge-v1.md). The host draws them in place, with no copy of the geometry data, into a view texture. The applet shows the view with `ImGui::Image`. [Rendering](../rendering.md#geometry-path) describes the draw on each backend.

Geometry is a separate service and not a tensor-bridge revision. The bridge means "a tensor becomes an image", and cameras and draw calls are a different vocabulary. The two services share id spaces. A view is a `CaliperTextureId` in the bridge texture table, and a geometry source is a `(CaliperAllocId, byte offset)` pair. Imports, caches, gates, and lifecycle work as in the bridge.

## Versions

Each revision is additive. The earlier struct prefix stays frozen, and a revision adds no entry point after `v1_1`.

| Service id | Adds | Caps bit | Draw record size |
|---|---|---|---|
| `caliper.geometry.v1` | Instanced points | 0 (`CALIPER_GEOM_CAP_IMPORTED_POINTS`) | None |
| `caliper.geometry.v1_1` | General primitives and depth views | 1 (`CALIPER_GEOM_CAP_PRIMITIVES`) | 192 bytes |
| `caliper.geometry.v1_2` | Textures on meshes | 2 (`CALIPER_GEOM_CAP_TEXTURED`) | 216 bytes |
| `caliper.geometry.v1_3` | Instanced transforms | 3 (`CALIPER_GEOM_CAP_INSTANCED`) | 256 bytes |

!!! info "Platform support"
    Points (`v1`) run on Metal and Vulkan. Primitives, textures, and instanced transforms run on Metal and Vulkan with CUDA.
    Both backends match one shared CPU reference byte for byte.
    If a backend lacks a path, or the host has no renderer bound, the matching caps bit is unset and every entry point is inert.
    Ship a CPU fallback for that case, as in the worked example below. An absent capability never gives a wrong image.

## Non-Goals

The service draws geometry from a fixed menu. The table lists what it does not provide.

| Not provided | Why |
|---|---|
| Applet-supplied shaders | Applets see only ImGui and bridge ids, and shaders open a large security surface |
| Render-to-tensor or readback | Data flows one way, from tensors to pixels |
| MSAA | Views are single-sample RGBA8 |
| Line width above 1 px | Portable thick lines are triangles. `size_px` applies to points only |
| Per-pixel lighting | Lighting is evaluated per vertex |
| Transparency sorting | The applet orders its draw array, and unsorted alpha shows artifacts |

## `caliper.geometry.v1` Instanced Points

The v1 service draws instanced points from imported allocations, built for particle clouds. Meshes and lines come with v1_1.

### Point Calls

<!-- c-api calls sdk/include/caliper/services/geometry_v1.h -->

### Camera Type

<!-- c-api types sdk/include/caliper/services/geometry_v1.h level=4 -->

### Point Ids

<!-- c-api ids sdk/include/caliper/services/geometry_v1.h -->

### Offscreen View Pattern

1. Call `create_view(width, height)`. It returns a `CaliperTextureId` in the tensor-bridge texture table, or `0` on failure. Sizes are physical (framebuffer) pixels. Recreate the view when the content region changes by more than a few pixels.
2. Call `draw_points(view, cam, …)` each frame. It renders one frame of the view atomically. It clears to `clear_rgba`, then draws `count` points.
3. Cast the id with `caliper::Bridge::imtex(view)` and show it with `ImGui::Image`. Use the logical size, which is `physical / DisplayFramebufferScale`, so one texel maps to one framebuffer pixel.
4. Call `release_view(view)` on the frame thread. Do not release the view through the bridge.

### Memory-Stability Contract

The host reads the addressed bytes in place. The applet must not rewrite them until the next draw of the same view. On the worker side, drain the producer stream or queue once at the handoff, before it publishes the slot. The exemplars use this triple-buffer discipline with [`caliper.jobs.v1`](caliper-jobs-v1.md). The frame thread does every draw and never touches the learner's tensors. [Zero-copy tensor visualization](../../python/zero-copy-tensors.md#drain-rung-geometry) explains the reasons for both halves of this rule.

## `caliper.geometry.v1_1` General Primitives

The first five members of the v1 table are identical in v1_1. Three members follow them. The ABI epoch does not change.

Hosts set caps bit 0 and bit 1 together when the primitives path exists.

### Primitive Calls

<!-- c-api calls sdk/include/caliper/services/geometry_v1_1.h -->

### Primitive Types

<!-- c-api types sdk/include/caliper/services/geometry_v1_1.h level=4 -->

### Primitive Ids

<!-- c-api ids sdk/include/caliper/services/geometry_v1_1.h -->

### Depth Views

`CALIPER_GEOM_VIEW_DEPTH` views use the same texture table and `release_view` as plain views. The renderer never samples the depth buffer, and it never gives the buffer an id.

### Atomic Gated Frame

`draw_primitives` renders one frame of the view atomically:

1. Gate every draw (see [Draw Gates](#draw-gates)).
2. Clear the color to `clear_rgba`, and clear depth to `1.0` if the view has depth.
3. Encode `draws[0..draw_count)` in array order into a single render pass. Later draws paint over earlier ones, subject to their own depth and blend state.

If any gate fails, the host writes one `caliper.log.v1` line that names the gate. The API is one atomic call with a descriptor array and not a begin/end pair, because a begin/end pair can fail after the clear.

The host reads `min(stride, its own sizeof)`, so the struct can grow additively. The [C++ sugar](#c-sugar) passes `sizeof` for you.

### Draw Descriptor Semantics

The `COLORMAP` index rule is byte-identical to `texture_from_tensor_mapped` and `map_f32_to_rgba8` in the tensor bridge:

```
idx = uint(clamp((v - vmin) / (vmax - vmin), 0, 1) * 255.0 + 0.5)
```

`NaN` maps to `0`, and a degenerate range with `vmin == vmax` maps to `0`. The table has 256 entries. The bridge LUT ids are `CALIPER_CMAP_VIRIDIS` (0), `_MAGMA` (1), and `_RDBU` (2).

`LAMBERT` uses one headlight with the view-space light direction `(0,0,1)`. The host shades per vertex and interpolates:

```glsl
n_vs = normalize(nmat * n);                        // nmat: host-computed 3x3
lit  = 0.30 + 0.70 * max(dot(n_vs, vec3(0,0,1)), 0.0);
color.rgb *= lit;                                  // alpha untouched
```

The host computes `nmat = transpose(inverse(upper3x3(view * model)))` on the CPU for each draw. The `0.30` ambient term is fixed and is not a parameter. For faceted shading, duplicate the vertices with face normals.

The depth compare is fixed at `LESS_OR_EQUAL`, so coplanar edges win. For a wireframe over a mesh, make draw 0 the triangles (`TEST | WRITE`) and draw 1 the lines (`TEST` only).

The host premultiplies `mvp = proj * view * model`, and the normal matrix for `LAMBERT`. The sugar function `geom_draw_defaults()` sets an identity `model`, so you do not hit the all-zero case. The camera convention is +Y up and clip Z in `[0, 1]`, as in Vulkan, Metal, and D3D. It is not the GL range `[-1,1]`.

### Fixed-Function State

The render state is fixed, and both backends use the same values.

| State | Value |
|---|---|
| Depth compare | `LESS_OR_EQUAL` when testing, `ALWAYS` when only writing |
| Depth format | D32 float (`VK_FORMAT_D32_SFLOAT`, `MTLPixelFormatDepth32Float`) |
| Culling | None, two-sided |
| Line width | 1 px |

### Draw Gates

All gates run before any encoding. Any failure refuses the whole frame: the call returns `false`, pixels stay untouched, and the host writes one `caliper.log.v1` line. This includes the rigidity check for instanced `LAMBERT` draws.

| Scope | Gate |
|---|---|
| Per frame | Live view, `cam != NULL`, and `draw_stride >=` host minimum |
| Per draw | `topology`, `color_mode`, `shade_mode`, and `blend_mode` in range, and `reserved` zero |
| | `pos_alloc` live, `pos_offset % 4 == 0`, `vertex_count > 0`, and overflow-safe bounds |
| | Indexed (`index_alloc != 0`): alloc live, `index_offset % 4 == 0`, `index_count > 0`, and bounds-checked |
| | `LINE_*` needs at least 2 consumed vertices and `TRIANGLE*` needs at least 3, where consumed is `index_count` when indexed and `vertex_count` otherwise |
| | `LAMBERT` requires `normal_alloc != 0`, live, aligned, and bounds-checked |
| | `color_mode != FLAT` requires `attr_alloc != 0`, live, aligned, and bounds-checked, and `COLORMAP` also needs a resolvable `colormap` |
| | Any `depth_flags` bit against a depthless view |

### Index-Clamp Contract

Index values live in device memory, so the host cannot gate them. Both backends clamp in the vertex shader: `vi = min(index[i], vertex_count - 1)`. An out-of-range index gives a wrong-looking but defined image and never an out-of-bounds read. A bad index is a visual bug in your data and not a crash.

## `caliper.geometry.v1_2` Textured Meshes

`CaliperGeomDrawV1_2` uses the same `draw_primitives` and `draw_stride` slot as v1_1, and `reserved0` stays `NULL`. If caps bit 2 (`CALIPER_GEOM_CAP_TEXTURED`) is set, `COLOR_TEXTURE` draws are live. If the bit is absent, the textured path is inert and the draw behaves as in v1_1.

Calls through the v1.1 table accept a minimum stride of `192` and cannot request textured color. Calls through the v1.2 table need a minimum stride of `216`, and the host refuses a shorter stride.

### Texture Calls

<!-- c-api calls sdk/include/caliper/services/geometry_v1_2.h -->

### Texture Types

<!-- c-api types sdk/include/caliper/services/geometry_v1_2.h level=4 -->

### Texture Ids

<!-- c-api ids sdk/include/caliper/services/geometry_v1_2.h -->

### `COLOR_TEXTURE` Color Mode

`CALIPER_GEOM_COLOR_TEXTURE` is a fourth `color_mode` value. The draw pulls each UV pair with the same final vertex index as positions, normals, and attributes. The fragment samples the bridge texture in `texture` at that coordinate, and the sampled RGBA value is the draw color. `attr_alloc` is ignored under `COLOR_TEXTURE`.

The sampler is fixed, with no configurable materials or sampler state:

- Normalized coordinates.
- Linear minification and magnification.
- Clamp-to-edge on U and V, so UVs outside `[0, 1]` are defined and not wrapped.
- Base level only, with no mipmaps.

`SHADE_LAMBERT` multiplies the sampled RGB by the `0.30 + 0.70 * max(dot(n_vs, headlight), 0.0)` term and leaves alpha unchanged. Blend and depth modes then apply as in v1_1.

### Render-Target-View Refusal

`texture` names a bridge texture and never a geometry view. If it names any geometry view, including the view being drawn, the host refuses the whole frame. This prevents a feedback loop against the target.

### Texture Gates

`COLOR_TEXTURE` adds these gates to the v1_1 set. They run before any encoding, and any failure refuses the whole frame.

| Scope | Gate |
|---|---|
| Per draw | `uv_alloc` live, `uv_offset % 4 == 0`, and `uv_offset + vertex_count·2·sizeof(f32)` overflow-safe and in bounds |
| | `texture` resolves to a live tensor-bridge texture entry |

No gate checks UV values, because the clamp-to-edge sampler defines values outside `[0, 1]`.

### Fallback Ladder

If caps bit 2 is absent, `has_textured()` is false. The applet then falls back to per-vertex color (`COLORMAP` or `VERTEX_RGBA` over the same field) and says so. The `caliper::Geometry` sugar exposes `has_textured()` and a `draw_primitives` overload that takes `CaliperGeomDrawV1_2*`. A v1_2 record with a zero texture tail renders identically to the equivalent v1_1 draw.

## `caliper.geometry.v1_3` Instanced Transforms

`CaliperGeomDrawV1_3` uses the same `draw_primitives` and `draw_stride` slot as v1_2, and `reserved0` stays `NULL`. If caps bit 3 (`CALIPER_GEOM_CAP_INSTANCED`) is set, the instance tail is live. If the bit is absent, the tail is inert and the draw behaves as in v1_2.

### Instance Calls

<!-- c-api calls sdk/include/caliper/services/geometry_v1_3.h -->

### Instance Types

<!-- c-api types sdk/include/caliper/services/geometry_v1_3.h level=4 -->

### Instance Ids

<!-- c-api ids sdk/include/caliper/services/geometry_v1_3.h -->

One call draws one imported mesh N times. An imported `(N,16)` f32 column-major pose tensor supplies the model matrix of each instance. The vertex shader pulls it by instance index and applies it to the world position first, as `mvp · (M_i · v)`. An optional imported `(N,)` f32 per-instance scalar tints each instance through the draw's `colormap`, `vmin`, and `vmax`, with the same index rule as every other `COLORMAP` path. If `instance_attr_alloc == 0`, the base `color_mode` sets the color.

`LAMBERT` allows only rigid and uniform-scale instance transforms: rotation, uniform scale, and translation, with no shear and no non-uniform scale. Gate G14 refuses the whole frame if an instance matrix on a `LAMBERT` draw is outside this class. The tolerance is `CALIPER_GEOM_RIGID_TOL`. `UNLIT` instanced draws have no restriction.

If caps bit 3 is absent, `has_instanced()` is false. The applet then draws the unit alone or N plain draws, and says so. The `caliper::Geometry` sugar exposes `has_instanced()` and a `draw_primitives` overload that takes `CaliperGeomDrawV1_3*`. Seed descriptors from `caliper::geom_draw_v1_3_defaults()`, which gives a zero instance tail over the v1_2 defaults.

## C++ Sugar

`caliper::Geometry` (in `caliper.hpp`) wraps every geometry revision. Construct it from the `Host`. It is falsy when the host vends no geometry service, and every method null-guards, so it stays inert without the path. Use it on the frame thread only, as with `caliper::Bridge`.

```cpp
caliper::Geometry geometry(host);
uint32_t caps = geometry.caps();                 // 0 when absent
bool have_prims = geometry.has_primitives();     // caps bit 1

CaliperTextureId view = geometry.create_view_ex(w, h, CALIPER_GEOM_VIEW_DEPTH);
CaliperGeomDraw draws[3] = { /* … */ };
bool drew = geometry.draw_primitives(view, cam, draws, 3, 0xff05050au);
geometry.release_view(view);
```

`draw_primitives` passes `sizeof(CaliperGeomDraw)` as the stride for you.

Seed every descriptor from `caliper::geom_draw_defaults()`. It returns a zero-initialized `CaliperGeomDraw` with `flat_rgba = 0xffffffff`, `vmin/vmax = 0/1`, `size_px = 1`, and `model` set to identity. Set only the fields your draw needs.

### Torch Layout

- **positions**: `.contiguous()` f32 `(N, 3)`.
- **indices**: **int32** tensors with non-negative values. Torch has no uint32, and the bit pattern is what the u32 shader reads.
- **normals**: f32 `(N, 3)`.
- **packed colors** (`VERTEX_RGBA`, `flat_rgba`): int32 bit patterns.

## Learned Surface Example

A [`jobs.v1`](caliper-jobs-v1.md) worker writes the prediction of a small net over a grid into imported device tensors at each training step. The frame thread draws it as Lambert-lit triangles with a wireframe overlay. The worker drains the device once before it flips the ready slot, and the frame thread does every draw.

```cpp
// Frame thread. `geometry`, `bridge`, `pool` are set up at init; `draw_pos`,
// `draw_normal`, `draw_attr`, `draw_sample`, `tri_idx`, `line_idx` are the
// worker-published slot tensors snapshotted under the mutex.
const bool geom_live = geometry.has_primitives();

// Honest fallback ladder: no caps / no pool / no view / no surface -> CPU heatmap.
if (!geom_live || view == 0 || !pool || !draw_pos.defined()) {
    // …input-locked ImPlot heatmap of the same per-vertex error; never a blank
    // rectangle.
    return;
}

CaliperGeomCamera cam{};
look_at(eye, {0,0,0}, {0,1,0}, cam.view);               // applet-owned math
perspective(fovy, aspect, 0.05f, 50.f, cam.proj);

// Import-once per pool block (cached); resolves to (alloc, offset).
auto pref = pool->to_bridge(bridge, draw_pos);
auto nref = pool->to_bridge(bridge, draw_normal);
auto aref = pool->to_bridge(bridge, draw_attr);
auto tref = pool->to_bridge(bridge, tri_idx);
auto lref = pool->to_bridge(bridge, line_idx);
if (!(pref && nref && aref && tref && lref)) return;    // fall back

// Draw 0: the learned surface — indexed triangles, MAGMA over squared error,
// Lambert-lit from finite-difference normals, opaque, depth read+write.
CaliperGeomDraw surf = caliper::geom_draw_defaults();
surf.pos_alloc    = pref->alloc; surf.pos_offset    = pref->offset;
surf.vertex_count = vertex_count;
surf.index_alloc  = tref->alloc; surf.index_offset  = tref->offset;
surf.index_count  = tri_index_count;
surf.normal_alloc = nref->alloc; surf.normal_offset = nref->offset;
surf.attr_alloc   = aref->alloc; surf.attr_offset   = aref->offset;
surf.topology    = CALIPER_GEOM_TOPO_TRIANGLES;
surf.color_mode  = CALIPER_GEOM_COLOR_COLORMAP;
surf.shade_mode  = CALIPER_GEOM_SHADE_LAMBERT;
surf.blend_mode  = CALIPER_GEOM_BLEND_OPAQUE;
surf.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
surf.colormap    = CALIPER_CMAP_MAGMA;
surf.vmin = 0.0f; surf.vmax = color_vmax;

// Draw 1: coplanar wireframe overlay — indexed lines, flat white at low alpha,
// depth-TESTed only. LESS_OR_EQUAL lets the edges win over the surface.
CaliperGeomDraw wire = caliper::geom_draw_defaults();
wire.pos_alloc   = pref->alloc; wire.pos_offset   = pref->offset;
wire.vertex_count = vertex_count;
wire.index_alloc = lref->alloc; wire.index_offset = lref->offset;
wire.index_count = line_index_count;
wire.topology    = CALIPER_GEOM_TOPO_LINES;
wire.color_mode  = CALIPER_GEOM_COLOR_FLAT;
wire.blend_mode  = CALIPER_GEOM_BLEND_ALPHA;
wire.depth_flags = CALIPER_GEOM_DEPTH_TEST;
wire.flat_rgba   = 0x59ffffffu;                 // white, alpha ~0.35

CaliperGeomDraw draws[2] = { surf, wire };
bool drew = geometry.draw_primitives(view, cam, draws, 2, 0xff05050au);

if (drew)
    ImGui::Image(caliper::Bridge::imtex(view),
                 ImVec2(view_w / fb_scale, view_h / fb_scale));
```

Report "zero-copy (imported geometry)" in the status line only when `draw_primitives` drew this frame.

---

See also: [`caliper.tensor_bridge.v1`](caliper-tensor-bridge-v1.md) for the import machinery and [`caliper.jobs.v1`](caliper-jobs-v1.md) for the worker and frame threading.
