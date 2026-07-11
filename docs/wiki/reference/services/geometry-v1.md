# caliper.geometry.v1 / v1_1 / v1_2 / v1_3

Service ids `caliper.geometry.v1` (instanced points) and its additive revisions
`caliper.geometry.v1_1` (general primitives), `caliper.geometry.v1_2`
(textures on meshes), and `caliper.geometry.v1_3`
(instanced transforms). Imported 3-D geometry: an applet
writes vertices/indices/normals/attributes into device memory, exports them once
through [`caliper.tensor_bridge.v1_2`](tensor-bridge-v1.md), and the host draws
them **in place** — zero copies of the geometry data — into an offscreen view
texture you show with `ImGui::Image`. This page embeds the headers verbatim; the
docs build fails if any embedded file moves.

Deliberately a **new** service, not a tensor-bridge revision: the bridge's frozen
identity is "a tensor becomes an image"; cameras and draw calls are a different
vocabulary. The two share id spaces on purpose — a view is a `CaliperTextureId`
in the same table the bridge uses (drawable with `ImGui::Image` like any texture),
and geometry sources are addressed as `(CaliperAllocId, byte offset)`, reusing the
v1.2 import machinery, caches, gates, and lifecycle as-is.

!!! info "Platform status (honest, stated once)"
    **Points (`v1`):** Metal + Vulkan + GL-fallback ladder. **Primitives
    (`v1_1`):** shipped on **both** backends (Metal and Vulkan), byte-exact
    against one CPU reference on real hardware on each. **Textures on meshes
    (`v1_2`):** **run-proven byte-exact on both backends** — Vulkan/RTX 500 Ada
    and Metal/Apple Silicon (both 2026-07-10, the macOS hardware pass), against
    the same shared CPU reference. **Instanced transforms
    (`v1_3`):** **run-proven byte-exact on both backends** — Metal on Apple
    Silicon and Vulkan/CUDA on Windows (both 2026-07-11), against the same
    shared CPU reference. On a backend without
    the path (the GL fallback, or a host that doesn't vend the revision) the
    matching **caps bit is unset and every
    entry point is inert** — ship your fallback (see the worked example
    below). This is the degradation ladder, not a bug: absent capability →
    CPU path, never a wrong image.

```c
--8<-- "sdk/include/caliper/services/geometry_v1.h"
```

```c
--8<-- "sdk/include/caliper/services/geometry_v1_1.h"
```

```c
--8<-- "sdk/include/caliper/services/geometry_v1_2.h"
```

```c
--8<-- "sdk/include/caliper/services/geometry_v1_3.h"
```

## caliper.geometry.v1 — instanced points

Caps bit 0 (`CALIPER_GEOM_CAP_IMPORTED_POINTS`) set means `create_view` /
`draw_points` are live. Absent bit → both are inert, the applet keeps its CPU
fallback.

### The offscreen-view pattern

1. `create_view(width, height)` returns a `CaliperTextureId` in the tensor-bridge
   texture table. `0` = failure. Sizes are **physical (framebuffer) pixels**;
   recreate the view when the content region changes by more than a few pixels.
2. Each frame, `draw_points(view, cam, …)` renders **one** frame of the view
   atomically: clear to `clear_rgba` (packed little-endian `r | g<<8 | b<<16 |
   a<<24`), then draw `count` points whose positions are a contiguous `(count,3)`
   f32 array at `pos_offset` inside the imported allocation `pos_alloc`.
   `count == 0` is a pure clear.
3. `caliper::Bridge::imtex(view)` casts the id to `ImTextureID`; display it with
   `ImGui::Image` at **logical** size (`physical / DisplayFramebufferScale`) so one
   texel maps to one framebuffer pixel.
4. `release_view(view)` frees it — release it **here**, on the frame thread, not
   through the bridge.

`attr_alloc != 0` selects a contiguous `(count,)` f32 scalar per point at
`attr_offset`, colormapped through the shared LUTs over `[vmin, vmax]` (same index
rule as `texture_from_tensor_mapped`, see [tensor-bridge](tensor-bridge-v1.md));
`attr_alloc == 0` draws flat white and ignores `attr_offset`/`colormap`. `size_px`
is point size in pixels (clamped to device limits). Points blend **additively with
no depth test** — v1 is built for particle clouds: order-independent, no sort.

`cam` is a `CaliperGeomCamera` — column-major 4×4 `view` and `proj`, **applet-owned
math**. The service does no camera logic; orbit/zoom/ray-casting are UI and live in
the applet.

### Memory-stability contract

The addressed bytes are read **in place** and must not be rewritten until this
view's next draw. On the worker side, drain the producer stream/queue once at the
handoff **before** publishing the slot (this is the triple-buffer / ready-slot
discipline the exemplars use with [`caliper.jobs.v1`](jobs-v1.md)); the frame
thread does every draw and never touches the learner's tensors.

## caliper.geometry.v1_1 — general primitives

Additive revision of v1: the first five members are prefix-identical
(`struct_size`, `caps`, `create_view`, `release_view`, `draw_points`); three
members are appended (`create_view_ex`, `draw_primitives`, a reserved slot). No ABI
epoch bump, v1 untouched and frozen.

Caps bit 1 (`CALIPER_GEOM_CAP_PRIMITIVES`) set means `create_view_ex` /
`draw_primitives` are live. It implies nothing about bit 0; hosts set both when the
primitives path exists.

### Depth views

`create_view_ex(width, height, flags)` is `create_view` plus flags. The one flag,
`CALIPER_GEOM_VIEW_DEPTH`, attaches a D32-float depth buffer (renderer-internal —
never sampled, never an id). Same texture table, released via `release_view`. A
plain v1 view has no depth: setting any `depth_flags` bit against it is
**refused**, never silently ignored (degradation ladder — never a silently-wrong
image).

### One atomic gated frame

`draw_primitives(view, cam, draws, draw_count, draw_stride, clear_rgba)` renders
**one** frame of the view, atomically:

1. **Gate every draw first** (see below).
2. Clear color to `clear_rgba` (and depth to `1.0` when the view has depth).
3. Encode `draws[0..draw_count)` in **array order** into a single render pass.
   Later draws paint over earlier ones subject to their own depth/blend state.

`draw_count == 0` is a pure clear. If any gate fails, **nothing is drawn or
cleared** — prior pixels intact — and the host emits one `caliper.log.v1` line
naming the gate. This is why the API is one atomic call with a descriptor array
rather than a stateful begin/end: a begin/end can fail *after* the clear, which
would violate pixels-untouched.

`draw_stride` is the applet's compiled `sizeof(CaliperGeomDraw)`, so the struct can
grow additively later; the host reads `min(stride, its own sizeof)`. A stride below
the host's known minimum is refused. The [C++ sugar](#c-sugar) passes `sizeof`
for you.

### The `CaliperGeomDraw` descriptor

One fixed-layout struct per draw call. `static_assert(sizeof == 192)` in every
consumer — 8-byte fields first, no packing surprises.

**Geometry sources** — each an `(alloc, byte offset)` pair into a bridge-v1.2
import; alloc id `0` = source absent; all offsets 4-byte aligned:

| Field(s) | Meaning |
|---|---|
| `pos_alloc`, `pos_offset` | positions: contiguous `(vertex_count, 3)` f32 |
| `vertex_count` | vertex count (must be `> 0`) |
| `index_alloc`, `index_offset`, `index_count` | indices: `(index_count,)` **u32** bit patterns; consumed only when `index_alloc != 0` |
| `normal_alloc`, `normal_offset` | normals: `(vertex_count, 3)` f32; required for `LAMBERT` |
| `attr_alloc`, `attr_offset` | per-vertex attribute; meaning set by `color_mode` |

**`topology`** — how vertices assemble:

| Constant | |
|---|---|
| `CALIPER_GEOM_TOPO_POINTS` | `0u` |
| `CALIPER_GEOM_TOPO_LINES` | `1u` |
| `CALIPER_GEOM_TOPO_LINE_STRIP` | `2u` |
| `CALIPER_GEOM_TOPO_TRIANGLES` | `3u` |
| `CALIPER_GEOM_TOPO_TRIANGLE_STRIP` | `4u` |

**`color_mode`** — where each vertex's colour comes from:

| Constant | `attr` layout | Colour |
|---|---|---|
| `CALIPER_GEOM_COLOR_FLAT` (`0u`) | (none) | `flat_rgba` for every vertex, unpacked LE bytes `/255` |
| `CALIPER_GEOM_COLOR_COLORMAP` (`1u`) | `(vertex_count,)` f32 | `attr` value through the LUT `colormap` over `[vmin, vmax]` |
| `CALIPER_GEOM_COLOR_VERTEX_RGBA` (`2u`) | `(vertex_count,)` u32 | packed LE `r\|g<<8\|b<<16\|a<<24` per vertex, unpacked `/255` |

The `COLORMAP` index rule is **byte-identical** to the tensor-bridge's
`texture_from_tensor_mapped` / `map_f32_to_rgba8`:

```
idx = uint(clamp((v - vmin) / (vmax - vmin), 0, 1) * 255.0 + 0.5)
```

with `NaN → 0`, a degenerate `vmin == vmax` range → `0`, and the shared 256-entry
table. `colormap` is a bridge LUT id (`CALIPER_CMAP_VIRIDIS` = 0, `_MAGMA` = 1,
`_RDBU` = 2).

**`shade_mode`**:

| Constant | |
|---|---|
| `CALIPER_GEOM_SHADE_UNLIT` (`0u`) | colour as computed above |
| `CALIPER_GEOM_SHADE_LAMBERT` (`1u`) | single headlight, Gouraud; **requires `normal_alloc != 0`** |

`LAMBERT` is a single **headlight** — view-space light direction `(0,0,1)` — shaded
per-vertex and interpolated:

```glsl
n_vs = normalize(nmat * n);                        // nmat: host-computed 3x3
lit  = 0.30 + 0.70 * max(dot(n_vs, vec3(0,0,1)), 0.0);
color.rgb *= lit;                                  // alpha untouched
```

`nmat = transpose(inverse(upper3x3(view * model)))`, computed on the CPU per draw —
no shader inverses. The `0.30` ambient is a spec constant, not a parameter. Faceted
shading = the applet duplicates vertices with face normals (applet choice, zero
host work); per-pixel (Phong) lighting is not offered.

**`blend_mode`**:

| Constant | Equation |
|---|---|
| `CALIPER_GEOM_BLEND_OPAQUE` (`0u`) | blending disabled |
| `CALIPER_GEOM_BLEND_ALPHA` (`1u`) | `SRC_ALPHA / ONE_MINUS_SRC_ALPHA` |
| `CALIPER_GEOM_BLEND_ADDITIVE` (`2u`) | `ONE / ONE`, both channels (byte-identical to the v1 points look) |

Transparency is **not** sorted — the applet orders its draw array; unsorted alpha
artifacts are documented, not fixed.

**`depth_flags`** (bits) — valid only on a `CALIPER_GEOM_VIEW_DEPTH` view:

| Constant | |
|---|---|
| `CALIPER_GEOM_DEPTH_TEST` (`1u << 0`) | depth test on (`LESS_OR_EQUAL` compare) |
| `CALIPER_GEOM_DEPTH_WRITE` (`1u << 1`) | write depth |

The `LESS_OR_EQUAL` compare is fixed; it lets coplanar edges win, so a
wireframe-over-mesh overlay is draw 0 = triangles (`TEST | WRITE`), draw 1 = lines
(`TEST` only).

**Remaining scalar/state fields:** `flat_rgba` (packed LE, used when `COLOR_FLAT`),
`colormap` / `vmin` / `vmax` (used when `COLOR_COLORMAP`), `size_px` (point size;
ignored for non-point topologies), and `reserved[2]` which **must be zero**
(nonzero is refused).

**`model[16]`** — per-draw column-major model transform. Applets pass an **identity
matrix** for world-space vertices; `mvp = proj * view * model` and (for `LAMBERT`)
the normal matrix are premultiplied host-side. An **all-zero `model` renders
nothing** — the classic footgun; the sugar's `geom_draw_defaults()` sets identity
so you never hit it.

Clip-space convention (applet-owned camera math): **+Y up, clip Z in `[0, 1]`**
(Vulkan/Metal/D3D, not GL's `[-1,1]`).

### Gates and refusals

All gates run **before any encoding** (on Metal, before the encoder exists —
creating it performs the clear). Any failure refuses the **whole frame**: `false`,
pixels untouched, one `caliper.log.v1` line.

| Scope | Gate |
|---|---|
| Per frame | live view; `cam != NULL`; `draw_stride >=` host minimum |
| Per draw | `topology` / `color_mode` / `shade_mode` / `blend_mode` in range; `reserved` zero |
| | `pos_alloc` live; `pos_offset % 4 == 0`; `vertex_count > 0`; overflow-safe bounds |
| | indexed (`index_alloc != 0`): alloc live, `index_offset % 4 == 0`, `index_count > 0`, bounds-checked |
| | `LINE_*` need ≥2 consumed vertices, `TRIANGLE*` need ≥3 (consumed = `index_count` when indexed, else `vertex_count`) |
| | `LAMBERT` requires `normal_alloc != 0`, live, aligned, bounds-checked |
| | `color_mode != FLAT` requires `attr_alloc != 0`, live, aligned, bounds-checked, and a resolvable `colormap` (COLORMAP) |
| | any `depth_flags` bit against a depthless view → refuse |

### The index-clamp contract

Index **values** live in device memory and cannot be gated host-side. Both
backends' vertex shaders clamp: `vi = min(index[i], vertex_count - 1)`. An
out-of-range index therefore produces a **wrong-looking but defined** image — never
an out-of-bounds read or UB. This is the applet's contract: a bad index is a visual
bug in your data, not a crash.

## caliper.geometry.v1_2 — textures on meshes

Additive revision of v1_1: `CaliperGeomDrawV1_2` = the frozen **192-byte**
`CaliperGeomDraw` record (`base`) plus an appended **texture tail**
(`uv_alloc`, `uv_offset`, `texture`), carried by the same `draw_primitives` +
`draw_stride` slot. No new entry point, `reserved0` still NULL, the v1.1 prefix
untouched and frozen. `static_assert(sizeof(CaliperGeomDrawV1_2) == 216)` and
`offsetof(uv_alloc) == 192` pin the tail. Caps bit **2**
(`CALIPER_GEOM_CAP_TEXTURED`, `1u << 2`) set means `COLOR_TEXTURE` draws are
live; absent → the textured path is inert and the draw behaves exactly as v1_1.

The host vends both revisions side by side: calls through the v1.1 table accept
a minimum stride of `192` and cannot request textured colour; calls through the
v1.2 table accept a minimum stride of `216`. Old binaries keep working while a
v1.2 caller cannot expose an absent tail (short stride → refused).

### The `COLOR_TEXTURE` color mode

A fourth `color_mode` value, `CALIPER_GEOM_COLOR_TEXTURE` (`3u`), extends the
v1_1 `FLAT` / `COLORMAP` / `VERTEX_RGBA` set. Each vertex carries a
`(vertex_count, 2)` f32 UV pair at `(uv_alloc, uv_offset)` — another
`(alloc, byte offset)` into a bridge-v1.2 import, pulled by the **same** final
vertex index as positions/normals/attributes — and the fragment samples a
**bridge `CaliperTextureId`** (`texture`) at that coordinate. The sampled RGBA
value is the draw colour. `attr_alloc` is ignored under `COLOR_TEXTURE` and
need not be present.

Sampling is a **fixed** sampler — no configurable materials or sampler state:

- normalized coordinates;
- bilinear minification and magnification (**linear**);
- **clamp-to-edge** on U and V (UVs outside `[0, 1]` are defined, not wrapped);
- base level only, **no mipmaps**.

`SHADE_LAMBERT` multiplies the sampled RGB by the existing
`0.30 + 0.70 * max(dot(n_vs, headlight), 0.0)` headlight term and leaves alpha
unchanged; blend and depth modes then apply exactly as in v1_1.

### The render-target-view refusal

`texture` names a **bridge texture**, never a geometry view. Naming any geometry
view — including the view currently being drawn — is **refused whole-frame**
(pixels untouched, one `caliper.log.v1` line), not silently sampled. A geometry
view is never sampleable by construction, which forecloses a read-after-write
feedback loop against the target you are painting. This is the degradation
ladder's usual shape: a wrong request refuses, never renders a wrong image.

### Gates and refusals

`COLOR_TEXTURE` adds these to the v1_1 gate set (all before any encoding; any
failure refuses the **whole frame**):

| Scope | Gate |
|---|---|
| Per draw | `uv_alloc` live; `uv_offset % 4 == 0`; `uv_offset + vertex_count·2·sizeof(f32)` overflow-safe and in-bounds |
| | `texture` resolves to a live tensor-bridge texture entry |
| | `texture` is **not** a geometry view (including the target view) → refuse |

UV **values** are finite shader inputs like any other imported bytes; values
outside `[0, 1]` are defined by the clamp-to-edge sampler, not a gate.

### Honest ladder

When caps bit 2 is absent (the GL fallback, or a host vending only v1/v1_1),
`has_textured()` is false and the textured path is inert — the applet ladders
down to a per-vertex colour fallback (`COLORMAP` or `VERTEX_RGBA` over the same
field, the field sampled at vertex resolution instead of texture resolution) and
says so; never a wrong image, never a false status line. The `caliper::Geometry`
sugar exposes `has_textured()` (caps bit 2) and a `draw_primitives` overload
taking `CaliperGeomDrawV1_2*`; seed descriptors so the texture tail is zero over
the v1_1 defaults (a zero-tail v1_2 record renders byte-identically to the
equivalent v1_1 draw). **TwinScope's surface twin** — a learned thermal field
draped on a heatsink housing at texture resolution — is the exemplar: the twin's
*state painted on its shape*.

## caliper.geometry.v1_3 — instanced transforms

Additive revision of v1_1/v1_2: `CaliperGeomDrawV1_3` = the frozen 216-byte
`CaliperGeomDrawV1_2` record (`base`) plus an appended **instance tail**, carried
by the same `draw_primitives` + `draw_stride` slot. No new entry point, `reserved0`
still NULL, the v1.1/v1.2 prefixes untouched. Caps bit **3**
(`CALIPER_GEOM_CAP_INSTANCED`, `1u << 3`) set means the instance tail is live;
absent → the tail is inert and the draw behaves exactly as v1_2.

One imported mesh drawn N times in **one call**: an imported `(N,16)` f32
column-major pose tensor (`instance_alloc`/`instance_offset`, `instance_count = N`)
supplies each instance's model matrix, pulled by instance index in the vertex
shader and applied to the world position **first** (`mvp · (M_i · v)`). An optional
imported `(N,)` f32 per-instance scalar (`instance_attr_alloc`/`instance_attr_offset`)
tints each unit through the draw's existing `colormap`/`vmin`/`vmax` LUT — same
index rule as every other COLORMAP path on this page. `instance_attr_alloc == 0`
leaves coloring to the base `color_mode`.

**Additive default (zero tail == a v1_2 draw).** `instance_count == 0` **or**
`instance_alloc == 0` takes the exact non-instanced path — same shader code path,
same draw arity — so a v1_3 record with a zero instance tail is byte-identical at
the pixel level to the equivalent v1_2 record. Widening a v1_1/v1_2 record up to
v1_3 (zero tail) draws the same pixels.

**LAMBERT is restricted to rigid + uniform-scale instance transforms**
(rotation, uniform scale, translation — no shear, no non-uniform scale). Under that
class the correct normal map is the instance rotation up to a scalar `normalize`
removes, so the shader composes the instance upper-3×3 with the per-draw normal
matrix and stays a ±2-LSB tolerance row. Out-of-class instance matrices on a
LAMBERT-instanced draw are **refused whole-frame** (never silently mis-lit) by gate
**G14**, whose tolerance is pinned in the header as `CALIPER_GEOM_RIGID_TOL`
(`1e-4f`) — part of the byte-exact contract, not a tunable. UNLIT instanced draws
never touch normals and are unrestricted.

**Honest ladder.** When caps bit 3 is absent (GL fallback, or a host vending only
v1/v1_1/v1_2), `has_instanced()` is false and the instanced path is inert — the
applet ladders down to a non-instanced fallback (draw the hero unit alone, or N
plain draws) and says so; never a wrong image, never a false status line. The
`caliper::Geometry` sugar exposes `has_instanced()` (caps bit 3) and a
`draw_primitives` overload taking `CaliperGeomDrawV1_3*`; seed descriptors from
`caliper::geom_draw_v1_3_defaults()` (a zero instance tail over the v1_2 defaults).

## C++ sugar

`caliper::Geometry` (in `caliper.hpp`) wraps every geometry revision. Construct it
from the
`Host`; it is falsy when the host vends no geometry service, and every method
null-guards so it stays inert on hosts without the path. **Frame-thread only**, same
as `caliper::Bridge`.

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

Always seed a descriptor from **`caliper::geom_draw_defaults()`** — it returns a
zero-initialised `CaliperGeomDraw` with `flat_rgba = 0xffffffff`, `vmin/vmax =
0/1`, `size_px = 1`, and `model` = identity, removing the all-zero-model footgun at
the source. Set only the fields your draw needs.

### Torch layout notes

- **positions**: `.contiguous()` f32 `(N, 3)`.
- **indices**: **int32** tensors with non-negative values — torch has no uint32;
  the bit pattern is exactly what the u32 shader reads.
- **normals**: f32 `(N, 3)`.
- **packed colours** (`VERTEX_RGBA`, `flat_rgba`): int32 bit patterns.

## Worked example — a learned surface, three draws

Modelled on `applets/mesh_scope` (the v1_1 exemplar): a small net's prediction over
a grid is written into imported device tensors each training step by a
[`jobs.v1`](jobs-v1.md) worker and drawn the **same frame** as Lambert-lit
triangles + a wireframe overlay + the training minibatch as additive points. The
worker publishes into triple-buffered slots and drains the device once before
flipping the ready slot; the frame thread snapshots the display slot under one
mutex and does every draw.

```cpp
// Frame thread. `geometry`, `bridge`, `pool` are set up at init; `draw_pos`,
// `draw_normal`, `draw_attr`, `draw_sample`, `tri_idx`, `line_idx` are the
// worker-published slot tensors snapshotted under the mutex.
const bool geom_live = geometry.has_primitives();

// Honest fallback ladder: no caps / no pool / no view / no surface -> CPU heatmap.
if (!geom_live || view == 0 || !pool || !draw_pos.defined()) {
    // …input-locked ImPlot heatmap of the same per-vertex error; never a blank
    // rectangle. See mesh_scope.cpp for the full fallback.
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

The status line reports "zero-copy (imported geometry)" **only when
`draw_primitives` actually drew this frame** — the honest-provenance discipline the
exemplars follow verbatim; on a non-primitives backend `has_primitives()` is false
and the CPU fallback runs instead.

---

See also: [`caliper.tensor_bridge.v1`](tensor-bridge-v1.md) for the import
machinery geometry draws from, and [`caliper.jobs.v1`](jobs-v1.md) for the
worker/frame threading the exemplars use to feed it.
