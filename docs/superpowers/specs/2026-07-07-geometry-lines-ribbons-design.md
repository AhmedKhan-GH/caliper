# Design — `caliper.geometry.v1_1` ribbons + ThoughtSpace as a thread bundle

**Status:** approved design, ready for implementation planning (handoff spec —
the implementing agent is expected to have NO context beyond this file and the
pointers in §2). **Branch:** a child of `main` (geometry.v1 is merged at
`08b1848`). **Decided:** 2026-07-07 with the user: the ThoughtSpace point cloud
reads as noise; add a **line/ribbon primitive** and re-render the residual
stream as a legible bundle of flowing threads (chosen over manifold-surface
meshes, which are a heavier later increment).

## 1. What this is

Two coupled deliverables:

1. **Platform:** an additive geometry-service revision `caliper.geometry.v1_1`
   adding `draw_lines` — connected polylines rendered as camera-facing
   **ribbons** (pixel-width triangle strips), vertex-pulled from imported
   allocations, per-vertex colormapped, depth-tested for real 3-D occlusion.
2. **Applet:** GPTScope ThoughtSpace switches from 232k additive points to
   ~1–2k **ribbon threads** — one polyline per token, arcing through the 5
   depth stations, colored by loss/confidence/depth. Fewer primitives,
   connected, occluding: a shape you can read instead of a glowing haze.

The point path (`draw_points`) is unchanged and still used by flow_scope; the
two primitives coexist in one service.

## 2. Required reading for the implementing agent

- `sdk/include/caliper/services/geometry_v1.h` — the current service
  (`create_view`/`release_view`/`draw_points`, cap bit `IMPORTED_POINTS`).
  `draw_lines` is added the D24 way: a NEW struct `CaliperGeometryV1_1`,
  prefix-identical to `CaliperGeometryV1`, plus one member.
- `sdk/include/caliper/services/tensor_bridge_v1_2.h` and `..._v1_1.h` — the
  EXACT additive-revision pattern to copy (new struct, prefix-identical,
  `struct_size`, new cap bit). `tests/test_abi.cpp` shows how such a revision
  is layout-pinned; copy that test shape.
- `src/host/renderer/vulkan_renderer.cpp` — the point pipeline is the template:
  `ensure_geom_objects()` (render pass + POINT_LIST pipeline + descriptor set +
  LUT buffer), `geom_create_view` (offscreen RGBA8 view + framebuffer),
  `geom_draw_points` (fenced submit, vertex-pulled positions, push constants,
  `GeomPush`), `destroy_geom`. `shaders/points.vert` / `points.frag` and the
  CMake SPIR-V rules. The lines pipeline mirrors all of this.
- `src/host/renderer/host_renderer.h` — the defaulted-unsupported geometry
  virtuals (`supports_geometry`, `geom_create_view`, `geom_draw_points`); add
  siblings for lines.
- `src/host/tensor_bridge.{h,cpp}` — `geom_*` host bookkeeping (view table,
  gates, `to_bridge`-fed alloc ids). `geom_draw_lines` mirrors
  `geom_draw_points`'s gate structure.
- `src/host/host_services.cpp` — where `kGeom1` is built and vended; add
  `kGeom1_1`.
- `sdk/include/caliper/caliper.hpp` — the `caliper::Geometry` wrapper; add
  `draw_lines` null-guarded via the v1_1 table.
- `tests/gfx/gfx_main.cpp` — the geometry rows (`gfx/geometry:` cases) are the
  template for the line rows; note `VmmBlock`, `vmm_rows_ready()`, `Backend`,
  `geom_ref`, `ndc_for_pixel`.
- `applets/gpt_scope/thoughtspace.h` and the ThoughtSpace block in
  `applets/gpt_scope/gpt_scope.cpp` (the `"GPTScope: ThoughtSpace"` window,
  `publish_thoughtspace`, the pool slots) — what gets re-pointed at ribbons.
- `applets/flow_scope/flow_scope.cpp` — still uses `draw_points`; MUST keep
  working unchanged (a regression check).

## 3. Non-goals (v1)

- No variable-length polylines: every polyline in a `draw_lines` call has the
  SAME vertex count (uniform L). ThoughtSpace threads are all D vertices, so
  this suffices; an index/offset array for ragged polylines is a later revision.
- No tubes (extruded circles) or meshes/surfaces — ribbons only. Surfaces are
  the documented next increment.
- No line joins/caps beyond simple per-segment quads (miter joints are a polish
  increment; at these widths/segment counts the gaps are invisible).
- ThoughtSpace keeps ONE object (ribbons); no points/ribbons toggle in v1.
- No change to `draw_points`, its tests, flow_scope, or the bridge.

## 4. ABI — `caliper.geometry.v1_1`

Add to `geometry_v1.h` (do NOT modify `CaliperGeometryV1`):

```c
#define CALIPER_GEOMETRY_V1_1 "caliper.geometry.v1_1"
#define CALIPER_GEOM_CAP_IMPORTED_LINES (1u << 1)   // ribbons available

typedef struct CaliperGeometryV1_1 {
    /* --- prefix-identical to CaliperGeometryV1 --- */
    uint32_t struct_size;
    uint32_t (*caps)(void);
    CaliperTextureId (*create_view)(uint32_t width, uint32_t height);
    void (*release_view)(CaliperTextureId view);
    bool (*draw_points)(CaliperTextureId view, const CaliperGeomCamera* cam,
                        CaliperAllocId pos_alloc, uint64_t pos_offset,
                        uint64_t count,
                        CaliperAllocId attr_alloc, uint64_t attr_offset,
                        int32_t colormap, float vmin, float vmax,
                        float size_px, uint32_t clear_rgba);
    /* --- v1_1 addition --- */
    /* Draw `polyline_count` polylines, each `verts_per_polyline` vertices,
       laid contiguously in pos (polyline i = vertices [i*L, (i+1)*L), 3-float
       rows) as camera-facing ribbons of width_px pixels. Segments (L-1 per
       polyline) are depth-tested so nearer threads occlude farther ones.
       attr_alloc != 0 selects one f32 per vertex (same layout), colormapped
       per-vertex through the shared LUTs and Gouraud-interpolated along
       segments; attr_alloc == 0 draws flat white. Clears color+depth to
       clear_rgba first. Same gates as draw_points (live view/allocs, 4-byte
       offsets, overflow-safe bounds over polyline_count*verts_per_polyline*12
       bytes). L must be >= 2. false = view unchanged. */
    bool (*draw_lines)(CaliperTextureId view, const CaliperGeomCamera* cam,
                       CaliperAllocId pos_alloc, uint64_t pos_offset,
                       uint64_t polyline_count, uint64_t verts_per_polyline,
                       CaliperAllocId attr_alloc, uint64_t attr_offset,
                       int32_t colormap, float vmin, float vmax,
                       float width_px, uint32_t clear_rgba);
} CaliperGeometryV1_1;
```

`caliper::Geometry` gains `draw_lines(...)` guarded on the v1_1 table (falsy →
false), and `caps()` prefers v1_1's when present. Host vends BOTH `kGeom1`
and `kGeom1_1` (same backing `TensorBridge`), and adds `CALIPER_GEOMETRY_V1_1`
to `kIds` and the `service()` switch.

`test_abi.cpp`: a `geometry v1_1` case pinning prefix-identical offsets vs
`CaliperGeometryV1`, `draw_lines` appended contiguously, total size, the id
string, and `CALIPER_GEOM_CAP_IMPORTED_LINES == (1u<<1)`.

## 5. Host bookkeeping (`tensor_bridge.{h,cpp}`)

`geom_caps()` gains the LINES bit when the renderer reports line support
(`renderer_.supports_geometry_lines()` — a new virtual, default false; in
practice a Vulkan backend that supports points supports lines, so it can
return the same condition). `geom_draw_lines(...)` mirrors `geom_draw_points`:
resolve the view (must be a geometry view), gate `cam`/`width_px>0`, and when
`polyline_count>0 && verts_per_polyline>=2`: resolve pos (and attr) allocs,
4-byte-align offsets, overflow-safe bounds over `polyline_count *
verts_per_polyline` vertices (×12 bytes pos, ×4 attr), resolve the LUT, then
`renderer_.geom_draw_lines(...)`. Fail closed with a logged reason.

## 6. Renderer — the ribbon pipeline (Vulkan)

### 6.1 Views gain a depth attachment
`geom_create_view` additionally allocates a depth image
(`VK_FORMAT_D32_SFLOAT`, DEPTH_STENCIL_ATTACHMENT) + view, and the geometry
render pass gains a depth attachment (clear on load, don't-store). The point
pipeline sets `depthTestEnable=false, depthWriteEnable=false` (unchanged
behavior — points stay additive, order-independent). The line pipeline sets
`depthTestEnable=true, depthWriteEnable=true` (opaque, occluding). Both share
the render pass. Existing point gfx rows must stay byte-exact (depth present
but disabled for points → identical output); re-run them.

### 6.2 The lines pipeline (lazy, once)
`shaders/lines.vert` + `lines.frag`, compiled to SPIR-V headers by new
`add_custom_command` rules (copy the points rules). Topology
`TRIANGLE_LIST`; no vertex input; **no blend** (opaque); depth test+write on;
cull none. Draw call: `vkCmdDraw(cb, total_segments*6, 1, 0, 0)` where
`total_segments = polyline_count*(L-1)`.

`lines.vert` decodes `gl_VertexIndex`:
- `seg = vid / 6; corner = vid % 6;`
- `p = seg / (L-1); local = seg % (L-1);`  (L from push constant)
- endpoint indices `iA = p*L + local`, `iB = iA + 1`; pull `posA`, `posB`
  (3 floats each) from the storage buffer at `pos_base + 3*i`.
- project both by `mvp` → clip → NDC; compute screen-space segment direction
  (apply aspect = viewport.x/viewport.y), perpendicular `n = normalize(perp)`;
  offset each endpoint by `± n * (width_px / viewport.y)` in NDC (pixel→NDC).
- the 6 corners select {A-, A+, B-, B+} into two triangles; emit clip-space
  position (keep each endpoint's own clip.w for correct perspective) and the
  endpoint's color (so the fragment Gouraud-interpolates A↔B along the ribbon).
- color: `use_attr? LUT[idx(attr[attr_base+i], vmin, vmax)] : white`, same
  index rule as points (NaN→0, degenerate→0, `t*255+0.5` floored).

`GeomLinePush` (fits 128 B): `float mvp[16]; uint pos_base, attr_base,
use_attr, L; float vmin, vmax, width_px; float vp_x, vp_y;` = 64 + 40 = 104 B.

`destroy_geom` also tears down the line pipeline/pass-depth resources;
`destroy_tex` frees the depth image/view.

## 7. ThoughtSpace re-render (`gpt_scope.cpp` + `thoughtspace.h`)

- **Thread set:** subsample the S×T probe to `N_threads` (default 1536: e.g.
  every ⌈S·T / N_threads⌉-th token, deterministic). Each thread is the token's
  D residual stations — NO interpolated trail points (the ribbon segments ARE
  the interpolation). Buffer: pos `(N_threads*D, 3)` f32, attr `(N_threads*D,)`,
  row-major (thread, depth).
- **thoughtspace.h:** add `write_thread_positions(pos_out, resid_sub, basis,
  dims, raw, scale)` — project the subsampled `(N_threads, D, C)` residuals to
  `(N_threads*D, 3)`, no trails. Add `write_thread_attr_*` (per-vertex: loss
  broadcast along a thread's depths; confidence = per-station value; depth =
  d/(D-1)). The station/projection math is the existing `project`/normalization
  — reuse it; just drop the trail expansion. Keep the old point-layout
  functions (flow_scope-independent, but leave them for the record / tests).
- **Draw:** `geometry.draw_lines(view, &cam, pos.alloc, pos.offset, N_threads,
  D, attr.alloc, attr.offset, cmap, vmin, vmax, width_px≈2.5, clear)`.
  Toolbar gains a "width" slider (1–6). Color-mode combo, vmax, raw-norms,
  camera, pool/triple-buffer, honest status, and the CPU-subsample ImPlot3D
  fallback are UNCHANGED (fallback still scatters station points — acceptable).
- **Gen thread:** the sampled sequence's last-T tokens become T extra white-hot
  threads appended after the probe threads (count math: `polyline_count =
  N_threads (+ T when a sample exists)`; the gen region occupies the tail of
  the same slot buffers, sized for the max). Pin their attr to white.
- **Caps gate:** ThoughtSpace now needs `CALIPER_GEOM_CAP_IMPORTED_LINES`
  (fall back to the ImPlot3D scatter when absent, exactly as today).
- Status wording stays honest: "N threads — zero-copy (imported geometry)"
  only when `draw_lines` returned true this frame.

## 8. Verification (the definition of done)

- [ ] `test_abi` v1_1 layout case green; full unit suite green.
- [ ] Stub-renderer cases (mirror the point gates): caps-off inert; every
      draw_lines gate (unknown/misaligned/OOB pos & attr, L<2, unknown view,
      released alloc, null cam, width 0, count 0) returns false with pixels
      untouched.
- [ ] Hardware gfx rows (byte-exact where rasterization is deterministic):
      an axis-aligned width-1 ribbon between two pixel-center endpoints covers
      exactly the expected pixel run in the exact LUT color; a flat-white
      case; count-0 = pure clear; the negative gates keep prior pixels.
      (Angled/wide ribbons have AA edges — assert only the deterministic
      axis-aligned width-1 geometry + the gates, per the points-row precedent.)
- [ ] The EXISTING point gfx rows stay byte-exact after the depth-attachment
      change (points depth-disabled → identical output).
- [ ] Live (RTX 500 Ada): ThoughtSpace shows a legible ribbon bundle that
      occludes correctly (front threads hide back), reorganizes during
      training, switches color modes, and the width slider works; renderer
      logs a lines path ("lines path OK …"); green zero-copy status gated on
      the draw result. Screenshot early vs late training.
- [ ] flow_scope still renders its million points (regression) — the depth
      attachment and shared pass didn't disturb the point path.
- [ ] `CALIPER_RENDERER=gl`: ThoughtSpace + flow_scope fall back cleanly.
- [ ] Clean mid-run teardown (both applets), full ctest 3/3 green.

## 9. Risks & notes

- **Ribbon expansion math** (screen-space perpendicular with perspective) is
  the tricky part; the mitigations are the well-trodden "instanced wide line"
  technique and the deterministic axis-aligned test row that pins it.
- **Depth attachment touches the shared render pass** — the one place this can
  regress the verified point path; the point pipeline must explicitly disable
  depth test+write, and the existing point rows are the guard.
- **Additive vs opaque:** ribbons are opaque+depth (occlusion = legibility);
  points stay additive+no-depth. Do not unify them.
- **Thread count** is a constant (default 1536); if the bundle is too sparse
  or dense, it's a one-line tune, not a redesign.
- **Two push-constant structs** (points 88 B, lines 104 B) — keep them distinct
  and static_assert both sizes.

## 10. Future increments (out of scope, listed to keep them out of v1)

Variable-length polylines (offset array); miter/round joins and caps; tubes;
manifold-surface meshes (the option-2 object); a points/ribbons toggle in
ThoughtSpace; per-segment attn-vs-MLP decomposition of each thread (the data is
in `forward_full`); learned/PCA projection with smooth basis interpolation.
