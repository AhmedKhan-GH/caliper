# geometry.v1_1 Phase B — the Vulkan backend (S4, Windows box)

**Date:** 2026-07-09
**Status:** approved (design), execution pending — requires the Windows/NVIDIA machine
**Design authority:** `GEOMETRY.md` §5 (Vulkan backend), §4 (semantics), §2.3 (gates) —
all CLOSED; this spec sequences the Windows work and pins the parity discipline.
**Governing docs:** `docs/m2a-windows-verification.md` (D24: how work is verified on
that box), `ZEROCOPY.md` (CUDA VMM import), `docs/superpowers/specs/
2026-07-09-geometry-v1_1-execution-plan.md` (S4 slot).
**Reference implementation:** the shipped Metal backend
(`src/host/renderer/metal_renderer.mm`, `kGeomShaderSrc` + `geom_draw_primitives`)
and its 13 §9.2 gfx rows — Vulkan mirrors Metal, not the spec alone.

## One-line

Implement `geom_create_view_ex` / `geom_draw_primitives` in
`src/host/renderer/vulkan_renderer.cpp` so the same applet binaries
(mesh_scope unchanged) draw zero-copy on Windows/NVIDIA, verified by
byte-exact mirrors of every Metal §9.2 row against the *same* CPU references.

## Deliverables (files)

| File | Change |
|---|---|
| `src/host/renderer/shaders/geom.vert`, `geom.frag` | NEW — GLSL twins of the Metal `kGeomShaderSrc` semantics (§ below). `points.vert/.frag` stay frozen. |
| `CMakeLists.txt` | Two more glslang steps (pattern at `CMakeLists.txt:232-252`): `geom.vert → geom_vert_spv.h` (`--vn kGeomVertSpv`), `geom.frag → geom_frag_spv.h`, `--target-env vulkan1.1`. |
| `src/host/renderer/vulkan_renderer.cpp` | `supports_geometry_primitives() → true`; `geom_create_view_ex`; `geom_draw_primitives`; depth render pass; pipeline cache; per-frame descriptor pool; params UBO ring. v1 members (`geom_pass_`, `geom_pipeline_`, points shaders) untouched. |
| `tests/gfx/gfx_main.cpp` | Vulkan §9.2 rows mirroring the 13 Metal rows, reusing the SAME file-scope CPU reference helpers and constants. |

## Shader semantics (pin these, byte-for-byte where stated)

One vertex/fragment pair, vertex-pulled exactly like the Metal geom shader:

- Pull by `gl_VertexIndex` from whole-bound SSBOs at element bases
  (`byte offset / 4`); indexed draws issued as **non-indexed** draws of
  `index_count` vertices; index clamp `vi = min(idx[base+i], vertex_count-1)`.
- COLORMAP index math byte-identical to v1/`map_f32_to_rgba8`:
  `t = (v==v && vmax>vmin) ? clamp((v-vmin)/(vmax-vmin),0,1) : 0`,
  `lut[uint(t*255+0.5)]`; FLAT unpacks `flat_rgba` LE; VERTEX_RGBA per-vertex u32.
- LAMBERT: `lit = 0.30 + 0.70*max(dot(normalize(nmat*n), vec3(0,0,1)), 0)`,
  applied to rgb only; `nmat` columns come premultiplied from the host (CPU,
  double precision) — no shader inverses.
- `gl_PointSize = size_px` written **unconditionally** — legal in Vulkan
  (consumed only by point topologies), so the Metal two-entry-point split is
  NOT needed. If validation layers complain on the box, fall back to the
  Metal-style split (two entry points, same body); semantics identical either way.
- FS is `out = in.color` — nothing else.

## Backend structure (per GEOMETRY.md §5, concretized)

1. **Views with depth** (`geom_create_view_ex(w, h, DEPTH)`): existing color
   image/view/descset unchanged (same texture table); plus per-view
   `VK_FORMAT_D32_SFLOAT` image + view, `DEPTH_STENCIL_ATTACHMENT`,
   device-local, never sampled. New render pass `geom_pass_depth_`
   (color: `loadOp CLEAR`, final `SHADER_READ_ONLY_OPTIMAL` — same as
   `geom_pass_`; depth: `CLEAR`/`storeOp DONT_CARE`). Track `has_depth` in
   `Tex`; `tex_release` destroys the depth image with the view. Plain
   `create_view` views remain depthless; depth_flags on them refuse (§2.3.7).
2. **Pipeline cache** `geom_prim_pipelines_`: key
   `(topology, blend_mode, depth_flags, has_depth_pass)` → `VkPipeline`,
   created lazily from the single geom vert/frag pair; one shared pipeline
   layout. Topology maps 1:1 (`POINT_LIST/LINE_LIST/LINE_STRIP/TRIANGLE_LIST/
   TRIANGLE_STRIP`, `primitiveRestartEnable = VK_FALSE`). Depth-stencil state:
   compare `LESS_OR_EQUAL` when testing, write per flag. Blend states exactly
   v1's table (OPAQUE off; ALPHA src/1-src + one/1-src alpha; ADDITIVE one/one).
   Negative-viewport y-flip and dynamic viewport+scissor exactly as
   `geom_pipeline_` does today.
3. **Descriptors & params** (§5.3): per-frame descriptor pool reset at each
   `draw_primitives`; one set per draw — bindings 0-3 SSBO pos/idx/nrm/attr
   (absent source → bind pos as harmless placeholder, shader never reads it),
   binding 4 LUT (as v1), binding 5 params as `UNIFORM_BUFFER_DYNAMIC`.
   **Params ring**: one HOST_VISIBLE|COHERENT buffer, 256-aligned slot per
   draw, grown ×2, written once per frame, bound via dynamic offset. Params
   block layout mirrors the Metal `PrimParams` (~160 B — exceeds the 128-B
   push budget, hence the ring; do NOT use push constants).
4. **Encoding order** (§5.4, gates-before-everything): ALL §2.3 gates → memory
   barriers over the union of referenced imported buffers
   (`ALL_COMMANDS → VERTEX_SHADER`, the `geom_draw_points` discipline) → begin
   `geom_pass[_depth_]` with clear values (depth 1.0) → per draw: bind
   pipeline, set + dynamic offset, `vkCmdDraw(consumed, 1, 0, 0)` → end pass.
   Timeline-semaphore pipelined ordering (D24/M2a) when live, drain otherwise.
   `last_device_path_ = "primitives-imported"` + one `dev_note`.
5. **Re-gating**: the renderer re-checks liveness/alignment/bounds against its
   own alloc table (defense in depth, as `geom_draw_points` does) — the bridge
   already gated at the ABI layer.

## Test mirrors (the parity contract)

Mirror ALL 13 Metal §9.2 rows in the Vulkan section of `gfx_main.cpp`:

- **Reuse the same file-scope CPU reference helpers and expected-value
  constants** the Metal rows use — parity means both backends are compared
  against ONE reference, never against each other's readbacks.
- Rows needing imported device buffers are **hardware-gated** with the
  existing `vk_cuda_ready()` pattern (UUID-paired single-GPU, primary ctx
  retained; skip with MESSAGE otherwise). Source buffers come from CUDA VMM
  allocations imported via bridge-v1.2, as the existing `points-imported`
  Vulkan tests do.
- Gate-refusal rows that need no live allocs (dead view, null cam, short
  stride, nonzero reserved) run portable (any Vulkan ICD, no CUDA).
- Byte-exact rows stay byte-exact; the two masked rows keep their masks
  (line endpoints; overlay edges); Lambert keeps ±2 LSB.
- v1 rows (`points-imported`, CPU-staged matrix) must remain green untouched.

## Acceptance (all by artifacts, per D24)

1. Build green on the Windows box; glslang steps produce the two new headers.
2. `caliper_gfx_tests`: all Vulkan v1_1 mirrors pass on real NVIDIA hardware
   (not skipped — the box is UUID-paired); all pre-existing rows still green.
3. Full suite 100% (all 6 suites).
4. mesh_scope run-proof: launch with `CALIPER_AUTOLAUNCH=dev.caliper.mesh-scope`;
   the one-shot provenance log line "first zero-copy frame drawn (imported
   geometry, 3 draws)" appears with the Vulkan renderer active;
   `last_device_path == "primitives-imported"`. Paint-the-target works (manual).
5. `CALIPER_RENDERER=gl` still falls back to the heatmap, honestly labeled.
6. ROADMAP.md §4 boxes checked; GEOMETRY.md status header → shipped both
   platforms; ZEROCOPY.md imported-geometry table gains the primitives row
   (this completes S5).

## Out of scope

No ABI changes (the header is frozen); no applet changes (mesh_scope runs
unchanged — that's the portability claim); GL backend untouched (stays inert
for primitives); no wide lines / MSAA / culling (GEOMETRY.md §1.2); no
MoltenVK; no Windows CI wiring beyond the existing D24 manual discipline.

## Execution notes for the Windows session

Single session, one branch (`feat/geometry-v1_1-vulkan`), phases: shaders+CMake
→ views/pipelines → descriptors/encode → portable gate rows → CUDA-gated rows
→ mesh_scope run-proof → docs. The Metal implementation is the map — when in
doubt, open `metal_renderer.mm` and mirror its structure; when the spec and
Metal disagree, STOP and flag it (do not silently pick one).
