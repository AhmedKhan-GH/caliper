# TwinScope — a trained thermal twin, watched live (the R2/R3 forcing function)

**Date:** 2026-07-10
**Status:** approved (design). Scope A: this doc designs the exemplar and
**derives** the R2/R3 requirement contracts; R2 and R3 each still get their
own full spec → implement → verify pass (the v1_1 discipline verbatim)
before any backend code.
**Design authority upstream:** `GEOMETRY.md` §11 (R2/R3 rows, the no-new-ABI
list), §1.1/§1.2 (invariants and non-goals — binding here), `ROADMAP.md` §6.
**Exemplar precedent:** `applets/mesh_scope` (learned surface, paint-the-target),
`applets/flow_scope` (triple-buffered pool slots, applet-side ray casting),
`applets/field_scope` (torch-ops physics solver).
**Brainstormed:** 2026-07-10; scope A, field story A (sim ground truth +
live-learning net), mesh A (committed OBJ + minimal loader), fleet A (pose +
per-instance scalar) — recommendations executed per Ahmed's standing
preference.

## One-line

A finned electronics housing whose live-simulated heat field a small net
learns while you watch — the field draped on the housing's skin at texture
resolution (R2), and a 50-unit fleet of boundary-condition variants each
tinted by its own state (R3) — the concrete applet need that R2 and R3 are
built against, never on spec.

## Why this exemplar (what it forces)

GEOMETRY.md §11 gates R2 (textures-on-meshes) and R3 (instanced transforms)
on *demonstrated applet need*. TwinScope is that demonstration, chosen so
each revision carries exactly one load:

- The **hero unit** needs the field at a resolution the mesh doesn't have —
  ~500 vertices cannot carry a 256×256 temperature field per-vertex. That is
  R2's decoupling (shape loaded once; state re-draped every step at field
  resolution), and nothing short of sampled textures provides it.
- The **fleet** needs 50 units in one draw with per-unit pose AND per-unit
  state. That is R3's instanced transforms — and it surfaces the one R3
  design question a pose-only demo would hide: per-instance attributes.

## 1. The applet: `applets/twin_scope`

### 1.1 The subject

A finned electronics housing: a committed OBJ asset
(`applets/twin_scope/assets/housing.obj`, ~500 triangles, with `vt` UVs and
`vn` normals). We generate the asset ourselves (procedural finned box,
UV-unwrapped programmatically, exported once) — no licensing questions, and
the generator script is NOT shipped; the OBJ is the artifact.

### 1.2 The physics (ground truth)

A synthetic heat-diffusion solver in pure torch ops (the field_scope
precedent), operating on the housing's **UV texture space**:

- State: `(B, H, W)` f32 temperature grid, `B = 50` boundary-condition
  variants, `H = W = 256` (tunable), solved as ONE batched step —
  explicit finite-difference diffusion + source injection + boundary loss,
  a handful of torch ops per step.
- Sources: K fixed sites in UV space (the bolt holes + core), per-variant
  intensity vector `(B, K)` — variant 0 is the hero unit. Sources toggle /
  scale live (§1.5).
- The solver runs on the training device (CUDA / MPS), on the jobs.v1
  worker — never the frame thread (the frame-thread discipline is absolute;
  see the embed_scope postmortem).

Physical honesty note: this is a *plausible* diffusion field for
visualization, not a validated thermal model — the doc and the UI say
"synthetic heat field," never "FEA." The twin claim is about the
*dataflow*, not thermal engineering.

### 1.3 The learner

A small MLP `f_θ(u, v, s_1..s_K) → T` (UV position + that variant's source
intensities → temperature), trained every step on random samples across ALL
50 variants — one batched forward/backward. The net is deliberately small
(mesh_scope scale): convergence in ~a minute, visible improvement in
seconds.

Displayed quantities (hero toggle): **sim field** (ground truth), **net
prediction** (rendered by evaluating f_θ on the same `(H, W)` UV grid), and
**|error|** — all three are `(H, W)` f32 tensors through the same path.

### 1.4 The two views (what each revision draws)

- **Hero (R2):** the housing mesh, `COLOR_TEXTURE` draw: the selected
  `(H, W)` field tensor updates a bridge texture zero-copy (the shipped
  tensor_bridge pool path — unchanged), and geometry.v1_2 samples that
  texture on the mesh via the OBJ's UVs. Lambert-lit, wireframe overlay
  optional (the v1_1 coplanar pattern). One draw.
- **Fleet (R3):** the same mesh, one instanced draw: imported `(50, 16)` f32
  column-major model matrices (grid layout, hero at the front), plus
  imported `(50,)` f32 per-instance attr — each unit's scalar summary (peak
  T or peak |error|, a toggle) through the existing LUT. Depth on, OPAQUE.
- Camera: applet-owned orbit (mesh_scope math reused). Both views in one
  geometry view texture; ImGui composites the toolbar/HUD.

### 1.5 Interaction (zero ABI, the §11 list held)

Click a heat source on the hero → toggle it; drag → scale its intensity
(variant 0 only). Applet-side ray cast against its own tensors (flow_scope
pattern). The sim responds next step; the net's prediction visibly lags and
chases — the paint-the-target drama generalized to 3-D. Stroke/click events
queue to the worker; torch never runs on the frame thread.

### 1.6 Data & buffers (all existing machinery)

- Mesh: pos/normal/uv/index tensors filled by the OBJ loader (§2), moved to
  the training device, exported once via the ExportablePool → bridge-v1.2
  `import_allocation`. Static thereafter.
- Field textures: `(H, W)` f32 → existing bridge texture with LUT (MAGMA),
  updated zero-copy from the pool per publish (the gpt_scope heatmap path).
- Fleet transforms + attrs: pool-born tensors, triple-buffered slots
  (flow_scope contract) — the worker writes slot k+1 while the frame reads
  slot k; the memory-stability contract holds per view.
- Publish cadence: worker publishes at display cadence (~30 Hz cap), sim
  steps uncapped — the metrics/jobs pattern.

### 1.7 Degradation ladder (honest at every rung)

| Missing | Behavior | Status line says |
|---|---|---|
| R2 cap (bit 2) | hero falls back to per-vertex COLORMAP: field sampled at mesh vertices (torch grid_sample, worker-side), drawn via v1_1 | "per-vertex fallback (no textured geometry)" |
| R3 cap (bit 3) | hero only; fleet panel explains | "fleet needs instanced geometry (cap absent)" |
| primitives cap (bit 1) | CPU ImPlot heatmap of the field, hero only | mesh_scope's ladder verbatim |
| import/pool | v1 CPU-staged texture updates still work for the heatmap rung | never a wrong image |

Zero-copy is claimed only when the path drew (flow_scope discipline).

## 2. The OBJ loader (no ABI growth — §11's list is binding)

`sdk/include/caliper/adapters/obj.hpp`, header-only, applet-side:

- Parses `v`, `vt`, `vn`, `f` (triangulates fans; polygon faces → triangle
  fans). Ignores materials, groups, everything else. OBJ only — glTF is
  REJECTED for this loader (JSON + buffer plumbing, zero exemplar benefit).
- Output: contiguous f32 `(V, 3)` positions, `(V, 3)` normals, `(V, 2)` UVs,
  int32 `(F*3,)` indices — **vertices de-duplicated on the full
  (v, vt, vn) triple** (OBJ indexes the three streams independently; the
  GPU wants one index per vertex). Missing `vt`/`vn` → zero-filled stream +
  a flag the caller can gate on.
- Contract: it loads *the committed asset class* correctly (tested against
  the shipped housing + synthetic edge cases); it is not a general importer.
  Malformed input → error return, never UB.
- The host never learns about file formats — this lives entirely SDK/applet
  side.

## 3. R2 requirement contract → `caliper.geometry.v1_2` (own spec pass)

What TwinScope demonstrably needs, and therefore what the R2 spec pass must
provide — the shape is pinned here so that pass is mechanical:

- **No new entry points.** `reserved0` stays NULL. Growth rides the two
  mechanisms v1_1 already shipped: `CaliperGeomDraw` grows appended fields
  and `draw_stride` carries the growth (the forward-compat row already
  proves the mechanism end-to-end on both backends).
- **Appended draw fields:** `uv_alloc`, `uv_offset` — `(vertex_count, 2)`
  f32, imported, 4-byte-aligned, vertex-pulled like every other stream —
  and `texture` — a **bridge `CaliperTextureId`**, the same id
  `update_texture[_from_alloc]` already services. R2 composes the two
  shipped services; it does not invent texture machinery.
- **`color_mode` gains `COLOR_TEXTURE` (3):** fragment stage samples
  `texture` at the interpolated UV — **bilinear, clamp-to-edge, no mips**
  (fixed; no sampler menu). Lambert (when on) multiplies the sampled rgb
  exactly as it multiplies vertex color today. This is the one real backend
  change: the v1_1 FS is `out = in.color`; v1_2's textured pipelines gain a
  sample. Blend modes apply to the sampled result unchanged.
- **Caps bit 2** (`CALIPER_GEOM_CAP_TEXTURED`). Absent → inert, applet
  ladders down (§1.7).
- **Gates (whole-frame refusal, as always):** COLOR_TEXTURE requires live
  `uv_alloc` (aligned, `vertex_count*8` bounds-checked) AND a live bridge
  texture id owned by the same bridge; a geometry *view* id in `texture` is
  refused (views and textures stay distinct doors, the v1_1 rule). Sampling
  a texture INTO a view while that view is the render target is structurally
  impossible (views are never sampleable by draws) — no feedback loops.
- **Sync:** the sampled texture's content follows the existing texture
  update contract (pipelined timeline / drained) — the draw orders after the
  texture's last update exactly as ImGui sampling does today. No new sync
  vocabulary.
- **Verification bar:** §9.2-style byte-exact rows on both backends (UV
  pull at offsets, clamp-to-edge at borders, bilinear at exact texel
  centers = byte-exact, Lambert×texture, gate refusals). Pinned in the R2
  spec pass, not here.

## 4. R3 requirement contract → instanced transforms (own spec pass)

- **Pure struct growth again:** appended `instance_alloc`,
  `instance_offset`, `instance_count` — `(N, 16)` f32 column-major model
  matrices, imported — and `instance_attr_alloc`, `instance_attr_offset` —
  optional `(N,)` f32, whole-instance value through the draw's LUT
  (`colormap`/`vmin`/`vmax` reused). The per-instance attr is DECIDED here
  (fleet need §1.4): without it R3 ships pose-only and every real fleet
  immediately wants state.
- **Semantics:** effective transform = draw `model` × instance matrix
  (host premultiplies proj·view as today; the instance matrix applies in
  the vertex shader, pulled by instance index). `instance_count == 0` or
  `instance_alloc == 0` → non-instanced draw, exactly today's path (the
  additive default). Instanced + LAMBERT: normal matrix from the combined
  transform — the spec pass decides shader-side vs a second per-instance
  stream, against the byte-exactness bar. `instance_attr` present →
  overrides vertex color source with the instance tint (COLORMAP semantics);
  absent → vertex coloring as today.
- **One draw:** `vkCmdDraw(consumed, N, 0, 0)` /
  `drawPrimitives:...instanceCount:N` — both backends already draw
  instanced points in v1; this generalizes the existing mechanism.
- **Caps bit 3.** Gates: alignment, `N*64` (and `N*4`) bounds, N > 0 when
  alloc present — the §2.3 pattern extended.
- **REJECTED, not deferred: per-instance textures** (texture arrays/atlas
  ABI). The hero carries the draped field; the fleet carries scalar state.
  A future need must bring its own exemplar.
- Naming (`v1_3` vs combined) is the spec pass's call; contracts here are
  independent either way.

## 5. Sequencing (each step ships alone)

1. **R2 spec pass → implement (Metal + Vulkan, byte-exact rows) → hero-only
   TwinScope** — already a complete, compelling twin: loaded CAD shell,
   live field on its skin, learning net, click-the-source interaction.
2. **R3 spec pass → implement → fleet joins TwinScope.**
3. **Polish pass:** the flagship demo — run-proven both platforms, docs,
   whitepaper figure candidate.

Implementation protocol per pass: the v1_1 discipline verbatim — subagent-
driven (Opus implementers for token-heavy transcription), byte-exact
verification by artifacts on both platforms, honest ladders demonstrated.

## 6. Tests (physics/logic only — the TDD-by-stakes rule)

- **Solver:** energy bounded under zero sources; hot-disc steady state
  matches the analytic profile within tolerance; batched B=50 step ==
  50 independent steps (batching is pure vectorization).
- **Loader:** round-trips the committed housing (counts, bounds, UV range,
  dedup correctness on a synthetic (v,vt,vn)-mismatch case); malformed
  inputs error cleanly.
- **Learner:** loss decreases on a fixed tiny run; prediction tensor finite
  and in-bounds.
- No pixel tests for applet UI; the R2/R3 §9.2 rows live in their passes.

## 7. Out of scope (named, so nobody re-opens them here)

- Telemetry ingestion for physical twins — ROADMAP §7's decision, not this
  applet's. TwinScope's "sensor" is its own solver, deliberately.
- em-controller (RL steering) — retained in specs, unbuilt.
- PBR/shadows/photorealism, picking-as-ABI, render-to-tensor — §1.2
  invariants, rejected forever.
- glTF, general-purpose asset importing, per-instance textures — rejected
  above.
- Gaussian-splat instancing (§11 mentions R3 could draw splat captures) —
  a *consumer* of R3, not a requirement on it; no splat work here.
