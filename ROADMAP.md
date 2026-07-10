# Caliper Roadmap — the graphics API, the exemplars, and the twin story

Checkbox discipline: an item is checked only when verified by artifacts
(suites green, run-proven, commit named). Invariants at the bottom are never
checkboxes — they don't ship, they hold.

## Direction (read this first — the why behind the boxes)

**The goal** (GEOMETRY.md §11, decided 2026-07-07): *complex simulations and
digital twins, trained on, watched live.* The whitepaper's thesis is the
mechanism — model state drawn the same frame it's computed, no round trip —
and the graphics API grows that mechanism through a ladder of **nouns** the
instrument can draw live, while the **verbs** never change (zero-copy, gated,
byte-verified against one CPU reference, honestly degraded):

  images (bridge) → point clouds (v1) → connected shapes (v1_1, HERE)
  → state painted ON shapes (R2 textures) → populations of shapes
  (R3 instancing) → other people's hosts (R4 libcaliper/Compass)

What each remaining rung buys, in twin terms:
- **S4 Vulkan (§4)** adds no capability — it makes the existing ones true on
  the hardware that matters for serious training. Same applets, unchanged.
  It converts a Mac demo into the portability claim.
- **R2 textures-on-meshes** decouples shape from state-on-the-shape: a CAD
  housing loaded once, its live temperature/stress/error field re-draped
  every optimizer step at any resolution. The twin's state painted on its skin.
- **R3 instanced transforms** decouples one object from many: one mesh ×
  an imported `(N,16)` pose tensor = a fleet in one draw, moved per-frame by
  whatever writes the tensor (sim or policy). Also the rung that can draw
  Gaussian-splat captures as instanced primitives.
- **R2+R3 together ARE the twin demo** (§6) — which is why they're gated on
  that exemplar's design doc, not built on spec.

**The competitive wedge, one line:** everyone else moves data to the picture
(TensorBoard/W&B log-and-poll; rerun.io logs-then-views; Omniverse renders
then imports; game engines bake ONNX) — **Caliper makes the data BE the
picture**, verified byte-exact on both hardware ecosystems, behind a frozen
~200-byte ABI. Isaac/MuJoCo/Genesis run the opposite dataflow (render→tensor)
and are explicitly not competitors — that direction is a rejected invariant.

**Open strategic decisions** (§7 — decisions before code): telemetry
ingestion for *physical* twins (feed-applet pattern vs. new service; touches
the whitepaper's local-loop claim), and the R4/libcaliper platform call
(intent lives at PLATFORM.md:850 — Compass, the wx sibling host; no spec yet
by design). The next spec to write, per this roadmap's own ordering, is the
**twin exemplar design doc** (§6) — it drags the R2/R3 spec passes behind it.

## 0 · Platform spine (done)

- [x] `tensor_bridge.v1 → v1_2` — zero-copy textures + `import_allocation`, both memory models (Metal/MPS unified, Vulkan/CUDA external-memory)
- [x] `geometry.v1` — instanced points, both platforms, byte-exact gfx rows
- [x] 8 services live (ui, log, jobs, device, metrics, artifacts, data, bridge); ImGui docking shell
- [x] Whitepaper draft v0.1 (accurate to the R0 state)

## 1 · Exemplar cleanup (done 2026-07-09)

- [x] field_scope — PIC plasma physics exemplar on the point API (`test_em_pic`)
- [x] sculpt_scope — live-training net drawn from its own tensor (`test_sculpt`)
- [x] grok_scope deleted; branch renamed + merged to main (`0c7b3bd`); spec statuses truthful
- [x] geometry v1_1 execution plan spec committed

## 2 · geometry.v1_1 — macOS (done 2026-07-09, `feat/geometry-v1_1`)

- [x] S0 — branch rebased onto post-merge main
- [x] S1 — Metal `[[point_size]]` topology-class fix (`c2febe0`); first correct v1_1 frame
- [x] S2 — full §9.2 matrix: 13 cases / 173 assertions on live Metal (coverage, LUT-at-offsets, v1 parity, pure clear+depth, depth order-independence, ALPHA byte-exact, 1-px lines, Lambert ±2 LSB, coplanar overlay, index clamp, 22-case gate battery, stride forward-compat)
- [x] S3 — MeshScope: learned-surface exemplar per design doc (jobs.v1 worker, triple-buffered slots, 3-draw frame, `caliper_mesh_tests`); run-proven: "first zero-copy frame drawn" on Metal+MPS, honest GL fallback
- [x] S3b — MeshScope "paint the target": left-drag sculpts the target grid, the net chases the edit live (stroke queue keeps torch off the frame thread; chase test green; run-proven)

## 3 · Ship it (done 2026-07-09)

- [x] Push `main` to origin
- [x] Code-review pass over `feat/geometry-v1_1` (15 commits) — no blockers, 2 cosmetic NOTEs
- [x] Merge `feat/geometry-v1_1` → main, caps-gated (`d0b61f1`); 6/6 suites green on merged main; branch deleted
- [x] Push merged main (origin in sync)

## 4 · geometry.v1_1 — Windows (S4, done 2026-07-09, `feat/geometry-v1_1-vulkan`)

Spec: `docs/superpowers/specs/2026-07-09-geometry-v1_1-vulkan-phase-b-design.md`
(self-contained for a fresh session on that machine; folds the S5 doc rows into
its acceptance).

- [x] Vulkan backend: `geom.vert/.frag` SPIR-V, per-frame descriptor pool + dynamic-UBO params ring, depth pass, pipeline cache
- [x] Mirror EVERY Metal §9.2 row byte-exact against the same CPU references (D24 verification discipline) — all 13 rows green on RTX 500 Ada, every drawing row byte-exact first try; plus a portable no-CUDA gate-refusal set
- [x] MeshScope run-proven on Windows: zero-copy log line on Vulkan+CUDA; `points-imported` path still green
- [x] S5 docs: ZEROCOPY.md primitives row, STATUS.md, GEOMETRY.md status → shipped

## 5 · Whitepaper v0.2 (done 2026-07-09)

- [x] Rewrite §4/§9: coordinate-style graphics no longer CPU-array-only — geometry is now a zero-copy class on both ecosystems
- [x] Add MeshScope as the geometry exemplar figure; keep the never-claim-unverified discipline

## 6 · The twin exemplar (forcing function for R2/R3)

- [ ] Design doc: one concrete twin — learned thermal/stress field over a CAD housing, fleet of ~50 instanced units
- [ ] Applet-side OBJ/glTF loader helper (fills vertex/index tensors; NO ABI growth)
- [x] R2 `geometry.v1_2` — textures-on-meshes (the `reserved0` slot): spec → Metal → matrix rows → Vulkan mirror — shipped both backends on `feat/geometry-v1_2` (COLOR_TEXTURE, appended uv/texture draw fields, caps bit 2); Vulkan run-proven byte-exact on this box, Metal transcribed + reviewed, hardware verification pending macOS
- [ ] R3 — instanced transforms from an imported `(N,16)` alloc: spec → backends → rows
- [ ] Twin applet ships run-proven on both platforms — the flagship demo — TwinScope v2 surface twin run-proven zero-copy on Vulkan+CUDA (GL fallback proven); Metal/MPS pass pending macOS

## 7 · Strategic (decisions before code)

- [ ] Telemetry ingestion decision: physical twins need live external data (feed-applet pattern vs. new service) — touches the whitepaper's local-loop claim; positioning call, not an engineering default
- [ ] Optional second flagship: em-controller (actor-critic RL steering the PIC plasma — design retained in specs, never implemented); slot after the twin exemplar
- [ ] R4 `libcaliper` second host — platform call (PLATFORM.md Phases 3–6), not geometry's

## Invariants (hold forever, never relitigate)

- Data flows **tensors → pixels → ImGui**, one way. No render-to-tensor, ever.
- No applet-supplied shaders; appearance is the fixed menu.
- Honest ladders: a missing capability degrades to a working slow path and says so; never a wrong image, never a false status line.
- Increments ship against a demonstrated applet need, byte-exact-verified on both ecosystems, as prefix-identical additive revisions.
