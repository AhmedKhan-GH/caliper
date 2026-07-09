# Caliper Roadmap — the graphics API, the exemplars, and the twin story

Checkbox discipline: an item is checked only when verified by artifacts
(suites green, run-proven, commit named). Invariants at the bottom are never
checkboxes — they don't ship, they hold.

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

## 3 · Ship it (next actions, in order)

- [ ] Push `main` to origin (currently 13 commits ahead, local only)
- [ ] Code-review pass over `feat/geometry-v1_1` (11 commits)
- [ ] Merge `feat/geometry-v1_1` → main, caps-gated (Windows honestly reports no primitives until S4; the ladder covers it) — amends the S5 both-platforms gate deliberately
- [ ] Push merged main

## 4 · geometry.v1_1 — Windows (S4, needs the Windows box)

- [ ] Vulkan backend: `geom.vert/.frag` SPIR-V, per-frame descriptor pool + dynamic-UBO params ring, depth pass, pipeline cache
- [ ] Mirror EVERY Metal §9.2 row byte-exact against the same CPU references (D24 verification discipline)
- [ ] MeshScope run-proven on Windows: zero-copy log line on Vulkan+CUDA; `points-imported` path still green
- [ ] S5 docs: ZEROCOPY.md primitives row, STATUS.md, GEOMETRY.md status → shipped

## 5 · Whitepaper v0.2 (after S4)

- [ ] Rewrite §4/§9: coordinate-style graphics no longer CPU-array-only — geometry is now a zero-copy class on both ecosystems
- [ ] Add MeshScope as the geometry exemplar figure; keep the never-claim-unverified discipline

## 6 · The twin exemplar (forcing function for R2/R3)

- [ ] Design doc: one concrete twin — learned thermal/stress field over a CAD housing, fleet of ~50 instanced units
- [ ] Applet-side OBJ/glTF loader helper (fills vertex/index tensors; NO ABI growth)
- [ ] R2 `geometry.v1_2` — textures-on-meshes (the `reserved0` slot): spec → Metal → matrix rows → Vulkan mirror
- [ ] R3 — instanced transforms from an imported `(N,16)` alloc: spec → backends → rows
- [ ] Twin applet ships run-proven on both platforms — the flagship demo

## 7 · Strategic (decisions before code)

- [ ] Telemetry ingestion decision: physical twins need live external data (feed-applet pattern vs. new service) — touches the whitepaper's local-loop claim; positioning call, not an engineering default
- [ ] Optional second flagship: em-controller (actor-critic RL steering the PIC plasma — design retained in specs, never implemented); slot after the twin exemplar
- [ ] R4 `libcaliper` second host — platform call (PLATFORM.md Phases 3–6), not geometry's

## Invariants (hold forever, never relitigate)

- Data flows **tensors → pixels → ImGui**, one way. No render-to-tensor, ever.
- No applet-supplied shaders; appearance is the fixed menu.
- Honest ladders: a missing capability degrades to a working slow path and says so; never a wrong image, never a false status line.
- Increments ship against a demonstrated applet need, byte-exact-verified on both ecosystems, as prefix-identical additive revisions.
