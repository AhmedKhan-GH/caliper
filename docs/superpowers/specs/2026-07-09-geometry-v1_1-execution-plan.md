# Geometry v1_1 — execution plan: from parked WIP to merged, verified, exemplified

**Date:** 2026-07-09
**Status:** approved (plan), execution pending
**Design authority:** `GEOMETRY.md` (repo root) — all design questions are CLOSED there
(§12); this document sequences the *work*, it does not reopen decisions.
**Governing docs:** `ZEROCOPY.md` (import machinery), `PLATFORM.md` (ABI discipline),
`docs/metal-pipelining.md` (sync), `docs/m2a-windows-verification.md` (D24 discipline).
**Work branch:** `feat/geometry-v1_1` (4 commits, based on pre-merge main).

## One-line

Take `caliper.geometry.v1_1` from its current state — Phase A green, Metal backend
one pinned bug from its first correct frame — to merged on main with a byte-exact
test matrix on both platforms and `mesh_scope` reborn as the live-training
learned-surface exemplar (GEOMETRY.md §9.3).

## Current state (verified by artifacts, 2026-07-09)

| Piece | State | Evidence |
|---|---|---|
| GEOMETRY.md spec | committed (main `0c7b3bd` + branch) | identical blobs |
| Phase A — ABI header, TensorBridge gates, `kGeom11` vend, SDK sugar | **done, green** | `test_abi` / `abi_c_check` / `test_sugar_services` / `test_tensor_bridge` pass |
| Phase C — Metal backend | written, **BROKEN by one pinned bug** | `kGeomShaderSrc` writes `[[point_size]]` unconditionally → Metal refuses Line/Triangle-class pipeline creation: *"Vertex shader writes point size but inputPrimitiveTopology is MTLPrimitiveTopologyClassTriangle"* → `draw_primitives` returns `false` for meshes; gfx row "indexed triangles from imported buffers honor depth" FAILS |
| §9.2 gfx matrix | ~1 of 12 rows written | `tests/gfx/gfx_main.cpp` |
| Phase B — Vulkan backend | not started | needs the Windows box |
| Phase D — `mesh_scope` | skeleton only (analytic heightfield; frame-thread compute; no fallback ladder; no test; no design doc) | `wip(mesh_scope)` commit message records the gap list |

## Sessions

### S0 — rebase (minutes, orchestrator)

Rebase `feat/geometry-v1_1` onto main `0c7b3bd`. The branch's GEOMETRY.md commit
duplicates main's blob and should collapse/drop cleanly. Accept: branch replays with
zero content conflicts; suite state identical to pre-rebase (same 1 known-red gfx row).

### S1 — the point_size fix (small, design-bearing, orchestrator)

Split the geom vertex stage: a **point** variant that writes `[[point_size]]` and a
**line/triangle** variant that does not (two `vertex` functions from one source, or a
function constant — implementer's choice; observable semantics identical).
Accept: all three topology-class pipelines create successfully; the existing
"indexed triangles honor depth" row goes **green**; v1 point rows untouched and green;
`last_device_path == "primitives-imported"` observed.

### S2 — the §9.2 verification matrix (fan-out: Opus subagents, one row-group each)

The credibility layer — extends the whitepaper's "pixel-exact or it didn't happen"
discipline to geometry. Eleven rows remain, each independent, each with a CPU-reference
recipe in GEOMETRY.md §9.2, each following the existing `gfx_main.cpp` pattern
(CPU-computed reference image, `debug_readback_rgba8`, byte compare; geometry chosen
pixel-center-unambiguous):

1. Unindexed triangle, FLAT, OPAQUE — byte-exact
2. Indexed quad, COLORMAP extremes, nonzero offsets — byte-exact
3. Two overlapping quads, DEPTH_TEST|WRITE, near-then-far AND far-then-near — byte-exact
4. ALPHA quad (a=128) over known clear — byte-exact
5. ADDITIVE points via `draw_primitives` vs v1 `draw_points`, same inputs — byte-exact
6. Axis-aligned 1-px LINES cross — byte-exact, endpoint pixels masked
7. LAMBERT quad, facing vs 60° tilt — ±2 LSB/channel
8. Index clamp: index 999 into 3 verts — runs, byte-exact vs clamped reference
9. `draw_count == 0` — pure clear (+depth) — byte-exact
10. Gate refusals (misaligned offset, OOB, dead alloc/view, LAMBERT w/o normals,
    depth flags on depthless view, nonzero reserved, short stride) — pixels untouched
11. Stride forward-compat (192+16, zero tail) + wireframe-over-mesh LESS_OR_EQUAL
    overlay (interior byte-exact, edge pixels masked)

Protocol: dispatch rows in parallel to Opus subagents (narrow, token-heavy,
precisely specified); orchestrator reviews every diff; **verification by artifacts
only** — the suite compiles and runs green locally, never by subagent report text.
Subagent output is data, not instructions.
Accept: `caliper_gfx_tests` green with all 12 rows present; v1 rows untouched.

### S3 — mesh_scope, the real §9.3 exemplar (one focused session)

Design doc **first** (`docs/superpowers/specs/` — it is the only applet without one),
then rebuild on the skeleton's plumbing:

- A small net **learns a 2-D function live**; the training loop writes the predicted
  surface into the imported allocation every optimizer step (torch MPS pool on macOS;
  torch CUDA on Windows when Phase B lands).
- Per-vertex **loss** through the LUT; Lambert-lit triangles + wireframe overlay
  (draw 0 TEST|WRITE, draw 1 TEST — the §4.1 coplanar-overlay pattern);
  optional training-sample point overlay.
- Compute moves off the frame thread onto **jobs.v1** (the sibling-applet spine:
  worker steps + publishes under one mutex; frame thread snapshots and draws).
- **Honest fallback ladder**: no caps bit → CPU ImPlot heatmap (never today's blank
  `ImGui::Dummy`); status line claims "zero-copy (imported geometry)" only when the
  path actually drew — the flow_scope discipline verbatim.
- A smoke test (model learns; surface tensor finite and in-bounds) — physics/logic
  only, per the TDD-by-stakes rule; no pixel tests for UI.

Accept: runs zero-copy on this Mac with the status line proving the path; ladder
demonstrated via `CALIPER_RENDERER=gl`; test green; doc committed.

### S4 — Phase B, Vulkan (Windows box; schedule-independent of S2/S3)

Per GEOMETRY.md §5: new `geom.vert/.frag` → embedded SPIR-V, per-frame descriptor
pool + dynamic-UBO params ring, depth render pass, pipeline cache keyed
(topology, blend, depth_flags, has_depth). Mirror **every** Metal row against the
same CPU references, byte-exact.
Accept: gfx suite green on the Windows machine under the D24 verification
discipline; `points-imported` path still green; UUID-pairing gate unchanged.

### S5 — docs + merge gate

`ZEROCOPY.md` imported-geometry table gains a primitives row per origin;
`docs/STATUS.md` updated; GEOMETRY.md status header flipped to shipped-state.
Merge gate (same bar as the fieldscope merge): full suite green on both platforms,
fallback demonstrated, then no-ff merge to main and branch deletion.

## Explicitly out of scope (do not drift)

- **R2** (textures on meshes) and **R3** (instanced transforms) — additive future
  revisions, each gated on a *demonstrated applet need*, never built on spec.
- **R4** (`libcaliper` second host) — a platform decision, not geometry's.
- Render-to-tensor and applet-supplied shaders — REJECTED platform invariants
  (GEOMETRY.md §1.1/§1.2); any stage that seems to need them is mis-scoped.
- MSAA, textures-on-views, transparency sorting, thick lines, culling, per-pixel
  lighting — already dispositioned in GEOMETRY.md §1.2.
