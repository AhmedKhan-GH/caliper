# MeshScope — a net's learned surface, drawn as live zero-copy geometry

**Date:** 2026-07-09
**Status:** approved (design), implementation next (execution plan S3)
**Exemplar for:** `caliper.geometry.v1_1` (GEOMETRY.md §9.3 — this doc is that
section made concrete)
**Built on the zero-copy spine of:** `applets/flow_scope` (triple-buffered pool
slots), `applets/sculpt_scope` (live-training net → rendered tensor)
**Replaces:** the analytic-heightfield placeholder committed as
`wip(mesh_scope)` — its pool/import/draw plumbing survives; its sin/cos surface
and frame-thread compute do not.

## One-line

A small MLP `f_θ(x, y) → z` trains live against a fixed target surface; every
optimizer step the net's prediction over a 72×72 grid is written into imported
device tensors and drawn the same frame as Lambert-lit indexed triangles
colored by **per-vertex squared error** through the LUT, with a wireframe
overlay and the training minibatch shown as points — watching a function being
learned, as geometry.

## Why this is the exemplar

geometry.v1's exemplars show *point clouds* learning (sculpt_scope) and
physics (field_scope). v1_1's claim is *any geometry*; the load-bearing demo is
therefore a **connected surface** whose shape is the model's live internal
state — training dynamics (early chaos, bump-by-bump capture, residual error
cooling from hot to uniform) made directly visible. Pure §1.1 purpose: tensors
→ pixels, one way; training consumes the same tensors the renderer reads.

## Architecture

### A. Model + target (worker, torch)

- **Target** `z*(x, y)`: fixed two-bump-plus-ripple analytic surface over
  `[-1.6, 1.6]²`, amplitude ≈ ±0.4 — visually distinct lobes so capture order
  is watchable. Evaluated once on the grid at init (device tensor, constant).
- **Net** `f_θ`: MLP `2 → 64 → 64 → 1`, `tanh` activations, float32, on the
  torch device (MPS/CUDA/CPU). Adam, lr 3e-3 (UI-tunable), seeded for
  reproducibility.
- **Step** (every worker tick):
  1. Sample a **minibatch of 512 continuous coords** uniform over the domain
     (not grid-locked — the grid render then shows *generalization*, and the
     batch is drawn as the "where is supervision" point overlay).
  2. `loss = mse(f_θ(batch), z*(batch))`; backward; Adam step.
  3. Under `NoGradGuard`: `pred = f_θ(grid)` (5 184 points, full grid);
     `err = (pred − z*_grid)²` → the render attr.
  4. Normals from `pred` by **central finite differences** on the regular grid
     (`torch.roll`-style shifts; one-sided at borders), normalized. No autograd
     for normals.
- Training toggle, reset (re-init θ + optimizer), lr slider. Policy runs
  (inference) even when training is off — the surface holds still.

### B. Render-facing state (the flow_scope contract, verbatim)

- **Triple-buffered pool slots**: 3 × {`pos (N,3)`, `normal (N,3)`,
  `attr (N,)`, `sample_pos (512,3)`} born in the `ExportablePool`, imported via
  bridge-v1.2 once, written in place thereafter. Static `tri_idx`, `line_idx`
  (int32) imported once at init.
- Worker writes slot *w*, **device-syncs** (`mps_synchronize_serialized` /
  `cuda::synchronize`), then publishes `ready_slot = w` under the one mutex.
  Frame thread snapshots `display_slot = ready_slot` and never blocks on the
  worker. This satisfies draw_primitives' memory-stability contract (an
  addressed range is never rewritten before the view's next draw).
- Worker runs via **`caliper.jobs.v1`** (manifest gains the service), matching
  the sibling applets. Frame thread does zero torch work.

### C. The frame (one `draw_primitives` call, three draws + clear)

| # | Draw | State |
|---|---|---|
| 0 | learned surface, indexed TRIANGLES | COLORMAP = **per-vertex err²**, MAGMA, vmin 0 / vmax UI-tunable (default 0.05); SHADE_LAMBERT (finite-diff normals); OPAQUE; DEPTH_TEST\|WRITE |
| 1 | learned wireframe, indexed LINES (same pos, line_idx) | FLAT white α≈0.35; UNLIT; ALPHA; DEPTH_TEST only — the §4.1 coplanar LESS_OR_EQUAL overlay |
| 2 | training minibatch, POINTS at `(x, f_θ(x,y), y)` | FLAT amber; UNLIT; ADDITIVE; DEPTH_TEST; size ≈ 3 px |

Orbit (right-drag) / zoom (wheel) camera — applet-owned math, unchanged from
the skeleton. Clear = near-black. Status line: **"zero-copy (imported
geometry)"** only when `draw_primitives` returned true this frame, plus
`step · train-loss · grid-MSE` — the flow_scope honesty discipline verbatim.

### D. Fallback ladder (replaces the skeleton's blank `ImGui::Dummy`)

Any rung missing (no `CALIPER_GEOM_CAP_PRIMITIVES`, no import cap, no pool, no
GPU torch, view creation failed) → **CPU ImPlot heatmap** of the same
per-vertex err² grid (worker still runs; it just also stages one small CPU
copy), honestly labeled "fallback: <reason> — CPU heatmap". The plot is
**input-locked** (the read-only-viewer ImPlot flags) — it is a viewer, not a
widget. The applet is never a blank rectangle.

## Testing (TDD by stakes — logic, not pixels)

`tests/test_mesh_surface.cpp` (CPU, seeded), against a `mesh_model.h` header
holding the model/target/normals code (the sibling pattern: logic in a
testable header, applet file does UI + services):

1. **Learns.** 400 full steps on CPU: final grid-MSE < 25 % of the step-0
   grid-MSE (deterministic seed; generous margin — this is a smoke test of the
   training loop's sign, not a benchmark).
2. **Error map is honest.** `err²` tensor: shape (N,), all finite, all ≥ 0,
   and `mean(err²) == grid-MSE` to float tolerance.
3. **Normals are sane.** Finite-difference normals on a known analytic
   heightfield: unit length within 1e-3, no NaN, and match the analytic
   normal within tolerance away from borders.
4. **Determinism of reset.** Two runs from `reset(seed)` produce identical
   step-10 loss (guards the reproducible-demo property grok_scope had to
   retrofit).

Rendering, colors, camera, and UI get no tests (per the stakes rule); the
draw path itself is pinned by the §9.2 gfx matrix, not by the applet.

## Explicitly out of scope (YAGNI)

- Streaming to `caliper.metrics.v1` / artifacts checkpoints — sculpt_scope
  ships without them; add only if a real session wants cross-run comparison.
- Learnable/selectable target functions beyond the fixed one (a selector is
  UI sugar; one good target tells the story).
- SIREN/positional encodings — tanh MLP learns this target in seconds; fancier
  nets are a different demo.
- Texture-mapped or >72² surfaces, LOD — R2+ territory.
- Windows/CUDA verification — lands with execution-plan S4, no applet change
  expected (the point of the seam).
