# MeshScope: paint the target — fencing with an optimizer

**Date:** 2026-07-09
**Status:** approved (design), implementation next
**Amends:** `2026-07-09-meshscope-learned-surface-design.md` (the learned-surface
exemplar, shipped `063d617`/`e2d7e09`) — this is an interaction upgrade on top;
everything not named here is unchanged from that spec.
**API impact:** none. Rides entirely on shipped `geometry.v1_1`.

## One-line

The target surface becomes yours: left-drag raises (Alt lowers) a Gaussian bump
in the target grid and the live-training net chases your edit — a hot
error scar blooms where you painted, then the surface swells to meet it and
cools, making generalization *visible where you didn't touch*.

## What changes (and only this)

### A. `mesh_model.h` — target becomes a mutable grid

Replace the analytic-only target with a `TargetGrid`:

- `torch::Tensor grid` — `(kGrid*kGrid,)` f32 on the model's device, initialized
  from the existing two-lobe+ripple preset (cold-open demo unchanged).
- `sample(xy)` — **bilinear** interpolation of `grid` at continuous domain
  coords `(B,2)`; replaces the analytic call in `train_step`'s minibatch loss.
  Clamp coords to the domain (brush can't push samples outside anyway).
- `brush(cx, cy, radius, amp)` — `grid += amp * exp(-d²/(2·(radius/2)²))`
  where `d` is domain-space distance from `(cx, cy)` to each grid node;
  result clamped to `[-1.0, 1.0]` (keeps camera framing sane).
- `reset_preset()` — restore the analytic preset.
- Grid-aligned paths (`err_sq`, `grid_mse`) compare against `grid` directly —
  no interpolation needed there.

### B. `mesh_scope.cpp` — stroke queue, ray pick, controls

- **Stroke queue (the threading rule).** The frame thread never touches torch:
  it pushes `{cx, cy, radius, amp}` structs onto a small `std::vector` under
  the existing publish mutex. The worker drains the queue at the top of each
  step (before `train_step`), applying brushes in order. One-step latency;
  single-writer invariant preserved.
- **Picking.** Left-drag over the zero-copy image → unproject the cursor
  through the applet's own view/proj (inverse ray, the flow_scope pattern) →
  intersect with the domain base plane `y = 0` → `(cx, cy)` in domain coords.
  Ignore misses (ray parallel / outside domain). While left-drag paints,
  it does NOT orbit; right-drag orbit and wheel zoom unchanged.
- **Brush sign.** Plain left-drag raises (`amp = +strength`); Alt+left-drag
  lowers (`-strength`). Strength is per-frame while dragging (continuous
  strokes), scaled by `ImGui::GetIO().DeltaTime` so paint rate is
  framerate-independent.
- **Controls added:** brush radius slider (domain units, default 0.35),
  brush strength slider (units/sec, default 0.8), `reset target` button.
  Existing train/reset/lr/err-vmax controls unchanged.
- **Status line** gains nothing (the hot scar is the feedback). One-shot
  provenance log unchanged.
- **Fallback mode:** painting disabled (the CPU heatmap stays a locked
  viewer); no attempt at plot-click painting.

## What you feel (acceptance narrative)

Converged surface, mostly dark. Drag a mountain under the cursor: a MAGMA-hot
scar blooms exactly there (net now wrong), amber minibatch dots keep raining
everywhere, and over the next seconds the surface inflates to meet the edit
while the scar cools outward — training and generalization as a visible chase.
`reset target` snaps the answer key back; the whole surface re-heats and
re-converges.

## Testing (logic only, in `tests/test_mesh_surface.cpp`)

1. **Bilinear sampler exact** at grid nodes and midpoints of a known small
   grid (hand-computed values).
2. **Brush is local, bounded, signed:** far cells unchanged (beyond ~4·radius),
   peak ≈ amp at center, negative amp lowers, results clamped to ±1.
3. **The chase:** seeded CPU run to convergence (grid-MSE small), apply one
   brush stroke → grid-MSE strictly increases; continue training → grid-MSE
   falls below 25 % of the post-stroke value. Proves the loop learns the
   *edited* target.
4. Existing tests (learns, err-map honesty, normals, reset determinism) stay
   green — the preset init keeps their setup valid.

No tests for picking, brush UI, or rendering (stakes rule; the draw path is
pinned by the §9.2 matrix).

## Explicitly out of scope (YAGNI)

Multi-brush shapes, undo/redo, target import/export or save, painting in
fallback mode, target ghost overlay, any ABI/API change, Windows verification
(S4 covers the applet unchanged).
