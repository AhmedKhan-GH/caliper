# SculptScope — a neural network sculpts a point cloud, rendered from its own tensor with zero copies

**Date:** 2026-07-08
**Status:** approved (design), pending implementation plan
**Built on the zero-copy spine of:** `applets/field_scope` / `applets/flow_scope`
**Sibling ML exemplars (training, but copy-to-plot):** `applets/embed_scope`, `applets/gpt_scope`

## One-line

A small libtorch MLP `g_θ : R^k → R³` maps N fixed latent codes to N 3-D points and
**trains live** (Adam, real `.backward()`) to match a target shape. Every publish its
forward pass **writes its (N,3) output directly into the pool-born slot the renderer
draws in place** — the buffer the final `Linear` layer writes is the buffer the GPU
draws, zero copies between "the ML" and "the picture." You watch a formless blob flow
into a crisp shape as the network learns.

## Motivation — the gap this closes

The suite splits into two camps that never touch the same allocation:

| Applet | libtorch training | Zero-copy render |
|---|---|---|
| `flow_scope` / `field_scope` | ✗ (analytic physics) | ✓ pool-born tensor → `geometry.draw_points`, drawn in place |
| `embed_scope` / `gpt_scope`  | ✓ (`nn::Module` + Adam + `.backward()`) | ✗ copies coords **DtoH** to `std::vector`, drawn via ImPlot3D |

No applet renders a **live-training network's own device tensor** with zero copies.
`embed_scope` even trains a 3-D embedding and animates it — but `DtoH`-copies the
coordinates to plot them. SculptScope is the missing fusion: the exemplar for the
platform's stated mission ("real-time visualization *of* ML state"), where the ML state
being visualized *is literally the rendered buffer*. It is the geometry-service
counterpart of `field_scope`, with the analytic field function replaced by a trained net.

## Invariants (do not violate)

- **Zero-copy contract, made literal.** The rendered `pts` slot is written by the
  network's final layer via an `out=` op (`torch::addmm_out`) under `NoGradGuard`; it is
  pool-born, triple-buffered, imported once via `to_bridge`, and drawn with
  `Geometry::draw_points`. No `.copy_()` sits between the net's output and the draw.
- **Honest fallback ladder.** No geometry caps / no pool / CPU torch / GL renderer all
  fall back to the ImPlot3D subsampled scatter, honestly labeled — verbatim from
  `field_scope`.
- **Threading spine.** One worker trains + runs the display forward + publishes slots
  under one mutex with the `ready_slot`/`display_slot` triple-buffer invariant; the
  frame thread snapshots, draws, and never launches torch ops.
- **Direction of flow (GEOMETRY.md §1.1).** tensors → pixels → ImGui. No readback on any
  hot path. Training reads the latent/target tensors and the net's own output; it never
  reads rendered pixels.

## Architecture

### What is reused verbatim from `field_scope`
Triple-buffered pool slots + `ready/display` invariant; worker/frame threading + single
publish mutex; the CUDA/MPS/CPU device pick and `ExportablePool` opt-in; the orbit
(right-drag) / zoom (wheel) camera and its hand-rolled `look_at`/`perspective`;
DPI-correct physical-pixel view sizing + recreate logging; magma-by-`speed` colormap;
the ImPlot3D CPU-subsample fallback; the greppable provenance status line and worker-start
log lines; the cleanup/leak-on-live-worker discipline.

### What is new
1. A libtorch **generator MLP** replaces the physics `sim_step`.
2. A **live training loop** (minibatch energy-distance loss, Adam) runs in the worker.
3. **Target-shape samplers** (analytic, on-device) provide the training target.
4. The display forward writes **into the slot via `addmm_out`** (the literal-fusion op).

### Components

**A. Generator network — `SculptNet` (`sculpt_model.h`, header-only `nn::Module`)**
- `g_θ : R^k → R³`, `k = 3`. Layers:
  `Linear(k,128) → SiLU → Linear(128,128) → SiLU → Linear(128,128) → SiLU → Linear(128,3)`.
  Small (~50k params); forward on N points is cheap enough to run full-N every publish.
- `hidden(z)` returns the last SiLU activation `(M,128)`; `forward(z)` = `fc_out(hidden(z))`.
  Splitting out `hidden` lets the display path call `addmm_out(slot, b_out, h, W_outᵀ)`
  to write the final `(N,3)` **directly into the pool slot** (§Data flow).

**B. Latents & targets (worker, torch)**
- **Latents:** N codes `z_i ~ N(0, I_k)`, sampled once at worker start, persistent
  (default allocator — not rendered, so not pooled). N = 150 000 on GPU, 20 000 on CPU.
- **Target samplers** — `sample_target(shape, M, device) → (M,3)`, analytic, on-device,
  fresh each train step (infinite target support, no correspondence needed):
  `SPHERE` (r=1 shell), `TORUS` (R=0.9, r=0.35), `HELIX` (spiral tube), `TWO_LOBES`
  (a 2-Gaussian mixture — the "embed_scope look" without data). Combo-selectable; changing
  it just changes the target the same net now chases (visible re-sculpt).

**C. Live trainer — minibatch energy-distance (worker, torch)**
- **Loss:** the energy distance between generated and target batches — a proper,
  correspondence-free distribution metric, differentiable, no nearest-neighbour graph:
  `E(X,Y) = 2·mean‖xᵢ−yⱼ‖ − mean‖xᵢ−xᵢ′‖ − mean‖yⱼ−yⱼ′‖`, pairwise via `torch::cdist`.
  Batch `B = 1024` (GPU) / `256` (CPU): `cdist` is `B×B`, ~1 M distances, trivial.
- **Step:** sample `B` latent indices → `gen = g_θ(z_batch)` **with grad** (default
  allocator) → `tgt = sample_target(shape, B)` → `loss = energy(gen, tgt)` →
  `loss.backward()` → `adam.step()`. One step per worker iteration when `train` is on.
  Entirely decoupled from the display forward and the render slots.
- **Metrics (optional, mirrors `embed_scope`).** When `caliper.metrics.v1` is present,
  `begin_run("sculptscope","generator")` and stream `train/loss` each eval cadence, so
  the host Runs dashboard renders the learning curve for free. Absent → skipped silently.

**D. Display forward + publish (worker, torch, the fusion)**
- Persistent pool-born slots `pts[3]` `(N,3)` and `speed[3]` `(N,)`, allocated once inside
  `pool->use()` (exactly `field_scope`'s slot allocation).
- Each publish, under `NoGradGuard`: compute `h = g_θ.hidden(z_all)` `(N,128)` (default
  alloc), then **`torch::addmm_out(pts[write], b_out, h, W_outᵀ)`** — the final layer's
  matmul lands in the pool slot itself. `speed[write] = ‖pts[write] − prev‖`; `prev.copy_(pts[write])`.
  Sync (CUDA `synchronize()` / MPS serialized drain — `field_scope`'s contract) **before**
  publishing the slot.
- **Color = `speed` through magma** (verbatim `field_scope` channel): points still moving
  (early training / after a shape change) glow; a converged cloud dims to the LUT floor.
  Honest, cheap, needs no per-shape SDF, and reuses the proven color path.

### Data flow (per worker step)
```
if train:                                   # ordinary training, default allocator
    gen ← g_θ(z[batch])                      # WITH grad
    tgt ← sample_target(shape, B)
    loss ← energy(gen, tgt);  loss.backward();  adam.step()
    metrics.scalar(run, "train/loss", step, loss)     # optional

# publish (every K steps) — the fusion, NoGrad:
h            ← g_θ.hidden(z_all)             # (N,128), default alloc
addmm_out(pts[write], b_out, h, W_outᵀ)      # (N,3) written INTO the pool slot
speed[write] ← ‖pts[write] − prev‖ ;  prev ← pts[write]
sync();  publish(write)                      # renderer imports pts[write] in place
```

## Interaction
- **Right-drag orbit, wheel zoom** — unchanged from `field_scope`.
- **Controls:** `pause` · `train on/off` · `reset weights` (re-init the net → blob returns,
  re-sculpts) · `learning rate` · **target shape** combo · `color` scale. The net keeps
  *running* the display forward even when training is off (frozen net, static cloud).
- Status line (last frame's provenance): `N pts · zero-copy (imported geometry) · loss L
  · S steps · Metal` when the path drew, else the honest fallback string with the reason.

## Testing (per TDD-by-stakes — logic gets tests; render/UI/color do not)
`tests/` gains `test_sculpt.cpp` (a plain torch test, no host), covering:
1. **Energy distance is a metric-shaped loss.** `energy(X,X) ≈ 0` (up to the `‖xᵢ−xᵢ′‖`
   self-term bias on finite batches → within tolerance); `energy(X,Y) > 0` for a shifted
   `Y`; symmetric `energy(X,Y) == energy(Y,X)`; gradient w.r.t. `X` is finite and nonzero.
2. **Target samplers hit their manifold.** Sampled points satisfy the shape's analytic
   constraint to tolerance (SPHERE: `‖p‖≈1`; TORUS: `(√(x²+y²)−R)²+z² ≈ r²`).
3. **Training reduces loss (smoke).** From a fixed seed, K Adam steps on `SPHERE`
   measurably lower `energy(g_θ(z_batch), tgt)` over a short window (the net learns).
4. **`addmm_out` fusion is exact.** `addmm_out(slot,b,h,Wᵀ)` equals `g_θ.forward(z)` to
   bit tolerance — proves the display path renders the net's true output, not an approximation.

## Wiring
- New dir `applets/sculpt_scope/` (auto-globbed by the top-level `applets/*`
  `add_subdirectory`, `CONFIGURE_DEPENDS`): `plugin.cpp`, `sculpt_scope.{h,cpp}`,
  `sculpt_model.h`, `sculpt_scope.caliper.toml`, `CMakeLists.txt` (copy of
  `field_scope`'s, links `caliper::sdk`, `caliper::ui_stack`, `TORCH_LIBRARIES`).
- id `dev.caliper.sculpt-scope`, name **SculptScope**, tag `ML`; window title
  **"SculptScope: Learned Cloud"** added to `src/main.cpp` `central_windows`.
- Services: required `ui/log/jobs/device`; optional `tensor_bridge/geometry/metrics`.

## Explicitly out of scope (YAGNI)
- Loading external datasets (self-contained analytic targets only — that is the point of
  robustness; MNIST-embedding was considered and cut: more code, download fragility, no
  gain to the fusion being proved).
- Artifacts save/load of the trained net (a trivial follow-up mirroring `embed_scope`;
  not needed to demonstrate the fusion — live training from `reset` is the show).
- Colouring by analytic distance-to-manifold (deferred; `speed` is cheaper, universal,
  and already proven).
- Per-point picking / hover, custom shaders, render-to-tensor (GEOMETRY.md invariants).
