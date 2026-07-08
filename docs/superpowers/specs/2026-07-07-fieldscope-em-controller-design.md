# FieldScope — a charged cloud steered by a learned controller

**Date:** 2026-07-07
**Status:** approved (design), pending implementation plan
**Supersedes the physics of:** `applets/flow_scope` (FlowScope, `dev.caliper.flow-scope`)
**Prior spec:** `docs/superpowers/specs/2026-07-07-geometry-flowscope-design.md`

## One-line

A million charged particles move under real Lorentz-force physics `F = q(E + v×B)`;
a small neural policy watches the cloud and sets the E/B field each step to drive
the cloud's centroid onto a target the user drags around. The policy **trains live**
and visibly improves. The `caliper.geometry.v1` zero-copy render path is untouched.

## Motivation

FlowScope exists to showcase `caliper.geometry.v1` zero-copy rendering (pool-born
torch tensors imported once by the host and drawn in place, triple-buffered). Its
physics — an analytic `sin/cos` curl field — is incidental and not "practical."

This redesign keeps the entire zero-copy spine and replaces only the *field function*
with a fusion the platform's stated mission ("state-based ML viz") actually wants:
**real electromagnetic physics closed-loop-controlled by a real, live-training ML
policy** — the literal shape of plasma-confinement / accelerator beam-steering control.

Both halves are load-bearing: the physics is genuine (Boris-pusher Lorentz
integration), and the ML does real work you can watch (an analytic-policy-gradient
controller whose loss falls in real time).

## Invariants (do not violate)

- **Zero-copy geometry contract.** The rendered cloud's `pos`/`speed` tensors remain
  pool-born, triple-buffered, imported via `to_bridge`, and drawn with
  `Geometry::draw_points`. Training cost must not touch this path.
- **Honest fallback ladder.** No geometry caps / no pool / CPU torch / GL renderer all
  still fall back to the ImPlot3D subsampled scatter, honestly labeled.
- **Threading spine.** One worker steps sim + policy and publishes slots under one
  mutex; the frame thread snapshots, draws, and never blocks on the worker.

## Architecture

### What stays (unchanged from FlowScope)
- Triple-buffered pool-born `pos`/`speed` slots; `ready_slot`/`display_slot` invariant.
- Worker/frame threading and the single publish mutex.
- Orbit (right-drag) / zoom (wheel) camera; magma-by-speed colormap.
- ImPlot3D CPU-subsample fallback; provenance status line; CUDA/MPS/CPU device pick.

### What changes
1. Particles gain a **velocity** tensor that participates in the render-facing state
   (positions stay the rendered attribute; velocity magnitude still feeds `speed`/color).
2. The analytic curl field is replaced by **Lorentz-force integration under a global
   E/B field** produced by the policy.
3. A new **tiny ML control path** runs in the worker alongside the sim.

### Components

**A. Physics integrator — Boris pusher (worker, torch)**
- State per particle: position `p` (N×3), velocity `v` (N×3). Charge-to-mass `q/m`
  is a uniform scalar constant (single species — multiple species cut as YAGNI).
- Field: **global** `E` (3-vector) and `B` (3-vector), the policy's output. No discrete
  coils (cut as YAGNI — global E/B gives enough control authority: E accelerates the
  centroid, B curves it).
- Integration: the **Boris pusher** — half electric kick, full magnetic rotation, half
  electric kick. Standard stable integrator for `v×B`; conserves energy in a static
  field and prevents the blow-up a naive Euler `v×B` step causes.
- Boundary: mild velocity damping + a soft restoring box so the cloud stays in view;
  centroid remains free to move toward the target. (No hard wrap — wrap would fight
  the steering objective.)
- `speed_out = ‖v‖` as today, feeding the color LUT.

**B. Control policy — tiny MLP (worker, torch)**
- Shape: `state(~10) → 64 → 64 → 6`, `tanh`-bounded outputs scaled to physical E/B ranges.
- **State (input):** cloud centroid (3), centroid velocity (3), cloud spread (scalar or
  3), target position (3) — all cheap reductions over the cloud.
- **Action (output):** `E` (3) and `B` (3) for the next step.
- Runs every sim step to produce the field the full cloud integrates under.

**C. Live trainer — analytic policy gradient through differentiable physics (worker)**
- Every frame, sample a **256-particle tracer batch** from the cloud and roll it forward
  ~8 steps **differentiably through the same Boris physics**, loss
  `Σ‖centroid_t − target‖² + λ‖action‖²`. One Adam step; gradients flow through the
  physics into the policy.
- The **full 1M cloud is forward-only** (no grad) and simply consumes the policy's E/B,
  so training cost is decoupled from the render cloud and the zero-copy showcase is
  intact. Tracer count and horizon are tunable constants.
- Controls: `train on/off`, `reset policy`, learning rate. The policy runs (inference)
  even when training is off.

### Data flow (per worker step)
```
state ← reduce(cloud pos/vel, target)
E,B   ← policy(state)                    # inference, no grad
cloud ← boris_push(cloud, E, B, dt)      # 1M particles, forward-only  → publish slot
if training:
    tracers ← sample(cloud, 256)
    loss ← Σ‖centroid(rollout(tracers,E,B,H)) − target‖² + λ‖E,B‖²   # differentiable
    adam.step(∇loss)                     # updates policy weights
```

## Interaction

- **Left-drag → move the target.** Cursor ray projected to the cloud's depth plane sets
  the target position (replaces FlowScope's direct force-splat). You drag the goalpost;
  the controller scrambles to re-aim the field.
- **Right-drag orbit, wheel zoom** — unchanged.
- Target rendered as an **overlay crosshair**: the 3D target projected to screen via
  `ImDrawList` (the geometry service draws points only; an overlay avoids a second pass).
- **Controls:** pause · train on/off · reset policy · learning rate · field clamp ·
  **manual override** (freeze the policy and set E/B by hand to watch the cloud lose
  control). Status line gains `policy loss · steps trained · ‖centroid − target‖`.

## Testing (physics invariants — real tests, not change-detectors)

Per the TDD-by-stakes rule, the physics and training logic get tests; rendering, UI,
color, and the target overlay do not.

1. **Gyration.** A charged particle in a uniform `B` (E=0) traces a circle at the correct
   gyrofrequency `ω = qB/m` and gyroradius `r = mv⊥/qB`; kinetic energy is conserved to
   tolerance over a full orbit (validates the Boris pusher).
2. **E×B drift.** Under crossed uniform E and B, the guiding-center drift velocity matches
   the analytic `v_drift = E×B/‖B‖²`.
3. **Training gradient smoke test.** One APG step on a fixed target measurably *reduces*
   the steering loss (gradients flow, sign is correct).

## Naming

Keep the internal target `flow_scope` and id `dev.caliper.flow-scope` (zero CMake /
registration churn). Retitle the window and header to
**"FieldScope — learned control of a charged cloud."** A full file/target rename is an
optional mechanical follow-up, out of scope here.

## Explicitly out of scope (YAGNI)

- Multiple charge species / sign-colored particles.
- Discrete current-loop coils or spatially-varying analytic fields.
- Differentiating through the full 1M-particle rollout (tracer batch replaces it).
- Full applet/target rename.
- Pretrained / checkpointed policy weights (live training from reset only).
