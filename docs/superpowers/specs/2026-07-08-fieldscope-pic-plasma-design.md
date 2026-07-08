# FieldScope — a self-consistent electrostatic PIC plasma, drawn zero-copy

**Date:** 2026-07-08
**Status:** approved (design), pending implementation plan
**Replaces the physics of:** `applets/field_scope` (the analytic-dipole sim — test
particles in a canned field, not a plasma)
**Supersedes (physics direction):** `docs/superpowers/specs/2026-07-07-fieldscope-em-controller-design.md`
— that spec steered the cloud with a learned RL controller (a different, ML-first
project). This spec is a *proper self-consistent simulation*: no ML, the particles
generate the field that pushes them.
**Shares the zero-copy spine of:** `applets/sculpt_scope`, `applets/flow_scope`

## One-line

A few hundred thousand charged particles evolve under their **own** electric field —
each step the particles deposit charge onto a grid, a Poisson FFT solve produces the
self-consistent field, that field is gathered back and a Boris pusher advances them —
so real plasma phenomena (two-stream instability, plasma oscillations) *emerge* instead
of being scripted. Positions render zero-copy (pool-born tensor drawn in place by
`geometry.v1`), coloured by speed — SculptScope's visual twin, a genuine EM simulation.

## Motivation — what was wrong

The prior `field_scope` sim pushed particles with the Lorentz force under a **fixed
analytic dipole E and a uniform B**, kept in frame by **artificial viscous drag** and
**box-wrapping**, integrated with **semi-implicit Euler**. Four physics objections:

1. **Fields are prescribed, not self-consistent** — particles move *in* a field but
   never *create* one. It is test particles in a canned field, not a plasma.
2. **No particle–particle interaction** — the charges never feel each other.
3. **Euler on `v×B` is the wrong integrator** — it does not conserve energy in a
   magnetic field. The Boris pusher is the standard.
4. **Viscous drag is unphysical** — a hack to stop blow-up; vacuum EM has no friction.

This redesign fixes all four with the canonical, GPU-scalable method: **electrostatic
Particle-In-Cell (PIC)** on a periodic grid with an FFT Poisson solve and a Boris pusher.

## Invariants (do not violate)
- **Zero-copy contract.** The rendered `pos`/`speed` tensors stay pool-born,
  triple-buffered, imported via `to_bridge`, drawn with `Geometry::draw_points`. The
  field solve touches only the small grid, never the particle render path.
- **Honest fallback ladder.** No geometry caps / no pool / CPU torch / GL → the
  ImPlot3D subsampled scatter, honestly labelled — verbatim from `field_scope`.
- **Threading spine.** One worker steps the sim and publishes slots under one mutex with
  the `ready_slot`/`display_slot` triple-buffer invariant; the frame thread never
  launches torch ops.
- **Direction of flow (GEOMETRY.md §1.1).** tensors → pixels → ImGui. No readback on the
  render hot path.

## Architecture

### What stays (verbatim from field_scope / sculpt_scope)
Triple-buffered pool slots + ready/display invariant; worker/frame threading + one
publish mutex; CUDA/MPS/CPU device pick + `ExportablePool` opt-in; orbit/zoom camera
and hand-rolled `look_at`/`perspective`; DPI-correct physical-pixel view sizing;
magma-by-speed colour; ImPlot3D CPU-subsample fallback; provenance status line; the
cursor-ray impulse (left-drag) as an external perturbation; the cleanup/leak discipline.

### What changes
The analytic `sim_step` is replaced by one PIC step. Physics core lives in a host-free
header `em_pic.h` (unit-testable like `sculpt_model.h`); `field_scope.cpp` calls it.

### PIC core — `em_pic.h` (pure torch, host-free)

Periodic box `[0,L)³`, `L = 2π`, grid `G³` (`G = 32`; ~6 particles/cell at N≈200k for
good statistics). Normalised units: `ε₀ = 1`, `q/m = 1`; a tunable `coupling` scales
the field (sets the effective plasma frequency `ω_p ≈ √coupling`).

- **`deposit_cic(pos, G, L) → rho (G,G,G)`** — Cloud-In-Cell: each particle contributes
  to its 8 surrounding nodes with weights `∏(1−f or f)` per axis; scatter via
  `index_add_` on the flattened grid. Charge-conserving; per-particle charge normalised
  so mean density ≈ 1.
- **`solve_field(rho, L) → E (G,G,G,3)`** — neutralise `rho -= rho.mean()`; `rho_hat =
  fftn(rho)`; `k` from `fftfreq(G)`; `phi_hat = rho_hat / k²` with `k²[0]=1`,
  `phi_hat[0]=0`; `E_hat = −i·k·phi_hat` per axis; `E = real(ifftn(E_hat))`. Device-first
  with a CPU fallback (caught once) for backends lacking FFT — the grid is `32³`, tiny.
- **`gather_cic(E, pos, L) → Epart (N,3)`** — same 8-node CIC weights, `Σ w·E[node]`.
- **`boris_push(pos, vel, Epart, B, qm, dt, L)`** — half E-kick, magnetic rotation
  (`t=qm·B·dt/2`, `s=2t/(1+|t|²)`, `v⁻→v'→v⁺`), half E-kick; `pos += vel·dt`; wrap mod L.
  Standard, energy-conserving in static B.

### Initial conditions — `init_state(kind, N, L, device) → (pos, vel)`
Selector mirroring SculptScope's shape combo (the collective result *emerges*):
- **`kTwoStream`** — uniform in space; half the particles at `+v₀ x̂`, half at `−v₀ x̂`,
  small thermal spread. The two-stream instability grows into phase-space vortices.
- **`kThermal`** — uniform positions, Maxwellian velocities → plasma oscillations.
- **`kBeam`** — a drifting warm beam.
- **`kBlob`** — a localised warm clump that expands and rings against the background.

### Data flow (per worker step)
```
rho   ← deposit_cic(pos, G, L)              # 1M particles → 32³ grid (scatter)
E     ← solve_field(rho, L)                 # FFT Poisson: particles' OWN field
Epart ← gather_cic(E, pos, L)               # field → particles (interp)
boris_push(pos, vel, Epart + impulse, B)    # self-field (+ optional bg B, + cursor)
speed ← ‖vel‖                               # colour channel
sync(); publish(write)                      # renderer imports pos[write] in place
```
`E`/`rho` are grid-sized (default allocator); only `pos`/`speed` slots are pool-born.

## Interaction
Right-drag orbit, wheel zoom (unchanged). **Initial-condition combo** (re-seeds the
plasma, like `reset`). Left-drag → cursor-ray impulse added to `Epart` (perturb the
plasma, watch it respond). Sliders: **B** (background field), **coupling** (ω_p),
**temperature** (thermal spread on re-seed), **color**. Status line reports
`N particles · zero-copy (imported geometry) · <IC> · KE <energy> · N steps · Metal`.

## Testing (physics invariants — real tests, not change-detectors)
`tests/test_em_pic.cpp` (pure torch, CPU, `REQUIRE` — c10 shadows doctest's `CHECK`):
1. **Poisson solve is correct.** For `ρ = cos(k·x)` on the grid, the solved `E` matches
   the analytic `E = −∇φ`, `φ = ρ/|k|²`, to tolerance (the core of the whole method).
2. **CIC conserves charge and partitions unity.** `Σ rho·cell_vol ≈ N·q`; the 8 CIC
   weights sum to 1 for arbitrary positions.
3. **Boris gyration.** In uniform B (E=0) a particle traces a circle at `ω = qB/m`,
   kinetic energy conserved over a full orbit to tolerance.
4. **E×B drift.** Under crossed uniform E,B the guiding-centre drift matches `E×B/|B|²`.
5. **Neutral plasma conserves momentum.** A periodic OCP with no background B conserves
   total momentum over many steps (no external force) to tolerance.

Rendering, UI, colour, camera get no tests (per TDD-by-stakes).

## Naming / wiring
Keep target `field_scope`, id `dev.caliper.field-scope`, window title
**"FieldScope: EM Field"** (zero CMake / `central_windows` churn). Update the summary,
header comment, and `.caliper.toml` to describe the PIC plasma. `em_pic.h` added to the
applet dir; `tests/test_em_pic.cpp` gets its own torch test executable (mirrors
`caliper_sculpt_tests`) with the applet dir on its include path.

## Explicitly out of scope (YAGNI)
- Full electromagnetic PIC (Yee-grid FDTD Maxwell) — much heavier; electrostatic PIC is
  the right scope for this particle count / framerate.
- Multiple explicit species (electrons + ions) — the one-component-plasma neutralising
  background is the standard simplification and enough for the target phenomena.
- Collisions / higher-order shape functions (TSC) — CIC is the standard baseline.
- The learned RL controller (the superseded 2026-07-07 spec) — this is a simulation.
- Render-to-tensor, custom shaders, picking (GEOMETRY.md invariants).
