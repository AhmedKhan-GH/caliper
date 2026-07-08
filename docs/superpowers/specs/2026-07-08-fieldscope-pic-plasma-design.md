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
each step the particles deposit charge onto a grid, a **free-space (open-boundary)**
Poisson FFT solve produces the self-consistent field of the isolated cloud, that field
is gathered back and a Boris pusher advances them under it plus a strong background **B**
and a soft axial trap. In that **magnetized** regime the transverse motion is E×B drift,
so the cloud behaves as a **2-D vortex fluid**: a handful of charge clumps **orbit their
common centre and merge** (a real vortex-merger), a hollow ring breaks into a rotating
necklace (the *diocotron instability*) — genuine emergent structure, not a spring
breathing about its centre of mass (the failure mode of the plain harmonic-trap version).
The cloud **floats freely in open space** — like SculptScope's cloud, not a box.
Positions render zero-copy (pool-born tensor drawn in place by `geometry.v1`), by speed.

**Design note (open vs periodic).** An earlier draft used a *periodic* solver with a
neutralising background (a one-component plasma). That is mathematically a **cube**: the
background fills the box and the cloud spreads to tile it. To get a free-floating cloud
the solver must be **open-boundary** (free-space Green's function, Hockney's doubled-grid
FFT) and the plasma **single-species** (like charges repel, so a harmonic trap yields a
stable floating equilibrium instead of the two-species attractive collapse that heats a
coarse grid to blow-up). Both were found during implementation and verified headless.

**Third fix (dynamics).** A *harmonic* trap is a linear spring: a repulsive cloud in it
only breathes about its centre of mass — no structure. The interesting regime is
**magnetized** (strong B, trap acting mainly in z): the transverse E×B dynamics are 2-D
vortex dynamics, where clumps orbit and merge and rings go diocotron-unstable. Verified
headless with an azimuthal-clumpiness metric: 4 clumps persist (clumpiness ≈ 2.6–2.9)
for ~15 s, then merge into a ring, with KE **bounded** throughout (≈[0.3k, 18k], no
blow-up). Grid resolution (`G=48`) and a finite slab thickness were needed to stop the
under-resolved thin disk from finite-grid heating.

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

Solver domain `[0,L)³`, `L = 10`, grid `G³` (`G = 32`). The cloud is **single-species**
(charge `+1`), floats near the box centre, and is confined by a soft harmonic trap. The
FFT Poisson solve runs on CPU (grid tiny; robust across backends incl. MPS whose FFT
support is spotty); the `N ≈ 200k` scatter/gather stay on the GPU. A tunable `coupling`
scales the total space charge (`E ×= coupling/N`); a tunable `trap` sets the confinement.

- **`deposit_cic(pos, G, L, charge) → rho (G,G,G)`** — Cloud-In-Cell: each particle adds
  `charge` split across its 8 surrounding nodes (`∏(1−f or f)` weights) via `index_add_`.
  Weights partition unity → total charge conserved.
- **`poisson_E_free(rho, L) → E (G,G,G,3)`** — **free-space (open-boundary)** Poisson by
  Hockney's doubled-grid method: `φ = rho (*) G_free`, the *linear* convolution with the
  free-space Green's function `G_free(r)=1/(4π r)`, computed as a cyclic convolution on a
  `2G` grid with `rho` zero-padded into one octant (Green's-function FFT cached; self-cell
  softened). `E = −∇φ` by central differences. This is the field of the *isolated* cloud,
  decaying at infinity — no periodic images, no cube. (The periodic `poisson_E` still
  exists and is unit-tested, but the applet uses the free-space solver.)
- **`gather_cic(E, pos, L) → (N,3)`** — same 8-node CIC weights, `Σ w·E[node]`.
- **`boris_push(pos, vel, accel, charge, B, dt)`** — half kick with the charge-inclusive
  acceleration `accel` (`= charge·E + trap + impulse`), magnetic rotation driven by
  per-particle `charge` (`t=charge·B·dt/2`, `s=2t/(1+|t|²)`, `v⁻→v'→v⁺`), half kick;
  `pos += vel·dt`. **Open** — no wrap; the trap keeps the cloud inside the grid.

### Initial conditions — `init_state(kind, N, L, temp, v0, dev) → (pos, vel, charge)`
All free-floating blobs centred at `L/2` (single species). Selector mirroring
SculptScope's shape combo:
- **`kBlob`** — a warm Gaussian clump (settles into a breathing trapped ball).
- **`kTwoStream`** — one blob, halves counter-streaming ±v₀ x̂ (beam dynamics).
- **`kSphere`** — a spherical shell.
- **`kRing`** — a spinning ring (striking under background B).

### Data flow (per worker step)
```
rho   ← deposit_cic(pos, G, L, charge)      # 200k particles → 32³ grid (scatter)
E     ← poisson_E_free(rho, L) * coupling/N # free-space FFT Poisson: cloud's OWN field
accel ← charge*gather_cic(E,pos) − trap*(pos−centre) [+ cursor impulse]
boris_push(pos, vel, accel, charge, B, dt)  # self-field + trap + optional bg B
speed ← ‖vel‖                               # colour channel
sync(); publish(write)                      # renderer imports pos[write] in place
```
`E`/`rho` are grid-sized (default allocator); only `pos`/`speed` slots are pool-born.

## Interaction
Right-drag orbit, wheel zoom. **Initial-condition combo** (re-seeds). Left-drag →
cursor-ray impulse (an external, charge-independent perturbation). Sliders: **charge**
(space-charge `coupling`), **trap** (confinement), **B** (background field),
**temperature** (on re-seed), **color**. Status line reports
`N particles · zero-copy (imported geometry) · <IC> · KE <energy> · N steps/s`.

## Testing (physics invariants — real tests, not change-detectors)
`tests/test_em_pic.cpp` (pure torch, CPU, `REQUIRE` — c10 shadows doctest's `CHECK`):
1. **Periodic Poisson solve is correct.** For `ρ = cos(k·x)`, the solved `E` matches the
   analytic `E = −∇φ`, `φ = ρ/|k|²`, to tolerance.
2. **Free-space Poisson gives `1/r`.** A unit point charge produces `φ(r) ≈ 1/(4π r)` at
   several radii (validates the Hockney free-space Green's function — the applet's solver).
3. **CIC conserves charge.** Unsigned deposit sums to `N`; a signed (neutral) set deposits
   ≈ zero net charge.
4. **Boris gyration.** In uniform B (E=0) a particle traces a circle at `ω = qB/m`,
   kinetic energy conserved over a full orbit to tolerance.
5. **E×B drift.** Under crossed uniform E,B the guiding-centre drift matches `E×B/|B|²`.
6. **Momentum conservation.** A closed system with no external force conserves total
   momentum over many steps (Newton's third law) to tolerance.

Headless verification (`CALIPER_AUTOLAUNCH`): zero-copy pool + draw on MPS, and KE stays
**bounded** (a stable breathing oscillation, ~[3k,17k] over 20 s) — no blow-up.

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
- Two-species neutral plasma (electrons + ions) — the attractive close-encounter collapse
  heats a coarse grid to blow-up (found in testing); a single-species trapped cloud is the
  stable, free-floating choice here. (A finer grid + smoothing could revisit this later.)
- Collisions / higher-order shape functions (TSC) — CIC is the standard baseline.
- The learned RL controller (the superseded 2026-07-07 spec) — this is a simulation.
- Render-to-tensor, custom shaders, picking (GEOMETRY.md invariants).
