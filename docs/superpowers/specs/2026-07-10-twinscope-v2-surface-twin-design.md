# TwinScope v2 — the surface twin (R2 exemplar redirect)

**Date:** 2026-07-10
**Status:** implemented (feat/geometry-v1_2; Metal hardware pass pending). Supersedes §1 (the applet) of
`2026-07-10-twinscope-twin-exemplar-design.md`; that doc's §3 (R2 contract),
§4 (R3 contract), and §11-derived invariants remain binding and unchanged.
The v1_2 implementation contract
(`2026-07-10-geometry-v1_2-textured-mesh-design.md`) remains binding and
unchanged — the ABI is identical.
**Donor:** branch `codex/twinscope-implementation` @ `2bb13c5` — a complete
but big-bang R2 implementation, reviewed 2026-07-10 (6 confirmed findings,
4 refuted; conformance audit largely MET; verification bar incomplete).
Implementers TRANSCRIBE from the donor wherever it survived review — do not
re-derive working code. This branch exists to (a) rebuild incrementally with
per-task review gates, (b) fold every confirmed fix in at construction time,
and (c) replace the exemplar's demo direction, which read as a texture-space
screensaver, not a twin.
**Brainstormed:** 2026-07-10 with Ahmed. Decision: upgraded thermal twin now
(this doc); an LLM-internals scope consuming R2/R3 follows later (out of
scope here, recorded in §10).

## One-line

A finned heatsink housing whose heat flows *on the actual surface* — solved
on a subdivided sim mesh by batched torch sparse ops, draped on the coarse
render mesh at texture resolution (R2), while a small net learns the field
and visibly chases it — with the libtorch↔Vulkan fusion itself on screen:
split physics|belief view, live textured↔per-vertex toggle, zero-copy
provenance, uncapped sim/train counters.

## 1. Why the redirect (what v1 got wrong as a demo)

- v1 diffused heat in flat UV space: motion ignored the 3-D surface, seams
  meant nothing, fins had no thermal role — "nonsensical evolution."
- The committed housing was ~120 triangles — "badly made figure" — and
  shrank the texture-vs-vertex resolution gap that justifies R2.
- The net converged in ~a minute and the drama ended; nothing on screen
  showed WHY textures-on-meshes or zero-copy mattered.

v2 keeps v1's load-bearing choices (R2 forcing function, jobs.v1 worker,
tensor-bridge texture path, degradation ladder, OBJ loader) and replaces the
physics, the asset, and the presentation.

## 2. The subject and the resolution story

- **Render mesh:** committed OBJ heatsink housing, ~2,500-4,000 triangles,
  dense fin array, `vt` UVs (charted; gutters ≥ 4 texels), `vn` normals.
  Procedurally generated; the generator is NOT shipped, the OBJ is the
  artifact (v1 rule kept). Replaces the donor asset.
- **Sim mesh:** the render mesh midpoint-subdivided ×2 at init
  (V_sim ≈ 16× V_render ≈ 40-60k vertices). Midpoint subdivision keeps the
  original vertices as the first V_render entries — the per-vertex fallback
  reads `state[:, :V_render]` for free.
- **Field:** temperature lives on sim-mesh vertices, `(B, V_sim)` f32,
  B = 50 boundary-condition variants, one batched step. The hero texture is
  `(H, W) = (256, 256)`.
- **The R2 argument, now visible:** the render mesh is deliberately coarse;
  the field genuinely carries ~16× more spatial detail than the render
  vertices can. The textured↔per-vertex toggle (§6) shows exactly that gap.

## 3. The physics (surface-aware, still honest)

Explicit heat step on the sim mesh, pure batched torch ops on the worker:

- **Operator:** cotangent-weighted graph Laplacian `L` (sparse, V_sim²,
  ~7 nnz/row) with Voronoi-third vertex masses `M` (per-vertex area =
  1/3 · sum of incident triangle areas). Built once at init from the
  subdivided mesh, on CPU, moved to the training device as
  `torch.sparse_csr`.
- **Step:** `T ← T + dt·M⁻¹·(−κ·L·T + inject(s) − h·A·(T − T_amb))` where
  `inject` deposits per-variant source intensities `(B, K)` as Gaussian
  bumps (3-D Euclidean distance, radius ≪ feature size) around K = 4 fixed
  3-D source sites (bolt bosses + core), and the loss term is area-weighted
  (`A` = vertex mass) so FINS VISIBLY SHED HEAT — the geometry is thermally
  load-bearing. `dt` fixed and stable for the operator's spectral bound
  (explicit scheme; dt chosen from the max row sum at init, ×0.9 safety).
  (Clarified after T7: with the lumped mass, `M⁻¹·h·A` reduces to uniform
  Newton cooling per vertex — the fins' thermal role is carried by the
  diffusion operator and mass-weighted geometry, not by a per-vertex loss
  coefficient. Synthetic, never FEA — unchanged.)
- **Batched:** one `sparse.mm` per step services all 50 variants
  (`L @ Tᵀ` layout). Sim steps uncapped; publish at ≤30 Hz (v1 pattern).
- **Honesty line unchanged:** UI and docs say "synthetic surface heat
  field," never FEA. The twin claim is the dataflow.
- **Drama:** sources duty-cycle on a fixed seeded schedule (each source has
  an on/off period, a few seconds, mutually offset), so the field never
  settles and the net chases forever. User interaction (§7) overrides the
  schedule for the touched source.

## 4. The learner

- MLP `f_θ(x, y, z, s_1..s_K) → T` — 3-D position input (seam-free;
  UV inputs would alias across charts), K = 4 source intensities.
  mesh_scope-scale net (3+K → 64 → 64 → 1), Adam; trained every step on
  random sim-vertex samples across ALL variants, one batched fwd/bwd.
- Displayed quantities (hero): **sim** (bake of the state), **net**
  (f_θ evaluated at texel 3-D positions — note: at TEXTURE resolution,
  finer than the render mesh), **|error|**. All `(H, W)` f32 through the
  same texture path.
- Per-learner seeding must not perturb process-global RNG the way the donor
  did (`torch::manual_seed` in reset — review Minor): seed via a local
  `torch::Generator` for sampling, and accept deterministic-enough init.

## 5. The bake (state → texture, precomputed)

At init, rasterize the UV atlas at 256²: for each inside-chart texel, the
covering triangle and barycentric weights → a sparse bake matrix
`S (H·W × V_sim, 3 nnz/row)`. Per publish: `tex = S @ T[hero]` (one sparse
mv). Outside-chart texels get a precomputed gutter map (index of nearest
inside texel, `index_select` at publish) so bilinear/clamp sampling never
bleeds garbage. The rasterizer is a small header-only helper next to the
OBJ loader, unit-tested (partition of unity: each inside texel's weights
sum to 1; coverage: every triangle whose UV extent is ≥1 texel owns ≥1
texel at 256². Sub-texel UV slivers — thin fin-thickness faces — cannot be
owned by any texel-center rasterizer (Nyquist); they are permitted and must
be neighbor-filled via the gutter/partition-of-unity path. Amended
2026-07-10 after T6 measured 445/3184 sliver triangles on the committed
asset; the original "every nonzero-UV-area triangle" wording was
unsatisfiable at 256².)

## 6. The views (what the demo shows)

One geometry view texture, applet-owned orbit camera (shared
`orbit_camera.hpp`, §8.f):

- **Split hero (default):** the housing drawn TWICE side by side (two
  draws, model matrices offset ±x): left = sim texture, right = net
  texture. The net side visibly lags and sharpens as it learns — physics
  vs. belief in one glance. A third mode collapses to a single mesh with
  the sim/net/|error| radio (donor behavior).
- **The R2 toggle:** toolbar switch "textured (v1.2)" ↔ "per-vertex
  (v1.1)". Per-vertex mode draws the SAME field sampled at the render
  vertices (`state[:, :V_render]` through COLORMAP) — the visible blur vs.
  the 256² drape IS the R2 justification, on screen. Toggle available only
  when both caps present; ladder (§9) otherwise.
- **The fusion HUD:** status line (zero-copy provenance, flow_scope
  discipline: claimed only when that path drew, 1-frame-lag reporting
  pattern), sim steps/s and train steps/s counters (uncapped, on the
  worker), loss sparkline, and the MAGMA legend with actual °C range.
- Lambert on, wireframe overlay optional (donor pattern). Depth on, OPAQUE.

## 7. Interaction

Click a source site on either mesh half → toggle it; drag → scale intensity
(hero variant only). Ray cast applet-side against the RENDER mesh
(donor Möller–Trumbore, transcribed), hit point → nearest 3-D source site —
sites defined once in twin_model (v1 review finding: the donor duplicated
site constants in two files; v2 has exactly one definition). Events queue
to the worker via atomics (donor pattern); a touched source leaves the duty
cycle and holds the user's setting. Torch never runs on the frame thread.

## 8. Review fixes folded in at construction (the donor's confirmed findings)

a. **SDK widening** (donor fix `2bb13c5`, transcribe): v1.1-shaped draws
   widen to zero-tailed 216-byte records on a v1_2-only host; regression
   test rides along.
b. **Frame-thread sync:** `stream_to_tensor` in any `draw_ui` is gated on
   `CALIPER_BRIDGE_CAP_STREAM_ORDERED`; cap absent → CPU-staged rung with
   an honest status line. Adapter gets the doc contract. gpt_scope's
   identical call site gets the same gate (small cross-applet fix, rides
   the applet task).
c. **Asset path:** housing.obj resolves relative to the applet DLL (staged
   copy) with the source-tree macro as dev fallback; the CMake staging step
   finally has a consumer.
d. **Mode-tagged publishes:** the worker tags each publish with the mode
   that selected it; the frame thread never maps mode-A data through
   mode-B's LUT range (donor flash bug).
e. **Metal parity:** the Metal textured gate refuses uv_base > 32 bits and
   refuses sampling ANY geometry view (matching Vulkan) from day one.
f. **Cleanup class:** shared `sdk/include/caliper/adapters/orbit_camera.hpp`
   (one eye computation for draw AND pick); single-source site constants;
   1-sync publish (one `.to(kCPU)`, peak computed CPU-side); no hot-loop
   sleep (idle branch only); direct store protocol for source widgets;
   `geom.vert` params comment says 176 and names the three synced copies;
   host validation takes one revision axis (not stride+bool);
   CMake shader-compile and torch-test helper functions.

## 9. Degradation ladder (v1 §1.7 kept verbatim, plus one rung)

| Missing | Behavior | Status line |
|---|---|---|
| R2 cap | per-vertex COLORMAP on the render mesh (v1_1), toggle disabled | "per-vertex fallback (no textured geometry)" |
| STREAM_ORDERED | textured path continues via CPU-staged texture updates | "CPU-staged texture (no stream-ordered handoff)" |
| primitives cap | ImPlot heatmap of the hero texture, 2-D | mesh_scope ladder verbatim |
| import/pool | v1 CPU-staged updates | never a wrong image |

Additionally (donor PLAUSIBLE finding): if the textured draw is refused at
runtime while the per-vertex rung is viable, fall to per-vertex, not to the
heatmap — the rung selection re-evaluates on draw failure instead of
snapshotting caps once.

## 10. Verification bar

- **Host/ABI:** donor's abi/bridge tests transcribed (offsets, caps, gates,
  v1.1-tail isolation, mixed multi-draw) PLUS the widening regression (§8.a).
- **gfx rows (both backends; Vulkan runs on this box, Metal twins
  transcribed, hardware pending macOS):** donor's texel-center/bilinear/
  Lambert/uv-offset rows PLUS the four the donor left out — clamp-to-edge
  with out-of-range UVs; v1.1 draw == v1.2 zero-tail non-textured draw
  byte-identical; gate refusals leave pixels bit-for-bit untouched
  (incl. released-UV-alloc); short-stride refusal.
- **Engine tests (pure torch, the TDD core of v2):** subdivision (counts,
  original-vertex prefix, manifold edges); Laplacian (symmetry, zero row
  sum, PSD via smallest-eigenvalue probe on a small mesh); bake matrix
  (partition of unity, coverage); analytic steady state — flat strip mesh,
  two pinned ends → linear profile within 2%; energy decay under zero
  sources; batched B=50 step == 50 independent steps; learner loss
  decreases on a fixed run, prediction finite/in-bounds.
- **Loader:** donor tests transcribed against the NEW asset (counts,
  bounds, UV range, dedup on the (v,vt,vn) triple, malformed set).
- Applet UI: no pixel tests (v1 rule); log-based run proof both render
  paths (ledger lesson: no GUI clicking).

## 11. Out of scope (named)

- **R3 / the fleet** — next pass, own spec (parent doc §4 contract stands).
  v2's split view is designed so the fleet replaces the right half's slot
  cleanly later.
- **The LLM-internals consumer scope** (attention fields via R2, fleet of
  heads via R3) — follow-on after R3; Fable-authored end to end.
- Telemetry ingestion, PBR/shadows, picking-as-ABI, render-to-tensor, glTF,
  per-instance textures — all v1 rejections stand.
- Migrating the other five applets to `orbit_camera.hpp` — deferred.

## 12. Sequencing

1. Host/ABI + SDK (donor transcription + fixes) → 2. Vulkan backend →
3. Metal backend (parallel-safe with 2: disjoint files) → 4. gfx rows →
5. OBJ loader + new asset → 6. surface engine (subdivision/Laplacian/bake,
parallel-safe with 5 once the loader interface is pinned) → 7. thermal
model v2 + learner → 8. applet v2 → 9. docs/polish + final whole-branch
review. Implementation protocol: subagent-driven, Opus implementers,
transcribe-from-donor briefs, per-task review gates, ledger discipline.
