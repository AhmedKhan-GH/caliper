# Caliper framework — remaining internal work

**Date:** 2026-07-10
**Status:** planning spec (framework-internal). Enumerates the engineering
still owed by the *framework itself* — capability rungs, cross-platform
closeout, and tech debt — to reach the roadmap's stated goal. Deliberately
**excludes external applications** (the voltage classifier, any research
applet): those consume the ladder, they don't extend it, and each gets its
own application note. See `docs/superpowers/specs/2026-07-10-voltage-classifier-caliper-application.md`
for how one such consumer maps onto the rungs below.
**Authority:** `ROADMAP.md` (checkbox discipline — an item is checked only
when verified by artifacts), `GEOMETRY.md §11` (the goal and the no-new-ABI
list), the R3 requirement contract in
`docs/superpowers/specs/2026-07-10-twinscope-twin-exemplar-design.md §4`.
**Checkbox discipline (inherited):** a box is checked only when suites are
green, the path is run-proven, and the commit is named. Invariants at the
bottom never become checkboxes.

## 0. Where the framework stands (the ladder)

The goal is *complex simulations and digital twins, trained on, watched
live* — grown through a ladder of **nouns** the instrument draws live, while
the **verbs** never change (zero-copy, gated, byte-verified against one CPU
reference, honestly degraded):

```
images → point clouds → connected shapes → state-on-shapes → populations → other hosts
(bridge)    (v1)          (v1_1) ✅          (R2/v1_2) ✅        (R3) ▢        (R4) ▢
```

- **Shipped and merged:** platform spine, `geometry.v1` (points), `v1_1`
  (connected shapes) byte-exact on Metal **and** Vulkan, the MeshScope
  exemplar, whitepaper v0.2.
- **On `feat/geometry-v1_2` (final review: READY TO MERGE):** R2
  `geometry.v1_2` (textures-on-meshes) — both backends; Vulkan run-proven
  byte-exact, Metal run-proven byte-exact on Apple Silicon (macOS hardware
  pass 2026-07-10). TwinScope v2 surface-twin exemplar, run-proven zero-copy
  on Vulkan+CUDA and Metal/MPS, honest GL fallback.
- **Not started:** R3 (populations), R4 (second host), and the strategic
  decisions in §4.

This doc covers everything from "merge the R2 branch" through "R3 shipped,"
plus the tech debt the R2 build ledgered, plus the internal strategic
decisions that gate later rungs. It stops at the framework boundary.

---

## 1. R2 closeout (finish what the branch started)

R2 is functionally done; these are the steps that make it *true on the
roadmap* rather than true on this one machine.

- [ ] **1.1 Merge `feat/geometry-v1_2` → main**, caps-gated, no-ff (house
  pattern). Final whole-branch review returned READY TO MERGE, 0
  Critical/0 Important across all six cross-task seams; full ctest 8/8 at
  every task boundary. Precondition: none technical — this is the human's
  call. After merge: delete the branch; the donor
  `codex/twinscope-implementation` may be archived or deleted (it served its
  transcription purpose).
- [x] **1.2 macOS hardware pass** *(done 2026-07-10, Apple Silicon Metal/MPS —
  commits `5164f99` compile-gate test fix, `c92da48` DeviceSparse MPS carrier,
  `f885479` §6.1 log tag, `edddb26` §6.2 varying-integrity row;
  `caliper_gfx_tests` 43/43 on live Metal, full suite 8/8 ctest, TwinScope v2
  run-proven zero-copy on MPS. Recorded in the `progress.md` MAC HARDWARE PASS
  section, which stands in for the never-transferred Windows MAC-PENDING
  ledger.)* (the recorded platform protocol; v1_1 ran
  this direction in reverse). The MAC-PENDING CHECKLIST in
  `.superpowers/sdd/progress.md` is the acceptance list:
  - Metal build of `feat/geometry-v1_2` (the `.mm` never compiled on the
    Windows box).
  - T3 parity gates fire on hardware: `uv_base` 32-bit refusal; the
    render-target-view sampling refusal (`MTLTextureUsageRenderTarget`
    marker).
  - The four Metal gfx twin cases from T4 (clamp-to-edge, v1.1/v1.2 compat,
    refusal purity) pass byte-exact, plus the donor-row Metal twins.
  - TwinScope v2 runs on Metal/MPS — **verify device selection**: the
    thermal model / applet assume CUDA-or-CPU; confirm the MPS pick and the
    STREAM_ORDERED path (Metal always reports the cap true).
  - Fix the T3-ledgered cosmetic on hardware: the Metal refusal log
    double-prefix `geom_prims: primitives:` (drop the redundant category
    tag), and confirm `geom_tex_fs`'s `VOut` stage-in is never reached by a
    textured POINT draw (unexercised; document or guard).
- [ ] **1.3 Reconcile ROADMAP §6 checkboxes.** The roadmap text predates this
  branch: tick the twin-exemplar design doc and the OBJ-loader items (both
  shipped on the branch). "Twin applet ships run-proven on both platforms" is
  now ticked — the macOS hardware pass (§1.2) run-proved TwinScope v2 zero-copy
  on Metal/MPS, so both platforms are proven independent of R3. **R3** stays
  unchecked until §2 lands; this box stays open until then.

---

## 2. R3 — instanced transforms (the next capability rung)

The last rung of the twin story: **populations of shapes.** One mesh × an
imported `(N,16)` pose tensor = a fleet in one draw, moved per frame by
whatever writes the tensor. This is the rung that turns TwinScope's *hidden*
50-variant batch into fifty housings on screen, and the roadmap's own rule is
that **R2+R3 together ARE the twin demo**. It is gated on demonstrated applet
need — TwinScope already provides it (the fleet the split view stands in for).

**Protocol:** the v1_1 / v1_2 discipline verbatim — own spec pass →
implement (Metal + Vulkan, byte-exact §9.2-style rows) → run-proven applet →
docs. Subagent-driven, Opus implementers, per-task review gates, incremental
commits on a fresh `feat/geometry-v1_3` (or combined-name — the spec pass's
call, see 2.1).

- [ ] **2.1 R3 spec pass** — write the implementation contract (this doc pins
  the requirement contract; the spec pass makes it mechanical). From the
  parent exemplar §4:
  - **Pure struct growth**, same mechanism R2 used: appended draw fields
    carried by `draw_stride`, `reserved0` stays NULL, no new entry points.
    Fields: `instance_alloc`, `instance_offset`, `instance_count` —
    `(N,16)` f32 column-major model matrices, imported; and optional
    `instance_attr_alloc`, `instance_attr_offset` — `(N,)` f32 whole-instance
    value through the draw's existing LUT (`colormap`/`vmin`/`vmax` reused).
    Per-instance attr is **decided in**, not deferred (pose-only fleets
    immediately want state).
  - **Naming decision:** `CaliperGeomDrawV1_3` = `V1_2` + instance tail (new
    stride), vs. a combined revision. Either way the contracts are
    stride-additive and prefix-identical. **Fold in the single-axis
    refactor** the R2 hardening already prepared: the shared host validator
    now takes one `bool v12` axis (deriving min_stride + color-mode ceiling)
    — R3 should generalize that to an **enum revision axis** (`v1_1 / v1_2 /
    v1_3`), NOT add a second bool. The reviewed defect this replaced was
    exactly a two-bool encoding; do not reintroduce it.
  - **Semantics:** effective transform = draw `model` × instance matrix (host
    premultiplies proj·view as today; instance matrix pulled by instance
    index in the vertex shader). `instance_count == 0` or
    `instance_alloc == 0` → today's non-instanced path (additive default).
    Instanced + LAMBERT: normal matrix from the combined transform — the spec
    pass decides shader-side derivation vs. a second per-instance stream,
    **against the byte-exactness bar**. `instance_attr` present → overrides
    the vertex color source with the instance tint (COLORMAP semantics);
    absent → vertex coloring as today.
  - **Caps bit 3.** Gates (the §2.3 pattern extended): alignment; `N*64` and
    `N*4` bounds; `N > 0` when an alloc is present; whole-frame refusal;
    backend re-gate for parity (Metal↔Vulkan, binding).
  - **REJECTED, not deferred:** per-instance textures (texture-array/atlas
    ABI). The hero carries the draped field (R2); the fleet carries scalar
    state (R3). A future need brings its own exemplar.
- [ ] **2.2 Host/ABI + SDK** — the `V1_3` record (or combined), the enum
  revision axis in the shared validator, caps bit 3, gate battery in
  `test_tensor_bridge.cpp`, ABI offset pins + a widening/compat regression in
  `test_abi.cpp` (the SDK wrapper must keep old-stride draws working — the
  v1_2 widening precedent).
- [ ] **2.3 Vulkan backend** — `vkCmdDraw(consumed, N, 0, 0)`, instance
  matrix pulled in `geom.vert` by `gl_InstanceIndex`, the LAMBERT
  normal-matrix decision from 2.1, the instance-attr LUT path. Both backends
  already draw instanced *points* in v1 — this generalizes the mechanism, it
  does not invent it. `PrimParams` growth stays synced across the three
  copies (GLSL / Vulkan / Metal — the comment named them in R2).
- [ ] **2.4 Metal backend** — `drawPrimitives:...instanceCount:N`, MSL
  instance pull, day-one gate parity (transcription on the Windows box;
  hardware pass folds into the next macOS session — do not claim verified).
- [ ] **2.5 gfx §9.2 rows (both backends)** — byte-exact against one CPU
  reference: pose-only fleet at `N` distinct transforms; per-instance-attr
  tint through the LUT; `instance_count==0` == non-instanced draw
  byte-identical (the additive-default compat row); gate refusals leave
  pixels untouched (bad alignment, `N*64` overflow, `N==0`-with-alloc,
  released instance alloc). Vulkan runs on the box; Metal twins transcribed.
- [ ] **2.6 TwinScope fleet** — replace the split view's stand-in with the
  real fleet: the 50 variants the sim already computes, drawn as one
  instanced draw, each unit tinted by its live peak-T or peak-|error| via the
  instance-attr path; click-to-promote a fleet unit to the hero slot. The
  applet already batches all 50 variants (T7) — this is the draw path, not
  new physics. Run-proven both platforms (Vulkan here; Metal in the mac
  pass). This is the flagship demo the roadmap §6 names.
- [ ] **2.7 Docs** — GEOMETRY.md status (R3 shipped), ROADMAP §6 ticks,
  ZEROCOPY.md instanced row, whitepaper figure candidate.

**Definition of done for the twin story:** with R3 merged and run-proven both
platforms, the roadmap's §6 ("R2+R3 together ARE the twin demo") is complete
and the framework has reached its stated goal on the 2-manifold ladder.

---

## 3. Framework hygiene (tech debt ledgered during the R2 build)

Non-blocking, but framework-internal and worth clearing before or alongside
R3 so they don't compound. Each was surfaced and triaged during the
`feat/geometry-v1_2` reviews (`.superpowers/sdd/progress.md`).

- [x] **3.1 Ungated frame-thread `stream_to_tensor` sweep.** *Done — commit
  `14e143a`: audit found exactly four call sites (all frame-thread, no worker
  callers); the two ungated ones (gpt_scope `upload_mapped`, embed_scope
  `update_or_create`) now gate on STREAM_ORDERED with CPU-staged fallbacks
  matching the R2 pattern. ctest 8/8.* R2 gated
  twin_scope and gpt_scope's draw-path calls on `STREAM_ORDERED`, but the
  final review found **two pre-existing ungated callers**:
  `applets/gpt_scope/gpt_scope.cpp:445` (`upload_mapped`) and
  `applets/embed_scope/embed_scope.cpp:1014`. Same freeze-risk class the
  branch fixed at its assigned sites. Audit every frame-thread
  `stream_to_tensor` caller across applets; gate each on the cap with the CPU-
  staged fallback. This is the codebase-wide close of the discipline the
  postmortem established.
- [x] **3.2 Per-vertex attr path handshake (latent, codebase-wide).** *Done —
  commits `bfe30e1` + `335d0b1`. Verdict: safe by construction via TWO
  invariants — temporal (worker drains its device before flipping
  `ready_slot`) + spatial (triple-buffer rotation); the STREAM_ORDERED gate is
  structurally impossible on the geometry ABI (no stream channel) and
  unnecessary given the drain. The audit falsified the temporal half at three
  sites (twin_scope every publish; flow/field initial publish) — drains
  enforced there, so the documented rule now holds at every publish site.
  Contract written into `geometry_v1.h`/`geometry_v1_1.h` headers +
  ZEROCOPY.md/GEOMETRY.md — R3's pose/attr publish path inherits it. ctest
  8/8; twin/flow/field run-proven live post-fix. Note for §1.2: the new
  twin_scope MPS drain branch is compile-verified on Windows only.* The
  final review noted the per-vertex COLORMAP draw path writes device tensors
  from the worker with **no** STREAM_ORDERED handshake (only the *texture*
  path is gated). Empirically clean in every run-proof, but it is a latent
  synchronization gap in a shared pattern. Decide: is the point/vertex attr
  path safe by construction (the pool's triple-buffer stability contract), or
  does it need the same gate? Document the answer where the contract lives;
  gate if the answer is "needs it."
- [x] **3.3 Optional test-coverage tightening** *(done — commit `a087b03`: all
  four tightened as spec-derived pins, none accepted-out; per-vertex masses =
  incident-area/3, midpoint UV = parent-edge mean, exact OBJ counts
  3184/2430, 6/256-uv gutter pin. ctest 8/8.)* Original items:
  surface-engine `vertex_masses` per-vertex distribution (currently sum-only);
  midpoint-UV-mean assertion in subdivision; asset test exact-count pins
  (currently ranges); inter-chart gutter-width pin in the bake test. None
  block; batch into the next test-touching task if convenient.
- [x] **3.4 TwinScope efficiency/honesty minors** *(done — commit `35b2661`:
  provenance now per-half — zero-copy claimed only when every drawn half
  imported; publish gated on `sim_on || train_on` with drain-before-publish
  preserved. M1 + startup run-proven live; the idle-off branch is
  reading-verified only — headless runs can't toggle the checkboxes.)*
  Original items (M1/M2 from the T8b review):
  split-view provenance ORs both halves (over-claims zero-copy when one half
  falls back — make it per-draw); the idle worker still publishes + syncs
  every 33 ms when both sim and train are off (gate the publish on
  `sim_on || train_on`). Cosmetic; fold into an applet-polish pass.

---

## 4. Internal strategic decisions (decisions before code)

These are framework-positioning calls the roadmap §7 lists. They are
**decisions**, not scheduled engineering — but they are internal (they shape
the framework, not any one applet), so they belong in this doc. None should be
coded before the decision is made.

- [ ] **4.1 Telemetry-ingestion decision.** Physical twins (and any live
  external feed) need data *into* a published tensor. Two shapes: the
  **feed-applet pattern** (an applet owns the ingest on its worker, no new
  service) vs. a **new ingestion service**. Touches the whitepaper's
  local-loop claim, so it is a positioning call, not an engineering default.
  *Boundary note:* the ingest of external data is where "framework-internal"
  meets "external application" — the **decision** is internal; a specific
  feed is external. Resolve the pattern here; leave feeds to their apps.
- [ ] **4.2 R4 / `libcaliper` second host** — the top of the ladder ("other
  people's hosts"). Platform call, not geometry's; intent lives at
  `PLATFORM.md` (Compass, the wx sibling host). No spec by design until the
  call is made. Sequenced after the twin story (R3) completes — a second host
  that draws an incomplete ladder is premature.
- [ ] **4.3 Optional second flagship: em-controller** (actor-critic RL
  steering the PIC plasma; design retained in specs, never implemented). Not a
  capability rung — an exemplar that would exercise the existing ladder from
  the control/RL direction. Slot after the twin exemplar if a second flagship
  earns its cost; otherwise leave in specs.

---

## 5. Conditional rungs (gated on demonstrated need — NOT scheduled)

Named so nobody schedules them prematurely. The ladder grows **only** against
a demonstrated applet need, byte-exact-verified on both ecosystems, as
prefix-identical additive revisions. These have no such need *yet*.

- **Volumetric / voxel-field primitive.** The entire geometry ladder is
  2-manifold (points → surfaces → textures-on-surfaces → instanced surfaces).
  A scalar field defined through a **solid** (e.g. transmural propagation
  through an organ wall) is not on it — drawing it means volume rendering
  (ray-marched 3-D texture, transfer functions, sampling *through* the
  object), a genuinely new rung, not a re-skin. This is currently the leading
  candidate for a future rung **iff** an external consumer demonstrates it is
  scientifically required (the voltage-classifier note §8 flags exactly this
  fork). Until a consumer's design doc forces it, it stays here: named, not
  built. It would be a capability increment comparable to R2/R3 in scope.
- **Anything requiring render-to-tensor, applet-supplied shaders, or a
  non-additive ABI break** — forbidden by the invariants, not conditional.
  These never become rungs.

---

## Invariants (hold forever, never relitigate — restated from ROADMAP.md)

- Data flows **tensors → pixels → ImGui**, one way. No render-to-tensor, ever.
- No applet-supplied shaders; appearance is the fixed menu.
- Honest ladders: a missing capability degrades to a working slow path and
  says so; never a wrong image, never a false status line.
- Increments ship against a demonstrated applet need, byte-exact-verified on
  both ecosystems, as prefix-identical additive revisions.

---

## Sequencing (framework-internal)

1. **§1 R2 closeout** — merge; then the macOS pass at the next mac session
   (§1.2 can lag the merge; it does not block R3 spec work, which is
   Windows/Vulkan-first).
2. **§2 R3** — the spec pass first (§2.1), then implement→verify→fleet→docs.
   This is the largest remaining engineering block and completes the twin
   story. Clear §3.1 (the sync sweep) before or during, since R3 adds another
   worker→frame publish path.
3. **§4 decisions** — resolve 4.1 (telemetry pattern) when a physical-data
   consumer is imminent; 4.2 (R4) only after R3; 4.3 (em-controller) optional.
4. **§5** — never, unless a consumer's design doc forces the volumetric rung.

**One line:** the framework is one merge and one rung (R3) from its stated
goal; everything past that is a positioning decision or a need-gated
conditional, not scheduled engineering.
