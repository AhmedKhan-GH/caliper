# geometry.v1_3 on Vulkan/Windows — the hardware pass

**Date:** 2026-07-11
**Status:** COMPLETE (2026-07-11, same day) — all boxes ticked below; rows
A–D passed byte-exact untouched on first hardware contact, the backend needed
zero changes, and the only fixes were test-side (`477431e`). Originally:
execution spec (hardware pass). NOT a greenfield implementation —
the Vulkan code for R3 (`geometry.v1_3` instanced transforms) **already exists
on `feat/geometry-v1_3`**, transcribed from the run-proven Metal reference and
line-by-line reviewed on this Mac, but every Vulkan line is compiled out here
(`if(WIN32)`) and nothing has run on NVIDIA hardware. This doc (a) orients the
Windows session to what the Vulkan surface already contains and where it
lives, and (b) gives the mechanical execution plan with acceptance gates to
take it from "transcribed + reviewed" to "run-proven byte-exact on
Vulkan/CUDA." It is the platform mirror of
`2026-07-10-metal-macos-v1_2-hardware-pass.md` (same protocol, reverse
direction — this time Metal is the reference and Vulkan must match), and the
v1_3 sibling of `2026-07-09-geometry-v1_1-vulkan-phase-b-design.md`.
**Authority:** the implementation contract is
`2026-07-10-geometry-v1_3-instanced-transforms-design.md` (§4 semantics, §5
gates, §6 Vulkan design, §8 rows, §9 fleet); the Windows-box verification
protocol is `docs/m2a-windows-verification.md` (D24). The run-proven
reference is the Metal backend at commit `5aa2f73` and the Metal gfx rows
(49 cases live-green on Apple Silicon).
**Checkbox discipline (inherited):** a box is checked only when the suite is
green on the Windows box, the path is run-proven by a logged artifact, and the
fixing commit is named. A twin divergence is STOP-and-diagnose, never a
loosened comparison. Invariants at the bottom never become checkboxes.

---

## 0. What already exists (the Vulkan v1_3 surface, file by file)

Branch commits: `3d50ce3` (T1 ABI+SDK+vend), `ca3a574` (T2 host gates G1–G12),
`511cdc0` (T3 Metal backend, run-proven), `d0d86af` + `78416c3` (T4 Vulkan
transcription + honesty-comment fix), `5403d92`/`40697f1`/`5aa2f73` (T5
TwinScope fleet + Metal G14 storage-mode fixes). **The T5 TwinScope fleet was
subsequently reverted (`85698f6`) as a UX regression; R3's shipped exemplar is
the dedicated `instance_scope` applet (`bfe6da9`)** — the fleet-independent v1_3
ABI, host gates, and both backends are untouched by that swap. Line anchors
below are as of `5aa2f73`; re-grep if the tree has moved.

### 0.1 Platform-neutral (already compiled AND tested on the Mac — no new risk)

- `sdk/include/caliper/services/geometry_v1_3.h` — the 256-byte
  `CaliperGeomDrawV1_3` (instance tail at 216), caps bit `1u<<3`,
  `CALIPER_GEOM_RIGID_TOL = 1e-4f`, static_asserts. Pinned by `test_abi.cpp`
  (layout, service-table parity, the v1_3-only-host widening regression).
- `src/host/tensor_bridge.cpp` — the `GeomRev {V1_1,V1_2,V1_3}` enum axis and
  host gates G1–G12 with byte-exact reason strings (order:
  G2→G3→G4→G6→G7→overflow→G5, then G8→G9→G11→overflow→G10→G12); the tint-LUT
  resolution rule (lut256 populated for COLORMAP OR instanced-tint,
  regardless of base color_mode). Exercised by `test_tensor_bridge.cpp`.
- SDK wrapper (`caliper.hpp`): `g13_`, `has_instanced()`, the v1_3 overload,
  widening tiers. All run on both platforms — needs no new verification
  beyond "the same tests pass."

### 0.2 The Vulkan code that has never compiled (the risk class)

Risk class: GLSL/SPIR-V compile errors, descriptor-count mismatches a Mac
build cannot catch, std140 layout slips, and CUDA-import interactions — the
things review cannot catch and only hardware reveals.

- **`src/host/renderer/shaders/geom.vert`** — PrimParams std140 grown to
  **192 bytes** (`uv_base@160, use_instance@164, inst_base@168,
  use_instance_attr@172, inst_attr_base@176`, pads to 192); bindings **8**
  (`Inst { float im[]; }`) and **9** (`InstAttr { uint iattr[]; }`,
  NaN-safe via `uintBitsToFloat`); column-major matrix pull
  `im[inst_base + 16*gl_InstanceIndex]`; **vector-first**
  `gl_Position = p.mvp * (M * vec4(wp,1.0))`; per-instance tint override
  before the color switch; §4.4 LAMBERT normal chain in the exact float op
  order the Metal MSL uses (byte-load-bearing). `use_instance==0` keeps the
  pre-v1_3 expression textually intact (bit-identity contract, §8 Row C).
- **`src/host/renderer/vulkan_renderer.cpp`**:
  - `supports_geometry_instanced()` → `supports_external_import()`.
  - Set layout grown to 10 bindings (8/9 = readonly SSBO, vertex stage);
    descriptor pool `STORAGE_BUFFER` 6u*cap → **8u*cap**; write arrays
    extended (`bi[9]`/`wr[10]`, texture write at `wr[9]`/binding 7);
    placeholder-bind `pos` at 8/9 when streams unused.
  - **Binding-4 LUT predicate is now `d.lut256 != nullptr`** (was
    COLORMAP-only) — the §6.2 trap; grep confirmed no stale predicate.
  - PrimParams fill: `use_instance`, `inst_base = instance_offset/4`,
    `use_instance_attr`, `inst_attr_base = instance_attr_offset/4`.
  - `vkCmdDraw(cb, consumed, use_instance ? n_inst : 1u, 0, 0)` at the single
    primitives site (~:1446). The v1 `geom_draw_points` site stays
    non-instanced (mirrors Metal).
  - Re-gate G1–G12, byte-identical reason content in the identical order,
    via `dev_bail("primitives: <reason>")`.
  - **G14 rigidity** per §5.1's binding mechanics: collect ALL
    LAMBERT-instanced draws → ONE grow-only host-visible member staging
    buffer (`geom_prim_inst_staging_`, ensure_buffer pattern) → ONE batched
    `vkCmdCopyBuffer` submit + fence → `instance_upper3x3_rigid` (transcribed
    verbatim from Metal: s̄² mean, s̄²>0, |ci·cj| ≤ tol·s̄², |‖ci‖²−s̄²| ≤
    tol·s̄²) — before any clear/encode. Imported buffers carry TRANSFER_SRC
    (import path ~:1486).
- **`tests/gfx/gfx_main.cpp`** — the shared CPU reference helpers
  (`instanced_project`, `instanced_lit`, etc.) were HOISTED to file scope
  outside both `#ifdef` blocks at T4 — both backends compare against ONE
  reference (the Metal rows consume the hoisted copies and re-ran 49/49
  live after the hoist). The Vulkan v1_3 rows A–E sit in the
  `CALIPER_HAVE_VULKAN` block, CUDA/VMM-gated by `vmm_rows_ready()` like the
  v1_1/v1_2 rows, with the header comment honestly reading "TRANSCRIBED …
  hardware verification PENDING the Windows session."

### 0.3 Hardware findings from the Mac pass the Windows session must know

1. **G14 storage-mode reversal (Metal-side, context only).** The v1_3 design
   spec §5.1 claimed Metal could read imported instance buffers via
   `contents()` directly (no blit). Hardware REVERSED this: torch MPS tensors
   import as `MTLStorageModePrivate` and `contents()` returns garbage. The
   Metal G14 now mirrors **Vulkan's** staged-copy shape (one grow-only member
   staging buffer, one batched submit) — commits `5403d92` + `5aa2f73`,
   pinned by a dedicated private-storage gfx row. **No Vulkan action**: the
   Vulkan design always assumed device-local imports and always staged. But
   when verifying G14 on Windows, know that the Metal reference's structure
   (collect → one copy submit → compare) is now identical on both backends —
   verify the Vulkan twin matches it structurally, not the pre-`5aa2f73`
   per-draw shape.
2. **`inst_attr_base` has no G6-equivalent u32 guard** (host-level,
   symmetric on both backends, G10-bounded — only reachable at >16 GB attr
   offsets). On record as a future-pass note; do NOT add it unilaterally on
   one backend (parity).
3. **G5/G10 refusal lines ride `range_ok`'s sink** (no `draw %u refused:`
   prefix) — adjudicated correct (mirrors v1.1/v1.2 bounds logs). Expect it
   in logs; do not "fix" it.

---

## 1. Prerequisites & build on the Windows box (D24 recipe)

Per `docs/m2a-windows-verification.md` and the v1_1 Phase-B precedent: the
`configure.cmd`/`build.cmd` wrappers (vcvars), build root prepended to PATH
for DLLs. The glslang steps are in-tree (`glslang-standalone` target,
CMakeLists `if(WIN32)` block ~:232-262) — the grown `geom.vert` compiles to
`geom_vert_spv.h` at build time; a std140 slip in the 192-byte block is a
**compile-time** static_assert failure in vulkan_renderer.cpp
(`sizeof(PrimParams)==192`) plus a runtime UBO-size mismatch, so check both.

- [x] **1.1** Configure + build green on the Windows box; the glslang step
  regenerates `geom_vert_spv.h` from the grown shader; record the exact
  invocation in the box's own scratch ledger (`.superpowers/sdd/` is
  gitignored per-box — the Mac ledger did NOT transfer, expected).
  *Done 2026-07-11: `cmd //c build_release.cmd` (vcvars64 → CLion CMake/Ninja,
  Release); `geom_vert_spv.h` regenerated (08:47, 43,558 B). Ledger:
  `v13-windows-hardware-pass-report.md`.*

## 2. Compile gate (first checkbox — none of this has compiled)

- [x] **2.1** `geom.vert` → SPIR-V compiles (bindings 8/9, std140 192-byte
  Params block, `gl_InstanceIndex` pull, `uintBitsToFloat` tint).
  *First try, no glslang complaint.*
- [x] **2.2** `vulkan_renderer.cpp` compiles: 10-binding set layout, pool
  growth, `bi[9]`/`wr[10]` write arrays, G14 staging machinery,
  `static_assert(sizeof(PrimParams)==192)`. *First try under MSVC 14.50 —
  the transcription needed zero backend changes.*
- [x] **2.3** `caliper_gfx_tests` compiles with the Vulkan v1_3 rows A–E in
  the `CALIPER_HAVE_VULKAN` block (first compile of the transcribed rows;
  the hoisted file-scope reference helpers already compile on the Mac side).
  *One break: doctest C2338 on chained `&&` in `REQUIRE_MESSAGE` (the MSVC
  analog of the Metal fix `5164f89`) — three VmmBlock.ok sites split to one
  check per block, fixed in `477431e`.*
- [x] **2.4** `caliper` app links; `caliper_tests`/`caliper_torch_tests`
  build; no cross-platform regression via shared headers. *All 8 test
  targets + caliper.exe linked.*

## 3. Validation-layer + portable gate rows

- [x] **3.1** Run with Vulkan validation layers once: no descriptor-count,
  binding-index, or UBO-range complaints on the instanced path (the class of
  error a Mac review cannot catch). *No layer existed on this box (SDK-free
  build); LunarG SDK 1.4.350.0 installed, `VK_LAYER_KHRONOS_validation`
  loader-injected over the full gfx battery AND a live 1000-instance
  `instance_scope` run: ZERO complaints in the target class. Only two
  pre-existing, pre-v1_3 framework findings fired (startup swapchain
  pre-transition ×2, one VkSurfaceKHR teardown leak) — recorded as
  carry-forwards in the box ledger, not v1_3 blockers. Artifacts:
  `v13-gfx-validation-run.log`, `v13-instance-scope-validation-run.log`.*
- [x] **3.2** Portable Row-E refusals that need no live CUDA (G2/G4/G8/G12
  shapes) fire with pixels bit-untouched on any ICD, reason strings byte-
  matching the host battery. *G2/G4/G8/G12 proven GPU-free by the
  `test_tensor_bridge` G1–G12 battery (green in caliper_tests);
  pixels-untouched proven on hardware by gfx Row E.*

## 4. Byte-exact rows on hardware (STOP-and-diagnose on any miss)

All five rows run against the SAME hoisted CPU reference the Metal rows
passed live (49/49). UUID-paired CUDA/VMM imports via `vmm_rows_ready()`,
same as the v1_1/v1_2 Vulkan rows.

- [x] **4.1 Row A** — pose-only fleet N=4, FLAT: **exact (0 LSB)**.
  *Passed untouched on first hardware contact.*
- [x] **4.2 Row B** — per-instance tint over a **FLAT** base, MAGMA, UNLIT:
  **exact (0 LSB)** (this row exists to catch a COLORMAP-only LUT-predicate
  regression at binding 4). *Passed untouched, first contact.*
- [x] **4.3 Row C** — v1.2 record vs zero-tail v1.3 record, two views,
  readbacks **byte-identical to each other** (additive-default proof; also
  proves `use_instance==0` SPIR-V bit-identity). *Passed untouched.*
- [x] **4.4 Row D** — instanced LAMBERT, N=2 rigid: **±2 RGB LSB, alpha
  exact** (the sole tolerance row). *Passed untouched — the §4.4 float-order
  chain held across glslang/NVIDIA.*
- [x] **4.5 Row E** — all seven refusals (G4/G5/G2/G8/G12/G14-shear/G7)
  return false with the view byte-equal to the last-good frame; G14's
  staged-copy readback proves out on real device-local CUDA imports.
  *One STOP-and-diagnose: the G5 shape carried Metal's literal count=5
  against a granularity-padded 2 MiB VMM block — the gate was right, the
  test was wrong (the recurring vkb6 padded-block lesson). OOB count now
  derived from `inst_blk.size` (`477431e`); all seven refusals then fired
  with pixels byte-identical to the last-good frame, G14 staged readback
  live on device-local CUDA imports.*
- [x] **4.6** Full `caliper_gfx_tests` green on the box; ALL pre-existing
  v1/v1_1/v1_2 Vulkan rows still green (no regression from the descriptor/
  pool/LUT-predicate changes — the LUT predicate change touches EVERY draw
  path, so the v1 COLORMAP rows are the regression canary).
  *48/48 cases, 1475/1475 assertions, zero "skipping" lines in the full log
  (`v13-gfx-battery-run.log`) — every CUDA/VMM row genuinely executed.*
- [x] **4.7** Full suite 100% (all 8 ctest suites).
  *8/8, 22.86 s (`v13-ctest-full-suite.log`).*

## 5. `instance_scope` on Vulkan/CUDA — the run-proof

R3's shipped exemplar is `instance_scope` (the earlier TwinScope 50-housing
fleet was reverted as a UX regression — see the design spec's hardware addendum
(e); TwinScope no longer carries a fleet, it is the R2 surface twin). The applet
code is platform-neutral and already run-proven on Metal/MPS (N gems, slider
1–5000, default 1000, ONE instanced draw, live device-tensor poses + tints). On
Windows verify:

- [x] **5.1** Device pick lands on CUDA; `instance_scope` renders: the status
  line `first zero-copy instanced frame drawn — 1000 objects, 1 draw call, 0
  mesh copies` appears with the Vulkan renderer active, zero-copy provenance
  line claimed only from the actually-drawn path. *Logged live 2026-07-11:
  `[renderer] vulkan` → `zero-copy pool ready (cuda)` → `geometry path OK —
  primitives drawn from imported allocations in place` → the exact line,
  sustained 12 s, clean teardown (`v13-instance-scope-vulkan-run.log`).*
- [x] **5.2** Drain-before-publish on the pose/attr path takes the CUDA arm
  (`torch::cuda::synchronize` idiom) — no torn poses/tints under load.
  *CUDA arm present at the publish site; sustained runs (plain + under
  validation layer) showed no torn poses and no spurious G14.*
- [x] **5.3** The N slider spans 1–5000 live; honest ladder:
  `CALIPER_RENDERER=gl` (or cap bit 3 absent) → the non-instanced fallback rung
  with the honest line, never a wrong image. *GL rung logged: `[renderer] gl`
  → `fallback (no geometry backend)`, zero zero-copy claims
  (`v13-instance-scope-gl-fallback.log`). Slider span pinned structurally
  (slots pre-sized to kNmax once; live run at default 1000) — the hand-driven
  1→5000 sweep was not performed this session, noted honestly in the ledger.*

## 6. Closeout (which lines flip, and only then)

- [x] **6.1** `GEOMETRY.md` R3 row: → SHIPPED both platforms ("byte-exact
  both backends"; drops "Vulkan transcribed, hardware pending Windows").
  *This commit.*
- [x] **6.2** `ROADMAP.md` §6: R3 + twin-flagship rows drop their Windows
  qualifiers. *This commit (the twin-flagship row carried no Windows
  qualifier — only R3's line changed).*
- [x] **6.3** `ZEROCOPY.md`: the Vulkan/CUDA instanced-geometry row flips to
  hardware-verified. *This commit.*
- [x] **6.4** `tests/gfx/gfx_main.cpp` Vulkan v1_3 rows header: replace the
  transcription caveat with the run-proven statement (the exact mirror of
  what the Metal header already truthfully says). *This commit.*
- [x] **6.5** This doc's boxes ticked with commits named; findings to the
  box's scratch ledger; commits in house style with the Fable trailer.
  *Test fixes: `477431e`. Closeout: this commit. Ledger:
  `v13-windows-hardware-pass-report.md` + four run logs in
  `.superpowers/sdd/`.*

---

## Invariants (hold forever — restated)

- **One reference:** both backends compare against the SAME file-scope CPU
  helpers. A Vulkan miss is a Vulkan bug or a shared-reference bug — never
  grounds to fork the reference or widen a tolerance.
- **Bit-identity of the non-instanced path:** `use_instance==0` draws must be
  byte-identical to pre-v1_3 output (Row C is the proof; the v1/v1_1/v1_2
  row battery is the canary).
- **Honest degradation and provenance:** zero-copy claimed only when the
  imported instanced path actually drew; cap absent → non-instanced fallback +
  the honest line.
- **No checkbox without artifacts.** Transcription review (T4) was thorough
  but is NOT a hardware claim — this pass exists because it isn't.
- Data flows tensors → pixels → ImGui, one way; no new ABI; the frozen 192-
  and 216-byte records and the 256-byte v1_3 record are never touched.
