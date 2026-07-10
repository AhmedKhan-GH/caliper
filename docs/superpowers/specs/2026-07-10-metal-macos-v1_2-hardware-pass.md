# geometry.v1_2 on Metal/macOS — the hardware pass

**Date:** 2026-07-10
**Status:** execution spec (hardware pass). NOT a greenfield implementation —
the Metal code for R2 (`geometry.v1_2` textures-on-meshes) and the TwinScope v2
surface twin **already exists in the tree**, transcribed from the donor and
code-reviewed on the Windows box, but the `.mm` translation unit was compiled
out there (`CALIPER_HAVE_METAL` unset) and nothing ran on Apple hardware. This
doc (a) orients a macOS session to what the Metal surface already contains and
where it lives, and (b) gives the mechanical execution plan with acceptance
gates to take it from "transcribed + reviewed" to "run-proven byte-exact on
Metal/MPS."
**Authority:** the MAC-PENDING acceptance list is
`docs/superpowers/specs/2026-07-10-caliper-framework-remaining-work.md §1.2`
and the `## MAC-PENDING CHECKLIST` in `.superpowers/sdd/progress.md` (lines
484–496). Byte-exact bar and rows: the v1_2 implementation contract
(`2026-07-10-geometry-v1_2-textured-mesh-design.md §Verification rows`) and the
TwinScope v2 design (`2026-07-10-twinscope-v2-surface-twin-design.md §10`).
Platform protocol mirrors v1_1, which ran this same protocol in the **reverse**
direction (Metal-first, Vulkan pass second — `GEOMETRY.md §10 Phase C`,
`ROADMAP §2/§4`).
**Checkbox discipline (inherited):** a box is checked only when the suite is
green on this Mac, the path is run-proven by a logged/screenshotted artifact,
and the fixing commit is named. Invariants at the bottom never become
checkboxes. A twin failure is a **STOP-and-diagnose**, never a loosened
comparison.

---

## 0. What already exists (the Metal v1_2 surface, file by file)

Everything below is on `main` (merged at `f9397db`). "Transcribed + reviewed,
never compiled" means: the code passed line-by-line review against the Vulkan
reference and the donor, the reviewer confirmed structural parity, but no Metal
compiler has seen it and no Apple GPU has executed it. The **risk class** is
therefore: MSL compile errors, stage-in signature mismatches, pipeline-creation
failures, viewport/NDC sign slips, and MPS device-selection surprises — the
things a review cannot catch and only hardware reveals.

### 0.1 Host validator (platform-neutral — already compiled on Windows)

`src/host/tensor_bridge.cpp` — `TensorBridge::geom_draw_primitives_impl`
(line 587) is the **single revision axis** the R2 hardening introduced: one
`bool v12` parameter (line 590) derives both `min_stride`
(`sizeof(CaliperGeomDrawV1_2)` vs `sizeof(CaliperGeomDraw)`, line 593–594) and
the color-mode ceiling (`COLOR_TEXTURE` vs `VERTEX_RGBA`, line 596). The v1.1
and v1.2 public entry points call it with `/*v12=*/false` (line 576) and
`/*v12=*/true` (line 584). This validator resolves both public ids and runs on
both platforms — it is already exercised by the Windows/Vulkan run-proof, so on
the Mac it needs no new verification beyond "the same gates fire." The Metal
backend **re-gates** every allocation against its native `id<MTLBuffer>` before
encoding (§0.2), which is the part that has never run.

### 0.2 The `.mm` draw path (the code that never compiled)

`src/host/renderer/metal_renderer.mm` (1308 lines, Apple-only TU). The v1_2
surface inside it:

- **`geom_draw_primitives`** (lines 767–980): gate-everything-then-encode. The
  textured branch (`CALIPER_GEOM_COLOR_TEXTURE`, lines 863–883) resolves the UV
  import and the sampled texture, and carries the two donor parity fixes:
  - **uv_base 32-bit refusal** (lines 933–934): `if (d.uv_offset / 4 > UINT32_MAX)`
    → `metal_geom_fail("primitives: uv base exceeds 32 bits")`.
  - **render-target-view sampling refusal** (lines 867–875): refuses sampling
    any texture whose `usage & MTLTextureUsageRenderTarget` is set, the unknown
    id, or the current target (`d.texture == view_tex`). The `RenderTarget`
    usage bit is the marker that uniquely tags a `geom_create_view*` texture —
    `tex_create_rgba8` never sets it (lines 494–509 vs 606–682). Mirrors
    Vulkan's `fb != VK_NULL_HANDLE` refusal.
- **`metal_geom_fail`** (lines 358–361): the refusal helper, prints
  `[metal] geom_prims: %s` to stderr and returns `false`.
- **`honors_stream_ordered_handoff`** (line 384): returns `true`
  **unconditionally** — so Metal always lights `CALIPER_BRIDGE_CAP_STREAM_ORDERED`
  (`tensor_bridge.cpp:215`), unlike Vulkan which gates it on `pipelined_ok_`.
  This is the "Metal always reports the cap true" caveat (§5).
- **Runtime-compiled MSL, no `.metal` files** (repo convention): three shader
  strings — `kColormapShaderSrc` (52–84), `kPointsShaderSrc` (96–144), and the
  geometry pair `kGeomShaderSrc` (157–296). The textured fragment function
  `geom_tex_fs` is at lines 288–295 (clamp-to-edge, linear, no-mip sampler —
  matching the v1_2 contract §Rendering). Pipeline selection in `geom_pipeline`
  (1088–1144) pairs `geom_vs`/`geom_vs_point` with `geom_fs`/`geom_tex_fs`.

### 0.3 `PrimParams` lives in three synced copies

The params block is hand-synced across three files; all three are pinned to
**176 bytes** and must move together (the comment naming them is
`geom.vert:26–29`):

1. **GLSL std140** — `src/host/renderer/shaders/geom.vert:30–51` (block size
   176 at line 50). Compiled to SPIR-V; Vulkan-only.
2. **Vulkan C++ mirror** — `src/host/renderer/vulkan_renderer.cpp:136–147`
   (`static_assert(sizeof(PrimParams) == 176)` at 147).
3. **Metal** — two sub-copies in `metal_renderer.mm`: the MSL struct inside
   `kGeomShaderSrc` (lines 161–182) and its C++ mirror (lines 298–309,
   `static_assert(sizeof(PrimParams) == 176)` at 309). Only these two are
   compiled by the Metal TU; the GLSL/Vulkan copies are compiled out on mac.

The `uv_base` field is the v1_2 tail addition (offset 160, `pad0..pad2`
following). Because copies (1) and (2) are compiled out on mac, a divergence
between them and the Metal copy can ONLY surface as a byte-exactness failure in
the cross-platform gfx rows (§4) — there is no compile-time guard across the
`#ifdef` seam.

### 0.4 The gfx twin rows (compiled out on Windows)

`tests/gfx/gfx_main.cpp` is backend-partitioned by `#ifdef`:
- **`CALIPER_HAVE_METAL` block: lines 350–2358** — the Metal rows, including the
  four T4 v1_2 twins. These were transcribed but the whole block was compiled
  out on the Windows box.
- **`CALIPER_HAVE_VULKAN` block: lines 2365–4757** — the Vulkan rows that RAN
  and are byte-exact on this hardware; they are the **references** the Metal
  twins must match.

The v1_2 Metal twin rows: clamp-to-edge OOB-UV (~line 2121), short-`draw_stride`
refusal (~2267/2344), the RenderTarget-usage view refusal exercised via a
render-target texture (~line 701). Their Vulkan references are the mirror rows
in the Vulkan block (clamp-to-edge ~4511, short-stride ~4661/4741). Same CPU
reference, same expected bytes.

### 0.5 The applet (TwinScope v2) — already has an MPS branch

`applets/twin_scope/twin_scope.cpp` — `twin_job` (line 258) selects the device
with an explicit `#if defined(__APPLE__)` MPS branch (lines 259–268): `cuda ?
kCUDA : mps ? kMPS : kCPU`, where `mps = !cuda && torch::mps::is_available()`.
The zero-copy update path (`update_tex`, lines 823–839) uses `stream_to_tensor`
+ `update_texture_from_alloc` gated on `CALIPER_BRIDGE_CAP_STREAM_ORDERED`
(lines 617–618), else CPU-staged. See §5 — and the contradiction note there
about the "assume CUDA-or-CPU" framing in the authority docs.

---

## 1. Prerequisites & build on the mac (the recorded recipe)

The mac session's own `.superpowers/sdd/` scratch was **never committed and did
not transfer** to the Windows machine (`progress.md:1–7`), so the recipe below
is reconstructed from the last place it WAS recorded: the v1_1-era Metal plan
`docs/superpowers/plans/2026-07-07-geometry-metal.md`. Do not invent flags —
if this recipe is stale on the actual box, rediscover from CLion's CMake profile
and re-record it in `progress.md`.

- **Toolchain** (from `2026-07-07-geometry-metal.md:9,20`): ObjC++/ARC
  (`metal_renderer.mm`, `-fobjc-arc` set in CMakeLists), MSL compiled at runtime
  via `newLibraryWithSource` (no `.metal` files), libtorch 2.5.1 **MPS**,
  doctest suites `caliper_tests` / `caliper_gfx_tests` / `caliper_torch_tests`.
- **Build dir:** `cmake-build-debug` (CLion-managed). Configure via the CLion
  CMake profile; `CALIPER_HAVE_METAL` is defined on Apple, which is what pulls
  `metal_renderer.mm` and the gfx Metal block (§0.4) into the build for the
  first time.
- **Build targets** (recorded command form,
  `2026-07-07-geometry-metal.md:69,781`):
  `cmake --build cmake-build-debug --target caliper_tests`,
  `... --target caliper_gfx_tests`, `... --target caliper_torch_tests`,
  `... --target caliper`. Binaries land under `cmake-build-debug/tests/` (find
  with `find cmake-build-debug -name caliper_gfx_tests -type f`).
- **MPS torch header:** `torch::mps::is_available()` comes via
  `<torch/torch.h>`; if the build disagrees, add `#include <torch/mps.h>` under
  `#if defined(__APPLE__)` (`2026-07-07-geometry-metal.md:733`). The
  `caliper_torch_tests` MPS cases skip when `!torch::mps::is_available()`.
- **Note vs Windows:** the Windows env needs a build root prepended to PATH for
  DLLs and a vcvars wrapper (`progress.md:414–419`); on the mac there is no DLL
  copy step and no `build.cmd` — CLion drives the build directly. The Windows
  `configure.cmd`/`build.cmd` wrappers are Windows-only.

- [ ] **1.1** CLion CMake profile configures clean on this Mac with
  `CALIPER_HAVE_METAL` defined; record the exact configure/build invocation in
  `progress.md` (close the recipe gap the lost scratch left).

---

## 2. Compile gate (the first checkbox — the `.mm` has never compiled)

The single most likely place to fail: MSL and ObjC++ that no compiler has read.

- [ ] **2.1** `metal_renderer.mm` compiles: the three runtime-MSL strings
  (`kColormapShaderSrc`, `kPointsShaderSrc`, `kGeomShaderSrc`) and the ObjC++
  encode path build with `-fobjc-arc`. The two Metal `PrimParams` copies
  (§0.3) hit their `static_assert(... == 176)` — a mismatch here is a compile
  error, not a runtime bug.
- [ ] **2.2** `caliper_gfx_tests` compiles with the `CALIPER_HAVE_METAL` block
  (lines 350–2358) now IN the build for the first time. Transcription typos in
  the Metal twin rows surface here.
- [ ] **2.3** `caliper_torch_tests` compiles (the MPS `storage_ref`/pool path,
  `exportable_pool.hpp:513–609`, is Apple-guarded and never compiled on
  Windows).
- [ ] **2.4** `caliper` app links and `caliper_tests` builds — confirm no
  non-Apple regression leaked in via a shared header.

---

## 3. Parity / refusal gates on hardware (T3)

Each refusal must fire on live Metal AND leave the target pixels **bit-for-bit
untouched** (the gate runs before the render encoder exists, so no clear
happened — `metal_renderer.mm:459` house rule, GEOMETRY.md §6.3).

- [ ] **3.1 uv_base 32-bit refusal.** A textured draw with `uv_offset/4 >
  UINT32_MAX` is refused (`metal_renderer.mm:933–934`); the log reads the
  `geom_prims:` line (see §6.1 for the cosmetic on that exact string). Pixels
  untouched.
- [ ] **3.2 render-target-view sampling refusal.** A draw naming a
  `geom_create_view*` texture (or the current target) as its sampled
  `d.texture` is refused via the `MTLTextureUsageRenderTarget` marker
  (`metal_renderer.mm:867–875`). Confirm on hardware that the marker is
  genuinely unique to view textures (`tex_create_rgba8` sets only
  `ShaderRead|ShaderWrite`, never `RenderTarget`). Pixels untouched.
- [ ] **3.3 short-`draw_stride` refusal.** A v1.2 submission with a 192-byte
  stride (the frozen v1.1 prefix, missing the tail) is refused by the host
  validator (§0.1) before any backend work. The Metal gfx row exercising this
  is ~`gfx_main.cpp:2267/2344`.
- [ ] **3.4 the rest of the gate battery** fires identically to the Vulkan
  reference: unknown/released UV alloc, UV misalignment, UV overflow,
  unknown/released texture, mixed valid/invalid multi-draw refuses atomically
  (v1_2 design §Verification rows / row 2). Byte-exact refusal each time.

---

## 4. Byte-exact twin rows (T4) — STOP-and-diagnose on any miss

The four T4 Metal gfx twins plus the transcribed donor rows must run LIVE-GREEN
as byte-exact mirrors of the Vulkan references (same CPU reference, same
expected bytes — the v1_1 Phase B discipline where every drawing row passed
byte-exact first try, `progress.md:388`). **A twin failure means STOP and
diagnose the backend divergence; never loosen the comparison, widen a
tolerance, or mask a pixel to make a row pass.** The rows exist precisely to
catch an MSL/std140 divergence the compiled-out `#ifdef` seam (§0.3) cannot.

- [ ] **4.1 texel-center / bilinear / Lambert / uv-offset** (the four donor
  rows the Vulkan side already passes): exact texel-center samples byte-exact;
  2×2 bilinear center within 1 RGBA8 LSB; Lambert×texture within 2 RGB LSB,
  alpha unchanged; nonzero UV byte offset selects the intended coordinates.
- [ ] **4.2 clamp-to-edge with out-of-range UVs** (`gfx_main.cpp` ~2121):
  out-of-`[0,1]` UVs clamp to the nearest edge texel, no bilinear mix.
- [ ] **4.3 v1.1 draw == v1.2 zero-tail non-textured draw** byte-identical (the
  additive-compat row): a v1.1-shaped draw and a v1.2 record with a zeroed
  UV/texture tail render bit-identical.
- [ ] **4.4 refusal purity incl. released-UV-alloc + short-stride** leave pixels
  bit-for-bit untouched (this is the row that also covers §3.3).
- [ ] **4.5 donor-row Metal twins** (the transcribed donor gfx rows) confirmed
  live-green on Metal.
- [ ] **4.6 full `caliper_gfx_tests` green** on this Mac with the Metal block
  live; the pre-existing v1 / v1_1 Metal rows still pass (no regression from the
  v1_2 additions).

---

## 5. TwinScope v2 on Metal/MPS — device selection is the real risk

The applet already has an MPS branch (§0.5), the STREAM_ORDERED path is fully
wired for MPS (`stream_to_tensor` returns a Metal `id<MTLBuffer>` pointer and a
producer-queue `.stream`, `torch.hpp:148–155,189+`; Metal's
`order_after_producer`, `metal_renderer.mm:1165`, consumes it), and Metal always
reports `STREAM_ORDERED` true (§0.2). None of it has run on an Apple GPU. The
one thing a review cannot settle: **does the device pick actually land on MPS at
runtime, or silently fall to CPU?**

- [ ] **5.1 device pick lands on MPS.** Confirm `torch::mps::is_available()`
  returns true on this box and the logged `device_name` reads **"MPS"** (set at
  `twin_scope.cpp:379,384`), not "CPU". A silent CPU fall makes every zero-copy
  claim below false.
- [ ] **5.2 STREAM_ORDERED handoff runs.** With MPS selected, the imported
  update path (`update_tex`, `twin_scope.cpp:823–831`) takes the zero-copy
  branch: verify the MPS producer-queue handoff orders correctly (the Metal
  `MTLSharedEvent` path, `metal_renderer.mm:1165–1183`) — texels reflect the
  latest sim step, no torn field. This is the MPS analog of the CUDA
  STREAM_ORDERED proof; it has NEVER executed.
- [ ] **5.3 zero-copy textured split view draws** with an honest provenance
  line (claimed only when that path actually drew — flow_scope discipline). The
  Vulkan+CUDA reference proof: `progress.md:478–479` (`V_render 2430 / V_sim
  28590`, "geometry path OK — imported allocations in place"). Mirror that as
  the Metal proof: a logged/screenshotted status line.
- [ ] **5.4 honest ladder holds.** `CALIPER_RENDERER=gl` (or MPS zero-copy
  genuinely unavailable) → per-vertex or heatmap rung with the honest status
  line, never a wrong image (design §9). If MPS is present but the pool import
  fails, the runtime re-eval falls to per-vertex, not to the heatmap.

**Contradiction to flag (do not silently reconcile).** Both authority docs —
remaining-work §1.2 and `progress.md:493–496` — describe the applet / thermal
model as assuming **"CUDA-or-CPU."** The tree does not match: `twin_scope.cpp:
259–268` has a real `#if defined(__APPLE__)` MPS branch selecting `kMPS`, and
`twin_model.h` is device-agnostic (it takes a `device` parameter,
`twin_model.h:15`). So the MPS path is present, not absent — the genuine open
question the docs are gesturing at is only **runtime selection** (5.1) and the
**never-run MPS STREAM_ORDERED ordering** (5.2), not a missing branch. Treat the
docs' "assume CUDA-or-CPU" as imprecise wording, and verify the runtime pick
rather than expecting to add an MPS branch.

---

## 6. The two ledgered cosmetics (fix on hardware, anchors pinned)

Both were triaged during the T3 Metal review and pushed to this list
(`progress.md:436–437,503`). Anchors below were grepped from the current tree so
the mac session does not hunt.

### 6.1 Refusal-log double prefix `geom_prims: primitives:`

- **Anchor:** `src/host/renderer/metal_renderer.mm:359` — the helper prints
  `[metal] geom_prims: %s`. The only call site that passes a `primitives:`-
  prefixed reason is **`metal_renderer.mm:934`**:
  `metal_geom_fail("primitives: uv base exceeds 32 bits")`, which emits
  `[metal] geom_prims: primitives: uv base exceeds 32 bits` — the redundant
  double tag.
- **Fix:** drop the leading `primitives: ` from the string at line 934 (the
  helper already supplies the `geom_prims:` category). No other call site is
  affected (verified by grep: only line 934 double-prefixes). Log-only,
  behavior unchanged.
- [ ] **6.1** fixed and confirmed in the §3.1 refusal log.

### 6.2 `geom_tex_fs` VOut stage-in unreached by textured POINT draws

- **Anchor:** `src/host/renderer/metal_renderer.mm:288–295` — `geom_tex_fs`
  takes `VOut in [[stage_in]]` (the non-point VOut, lines 188–192). The point
  vertex function `geom_vs_point` (lines 269–285) emits `VOutPoint` (with
  `[[point_size]]`, lines 194–199). `geom_pipeline` (lines 1097–1100) pairs
  `geom_vs_point` (when `cls==0`) with `geom_tex_fs` (when `textured`) — a
  stage-in signature mismatch (`VOutPoint` producer, `VOut` consumer) that would
  only be constructed by a **textured POINT draw**.
- **Why unexercised:** TwinScope drapes textures on triangle meshes only; it
  never issues a `COLOR_TEXTURE` draw with `TOPO_POINTS`. So the mismatched
  pipeline is never created. This is a donor property, adjudicated defensible.
- **On hardware:** confirm no path builds that pipeline (a textured POINT draw
  would attempt `geom_pipeline(cls=0, ..., textured=true)` and could fail
  pipeline creation → clean `metal_geom_fail`, pixels untouched). Then either
  **document** the constraint in a comment at `geom_tex_fs` / `geom_pipeline`, or
  **guard** it (refuse `COLOR_TEXTURE + TOPO_POINTS` at the gate in
  `geom_draw_primitives`, ~lines 863/885). Documenting is sufficient if
  pipeline creation fails closed on hardware; guard only if it does something
  worse.
- [ ] **6.2** verified never-reached, and documented (or guarded) accordingly.

---

## 7. Closeout (which lines flip, and the protocol)

When §2–§6 are all green with artifacts, the following flip — and only then:

- [ ] **7.1 `ROADMAP.md §6`** — line 99 (`Twin applet ships run-proven on both
  platforms`) is **the box this pass unlocks**: it currently reads "Metal/MPS
  pass pending macOS"; flip to run-proven both platforms. Line 97 (R2 shipped
  both backends) drops the "hardware verification pending macOS" qualifier.
- [ ] **7.2 `GEOMETRY.md`** — the R2 row (line 611) drops "Metal: transcribed +
  reviewed, hardware verification pending macOS" → "byte-exact both backends."
- [ ] **7.3 `ZEROCOPY.md`** — add the Metal/MPS imported-geometry **textured**
  status alongside the existing rows (the primitives rows are at lines 288–294);
  the v1_2 textured path is now "byte-exact verified on Apple Silicon."
- [ ] **7.4 `.superpowers/sdd/progress.md`** — tick the `## MAC-PENDING
  CHECKLIST` (lines 484–496); record the build recipe (§1.1) and any hardware
  finding, in the honesty-ledger style of the existing entries.
- [ ] **7.5 the remaining-work plan §1.2** — mark the macOS hardware pass done;
  §1.3's "twin applet on both platforms" box is now truthfully checkable.
- [ ] **7.6 commit** in house style: `docs(specs):` for doc flips, `fix(metal):`
  for §6 cosmetics, each its own commit; end every message with the
  `Co-Authored-By: Claude Fable 5` trailer. Any code fix rides a full
  `caliper_gfx_tests` + live re-proof, not a spot check.

---

## 8. Addendum — code merged after this spec was written (tech-debt merge `30e52f6`)

The §3 tech-debt branch (`fix/r2-tech-debt`, merged the same day) landed applet
code with `__APPLE__`-guarded branches that **no compiler has seen** — the same
risk class as §2. The mac session must fold these into the pass:

- [ ] **8.1 Compile gate additions.** New Apple-guarded drain sites, all calling
  `caliper::adapters::detail::mps_synchronize_serialized()` in the mesh_scope
  idiom: `applets/twin_scope/twin_scope.cpp:497-500` (the drain between the
  slot copies and the `ready_slot` flip — commit `335d0b1`),
  `applets/flow_scope/flow_scope.cpp:255-258` (initial-publish drain), and the
  hoisted `sync` lambda call at `applets/field_scope/field_scope.cpp:206-211`.
  These must compile on macOS; they are one-line siblings of code that already
  compiles there, but verify, don't assume.
- [ ] **8.2 Contract holds on MPS.** The drain-before-publish invariant is now
  documented contract (`ZEROCOPY.md`, `geometry_v1.h:66-90`): every worker
  drains its device before flipping `ready_slot`. On MPS the drain is the
  serialized-sync helper — confirm during the §5 TwinScope run that publishes
  are drained (no torn per-vertex COLORMAP frames) with the sim under load.
- [ ] **8.3 Frame-thread gates behave on Metal.** `14e143a` gated the last two
  frame-thread `stream_to_tensor` callers (gpt_scope `upload_mapped`,
  embed_scope `update_or_create`) on STREAM_ORDERED. Metal reports that cap
  unconditionally (`metal_renderer.mm:384`), so on this Mac the gated *fast*
  arm is what executes — confirm gpt_scope and embed_scope draw without stall
  as part of the §5-adjacent applet sweep. (`gpt_scope.cpp:818`'s drain is
  CUDA-only by design; its imported path is CUDA-gated at `:719` — not a gap.)

Line anchors are as of `30e52f6`; re-grep if the tree has moved.

---

## Invariants (hold forever — restated from ROADMAP.md / GEOMETRY.md §12)

- **Byte-exact bar:** the Metal twin is byte-identical to the one CPU reference
  the Vulkan side already matches. A miss is a backend bug to diagnose, never a
  tolerance to widen or a pixel to mask.
- **Honest degradation:** MPS-zero-copy absent → a working slower rung
  (per-vertex, then CPU-staged, then heatmap) with a status line that says so.
  Never a wrong image, never a false provenance claim.
- **No checkbox without artifacts:** a box is checked only when the suite is
  green on this Mac, the path is logged/screenshotted run-proven, and the commit
  is named. Review is not a substitute for a hardware run — this whole pass
  exists because it isn't.
- Data flows tensors → pixels → ImGui, one way. No render-to-tensor, no
  applet-supplied shaders, no new ABI — the hardware pass changes zero contracts.
