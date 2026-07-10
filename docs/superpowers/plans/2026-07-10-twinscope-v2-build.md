# TwinScope v2 Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild geometry.v1_2 + the TwinScope exemplar incrementally on `feat/geometry-v1_2`, transcribing surviving code from the donor branch, folding in every confirmed review fix at construction, and shipping the redirected surface-twin demo.

**Architecture:** Per `docs/superpowers/specs/2026-07-10-twinscope-v2-surface-twin-design.md` (READ IT — it is the requirements authority; its §8 fix list and §10 verification bar bind every task). ABI identical to `docs/superpowers/specs/2026-07-10-geometry-v1_2-textured-mesh-design.md`.

**Tech Stack:** C++20, Vulkan (runnable here), Metal (transcription, macOS pending), libtorch (incl. sparse CSR), doctest, CMake/ninja.

## Global Constraints

- Branch `feat/geometry-v1_2` (off main `0b84c57`). Never touch `main` or the donor branch.
- **Donor access (token discipline):** read donor files WITHOUT switching branches: `git show codex/twinscope-implementation:<path>` (e.g. `git show codex/twinscope-implementation:sdk/include/caliper/services/geometry_v1_2.h`). Donor tip `2bb13c5` already contains the SDK widening fix. TRANSCRIBE what survived review; the brief tells you what to change during transcription. Do not re-derive working code; do not copy donor code the brief says to replace.
- Frozen ABI: `CaliperGeomDraw` 192-byte prefix frozen; `CaliperGeomDrawV1_2` 216 bytes, tail offsets 192/200/208; no new entry points; `reserved0` NULL; growth stride-carried.
- Frame-thread discipline: no torch ops / GPU syncs / file I/O in `draw_ui`.
- Cross-backend byte-exact contract; gates refuse whole-frame, pixels untouched; gate parity Metal↔Vulkan is binding.
- Metal files cannot compile here — transcribe, self-review, ledger "hardware verification pending macOS".
- Tests: `cmake --build cmake-build-debug` then `PATH="/c/Users/ahmed/CLionProjects/caliper/cmake-build-debug:$PATH" ctest --test-dir cmake-build-debug --output-on-failure` (unit target name is `caliper_tests`).
- TDD by stakes: engine/model/host logic test-first; renderer changes get gfx rows; applet threading verified by build + log-run + review.
- One task = one commit (or a few coherent ones), conventional style, `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Parallel dispatch allowed ONLY for the marked pairs (disjoint files; ThoughtSpace precedent): (T2 ∥ T3), (T7 ∥ T8a). Everything else sequential.

---

### Task 1: v1_2 ABI + host gates + SDK widening (donor transcription + one fix)

**Files:**
- Create: `sdk/include/caliper/services/geometry_v1_2.h` [donor verbatim]
- Modify: `sdk/include/caliper/caliper.hpp` [donor @2bb13c5 Geometry wrapper hunks verbatim — includes the widening fix + `widen_v11_draws_`]
- Modify: `src/host/host_services.cpp` [donor: kGeom12 table, id in kIds/service_ids]
- Modify: `src/host/tensor_bridge.h`, `src/host/tensor_bridge.cpp` [donor gates/texture-lookup/caps-bit-2 logic, ONE CHANGE: the shared impl takes a single `bool v12` axis and derives `min_stride = v12 ? sizeof(CaliperGeomDrawV1_2) : sizeof(CaliperGeomDraw)` and the color-mode ceiling inside — the donor's `(uint32_t min_stride, bool allow_textured)` pair is the reviewed defect, do not transcribe it]
- Modify: `src/host/renderer/host_renderer.h` [donor: HostGeomDraw uv/texture fields + `supports_geometry_textured()` default-false virtual]
- Tests: `tests/test_abi.cpp` [donor @2bb13c5: offset pins + widening regression verbatim], `tests/test_tensor_bridge.cpp` [donor: full v1_2 gate battery — unknown/released UV alloc, misalign, overflow, unknown/released texture, view-as-texture, short stride, v1.1-tail isolation, mixed multi-draw, caps bit], `tests/abi_c_check.c` [donor one-liner]

**Interfaces:** Produces the complete host v1_2 surface later tasks consume; caps bit 2 stays unset on real renderers until T2/T3 (stub-renderer tests exercise the gates — donor pattern).

- [ ] Step 1: Transcribe test files first; build; RED (missing header/entry points).
- [ ] Step 2: Transcribe implementation with the single-axis change; build; GREEN — full ctest green.
- [ ] Step 3: Commit `feat(geometry): caliper.geometry.v1_2 ABI + host gates (single revision axis) + SDK widening`.

### Task 2: Vulkan textured backend (∥ Task 3)

**Files:**
- Modify: `src/host/renderer/vulkan_renderer.cpp` [donor: textured pipeline-key bit, binding-7 combined sampler (fixed linear/clamp/no-mips), gates incl. `uv_base > 32 bits` refusal and view/layout refusal, PrimParams 176 assert]
- Modify: `src/host/renderer/shaders/geom.vert` [donor uv additions; FIX the comment: params block is 176 bytes, and add the three-synced-copies note (GLSL block / vulkan PrimParams / metal MSL string)]
- Create: `src/host/renderer/shaders/geom_tex.frag` [donor verbatim]
- Modify: `CMakeLists.txt` [donor adds a sixth SPIR-V rule — instead extract function `caliper_compile_shader(NAME SRC VN)` and convert ALL six shader blocks to it]

**Interfaces:** `supports_geometry_textured()` returns true → caps bit 2 live on Vulkan.

- [ ] Step 1: Transcribe; clean-configure + build; verify regenerated `*_spv.h` for the five pre-existing shaders are byte-identical to before the helper conversion (diff against pre-change copies).
- [ ] Step 2: Full gfx suite green (pre-existing rows must be untouched by the shared-shader change — if any §9.2 row breaks, that is a real regression: stop, report).
- [ ] Step 3: Commit `feat(vulkan): geometry.v1_2 textured path + shader-compile CMake helper`.

### Task 3: Metal textured backend (∥ Task 2; transcription only, no build)

**Files:**
- Modify: `src/host/renderer/metal_renderer.mm` [donor: textured pipeline variant, constexpr sampler, PrimParams 176, MSL geom_tex fs; PLUS the two parity fixes the donor lacks — (1) `if (d.uv_offset / 4 > UINT32_MAX) return metal_geom_fail("primitives: uv base exceeds 32 bits");` before uv_base assignment; (2) the sampled-texture gate refuses ANY geometry view (find the Tex-entry flag `create_view` sets and refuse on it), not just `d.texture == view_tex`]

- [ ] Step 1: Transcribe + apply both fixes; self-review hunk-by-hunk against the donor AND against vulkan_renderer.cpp's gate block for parity.
- [ ] Step 2: Commit `feat(metal): geometry.v1_2 textured path with day-one gate parity (mac verification pending)`. Ledger the pending status.

### Task 4: gfx verification rows — donor set + the four missing (both backends)

**Files:** Modify `tests/gfx/gfx_main.cpp`.

Rows (each on BOTH backends; Vulkan runs here, Metal transcribed):
1. [donor] texel-center byte-exact red / bilinear-center gray / Lambert×texture 2-LSB / uv-offset-64-with-poison — transcribe donor's Metal (~2038) and Vulkan (~4183) cases.
2. [new] clamp-to-edge: quad UVs spanning (-0.5..1.5) over the 2×2 texture, FLAT, readback points deep in each out-of-range region == nearest edge texel exactly.
3. [new] compat: identical non-textured indexed COLORMAP+LAMBERT draw via v1_1 entry (stride 192) into view A and via v1_2 entry (zero tail, stride 216) into view B → full-image memcmp equal.
4. [new] gate refusals leave pixels untouched (cumulative pre-image memcmp): released-after-import uv_alloc; geometry-view id in `texture`; released texture id; stride-192 v1_2 submission.
- [ ] Step 1: Write rows; run `caliper_gfx_tests` — all new Vulkan rows + every pre-existing row green. A row-3 failure is a real shared-shader regression: stop, report.
- [ ] Step 2: Commit `test(gfx): v1_2 rows — donor set + clamp-to-edge, v1_1/v1_2 compat, refusal purity`.

### Task 5: OBJ loader + the v2 heatsink asset

**Files:**
- Create: `sdk/include/caliper/adapters/obj.hpp` [donor verbatim — it passed review]
- Create: `tests/test_obj.cpp` [donor, adapted to the new asset's counts/bounds]
- Create: `applets/twin_scope/assets/housing.obj` — NEW procedural heatsink housing per spec §2: 2,500-4,000 triangles, dense fin array, per-chart UVs with ≥4-texel gutters at 256², `vt`+`vn` throughout, vertices deduplicated. Write a scratch generator (python or C++ in a temp dir — NOT committed; the OBJ is the artifact), run it once, sanity-render the numbers (triangle count, UV range, chart count) into the test expectations.
- Modify: `tests/CMakeLists.txt` [donor's caliper_obj_tests block]

**Interfaces:** `caliper::obj::Mesh {positions (V,3) f32, normals, uvs (V,2), indices (F*3) i32, has_uvs/has_normals}` — T6/T8 consume exactly this (donor shape).

- [ ] Step 1: Loader + donor tests RED→GREEN against synthetic cases; then the committed asset row (counts, UV∈[0,1], dedup, malformed set).
- [ ] Step 2: Commit `feat(sdk): OBJ loader (donor) + v2 heatsink asset (~Nk tris, charted UVs)`.

### Task 6: Surface engine — subdivision, cotan Laplacian, bake matrix (NEW, the v2 core)

**Files:**
- Create: `applets/twin_scope/twin_surface.h` — header-only, pure torch/CPU-precompute per spec §2/§3/§5: `subdivide_midpoint(mesh, levels=2)` (originals stay prefix), `cotan_laplacian(mesh) -> sparse CSR L + dense masses M` (Voronoi-third areas), `stable_dt(L, M)` (0.9/max row ratio), `bake_matrix(mesh_sim, H, W) -> sparse S + gutter index map` (UV edge-function rasterizer, barycentric weights).
- Create: `tests/test_twin_surface.cpp` — spec §10 engine rows: subdivision counts/prefix/manifold; L symmetry, zero row sums, PSD probe (small mesh, smallest eigenvalue ≥ −1e-5); bake partition-of-unity + coverage (uses T5 asset via loader); flat-strip analytic: in-test 20×2-quad strip mesh, two ends pinned each step, run to convergence, interior linear within 2%.
- Modify: `tests/CMakeLists.txt` — FIRST extract `add_caliper_torch_model_test(name src applet_dir)` from the four existing verbatim blocks (sculpt/mesh/em_pic/obj-if-torch — check which qualify), convert them, then add this suite through it. `ctest -N` before/after lists identical pre-existing tests.

- [ ] Step 1: TDD each unit in the order subdivision → Laplacian → dt → bake.
- [ ] Step 2: Full ctest green. Commit `feat(twin_scope): surface engine — subdivision, cotan Laplacian, texture bake (+ torch-test CMake helper)`.

### Task 7: Thermal model v2 + learner (∥ Task 8a)

**Files:**
- Create: `applets/twin_scope/twin_model.h` [donor as skeleton, physics replaced per spec §3/§4]: single 3-D source-site table (K=4, the ONE definition — spec §7), Gaussian 3-D injection, duty-cycle schedule (seeded, per-source offset periods, user-override flag per source), batched explicit step on `(B=50, V_sim)` using T6's L/M/dt, area-weighted loss term; `ThermalLearner` 3+K→64→64→1 with a LOCAL `torch::Generator` for sampling (no global `manual_seed` — donor review Minor).
- Create: `tests/test_twin_thermal.cpp` [donor adapted]: energy decay to ambient under zero sources; batched step == 50 independent steps; duty-cycle determinism (same seed → same schedule); learner loss decreases on a fixed run, prediction finite/in-bounds at texel positions.
- Modify: `tests/CMakeLists.txt` (one `add_caliper_torch_model_test` line).

- [ ] Step 1: TDD; full ctest green. Commit `feat(twin_scope): surface thermal model + chasing learner`.

### Task 8a: Shared camera header + adapter contract + gpt_scope gate (small, ∥ Task 7)

**Files:**
- Create: `sdk/include/caliper/adapters/orbit_camera.hpp` — hoist the donor applet math verbatim (`git show codex/twinscope-implementation:applets/twin_scope/twin_scope.cpp` lines ~35-90): V3 ops, look_at, perspective (document the Vulkan z-[0,1]/negative-viewport convention, flow_scope.cpp:74-76 wording), plus `orbit_eye(azimuth, elevation, distance, target)` and `cursor_ray(eye, target, fov_deg, aspect, ndc_x, ndc_y)`.
- Modify: `sdk/include/caliper/adapters/torch.hpp` — doc contract on `stream_to_tensor`: "FRAME-THREAD WARNING: without STREAM_ORDERED this degrades to synced_to_tensor (full device barrier). Gate on the cap before calling from draw_ui." No behavior change.
- Modify: `applets/gpt_scope/gpt_scope.cpp` (~line 1317) — gate its draw-path `stream_to_tensor` on its existing `bridge_caps` STREAM_ORDERED snapshot; cap absent → its existing v1 update_texture fallback.

- [ ] Step 1: Implement; build gpt_scope + full ctest green; grep proof `stream_to_tensor` has no ungated draw_ui caller. Commit `fix(sdk,gpt_scope): orbit camera header; frame-thread contract on stream_to_tensor; gate gpt_scope`.

### Task 8b: The TwinScope v2 applet

**Files:**
- Create: `applets/twin_scope/twin_scope.h`, `twin_scope.cpp`, `plugin.cpp`, `twin_scope.caliper.toml`, `applets/twin_scope/CMakeLists.txt` [donor as skeleton — worker/job structure, triple-buffered publish, ExportablePool use, ladder, ray-cast picking, ImGui toolbar are all transcribable; REPLACE per spec §6-§9 and fold §8 fixes]:
  - init: DLL-relative asset resolution (module_dir() helper; staged CMake copy is the consumer; source-tree macro fallback; log the choice) [spec §8.c]
  - init job: load OBJ → subdivide → L/M/dt → bake S/gutter → move to device
  - worker: sim+train uncapped, publish ≤30 Hz, publishes tagged with mode [§8.d], ONE `.to(kCPU)` per publish with CPU-side peak [§8.f], no hot-loop sleep (idle branch only), duty-cycle sources with user overrides
  - draw_ui: split hero view (two draws, ±x model offsets, sim|net textures), sim/net/|error| single-mesh mode, R2 textured↔per-vertex toggle (per-vertex = `state[:, :V_render]` COLORMAP), HUD (provenance line, steps/s + train/s, loss sparkline, MAGMA °C legend), wireframe overlay; STREAM_ORDERED gate on the imported-texture path with honest status [§8.b]; ladder rungs re-evaluated on draw failure (textured refusal → per-vertex, not heatmap) [spec §9]; orbit_camera.hpp for draw AND pick (one eye); direct store protocol on source widgets; shared_ptr field snapshot under the mutex.
- Modify: root `CMakeLists.txt` (add_subdirectory) following how sibling applets register.

- [ ] Step 1: Transcribe skeleton, integrate T5/T6/T7 pieces, implement views/HUD.
- [ ] Step 2: Build all; full ctest green; log-based run proof — Vulkan+CUDA autolaunch (expect: renderer "geometry path OK", applet zero-copy provenance line, both split halves rendering, toggle switches paths in the log) and `CALIPER_RENDERER=gl` (expect honest ladder line). No GUI clicking (ledger lesson); short sessions.
- [ ] Step 3: Commit `feat(twin_scope): the surface twin — split sim|net view, R2 toggle, honest ladder`.

### Task 9: Docs, ledger, final whole-branch review

- [ ] Step 1: Update `GEOMETRY.md` (v1_2 status → shipped-with-caveat mac-pending), `ROADMAP.md` (R2 row), ledger (per-task lines already appended; add the mac-pending checklist: Metal build, Metal gfx rows, Task-3 parity gates).
- [ ] Step 2: `scripts/review-package $(git merge-base main HEAD) HEAD`; dispatch the final whole-branch reviewer (most capable model) per superpowers:requesting-code-review; fix wave (ONE fixer, complete findings list) if needed.
- [ ] Step 3: Commit docs; stop — merge decision is the human's (superpowers:finishing-a-development-branch).

## Self-Review (write time)

- Spec coverage: §2→T5/T6, §3→T6/T7, §4→T7, §5→T6, §6/§7/§9→T8b, §8.a→T1, §8.b→T8a+T8b, §8.c/§8.d→T8b, §8.e→T3, §8.f→T2/T6/T8a/T8b, §10→T1/T4/T5/T6/T7/T8b. All twelve spec sections land in a task.
- No dangling names: `twin_surface.h` API named once (T6) and consumed by T7/T8b; loader Mesh shape pinned in T5.
- Order-of-need: tests/CMakeLists touched by T5, T6, T7 — sequential by design; only (T2∥T3) and (T7∥T8a) are parallel, both disjoint-file pairs.
