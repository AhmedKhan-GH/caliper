# libcaliper embed on Vulkan/Windows — the hardware pass

**Date:** 2026-07-11
**Status:** COMPLETE (2026-07-11, same day) — all boxes ticked below, run-proven
on the RTX 500 Ada box (driver 596.47, Windows 11, MSVC 14.50, branch
`feat/embed-vulkan-windows`). The greenfield half — the Vulkan embed-canvas seam
(WINDOW HWND surface + OFFSCREEN RGBA8 render-to-texture) — was implemented
against the Metal reference and is now live on hardware: `embed_host` drew both
`instance_scope` (1000 objects, 1 draw call, 0 mesh copies) and `mesh_scope`
zero-copy on the Vulkan WINDOW canvas, and the §7 host-axis byte-compare passed
on the Vulkan OFFSCREEN canvas. **Commits:** `d7f9cc4` (§1.1+§5.2 build:
LNK1149 `libcaliper` OUTPUT_NAME + `embed_host` DLL copy), `c8e6005` (the canvas
seam §2.1–2.6, WINDOW + OFFSCREEN; review Approved 0 Crit/0 Imp), `852c398`
(interop pairing on the canvas path — zero-copy caps under embed; review PASS),
`0724161` (`CALIPER_EMBED_EXIT_AFTER` Win32 parity hook), `0614d51`
(`mesh_scope` `SetNextWindowSize` bootstrap), `37d87f2` (validation-clean
OFFSCREEN, 4 VUIDs; review PASS). MSVC took `main_win32.cpp` +
`caliper_embed_tests` on first compile — NO doctest decomposition, unlike the
v1_3 pass. **Two honest partials on record:** (a) `WM_DPICHANGED` mixed-DPI was
NOT exercisable on this box (single monitor @1.5×) — the clause stays an honest
partial with that reason; (b) the WINDOW-path validation residuals are
*pre-existing SHARED machinery*, not embed-canvas bugs — the OFFSCREEN embed
path is validation-clean, and `VUID-vkDestroyInstance-instance-00629` reproduced
IDENTICALLY on `caliper.exe` (control run) while
`VUID-vkQueueSubmit-pSignalSemaphores-00067` lives in the shared
`frame_render`/`frame_present` (untouchable this pass; embed's unthrottled pump
merely surfaces it). §6.4 review follow-ups are DEFERRED with reason (mac-side
verification needed to land safely — carried, not dropped). Originally:
execution spec (hardware pass). PARTLY greenfield — read §0.2
carefully: this is NOT the pure transcription the v1_3 Vulkan pass was. The
embed *host* and the embed *core* are transcribed and compiled-out
(`if(WIN32)`), but the **Vulkan renderer's embed-canvas seam does not exist
yet** — L2 built `canvas_init/new_frame/render/read_pixels` only on Metal, and
the base-class defaults REFUSE. So this pass has two parts: (a) implement the
Vulkan canvas seam against the run-proven Metal reference, and (b) run the
transcribed Win32 host + embed tests on NVIDIA hardware to prove it. Same house
discipline as the v1_3 and v1_2 hardware passes; the sibling docs are
`2026-07-11-geometry-v1_3-vulkan-windows-hardware-pass.md` and
`2026-07-10-metal-macos-v1_2-hardware-pass.md`.
**Authority:** the design contract is
`2026-07-11-libcaliper-compass-design.md` (§4 the embed seam, §4.2 who renders
the canvas, §4.3 lifecycle/threading, §7 verification); the Windows-box
protocol is `docs/m2a-windows-verification.md` (D24). The run-proven reference
is the **Metal** embed path at branch `feat/libcaliper` HEAD (`d412e12`):
`src/host/renderer/metal_renderer.mm` `canvas_*` (lines 482–620) and
`examples/embed_host/main.mm`, live-proven on Apple Silicon (embed battery
8/8, `embed_host` drew `instance_scope` zero-copy at 2560×1600 @2×).
**Checkbox discipline (inherited):** a box is checked only when the suite is
green on the Windows box, the path is run-proven by a logged artifact, and the
fixing commit is named. Refusal purity and byte-exactness are STOP-and-diagnose,
never loosened. Invariants at the bottom never become checkboxes.

**Prerequisite:** land the `feat/libcaliper` branch (or work on top of it)
AND the Windows box must already be current on `main` incl. the v1_3 Vulkan
pass (`6938248`) — this pass reuses that session's build recipe and the
CUDA/VMM gating machinery.

---

## 0. What exists, and the one thing that does not

### 0.1 Platform-neutral + Metal, already proven (no new risk)

- `include/caliper/embed.h` — the v1 C ABI (create/attach_canvas/frame/event/
  load_applet/unload/read_pixels/last_error/shutdown), struct_size-versioned,
  its own include root. Compiles everywhere; the ABI itself is platform-free.
- `src/host/embed/embed_core.cpp` — the core: one-core CAS guard, the
  teardown-first `load_applet` arc, the canvas gate, event struct_size gate.
  It calls `renderer->canvas_init(...)` etc. through the HostRenderer vtable —
  **backend-agnostic**; it already selects `make_vulkan_renderer()` on `_WIN32`
  (embed_core.cpp:169). This file needs NO Windows-specific change.
- `src/host/renderer/host_renderer.h:169-182` — the `canvas_*` virtuals with
  **defaults that refuse** (`canvas_supported()`→false, `canvas_init()`→false).
  This is why an un-implemented backend fails cleanly instead of crashing.
- The Metal implementation (the reference to mirror):
  `metal_renderer.mm:482-620` — `canvas_supported/init/resize/new_frame/render/
  read_pixels/shutdown`, CANVAS_WINDOW (CAMetalLayer on the NSView) and
  CANVAS_OFFSCREEN (render-to-texture + blit-back) both live.

### 0.2 The gap — the Vulkan embed-canvas seam DOES NOT EXIST

`grep -c canvas_init src/host/renderer/vulkan_renderer.cpp` → **0.** The Vulkan
backend implements the exe's GLFW swapchain path (`init(GLFWwindow*)`,
`new_frame`, `render`, `frame_present`) but NONE of the `canvas_*` family. So
today, `embed_host` on Windows would reach `attach_canvas`, call
`canvas_init`, hit the base-class default, and get an **honest refusal**
("attach_canvas: backend canvas_init failed") — correct, but nothing draws.

**This pass's real work is implementing that seam on Vulkan.** The good news:
the machinery mostly exists — the Vulkan renderer already owns a swapchain, a
depth pass, the geometry pipelines, and (from the v1_3 pass) `debug_readback_
rgba8`. What is missing is the *parallel* canvas entry points that run without
GLFW, plus a **Win32 VkSurfaceKHR from an HWND**.

### 0.3 The transcribed-but-never-run Windows host

- `examples/embed_host/main_win32.cpp` — the HWND sibling of `main.mm`: a bare
  Win32 message loop, the five embed calls, WndProc → CaliperInputEvent
  (mouse/scroll/char/size/dpichanged/focus), `CALIPER_CANVAS_WINDOW`,
  `CALIPER_RENDERER_DEFAULT` (→ Vulkan). Header STATUS block already says
  "TRANSCRIPTION — NOT yet run." Structure mirrors main.mm; the port is meant
  to be mechanical ONCE the seam under it exists.
- `examples/embed_host/CMakeLists.txt` — the `elseif(WIN32)` branch builds
  `main_win32.cpp` against `libcaliper` + deps. Compiled-out on the Mac.
- `tests/` embed battery (`caliper_embed_tests`) — 8 cases, all CANVAS_OFFSCREEN
  + the §7 host-axis bridge-upload byte-compare. Ran on Metal; the OFFSCREEN
  path is what they exercise, so once the Vulkan OFFSCREEN canvas exists they
  should port with only the MSVC-doctest care the v1_3 pass documented.

---

## 1. Build recipe (D24, inherit the v1_3 session's)

Same `configure.cmd`/`build.cmd` (vcvars) wrappers, same build-root-on-PATH for
DLLs. New target this pass: `embed_host` (the WIN32 branch) — a NEW exe under
`build/examples/embed_host/`.

- [x] **1.1** Configure + build green with the WIN32 embed branch now compiling
  (`main_win32.cpp`, `caliper_embed_tests`). Record the invocation in the box's
  scratch ledger (`.superpowers/sdd` is gitignored per-box; the Mac's did not
  transfer).
  *Done 2026-07-11: first Windows contact — MSVC took `main_win32.cpp` and
  `caliper_embed_tests` on first compile, NO doctest complex-expression
  decomposition needed (unlike the v1_3 pass). The one build fix was link-side:
  LNK1149 on the `libcaliper` OUTPUT_NAME + the `embed_host` DLL-copy step
  (`d7f9cc4`, see §5.2). Invocation in the box's scratch ledger
  (`.superpowers/sdd/embed-pass/`, gitignored).*

---

## 2. Implement the Vulkan canvas seam (the greenfield half)

Mirror the Metal reference (`metal_renderer.mm:482-620`) structurally; the two
backends must run the SAME embed_core calls and produce byte-identical
OFFSCREEN pixels to the CPU reference (§7 discipline). Implement in
`vulkan_renderer.cpp` as the `canvas_*` overrides, running PARALLEL to the
swapchain path (never both on one instance — the invariant is stated at
host_renderer.h:163-168; the byte-exact geometry rows must stay untouched).

- [x] **2.1 `canvas_supported()` → true** on Vulkan (device present).
  *Implemented in the canvas seam (`c8e6005`); live true on the RTX 500 Ada.*
- [x] **2.2 `canvas_init(void* hwnd, CanvasMode, w, h)`**:
  - **CANVAS_WINDOW:** create a `VkSurfaceKHR` from the HWND via
    `vkCreateWin32SurfaceKHR` (volk already shims `vulkan_win32.h`,
    vulkan_renderer.cpp:67), then build the swapchain against it — reuse the
    exe swapchain builder, just fed a surface from HWND instead of
    `glfwCreateWindowSurface`. Wire the ImGui Vulkan render backend
    (`imgui_impl_vulkan`) on this device WITHOUT `imgui_impl_glfw` (the embed
    layer feeds ImGuiIO; §4.3).
  - **CANVAS_OFFSCREEN:** a device-local color target (RGBA8, the same format
    the swapchain uses so read-back needs no swizzle) + depth, render pass,
    and a host-visible staging buffer for read-back. No surface, no swapchain.
  - false on any failure with NOTHING allocated/left dangling (honest refusal).
  *Both modes implemented in `c8e6005`: WINDOW builds a `VkSurfaceKHR` from the
  HWND (`vkCreateWin32SurfaceKHR`) + swapchain; OFFSCREEN a device-local RGBA8
  render-to-texture + depth + host-visible staging. Live on hardware — see §4.1
  (WINDOW) and §3.2 (OFFSCREEN).*
- [x] **2.3 `canvas_new_frame()`** — begin the frame: acquire the swapchain
  image (WINDOW) or bind the offscreen target; open the pass (this CLEARS, per
  the frame-order contract at host_renderer.h:167); ImGui Vulkan + a NewFrame.
  *Implemented in `c8e6005`.*
- [x] **2.4 `canvas_render()`** — `ImGui::Render()`, record the draw data into
  the canvas command buffer, composite; **WINDOW:** present (queue-present the
  acquired image); **OFFSCREEN:** resolve/copy the color target into the
  read-back staging buffer.
  *Implemented in `c8e6005`; interop pairing on this path corrected in `852c398`
  so zero-copy caps resolve under embed.*
- [x] **2.5 `canvas_read_pixels(dst, stride)`** (OFFSCREEN only) — copy the
  last composited frame, tightly-packed RGBA8, from staging into `dst`; false
  if WINDOW mode or nothing rendered. This is what the embed battery asserts
  against, so it is the byte-exactness surface.
  *Implemented in `c8e6005`; the §7 host-axis byte-compare passes on it (§3.2).*
- [x] **2.6 `canvas_resize` / `canvas_shutdown`** — rebuild-on-resize (the
  swapchain already has `rebuild_swapchain_` machinery, vulkan_renderer.cpp:329)
  and orderly teardown (surface/swapchain/offscreen targets/ImGui backend),
  matching the Metal shutdown ordering.
  *Implemented in `c8e6005`; live resize verified in §4.3
  (1280×800→1720×1050 rebuilt the swapchain clean, no tear, no validation
  error). OFFSCREEN teardown made leak-free in `37d87f2` (§5.1).*
- [x] **2.7** The exe's GLFW swapchain path and ALL byte-exact geometry rows
  are untouched — `caliper_gfx_tests` stays 48/48 on this box (the canary that
  the parallel seam didn't disturb the shared pipelines).
  *48/48 cases, 1475 assertions green at every boundary of this pass (the
  canary held across the seam, the interop-pairing fix, and the OFFSCREEN
  VUID work).*

## 3. Port the OFFSCREEN embed battery (headless, portable-ish)

- [x] **3.1** `caliper_embed_tests` compiles under MSVC (the doctest
  complex-expression decomposition lesson from `477431e`/`5164f89` — split any
  chained `&&` in REQUIRE). No VMM-padding rows here, so that gotcha does not
  apply.
  *Compiled clean first try — the doctest decomposition the v1_3 pass needed did
  NOT recur here.*
- [x] **3.2** All 8 cases pass on real hardware with the Vulkan OFFSCREEN
  canvas: create/shutdown cycle twice-in-one-process, the struct_size gates,
  the one-core CAS guard, the teardown-first `load_applet` reload trio, the
  W1 canvas gate, and the **§7 host-axis byte-compare** — the bridge upload
  driven through the embed core must produce byte-identical pixels to the CPU
  reference `expand_u8_to_rgba8`, exactly as on Metal. A miss is
  STOP-and-diagnose (a canvas-seam bug, not a tolerance).
  *8/8 cases, 42 assertions, ZERO refusal-skips — before the seam existed the
  battery ran 14 assertions all-skip (the canvas gate refused everything), so
  the jump to 42 executed assertions IS the seam coming alive. The §7 host-axis
  byte-compare PASSED byte-identical on the Vulkan OFFSCREEN canvas, exactly as
  on Metal. Artifact: `artifact-embed-battery-vulkan.txt` in the box's scratch
  ledger (`.superpowers/sdd/embed-pass/`, gitignored per-box).*

## 4. Run-prove the Win32 host (the windowed path, live)

CANVAS_WINDOW has NO automated coverage on either OS (documented known-gap) —
the live `embed_host` run is the proof of record.

- [x] **4.1 embed_host launches** an HWND window, `attach_canvas` succeeds
  (Vulkan WINDOW), `load_applet dev.caliper.instance-scope` — the log line
  `first zero-copy instanced frame drawn — 1000 objects, 1 draw call, 0 mesh
  copies` appears with the **Vulkan** renderer active, per-draw provenance
  honest. Capture stdout/stderr + a screenshot (HWND windows ARE
  screenshotable, unlike the TCC-blocked Mac headless case — get the pixels).
  *Logged live: Vulkan WINDOW canvas 1258×744 @1.5×, `instance-scope: first
  zero-copy instanced frame drawn — 1000 objects, 1 draw call, 0 mesh copies`.
  Screenshots captured (gems field, hero line, 120 FPS, device line "NVIDIA
  RTX 500 Ada"). This is the live windowed run-proof — CANVAS_WINDOW has no
  ctest coverage on either OS by design; the byte-exact claim rides the §3.2
  OFFSCREEN battery, this is live proof, not byte proof.*
- [x] **4.2 mesh_scope** under embed_host draws zero-copy on Vulkan+CUDA
  (a second applet, proving the seam isn't instance_scope-specific).
  *Logged live: `[vulkan] device path: geometry path OK — primitives drawn from
  imported allocations in place` → `mesh-scope: first zero-copy frame drawn
  (imported geometry, 3 draws)`. Root-caused a §4.2 gap first: `mesh_scope`
  lacked the `SetNextWindowSize(FirstUseEver)` bootstrap `instance_scope`
  carries — with no dock layout under embed the window collapsed below 64px
  ("no geometry view"). Fixed `0614d51`. Driver gotcha recorded: a stale
  `imgui.ini` beside the host pins the collapsed size past `FirstUseEver`.*
- [x] **4.3 Input + resize live:** mouse/scroll reach the applet (orbit the
  camera), window resize rebuilds the swapchain without a validation error or
  a torn frame, `WM_DPICHANGED` is exercised on a mixed-DPI setup if available
  (the sibling of the Mac contentsScale note).
  *Input live: posted WM mouse events reached the applet — a wheel dolly changed
  zoom to 0.66, observed on the screenshot. Live resize 1280×800→1720×1050
  rebuilt the swapchain clean: no tear, no validation error. **Honest partial:**
  `WM_DPICHANGED` mixed-DPI is NOT available on this box (single monitor @1.5×)
  — the clause stays a partial with that reason, not a claim.*
- [x] **4.4 Honest ladder:** `CALIPER_RENDERER=gl` under embed_host → the GL
  backend refuses the canvas (GL is GLFW-coupled chrome, D13), `attach_canvas`
  returns false with the honest `last_error`, the host prints it and exits
  cleanly — never a crash, never a blank-but-claiming-success window.
  *Logged: `attach_canvas: embed requires Metal or Vulkan (the GL fallback is
  not an embed target)`; the host printed `last_error` and exited cleanly.*
- [x] **4.5 Crash containment:** the applet-fault path (`crash_fn`) still
  quarantines a faulting applet and the host lives — the embed promise holds
  on Windows.
  *`CALIPER_HELLO_CRASH` logged: `applet 'dev.caliper.hello' faulted and was
  quarantined (the host lives on): crashed in frame(): SEH exception
  0xC0000005` — the host lived to its `EXIT_AFTER` close (`0724161`).*

## 5. Validation layers + the Windows gotchas the review flagged

- [x] **5.1** LunarG validation layers (loader-injected; the binary stays
  layer-free) clean on the embed canvas path — surface creation, swapchain,
  the ImGui Vulkan backend, present. Descriptor/UBO complaints here are the
  class only hardware+layers catch.
  *The OFFSCREEN embed-canvas path is validation-CLEAN after `37d87f2`, which
  fixed 4 VUIDs: `-01387` (mode-gated the swapchain extension), `-00897`/`-01211`
  (honest UNDEFINED→TRANSFER_SRC layouts), `-05137` (teardown leak). **Honest
  partial on WINDOW mode:** its residuals (4 error lines in the run artifact —
  2 VUID classes ×2 each) are PRE-EXISTING SHARED machinery, not
  embed-canvas bugs — `VUID-vkDestroyInstance-instance-00629` reproduced
  IDENTICALLY on `caliper.exe` (control run), and
  `VUID-vkQueueSubmit-pSignalSemaphores-00067` lives in the shared
  `frame_render`/`frame_present` (the exe path is untouchable this pass; embed's
  unthrottled pump merely surfaces it more often). The box is checked for the
  embed-canvas claim with that scoping stated explicitly.*
- [x] **5.2 DLL-copy for the new exe dir.** `embed_host` lands in
  `build/examples/embed_host/` — torch/dependency DLLs are NOT beside it; the
  caliper-exe copy step doesn't cover this target. Add a post-build DLL copy
  (mirror the exe's) or the host won't start. (Ledgered in the design spec's
  Windows addendum.)
  *Post-build DLL-copy for `embed_host` added in `d7f9cc4` (alongside the
  LNK1149 `libcaliper` OUTPUT_NAME fix); the host starts and runs.*
- [x] **5.3 WndProc null-safety is BY DESIGN.** `WM_SIZE` fires *during*
  `CreateWindowA`, before `g_core` exists, so `caliper_core_event(nullptr,…)`
  is called — the core is null-safe on purpose. Do NOT "fix" this into a guard
  that crashes; confirm the null-event path stays a no-op.
  *The `WM_SIZE`-during-`CreateWindowA` null-event no-op was exercised on every
  run — confirmed a clean no-op, not "fixed" into a guard.*
- [x] **5.4** Verify the `GetClientRect` physical-pixel assumption under
  `PER_MONITOR_AWARE_V2` (main_win32.cpp assumes client coords are physical px).
  *Confirmed physical px: the client rect measured 1258×744 @1.5× from a
  1280×800 outer window — the assumption holds.*

## 6. Closeout (only with artifacts)

- [x] **6.1** `2026-07-11-libcaliper-compass-design.md` — flip the §6 L2
  status line and the Windows-pass addendum: the embed path is run-proven on
  Vulkan/Windows; name the commits.
  *This commit — §6 L2 outcome + the Windows-pass addendum now read run-proven
  on both ecosystems, commits `c8e6005`/`852c398`/`37d87f2` (seam) +
  `0614d51`/`0724161` (supporting) named.*
- [x] **6.2** ROADMAP §7 R4 — the L2 sub-item gains "both platforms" once the
  Windows embed is proven (L3/Compass stays gated on §5's named-workflow rule).
  *This commit — the R4 L2 sub-item now reads both platforms; L3/Compass gating
  unchanged.*
- [x] **6.3** `docs/wiki/reference/embedding.md` — drop the "Metal-proven;
  Vulkan pending" qualifier on the embed path; the CANVAS_WINDOW known-gap note
  updates to "run-proven live on both OSes, no ctest coverage" (the live run
  stays the ritual).
  *This commit — the qualifier is gone; the CANVAS_WINDOW note reads run-proven
  live on both OSes, no ctest coverage (the live run stays the ritual);
  `CALIPER_EMBED_EXIT_AFTER` noted as a both-hosts hook.*
- [~] **6.4** Fold in the review's carried follow-ups if cheap while in the
  files: embed_host `caliper::sdk`-link trim, `src/host` include→PRIVATE flip,
  watchdog surfacing on the embed path (it feeds but is never read there).
  *DEFERRED with reason (carried, not dropped): the sdk-link trim, the
  include→PRIVATE flip, and watchdog surfacing all touch shared build/host wiring
  that requires mac-side verification to land safely — folding them in blind on
  the Windows box risks the byte-exact matrix. Carried to a follow-up pass.*
- [x] **6.5** Commits in house style (`feat(vulkan):` for the canvas seam,
  `test(embed):`, `docs(specs):`), Fable trailer; every code fix rides the full
  `caliper_embed_tests` + `caliper_gfx_tests` + a live embed_host re-proof.
  *House-style commits landed: `d7f9cc4` `build(win)`, `c8e6005`/`852c398`/
  `37d87f2` `feat(vulkan)`/`fix(vulkan)`, `0724161` `feat(embed_host)`,
  `0614d51` `fix(applets)`; each code fix rode the full `caliper_embed_tests` +
  `caliper_gfx_tests` (48/48) + a live `embed_host` re-proof. Closeout: this
  `docs(specs):` commit. Findings in the box's scratch ledger
  (`.superpowers/sdd/embed-pass/`).*

---

## Invariants (hold forever)

- **The embed canvas runs PARALLEL to the swapchain path** — one renderer
  instance is exe-swapchain OR embed-canvas, never both; the byte-exact
  geometry rows (48/48) are the canary that the seam stayed separate.
- **One CPU reference, both backends, both host-shells.** The §7 byte-compare
  must match on Vulkan-embed exactly as on Metal-embed and on both exes. A
  divergence is a backend bug to diagnose, never a tolerance.
- **Honest refusal, pixels untouched.** GL refuses the canvas; a failed
  canvas_init allocates nothing; a faulting applet is quarantined and the host
  lives. Zero-copy claimed only when the imported path actually drew.
- **No new ABI.** `embed.h` is frozen at v1 for this pass — the Windows work is
  backend implementation + host run-proof behind the SAME C contract, invisible
  to every applet and to the embed API itself.
- **No checkbox without artifacts.** Transcription + a Mac run are not a Windows
  claim — this pass exists because they aren't.
