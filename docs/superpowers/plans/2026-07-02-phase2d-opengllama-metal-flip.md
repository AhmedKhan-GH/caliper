# Phase 2D — opengllama Sheds Raw GL + Metal Default Flip + Real-Data Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Step 4 of the ratified Phase-2 sequencing plus a user directive: (1) opengllama's raw-GL heatmap textures migrate onto `tensor_bridge.v1` — the §6c grandfather clause expires, making it the bridge's first non-torch consumer; (2) with no raw-GL applet left, **the macOS default renderer flips GL→Metal** (the gate the spec records); (3) MLScope's demo gains **real-data visualization** — sample digits and conv feature maps as live textures, not just 3×3 kernel tiles.

**Architecture:** opengllama already composes RGBA u8 pixel buffers on CPU (`opengllama.cpp:659-673` uploads via `glTexImage2D`) — each site becomes a 3-D (H,W,4) u8 `CaliperTensor` → `texture_from_tensor`/`update_texture`, `ImGui::Image(bridge.imtex(id))`. MLScope's enrichment reuses the ML-EXEMPLAR 7 worker-snapshot/frame-upload pattern with bigger, real tensors (28×28 probe digit; 8×26×26 conv1 feature maps). The flip changes only `make_renderer`'s default on APPLE + landing-page expectations (cards work, GL-only 3D background absent — documented).

## Global Constraints

- All prior plan constraints carry over (trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`, explicit-path staging, no merge by agents, build/, strict mkdocs when docs change, TDD where units exist — these are ports/glue + one flip, so the §16 gfx suite and existing ctest are the regression net; no new unit frameworks).
- **Branch:** `platform/phase-2d` from `main`.
- **Exit gates:** `grep -rn 'glGenTextures\|glTexImage\|glDeleteTextures\|glBindTexture' applets/` → EMPTY (the §6c sweep); default `./build/caliper` on macOS prints `[renderer] metal`; `CALIPER_RENDERER=gl` still fully works (fallback preserved, spec §5.4 frozen-fallback).
- **Bridge is REQUIRED for opengllama after D1** (manifest moves it from absent to required — it cannot render its heatmaps without it; negotiation enforces).
- **Do not touch:** frozen headers, `src/host/` internals (except `make_renderer` default in D2), other applets, `third_party/`, `cmake-build-debug/`.

## Tasks

### Task D1: opengllama → tensor_bridge.v1 (the §6c expiry)
**Files:** `applets/opengllama/opengllama.cpp` (+`.h` if members change), `applets/opengllama/plugin.cpp` (Bridge probe), `applets/opengllama/opengllama.caliper.toml` (required += tensor_bridge.v1 + ui note), CMakeLists (drop `libglew_static` link if no GL remains).
- Enumerate ALL raw-GL texture sites (`grep -n 'glGenTextures\|glTexImage\|glDeleteTextures' applets/opengllama/` — expect the ctx_text_heatmap site ~659 plus any siblings); replace each: RGBA u8 pixel buffer → `(H,W,4)` u8 CaliperTensor (contiguous, CPU) → create-once/`update_texture`-after (the C8 lifecycle), release in cleanup; `ImTextureID` usages switch to `bridge_.imtex(id)`.
- `caliper::Bridge bridge_` acquired in `on_init` via the plugin adapter (bridge REQUIRED in manifest → present; no null ceremony, but keep the falsy guard for fixture-driven tests of the class if any).
- The §6c grandfather comment in plugin.cpp is DELETED — replaced by one line noting the applet is bridge-native since Phase 2D.
- Exit: applet-dir GL grep empty; build green; app (GL renderer) headless 10s; visual = human checklist (D4).
- Commit: `feat(opengllama): heatmaps via tensor_bridge.v1 — raw GL removed, §6c grandfather expired`.

### Task D2: macOS default renderer flips to Metal
**Files:** `src/main.cpp` (or wherever `make_renderer`'s default resolves), `docs/wiki/explanation/rendering.md` (status table row flips).
- Default on APPLE: Metal; `CALIPER_RENDERER=gl` selects the frozen fallback; non-APPLE default unchanged (GL). Fallback-on-init-failure path retained.
- Landing page on Metal: cards/launch/dashboard work; 3D background absent (GL-only IntroScreen) — the stderr line + rendering.md note state it plainly; full landing parity is recorded as a 2D-migration follow-up in the plan's notes, NOT silently regressed.
- Exit: bare `./build/caliper` → `[renderer] metal`; `-L gfx` + full ctest green; both env overrides verified headless 10s.
- Commit: `feat(host): Metal is the macOS default renderer — GL demoted to frozen fallback (§5.4, gate cleared by opengllama migration)`.

### Task D3: MLScope real-data visualization (user directive)
**Files:** `examples/ml_scope/ml_scope.cpp`.
- New "data" panel via the established worker-snapshot pattern: (a) the CURRENT probe digit (the test image whose prediction is shown) as a 28×28 f32 tensor → `texture_from_tensor_mapped` grayscale-ish (VIRIDIS, vmin 0 vmax 1), rendered ~112×112; (b) its **conv1 feature maps** — forward the probe through conv1 only, snapshot `(8,26,26)` → 8 mapped textures in a 4×2 grid beside the kernels (same RdBu symmetric treatment); (c) predicted vs true label caption, updating every eval tick.
- Worker computes + syncs + publishes under the mutex (never calls the bridge); frame uploads on generation change (exact C8 discipline); textures create-once/update-after; released after the cleanup wait.
- On Metal these are device tensors (zero CPU staging); on GL the C8 relocate-fallback already generalizes — reuse it.
- Exit: build + all ctest labels green; both renderers headless 10s; visuals = human checklist.
- Commit: `feat(ml_scope): real-data visualization — probe digit + conv1 feature maps live via the bridge`.

### Task D4: docs + demo checklist + (orchestrator) merge
**Files:** `docs/wiki/reference/services/tensor-bridge-v1.md` (demo checklist extended: opengllama heatmaps + MLScope data panel + the default-flip expectations), `docs/wiki/explanation/rendering.md` (if D2 left anything), `docs/wiki/howto/port-v1-applet.md` (short "porting raw GL to the bridge" subsection distilled from D1), `PLATFORM.md` §17 Phase-2 sequencing item (4) marked done-style note optional — NO, leave spec untouched; strict mkdocs green.
- Commit: `docs(wiki): Phase-2D semantics — bridge-native opengllama, Metal default, real-data demo checklist`.

## Exit Criteria
| Requirement | Proof |
|---|---|
| No raw GL in any applet (§6c fully enforced) | D1 grep gate |
| macOS default = Metal, GL fallback intact | D2 stderr + env-override checks |
| Bridge's first non-torch consumer works | D1 + human checklist (heatmaps identical to pre-port) |
| Real-data ML visualization | D3 + human checklist |
| Suite green throughout (unit + gfx + torch labels) | every task |

## Risks / Notes
- opengllama's heatmap composes on CPU already — this port does NOT regress performance (same CPU compose, upload path swaps); GPU-side compose is a possible later enhancement, out of scope.
- The Metal landing page loses the 3D background (IntroScreen is GL-only) — accepted, documented; full parity is a follow-up when the intro migrates (post-2F polish or Phase 4 dev-mode work).
- MLScope feature-map forward pass runs on the worker at eval cadence (~18×/run) — negligible cost, no frame-thread work.
