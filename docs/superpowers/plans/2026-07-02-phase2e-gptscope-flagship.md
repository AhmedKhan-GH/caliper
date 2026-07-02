# Phase 2E′ — GPTScope: the Flagship Applet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** The flagship (D16): `applets/gpt_scope` — a nanoGPT-style char-level transformer trained on TinyShakespeare, born native on the full service stack. The two demo arcs: **live sampled text evolving from noise → Shakespeare-ish as the model learns**, and **per-head attention as live GPU heatmaps**. This replaces repnet's migration as the Phase-2 generality proof; its exit criterion is the spec's amended Phase-2 exit.

**Architecture:** Manual (non-SDPA) attention so per-head weights are exposable for visualization. Training on `jobs.v1` (worker owns the loop, cooperative cancel), curves via `metrics.v1`, attention maps + any tensor viz via `tensor_bridge.v1` on the frame thread (C8 worker-snapshot/generation discipline), device via `device.v1`. Dataset: single ~1.1 MB text file, downloaded once into data_dir with the B1 recipe (curl, cancellable, atomic cache). All established exemplar contracts (bounded-wait cleanup, curl global init, mutex-published state) carry over from MLScope — GPTScope is written as a real applet, not an exemplar, so comments are normal engineering comments, not numbered teaching points.

**Model config (fixed for v1):** n_layer=4, n_head=4, n_embd=128, block_size=128, dropout 0.1, AdamW lr 3e-4, batch 64, 90/10 train/val split on characters, max_steps 3000 (visible learning in ~2–4 min on M5; val loss every 100 steps).

## Global Constraints
- All prior plan constraints carry over (trailer, explicit-path staging, no agent merges, build/, strict mkdocs on docs changes).
- **Branch:** `platform/phase-2e` from `main`.
- Dataset URL (fixed): `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` → cached as `<data_dir>/tinyshakespeare.txt`.
- Applet id `dev.caliper.gpt-scope`, name `GPTScope`, version `0.1.0`, tag `LLM`; required services ui/log/jobs/device, optional metrics/tensor_bridge (probe patterns).
- The `applets/*` glob auto-registers the target — manifest POST_BUILD copy required as usual; torch + curl linked like ml_scope (no zlib — plain text).
- Checkpoint save/load is EXPLICITLY DEFERRED — it is the designated trigger for `artifacts.v1` (D16 demand-driven clause); a disabled UI button with a tooltip saying so is the honest placeholder.
- **Do not touch:** frozen headers, `src/`, other applets/examples, `third_party/`, `cmake-build-debug/`.

## Tasks

### Task E1: GPTScope — model + training + live sampling
**Files:** `applets/gpt_scope/{gpt_model.h, gpt_scope.cpp, plugin.cpp, gpt_scope.caliper.toml, CMakeLists.txt}`.
- `gpt_model.h`: the transformer (token+pos embedding, N blocks of {manual multi-head causal attention with optional attention-weight output, MLP 4×, layernorms, residuals}, LM head). Manual attention: `att = softmax(mask(QKᵀ/√d))` kept as a tensor so a probe forward can return per-layer `(n_head, T, T)` weights; training forwards skip weight retention.
- `gpt_scope.cpp`: char vocab built from the text; dataset download (B1 recipe: cancellable xferinfo, atomic .tmp+rename, self-heal, offline message) + 90/10 split; training job (cancel per step; loss point per step under mutex; val loss + **200-char sample at temperature 0.8** every 100 steps, published under mutex; metrics streams `train/loss` + `val/loss` per the probe-optional pattern); UI: config header, device line, start/cancel + tray progress, train/val loss plot (step axis), the **live sample panel** (monospace, updating each eval tick — the first demo arc), disabled "save checkpoint" button with the artifacts.v1 tooltip. Cleanup: cancel + bounded wait (contract comment), curl global pairing.
- `plugin.cpp` + manifest per the established adapter shape; byte-identical id/version macro↔toml.
- Verification: build (torch link), full ctest + gfx + torch labels green, both renderers headless 10s; sampling quality = human checklist (E3). Commit: `feat(gpt_scope): flagship — mini-GPT on TinyShakespeare, jobs-trained, live sampling, metrics-streamed`.

### Task E2: Attention visualization via the bridge
**Files:** `applets/gpt_scope/{gpt_model.h, gpt_scope.cpp}` (+toml only if services change — they don't).
- Worker, each eval tick: probe forward on a fixed 64-char excerpt (val text, chosen once per run) with attention retention → snapshot the SELECTED layer's 4 heads as owned `(64,64)` f32 clones (MPS-synced in worker), publish with generation + the probe string + current layer index.
- Frame, on generation/selection change: 4 mapped VIRIDIS textures (vmin 0, vmax per-head max), 2×2 grid ~140px cells, head captions; layer selector (0–3) triggers re-snapshot request (atomic desired-layer the worker reads); the probe text rendered beneath with a hover-highlight: hovering a map row/col highlights the corresponding characters (ImGui text with per-char coloring — the touch that makes attention legible). C8 create/update/recreate lifecycle; release after cleanup wait; bridge-absent → panel says so.
- Verification: suites green, both renderers headless; the visual = human checklist. Commit: `feat(gpt_scope): live per-head attention heatmaps via tensor_bridge (manual attention exposes weights)`.

### Task E3: Polish + docs + merge (orchestrator merges)
**Files:** `applets/gpt_scope/gpt_scope.cpp` (temperature slider 0.2–1.5 affecting next samples; val perplexity readout `exp(val_loss)`), `docs/wiki/index.md` (flagship mention), `docs/wiki/reference/services/tensor-bridge-v1.md` (demo checklist gains the GPTScope items: sample-text evolution, attention grid, layer switching, hover-highlight), `docs/wiki/tutorials/first-applet.md` (cross-link as "the flagship, built entirely on public services").
- Demo checklist (human): (1) launch GPTScope → dataset downloads once → start → loss falls, samples evolve gibberish→words→cadence across ~3 min; (2) attention grid live, layer switch works, hover highlights characters; (3) temperature slider changes sample character; (4) cancel/relaunch clean; (5) run appears in Runs dashboard alongside MLScope history; (6) both renderers.
- Strict mkdocs green; full suites green. Commit: `feat(gpt_scope)+docs: temperature + perplexity; flagship demo checklist`.

## Exit Criteria (= the spec's amended Phase-2 exit, flagship half)
| Requirement | Proof |
|---|---|
| Flagship entirely on public services (ui/log/jobs/device + metrics/bridge) | E1+E2 code; no private service-shaped machinery |
| Attention path zero CPU staging on Metal | bridge device path (§16-verified infra) + human checklist |
| Live sampling demo arc | E1 + human checklist |
| artifacts.v1 demand recorded, not faked | the disabled-button tooltip + ledger note |
| Suites green throughout | every task |

## Risks / Notes
- MPS op coverage for the manual-attention transformer (matmul/softmax/layernorm/embedding/dropout) is solid in libtorch 2.5.1; if any op falls back to CPU with a console warning, note it — don't chase it (torch's known MPS chatter).
- 3000 steps ≈ visible learning, not literary genius — the checklist language sets expectations ("Shakespeare-ish cadence", not sonnets).
- Attention probe forward retains weights only on the probe path — training speed unaffected.
