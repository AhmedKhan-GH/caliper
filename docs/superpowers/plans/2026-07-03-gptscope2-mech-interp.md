# GPTScope 2 — mechanistic insight into a live-training char GPT

**Goal:** replace the archived GPTScope's "visual slop" (wall-of-attention-heatmaps)
with visualizations of the transformer's ACTUAL mechanics, live during training on
TinyShakespeare. Every panel answers a named question about the architecture or the
data. New applet at `applets/gpt_scope/` (id `dev.caliper.gpt-scope`, version
**0.2.0**, name `GPTScope`, tag `ML`). The archive (`applets/legacy-dev/gpt_scope/`)
stays untouched; copy its `gpt_model.h` + download/training skeleton as the base.

## Model (same scale, richer forward)

4L/4H/128d char transformer, block 128, TinyShakespeare (same S3-tolerant download
recipe as archived version — url in its source; atomic .tmp+rename; cancellable).
The hand-written attention is the asset: extend the model with a
**`forward_full`** that returns, for a probe batch (1,T):
- `resid[l]` — the residual stream AFTER each block, l = 0..4 (0 = embeddings+pos)
- `attn[l]` — per-layer attention weights (H,T,T)
- `attn_out_norm[l]`, `mlp_out_norm[l]` — L2 norm of each sublayer's WRITE into the
  residual stream (mean over positions)
Normal `forward` for training stays as-is (loss path unchanged, no perf cost).

## The panels (each its own docked ImGui window; host tiles them)

### 1. `GPTScope: Logit Lens` — "when does the model decide?"
THE centerpiece. Fixed probe text (~48 chars of Shakespeare, constant across the
run). At each depth l, project the residual through the final LayerNorm + unembed:
`logits_l = ln_f(resid[l]) @ W_U`. Render a **grid: rows = depth (emb, L1..L4),
cols = position**: each cell shows the top-1 predicted next char, colored GREEN if
it matches the actual next char, else a red→yellow ramp by the probability it
assigns the correct char. Rendered with ImGui text cells (monospace, colored) — no
bridge needed, it must be READABLE. Bottom row = the actual text.
**Insight:** predictions crystallize with depth; early in training all rows are
noise, then lower rows (deep layers) organize first... or do they? — that's the
show. Publish cadence ~1 Hz (probe forward is tiny).

### 2. `GPTScope: Heads` — "what did each head become?"
The anti-slop attention view: a scatter of ALL 16 heads, x = **mean attended
distance** (Σ p(i,j)·(i−j), averaged over queries), y = **attention entropy**
(mean −Σ p log p), point color by layer, computed on the probe. Watch heads
DIFFERENTIATE during training: local/positional heads drop toward (1, low),
diffuse heads sit high, prev-token heads pin x≈1.
**Drill-down redeems the heatmap:** hover/click a head point → that ONE head's
(T,T) attention pattern for the probe as a bridge texture (MAGMA, row-normalized)
beside the scatter, with the probe text on both axes explained in a tooltip.
Heatmap-on-demand-with-a-question ≠ wallpaper.

### 3. `GPTScope: Embeddings` — "what has it learned about characters?"
The token embedding matrix W_E (vocab≈65 × 128) PCA-projected to 3-D (top-3 PCs,
torch SVD on CPU — 65×128 is trivial), drawn with **ImPlot3D::PlotText: each point
IS its character glyph** (space shown as '␣', newline '⏎'). Color classes: vowels /
consonants / digits / punctuation / uppercase. Fixed axes after first fit
(EmbedScope's policy; Refit button).
**Insight:** watch vowels find each other, case pair up, punctuation exile itself —
the model's phonotactics emerging from nothing.

### 4. `GPTScope: Residual` — "who writes what, where?"
Two small plots: (a) bar chart per layer of attn-write vs MLP-write norms into the
residual stream (from forward_full) — division of labor between attention and MLP
across depth; (b) per-layer gradient norm lines over training steps (per-step or
every N; grouped by parameter name prefix) — which layers are learning NOW.

### 5. `GPTScope: Sample` — "what does it believe as it speaks?"
Live sampling (~every 2 s or on-demand button + temperature slider), but
**confidence-colored**: each generated char tinted by the probability it was
sampled with (dim red = desperate guess, bright green = confident). Below: for the
LAST generated position, a top-8 bar chart of candidate next chars.
Also the probe text rendered with **per-position loss coloring** (which characters
does the model find hard?).

### 6. `GPTScope: Training` — controls + curves
Train/Cancel/progress, loss + val perplexity curves (follow-toggle idiom),
metrics.v1 streaming (experiment `tinyshakespeare`, run name `gpt2-mech`),
**Save/Load via artifacts.v1** (EmbedScope's exact pattern: save = torch::save to
ostringstream + put("gptscope-model", ..., run); load = path_of on frame thread →
job that loads + runs one probe pass, NO training).

## Cadences (cookbook §2)
- loss scalar: every step. grad-norm lines: every step (cheap norms).
- probe bundle (lens, heads, residual writes): ~1 Hz (time-gated in worker).
- embeddings PCA: every ~5 s (SVD on CPU, still cheap; no need for faster).
- sample: every ~2 s when auto is on (checkbox), or button.
- ALL publishes: owned CPU copies under the one mutex + gen counters (probe tensors
  are tiny — no device-handle machinery needed here EXCEPT the selected-head
  heatmap texture which follows EmbedScope's disp-tensor pattern or plain CPU f32).

## Services
required: ui, log, jobs, device. optional: metrics, tensor_bridge (head drill-down
only — degrade with "bridge absent (ok)"), artifacts (Save/Load), data.v1 NOT used
(no honest tabular need — do not force it).

## Constraints
- Follow ALL cookbook idioms (threading spine, cancel ≤100ms incl. sampling loops,
  atomic download cache reusing the archived recipe, texture lifecycle, viewport
  policy per panel type, visible degradation).
- Dev hooks: honor `CALIPER_GPT_AUTOTRAIN=1` (press Train frame 1) mirroring
  EmbedScope's.
- Window titles exactly: `GPTScope: Logit Lens`, `GPTScope: Heads`,
  `GPTScope: Embeddings`, `GPTScope: Residual`, `GPTScope: Sample`,
  `GPTScope: Training` (host dock list will be updated to tile them).
- id/version byte-match manifest↔descriptor; glob auto-builds; torch link + rpath
  per exemplar CMake.
- No SetNextWindowPos/Size. No raw GL. ~1100 lines target; clarity over cleverness;
  comment each panel with the QUESTION it answers.

## Verification (artifacts only)
Full build; all 3 ctest suites; `CALIPER_AUTOLAUNCH=dev.caliper.gpt-scope
CALIPER_GPT_AUTOTRAIN=1 CALIPER_EXIT_AFTER=60 ./build/caliper` exits 0 on BOTH
renderers (dataset downloads or is cached at
`~/Library/Application Support/Caliper/data/dev.caliper.gpt-scope/` — the archived
version's cache may exist as sibling dir; REUSE via sibling lookup like EmbedScope
does for MNIST); metrics run row appears (check via duckdb if unlocked, else log);
clean-exit stress x3.

## Out of scope
Induction-head detection metrics, activation patching, SAE features — future work;
the architecture (forward_full seam) leaves room.
