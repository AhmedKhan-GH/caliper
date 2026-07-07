# Design — GPTScope "ThoughtSpace": the residual stream as a live 3-D constellation

**Status:** approved design, ready for implementation planning (handoff spec —
the implementing agent is expected to have NO other context than this file and
the pointers in §2). **Branch:** work on `feat/flowscope` or a child of it
(needs the `caliper.geometry.v1` machinery merged there).
**Decided:** 2026-07-07 with the user (option 1 of 3: residual constellation;
alternatives — neuron lattice, attention galaxy — rejected because their
geometry is static/derived; here the geometry IS the learned representation).

## 1. What this is

GPTScope today shows attention heatmaps, a logit lens, head-role scatters —
2-D readouts. ThoughtSpace adds the flowscope-class view: **every token's path
through the network, drawn as a thread in 3-D space, hundreds of thousands of
points, zero copies, live during training.**

The transformer's computation is literally a trajectory: a token starts as an
embedding and each block's attention+MLP write moves it through the residual
stream until it becomes a prediction. ThoughtSpace projects every
(sequence, token, depth) residual state of a fixed probe batch into 3-D and
draws stations plus interpolated **trail points** between consecutive depths,
so each token reads as a thread. Two kinds of motion, both live:

- **Training** reorganizes the entire constellation — threads start as noise,
  then straighten/cluster as representations form (embed_scope's blob→lobes
  moment, but for the full depth of a transformer).
- **Generation** (the existing Sample feature) appends one bright white-hot
  thread that grows token by token — the model visibly "thinking" through the
  space it has learned.

Everything rides the verified zero-copy path: the probe tensors are born in
the ExportablePool, imported once, and `geometry.draw_points` reads them in
place. No new host or SDK capability is required.

## 2. Required reading for the implementing agent

- `applets/flow_scope/flow_scope.cpp` — THE reference implementation. Copy its
  patterns verbatim: pool-slot allocation inside `pool.use()` on the worker,
  triple-buffered slots + display_slot invariant, `to_bridge` caching, orbit
  camera + `look_at`/`perspective` helpers, honest status lines, ImPlot3D
  fallback, and the cleanup teardown order (tensors → pool, deliberate leak if
  the worker outlives the cancel grace).
- `applets/gpt_scope/gpt_scope.cpp` — `publish_probe()` (~line 387): the
  existing probe cadence, worker→frame publish-under-mutex + generation
  counter pattern. ThoughtSpace hooks the SAME trigger.
- `applets/gpt_scope/gpt_model.h` — `forward_full()` (~line 194): the existing
  single-sequence residual probe. ThoughtSpace adds a batched sibling (§4.1).
- `sdk/include/caliper/services/geometry_v1.h` — the draw contract (one call =
  clear + draw; count is a per-call parameter; additive blend, no depth).
- `sdk/include/caliper/adapters/exportable_pool.hpp` — pool rules (allocate
  once inside `use()`, worker thread only; `to_bridge` is frame-thread-only).
- `src/main.cpp` `central_windows` list (~line 247) — the ONE host-side line.
- `docs/superpowers/specs/2026-07-07-geometry-flowscope-design.md` — why the
  machinery looks the way it does.

## 3. Non-goals (v1)

- No line/mesh primitives (trails are interpolated points; real polylines are
  a future geometry.v1 additive revision).
- No point picking / hover tooltips (needs an ID buffer or CPU nearest-point
  query; deferred).
- No live resize of the probe batch (S, T, K are compile-time constants; a
  restart applies changes).
- No PCA/learned projection in v1 (fixed random orthonormal basis — stable by
  construction; a "refit PCA" button is a documented future increment with a
  visual-discontinuity warning).
- No second draw pass / compositing: `draw_points` clears every call, so ALL
  points (probe + generation overlay) live in ONE buffer per slot.

## 4. Data model

Constants (top of file, tuned for RTX 500 Ada headroom while training runs):

```
S  = 96        // probe sequences (fixed val slice, deterministic seed)
T  = 96        // tokens per sequence (<= block_size 128)
D  = n_layer+1 // depth stations = 5 (embeddings + after each block)
K  = 5         // interpolated trail points between consecutive stations
N_probe = S*T*(D + (D-1)*K) = 96*96*25 = 230,400
N_gen   = T*(D + (D-1)*K)   =    2,400   // one generation thread
N_max   = N_probe + N_gen   =  232,800 points
```

### 4.1 Model change (additive, `gpt_model.h`)

Add beside `forward_full()`:

```cpp
// Batched residual probe for ThoughtSpace: eval+no_grad forward of idx (S,T)
// keeping EVERY batch element, returning the stacked per-depth residual
// stream (D, S, T, C) on the model's device. No attention weights, no write
// norms — cheaper than forward_full. Training forward is untouched.
torch::Tensor forward_resid(const torch::Tensor& idx);
```

Implementation mirrors `forward_full` (embeddings + per-block loop, stack the
D station tensors), minus the `select(0,0)`, `need_weights`, and norm
plumbing. Restores train/eval mode the same way.

### 4.2 Projection: 128-d → 3-D

- Basis: fixed random orthonormal `P (C,3)` — `torch::randn({C,3})` with a
  hard-coded seed, QR-orthonormalized once at init, kept on device. Stable
  across steps ⇒ all motion on screen is MODEL change, never basis change.
- Depth scale normalization: residual norms grow with depth, which would read
  as depth = radius and swamp shape. Divide each depth station by its
  per-depth mean L2 norm (computed on the current probe, detached) before
  projecting. Toggle "raw norms" in the toolbar disables this (radius then
  honestly shows residual growth).
- Station positions: `stations (D,S,T,3) = normalize_d(resid) @ P`, then
  scaled so the cloud fits ~[-1.5, 1.5]^3 (divide by a running max, smoothed).

### 4.3 Trails

For each consecutive station pair (d, d+1): K points linearly interpolated in
projected 3-D space (lerp AFTER projection — projection is linear so this
equals projecting the lerped 128-d states; do it in 3-D because it's 40×
cheaper). Precompute lerp weights `(K,)` once; the expansion is pure tensor
broadcasting on device, written in place into the slot.

Point order inside the slot buffer (documented so attr writes match):
`[probe stations (D,S,T)] [probe trails (D-1,K,S,T)] [gen stations (D,T)]
[gen trails ((D-1)*K,T)]`, all flattened row-major, positions `(N_max,3) f32`
contiguous, attr `(N_max,) f32`.

### 4.4 Color modes (toolbar combo; per-point f32 attr through the shared LUTs)

1. **loss** (default) — per-token cross-entropy of the FINAL depth's
   next-token prediction, broadcast along that token's entire thread. Hot
   thread = the model is confused about that token. MAGMA, vmin = −0.33·vmax
   (the flowscope baseline-floor trick so cold points stay visible),
   vmax = 6.0 (nats, slider "color").
2. **confidence** — logit-lens p(target) at EACH depth station (trail points
   lerp endpoint values): the 3-D logit lens — you see WHERE along the thread
   the model decides. VIRIDIS, vmin −0.33, vmax 1.0.
3. **depth** — station index / (D−1), trails lerped. MAGMA, fixed 0..1 window
   (plus baseline floor). Reads flow direction at a glance.

Loss/confidence come from one extra `lm_head(ln_f(resid_d))` pass over the
stations (the logit-lens math `publish_probe` already does — reuse its
approach, but batched and kept on device; only per-token scalars survive).

The generation thread's attr is pinned to vmax in every mode (white-hot).

## 5. Architecture

One new window in the EXISTING gpt_scope applet (not a new applet — it shares
the model, the training job, and the probe cadence).

### 5.1 Worker side (extends the existing training job)

- At job start, after the model exists: construct the pool exactly as
  gpt_scope already does for the heatmap path (same caps gates); inside
  `pool.use()` allocate the ThoughtSpace slots ONCE:
  `pos[3] (N_max,3) f32`, `attr[3] (N_max,) f32`. Velocities/temporaries
  don't exist here — this is a readout, not a sim.
- On the existing probe trigger (same cadence as `publish_probe`), and ONLY
  when the frame has set `thoughtspace_wanted` (atomic; the window is open
  and uncollapsed — "the interp tax is paid only when someone is looking",
  gpt_model.h's stated philosophy):
  1. `forward_resid(probe_batch)` → stations; project, normalize, expand
     trails; compute the active color mode's attr — all in-place into the
     write slot's tensors (no allocation in pool scope after init).
  2. If a sampled sequence exists (published by the existing Sample flow),
     run its stations/trails into the gen region and set the live point
     count to `N_max`; otherwise count = `N_probe`.
  3. `torch::cuda::synchronize()`, then flip write→ready under the mutex
     (triple-buffer invariant copied from flow_scope: never write the slot
     the frame is displaying).
- Probe batch: a fixed `(S,T)` int64 tensor cut deterministically from the
  val split at data-load time (seeded; same slice every run so training
  motion is comparable across runs).
- CPU-torch hosts: S drops to 16 and everything still runs (fallback draws).

### 5.2 Frame side (new window `"GPTScope: ThoughtSpace"`)

Copy flow_scope's frame structure wholesale:

- Toolbar child (bordered, fixed height): color-mode combo, "color" vmax
  slider, "raw norms" toggle, point-size slider (1–4 px), status text.
- View child fills `GetContentRegionAvail()`; offscreen view resized to it
  (≥3 px change threshold, clamp 64..4096) — flow_scope's exact logic.
- Per frame: snapshot ready slot + live count under mutex; `to_bridge` both
  tensors (cached import); one `geometry.draw_points(view, cam, pos, count,
  attr, colormap, vmin, vmax, size_px, black)`; `ImGui::Image`; orbit/zoom
  interaction identical to flow_scope (no impulse — left-drag also orbits).
- Honest status (flowscope wording discipline, green vs amber):
  - green: `"232,800 thought-points — zero-copy (imported geometry)"` only
    when THIS frame's draw returned true from the imported path;
  - amber fallback: ImPlot3D scatter of an 8k CPU subsample of the stations
    (worker publishes it alongside, flow_scope's sub_x/y/z pattern),
    labeled `"CPU fallback (subsampled)"` with the reason
    (`no geometry service` / `pool unavailable` / `torch CPU`).

### 5.3 Host-side change (exactly one line)

Add `"GPTScope: ThoughtSpace"` to `central_windows[]` in `src/main.cpp` so
the window docks into the central node (precedent: `"FlowScope: Field"`).
Nothing else in the host or SDK changes.

## 6. Memory & perf budget

- Pool: 3 slots × (232,800×3 + 232,800) × 4 B ≈ **11 MB** — trivial next to
  the model.
- `forward_resid` on (96,96), 4L/128d: a few ms on the target GPU, at probe
  cadence (~1 Hz), off the frame thread. Attr pass adds one (D,S,T,V) logit
  computation at stations only — keep it staged per depth to bound memory
  (V≈65, so even naive is ~12 MB transient; fine).
- Draw: 233k additive 1–3 px points ≈ a quarter of flowscope's verified 1M.
- Zero impact when the window is closed (`thoughtspace_wanted` gate) and
  zero impact on the training forward path (probe is eval+no_grad).

## 7. Failure ladder (all fail-closed, never a wrong image)

Identical to flow_scope: geometry caps absent → fallback scatter; pool ctor
fails → fallback; `to_bridge` miss or `draw_points` false on any frame →
that frame falls back and the label says so; released/torn-down mid-run →
the existing pool teardown pattern (tensors dropped under mutex first, pool
reset after worker join, deliberate leak + log line if the worker outlives
the cancel grace — copy flow_scope::cleanup verbatim).

## 8. Verification checklist (the implementing agent's definition of done)

- [ ] Full ctest 3/3 suites green, before and after (no host/renderer/test
      changes are expected beyond none — this is applet + 1 line of main.cpp;
      existing gfx geometry rows already cover the draw path).
- [ ] `gpt_model.h` diff is additive only; a stash A/B build shows the
      training-path code and warning count unchanged.
- [ ] Live (Vulkan+CUDA): train; open ThoughtSpace; status shows the green
      zero-copy line; the constellation visibly reorganizes within ~2 min of
      training (screenshot early + late; threads should straighten/cluster).
- [ ] Color modes switch live without re-import (same buffers, new attr on
      next probe; combo change may take one probe interval to apply — that's
      fine, document in a comment).
- [ ] Run Sample: a single bright thread appears and grows token-by-token.
- [ ] Cancel / relaunch Train in-process: clean, constellation resumes.
- [ ] Close the window: `thoughtspace_wanted` false ⇒ probe skips the
      ThoughtSpace work (verify via a counter/log, not vibes).
- [ ] `CALIPER_RENDERER=gl`: amber fallback scatter renders, labeled with the
      reason; clean exit.
- [ ] Mid-run applet unload (Home during training): no crash, no validation
      errors (the flow_scope teardown pattern).
- [ ] Honest-labeling audit: the words "zero-copy" appear ONLY on the
      imported-path frames (grep the new code for the string and check every
      site is gated on the draw result).

## 9. Risks & mitigations

- **Residual norm growth dominating shape** → per-depth RMS normalization
  (default on, §4.2); the raw toggle keeps honesty available.
- **Basis luck** (random projection hides structure) → seeded, so it's at
  least reproducible; PCA refit is the documented next increment if the
  default view underwhelms.
- **Probe cost spikes at large S** → constants chosen small (96×96); the
  gate keeps it zero when unwatched.
- **One-buffer constraint** (draw clears per call) → gen thread lives in the
  same buffer (§4.3 layout); never attempt a second draw call per frame.
- **Trail aliasing at 1 px** → default point size 2 px; slider up to 4.

## 10. Future increments (explicitly out of scope, listed so nobody sneaks
them into v1)

Real polylines/meshes in geometry.v1 (additive revision); a no-clear/overlay
draw flag; point picking + token tooltips; PCA/learned-probe projection with
smooth basis interpolation; attention-edge rendering between threads;
per-head decomposition of the trail segments (attn write vs MLP write as
separate colored sub-segments — the data is already in `forward_full`).
