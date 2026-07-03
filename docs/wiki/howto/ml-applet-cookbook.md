# ML applet cookbook

The composition idioms for **featurized, live ML demonstrations** — the
patterns that make an applet feel alive rather than merely instrumented.
Every one of them is embodied in the exemplar (`applets/embed_scope/`); this
page names them, explains *why* they're shaped that way, and tells you which
part of the exemplar to copy. Contracts live in the per-service reference
pages; this is the page about putting them together.

## 1. The threading spine

One training **worker** (a `jobs.v1` job), one **frame thread** (your
`draw_ui`). Everything they share lives under **one mutex**, published with
**generation counters**:

```
worker:  compute -> lock -> swap results in, gen++ -> unlock
frame:   lock -> if (gen != last_seen) copy out -> unlock -> render copies
```

- The worker publishes **owned copies** (`std::vector`) for plot data, or
  **tensor handles** for device-resident display tensors (§3) — never raw
  pointers into transient worker state.
- The frame thread is the ONLY thread that touches `tensor_bridge.v1` and
  `data.v1`. All torch *operations* stay on the worker; the frame thread may
  hold and pass tensor handles but never launches kernels.
- Generation counters make every downstream stage cheap: nothing re-uploads,
  re-splits, or re-queries unless its `gen` moved.

*Copy from:* `EmbedScopeState` (the `mtx` block), `train_job`'s publish
blocks, the top of `draw_ui`.

## 2. Three cadences — match update rate to what the data is

A rich demo has streams with different natural rates. Name them explicitly:

| Stream | Cadence | Why |
|---|---|---|
| Live layer: current batch, weight tensors | **every optimizer step** | weights only change when the optimizer steps — per-step IS every change |
| Derived aggregates: SQL centroids, misclassified counts | **throttled (~2 Hz)** | rebuilt on the *frame thread*; at step rate it would stutter the UI |
| Quality metrics: test accuracy, val loss → `metrics.v1` | **every N steps** (`kEvalEvery`) | metrics want statistical stability, not liveness; they also persist to the Runs dashboard |

Don't add a config knob for cadence unless a real cost forces one — measure
first. (The exemplar once had a "cloud refresh" slider defending against a
6× slowdown that turned out to be ~nothing: step time was dominated by GPU
sync overhead, not compute. The knob died.)

*Copy from:* the step loop in `train_job` (per-step publishes + `kEvalEvery`
gate) and the `last_sql` throttle in `draw_ui`.

## 3. The device-resident pull (the USP pattern)

The platform's reason to exist: tensors go from training memory to pixels
**without CPU staging**. The recipe:

1. **Worker, per step:** build *display tensors* on the training device
   (slice/reshape/upscale — all GPU ops), `.contiguous()`, then under the
   mutex swap the **handles** into shared state and bump `live_gen`.
   Handle swaps are refcount bumps — no data moves.
2. **Frame thread, at most once per frame:** if `live_gen` advanced, take
   co-owning handle copies under the mutex, convert via
   `caliper::adapters::to_tensor`, and hand them to the bridge
   (`texture_from_tensor_mapped` for f32 heatmaps). On Metal the bridge
   colormaps **on-GPU** (`tex_update_from_device`); the weights never visit
   the CPU.
3. **Decoupling is the point:** if steps outpace frames you render 60 fps of
   *latest* states; if frames outpace steps, identical memory renders
   identical pixels. Display never throttles training, training never blocks
   display.

Rules that bite: MPS tensors must be `contiguous()` with
`storage_offset() == 0` (fresh results of GPU ops are; views may not be —
see [adapters](../reference/adapters.md)). And degrade gracefully: if the
bridge rejects device tensors (GL fallback), flip an atomic so the worker
hands over CPU tensors from the next step (`disp_force_cpu` in the exemplar).

*Copy from:* the `disp_conv`/`disp_embw` blocks in `EmbedScopeState`,
`train_job`'s live-publish block, the Tensors-window rebuild in `draw_ui`.

## 4. Sharp tiny tensors: block-upscale on device

Bridge textures sample linearly (v1 has no filter flag), so a 3×3 kernel
drawn at 40 px is interpolated mush. Fix it where the data lives — on the
GPU: `repeat_interleave(k, 0).repeat_interleave(k, 1)` turns 3×3 into
48×48 **hard blocks** before upload. Draw at 1:1 or integer multiples.
Sharp under any sampler, zero ABI involvement, stays device-resident.

## 5. Texture lifecycle

- Bridge textures are **frame-thread-owned**. Create/update/release only in
  `draw_ui` and `cleanup()`.
- Rebuild gated on a gen counter (`tex_gen`), release-then-create per
  rebuild for mapped textures whose value range evolves (the range is baked
  at creation).
- `cleanup()` releases every texture **after** the worker join, **before**
  returning (renderer teardown follows). Miss one and the leak is silent;
  release early and you race the worker.

## 6. Viewport policy: who owns the camera

Three plot situations, three correct answers — and every "why is my plot
fighting me" bug is one of these mismatched:

| Data behavior | Correct axes | Exemplar |
|---|---|---|
| Points **move** through space (embeddings, particles) | **Fixed**: fit once, hold still — motion is only visible against a static frame. Offer `auto-fit` toggle + one-shot `Refit`. | the 3-D Cloud |
| Series **grows** (loss, accuracy) | **Following**: `ImPlotAxisFlags_AutoFit` traces the whole curve. But AutoFit **input-locks** the axis — so offer a `follow` toggle (the log-console auto-scroll idiom); unchecked = free zoom/pan. | the Training curves |
| Static content (a snapshot, a matrix) | Default fit; double-click re-fits (native ImPlot/ImPlot3D gesture — leave it enabled). | the Tensors heatmaps |

One visible toggle per plot, sensible motion by default, double-click resets.

## 7. The cancel contract, in practice

`jobs.v1` promises cancel is honored **≤ 100 ms**. In a training loop that
means: check `ctl->cancelled(ctl)` **every batch**, inside **every eval
sub-loop**, and per-transfer via curl's xferinfo callback during downloads.
Your `cleanup()` then does: request cancel → bounded wait (poll
`is_running`) → release textures → return. The host additionally joins all
workers before teardown, so sloppiness here degrades instead of crashing —
but honor the contract anyway; the bounded wait is what keeps app exit fast.

## 8. Data acquisition (the download recipe)

Datasets are fetched **inside the job** (never the frame thread), with:
atomic `.tmp` + rename writes (a crash mid-download can't poison the
cache), corrupt-cache self-heal (parse failure deletes and re-downloads),
cancellable transfers, and `curl_global_init`/`cleanup` paired in applet
init/cleanup. Prefer mirrors that tolerate automation (the exemplar uses the
S3 MNIST mirror; the classic host 403s).

*Copy from:* `ensure_dataset` + `mnist_path` in the exemplar.

## 9. Checkpoints via `artifacts.v1`

Save on the frame thread: serialize to a byte buffer (`torch::save` into a
`std::ostringstream`), `Artifacts::put(name, bytes, run_id)` — passing the
`metrics.v1` run id buys you lineage for free. Load: resolve `path_of` **on
the frame thread** (host strings are valid-until-next-call), hand the path
to a job that `torch::load`s and runs one eval pass — restoring the visuals
**without training** is what makes Save/Load feel magical in a demo.

## 10. SQL over live data (`data.v1`)

Rebuild your table from the latest published snapshot (CREATE OR REPLACE +
batched INSERT through `query()` — release the empty DDL streams), then ask
real questions (GROUP BY centroids, misclassification counts) and drain with
`Data::drain_numeric`. Frame-thread only, throttled (§2). If the service is
absent, say so in the panel and keep running — every optional service
degrades to a visible "absent (ok)" line, never a crash.

---

**The one-sentence summary:** worker computes and publishes generations;
frame pulls the latest and renders; every stream updates at the rate its
nature dictates; the camera has an owner; and everything optional degrades
visibly instead of failing silently. That's a Caliper ML demo.
