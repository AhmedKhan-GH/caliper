# caliper.metrics.v1

Service id `caliper.metrics.v1` — TensorBoard vocabulary with ImPlot immediacy (PLATFORM.md §7.6). This page embeds the header verbatim; the docs build fails if the file moves.

```c
--8<-- "sdk/include/caliper/services/metrics_v1.h"
```

## Semantics

The vocabulary is TensorBoard's, deliberately: **experiment / run / tag / step**.

- **`begin_run(experiment, run_name)`** opens a run and returns its id. `0` means
  error — check for it and skip streaming (a `0` id is never a valid run). Every
  scalar, histogram, image, and hyperparameter you log is scoped to that id.
- **`end_run(run)`** closes it. Call it on *every* exit path of your training
  job — completion **and** cancellation — so a run never dangles as "still
  running". Partial curves logged before the end are preserved.
- **`scalar(run, tag, step, value)`** appends one point. The **tag** is the
  series name (`"train/loss"`, `"test/accuracy"`); the `/` groups tags into
  panes in the dashboard. The **step** is the x-axis: a global batch index for
  per-batch loss, an epoch index for per-epoch accuracy — you choose the axis by
  choosing the step. The store keeps points ordered by step and queries them back
  ordered (the §16 contract: 10k scalars written and read back in order).
- **`hparams_json(run, json_utf8)`** attaches a flat JSON blob of hyperparameters
  to the run (`{"lr":0.001,"batch":256,...}`) so runs are comparable.
- **`histogram` / `image`** log a distribution or a picture at a step.

### Thread-callability

Every entry point is callable **from an applet job thread** — which is where
training lives (never the frame thread). The host serializes writes internally
(a mutex over one DuckDB connection in v1), so concurrent jobs are safe. The host
also destroys the metrics store *after* it joins job threads, so a scalar logged
in the last instant before a cancel lands cannot fault.

### v1 image limitation

`image()` accepts **CPU-resident, contiguous, HWC `u8`** tensors only. The host
gate enforces this: a tensor that is non-contiguous, not on the CPU, or not
`u8`-HWC is logged and **dropped**, never misinterpreted. GPU-resident image
paths (no CPU staging) arrive with `caliper.tensor_bridge.v1` in Phase 2C.

### The payoff

Metrics is an **optional** service: probe it, and stream only when present. Every
applet that logs a scalar this way inherits the **Runs dashboard** for free —
run list, per-tag plots, EMA smoothing — with no dashboard code of its own. See
[MLScope](../../tutorials/first-applet.md) for the exemplar: MNIST training that
streams `train/loss` per batch and `test/accuracy` per epoch to this service.
