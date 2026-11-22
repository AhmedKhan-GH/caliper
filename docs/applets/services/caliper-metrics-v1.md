# caliper.metrics.v1 and v1_1

Service id `caliper.metrics.v1` is a write-only metrics log in TensorBoard's vocabulary. `caliper.metrics.v1_1` adds one read entry, `query()`, and is described [below](#calipermetricsv1_1). This page embeds both headers verbatim; the docs build fails if a file moves.

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
  per-batch loss, and the *same* global step for accuracy sampled every N batches
  (per-epoch cadence hides the learning transient on fast-converging datasets) —
  you choose the axis by choosing the step. Sharing the step domain lets loss and
  accuracy line up. The store keeps points ordered by step and queries them back
  ordered (10k scalars written are read back in order).
- **`hparams_json(run, json_utf8)`** attaches a flat JSON blob of hyperparameters
  to the run (`{"lr":0.001,"batch":256,...}`) so runs are comparable.
- **`histogram` / `image`** log a distribution or a picture at a step.

### Thread Callability

Every entry point is callable **from an applet job thread** — which is where
training lives (never the frame thread). The host serializes writes internally
(a mutex over one DuckDB connection in v1), so concurrent jobs are safe. The host
also destroys the metrics store *after* it joins job threads, so a scalar logged
in the last instant before a cancel lands cannot fault — for a job that honours
cancel; the join gives up after 5 s (see [`caliper.jobs.v1`](caliper-jobs-v1.md)).

### Resource Limits

The metrics store (`metrics.duckdb`) opens with its own DuckDB thread count and
memory limit, sized from the machine's RAM and cores. Override them with
`CALIPER_DUCKDB_METRICS_THREADS` and `CALIPER_DUCKDB_METRICS_MEMORY_LIMIT`, or
the all-stores `CALIPER_DUCKDB_THREADS` and `CALIPER_DUCKDB_MEMORY_LIMIT`. The
defaults and the precedence are in
[`caliper.data.v1`: Resource limits](caliper-data-v1.md#resource-limits).

### Image Limitation

`image()` accepts **CPU-resident, contiguous, HWC `u8`** tensors only. The host
gate enforces this: a tensor that is non-contiguous, not on the CPU, or not
`u8`-HWC is logged and **dropped**, never misinterpreted. There is no GPU-resident
path for metrics images.

### Persisted Runs

Metrics is an **optional** service: probe it, and stream only when present. Every
applet that logs a scalar this way has its runs persisted in the host's metrics
store, with no storage code of its own. The host has no run browser, so stored
runs are read back through `caliper.metrics.v1_1.query` or from SQL tools.

## `caliper.metrics.v1_1`

`caliper.metrics.v1` is write-only across the ABI and immutable, so a consumer
such as an embedding host ([Embedding Caliper](../../web/embedding-api.md#reaching-services-from-a-host))
cannot list runs or read scalars through it. `caliper.metrics.v1_1` is the
additive read revision.
Its table starts with the byte-identical v1 writer prefix, and appends two
entries: `query` and `last_error`.

```c
--8<-- "sdk/include/caliper/services/metrics_v1_1.h"
```

### `query`

`query(sql_utf8, out)` runs one SQL statement against the metrics store and
streams the result as Arrow ([Arrow interchange](../tensor-and-arrow-headers.md#arrow-header)).

- **One SELECT only.** The host parses the text without executing it and refuses
  anything that is not exactly one `SELECT` statement. That blocks statement
  chaining (`SELECT 1; DROP TABLE runs`) and every `INSERT`, `UPDATE`, `DELETE`
  and DDL statement. A read-only introspection `PRAGMA` that DuckDB rewrites into
  a `SELECT` is accepted; setter pragmas parse as non-`SELECT` and are refused.
  This differs from `caliper.data.v1`'s `query`, which runs any SQL against a
  separate store, because this one runs on the live writer connection.
- **Stream ownership.** On success the host fills `*out` with a live
  `ArrowArrayStream`. The caller drains it (`get_schema`, then `get_next` until a
  released empty array) and must call `out->release(out)` exactly once. This is
  the same rule as `caliper.data.v1`.
- **Failure.** `query` returns `false`, leaves `*out` untouched and sets the
  error.
- **`last_error()`** returns the reason the last `query` on the calling thread
  failed. The string is host-owned, never `NULL`, thread-local, and valid until
  the next `metrics.v1_1` call on the same thread.
- **Threads.** Applet job threads and a host UI thread that is not the frame
  thread may all call it. The store serializes every writer and reader through
  one DuckDB connection under one mutex.

### Read Access via `query`

A consumer could in principle open the metrics database file read-only with
DuckDB `ATTACH`. That second instance sees only checkpointed state, so rows an
applet logged a moment ago would be missing. `query` runs on the writer's own
connection and sees fresh rows immediately, and it keeps DuckDB types on the
host side of the ABI.

The header is immutable once published. A future read entry would arrive as
`caliper.metrics.v1_2`.
