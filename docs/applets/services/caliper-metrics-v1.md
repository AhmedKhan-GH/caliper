# caliper.metrics.v1 and v1_1

Service id `caliper.metrics.v1` is a write-only metrics log that uses the TensorBoard vocabulary. `caliper.metrics.v1_1` adds one read entry, `query()`, and is described [below](#calipermetricsv1_1). Each header below is embedded verbatim.

```c
--8<-- "sdk/include/caliper/services/metrics_v1.h"
```

## Semantics

The vocabulary is **experiment / run / tag / step**, as in TensorBoard.

- **`begin_run(experiment, run_name)`** opens a run and returns its id. A `0` id means error and is never a valid run, so check for it and skip streaming. Every scalar, histogram, image and hyperparameter you log is scoped to that id.
- **`end_run(run)`** closes the run. Call it on every exit path of your training job, both completion and cancellation, so a run never stays "running". Partial curves logged before the end are kept.
- **`scalar(run, tag, step, value)`** appends one point. The **tag** is the series name, for example `"train/loss"` or `"test/accuracy"`, and the `/` groups tags into panes in the dashboard. The **step** is the x-axis, and you choose the axis when you choose the step.
- **`hparams_json(run, json_utf8)`** attaches a flat JSON object of hyperparameters to the run, for example `{"lr":0.001,"batch":256}`, so runs are comparable.
- **`histogram` / `image`** log a distribution or a picture at a step.

Use a global batch index as the step for per-batch loss. Use the same global step for accuracy that you sample every N batches, so loss and accuracy line up. A per-epoch step hides the learning transient on fast-converging datasets. The store keeps points ordered by step and returns them in that order.

### Thread Callability

Every entry point is callable from an applet job thread, which is where training runs. Do not call it from the frame thread. The host serializes writes, so concurrent jobs are safe. The host destroys the metrics store after it joins job threads. Thus a scalar logged just before a cancel lands cannot fault, if the job honors cancel. The join gives up after 5 s (see [`caliper.jobs.v1`](caliper-jobs-v1.md)).

### Resource Limits

The metrics store (`metrics.duckdb`) opens with its own DuckDB thread count and memory limit. The host sizes both from the RAM and cores of the machine. Override them with `CALIPER_DUCKDB_METRICS_THREADS` and `CALIPER_DUCKDB_METRICS_MEMORY_LIMIT`, or with the all-stores `CALIPER_DUCKDB_THREADS` and `CALIPER_DUCKDB_MEMORY_LIMIT`. The defaults and the precedence are in [`caliper.data.v1`: Resource limits](caliper-data-v1.md#resource-limits).

### Image Limitation

`image()` accepts only CPU-resident, contiguous, HWC `u8` tensors. The host drops and logs any tensor that is non-contiguous, not on the CPU, or not `u8`-HWC, so it never misreads the data. Metrics images have no GPU-resident path.

### Persisted Runs

Metrics is an optional service, so probe for it and stream only when it is present. The host persists the runs of every applet that logs a scalar, and the applet needs no storage code. The host has no run browser. Read stored runs back with `caliper.metrics.v1_1.query` or with SQL tools.

## `caliper.metrics.v1_1`

`caliper.metrics.v1` is write-only across the ABI and immutable. A consumer such as an embedding host ([Embedding Caliper](../../web/embedding-api.md#host-service-access)) cannot list runs or read scalars through it. `caliper.metrics.v1_1` is the additive read revision. Its table starts with the byte-identical v1 writer prefix and appends two entries: `query` and `last_error`.

```c
--8<-- "sdk/include/caliper/services/metrics_v1_1.h"
```

### `query`

`query(sql_utf8, out)` runs one SQL statement against the metrics store and streams the result as Arrow ([Arrow interchange](../tensor-and-arrow-headers.md#arrow-header)).

- **One SELECT only.** The host parses the text without executing it and refuses anything that is not exactly one `SELECT` statement. This blocks statement chaining (`SELECT 1; DROP TABLE runs`) and every `INSERT`, `UPDATE`, `DELETE` and DDL statement. The host accepts a read-only introspection `PRAGMA` that DuckDB rewrites into a `SELECT`. It refuses setter pragmas. This differs from `query` in `caliper.data.v1`, which runs any SQL against a separate store, because this `query` runs on the live writer connection.
- **Stream ownership.** On success the host fills `*out` with a live `ArrowArrayStream`. The caller drains it (`get_schema`, then `get_next` until a released empty array) and calls `out->release(out)` exactly once. This is the same rule as `caliper.data.v1`.
- **Failure.** `query` returns `false`, leaves `*out` untouched and sets the error.
- **`last_error()`** returns the reason the last `query` on the calling thread failed. The string is host-owned, never `NULL` and thread-local. It stays valid until the next `metrics.v1_1` call on the same thread.
- **Threads.** Applet job threads and a host UI thread that is not the frame thread may all call `query`. The host serializes every writer and reader.

### Read Access via `query`

A consumer could open the metrics database file read-only with DuckDB `ATTACH`. That second instance sees only checkpointed state, so it misses rows that an applet logged a moment ago. `query` runs on the writer connection and sees fresh rows immediately. It also keeps DuckDB types on the host side of the ABI.

The header is immutable once published. A future read entry would arrive as `caliper.metrics.v1_2`.
