# caliper.data.v1

SQL over the host's embedded analytical store — datasets as named, shared,
queryable resources, with results crossing the ABI as **Arrow C streams**.
Service id: `caliper.data.v1`.

```c
--8<-- "sdk/include/caliper/services/data_v1.h"
```

## Semantics

Datasets become **named, shared, queryable resources** instead of per-applet
private downloads. The query engine is the host's embedded DuckDB; results cross
the ABI as [**Arrow C streams**](../arrow.md) — no DuckDB or Arrow C++ type ever
appears in the frozen header. Four entry points, all callable from a **job
thread**.

- **`query(sql_utf8, out)`** runs a SQL statement against the host store. On
  success it fills `*out` with a live `ArrowArrayStream` and returns `true`; on
  failure it returns `false`, leaves `*out` **untouched**, and sets
  `last_error()`. DDL/INSERT statements that produce no rows still succeed with an
  empty stream.
- **`register_dataset(name, uri)`** names a dataset. The **`uri`** may be a
  `.parquet` path, a `.csv` path, or the name of an existing table; the host
  wraps it in a view. Re-registering a name **replaces** it. The **`name` must be
  a SQL identifier** — `[A-Za-z_][A-Za-z0-9_]*` — because it is spliced into SQL
  as an identifier; anything else is rejected (`false` + `last_error`), which is
  also the injection guard.
- **`open_dataset(name, out)`** streams a registered dataset back as a
  full-table `ArrowArrayStream` — the `SELECT *` convenience over `query`.
- **`last_error()`** returns why **this thread's** last call returned `false`.
  It is never NULL (`""` when nothing has failed yet).

### Stream ownership protocol

This is the sharp edge — read it before you drain a stream.

- On success the host hands back a **live** `ArrowArrayStream`. The **caller
  drains it** (`get_schema`, then `get_next` until end) and **MUST call
  `out->release(out)` exactly once** when done. Releasing sets the callback to
  NULL; a stream whose `release` is NULL is already empty.
- **End of stream** is signalled by `get_next` yielding an array whose own
  `release` is NULL (an empty array) — not by an error. Stop there.
- **Materialized-stream independence.** `query`/`open_dataset` **fully
  materialize** the result before returning; the stream's producer state owns
  those rows. Draining the stream therefore **never touches the DuckDB
  connection** — one job thread can drain a result while another thread issues a
  new `query`, with no lock contention between them.

### Threading and errors

`query`/`register_dataset`/`open_dataset` serialize on one internal mutex (the
[artifacts](artifacts-v1.md#threading-and-string-lifetime)/metrics model).
`last_error()` is **thread-local**: each thread sees only the error of *its* last
failing call, so one thread's failure never clobbers another's diagnostic. If the
store failed to open at start-up the service is still vended but inert (every call
`false` + a `last_error`).

### C++ sugar

The [`caliper::Data`](../sugar.md) wrapper is falsy-inert when absent and exposes
the raw stream API plus one helper — `drain_numeric` — that drains an
all-numeric-columns result into column-major `double`s **and releases the
stream**, so simple consumers never touch Arrow buffers:

```cpp
caliper::Data data(host);                       // falsy if the host doesn't vend it
// DDL/INSERT still hand back a (empty) stream — release it (EmbedScope's
// data_exec helper wraps exactly this):
ArrowArrayStream ddl{};
if (data.query("CREATE OR REPLACE TABLE embed_points(label INT, x REAL, y REAL, z REAL)", &ddl)
    && ddl.release)
    ddl.release(&ddl);
ArrowArrayStream s{};
std::vector<std::string> names;
std::vector<std::vector<double>> cols;
if (data.query("SELECT label, AVG(x), AVG(y), AVG(z) FROM embed_points GROUP BY label", &s))
    caliper::Data::drain_numeric(&s, &names, &cols);   // releases s for you
```

### The reference consumer

[EmbedScope](../../tutorials/first-applet.md) registers its published test-set
embeddings as a table each eval tick, then runs live SQL over that genuinely
tabular data: **per-class centroids** (`AVG(x), AVG(y), AVG(z) GROUP BY label`,
drawn as 3-D diamonds) and a **misclassified count** (`SUM(CASE ...)`), both
drained through `drain_numeric`. It is demonstrative but honest — real learned
data, real aggregation — and degrades gracefully: absent, the panels say so and
the applet still runs.
