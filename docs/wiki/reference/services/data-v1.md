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

### Resource limits

The host runs three DuckDB stores: `data.duckdb` for this service,
`metrics.duckdb` for [`caliper.metrics.v1`](metrics-v1.md) and the
[`caliper.artifacts.v1`](artifacts-v1.md) index. Each opens with its own thread
count and memory limit instead of DuckDB's defaults (every hardware thread and
80% of RAM, per instance). This section is the one place the limits are
described; the metrics and artifacts pages link here.

**Defaults.** Computed from the machine's physical RAM and core count when the
store opens:

<!-- DuckDB default limits: the rule in src/host/duckdb_sizing.h (patch 10;
     the lower bounds are patch 06's former fixed values). This table is the
     only copy of the numbers in the docs. -->

| Store | Memory limit | Threads |
|---|---|---|
| data | 25% of RAM, between 2 GB and 32 GB | cores ÷ 2, between 2 and 16 |
| metrics | 2% of RAM, between 512 MB and 4 GB | cores ÷ 4, between 1 and 4 |
| artifacts | 1% of RAM, between 256 MB and 2 GB | cores ÷ 16, between 1 and 2 |

MB and GB are DuckDB's decimal units (10⁶ and 10⁹ bytes), and the core division
rounds down. When the RAM or the core count cannot be read, the lower bound
applies. On a 64 GiB, 18-core Mac this gives data 17179MB and 9 threads,
metrics 1374MB and 4 threads, artifacts 687MB and 1 thread; an 8-core machine
gets 4, 2 and 1 threads. On Linux the host's RAM and cores are read, not a
container's cgroup limits, so set the overrides below in a constrained
container.

The memory limit caps DuckDB's buffer manager; it reserves nothing up front. A
query that needs more memory than its store's limit spills to DuckDB's temporary
directory beside the database file.

**Overrides.** Environment variables, read when each store opens:

| Variable | Applies to | Value |
|---|---|---|
| `CALIPER_DUCKDB_THREADS` | all three stores | an integer, 1 to 1024 |
| `CALIPER_DUCKDB_MEMORY_LIMIT` | all three stores | a DuckDB size string, e.g. `4GB` |
| `CALIPER_DUCKDB_<STORE>_THREADS` | one store: `<STORE>` is `DATA`, `METRICS` or `ARTIFACTS` | an integer, 1 to 1024 |
| `CALIPER_DUCKDB_<STORE>_MEMORY_LIMIT` | one store | a DuckDB size string |

For each setting the per-store variable wins, then the all-stores variable, then
the computed default; an empty variable counts as unset. An invalid value (a
thread count outside 1–1024, or a size DuckDB rejects) falls back to the
computed default, not to the other variable, with one line on stderr, and the
store still opens:

```
[duckdb] DATA: ignoring threads override "lots" (want an integer 1..1024); using 9
```

For example, this caps metrics and artifacts at 1 GB and gives the data store
8 GB:

```bash
CALIPER_DUCKDB_MEMORY_LIMIT=1GB CALIPER_DUCKDB_DATA_MEMORY_LIMIT=8GB ./build/caliper
```

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
