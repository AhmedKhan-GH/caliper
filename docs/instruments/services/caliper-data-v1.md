# caliper.data.v1

The service runs SQL over the host's embedded analytical store. Datasets are named, shared resources, and results cross the ABI as **Arrow C streams**. Service id: `caliper.data.v1`.

## Calls

<!-- c-api calls sdk/include/caliper/services/data_v1.h -->

## Ids

<!-- c-api ids sdk/include/caliper/services/data_v1.h -->

## Semantics

The query engine is the host's embedded DuckDB. Results cross the ABI as [**Arrow C streams**](../tensor-and-arrow-headers.md#arrow-header), so no DuckDB or Arrow C++ type appears in the frozen header. The service has four entry points, and a **job thread** can call all of them. A statement that returns no rows, such as DDL or `INSERT`, still succeeds with an empty stream. The host wraps a registered `uri` in a view, and it splices the dataset `name` into SQL. A name that is not a SQL identifier (`[A-Za-z_][A-Za-z0-9_]*`) is thus refused, which also guards against injection.

### Stream Ownership Protocol

Read this section before you drain a stream.

On success the host hands back a **live** `ArrowArrayStream`. The **caller drains it** with `get_schema`, then `get_next` until the end. The caller **MUST call `out->release(out)` exactly once** when done. Release sets the callback to NULL, and a stream whose `release` is NULL is already empty.

`get_next` signals the **end of stream** with an array whose own `release` is NULL (an empty array). It does not signal the end with an error, so stop there.

`query` and `open_dataset` **fully materialize** the result before they return, and the stream owns those rows. Thus draining a stream **never touches the DuckDB connection**. One job thread can drain a result while another thread issues a new `query`, with no lock contention.

### Threading and Errors

`query`, `register_dataset` and `open_dataset` serialize on one internal mutex, the same model as [artifacts](caliper-artifacts-v1.md#threading-and-string-lifetime) and metrics. `last_error()` is **thread-local**, so each thread sees only the error of its own last failing call. If the store fails to open at start-up, the host still vends the service, but every call returns `false` with a `last_error()`.

### Persistence and Rebuilds

The store is the file `data.duckdb` in the app-data folder. DuckDB writes every ordinary table in it to disk through its write-ahead log. This suits a dataset that you register once. It does not suit a table that an instrument rebuilds several times a second to show live state, because each `CREATE OR REPLACE TABLE` appends to the log and grows the catalog. As a result, the cost of a rebuild grows over a session.

Build live-visualization tables as `TEMP` tables instead:

```sql
CREATE OR REPLACE TEMP TABLE embed_points(label INT, pred INT, x REAL, y REAL, z REAL)
```

A `TEMP` table lives in memory and disappears when the host exits, so nothing reaches the log. The service keeps one connection, so every later call from any thread sees a `TEMP` table that one call made. Use an ordinary table only for data that must outlive the session.

Run the rebuild on a thread that the instrument owns, never on the frame thread. A query is file and catalog work with a cost that differs by platform, and a slow query on the frame thread stalls every window. See [The frame loop](../../introduction/architecture.md#frame-loop).

### Resource Limits

The host runs three DuckDB stores: `data.duckdb` for this service, `metrics.duckdb` for [`caliper.metrics.v1`](caliper-metrics-v1.md), and the [`caliper.artifacts.v1`](caliper-artifacts-v1.md) index. Each store opens with its own thread count and memory limit. DuckDB's defaults would give every instance all hardware threads and 80% of RAM. The metrics and artifacts pages link here for these limits.

**Defaults.** The host computes them from the physical RAM and core count when a store opens:

| Store | Memory limit | Threads |
|---|---|---|
| data | 25% of RAM, between 2 GB and 32 GB | cores ÷ 2, between 2 and 16 |
| metrics | 2% of RAM, between 512 MB and 4 GB | cores ÷ 4, between 1 and 4 |
| artifacts | 1% of RAM, between 256 MB and 2 GB | cores ÷ 16, between 1 and 2 |

MB and GB are DuckDB's decimal units (10⁶ and 10⁹ bytes), and the core division rounds down. If the host cannot read the RAM or the core count, the lower bound applies. On a 64 GiB, 18-core Mac, the data store gets 17179MB and 9 threads, metrics gets 1374MB and 4 threads, and artifacts gets 687MB and 1 thread. On Linux the host reads the machine's RAM and cores, not the limits of a container's cgroup. In a constrained container, set the overrides below.

The memory limit caps DuckDB's buffer manager and reserves nothing up front. A query that needs more memory than the limit spills to DuckDB's temporary directory beside the database file.

**Overrides.** The host reads these environment variables when each store opens:

| Variable | Applies to | Value |
|---|---|---|
| `CALIPER_DUCKDB_THREADS` | all three stores | an integer, 1 to 1024 |
| `CALIPER_DUCKDB_MEMORY_LIMIT` | all three stores | a DuckDB size string, e.g. `4GB` |
| `CALIPER_DUCKDB_<STORE>_THREADS` | one store: `<STORE>` is `DATA`, `METRICS` or `ARTIFACTS` | an integer, 1 to 1024 |
| `CALIPER_DUCKDB_<STORE>_MEMORY_LIMIT` | one store | a DuckDB size string |

For each setting, the per-store variable wins, then the all-stores variable, then the computed default. An empty variable counts as unset. An invalid value falls back to the computed default, not to the other variable. A thread count outside 1 to 1024 is invalid, and so is a size that DuckDB rejects. The host prints one line on stderr and the store still opens:

```
[duckdb] DATA: ignoring threads override "lots" (want an integer 1..1024); using 9
```

This example caps metrics and artifacts at 1 GB and gives the data store 8 GB:

```bash
CALIPER_DUCKDB_MEMORY_LIMIT=1GB CALIPER_DUCKDB_DATA_MEMORY_LIMIT=8GB ./build/caliper
```

### C++ Sugar

The [`caliper::Data`](../cpp-sugar.md) wrapper is falsy-inert when the host does not vend the service. It exposes the raw stream API and one helper, `drain_numeric`. This helper drains an all-numeric result into column-major `double` values **and releases the stream**, so simple consumers never touch Arrow buffers:

```cpp
caliper::Data data(host);                       // falsy if the host does not vend it
// DDL and INSERT still hand back an empty stream, so release it:
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
