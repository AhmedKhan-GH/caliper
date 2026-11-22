# caliper.artifacts.v1

Service id `caliper.artifacts.v1` is a content-addressed artifact store. It keeps deduplicated, lineage-tracked checkpoints and exports.

## Calls

<!-- c-api calls sdk/include/caliper/services/artifacts_v1.h -->

## Ids

<!-- c-api ids sdk/include/caliper/services/artifacts_v1.h -->

## Semantics

The store keeps checkpoints and exports under a key that comes from their content. It deduplicates them on disk and links each one to the run that produced it. An applet calls all three entry points from a **job thread**. Hashing uses **sha256**, and a `put` writes the digest into `out_digest[65]` as **64 hex chars + NUL**.

The digest is a pure function of the bytes, so identical bytes always land at the same path.

**Deduplication.** If an applet stores the same bytes twice, the second `put` finds the file on disk and **skips the write**. The store keeps one file, not two. The index still records the newer `(name, run, ts)` against the same digest.

**Name Resolution.** A `name` is a mutable label, not a key. If an applet reuses a name for new bytes, the store adds a new row with a fresh `ts`. `path_of` and `exists` accept **either** a digest **or** a name. A name resolves to the **newest** row with that name, and a digest resolves to exactly that blob.

**Run Lineage.** The `run` argument links a blob to the `caliper.metrics.v1` run that produced it. The value `run = 0` means the blob is unlinked, for example a standalone export. The host can list the artifacts of a run with its `by_run` query, so a checkpoint always knows which run made it.

**Unknown Values.** An unknown digest or name returns `false` or `nullptr`. No exception crosses the C boundary. If the store fails to open at host start-up, the host still vends the service, but every call does nothing: `put` returns `false`, `path_of` returns `nullptr` and `exists` returns `false`. The applet degrades and does not crash.

### Threading and String Lifetime

The host serializes every method internally, in the same way as [MetricsStore](caliper-metrics-v1.md#thread-callability). Job threads can thus call `put` concurrently. The host destroys the store after it joins its job threads, so a `put` just before a cancel cannot fault. This holds for a job that honors cancel, and the join gives up after 5 s (see [`caliper.jobs.v1`](caliper-jobs-v1.md)).

The artifacts index opens with its own DuckDB thread count and memory limit. Set them with `CALIPER_DUCKDB_ARTIFACTS_THREADS` and `CALIPER_DUCKDB_ARTIFACTS_MEMORY_LIMIT`, or set both for all stores with `CALIPER_DUCKDB_THREADS` and `CALIPER_DUCKDB_MEMORY_LIMIT`. The defaults and the precedence are in [`caliper.data.v1`: Resource limits](caliper-data-v1.md#resource-limits).

!!! warning "`path_of` returns a host-owned string, valid only until your next call"
    The `const char*` from `path_of` points at host-owned storage. It stays valid until the next `artifacts.v1` call. Copy it into a `std::string` at once if you call the service again before you use it.
    The host keeps this storage per thread, so two threads that call `path_of` get independent buffers. A second call on the same thread still overwrites the first. To avoid the problem, resolve `path_of` on the frame thread and give the copied path to the worker.

### C++ Sugar

The [`caliper::Artifacts`](../cpp-sugar.md) wrapper does nothing and tests false when the service is absent. It also hides the `out_digest` buffer.

```cpp
caliper::Artifacts art(host);           // falsy if the host doesn't vend it
std::string digest = art.put("embedscope-model", bytes.data(), bytes.size(), run);
if (art.exists("embedscope-model"))     // digest OR name
    const char* path = art.path_of("embedscope-model");   // copy before next call
```
