# caliper.artifacts.v1

Service id `caliper.artifacts.v1` — content-addressed artifact store: deduplicated, lineage-tracked checkpoints and exports (PLATFORM.md §7.8). This page embeds the header verbatim.

```c
--8<-- "sdk/include/caliper/services/artifacts_v1.h"
```

## Semantics

The idea is **MLflow's artifact store without the server**: checkpoints and
exports keyed by their content, deduplicated on disk, and lineage-linked to the
run that produced them. Three entry points, all callable from an applet **job
thread**.

- **`put(name, bytes, len, run, out_digest)`** hashes the bytes with **sha256**,
  writes the blob to a file named by that hash, and upserts an index row
  `(digest, name, run, len, ts)`. It writes the **64 hex chars + NUL** of the
  digest into `out_digest[65]` and returns `true`. The blob is
  **content-addressed**: the digest is a pure function of the bytes, so identical
  bytes always land at the same path.
- **Dedup.** Storing the same bytes twice computes the same digest, sees the file
  already on disk, and **skips the second write** — one file, not two. The index
  row is still upserted, so the newer `(name, run, ts)` is recorded against the
  same digest.
- **Name → newest resolution.** A `name` is a mutable label, not a key: reusing a
  name for new bytes adds a new row with a fresh `ts`. `path_of`/`exists` accept
  **either** a digest **or** a name; a name resolves to the **newest** matching
  row (`ORDER BY ts DESC LIMIT 1`). A digest resolves to exactly that blob.
- **Run lineage.** The `run` argument links a blob to the `caliper.metrics.v1`
  run that produced it (**`run = 0` means unlinked** — a standalone export with no
  training provenance). The host can list a run's artifacts back (the
  `by_run` query the §16 contract exercises), so a checkpoint always knows which
  run made it.
- **Unknown is inert, never fatal.** An unknown digest or name returns
  `false`/`nullptr` — never an exception across the C boundary. If the store
  failed to open at host start-up, the service is still vended but every thunk
  no-ops (`put` → `false`, `path_of` → `nullptr`, `exists` → `false`); the applet
  degrades, it does not crash.

### Threading and string lifetime

Every method is host-**serialized** internally (one mutex over one DuckDB
connection, the same sanctioned model as [MetricsStore](metrics-v1.md#thread-callability)),
so job threads calling `put` concurrently are safe. The store is destroyed
*after* the host joins its job threads, so a `put` in the last instant before a
cancel lands cannot fault.

!!! warning "`path_of` returns a host-owned string, valid only until your next call"
    The `const char*` from `path_of` points at **host-owned** backing storage,
    documented **valid until the next `artifacts.v1` call**. Copy it into a
    `std::string` immediately if you will call the service again before using it.
    The host backs this with **thread-local** storage, so two threads each calling
    `path_of` get independent buffers and cannot stomp each other's result — but
    a *second* call **on the same thread** still overwrites the first. EmbedScope
    sidesteps the race entirely by resolving `path_of` on the frame thread and
    handing the *copied* path to its worker.

### C++ sugar

The [`caliper::Artifacts`](../sugar.md) wrapper is falsy-inert when the service is
absent (every call no-ops) and hides the `out_digest` buffer:

```cpp
caliper::Artifacts art(host);           // falsy if the host doesn't vend it
std::string digest = art.put("embedscope-model", bytes.data(), bytes.size(), run);
if (art.exists("embedscope-model"))     // digest OR name
    const char* path = art.path_of("embedscope-model");   // copy before next call
```

### The reference consumer

[EmbedScope](../../tutorials/first-applet.md) uses artifacts.v1 as its
**load-bearing** demand: **Save** serializes the trained module to a byte buffer
(`torch::save`) and `put`s it under `"embedscope-model"` linked to the current
run; **Load** resolves the path via `path_of`, `torch::load`s the module, and runs
**one eval pass — skipping training entirely**. Quit, relaunch, Load: the 3-D
cloud is restored without ever re-training. That "no reload without it" is why the
service is load-bearing rather than merely demonstrative.
