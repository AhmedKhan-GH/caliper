# Catalog

The catalog lists every source that a spec can bind, with its schema, its shape and its rate, and gives a small sample of each. It is the first thing an agent reads when it builds a visual: `catalog.list` says what exists, `catalog.describe` says what one source is, and `catalog.sample` shows a little of its data. This page elaborates section 3 of the sprint 3 contracts. The commands themselves are part of the [Command layer](command-layer.md).

## What is built and what is planned

Built on this branch, in the library `caliper_builder_catalog` (`src/builder/catalog/`, headers in `src/builder/include/caliper_builder/catalog/`):

- `CatalogEntry`, its strict JSON form and the parser of source ids;
- the `CatalogProvider` interface, with `list`, `describe`, `sample` and `revision`;
- `Catalog`, which merges providers, resolves duplicate ids and reports them;
- `InMemoryCatalogProvider`, a provider for tests that holds entries, table rows and tensor bytes;
- `sample_tensor()`, which computes the statistics and the slice of a tensor from its dtype, shape and little-endian bytes, for any provider to use.

Planned:

- The provider for loader outputs (tables and tensors) belongs to the loaders library.
- The providers for live views, feed channels and metrics come when the libraries are wired into the app, because they read the app's live session and services.
- The handlers of `catalog.list`, `catalog.describe` and `catalog.sample` are bound by a later stream; each calls the matching function of `Catalog`.

## The entry

An entry describes one source. Its JSON form always has all fifteen keys, with `null` where a value does not apply or is not known:

```json
{
  "id": "tensor:attn_weights",
  "kind": "tensor",
  "name": "attn_weights",
  "title": "attn_weights",
  "origin": {"loader": "run42-weights", "file": "model.safetensors"},
  "columns": null,
  "rows": null,
  "dtype": "f32",
  "shape": [12, 12, 64, 64],
  "rate_hz": null,
  "bytes": 28311552,
  "wire": true,
  "warnings": [],
  "updated_at": "2026-11-15T12:00:00Z"
}
```

| Key | Value |
|---|---|
| `id` | `<kind>:<name>`. |
| `kind` | `table`, `tensor`, `live`, `feed` or `metric`. |
| `name` | 1 to 256 bytes of UTF-8, with no control characters and no space at either end. It may hold `.`, `/` and `:`. |
| `title` | Text for a person, with the same rules as a name. It is the name when the origin has nothing better. |
| `origin` | An object that says where the source comes from, with the keys listed [below](#origins). |
| `columns` | For a tabular kind, a list of 0 to 4096 `{"name", "type"}` objects, where `type` is a DuckDB type name such as `BIGINT`, `DOUBLE` or `VARCHAR`. `null` for the other kinds. |
| `rows` | For a tabular kind, the number of rows, or `null` when it is not known without a scan. `null` for the other kinds. |
| `dtype` | For a tensor-shaped kind, one of `u8 i8 u16 i16 u32 i32 u64 i64 f16 f32 f64 bool`. `null` for the tabular kinds. |
| `shape` | For a tensor-shaped kind, a list of 0 to 8 sizes, each at least 0, whose product fits in 2^62. `null` for the tabular kinds. |
| `rate_hz` | A positive rate in hertz, or `null`; its meaning depends on the kind (see the table below). |
| `bytes` | The size of the data in bytes, or `null` when it is not known. |
| `wire` | `true` when the source has a wire form, so that a spec that binds it can run in the browser viewer. |
| `warnings` | 0 to 16 strings of 1 to 1024 bytes, such as "loaded weights-only from a pickle file". |
| `updated_at` | When the source's data last changed, in UTC ISO 8601 (`2026-11-15T12:00:00Z`, with optional fractions of a second), or `null`. |

The parser is strict: a missing key, an unknown key, a value of the wrong type or a value outside its bounds is an error with the code `catalog.invalid_entry` and `where` naming the key (`columns[3].type`). The same checks run on every entry a provider gives, so a provider's fault never reaches an agent as a malformed entry.

### Ids

A source id is `<kind>:<name>`, split at the first colon. The kind must be one of the five, and the name follows the rules above, so an id has at most 263 bytes. `parse_catalog_id()` returns the kind and the name, or `catalog.invalid_id`. The names each origin gives are in the table of [origins](#origins).

### Kinds

| Kind | Shape | `columns`, `rows` | `dtype`, `shape` | `rate_hz` | `bytes` |
|---|---|---|---|---|---|
| `table` | Tabular | Required; `rows` may be `null` | `null` | `null`: a table changes when a loader runs, not at a rate | The stored size of a table-mode output, else `null` |
| `tensor` | Tensor | `null` | Required | The sampling rate along axis 0 for a signal such as audio (the `wav` reader sets it), else `null` | Element count times the dtype's size |
| `live` | Tensor | `null` | Both `null` until the first snapshot, then both set | Snapshots per second measured over the last 5 s; `null` until two snapshots have arrived | The size of the newest snapshot |
| `feed` | Tabular | Required: `seq BIGINT`, `t_ns BIGINT`, `value FLOAT`; `rows` is the number of samples buffered | `null` | The channel's nominal rate from `caliper.feed.v1` | `null` |
| `metric` | Tabular | Required: `step BIGINT`, `value DOUBLE`; `rows` is the number of points | `null` | `null`: points arrive per training step | `null` |

### Origins

Each origin fills the fields as follows. The keys of `origin` are a closed set, so that the entry's JSON Schema can be closed too; each value is a string of at most 1024 bytes, except `run`, which is an integer.

| Origin | Kinds | Name | `origin` keys | `title` | `updated_at` |
|---|---|---|---|---|---|
| A loader's output | `table`, `tensor` | The output's `name`; with `tensor = "*"`, the prefix and the tensor's name in the file | `loader`, `file` (relative to the workspace, or the dropped file), `output`, and `tensor` for a tensor | The output's name | The end of the run that made it |
| A table of the database that no loader made | `table` | The table's name; `<schema>.<table>` outside the schema `main` | `database` (`workspace`), `schema`, `table` | The table's name | `null` |
| A live view of the attached live session | `live` | The view's name as Caliper LIB published it | `session`, `view`, `view_type` (`heatmap`, `points`, `histogram` or `series`) | The view's name | The newest snapshot |
| A channel of `caliper.feed.v1` | `feed` | The channel's id, such as `sys.cpu.util` | `service` (`feed`), `channel`, `units` | The channel's display name | The newest sample |
| A run and tag of `caliper.metrics.v1` | `metric` | `<run id>/<tag>`, such as `7/train/loss` | `service` (`metrics`), `experiment`, `run`, `run_name`, `tag` | `<run name> <tag>` | The newest point |

Only one live session is attached at a time, so a live view's name needs no session prefix. A metric's name uses the run's id rather than its name, because two runs may share a name; the title carries the name.

## The commands

| Command | Arguments | Result |
|---|---|---|
| `catalog.list` | `kind` (optional), `prefix` (optional, matched against the start of the name, case-sensitive), `limit` (1 to 1000, default 100) | `entries` (full entries, sorted by id), `truncated` (more entries matched than `limit`), `warnings` (strings that report duplicate ids and providers that failed) |
| `catalog.describe` | `id` | `entry`; or `catalog.invalid_id`, `catalog.not_found` |
| `catalog.sample` | `id`; `limit` (rows of a tabular source, 1 to 1000, default 20); `start` and `count` (for a tensor-shaped source, one integer per axis) | A sample (below); or `catalog.invalid_id`, `catalog.not_found`, `catalog.no_data`, `catalog.bad_request`, `catalog.provider_failed` |

A sample has the same keys for every kind, with `null` for those that do not apply:

```json
{
  "id": "tensor:attn_weights", "kind": "tensor",
  "columns": null, "rows": null, "total_rows": null, "truncated": false, "cut_cells": 0,
  "dtype": "f32", "shape": [12, 12, 64, 64],
  "stats": {"count": 7077888, "sampled": 7077888, "exact": true, "nonfinite": 0,
            "min": 0.0, "max": 0.93, "mean": 0.0156, "std": 0.041},
  "slice": {"start": [0, 0, 0, 0], "shape": [8, 8, 8, 8], "values": [0.012, 0.018, "..."]}
}
```

## Sampling rules and limits

**Tabular sources.**

- A table gives its first `limit` rows in the table's own order, with no sort. A feed gives its newest `limit` samples, oldest first. A metric gives its last `limit` points by step, in step order.
- `rows` is a list of rows, each a list of cells in the order of `columns`. A cell is a number, a string, a boolean or `null`. Integers are numbers; floating-point values are numbers, with NaN and infinities written as the strings `nan`, `inf` and `-inf`; dates and times are ISO 8601 strings; nested values (`LIST`, `STRUCT`, `MAP`) are their JSON text as a string; a `BLOB` is the string `blob of <n> bytes`.
- A string cell longer than 1024 bytes is cut at a character boundary to at most 1024 bytes, and `cut_cells` counts the cells cut.
- `total_rows` is the source's row count when it is known. `truncated` is true when the source has more rows than were given.
- `Catalog` stops adding rows once the sample's JSON passes 4 MiB and sets `truncated`, whatever the provider gave.

**Tensor-shaped sources** (`tensor`, and `live` from its newest snapshot):

- `stats` is computed over the whole tensor when it has at most 2^26 (67,108,864) elements. A larger tensor is read with a stride: every k-th element, with k the smallest integer that keeps the count within 2^26. `count` is the tensor's element count, `sampled` the number of elements read, and `exact` is true when they are equal.
- `nonfinite` counts NaN and infinite values among the elements read. `min`, `max`, `mean` and `std` cover the finite ones and are `null` when there are none. `std` is the population standard deviation. A `bool` tensor counts as 0 and 1.
- `slice` is a box of the tensor: `start` and `shape` give its corner and its size on each axis, and `values` lists its elements in row-major order. The box has at most 64 elements on each axis and at most 4096 in all. When `count` is absent, each axis takes as many elements as fit: 64 for a tensor of rank 1 or 2, 16 for rank 3, 8 for rank 4, then 5, 4, 3 and 2 for ranks 5 to 8, or the axis's size when that is smaller. `start` defaults to zeros. A rank-0 tensor gives its one value.
- `start` and `count`, when given, must have one entry per axis, each `count` from 1 to 64 (0 is allowed only on an axis of size 0), each `start` within the axis, and `start + count` at most the axis's size; otherwise the answer is `catalog.bad_request`.
- Values are numbers, `true` and `false` for `bool`, and the strings `nan`, `inf` and `-inf` for non-finite values. `f16` is decoded to its exact value.
- A live view that has published nothing yet answers `catalog.no_data`.

`sample_tensor()` implements these rules over a dtype, a shape and a buffer of little-endian bytes. It checks that the buffer's size is exactly the element count times the dtype's size before it reads anything, and answers `catalog.bad_tensor` otherwise, so a provider can hand it bytes mapped from a file.

## Providers, merging and refreshing

### The provider interface

A provider implements four functions. None may throw; `Catalog` catches an exception anyway and treats it as a failure of that provider.

| Function | Contract |
|---|---|
| `list(filter)` | The provider's entries that match the filter's `kind` and `prefix`, sorted by id, at most `filter.limit` of them, with `truncated` set when it has more. |
| `describe(id)` | The entry with that id, or a not-found answer, or an error. |
| `sample(id, request)` | A sample by the rules above, or an error. |
| `revision()` | A counter that increases whenever an entry is added or removed, or an entry's schema changes (its columns, dtype, shape or kind). A change of data alone, of `rows`, `bytes`, `rate_hz` or `updated_at`, does not count, so a live view at 60 Hz does not churn it. |

Each provider has a name (`[a-z][a-z0-9_-]{0,31}`, unique in its catalog) and a priority. A catalog holds at most 32 providers.

### Merging

- `Catalog::list` asks each provider for `limit + 1` entries, so that it can tell whether more exist. Asking for that many is enough: an entry that ranks beyond `limit` in its own provider ranks beyond `limit` in the merge too.
- Every entry is checked with the strict rules. An entry that fails is dropped, with a warning that names the provider, the entry and the error. The filter is applied again, in case a provider ignored it.
- **Duplicate ids.** When two providers give the same id, the provider with the higher priority wins, and between equal priorities the one added first wins. The result reports each duplicate as a warning ("table:run42 is also offered by data; the entry of loaders is shown") and, in C++, as a `CatalogDuplicate` with the id, the provider kept and the providers hidden.
- The merged entries are sorted by id in byte order and cut to `limit`; `truncated` is set when more entries matched.
- A provider that fails adds a warning and contributes nothing; the others still answer.
- `describe` and `sample` ask the providers in priority order (equal priorities in the order they were added) and use the first that has the id, so the same provider wins as in `list`. When no provider has the id and one of them failed, the answer is `catalog.provider_failed` rather than `catalog.not_found`, because the source may exist.
- `Catalog::sample` checks the request's bounds before it asks a provider, and checks the sample it gets back: it cuts rows beyond `limit` or past 4 MiB, and refuses a slice of more than 4096 values with `catalog.provider_failed`.

### Refreshing

`Catalog` keeps no cache. Every `list`, `describe` and `sample` asks the providers, so the catalog is always as fresh as they are. Each provider keeps its own state current: the loader provider updates its entries when a loader runs, the live provider when a session attaches or detaches and when a view first publishes, and the feed and metric providers read their services when they are asked. `Catalog::revision()` is the sum of the providers' revisions plus the number of times a provider was added or removed. Since every term only grows, a pane that shows the catalog can poll it once a frame and list again only when it changes.

## Decisions

- **Every key is present, and the parser is strict.** An agent reads the same fifteen keys for every source, and a provider's mistake is caught at the catalog rather than in a spec.
- **The `origin` keys are a closed set.** The command table's schemas are closed for strict tools, so the entry's schema must be closed too. A provider that needs a new key adds it here and in the parser.
- **A metric's name is `<run id>/<tag>`.** Run names are not unique.
- **A slice is flat, with its shape.** It keeps the sample's schema closed and non-recursive, and `shape` says how to read it.
- **Statistics are exact up to 2^26 elements, then strided.** A sample must stay cheap on a tensor of many gigabytes that is mapped from a file.
- **Duplicates go to the highest priority, then to the provider added first.** The rule is deterministic, and the wiring sets the priorities (loader outputs above plain database tables, for example).
- **A failing provider does not fail the catalog.** One broken origin must not hide every other source from an agent.
- **No cache in `Catalog`.** The providers know when their data changes; a second cache would only go stale.
