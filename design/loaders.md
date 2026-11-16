# Loaders

This page designs the loaders of the applet builder: saved, declarative recipes that turn files into catalog sources. A loader names a file or a glob, a reader and either SQL or tensor transforms, and running it produces tables in the workspace database and tensors that the catalog lists. It elaborates the loader part of the sprint 3 contracts, and the [Applet builder](applet-builder.md) page gives the wider design.

The library `caliper_builder_loaders` in `src/builder/loaders/` is built on the `sprint3-loaders` branch, with its public headers in `src/builder/include/caliper_builder/loaders/`, and the `sprint3-pyloader` branch adds [the Python loader](#the-python-loader). It is not wired into the app yet; that waits for sprint 2 to merge. The [last section](#what-is-built-and-what-is-planned) lists what is built and what is only planned.

## The loader format v0

A loader is a TOML file, `loaders/<id>.loader.toml` in the workspace. The text is the source of truth. Parsing is strict: an unknown key, a value of the wrong type or a missing required key is an error that names the key and carries the line and the column of the fault.

```toml
caliper_loader = 0

[loader]
id = "run42-metrics"
title = "Run 42 metrics"

[input]
path = "data/run42/*.parquet"
reader = "parquet"

[reader]
union_by_name = true

[[output]]
name = "run42_metrics"
mode = "view"
sql = "SELECT step, loss FROM input WHERE step > 0"
```

A tensor file selects tensors and may transform them:

```toml
caliper_loader = 0

[loader]
id = "run42-weights"

[input]
path = "checkpoints/run42.safetensors"

[[output]]
name = "attn_entropy"
tensor = "layers.0.attn.weights"
  [[output.transform]]
  op = "reduce"
  axis = -1
  fn = "entropy"
```

### Keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `caliper_loader` | integer | required | The format version. Only `0` is accepted. |
| `loader.id` | string | required | The loader id, `[a-z][a-z0-9_-]{0,63}`. It also names the file. |
| `loader.title` | string | the id | A title for people. |
| `loader.description` | string | empty | One or more sentences about the loader. |
| `input.path` | string | required | A file or a glob. A relative path resolves against the workspace. |
| `input.reader` | string | inferred | One of the readers below. When it is absent, the reader is inferred from the first matched file. |
| `reader` | table | empty | Options of the chosen reader. A key that the reader does not take is an error. |
| `output` | array of tables | required | At least one output, at most 256. |
| `output.name` | string | required | The source name, `[a-z][a-z0-9_-]{0,63}`. A table output becomes `table:<name>`, a tensor output `tensor:<name>`. With `tensor = "*"` the name is a prefix. |
| `output.title` | string | the name | A title for people. |
| `output.mode` | string | `"view"` | Tabular readers, and a Python loader's table outputs. `"view"` reads the files on each query; `"table"` copies the rows into the database when the loader runs. |
| `output.sql` | string | none | Tabular readers, and a Python loader's table outputs that name one table. One `SELECT` statement over the relation `input`, which holds every row the reader read. |
| `output.tensor` | string | none | Tensor readers and the Python loader. The name of one tensor in the file, or `"*"` for all of them. Required for `npz`, `safetensors` and `hdf5`; refused for `npy`, `image` and `wav`, whose files hold one tensor. A Python loader's output takes exactly one of `tensor` and `table`. |
| `output.table` | string | none | The Python loader only. The name of one table the script wrote, or `"*"` for all of them. |
| `output.transform` | array of tables | empty | Tensor readers, and a Python loader's tensor outputs. The tensor transforms of [Tensor outputs and transforms](#tensor-outputs-and-transforms), applied in order. |

A recipe is at most 1 MiB of text, and an output's `sql` at most 64 KiB. The canonical writer emits the keys in the order of the table above and does not keep comments, as the spec writer does not; a `description` is the place for prose. Unknown keys are errors everywhere, and the loader format reserves no `x` table for extensions, unlike the spec: a recipe is small and a typo in it should never pass silently.

## Readers

| Reader | Extensions | Magic bytes | Through | Gives | Options |
|---|---|---|---|---|---|
| `csv` | `.csv`, `.tsv` | text | DuckDB `read_csv` | table | `header`, `delimiter`, `quote`, `escape`, `skip`, `null`, `union_by_name` |
| `parquet` | `.parquet`, `.pq` | `PAR1` | DuckDB `read_parquet` | table | `union_by_name` |
| `json` | `.json` | text | DuckDB `read_json` | table | `format`, `union_by_name` |
| `jsonl` | `.jsonl`, `.ndjson` | text | DuckDB `read_json`, newline delimited | table | `union_by_name` |
| `arrow` | `.arrow`, `.feather` | `ARROW1` | see [Arrow IPC files](#arrow-ipc-files) | table | none |
| `npy` | `.npy` | `\x93NUMPY` | native | one tensor | none |
| `npz` | `.npz` | `PK\x03\x04` | native zip reader, stored or deflate members | tensors | none |
| `safetensors` | `.safetensors` | a little-endian length, then `{` | native, header JSON plus mapped bytes | tensors | none |
| `hdf5` | `.h5`, `.hdf5` | `\x89HDF\r\n\x1a\n` | a refusal on this branch; libhdf5 when a build finds it, planned | tensors | `dataset` |
| `image` | `.png`, `.jpg`, `.jpeg` | `\x89PNG`, `\xFF\xD8\xFF` | stb_image | one tensor `u8 [H, W, C]` | `channels` |
| `wav` | `.wav` | `RIFF....WAVE` | native | one tensor `[frames, channels]` with `rate_hz` | none |
| `python` | anything, named explicitly | none | a Python loader, out of process | tables and tensors | `script`, `python`, `timeout_s` |

The reader options:

| Reader | Option | Type | Default | Meaning |
|---|---|---|---|---|
| `csv` | `header` | boolean | detected | Whether the first row names the columns. |
| `csv` | `delimiter` | string, one character | detected; a tab for `.tsv` | The field separator. |
| `csv` | `quote` | string, one character | `"` | The quote character. |
| `csv` | `escape` | string, one character | `"` | The escape character inside quotes. |
| `csv` | `skip` | integer, 0 to 1,000,000 | 0 | Lines skipped at the start of each file. |
| `csv` | `null` | string | empty | The text read as NULL. |
| `csv`, `parquet`, `json`, `jsonl` | `union_by_name` | boolean | false | Unify the columns of several files by name rather than by position. |
| `json` | `format` | string | `"auto"` | `"auto"`, `"array"` (one array of records) or `"unstructured"` (one or more JSON documents). |
| `hdf5` | `dataset` | string | all datasets | The path of one dataset in the file. |
| `image` | `channels` | integer, 0 to 4 | 0 | 0 keeps the file's channels; 1, 2, 3 or 4 converts to grey, grey and alpha, RGB or RGBA. |
| `python` | `script` | string | required | The script that loads the file, inside the workspace. |
| `python` | `python` | string | `"python3"` | The interpreter: an absolute path, or a command name looked up on `PATH`. |
| `python` | `timeout_s` | integer, 1 to 3600 | 120 | The time limit of one run, in seconds. |

### What each reader produces

- **`npy`** reads format versions 1, 2 and 3, in C or Fortran order and in little or big byte order. The dtypes are `bool`, `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`, `f16`, `f32` and `f64`. A little-endian file in C order is mapped where it lies; a big-endian or Fortran-order file is converted into a derived tensor. An object dtype is a pickle and is refused; structured, string and complex dtypes are refused as unsupported. The one tensor in the file is named after the file's stem.
- **`npz`** reads a zip archive of `.npy` members, stored or compressed with deflate, including the zip64 records that numpy writes. Each member `<name>.npy` is the tensor `<name>`. A stored member in little-endian C order is mapped inside the archive; a deflated member is inflated into a derived tensor after its CRC-32 is checked. Members that are not `.npy` files are skipped with a warning, and an encrypted member or another compression method is refused.
- **`safetensors`** reads the 8-byte header length, the JSON header and the byte buffer, and maps every tensor's bytes. The dtypes `BOOL`, `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `U64`, `I64`, `F16`, `F32` and `F64` map onto the catalog's dtypes. `BF16` is converted to a derived `f32` tensor with a warning, because bfloat16 is common in model weights and the catalog has no such dtype; the `F8` types are skipped with a warning.
- **`image`** decodes a PNG or a JPEG into `u8 [H, W, C]`, converting 16-bit PNG samples to 8 bits, and stores it as a derived tensor.
- **`wav`** reads PCM with 8, 16, 24 or 32 bits per sample, IEEE float with 32 or 64 bits, and `WAVE_FORMAT_EXTENSIBLE` with either sub-format. It produces `[frames, channels]` with `rate_hz` set to the sample rate. 8-bit PCM is `u8` (unsigned, as WAV stores it), 16-bit `i16`, 24-bit `i32` (sign-extended, so the values keep their 24-bit range), 32-bit `i32`, and float `f32` or `f64`. Every format except 24-bit PCM is mapped where it lies.
- **`hdf5`** is a refusal on this branch; see [HDF5](#hdf5).
- The tabular readers produce one `table` source per output, with the columns and DuckDB types read back from the database and the row count. The `json` and `jsonl` readers need DuckDB's JSON extension. The DuckDB release this branch builds against has it built in; the host's static DuckDB links only the `parquet` and `core_functions` extensions, so there both readers fail with `loader.json_not_available` until the host links `json` too. No reader needs any other extension, and none is ever loaded or installed at run time.

### Inferring the reader

When `input.reader` is absent, and when `data.import` makes a loader for a dropped file, the reader is inferred from the first matched file.

1. The extension, compared without case, picks a reader from the table above.
2. The first 16 bytes are compared with the magic bytes. A pickle or a torch archive is refused at this step, whatever the extension says ([The pickle policy](#the-pickle-policy)).
3. If the extension picked a reader and the bytes identify a different binary format, inference fails with `loader.reader_mismatch`, which names both. A text format (`csv`, `json`, `jsonl`) has no signature of its own, so it conflicts only with bytes that identify a binary format.
4. If the extension is unknown, the bytes decide, for the binary formats only; a file that matches none is refused with `loader.no_reader`, which points to the Python loader.

A recipe that names its reader skips inference, and the reader's own checks then report a file that is not what it claims to be. Every file of a glob must infer to the same reader.

For a dropped file, `infer_loader()` makes the default loader: its id is the file's stem made into an id (lower case, other characters replaced by `_`, `f_` in front of a leading digit), with `-2`, `-3` and so on when the id is taken; its input path is relative to the workspace when the file lies inside it and absolute otherwise; it names its reader explicitly, so a later run does not depend on inference; and it has one output named after the stem with `-` replaced by `_`, so that SQL can name it unquoted. A multi-tensor file gets `tensor = "*"` with the prefix `<stem>_`, so a safetensors file `model.safetensors` with a tensor `layers.0.attn` gives the source `tensor:model_layers_0_attn`.

### Arrow IPC files

The `arrow` reader is refused with `loader.arrow_not_available` on this branch. nanoarrow's IPC reader (Apache 2.0) was evaluated and not built here, for three reasons:

- nanoarrow's IPC module needs flatcc, a second fetched dependency, and reads the stream format, so the file format's footer and the old Feather v1 layout need code of their own.
- The DuckDB in this build has no `read_arrow`. Its C++ route for an `ArrowArrayStream` is the `arrow_scan` table function, which takes raw pointers, is not a stable interface between DuckDB releases, and can scan a stream only once.
- The other route, decoding with nanoarrow and inserting the rows through DuckDB's appender, needs a mapping of every Arrow type to a DuckDB type and its tests.

The plan stays the same: a later stream fetches nanoarrow at a pinned release, decodes the file into an `ArrowArrayStream` and copies it into a table, so an `arrow` output takes only `mode = "table"`. Until then, Arrow IPC files go through the Python loader or are converted to Parquet. `arrow_scan` is not reachable from SQL: its pointer arguments have no SQL type, so a query cannot pass it an address.

### HDF5

On this branch the `hdf5` reader is a refusal, `loader.hdf5_not_available`, whose message points to the Python loader (which reads the file with h5py) or to a conversion to safetensors or `.npy`. `find_package(HDF5)` finds no library on the machine this branch was built on, and the build never fetches or installs one, so no HDF5 code is compiled and `hdf5_available()` returns false. The plan is a reader behind `find_package(HDF5 COMPONENTS C)` that, when the library is found, reads one dataset (`reader.dataset`) or every dataset, with the same size checks as the other readers; a build without the library keeps the refusal.

## Globs

`input.path` may be a glob. `*` matches any characters within one path segment, `?` matches one character, and a segment that is exactly `**` matches zero or more folders. The rest of the path is literal. As in a shell, `*` and `?` do not match a leading dot, so `*.csv` skips `.hidden.csv`; a pattern that starts with a dot matches it.

- The fixed part of the pattern, before the first segment with a wildcard, is resolved first. The walk below it does not follow symbolic links to folders, goes at most 32 folders deep and visits at most 100,000 entries.
- Every match is a regular file, and every match must be allowed by the rules of [Paths and confinement](#paths-and-confinement); one match outside them fails the run with `loader.path_not_allowed`. A glob that matches nothing fails with `loader.no_match`, and one that matches more than 10,000 files fails with `loader.too_many_files`.
- Matches are sorted by path, so a run is deterministic.
- A tabular reader gets the sorted list of files as one relation, so a glob of Parquet files is one table. A view keeps that list, and a file added later appears when the loader runs again.
- A tensor reader reads each file on its own. When a glob matches several files, each source name gets the file's stem: `<name>_<stem>` for one tensor per file, and `<prefix><stem>_<tensor>` with `tensor = "*"`; the title gets the file name in parentheses. A glob `images/*.png` with the output name `img` gives `tensor:img_cat` and `tensor:img_dog`.

## Tables, views and SQL

Each table output becomes a view or a table in the main schema of `workspace.duckdb`, named after the output, so a spec's SQL refers to it by that name.

```sql
CREATE VIEW "run42_metrics" AS
WITH input AS (SELECT * FROM read_parquet(['/abs/data/run42/a.parquet', '/abs/data/run42/b.parquet'], union_by_name = true))
SELECT * FROM (
SELECT step, loss FROM input WHERE step > 0
)
```

- **`mode = "view"`** is the default. The view reads the files on each query, so it always shows their current contents, and the database stays small. It keeps the absolute paths of the files.
- **`mode = "table"`** copies the rows when the loader runs. Queries are faster and the files may move, at the cost of space in the database.
- **`sql`** is checked before it is used: DuckDB's own parser must find exactly one statement, and it must be a `SELECT` (a `FROM`-first query, `VALUES`, `DESCRIBE`, `SUMMARIZE` and `SHOW` parse as one). The whole `CREATE` statement is then parsed again and must be one `CREATE`. Without `sql` the output is `SELECT * FROM input`.
- Generated SQL quotes every identifier and every string literal, and a reader option that holds a NUL or another control character is refused.
- After the run the library reads the columns and their DuckDB types with a `LIMIT 0` query, and the row count with `count(*)`, which scans a view's files once.
- An output name that is already a table or view the loader did not make fails with `loader.source_exists`.
- An output name that is a reserved SQL keyword, such as `order` or `semi`, is valid but must be quoted in SQL (`SELECT * FROM "order"`); the source carries a warning that says so.
- The semicolon that may end an output's `sql` is cut before the SQL is embedded. The cut is made at the first semicolon outside strings, quoted identifiers and comments, which in one statement can only be its end; a misjudged cut can only produce SQL that fails the `CREATE` check.

## Tensor outputs and transforms

A tensor output selects tensors by `tensor`:

- `tensor = "<name in the file>"` selects one tensor, and the source is `tensor:<name>`. A file without that tensor fails the run with `loader.tensor_not_found`. A single-tensor file (`npy`, `image`, `wav`) takes no `tensor` key, and its source is `tensor:<name>`.
- `tensor = "*"` selects every tensor in the file, and each source is `tensor:<prefix><tensor name>`, with the tensor name lowered and every character outside `[a-z0-9_-]` replaced by `_`. A name longer than 64 characters keeps its first 55 and ends with `-` and 8 hexadecimal digits of a hash of the full name. Two sources that end with the same name fail the run with `loader.name_collision`.

`output.transform` lists the tensor transforms of the contracts, the same vocabulary that a spec uses:

| `op` | Fields | Meaning |
|---|---|---|
| `slice` | `axis`, and either `index` (drops the axis) or `start`, `stop`, `step` | Selects along one axis, with Python semantics: negative indices count from the end, out-of-range bounds are clamped, and `step` is at least 1. |
| `reduce` | `axis` (an integer or a list), `fn`, `keepdims` (default false) | Reduces over the axes with `mean`, `sum`, `min`, `max`, `std`, `var`, `norm`, `argmax`, `argmin` or `entropy`. |
| `normalise` | `mode`, optional `axis` (an integer or a list; default the whole tensor) | Rescales with `minmax`, `zscore`, `softmax` or `l2`. |

The reference implementation is on the CPU, in `tensor_ops.h`, and the app will run the same operations on the GPU later. Its rules, which a GPU version must match:

- An axis may be negative and counts from the end. `axis` is required for `slice` and `reduce`. A list of axes may not repeat one.
- `slice` keeps the input dtype and copies the selected elements into a contiguous tensor.
- `reduce` and `normalise` compute in double precision and return `f32`, or `f64` for `f64` input; `argmax` and `argmin` return `i64`, take exactly one axis and return the first index of the extreme value, where NaN counts as the extreme, as in numpy.
- `std` and `var` are population statistics (numpy's default, `ddof = 0`), and `norm` is the Euclidean norm.
- `entropy` treats the reduced axes as one distribution: the values are divided by their sum and the result is `-Σ p ln p` in nats, where a value of 0 or less contributes nothing and a sum of 0 or less gives NaN, as `scipy.stats.entropy` does.
- A reduction over zero elements gives 0 for `sum` and `norm`, NaN for the other functions, and fails with `transform.empty_reduce` for `argmax` and `argmin`.
- `minmax` maps a constant input to 0, `zscore` maps an input with zero spread to 0, `l2` maps a zero vector to 0, and `softmax` subtracts the maximum before it exponentiates.
- NaN propagates through every function except `entropy`, whose non-positive terms are skipped. Inputs of type `bool` count as 0 and 1, and `f16` is widened to double.

A tensor has at most 8 dimensions. A transform that names an axis the tensor does not have, or whose result would exceed the limits, fails with a `transform.*` error that names the transform's position in the list.

### Statistics for `catalog.sample`

`tensor_stats` gives the count of elements, the count of non-finite values, and the minimum, maximum, mean and population standard deviation over the finite values (null when there are none). `tensor_sample` gives a slice from the start of the tensor of at most 64 values along each axis and at most 4,096 values in all: when 64 per axis would exceed that, the leading axes are cut to one index each, from the first axis on, until it fits. The slice is nested JSON arrays, with its start and shape, and a non-finite value appears as the string `"NaN"`, `"Infinity"` or `"-Infinity"`.

## Where loaders and their outputs are stored

```
<workspace>/
  loaders/<loader-id>.loader.toml      one saved loader each
  tensors/<source-name>.npy            derived tensors
  loader-outputs/<loader-id>/run-<n>/  the checked outputs of a Python loader's last run
  inputs.toml                          the files and folders the user granted
  workspace.duckdb                     views, tables, caliper.loaders and caliper.sources
```

The text files are the source of truth, and `LoaderStore::rebuild()` recreates every database row from them: it drops the views and tables that loaders made, empties the two tables, reads every `.loader.toml` file, runs each loader again (a Python loader's sources are read back from its kept outputs instead; see [Where the outputs are kept](#where-the-outputs-are-kept)) and deletes the derived tensors that no source uses. Opening a workspace whose database is missing, or lacks the `caliper` schema, rebuilds it. Opening one whose database exists brings the rows in line with the files without running anything: a row whose file is gone is removed with its sources, and a file without a row, or whose text changed since it was saved (an edit by hand), is indexed with `last_status = "never"`. A loader file that does not parse, or whose id differs from its file name, is listed with `last_status = "error"` and the error, and makes no sources.

Times are UTC in ISO 8601 (`2026-11-15T12:00:00Z`), stored as text so that they sort and read the same everywhere.

### `caliper.loaders`

| Column | Type | Meaning |
|---|---|---|
| `id` | VARCHAR, primary key | The loader id. |
| `title` | VARCHAR | The loader's title. |
| `recipe` | VARCHAR | The text of `loaders/<id>.loader.toml`. |
| `created_at` | VARCHAR | When the loader was first saved. |
| `updated_at` | VARCHAR | When it was last saved. |
| `last_run_at` | VARCHAR | When it last ran, or NULL. |
| `last_status` | VARCHAR | `never`, `ok` or `error`. |
| `last_error_code` | VARCHAR | The error code of the last failed run, or NULL. |
| `last_error_message` | VARCHAR | Its message, or NULL. |

### `caliper.sources`

| Column | Type | Meaning |
|---|---|---|
| `id` | VARCHAR, primary key | `table:<name>` or `tensor:<name>`. |
| `kind` | VARCHAR | `table` or `tensor`. |
| `name` | VARCHAR | The source name. |
| `title` | VARCHAR | Its title. |
| `loader` | VARCHAR | The id of the loader that made it. |
| `file` | VARCHAR | The input as the catalog shows it: for a table, the recipe's `input.path`; for a tensor, the file it came from, relative to the workspace when it lies inside it. |
| `columns` | VARCHAR | The columns as JSON, `[{"name": ..., "type": ...}]`, or NULL for a tensor. |
| `rows` | BIGINT | The row count, or NULL for a tensor. |
| `dtype` | VARCHAR | The dtype, or NULL for a table. |
| `shape` | VARCHAR | The shape as a JSON list, or NULL for a table. |
| `rate_hz` | DOUBLE | The sample rate of an audio tensor, or NULL. |
| `bytes` | BIGINT | A tensor's size in bytes; for a table, the total size of its input files. |
| `wire` | BOOLEAN | Always true for loader outputs: the host can send a table's rows or a tensor's bytes to the browser viewer as a snapshot. |
| `warnings` | VARCHAR | A JSON list of warning texts. |
| `updated_at` | VARCHAR | When the run that made it finished. |
| `table_mode` | VARCHAR | `view` or `table` for a table, NULL for a tensor. |
| `tensor` | VARCHAR | The tensor's name in its file, or NULL. |
| `data_file` | VARCHAR | The absolute path of the file that holds a tensor's bytes. |
| `data_offset` | BIGINT | The byte offset of those bytes in `data_file`. |
| `derived` | BOOLEAN | Whether `data_file` is a derived tensor under `tensors/`. |

A tensor's bytes stay in its file and are mapped when a panel or `catalog.sample` uses them. A tensor that had to be converted (a deflated `npz` member, a big-endian or Fortran-order `npy`, `bf16`, 24-bit audio, a decoded image) or transformed is written to `tensors/<source-name>.npy`, a version 1 `.npy` file whose header is padded so that the data starts on a 64-byte boundary, and the row points into it. The library writes a derived file under a temporary name and renames it once the run's database transaction has committed, so a failed run leaves the previous outputs whole.

The C++ side carries each output as a `LoadedSource`, whose fields are exactly those of the catalog entry of the contracts, plus the storage columns above. Its `to_json()` gives the catalog entry's JSON form, and the catalog's `CatalogProvider` for loader outputs is a thin adapter over `LoaderStore::sources()`, `describe()` and `sample()`, added when the catalog module and this one merge. `sample()` gives `{"id", "columns", "rows", "truncated"}` for a table, with at most `limit` rows (1 to 1,000), and `{"id", "dtype", "shape", "stats", "slice"}` for a tensor, from `tensor_stats` and `tensor_sample`. A source that does not exist fails with `catalog.not_found`.

A run is atomic. The store drops the loader's previous views and tables and deletes its previous rows, creates the new outputs and records them in one DuckDB transaction, and commits; only then are the derived tensors renamed into place and the previous derived files that the run did not remake deleted. A failed run rolls the transaction back, deletes its temporary files and records the error in `caliper.loaders`, so the previous sources stay whole and queryable.

## The C++ interface

The public headers are in `src/builder/include/caliper_builder/loaders/`:

| Header | What it gives |
|---|---|
| `tensor.h` | `DType`, `TensorData` (owned or mapped), `check_shape`, `half_to_float` |
| `tensor_ops.h` | `TensorTransform`, its JSON form, `apply_transforms`, `tensor_stats`, `tensor_sample` |
| `tensor_readers.h` | `read_npy`, `read_npz`, `read_safetensors`, `read_image`, `read_wav`, `read_hdf5`, `write_npy`, `ReaderLimits` |
| `readers.h` | The reader names, their extensions, `sniff_format` and `validate_reader_options` |
| `python_loader.h` | `PythonReaderOptions`, `resolve_python`, `python_environment`, `embedded_loader_host`, `PythonManifest`, `read_python_manifest`, `read_typed_csv_header`, `run_python_script` and their limits |
| `pickle.h` | `check_not_pickle` |
| `loader.h` | `Loader`, `parse_loader`, `loader_from_json`, `write_loader`, `make_id` |
| `inference.h` | `infer_reader`, `infer_loader` |
| `paths.h` | `AllowedRoots`, `expand_input`, `path_allowed` |
| `query.h` | `QueryResult`, `ColumnInfo` |
| `loaded_source.h` | `LoadedSource` |
| `store.h` | `LoaderStore`: `open`, `grant`, `save`, `get`, `list`, `remove`, `run` (with `RunOptions::allow_python`), `import_files`, `sources`, `describe`, `sample`, `load_tensor`, `query`, `rebuild` |

DuckDB stays behind one source file, `db.cpp`, so no public header includes `duckdb.hpp`. Every method of `LoaderStore` takes one mutex, so the command handlers may call it from any thread; one store opens a workspace at a time in a process. A Python loader's run releases the mutex while its script runs. The library links `caliper_builder_acp` for `acp::run_process`.

## The pickle policy

Loading a pickle runs code that the file chooses, so the native readers never load one. They refuse with `loader.pickle_refused`, whose message names the safe routes: convert the file to safetensors, or load it with a Python loader (`reader = "python"`) whose script calls `out.torch_weights(path)`, which reads torch files weights-only (`torch.load(weights_only=True)`) and marks each catalog entry with the warning "loaded weights-only from a pickle file".

The refusal applies to:

- the extensions `.pkl`, `.pickle`, `.pt`, `.pth`, `.ckpt` and `.joblib`, whatever the bytes are;
- any file whose bytes start with the pickle protocol marker, `\x80` followed by a protocol number from 2 to 5, which covers a `.bin` checkpoint written by `torch.save` with the legacy format;
- any zip archive with a member named `data.pkl` or ending in `/data.pkl`, which is the layout of `torch.save`;
- an `.npy` file, or an `.npz` member, whose dtype is `object` (`|O`), because numpy stores those as pickles;
- an `.npz` member that is a pickle by its bytes or by its name;
- a file whose extension has no reader, when its first byte is one of the opcodes `(`, `]`, `}` or `c` and its last byte is the STOP opcode `.`, which is how pickles of protocols 0 and 1 look, since they have no marker.

The last rule is a heuristic, so it applies only where no reader claims the extension; a CSV file that happens to end with a full stop is never refused. `check_not_pickle()` in `pickle.h` applies all the rules. Inference runs it on every dropped file, and `LoaderStore::run()` runs it on every matched file before any reader opens it, whatever reader the recipe names, except for a Python loader's input, which only the user's script opens; every file a Python loader hands back is checked instead. The `npy` and `npz` readers also refuse the object dtype themselves.

## The Python loader

A Python loader reads a file that the native readers do not know, or refuse, by running a script of the user's in a separate Python process. The process hands back only files of formats that the native readers already read safely (`.npy`, Parquet and CSV), and Caliper checks those files as it checks any other untrusted input. It is built on the `sprint3-pyloader` branch: `python_loader.h` and `LoaderStore::run()` in the library, and the host script `python/caliper/loader_host.py`.

It is the route for torch checkpoints (weights-only), Arrow IPC files until the native reader exists, HDF5 files without libhdf5, and any format the native readers do not know.

### The recipe

```toml
caliper_loader = 0

[loader]
id = "run42-ckpt"

[input]
path = "checkpoints/run42.pt"
reader = "python"

[reader]
script = "scripts/load_ckpt.py"
python = "/opt/envs/ml/bin/python"
timeout_s = 300

[[output]]
name = "run42_"
tensor = "*"

[[output]]
name = "run42_metrics"
table = "metrics"
sql = "SELECT step, loss FROM input WHERE step > 0"
```

- `reader = "python"` must be named; inference never chooses it.
- `input.path` must match exactly one file, inside the workspace or a grant like any other input; a glob that matches more fails with `loader.bad_value`. The script receives the file's canonical path.
- `reader.script` is required. It is a path inside the workspace, relative to it or absolute inside it, without a `..` segment, and its canonical path, with every symbolic link followed, must lie inside the workspace. A granted folder does not count, because a grant lets a loader read data, and a script is code. The path is checked when the loader is saved (by its text, so a script written later is allowed) and again when it runs, when the script must exist.
- `reader.python` is optional: an absolute path to an interpreter, or a command name looked up on the absolute entries of `PATH`. The default is `python3`. A relative path such as `.venv/bin/python` is refused with `loader.bad_value`, because its meaning would depend on the app's working folder.
- `reader.timeout_s` is a whole number of seconds from 1 to 3600, and 120 by default.
- Each `[[output]]` selects outputs of the script by kind and name, with exactly one of `tensor` and `table`: a name the script wrote, or `"*"` for every output of that kind. As for a multi-tensor file, `"*"` makes `name` a prefix, and each source is `<kind>:<prefix><the script's name>`, made into an id. A tensor output may carry transforms; a table output may carry `mode` and `sql`, except that `table = "*"` takes no `sql`. The `table` key is new in the loader format, and only the Python reader takes it.
- A selected name that the script did not write fails the run with `loader.tensor_not_found` or `loader.table_not_found`. An output of the script that no `[[output]]` selects is skipped, with a warning in the run's result.

### The script

The script defines one function and imports nothing from Caliper:

```python
def load(path, out):
    data = read_my_format(path)
    out.tensor("embeddings", data.vectors)
    out.table("metrics", [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.25}])
    out.warn("The file had no header, so the default column names were used.")
```

| Method | What it does |
|---|---|
| `out.tensor(name, array)` | Writes anything that `numpy.asarray` accepts as `<name>.npy`, little-endian and in row-major order, with `numpy.save(..., allow_pickle=False)`. The dtype must be one of the catalog's; the object dtype, strings, complex numbers, dates and structured dtypes raise `TypeError`, and more than 8 dimensions raise `ValueError`. |
| `out.table(name, rows)` | Writes a list of dicts, a dict of columns, or a pandas or pyarrow table as `<name>.parquet` through pyarrow when pyarrow can be imported, and as `<name>.csv` with the typed header below otherwise. A list of dicts gets every key of any row as a column, in the order first seen, with NULL where a row lacks it. |
| `out.warn(text)` | Adds a warning that every source of the run carries in its catalog entry: one line of at most 1,024 characters, and at most 64 per run. |
| `out.torch_weights(file)` | Calls `torch.load(file, map_location="cpu", weights_only=True)`, which refuses every pickled object that is not a tensor or a plain container, and calls no other loader. It walks the dicts and lists that come back and writes each tensor with its key path (`layers.0.attn.weight`) as its title and that path made into an id as its name. Each tensor carries the warning "loaded weights-only from a pickle file"; a `bf16` tensor becomes `f32` and also carries "converted from bf16 to f32"; a tensor whose dtype the catalog lacks, and every value that is not a tensor, is skipped with a warning of the run. It returns the names it wrote. |

Names match the id grammar and are unique per kind; a second output of the same kind and name raises `ValueError`. Each method writes its file at once, opened for exclusive creation, so a file the script planted under that name is never followed or overwritten. The script runs as a module with its own folder first on `sys.path`, so it can import modules that lie beside it. Its working folder is the run's output folder, so a file it writes by a relative path lands there and is discarded unless a method listed it. `make_id` in the host follows the library's rule exactly, so a checkpoint loaded through this route and the same weights in a safetensors file give the same source names, apart from names longer than 64 characters, whose hashes are taken over different text.

Without pyarrow, a table is written as CSV with a typed header. Every header cell is `"<column>:<type>"`, where the type is `BIGINT`, `DOUBLE`, `BOOLEAN` or `VARCHAR`. Every header cell and every `VARCHAR` value is quoted, with a quote inside doubled, so an empty string is `""`; NULL is an empty unquoted field; booleans are `true` and `false`; doubles use Python's shortest round-trip form, with `nan`, `inf` and `-inf`; fields are separated by commas and rows end with a line feed. A column of integers is `BIGINT`, of integers and floats `DOUBLE`, of booleans `BOOLEAN`, of strings or only NULLs `VARCHAR`; any other mixture, an integer outside 64 bits, a NUL character or a value of another type raises an error that suggests installing pyarrow. Caliper reads the header with its own parser and then the file with DuckDB's `read_csv`, with the columns and types from the header and every detection turned off.

### The host and the process

`loader_host.py` is one file that needs only the standard library and numpy, so it runs by path without Caliper LIB; with Caliper LIB installed it also runs as `python -m caliper.loader_host`. The library embeds it at build time (CMake turns it into a byte array), so the app needs no install path for it: for each run the store writes the embedded copy to `tmp-<n>/host/loader_host.py` and runs

```
<python> -I <workspace>/loader-outputs/<id>/tmp-<n>/host/loader_host.py \
    --script <script> --input <file> --output <workspace>/loader-outputs/<id>/tmp-<n>/out
```

through `acp::run_process`, never through a shell, with:

- a scrubbed environment: no inherited variable except `PATH` (its absolute entries only), `HOME`, `TMPDIR` and `LANG`, and, for an interpreter that lies in a virtual environment (a `pyvenv.cfg` above its `bin` folder), `VIRTUAL_ENV` with the environment's `bin` folder first on `PATH`. `-I` also makes Python ignore every `PYTHON*` variable and the user's site-packages. A conda environment runs its Python without variables, so none are passed for it;
- the empty output folder as its working folder;
- the time limit, after which the run fails with `loader.python_timeout`: the process group gets SIGTERM and then, within two seconds, SIGKILL, so processes the script started in turn are stopped too;
- at most 1 MiB kept of each output stream; the rest is read and dropped, so a script that prints gigabytes neither blocks on a full pipe nor fills memory.

The host exits with 0 when the outputs and `manifest.json` are written, with 1 when the script raised (it writes `error.json` with the exception's type and message and prints the traceback), with 2 for wrong arguments, and with 3 when numpy cannot be imported. The store reports exit 3, an interpreter that cannot be found or started, and a platform without process spawning as `loader.python_not_available`; any other failure as `loader.python_failed`, whose message carries the exception from `error.json`, or else the last line of standard error. The store's lock is released while the script runs, so the catalog stays usable; the run then takes the lock again to check and commit.

### The manifest and what Caliper checks

```json
{
  "caliper_python_loader": 0,
  "outputs": [
    {"name": "layers_0_attn_weight", "kind": "tensor", "file": "layers_0_attn_weight.npy", "bytes": 2176,
     "dtype": "f32", "shape": [16, 32], "title": "layers.0.attn.weight",
     "warnings": ["loaded weights-only from a pickle file"]},
    {"name": "metrics", "kind": "table", "file": "metrics.parquet", "bytes": 1534}
  ],
  "warnings": ["1 value of the checkpoint is not a tensor and was skipped."]
}
```

`read_python_manifest()` treats the folder as hostile input:

- The output folder and `manifest.json` must not be symbolic links. The manifest is at most 1 MiB, nested at most 64 deep (counted before parsing), and valid JSON with exactly the keys above; an unknown key is an error, and the version must be 0.
- Every name matches `[a-z][a-z0-9_-]{0,63}` and is unique per kind; the kind is `tensor` or `table`.
- Every file is a plain name of at most 128 characters from `[A-Za-z0-9_.-]`, not starting with a dot or a dash, so no `..`, separator or absolute path can pass (`loader.path_not_allowed`). It ends in `.npy` for a tensor, and `.parquet` or `.csv` for a table, and no two outputs share it.
- Every file is a regular file in the folder, not a symbolic link, with exactly one hard link (`loader.path_not_allowed`). Its declared `bytes` must equal its real size and stay within 4 GiB, and all files within 16 GiB (`loader.too_large`).
- A tensor declares a catalog dtype and a shape of at most 8 sizes that fits its file; a table declares neither.
- Titles and warnings are single lines of bounded length, at most 64 warnings per output and per run.

The store then moves the listed files into `run-<n>/`, checks them there again, and reads each with the native readers inside the run's transaction: `check_not_pickle()` on every file; the `npy` reader for a tensor, which refuses the object dtype, after which the dtype and shape must equal the manifest's (`loader.malformed`); DuckDB's `read_parquet` for a Parquet table; and the typed header and `read_csv` for a CSV table. The Python process never hands the app anything else.

### Where the outputs are kept

```
<workspace>/loader-outputs/<loader-id>/
  tmp-<n>/host/loader_host.py   during a run only: the embedded host script
  tmp-<n>/out/                  during a run only: the script's empty output folder
  run-<n>/manifest.json         the checked manifest, as the host rewrote it
  run-<n>/<name>.npy            the checked tensors
  run-<n>/<name>.parquet        the checked tables, or <name>.csv
```

Each run takes the next number `n`. After the checks the listed files move from `tmp-<n>/out/` to a new `run-<n>/`, the manifest is written there from the checked values, and `tmp-<n>/` is deleted with anything else the script left in it. The sources point into `run-<n>/`: a tensor is mapped from its `.npy` file, and a table is a view or a table over its Parquet or CSV file. A transform makes a derived tensor under `tensors/`, as for any other loader. When the run's transaction commits, the older `run-*` folders of the loader are deleted; a failed run deletes its own folders and leaves the previous `run-<n>/` and its sources whole.

A restart keeps the database rows, which point into `run-<n>/`, so nothing runs. `rebuild()`, and opening a workspace whose database is missing, reads the newest `run-<n>/` that has a manifest, checks it again as untrusted input (the files lie in the workspace, where anything can edit them) and makes the sources from it; the script does not run and needs no approval. A Python loader without kept outputs keeps `last_status = "never"`. Removing a loader deletes its folder; a successful run of a loader that is no longer a Python loader deletes it too; and `rebuild()` deletes the folders of loaders that are gone or are no longer Python loaders, and unfinished `tmp-<n>/` folders.

### Approval

Running a Python loader executes the user's code with the user's privileges, out of process. `LoaderStore::run(id, options)` therefore refuses a Python loader unless `RunOptions::allow_python` is set for that run, and fails with `loader.python_needs_approval` without touching any file or recording a status. The command layer sets the flag only after the user approved that run: the user's own click on run in the app is that approval, and `loader.run` of a Python loader counts as destructive, so the approval policy asks the user before it runs one sent by an agent, MCP or Caliper API. Reading kept outputs back in `rebuild()` runs nothing and needs no approval. The script is not sandboxed; the approval is what stands between an agent and running code.

### Limits

| Limit | Default |
|---|---|
| Time of one run | 120 seconds, at most 3,600 (`reader.timeout_s`) |
| Output kept of each stream | 1 MiB |
| Grace between SIGTERM and SIGKILL | 2 seconds |
| `manifest.json` | 1 MiB, nested at most 64 deep |
| Outputs of one run | 10,000 |
| One output file | 4 GiB |
| All output files | 16 GiB |
| Warnings per output, and per run | 64, of at most 1,024 characters |
| `error.json` | 64 KiB |
| CSV header | 1 MiB and 10,000 columns |
| Input files | 1 |

`PythonOutputLimits` and `PythonProcessOptions` in `StoreOptions` hold them.

### What the Python loader does not do yet

- Process spawning is POSIX only in `acp::run_process`, so on Windows every run fails with `loader.python_not_available`.
- The process group is killed when the time limit passes. A script that finishes normally but leaves a process behind, or a process that left the group (with `setsid`), keeps running; such a process could still change files after the checks, as any program of the user can change any file in the workspace.
- A Python loader reads one file. A glob of files, one run per file, is a later addition.

## Safety

### Paths and confinement

A loader may read:

- files inside the workspace;
- files and folders that the user granted, by dropping them on the app or naming them in a file dialog. `LoaderStore::grant()` records each one in `inputs.toml`, and only a user action calls it.

Every input path is checked before it is read. A path with a `..` segment is refused. Each match of a glob is resolved to its canonical path, with every symbolic link followed, and must then lie inside the workspace, inside a granted folder, or be a granted file; so a symbolic link inside the workspace that points elsewhere is refused. A path that fails fails with `loader.path_not_allowed`.

`inputs.toml` is an addition to the workspace layout of the contracts, which say which paths a loader may read but not where the user's grants are kept:

```toml
caliper_inputs = 0
files = ["/abs/path/run42.parquet"]
folders = ["/abs/path/datasets"]
```

The command layer must not let an agent widen it: `data.import` from the `agent`, `mcp` or `api` origin should grant a path only after the user approves.

### Sizes

Every reader checks every size it reads from a header against the file's real size before it allocates or maps anything, and checks every multiplication for overflow. No reader trusts a header's sizes, counts or offsets.

| Limit | Default |
|---|---|
| Dimensions of a tensor | 8 |
| Bytes of one derived tensor (inflated, decoded or converted) | 4 GiB |
| `.npy` header | 1 MiB |
| safetensors header | 100 MiB, nested at most 64 deep |
| Members of a zip archive | 65,536 |
| Image side, and pixels | 32,768 each side, 2<sup>28</sup> pixels |
| Channels of a WAV file | 1,024 |
| Files matched by a glob | 10,000 |
| Sources made by one run | 10,000 |
| Rows returned by one `query` | 100,000 (the caller's limit, clamped) |
| Time of one `query` | 60 seconds (the caller's limit, clamped) |

### What the database may read

`workspace.duckdb` is opened with these settings, which were tested on DuckDB v1.5.6:

| Setting | Value | Effect |
|---|---|---|
| `enable_external_access` | false | SQL can read no file, install no extension and attach no database outside the allowed paths. |
| `allowed_directories` | the workspace and the granted folders, each with a trailing separator | SQL can read files below them. |
| `allowed_paths` | the granted files | SQL can read these files. |
| `autoinstall_known_extensions` | false | No query downloads an extension. |
| `autoload_known_extensions` | false | No query loads an extension from disk. |
| `allow_community_extensions` | false | No community extension loads. |
| `allow_persistent_secrets` | false | No secret is written to disk. |
| `temp_directory` | `<workspace>/workspace.duckdb.tmp` | Spilling stays inside the workspace. |
| `threads`, `memory_limit` | 2 and 1 GB by default | The store does not take the machine. |
| `lock_configuration` | true | No `SET` can change any of the above. |

What the tests showed:

- `read_csv`, `read_text`, `read_blob`, `glob`, `sniff_csv` and `parquet_metadata` on a path outside the allowed ones fail with a permission error, and so do a path with `..` that leaves the workspace and a symbolic link inside the workspace whose target lies outside.
- `INSTALL` fails, and `LOAD` of an extension that is not built in fails. `LOAD json` succeeds where JSON is built in, which loads nothing new.
- `SET enable_external_access`, `SET allowed_directories` and every other `SET` fail once the configuration is locked.
- The `query()` table function accepts only a single `SELECT`.
- DuckDB refuses to change `temp_directory` once external access is disabled, so the store sets it first.
- DuckDB parses the table-valued pragmas, such as `PRAGMA database_list` and `PRAGMA version`, as `SELECT` statements over pragma functions. They pass the check and only read.

Two gaps remain, and both are closed by the statement check rather than by a setting. DuckDB v1.5.6 has no setting that forbids `ATTACH`: a database file inside an allowed path, or `:memory:`, can be attached. And `COPY ... TO` may write inside an allowed folder, which includes the workspace. Neither is a `SELECT`, so neither passes the check that every SQL text of a loader or of `query` must pass; the store's own statements are the only others that run. A third, smaller gap is not closed: a `SELECT` can call `nextval()` on a sequence, and the rollback does not undo that. The workspace database holds no sequence, and SQL cannot create one.

A changed grant reopens the database with the new lists, because a locked configuration cannot be changed while it is open.

### The query helper

`LoaderStore::query(sql, row_limit, timeout_ms)` is the read-only query of `data.query` and the table side of `catalog.sample`:

- The text must be exactly one `SELECT`, checked by DuckDB's parser, and the parsed statement is what runs.
- It runs inside a transaction that is always rolled back, so it changes nothing even through a function with a side effect.
- It streams the result and stops after `row_limit` rows, then reports whether more rows existed.
- A watchdog thread interrupts the query when `timeout_ms` passes (`query.timeout`); the connection stays usable.
- Values come back as JSON: numbers, strings, booleans and null, with other DuckDB types as their text form, and a non-finite float as `"NaN"`, `"Infinity"` or `"-Infinity"`. A 128-bit integer, which is what `sum()` of a `BIGINT` column gives, is a number when it fits 64 bits and text otherwise.

## Errors

| Code | When |
|---|---|
| `loader.parse` | The recipe is not valid TOML, or a value has the wrong type. |
| `loader.unknown_key`, `loader.missing_key` | A key is unknown or missing. |
| `loader.bad_id`, `loader.bad_value` | An id or a value breaks its rule. |
| `loader.no_reader`, `loader.reader_mismatch` | Inference found no reader, or the extension and the bytes disagree. |
| `loader.path_not_allowed`, `loader.no_match`, `loader.too_many_files` | A path or a glob fails confinement or its limits. |
| `loader.pickle_refused` | A pickle or a torch archive. |
| `loader.arrow_not_available`, `loader.hdf5_not_available` | A reader that this build does not have. |
| `loader.python_needs_approval` | A Python loader was run without `RunOptions::allow_python`. |
| `loader.python_not_available` | The Python loader's interpreter is missing or cannot start, numpy is not installed for it, or the platform cannot start a process. |
| `loader.python_failed`, `loader.python_timeout` | The script failed, or ran past its time limit and was stopped. |
| `loader.json_not_available` | The DuckDB build lacks its JSON extension. |
| `loader.truncated`, `loader.malformed`, `loader.unsupported`, `loader.too_large` | A file is cut short, breaks its format, uses a feature the reader does not support, or exceeds a limit. |
| `loader.tensor_not_found`, `loader.table_not_found`, `loader.name_collision`, `loader.source_exists` | An output names a tensor or a table that is not there, or a name clashes. |
| `loader.not_found` | No saved loader has the id. |
| `loader.io`, `loader.database` | A file or the database could not be opened, read or written. |
| `catalog.not_found`, `catalog.not_a_tensor` | No source has the id, or a table was asked for as a tensor. |
| `tensor.rank`, `tensor.too_large`, `tensor.shape` | A shape has more than 8 dimensions, overflows or is negative. |
| `loader.sql_not_select` | An output's SQL, or a query, is not exactly one `SELECT`. |
| `loader.sql` | DuckDB rejected the SQL; the message is DuckDB's. |
| `query.timeout` | A query ran past its time limit. |
| `transform.*` | A tensor transform is invalid for its input. |

## What is built and what is planned

Built on this branch, with tests: the tensor type and the three transforms with the statistics; the `npy`, `npz`, `safetensors`, `image` and `wav` readers; the HDF5 refusal; the pickle refusal; the tabular readers through DuckDB with the SQL check; the loader format's parser and writer, inference and globs; `LoaderStore` with its runs, grants, rebuild, the safety settings and the query helper; and an end-to-end test that imports a Parquet file and a safetensors file into a workspace, runs both loaders, reads back a table source and a tensor source, and finds them again after reopening the workspace and after deleting its database.

Built on the `sprint3-pyloader` branch, with tests: the Python loader, that is `loader_host.py` with its writer, its typed CSV and `out.torch_weights`; the interpreter lookup and the scrubbed environment; the run through `acp::run_process` with its time limit and bounded output; the manifest checks; the approval flag; the kept outputs and their reuse by `rebuild()`; and the pickle refusal's pointer to this route. The tests run every dtype and both table formats through a whole run; a script that raises, hangs (its child process is stopped with it), prints megabytes, forges a manifest with path tricks, an object array, a symbolic link out of the folder or a huge declared size; a missing interpreter and a missing numpy; a run without approval; and the scrubbed environment. A `.pt` file that the native readers refuse as a pickle loads through a Python loader whose script calls `out.torch_weights`, with a stand-in `torch` module beside the script that records the call, so the test needs no torch and shows that `weights_only=True` is what is called; a second test uses the real torch when the test interpreter has it, and is skipped otherwise. `loader_host.py` has unit tests of its own, which ctest runs with the test interpreter. The tests that run Python need `-DCALIPER_BUILDER_TEST_PYTHON=<interpreter with numpy>`, and pyarrow for the Parquet path; without it they say they are skipped and the rest still run.

Planned: the `CatalogProvider` adapter (after the catalog module merges), the command handlers (a later stream), the Arrow IPC reader, the HDF5 reader where libhdf5 is found, stacking a glob of images into one tensor, the GPU version of the transforms, Python loaders on Windows, and a Python loader over a glob, with one run per file.

## Decisions

| Decision | Reason |
|---|---|
| Granted paths live in `inputs.toml`, written only by `grant()`. | The contracts require a user grant for paths outside the workspace but give it no place; a saved loader cannot be its own grant, or an agent could name any file. |
| Views hold absolute paths of the matched files. | DuckDB resolves a relative path against the process's folder, not the workspace. A moved workspace runs its loaders again through `rebuild()`. |
| A glob is expanded by the library, not by DuckDB. | Each match must pass the confinement check, and DuckDB then sees only allowed files. |
| `bf16` becomes a derived `f32` tensor with a warning. | bfloat16 is common in weights and the catalog's dtype list lacks it. |
| Derived tensors are `.npy` files. | One format the `npy` reader already maps, readable by numpy. |
| Times are ISO 8601 text. | The host's DuckDB has no ICU, and text sorts and reads the same everywhere. |
| `query` rolls back its transaction. | It keeps `query` read-only on a connection that also writes. |
| A name that is an SQL keyword is allowed, with a warning. | The id grammar allows it, and quoting it works. |
| Unknown keys are errors, with no `x` table. | A recipe is small, and a typo must not pass. |
| `axis` is required for `slice` and `reduce`. | A missing axis is more often a mistake than a wish to reduce everything. |
| Stored `npz` members are mapped without a CRC check. | Checking would read the whole archive at load time; a deflated member is checked as it is inflated. |
| The Arrow IPC reader is a refusal for now. | See [Arrow IPC files](#arrow-ipc-files): it needs flatcc and an unstable DuckDB interface or a full type mapping. |
| `.ckpt` is refused as a pickle, and `.pq` reads as Parquet. | The contracts' lists lack them; a Lightning checkpoint is a torch pickle, and `.pq` is a common Parquet extension. |
| A single-tensor file refuses `tensor`, and a multi-tensor file needs it. | A missing selection on a checkpoint is more often a mistake than a wish for every tensor, and `"*"` says that wish plainly. |
| The default loader names its reader. | A saved recipe then reads the same way even when a file's bytes change. |
| Source names from file stems use `_`, not `-`. | A table name with `-` must be quoted in every query. |
| A WAV data chunk that claims more bytes than the file holds is cut to what is there, with a warning. | Recorders that stop early leave such files, and the bytes that are there are valid. |
| Loader outputs have `wire = true`. | The host can send a table's rows or a tensor's bytes to the browser viewer as a snapshot. |
| A missing source is `catalog.not_found`. | `catalog.describe` and `catalog.sample` report it, and the code should not depend on the provider. |
| The library embeds `loader_host.py` and writes it next to each run. | The app needs no install path for it, and the copy that runs always matches the code that checks its output. |
| A Python loader's output selects with `tensor` or `table`, and `"*"` makes its name a prefix. | It follows the multi-tensor files, so one vocabulary covers both kinds; `table` is the only new key. |
| The manifest also declares each output's kind, size in bytes and, for a tensor, its dtype and shape, and the host compares all of them with the files. | A huge or inconsistent declaration is refused before any file is read, and a tensor file that is not what the manifest says is caught. |
| Without pyarrow, a table is CSV with the typed header `"<name>:<type>"`, read by Caliper's own parser. | Otherwise DuckDB would guess the types, and a NULL and an empty string would look the same. |
| Checked outputs move to `loader-outputs/<id>/run-<n>/` with the manifest rewritten, and older runs are deleted after the commit. | The sources survive a restart and a rebuild without Python, a failed run leaves the previous outputs whole, and the database never points into a folder that is being replaced. |
| `rebuild()` reads kept outputs back and never runs a script. | A rebuild has no approval to give, and the kept files are the results of the last approved run. |
| A run refused for want of approval records no status. | Nothing ran, so the previous status and sources stay true. |
| The script must lie inside the workspace, and a grant does not cover it. | A grant lets a loader read data; running code from a dropped folder would widen what a grant means. |
| `reader.python` is an absolute path or a command name, and the lookup uses only the absolute entries of `PATH`. | A relative path, or a relative entry of `PATH`, would depend on the app's working folder. |
| The script's folder comes first on `sys.path`. | This is how `python script.py` behaves, so a script can import its own helper modules; the tests use it to put a stand-in torch beside the script. |
| The store's lock is released while a script runs. | A run may take up to an hour, and the catalog and queries must stay usable meanwhile. |
| A Python loader reads exactly one file. | `load(path, out)` takes one path; a glob would need one process per file and naming rules of its own. |
| The pickle check does not apply to a Python loader's input. | Only the user's script opens it, and every file that comes back is checked. |
| Every output file must have exactly one hard link. | Otherwise a later change to another file could change a source after the checks. |
| A Python source's `origin.file` is the recipe's `input.path`. | It is what tables already show, and it stays right when the sources are read back from kept outputs after the input moved. |
