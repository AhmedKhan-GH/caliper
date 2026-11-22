# Recordings

This page plans timelines and recordings: how every logged item is placed on the timelines of its session, how a recorder keeps what it receives within a budget, how the workspace stores a recording beside sprint 3's loaders, the recording file, an MCAP file that a person shares, how a recording is opened in the Caliper app, in a browser and through the hub, how it is exported for other tools, and the scrubber that moves the cursor over a recording or a live session. The messages that carry stamps, scenes and the cursor are drafted in [Wire protocol 3.2](wire-protocol-3-2.md), and the scene model is in [Scenes](scenes.md).

## What is built and what is planned

Nothing on this page is built. It rests on these built pieces:

- **Wire protocol 3.1** on the sprint 2 branch: frames of a fixed layout with 8-aligned sections, snapshots that name their data view, series sent as deltas with absolute indices, and a decoding order that every implementation applies to frames from a hostile peer.
- **Series retention.** `caliper.serve()` keeps the last 1,048,576 values of each series whether or not anyone watches, and the hub keeps as many per series view, so a late viewer draws the whole curve.
- **The hub's ended sessions.** The hub keeps an ended session with its last snapshots, series and controls, and answers a data view that nobody watched with the one watched last.
- **Sprint 3's workspace**: `workspace.duckdb` with the loader and catalog tables in the schema `caliper`, the catalog's `CatalogEntry`, and the names `recording` (a catalog kind) and `caliper_recordings` (a schema), which sprint 3 reserved for this page.
- **The metrics store** behind `caliper.metrics.v1` (`docs/applets/services/caliper-metrics-v1.md`) and the data service of `caliper.data.v1` (`docs/applets/services/caliper-data-v1.md`).
- **The web package**, whose frame codec already decodes every frame of the protocol in the browser.

The evidence for the recording file is a **structural check of MCAP**, kept outside the repository. A recorder prototype on the official MCAP C++ library 2.2.0 wrote the first 1,500 steps of a synthetic training session, 1,170 real frames of protocol 3.2 and 14.1 MB, with four series, a text log, two heatmaps, an image, points, a histogram, bars and a scene with entity transforms, clears, a class table, static items and five entity parts, on the timelines `step`, `time` and a timeline of the script's own. Its encoder reproduced the golden frames of protocols 3.0, 3.1 and 3.2 byte for byte. The files were read back with Caliper's own reader, the official C++, Python and TypeScript readers and the `mcap` command-line tool 0.4.0, on an Apple M5 Max with macOS 27.0.1 and APFS. The check answered eight structural questions and measured no timing. Where this page relies on it, it says so; everything else about the file is designed and untested.

## The recording model

### Timelines

A **timeline** is a named, ordered axis on which logged data is stamped. Every session has two built-in timelines and may declare up to six more.

| Timeline | Kind | Values | How it advances |
|---|---|---|---|
| `step` | sequence | Signed 64-bit integers | `caliper.step()` adds 1, `caliper.step(n)` sets it to n, `caliper.set_time("step", n)` sets it without a checkpoint |
| `time` | time | Signed 64-bit nanoseconds since the Unix epoch | The wall clock, read at every call. `caliper.set_time("time", t)` pins it, for example to log old data with its own times; `None` returns it to the clock |
| Up to 6 of the script's own | sequence or time | Signed 64-bit integers; a time in nanoseconds, from the epoch or from any origin | `caliper.set_time(name, value)` |

A timeline's name is 1 to 63 bytes of `[A-Za-z0-9_.-]`; `step` and `time` are taken. Timelines get the ids 0 to 7 in the order they are declared, `step` 0 and `time` 1, and are never removed during a session. A time timeline that does not count from the epoch, such as simulated time, is shown as a duration rather than a date.

### Stamps and static items

The **stamp** of an item is its position on every timeline of the session. Caliper LIB reads it at the call, on the calling thread, before any device work: the current value of `step` and of each declared timeline, and the wall clock for `time`. The current values are process-wide, so a call on a data-loading thread gets the values the training thread set last. The stamp travels with the snapshot, so a snapshot drawn a frame later still carries the moment of its call, and a batch dropped by the rate limit leaves no stamp. Values appended by one `series` call share one stamp. The monotonic `stamp_ns` of every snapshot stays as it is, for latency display.

A timeline may have no value yet: `step` before the first `caliper.step`, a timeline declared after some items were logged, or one cleared with `caliper.set_time(name, None)`. An item made then is **absent** on that timeline. It is placed on the other timelines and does not appear when that timeline is active.

A **static** item, logged with `static=True`, has no stamp. It is present at every position of every timeline, a newer static item of the same view or entity part replaces an older one, and wherever a stamped item lies at or before the cursor, the stamped item wins. Static suits a scene's setup, a class table or a header line of a text log. A series takes no `static`, because its values are appended, not replaced.

### The calls in Caliper LIB

```python
caliper.step(n=None)                                  # returns the step
caliper.timeline(name, kind="sequence", *, epoch=False)
caliper.set_time(name, value)
recorder = caliper.record(path, *, views=None, rates=None, data_views=None,
                          budget="4GiB", horizon=1024, sync=1.0, full=False)
```

| Call | Meaning |
|---|---|
| `step(n=None)` | The checkpoint of protocol 3.1, which this call already names, and the `step` timeline: without an argument it adds 1, with `n` it sets the step to `n`. It returns the step. A script that never calls it has no `step` stamps |
| `timeline(name, kind, *, epoch)` | Declares a timeline of kind `"sequence"` or `"time"`; `epoch=True` marks a time timeline that counts from the Unix epoch. Declaring a name again with the same kind does nothing; another kind, a seventh timeline or a bad name raises `ValueError` |
| `set_time(name, value)` | Sets a timeline's current value: an integer, or for a time timeline also a float in seconds or a `datetime`. `None` makes later items absent on it. An undeclared name raises `ValueError` |
| `record(path, ...)` | Starts a recorder in the script and returns it, below |
| `static=True` | A keyword of every view call except `series`: `show`, `points`, `histogram`, `bars`, `text` and every call of a scene |

None of the names `step`, `timeline`, `set_time` and `record` is taken in `python/caliper/__init__.py` on the sprint 2 branch, and protocol 3.1 already reserves `caliper.step()` for its checkpoint. The scene calls, `caliper.text` and `caliper.bars` are planned in [Scenes](scenes.md).

`record()` arguments: `path` names the recording, a path ending in `.mcap` that does not exist yet (`ValueError` for another ending, `FileExistsError` for an existing path); it is a single file and becomes a folder of the same name past the segment limit ([Segments and the manifest](#segments-and-the-manifest)). `views` is a list of view names or `fnmatch` patterns (default: every view, including views created later); `rates` a dict from view name to snapshots per second; `data_views` a dict from view name to a data view request; `budget` the recording's size budget, as bytes or a string such as `"4GiB"`; `horizon` the thinning horizon of [Thinning](#thinning); `sync` and `full` the commit and sync cadence of [Guarantees](#guarantees). The `Recorder` has `path`, `stats()` (items, bytes, segments, commits, thinning level, dropped counts), `stop()`, `closed`, and works as a context manager. `caliper.close()` and interpreter exit stop it, which commits, writes the final index and closes the file.

### Recorders

A **recorder** is a subscriber. It subscribes to the views it records, each under a data view and a rate of its own, and writes every message it receives. The producer therefore still reduces each snapshot on its device and sends only the result, and a view that neither a viewer nor a recorder subscribes to costs nothing. Unlike a viewer, a recorder makes the producer work for views that nobody is looking at, at the recording rate, which is why recording is something a person starts.

Three places can record:

| Recorder | Started by | Receives through | Writes |
|---|---|---|---|
| Caliper LIB | `caliper.record(path)` in the script | A route of its own, like `caliper.serve()`: the device-side snapshot is copied into pinned host memory without waiting, and a writer thread encodes and writes it | The recording at `path`, a `.mcap` file, or a folder of that name past the segment limit |
| The hub | `caliper serve --record <folder>` | Its producer connections: it subscribes to every view of every session at the recording rate | `<folder>/<session id>.mcap`, one recording per session |
| The Caliper app | Record in the session's toolbar, or the setting "Record sessions when they start", which is off by default | Its attachment: `NetSource` frames as received, or slots of the shared-memory session encoded as the same frames | `recordings/<id>.mcap` in the workspace, and the index in `workspace.duckdb` |

A recorded item is the frame of wire protocol 3.2 that carries it live, stored as one MCAP message: the same bytes when it arrived over a socket, and the same arrays and keys, encoded by the app, when it arrived through shared memory. The frame's `sub` is the recorder's own and means nothing in a file. A frame from a producer of protocol 3.1 carries no stamp; the recorder keeps the frame as it arrived and places it at its own receive time on `time`.

### What a recording stores

| Kind | Stored item | Arrays | Keys kept with it |
|---|---|---|---|
| Heatmap | SNAPSHOT | `data` f32 (H, W) after the data view and budget | `data_view`, `axes`, `tile`, `colormap`, `flags`, `vmin`, `vmax`, `stats`, `seq`, the stamp |
| Image | SNAPSHOT | `data` u8 (H, W, 4) | `data_view`, `axes`, the stamp |
| Points | SNAPSHOT | `data` f32 (N, D), `aux` f32 (N,) | `points_dims`, `label_count`, `colormap`, `stats` with the bounding box |
| Series | SERIES | `values` f32 (M,) and a stamp per value | `start`, `total` |
| Histogram | SNAPSHOT | `data` f32 (B,) | `stats`, the range of the bins |
| Bars | SNAPSHOT of kind 8 | `data` f32 (B,), `colors` u8 (B, 4) | `labels`, `horizontal` |
| Scene | Each entity update, batch of entity transforms, clear and class table | The arrays of each primitive after the scene's data view | `path`, `part`, the keys of the part |
| Text | TEXT | The lines' bytes, offsets, levels and stamps | `start`, `total` |
| The session | VIEWS and CONTROLS, each time they change | | The timelines, the views and their data views, the run state and every parameter's value |

Every stored item is a frame and nothing else. A scene's keyframes are not frames: they are entries of the index attachment that point to the items holding the scene's state ([Keyframes for scenes](#keyframes-for-scenes)). A recording does not store viewer events, hovers, selections or tokens, and it never holds a tensor beyond the reduced snapshot that travelled.

### What a recording can and cannot answer

A recording can show every recorded view at any position of any timeline, under latest-at, at the rate and under the data view it was recorded with; whole series curves, decimated only beyond the limits below; the text log up to any position; the state of a scene at any position; and the value of every parameter and the run state over the run, since CONTROLS is recorded too.

It cannot serve a data view that was never recorded: another layer, another head, a zoom finer than the recorded budget, or the full tensor. A viewer that asks for one gets the recorded data view of that view, and the reply names it, as the hub does today for an ended session. It holds no snapshot between two recorded ones, none that thinning removed, and fewer intermediate updates where the recorder fell behind, because a recorder obeys the same flow control as a viewer: it can miss a snapshot but never receive a wrong one. It records nothing from before it started, apart from what a new subscription receives at once: the retained series and text, and the scene state that the producer keeps. Controls cannot be replayed into a script.

## Retention and downsampling

### Rates

A recorder caps each view at a rate of its own, set per view by `rates` (Caliper LIB), `--record-rate <view>=<hz>` (the hub) or the session's recording settings (the app).

| Kind | Default rate | Default recording data view |
|---|---|---|
| Heatmap, image | 2 per second | The view's default, with `max_extent` 256 |
| Points | 2 per second | The view's default, with `max_points` 100,000 |
| Histogram, bars | 2 per second | The view's default |
| Scene | 10 per second for each entity part; entity transforms, cameras, class tables and clears at every change | The scene's default |
| Series, text, controls | Every value, every line, every change | |

At these defaults a 512 × 512 heatmap is recorded at 256 × 256, 256 KiB, twice a second: 0.5 MiB a second, or 1.8 GiB an hour before thinning.

### Budgets

| Recorder | Memory | Disk |
|---|---|---|
| Caliper LIB | A write queue of 64 MiB; beyond it, the same coalescing as a slow viewer | `budget`, default 4 GiB per recording |
| The hub | 64 MiB per recorded session | `--record-budget`, default 4 GiB per session |
| The Caliper app | 64 MiB per recorded session | 4 GiB per recording, a setting |
| A browser page's live buffer | 512 MiB by default, at most 2 GiB | None |

### Thinning

Thinning keeps recent data whole and old data at a density that halves with each doubling of its age. It works per view, and in a scene per entity part, by count rather than by time, so a view recorded at 2 Hz and one at 30 Hz are each kept at their own density.

Number the items of a view in arrival order with an ordinal n = 0, 1, 2, …, and let N be the number recorded so far. The age of item n is d = N − 1 − n. With the horizon h (default 1,024), its band is b = 0 when d < h and b = ⌊log<sub>2</sub>(d / h + 1)⌋ otherwise. Thinning keeps item n when n mod 2<sup>b</sup> = 0.

- Band b covers h · 2<sup>b</sup> ages and keeps every 2<sup>b</sup>-th ordinal, so every band holds h items, and a view keeps about h · (B + 1) items, where B is the band of its oldest item. With h = 1,024, a view of 72,000 snapshots (ten hours at 2 Hz) keeps 7,168, and one of a million keeps 10,240.
- An item's band only grows as N grows, and a multiple of 2<sup>b+1</sup> is a multiple of 2<sup>b</sup>, so thinning only ever removes items. It never needs one it removed before.
- It never removes static items, VIEWS, CONTROLS, class tables, clears, entity transforms, series or text values, or the first item of a view or entity part.

A browser page applies thinning continuously to its live buffer, against its memory budget. On disk nothing is thinned until a recording reaches its budget. Then the recorder compacts closed segments, oldest first: each compaction rewrites one segment without its thinned items and renames it over the old one, while appends continue into the open segment ([Segments and the manifest](#segments-and-the-manifest)). A compaction reads and writes at most one segment, 1 GiB, on a thread beside the writer. The structural check's prototype holds compaction, but the check did not test it.

### Series and text

Series values are kept whole up to 4,194,304 values per series, 80 MiB with two timelines. Beyond that the oldest values are decimated in buckets of 64: each bucket keeps its smallest and its largest value, in their original order and with their own stamps and absolute indices, so the curve's envelope survives at a 32nd of the size. Text is kept whole up to 1,048,576 lines or 256 MiB per text view; beyond that the oldest lines go, and the recording keeps their count, so the log shows how many earlier lines were dropped.

### Keyframes for scenes

Each entity part of a scene changes on its own, so the state at a cursor is the newest item of every entity part at or before it, and a static floor mesh logged at the start may lie a whole recording away. A recorder therefore adds a **keyframe** for each scene to the index every 2 seconds of `time` or every 512 items of that scene, whichever comes first. A keyframe is an entry of the index attachment, not a frame, and copies no payload: it lists the items that make up the scene's state at that moment, the newest update of every entity part not cleared since, the newest entity transforms of each entity and the class table, each by its channel and its ordinal on that channel. Latest-at at a cursor reads the newest keyframe at or before it and at most 512 index entries after it, then fetches only the newest frame of each entity part.

Keyframes follow arrival order. They answer latest-at on a timeline whose stamps never decrease in arrival order and are never absent within the scene, which the reader checks per scene and per timeline; this holds for `time` and for a `step` that only grows. On another timeline latest-at scans the scene's index entries, as the structural check's queries on its own timeline did.

### When the budget is reached

When a recording reaches its budget, the recorder acts in this order and records each step in its status:

1. Compact closed segments, oldest first, with thinning at the current horizon.
2. If the recording is still above three quarters of its budget, halve the horizon, down to 16, at each later compaction.
3. Decimate series beyond their whole limit and drop text beyond its limit, as above.
4. Drop the oldest band of snapshots and scene updates entirely. The recording's start moves later, and the scrubber hatches the part that is gone.
5. If series, text and the never-thinned items alone exceed the budget, stop recording snapshots and scene updates, keep series and text, and say so once in the status.

No recorder ever deletes a recording as a whole.

## Storage in the workspace

### Payloads in files, the index in the database

```text
<workspace>/
  recordings/<recording-id>.mcap     the recording, a file or a folder of segments: the source of truth
  workspace.duckdb                   schema caliper_recordings: index, series values, text lines
```

A recording's id follows the workspace's id grammar, `[a-z][a-z0-9_-]{0,63}`, and is made from the script's name and the start time, such as `train-20261115-120314`. As for loaders, the files are the source of truth and the tables can be rebuilt from them: opening a workspace whose `caliper_recordings` tables are missing, or whose `bytes`, `segments` or `generation` for a recording differ from its files, imports that recording's index again. While the app records, it adds the rows of each commit in one transaction, so its tables always end at a commit.

Payloads stay in the files, not in DuckDB. A frame is read in place from a mapping of its segment or fetched with one range request, which scrubbing ten views at 60 frames a second needs; the file in the workspace is the file a person shares, so nothing is exported to share it; and the hub, which has no DuckDB, writes the same thing. What SQL is good at goes into the database: one row per item, and the values of series and text.

### The schema `caliper_recordings`

Stamps are BIGINT columns: `step`, `time` (nanoseconds since the epoch), and `t2` to `t7` for the session's own timelines, NULL where an item is absent on a timeline. Times of the workspace's own bookkeeping are ISO 8601 text, as in sprint 3's loader tables.

| Table | Columns |
|---|---|
| `recordings` | `id` VARCHAR primary key; `title` VARCHAR; `file` VARCHAR (the `.mcap` file or folder, relative to the workspace, or the absolute path of a granted one); `bytes` BIGINT; `segments` INTEGER; `generation` BIGINT (the manifest's, 0 for a single file); `items` BIGINT; `uid` VARCHAR (the recording's `uid`); `session_id`, `script` VARCHAR; `recorder` VARCHAR (`lib`, `hub`, `app`); `started_at`, `ended_at`, `imported_at` VARCHAR; `status` VARCHAR (`recording`, `complete`, `recovered`, `error`); `error` VARCHAR |
| `timelines` | `recording` VARCHAR; `timeline` SMALLINT (0 to 7); `name` VARCHAR; `kind` VARCHAR; `epoch` BOOLEAN; `min_value`, `max_value` BIGINT; `monotonic` BOOLEAN |
| `views` | `recording` VARCHAR; `view` INTEGER (the id of VIEWS); `name` VARCHAR; `kind` VARCHAR; `data_views` VARCHAR (the recorded resolved data views as JSON); `items`, `bytes` BIGINT; `thinned` BIGINT |
| `items` | `recording` VARCHAR; `item` BIGINT; `view` INTEGER; `channel` INTEGER; `type` INTEGER (the message type); `entity` INTEGER; `flags` INTEGER; `step`, `time`, `t2` … `t7` BIGINT; `segment` INTEGER; `chunk_offset` BIGINT (the chunk record's offset in its segment); `message_offset` INTEGER (the frame's offset in the chunk's uncompressed records); `frame_len` INTEGER |
| `entities` | `recording` VARCHAR; `view` INTEGER; `entity` INTEGER; `path` VARCHAR; `parts` VARCHAR (JSON list) |
| `series_values` | `recording` VARCHAR; `view` INTEGER; `idx` BIGINT; `value` FLOAT; `step`, `time`, `t2` … `t7` BIGINT; `decimated` BOOLEAN |
| `text_lines` | `recording` VARCHAR; `view` INTEGER; `idx` BIGINT; `level` VARCHAR; `text` VARCHAR; `step`, `time`, `t2` … `t7` BIGINT |
| `controls` | `recording` VARCHAR; `item` BIGINT; `step`, `time` BIGINT; `run_state` VARCHAR; `params` VARCHAR (JSON) |

For each recorded series and text view, the import also creates a view named `"<recording-id>/<view name>"` in the schema, with the session's own timelines under their names and `time` as a TIMESTAMP_NS through `make_timestamp_ns` (checked on DuckDB 2.0 from PyPI, not yet on the host's 1.5.6):

```sql
SELECT step, value FROM caliper_recordings."train-20261115-120314/loss" ORDER BY step
```

This is how SQL reaches a recording: a spec source's `sql` names that view like any table, under sprint 3's rules for spec SQL, which run one read-only `SELECT` with a row limit and a time limit.

### The catalog kind `recording`

The recording provider lists one entry per recorded view, named `<recording-id>/<view name>`, and one per recording, named `<recording-id>`, whose rows are the recording's index. The entry follows sprint 3's `CatalogEntry`; its shape follows the view's kind:

| Recorded view | Shape | `columns` or `dtype` and `shape` |
|---|---|---|
| Series | Tabular | `step` BIGINT, `time` TIMESTAMP_NS, the session's own timelines, `value` FLOAT |
| Text | Tabular | The stamps, `level` VARCHAR, `text` VARCHAR |
| Scene | Tabular | Its entity index: `path`, `part` and the stamps. The scene itself is drawn by the spec primitive `scene` |
| Heatmap, image, points, histogram, bars | Tensor | The dtype and shape of the newest recorded snapshot. A panel bound to it draws the snapshot at the cursor |

```json
{"id": "recording:train-20261115-120314/loss", "kind": "recording", "name": "train-20261115-120314/loss",
 "title": "train.py loss, 15 Nov 12:03",
 "origin": {"recording": "train-20261115-120314", "file": "recordings/train-20261115-120314.mcap",
            "session": "clp-4242-a1b2c3", "view": "loss", "view_type": "series"},
 "columns": [{"name": "step", "type": "BIGINT"}, {"name": "time", "type": "TIMESTAMP_NS"}, {"name": "value", "type": "FLOAT"}],
 "rows": 20000, "dtype": null, "shape": null, "rate_hz": 200.0, "bytes": 400000,
 "wire": true, "warnings": [], "updated_at": "2026-11-15T12:05:00Z"}
```

`recording` is a new key of `origin`, which the catalog page on sprint 3 allows a provider to add. `rate_hz` is the recorded items per second over the recording, and `warnings` say where thinning or decimation began and whether a segment was recovered after a crash. `wire` is true, because the recorded frames are the wire form.

### The metrics store

The metrics store and recordings stay separate and meet in the catalog. The metrics store (`metrics.duckdb` in the app-data folder) is written by applets through `caliper.metrics.v1`, with a run and a tag per scalar; a recording is written by a recorder of a session and holds every kind of view. Neither copies into the other. Both appear as catalog sources, `metric:` and `recording:`, and a spec can show a metric and a recorded series side by side. A spec's SQL runs on `workspace.duckdb` and cannot join the metrics store, which it reaches through its catalog entries. The run history of [Data and Cloud](data-and-cloud.md) lists both.

## The recording file

### Why MCAP

The recording file is MCAP, an open container format for timestamped messages, with the extension `.mcap`; the user chose it. MCAP has official readers in C++, Python and TypeScript and a command-line tool, `mcap`, that lists, checks and repairs files. Caliper stores its own wire frames in it, so a recorded item is still the live message, byte for byte, and the app, the hub, Caliper LIB and the browser decode a recording with the codec they already use for the live protocol. The cost is that generic MCAP tools list Caliper's channels and time its messages but cannot draw them, because the message encoding is Caliper's own. An earlier comparison of containers was made; the choice of MCAP is the user's and does not rest on it.

### The file

Every file Caliper writes is a complete MCAP file. Its header names the profile `caliper` and, as its library, Caliper's recorder and version. Its messages lie on channels, one per view:

| Channel | Topic | Holds | Channel metadata |
|---|---|---|---|
| One per heatmap, image, points, series, histogram, bars or text view | The view's name | Its SNAPSHOT, SERIES or TEXT frames | `caliper.role` `view`, `caliper.kind` the kind, `caliper.view` the name |
| One per scene, for the scene's own messages | The scene's name | Its SCENE_TRANSFORMS, SCENE_CLEAR and SCENE_CLASSES frames | `caliper.role` `scene_control`, `caliper.kind` `scene`, `caliper.view` |
| One per entity part of a scene | `<scene>/<entity path>#<part>`, such as `world/robot/arm#mesh` | That part's SCENE_UPDATE frames | `caliper.role` `scene_part`, `caliper.kind` `scene`, `caliper.view`, `caliper.path`, `caliper.part`, `caliper.scene_channel` (the id of the scene's channel) |
| The session's views | `caliper/views` | VIEWS | `caliper.role` `session`, `caliper.kind` `session` |
| The session's controls | `caliper/controls` | CONTROLS | As for views |

A channel is created when its first item arrives and lasts for the whole recording. Topics are for people and generic tools: Caliper's reader takes the view, path and part from the channel metadata and never parses a topic, and since MCAP allows several channels with one topic, a view that happens to be named `caliper/views` collides with nothing. In the structural check, twelve views, among them a scene with five entity parts, made 19 channels.

**Encoding and schemas.** Every channel has the message encoding `caliper.wire3`. Each kind has one schema record, `caliper.wire3.<kind>` for `heatmap`, `image`, `points`, `series`, `histogram`, `bars`, `scene`, `text` and `session`, with the schema encoding `caliper.wire3` and a short JSON text that names the protocol version and the frame layout of [Wire protocol 3.2](wire-protocol-3-2.md). A message's data is exactly one frame of protocol 3.2, from its magic `CALW` to its last section, and nothing else: no wrapper, header or padding. The structural check showed that 1,170 frames written this way came back from the unpatched official C++ reader and from the official Python reader with the same SHA-256 over every message, and that the file carried the profile `caliper`, the encoding `caliper.wire3` on all 19 channels and 9 schema records.

| Message field | Value |
|---|---|
| `log_time` | The item's `time` stamp, in nanoseconds since the Unix epoch; for a frame with a stamp per value, such as SERIES, the stamp of its last value. A static item gets the recording's start time. An item without a stamp gets the newest `time` the recorder has seen, or the recorder's receive time from a producer of protocol 3.1 |
| `publish_time` | When the recorder received the frame |
| `sequence` | The item's ordinal on its channel, from 0, which the index keeps too |

**Metadata.** Each file holds, after its header and its schema and channel preamble, one metadata record named `caliper.segment`:

| Key | Value |
|---|---|
| `uid` | 128 random bits in hex that identify the recording across copies and segments |
| `segment`, `first_item` | The segment's number, 0 for the first segment and for a single-file recording, and the arrival number of its first item in the recording |
| `started_unix_ns` | When the recording started |
| `session`, `recorder`, `title` | The session's identity; `lib`, `hub` or `app` with the software's version; the recording's title |
| `wire_protocol` | `3.2` |
| `timelines`, `views` | JSON: the timelines in id order with name, kind and `epoch`; the newest VIEWS metadata when the segment opened, so that a segment alone knows its views |
| `settings` | JSON: the rates, budget, horizon and sync setting |

A segment that a roll-over closes also carries, before its summary, a metadata record `caliper.segment_end` that names the next segment's file, so that a reader following the recording knows to go on; the last segment has none (designed and untested). The index is in attachments, below.

### The timeline and keyframe index

MCAP indexes messages only by `log_time`, per chunk and per channel. Caliper's index adds what a scrubber needs beyond that: every item's stamp on every timeline, where its frame lies, the scenes' entity paths and keyframes. It is stored in attachments of the media type `application/vnd.caliper.index`:

- `caliper.index` holds the whole segment's index and is written when the segment closes, just before its summary.
- `caliper.index.delta` holds the entries since the previous delta and is written at every commit point. A tailing reader learns of new items from the deltas, and recovery rebuilds a segment's index from them.

An attachment's data is the magic `CALIDXZ1`, the raw length as a u64, and the raw index compressed with zstd. The raw index starts with the magic `CALIDX01` and a JSON header, followed by columns, each padded to 8 bytes; sorted columns are delta-coded, and every column is byte-shuffled before compression.

| Part | Content |
|---|---|
| Header | JSON: version 1; whether it is a final index or a delta; the segment; the timelines; the recording's `uid`; the first entry, chunk and entity it covers and their counts; the definitions of the channels it adds |
| Chunks | The file offset of every chunk record it covers |
| Entries, one row per item | `item`, the arrival number in the recording; `chunk`, its chunk within the segment; `offset`, the frame's offset in the chunk's uncompressed records; `len`; `channel`; `type`, the message type; `flags` (static, state burst, reset, recursive, unstamped); `ordinal`, the MCAP `sequence`; and for scenes `entity` and the parts mask of a clear |
| Stamps | One i64 column per timeline, −2<sup>63</sup> where the item is absent |
| Entities and entity transforms | The entity paths of each scene with recording-wide ids, and for each SCENE_TRANSFORMS the entities it moves |
| Keyframes | For each: its scene, the item after which it holds, its stamp on every timeline, and its items, each a key (an entity part's channel, an entity's transforms or the class table) with an ordinal |

In the structural check the final index of 1,170 items took 5,852 bytes.

**Latest-at.** On `step` and on the script's own timelines, latest-at goes through the index: the reader keeps each channel's present entries sorted by stamp and arrival and finds the newest at or before the cursor by binary search, with the newest static item as the fallback. On `time`, MCAP's own index answers as well, because `log_time` is the `time` stamp: generic tools seek on it, and Caliper's readers use it where the attachment is missing or stale and a scan would cost too much, as over HTTP (designed and untested); otherwise they use the attachment's `time` column, so that every timeline follows one rule. The structural check compared 1,000 random latest-at queries with a brute force on each of `step`, `time` and a timeline of the script's own that is absent before step 1,000, and all agreed.

**Scene state.** The state of a scene at a cursor comes from the newest keyframe at or before it and at most 512 entries after it ([Keyframes for scenes](#keyframes-for-scenes)). A keyframe names each item by channel and ordinal, and the reader takes the newest kept item at or before that ordinal, so thinning never invalidates a keyframe. The structural check compared 200 scene-state queries per timeline with a brute force, and all agreed: on `step` and `time` through keyframes, and on the script's timeline by a scan of the scene's entries, because that timeline is absent before step 1,000.

**Validation and rebuilding.** A reader uses an index only when it describes the file: a final index's chunk offsets must equal the chunk index in the file's summary, and a delta's chunks must be the file's next chunks. Otherwise, or when there is no index, the reader rebuilds the index by scanning the chunks and reading each frame's metadata, and the app keeps the rebuilt index in `caliper_recordings` so that it scans once. The structural check found this rule necessary: after `mcap recover` rewrote a file, the attachment survived but described chunks that had moved, and the prototype's reader, which first trusted it, read nothing correctly; with the rule it rebuilt the index and read 1,170 of 1,170 frames byte-identical. A file written with no attachment opened by scanning and answered every query alike, in 4.4 ms instead of 0.35 ms, times from one run each that the check gives as context, not as a measurement.

### Alignment

A no-copy read needs a frame's arrays at aligned addresses. A wire frame's sections are 8-aligned from the frame's start, but MCAP places a message wherever the records before it end. Before each frame of 64 KiB or more, which goes raw and alone into a chunk of its own ([Compression](#compression)), the writer therefore writes a private record with the opcode `0xCA`, filled with zeros and sized from the file offset, so that the frame's largest section, its largest array, starts on a 16 KiB boundary in the file. 16 KiB is the page size of Apple silicon, and a 16 KiB boundary is also 8- and 16-aligned. A frame that already lands aligned needs no record; otherwise the record takes from 9 bytes, its opcode and length, to 16 KiB plus 8.

The padding is a record of its own, not padding inside the frame, so that the recorded frame stays exactly the live frame. A receiver would accept extra spaces in a frame's metadata JSON, which could align the sections as well, but the recorded bytes would then differ from the live ones.

A no-copy read is then a pointer into a mapping of the file. A mapping starts on a page boundary, so the array's address is 16 KiB-aligned, which typed views, vector loops and a no-copy GPU buffer over the mapping need; Metal's no-copy buffers take page-aligned memory. In the browser the reader fetches exactly the frame's bytes, and its sections, 8-aligned from the frame's start, become typed-array views over the fetched buffer without a copy.

The structural check wrote a 16 MiB tensor whose bytes started at file offset 81,920, 5 × 16,384, after 3,591 bytes of padding in such a record, at an address that was 0 modulo 8, 16 and 16,384 in the mapping, and the official reader returned the same bytes. In its training-session file, all 82 frames of 64 KiB or more were 16-aligned with the prototype's default of 16 bytes; 16 KiB for every large frame is designed and was checked on the one tensor. The padding averages 8 KiB per large frame, about 3 % of a 256 KiB heatmap and up to a quarter of a 64 KiB frame ([Open decisions](#open-decisions)).

Standard tools tolerate the record. The structural check showed that the official C++ and Python readers skip it, that `mcap doctor` warns once per record that the opcode is unknown and still exits with 0, and that `mcap recover` drops it. The mixed file of the check's browser test holds 23 such records, and the official TypeScript reader opened it and returned the expected frames.

### Compression

Small frames go together into chunks compressed with zstd: series, text, entity transforms, controls and small snapshots. Such a chunk is closed when it holds 1 MiB of records, at every commit point, and before each large frame. A frame of 64 KiB or more goes raw and alone into a chunk of its own, after its padding record, so that it is read in place and costs the recorder no compression. MCAP chooses the compression per chunk, so a reader that follows the specification reads such a file; the structural check showed a file of 46 zstd chunks and 101 raw ones read by the unpatched official C++ reader, the Python reader and `mcap info`, with all 82 large frames read in place.

**The writer.** The official C++ writer fixes one compression when it opens a file, and its class is final with private members, so mixed chunks need a change to it. The two ways are a patch that adds `setCompression()`, which closes the open chunk and switches the compression of the following chunks, 50 added lines in the structural check, or Caliper's own chunk writer over the library's record types and serializers, which would re-implement the writer's chunks, message indexes, statistics and summary. The design takes the patch, because the structural check showed that it works and that the official readers read its files, and because it keeps the official writer, with its message indexes, statistics and summary, for everything else. The patch also gains a small accessor that reports where the last message landed in its chunk, about 15 lines, designed and untested: the official writer does not report it, and the prototype had to mirror the library's record sizes to know, which ties the code to one version. Caliper vendors the header-only library at version 2.2.0 with the patch applied, builds it with zstd and without lz4, and offers both changes upstream ([Open decisions](#open-decisions)). Caliper's own code writes only the padding records, the schema and channel preamble of each segment and, in recovery, a summary. In the browser, where not every current browser decodes zstd through `DecompressionStream`, the web package bundles a small decoder in JavaScript, `fzstd`, as the check's browser test did ([Opening a recording](#opening-a-recording)).

### Segments and the manifest

**What a recording is on disk.** A recording is one path ending in `.mcap`. While it is smaller than the segment limit, it is a single MCAP file at that path. When it reaches the limit, the recorder turns the path into a folder of the same name, which holds the file as its first segment, the segments after it and a manifest. A short run is therefore one file that a person sends as it is, and a long one a folder that they zip; `caliper.record`, `caliper open`, the catalog and the hub name both by the same path.

| | A short recording | A long recording |
|---|---|---|
| On disk | `run.mcap`, one MCAP file | `run.mcap/`, a folder with `manifest.json`, `seg-00000.mcap`, `seg-00001.mcap` and so on, and `LOCK` |
| What a person shares | The file | The folder, zipped; any segment alone also opens in generic tools |
| What `caliper open` takes | `run.mcap` | `run.mcap` or its `manifest.json`; a segment alone opens as its part of the recording |
| The writer's lock | An exclusive lock on the file itself | An exclusive lock on `LOCK` |
| In a browser by URL | The file's URL | The manifest's URL |

**The limit.** The segment limit is 1 GiB, or a quarter of the budget when that is smaller, so that a recording at its budget has at least four segments to compact. 1 GiB bounds one compaction to 1 GiB of reading and writing, keeps a segment below the 4 GiB file limit of FAT32 storage, and keeps a run that records 0.5 MiB a second, the default heatmap's rate, one file for about half an hour. It is fixed in Caliper LIB, the hub and the app; the C++ recorder takes it as an option for its tests. A single-file recording is never compacted, since it rolls over before it reaches its budget.

**The roll-over** is designed and untested, because the check's prototype always wrote a folder. At the first commit point past the limit, the recorder:

1. Writes the file's final index, its `caliper.segment_end` record, summary and footer, and syncs it.
2. Creates the folder `run.mcap.tmp`, writes in it the manifest that lists `seg-00000.mcap` as closed, syncs it, moves the file into it under that name and syncs the folder.
3. Renames `run.mcap.tmp` to `run.mcap` and syncs the parent folder.
4. Opens `seg-00001.mcap` and goes on.

A recorder or reader that finds a `run.mcap.tmp` folder completes the roll-over when the folder holds a manifest, moving the file in if it is still outside, and otherwise removes the folder.

**Each segment is a valid MCAP file on its own.** It starts with its header, then every schema and channel record known so far, then its `caliper.segment` metadata, and ends with its own index attachment, summary and footer. The preamble is necessary. The structural check found that the official writer, reused across files, lists every channel it was ever given in each file's summary, so that later segments failed `mcap doctor` with "Channel with ID n in summary section does not exist in data section"; with every schema and channel record written at the start of each segment, which the specification allows, `doctor` accepted every segment. The check split a recording at a 4 MiB limit into four segments, each of which both official readers read alone and `doctor` passed with exit 0, and which gave the writer's digest when read in the manifest's order.

**The manifest**, `manifest.json`:

| Key | Value |
|---|---|
| `format`, `version` | `caliper.recording`, 1 |
| `profile`, `uid`, `created_unix_ns` | `caliper`, the recording's `uid`, its start |
| `generation` | A number that grows with every replacement |
| `config` | The recorder's settings |
| `segments` | In order, for each: `file`, `state` (`open` or `closed`), `bytes`, `first_item`, `items`, `recovered`, `compactions` |

The manifest is replaced as a whole: written to `manifest.json.tmp`, synced, renamed over `manifest.json`, and the folder synced. The recorder replaces it when a segment opens or closes, after a recovery and after a compaction. A reader that finds segment files the manifest does not list, as after a crash during a roll-over, opens them too, in the order of their names.

**Thinning and compaction** rewrite one closed segment at a time. The recorder writes the kept items, with their item numbers, ordinals and stamps unchanged and with a new index, into `seg-NNNNN.mcap.tmp`, syncs it, renames it over the segment, syncs the folder and replaces the manifest with the segment's new size and compaction count. A reader that has the old segment mapped keeps reading it until it sees the manifest's new generation.

**Opening through the manifest.** A reader opens a folder through its manifest: it maps each segment, loads each segment's index (the final one, else the deltas, else a scan of what neither covers), and continues each segment's item numbers from its `first_item`. A reader given one segment reads its `caliper.segment` metadata and says which part of which recording it shows.

### Guarantees

A recording keeps four guarantees, sized for runs of days.

- **Atomic.** Items join a recording at commit points. A commit closes the open chunk, syncs as `sync` asks, and then writes a delta of the index listing the chunks before it, so a reader sees all of a commit or none of it. Every MCAP record carries its length and every chunk a CRC of its records, so a torn record is recognised, and the torn tail is cut when the recording is next opened by a writer or recovered. The manifest and a compacted segment are each replaced by one rename.
- **Consistent.** No data record is rewritten in place: a writer only appends, recovery cuts a torn tail and appends, and compaction writes a new file. The files are the truth: a reader uses an index only after checking it against the file, and the app's tables in `workspace.duckdb`, written in one transaction per commit, are rebuilt from the files whenever they disagree.
- **Isolated.** One writer per recording, through an exclusive lock: `flock` on the file of a single-file recording or on `LOCK` in a folder, and on Windows a file opened without write sharing. A second writer fails at once with `caliper: recording is locked by another writer: <path>`. Tailing readers, such as the app following what Caliper LIB writes, the hub serving a session it records and a browser page reading a growing file, never read past the newest commit point, which they learn from the deltas.
- **Durable.** A commit survives what its sync covers, as below.

**The sync cadence** is set by `caliper.record(..., sync=..., full=...)`, by `--record-sync` and `--record-full-sync` on the hub, and by a setting of the app:

| `sync=` | Commits | Syncs |
|---|---|---|
| A number of seconds from 0.1 to 3,600; the default is 1.0 | Every that many seconds, and at a roll-over and at the close | At every commit |
| `"commit"` | At every chunk close, so at least once a second, at every 1 MiB of small frames and before and after each large frame | At every commit; chunks closed while a sync runs share the next one |
| `None` | Every second, so that tailing readers keep up | Never; the system writes the data back when it chooses |

`full=False` is the default. With `full=True`, each sync on macOS is `fcntl(F_FULLFSYNC)` instead of `fsync`. On Windows a sync is `FlushFileBuffers`, which already asks the drive to write its cache, so `full` changes nothing there.

What each failure can lose:

| Failure | `sync=None` | `sync=1.0`, the default | `sync="commit"` |
|---|---|---|---|
| A crash of the script or of the recorder's process | The open chunk and the queue: small frames since the last commit, about a second, and what waited in the queue. What was written survives in the system's cache | The same | The open chunk and the queue |
| A kernel panic | Whatever the system had not written back, possibly far behind the last commit; recovery cuts at the first damaged record | The items since the last commit, about a second, and the queue: a finished `fsync` has handed the data to the drive, which keeps its power | The open chunk and the queue |
| A power cut on macOS with `full=False` | As for a kernel panic | Not bounded: `fsync` does not make the drive write its cache, which it may write later and out of order, so data before the last commit can be lost; recovery cuts at the first damaged record | As for the default |
| A power cut on macOS with `full=True`, or on Windows | As for a kernel panic | The items since the last commit and the queue | The open chunk and the queue |

When the hub or the app records, a crash of the script loses nothing they received: the recording ends there and the recorder closes it. On a full disk the recorder stops writing, leaves the file for recovery and reports the error in its status (designed and untested). **The evidence is limited.** The structural check killed writers with `kill -9` after 150, 520 and 930 frames; they recovered 138, 514 and 903 frames, every frame of their complete chunks and every committed one, all byte-identical. A fourth writer, killed after 700 frames with 5,000 bytes then torn off its end, was cut at its last complete record and kept 668 frames, two fewer than it had committed, because it ran without a sync, which is all that `sync=None` promises. Power cuts, kernel panics, a full disk and the cost of syncing during a training loop were **not** tested, because the user chose a structural check. The failure table is therefore reasoned from the platforms' documented behaviour: macOS documents that `fsync` moves data to the drive without making the drive write it to permanent storage, and offers `F_FULLFSYNC` to ask for that; Windows documents that `FlushFileBuffers` writes a file's buffers and flushes the device. The cost of syncing is measured when the recorder is built, against the plan's budget of 3 % of a training step. The writer runs on its own thread, fed by a bounded queue of 64 MiB that never blocks the producer: when the queue is full, a newer snapshot of a view replaces its pending one, as for a slow viewer, and anything else that does not fit is dropped and counted in `stats()`.

### Recovery and the standard tools

Caliper repairs its own files and never uses `mcap recover`. When a recorder resuming a recording, the app or `caliper open` finds a segment without a valid footer and can take the writer's lock, which shows that its writer is gone, it walks the records from the start and cuts the file after the last complete and valid record, whose chunk CRC matches; regenerates the message indexes of the last chunk when they were torn; rebuilds the Caliper index from the deltas and a scan of the chunks after the last delta and appends it as `caliper.index`; appends the summary and the footer; and marks the segment recovered in the manifest. It never rewrites a data record, so alignment and mixed compression survive. The structural check's recovered files, from every crash run, passed `mcap doctor` with exit 0. A reader that cannot take the lock, such as a tailing reader, the hub serving a folder or a browser page, reads up to the last complete chunk and changes nothing.

What the `mcap` command-line tool does with Caliper's files, from the structural check:

| Command | Result |
|---|---|
| `info` | Works: the library, the profile `caliper`, the message count, start, end and duration from `log_time`, the chunks per compression, the channels with counts, rates and schema names, the attachments and metadata |
| `list channels`, `schemas`, `attachments`, `metadata`, `chunks` | Work, showing the channel metadata and the schemas' JSON |
| `cat` | One line per message with its log time, topic, schema and a preview of its bytes, which start with `CALW`. `--format ndjson` refuses the custom encoding |
| `doctor` | Exits with 0 and warns that the profile is not well known, that log times decrease, because static items lie at the start, and that opcode `0xCA` is unknown, once per padding record |
| `recover` | Rewrites the whole file. It keeps every message's bytes, the metadata and the index attachment, which no longer matches the file, writes every chunk with zstd and drops the padding, so mixed compression and alignment are lost. Caliper's reader then rebuilds the index by scanning |
| `compress`, `decompress`, `filter`, `merge`, `sort` | Not run. They rewrite chunks as `recover` does, so the same losses are expected |

A generic MCAP viewer, judged from the format's rules, since the check launched none, can open a segment, list its channels, schemas, attachments and metadata, and show message times and per-channel rates on `log_time`, which is the `time` stamp. It cannot draw Caliper's frames without a decoder for `caliper.wire3`, seek on `step` or the script's timelines, or follow a manifest, and it shows static items at the recording's start.

### Standard MCAP and Caliper's additions

| Addition | Within the MCAP specification | What a standard reader does |
|---|---|---|
| The message encoding `caliper.wire3` and schemas `caliper.wire3.<kind>` | Yes; any encoding is allowed, and the registry lists only well-known ones | Reads the messages as opaque bytes and lists channels and schemas; `cat --format ndjson` refuses them |
| The index attachments | Yes; attachments may hold any data | Ignores them, and `list attachments` shows them. Tools that rewrite the file keep them, no longer matching it |
| The metadata records | Yes | Ignores them, and `list metadata` shows them |
| The profile `caliper` | Yes; the profile is a free string | Ignores it; `doctor` warns that it is not well known |
| The padding record, opcode `0xCA` | Yes; opcodes 0x80 to 0xFF are private | Skips it; `doctor` warns; rewriting tools drop it |
| Segments and the manifest | No; each segment is a complete MCAP file, and the manifest is Caliper's own JSON | Reads each segment as a file of its own and knows nothing of the manifest |
| Mixed compression | Yes; compression is chosen per chunk | Reads it; rewriting tools turn it into one compression |

**Standard payloads, a decision not taken.** The messages could instead be Protobuf or FlatBuffers under the schemas published for robotics, and generic tools could then draw some views. The cost is recalled from an earlier measurement that was lost and not repeated: dense tensors stayed within a few percent of the size of Caliper's frames, but Protobuf always copies on decode, so nothing is read in place, and FlatBuffers reads in place with unaligned arrays; scene schemas took about 2 to 3 times the bytes; and every standard schema drops `step`, the script's timelines, data views and axes, which Caliper would have to carry beside them. A recorded item would no longer be the live message, and every reader would need a second codec.

### Exporting to Parquet

`caliper export` writes a recording as Parquet tables for analysis in other tools. It is a subcommand of Caliper CLI (`docs/reference/cli.md`) and, like the other subcommands except `dev` and `open`, never opens a window.

```text
usage: caliper export <recording> --to parquet [--out <folder>] [--view <pattern>]... [--arrays]
```

| File | Rows |
|---|---|
| `items.parquet` | One per recorded item: `item`, `view`, `kind`, `type`, for scenes `entity` (the path) and `part`, `static`, the stamps (`step`, `time` as TIMESTAMP_NS and the session's own timelines by name), `bytes`, and the `dtype` and `shape` of the item's largest array |
| `series.parquet` | One per series value: `view`, `idx`, `value`, the stamps, `decimated` |
| `text.parquet` | One per text line: `view`, `idx`, `level`, `text`, the stamps |
| `controls.parquet` | One per recorded CONTROLS: `item`, the stamps, `run_state`, `params` as JSON |
| `arrays/<view>.parquet` | With `--arrays`, one per snapshot of a heatmap, image, points, histogram or bars view: `item`, the stamps, and for each array a list column of its values, flattened, with its shape in a column `<array>_shape` |

| Option | Meaning |
|---|---|
| `--to parquet` | The format. Required; Parquet is the only one |
| `--out <folder>` | Where the tables go. It must not exist or must be empty. The default is the recording's path without `.mcap`, followed by `-parquet` |
| `--view <pattern>` | Only views whose names match the `fnmatch` pattern. It may be given more than once; the default is every view |
| `--arrays` | Also write the arrays of snapshots, as above. Scene arrays are not exported in this sprint; a scene's items are in `items.parquet` |

Each option takes its value as the next argument or after `=`, and each but `--view` may be given once. Export reads the recording through the C++ reader and changes nothing in it: a recording still being written is exported up to its last commit, and one with a torn tail up to its last complete chunk, with a line that says so. It writes into a hidden temporary folder beside `<folder>` and renames it into place at the end, so a failed export leaves nothing behind. The tables go through the app's DuckDB, an in-memory database filled with its appender and written with DuckDB's Parquet writer (`COPY ... (FORMAT parquet, COMPRESSION zstd)`), so export adds no dependency. It ends with one line, such as `exported 20,412 items of 14 views from runs/train.mcap into runs/train-parquet`. `export` exits with 0 when it has written the tables, with 2 for a wrong command line, including a missing `--to` or another format (`caliper: export writes only parquet, not <format>`), and with 1 when `<recording>` is not a recording (`caliper: not a recording: <path>`), when `<folder>` exists and is not empty (`caliper: <folder> exists and is not empty`), when no view matches (`caliper: no view matches <pattern>`), or when a table cannot be written (`caliper: cannot write <file>: <reason>`). It takes 2 builder days within the 25 of recordings. It needs the DuckDB that sprint 3 links into the executable; if sprint 3 is not merged when increment 7 is built, export moves to increment 9 with the workspace tables.

### Limits of the format

- **MCAP's version.** The specification's major version is 0, and it promises compatibility only within a major version. A later major version may need a converter.
- **One writer and one native index.** MCAP has no concurrent writers, and Caliper enforces one writer per recording. MCAP indexes only `log_time`. Every other timeline and the state of scenes need Caliper's index, which standard tools neither read nor keep current.
- **Sizes.** A frame is at most 64 MiB, as on the wire; a chunk of small frames holds 1 MiB of records; a segment ends at the first commit past its limit. A recording has at most 65,535 channels, since MCAP's channel ids are 16 bits, after which the recorder records no new entity part and says so; at most 8 timelines; and at most 2<sup>32</sup> items per channel, since `sequence` is 32 bits.
- **Compression.** Caliper reads raw and zstd chunks. A file that a tool rewrote with lz4 is refused with a message that names the compression.

A reader checks a hostile file before it allocates or maps anything:

| What | Check |
|---|---|
| The file | The MCAP magic at the start, and at the end of a closed file; a header with the profile `caliper`. Another profile is not a recording |
| Every record | Its length within the bytes left; an unknown opcode skipped by its length; a padding record of zeros only |
| Chunks | Compression none or zstd; compressed and uncompressed sizes at most 80 MiB; decompression stops at the uncompressed size and fails when the stream ends early or leaves bytes over; the CRC checked when present |
| Messages | Inside their chunk, on a known channel; the data decoded with the protocol's checks 1 to 9 |
| Summary | Every offset inside the file; chunk index entries in order and not overlapping |
| Index attachments | The attachment's CRC; a raw length at most 64 times the compressed one and within the reader's memory budget; every column of its declared length; at most 8 timelines; every chunk, channel, entity and keyframe reference inside its table; the chunk offsets equal to the summary's |
| Manifest | At most 1 MiB and 65,536 segments; file names of the form `seg-` with five digits and `.mcap`, never a path; `first_item` increasing |

A file holds no code and no reference to anything outside itself; a mesh's texture is inside it. Text lines, labels and names are shown as text, never as markup.

## Opening a recording

| Where | How | What runs where |
|---|---|---|
| The Caliper app | Open a `recording:` entry of the catalog; drop a `.mcap` file or folder on the window; `caliper open <recording>` | Everything in the app: it maps each segment, reads the index from `caliper_recordings` or from the files, and draws with its own views |
| A browser, no server | Drop a file or folder on the viewer page, or open the page with `?recording=<url>`, the URL of a file or of a manifest | Everything in the page: a reader in TypeScript reads the index and the frames, the existing codec decodes them, and the views draw them as if they had arrived live |
| A browser or the app, through the hub | `caliper serve --recordings <folder>`; the viewer picks a recording from the list | The hub reads the index and the frames and serves the recording as an ended session; the viewer draws what it receives |

**In the app.** A dropped recording is granted as a loader input is, recorded in `inputs.toml`, and registered where it lies; `recording.copy` copies it into `recordings/`. The recording opens as a session in the session list, with its views laid out as for a live session and the scrubber below them. `caliper open <recording>` is a new subcommand that starts the app, as `caliper dev` does, with the recording opened; it takes a `.mcap` file or folder, a manifest or a single segment, and with anything else it prints `caliper: not a recording: <path>` and exits with 1.

**In a browser.** The page reads a recording with `@mcap/core`, the official TypeScript reader, and Caliper's index attachment. The structural check showed the first half: `@mcap/core` 2.4.0 in headless Chromium opened two small Caliper files through a local server that honours `Range` in 4 range requests each, and read the profile, the channels and the index attachment's entry. The page then fetches the attachment with one more request and fetches each frame it needs: one range request for exactly the frame in a raw chunk, or the whole chunk for a small frame in a compressed one, which it decompresses and keeps in a cache of 256 MiB. The check showed why the attachment matters: its page, seeking through the official reader alone and reading each frame's stamp, needed 10 to 28 requests for one view and fetched whole chunks, most of the bytes of the uncompressed file. A seek through the attachment, one request per frame, is designed and was not built in the check. A segmented recording by URL needs its manifest's URL; the page reads the manifest and opens each segment at its URL relative to it. A dropped file is read in slices of the `File`, and a dropped folder through its entries. A host on another origin must send `Access-Control-Allow-Origin` and honour `Range`; without range support, or for a segment without a footer, the page downloads the whole file, up to 2 GiB. The reader adds `@mcap/core` and `fzstd` to the web package; with both, the check's reader bundle was 54,526 bytes minified and 15,179 gzipped. The page keeps the index in memory, about 40 bytes per item and 12 more per timeline, an estimate from the index's columns. The WebAssembly views live in a heap of at most 4 GB, which bounds what is drawn at once, not the size of the recording.

**Through the hub.** `--recordings <folder>` lists the `.mcap` files and `.mcap` folders in the folder, not deeper, under ids made from their names; no path ever travels on the wire. A viewer that announced the capability `recordings` gets the list and opens one, which becomes an ended session with a cursor of its own for each viewer. A viewer of protocol 3.1 sees an opened recording as an ended session and is shown its state at the end. The hub reads indexes and frame metadata only, as it does when it forwards.

## The scrubber

### The control

```text
+--------+------------------------+------+------+---------------------------------+---------+
| step v | |<   <   ||   >   >|   | 1x v | loop | step 12,480  12:04:51.207      | live >| |
+--------+------------------------+------+------+---------------------------------+---------+
 0           5,000          10,000          15,000          20,000              24,812
 |''''''''''''|''''''''''''''|''''''''''''''|''''''''''''''|''''''''''''''''''''|
 :::::::::::::::::::::::::::::.:::::::::::::::::::::::::::::::::::::::::::::::::::  items
                    [================ range ================]
                                    ^ cursor                                     ^ live edge
```

From the left: the active timeline, first, step back, play or pause, step forward, last, the speed, loop, the cursor's position on the active timeline with the wall time of the newest item at it, and, in a live session, the live button. Below them the timeline's axis, a row of marks where items lie, darker where they are dense, the range, the cursor and the live edge. Dragging on the axis moves the cursor, a drag with Shift sets the range, and the mouse wheel with Ctrl zooms the axis around the pointer. A part that thinning or the budget removed is hatched, and its hover says so.

### Timelines and the cursor

The **cursor** is a position on the active timeline. The menu lists the timelines on which the session has items; the default is `step` when it has items, otherwise `time`. Switching timelines keeps the moment: the new cursor is the position, on the new timeline, of the newest item at or before the old cursor, and falls back to the start when that item is absent on the new timeline.

### What each view shows at the cursor

Every view shows its data under **latest-at**: for each item, the newest value stamped at or before the cursor, with static items as the fallback, and among equal stamps the one recorded last.

| View | At the cursor |
|---|---|
| Heatmap, image, points, histogram, bars | The latest-at snapshot, with its stamp and its distance from the cursor in the header, such as "step 12,460, 20 before the cursor". A histogram shows the bins of that one snapshot |
| Series | The whole recorded curve, plotted against the active timeline, with a vertical line at the cursor; the part after the cursor is dimmed, and the header shows the latest-at value |
| Text | Every line stamped at or before the cursor, scrolled so that the newest of them is the last line shown; later lines are hidden, or dimmed when the person asks to see them. A click on a line moves the cursor to its stamp |
| Scene | For each entity part, its latest-at item, with clears applied, composed through the entity transforms as they stood at the cursor |
| Controls | The run state and parameter values recorded at the cursor, read only, beside the live ones |

Before its first item a view keeps its frame and says "No data at step 40; first at step 120", with a link that moves the cursor there. A series draws its axes and nothing before its first value, and a scene draws only its static items.

### Playing

Play moves the cursor forward over time. On a time timeline the cursor advances by the wall time elapsed times the speed. On a sequence timeline it follows the pace at which the items were recorded, through their `time` stamps, with gaps longer than 2 seconds shortened to 2 seconds, or 30 positions a second times the speed where items have no `time`. The speed goes from 1/16 to 64, in powers of two. Step forward and step back move the cursor to the next or previous position at which any shown view has an item. With loop on, playback that reaches the end of the range, or of the recording, starts again at the range's start; with loop off, it stops there. A range limits playback and looping, and series can fit their x axis to it.

### Live sessions

In a live session the cursor follows the **live edge** by default, and every view shows its newest data, as without timelines. Moving the cursor leaves the edge: the views hold at the cursor while new data keeps arriving and being kept, and the axis keeps growing on the right. The live button or End returns to the edge, and playback that reaches the edge without loop follows it again. Recording is deliberate: the app does not record a session unless the person presses Record in the session's toolbar or has turned on the setting "Record sessions when they start", which is off by default. Without a recording, the app scrubs only its live buffer, which is what it already holds of the session: the retained series and text, the scene's kept state, and the snapshots it has received, thinned against its memory budget as a browser page thins its own. The scrubber says how far back that buffer reaches, as "Live buffer from step 11,200" at the left end of its axis, and the part before it is drawn empty with a note that it was not recorded. Pressing Record starts a recording from that moment on; it does not reach back before the live buffer. A browser page scrubs its live buffer in the same way, and through a server that records the session, older data from that server ([Wire protocol 3.2](wire-protocol-3-2.md)).

### Selections and controls while scrubbing

A selection belongs to an item space, not to a moment, so it stays as the cursor moves, and every view highlights the selected items in whatever it shows at the cursor. In a live session a selection made on a scrubbed view reaches the script as any select event does, since item ids do not depend on time. Controls act on the live script only. The controls panel always shows the live run state and live values, which a person may change as usual, with the recorded values at the cursor beside them; in a recorded session the controls are read only. Pausing the run and pausing the scrubber are separate controls, labelled "Pause run" and "Pause", and neither moves the other.

### Keyboard

The keys act while the session's page has focus and no text field is active. They avoid the keys the app already uses: F in a live view, Escape, F4, and Ctrl with R, K and 1 to 4.

| Key | Action |
|---|---|
| Space | Play or pause |
| Left, Right | Step back or forward |
| Shift+Left, Shift+Right | Ten steps |
| Home | The first position |
| End | The last position; in a live session, follow the live edge |
| `[`, `]` | Set the range's start or end at the cursor |
| `\` | Clear the range |
| L | Loop on or off |
| `-`, `=` | Halve or double the speed |
| T | The next timeline |

### Several views in step

A session page has one cursor, and every view of the page reads it. For each frame, the app resolves latest-at for every view at that frame's cursor on a worker thread, and swaps in the new frames together once all have arrived, or after 50 ms with those that have. A browser page does the same; through a hub, each data message carries the generation of the cursor that it answers, and the page drops any of an older generation. Views of different sessions, such as two runs side by side, have a cursor each, and the person can link them on a shared timeline such as `step`.

## Size

Recordings on MCAP take 25 builder days in increment 7 of [Sprint 4](sprint-4.md), a provisional figure, which the design fits with no margin. The structural check's prototype covers most of the C++ work in outline but is not production code, and the risk lies in what it did not test: compaction, tailing, the roll-over and the browser's seek through the attachment. The hub's and the app's recorders, the workspace tables, the catalog provider and `caliper open` are in increment 9.

| Item | Builder days |
|---|---|
| The C++ recorder and reader: the writer with its patch, padding, mixed chunks, thread and queue (3); commit points, sync modes and the lock (1.5); the index with keyframes, validation and rebuilding (4); the reader with its checks and tailing (3); segments, the manifest and the roll-over (2); recovery (1.5); thinning and compaction (2); checks against the official tools and the sync cost (1) | 18 |
| Caliper LIB: `caliper.record`, its route, `Recorder`, `sync` and `full` | 2 |
| The browser reader: `@mcap/core`, the index attachment, frame fetches, `fzstd`, manifests and dropped folders | 3 |
| `caliper export --to parquet` | 2 |

## Limits

- **Only what was recorded.** A data view, a resolution or a moment that the recorder did not receive cannot be shown later.
- **Copies.** Recording is a copy off the device, as every network route is, even when the viewer reads the same session zero-copy through shared memory.
- **Budgets thin old data.** A long run keeps recent data whole and old data at falling density; the scrubber shows where.
- **The index lives in memory.** About 40 bytes per item and 12 more per timeline: a reader of a recording with 50 million items and three timelines needs about 3.8 GB for its index, beyond a browser's means.
- **Generic tools do not draw.** They list Caliper's channels and time its messages but cannot draw them, and tools that rewrite a file lose its alignment and mixed compression and leave its index stale, which Caliper then rebuilds.
- **Durability rests on reasoning.** Only killed writers were tested; power cuts, kernel panics and a full disk were not ([Guarantees](#guarantees)).
- **One writer per recording.** Two recorders never write one recording; the hub and the app each write their own.
- **macOS first.** Caliper LIB and the app are built and tested on macOS. The roll-over renames a folder, which on Windows fails while another process holds a file in it open, so it is checked again with [Windows parity](windows-parity.md). The file's own limits are in [Limits of the format](#limits-of-the-format).

## Changes to the shared contracts

- **Absent stamps.** The contracts say every item carries a stamp on every timeline. An item made before a timeline has a value, or after the script cleared it, is absent on that timeline and is not placed on it. This avoids inventing positions, such as a step 0 for everything logged before the first `caliper.step`.
- **`caliper.step` is the checkpoint of protocol 3.1.** The contracts' "or automatically" is read as `caliper.step()` without an argument, which adds one; a script that never calls it has no `step` stamps.
- **Pinning `time`.** `caliper.set_time("time", t)` may pin wall time, so that old data can be logged with its own times.

## Decisions

| Decision | Reason |
|---|---|
| The container is MCAP and the extension is `.mcap` | Decided by the user. An open format with official readers in C++, Python and TypeScript and a command-line tool |
| A recorded item is the live frame of protocol 3.2 | The data plan requires that recordings share their snapshots with the live protocol, and one codec then serves the app, the hub, Caliper LIB and the browser |
| Frames under the custom encoding `caliper.wire3`, on one channel per view and per entity part, not standard payloads | Byte-identical with the live message and read in place; standard schemas copy or misalign arrays and drop timelines, data views and axes. A channel is the unit that replaces and is thinned, and generic tools show each with its rate, though they cannot draw the frames |
| The timeline and keyframe index as attachments, used only after validation against the file | MCAP indexes only `log_time`, and attachments are within the specification and ignored by standard readers. Validation, because tools that rewrite a file keep the index stale, which the structural check found |
| A private padding record before each large frame | The recorded frame stays exactly the live frame while its largest array lands on a 16 KiB boundary for no-copy reads; standard readers skip it |
| Mixed compression through a 50-line patch of the official writer, pinned at 2.2.0 | Large frames are read in place and cost no compression, small frames are compressed, and the official readers read the result, as the structural check showed; far less code than a chunk writer of Caliper's own |
| A single file up to 1 GiB, then a folder of segments with a manifest | A short run stays one shareable file; segments bound compaction and file size, and each is a valid MCAP file alone |
| Atomic, consistent, isolated and durable, with a commit and an `fsync` every second by default | Loss bounded to about a second on a crash or a kernel panic with one sync a second; `"commit"` and `full=True` for more, `None` for less. The cost is measured when the recorder is built, against 3 % of a training step |
| Recovery of Caliper's own, never `mcap recover` | `recover` rewrites the file, loses alignment and mixed compression and leaves the index stale |
| Payloads in the files, the index and small data in `workspace.duckdb` | Frames read in place or with one range request, the workspace's file is the shared file, and the hub writes the same without DuckDB |
| Logarithmic thinning by count, per view and entity part, segment by segment | Deterministic, only ever removes, keeps h items per doubling of age, treats slow and fast views alike, and rewrites at most one segment at a time |
| Keyframes in the index that point to items | Latest-at reads at most 512 index entries after a keyframe without copying large static meshes again, and thinning never invalidates one |
| `caliper export --to parquet` through the app's DuckDB, with arrays as list columns on request | Analysis in other tools with no new dependency and one output format |
| Stamps read at the call, process-wide | No device sync, and a snapshot keeps the moment it was taken |
| One cursor per session page, generations through a hub | Every view of a page stays in step, and late messages of an earlier cursor are dropped |
| No automatic deletion of recordings | A recording is the person's data |
| Recording is deliberate | Decided by the user. The app records only after Record or with the setting "Record sessions when they start", which is off by default; a live session without a recording is scrubbed from the live buffer alone, and the scrubber says how far back it reaches |
| `caliper open` starts the app | It matches `caliper dev`, the other subcommand that starts the app |
| The structural check is the evidence for the file | It showed byte-identical round trips, the index's answers and its rebuild, alignment, mixed compression, valid segments, the command-line tool's behaviour, recovery after killed writers and opening in a browser. It did not test power cuts, kernel panics, a full disk, compaction, tailing, the roll-over, or any performance |

## Open decisions

- **Time windows.** Latest-at is the only rule in this sprint. A per-view window of items before the cursor, for trails of points and scene entities, is planned in [Sprint 5](sprint-5.md).
- **The default rates and budget.** Options: 2 Hz and 4 GiB, or a single bytes-per-second target that the recorder divides among views. Recommendation: per-kind rates now, a bytes-per-second target after measuring real runs.
- **Copying dropped files.** Options: register in place, or copy into the workspace. Recommendation: register in place, with `recording.copy` to keep a copy.
- **The writer change upstream.** Options: carry the patch privately; offer `setCompression()` and the accessor upstream and drop the patch once a release has them; or replace the patch with a chunk writer of Caliper's own. Recommendation: offer both changes upstream now and carry the pinned patch meanwhile; write Caliper's own chunk writer only if upstream declines and a later release makes the patch hard to carry.
- **A decoder of `caliper.wire3` for generic viewers.** Options: none; or an extension for a generic MCAP viewer that decodes series, heatmaps, images and points into its standard types. Recommendation: none in this sprint, since Caliper's own viewers draw every recording; reconsider when users ask to see recordings beside other robotics data.
- **Alignment of mid-sized frames.** Options: 16 KiB for every raw frame, or 16 bytes for frames under 1 MiB, whose copy is cheap, and 16 KiB from 1 MiB. Recommendation: 16 KiB for all until the recorder measures the padding on real runs, since the default heatmap pays about 3 %.
