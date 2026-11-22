# Wire protocol 3.2 draft

This page drafts version 3.2 of wire protocol v3, an additive minor version after 3.1. It adds timelines and the stamps that place every item on them, three kinds of live view (scenes, text logs and bar charts) and the messages that list, open and seek recordings. It follows the reference page of wire protocol v3 on the sprint 2 branch, uses its terms and refers to its sections by name. [Scenes](scenes.md) defines the scene model and Recordings the timelines, the recording file and the scrubber that these messages serve.

!!! note "Status"
    A draft for sprint 2 to review. Sprint 2 owns the wire protocol reference and `include/caliper/live_net.h`; the number 3.2 and every id, kind, dtype, key and code on this page are proposals until sprint 2 takes them into both. [What sprint 2 must confirm](#what-sprint-2-must-confirm) lists them.

## What is built and what is planned

Nothing of 3.2 is built. It rests on 3.1 as the sprint 2 branch builds it: the frame layout and its decoding order, VIEWS, SUBSCRIBE, SNAPSHOT and SERIES, data views, events, controls, linked selection, flow control, the hub and the golden frames under `tests/fixtures/wire_v3`. The example frames below were made by a prototype encoder, kept outside the repository, that reproduces the golden frames `series.bin`, `snapshot.bin`, `snapshot_points.bin`, `views.bin` and `v3_1/views_items.bin` byte for byte.

## What 3.2 adds

Every addition belongs to a capability. A sender uses it only with a peer that announced the capability, as the Versioning section of 3.1 requires, and every 3.1 message keeps its meaning and encoding.

| Capability | Adds |
|---|---|
| `timelines` | The VIEWS key `timelines`; the section `stamp` of SNAPSHOT, CONTROLS and the scene messages; the sections `stamp` or `stamps` of SERIES, SCENE_TRANSFORMS and TEXT; the key `static` |
| `scenes` | The kind `scene` (6), the VIEWS key `scene`, the scene data view, SCENE_UPDATE, SCENE_TRANSFORMS, SCENE_CLEAR and SCENE_CLASSES, the dtype `u16` (6) and the scene events |
| `text` | The kind `text` (7) and TEXT |
| `bars` | The kind `bars` (8), a SNAPSHOT of that kind, and the bar events |
| `recordings` | RECORDINGS, RECORDING_OPEN, CURSOR and TIME_STATE; the session key `recording`; the VIEWS key `recorded`; the keys `cursor` and `empty` of data messages; the SERIES section `indices` and key `decimated` |

A viewer and a hub announce what they understand, and a producer and a hub what they implement. A producer that streams scenes without timelines announces `scenes` alone, and its scene messages then carry no stamp.

## Checking for clashes

Every `sprint2*` branch, nineteen of them, was searched on the day of this draft:

- **Message types `0x0070` to `0x007F`.** None is used. The only matches of `0x007` are the float mask `0x007fffff` in the NaN tests of `webgpu_renderer.cpp`, `web/src/render/gpu.ts` and the bundled views.
- **Kinds 6 to 8.** None is used. `CaliperLiveKind` and `CaliperNetKind` stop at 5, and so do the Python and TypeScript codecs. The future table kind that 3.0 reserves has no id, so 9 stays free for it.
- **Dtype ids above 5.** None, in `CaliperDType` (`sdk/include/caliper/tensor.h` on `main`) or `CaliperNetDType` on any branch.
- **The string "3.2".** `CALIPER_NET_PROTOCOL_MINOR` is 1 everywhere. The only "3.2" is a section reference, "§3.2", in a comment of `include/caliper/embed.h`, which has nothing to do with the protocol.

## Timelines and stamps

### Declaring timelines

A producer with `timelines` adds the key `timelines` to VIEWS: one object per timeline, in the order of their ids.

| Key | Type | Meaning |
|---|---|---|
| `name` | string | 1 to 63 bytes of `[A-Za-z0-9_.-]`, unique in the session. The first two are always `step` and `time` |
| `kind` | string | `sequence` or `time` |
| `epoch` | boolean, optional | `time` only: the values count nanoseconds from the Unix epoch. True for `time` |

A session has at most 8 timelines. The producer sends the whole VIEWS again when the script declares one. Timelines are never removed or reordered during a session, so a timeline's id is its position in this list.

### The stamp

A **stamp** is T signed 64-bit integers, one per timeline in id order, where T is the number of timelines declared when the item was made. A receiver treats missing trailing values as absent. The value −2<sup>63</sup> (`CALIPER_NET_STAMP_ABSENT`) means that the item is absent on that timeline.

Stamps travel in sections of dtype `i64`, never as JSON numbers, because `time` is about 1.76 × 10<sup>18</sup> and a browser reads JSON numbers exactly only up to 2<sup>53</sup>. Where a stamp value appears in metadata, in CURSOR, TIME_STATE and RECORDINGS, it is **stamp text**: a string of decimal digits with an optional leading minus sign, at most 20 characters, such as `"1763200000123456789"`.

| Message | 3.2 addition |
|---|---|
| SNAPSHOT | Section `stamp` i64 (T,), and the key `static`. `section_count` is 1 to 3, or 0 with the key `empty` of a recorded session |
| SERIES | Section `stamps` i64 (M, T), one stamp per value, or section `stamp` i64 (T,), shared by all M values |
| CONTROLS | Section `stamp` i64 (T,): the moment the run state or a value changed. `section_count` is 0 or 1 |
| SCENE_UPDATE, SCENE_CLEAR, SCENE_CLASSES | Section `stamp` i64 (T,) |
| SCENE_TRANSFORMS, TEXT | Section `stamps` i64 (K or M, T), or `stamp` i64 (T,) shared by all |

A message with `static` true carries no stamp section: it holds a static item, present at every position of every timeline. A message from a producer without `timelines` has no stamp either, and a recorder of such a session stamps it with its own receive time on `time` and leaves it absent on `step`.

## The dtype u16

| Id | Name | Item size | numpy | TypedArray |
|---|---|---|---|---|
| 6 | `u16` | 2 | `<u2` | `Uint16Array` |

`u16` carries depth images, which sensors give in millimetres, and segmentation images, whose class ids run from 0 to 65,535. The 3.1 rule that the ids are the values of `CaliperDType` holds only if `CALIPER_DT_U16 = 6` is added to the SDK's `tensor.h` as well. No unsigned 32-bit type is added: mesh indices travel as `i32`, whose non-negative values have the bits of the `uint32` index format of WebGPU and Metal, and a mesh within the limits of [Scenes](scenes.md) has at most 4,194,304 vertices.

## New kinds in VIEWS

| Kind | Name | Payload |
|---|---|---|
| 6 | `scene` | SCENE_UPDATE, SCENE_TRANSFORMS, SCENE_CLEAR and SCENE_CLASSES |
| 7 | `text` | TEXT messages only |
| 8 | `bars` | SNAPSHOT: `data` f32 (B,), optional `colors` u8 (B, 4) |

A producer lists a view of a new kind only to a peer that announced its capability. A view object of 3.2 may also carry:

| Key | Kinds | Meaning |
|---|---|---|
| `scene` | scene | `{"space", "up", "units", "view_camera"}`: `"3d"` or `"2d"`, the up axis (`"+x"` … `"-z"`, or `"-y"` and `"+y"` in 2-D), a unit note of at most 15 bytes, and optionally, in a 3-D scene, [the starting camera](scenes.md#the-starting-camera): an object with any of `target` (3 numbers), `distance`, `azimuth` and `elevation` (numbers, the angles in degrees), or a string, the entity path of a camera to look through. A viewer ignores a `view_camera` it cannot read and opens on the fit |
| `sources` | all | Empty for a scene and a text log. A bars view lists sources as a histogram does |
| `recorded` | all, `recordings` | In a session that a server serves from a recording, the resolved data views the recording holds of this view, at most 64 |

**Data views.** A scene's data view has the keys `entities` and `budget` of [Data views of a scene](scenes.md#data-views-of-a-scene). `entities` is `{"include": [...], "exclude": [...]}`, at most 16 filters each, every filter an entity path, `<path>/**` or `**`; a filter outside the path grammar is `bad_data_view`. The budget keys are `max_points` (default 100,000, limit 1,000,000), `max_items` (100,000, 1,048,576), `max_vertices` (1,000,000, 4,194,304) and `max_extent` (1,024, 8,192), resolved key by key as for the other kinds. The resolved form has both filter lists and every budget key. A text log's data view is `{"budget": {"max_backfill": n}}`, as a series'. A bars view's data view is a histogram's with exactly one kept dim of at most 4,096 positions.

## Message types

| Id | Name | Direction | Sections | Answer |
|---|---|---|---|---|
| `0x0070` | `scene_update` | P to V/H, H to V | The arrays of one part, `stamp` | |
| `0x0071` | `scene_transforms` | P to V/H, H to V | `matrices`, `stamps` or `stamp` | |
| `0x0072` | `scene_clear` | P to V/H, H to V | `stamp` | |
| `0x0073` | `scene_classes` | P to V/H, H to V | `stamp` | |
| `0x0074` | `text` | P to V/H, H to V | `text`, `offsets`, `levels`, `stamps` or `stamp` | |
| `0x0078` | `recordings` | server to viewer | none | |
| `0x0079` | `recording_open` | viewer to server | none | `views` or `error` |
| `0x007A` | `cursor` | viewer to server | none | `time_state` |
| `0x007B` | `time_state` | server to viewer | none | |

A recording stores these messages as they are, in an MCAP file (Recordings), so the protocol reserves no message types for a recording file.

### scene_update

One part of one entity: a primitive batch or a camera. `section_count` is 0 to 6.

| Key | Type | Meaning |
|---|---|---|
| `sub` | integer | The subscription |
| `view` | string | The scene's name |
| `seq` | integer | The producer's counter of the scene's messages; it grows with every message of the scene |
| `stamp_ns` | integer | The producer's monotonic clock at the call, as in SNAPSHOT |
| `path` | string | The entity path: 0 to 16 segments of `[A-Za-z0-9_.-]{1,63}` joined by `/`, at most 255 bytes, never `.` or `..`. The empty path is the root |
| `part` | string | `points`, `lines`, `arrows`, `boxes`, `mesh`, `image` or `camera` |
| `static` | boolean, optional | A static item. Default false |
| `state` | boolean, optional | Part of the state sent to a new subscription. Default false |
| `axes` | array, optional | For a batch that the budget sampled, the payload axis of its item dim, as in SNAPSHOT |
| `items` | string, optional | The item space of 3.1 whose items the batch's rows are; only to a peer with `selection` |
| `split` | object, optional | See [Splitting a large update](#splitting-a-large-update) |

The keys and sections of each part, with N items and D = 3 in a 3-D scene, 2 in a 2-D scene and under a camera:

| Part | Sections | Keys |
|---|---|---|
| `points` | `positions` f32 (N, D); `colors` u8 (N, 4); `values` f32 (N,); `class_ids` i32 (N,); `radii` f32 (N,) | `color`, `radius`, `labels`, `colormap`, `flags`, `vmin`, `vmax`, `stats` |
| `lines` | `positions` f32 (V, D); `strip_lengths` i32 (S,) or `indices` i32 (E, 2); `colors` u8 (V, 4), (S, 4) or (E, 4); `values` f32 (V,), (S,) or (E,) | `mode` (`segments`, `strips` or `edges`), `color_per` (`vertex`, `strip` or `edge`, for both `colors` and `values`), `color`, `width`, `colormap`, `flags`, `vmin`, `vmax`, `stats` |
| `arrows` | `origins` f32 (N, D); `vectors` f32 (N, D); `colors` u8 (N, 4); `values` f32 (N,) | `color`, `width`, `colormap`, `flags`, `vmin`, `vmax`, `stats` |
| `boxes` | `centers` f32 (N, D); `half_sizes` f32 (N, D); `rotations` f32 (N, 4); `colors` u8 (N, 4); `class_ids` i32 (N,) | `color`, `labels`, `fill`, `width` |
| `mesh` | `positions` f32 (V, D); `indices` i32 (T, 3); `normals` f32 (V, 3); one of `colors` u8 (V, 4), `values` f32 (V,), `uvs` f32 (V, 2); `texture` u8 (H, W, 4) with `uvs` | `color`, `colormap`, `flags`, `vmin`, `vmax` |
| `image` | `pixels`: u8 (H, W, 4) or (H, W) for `color`; f32 or u16 (H, W) for `depth`; u16 or u8 (H, W) for `segmentation`; u8 (B,) for `encoded` | `form`, `opacity`; `scale`, `cloud`, `colormap`, `flags`, `vmin`, `vmax` for depth; `format` (`jpeg` or `png`) for encoded |
| `camera` | none | `fx`, `fy`, `cx`, `cy`, `width`, `height`, `convention` (`rdf` or `rub`), `plane_distance` (optional) |

Every section but `stamp` is optional except the first array of each part. Colours travel as RGBA: the producer expands RGB to RGBA on its device, and one colour for all items travels as `color`, `[r, g, b, a]` from 0 to 255. `labels` is one string for all or one per item, at most 4,096 strings of at most 63 bytes. `colormap`, `flags`, `vmin`, `vmax` and `stats` mean what they mean in SNAPSHOT, and `stats` holds `[values min, values max, x min, x max, y min, y max, z min, z max]`, computed on the device. With the `stamp` section, a part uses at most 6 of the 8 sections a frame may hold. A `lines` part of [index-pair edges](scenes.md#index-pair-edges) uses at most 5, `positions`, `indices`, `colors`, `values` and `stamp`; when the budget sampled it, its `axes` is the edge dim, and its positions travel whole.

### Splitting a large update

An update whose frame would pass 64 MiB, such as a mesh of four million vertices with its texture, travels in pieces. Each piece is a SCENE_UPDATE with the same `path`, `part`, `seq` and stamp and the key `split`:

| `split` key | Meaning |
|---|---|
| `update` | The producer's counter of split updates in the scene |
| `piece`, `pieces` | This piece's number from 0, and the number of pieces, 2 to 16 |
| `first` | Per section in this piece, the first row of the whole array that it holds |
| `rows` | Per array of the update, its rows in the whole update; in piece 0 only |

A piece carries a contiguous range of rows of some of the arrays, cut along their first axis; an image is cut by rows. The keys of the part are in piece 0 only. An encoded frame, at most 16 MiB, is never split. A receiver checks that the whole update stays within `CALIPER_NET_MAX_UPDATE_BYTES`, allocates its arrays, fills the ranges as pieces arrive and applies the update once every row of every array has arrived. A newer update of the same entity part discards an incomplete one, and a receiver holds at most 2 incomplete updates per subscription, discarding the oldest. A sender sends the pieces of an update consecutively on the subscription and never replaces part of one.

### scene_transforms

The entity transforms of up to 1,024 entities of one scene. Every entity transform travels as a 4 × 4 matrix, one form for the viewer to compose; the producer turns translation, rotation and scale, or a 2-D transform, into the matrix, on the host for numbers and on the device for a tensor.

| Key or section | Meaning |
|---|---|
| `sub`, `view`, `seq`, `stamp_ns`, `static`, `state` | As in SCENE_UPDATE |
| `paths` | The K entity paths, distinct, never the root |
| Section `matrices` f32 (K, 16) | Row k is the entity transform of `paths[k]`, column-major, the layout of the geometry service's instance poses |
| Section `stamps` i64 (K, T) or `stamp` i64 (T,) | Each entity transform's stamp, or one for all |

A producer merges the pending entity transforms of a scene into one message, keeping the newest per path, so a robot of 50 links updated at 60 Hz costs one message per frame. A static entity transform travels in a message of its own with `static` true. SCENE_CLEAR with `parts` `["transform"]` removes one.

### scene_clear

| Key | Type | Meaning |
|---|---|---|
| `sub`, `view`, `seq`, `stamp_ns`, `static`, `state` | | As in SCENE_UPDATE |
| `path` | string | The entity to clear |
| `recursive` | boolean, optional | Its subtree too. Default false |
| `parts` | array of strings, optional | Only these parts: `transform`, `camera` or a primitive. Default all |
| `reset` | boolean, optional | Empty the whole scene, static items included. Only the first message of a scene's state carries it |

A clear removes stamped parts from its stamp on, and static parts only when it is itself static, as [Clearing and static items](scenes.md#clearing-and-static-items) states.

### scene_classes

| Key | Type | Meaning |
|---|---|---|
| `sub`, `view`, `seq`, `stamp_ns`, `static`, `state` | | As in SCENE_UPDATE |
| `classes` | array of objects | At most 4,096 entries in ascending `id` order without repeats: `id` 0 to 65,535, `label` of at most 63 bytes, `color` `[r, g, b, a]` |

The table replaces the scene's table whole.

### The state of a scene for a new subscription

When a subscription to a scene opens, or its data view changes, the sender first sends the scene's state, each message with `state` true and its own stamp or `static`: SCENE_CLEAR with `path` `""` and `reset` true; SCENE_CLASSES; SCENE_TRANSFORMS of every entity that has an entity transform, static ones and stamped ones in separate messages; and SCENE_UPDATE of every camera and of every batch it holds that the data view passes, in the byte order of the paths. A producer holds what [Scenes](scenes.md) says it retains: static items, entity transforms, cameras, the class table, clears, and the newest snapshot it still has of each stamped batch. A viewer draws an entity whose batch was not held from its next update. A viewer may draw the state as it arrives.

### text

New lines of a text log, appended as a series' values are. A line is the text of one `caliper.text` call and may itself hold line breaks. `section_count` is 3, or 4 with a stamp section.

| Key or section | Meaning |
|---|---|
| `sub`, `view`, `stamp_ns` | As in SERIES |
| `start` | The absolute index of the first line: the lines appended before it |
| `total` | The lines appended so far. `start` + M ≤ `total` |
| `dropped` | Optional: lines before `start` that the sender no longer held when this viewer needed them |
| Section `text` u8 (L,) | The lines' UTF-8 bytes, concatenated |
| Section `offsets` i32 (M + 1,) | Line i is bytes `offsets[i]` to `offsets[i + 1]` of `text`; `offsets[0]` is 0 and `offsets[M]` is L |
| Section `levels` u8 (M,) | 0 debug, 1 info, 2 warning, 3 error |
| Section `stamps` i64 (M, T) or `stamp` i64 (T,) | The lines' stamps |

A line is at most 4,096 bytes of valid UTF-8; the producer cuts a longer one and marks the cut. A message holds at most 65,536 lines. On subscribe the sender first sends the lines it retains, at least the last 4,096, under `max_backfill`, and then the new ones. A viewer stores lines by index, as series values. Lines are never dropped silently: a viewer so far behind that its next lines left the sender's retention receives a `start` past what it holds, with `dropped`.

### bars

A bars snapshot is a SNAPSHOT of kind 8, with the 3.1 keys `sub`, `view`, `seq`, `stamp_ns`, `kind`, `data_view` and `axes`, and:

| Key or section | Meaning |
|---|---|
| Section `data` f32 (B,) | The bar values, 1 ≤ B ≤ 4,096 |
| Section `colors` u8 (B, 4) | Optional, one RGBA colour per bar |
| `labels` | Optional: B strings of at most 63 bytes, the labels of the positions the data view selected |
| `horizontal` | Optional boolean: bars from left to right |
| `stats` | `[min, max]` of the values |

### Events on the new kinds

| Kind | `pick` and `hover` detail | `select` detail |
|---|---|---|
| Scene | `entity`, the path; `primitive`; `index`, the item in the sent batch; `row_id`, its source index through `axes`; `position`, optional `[x, y, z]` of the hit in world space | `entities`: a list of `{"entity", "primitive", "indices", "row_ids"}`, or `{"entity"}` for a whole entity, at most 32,768 indices in all, with `truncated` when cut |
| Bars | `bar`, the payload index; `row_id`, its source index; `value` | `bars`: a `[start, stop]` range of source indices |
| Text | none | none |

With an item space, a select also carries `space` and `items`, as in 3.1.

### recordings

The recordings a server can open. `section_count` is 0. The server sends it right after SESSIONS to a viewer that announced `recordings`, and again whenever the list changes.

| Key | Type | Meaning |
|---|---|---|
| `recordings` | array of objects | At most 1,024 |

| Recording key | Type | Meaning |
|---|---|---|
| `id` | string | 1 to 63 bytes of `[A-Za-z0-9._-]`, made by the server from the file's name. No path travels |
| `title`, `script` | string | For display |
| `bytes`, `items` | integer | Its size and its number of items |
| `created_unix_ms`, `ended_unix_ms` | integer or null | When it started and ended; `ended_unix_ms` is null while a recorder still writes it |
| `complete` | boolean | The file has its footer |
| `timelines` | array of objects | `name`, `kind`, `epoch`, and `min` and `max` as stamp text |

A session object in SESSIONS may carry `recording`, `{"id", "seekable", "complete"}`: a session opened from a recording, or a live session of which the server holds a recording. A viewer may send CURSOR only for a session whose `seekable` is true.

### recording_open

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer (uint32) | Chosen by the viewer, echoed by TIME_STATE |
| `recording` | string | A recording's id |

The server opens the recording as an ended session, unless it is already open, and attaches the viewer to it as ATTACH would. It answers with SESSIONS when the list changed, then VIEWS, which confirms the attachment, CONTROLS as recorded at the end, and TIME_STATE with the `request_id`, the cursor at the end of the recording. A failure is answered with ERROR `unknown_recording` or `bad_recording` and leaves the viewer as it was. The session's id is the recording's id, with `-2`, `-3` and so on when that id is taken. An opened recording stays in the list while a viewer is attached and for 10 minutes after the last one leaves.

### cursor

A viewer's request to move its cursor. `section_count` is 0.

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer (uint32) | Echoed by TIME_STATE |
| `op` | string | `seek`, `play`, `pause`, `step` or `follow` |
| `timeline` | string | The active timeline; required with `seek`, `play` and `step` |
| `value` | stamp text | `seek`: the new cursor |
| `count` | integer | `step`: the positions to move, negative backwards, from −1,000 to 1,000 and not 0 |
| `speed` | number | `play`: from 1/16 to 64. Default 1 |
| `pace` | string | `play` on a sequence timeline: `recorded` (default) or `even` |
| `loop` | array or null | `play`: `[start, end]` as stamp text to loop over, or null for no loop |

A cursor belongs to one viewer connection and the session it is attached to; attaching to another session drops it. `pause` holds the cursor where it is, `step` moves it to the next or previous position with an item of a view the viewer subscribes to, and `follow` returns a live session's viewer to the live edge. Playback follows Playing. A viewer sends at most 30 CURSOR messages a second, for example while a person drags the scrubber, ending with the position where the drag stops, and a server may skip a `seek` it has not started when a newer one arrives. The server answers each CURSOR with TIME_STATE, or with ERROR `not_seekable` or `bad_cursor`: an unknown timeline, malformed stamp text, a speed or count out of range, or `follow` on an ended session.

While the cursor does not follow the live edge, the server sends each of the viewer's subscriptions to that session what lies at the cursor, each data message with the key `cursor`, the generation of the TIME_STATE it answers:

- For a heatmap, image, points, histogram or bars view, the latest-at snapshot under the subscription's data view, or the recorded data view the reply named, when it differs from the one last sent; or a SNAPSHOT with `empty` true and no section when nothing lies at or before the cursor.
- For a series or a text log, nothing more: the backfill of a recorded session holds every recorded value with its stamp, and the viewer draws the cursor itself. A recorded series that was decimated arrives with the section `indices` i64 (M,), the absolute index of each value, and the key `decimated`, the bucket size.
- For a scene, during continuous playback the scene messages between the previous position and the new one, in order; after any other move, the scene's state at the new cursor.

Live data for those subscriptions stops until `follow`, which first sends the newest snapshot of each and each scene's state.

### time_state

The state of a viewer's cursor. `section_count` is 0.

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer, optional | The CURSOR or RECORDING_OPEN it answers |
| `session` | string | The session |
| `generation` | integer (uint32) | Grows at every move that is not continuous playback: a seek, a step, a wrap of the loop, a follow, a change of timeline |
| `state` | string | `following`, `paused` or `playing` |
| `timeline` | string | The active timeline |
| `cursor` | stamp text or null | The cursor; null while following |
| `speed`, `pace`, `loop` | | As requested |
| `ranges` | object | Per timeline name, `[min, max]` as stamp text, over the session's data so far |
| `live` | boolean | The session is live |

A server sends it as the answer to CURSOR and RECORDING_OPEN, at most 30 times a second while playing, and at most once a second when the ranges of a live session grow. A viewer drops a data message whose `cursor` generation is older than the newest TIME_STATE it has received.

## Flow control

The rules of 3.1 hold, with these additions:

- **Scene updates coalesce per entity part.** Per subscription a sender keeps at most one pending SCENE_UPDATE for each entity path and part, which a newer one replaces in place. Pending entity transforms of a scene merge into one SCENE_TRANSFORMS, the newest per path, and a newer SCENE_CLASSES replaces a pending one.
- **Clears keep their order.** A SCENE_CLEAR is never dropped, and it drops the pending updates and entity transforms of the entities it clears that were queued before it, so that nothing it clears is sent after it.
- **The pieces of a split update** go together and are replaced only as a whole.
- **Text lines never drop silently.** TEXT coalesces into one pending message per subscription and is never dropped while the sender retains its lines; beyond that, `dropped` says how many were lost.
- **Rates.** For a scene, `max_rate` applies to each entity part; entity transforms, cameras, class tables and clears are not limited.
- **Order.** Among pending data, series and text go first, then scene messages, then snapshots, and subscriptions take turns. A scene's state goes before any later message of that scene.
- **TIME_STATE and RECORDINGS** are control messages: they go in order and before pending data, and a newer one replaces an unsent older one.
- **Playback** obeys `max_rate` per subscription and the one-pending-snapshot rule, so a slow viewer sees fewer frames during playback, never older ones.

## Through a hub

A 3.2 hub announces the capabilities it implements. It removes the additions from what it forwards to a viewer that lacks their capabilities: views of unknown kinds and the key `timelines` leave VIEWS, stamp sections and new keys leave SNAPSHOT, SERIES and CONTROLS, and the payload sections stay shared, since only the metadata is encoded again, as it already is to rewrite `sub`.

| Message | Forwarded | Kept for a late viewer |
|---|---|---|
| VIEWS with `timelines` | To every viewer of the session, filtered as above | The latest, as in 3.1 |
| SNAPSHOT, bars included | As in 3.1 | The latest of each producer subscription |
| SERIES with stamps | As in 3.1 | The newest 1,048,576 values of each series view with their stamps, 8 × T more bytes per value |
| Scene messages | To the viewers of the subscription that announced `scenes` | Per producer subscription, the scene's state: the class table, entity transforms, cameras, the newest batch of each entity part and the effect of clears, sent as the state to a late viewer |
| TEXT | As SERIES | The newest 65,536 lines or 16 MiB of each text view |
| CONTROLS with a stamp | As in 3.1 | The latest |
| RECORDINGS, TIME_STATE | Made by the hub | |
| RECORDING_OPEN, CURSOR | Answered by the hub | |

Everything kept counts toward the hub's budget of 1 GiB. When the hub has forgotten a scene's state, it subscribes again at the producer to receive a fresh state for a late viewer. The hub answers CURSOR only for sessions of which it holds a recording: a recording opened from its `--recordings` folder, or a session it records with `--record`. It never forwards CURSOR to a producer, so a viewer of a live session that the hub does not record scrubs its own buffer. With `--record`, the hub subscribes to every view at its recording rate, as its own recorder.

## Limits

| Constant | Value | Meaning |
|---|---|---|
| `CALIPER_NET_MAX_TIMELINES` | 8 | Timelines of a session |
| `CALIPER_NET_STAMP_ABSENT` | −2<sup>63</sup> | A stamp value meaning absent |
| `CALIPER_NET_STAMP_TEXT_MAX` | 20 | Characters of stamp text |
| `CALIPER_NET_PATH_MAX`, `CALIPER_NET_PATH_SEGMENTS_MAX` | 255, 16 | Bytes and segments of an entity path |
| `CALIPER_NET_MAX_ENTITIES`, `CALIPER_NET_MAX_BATCHES` | 1,024, 2,048 | Entities and batches of a scene |
| `CALIPER_NET_MAX_BATCH_POINTS` | 4,194,304 | Points, line and mesh vertices, index-pair edges or triangles of a batch |
| `CALIPER_NET_MAX_BATCH_ITEMS` | 1,048,576 | Boxes, arrows or line strips of a batch |
| `CALIPER_NET_MAX_LABELS` | 4,096 | Labels of a batch, of 63 bytes each |
| `CALIPER_NET_MAX_CLASSES` | 4,096 | Entries of a class table, ids 0 to 65,535 |
| `CALIPER_NET_MAX_IMAGE_SIDE` | 8,192 | Pixels on a side of an image |
| `CALIPER_NET_MAX_ENCODED_BYTES` | 16,777,216 | Bytes of an encoded frame |
| `CALIPER_NET_MAX_TRANSFORMS` | 1,024 | Entity transforms in one SCENE_TRANSFORMS |
| `CALIPER_NET_MAX_PIECES`, `CALIPER_NET_MAX_UPDATE_BYTES` | 16, 536,870,912 | Pieces and bytes of a whole split update |
| `CALIPER_NET_MAX_INCOMPLETE_UPDATES` | 2 | Incomplete split updates a receiver holds per subscription |
| `CALIPER_NET_TEXT_LINE_MAX`, `CALIPER_NET_TEXT_MAX_LINES` | 4,096, 65,536 | Bytes of a text line, and lines in one TEXT |
| `CALIPER_NET_TEXT_RETAIN_MIN` | 4,096 | Lines a producer retains per text log, at least |
| `CALIPER_NET_MAX_BARS` | 4,096 | Bars of a snapshot |
| `CALIPER_NET_MAX_RECORDINGS` | 1,024 | Recordings in RECORDINGS |
| `CALIPER_NET_CURSOR_RATE_HZ`, `CALIPER_NET_TIME_STATE_RATE_HZ` | 30, 30 | CURSOR messages a viewer sends, and TIME_STATE messages a server sends while playing, per second at most |
| `CALIPER_NET_MIN_SPEED`, `CALIPER_NET_MAX_SPEED` | 1/16, 64 | Playback speed |
| `CALIPER_NET_RECORDING_LINGER_MS` | 600,000 | An opened recording without viewers stays listed this long |

The scene limits are those of [Scenes](scenes.md#limits); that page is their authority.

## Error codes

| Code | Meaning |
|---|---|
| `unknown_recording` | RECORDING_OPEN named a recording the server does not list |
| `bad_recording` | The recording's file failed a check of Recordings |
| `not_seekable` | CURSOR for a session without a recording the server holds |
| `bad_cursor` | CURSOR with an unknown timeline, malformed stamp text, a value out of range, or `follow` on an ended session |

A scene message that breaks this page, such as a bad entity path or a part over its limit, is answered by the viewer with `bad_request`, as any invalid message. No close code is added.

## Versioning and compatibility

The minor version is 2. A 3.2 peer decides by capabilities, never by the minor alone.

| Viewer | Server | What happens |
|---|---|---|
| 3.1 | `caliper.serve()` of 3.2 | The viewer announces none of the 3.2 capabilities, so it receives no new kinds, no `timelines` and no stamp sections, and draws the 3.1 views as before |
| 3.2 | `caliper.serve()` of 3.1 | WELCOME announces none of them. The viewer draws what it gets and scrubs its own buffer, placed by its receive time on `time` |
| 3.2 | A 3.1 hub with a 3.2 producer | The hub's WELCOME to both announces none of them, so the producer sends 3.1 frames |
| 3.2 | A 3.2 hub with a 3.1 producer | The session lists no `timelines`. The hub forwards 3.1 frames; with `--record` it stamps them with its receive time on `time` |
| 3.1 | A 3.2 hub serving recordings | The viewer sees an opened recording as an ended session and gets its state at the end. It receives no RECORDINGS and sends no CURSOR |
| 3.2 with `timelines` but without `scenes` | A 3.2 producer | The scene views are left out of VIEWS; the other views carry stamps |

The golden frames of 3.2 belong in `tests/fixtures/wire_v3/v3_2/`, one per new message, op and part, and the 3.0 and 3.1 messages with their 3.2 keys and sections. A 3.1 decoder finds them well formed, with the new types unknown to it.

## Worked example

A viewer of a hub started with `--recordings` opens a recording of a training run with a series `loss`, a scene `world` and a text log `log`, seeks to step 5,000 and plays.

| # | From | Message | Content |
|---|---|---|---|
| 1 | V, H | HELLO, WELCOME | `capabilities` `["bars", "controls", "recordings", "scenes", "selection", "text", "timelines"]` both ways; `control` false |
| 2 | H | SESSIONS, RECORDINGS | The live sessions; `train-20261115-120314`, complete, `step` from `"0"` to `"19999"` |
| 3 | V | RECORDING_OPEN | `request_id` 1, `recording` `train-20261115-120314` |
| 4 | H | SESSIONS | The new ended session, with `recording` `{"seekable": true, "complete": true}` |
| 5 | H | VIEWS | `timelines` `step` and `time`; `loss` (4), `world` (6) and `log` (7) |
| 6 | H | CONTROLS | As recorded at the end, with its `stamp` |
| 7 | H | TIME_STATE | `request_id` 1, `generation` 1, `paused`, `step` at `"19999"` |
| 8 | V, H | SUBSCRIBE, VIEW_REPLY | `sub` 1 `loss`, `sub` 2 `world`, `sub` 3 `log`; three replies with the recorded data views |
| 9 | H | SERIES, TEXT | Every recorded value and line, with their `stamps` |
| 10 | H | Scene state | SCENE_CLEAR `reset`, SCENE_CLASSES, SCENE_TRANSFORMS, SCENE_UPDATE…, each with `state` true and `cursor` 1 |
| 11 | V | CURSOR | `request_id` 2, `seek`, `step`, `value` `"5000"` |
| 12 | H | TIME_STATE, scene state | `request_id` 2, `generation` 2, `paused` at `"5000"`; then the scene at step 5,000 with `cursor` 2 |
| 13 | V | CURSOR | `request_id` 3, `play`, `step`, `speed` 2 |
| 14 | H | TIME_STATE, scene messages | `request_id` 3, `generation` 3, `playing`; then the scene messages after step 5,000 at twice the recorded pace, each with `cursor` 3, and TIME_STATE at most 30 times a second |
| 15 | V | CURSOR | `request_id` 4, `pause`; answered by TIME_STATE `paused` at the step reached |

## Example frames

Both frames follow the frame layout of 3.1 exactly and are written in canonical form. Each of their stamps is step 1,200 at 1,763,200,000,123,456,789 ns.

**A series with a stamp per value.** The golden frame `series.bin` of 3.0 with minor 2 and the section `stamps` i64 (3, 2): the values 2.5, 1.25 and 0.75 at steps 1,200 to 1,202, 5 ms apart.

```text
0000  43 41 4c 57 03 00 02 00 31 00 00 00 08 01 00 00
0010  a4 00 00 00 02 00 00 00 0c 00 00 00 30 00 00 00
0020  7b 22 73 65 63 74 69 6f 6e 73 22 3a 5b 7b 22 64
0030  74 79 70 65 22 3a 30 2c 22 6e 61 6d 65 22 3a 22
0040  76 61 6c 75 65 73 22 2c 22 73 68 61 70 65 22 3a
0050  5b 33 5d 7d 2c 7b 22 64 74 79 70 65 22 3a 33 2c
0060  22 6e 61 6d 65 22 3a 22 73 74 61 6d 70 73 22 2c
0070  22 73 68 61 70 65 22 3a 5b 33 2c 32 5d 7d 5d 2c
0080  22 73 74 61 6d 70 5f 6e 73 22 3a 38 31 32 33 34
0090  35 36 37 38 39 31 31 32 33 2c 22 73 74 61 72 74
00a0  22 3a 30 2c 22 73 75 62 22 3a 33 2c 22 74 6f 74
00b0  61 6c 22 3a 33 2c 22 76 69 65 77 22 3a 22 6c 6f
00c0  73 73 22 7d 00 00 00 00 00 00 20 40 00 00 a0 3f
00d0  00 00 40 3f 00 00 00 00 b0 04 00 00 00 00 00 00
00e0  15 cd 53 eb 0e 25 78 18 b1 04 00 00 00 00 00 00
00f0  55 18 a0 eb 0e 25 78 18 b2 04 00 00 00 00 00 00
0100  95 63 ec eb 0e 25 78 18
```

| Offset | Bytes | Field | Value |
|---|---|---|---|
| 0 | `43 41 4c 57` | `magic` | `CALW` |
| 4 | `03 00`, `02 00` | `major`, `minor` | 3, 2 |
| 8 | `31 00` | `type` | `0x0031`, series |
| 12 | `08 01 00 00` | `frame_len` | 264 |
| 16 | `a4 00 00 00` | `meta_len` | 164 |
| 20 | `02 00` | `section_count` | 2 |
| 24 | `0c 00 00 00`, `30 00 00 00` | `section_len` | 12 `values`, 48 `stamps` |
| 32 | `7b 22` to `22 7d` | metadata | 164 bytes, ending at offset 196, then 4 bytes of padding |
| 200 | `00 00 20 40` … | `values` | 2.5, 1.25, 0.75, then 4 bytes of padding |
| 216 | `b0 04 … 78 18` … | `stamps` | (1,200, 1,763,200,000,123,456,789), (1,201, … 128,456,789), (1,202, … 133,456,789) |

```json
{"sections":[{"dtype":0,"name":"values","shape":[3]},{"dtype":3,"name":"stamps","shape":[3,2]}],"stamp_ns":81234567891123,"start":0,"sub":3,"total":3,"view":"loss"}
```

**A scene update of three points.** The entity `world/cloud` of the scene `world`, with positions, one colour per point and the stamp.

```text
0000  43 41 4c 57 03 00 02 00 70 00 00 00 58 01 00 00
0010  eb 00 00 00 03 00 00 00 24 00 00 00 0c 00 00 00
0020  10 00 00 00 7b 22 70 61 72 74 22 3a 22 70 6f 69
0030  6e 74 73 22 2c 22 70 61 74 68 22 3a 22 77 6f 72
0040  6c 64 2f 63 6c 6f 75 64 22 2c 22 73 65 63 74 69
0050  6f 6e 73 22 3a 5b 7b 22 64 74 79 70 65 22 3a 30
0060  2c 22 6e 61 6d 65 22 3a 22 70 6f 73 69 74 69 6f
0070  6e 73 22 2c 22 73 68 61 70 65 22 3a 5b 33 2c 33
0080  5d 7d 2c 7b 22 64 74 79 70 65 22 3a 35 2c 22 6e
0090  61 6d 65 22 3a 22 63 6f 6c 6f 72 73 22 2c 22 73
00a0  68 61 70 65 22 3a 5b 33 2c 34 5d 7d 2c 7b 22 64
00b0  74 79 70 65 22 3a 33 2c 22 6e 61 6d 65 22 3a 22
00c0  73 74 61 6d 70 22 2c 22 73 68 61 70 65 22 3a 5b
00d0  32 5d 7d 5d 2c 22 73 65 71 22 3a 31 32 2c 22 73
00e0  74 61 6d 70 5f 6e 73 22 3a 38 31 32 33 34 35 36
00f0  37 38 39 31 35 30 30 2c 22 73 75 62 22 3a 32 2c
0100  22 76 69 65 77 22 3a 22 77 6f 72 6c 64 22 7d 00
0110  00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 3f
0120  00 00 00 00 00 00 00 3f 00 00 00 00 00 00 00 40
0130  00 00 80 3e 00 00 00 00 ff 00 00 ff 00 ff 00 ff
0140  00 00 ff ff 00 00 00 00 b0 04 00 00 00 00 00 00
0150  15 cd 53 eb 0e 25 78 18
```

| Offset | Bytes | Field | Value |
|---|---|---|---|
| 4 | `03 00`, `02 00` | `major`, `minor` | 3, 2 |
| 8 | `70 00` | `type` | `0x0070`, scene_update |
| 12 | `58 01 00 00` | `frame_len` | 344 |
| 16 | `eb 00 00 00` | `meta_len` | 235 |
| 20 | `03 00` | `section_count` | 3 |
| 24 | `24 00 00 00`, `0c 00 00 00`, `10 00 00 00` | `section_len` | 36 `positions`, 12 `colors`, 16 `stamp` |
| 36 | `7b 22` to `22 7d` | metadata | 235 bytes, ending at offset 271, then 1 byte of padding |
| 272 | `00 00 00 00` … `00 00 80 3e` | `positions` | (0, 0, 0), (1, 0, 0.5), (0, 2, 0.25), then 4 bytes of padding |
| 312 | `ff 00 00 ff` … | `colors` | red, green, blue, opaque, then 4 bytes of padding |
| 328 | `b0 04 … 78 18` | `stamp` | 1,200 and 1,763,200,000,123,456,789 |

```json
{"part":"points","path":"world/cloud","sections":[{"dtype":0,"name":"positions","shape":[3,3]},{"dtype":5,"name":"colors","shape":[3,4]},{"dtype":3,"name":"stamp","shape":[2]}],"seq":12,"stamp_ns":81234567891500,"sub":2,"view":"world"}
```

## What sprint 2 must confirm

1. The minor version 2; the kinds `scene` 6, `text` 7 and `bars` 8, beside the table kind that 3.0 reserves without an id; the message types `0x0070` to `0x0074` and `0x0078` to `0x007B`.
2. The dtype `u16` 6, with `CALIPER_DT_U16 = 6` in `CaliperDType`, which the SDK owns, and stamps in `i64` sections, which makes `i64` a dtype that producers send.
3. The capabilities `timelines`, `scenes`, `text`, `bars` and `recordings`, the error codes and the limits above.
4. The hub's new duties: filtering for older viewers, keeping scene state and text, serving and recording sessions, and answering CURSOR.
5. The `caliper.step(n)` form of the 3.1 checkpoint in Caliper LIB.

## Decisions

| Decision | Reason |
|---|---|
| Stamps in `i64` sections, stamp text in metadata | Nanoseconds since the epoch exceed the 2<sup>53</sup> that a browser reads exactly from JSON |
| The timeline declaration is a key of VIEWS | Timelines change as rarely as views, and VIEWS already reaches every viewer and is kept by the hub |
| A stamp per series value and text line, or one for all | Pending values coalesce across calls, so one stamp per message would be wrong; a shared stamp keeps the common case small |
| One scene message per entity part | It is the unit that replaces, coalesces and is recorded, and every part fits within 8 sections |
| Entity transforms as matrices, batched per scene | One form for the viewer to compose, and one message per frame for an articulated body |
| Labels and class tables in metadata | They are short host strings, bounded well under 1 MiB |
| Only `u16` added | Depth and segmentation need it; `i32` already carries mesh indices with the bits of `uint32` |
| Index-pair edges as an `indices` section of `lines`, decided by the user | Edges reuse their positions instead of repeating them, and the part stays within 5 sections |
| Bars as a SNAPSHOT of kind 8 | A bar chart is replaced whole, as a histogram is, and needs no message of its own |
| Text in binary sections | No JSON escaping of log text, and up to 65,536 lines per message |
| A cursor per viewer connection, with generations | Several viewers of one recording scrub independently, and late data of an earlier cursor is dropped |
| The hub never forwards CURSOR to a producer | A producer serves one shared stream; seeking stays where a recording is held |
| A recorded session is an ended session to a 3.1 viewer | Older viewers still open recordings, at their end |

## Open decisions

- **The timeline declaration.** Options: a key of VIEWS, as drafted, or a message of its own. Recommendation: the key, unless sprint 2 prefers VIEWS unchanged.
- **Server-side playback.** Options: the server plays, as drafted, or the viewer sends a seek per frame. Recommendation: the server plays, which takes one message per view per frame off the viewer's link.
- **Time windows.** Options: latest-at only, or a subscription key that asks for every item within a window before the cursor, for trails. Recommendation: a key `window` in a later minor version.
- **The text log's levels.** Options: the four of [Scenes](scenes.md), or a fifth below debug. Recommendation: the four.
- **Scene state at the hub.** Options: the hub keeps it, as drafted, or always subscribes again at the producer for a late viewer. Recommendation: keep it, since a new subscription costs the producer device work.
