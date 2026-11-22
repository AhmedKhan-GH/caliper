# Wire Protocol 3.2

This page specifies what is built of wire protocol 3.2. Producers, the hub and viewers build scenes: the capability `scenes`, the live view kind `scene`, the messages SCENE_UPDATE with every part, SCENE_TRANSFORMS, SCENE_CLEAR and SCENE_CLASSES, the VIEWS key `scene` with its starting camera, and the default data view of a scene. The codecs build the whole draft besides: timelines and stamps, the dtype `u16`, text logs, bar charts, split updates and the messages of recordings, which no producer or server sends yet, except the stamps of snapshots from the two senders of `caliper.serve()` ([Timelines, text logs, bar charts and recordings](#timelines-text-logs-bar-charts-and-recordings)). Everything else follows [Wire protocol v3](wire-protocol.md), whose frame layout, decoding order, terms and rules apply unchanged.

!!! note "Status"
    A draft. The minor version 2, the capabilities, the kinds 6 to 8, the message types `0x0070` to `0x0074`, `0x0078` to `0x007B` and `0x0080` to `0x0082`, the dtype 6 and the error codes are proposals that the owners of the protocol have not confirmed, so they may still be renumbered. They are one line each in a separate block at the end of `include/caliper/live_net.h` and of each codec. Senders keep writing minor 1 in their frames until 3.2 is confirmed; a peer decides by a capability, never by the minor version. Every 3.1 message keeps its meaning and its encoding.

## Implementation Status

| Piece | Built |
|---|---|
| Capability | `scenes` |
| Kind | `scene` (6), with the VIEWS key `scene` and its `view_camera` |
| Messages | SCENE_UPDATE with the parts `points`, `lines` (segments, strips and index-pair edges), `arrows`, `boxes`, `mesh`, `image` (colour, depth, segmentation and encoded) and `camera`; SCENE_TRANSFORMS; SCENE_CLEAR; SCENE_CLASSES; the section `stamp` of each, and `stamps` of SCENE_TRANSFORMS |
| Dtype | `u16` (6) |
| Data view | The default scene data view, resolved |
| Codecs | `python/caliper/_wire.py`, `src/host/net/wire.h` and `web/src/protocol/codec.ts`, with the golden frames of `tests/fixtures/wire_v3/v3_2`: every message, key and section of the draft, the assembly of split updates in each, and their division in Python |
| Producer | `caliper.scene` in Caliper LIB on `caliper.serve()`, the Python and the native sender ([Caliper LIB](../python/scene-calls.md)) |
| Hub | Forwards the scene messages to the viewers that announced `scenes` and keeps a scene's state for a late viewer ([Hub Forwarding](#hub-forwarding)) |
| GPU programs | The capabilities `programs` and `interventions`, PROGRAM and PROGRAM_STATE, the CONTROL op `intervene` and the CONTROLS key `interventions` in the three codecs; Caliper LIB on both senders, the hub and the Caliper app send and forward them ([GPU Programs and Interventions](#gpu-programs-and-interventions)) |
| Programmable graphics | The capability `resources`, RESOURCE and the instance form of PROGRAM in the three codecs; the hub forwards them and keeps them for a late viewer, and the Caliper app announces `resources`, sends instances and keeps the versions of each resource ([Programmable Graphics](#programmable-graphics)) |
| Viewers | The Caliper app, `caliper-viewer` and the WebAssembly views announce `scenes` and draw every part with one scene view, all but encoded images, and apply SCENE_TRANSFORMS and SCENE_CLASSES; the TypeScript client announces it so that the `<caliper-view>` element and the viewer page hand a scene to the WebAssembly views |

Both senders of `caliper.serve()`, the native sender and the Python sender, and the Caliper app build timelines for snapshots ([Timelines](#timelines-text-logs-bar-charts-and-recordings)): to a peer that announces `timelines`, the sender declares the timelines `step` and `time` and stamps each SNAPSHOT with the step of the run at the view call that took it, and the app announces `timelines` and keeps the step of the newest snapshot of each live view. Not built beyond the codecs: timelines in the other senders (they declare none and send no stamp, and a receiver reads one and keeps it), timelines of series, text logs and bar charts in Caliper LIB and the viewers, split updates in the senders and viewers, and recordings in a server. Not built at all: scene data views other than the default, and scene events. [Versioning](#versioning) says what a receiver does when one of them arrives, and what a viewer does with a part it does not draw yet.

## Capability

A viewer announces `scenes` in HELLO when it draws scenes, and a producer and a hub announce it in WELCOME when they send them. A producer lists a view of kind 6 in VIEWS, and sends the scene messages, only to a peer that announced `scenes`; a SUBSCRIBE to a scene from another peer is answered with `unknown_view`. A hub announces `scenes` to the producers that connect to it. The key `static` of the scene messages comes with `scenes` alone, since a scene's state needs it while no timeline is built.

## Scene Kind

A scene is one live view: a tree of entities, named by entity paths, in one 2-D or 3-D space. Its entry in VIEWS has kind 6, an empty `sources` list and the key `scene`:

| Key of `scene` | Type | Meaning |
|---|---|---|
| `space` | string | `"3d"` or `"2d"`. Default `"3d"` |
| `up` | string | The axis a 3-D view keeps upright when it opens: `"+x"`, `"-x"`, `"+y"`, `"-y"`, `"+z"` or `"-z"`. In 2-D, the direction of y on screen, `"-y"` (image pixels) or `"+y"` (a plot). Default `"+y"` in 3-D and `"-y"` in 2-D |
| `units` | string | A unit note for display, at most 15 bytes. Nothing is converted. Default `"m"` |
| `view_camera` | object or string, optional | The camera a 3-D view opens on: an object with any of `target` (3 numbers), `distance` (above 0), `azimuth` and `elevation` (degrees, the elevation from −89 to 89), the orbit's angles read against `up`; or the entity path of a camera to look through. A viewer ignores a `view_camera` it cannot read, and one in a 2-D scene, and opens on the fit |

An **entity path** is 0 to 16 segments of `[A-Za-z0-9_.-]{1,63}` joined by `/`, at most 255 bytes, with no segment `.` or `..`. The empty path is the root, the scene's own space. An entity holds at most one batch of each part, and logging a part again at the same path replaces it.

## Dtype `u16`

| Id | Name | Item size | numpy | TypedArray |
|---|---|---|---|---|
| 6 | `u16` | 2 | `<u2` | `Uint16Array` |

`u16` carries depth images in stored units, such as millimetres, and segmentation images with class ids up to 65,535. From 3.2 a decoder checks the length of a `u16` section against its shape as it does for the other dtypes. The SDK's `CaliperDType` in `sdk/include/caliper/tensor.h` has no `u16` yet.

## Scene Data View

A subscription to a scene takes the default data view only: `data_view` absent, `{}`, or the resolved default itself. It resolves to:

```json
{"budget":{"max_extent":1024,"max_items":100000,"max_points":100000,"max_vertices":1000000},"entities":{"exclude":[],"include":["**"]}}
```

Any other data view is answered with `bad_data_view`. The producer applies the budget to points and to line segments by stride sampling, with every array of the batch sampled alike, and then sends `axes` with the batch, as a points snapshot does ([Payload axes](data-view-messages.md#payload-axes)); the other parts travel whole.

## Message Types

| Id | Name | Direction | Sections | Answer |
|---|---|---|---|---|
| `0x0070` | `scene_update` | producer to viewer or hub, hub to viewer | The arrays of one part, `stamp` | |
| `0x0071` | `scene_transforms` | producer to viewer or hub, hub to viewer | `matrices`, `stamps` or `stamp` | |
| `0x0072` | `scene_clear` | producer to viewer or hub, hub to viewer | `stamp` | |
| `0x0073` | `scene_classes` | producer to viewer or hub, hub to viewer | `stamp` | |
| `0x0074` | `text` | producer to viewer or hub, hub to viewer | `text`, `offsets`, `levels`, `stamps` or `stamp` | |
| `0x0078` | `recordings` | server to viewer | none | |
| `0x0079` | `recording_open` | viewer to server | none | `views` or `error` |
| `0x007a` | `cursor` | viewer to server | none | `time_state` |
| `0x007b` | `time_state` | server to viewer | none | |
| `0x0080` | `program` | viewer to producer or hub, hub to producer; for `resources`, also producer to viewer or hub, hub to viewer | `wgsl`, or `common`, `derive` and `draw` | `program_state` or `error`, from a producer only |
| `0x0081` | `program_state` | producer to viewer or hub, hub to viewer | none | |
| `0x0082` | `resource` | producer to viewer or hub, hub to viewer | `resource` | |

Every scene message has the keys `sub`, `view`, `seq` and `stamp_ns`, and may carry `static`, `state` and `cursor`:

| Key | Type | Meaning |
|---|---|---|
| `sub` | integer | The subscription |
| `view` | string | The scene's name |
| `seq` | integer | The producer's counter of the scene's messages; it grows with every new message of the scene, and a message sent again in a scene's state keeps its own |
| `stamp_ns` | integer | The producer's monotonic clock at the call, as in SNAPSHOT |
| `static` | boolean, optional | A static item: the scene's setup, which the producer retains. It carries no stamp. Default false |
| `state` | boolean, optional | Part of a scene's state for a new subscription. Default false |
| `cursor` | integer, optional | The generation of the TIME_STATE that a recorded message answers; unsigned 32-bit |

### Stamp

A **stamp** is T signed 64-bit integers, one per timeline in the order of their ids, from 1 to 8 (`CALIPER_NET_MAX_TIMELINES`) of them; the value −2<sup>63</sup> (`CALIPER_NET_STAMP_ABSENT`) means absent on that timeline. A scene message carries its stamp in the section `stamp`, i64 (T,), written after every other section; SCENE_TRANSFORMS may instead carry `stamps`, i64 (K, T), one stamp per entity transform. A message with `static` true carries neither. The native sender and the Python sender of `caliper.serve()` announce `timelines`. They declare the timelines `step` (sequence) and `time` (time, with an epoch) in VIEWS when the peer announces `timelines`. They stamp each SNAPSHOT to that peer with the step of the run at the view call that took the snapshot, or the absent value before the first step, and the time of the call in nanoseconds. A snapshot thus keeps its step when it arrives steps later. The other senders declare no timelines and send no stamp. A receiver checks a stamp and keeps it with the message.

### scene_update

One part of one entity. `section_count` is 0 to 8.

| Key | Type | Meaning |
|---|---|---|
| `path` | string | The entity path |
| `part` | string | `points`, `lines`, `arrows`, `boxes`, `mesh`, `image` or `camera` |
| `axes` | array, optional | For a batch the budget sampled, the payload axis of its item dim, as in SNAPSHOT |
| `items` | string, optional | The item space of 3.1 whose items the batch's rows are; only to a peer with `selection` |
| `split` | object, optional | A piece of a split update ([Split updates](#split-updates)) |

The sections and keys of each part, with N items and D = 3 in a 3-D scene or 2 in a 2-D scene and under a camera, in the order a sender writes them:

| Part | Sections | Keys |
|---|---|---|
| `points` | `positions` f32 (N, D); `colors` u8 (N, 4); `values` f32 (N,); `class_ids` i32 (N,); `radii` f32 (N,) | `color`, `radius`, `labels`, `colormap`, `flags`, `vmin`, `vmax`, `stats` |
| `lines` | `positions` f32 (V, D); `strip_lengths` i32 (S,) for strips or `indices` i32 (E, 2) for edges; `colors` u8 (V, 4), (S, 4) or (E, 4); `values` f32 (V,), (S,) or (E,) | `mode`, `color_per`, `color`, `width`, `colormap`, `flags`, `vmin`, `vmax`, `stats` |
| `arrows` | `origins` f32 (N, D); `vectors` f32 (N, D); `colors` u8 (N, 4); `values` f32 (N,) | `color`, `width`, `colormap`, `flags`, `vmin`, `vmax`, `stats` |
| `boxes` | `centers` f32 (N, D); `half_sizes` f32 (N, D); `rotations` f32 (N, 4), 3-D only; `colors` u8 (N, 4); `class_ids` i32 (N,) | `color`, `labels`, `fill`, `width`, `stats` |
| `mesh` | `positions` f32 (V, D); `indices` i32 (T, 3); `normals` f32 (V, 3); `colors` u8 (V, 4); `values` f32 (V,); `uvs` f32 (V, 2); `texture` u8 (H, W, 4) | `color`, `colormap`, `flags`, `vmin`, `vmax`, `stats` |
| `image` | `pixels`: u8 (H, W, 4) or (H, W) for `color`; f32 or u16 (H, W) for `depth`; u16 or u8 (H, W) for `segmentation`; u8 (B,) for `encoded` | `form`, `opacity`, `scale`, `cloud`, `colormap`, `flags`, `vmin`, `vmax`, `format`, `stats` |
| `camera` | none | `fx`, `fy`, `cx`, `cy`, `width`, `height`, `convention`, `plane_distance` |

| Key | Type | Meaning |
|---|---|---|
| `mode` | string | Lines, required: `segments`, where vertices 2k and 2k + 1 form segment k and V is even; `strips`, where `strip_lengths` gives the vertices of each strip, each length at least 1 and their sum V; or `edges`, where row k of `indices` is edge k from vertex `indices[k, 0]` to vertex `indices[k, 1]` |
| `color_per` | string, optional | Lines: `vertex` (default), `strip` (strips only) or `edge` (edges only), what the rows of `colors` and `values` belong to |
| `color` | `[r, g, b, a]`, optional | One colour for every item, from 0 to 255 |
| `radius` | number, optional | Points: one radius for every point, in scene units; above 0. `radii` gives one per point. Without either, points are discs of a few pixels |
| `width` | number, optional | Lines, arrows and box edges: the width in pixels, above 0. Default 2. A camera's `width` is its image width instead |
| `labels` | array of strings, optional | Points and boxes: 1 to 4,096 strings of at most 63 bytes, one for all items or one per item |
| `fill` | boolean, optional | Boxes: also draw translucent faces |
| `colormap`, `flags`, `vmin`, `vmax` | | As in SNAPSHOT: the colormap of `values`, or of a depth image, and its range |
| `stats` | array, optional | `[values min, values max, x min, x max, y min, y max, z min, z max]`, computed on the producer's device; z is 0 in 2-D and the values entries are 0 without `values`. A depth image's are `[min, max]` of its depths in scene units. A non-finite entry is null |
| `form` | string | Images, required: `color`, `depth`, `segmentation` or `encoded` |
| `opacity` | number, optional | Images: from 0 to 1 |
| `scale` | number, optional | Depth images: scene units per stored unit, above 0 |
| `cloud` | boolean, optional | Depth images: also draw the back-projected point cloud |
| `format` | string | Encoded images, required: `jpeg` or `png`. The frame is at most 16 MiB |
| `fx`, `fy` | number | Cameras, required: the focal lengths in pixels, above 0 |
| `cx`, `cy` | number | Cameras, required: the principal point in pixels |
| `width`, `height` | integer | Cameras, required: the image size, from 1 to 8,192 pixels |
| `convention` | string | Cameras, required: `rdf` (x right, y down, z forward) or `rub` (x right, y up, looking along −z) |
| `plane_distance` | number, optional | Cameras: how far in front of the camera its image plane is drawn, in scene units, above 0 |

The first array of each part is required, and so are `vectors` of arrows, `half_sizes` of boxes, `strip_lengths` of strips and `indices` of edges; the other sections are optional. An item's colour comes from `colors`, else from `values` through the colormap, else from its class id, else from `color`, else from the viewer's default; `colors` and `values` together are refused, and a mesh takes at most one of `colors`, `values` and `uvs`, with `uvs` and `texture` together or not at all. A mesh without `indices` takes each three vertices as a triangle, so V is a multiple of 3. Index values are never checked by a receiver: an index out of range draws as the last vertex, by the index-clamp contract. An image and a texture are at most 8,192 pixels on a side. A camera part carries no array. N, V, T and E are at most 4,194,304 for points, vertices, triangles and edges, and N and S at most 1,048,576 for arrows, boxes and strips.

### scene_transforms

The entity transforms of up to 1,024 entities of one scene, each a 4 × 4 matrix.

| Key or section | Meaning |
|---|---|
| `paths` | The K entity paths, 1 to 1,024 of them, distinct, never the root |
| Section `matrices` f32 (K, 16) | Row k is the entity transform of `paths[k]`, column-major, the layout of the geometry service's instance poses |
| Section `stamps` i64 (K, T) or `stamp` i64 (T,) | Each entity transform's stamp, or one for all |

A static entity transform travels in a message of its own with `static` true. SCENE_CLEAR with `parts` `["transform"]` removes one.

### scene_clear

| Key | Type | Meaning |
|---|---|---|
| `path` | string | The entity to clear |
| `recursive` | boolean, optional | Its subtree too. Default false |
| `parts` | array of strings, optional | Only these parts: a part of SCENE_UPDATE or `transform`. Default all |
| `reset` | boolean, optional | Empty the whole scene, static items included. Only with `path` `""`, and only as the first message of a scene's state |

A clear removes the parts it names from its entity, or its subtree, and leaves static parts unless it is itself static.

### scene_classes

| Key | Type | Meaning |
|---|---|---|
| `classes` | array of objects | At most 4,096 entries in ascending `id` order without repeats: `id` 0 to 65,535, `label` of at most 63 bytes, `color` `[r, g, b, a]` |

The table replaces the scene's table whole; an empty list empties it.

## Scene State for a New Subscription

When a subscription to a scene opens, the producer first sends the scene's state, each message with `state` true: SCENE_CLEAR with `path` `""` and `reset` true; SCENE_CLASSES, when the scene has a class table; SCENE_TRANSFORMS of every entity that has an entity transform, the static ones and the others in separate messages; then a SCENE_UPDATE of every camera and every batch it holds, in the byte order of their paths and, within a path, in the order of the parts in the table above. It holds the class table, the entity transforms, the cameras and every static item, which it copies once at the call even while nothing is attached, and the newest batch it still has of each other entity part, which exists only for the batches it sent while some subscription was open. A viewer draws an entity whose batch was not held from its next update.

## Flow Control

The rules of [Flow control](wire-protocol.md#flow-control) hold, with these additions:

- Per subscription, a sender keeps at most one pending SCENE_UPDATE for each entity path and part, which a newer one replaces. Pending entity transforms merge into one SCENE_TRANSFORMS, the newest per path and the static ones apart, and a newer SCENE_CLASSES replaces a pending one.
- A scene's state goes before any later message of that scene, and a SCENE_CLEAR is never dropped; it drops the pending updates and entity transforms it clears that were queued before it.
- `max_rate` applies to each entity part of a scene. Entity transforms, cameras, class tables and clears are not limited.
- Among pending data, series go first, then scene messages, then snapshots. The native sender of Caliper LIB hands its scene messages to its connection as control frames instead, at most two waiting at once, so that they still coalesce in the producer while a viewer drains slowly.

## Hub Forwarding

The hub announces `scenes` to producers and viewers. It removes the views of kind 6 from the VIEWS it sends to a viewer without `scenes`, and refuses that viewer's SUBSCRIBE to one with `unknown_view`. It forwards the scene messages to the viewers of the producer subscription, rewriting `sub` and sharing the payload sections, with the flow control above per viewer. Of the messages that go to a viewer in order, clears and states, it keeps at most 2 × `CALIPER_NET_MAX_BATCHES` waiting: a reset drops what waits before it, and past that bound the hub's own state replaces what waits. For a late viewer it keeps, per producer subscription, the newest batch of each entity part, the class table and the newest entity transform of each entity, a clear removing what it clears and a reset all of it, and sends them as the scene's state, opened by its own SCENE_CLEAR with `reset`. A viewer's pending entity transforms merge into one SCENE_TRANSFORMS, the newest per entity, and a camera and a class table are not held to the viewer's `max_rate`. The hub puts the pieces of a split update together before it forwards them, and then sends and keeps them as one update, in order and replaced only as a whole; it drops a piece whose piece 0 it does not hold. What it keeps counts toward its budget of retained bytes. Over that budget, or past `CALIPER_NET_MAX_BATCHES` batches of one scene, it forgets a scene's state, even one that viewers watch; they keep receiving the scene, and when a later viewer subscribes, the hub subscribes again at the producer, whose fresh state reaches that viewer.

## Limits

| Constant | Value | Meaning |
|---|---|---|
| `CALIPER_NET_PATH_MAX` | 255 | Bytes of an entity path |
| `CALIPER_NET_PATH_SEGMENTS_MAX` | 16 | Segments of an entity path |
| `CALIPER_NET_UNITS_MAX` | 15 | Bytes of a scene's unit note |
| `CALIPER_NET_MAX_ENTITIES` | 1,024 | Entities of a scene, which the producer enforces |
| `CALIPER_NET_MAX_BATCHES` | 2,048 | Batches of a scene, which the producer enforces |
| `CALIPER_NET_MAX_BATCH_POINTS` | 4,194,304 | Points, line vertices, index-pair edges, mesh vertices and triangles of a batch |
| `CALIPER_NET_MAX_BATCH_ITEMS` | 1,048,576 | Arrows, boxes and line strips of a batch |
| `CALIPER_NET_MAX_LABELS` | 4,096 | Labels of a batch, each of at most 63 bytes |
| `CALIPER_NET_MAX_CLASSES` | 4,096 | Entries of a class table |
| `CALIPER_NET_MAX_CLASS_ID` | 65,535 | The largest class id |
| `CALIPER_NET_MAX_IMAGE_SIDE` | 8,192 | Pixels on a side of an image or a texture |
| `CALIPER_NET_MAX_ENCODED_BYTES` | 16,777,216 | Bytes of an encoded frame |
| `CALIPER_NET_MAX_TRANSFORMS` | 1,024 | Entity transforms in one SCENE_TRANSFORMS |
| `CALIPER_NET_MAX_TIMELINES` | 8 | Values of a stamp |
| `CALIPER_NET_STAMP_TEXT_MAX` | 20 | Characters of stamp text |
| `CALIPER_NET_MAX_PIECES` | 16 | Pieces of a split update |
| `CALIPER_NET_MAX_UPDATE_BYTES` | 536,870,912 | Bytes of a whole split update |
| `CALIPER_NET_MAX_INCOMPLETE_UPDATES` | 2 | Incomplete split updates a receiver holds per subscription |
| `CALIPER_NET_TEXT_LINE_MAX` | 4,096 | Bytes of a text line |
| `CALIPER_NET_TEXT_MAX_LINES` | 65,536 | Lines in one TEXT |
| `CALIPER_NET_TEXT_RETAIN_MIN` | 4,096 | Lines a producer retains per text log, at least |
| `CALIPER_NET_MAX_BARS` | 4,096 | Bars of a snapshot |
| `CALIPER_NET_MAX_RECORDINGS` | 1,024 | Recordings in RECORDINGS |
| `CALIPER_NET_MAX_RECORDED` | 64 | Recorded data views of a view in VIEWS |
| `CALIPER_NET_MAX_CURSOR_STEP` | 1,000 | Positions a CURSOR `step` moves, either way |
| `CALIPER_NET_CURSOR_RATE_HZ` | 30 | CURSOR messages a viewer sends a second, at most |
| `CALIPER_NET_TIME_STATE_RATE_HZ` | 30 | TIME_STATE messages a server sends a second while playing, at most |
| `CALIPER_NET_MAX_SPEED` | 64 | The fastest playback; the slowest, `CALIPER_NET_MIN_SPEED`, is 1/16 |
| `CALIPER_NET_RECORDING_LINGER_MS` | 600,000 | How long an opened recording without viewers stays listed |
| `CALIPER_NET_SCENE_DEFAULT_MAX_ITEMS` | 100,000 | Default `max_items` of a scene data view |
| `CALIPER_NET_SCENE_LIMIT_MAX_ITEMS` | 1,048,576 | Largest `max_items` |
| `CALIPER_NET_SCENE_DEFAULT_MAX_VERTICES` | 1,000,000 | Default `max_vertices` |
| `CALIPER_NET_SCENE_LIMIT_MAX_VERTICES` | 4,194,304 | Largest `max_vertices` |
| `CALIPER_NET_SCENE_DEFAULT_MAX_EXTENT` | 1,024 | Default `max_extent` |
| `CALIPER_NET_SCENE_LIMIT_MAX_EXTENT` | 8,192 | Largest `max_extent` |

A batch must also fit in one frame of 64 MiB, since Caliper LIB sends no split update yet; it refuses a larger one at the call.

## Error Codes and Check Order

| Code | Meaning |
|---|---|
| `unknown_recording` | RECORDING_OPEN named a recording the server does not list |
| `bad_recording` | The recording's file failed a check of the recording format |
| `not_seekable` | CURSOR for a session without a recording the server holds |
| `bad_cursor` | CURSOR with malformed stamp text, a field of its op missing or out of range, an unknown timeline, or `follow` on an ended session |

A scene message that breaks this page is answered with ERROR `bad_request`, as any invalid message. The three codecs check a message's keys and sections in the same order, so that all of them answer a frame with the same code: the common keys (`sub`, `view`, `seq`, `stamp_ns`), then for SCENE_UPDATE `path` and `part`, then `static`, `state`, `cursor`, `axes` and `items`, then the part's keys in the order of the table, then its sections in the order of the table, and last the stamp.

## Versioning

A receiver of this version handles the rest of the 3.2 draft by the rules of [Versioning](wire-protocol.md#versioning):

| What arrives | What the receiver does |
|---|---|
| A piece of a split update (the key `split`) at a viewer that does not assemble them yet | Ignores the message and answers ERROR `unsupported` |
| A part or a form that the viewer does not draw yet: the native and browser scene views draw every part and form but encoded images | Ignores the message, answers ERROR `unsupported` and says so in the scene view |
| A section the viewer does not use, such as `stamp` | Ignores the section, says so in the scene view, and draws the rest of the message |
| An unknown part, a bad entity path, a section of the wrong dtype or shape, strip lengths that do not add up, or a batch over a limit | Answers ERROR `bad_request` |

A 3.1 peer announces no `scenes`, so it receives no scene view and no scene message; a 3.1 decoder finds the 3.2 frames well formed, with their types unknown to it.

## Readings of the Draft

Where the draft of 3.2 is silent or ambiguous, this version reads it as follows:

| Point | Reading |
|---|---|
| The number of values of a stamp | From 1 to `CALIPER_NET_MAX_TIMELINES` |
| `labels` | Always an array: one string for all items, or one per item |
| `stats` | Accepted on every part but `camera`, since the scene view fits its camera to the bounds; a depth image's are its range in scene units |
| Image keys | One list for every form; a receiver uses the keys that apply to the form |
| `format` of an encoded image and `convention` of a camera | Required, as the draft lists them without "optional" |
| `cx` and `cy` of a camera | Required; Caliper LIB fills the image centre |
| An entry of the class table | `id`, `label` and `color` are all required; Caliper LIB fills a colour from the categorical palette when the script gives none |
| `normals` of a mesh | (V, 3) in every space |
| An empty class table, an empty batch | Well formed: they empty the table or remove the part |
| The keys `first` and `rows` of `split` | Objects from array names to rows, so that a piece names the arrays it holds; the stamp is not cut and travels whole in every piece |
| The order of the pieces | A sender sends piece 0 first; a receiver drops a piece of an update whose piece 0 it does not hold |
| CURSOR | A missing or wrong `request_id` or `op` is `bad_request`; any other field of the op, `bad_cursor`. Fields of other ops are dropped, and `count` of `step` is required |
| A decimated series | `indices` and `decimated` go together, and its values need not follow `start` one by one |
| `static` and stamps | A static item carries no stamp section; TEXT and CONTROLS carry no `static`, as drafted |
| TIME_STATE | `cursor` is always written, null while following and stamp text otherwise |
| An empty snapshot | Carries no section at all |
| An image's `opacity`, `scale` and `cloud` when absent | The scene view takes the scene model's defaults: opacity 1, or 0.5 for segmentation; scale 1 for f32 depths and 0.001 for u16; a cloud for a depth image under a camera |

## Timelines, Text Logs, Bar Charts and Recordings

The codecs build these parts of the draft; no producer or server sends them yet. Each belongs to a capability, which a peer announces as it does `scenes`:

| Capability | Adds |
|---|---|
| `timelines` | The VIEWS key `timelines`; the section `stamp` of SNAPSHOT and CONTROLS, `stamp` or `stamps` of SERIES and TEXT; the key `static` of SNAPSHOT |
| `text` | The kind `text` (7) and TEXT |
| `bars` | The kind `bars` (8), a SNAPSHOT of that kind |
| `recordings` | RECORDINGS, RECORDING_OPEN, CURSOR and TIME_STATE; the session key `recording`; the VIEWS key `recorded`; the keys `cursor` and `empty` of data messages; the SERIES section `indices` and key `decimated` |

### Timelines and Stamps

VIEWS may carry `timelines`: 2 to 8 objects `{"name", "kind", "epoch"}` in the order of their ids, `name` 1 to 63 bytes of `[A-Za-z0-9_.-]` and unique, `kind` `sequence` or `time`, `epoch` an optional boolean; the first two are `step` (sequence) and `time` (time). A stamp has one value per timeline. SNAPSHOT and CONTROLS take the section `stamp` i64 (T,); SERIES takes `stamps` i64 (M, T), a stamp per value, or `stamp` i64 (T,) for all; a SNAPSHOT with `static` true carries no stamp. **Stamp text**, where a stamp value travels in metadata, is decimal digits with an optional leading minus, at most 20 characters, within the signed 64-bit range, such as `"1763200000123456789"`.

### `text`

New lines of a text log of kind 7, appended as a series' values are. `section_count` is 3, or 4 with a stamp section.

| Key or section | Meaning |
|---|---|
| `sub`, `view`, `stamp_ns` | As in SERIES |
| `start`, `total` | The absolute index of the first line, and the lines appended so far; `start` + M ≤ `total` |
| `dropped` | Optional: lines before `start` that the sender no longer held |
| `cursor` | Optional, as in the scene messages |
| Section `text` u8 (L,) | The lines' UTF-8 bytes, concatenated |
| Section `offsets` i32 (M + 1,) | Line i is bytes `offsets[i]` to `offsets[i + 1]`; `offsets[0]` is 0, `offsets[M]` is L, and they never decrease |
| Section `levels` u8 (M,) | 0 debug, 1 info, 2 warning, 3 error |
| Section `stamps` i64 (M, T) or `stamp` i64 (T,) | The lines' stamps |

M is 1 to 65,536, and each line is at most 4,096 bytes of valid UTF-8. A text log's data view is a series': `{"budget": {"max_backfill": n}}`.

### Bar Charts

A bar chart is a SNAPSHOT of kind 8 with the section `data` f32 (B,), 1 ≤ B ≤ 4,096, and optionally the section `colors` u8 (B, 4), `labels`, exactly B strings of at most 63 bytes, `horizontal`, a boolean, and `stats` `[min, max]`. Its data view is a histogram's with exactly one kept dim of at most 4,096 positions, and its payload axis is that dim.

### Split Updates

An update whose frame would pass 64 MiB travels as 2 to 16 SCENE_UPDATE pieces with the same `path`, `part`, `seq` and stamp and the key `split`:

| `split` key | Meaning |
|---|---|
| `update` | The producer's counter of split updates in the scene |
| `piece`, `pieces` | This piece's number from 0, and the number of pieces, 2 to 16 |
| `first` | An object from the name of each array section of this piece to the first row of the whole array that it holds |
| `rows` | Piece 0 only: an object from the name of each array of the update to its rows |

A piece holds a contiguous range of rows of some arrays, cut along their first axis, and the stamp whole; piece 0 carries the part's keys. The codecs check a piece's `split` and that `first` names exactly its arrays, and put the update together: `SplitUpdates` in Python and TypeScript and `wire::SplitUpdates` in C++ allocate each array when its first rows arrive, refuse a whole update over 512 MiB (`CALIPER_NET_MAX_UPDATE_BYTES`), overlapping rows, a piece twice or a piece that differs from piece 0, and check the whole update as an unsplit one once every piece and row has arrived. A newer update of the same entity part discards an incomplete one, and at most 2 incomplete updates are held per subscription, the oldest going first. `split_update` in Python divides an update into pieces.

### Recording Messages

| Message | Keys |
|---|---|
| RECORDINGS | `recordings`: at most 1,024 objects with `id` (1 to 63 bytes of `[A-Za-z0-9._-]`), `title`, `script`, `bytes`, `items`, `created_unix_ms` and `ended_unix_ms` (integers or null), `complete`, and `timelines`, each `{"name", "kind", "epoch", "min", "max"}` with `min` and `max` as stamp text |
| RECORDING_OPEN | `request_id` (uint32) and `recording`, a recording's id |
| CURSOR | `request_id`, `op` (`seek`, `play`, `pause`, `step` or `follow`); `seek` takes `timeline` and `value` (stamp text); `play` takes `timeline`, and optionally `speed` (1/16 to 64), `pace` (`recorded` or `even`) and `loop` (`[start, end]` as stamp text, start ≤ end, or null); `step` takes `timeline` and `count` (−1,000 to 1,000, not 0) |
| TIME_STATE | `session`, `generation` (uint32), `state` (`following`, `paused` or `playing`), `timeline`, `cursor` (stamp text, null exactly while following), `ranges` (an object from timeline names to `[min, max]` stamp text), `live`; optionally `request_id`, `speed`, `pace` and `loop` |

A session in SESSIONS may carry `recording`, `{"id", "seekable", "complete"}`. A view in VIEWS may carry `recorded`, at most 64 resolved data views. SNAPSHOT, SERIES, TEXT and the scene messages may carry `cursor`, the generation of the TIME_STATE they answer; a SNAPSHOT with `empty` true carries no section; SERIES may carry the section `indices` i64 (M,), the absolute index of each value of a decimated series, with the key `decimated`, the bucket size.

## GPU Programs and Interventions

An agent writes a GPU program in WGSL over a live tensor of a training script, and the script runs it on the GPU every few steps. The program's outputs come back as SCENE_UPDATE and SNAPSHOT messages that exist already. Two capabilities guard the additions:

| Capability | Adds |
|---|---|
| `programs` | PROGRAM and PROGRAM_STATE, and the SNAPSHOT key `cadence` |
| `interventions` | The CONTROLS key `interventions` and the CONTROL op `intervene` ([Run control messages](run-control-messages.md#interventions)) |

Caliper LIB announces both only while a `caliper.watch(model, optimizer)` is on, and only beside `controls`. A peer that connected before that call does not get them. A viewer sends PROGRAM only to a server that announced `programs`, and through a hub only to a session whose producer announced it. A PROGRAM needs the same permission as CONTROL: a server that does not allow control answers it with ERROR `forbidden`.

The wire protocol says that no message carries code. PROGRAM is the one exception. Its WGSL runs only on the GPU, after the sandbox checks of the script, with each tensor bound read-only, and never on the CPU.

### `program`

PROGRAM installs, replaces or removes one program. A `set` carries one section, `wgsl`, of dtype u8 and shape (B,): the source as UTF-8, 1 to 65,536 bytes. A `remove` carries no section.

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer (uint32) | Chosen by the viewer, echoed by PROGRAM_STATE or ERROR |
| `op` | string | `set` or `remove` |
| `id` | string | 1 to 55 bytes of `[a-z0-9_-]` that start with a letter, so that the view `program.<id>` fits a view name |
| `declaration` | object | `set` only: the program declaration without `wgsl` |
| `scene` | string | `set` of a geometry output only: the scene name of the target panel |
| `sha256` | string | `set` only: the SHA-256 of the source, 64 lowercase hex digits |

The codecs check the parts of the declaration that size a buffer in the script, in this order: `output` with `kind` (`geometry` or `pixels`); for geometry, `part` (`points` or `lines`), `max_count` (1 to 1,000,000), `arrays` (unique names of `positions`, `colors`, `values` and `radii`, with `positions`) and an optional `entity` path; for pixels, `width` and `height` (1 to 2,048) and an optional `format` (`rgba8unorm`); then `inputs` (at most 8 objects `{var, source}`, `source` 1 to 255 bytes), `limits` (`time_ms` above 0 and at most 50, `memory_mb` 1 to 512, `every` 1 to 10,000), `workgroups` (three integers from 1 to 65,535) and `entry`. The script checks the hash against the source and refuses a mismatch with ERROR `bad_request`. Its sandbox checks the rest before any run.

### `program_state`

PROGRAM_STATE carries no section. The script sends one after each PROGRAM, with its `request_id`, and one without `request_id` after a change of a state and at most twice a second while a program runs. A viewer that attaches also gets one, so that it can remove the programs it no longer wants.

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer (uint32), optional | The PROGRAM it answers |
| `step` | integer, optional | The script's step |
| `step_ms` | number, optional | The base step time in milliseconds |
| `programs` | array of objects | At most 16 status items with unique ids, in the order of installation |

A status item has the keys of the status item of `program.status` without `host`: `id`, `kind` (`geometry` or `pixels`) and `state` (`waiting`, `review`, `compiled`, `running`, `refused` or `stopped`) are necessary. `every` and `every_min` (1 to 10,000), `last_pass_ms`, `last_cost_ms`, `step_ms`, `overhead_share` and `passes_per_s` (numbers of at least 0), `last_step`, `runs`, `memory_bytes` and `memory_limit_bytes` (integers from 0 to 2^53 − 1), `error` (`{code, message, line, column}`) and `sha256` (empty or 64 hex digits) are optional and null when unknown. A script that cannot load the producer runtime answers with a refused item of code `program.unavailable`.

A server refuses a PROGRAM with a non-fatal ERROR that names it in `in_reply_to` and `request_id`:

| Code | Cause |
|---|---|
| `unsupported` | The viewer or the producer did not announce `programs`, or the session has ended |
| `forbidden` | The server does not allow control, or the token may only watch |
| `not_attached` | The viewer of a hub is not attached to a session |
| `bad_request` | The message breaks this page, or `sha256` is not the hash of the source |

A hub maps the `request_id` of a PROGRAM as it maps the one of a CONTROL, and keeps the latest PROGRAM_STATE without `request_id` for a viewer that attaches later. The script removes every program when no viewer with `programs` stayed connected for 5 seconds, and the app sends each program again after it attaches.

### Snapshot Cadence

The script runs the programs, the snapshots of the subscribed views and the chunks of the watched series at the passes of one cadence, which keeps their overhead inside a share of the training time: 2 % while a viewer subscribes, 4 % for programs alone. To a peer with `programs`, a SNAPSHOT taken at a pass may carry the key `cadence`, an object with these members:

| Key | Type | Meaning |
|---|---|---|
| `every` | integer | Steps between two passes of the cadence |
| `passes_per_s` | number or null | Passes that took something, a second, over the last 8 |
| `step_ms`, `pass_ms` | number or null | The measured time of a step without a pass, and the overhead of a pass |
| `share` | number or null | The measured share of the training time that the passes take |
| `budget` | number | The share that the cadence keeps to, 0.02 while a viewer subscribes |
| `limit` | string | `budget` when the passes come less often than the views ask, `floor` for a pass of cheap views on their own, `rate` when the views' own rate sets the pace |

A reader that does not know the key ignores it.

### Program Limits

| Constant | Value | Meaning |
|---|---|---|
| `CALIPER_NET_MAX_PROGRAMS` | 16 | Programs of a producer, and status items of PROGRAM_STATE |
| `CALIPER_NET_PROGRAM_ID_MAX` | 55 | Bytes of a program id |
| `CALIPER_NET_PROGRAM_SOURCE_MAX` | 65,536 | Bytes of the section `wgsl` |
| `CALIPER_NET_PROGRAM_STATE_RATE_HZ` | 2 | Unrequested PROGRAM_STATE messages a second, at most |
| `CALIPER_NET_PROGRAM_MAX_INPUTS` | 8 | Inputs of a declaration |
| `CALIPER_NET_PROGRAM_MAX_COUNT` | 1,000,000 | The largest `max_count` of a geometry output |
| `CALIPER_NET_PROGRAM_SIDE_MAX` | 2,048 | The largest `width` and `height` of a pixels output |
| `CALIPER_NET_PROGRAM_TIME_MS_MAX` | 50 | The largest `time_ms` |
| `CALIPER_NET_PROGRAM_MEMORY_MB_MAX` | 512 | The largest `memory_mb` |
| `CALIPER_NET_PROGRAM_EVERY_MAX` | 10,000 | The largest `every`, and the largest cadence |
| `CALIPER_NET_PROGRAM_WORKGROUPS_MAX` | 65,535 | The largest count of each of the three `workgroups` |
| `CALIPER_NET_TENSOR_NAME_MAX` | 255 | Bytes of an input's `source` |
| `CALIPER_NET_POINT_MAX` | 255 | Bytes of an intervention point |
| `CALIPER_NET_MAX_POINTS_LISTED` | 4,096 | Points in the CONTROLS key `interventions` |
| `CALIPER_NET_MAX_INTERVENTIONS` | 64 | Interventions in the CONTROLS key `interventions` |

## Programmable Graphics

An agent declares a whole graphic as a piece: a manifest and WGSL files with a derive half, which runs in the training script, and a draw half, which runs in the viewer. A program of an instrument is an instance of a piece. The capability `resources` adds the instance form of PROGRAM and the message RESOURCE, which carries the arrays that the derive half publishes. Caliper LIB announces `resources` only beside `programs`, under `caliper.watch(model, optimizer)`. A viewer that draws pieces announces it in HELLO, and a hub announces it to producers and to viewers.

### Instance Form of `program`

A PROGRAM `set` that holds the key `piece` installs an instance of a piece. It has no `declaration`, `scene` or `sha256` and no section `wgsl`. A viewer sends it only to a producer that announced `resources`, else the producer or the hub answers with ERROR `unsupported`.

| Key | Type | Meaning |
|---|---|---|
| `piece` | object | The pin: `name` (`[a-z][a-z0-9_]*`, 1 to 32 bytes), `version` (1 to 1,000,000) and `hash` (`sha256:` and 64 lowercase hex digits) |
| `manifest` | object | The piece's manifest, which the receiver checks before it runs a half |
| `bind` | object, optional | At most 8 inputs, each a WGSL name with `live:<tensor>` (a name of 1 to 255 bytes) or `program:<id>/<resource>` |
| `params` | object, optional | 1 to 16 parameter values, as in the op `params` |
| `panel` | string, optional | The scene panel that the draw half writes, 1 to 63 bytes |
| `segments` | array, optional | At most 16 objects `{part, first_line, lines}` that give the lines of each lib module and of `common.wgsl` in the section `common`, with `first_line` from 1 |

The sections are UTF-8 text of dtype u8 and shape (B,), with the imports resolved. `common` holds the lib modules and `common.wgsl`, 1 to 589,824 bytes. `derive` and `draw` hold the halves, 1 to 65,536 bytes each. An instance carries `derive` or `draw` or both, and each receiver composes the prelude and the generated header itself.

The producer also sends this form to each viewer that announced `resources`, so that a viewer gets each draw half from the same place as its resources. It sends each installed instance with `request_id` 0 and without `derive` when the viewer attaches and after each install, and it sends PROGRAM `remove` with `request_id` 0 after a removal. A viewer does not answer these messages. PROGRAM_STATE gives an instance the status kind `piece`.

### `resource`

RESOURCE carries one version of one published resource of a program: the elements that the derive half wrote at one pass of the run's cadence. The producer sends it only to a viewer or a hub that announced `resources`. One message carries one resource, thus a large resource does not hold a small one back.

| Key | Type | Meaning |
|---|---|---|
| `program` | string | The program's id |
| `resource` | string | The resource's id, a WGSL name of 1 to 63 bytes that does not start with `__` |
| `step` | integer | The run's step at the pass, from 0 to 2^53 − 1: the step stamp of the version |
| `count` | integer (uint32) | The valid elements |
| `stride` | integer | The bytes of one element, a multiple of 4 from 4 to 256 |
| `width`, `height` | integer, optional | A texture's size, 1 to 4,096 each, both or neither; a texture has a stride of 4 and `count` equal to `width` times `height` |

The section `resource` is of dtype u8 and shape (B,), and B is exactly `count` times `stride`, at most 67,043,328 bytes. A counted buffer with no valid element has `count` 0 and an empty section. The codecs check the keys in the order of the table, then that `count` is at most 67,043,328 divided by `stride`, and only then the section's length, thus no product of two values from the network can wrap. A texture's bytes are `rgba8unorm` texels, one packed `u32` each, in row order.

### Resource Flow

A producer keeps at most one unsent RESOURCE for each resource of each program and connection, and a newer version replaces it, as with snapshots. A viewer that falls behind thus gets the newest version of each resource and loses the older ones. A hub forwards RESOURCE to each attached viewer that announced `resources`, with the same rule for each viewer, and it never sends one to another viewer. It forwards a viewer's PROGRAM to the producer as before, with its permission rules, and answers an instance with ERROR `unsupported` when the producer did not announce `resources`. It forwards a producer's PROGRAM to each attached viewer that announced `resources`. For a viewer that attaches later, the hub keeps each program's last draw half and the newest version of each of its resources, and sends them to that viewer after its WELCOME. A PROGRAM `remove`, or the draw half of another piece, drops what the hub keeps of the program. The same piece relayed again keeps its resources, thus a static resource, which the script publishes once, stays. The kept versions count toward the hub's budget of retained bytes, and a version over the budget is forwarded but not kept.

| Constant | Value | Meaning |
|---|---|---|
| `CALIPER_NET_MAX_RESOURCES` | 8 | Resources of one program |
| `CALIPER_NET_RESOURCE_MAX_BYTES` | 67,043,328 | Bytes of the section `resource`: 64 MiB less 64 KiB, thus a version fits one frame with its metadata |
| `CALIPER_NET_RESOURCE_STRIDE_MAX` | 256 | The largest `stride` |
| `CALIPER_NET_RESOURCE_SIDE_MAX` | 4,096 | The largest `width` and `height` of a texture |
| `CALIPER_NET_PIECE_NAME_MAX` | 32 | Bytes of a piece's name |
| `CALIPER_NET_PIECE_VERSION_MAX` | 1,000,000 | The largest version of a piece |
| `CALIPER_NET_PIECE_COMMON_MAX` | 589,824 | Bytes of the section `common`: eight lib modules and `common.wgsl` of 64 KiB each |
| `CALIPER_NET_PIECE_SEGMENTS_MAX` | 16 | Items of `segments` |

## Golden Frames

`tests/fixtures/wire_v3/v3_2` holds the frames of a 3.2 sender, with minor 2 in their header, which `gen_v3_2.py` writes for `generate.py`: HELLO and WELCOME with `scenes`, VIEWS with scenes and a starting camera, VIEWS with a scene whose keys are null (which mean the same as absent keys, so the scene takes the defaults), the resolved default scene data view, SCENE_UPDATE of each part and form, with and without stamps, the five pieces of a split update, SCENE_TRANSFORMS with one stamp, a stamp each and none, SCENE_CLEAR with and without `reset`, SCENE_CLASSES, VIEWS with timelines and recorded data views, SESSIONS with a recording, SNAPSHOT with a stamp, static, empty and of a bar chart, SERIES with a stamp per value (the draft's example frame), one stamp and decimated, CONTROLS with a stamp, TEXT, every recording message and CURSOR op, PROGRAM `set` of each output and `remove`, PROGRAM `set` of an instance of a piece with both halves and as a draw half alone, PROGRAM_STATE as an answer and on its own and of an instance, RESOURCE of a buffer, a struct buffer, a texture and an empty counted buffer, CONTROL `intervene` of each form, its CONTROL_REPLY, and CONTROLS with `interventions`. Each implementation decodes every frame to what its `.json` says and encodes it back to the same bytes, except `scene_update_points_future`, which carries a section no part has. `v3_2/invalid` holds frames every implementation refuses; its `expect.json` gives, for each, the reason and close code of a frame that is not well formed or the ERROR code of an invalid message.

This is `v3_2/scene_update_mesh.bin`: the surface of two triangles over four vertices, coloured by their values through viridis on the fixed range −4 to 2.

```text
0000  43 41 4c 57 03 00 02 00 70 00 00 00 c8 01 00 00
0010  48 01 00 00 03 00 00 00 30 00 00 00 18 00 00 00
0020  10 00 00 00 7b 22 63 6f 6c 6f 72 6d 61 70 22 3a
0030  30 2c 22 66 6c 61 67 73 22 3a 30 2c 22 70 61 72
0040  74 22 3a 22 6d 65 73 68 22 2c 22 70 61 74 68 22
0050  3a 22 73 75 72 66 61 63 65 22 2c 22 73 65 63 74
0060  69 6f 6e 73 22 3a 5b 7b 22 64 74 79 70 65 22 3a
0070  30 2c 22 6e 61 6d 65 22 3a 22 70 6f 73 69 74 69
0080  6f 6e 73 22 2c 22 73 68 61 70 65 22 3a 5b 34 2c
0090  33 5d 7d 2c 7b 22 64 74 79 70 65 22 3a 34 2c 22
00a0  6e 61 6d 65 22 3a 22 69 6e 64 69 63 65 73 22 2c
00b0  22 73 68 61 70 65 22 3a 5b 32 2c 33 5d 7d 2c 7b
00c0  22 64 74 79 70 65 22 3a 30 2c 22 6e 61 6d 65 22
00d0  3a 22 76 61 6c 75 65 73 22 2c 22 73 68 61 70 65
00e0  22 3a 5b 34 5d 7d 5d 2c 22 73 65 71 22 3a 33 39
00f0  2c 22 73 74 61 6d 70 5f 6e 73 22 3a 38 31 32 33
0100  34 35 36 37 38 39 33 36 32 33 2c 22 73 74 61 74
0110  73 22 3a 5b 2d 31 2e 35 2c 31 2e 30 2c 2d 31 2e
0120  30 2c 31 2e 30 2c 2d 31 2e 30 2c 31 2e 30 2c 2d
0130  30 2e 37 35 2c 30 2e 35 5d 2c 22 73 75 62 22 3a
0140  32 2c 22 76 69 65 77 22 3a 22 6c 61 6e 64 73 63
0150  61 70 65 22 2c 22 76 6d 61 78 22 3a 32 2e 30 2c
0160  22 76 6d 69 6e 22 3a 2d 34 2e 30 7d 00 00 00 00
0170  00 00 80 bf 00 00 80 bf 00 00 40 bf 00 00 80 3f
0180  00 00 80 bf 00 00 80 be 00 00 80 bf 00 00 80 3f
0190  00 00 00 3e 00 00 80 3f 00 00 80 3f 00 00 00 3f
01a0  00 00 00 00 01 00 00 00 02 00 00 00 01 00 00 00
01b0  03 00 00 00 02 00 00 00 00 00 c0 bf 00 00 00 bf
01c0  00 00 80 3e 00 00 80 3f
```

| Offset | Bytes | Field | Value |
|---|---|---|---|
| 0 | `43 41 4c 57` | `magic` | `CALW` |
| 4 | `03 00`, `02 00` | `major`, `minor` | 3, 2 |
| 8 | `70 00` | `type` | `0x0070`, scene_update |
| 12 | `c8 01 00 00` | `frame_len` | 456 |
| 16 | `48 01 00 00` | `meta_len` | 328 |
| 20 | `03 00` | `section_count` | 3 |
| 24 | `30 00 00 00`, `18 00 00 00`, `10 00 00 00` | `section_len` | 48 `positions`, 24 `indices`, 16 `values` |
| 36 | `7b 22` to `30 7d` | metadata | 328 bytes, ending at offset 364, then 4 bytes of padding |
| 368 | `00 00 80 bf` … | `positions` f32 (4, 3) | (−1, −1, −0.75), (1, −1, −0.25), (−1, 1, 0.125), (1, 1, 0.5) |
| 416 | `00 00 00 00` … | `indices` i32 (2, 3) | (0, 1, 2), (1, 3, 2) |
| 440 | `00 00 c0 bf` … | `values` f32 (4,) | −1.5, −0.5, 0.25, 1 |

The metadata, in canonical form:

```json
{"colormap":0,"flags":0,"part":"mesh","path":"surface","sections":[{"dtype":0,"name":"positions","shape":[4,3]},{"dtype":4,"name":"indices","shape":[2,3]},{"dtype":0,"name":"values","shape":[4]}],"seq":39,"stamp_ns":81234567893623,"stats":[-1.5,1.0,-1.0,1.0,-1.0,1.0,-0.75,0.5],"sub":2,"view":"landscape","vmax":2.0,"vmin":-4.0}
```
