# Wire protocol 3.2 draft

This page specifies the part of wire protocol 3.2 that is built: the capability `scenes`, the live view kind `scene`, the messages SCENE_UPDATE and SCENE_CLEAR with the parts points, lines and mesh, and the default data view of a scene. Everything else follows [Wire protocol v3](wire-protocol.md), whose frame layout, decoding order, terms and rules apply unchanged.

!!! note "Status"
    A draft. The minor version 2, the capability `scenes`, the kind 6 and the message types `0x0070` and `0x0072` are proposals that the owners of the protocol have not confirmed, so they may still be renumbered. They are one line each in a separate block at the end of `include/caliper/live_net.h` and of each codec. Senders keep writing minor 1 in their frames until 3.2 is confirmed; a peer decides by the capability `scenes`, never by the minor version. Every 3.1 message keeps its meaning and its encoding.

## What is built

| Piece | Built |
|---|---|
| Capability | `scenes` |
| Kind | `scene` (6), with the VIEWS key `scene` |
| Messages | SCENE_UPDATE with the parts `points`, `lines` (segments and strips) and `mesh`; SCENE_CLEAR |
| Data view | The default scene data view, resolved |
| Codecs | `python/caliper/_wire.py`, `src/host/net/wire.h` and `web/src/protocol/codec.ts`, with the golden frames of `tests/fixtures/wire_v3/v3_2` |
| Producer | `caliper.scene` in Caliper LIB on `caliper.serve()`, the Python and the native sender ([Caliper LIB](python-api.md#scenes)) |
| Hub | Forwards both messages to the viewers that announced `scenes` and keeps a scene's state for a late viewer ([Through a hub](#through-a-hub)) |
| Viewers | The Caliper app, `caliper-viewer` and the WebAssembly views announce `scenes` and draw scenes with one scene view; the TypeScript client announces it so that the `<caliper-view>` element and the viewer page hand a scene to the WebAssembly views |

Not built: timelines and stamps, the parts `arrows`, `boxes`, `image` and `camera`, lines of mode `edges`, entity transforms (SCENE_TRANSFORMS), class tables (SCENE_CLASSES), the dtype `u16`, split updates, labels, class ids, normals, texture coordinates, scene data views other than the default, the starting camera and scene events. [Versioning](#versioning) says what a receiver does when one of them arrives.

## The capability

A viewer announces `scenes` in HELLO when it draws scenes, and a producer and a hub announce it in WELCOME when they send them. A producer lists a view of kind 6 in VIEWS, and sends SCENE_UPDATE and SCENE_CLEAR, only to a peer that announced `scenes`; a SUBSCRIBE to a scene from another peer is answered with `unknown_view`. A hub announces `scenes` to the producers that connect to it. The key `static` of SCENE_UPDATE and SCENE_CLEAR comes with `scenes` alone, since a scene's state needs it while no timeline is built.

## The kind scene

A scene is one live view: a tree of entities, named by entity paths, in one 2-D or 3-D space. Its entry in VIEWS has kind 6, an empty `sources` list and the key `scene`:

| Key of `scene` | Type | Meaning |
|---|---|---|
| `space` | string | `"3d"` or `"2d"`. Default `"3d"` |
| `up` | string | The axis a 3-D view keeps upright when it opens: `"+x"`, `"-x"`, `"+y"`, `"-y"`, `"+z"` or `"-z"`. In 2-D, the direction of y on screen, `"-y"` (image pixels) or `"+y"` (a plot). Default `"+y"` in 3-D and `"-y"` in 2-D |
| `units` | string | A unit note for display, at most 15 bytes. Nothing is converted. Default `"m"` |

An **entity path** is 0 to 16 segments of `[A-Za-z0-9_.-]{1,63}` joined by `/`, at most 255 bytes, with no segment `.` or `..`. The empty path is the root, the scene's own space. An entity holds at most one batch of each part, and logging a part again at the same path replaces it.

## The scene data view

A subscription to a scene takes the default data view only: `data_view` absent, `{}`, or the resolved default itself. It resolves to:

```json
{"budget":{"max_extent":1024,"max_items":100000,"max_points":100000,"max_vertices":1000000},"entities":{"exclude":[],"include":["**"]}}
```

Any other data view is answered with `bad_data_view`. The producer applies the budget to points and to line segments by stride sampling, with every array of the batch sampled alike, and then sends `axes` with the batch, as a points snapshot does ([Payload axes](wire-protocol.md#payload-axes)); line strips and meshes travel whole.

## Message types

| Id | Name | Direction | Sections | Answer |
|---|---|---|---|---|
| `0x0070` | `scene_update` | producer to viewer or hub, hub to viewer | The arrays of one part | |
| `0x0072` | `scene_clear` | producer to viewer or hub, hub to viewer | none | |

### scene_update

One part of one entity. `section_count` is 1 to 4.

| Key | Type | Meaning |
|---|---|---|
| `sub` | integer | The subscription |
| `view` | string | The scene's name |
| `seq` | integer | The producer's counter of the scene's batches; it grows with every new batch of the scene, and a batch sent again in a scene's state keeps its own |
| `stamp_ns` | integer | The producer's monotonic clock at the call, as in SNAPSHOT |
| `path` | string | The entity path |
| `part` | string | `points`, `lines` or `mesh` |
| `static` | boolean, optional | A static item: the scene's setup, which the producer retains. Default false |
| `state` | boolean, optional | Part of a scene's state for a new subscription. Default false |
| `axes` | array, optional | For a batch the budget sampled, the payload axis of its item dim, as in SNAPSHOT |
| `items` | string, optional | The item space of 3.1 whose items the batch's rows are; only to a peer with `selection` |

The sections and keys of each part, with N items and D = 3 in a 3-D scene or 2 in a 2-D scene:

| Part | Sections | Keys |
|---|---|---|
| `points` | `positions` f32 (N, D); `colors` u8 (N, 4); `values` f32 (N,); `radii` f32 (N,) | `color`, `radius`, `colormap`, `flags`, `vmin`, `vmax`, `stats` |
| `lines` | `positions` f32 (V, D); `strip_lengths` i32 (S,) for strips; `colors` u8 (V, 4) or (S, 4); `values` f32 (V,) or (S,) | `mode`, `color_per`, `color`, `width`, `colormap`, `flags`, `vmin`, `vmax`, `stats` |
| `mesh` | `positions` f32 (V, D); `indices` i32 (T, 3); `colors` u8 (V, 4); `values` f32 (V,) | `color`, `colormap`, `flags`, `vmin`, `vmax`, `stats` |

| Key | Type | Meaning |
|---|---|---|
| `mode` | string | Lines, required: `segments`, where vertices 2k and 2k + 1 form segment k and V is even, or `strips`, where `strip_lengths` gives the vertices of each strip, each length at least 1 and their sum V |
| `color_per` | string, optional | Lines: `vertex` (default) or `strip` (strips only), what the rows of `colors` and `values` belong to |
| `color` | `[r, g, b, a]`, optional | One colour for every item, from 0 to 255 |
| `radius` | number, optional | Points: one radius for every point, in scene units; above 0. `radii` gives one per point. Without either, points are discs of a few pixels |
| `width` | number, optional | Lines: the width in pixels, above 0. Default 2 |
| `colormap`, `flags`, `vmin`, `vmax` | | As in SNAPSHOT: the colormap of `values` and its range |
| `stats` | array, optional | `[values min, values max, x min, x max, y min, y max, z min, z max]`, computed on the producer's device; z is 0 in 2-D and the values entries are 0 without `values`. A non-finite entry is null |

`positions` is required, and the other sections are optional. An item's colour comes from `colors`, else from `values` through the colormap, else from `color`, else from the viewer's default; `colors` and `values` together are refused. A mesh without `indices` takes each three vertices as a triangle, so V is a multiple of 3. Index values are never checked by a receiver: an index out of range draws as the last vertex, by the index-clamp contract of the geometry service. N, V and T are at most 4,194,304 and S at most 1,048,576.

### scene_clear

| Key | Type | Meaning |
|---|---|---|
| `sub`, `view`, `seq`, `stamp_ns`, `static`, `state` | | As in SCENE_UPDATE |
| `path` | string | The entity to clear |
| `recursive` | boolean, optional | Its subtree too. Default false |
| `parts` | array of strings, optional | Only these parts. Default all |
| `reset` | boolean, optional | Empty the whole scene, static items included. Only with `path` `""`, and only as the first message of a scene's state |

The producer of Caliper LIB sends SCENE_CLEAR only as the reset that opens a scene's state.

## The state of a scene for a new subscription

When a subscription to a scene opens, the producer first sends the scene's state, each message with `state` true: SCENE_CLEAR with `path` `""` and `reset` true, then a SCENE_UPDATE of every batch it holds, in the byte order of their paths and, within a path, in the order points, lines, mesh. It holds every static item, which it copies once at the call even while nothing is attached, and the newest batch it still has of each other entity part, which exists only for the batches it sent while some subscription was open. A viewer draws an entity whose batch was not held from its next update.

## Flow control

The rules of [Flow control](wire-protocol.md#flow-control) hold, with these additions:

- Per subscription, a sender keeps at most one pending SCENE_UPDATE for each entity path and part, which a newer one replaces.
- A scene's state goes before any later message of that scene, and a SCENE_CLEAR is never dropped.
- `max_rate` applies to each entity part of a scene.
- Among pending data, series go first, then scene messages, then snapshots. The native sender of Caliper LIB hands its scene messages to its connection as control frames instead, at most two waiting at once, so that they still coalesce in the producer while a viewer drains slowly.

## Through a hub

The hub announces `scenes` to producers and viewers. It removes the views of kind 6 from the VIEWS it sends to a viewer without `scenes`, and refuses that viewer's SUBSCRIBE to one with `unknown_view`. It forwards SCENE_UPDATE and SCENE_CLEAR to the viewers of the producer subscription, rewriting `sub` and sharing the payload sections, with the flow control above per viewer. For a late viewer it keeps, per producer subscription, the newest batch of each entity part, a reset clearing what it kept, and sends them as the scene's state, opened by its own SCENE_CLEAR with `reset`. What it keeps counts toward its budget of retained bytes.

## Limits

| Constant | Value | Meaning |
|---|---|---|
| `CALIPER_NET_PATH_MAX` | 255 | Bytes of an entity path |
| `CALIPER_NET_PATH_SEGMENTS_MAX` | 16 | Segments of an entity path |
| `CALIPER_NET_UNITS_MAX` | 15 | Bytes of a scene's unit note |
| `CALIPER_NET_MAX_ENTITIES` | 1,024 | Entities of a scene, which the producer enforces |
| `CALIPER_NET_MAX_BATCHES` | 2,048 | Batches of a scene, which the producer enforces |
| `CALIPER_NET_MAX_BATCH_POINTS` | 4,194,304 | Points, line vertices, mesh vertices and triangles of a batch |
| `CALIPER_NET_MAX_BATCH_ITEMS` | 1,048,576 | Line strips of a batch |
| `CALIPER_NET_SCENE_DEFAULT_MAX_ITEMS` | 100,000 | Default `max_items` of a scene data view |
| `CALIPER_NET_SCENE_LIMIT_MAX_ITEMS` | 1,048,576 | Largest `max_items` |
| `CALIPER_NET_SCENE_DEFAULT_MAX_VERTICES` | 1,000,000 | Default `max_vertices` |
| `CALIPER_NET_SCENE_LIMIT_MAX_VERTICES` | 4,194,304 | Largest `max_vertices` |
| `CALIPER_NET_SCENE_DEFAULT_MAX_EXTENT` | 1,024 | Default `max_extent` |
| `CALIPER_NET_SCENE_LIMIT_MAX_EXTENT` | 8,192 | Largest `max_extent` |

A batch must also fit in one frame of 64 MiB, since split updates are not built; Caliper LIB refuses a larger one at the call.

## Versioning

A receiver of this subset handles the rest of the 3.2 draft by the rules of [Versioning](wire-protocol.md#versioning):

| What arrives | What the receiver does |
|---|---|
| A part of the draft that is not built: `arrows`, `boxes`, `image` or `camera` | Ignores the message and answers ERROR `unsupported` |
| A piece of a split update (the key `split`) | Ignores the message and answers ERROR `unsupported` |
| Lines of mode `edges` | Ignores the message and answers ERROR `unsupported` |
| A section or key it does not build: `stamp`, `class_ids`, `labels`, `normals`, `uvs`, `texture` | Ignores the section or key, and reads the rest of the message |
| An unknown part, a bad entity path, a section of the wrong dtype or shape, strip lengths that do not add up, or a batch over a limit | Answers ERROR `bad_request` |

A 3.1 peer announces no `scenes`, so it receives no scene view and no scene message; a 3.1 decoder finds the 3.2 frames well formed, with their types unknown to it.

## Golden frames

`tests/fixtures/wire_v3/v3_2` holds the frames of a 3.2 sender, with minor 2 in their header, which `gen_v3_2.py` writes for `generate.py`: HELLO and WELCOME with `scenes`, VIEWS with a scene, the resolved default scene data view, SCENE_UPDATE of each part and form, and SCENE_CLEAR with and without `reset`. Each implementation decodes every frame to what its `.json` says and encodes it back to the same bytes, except `scene_update_points_stamped` and `scene_update_mesh_normals`, which carry sections this subset ignores. `v3_2/invalid` holds frames every implementation refuses; its `expect.json` gives, for each, the reason and close code of a frame that is not well formed or the ERROR code of an invalid message.

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
