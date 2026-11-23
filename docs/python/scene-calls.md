---
kind: reference
figure: none
---

# Scene Calls

The scene calls of the `caliper` package log 3-D and 2-D geometry into a scene that every viewer of the network route draws.

## Scene Model

A scene is a live view of 3-D or 2-D geometry: a tree of entities, named by
entity paths such as `"world/cloud"`, each holding batches of points, lines,
arrows, boxes, a mesh or an image, or a pinhole camera, and each placed by an
entity transform. Scenes travel on the network route only, over
`caliper.serve()` and a hub that it connects to, with the
[Wire Protocol 3.2 draft](../reference/wire-protocol-3-2.md). With only the live session or
the in-process window open, the first scene call warns once and scene calls
do nothing there. The Caliper app and `caliper-viewer` attached to the stream
([Viewer](viewer.md#scene-windows)),
the notebook widget and the viewer page draw a scene with the same scene view
on the GPU; in a browser it needs WebGPU and Caliper's WebAssembly views, and
without them the scene's plate shows a notice in its place. The scene view
draws points, lines (segments and strips) and meshes so far; for the other
parts and arguments it says what it does not draw yet
([Versioning](../reference/wire-protocol-3-2.md#versioning)).

```python
scene = caliper.scene("world", up="+z", view_camera={"target": (0, 0, 0.5), "azimuth": 30.0})
scene.classes({0: "floor", 1: "obstacle"})
scene.points("room/cloud", xyz, class_ids=(xyz[:, 2] > 0.05).int(), static=True)
scene.transform("rig/left", translation=(-0.06, -2.5, 1.2), rotation=forward_y, static=True)
scene.camera("rig/left", width=640, height=480, focal=525.0, static=True)
scene.image("rig/left/rgb", frame)                      # in the camera's pixel space
scene.lines("graph/edges", pos, indices=edge_index.T, values=weight, cmap="magma")
caliper.widget("world", "scene", height=480)
```

[](){#caliper.scene}
## `scene`

::: caliper.scene
    options:
      show_root_heading: false
      show_root_toc_entry: false

Like every view call, the first call of a script opens the live session when no route is open. The view shows `units`
on its scale bar, with an SI prefix for m, s and g, and `units=""` shows the number alone. The starting camera
travels in the scene's entry in VIEWS, and a view moves to a new one only until a person moves its camera.
`caliper.remove(name)` removes the scene and what it retains, and `caliper.widget(name, "scene")` shows it in a
notebook.

## Primitives

Each call logs one part of the entity at `path` and replaces the batch that
part held. The empty path is the scene's root, which may hold batches too. A
call returns `True` when a route took a snapshot and `False` when none did:
no viewer of the scene is attached, or the batch's rate limit dropped it.

The arguments of each call are in the [`Scene`][caliper.Scene] reference below.

The rows of `colors` and `values` are N for points and arrows, V, S or E for
lines, and V for a mesh. `colors` and `values` together raise `ValueError`,
and so do a shape, dtype or row count that does not fit, a bad path, an odd
number of segment vertices, strip lengths that do not add up (when they are
host data), `indices` with `strip_lengths`, `uvs` without `texture`, rotations
of 2-D boxes, a mesh without indices whose vertex count is not a multiple of 3,
a batch over the limits (4,194,304 points, vertices, edges or triangles,
1,048,576 arrows, boxes or strips, images of 8,192 pixels on a side, 1,024
entities and 2,048 batches in a scene), and a part the entity cannot hold: a
primitive at an entity that holds a camera, 3-D data under a camera, whose
children live in its 2-D pixel space, and in a 3-D scene an image that is not
under a camera logged before it. These checks read host metadata only, so
they run on every call, attached or not. An argument of the wrong type raises
`TypeError`. Strip lengths in a device tensor are checked once their copy
reaches host memory, on the network thread: a batch whose lengths do not add
up to V, or hold a length below 1, is dropped there with a warning, and the
call has already returned `True`. `items=`, `blend=` and the instance arrays of
a mesh are not built yet and raise `NotImplementedError`, for example
`items= is not built yet (picking and linked selection)`.

### Call Cost

With no viewer of the scene attached, a call costs its
checks and a dictionary update and does no device work, unless the batch is
static. With a viewer attached and the batch due, the snapshot is built once
on the arrays' own device: the casts, colours as RGBA, 2-D box corners turned
into centres, the per-axis bounds and the value range (sent as `stats`; a
depth image's range leaves out zeros and non-finite depths), and the budget
of the default data view, which samples more than 100,000 points or segments
by stride. Each array is then copied into pinned host memory without waiting,
one `torch.mps.Event` is recorded, and the network thread reads the copies
once the event has completed. Nothing reads the device on the training
thread. A batch whose arrays pass one frame of 64 MiB travels as a split
update of up to 16 pieces; a batch over 512 MiB raises `ValueError`.

### Piece Form

The Caliper app, `caliper-viewer` and Caliper's WebAssembly views announce the capability `scene_pieces`. To such a viewer, a batch of points and a color image travel in the piece form: the network thread writes the arrays from the host copies as the resources of the part's library piece, `points@1` or `image@1`, and sends them as RESOURCE before the update that names them ([Piece Form](../reference/wire-protocol-3-2.md#piece-form)). It writes each batch once for every such viewer, and only when the batch is sent, thus a batch that a newer one replaced is never written, and the training thread does the same work as before. The viewer draws the received bytes with its own library piece. A viewer without the capability, and the other parts and forms, get the arrays as before. `CALIPER_SCENE_SHAPES=builtin` in the script's environment sends every batch with its arrays to every viewer.

### Static Items

A batch logged with `static=True` is copied once at the call,
even with nothing attached, and the producer retains it, a newer static batch
of the same entity part replacing it. The rate limit never drops one. A
batch logged without `static` is snapshotted only while a viewer watches the
scene; the producer keeps the newest one it sent of each entity part.

### Rate Limits

Each batch, an entity path and part, has its own rate limit
(`rate`, else the scene's, else 60 per second), and a viewer's `max_rate`
applies to each entity part of its subscription, the newest batch replacing
one still waiting.

## Cameras, Entity Transforms, Class Table and Clears

The arguments of `camera`, `transform`, `classes` and `clear` are in the [`Scene`][caliper.Scene] reference below.

A camera sits at an entity of its own, posed by that entity's transform; the
entity holds no primitive, the root holds no camera, and a camera is never
under another. Its children live in its pixel space, where images, 2-D points,
lines, arrows and boxes sit. An entity transform of tensors is composed on
their device and copied without waiting, as an array is; one of numbers is
composed on the host. A clear removes the entity's primitives, camera and
transform, and its subtree's with `recursive`; static parts stay unless the
clear is static too, and `clear("", recursive=True)` empties the scene's live
data. These four calls are small: the producer keeps them whether or not a
viewer is attached, the rate limit never drops them, a clear applies at once
to what it keeps, and each returns `True`. A viewer's pending entity
transforms merge into one message, the newest of each entity.

## Late Viewers

When a viewer subscribes to a scene, it first receives the
scene's state: a reset, the class table, the static and then the other entity
transforms, then every camera, every static batch and the newest batch of
every other entity part, in the byte order of their paths. It draws an entity
whose batch was not kept from its next call.

[](){#caliper.Scene}
## `Scene`

::: caliper.Scene
    options:
      show_root_heading: false
      show_root_toc_entry: false
