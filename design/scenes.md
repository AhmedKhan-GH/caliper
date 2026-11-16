# Scenes

This page plans scenes: a new kind of live view that draws a tree of entities in one 2-D or 3-D space, with points, lines, arrows, boxes, meshes, images and pinhole cameras, each entity placed by an entity transform. It defines the scene model, the Caliper LIB calls that log to a scene, what each route copies when a scene is viewed on the same machine, how the native view draws a scene through the scene core and graphics layer of the [renderer core](renderer-core.md), and how picks and selections in a scene reach the script. It also plans the two small views of the same sprint, the text log and the bar chart. Timelines, stamps and recordings are planned in [Recordings](recordings.md), the messages on the wire in [Wire protocol 3.2](wire-protocol-3-2.md), and the browser's pipeline in [Web geometry](web-geometry.md).

## What is built and what is planned

Nothing on this page is built. It rests on these built pieces, and on one planned piece, the renderer core:

- **The points live view** of Caliper LIB, `caliper.points`. A snapshot is one copy on torch's own stream into a CPU-readable shared slot, and the native view draws the slot in place. The points primitive of a scene is this view with many entities.
- **Protocol v2**, the shared-memory session between a script and the Caliper app (Live streaming (`docs/explanations/live-streaming.md`)), and **wire protocol 3.1** on the sprint 2 branch, with data views, events, controls and linked selection.
- **The geometry service**, `caliper.geometry.v1` to `v1_3` (`docs/references/service-geometry-v1.md`): points, lines and triangles, depth views, textured meshes and instanced transforms, drawn in place from imported allocations on Metal and on Vulkan with CUDA. The scene service for applets replaces it, as the user decided ([The scene service for applets](#the-scene-service-for-applets)).
- **Sprint 3's spec format**, which reserves the panel primitive name `scene` and the catalog kind `recording`.
- **The renderer core**, planned and built first in sprint 4: a graphics layer that each backend implements once, and a scene core that every view uses.

## The scene model

### Scenes and spaces

A scene is one live view of a session, of the new kind `scene`. The kind id 6 is a proposal, because sprint 2 owns the kind table. A script creates a scene by name; the name follows the rules of every view name, 1 to 63 bytes and unique in the session.

| Property | Values | Default | Meaning |
|---|---|---|---|
| space | `"3d"` or `"2d"` | `"3d"` | The space every entity of the scene is drawn in |
| up | 3-D: `"+x"`, `"-x"`, `"+y"`, `"-y"`, `"+z"`, `"-z"`. 2-D: `"-y"` or `"+y"` | `"+y"` in 3-D, `"-y"` in 2-D | In 3-D, the axis the orbit camera keeps upright when the view opens; the person can switch it in the view ([The up-axis switch](#the-up-axis-switch)). In 2-D, the direction of y on screen |
| units | Text, at most 15 bytes | `"m"` | A note shown on the view's [scale bar](#the-scale-bar) and in hovers. Nothing is converted |

The 3-D space is right-handed. The built points view and the browser's points camera both build their view matrix with a right-handed look-at, keep +y upright and project with clip z in [0, 1], which is the geometry service's convention. A scene with the default up axis therefore shows a point cloud exactly as `caliper.points` does. Robotics data usually has z up and passes `up="+z"`. The up axis changes only the camera, never the logged numbers.

### The up-axis switch

The up axis is a setting of each view, not a fixed property of the data. Every 3-D view, the scene view and the 3-D points view alike, has a control in its header that switches between Y up and Z up. The scene's `up` argument sets the mode in which a view starts, and the default is `"+y"`.

| Aspect | Behaviour |
|---|---|
| The control | Two segments in the view's header, labelled "Y up" and "Z up", beside the fit button, with the active one highlighted. A scene that starts with another axis, such as `"-y"`, shows that axis as a third segment, so that the person can return to it. The key U switches while the view has focus. A 2-D view has no such control |
| What changes | Only the view's camera basis: the axis that the orbit camera keeps upright and around which a horizontal drag turns. The logged arrays, the entity transforms, the bounds, picks, selections and every row id stay as they are. The producer is told nothing and takes no new snapshot, and nothing is written to a recording |
| The camera when switching | The orbit keeps its target, its distance and the position of its eye. The azimuth and elevation are computed again from that eye position in the new basis, so the picture only rolls about the line of sight until the new up axis is upright. An eye within 1° of the new axis is held at an elevation of 89°, so that the orbit stays defined. The change is immediate and is not animated |
| What follows the switch | The ground grid, the axes mark in the corner and the direction of a fit. A fit keeps the current mode |
| Looking through a camera | The pinhole's pose fixes the picture, so the switch has no effect while the person looks through a camera; the control stays active and its mode applies on return to the orbit |
| Remembered per view | The mode is kept with the view's other settings, such as its camera and its display options, and is restored when the view is shown again. A remembered mode wins over the scene's `up`; resetting the view returns to the scene's `up` |
| In a notebook and a browser | The widget and the viewer page show the same control. Two views of one scene may be in different modes |

The switch is tested headless at three levels. A test of the scene core switches an orbit from Y up to Z up and back, and checks that the target, the distance and the eye position are unchanged within 10<sup>-6</sup>, that the view matrix's up vector is the new axis projected perpendicular to the line of sight, that an eye on the new axis is held at 89°, and that a round trip restores the first view matrix. A test of the view logs a point one unit along +z above a target at the origin, switches to Z up, and checks that the point is drawn above the target on screen, that the batch's sequence number and bytes are unchanged, that a pick returns the same entity and row id in both modes, and that the mode is restored after the view is closed and shown again while a reset returns to the scene's `up`. A capture row draws one fixture in both modes against references, natively and in headless Chromium.

The 2-D space has x to the right and, by default, y down, the convention of image pixels, so that an image, its boxes and its segmentation line up in pixel coordinates. `up="+y"` gives the convention of a plot and of the 2-D points view.

Positions are 32-bit floats. Coordinates far from the origin lose precision, so a script subtracts a local origin before it logs world-scale coordinates.

### The scale bar

Every scene view shows a scale bar in the bottom-right corner of its picture, so that a person reads distances in the scene's `units` while zooming. The axes mark keeps the bottom-left corner.

| Aspect | Behaviour |
|---|---|
| Look | A horizontal bar 2 logical pixels thick with end ticks 6 pixels tall, 12 logical pixels in from the right and bottom edges, with its length above it in the view's small text, such as "0.5 m". Both are drawn in the view's text colour over a translucent band of its background, so that they stay legible over any content |
| Length | The longest of 1, 2 and 5 times a power of ten, in scene units, whose bar fits in a fifth of the picture's width. The bar is therefore between 8 % and 20 % of the width, and its length and label change as the person zooms |
| Label | The number, then the unit as the script gave it. For the units the view recognises, m, s and g with the prefixes n, µ (or u), m, c and k, a number outside 0.01 to 1,000 takes the SI prefix, in steps of 1,000, that brings it between 1 and 1,000, a prefixed unit such as cm being converted through its base: 0.005 m reads "5 mm", 2,000 m "2 km" and 0.0002 s "200 µs", while 0.5 m stays "0.5 m". Any other unit, such as "px" or "tokens", follows the number as given, in plain decimals from 0.001 to 999,999 and in the form "5e-7" outside them |
| No unit | With `units=""`, or the words "arbitrary" or "a.u.", which say that distances have no unit, the bar shows the number alone |
| 2-D | Exact: a pixel spans the view's half height divided by half its height in pixels. A 2-D view that draws its axes titles them with the same unit, as "x (m)" and "y (m)" |
| 3-D | Perspective makes the scale vary with depth, so the bar is measured at the depth of the orbit target, the centre of the picture, and its label says so: "0.5 m at target". There a pixel spans 2 · distance · tan(22.5°) divided by the picture's height in pixels, for the orbit's 45° field of view. Nothing nearer or farther is to scale, and there is no orthographic projection in this sprint |
| Through a pinhole | Hidden, because the picture has no single depth to measure at: the image plane is drawn at a distance chosen for display, and the projected scene lies at every depth. The bar returns with the orbit |
| Up-axis switch | Unaffected: the switch keeps the target and the distance, so the bar keeps its length and its label |
| Hiding it | A "Scale bar" item in the view's display menu, on by default and remembered per view with its other display options. The bar also hides while the picture is narrower than 160 logical pixels and before the scene has bounds |
| Native and browser | Overlay lines and text that the view layer draws with ImGui over the picture, never a scene draw, so the native view and the browser's WebGPU view, which is the same view layer, draw it alike. The TypeScript fallback draws it on its canvas by the same rule ([Web geometry](web-geometry.md)) |

Nothing new travels: `units` is already in the scene's entry in VIEWS ([Wire protocol 3.2](wire-protocol-3-2.md)), and a recording keeps it with the views list.

The tests are headless. One shared table of zoom levels, from 10<sup>-9</sup> to 10<sup>9</sup> scene units per pixel at widths from 160 to 4,000 pixels, checks the chosen length (1, 2 or 5 × 10<sup>k</sup>, the longest within a fifth of the width), the bar's share of 8 % to 20 % and the label with its prefixes, in C++ and in TypeScript. A test of the view checks, over a grid of orbit cameras, that the bar's length in pixels equals within one pixel the projected length of that distance laid across the picture at the target; in 2-D, that it equals the distance divided by the units per pixel within half a pixel; that an up-axis switch leaves the label and the length alone; that the bar hides through a pinhole and returns after it; and that hiding it survives a close and reopen. Capture rows draw a 3-D and a 2-D fixture with the bar natively and in headless Chromium, and the fallback's bar in Canvas 2D. The work is about 0.75 builder days natively and 0.25 in the browser, for the fallback's bar and the captures.

### Entities and entity paths

An entity is a node of the scene's tree, named by its entity path: 1 to 16 segments, each matching `[A-Za-z0-9_.-]{1,63}`, joined by `/`, at most 255 bytes in all. The segments `.` and `..` are refused, because they would read as relative steps. Paths are relative to the scene, and the root, the empty path, is the scene's own space; it may hold primitives but no entity transform or camera. Logging to a path creates its missing parents as empty entities.

| An entity holds | At most | Set by |
|---|---|---|
| An entity transform | One | `transform` |
| A camera | One | `camera` |
| A batch of one primitive type | One per type: points, lines, arrows, boxes, mesh, image | The primitive calls |

Logging a part again at the same path replaces it: the latest wins, as for every live view. An entity that holds a camera holds no primitive, so that every primitive has one unambiguous space; the camera's children hold the image and the 2-D drawing. Depth and segmentation images are forms of the image primitive, so a colour image and its segmentation are two sibling entities under the camera. Entities draw in the byte order of their paths, opaque batches first and translucent ones after.

### Entity transforms

An entity transform maps the entity's space into its parent's. It is given in one of two forms.

| Form | Values | Meaning |
|---|---|---|
| Parts | `translation`, 3 f32; `rotation`, a unit quaternion in the order x y z w; `scale`, 1 or 3 f32. Each part is optional | A point p maps to T · R · S · p: scaled, then rotated, then translated |
| Matrix | One 4 x 4 f32 matrix | Stored column-major, the layout of a mesh's instance poses |

The world pose of an entity is the product of the transforms down its path, M<sub>1</sub> · M<sub>2</sub> · … · M<sub>n</sub>, from the root's child to the entity. A missing transform is the identity. The view normalises a quaternion and hides an entity whose world pose is not finite, saying so in the entity's hover. In a 2-D scene the transform takes a translation of 2 numbers, a rotation of one angle in radians, counter-clockwise when y is up, and a scale of 1 or 2 numbers, and is stored as the same 4 x 4 matrix.

A transform may be given as Python numbers, a numpy array or a tensor on any device. A device tensor is copied into a slot on its device like any other array, and the view reads its 64 bytes from the slot's CPU side ([What each route copies](#what-each-route-copies)), so the script never waits for the device.

### Cameras and the camera's pixel space

A camera is a pinhole held by an entity:

| Field | Meaning |
|---|---|
| `fx`, `fy` | Focal lengths in pixels |
| `cx`, `cy` | The principal point in pixels; by default the image centre |
| `width`, `height` | The image size in pixels |
| convention | The camera frame: `"rdf"` (default: x right, y down, z forward) or `"rub"` (x right, y up, looking along −z) |
| plane distance | How far in front of the camera its image plane is drawn, in scene units; by default a twentieth of the scene's bounding radius |

The camera entity's own transform is the camera's pose. The view draws the camera as a frustum: four edges from the centre to the corners of the image plane, and the plane's rectangle.

The children of a camera entity live in the camera's 2-D pixel space: x right and y down from the top-left corner of the image, so that pixel (row i, column j) covers [j, j + 1) × [i, i + 1). A pixel point (u, v) lies on the image plane at ((u − cx) / fx · d, (v − cy) / fy · d, d) in the camera's frame, d being the plane distance. Images, depth and segmentation images and 2-D points, lines, arrows and boxes may sit under a camera; 3-D data there is refused. A child may hold a 2-D entity transform, in pixels. To attach 3-D content to a camera's pose, a script gives the pose to a parent entity and puts the camera and the 3-D content beside each other under it.

A person can look through a camera. The view then takes the camera's pose and builds its projection from `fx`, `fy`, `cx`, `cy`, the width and the height, with near and far planes from the scene's bounds, and letterboxes to the camera's aspect. The pixel-space children are drawn flat over the projected 3-D content, a colour image covering it unless its opacity is below 1.

A depth image under a camera is also drawn as a point cloud, back-projected through the pinhole: pixel (u, v) with depth z, in scene units after its scale, becomes the camera point ((u + 0.5 − cx) / fx · z, (v + 0.5 − cy) / fy · z, z). Zero and non-finite depths are skipped. Lens distortion and camera models other than the pinhole are not offered; a script undistorts its images first.

### The class table

A scene's class table maps a class id, 0 to 65,535, to a label of at most 63 bytes and an RGBA colour, with at most 4,096 entries. Points and boxes with `class_ids` and segmentation images use it, and hovers show the label. An id without an entry draws in colour id mod 16 of the fixed 16-colour palette that categorical points use today, with its number as its label. The table is logged as static by default, and logging it again replaces it whole.

### Primitives

N is the number of items and D the dimension of positions: 3 in a 3-D scene, 2 in a 2-D scene and under a camera. Every array is a tensor and stays on its device for local viewing. During the snapshot, positions, radii and values of any float dtype are cast to f32, and indices and class ids of any integer dtype to i32, on the device.

| Primitive | Arrays | Optional | Defaults | Limit per batch |
|---|---|---|---|---|
| points | `positions` f32 (N, D) | `colors`; `values` f32 (N,) with a colormap and range; `class_ids` i32 (N,); `radii` f32 (N,) or one, in scene units; `labels`; `blend` | Discs of 5, 3 or 2 pixels by count, as the points view draws, in flat light grey | N ≤ 4,194,304 |
| lines | `positions` f32 (V, D) | `strip_lengths` i32 (S,), which makes strips, or `indices` i32 (E, 2), which makes [index-pair edges](#index-pair-edges); `colors` per vertex (V, 4), per strip (S, 4), per edge (E, 4) or one; `values` f32 per vertex, per strip or per edge, with a colormap and range; `width` in pixels; `blend` | Segments: vertices 2k and 2k + 1 form segment k. Width 2 pixels | V ≤ 4,194,304, S ≤ 1,048,576, E ≤ 4,194,304 |
| arrows | `origins` f32 (N, D), `vectors` f32 (N, D) | `colors`; `values` with a colormap; `width` in pixels | Width 2 pixels; the head is a fifth of the arrow's length | N ≤ 1,048,576 |
| boxes | `centers` f32 (N, D), `half_sizes` f32 (N, D) | `rotations` f32 (N, 4), quaternions, 3-D only; `colors`; `class_ids`; `labels`; `fill`; `width` | Edges only, 2 pixels wide. 2-D boxes are axis-aligned and may be given as corners `(x0, y0, x1, y1)`, converted on the device | N ≤ 1,048,576 |
| mesh | `positions` f32 (V, D) | `indices` i32 (T, 3); `normals` f32 (V, 3); one of `colors` (V, 4), `values` (V,) with a colormap, or `uvs` f32 (V, 2) with a `texture` image; `instance_poses` f32 (N, 4, 4), with `instance_colors` (N, 4) or `instance_values` (N,) through the mesh's colormap | Without indices, each three vertices form a triangle. Without normals, faces are shaded flat. Flat light grey. Without instance poses, one instance at the entity's pose | V ≤ 4,194,304, T ≤ 4,194,304, N ≤ 1,048,576 |
| image, colour | u8 (H, W, 3 or 4) or (H, W); floats in [0, 1] are converted on the device | `opacity` | Opacity 1; a missing alpha is 255 | H, W ≤ 8,192 |
| image, depth | f32 or u16 (H, W) | `scale` (scene units per stored unit); a colormap and range; `cloud` | Scale 1 for f32 and 0.001 for u16, which reads millimetres as metres. `cloud` true under a camera | H, W ≤ 8,192 |
| image, segmentation | Integer class ids (H, W) | `opacity` | Opacity 0.5 | H, W ≤ 8,192 |
| image, encoded | JPEG or PNG bytes, one video frame each | `opacity` | | 16 MiB |

A colour is u8 (N, 3) or (N, 4), floats in [0, 1] converted to u8 on the device, or one colour for all as a tuple of 3 or 4 integers from 0 to 255 or floats from 0 to 1. An item's colour comes from `colors`, else from `values` through the colormap, else from its class id, else from the default. `colors` and `values` together are refused. A colormap is one of the three of `caliper.show`, `viridis`, `magma` and `rdbu`, and an open end of the range is computed on the device, as for points today.

`blend` is `"normal"`, the default, or `"additive"`. Additive points and lines are drawn after the opaque batches with the depth test on and depth writes off, adding their colour, so that a dense cloud or a bundle of trajectories brightens where it overlaps and the order of its items does not matter. A mesh with `instance_poses` is drawn once per pose, each pose applied before the entity's world pose, and its normals are transformed by the cofactor of each pose's upper 3 × 3, which lights any invertible pose correctly. `instance_colors` or `instance_values` colour each instance and take the place of the mesh's own colours. Both are additions to the shared contracts, made when the scene service replaced the geometry service, because each is a general feature of a scene model that the applets' draws needed. [Wire protocol 3.2](wire-protocol-3-2.md) carries them as a `blend` key of the points and lines parts and as instance arrays of the mesh part.

Labels are host strings: one per item or one for all, at most 4,096 per batch and 63 bytes each. Where an item has no label, its class label is shown. The view draws at most 512 labels at once, the nearest to the camera first.

Index values cannot be checked without reading the device. The geometry service's index-clamp contract holds for scenes as well: an index out of range draws the last vertex, which gives a wrong-looking but defined picture, never an out-of-bounds read. A hardware index fetch cannot clamp, so the scene core clamps a batch's indices once per update, in a small compute pass into an index buffer the layer owns. The same pass gathers the ends of [index-pair edges](#index-pair-edges).

In a 3-D scene an image of any form sits under a camera, because its place comes from the intrinsics. In a 2-D scene it sits at its entity's origin, one scene unit per pixel.

### Clearing and static items

A clear removes an entity's primitives, transform and camera from the clear's stamp on; a recursive clear also removes those of its whole subtree. A later log at the path brings the part back, and clearing the root recursively empties the scene. A clear affects stamped data only, and static data stays unless the clear is itself static, because static items are a scene's setup, such as a floor mesh or a class table, and a reset at the end of an episode should not remove them.

A static item has no stamp and is present at every position of every timeline. A newer static item replaces an older one, and at a cursor where stamped data of the same part exists at or before the cursor, the stamped data wins (latest-at).

The producer retains every static item: it makes one device-side copy at the call, even with nothing attached, so that a viewer that attaches later receives it. Entity transforms, cameras, class tables and clears are retained too, because they are small. A stamped primitive batch logged while nothing is attached is not retained, as for every view today. A late viewer lists such an entity as waiting and draws it from its next update. A script therefore logs what does not change as static.

### Limits

| What | Limit | Reason |
|---|---|---|
| Scenes in a session | Up to 64, shared with the session's other views | The control table of protocol v2 |
| Entities in a scene | 1,024 | Each entity costs at least one draw and one per-draw uniform, and 1,024 draws a frame stay cheap on every backend, the browser's included. Many objects of one kind belong in one batch |
| Batches in a scene | 2,048 | Sizes the scene's directory in shared memory |
| Points, line vertices, index-pair edges, mesh vertices and triangles in a batch | 4,194,304 | 48 MiB of positions per slot, and three slots per batch; for edges, the gathered positions stay within WebGPU's 128 MiB storage binding |
| Boxes, arrows, line strips and mesh instances in a batch | 1,048,576 | Each draws several quads or triangles |
| Labels | 4,096 per batch, 63 bytes each, 512 drawn at once | Labels are host strings drawn with ImGui every frame |
| Class table | 4,096 entries, ids 0 to 65,535 | Fits a small lookup texture |
| Image side | 8,192 pixels | The default 2-D texture limit of WebGPU |
| Encoded frame | 16 MiB | A quarter of a wire frame |

## The text log

The text log is a live view of the new kind `text`, proposed id 7. It keeps lines of text. Each line has a text of at most 4,096 bytes of UTF-8 (a longer one is cut and marked), a level, `debug`, `info`, `warning` or `error`, and a stamp. Like a series, it is append-only: a viewer receives the lines it lacks, never a replacement. Lines are host data and cost no device work, so the producer retains the last 4,096 lines of each text log whether or not anything is attached, and a late viewer receives them first. The view scrolls with the newest line, filters by level and searches by substring. When a recording is open, a click on a line moves the cursor to its stamp. In the shared-memory session a text log is one segment with one slot that is never claimed, as for a series: a byte ring of the last 4,096 lines and an index of their offsets, levels and stamps.

## The bar chart

The bar chart is a live view of the new kind `bars`, proposed id 8: one vector of at most 4,096 values with a label per bar, replaced whole by each snapshot, as a histogram is. The values are a tensor and stay on its device; the labels are host strings of at most 63 bytes. Bars rise from zero, so negative values point down, and a hover shows the label and the value. With named dimensions, the values offer data views as a histogram's do, with one kept dimension, the bar dimension. A pick reports `bar`, `row_id` and `value`, and a select a `[start, stop]` range of bars; an item space on the bar dimension links them to other views. In the shared-memory session it has three slots, each a (B,) f32 payload and a host region for the labels.

## Caliper LIB

### scene

```python
scene = caliper.scene(name="scene", *, space="3d", up=None, units="m", rate=None, data_view=None,
                      view_camera=None)
```

Return the scene `name`, creating it on the first call. Like every view call, the first call of a script opens the live session when no route is open.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `"scene"` | The view's name; non-empty, no `##`. A view of another kind with this name is replaced, as for every view |
| `space` | `str` | `"3d"` | `"3d"` or `"2d"` |
| `up` | `str` or `None` | `None` | The up axis in which a view of the scene starts; see [The up-axis switch](#the-up-axis-switch). `None` is `"+y"` in 3-D and `"-y"` in 2-D |
| `units` | `str` | `"m"` | The unit note, which the [scale bar](#the-scale-bar) shows |
| `rate` | `float` or `None` | `None` | The default rate limit of each batch, as in Caliper LIB's rate limit rules |
| `data_view` | `dict` or `None` | `None` | The scene's default data view, in [Data views of a scene](#data-views-of-a-scene) |
| `view_camera` | `dict`, `str` or `None` | `None` | The camera on which a 3-D view of the scene opens, in [The starting camera](#the-starting-camera) |

A second call with the same name returns the same object and updates `up`, `units`, `rate`, `data_view` and `view_camera`. A new `up` changes the mode in which views start and in which a reset ends; it does not override a mode that a person chose in a view, and a new `view_camera` behaves in the same way. It raises `ValueError` for a bad name, space, up axis or unit note, and for a `space` that differs from the scene's. `caliper.remove(name)` removes the scene, and `caliper.widget(name, kind="scene")` shows it in a notebook.

### The starting camera

`view_camera` sets the camera on which every 3-D view of the scene opens, so that a demo or an applet opens on a good angle without a person arranging it. It takes one of two forms.

| Form | Value | Meaning |
|---|---|---|
| Orbit | A `dict` with any of the keys below | The orbit camera of the scene core, its angles defined against the scene's `up` |
| Camera entity | A `str`, an entity path | The view opens looking through the camera of that entity |

| Key | Type and unit | Default | Meaning |
|---|---|---|---|
| `target` | 3 numbers, in scene units | The centre of the fit's bounds | The point the orbit turns around and looks at |
| `distance` | A number above 0, in scene units | The fit's distance, which frames the bounding sphere | From the target to the eye |
| `azimuth` | Degrees, any finite number | 45 | The angle around the up axis, defined below |
| `elevation` | Degrees, from −89 to 89 | 28.6479, which is 0.5 radian | The angle of the eye above the plane perpendicular to the up axis |

The eye is at target + distance · e, where e = cos(elevation) · (cos(azimuth) · a + sin(azimuth) · b) + sin(elevation) · u. Here u is the scene's `up` axis, a is the first of +x, +y and +z that is perpendicular to u, and b = a × u. With `up="+y"`, azimuth 0 puts the eye on the +x side and azimuth 90° on the +z side, which is the points view's convention, so the defaults open a scene exactly as `caliper.points` opens a cloud. With `up="+z"`, azimuth 0 is the +x side and 90° the −y side. The field of view is the orbit's fixed 45°, and the near and far planes follow from the bounds, as for every orbit. Keys that are given are used as given. Keys left out are filled from the fit once the scene has bounds, so a script may set only the angles and let the distance frame whatever it logs, and an empty dict is the fit at the default angles.

`caliper.scene` raises `TypeError` for a value that is neither a dict nor a string and for a key of the wrong type. It raises `ValueError` for an unknown key, a number that is not finite, a distance not above 0, an elevation outside −89° to 89° (the bound the up-axis switch keeps the orbit within), a path outside the entity path grammar, and any `view_camera` in a 2-D scene, whose fit already frames it whole. Whether the named entity holds a camera cannot be known at the call, because a script may log the camera later; if it holds none when a view opens, the view opens on the fit and says so in its header.

`view_camera` is a starting value only, like `up`. It is where a view starts and where its reset ends. Once a person moves the camera, by orbiting, panning, zooming, fitting or looking through a camera, the view remembers its own camera with its other settings, and that camera wins when the view is shown again. Resetting the view returns to `view_camera`, or to the fit when there is none. A second `caliper.scene` call with a new `view_camera` replaces the starting camera: a view whose camera the person has not moved since it opened or was reset moves to the new one at once, and a moved camera stays where the person left it until its next reset, which goes to the new one. `view_camera=None` leaves the starting camera as it is, and `view_camera={}` returns it to the fit.

The angles are always read against the scene's `up`. In a view that the person has switched to another mode, the starting camera follows the rule of [The up-axis switch](#the-up-axis-switch): the eye position and the target stay, and only the picture rolls. A reset returns to the scene's `up` and to `view_camera` together. The camera-entity form opens the view through that entity's pinhole, with its pose at the cursor. The up-axis switch has no effect there, as for any pinhole; leaving the pinhole goes to the orbit of the fit, and a reset looks through the camera again.

The starting camera is small host data that travels with the scene's description: on the network it is a key of the scene's entry in VIEWS ([Wire protocol 3.2](wire-protocol-3-2.md)), and in the shared-memory session it sits in the header page of the scene's directory segment. A recording keeps it with the views list, so a recorded scene opens on the same camera. A spec panel sets the same starting camera through its option `view_camera`, in the same two forms, which this page proposes to sprint 3. Nothing travels the other way: a person's camera movement is never sent to the script, and there is no camera event.

The tests are headless. A test of the scene core builds, for both forms, both up modes and a grid of angles, the view matrix of a look-at from target + distance · e to the target within 10<sup>-6</sup>, and fills the keys left out from a fit. A test of Caliper LIB checks every refusal without a window. A test of the view opens a scene with a `view_camera` and checks that a point logged at the target is drawn at the view's centre and picked there; that a camera the person moved survives a second call with another `view_camera` and a close and reopen; that a reset returns to `view_camera`, and to the fit without one; that a switched view keeps the eye position and the target within 10<sup>-6</sup>; and that the camera-entity form opens through the pinhole and falls back to the fit, with its note, when the entity holds no camera. A capture row draws one fixture from a `view_camera` against a reference, natively and in headless Chromium.

The work is about 1.5 builder days natively (the argument and its checks in Caliper LIB, the starting state and the reset in the scene core and the view, and the tests) and 0.5 in the browser, where the viewer reads the key from VIEWS and the capture row runs in Chromium. It needs from the scene core only the azimuth convention above and a stored starting state for reset, and from recordings nothing beyond keeping the views list.

Not in this sprint: preset views, an orthographic projection, following an entity, saved viewpoints, linked cameras and a fly mode, which [Sprint 5](sprint-5.md) plans as its viewport step, apart from the fly mode, which it leaves out.

### The scene's calls

Every call below takes an entity path first, logs one part of that entity and returns what `caliper.show` returns: `True` if at least one route took a snapshot, `False` when every route skipped it. Every call but `classes` takes `static`, and the primitive calls also take `rate`:

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `static` | `bool` | `False` | Log with no stamp; see [Clearing and static items](#clearing-and-static-items) |
| `rate` | `float` or `None` | `None` | This batch's rate limit; `None` takes the scene's. Transforms, cameras, class tables and clears are never dropped by a rate limit: the newest value always replaces the stored one |

Each call raises `ValueError` for a path outside the grammar, a shape, dtype or device that does not fit, a part the entity cannot hold (a primitive at a camera entity, a camera at an entity with primitives, 3-D data under a camera, an image outside a camera in a 3-D scene) and a limit that is exceeded, and `TypeError` for an argument of the wrong type. Unlike the checks of `show`, these run on every call, because they read host metadata only (shapes, dtypes and devices), never a value on the device. With nothing attached a call costs these checks and a dictionary update, estimated at 2 µs, and no device work, except that a static batch is copied once on its device.

```python
scene.points(path, positions, *, colors=None, values=None, cmap="viridis", vmin=None, vmax=None,
             class_ids=None, radii=None, labels=None, items=None, blend="normal", static=False, rate=None)
scene.lines(path, positions, *, indices=None, strip_lengths=None, colors=None, values=None,
            cmap="viridis", vmin=None, vmax=None, color_per=None, width=2.0, blend="normal", static=False,
            rate=None)
scene.arrows(path, origins, vectors, *, colors=None, values=None, cmap="viridis", vmin=None,
             vmax=None, width=2.0, static=False, rate=None)
scene.boxes(path, centers=None, half_sizes=None, *, corners=None, rotations=None, colors=None,
            class_ids=None, labels=None, fill=False, width=2.0, items=None, static=False, rate=None)
scene.mesh(path, positions, indices=None, *, normals=None, colors=None, values=None, cmap="viridis",
           vmin=None, vmax=None, uvs=None, texture=None, instance_poses=None, instance_colors=None,
           instance_values=None, static=False, rate=None)
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `positions`, `origins`, `vectors`, `centers`, `half_sizes`, `normals`, `uvs`, `strip_lengths`, `rotations`, `class_ids`, `radii` | tensor | | The arrays of [Primitives](#primitives). `radii` may also be one number |
| `indices` | tensor, i32 or i64 | `None` | Lines: (E, 2) index pairs, in [Index-pair edges](#index-pair-edges). Mesh: (T, 3) triangles |
| `colors` | tensor or tuple | `None` | Per item, per vertex, per strip or per edge, or one colour |
| `color_per` | `str` or `None` | `None` | Lines: `"vertex"`, `"strip"` or `"edge"`, what `colors` and `values` belong to. `None` reads it from their length, and a length that fits two of them raises `ValueError` |
| `values`, `cmap`, `vmin`, `vmax` | tensor, `str` or `int`, `float` | `None`, `"viridis"` | Values through a colormap; each open end of the range is computed on the device |
| `corners` | tensor (N, 4) | `None` | 2-D boxes as `(x0, y0, x1, y1)`, in place of `centers` and `half_sizes` |
| `labels` | `str` or list of `str` | `None` | One per item, or one for all |
| `fill` | `bool` | `False` | Boxes: also draw translucent faces |
| `width` | `float` | 2.0 | Line, arrow and box-edge width in pixels |
| `texture` | tensor (H, W, 3 or 4) | `None` | The mesh's texture, sampled at `uvs` |
| `items` | `str` or `None` | `None` | The item space of protocol 3.1 whose items the rows of this batch are, for linked selection |
| `blend` | `str` | `"normal"` | Points and lines: `"normal"` or `"additive"` |
| `instance_poses` | tensor (N, 4, 4) or (N, 16) | `None` | Mesh: one pose per instance, given in the usual row and column sense and stored column-major |
| `instance_colors`, `instance_values` | tensor (N, 4) or (N,) | `None` | Mesh: a colour or a value through `cmap` per instance |

```python
scene.image(path, image, *, opacity=1.0, static=False, rate=None)
scene.depth(path, depth, *, scale=None, cmap="viridis", vmin=None, vmax=None, cloud=None,
            static=False, rate=None)
scene.segmentation(path, class_ids, *, opacity=0.5, static=False, rate=None)
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `image` | tensor, or `bytes` | | A colour image, or the bytes of one JPEG or PNG frame, recognised by its signature |
| `depth` | tensor (H, W), f32 or u16 | | A depth image |
| `scale` | `float` or `None` | `None` | Scene units per stored unit; `None` is 1 for f32 and 0.001 for u16 |
| `cloud` | `bool` or `None` | `None` | Also draw the back-projected point cloud; `None` is true under a camera |
| `class_ids` | integer tensor (H, W) | | A segmentation image, coloured by the class table |
| `opacity` | `float` | 1.0, 0.5 | From 0 to 1 |

```python
scene.camera(path, *, width, height, focal=None, fov=None, principal=None, convention="rdf",
             plane_distance=None, static=False)
scene.transform(path, *, translation=None, rotation=None, scale=None, matrix=None, static=False)
scene.classes(table, *, static=True)
scene.clear(path="", *, recursive=False, static=False)
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `width`, `height` | `int` | | The camera's image size in pixels |
| `focal` | `float` or pair | `None` | `fx` and `fy`, or one value for both. Exactly one of `focal` and `fov` is given |
| `fov` | `float` | `None` | The vertical field of view in degrees, which sets `fx = fy` |
| `principal` | pair | `None` | `(cx, cy)`; `None` is the image centre |
| `convention` | `str` | `"rdf"` | `"rdf"` or `"rub"` |
| `plane_distance` | `float` | `None` | See [Cameras and the camera's pixel space](#cameras-and-the-cameras-pixel-space) |
| `translation`, `rotation`, `scale` | numbers, array or tensor | `None` | The parts of an entity transform. In 2-D, `rotation` is one angle |
| `matrix` | (4, 4) array or tensor | `None` | Given in the usual row and column sense and stored column-major. It excludes the parts; a call with none of them resets the transform to the identity |
| `table` | `dict` | | Class id to a label, or to `(label, colour)` |
| `recursive` | `bool` | `False` | Clear the subtree too |

### Index-pair edges

`scene.lines` also draws edges given as pairs of indices into `positions`, so that graph edges, bonds, skeletons and simulation meshes are drawn without repeating two positions for every edge. The form is PyTorch Geometric's `edge_index` transposed:

```python
scene.points("graph/nodes", pos, class_ids=data.y)
scene.lines("graph/edges", pos, indices=data.edge_index.T, values=edge_weight, cmap="magma")
```

| Aspect | Rule |
|---|---|
| Shape | `indices` is (E, 2), i32 or i64, and row k is edge k from vertex `indices[k, 0]` to vertex `indices[k, 1]`. An i64 array is cast to i32 on the device during the snapshot. A transposed view such as `edge_index.T` is taken as it is; the snapshot copies it into the slot |
| (2, E) | Refused with `ValueError` when E is not 2, with a message that names the fix: "indices has shape (2, 5); pass edge_index.T, an (E, 2) array". A (2, 2) array is read as two edges. Accepting both forms would make a two-edge graph the one array read either way; refusing (2, E) gives every array one reading, and a script that passes `.T` for every graph never meets the case |
| Excludes | `strip_lengths`: a batch holds segments, strips or edges, and both together raise `ValueError` |
| Colours and values | Per vertex, (V, 4) or (V,): taken at each end of an edge, each end's value mapped through the colormap, and the colours interpolated along the edge, as for segments and strips. Per edge, (E, 4) or (E,): one colour along the whole edge, so a value's colour is exact everywhere on it. When V equals E the length cannot say which is meant, and the call raises `ValueError` unless `color_per` is given |
| Index values | Not checked at the call, because that would read the device. An index out of range, a negative one included, draws as the last vertex, by the index-clamp contract that meshes follow |
| Drawing | The scene core's index-clamp pass gains a second entry point, the edge gather, run once per batch update. It clamps each pair and copies the positions of its two ends, and their per-vertex colour sources, into the streams that the segment draw already reads, so that edge k is drawn as segment k. Per-edge colours and values are bound as they arrived, once for each end. The line shader does not change, and the gather uses storage buffers in the compute stage only, which every backend and WebGPU's compatibility mode allow |
| Memory | The gathered streams take 2 · D · 4 bytes of positions per edge, 24 in 3-D and 16 in 2-D, and 8 bytes of colour sources when colours are per vertex, in buffers the layer owns, beside the producer's 8 bytes per pair. A batch of 1,000,000 edges in 3-D takes up to 32 MiB more on the viewer's GPU |
| Limit | E ≤ 4,194,304, the limit of vertices and triangles, which keeps the gathered positions within WebGPU's 128 MiB storage binding |
| Data view | `max_items` samples the edges by stride, with their per-edge colours and values. The positions and per-vertex arrays travel whole and count against `max_vertices`, as a mesh's do, and a batch over it is sent as its bounding box. Sampling the vertices instead would need their number on the host, which means waiting for the device |
| Picking | The item is the edge's index in the sent batch, and `row_id` its source index through the payload axes |

The tests are headless. Tests of Caliper LIB accept (E, 2) in i32 and i64 and a transposed view, and refuse (2, E) with the message naming `.T`, `indices` together with `strip_lengths`, and a colour length that fits two readings without `color_per`. A graph of known edges is drawn against a reference natively and in headless Chromium; an index out of range draws as the clamp defines; per-edge values through the colormap are exact at the edges' centres; a pick returns the edge's index; and the transposed `edge_index` of a small graph in PyTorch Geometric's form draws the same picture as the same edges written as pairs. The work is about 1.5 builder days natively, for the argument and its checks, the cast, the gather in the scene core and the three codecs, and 0.5 in the browser, for the gather's WGSL and the capture rows.

### text and bars

```python
caliper.text(message, name="log", level="info")
caliper.bars(values, name="bars", labels=None, colors=None, horizontal=False, rate=None, *,
             dims=None, data_view=None)
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `message` | `str` | | One entry; a message with line breaks is one entry of several lines |
| `level` | `str` | `"info"` | `"debug"`, `"info"`, `"warning"` or `"error"` |
| `values` | tensor (B,) | | 1 ≤ B ≤ 4,096; any dtype, cast to f32 on its device |
| `labels` | list of `str` or `None` | `None` | One per bar; `None` labels the bars by index |
| `colors` | tensor (B, 3 or 4) or one colour | `None` | Bar colours |
| `horizontal` | `bool` | `False` | Draw the bars from left to right |
| `name`, `rate`, `dims`, `data_view` | | | As for `caliper.histogram` |

`text` raises `ValueError` for an unknown level, and `bars` for a shape or label count that does not fit. None of the names `scene`, `text` and `bars` is taken in the `caliper` package today.

### Stamps on the timelines

Every call of a scene, and every `text` and `bars` call, is stamped at the call with the session's position on each timeline: `step`, which the script advances with `caliper.step`; `time`, the wall clock; and the timelines the script defines. The calls that define and advance timelines belong to the [Recordings](recordings.md) page. A call with `static=True` has no stamp. A snapshot carries the stamp of the call that made it, so a batch dropped by its rate limit leaves no stamp. Each snapshot keeps its monotonic `stamp_ns` for latency display, as today.

### A worked example

A robot simulation trains a policy and shows a static point cloud of the room, a stereo rig of two cameras with their frustums and images, and an arm of three links whose joints move every step.

```python
import math
import torch
import caliper

scene = caliper.scene("world", up="+z", units="m",
                      view_camera={"target": (0.0, 0.0, 0.5), "azimuth": 30.0, "elevation": 25.0})
scene.classes({0: "floor", 1: "obstacle"})

xyz = torch.rand(200_000, 3, device="mps") * torch.tensor([4.0, 4.0, 1.0], device="mps")
xyz[:, :2] -= 2.0
scene.points("room/cloud", xyz, class_ids=(xyz[:, 2] > 0.05).int(), static=True)

forward_y = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))   # camera z along world +y
for side, x in (("left", -0.06), ("right", 0.06)):
    scene.transform(f"rig/{side}", translation=(x, -2.5, 1.2), rotation=forward_y, static=True)
    scene.camera(f"rig/{side}", width=640, height=480, focal=525.0, static=True)

links = (0.5, 0.4, 0.3)
path = "arm"
for i, length in enumerate(links):
    path += f"/link{i}"
    scene.boxes(path, centers=[(0.0, 0.0, length / 2)], half_sizes=[(0.04, 0.04, length / 2)],
                static=True)

for step in range(10_000):
    angles, frames = sim.step(policy)      # three floats, two (480, 640, 3) u8 MPS tensors
    caliper.step(step)
    path, offset = "arm", 0.0
    for i, (a, length) in enumerate(zip(angles, links)):
        path += f"/link{i}"
        scene.transform(path, translation=(0.0, 0.0, offset),
                        rotation=(0.0, math.sin(a / 2), 0.0, math.cos(a / 2)))
        offset = length
    scene.image("rig/left/rgb", frames[0])
    scene.image("rig/right/rgb", frames[1])
```

The view opens looking at the arm from the front and to the right, 25° above the floor, at the distance that frames the room. Each link is a child of the one before, so its entity transform, a rotation about y at the end of its parent, composes down the path. The images sit under their cameras and appear on the image planes; looking through `rig/left` shows the left image with the room projected behind it.

## Keeping scene data on the device

### What each route copies

Caliper LIB offers each call to every open route, as it does for the other views. The device-side work of a snapshot is built once and shared by the routes that take it. A scene snapshot is one batch: the arrays of one primitive of one entity.

| Route | Metal with MPS | Vulkan with CUDA |
|---|---|---|
| An applet in the host, drawing its own tensors through the scene service | Zero-copy: the imported allocation is drawn in place | Zero-copy for tensors in the exportable pool, once the Vulkan backend of the graphics layer lands |
| Caliper LIB's in-process window or canvas | One copy on torch's stream into a Caliper-made shared `MTLBuffer` slot; the graphics layer imports the slot and draws it with no further copy | Not built: Caliper LIB runs on macOS only. Planned as on Metal, the slot being exportable device memory |
| The shared-memory session to the Caliper app | One copy on torch's stream into a slot of the session's shared memory; the app imports the slot and draws it with no further copy | Not built. Planned with the Windows session of [Windows parity](windows-parity.md): the directory in named shared memory and the slots in exportable CUDA memory, imported in the app through Vulkan external memory |
| The network: `caliper.serve()`, the hub | The data view is computed on the device and copied into pinned host memory without waiting; the bytes cross the socket, and the viewer uploads them into its own GPU memory | The same |

A scene from Caliper LIB is therefore never zero-copy in the strict sense of Zero-copy tensor visualization (`docs/explanations/zero-copy.md`). Each snapshot is one copy inside the GPU, on Apple Silicon inside the one unified memory, and that copy is what lets the training loop rewrite its tensors while a viewer draws. From the slot on, nothing is copied, nothing reaches host memory, and nothing waits: completion is polled through a `torch.mps.Event`, as for points. Zero-copy holds from the slot to the pixels, through the graphics layer's import. A CPU tensor is copied once into the slot, and on macOS a tensor on any other device is moved to the CPU first, one host copy, as for every view. A network viewer always receives copies.

Four kinds of data travel through host-visible memory because the viewer needs them on its CPU: entity transforms, cameras, the class table and labels. They are small. On Metal the slot is the same memory the CPU reads; on Vulkan with CUDA a device-given transform is copied asynchronously into a pinned host slot, and its completion is polled.

An image becomes a texture by one pass on the viewer's GPU, as heatmaps do today. An encoded frame is decoded on the viewer's CPU and uploaded, so it is never zero-copy.

### Growing the shared-memory session

Protocol v2 gives each view one segment of three slots, each slot one payload, an optional f32 aux region and a stats region. A scene needs several arrays per batch, many batches, and per-batch publishing, because entities change at different rates. Its proposed layout keeps the v2 slot protocol unchanged and applies it per batch.

| Piece | Name | Content |
|---|---|---|
| Control table row | | Kind 6, the scene's name, and in `segment` its directory segment; `slot_count` 0, `dtype` and `shape` unused |
| Directory segment | `/<session>-v<id>` | A header page with a `generation` seqlock and the scene's starting camera, then the entity table, the batch table and the class table, about 2.8 MiB |
| Arena segments | `/<session>-v<id>a<k>`, k from 0 to 15 | Page-aligned slots of batches, added as the scene grows; a batch's slots are reallocated when its size changes and freed once both viewer claims on them are released |

| Entity row (1,024, 512 bytes each) | Meaning |
|---|---|
| `path`, `parent` | The entity path, and the row of its parent |
| `flags` | Holds a transform, holds a camera, which parts are static, cleared |
| `transform` | A host-given entity transform, 16 f32, or the row of the batch that carries a device-given one |
| `camera` | `fx`, `fy`, `cx`, `cy`, width, height, convention, plane distance |
| `stamp`, `clear_stamp` | The stamps of its last change and its last clear |

| Batch row (2,048, 1 KiB each) | Meaning |
|---|---|
| `entity`, `primitive`, `form` | The entity's row; the primitive, 1 to 6 in the order of the primitives table, or 7 for a device-given entity transform; the image form, the line mode (segments, strips or edges), `fill` |
| `arrays` | Up to 8 descriptors: role, dtype, shape of up to 3 dimensions, and byte offset within the slot. A lines batch of edges adds the role `indices`, (E, 2) i32 |
| `slot_arena`, `slot_offset` | For each of the three slots, its arena segment and its offset |
| `slot_bytes`, `host_bytes` | The device-written part of a slot and its host-written part (labels) |
| Sync words | `seq`, `published_slot`, `viewer_reading_slot`, `viewer_prev_slot`, `dropped_client`, `viewer_seq`, `skipped_viewer`, with the meanings and ordering of a v2 view segment |
| Slot metadata | For each slot: `seq`, `stamp_ns`, the timeline stamp, `colormap`, `vmin`, `vmax`, flags, item count, and 256 bytes of stats written on the device: value range and per-axis bounds |
| `viewer_skip` | Written by the viewer: 1 while the person hides the entity, so that the producer takes no snapshot of the batch |

The protocol keeps version 2, under its own append-only rules: the kinds 6 to 8, a `features` word appended to the control segment, and `"features": ["scenes", "text", "bars"]` in the session file. A v2 viewer from before this sprint does not know kind 6. It must report an error for that view and draw the others, and a test against the sprint 2 viewer has to confirm that before the additions ship. One rule has to relax for this: today a reader refuses a struct smaller than its own, and a reader of the additions must instead read a smaller v2 struct as one without them.

### Data views of a scene

A scene offers a data view, so that the producer still reduces on its device and sends only the result. A subscription of the network names one, as for any view; the window and the shared-memory session use the scene's default, which by default sends every batch whole.

| Key | Form | Meaning |
|---|---|---|
| `entities` | `{"include": [...], "exclude": [...]}`, up to 16 filters each | A filter is a path (the entity alone), `<path>/**` (the entity and its subtree) or `**`. A batch is sent when an include filter matches its entity and no exclude filter does. Default: include `**` |
| `budget` | The keys below | Taken key by key, as the budget of the other kinds |

| Budget key | Applies to | Default over the network | Limit | How the producer meets it |
|---|---|---|---|---|
| `max_points` | Points, and the point cloud of a depth image | 100,000 | 1,000,000 | Stride sampling, with every array of the batch sampled alike, by the rule of the points data view |
| `max_items` | Boxes, arrows, line segments, index-pair edges | 100,000 | 1,048,576 | Stride sampling; a strip keeps every k-th vertex and its two ends, and edges keep their per-edge colours and values |
| `max_vertices` | Meshes, and the positions of index-pair edges | 1,000,000 | 4,194,304 | Not reduced: a mesh or an edge batch over the budget is sent as its bounding box, with a note in its hover |
| `max_extent` | Images | 1,024 | 8,192 | Colour by mean pooling; depth and segmentation by stride, so that every value sent is a real depth or class id |

Transforms, cameras, class tables and clears always travel whole. Encoded frames are not reduced. An excluded or hidden entity costs the producer nothing: no snapshot, no copy. Points sampled under a budget report picks in source indices through the payload axes, as points do today.

## Drawing a scene natively

### The scene core and the graphics layer

The native scene view is the scene core compiling entities into a draw list for the graphics layer, both planned in [Renderer core](renderer-core.md). It calls no backend, and it is the one drawing path for scenes: applets reach the same view through the scene service, which replaces the geometry service ([Applets](#applets)). The browser runs the same scene core over the layer's WebGPU backend, which [Web geometry](web-geometry.md) plans.

The points live view becomes a scene of one entity in the scene core. Today it calls the host renderer directly, `HostRenderer::geom_draw_primitives` with the host-internal draw record, which carries a raw 256-entry colour table that the public record lacks. Its camera is the right-handed orbit with +y up, a 45° field of view and clip z in [0, 1]. It picks on the CPU from the CPU-visible payload. Platform details reach into it in several places, which the layer's import capability replaces: `bind_buffer` in `src/host/live/live_views.cpp` has one branch for an imported `MTLBuffer`, one for a renderer that copies host memory (the browser's) and one that stages a CPU payload into a new Metal buffer and refuses on any other renderer; the Metal helpers in `live_views_metal.mm`; and the `mtl_buffer` field of the snapshot in `live_source.h`. The WebGPU renderer of the sprint 2 branch draws points only.

### From primitives to draws

Points and lines are instanced quads on every backend, so the 1-pixel line limit of the geometry service does not apply to scenes.

| Primitive | Draw of the graphics layer | Colour | Notes |
|---|---|---|---|
| points | An instanced quad per point, expanded in the vertex stage to a disc of its radius in scene units, or of a pixel size | Per-instance RGBA, a value through the colormap, a class colour, or flat | Depth tested and written in 3-D; the disc is cut in the fragment stage with a hard edge, so that points keep depth writes and exact colours; an additive batch is drawn after the opaque ones with depth writes off and additive blending |
| lines, segments | An instanced quad per segment: instance k reads vertices 2k and 2k + 1 as two instance attributes of one buffer (stride 24, offsets 0 and 12), expanded to `width` pixels | Per vertex or flat | Lines, arrows and box edges are drawn with a small constant depth offset toward the camera, so that an edge lying on a mesh wins over the surface. Additive lines are drawn as additive points are |
| lines, strips | An instanced quad per vertex pair: instance k reads vertices k and k + 1 (stride 12, offsets 0 and 12). A per-vertex break mask, computed on the producer's device from `strip_lengths`, hides the quads that would join two strips | Per vertex (per-strip colours are expanded on the device) or flat | |
| lines, index-pair edges | The segment draw, after the edge gather has written each edge's two ends into one stream (stride 24, offsets 0 and 12 in 3-D), so that instance k is edge k | Per vertex as gathered, per edge bound at both ends, or flat | The gather also clamps the indices |
| arrows | The shaft as a line quad, and the head as a flat triangle in screen space, oriented in the vertex stage from the origin and the vector | Per instance | |
| boxes | An instanced unit cube or square: 12 or 4 edges as line quads, and with `fill` 12 or 2 translucent triangles. Centre, half sizes and rotation are instance attributes that the vertex stage turns into a matrix | Per instance or class colour | Labels in the overlay |
| mesh | Indexed triangles, with vertex buffers for positions, normals and colours, values or UVs; with instance poses, instanced, the pose and the instance colour source as instance-step attributes | Vertex RGBA, colormap, texture or flat | A per-vertex headlight with normals, as the geometry service lit; flat faces from fragment derivatives without; drawn two-sided; with instance poses, normals go through the cofactor of each pose |
| image, colour | A textured quad on the camera's image plane in 3-D, or at the entity's origin in 2-D | Texture, linear filtering | |
| image, depth | A textured quad through the colormap; with `cloud`, also instanced point quads whose vertex stage reads the depth texture at its pixel and back-projects it through the intrinsics | Colormap | No CPU work for the cloud |
| image, segmentation | A textured quad over an integer texture, sampled nearest, coloured from the class table in the fragment stage | Class colours at `opacity` | |
| camera | Line quads: the frustum's edges and the image plane's rectangle | One colour per camera | |
| entity axes | Three line quads, shown when the person turns axes on | Red, green, blue | |
| labels | No draw: ImGui text over the view, at anchors projected on the CPU | | At most 512 |

Entity transforms become per-draw matrices. The scene core composes the world matrices on the CPU when a transform changes, at most 1,024 products, and writes each draw's model matrix, its model-view-projection and its normal matrix into the draw's uniform block, as the geometry service computes its normal matrix on the CPU. Under a camera the model matrix is the camera's world matrix times the pixel-to-plane matrix, a scale by d / fx and d / fy and a shift by the principal point to the plane distance d, times the child's own 2-D transform.

### What the scene view needs from the renderer core

From the graphics layer:

1. Import of slot buffers (a shared `MTLBuffer` on Metal, exportable CUDA memory through Vulkan external memory) at 4-byte-aligned byte offsets, usable as vertex and instance buffers and as the source of a texture fill, and a reported copy where import is impossible, as in the browser.
2. Instanced draws of a small quad or unit mesh with instance-step attributes at any offset and stride within an imported buffer: f32 positions, normalised u8x4 colours, f32 values, i32 class ids, f32 radii and quaternions. Storage buffers in the vertex stage are not needed.
3. Indexed triangle draws with u32 indices.
4. Textures filled from a buffer region by one GPU pass, which also absorbs row alignment: RGBA8 for colour, R32 float for depth, R16 or R32 unsigned integer for segmentation with nearest reads, and a small lookup texture holding the colormaps and the class colours. A vertex stage that can read a texture by texel, for the depth cloud.
5. A D32 float depth attachment with the less-or-equal test, opaque and alpha blending, fragment derivatives and discard.
6. Per-draw uniform blocks at dynamic offsets, for a few thousand draws a frame.
7. An ID target of two 32-bit unsigned integers per pixel, drawn by a pick pass around the pointer, and an asynchronous readback of it, delivered a frame or two later.

From the scene core: the entity tree and its world matrices; the cameras, that is the orbit with any of the six up axes, 2-D pan and zoom with y down or up, and looking through a pinhole; bounds from the per-axis bounds the producer computes on the device, transformed to world space, and fitting; picking; and the compilation of batches into draws, opaque first, then translucent in entity order.

### Applets

Applets draw scenes too. The scene service gives them this page's model through the applet ABI and replaces the geometry service, as the user decided; [The scene service for applets](#the-scene-service-for-applets) specifies it, and its scenes are drawn by this same view.

### Picking, hover and selection

Hover and click use the pick pass of the scene core: on a pointer event the visible draws are drawn again into a small ID target around the pointer, each writing its draw id and the index of its instance, which for lines is the segment or the edge, or of one vertex of the triangle for a mesh. A readback of that target gives the entity, the primitive and the item exactly, for every primitive, with no CPU copy of the geometry, and the pass costs nothing while the pointer is still. Hovers read at most 30 pixels a second, and a result that arrives after the draw list has changed is dropped.

Lasso and rectangle selection pick points and box centres by projecting them on the CPU, as the points view does today, from the arrays where they are CPU-visible: the shared slots on Metal and the copies a network viewer holds. This selects occluded points too, as today. On a route whose slots are not CPU-visible, the planned Windows session, a rectangle selects the visible items from a readback of the rectangle, and the lasso is unavailable. Clicking an entity in the view's tree selects the whole entity. The view dims what is not selected, as the points view does.

What a pick, hover or select must report travels in the event of protocol 3.1, with the scene's name as its view; [Wire protocol 3.2](wire-protocol-3-2.md) gives its encoding.

| Event | Detail |
|---|---|
| `pick`, `hover` | `entity`, the entity path; `primitive`; `index`, the item's index in the sent batch; `row_id`, its source index through the batch's payload axes; optionally `position`, the world position of the hit, from the arrays where they are CPU-visible and otherwise from the depth at the pixel |
| `select` | `entities`: a list of `{entity, primitive, indices, row_ids}`, at most 32,768 indices in all, with `truncated` when cut; `{entity}` alone selects a whole entity; `{}` clears |
| Linked | A batch logged with `items` maps its row ids to the items of that space. A select on it adds `space` and `items`, as in 3.1, and a SELECTION of that space highlights the matching items in every batch, and every other view, that maps the space |

## Interaction with the script

The controls of protocol 3.1, pause, resume, step, parameters and actions, belong to the session, so they work unchanged with scenes; a run paused at its checkpoint holds its scene where the checkpoint left it. A scene's events reach Python as the events of every view do: through the `on_event` and `events()` of `caliper.serve()` and the notebook widget's `selection`, with the detail above. The script can set the selection of an item space, and every scene batch that maps the space highlights it. The shared-memory session has no channel from the viewer back to the script apart from the heartbeat, so the Caliper app attached through it shows hovers and selections without the script receiving them; a script that needs them is attached through `caliper.serve()`. A person's camera movement is never sent to the script, and there is no camera event; the script only sets [the starting camera](#the-starting-camera). Interventions through a scene, such as dragging an entity to set a parameter, need typed arguments in the key that 3.1 reserves for them and are not part of this sprint.

## The scene service for applets

An applet draws in 3-D through the scene service, `caliper.scene.v1`, which offers the scene model of this page to the applet ABI: scenes with a space, an up axis and units; entity paths; entity transforms; points, lines as segments, strips and index-pair edges, arrows, boxes, meshes and images in their colour, depth and segmentation forms; pinhole cameras; clears and static items; the class table; the starting camera; and the scale bar. It replaces the geometry service, as the user decided, and [Renderer core](renderer-core.md#migration) orders the steps that build it, move the applets onto it and delete the geometry service. Nothing of it is built.

The model is the one Caliper LIB logs, with four differences that follow from where an applet runs. Arrays name memory the applet already has, an imported allocation or a `CaliperTensor`, instead of a tensor in a Python process. An applet's updates carry no stamps of a script's timelines, because an applet has none. There are no rate limits or data views, because the applet draws in its own process at its own cadence and decides what to set. Encoded JPEG and PNG frames are not offered, because an applet holds pixels, not files.

### The service and its version

The service id is `caliper.scene.v1`, declared in `sdk/include/caliper/services/scene_v1.h`. It is one table, as every graphics service is from sprint 4 on ([One version of each graphics service](renderer-core.md#one-version-of-each-graphics-service)): within an ABI epoch it grows only by appending members, which its `struct_size` guards, and each record below grows only by appending fields, which the record's own `struct_size` guards. An incompatible change is an epoch change. Every member is called on the frame thread only, the thread that runs the applet's `frame`, as for the tensor bridge.

### Naming an array

Every array of a batch is one `CaliperSceneArray`. It names where the bytes are, as the geometry service's sources did with an allocation and a byte offset, and how to read them:

| Field | Meaning |
|---|---|
| `source` | `CALIPER_SCENE_SRC_ALLOC`: an imported allocation, read in place. `CALIPER_SCENE_SRC_TENSOR`: a `CaliperTensor`. `CALIPER_SCENE_SRC_NONE`: the optional array is absent |
| `alloc`, `offset` | With `SRC_ALLOC`: a `CaliperAllocId` from the tensor bridge's `import_allocation`, the same id space the bridge's textures use, and the byte offset of row 0 inside it, a multiple of 4 |
| `count` | The number of rows: points, vertices, strips, edges, triangles, boxes, instances or pixels |
| `dtype`, `components` | The `CaliperDType` of one component, and the components per row: 1, 2, 3, 4 or 16 |
| `stride` | The bytes from one row to the next, at most 2,048; 0 means packed. A stride lets an applet bind one column of a wider array without a copy |
| `tensor` | With `SRC_TENSOR`: a CPU tensor, copied at the call so that the applet may free it on return, or a device tensor of the host's backend (`CALIPER_DEV_METAL` today), imported in place as the bridge imports it |

An `SRC_ALLOC` array is imported through the graphics layer's `import_buffer` once per allocation and bound as a vertex, instance or index source at its offset, so the scene view draws the applet's memory in place and the zero-copy route of the geometry service is kept: the info line reports the route that ran, and the copy counters of the migration's step 0 read 0 on it. Index arrays may sit at any 4-byte offset: the index clamp binds the 256-byte-aligned offset below and starts at the remainder, so nothing is copied. Each role takes the dtypes and component counts that [Primitives](#primitives) gives after Caliper LIB's conversion: f32 positions, radii, values and depths, u8 colours and colour images, i32 indices and class ids, and u16 class ids and depths. Caliper LIB casts other dtypes on the device; the service converts nothing, so an applet gives these.

### The table

```c
#define CALIPER_SCENE_V1 "caliper.scene.v1"

#define CALIPER_SCENE_CAP_DRAW     (1u << 0)  /* the host draws scenes: it has a backend of the graphics layer */
#define CALIPER_SCENE_CAP_IN_PLACE (1u << 1)  /* SRC_ALLOC arrays are read in place; without it, copied at each call */

typedef uint64_t CaliperSceneId;      /* 0 = none */
typedef uint64_t CaliperSceneToken;   /* one accepted update of one part; 0 = refused */

enum { CALIPER_SCENE_SRC_NONE = 0, CALIPER_SCENE_SRC_ALLOC = 1, CALIPER_SCENE_SRC_TENSOR = 2 };
typedef struct CaliperSceneArray {    /* 48 bytes */
    uint32_t source, dtype;
    CaliperAllocId alloc;  uint64_t offset;
    uint64_t count;
    uint32_t components, stride;
    const CaliperTensor* tensor;
} CaliperSceneArray;

enum { CALIPER_SCENE_COLOR_DEFAULT = 0, CALIPER_SCENE_COLOR_FLAT, CALIPER_SCENE_COLOR_RGBA,
       CALIPER_SCENE_COLOR_VALUES, CALIPER_SCENE_COLOR_CLASS };
enum { CALIPER_SCENE_PER_ITEM = 0, CALIPER_SCENE_PER_VERTEX, CALIPER_SCENE_PER_STRIP, CALIPER_SCENE_PER_EDGE };
typedef struct CaliperSceneColor {    /* 72 bytes */
    uint32_t kind;                    /* CALIPER_SCENE_COLOR_* */
    uint32_t per;                     /* lines: what the array belongs to; else PER_ITEM */
    uint32_t flat_rgba;               /* FLAT: r | g<<8 | b<<16 | a<<24 */
    int32_t  colormap;                /* VALUES: CaliperColormap */
    float    vmin, vmax;              /* VALUES: the range; vmin == vmax maps every value to level 0 */
    CaliperSceneArray array;          /* RGBA: u8 x4; VALUES: f32 x1; CLASS: i32 or u16 x1 */
} CaliperSceneColor;

typedef struct CaliperScenePixels {   /* an image held in an array */
    CaliperSceneArray data;           /* count = width * height; components = channels */
    uint32_t width, height;           /* at most 8,192 each */
    uint32_t row_pitch, reserved;     /* bytes per image row; 0 = packed */
} CaliperScenePixels;

/* Flags of the records below. */
#define CALIPER_SCENE_STATIC    (1u << 0)  /* setup: a clear of the scene's live data leaves it */
#define CALIPER_SCENE_ADDITIVE  (1u << 1)  /* points and lines: additive blending */
#define CALIPER_SCENE_FILL      (1u << 2)  /* boxes: translucent faces */
#define CALIPER_SCENE_CLOUD     (1u << 3)  /* depth images: also the back-projected cloud */
#define CALIPER_SCENE_MATRIX    (1u << 4)  /* transforms: use matrix, not the parts */
#define CALIPER_SCENE_RUB       (1u << 5)  /* cameras: x right, y up, looking along -z */
#define CALIPER_SCENE_RECURSIVE (1u << 6)  /* clear: the whole subtree */
#define CALIPER_SCENE_VIEW_NO_HEADER (1u << 0)  /* view: the header's controls in the context menu */

typedef struct CaliperScenePoints {
    uint32_t struct_size, flags;              /* STATIC, ADDITIVE */
    CaliperSceneArray positions;              /* f32 (N, D) */
    CaliperSceneColor color;                  /* per point */
    CaliperSceneArray radii;                  /* f32 (N, 1) in scene units, or SRC_NONE */
    float radius;                             /* one radius in scene units; 0: a pixel size */
    float size_px;                            /* disc diameter in logical pixels; 0: by count, as the points view */
    const char* const* labels;                /* label_count strings, or NULL */
    uint32_t label_count, reserved;           /* 0, 1 (one for all) or N */
} CaliperScenePoints;

enum { CALIPER_SCENE_SEGMENTS = 0, CALIPER_SCENE_STRIPS, CALIPER_SCENE_EDGES };
typedef struct CaliperSceneLines {
    uint32_t struct_size, flags;              /* STATIC, ADDITIVE */
    uint32_t mode;  float width_px;           /* width 0: 2 pixels */
    CaliperSceneArray positions;              /* f32 (V, D) */
    CaliperSceneArray strip_lengths;          /* STRIPS: i32 (S, 1) */
    CaliperSceneArray indices;                /* EDGES: i32 (E, 2) */
    CaliperSceneColor color;                  /* per vertex, strip or edge */
} CaliperSceneLines;

typedef struct CaliperSceneArrows {
    uint32_t struct_size, flags;
    float width_px;  uint32_t reserved;
    CaliperSceneArray origins, vectors;       /* f32 (N, D) each */
    CaliperSceneColor color;                  /* per arrow */
} CaliperSceneArrows;

typedef struct CaliperSceneBoxes {
    uint32_t struct_size, flags;              /* STATIC, FILL */
    float width_px;  uint32_t label_count;
    CaliperSceneArray centers, half_sizes;    /* f32 (N, D) each */
    CaliperSceneArray rotations;              /* f32 (N, 4) quaternions x y z w, 3-D only, or SRC_NONE */
    CaliperSceneColor color;                  /* per box, class colours included */
    const char* const* labels;
} CaliperSceneBoxes;

typedef struct CaliperSceneMesh {
    uint32_t struct_size, flags;
    CaliperSceneArray positions;              /* f32 (V, D) */
    CaliperSceneArray indices;                /* i32 (T, 3), or SRC_NONE: rows 3k to 3k + 2 */
    CaliperSceneArray normals;                /* f32 (V, 3): the headlight; SRC_NONE: flat faces */
    CaliperSceneColor color;                  /* per vertex */
    CaliperSceneArray uvs;                    /* f32 (V, 2), with texture */
    CaliperScenePixels texture;               /* u8 (H, W, 3 or 4), or data SRC_NONE */
    CaliperSceneArray instance_poses;         /* f32 (N, 16) column-major, or SRC_NONE */
    CaliperSceneColor instance_color;         /* per instance; DEFAULT: the mesh's own colour */
} CaliperSceneMesh;

enum { CALIPER_SCENE_IMAGE_COLOR = 0, CALIPER_SCENE_IMAGE_DEPTH, CALIPER_SCENE_IMAGE_SEGMENTATION };
typedef struct CaliperSceneImage {
    uint32_t struct_size, flags;              /* STATIC, CLOUD */
    uint32_t form;  float opacity;            /* opacity < 0: the form's default */
    CaliperScenePixels pixels;                /* COLOR u8 x1, 3 or 4; DEPTH f32 or u16; SEGMENTATION i32 or u16 */
    float depth_scale;  int32_t colormap;     /* DEPTH; depth_scale 0: 1 for f32, 0.001 for u16 */
    float vmin, vmax;                         /* DEPTH */
} CaliperSceneImage;

typedef struct CaliperSceneTransform {
    uint32_t struct_size, flags;              /* STATIC, MATRIX */
    float translation[3];
    float rotation[4];                        /* x y z w; in 2-D, rotation[0] is the angle in radians */
    float scale[3];
    float matrix[16];                         /* column-major, with MATRIX */
} CaliperSceneTransform;

typedef struct CaliperScenePinhole {
    uint32_t struct_size, flags;              /* STATIC, RUB */
    uint32_t width, height;                   /* the image size in pixels */
    float fx, fy, cx, cy;                     /* cx or cy NaN: the image centre */
    float plane_distance;                     /* 0: a twentieth of the scene's bounding radius */
} CaliperScenePinhole;

typedef struct CaliperSceneClass { uint32_t id, rgba; const char* label; } CaliperSceneClass;

#define CALIPER_SCENE_VIEW_TARGET    (1u << 0)
#define CALIPER_SCENE_VIEW_DISTANCE  (1u << 1)
#define CALIPER_SCENE_VIEW_AZIMUTH   (1u << 2)
#define CALIPER_SCENE_VIEW_ELEVATION (1u << 3)
typedef struct CaliperSceneViewCamera {
    uint32_t struct_size, given;              /* CALIPER_SCENE_VIEW_*; what is not given comes from the fit */
    float target[3], distance, azimuth_deg, elevation_deg;
    const char* entity;                       /* look through this entity's camera instead, or NULL */
} CaliperSceneViewCamera;

enum { CALIPER_SCENE_UP_DEFAULT = 0, CALIPER_SCENE_UP_POS_X, CALIPER_SCENE_UP_NEG_X, CALIPER_SCENE_UP_POS_Y,
       CALIPER_SCENE_UP_NEG_Y, CALIPER_SCENE_UP_POS_Z, CALIPER_SCENE_UP_NEG_Z };
typedef struct CaliperSceneDesc {
    uint32_t struct_size;
    uint32_t space;                           /* 0: 3-D, 1: 2-D */
    uint32_t up;                              /* CALIPER_SCENE_UP_*; DEFAULT is +y in 3-D and -y in 2-D */
    uint32_t reserved;
    const char* name;                         /* 1 to 63 bytes, unique within the applet */
    const char* units;                        /* at most 15 bytes; NULL: "m" */
} CaliperSceneDesc;

enum { CALIPER_SCENE_EVENT_HOVER = 1, CALIPER_SCENE_EVENT_HOVER_END, CALIPER_SCENE_EVENT_PICK,
       CALIPER_SCENE_EVENT_SELECT };
enum { CALIPER_SCENE_POINTS = 1, CALIPER_SCENE_LINES, CALIPER_SCENE_ARROWS, CALIPER_SCENE_BOXES,
       CALIPER_SCENE_MESH, CALIPER_SCENE_IMAGE };
typedef struct CaliperSceneEvent {
    uint32_t struct_size, kind;               /* CALIPER_SCENE_EVENT_* */
    const char* view;                         /* the view's key */
    const char* entity;                       /* the entity path; NULL in an empty SELECT, which clears */
    uint32_t primitive, has_position;         /* CALIPER_SCENE_POINTS ... */
    uint64_t item;                            /* HOVER, PICK: the row of the batch's arrays */
    float position[3];                        /* the hit's world position, when has_position */
    uint32_t pixel[2];                        /* images: the pixel */
    const uint64_t* items;                    /* SELECT: item_count rows of this entity and primitive */
    uint64_t item_count;
} CaliperSceneEvent;                          /* strings and items are valid until the next poll_event */

typedef struct CaliperSceneV1 {
    uint32_t struct_size;
    uint32_t (*caps)(void);
    CaliperSceneId (*create)(const CaliperSceneDesc* desc);
    void (*remove)(CaliperSceneId scene);
    CaliperSceneToken (*points)(CaliperSceneId scene, const char* path, const CaliperScenePoints* b);
    CaliperSceneToken (*lines)(CaliperSceneId scene, const char* path, const CaliperSceneLines* b);
    CaliperSceneToken (*arrows)(CaliperSceneId scene, const char* path, const CaliperSceneArrows* b);
    CaliperSceneToken (*boxes)(CaliperSceneId scene, const char* path, const CaliperSceneBoxes* b);
    CaliperSceneToken (*mesh)(CaliperSceneId scene, const char* path, const CaliperSceneMesh* b);
    CaliperSceneToken (*image)(CaliperSceneId scene, const char* path, const CaliperSceneImage* b);
    bool (*transform)(CaliperSceneId scene, const char* path, const CaliperSceneTransform* t);
    bool (*camera)(CaliperSceneId scene, const char* path, const CaliperScenePinhole* c);
    bool (*classes)(CaliperSceneId scene, const CaliperSceneClass* table, uint32_t count, uint32_t flags);
    bool (*clear)(CaliperSceneId scene, const char* path, uint32_t flags);   /* RECURSIVE, STATIC */
    bool (*view_camera)(CaliperSceneId scene, const CaliperSceneViewCamera* c);   /* NULL: the fit */
    bool (*retired)(CaliperSceneId scene, CaliperSceneToken token);
    bool (*view)(CaliperSceneId scene, const char* key, float width, float height, uint32_t flags);
    bool (*poll_event)(CaliperSceneId scene, CaliperSceneEvent* out);
    const char* (*last_error)(void);
} CaliperSceneV1;
```

Each record means what the Caliper LIB call of the same name means in [The scene's calls](#the-scenes-calls), with the defaults given there, and the rules of [Primitives](#primitives), [Entity transforms](#entity-transforms), [Cameras and the camera's pixel space](#cameras-and-the-cameras-pixel-space), [The class table](#the-class-table), [Clearing and static items](#clearing-and-static-items) and [The starting camera](#the-starting-camera) apply unchanged. A colour's range is given by the applet, because the service computes nothing on the device. `remove` releases the scene and its imports once the last frame that read them completed.

### Frames, atomicity and memory stability

- **When an applet may update.** Only on the frame thread, during its `frame` or its `draw_ui`. A worker thread hands its slots to the frame thread, as the geometry service's applets do today.
- **One frame, together.** Every call made between two frames takes effect together in the next picture: the host compiles each changed scene once, after the applet's frame returns and before the ImGui pass. A later call at the same path and primitive in the same frame replaces an earlier one.
- **Refusal per part.** A refused call returns 0 or false, changes nothing of that part, leaves the previous batch drawn, and states its reason in `last_error` and in one `caliper.log.v1` line. The other calls of the frame still apply. The geometry service refused a whole frame, because its frame was a clear and a list of draws, and a refusal after the clear would have left a blank picture; a scene keeps every part it holds, so a refused part cannot blank the picture, and refusing the frame's other parts would protect nothing.
- **The spatial half of memory stability.** An `SRC_ALLOC` array is read in place whenever the view redraws: when data, the camera or the size changes, and for each pick. Its bytes must therefore not be rewritten while its batch is current, nor after the batch is replaced or cleared until `retired` returns true for the token of the update that set it, which happens once the last frame that read it completed. `retired` never blocks. An applet's triple-buffered slots fit this: it writes a slot, sets the batch from it, keeps the token with the slot, and writes the slot again only once that token has retired.
- **The temporal half.** The producer must have finished writing an array before the call, by draining its device, as for the geometry service: the service orders nothing after a producer that is still writing.
- **Without in-place import.** Where `CALIPER_SCENE_CAP_IN_PLACE` is unset, on the Dawn fallback, an `SRC_ALLOC` array is copied at the call, its info line says so, and its token retires at once.
- **A `SRC_TENSOR` array** on the CPU is copied at the call. A device tensor follows the rules of `SRC_ALLOC`.

### The view in the applet's window

`view(scene, key, width, height, flags)` draws the scene view as one ImGui item in the applet's current window, at the cursor and at the given logical size, or in the rest of the content region when a size is 0. It is the same scene view that the Caliper app shows for a script's scene: the picture, the header with the fit button, the Y up and Z up switch and the display menu, orbit, pan and zoom by mouse and keys, the hover tooltip, labels, the axes mark and the scale bar. The host remembers each view's camera, up mode and display options under the applet's id and the view's key, as for every view, and the key also lets one scene be shown in two views with two cameras; `NULL` takes the scene's name. It returns true when the picture drew. The flag `CALIPER_SCENE_VIEW_NO_HEADER` moves the header's controls into the picture's context menu, for a small view.

The service hands out no texture id: the picture exists only as this item and through export. This is the choice between three forms, and the reason is that the shared camera, the up-axis switch, picking and the overlays come for nothing only if the host owns the view's input, while an applet's layout stays the applet's. An offscreen texture that the applet draws with `ImGui::Image`, as the geometry service's views were, would leave every applet to write its own camera and picking again, as each one does today. A view that the host places in a window of its own would take the layout away from applets that put two views side by side or a view beside their controls.

### Events

`poll_event` returns the next event of the scene's views, oldest first, once per call, and false when none is left; the applet calls it in its frame. A hover reports the item under the pointer when it changes, and a hover end when the pointer leaves every item. A click reports a pick. A rectangle or lasso reports one select event for each entity and primitive it caught, with the rows, at most 32,768 in all, and an empty select clears. An item is the row of the batch's arrays: a point, a segment, strip segment or edge, an arrow, a box, a mesh instance (or the triangle's first vertex when the mesh has no instances), or an image's pixel. Picks and hovers come from the scene core's pick pass, exact for every primitive and a frame or two late. A lasso projects the points and box centres that the CPU can read, as for every scene; an `SRC_ALLOC` array in memory the CPU cannot read, such as an MPS tensor's private storage, is selected by a rectangle from a readback of the pick target, and the lasso is unavailable for it, as on the planned Windows session.

### Capabilities and refusals

Two capability bits remain, because a backend can truly lack each: `CALIPER_SCENE_CAP_DRAW`, unset where the host has no backend of the graphics layer, as on Windows until the Vulkan backend lands, where `create` returns 0 with that reason; and `CALIPER_SCENE_CAP_IN_PLACE`, unset on the Dawn fallback, which copies. Every primitive, instancing, textures, picking and the overlays are present wherever the first bit is set, so no bit marks them.

A call is refused for the reasons Caliper LIB raises `ValueError` for, read from host metadata only: a path outside the grammar; a part the entity cannot hold (a primitive at a camera entity, a camera at an entity with primitives, 3-D data under a camera, an image outside a camera in a 3-D scene); a limit of [Limits](#limits) exceeded; a dtype, component count or row count that does not fit the role or the batch; an allocation that is not live; an offset that is not a multiple of 4; a window that leaves its allocation, checked without overflow; a stride above 2,048; and a record whose `struct_size` is below the first version's. Index values are not checked, because that would read the device; the index-clamp contract holds as for every scene.

### What an applet's scene gains

An applet that moves from the geometry service to the scene service deletes its camera code, its picking code and its fallback ladder, and gains, from the host and with no code of its own:

- the shared camera: orbit, pan, zoom, fit, reset to the starting camera, and looking through a pinhole;
- the up-axis switch;
- picking and hover through the pick pass, with events back to the applet, exact for every primitive and with no CPU copy of the data;
- depth testing, the headlight on meshes with normals and flat faces without, and the ground grid and axes mark that follow the up axis;
- the scale bar in the scene's units, labels, and hovers that name the entity, the item and its class;
- export through the scene, at any size ([Export](#export));
- recordings, scrubbing and drawing in a browser, where the scene is wire-capable ([Wire-capable scenes](#wire-capable-scenes)).

### Export

`caliper.export.v1` draws a scene instead of geometry records. `view_png(path, width, height, scene, camera)` renders the scene offscreen at the requested size through the scene core, with the given starting camera or, when it is `NULL`, the camera of the scene's first view, reads it back with the graphics layer's `read_texture`, and writes the PNG and its sidecar as today; `begin_sequence`, `frame` and `end_sequence` take the scene in the same way. A second export of one scene and camera is byte-identical on the same backend. The export capability bit follows `CALIPER_SCENE_CAP_DRAW`.

### Wire-capable scenes

An applet's scene is the same model that protocol 3.2 carries, so the host can send it where a script's scene goes. A scene is wire-capable when every array can be read by the host and fits the wire: the host can read it back (an `SRC_ALLOC` array on a backend that copies buffers back, which Metal does, or a CPU tensor), and each batch is within the limits of a frame and of the receiving viewer's data view. The host then reads the arrays back in a job, never on the frame thread, packs strided columns into rows, and encodes the scene's parts as protocol 3.2 messages for the app's recorder and for a hub that relays the app's views. A recorded applet scene opens in the scrubber like any recorded scene, in the app and in a browser, and a browser attached to that hub draws it under the parity rules of [Web geometry](web-geometry.md). Native-only are zero-copy, since every array that leaves the process is copied; the applet's events, since a pick in a browser reaches no applet in this sprint; and the applet's cadence, since the host sends at most at the view's rate. [Sprint 4](sprint-4.md) sizes this, 3 builder days, with the app's recorder in its increment 9.

### The C++ sugar

`caliper.hpp` wraps the table. `caliper::Scene` is created from the `Host` and a description, removes its scene when destroyed, and is falsy without `CALIPER_SCENE_CAP_DRAW`. `caliper::SceneArray::alloc(ref, dtype, count, components, stride)` takes the `(alloc, offset)` reference that the exportable pool's `to_bridge` returns, and `caliper::SceneArray::tensor(t)` takes a `CaliperTensor`. Each record has a builder with the defaults of Caliper LIB, such as `caliper::Points().positions(a).values(v, CALIPER_CMAP_MAGMA, lo, hi).size_px(3).additive()`. `scene.points(path, p)` and the other primitive calls return the token; `caliper::SlotTokens` keeps the tokens of a pool's slots and says when a slot may be written again; `scene.view(key, size)` draws the view item; and `scene.events()` iterates the queued events. The torch adapter gains `adapters::scene_array(pool, tensor)`, which imports an exportable tensor and returns its array.

### Tests

Headless tests of the service run on the graphics layer's recording backend in both CI lanes: each record compiles to the expected draw list, every refusal above is hit with its reason, tokens retire after the frames that read them, and a scene removed with live imports releases them. Capture rows draw one fixture applet scene on Metal against references under the parity rules. An applet built against the installed SDK draws a scene, logs its events and exports it, in the SDK's install test.

### From the geometry service

The geometry service drew a fixed menu of draws into an offscreen texture, with the camera, the state and the order chosen by the applet for every draw. The scene model describes what is drawn and leaves the drawing to the view. Each capability of the geometry service is covered by the scene model as it is, covered after an addition that is a general feature of a scene model in its own right, or dropped, with the applet adapted:

| Geometry service | In the scene model | What changes, and who loses what |
|---|---|---|
| Points of a pixel size (`draw_points`, the points topology) | Points with `size_px` | Covered. Discs with a hard edge instead of hardware points, so the pixels change |
| One-pixel hardware lines, line strips, indexed or not | Lines as segments, strips and index-pair edges | Covered. Quads one pixel wide or wider replace the hardware rule; the rows that depended on it, such as crossing lines with masked ends, are rewritten under the parity rules |
| Triangles, indexed or not | Meshes | Covered |
| Triangle strips | None | Dropped. No applet draws one, and a scene model describes surfaces as indexed triangles; an applet with a strip writes its indices on its device |
| Flat colour, per-vertex RGBA, per-vertex values through the shared colormaps | Colours, `colors` and `values` | Covered. The colormap decides by integer keys and matches the reference byte for byte |
| Textured meshes: UVs and a bridge texture, with the fixed linear clamped sampler | Meshes with `uvs` and a `texture` image | Covered, with one change: the texture is an image array, not a bridge texture id, so that the scene is wire-capable. `twin_scope` colormaps its field into an RGBA image on its device, one gather in torch, before it sets the mesh |
| The headlight, per vertex, from normals, `0.30 + 0.70 · max(n · l, 0)` | Meshes with `normals` | Covered as it is |
| Unlit triangles | Meshes without normals are shaded flat | Dropped. No applet draws unlit triangles, and flat faces are the scene model's rule |
| Opaque and alpha blending, chosen per draw | Opacity from the colours, translucent batches after opaque ones | Covered by the scene's rule instead of per-draw state |
| Additive blending | `blend="additive"` for points and lines | Covered after an addition: additive batches draw after the opaque ones with the depth test on and depth writes off, so dense clouds and trajectories brighten where they overlap and their order does not matter. It is a general feature of point clouds and trails, and five applets draw additive points |
| Views with or without depth; depth test and write chosen per draw | Every view has depth; opaque batches test and write, translucent and additive ones test only | Changed. Points that the geometry service drew without any depth test now test against the scene's other batches, which changes nothing when they are alone. Lines, arrows and box edges gain a small constant depth offset toward the camera, so that an edge lying on a mesh, such as a wireframe, wins over the surface: a general rule of the scene model, added because quads, unlike hardware lines, do not rasterise at exactly the surface's depth |
| A model matrix per draw | Entity transforms | Covered |
| Camera matrices owned by the applet | The view's camera, its starting camera and the shared controls | Dropped. Every applet loses its own orbit code and the freedom to set any projection each frame, and gains the shared camera |
| Instanced draws: (N, 16) poses and a per-instance value through the colormap | Meshes with `instance_poses` and `instance_colors` or `instance_values` | Covered after an addition: many copies of one shape is a general feature. Normals are transformed by the cofactor of each pose's upper 3 × 3, which is exact for any invertible pose |
| The rigidity gate on lit instanced draws, with its tolerance | None | Dropped, because nothing needs it: the cofactor lights every invertible pose correctly, so no pose is refused, and the host no longer reads the instance matrices back before a frame, which on Metal cost a blit and a wait |
| The index-clamp contract | The same contract | Covered as it is |
| One atomic gated frame: every gate before encoding, the whole frame refused with its pixels untouched | Refusal per part, with the previous part kept; every accepted call of a frame appears together | Changed, as [Frames, atomicity and memory stability](#frames-atomicity-and-memory-stability) explains |
| The memory-stability contract, in place until the view's next draw, drained before the call | The same two halves, with tokens that say when bytes may be rewritten | Covered |
| An offscreen view as a bridge texture id, shown with `ImGui::Image` | The view item | Changed: no texture id is handed out |
| The refusal of a geometry view named as a texture | None | Not needed, because no view texture is handed out |
| Four caps bits, one per revision | `CALIPER_SCENE_CAP_DRAW` and `CALIPER_SCENE_CAP_IN_PLACE` | Changed |
| Export drawing geometry records | Export drawing a scene | Changed |
| Byte-exact rows against a CPU reference | Rows under the parity rules | Changed, as below |

**The change of contract.** The geometry service's rows compare every pixel with one CPU reference, byte for byte, with lit pixels within 2 levels, and they depend on hardware points and lines and on state the applet chose per draw. The rows of the graphics suite that replace them are written for the scene service and follow the parity rules of [Web geometry](web-geometry.md#parity-with-the-native-renderers) natively as in a browser: a pixel whose colour a lookup decides away from edges (a flat colour, an RGBA colour, a class colour, a value through the colormap by keys, a texel sampled nearest) is byte-exact; a pixel within one pixel of an edge may take either side; lit pixels may differ by 2 levels, and blended and linearly sampled ones by 1. The rows keep the geometry rows' fixtures where they still apply. The colormap rows, the texture path's rows and the graphics layer's own rows stay byte-exact. Export is deterministic, a second export of one scene being byte-identical, but no longer byte-exact against a CPU reference for every pixel. This is a real change of the graphics contract, which the user accepted in choosing a scene service for applets.

## Changes to the shared contracts

- **The up axis of a 2-D scene.** The contracts give an up axis to 3-D scenes only. A 2-D scene also takes one, `"-y"` (the default, image pixels) or `"+y"` (plots), because an image overlay and a plot need opposite conventions.
- **Camera entities hold no primitive.** The contracts let an entity hold a camera and primitives. A camera entity's children live in pixel space, so a primitive on the camera entity itself would have no clear space; it is refused.
- **2-D entity transforms.** In a 2-D scene an entity transform takes 2 numbers of translation, one angle and 1 or 2 numbers of scale, stored as the same 4 x 4 matrix.
- **The depth scale** is in scene units per stored unit, which is metres per unit in a scene whose unit is the metre.
- **Additive arrays.** Points take `labels`; arrows and box edges take a `width`; a mesh in a 2-D scene takes positions (V, 2), which gives filled 2-D shapes; lines take `indices` (E, 2) and `values`.
- **Additive blending and instanced meshes.** Points and lines take `blend`, and a mesh takes `instance_poses` with `instance_colors` or `instance_values`, so that the scene model covers what applets drew through the geometry service ([From the geometry service](#from-the-geometry-service)).

## Decisions

- **Right-handed with +y up by default, and a switch in every 3-D view.** Decided by the user. The built points view and the browser's camera assume +y, so a scene and `caliper.points` look the same when they open; the header control switches any 3-D view between Y up and Z up, `up` only sets the starting mode, the switch changes the camera basis and never the data, and the mode is remembered per view.
- **A starting camera from the script, and no other viewport work in this sprint.** Decided by the user. `view_camera` is a starting value like `up`: a person's own camera wins once it is moved and is remembered per view, and a reset returns to `view_camera`. The orbit form uses the angles the view itself keeps, with the points view's azimuth convention, so its defaults open a scene as `caliper.points` opens a cloud. A 2-D scene takes none, because its fit already frames it whole.
- **Images in a 3-D scene sit under a camera.** Their place and size come from the intrinsics, so there is no free-floating image to position by hand.
- **Static items and small parts are retained, stamped batches are not.** A view stays near free when nobody watches, and a late viewer still receives the setup of the scene.
- **Shapes are checked on every call.** They are host metadata, and checking them at the call avoids the late argument errors that the live streaming page lists as a limit.
- **One device-side copy per snapshot, no borrowing.** The copy decouples the training loop from the viewer; drawing a tensor that training rewrites would tear.
- **Index-pair edges on lines, (E, 2) only, drawn through a gather.** Decided by the user. Refusing (2, E) gives every array one reading, where accepting it would leave a two-edge graph ambiguous. Gathering the ends in the existing clamp pass keeps the validated line shader and needs no storage buffer in the vertex stage, at 24 to 32 bytes per edge.
- **1,024 entities per scene.** Each entity is at least one draw, and objects of one kind belong in one batch.
- **Device-given entity transforms travel through slots and are read on the CPU.** World matrices, bounds and frustums are composed on the CPU, and no read of the device is ever waited for.
- **A pick pass for hover and click, CPU projection for lasso.** The pick pass is exact for every primitive and costs nothing while the pointer is still, and the projection keeps the points view's behaviour for selections.
- **A mesh over its budget becomes its bounding box.** Decimating on the device is costly and changes the topology a person is looking at.
- **Depth and segmentation are reduced by stride.** Mean pooling would invent depths and class ids.
- **The text log is retained whether or not anything is attached.** Lines are cheap host strings and are often read after the fact.
- **The names `caliper.text`, `caliper.bars` and `caliper.scene`.** Each view call of Caliper LIB is named after its kind, as `points`, `series` and `histogram` are, and none of the names collides with the package.
- **Separate `depth` and `segmentation` calls** rather than a form argument of `image`, because each form has its own options.
- **The shared-memory protocol keeps version 2.** Its append-only rules allow the additions, and a viewer from before refuses only the new views.
- **A scale bar from the scene's `units`.** Decided by the user. It is measured at the orbit target and says so, because perspective has no single scale; it hides through a pinhole, whose picture has no depth to measure at; and it applies prefixes only to the units it recognises, so that free text is never rewritten.
- **A scene service for applets, replacing the geometry service.** Decided by the user. Applets get the scene model, its camera, picking, overlays and export through one service instead of a fixed menu of draws with their own cameras, at about 25.5 builder days of migration in place of about 8.5 for moving the geometry service onto the graphics layer and giving it one version, with eight applets rewritten, seven from the geometry service and one from ImPlot3D, the geometry service deleted and the graphics contract for applets changed from byte-exact rows to the parity rules; [Renderer core](renderer-core.md#migration) gives the steps.
- **The view as an ImGui item that the host draws in the applet's window.** The host then owns the input, so the camera, the switch, picking and the overlays come for nothing, and the applet keeps its layout. No texture id is handed out.
- **Refusal per part, not per frame.** A scene keeps every part it holds, so a refused part cannot blank the picture, and the parts that were accepted still apply.
- **Tokens for memory stability.** An applet learns without waiting when the bytes of a replaced batch may be rewritten, which fits the triple-buffered slots the applets already use.
- **Arrays name allocations or tensors, with the dtypes the model draws.** The service then converts nothing on the device, and an allocation is drawn in place through the same ids as the tensor bridge's.
- **Additive blending and instanced meshes added to the scene model; triangle strips, unlit triangles, per-draw depth state, applet-owned camera matrices and the rigidity gate dropped.** The additions are general features of a scene model that applets need; what is dropped belongs to a fixed-function draw menu, not to a description of a scene, and no applet loses a picture it needs.
- **A texture is an image array, not a bridge texture id.** The scene then stays wire-capable, and an applet that colours a field colormaps it on its device.

## Left to sprint 5

The user moved what this sprint leaves out to [Sprint 5](sprint-5.md), which plans it:

- **More shapes:** ellipsoids, capsules, cylinders and planes, drawn from their parameters.
- **Mesh files** (glTF, OBJ, STL) through sprint 3's loaders, and a URDF helper in Caliper LIB.
- **Events from the app back to the script over shared memory:** an event ring in the scene's directory segment. Until then a script that needs events is attached through `caliper.serve()`.
- **Named frames,** through the import of robot recordings, which translates them into the entity tree. A second transform system beside the entity paths stays out.
- **The viewport features:** preset views, an orthographic projection, following an entity, saved viewpoints and linked cameras.
- **Trails over a time window,** slice planes and isosurfaces of 3-D fields, and comparing runs.

Compressed video stays out entirely: Caliper streams data, not video, so only single JPEG and PNG frames travel, as [Primitives](#primitives) describes.

## Open decisions

- **Borrowing static tensors without a copy.** The options are always copying, or a `copy=False` flag for tensors the script promises never to change. Recommendation: always copy for now; the copy of a static batch happens once.
- **The kind ids 6, 7 and 8.** Sprint 2 owns the kind table and must confirm or replace them.
- **The entity limit.** The options are 1,024 or 4,096 entities per scene. Recommendation: 1,024, raised once the draw cost is measured on the browser backend.
