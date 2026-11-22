# Scene Rendering

The scene core of the renderer core holds scenes and compiles them into draw lists of the [graphics layer](graphics-layer.md), and two live views draw through it, the scene view and the points view. [Rendering](rendering.md) describes the backends below them.

## Scene Core

The renderer core's scene core is in `src/host/scene/`, in the namespace `caliper_scene` and the library
`caliper_scene`. It is the one camera and scene model that the views are to share, and it compiles a scene into a draw
list of the renderer core's graphics layer (`src/host/gfx/`). It is plain C++17 that never calls a GPU. The live views'
scene view and points view draw through it ([Scene View](#scene-view), [Points View](#points-view));
no service uses it yet.

It holds these parts:

- **The scene store.** A scene has a space (3-D or 2-D), an up axis and a unit note. Its entities are named by paths
  of 1 to 16 segments, each of the characters `A-Z a-z 0-9 _ . -`, at most 255 bytes in all, and a scene holds at most
  1,024 entities with the root among them. They are stored as arrays with each parent before its children. An entity
  holds an entity transform, a pinhole camera, and one batch of each primitive type. A batch describes arrays in
  graphics-layer buffers by role (the buffer, byte offset, count, stride and format) and never copies their data.
  Every part has a stamped layer and a static layer: a clear removes the stamped parts, and a static clear removes both.
  A clear may name its parts (the transform, the camera or one primitive), as a SCENE_CLEAR does. The class table maps
  ids to labels and colours, and the scene core keeps a 256 x 256 texture of it for the programs: an entry's colour,
  else colour id mod 16 of the 16-colour palette.
- **World matrices.** One forward pass over the changed subtrees composes each entity's world matrix in double
  precision. An entity whose world pose is not finite is hidden with its subtree. The children of a camera entity live
  in the camera's pixel space, on its image plane.
- **Cameras.** The orbit turns about any of the six axes with a 45° field of view, and its near and far planes follow
  the bounds as the points view computes them. Azimuth 0 lies along the first of +x, +y and +z that is perpendicular to
  the up axis. Switching the up axis keeps the target, the distance and the eye position. The 2-D pan and zoom has y up
  or y down. Looking through a pinhole projects with its intrinsics, letterboxed to the view. The starting camera of a
  scene and reset are part of a view's camera state. Clip space has x right, y up and depth from 0 at the near plane to
  1, and matrices are column-major.
- **Bounds and fitting.** A batch's bounds come from the producer, or else from a scan of its CPU-visible array. They
  are moved to world space by their eight corners. Fitting frames the bounding sphere with the orbit, or the larger
  extent with the pan and zoom.
- **Compiling.** For the visible entities, the compiler chooses each batch's program from the generated program table
  and binds its arrays as vertex or instance-step buffers at their offsets. An absent array reads a constant at stride
  0. The compiler writes the frame block, each draw's parameter block and the colormap blocks into the draw list's arena,
  and appends the draws: opaque ones first, then translucent ones, each in the byte order of the entity paths. When the
  view looks through a camera, that camera's pixel space is drawn flat over the picture last. On a camera's image plane
  the images come first and then that camera's 2-D entities, so that an annotation is drawn, and picked, over the
  image it annotates even when the image is translucent; a plane that holds a translucent image draws with the
  translucent ones. A separate step records
  the index clamp of meshes and the edge gather of index-pair edges as compute dispatches. Every primitive compiles:
  points, lines as segments, strips and index-pair edges, arrows, boxes with their faces, meshes, images in 2-D or on a
  camera's image plane with their depth clouds, segmentation, camera frustums and entity axes. The faces of a filled
  box are drawn at a quarter of the box colour's alpha, so that they stay translucent with an opaque colour. Labels are
  not draws, and the scene core gives their projected anchors instead. The draw map records each draw's entity,
  primitive and part, and its blend and depth state, which the CPU reference reads.
- **Picking.** The pick pass compiles the same draws with their pick entries into a 16 x 16 `RG32Uint` window around
  the pointer, which it reaches by zooming clip space. Decoding a read-back window gives the entity, the primitive and
  the item under the pointer, the world position from the depth, and for an image the pixel.

**One camera, the same bytes on every backend.** The scene core composes each view's camera in double precision and
rounds every draw's model-view-projection to f32 once, and it is compiled without floating-point contraction, so its
arithmetic is the same sequence of IEEE operations natively as in WebAssembly, which has no fused multiply-add. The
backends upload the draw list's arena as it is (the Metal backend copies it into its arena buffer, the WebGPU backend
writes it with one `wgpuQueueWriteBuffer`, and the recording backend keeps the frame), so a browser draws from the same
uniform bytes as Metal. `src/host/scene/uniform_check.h` compiles a fixed scene of points under an orbit and under a
pan and zoom and hashes its arena; the graphics suite checks the hash on the recording and Metal backends, and the
browser module's self-test checks it on WebGPU.

The scene core asks a `Resources` object for the handles its records name: pipelines of the generated programs, bind
groups, the buffer of its constant streams, the buffers it owns for clamped indices and gathered edges, and the
colormap blocks. `BackendResources` provides them over a backend of the graphics layer, with the backend's table of
generated programs. The colormap blocks come from the caller, normally through `colormap_source` over a
`KeyBlockCache` that the view owns ([Colormap by Keys](graphics-layer.md#colormap-by-keys)). After each prepare, the owned
buffers of batches that are gone, the compute groups that refer to a released buffer, and the image bind groups whose
image the view released go back to the layer, so a scene reset, which gives every entity a new id, and an image
replaced at a new size leave nothing behind.

Its tests run in two suites:

- `caliper_scene_tests` (`tests/scene/`) runs headless with no GPU. It checks the matrices, every camera, the fits,
  compiled draw lists read as data, clears and static items, the limits, and picks decoded from synthetic ID texels,
  and it submits every fixture scene to the layer's recording backend, whose validation must accept it. It builds and
  runs with any preset, for example `cmake --build cmake-build-notorch --target caliper_scene_tests` and then
  `ctest -R caliper_scene_tests` from the build folder.
- `caliper_gfx_tests --test-case="gfx/Metal scene*"` draws scenes through the whole renderer core on Metal: the scene
  core compiles them over `BackendResources` with the generated programs, and the layer's Metal backend renders them
  into offscreen targets that are read back through the layer. One row draws a one-entity 2-D scene of square points
  on pixel centres (`tests/scene/scene_row_fixture.h`) and compares it with its reference byte for byte. Another
  draws a 3-D scene at 512 x 512 (`tests/scene/surface_row_fixture.h`): a height field of 41 x 41 vertices coloured
  through viridis with flat faces, a line strip over it, a disc marker and a box of segments, under the fitted orbit
  camera with depth. It is compared with a CPU rasterisation of the same compiled draws under the rules of
  [Testing](../development/testing.md#pixel-exact-graphics-suite). A pick row draws that scene's pick variants into the
  16 x 16 window around a pixel on the surface, the marker and the line, and checks the entity, the primitive and the
  item that the picker decodes. `CALIPER_GFX_CAPTURE_DIR` writes the captures `first-scene-points.png` and
  `first-scene-surface.png`, and `CALIPER_GFX_SCENE_BENCH=1` runs a row that reports the compile time and the
  layer's encode time of the surface scene and of a scene of 1,000 entities.

## Scene View

The live views' scene view (`src/host/live/live_scene.h`, kind 6 of the view registry) draws a scene of the
[wire protocol 3.2](../reference/wire-protocol-3-2.md) through the scene core and the graphics layer, natively
and in the WebAssembly views alike. A source hands it the decoded SCENE_UPDATE and SCENE_CLEAR messages in their
order (`src/host/live/live_views_scene.h`), and the view applies every part but encoded images: points with class ids
and labels, lines as segments, strips or index-pair edges with colours per vertex, strip or edge, arrows, boxes with
rotations, class ids, labels and filled faces, meshes with normals or with UVs and a texture, images in their colour,
grey, depth and segmentation forms, and cameras; SCENE_TRANSFORMS, SCENE_CLASSES, static items, clears of entities and
of named parts, and the reset that opens a scene's state for a new subscription. The view keeps each entity's static
and stamped batches apart, so a clear that is not static leaves a static batch drawn; transforms, cameras and the class
table keep their layers in the scene store. An encoded (JPEG or PNG) image is answered ERROR `unsupported` by the source
and noted in the view.

While the display is paused, the source keeps the scene's messages (`src/host/live/scene_queue.h`), only what the
view would still show: a reset drops everything before it, a clear drops the waiting updates it removes and the
clears it covers, an update replaces the waiting one of its entity part and layer, entity transforms replace a waiting
message of their layer whose every path they set again, and a class table replaces the waiting one of its layer. Resets
and clears are never dropped. A sender that clears ever new paths could still grow the queue, so past 8,192 messages it empties, the source
subscribes again on the same subscription, and the queue waits for the sender's new state. A scene view that the core
refuses for good, because an embed v1.2 view already has its name, ends its subscription instead, and the source drops
what still arrives for it.

- **Buffers.** The arrays of a network message are host bytes. Each is written into a buffer of the layer that the view
  owns, one copy that the view counts in its copy counters and its header reports; strip lengths become the scene core's
  break mask, and per-strip colours or values are expanded to their vertices on the way. A batch replaced by one of the
  same or a smaller size is written into the buffers it has, and a larger one gets buffers with a capacity rounded up to
  a power of two of at least 4 KiB. Images and a mesh's texture go into textures the view owns, kept while the size
  and format stay: colour as `rgba8unorm`, grey as `r8unorm`, and depth and segmentation as `r32uint` bits, a u16 depth
  scaled to f32 scene units (by `scale`, 0.001 by default) and class ids widened on the way, so that the colormap and
  the class table decide on the values as they are. The class table's texture is uploaded once each time the table
  changes, 256 KiB, and counted as a copy too. The view keeps a CPU copy of the positions or centres and class ids of
  a batch with labels or class ids, from which the label anchors are projected. In-place import from the
  shared-memory session's slots is not built yet, and on the network route the bytes are on the host already, so the
  view writes each texture directly instead of filling it from a buffer by a GPU pass.
- **Colormaps.** A colormapped batch takes the colormap block of its colormap and range from the shared cache of key
  blocks: the view owns a `KeyBlockCache` of `src/host/gfx/colormap_keys.h` and hands the scene core
  `colormap_source` over it, so each range's keys are derived from the CPU reference once and the app's and the
  browser's scene views decide colours exactly as the other generated programs do
  ([Colormap by Keys](graphics-layer.md#colormap-by-keys)).
- **Drawing.** When the scene, the camera, the size or the clear colour changed, the view compiles the scene and submits
  it, with the index clamp of changed meshes, to an offscreen colour target with a depth target in 3-D, which its
  window shows as an image; otherwise it submits nothing. A frame that fails to compile or that the layer refuses is
  dropped with its clamps, and the next frame emits them again. On Metal each scene view has a backend of the layer of
  its own on the renderer's device and queue, so its submits are ordered with the app's frame and its statistics are
  its own; in a browser it borrows the WebGPU renderer's backend.
- **Camera.** The orbit about the scene's up axis (pan and zoom in a 2-D scene), turned, panned and zoomed with the
  points view's gestures. It opens at the points view's default three-quarter view, azimuth 0.95 and elevation 0.42
  radians at the fit's distance times 1.15, which `src/host/scene/camera_defaults.h` defines once for both views. It is
  fitted once when the scene first has bounds and again on F or a double click, and there the two views differ: the
  scene view's fit frames the bounds again at the angles the camera has, so a view the person turned stays turned,
  while the points view's fit returns to the default three-quarter view. This is the scene design, and
  `caliper_scene_tests` checks that a fit keeps the orbit's angles.
- **The up-axis switch.** In 3-D the header has a segment for Y up and one for Z up, and a third for the scene's own
  axis when it is another one, with the active one pressed; the key U steps through them while the view is hovered or
  focused. A switch changes only the camera: the orbit keeps its target, its distance and the position of its eye,
  and its azimuth and elevation are taken again about the new axis (`switch_up` in `src/host/scene/camera.h`), so the
  picture rolls until the new axis is upright. The scene, its batches and the producer are not touched. A mode the
  person chose wins over the scene's `up` until the view is reset, and looking through a pinhole it waits and applies
  when the view returns to its orbit. A 2-D scene has no switch. The view has no ground grid or axes mark yet, so
  nothing else follows the switch.
- **The scale bar.** The view layer draws a scale bar with ImGui over the picture's bottom-right corner, natively and
  in the WebAssembly views alike: a bar 2 logical pixels thick with end ticks 6 pixels tall, 12 pixels in from the
  edges, with its length above it over a translucent band. `src/host/scene/scale_bar.h` states the rule: the longest of
  1, 2 and 5 times a power of ten, in scene units, within a fifth of the picture's width, labelled in the scene's
  `units` with an SI prefix for m, s and g, such as "5 mm" or "200 µs". In 3-D the scale is measured at the orbit's
  target, 2 · distance · tan(22.5°) per picture height, and the label ends with "at target"; in 2-D it is exact. The
  bar hides through a pinhole, while the picture is narrower than 160 logical pixels and before the scene has bounds.
- **The display menu.** The header's `display` tool opens a menu with the scale bar's option, on by default, and
  the view's reset, which returns to the scene's up and the default angles at the fit. The live views keep a scene
  view's up mode and scale bar option by its name when it closes, and give them back when a view of that name opens
  again.
- **Looking through a camera.** A right click that does not drag opens a menu of the scene's cameras. Looking through
  one draws the scene from its pose with its intrinsics, letterboxed to the camera's aspect, and its pixel space flat
  over the picture. The view remembers the camera's path, so while the entity holds no camera it shows its own
  camera and says so, and a fit returns to the orbit.
- **Labels.** The labels of points and boxes, and the class labels of items without one, are drawn as text over the
  picture at the anchors the scene core projects, the nearest 512 first.

`caliper_gfx_tests --test-case="gfx/Metal scene view*"` applies every 3.2 golden frame to scene views and checks their
state, draws the surface row's scene from protocol messages and compares it with the same CPU rasterisation as the
surface row under the same parity rules, checks a late attach, clears and static items, and checks that an update of
one batch reuses its buffers while nothing is drawn when nothing changes. Two more rows check the up-axis switch, a
point at +z drawn straight above the target once the view is in Z up with every batch, sequence number and copied byte
unchanged, and a 2-D scene's exact scale bar. Its primitive rows draw one fixture per primitive
(`tests/scene/primitive_fixtures.h`) from protocol messages at 256 x 256 and compare each with the CPU reference of
`tests/scene/draw_reference.h`, as [Testing](../development/testing.md#pixel-exact-graphics-suite) describes, and
`CALIPER_GFX_CAPTURE_DIR` writes their pictures as `scene-<fixture>.png`. `web/test/scenes.spec.ts` runs the same
fixtures and reference in the browser, through the WebAssembly views' scene self-test. `caliper_scene_tests` checks the
scale bar's rule over a table of zoom levels from 10<sup>-9</sup> to 10<sup>9</sup> scene units per pixel at widths from
160 to 4,000 pixels, its labels, and its length against the projection of that distance at the target, and
`caliper_live_views_tests` checks the mode and the option across a close and a reopen, the key U and the bar's pixels on
the offscreen canvas. `CALIPER_GFX_CAPTURE_DIR` writes `scene-view-row.png`, `scene-view-row-update.png`,
`scene-view-up-y.png` and `scene-view-up-z.png`, and `CALIPER_GFX_SCENE_VIEW_BENCH=1` runs a row that reports
the view's compile, submit and GPU times for that scene and the cost of applying one mesh update.
`python/tests/test_scene_app.py` runs the whole path: the app, hidden and attached late to `caliper.serve()`, draws
a scene that the test reads back through Caliper API's capture, and `web/wasm/test/scene.mjs` draws one on the viewer
page in headless Chromium.

## Points View

The live views' points view (`caliper.points`, kind 3) is a scene of the scene core, drawn through the graphics layer by
the same machinery as the scene view (`src/host/live/live_points.h`), natively and in the WebAssembly views alike.

- **The payload.** The view's points are one points batch on the scene's root entity; a payload above the scene model's
  4,194,304 points a batch continues on child entities at later offsets of the same buffers, so that it draws in the
  same order. Each buffer of a snapshot, the positions and the optional values or labels, is bound with one
  `import_buffer` call of the layer. An `MTLBuffer`, such as an MPS tensor or a slot of the shared-memory session, is
  imported in place up to the end of its window, and every payload in the same buffer before that end shares the
  import. A CPU payload is a transient host import, which the layer copies, so the caller's bytes are free when the
  apply returns, and the copy counts in the view's copy counters.
- **The route and the path label.** The view's path label comes from the route the layer returned, never from the
  tensor's device: `points-imported` for an import in place, `points-cpu-staged` for a copy of host memory. The flag
  `imported` is set for an import in place.
- **Colours.** Flat points are light grey. Values map through the colormap blocks of the shared key cache
  ([Colormap by Keys](graphics-layer.md#colormap-by-keys)). Labels, which the producer sends as f32, map through the same keys
  over the range 0 to 255 with a table that holds the fixed 16-colour palette at every level, so a label above 255
  takes the colour of 255, one below 0 or NaN that of 0, as before, and the labels are read in place. A selection draws
  every point dimmed toward the plot's ground and then the selected points in their own colours over them.
- **Sizes.** Each point is a square of 7, 5.5, 4 or 3 logical pixels by count, as the browser's points view sizes them
  (`caliper_scene::default_point_size`), drawn as an instanced quad of the generated points program.
- **Depth cues.** In 3-D the points draw the view's depth cues, which the scene core passes to the points program
  (`CompileOptions`, the parameter block's `att_ref`, `fade_w0` and `fade_w1`): a point is its size times
  `clamp(d / w, 0.5, 2)` wide, d being the orbit's distance and w the point's distance along the view, so that nearer
  points are larger, and past the camera's target its colour falls linearly to half at the far side of the fitted
  bounding sphere, its hue kept. The selected points of a selection are never faded. `set_toggle` "depth" turns the
  cues off.
- **Camera.** The scene core's orbit about +y in 3-D and its pan and zoom with y up in 2-D, fitted by its `fit` to the
  producer's bounding box, or else to a scan of a CPU-visible payload, when the first payload, the first box or a change
  between 2-D and 3-D arrives, and again on F or a double click. A fit in 3-D returns to the default three-quarter view
  (`src/host/scene/camera_defaults.h`). The gestures keep their rates and bounds, and the header's rotate tool turns
  the orbit's azimuth until a drag stops it.
- **The box.** In 3-D the target's ground is transparent, and the window draws the box with its ticks and floor grid
  around the points through ImGui's draw list: its back layer under the target's image and its front layer over it,
  projected with the f32 matrix the draw takes.
- **Hover and lasso.** They project the CPU-visible payload on the CPU with the f32 matrix the draw takes, as before.
  The CPU view is the layer's mapping of the import: an `MTLBuffer` in shared storage or the copy of a CPU payload. A
  private-storage buffer has none, and the view says so instead of hovering.
- **Lifetime.** Each payload remembers the layer's serial of the last frame that drew it. A superseded payload is
  released when that frame has completed, which is when the producer may rewrite its slot (`read_done`).

The view drew through the Metal renderer's geometry draw with hardware point sprites before migration step 7 of the
renderer core, and in the browser with quads. 3-D points are now exact squares on Metal, as they already were in the
browser. A sprite of a whole or half size is the exact square, so in 2-D, for points at every sub-pixel phase and size
class, and for a single 3-D point, the pixels did not change. The depth cues make a 3-D point's size fractional, and a
Metal sprite of a fractional size can have an edge up to about 1/32 of a pixel from the exact square's, so a 3-D cloud
differs from the old view in pixels at the edges of its points: 152 to 717 pixels of 421,632 in the step's clouds,
every one within 1/32 of a pixel, plus the rasteriser's 1/256 snap, of a point's edge. That is the only difference of
the step: the old view's float camera, where the scene core composes in double, changes none of these pixels.

`caliper_gfx_views_tests` (`tests/gfx/gfx_points_view.mm`) checks the view through an offscreen core
([Testing](../development/testing.md#pixel-exact-graphics-suite)): 36 cases against the references recorded from the old view,
against a CPU rasterisation of the draws the scene core compiled, and, with a folder of the old view's images, pixel by
pixel.
