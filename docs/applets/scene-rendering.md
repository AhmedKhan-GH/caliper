# Scene Rendering

The scene and points views draw with library pieces through the [graphics layer](graphics-layer.md).
The scene core supplies the scene tree, camera, bounds, labels and pick results.
[Rendering](rendering.md) describes the platform renderers.

## Scene Core

The library `caliper_scene` is plain C++17 in `src/host/scene/`.
It holds CPU scene state in the namespace `caliper_scene`.
The draw runtime executes the library pieces that turn this state into pixels.

A scene has a 2-D or 3-D space, an up axis and a unit label.
It holds at most 1,024 entities, with each parent before its children.
Each entity has a path, a transform, an optional pinhole camera and one batch of each primitive type.
Paths contain 1 to 16 segments and at most 255 bytes.
Each segment uses the characters `A-Z a-z 0-9 _ . -`.

Each part has a stamped layer and a static layer.
A normal clear removes stamped parts, and a static clear removes both layers.
A clear can name a transform, a camera or a primitive.
The class table supplies class labels and colors, with the 16-color palette as its default.

The scene tree composes world matrices in double precision.
A nonfinite pose hides its entity and subtree.
Children of a camera entity use its pixel space on the image plane.
Bounds come from producer metadata or CPU-visible arrays, then move into world space.
The camera fits their sphere in 3-D or their larger extent in 2-D.

The orbit supports all six up axes and a 45-degree field of view.
A change of up axis preserves the target, distance and eye position.
The 2-D camera supports either direction for its y axis.
A pinhole camera uses its intrinsics and a letterbox that preserves the image aspect.
Clip space has x right, y up and depth from 0 to 1, and matrices are column-major.

Each view composes its camera in double precision and rounds the matrix to f32 once for the draw runtime.
The camera code disables floating-point contraction for equal arithmetic in native code and WebAssembly.
`src/host/scene/labels.h` projects labels without a draw compiler.

Each piece writes its pick id and item into the frame's `RG32Uint` target.
`FramePick` reads a 16 by 16 window around the pointer from that target and its depth target.
The nearest valid texel gives the program and item.
The view maps this result to the entity, primitive, image pixel and world position where depth is available.
A pick uses the displayed frame and needs no separate scene draw.

`caliper_scene_tests` checks transforms, cameras, bounds, clears, static parts, scene limits, labels and synthetic frame picks without a GPU.
The scene and piece graphics tests compare library output with saved reference images.
[Testing](../development/testing.md#pixel-exact-graphics-suite) describes their fixtures and tolerances.

## Scene View

The scene view is kind 6 in the live-view registry, with its implementation in `src/host/live/live_scene.h`.
It draws through the same library pieces in the app and WebAssembly views.
The source applies scene updates, transforms, class tables and clears in their received order.
Current senders supply piece resources, and the read-only legacy decoder accepts per-shape parts from saved recordings.
[Piece Form](../reference/wire-protocol-3-2.md#piece-form) describes resource transport.

The view supports points, lines, arrows, boxes, meshes, images and camera frustums.
Lines can be segments, strips or index-pair edges.
Meshes can use normals, vertex colors or textures.
Images include color, gray, depth and segmentation forms.
Static and stamped parts remain separate, and a normal clear preserves the static part.
The view reports unsupported encoded images with an `unsupported` error.

The view keeps only the pending messages that affect the visible scene while display is paused.
A reset removes earlier messages, and an update replaces the pending update for its part and layer.
Resets and clears remain ordered.
Above 8,192 pending messages, the source requests a new subscription state.

The scene converter in `src/host/draw/scene_pieces.h` binds scene state to library pieces.
This includes entity poses, depth offsets, class colors, camera intrinsics and panel data boxes.
It accepts received resources or prepares resources from CPU scene data.
The view submits color, depth and ID targets through the draw runtime.
A changed scene or camera causes a redraw, and external programs draw on each frame.
The view reports an unsupported part or a failed piece through its notes.

The orbit starts at azimuth 0.95 and elevation 0.42 radians, with the fit distance multiplied by 1.15.
F or a double click fits the bounds and preserves the scene camera's current angles.
The header and U key change the up axis.
A chosen up axis remains active until reset.
A right click opens the camera menu, and a pinhole view preserves its letterbox.
Images precede annotations in a camera's pixel space.

The display menu controls the scale bar and camera reset.
The scale bar uses the longest of 1, 2 or 5 times a power of ten within one fifth of the image width.
Its label uses the scene units, and a 3-D label includes “at target”.
The bar hides in a pinhole view, below 160 logical pixels of width and before bounds exist.
Point and box labels appear at their projected anchors, with the nearest 512 first.
Selection highlights also use library pieces.

`caliper_gfx_tests` applies protocol fixtures to scene views and checks updates, late attachment, static parts, camera state and saved reference pixels.
`web/test/scenes.spec.ts` checks that each browser fixture draws pieces, produces nonclear pixels and reports no notes.
`web/test/draw_parity.spec.ts` compares app and browser colors, IDs and picks for the same fixtures.
`python/tests/test_scene_app.py` checks the hidden app with a live scene source.

## Points View

The points view is kind 3 and draws with `points@1` in the app and WebAssembly module.
Its implementation is `src/host/live/live_points.h`.
Each instance holds at most 4,194,304 points, and larger payloads use more instances in order.
Selection instances follow the base instances.

The view imports each snapshot buffer through the graphics layer.
Shared Metal buffers can use an in-place import, and CPU payloads use a transient copy.
The path label reports `points-imported` or `points-cpu-staged` from the import result.
A private GPU payload passes through textures and asynchronous readback before the piece can draw it.
The copy counters record these transfers.

The view prepares each point's position and packed RGBA color for the piece.
Values use the shared colormap keys, and labels use the fixed categorical palette.
A selection dims the base points and draws the selected points above them.
Each point is a square of 7, 5.5, 4 or 3 logical pixels, according to the count.

The params `att_ref`, `fade_w0` and `fade_w1` control depth cues.
A point's size is multiplied by `clamp(att_ref / w, 0.5, 2)`, where w is its clip-space depth.
Past `fade_w0`, its color fades linearly to half strength at `fade_w1` without a hue change.
Selected points retain their full color, and the depth toggle disables the cues.

The points camera orbits about +y in 3-D and uses y-up pan and zoom in 2-D.
Its first payload fits the producer bounds or CPU positions.
F or a double click fits the bounds and restores the default three-quarter view.
The rotate tool turns the orbit until a drag stops it.
ImGui draws the box, ticks and floor grid around the transparent 3-D target.

Hover and lasso project CPU-visible positions with the same f32 camera matrix as the piece.
The view reports when a payload has no CPU view for these operations.
Each payload stays alive until the last GPU frame that uses it completes.

`caliper_gfx_views_tests` compares the library draw with saved images and checks CPU and device interactions.
The lattice cases also check point-center colors against the CPU colormap reference.
The Canvas 2D browser view supplies the points display when WebGPU is unavailable.
