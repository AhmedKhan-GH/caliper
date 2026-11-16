# Sprint 4

Sprint 4 gives Caliper 3-D scenes and recordings on a new renderer core. This page orders its work as increments, each of which a person can see and check when it lands, and sizes each one. The designs are in [Renderer core](renderer-core.md), [Scenes](scenes.md), [Wire protocol 3.2](wire-protocol-3-2.md), [Web geometry](web-geometry.md), [Recordings](recordings.md) and [Demos](demos.md), and the [roadmap](roadmap.md) holds the order and status. Nothing on this page is built, and nothing is built until sprint 2 merges.

## Where things stand

- **Sprint 4 is designed and not built.** Its pages wait for review. Recordings are MCAP files with the extension `.mcap`, as the user decided, and [Recordings](recordings.md#the-recording-file) specifies the file.
- **Sprint 2 is built on its branches and not merged, and sprint 3 is built as libraries and not wired into the app.** Every increment builds on sprint 2's `caliper.serve()`, notebook widget, controls of protocol 3.1, `web/wasm` module and WebGPU backend; only increment 9 needs sprint 3.
- **The user decided the shape of the sprint.** The renderer core's minimum cut comes first, then a loss-landscape demo in a notebook as the first visible increment, then the rest. Every remaining migration step follows in full, the Vulkan backend comes with [Windows parity](windows-parity.md), and what the sprint leaves out is in [Sprint 5](sprint-5.md).
- **The user decided that the graphics change keeps no backwards compatibility,** because nothing is deployed or commercialised yet. Each migration step replaces what it moves and deletes the old code, OpenGL is retired at the start of increment 2, each graphics service keeps one version under one ABI epoch bump in increment 6, with every applet in this repository updated, and the TypeScript WebGPU views go in increment 4 while the Canvas 2D fallback stays. The user then decided that applets draw in 3-D through a scene service that replaces the geometry service, and that one applet outside this repository is ported as the last step. [Migration](renderer-core.md#migration) gives the steps.

## How the sizes are given

Each increment lists its items with what each depends on and an estimated size in days of one builder. The sizes are the designers' estimates, kept unchanged, except the migration's, which [Renderer core](renderer-core.md#the-steps) planned again when the user decided that the graphics change keeps no backwards compatibility. Where an item is split between increments, a table gives its share as "of" its whole size, and the shares add up to it. Four items are new, because no page sized them: the scene view in the shared view layer (2 days), drawing the landscape in the app (1.5), demo 0 itself (3), and applet scenes in recordings and through the hub (3). The 25 days of recordings on MCAP are itemised in [Recordings](recordings.md#size) and rest on a structural check of MCAP, not on stress tests. Every migration step of the renderer core lands under the rule of [Migration](renderer-core.md#the-rule-of-every-step): its own series of commits, which replaces what it moves and deletes the old code with no switch back, measurements before and after against the fast-path thresholds, the byte-exact rows of the graphics suite, references recorded from the old path at the step's first commit, and offscreen captures for the user. The next step waits until it passes.

## The increments

### 0. The renderer core's minimum cut

- **What a person sees.** Captures and measurements for each step: a quad and a one-entity scene drawn through the graphics layer on Metal and read back byte for byte, the scene programs compiled on Metal and in a browser, and the sprint 2 heatmap and points drawn unchanged in a browser through the layer.
- **What it needs.** Sprint 2 merged, for step 4.
- **Done when** steps 0 to 4 have passed their checks in [Migration](renderer-core.md#migration) and their measurements are committed.

| Item | Depends on | Builder days |
|---|---|---|
| Step 0: counters and baseline | Nothing | 2 |
| Step 1: the layer's interface, validation, uniform arena, frames, serials, readback and recording backend | The interface header | 3 (of 6) |
| Step 1: the layer's Metal backend | The row above | 3 (of 6) |
| Step 2: the shader generator with pinned Naga and the scene programs | The interface header | 3 |
| Step 3: the scene core | The interface header | 6 |
| Step 4: the layer's WebGPU backend in the browser | Steps 1 and 2; sprint 2's backend | 5 |

### 1. A loss landscape in a notebook

- **What a person sees.** [Demo 0](demos.md#the-increment-demo-a-loss-landscape) in a notebook widget and in the browser viewer: the loss surface around the current weights, over the plane of the path's two principal components, as a mesh coloured by log loss, the optimizer's path across it as a line strip, the current point, a box and a floor grid, under the orbit camera and its fit, with the plane and the surface computed again on the device every N steps while the model trains.
- **What it needs.** Increment 0, and from sprint 2 the controls of protocol 3.1, `caliper.serve()`, the widget and the confirmation of the ids below.
- **What it pulls forward**, and nothing more:
  - From the renderer core, beyond the minimum cut, only the browser half of step 6: the colormap decided by integer keys, which the surface's colours need.
  - From Scenes, on the network route only: `caliper.scene` with `space`, `up`, `units` and `rate`; `scene.mesh` with positions, triangle indices, and colours or values through a colormap; `scene.lines` as segments or strips, with colours or values and a width; `scene.points` with colours or values and radii; `static=` and `rate=` on each; `caliper.widget(name, kind="scene")`. With only the shared-memory session or the in-process window open, a scene call warns once and does nothing there.
  - From Wire protocol 3.2: the capability `scenes`; the kind `scene` (6) in VIEWS with the key `scene` (`space`, `up`, `units`); SCENE_UPDATE (`0x0070`) with the parts `points`, `lines` (`mode` segments or strips) and `mesh`; SCENE_CLEAR (`0x0072`) with `reset`, which opens a scene's state for a new subscription; the default scene data view, resolved; golden frames for each, in the C++, Python and TypeScript codecs.
  - From Web geometry: the points, segment and mesh draws (`vs_points` with `fs_points`, `vs_lines` with `fs_color`, `vs_mesh` with `fs_mesh_flat` and `fs_color`), stride-0 layouts, the uniform arena, buffers with capacity and the spare pool, the index clamp with the grid-stride loop, the colormap blocks by keys, and their parity rows. Without WebGPU, the view shows a notice in its place.
- **What it defers.** Picking, events and selection; arrows, boxes, images, cameras, entity transforms, the class table, labels, normals and textures; index-pair edges, the up-axis switch, the scale bar and `view_camera=`, which costs 2 days and is not needed, since the default orbit opens the surface at a good angle with `up="+z"`; scene data views beyond the default; timelines and stamps, because producing a stamp needs timelines in Caliper LIB, 3 days; SCENE_TRANSFORMS, SCENE_CLASSES, text, bars, `u16` and split updates; the shared-memory session and the in-process window; the TypeScript fallback, budgets and rebuilding after a lost device.
- **The app** draws the same scene in increment 2, so that it does not delay this one ([Open questions](#open-questions)).
- **Done when** the acceptance of demo 0 in [Demos](demos.md#the-increment-demo-a-loss-landscape) passes headless and the user has reviewed its captures from the notebook and the viewer page.

| Item | Depends on | Builder days |
|---|---|---|
| The 3.2 subset above in the three codecs, with golden frames | Sprint 2's confirmation of kind 6, `0x0070`, `0x0072` and `scenes` | 2 (of 4) |
| The Caliper LIB subset above, with its checks, device-side casts and bounds, retained static items and the scene's state for a new subscription | The codec subset | 3 (of 6) |
| The scene view in the shared view layer: kind 6 in the view registry, the scene's state from decoded messages, compiling and drawing when something changed, orbit input and fit from the points view, the info line, the widget's kind and the viewer page | Step 3; the codec subset | 2 (new) |
| Points, segment and mesh draws on the WebGPU backend | Step 4 | 2 (of 4) |
| Their WGSL and the index clamp's, in the one shader source | Step 2 | 2 (of 4) |
| The index clamp pass | Step 4 | 0.5 (of 2) |
| The colormap blocks, cached by range | The row below | 0.5 (of 1) |
| Step 6, browser half: the key derivation in the shared C++ header, the exhaustive CPU test, the integer decision in the browser's WGSL and its boundary rows | Steps 2 and 4 | 1.5 (of 2) |
| Parity rows for the three draws in headless Chromium, on a default and a compatibility device | The draws | 2 (of 4) |
| Demo 0: the notebook, `landscape.py`, the headless test, the reference captures and the cost measurement | The rows above | 3 (new) |

### 2. The landscape in the app, the end of OpenGL, and the views on the layer

- **What a person sees.** The Caliper app, attached to the notebook's server, draws the same landscape on Metal. GLEW and the OpenGL renderer are gone, and where Metal cannot start the app says so and exits, which a test that forces the failure captures. `caliper.points` draws through the scene core, and heatmaps, images and the embed views take the texture path through the layer with the colormap decided by keys, each with pixels identical to the references recorded before its step and with its old Metal code deleted. On Windows the Vulkan renderer keeps the window, the canvas and ImGui, and bridge textures and geometry views are refused with a reason until Windows parity, a gap the user has accepted. It needs increment 1 and step 0's baseline. Retiring OpenGL depends on nothing, so a builder who is free during increments 0 and 1 may land it then.
- **Done when** a capture of the app's scene view matches the CPU reference rows on Metal, and steps 5 to 8 pass their checks.

| Item | Depends on | Builder days |
|---|---|---|
| The scene from the network route in the app, with capture rows on Metal | Increment 1 | 1.5 (new) |
| Step 5: retiring the GLEW OpenGL path, with the refusal where no renderer starts | Nothing | 2 |
| Step 6, native rows: the colormap by keys on Metal through the generated program | Step 6's browser half | 0.5 (of 2) |
| Step 7: the points live view on the scene core, with the graphics suite's reference mode | Step 3 | 3 |
| Step 8: the texture path on Metal, and the Vulkan renderer reduced | Steps 1 and 6 | 4 |

### 3. Every primitive natively, through shared memory

- **What a person sees.** The rest of [Scenes](scenes.md) in the app, attached through the shared-memory session with in-place import from the slots: arrows, boxes, images, depth and segmentation, pinhole cameras and looking through them, entity transforms, the class table, labels, clears, scene data views, index-pair edges, the up-axis switch, `view_camera=` and the scale bar. Every call also reaches the network route.
- **What it needs.** Increments 1 and 2.
- **Done when** the worked example of Scenes runs attached through shared memory with its info lines reporting in-place import, opens on its `view_camera`, switches between Y up and Z up and shows its scale bar, and a graph drawn with `indices=` matches its reference.

| Item | Depends on | Builder days |
|---|---|---|
| The rest of the Caliper LIB scene calls | Increment 1 | 3 (of 6) |
| The rest of protocol 3.2 in the three codecs: the other parts, SCENE_TRANSFORMS, SCENE_CLASSES, stamps, text, bars, `u16`, split updates and the recording messages | Increment 1 | 2 (of 4) |
| The scene layout of the shared-memory session | Step 5 | 5 |
| Scene data views | The codec row | 3 |
| Index-pair edges on lines | The index clamp | 2 |
| The up-axis switch (2), `view_camera=` (2) and the scale bar (1) | The scene view | 5 |

### 4. Every primitive in the browser, and the scene model's additions

- **What a person sees.** The browser draws everything of increment 3 under the parity rules, with budgets, the refusal line, rebuilding after a lost device and the points-only fallback with its notice. Step 10 adds to the scene model what the applets will need, natively and in a browser alike: additive points and lines, instanced meshes, and edges that win over the surface they lie on. Step 9 removes the TypeScript WebGPU views, so that the WebAssembly views are the browser's one WebGPU path and the element's own views draw with Canvas 2D. It needs increment 3.
- **Done when** the scene rows of [Web geometry](web-geometry.md#parity-with-the-native-renderers) pass in headless Chromium for every primitive, and steps 9 and 10 pass their checks.

| Item | Depends on | Builder days |
|---|---|---|
| The other draws, their layouts and the pick variants | Increment 1 | 2 (of 4) |
| Their WGSL, and moving the drafted shaders into the one source | Increment 1 | 2 (of 4) |
| Texture fills for the image kinds, and encoded frames | Increment 3 | 1.5 (of 2) |
| The class table texture | Increment 3 | 0.5 (of 1) |
| Budgets, the refusal line and the lost device | Scene data views | 2 |
| The remaining parity rows | The rows above | 2 (of 4) |
| The scene fallback in TypeScript | The codec rows | 2 |
| Step 9: the TypeScript WebGPU views removed | Step 4 | 1.5 |
| Step 10: the scene model's additions, additive blending, instanced meshes and the edges' depth offset, in the scene core, Caliper LIB, the codecs and the browser | Increment 3; the draws above | 3 |

### 5. Picking, events, the text log, the bar chart, and the scene service for applets

- **What a person sees.** Hover, click, rectangle and lasso in a scene, natively and in a browser; picks and selections reaching the script through `caliper.serve()` and the widget's `selection`; linked selection with the other views; and the text log and bar chart in both. Then the scene service for applets ([Scenes](scenes.md#the-scene-service-for-applets)), with `field_scope`, `flow_scope` and `sculpt_scope` drawing their particle clouds through it: in place from their slots, with the shared camera, the up-axis switch and the scale bar, and with picks that come back to the applet as events, their own camera code and fallbacks deleted, and their captures before and after shown side by side.
- **What it needs.** Increments 3 and 4, and sprint 2's confirmation of the kinds 7 and 8.
- **Done when** a pick in a scene returns its entity, item and row id to the script, a lasso on a batch with `items` highlights the same items in a points view, a text log and a bar chart update natively and in a browser, and step 11 passes its checks.

| Item | Depends on | Builder days |
|---|---|---|
| Picking and events natively | The scene core's pick pass | 4 |
| Picking in the browser | The row above | 2 |
| The text log and the bar chart natively | Increment 3 | 3 |
| The text log in DOM and the bar chart in Canvas 2D | The row above | 3 |
| Step 11: the scene service, with `field_scope`, `flow_scope` and `sculpt_scope` on it | Picking and events natively; step 10 | 12.5 |

### 6. Every applet on the scene service, one version of each graphics service, the final seam, Dawn and a port from outside

- **What a person sees.** `embed_scope`, `gpt_scope` and `instance_scope`, then `mesh_scope` and `twin_scope` with their exports, drawing through the scene service, each with captures before and after and every difference listed; the geometry service gone from the tree; every applet in this repository rebuilt at ABI epoch 3 against the SDK 0.3.0, and an applet built for epoch 2 refused with the epoch message; `caliper.show` and the viewer drawing through the typed live views alone; `HostRenderer` reduced to the window, the frame loop, the canvas and ImGui, with the docs updated; the app on Dawn over Metal and over SwiftShader, scenes included, saying that it copies, with the Windows lane's tests that need a renderer running again on SwiftShader; and, by the user's decision, one applet that lives outside this repository built against the new SDK, passing its tests, with its captures before and after compared. Other applets and embedding hosts outside this repository stop loading until they are rebuilt, which is the user's decision to act on. It needs increment 5.
- **Done when** steps 12 to 16 pass their checks.

| Item | Depends on | Builder days |
|---|---|---|
| Step 12: `embed_scope`, `gpt_scope` and `instance_scope` on the scene service | Step 11 | 4 |
| Step 13: `mesh_scope` and `twin_scope` on the scene service, export drawing scenes, the geometry service deleted, one version of each graphics service, the one epoch bump and every manifest updated | Step 12 | 11 |
| Step 14: `HostRenderer` at its final shape, and the docs | Step 13 | 1.5 |
| Step 15: the Dawn fallback | Increment 4 | 5 |
| Step 16: one applet outside this repository ported, by the user's decision | Steps 13 and 14 | 3 |

### 7. Timelines, recordings on MCAP and the scrubber

- **What a person sees.** Stamps on `step`, `time` and the script's own timelines; `caliper.record()` writing `.mcap`; the scrubber in the app, over its live buffer and over opened files, and in a browser, over a file opened with `?recording=`; recording only when a person starts it, as the user decided; and demo 4, the training run, as the acceptance.
- **What it needs.** Increments 3 to 5, for demo 4's class table, item space and lasso.
- **Done when** the acceptance of demo 4 in [Demos](demos.md) passes.

| Item | Depends on | Builder days |
|---|---|---|
| Timelines in Caliper LIB | The codec rows of increment 3 | 3 |
| Recordings on MCAP: the recorder and reader in C++, `caliper.record`, the browser reader, the timeline and keyframe index, commit points and tailing, the writer lock, segments with a manifest, recovery, thinning and `caliper export --to parquet` | Timelines | 25 |
| The scrubber in the app | Recordings | 5 |
| The scrubber in the browser | Recordings; increment 4 | 3 |
| Demo 4, the training run | The rows above; increment 5 | 2 |

### 8. Demos 2, 3 and 1

- **What a person sees.** The robot arm, the detection boxes and the structure-from-motion reconstruction, in the order of [Demos](demos.md). It needs increment 7.
- **Done when** each demo answers its question from a fresh checkout and passes its acceptance, with its references reviewed by the user.

| Item | Depends on | Builder days |
|---|---|---|
| Demo 2, the robot arm | Increment 7 | 2 |
| Demo 3, detection boxes | Increment 7 | 2 |
| Demo 1, structure from motion, with choosing its shot | Demo 3's film | 3 |

### 9. Recordings through the hub, in the workspace and in spec panels

- **What a person sees.** `caliper serve --record` and `--recordings`; Record in the app's session toolbar writing into the workspace; recordings in the catalog and `caliper open`; a spec panel that shows a scene and binds a recording; and an applet's scene recorded with the session and drawn in a browser through the hub, where it is wire-capable ([Scenes](scenes.md#wire-capable-scenes)).
- **What it needs.** Increment 7, sprint 3 merged with sprint 2, and sprint 3's acceptance of spec version 1.
- **Done when** a recording made by the hub opens in a browser through the hub and in the app from the catalog, a spec panel shows a recorded scene at the cursor, and a recording of `mesh_scope` scrubs in the app and in a browser.

| Item | Depends on | Builder days |
|---|---|---|
| The hub recording and serving recordings | Increment 7; sprint 2's hub | 6 |
| The app's recorder, the workspace tables, the catalog provider and `caliper open` | Increment 7; sprint 3 | 5 |
| Spec version 1 with the `scene` primitive (4), and recording bindings in spec panels (2) | Sprint 3's spec runtime | 6 |
| Applet scenes in recordings and through the hub: a buffer readback in the graphics layer, the host reading a wire-capable applet scene back in a job and encoding it as protocol 3.2 messages for the app's recorder and for a hub that relays the app's views | The app's recorder; step 13 | 3 (new) |

## Totals and the wait for the first increment

Increments 0 to 9 take 22, 18.5, 11, 20, 16.5, 24.5, 24.5, 38, 7 and 20 builder days, 202 in all: the 192.5 days that the design pages size, and 9.5 for the four new items. The migration takes 74.5 of them, steps 0 to 16, and it ends with increment 6, before recordings. The Vulkan step, 8 days, is counted with Windows parity. Against the plan before the user's decision on backwards compatibility, 170.5, the sprint grew by 8.5 days to 179. The user's decision on a scene service for applets adds 23 more: the scene service adds 17 to the migration ([Renderer core](renderer-core.md#the-steps)), the port of the applet outside this repository 3, and applet scenes in recordings and through the hub 3, in increment 9.

With one builder the first increment is seen after 40.5 builder days: the minimum cut, 22, then increment 1, 18.5. With two it is about 21 days: one builder takes steps 1 and 4 while the other takes steps 0, 2 and 3, which ends the cut in 11 days; then one takes the shaders, draws, clamp, colormap and parity rows, and the other the codec subset, the Caliper LIB subset, the scene view and the demo. With three it is about 16 days: during the cut the third builder takes the codec subset, the Caliper LIB subset, the demo's numerical module, the CPU half of step 6 and the three draws' WGSL, none of which touches renderer code, and what remains in a chain is step 1, step 4, the draws, the parity rows and the demo's captures, 6 + 5 + 2 + 2 + 1 days.

## Dependencies on sprints 2 and 3

- **Sprint 2 is merged first.** Increment 0's step 4 needs its WebGPU backend, and every later increment needs its protocol, `web/wasm`, the widget, the hub, the controls and item spaces, and the checkpoint behind `caliper.step`. Before increment 1, sprint 2 confirms kind 6, the message types `0x0070` and `0x0072` and the capability `scenes`; before increment 3, the rest of [What sprint 2 must confirm](wire-protocol-3-2.md#what-sprint-2-must-confirm).
- **Sprint 3** is needed by increment 9 only: the workspace, the catalog provider interface, the reserved names `scene` and `recording`, and the spec format, whose `scene` primitive needs spec version 1.

## Decisions

- **Windows and Vulkan wait for the Windows machine.** Decided by the user: Windows and Vulkan are deferred until the user moves to the Windows machine, and the gap is accepted. From step 8, Windows draws its ImGui pages and refuses bridge textures, geometry views, applet scenes, exports and embedding views with a reason, until the Vulkan backend of the layer lands with [Windows parity](windows-parity.md).
- **Increments in order, the whole minimum cut first, then the loss landscape, and every remaining migration step in full.** Decided by the user, who wants incremental progress and richer 3-D than point clouds in the notebook as soon as possible. A smaller slice of the cut is put to the user below.
- **The subset of increment 1 is what demo 0 draws and nothing more.** Three primitives, the protocol messages that carry them and their draws in the browser make the demo; each deferred piece is planned whole in a later increment.
- **The network route only in increment 1.** A notebook and a browser receive scenes over the network in any case, and the shared-memory layout and the in-process window would add 5 days and more without changing what the demo shows.
- **The box and the floor grid as ordinary line draws at the scene's root, logged by the script, not overlays of the scene core.** They need no new code in the scene core or in either backend and draw alike natively and in the browser. The script knows the fixed range the box shows, while a box from the data's bounds would move at every update. The view's own ground grid and axes mark, in Scenes, come with the up-axis switch.
- **Step 6 in two halves.** The surface's colours need the keys in the browser now. The native half adds only the boundary rows on Metal through the generated program, and edits no hand-written copy, because each copy is deleted by the step that moves its client; the fast path's measurement of the keys comes with step 8.
- **The graphics change with no backwards compatibility.** Decided by the user, because nothing is deployed or commercialised yet: no dual paths, OpenGL retired at the start of increment 2, one version of each graphics service with one epoch bump and every applet in this repository updated in increment 6, the TypeScript WebGPU views removed in increment 4 with the Canvas 2D fallback kept, and zero-copy kept as a hard requirement. Protocol compatibility and the UI service's revisions are not part of it. [Renderer core](renderer-core.md#decisions) records each point with its reason.
- **Strip break masks built by the viewer from `strip_lengths`.** The wire carries strip lengths and per-strip colours, while Web geometry expects masks and expanded colours from the producer; building a mask costs the viewer one byte per vertex.
- **The key `static` travels with `scenes` alone.** Increment 1 sends no stamps, but a scene's state for a new subscription and its clears need the key, which Wire protocol 3.2 lists under `timelines`; sprint 2 is asked to move it.
- **The plane of the path's principal components as demo 0's default, with filter-normalised random directions as a switch.** Decided by the user: the question includes how the optimizer moves across the surface, and in random directions, nearly orthogonal to the trajectory, the path is an almost vertical line.
- **Vulkan with Windows parity, MCAP, and recording only when started.** Decided by the user.
- **A scene service for applets, replacing the geometry service.** Decided by the user. Its additions to the scene model land with the browser's scene work in increment 4, the service and the particle applets after native picking in increment 5, and the other applets, the deletion of the geometry service and the one epoch bump in increment 6. The API is designed on its own merits, and the applets adapt to it.
- **One applet outside this repository ported, as the last step of the migration.** Decided by the user. It adapts to the API, and it is done when it builds against the new SDK, passes its tests, and its captures before and after have been compared with every difference listed and explained.

## Risks

- **The cost of the surface.** About 1.1 TFLOP per surface, estimated at 0.2 to 0.5 s on an Apple laptop GPU and not measured, to which a refit of the plane adds about 13 GFLOP, a few milliseconds; `vmap` batching rules on MPS are unverified for every operation of the model, and the fallback loop is several times slower. If the budget of demo 0 misses, the default N rises.
- **Demo 0's plane.** The plane's axes change at each refit, so successive surfaces are not comparable and the path is redrawn, and early in training the two components may hold little of the path's variance, which a series reports. In the random directions of the switch the path moves under 1 % of the grid's width and reads as a vertical line.
- **A slower fast path, pixels that move, and no switch back.** The thresholds and byte-exact rows of every migration step catch the first two, and a step that misses is fixed, or reverted with git, since no old path is kept to switch back to.
- **Windows and machines without a GPU.** With no old path kept and the Vulkan backend at Windows parity, Windows refuses bridge textures, geometry views and applet scenes from step 8 until then, or draws them through Dawn with a copy from step 15, a gap the user has accepted. From step 5 until step 15 a machine where neither Metal nor Vulkan starts cannot run the app, and the Windows lane's tests that need a renderer skip.
- **Applets outside this repository.** From step 13 they are refused until rebuilt against the new SDK, those that drew through the geometry service no longer build until rewritten onto the scene service, and embedding hosts that call the v1.2 views no longer compile against `caliper-embed` 2.0.0. The sprint ports one, by the user's decision, and touches no other.
- **The applets' pictures and the graphics contract.** Each port changes what its applet draws, so captures are compared by eye instead of byte for byte, and applets' scenes follow the parity rules, where the geometry service's rows were byte-exact. The user accepted this in choosing the scene service.
- **Ids under review, and one browser.** Three sprints edit the protocol, the roadmap and the web package, and the ids of 3.2 are proposals until sprint 2 takes them. Only headless Chromium was used for checks; Safari and Firefox are untested.
- **MCAP.** Its sizes hold with no margin and rest on a structural check only, its specification's major version is 0, and standard tools that rewrite a file do not keep Caliper's alignment and mixed compression.
- **Scope.** 202 builder days.

## Open questions

1. **Reordering inside the minimum cut.** Increment 1 is seen only in a notebook and a browser, so it needs the layer's interface and core with its recording backend, the generator, the scene core and the browser's WebGPU backend, 17 days, and not step 0 or the layer's Metal backend, 5 days, which no browser draw uses. Options: the cut whole, as decided; or step 0 and the Metal backend moved to the start of increment 2, before any native path changes. The second brings the first increment forward from 40.5 to 35.5 builder days with one builder, and to about 18 and 13 days with two and three, and moves no work out of the sprint. It risks that the interface is exercised by WebGPU alone until Metal lands, so that what Metal needs changed (a bind model without bind group objects, inline constants, import, readback) reaches back into the scene core and the browser backend already built on it; the cut's captures on Metal come later; and it departs from the order the user decided. Recommendation: the cut whole, because a second backend on another API checks the interface before two clients are built on it.
2. **The app in increment 1.** Drawing the landscape natively costs about 1.5 days once the minimum cut exists: the app's network source passes kind 6 to the same scene view, which is shared C++ code, the Metal backend draws the three programs from their generated MSL, and three capture rows compare them with the CPU reference. Options: in increment 2, as planned, or in increment 1. Recommendation: increment 2 with one or two builders; with three it fits beside the browser rows and can come in.
3. **Demo 0's default N and cost budget.** Options: N = 500 with the run at most 1.5 times as long as without the landscape, as planned; or a tighter budget with a larger N. Recommendation: as planned, with N set again from the first measurement if the budget misses.
