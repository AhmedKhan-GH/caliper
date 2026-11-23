# Renderer Core

This page plans what remains of the renderer core's migration: the move of the texture path onto the graphics layer, the scene model's additions, a scene service for applets that replaces the geometry service, one version of each graphics service under one bump of the ABI epoch, `HostRenderer` at its final shape, and the Vulkan backend of the layer with Windows parity.

The rest of the renderer core is built, and the docs describe it. Graphics Layer (`docs/applets/graphics-layer.md`) describes the layer with its Metal, recording and browser WebGPU backends and the colormap by keys, and Rendering (`docs/applets/rendering.md`) describes the backends, the copy counters and the absence of a fallback renderer since OpenGL was retired. Shader Authoring (`docs/applets/shader-authoring.md`) describes the one WGSL source and its translation by Naga. Scene Rendering (`docs/applets/scene-rendering.md`) describes the scene core and the points view, which now draws as a scene of one entity. Browser Elements (`docs/web/browser-elements.md`) says that the element's own views draw only with Canvas 2D, and Testing (`docs/development/testing.md`) describes the renderer core rows, the colormap proof and the WebAssembly views. Of the migration, steps 0 to 7 and step 9 are built, and step 8 is half built.

The zero-copy backends are the product path, Metal on macOS, Vulkan with CUDA on Windows and Linux, and WebGPU in the browser, so the decision is that the native fallback through Dawn, which step 15 planned, is dropped.

## Migration

The decision is that every remaining step follows in full. No step is optional. The decision is also that the migration keeps no backwards compatibility, because nothing is deployed or commercialised yet: no step keeps an old path; `HostRenderer` shrinks to the window, the frame loop and ImGui setup; each graphics service keeps one version, under one bump of the ABI epoch; the hand-written shader copies and the platform code in the views go as each client moves; and zero-copy stays a hard requirement. [Decisions](#decisions) gives each of these with its reason.

### Rule of Every Step

Each step is an increment that can be seen and checked on its own:

- it lands as its own series of commits, each of which builds and passes its tests;
- it replaces what it moves and deletes the old code in the same series, together with the `HostRenderer` methods and the tests that only the old code used, so no old path stays behind as a fallback or a switch, and a step is reverted with `git revert` of its series;
- it has measurements before and after against the thresholds of [Fast Path Thresholds](#fast-path-thresholds), and the byte-exact pixel checks of the graphics suite;
- it has offscreen captures of what it changed, taken without a window, for a reviewer to look at;
- the next step does not start until the current one passes all of these.

With no old path left at run time, a row can no longer draw one payload through the old and the new path and compare them, so the comparison moves in time. The first commit of a step changes no drawing code. It runs the rows, captures and benchmarks of what the step will change on the code as it stands, and commits their outputs: the readback bytes become fixtures that the tests read, as hashes, with the images kept beside the step's numbers under `python/bench/results/<step>/`, and the numbers are the step's "before". The commits that replace the client must then reproduce every fixture byte for byte where the step promises unchanged pixels, and meet the thresholds against the "before" numbers and against step 0's baseline. The fixtures stay in the suite as references, so a later step that moves the same pixels is caught too. Where a CPU reference row already defines the pixels, as for the colormap and the geometry rows, that row is the check and no fixture is needed. The graphics suite's reference mode, which `tests/gfx/gfx_reference.h` defines, and the browser harness's screenshots serve the remaining steps.

### Fast Path Thresholds

The zero-copy paths must stay exactly as fast through the migration. Each remaining step is measured before and after, on the same machine, with these existing benchmarks and counters:

- `python/bench/bench_live.py table --size 256|1024|2048 --repeat 3` on MPS: steps per second against the commit-matched control, the cost of `caliper.show` (median and p95 microseconds), step-to-screen latency (median and p95), the viewer's frames per second and its path label, such as `W:compute-imported`.
- `python/bench/bench_show.py --mode window|attach --device mps`: the same in the in-process window.
- The graphics suite's timing rows, `gfx/Vulkan+CUDA: timing — update cost by size` (call and completion microseconds, frame mean and maximum, by path and size from 256 to 4,096) and `create+release cycle vs update`.
- `CALIPER_VULKAN_STATS=1`: frame-time p50, p99 and maximum, direct and buffer update counts, kernel waits and retire blocks.
- `caliper-viewer --stats --latency-log`, and the WebGPU backend's counters in the browser harness (GPU milliseconds, dispatches, live textures and allocations).

| Metric | Acceptance after a step |
|---|---|
| Copies per frame and bytes copied on a zero-copy route | Exactly 0, as before |
| Path labels | Identical |
| Byte-exact graphics rows | All green, Lambert rows within ±2 LSB as today |
| Steps per second, as a share of the control | Within 2 points (the two repeats of the 1,024 sampler row differ by 1.1 points today) |
| `caliper.show` cost, median | Within 5 % or 2 µs, whichever is larger |
| Step-to-screen latency | Median within 1 ms, p95 within 2 ms |
| Frame time | p50 within 3 %, p99 within 5 % |
| GPU time of an update and of a view's draw | Within 5 % or 20 µs, whichever is larger |
| Update call time in the timing rows | Within 5 % or 2 µs, whichever is larger |
| The layer's CPU encode time per frame | At most 0.05 ms more |

Where a threshold names a percentage and an absolute figure, a result passes when it is within either, so the absolute figure is the allowance for small values: a GPU pass of 25 µs may grow by up to 20 µs, not by 5 % of 25 µs. This reading was settled when the colormap by keys was measured.

The before build is the step's first commit, which records the step's references and changes no drawing code ([Rule of Every Step](#rule-of-every-step)), and the after build is its last commit. Both are built in Release and run on the reference Mac, on mains power, with the same torch, interleaved three times each; the benchmarks already record the load average. The after numbers must also meet the thresholds against step 0's baseline, so that losses within the thresholds cannot add up across steps, and a zero-copy route keeps exactly 0 copies throughout, because zero-copy is a hard requirement. Results are committed under `python/bench/results/` with the step's name. A step that misses a threshold is fixed, or reverted with `git revert` of its series, before the next one starts. The Vulkan step at Windows parity compares with a build of the last commit before step 8, which still has the Vulkan renderer's paths, run interleaved with its own build on the same Windows machine.

### Remaining Steps

| Step | What changes and what is deleted | What proves it | Builder days |
|---|---|---|---|
| 8. Texture path on Metal, and the Vulkan renderer reduced | Built: the layer's texture operations `update_texture_from_tensor` and `fill_texture_from_buffer`, the generated program `fill_texture`, `HostRenderer::gfx()` with the Metal renderer's own layer backend, the adapter through which the geometry service's old code reads the layer's imported buffers, and the texture path's reference rows (`caliper_gfx_texture_tests`, 98 cases against the old path's record). Still to do: the tensor bridge, the embed views and the heatmap and image live views go through `update_texture_from_tensor`, which runs the generated colormap by keys; the bridge's allocation table moves to `import_buffer`; the path label comes from the returned route. Deleted: the texture, per-source update, completion, interop and import methods of `HostRenderer` in every backend, `cmap_f32` and the blit code of the Metal renderer, the embed views' Metal branch and their check of the renderer's device kind, and the WebGPU renderer's texture calls of step 4. Until step 13, which deletes the geometry service, its old code on the Metal renderer reads the layer's imported buffers through an adapter, so there is one allocation table. On Windows the Vulkan renderer's feature paths go ([Vulkan Renderer Until Windows Parity](#vulkan-renderer-until-windows-parity)) | Every gfx/Metal texture row; the boundary rows through the bridge; `bench_live` and `bench_show` within thresholds, which also measure the native cost of the integer colormap; identical path labels; the bridge refusing on a missing backend, in both lanes; the Windows lane green | 4 |
| 10. Scene model additions | Additive blending for points and lines; instanced meshes with instance poses and instance colours or values, their normals through the cofactor of each pose; and a constant depth offset for lines, arrows and box edges toward the camera ([Scenes](scenes.md#primitives)). Each lands in the scene core and its programs, Caliper LIB, the three codecs of protocol 3.2 and the browser backend, so that one scene model stays | Headless tests of the compiled draws; capture rows natively and in headless Chromium under the parity rules; an instanced mesh with a sheared pose lit like its CPU reference within 2 levels; a wireframe over a mesh drawn whole | 3 |
| 11. The scene service, and the particle applets on it | `caliper.scene.v1` ([Scene Service for Applets](scenes.md#scene-service-for-applets)): the table and its records; the host's implementation over the scene core, with arrays imported through the tensor bridge's allocation table; the refusals; the tokens; the view item with its remembered state; the events; the sugar and the torch adapter; its reference page and tests. It is a new service id beside the geometry service, so no epoch changes yet. `field_scope`, `flow_scope` and `sculpt_scope` move onto it, and their `draw_points` calls, their own cameras and their ImPlot3D fallbacks are deleted ([Applets on the Scene Service](#applets-on-the-scene-service)) | The service's headless tests on the recording backend, in both lanes; its capture rows on Metal; each applet's captures before and after and its tests, with 0 copies per frame on its zero-copy route | 12.5 |
| 12. Three more applets on the scene service | `embed_scope`, `gpt_scope` and `instance_scope` move onto the scene service; their ImPlot3D plots, `draw_points` and instanced geometry calls, CPU picking, cameras and fallback ladders are deleted | Each applet's captures before and after and its tests | 4 |
| 13. The last applets, one version of each graphics service, the geometry service deleted, and the one epoch bump | `mesh_scope` and `twin_scope` move onto the scene service, and the export service draws scenes. `caliper.geometry.*` then has no client and is deleted: its headers, its reference page, the host's implementation, its MSL shaders and the four copies of its parameter block, the geometry methods and capability booleans of `HostRenderer`, `HostGeomDraw`, the adapter of step 8, its rows of the graphics suite, its caps bits and its sugar, and with it the graphics layer's `kCapVertexStorage` and bind group 3, which nothing else uses. The tensor bridge and the export service each become one table, the embed v1.2 tensor views are folded into the typed live views, the slot record of `include/caliper/live.h` loses its Metal buffer, the ABI epoch goes from 2 to 3 with the simplified check, the SDK to 0.3.0 and `caliper-embed` to 2.0.0, and every manifest in this repository is updated ([One Version of Each Graphics Service](#one-version-of-each-graphics-service)). Deleted as well: the revision headers and tables, the sugar's widening of records, the Python package's switch between two view routes, and `native_device()` | The two applets' captures before and after and their tests; the export rows on scenes; the ABI tests pin the new layouts; every applet in this repository builds against the installed SDK in both lanes, loads, and its page matches its capture from the end of its port, or from before this step for an applet without 3-D; an applet built for epoch 2 is refused before its library opens; the embedding, live and Python tests; no mention of `caliper.geometry` left in the tree | 11 |
| 14. `HostRenderer` at its final shape | What is left of the special-purpose seam goes ([Renderer Seam at Its Final Shape](#renderer-seam-at-its-final-shape)): `canvas_supported`, `window_capture_supported`, `debug_readback_rgba8`, and step 0's copy counters, which the layer's `Stats` now carries. Rendering (`docs/applets/rendering.md`), the zero-copy pages, Compatibility and Epochs (`docs/development/compatibility-and-epochs.md`) and the [roadmap](roadmap.md#renderers) are updated | The whole suite in both lanes; the docs build | 1.5 |
| 16. An applet outside this repository, ported | Done when that applet needs it. The decision is that one applet that lives outside this repository, with 3-D curves, a mesh, measurement marks and a model view, is ported onto the scene service and the single-version services, built against the SDK installed from the build branch, with its own tests run and offscreen captures taken before and after. The applet adapts to the API: where it draws something the API does not offer, the applet changes, through another primitive, a computation of its own or a simpler picture. Caliper gains a feature from the port only where it is a proper general feature in its own right, recorded with its reason as an item of its own. Other applets outside this repository stay open for a later decision. It needs steps 13 and 14 | The applet builds against the new SDK and passes its tests, and its captures after the port have been compared with those before, with every difference listed and explained | 3 |
| 17. Vulkan backend, with [Windows parity](windows-parity.md) | The Vulkan backend of the layer, with the CUDA direct path, the buffer path and the timeline semaphores as private paths, written from the code that step 8 deleted, which the repository's history keeps; the tensor bridge, the scene service and export on it; the Windows refusals of step 8 go. The CUDA kernel by keys is a row of its own in Windows parity | Every gfx/Vulkan and Vulkan+CUDA row, written against the layer, on Windows hardware; the timing rows and `CALIPER_VULKAN_STATS` within thresholds against the last commit before step 8 | 8 |
| Total | Steps 8 to 14, 16 and 17 | | 47 |

The remaining steps come to 47 builder days, with all of step 8 counted, although its first half is built. Steps 10 to 14 run on macOS, step 16 when its applet needs it, and step 17 with [Windows parity](windows-parity.md).

### Order of the Remaining Steps

1. **Step 8 moves the bridge's allocation table to the layer.** The geometry service draws from the same allocations, so until step 13 its old code on the Metal renderer reads the layer's imported buffers through an adapter; it is never moved onto the layer, because it is deleted once its applets have moved. On Windows both the bridge and the geometry service lose their renderer at this step, which is why the Vulkan renderer is reduced once, here.
2. **Step 10 before the service.** The applets need additive points and instanced meshes, and adding them to the scene model first, for Caliper LIB and the browser too, keeps one model instead of an applet dialect.
3. **Step 11 after the native scene view and its picking.** The service is the applet face of the scene view, which is built with its picking. The three particle applets move first, because each draws one points batch and they prove the in-place route.
4. **Step 12, three applets that use labels, classes, picking and instancing,** while the geometry service still serves the last two.
5. **Step 13, where the service ids change.** The last two applets export, so the export service moves with them; the geometry service is deleted in the step that moves its last client; and the tensor bridge and the export service lose their revisions there too, so the epoch changes once, where the service ids change. Adding the scene service's id in step 11 needed no epoch change, because a new id breaks nothing.
6. **Step 14 after step 13,** because what is left of the seam goes only once no client calls it.
7. **Step 16 when its applet needs it,** by decision. It needs the final services and the final SDK, and it is done when an applet built elsewhere runs on them.
8. **Step 17 with Windows parity,** by decision.

### One Version of Each Graphics Service

The decision is that each graphics service keeps one version, that the ABI epoch is bumped once, and that every applet in this repository is updated in the same changes. Step 13 does it, after steps 11 and 12 have moved most applets onto the scene service. The decision covers graphics only: protocol compatibility (producers of protocol 3.0 and the `/v0/stream` alias) and the revisions of the UI service, like those of the other services that do not draw, stay as they are.

| Service | Today | The one version |
|---|---|---|
| Tensor bridge | `caliper.tensor_bridge.v1`, `v1_1` and `v1_2`, three tables, each a prefix of the next | `caliper.tensor_bridge.v1` with today's `v1_2` table: the six texture operations, `caps`, `import_allocation`, `release_allocation` and `update_texture_from_alloc`. Both caps bits stay, because a backend can truly lack each: stream-ordered handoff, which the browser cannot honour, and import, which the browser and Vulkan without a paired CUDA device cannot do |
| Geometry | `caliper.geometry.v1` to `v1_3`, four tables, and draw records of 192, 216 and 256 bytes | None. It is deleted in step 13, when its last applets have moved, and applets draw in 3-D through the scene service, by decision |
| Scene | None | `caliper.scene.v1`, one table from its first version, added in step 11 ([Scene Service for Applets](scenes.md#scene-service-for-applets)) |
| Export | `caliper.export.v1` and `v1_1` | `caliper.export.v1` with today's `v1_1` table, which adds `last_error`, drawing a scene instead of geometry records ([Export](scenes.md#export)) |
| Embedding views | The v1.2 tensor views (`caliper_core_view_tensor` and the four calls beside it) and the v1.3 typed live views, whose heatmap and image draw through a v1.2 view of the same name | The typed live views alone. The heatmap and image views own a layer texture; the five v1.2 calls go, and `caliper_core_colormap_lut` stays. The slot record of `include/caliper/live.h` hands the slot's host address instead of a Metal buffer, and the views import it as host memory, which the Metal backend wraps in place as today. `CALIPER_EMBED_API_VERSION` becomes 2 with minor version 0, and the package `caliper-embed` becomes 2.0.0 |

**The ids.** The one version of each service keeps the plain `v1` id. The epoch bump already refuses every applet built for the old tables before it can ask for an id, so a new major id would protect nothing, and the applets that list only `v1` ids keep their service lists.

**Growth after the migration.** Within epoch 3 a graphics service grows by appending to its one table, which `struct_size` guards, and its records grow by appending, which each record's own size guards; no new id and no caps bit mark the growth. An incompatible change is an epoch change. The other services keep the growth rules of Compatibility and Epochs (`docs/development/compatibility-and-epochs.md`), which step 13 rewrites for the graphics services. Whether to return to revision ids after a first release is an [open decision](#open-decisions).

**The epoch bump.** Step 13 bumps the ABI epoch from 2 to 3 once, where the service ids change: the geometry ids and the revision ids of the tensor bridge and the export service disappear there. The commit that changes the tables also changes every manifest in the repository, so each commit of the series builds. Any other epoch change that the migration might need, such as a change of the UI-stack pin, lands in the same commit or waits for the next epoch. The SDK becomes 0.3.0, as Compatibility and Epochs allows before 1.0.

**The simplified epoch check.** The check stays, and it becomes simpler. One number, `CALIPER_ABI_EPOCH` in `abi.h`, is read everywhere: by the host's capabilities, by one prototype of `CaliperHost` that the app and the embedding core share (each writes the literal 2 today, in `src/host/ui/applet_page.cpp` and `src/host/embed/embed_core.cpp`), by the SDK's configure check, by the template of `caliper new` and by the test fixtures. The host compares it once before it opens a library, with a refusal that names both epochs and says to rebuild against the SDK of the host's epoch, and after opening the descriptor must agree with its manifest, as today. For the graphics services nothing else is negotiated: there are no revision ids, no caps bit per revision and no widening of records in the sugar.

**Outside this repository.** No other repository is touched before step 16. Applets that live outside this repository build against the SDK, and each stops loading at step 13, refused before its library opens with the epoch message; one that uses the geometry service also stops building against the SDK 0.3.0 until it is rewritten onto the scene service. An embedding host outside this repository that calls the v1.2 views stops compiling against `caliper-embed` 2.0.0 until it moves to the typed live views. Step 16 ports one such applet, by decision; whether and when to update the others is open. The `inspector` branch of this repository is in the same position when it next takes the migration.

### Applets on the Scene Service

Every applet in this repository that draws in 3-D moves onto the scene service, decided on the API's merits: each picture becomes what the scene model describes, and where an applet drew something no general scene model has, the applet changes rather than the model. No ImPlot3D plot in this repository stays, because each is a 3-D point cloud or a fallback for a missing geometry path, not a small chart with axes; ImPlot3D stays in the UI stack for applets whose 3-D plot is such a chart.

| Applet | What it draws today, and through what | What it becomes | Why | Step | Builder days |
|---|---|---|---|---|---|
| `field_scope` | A particle cloud through the geometry service's `draw_points`: additive, speed through magma, 1.5 pixels, with its own orbit camera; an ImPlot3D scatter as its CPU fallback | One additive points batch with values through magma at a pixel size, from its pool's slots, and a starting camera; the camera code and the fallback go | A particle cloud is a scene. The fallback's cases are covered: a CPU tensor is an array the service copies, and a host with no backend refuses with its reason | 11 | 1 |
| `flow_scope` | The same | The same | The same | 11 | 0.75 |
| `sculpt_scope` | The same, coloured by motion | The same | The same | 11 | 0.75 |
| `embed_scope` | ImPlot3D: ten class clouds, centroids as diamonds, the live batch as white sparks, fixed axes, and a hover found by projecting every point on the CPU each frame | Points with class ids and a class table, the centroids as larger points with labels, the live batch as white points, a starting camera in place of the fixed axes, and the hover from the pick pass | A 3-D embedding is a scene, and the per-frame CPU projection goes. Diamonds become discs, and the axes' titles go | 12 | 1.5 |
| `gpt_scope` | ThoughtSpace's residual stations through `draw_points` in three colour modes, with an ImPlot3D fallback; the token embedding as ImPlot3D text glyphs at 3-D positions | Two scenes: the stations as additive points through the mode's colormap, and the embedding as points with one label per token | Both are 3-D clouds; a hovered station or token comes back as an event, and labels are drawn by the view, nearest first, 512 at a time | 12 | 1.5 |
| `instance_scope` | Instanced gems through geometry v1.3: lit, opaque, rigid poses, a per-instance tint through magma; a one-gem ladder without instancing | One instanced mesh with instance poses and instance values through magma | Instanced meshes are now part of the scene model; the rigidity gate and the ladder go | 12 | 1 |
| `mesh_scope` | A lit surface coloured by error, a translucent white wireframe, the minibatch as additive amber points, the three draws exported; an ImPlot heatmap as its fallback | A mesh with normals and values, the wireframe as index-pair edges one pixel wide and translucent, the minibatch as additive points, export of the scene; the fallback goes | Each draw is a scene primitive; the wireframe wins over the surface by the edges' depth offset | 13 | 1.5 |
| `twin_scope` | Two textured meshes side by side, each sampling a bridge texture of its field, a toggle to per-vertex colours, wireframes, export | Two mesh entities placed by entity transforms, each textured from an RGBA image that the applet colormaps on its device or coloured by per-vertex values, edges as index-pair edges, export of the scene | A texture is an image in the scene model, so the field is colormapped by the applet, one gather in torch, instead of by the bridge | 13 | 2 |
| `pulse_scope`, `examples/hello`, `examples/sine_scope` | No 3-D | Unchanged, apart from the manifest's epoch | | 13 | |

**The acceptance of each port.** Each port takes `caliper shot` captures of the applet's page before and after, offscreen. The pictures are not byte-identical, because the drawing path changes on purpose: hardware points become discs, hardware lines become quads, and the camera's defaults move to the shared ones. A reviewer compares them by eye, with every difference listed and explained in the port's report. The applet's own tests pass. Its info line reports the route that ran, and copies per frame on its zero-copy route stay exactly 0 against step 0's baseline. Its frame time and GPU time are measured before and after and reported, but the thresholds of [Fast Path Thresholds](#fast-path-thresholds) do not gate a port, because a port changes what is drawn; in particular, discs of six vertices cost more GPU time than hardware points for the largest particle clouds, and the report says by how much.

### Renderer Seam at Its Final Shape

`HostRenderer` keeps the window, the frame loop and ImGui setup, where the window includes the embed canvas, which is a host's view or an offscreen target, and window capture, which reads the frame that the swapchain presents. It owns one `Backend` of the layer and returns it from `gfx()`, or none where no backend exists. Each step deletes the methods it made unused, in every backend, and step 14 removes what is left.

| Group of the inventory | Methods | What happens | Step |
|---|---|---|---|
| Window and frame | 6 | Stays; `name()` is a label for the log and the export sidecar, never compared | |
| Embed canvas | 7 | Stays, with `canvas_read_pixels`, which `caliper shot` uses; `canvas_supported` goes, and `canvas_init` refuses with its reason instead | 14 |
| Window capture | 3 | Stays; `window_capture_supported` goes, and a request returns false with its reason instead | 14 |
| RGBA8 textures | 4 | Go, for `create_texture`, `write_texture`, `release` and `imgui_texture_id` | 8 |
| Texture updates, one per source | 4 | Go, for `update_texture_from_tensor` and `fill_texture_from_buffer`, which return the route and the path label | 8 |
| Completion, one family per kind of work | 6 | Go, for serials: `submit`, `completed`, `wait` and `last_use` | 7 and 8 |
| Device and interop | 3 | Go: `interop_device` for the caps bits, `native_device` for `import_buffer`, and `alloc_device_shared` for the Vulkan backend's private path behind `kCapCudaWritesTexture` at Windows parity | 8 and 13 |
| Import, one per source | 4 | Go, for `import_buffer`, `import_texture` and `release` | 7 and 8 |
| Geometry, one per ABI revision | 8 | Go with the geometry service, whose applets draw through the scene service | 13 |
| Tests only | 2 | Go: `debug_readback_rgba8` for `read_texture`, and `debug_force_buffer_path` with the Vulkan renderer's buffer path, to return as a test hook of the Vulkan backend | 8 and 14 |
| Measurement, from step 0 | 3 | The copy counters go to the layer's `Stats`; `frame_stats` stays for the frame loop's times and takes GPU times from `Stats` | 14 |

The seam ends with 16 methods: six for the window and the frame, six for the canvas, two for window capture, `gfx()` and `frame_stats`. The Metal, Vulkan and WebGPU renderers implement them.

### Vulkan Renderer Until Windows Parity

The Vulkan backend of the layer comes with Windows parity by decision, and no step keeps an old path. Together these leave the Vulkan renderer with no client from step 8 on: the tensor bridge then draws through the layer, the geometry service and later the scene service draw from the bridge's allocations, and Windows has no backend of the layer until step 17. This plan therefore reduces the Vulkan renderer in step 8, and states what Windows loses until step 17. The gap is accepted: Windows and Vulkan are deferred until work moves to the Windows machine.

- **What the Vulkan renderer keeps.** The window, the swapchain and the frame loop; ImGui, ImPlot and ImPlot3D through `imgui_impl_vulkan`; the embed canvas in a window and offscreen, with the offscreen readback that `caliper shot` uses; and its frame-time statistics. It implements the shrunk `HostRenderer` and nothing more, and its `gfx()` returns no backend.
- **How it keeps compiling.** The bridge, the services and the views are shared code written against the layer. On Windows they find no backend, and nothing in them names the Vulkan renderer, so they compile there unchanged. The Vulkan renderer's target loses the GLSL shaders, the build-time glslang and the generated SPIR-V and PTX headers. `cuda_driver.cpp` stays, because the device query and the SDK's exportable pool use it.
- **What Windows refuses meanwhile.** Every tensor bridge texture: the caps read 0, and creating a texture returns 0 with a log line saying that Windows has no graphics backend until the Vulkan backend of the graphics layer lands. Every geometry view, and from step 11 every applet scene, every PNG export and every embedding view is refused the same way: `caliper.scene.v1` reports no `CALIPER_SCENE_CAP_DRAW`. Applets load, draw their ImGui and ImPlot pages and take their CPU fallbacks, as the fallback rule asks of them, and `caliper shot` and the embed canvas work. The live views and the shared-memory session are macOS-only already.
- **What is unbuilt meanwhile.** The Vulkan renderer's texture, import, CUDA direct, buffer and geometry paths, its six GLSL shaders, the build-time glslang, the direct kernel and its PTX, and the gfx/Vulkan, gfx/Vulkan+CUDA and Vulkan geometry rows. The repository's history keeps them for step 17, except the geometry paths, which no step brings back.
- **What Windows gains.** The headless tests of the layer, the scene core and the services' gates run in the Windows lane on the recording backend, which no Windows test covered before.

## Vulkan Backend of the Graphics Layer

Step 17 adds the Vulkan backend, which implements the same interface as the Metal backend, so that the layer's clients run on Windows as they run on macOS. The table gives the Vulkan side of each concept.

| Concept | Vulkan |
|---|---|
| Buffer, texture, sampler | `VkBuffer` and memory, `VkImage` and view, `VkSampler`, one to one |
| Pipeline | `VkPipeline`, cached as today |
| Bind group | A descriptor set, one to one |
| Per-draw uniform | A dynamic uniform descriptor offset, about as cheap as a push constant |
| Import | `VkImportMemoryWin32HandleInfoKHR` into a `VkBuffer`, as today |
| Ordering | Barriers from the pass list; timeline semaphores shared with CUDA, as today |
| Draw list | Recorded record by record into one command buffer |

The device paths that make Caliper fast stay private paths behind the same interface: the Vulkan backend keeps the CUDA direct path, its second image per texture, and the buffer path below it.

On Vulkan an import of `ExternalKind::Win32Memory` duplicates the handle and owns the imported memory, and its release is deferred until the last submit that read it completed. `ExternalSync::CudaStream` orders the reads after the producer's CUDA stream on the GPU, which is the stream-ordered rung of the sync ladder (`docs/python/zero-copy-tensors.md`). The one inverse case, a CUDA kernel writing a layer texture on the direct path, is `kCapCudaWritesTexture`.

| Source | Route on Vulkan with CUDA |
|---|---|
| Shared-memory slot of the live session | In place once the Windows session puts slots in exportable CUDA memory ([Windows parity](windows-parity.md)) |
| CUDA tensor in the exportable pool | In place |
| CUDA tensor from torch's caching allocator | Textures: the direct kernel, no copy. Buffers: one copy inside the GPU into an exportable buffer |
| Host bytes: a CPU tensor or a network snapshot | One upload |

The route and its label come back from every import and update, and the view's info line shows them.

## Risks

- **A slower fast path.** The layer could add encoding cost or lose Metal's inline constants. The thresholds above catch it, against the commit before each step and against step 0's baseline, and the native backends keep their private paths behind the interface.
- **Pixels that move.** Shaders translated instead of hand-written could change float operation order. The rows of every step that promises unchanged pixels catch it, and a changed row stops the step. The applet ports change their pictures on purpose, and their differences are listed instead.
- **No switch back.** A regression found after a step has landed can no longer be switched off at run time. It is fixed forward, or the step's series is reverted with git, which may also need the later steps that build on it reverted. The fixtures recorded at each step's first commit inherit any defect of the old path; where a fixture and a CPU reference row disagree, the reference row decides.
- **A weaker graphics contract for applets.** The geometry service's rows were byte-exact against a CPU reference for every pixel; the scene service's rows follow the parity rules, with tolerances at edges and on lit and blended pixels ([Mapping from the Geometry Service](scenes.md#mapping-from-the-geometry-service)). This was accepted in the choice of the scene service.
- **Particle clouds cost more GPU time.** Discs of six vertices replace hardware points, so the largest clouds of `field_scope`, `flow_scope` and `sculpt_scope` take more GPU time per frame. Each port reports the numbers; the in-place route and its zero copies are unchanged.
- **The scene service freezes the scene model into the applet ABI.** Within epoch 3 the model can only grow by appending, and a change that cannot append costs an epoch bump and a rebuild of every applet.
- **Applets outside the repository.** From step 13 every applet built against the SDK 0.2.0 is refused before its library opens, one that draws through the geometry service no longer builds against the SDK 0.3.0 until it is rewritten, and an embedding host that calls the v1.2 views no longer compiles against `caliper-embed` 2.0.0. The steps before 16 touch none of them, and step 16 ports one, by decision.
- **Windows.** With the Vulkan backend at Windows parity and no old path kept, Windows refuses bridge textures, geometry views, applet scenes, exports and embedding views from step 8 until step 17 ([Vulkan Renderer Until Windows Parity](#vulkan-renderer-until-windows-parity)). The gap is accepted. The code that step 8 deletes comes back from history at step 17, where it is ported to the layer rather than moved, and the Windows numbers are then compared with a build of the last commit before step 8.
- **Large steps.** Step 11, the scene service with three applets, takes 12.5 builder days, and step 13 changes the ABI, the SDK, the embedding package, the Python package and two applets and deletes the geometry service, in 11. Each lands as a series whose commits all build: in step 13 the commit that bumps the epoch also changes every manifest, and the captures of every applet show that nothing else moved.

## Limits

- Zero-copy holds only on the same machine and only where the backend imports: Metal for `MTLBuffer`s and shared-memory slots, and, from step 17, Vulkan with CUDA for exportable allocations. The browser always copies. A tensor from torch's default CUDA allocator reaches a texture with no copy through the direct kernel, but a buffer use, such as a scene's points, copies it once inside the GPU.
- Until step 13 the geometry service needs storage buffers in the vertex stage; once it is deleted, nothing does, so every draw runs on a WebGPU device in compatibility mode. In the browser, WebGPU's default limits bound what is drawn: 256 MiB buffers, 128 MiB storage bindings and 8,192-pixel textures.
- Windows draws no bridge textures, geometry views, applet scenes, exports or embedding views from step 8 until step 17, an accepted gap.
- Applets lose what the scene model does not describe: triangle strips, unlit triangles, per-draw depth and blend state, camera matrices of their own and the rigidity gate ([Mapping from the Geometry Service](scenes.md#mapping-from-the-geometry-service)).
- The migration updates the applets and embedding hosts in this repository, and, by decision, one applet outside it.

## Changes to the Shared Contracts

- **Every line is an expanded quad.** The contracts say lines are expanded quads everywhere. That now holds without exception: the geometry service's one-pixel hardware lines go with it in step 13.
- **CUDA import covers exportable memory only.** The contracts name "CUDA memory through Vulkan external memory". Only allocations made exportable (the `ExportablePool` and future session slots) can be imported; memory from torch's caching allocator has no handle and reaches the layer by the direct kernel or one copy inside the GPU.
- **The scene model gains additive blending and instanced meshes** for the applets' sake, and they are part of the model for Caliper LIB and the browser too ([Scenes](scenes.md#primitives)).

## Decisions

- **The rest of the migration in full.** The decision is that nothing of the full migration is dropped. After the minimum cut and the steps built on it, every remaining step follows as planned work: the move of the texture path (step 8), the scene model's additions, the scene service and the applets on it (steps 10 to 12), one version of each graphics service with the geometry service deleted (step 13), the seam at its final shape (step 14), the port of one applet outside this repository when that applet needs it (step 16), and the Vulkan backend with Windows parity (step 17). The Dawn fallback that step 15 planned is dropped. Each step is an increment a person can see, under the rule in [Migration](#migration).
- **No backwards compatibility in the graphics change.** The reason is that nothing is deployed or commercialised yet, so nothing outside the repository has to keep working. The decisions below follow from it.
- **No dual paths.** Each step replaces the old code and deletes it in the same series, and a revert is `git revert`. One path at a time is maintained and tested, and the measurements before and after and the byte-exact checks still gate every step.
- **`HostRenderer` shrinks to the window, the frame loop and ImGui setup.** Its feature-shaped drawing, texture, import and geometry methods, its capability booleans and its debug readbacks are what the layer generalises; keeping them would keep a second interface to every backend.
- **One version of each graphics service, and one epoch bump.** Revision ladders, caps bits per revision and the sugar's widening of records exist only to keep older applets working, and no such applet is deployed. Every applet in this repository is updated in the same series; applets outside it are not touched, and whether to update them is open.
- **The hand-written shader copies and the platform code in the views deleted as each client moves.** The generated programs and `import_buffer` replace them, so no copy is left to drift and no view branches on a platform, a device kind or a renderer's name. The Metal compute colormap, the GLSL `colormap.comp` and the CUDA kernel are the copies left, and step 8 deletes them.
- **Zero-copy unchanged, and a hard requirement.** In-place import through `import_buffer` is measured in every step against the commit before it and against step 0's baseline.
- **Protocol compatibility and the UI service's revisions are not part of this.** Producers of protocol 3.0, the `/v0/stream` alias and the revisions of `caliper.ui` and of the other services that do not draw stay as they are.
- **The epoch bump in step 13, where the service ids change.** The geometry ids and the revision ids of the tensor bridge and the export service disappear there, so the epoch changes once; the scene service's id was added in step 11 without one, because a new id breaks nothing.
- **The one version keeps the plain `v1` id.** The epoch bump already refuses every applet built for the old tables, so a new major id would protect nothing.
- **Export draws scenes.** Its tables carried the geometry record, so they change with the geometry service anyway, and drawing a scene keeps one drawing path.
- **The typed live views are the one embedding view.** They already draw heatmaps and images through the v1.2 views by name, so folding the v1.2 views into them removes a layer and the Python package's switch between two routes.
- **A scene service for applets, replacing the geometry service.** Applets get the scene model, the shared camera, picking with events, the overlays, export and, where the scene is wire-capable, recordings and the browser, from one service instead of a fixed menu of draws with cameras of their own, and Caliper keeps one drawing world. It costs about 25.5 builder days of migration in place of about 8.5 for moving the geometry service onto the layer and giving it one version, rewrites eight applets, and changes the graphics contract for applets from byte-exact rows to the parity rules.
- **The applets adapt to the API, not the API to the applets.** The scene service is designed as a general API; a capability of the geometry service that no general scene model has is dropped and the applet changes, and the port of the applet outside this repository adds a feature to Caliper only where it is general in its own right.
- **The particle applets first, the exporting applets last.** The particle applets each draw one points batch and prove the in-place route; the exporting applets move with the export service, in the step that deletes the geometry service.
- **One applet outside this repository ported, as the last step.** It shows that an applet built elsewhere runs on the final services; the other applets outside this repository stay open for a later decision. It is later made a step done when that applet needs it.
- **Build a thin layer with three backends: Metal, Vulkan and WebGPU.** No existing library keeps both zero-copy paths under a licence Caliper can ship. The WebGPU backend serves the browser.
- **The Vulkan backend comes with Windows parity, not on macOS.** It needs Windows hardware with CUDA, and Caliper LIB's sessions are macOS-only today. [Windows parity](windows-parity.md) carries the step. With no old path kept, Windows loses its GPU drawing from step 8 until then.
- **The gap on Windows is accepted.** The decision is that Windows and Vulkan are deferred until work moves to the Windows machine, and the gap is accepted. From step 8, Windows draws its ImGui pages and refuses bridge textures, geometry views, applet scenes, exports and embedding views with a reason, until the Vulkan backend of the layer lands with [Windows parity](windows-parity.md).
- **Instance-step vertex buffers for per-item data, and no vertex storage once the geometry service is gone.** Vertex buffers take any 4-byte offset and any stride, so imported slots and array columns bind without a copy, an alignment rule or a bind group per draw. Vertex pulling served only the geometry service, so `kCapVertexStorage` and bind group 3 go with it in step 13.
- **CUDA kernels stay hand-written** and share the colormap's C++ header with the CPU. 

## Open Decisions

- **How a graphics service changes after the migration.** Options: (a) append to its one table and record within an epoch, and bump the epoch for an incompatible change, as [One Version of Each Graphics Service](#one-version-of-each-graphics-service) plans; (b) return to revision ids with a caps bit per revision. Recommendation: (a) until a first release reaches users outside this repository, then decide again.
- **Applets and embedding hosts outside this repository, apart from the one that step 16 ports.** Options: update each in its own repository once step 13 lands, or leave each until it is next worked on. Recommendation: update each when step 13 lands, since an applet without 3-D needs only its manifest and its service ids, while one that used the geometry service needs the rewrite that the applets here get. This plan touches none of them.
