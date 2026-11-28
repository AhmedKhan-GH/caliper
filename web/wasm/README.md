# Caliper's views in the browser

This folder builds Caliper's own view layer (the live views, Dear ImGui and ImPlot) for the browser. It is compiled to WebAssembly with Emscripten and drawn by the WebGPU backend behind `HostRenderer` (`src/host/renderer/webgpu_renderer.*`), which draws the views' resources through the graphics layer of the renderer core and its WebGPU backend (`src/host/renderer/webgpu_backend.*`). It is a separate Emscripten project, not part of the native CMake build. These views are the only WebGPU drawing in a browser: the `<caliper-view>` element's own views, written in TypeScript, draw with Canvas 2D.

## Layout

| Path | What it is |
|---|---|
| `CMakeLists.txt` | The Emscripten project: the view-layer library `caliper_view_layer` and the module `caliper-views` |
| `emscripten-version.txt` | The Emscripten release the module is built with, in CI and by `scripts/build_wasm.py` |
| `src/main.cpp` | The page host: the canvas, the ImGui and ImPlot contexts, the renderer and a hosted core that draws the views |
| `src/net_source.*` | The viewer side of wire protocol v3 on the browser's WebSocket, as a `caliper_live::Source` |
| `src/source_poll.*` | The per-frame poll of a `Source` into the live views, in the order `live_core.cpp` uses for shared memory |
| `src/web_input.*` | The canvas size, the device pixel ratio, pointer, wheel, keyboard and focus input, and the mouse cursor |
| `src/fonts.*` | The native app's two faces from the design token package, embedded in the module |
| `src/web_services.cpp` | The tensor bridge vending that `host_services.cpp` does natively |
| `src/gfx_selftest.*` | The graphics layer's self-test in the page, which the browser tests start ([Checking](#checking)) |
| `src/scene_selftest.*` | The scene rows in the page: one scene per primitive drawn by the scene view and compared with the CPU reference ([Checking](#checking)) |
| `src/scene_parity.*` | The browser's side of the scene comparison with the app: the scene fixtures drawn by the scene view with the library pieces ([Checking](#checking)) |
| `caliper-views-loader.js` | The loader: `mountCaliperViews(container, options)`, an ES module with no dependencies |
| `demo.html` | A page that loads the module onto one canvas and connects it to a stream named in its URL |
| `test/stream_server.py` | A synthetic producer: one session with a heatmap, a series, a histogram and an image, served over wire protocol v3 with `python/caliper/_wire.py` |
| `test/scene.mjs`, `test/scene_server.py` | The headless check of a scene (the wire protocol 3.2 draft): a script logs demo 0's kind of scene through `caliper.serve()`, and the viewer page draws it in headless Chromium |
| `test/check.mjs` | The headless check of one page (Playwright, headless Chromium only) |
| `test/multi.mjs`, `test/pages/multi.html` | The headless check of several mounts on one page through the loader |
| `test/cross_origin.mjs` | The headless check of the loader imported from another origin, as a notebook page does |

## What compiles

The module compiles the native app's view code unchanged, so the views in the browser are the app's views. Only the parts that a browser cannot have are left out or stood in for.

| Native files | In the module |
|---|---|
| `src/host/live/live_views.cpp`, `live_views_draw.cpp` | Compiled unchanged. The non-Apple block of `live_views.cpp` stands in for `live_views_metal.mm` |
| `src/host/live/live_scene.cpp`, `live_views_scene.cpp`, `src/host/scene/*.cpp` | Compiled unchanged: the scene view and the scene core it draws through |
| `src/host/draw/scene_pieces.cpp` | Compiled unchanged: the converter of each scene batch into a program of a library piece, with which the scene view draws its shapes, as in the app |
| `src/builder/spec/toml_form.cpp` and the spec sources it needs | Compiled unchanged, with toml++: the reader of the library pieces' manifests, which the converter calls |
| `src/host/live/live_unsupported.cpp` | Compiled: the stand-in for the shared-memory attach of `live_core.cpp` and `live_attach.cpp` |
| `src/host/embed/embed_views.cpp` | Compiled unchanged: the embed v1.2 views and the hosted core (`caliper_core_create_hosted`, `caliper_core_hosted_frame`), split out of `embed_core.cpp` |
| `src/host/tensor_bridge.cpp` | Compiled unchanged. The heatmap and image views keep the bridge, which colormaps through the renderer's `tex_update_mapped_from_host` when the renderer offers it, as the WebGPU renderer does; native renderers keep the default `false` |
| `src/host/net/wire.cpp`, `json.cpp` | Compiled unchanged into `caliper_wire`, the codec of wire protocol v3 that the native source and the hub use too |
| `src/host/ui/sternbild_fonts.cpp`, `sternbild_theme.cpp` | Compiled unchanged: the app's faces and style |
| `src/host/host_services.cpp` | Replaced by `src/web_services.cpp`, which provides the three functions the views use (`services_set_renderer`, `services_renderer`, `services_bridge`); the real file pulls in DuckDB, the stores and the job system |
| `embed_core.cpp`, `live_core.cpp`, `live_attach.cpp`, `live_net_source.mm`, `live_session.mm`, `live_client.mm`, `live_views_metal.mm`, `viewer_main.mm` | Not compiled: the instrument loader, the services, the jobs and the app paths, the native transports and the macOS viewer shell |
| `src/host/gfx/core.cpp`, `src/host/renderer/webgpu_backend.cpp` | Compiled: the graphics layer's core (validation, the uniform arena, frames, serials and readback) and its WebGPU backend on the standard `webgpu.h` |
| `metal_renderer.mm`, `vulkan_renderer.cpp` | Not compiled. `src/host/renderer/webgpu_renderer.cpp` draws instead |
| Dear ImGui, ImPlot, `imgui_impl_wgpu` | Compiled unchanged. ImPlot3D, the node editor and the text editor are not compiled |

The module has no GLFW: the renderer draws on the canvas's WebGPU surface, and `src/web_input.cpp` feeds ImGui from the canvas's own listeners.

## Building

The module builds with Emscripten at the version in `emscripten-version.txt`, 6.0.13 at present, with CMake and Ninja on the path. From the repository root:

```sh
python scripts/build_wasm.py                    # emcc on PATH, as a sourced emsdk_env.sh puts it there
EMSDK=<emsdk> python scripts/build_wasm.py      # or from an emsdk checkout with that version activated
```

The script refuses another Emscripten version unless it is given `--any-version`, because the sizes and frame times below, and the WebGPU binding, belong to the pinned one. It builds into `web/wasm/build` (`--build-dir`; `--config Debug` for a debug build) and copies the three files a page loads, `caliper-views-loader.js`, `caliper-views.js` and `caliper-views.wasm`, into `python/caliper/_web/` and into the caliper package that the running Python has installed, if any. `--out <dir>` copies them elsewhere instead, and `--no-copy` only builds. The first build downloads Emscripten's `emdawnwebgpu` port into the emsdk cache.

The output is `caliper-views.wasm` and its JavaScript glue `caliper-views.js`, an ES module whose default export is `createCaliperViews(moduleArgs)`, and beside them a copy of the loader and of `demo.html`. `moduleArgs.canvas` is the canvas the instance draws on; every instance looks up only its own canvas, so several instances can share a page.

The WebGPU binding is that port (`--use-port=emdawnwebgpu`), whose Dawn package each Emscripten release pins; this SDK no longer has the older `-sUSE_WEBGPU` binding. The port provides the standard `webgpu.h`, which Dawn also offers natively, so the same renderer can later serve the native Dawn fallback that the roadmap plans. A browser hands out a WebGPU device only through promises, so `src/main.cpp` requests the adapter and the device with callbacks and starts the frame loop from the device's callback; the module needs neither Asyncify nor JSPI. The Release link runs Closure over the glue, has no file system (`-sFILESYSTEM=0`; the views never open a file) and caps the heap, as [Memory](#memory) describes.

## Where the files come from

The three files are build outputs and are not checked in; git ignores them in `python/caliper/_web/`. CI builds them for every change to their sources and uploads them as the artifact `caliper-views` ([In CI](#in-ci)), and a wheel built in CI ships them. In a checkout, either command puts them into `python/caliper/_web/` and into the caliper package that the running Python has installed, if any:

```sh
python scripts/build_wasm.py      # builds them; needs the pinned Emscripten, CMake and Ninja
python scripts/fetch_wasm.py      # downloads CI's build; needs only the standard library
```

`fetch_wasm.py` takes CI's build of the checkout's commit, or of the newest earlier commit whose sources of the module are the same, and otherwise the asset `caliper-views.zip` of the latest release. GitHub hands out CI's builds only to signed-in clients, so the first source needs a token in `GITHUB_TOKEN` or `GH_TOKEN`, which the script sends only to the GitHub API; a release needs none. It checks the files before it copies them, and it says plainly when it is offline or when no build exists yet. A wheel built locally without the files warns, and one built in CI fails (`python/check_web_views.cmake`).

Without the files, nothing breaks: the `<caliper-view>` element, the notebook widgets and the viewer page draw every view with the views written in TypeScript, and `caliper.serve()` prints one line, once, naming the two commands.

## Mounting the views on a page

`caliper-views-loader.js` is the one entry point for a page, the `<caliper-view>` element and the notebook widget:

```js
import { mountCaliperViews } from '<assetBase>caliper-views-loader.js';
const handle = await mountCaliperViews(container, {
  src,          // 'ws://127.0.0.1:<port>/v3/stream?token=<token>'
  view,         // the view to show; '' or absent: every view of the session
  kind,         // 'heatmap', 'image', 'points', 'series', 'histogram' or 'scene'; optional
  height,       // CSS pixels; optional, the default fills the container
  header,       // false: the page draws the view's header, so the view shows itself alone (default true)
  assetBase,    // the folder of caliper-views.js and .wasm; default: the loader's own folder
  onSelect,     // (detail, view) => {}; every selection, {} when it was cleared
  onEvent,      // ({ view, type, detail }) => {}; every select, pick and hover
  onInfo,       // (facts, view) => {}; what a header says about the view, as it changes
  onDataView,   // (dataView, view) => {}; every data view the producer resolved
});
await handle.ready;                              // the first snapshot is drawn
handle.setDataView({ index: { layer: 2 } });     // a VIEW_REQUEST; false when it could not be sent
handle.stats;                                    // the module's counters
handle.destroy();                                // BYE, the GPU released, the canvas removed
```

With `header: false` a view shows only itself: no title bar, no diagnostics line (size, path, sequence number, latency, updates, drops) and no Pause or Fit button, while hover values stay (`CaliperCore::views_bare`; the native app keeps all of it). In the browser the views also take the page palette (`src/page_style.cpp`): the view's ground is the plate colour, a series line is parchment, hover marks and selections are bronze, a fitted y range reaches 6% of the data's span past it on each side, and rules and tick labels are drawn as the TypeScript views draw them, so the two renderers of the element look alike.

The loader exports `CALIPER_VIEWS_KINDS`, which lists `heatmap`, `image`, `points`, `series`, `histogram` and `scene`.
The module's scene view uses library pieces through `src/host/draw/scene_pieces.h` and the draw runtime on WebGPU.
Its facts include entity counts, item counts, copied bytes and GPU time.
The `<caliper-view>` element uses the kind list to select a renderer and uses `onInfo` facts for the header.
[Renderer Selection](../../docs/web/browser-elements.md#renderer-selection) describes this choice.

## Pan, zoom and follow

The views pan and zoom as the element's TypeScript views do, in the browser and in the native app alike, and only the plot area takes a drag, so a floating window in the app moves only by its title bar. A heatmap or image zooms about the pointer with the wheel, pans with a drag and fits again on a double click; magnified so far that a cell covers a physical pixel or more, it is sampled nearest so that cells stay cells, and hover, picks and selections report the cell under the pointer at any zoom. A series or histogram fits its axes to the data every frame while it follows; a drag or the wheel stops it following and keeps the user's view while new values arrive, and a double click follows again. A double click works at any frame rate: ImGui takes one change of a mouse button per frame, so its second press arrives two frames after the first, and the page's input allows those two frames when they outlast ImGui's 0.3 s, as they do below about 7 frames a second on a software adapter. The element's header has the same `log` and `follow` toggles for both renderers: the page sets them with `handle.setToggle(name, on)`, and the loader's `onToggle(name, on)` reports a view that changed one itself. In the browser (a bare view with an event sink), a heatmap or image also takes the element's layout: axes of row and column indices that follow pan and zoom and are labelled with source indices, the dims' names, a colorbar beside a heatmap, a heatmap stretched to the plot area and an image at equal aspect. In the app the cells stay square, between the view's info line and its telemetry bar. The axes' tick labels are the element's 10.5 px mono. ImGui sizes a face by its line height and CSS by its em, and Source Code Pro is 1.257 em from ascent to descent, so the views draw them at 13.2 ImGui px; at 10.5 they came out at about 8 CSS px.

## Selections, picks and hovers

The views select, pick and hover with the gestures of the element's TypeScript views (`web/src/views/interact.ts`), and report the same details (wire protocol v3, "Events"). A shift-drag selects: in points a lasso, or a rectangle with shift and alt; in a heatmap or an image a block of cells, kept inside the tile where it began in a mosaic; in a series a range of steps; in a histogram a range of bins. A shift-click clears the selection, which an empty detail reports. A click picks the point, cell, step or bin under the pointer, and a hover is reported when the hovered item changes, at most 30 times a second. In 3-D points a drag orbits and a right, middle, alt or ctrl drag pans; in 2-D points any drag pans and the wheel zooms. The points outside a selection are dimmed, and the count of selected points shows in the corner.

The details are in source indices: each snapshot's payload axes (`axes`, `tile`) go with it into the view layer, which maps a payload index to the source positions it stands for. Each event goes to the producer as an EVENT on the module's own connection, with the subscription, the `seq` and the data view of the snapshot the view shows, and to the page through `onEvent`, and `onSelect` for selections. The element passes a selection on as its `caliper-select` event, so a widget's `selection` and the server's `events()` receive the same detail whichever renderer drew the view.

The seam is the view layer's, so the native app can use it too: a host sets an event sink on the core (`caliper_live_views::set_event_sink`), and a `caliper_live::Source` sends what it hears with `send_event` and asks for data views with `request_data_view` (`src/host/live/live_source.h`). Without a sink, as in the native app today, a selection stays in its view and nothing is reported. A host that draws [linked selections](../../docs/reference/wire-protocol.md#linked-selection) gives the views the selected items of a space with `caliper_live_views::set_link`: points draw over the dimmed rest, the rows or columns of a heatmap or image that stand for a selected item are outlined, and a series gets markers at the selected steps.

The loader creates the canvas inside `container` and hands that element to the module, so a container inside a shadow root works and nothing is looked up in the document. `preloadCaliperViews(assetBase)` starts the download and compilation early.

Each mount is its own module instance with its own canvas, WebGPU device, heap and WebSocket. The compiled WebAssembly module is fetched and compiled once per page and shared by every mount. This was chosen over one instance drawing several canvases after measuring the two costs that matter, with three mounts made one after another on one page (`test/multi.mjs`). Start-up: each mount drew its first snapshot 33 to 43 ms after `mountCaliperViews` was called on the Mac's GPU, and 74 to 99 ms on SwiftShader, most of it the WebGPU device request and the connection; the second and third mounts were no slower than the first. Memory: an instance reserves a 16 MB heap, of which one view uses 1.2 to 1.9 MB. One instance with several canvases would need a surface, an ImGui context and a socket per canvas inside one module, which is not worth that saving.

## Serving

A server that hands the views to a page, such as `caliper.serve()`, serves three files from one folder, the `assetBase`:

| File | Content type |
|---|---|
| `caliper-views-loader.js` | `text/javascript` |
| `caliper-views.js` | `text/javascript` |
| `caliper-views.wasm` | `application/wasm` (the loader falls back to compiling from the bytes under another type, more slowly) |

A notebook page imports the loader from another origin, so each of the three needs `Access-Control-Allow-Origin` for the page's origin, for example the loopback origins that the stream's Origin rule accepts, or `*`. The requests are simple CORS GETs without credentials, so no preflight is answered and no other CORS header is needed. Nothing needs cross-origin isolation, because the module uses no threads and no SharedArrayBuffer. The fonts are inside the module, so there are no other files.

## Input, size and look

Every listener sits on the instance's own canvas: pointer events with pointer capture for drags, the wheel, keys while the canvas has focus (a click gives it focus) and focus changes. The page keeps a wheel turn or a key only when ImGui does not want it. Each frame the canvas's backing store is sized to its CSS box times the device pixel ratio, so the canvas follows its element and text is rasterised at the screen's density. The views look as they do in the native app: the module embeds the app's faces (IBM Plex Sans Condensed and Source Code Pro, unchanged from the design token package, both under the SIL Open Font License 1.1) and applies its style with `sternbild::load_fonts_from_memory` and `sternbild::apply_style`.

## Shaders

The renderer's textures are colormapped by a program written by hand in WGSL, `kColormap` in `src/host/renderer/webgpu_programs.h`, a `ProgramDesc` of the graphics layer with its bind group layout, which is to be replaced by the shader generator's table; the points and scene views draw with the generated programs of the scene core. The hand-written programs were not translated from the native shaders, because three limits of WebGPU change their design rather than their syntax:

- WebGPU has no push constants, so the parameters that the Vulkan shaders take as push constants are uniform blocks here, written into the layer's uniform arena and bound at a dynamic offset.
- WebGPU draws a point as one pixel, so points are instanced quads ([Points](#points)).
- A vertex stage may read at most eight storage buffers by default, and none in WebGPU's compatibility mode. The points therefore read their positions and values from instance-step vertex buffers and their colormap from the draw's uniform block of keys, so they also draw on a compatibility-mode device. Library pieces supply the scene draw path through the draw runtime.

WGSL also has no `isnan`, lets an implementation assume that no NaN or infinity reaches float arithmetic, and allows a division 2.5 ULP of error. The colormap below is built so that none of that changes a byte.

## The colormap

A heatmap is colormapped on the GPU by a compute shader written in WGSL (`kColormap` in `src/host/renderer/webgpu_programs.h`), one dispatch of the graphics layer per update, and it gives exactly the bytes of the CPU reference `map_f32_to_rgba8` on every adapter. Recomputing the reference's formula on the GPU did not: WGSL allows a division 2.5 ULP of error and lets an implementation assume that no NaN or infinity reaches float arithmetic, and on Metal a few texels of a large heatmap came out one level away from the reference. The shader therefore does no arithmetic on the value. For each index k from 1 to 255 the CPU finds, by a binary search over the floats that runs the reference itself, the threshold at which the reference first gives index k or more (the last such float for a decreasing range). The reference is monotone in the value, because each IEEE operation in it is, so a value's index is the number of thresholds it has passed, and the shader finds that number with eight exact float comparisons. NaN takes index 0, tested on the bits, and a flat range gives index 0 everywhere, as in the reference. The thresholds change only with the range, and computing them takes about 8,000 evaluations of the reference, well under a millisecond.

The module's check compares each heatmap texture with the reference over the same snapshot. It found no differing texel on SwiftShader and on Metal at 256 × 192, 1024 × 1024, 1536 × 1536 and 2048 × 2048, nor for the synthetic producer's `edges` view, whose values lie on both sides of every index boundary of an increasing and a decreasing range, with NaN and both infinities among them. The native Metal and Vulkan shaders keep their own formulas and their own byte-exact tests.

## Points

A points view draws through the scene core over the renderer's backend of the graphics layer, as the native view does (`src/host/live/live_points.h`): instances of `points@1`, each point an instanced quad, because WebGPU draws a point as one pixel, with the view's sizes, colours and depth test. A points payload arrives as bytes, so the view imports each buffer with the layer's `import_buffer`; a browser cannot wrap host memory, so the layer copies it into a GPU buffer and reports the route `copied-from-host`, and the view's path label reads `points-cpu-staged`. The layer keeps released buffers and reuses one of the same capacity, a power of two, for the next snapshot, because creating and destroying a buffer per snapshot backed the GPU queue up by seconds. Hover is the native one, over a copy of the positions. Measured headless on the Mac's GPU before the view moved onto the scene core, 100,000 points streamed at 60 snapshots per second orbited at 56 to 60 frames per second. The module's checks compare the render target with the palette colour of each sampled point that no other point covers, on both adapters; in 3-D they compare only the points in front of the camera's target, which the depth fade leaves at full strength, and size each neighbour by its depth.

In 3-D the shared view code draws a box around the data with ImGui, its back edges and floor grid before the points' image and its other edges and tick labels after it, and the render target's ground is transparent so that the back layer shows through. The points draw the view's depth cues through `points@1` (the parameter block's `att_ref`, `fade_w0` and `fade_w1`, which `PointsScene` sets from the orbit's distance and the fitted bounding sphere): a point is larger the nearer it is, and the points behind the camera's target are dimmed with their hue kept, natively and here alike. The rotate tool and `set_toggle(view, "rotate", on)` turn the view; `set_toggle(view, "depth", false)` turns the cues off.

## Memory

The heap starts at 16 MiB and can grow to 2 GiB. Growth allocates memory only when needed. The ceiling covers 512 MiB of retained resources and two resource assemblies of up to 256 MiB each. Pick copies, scene arrays, and frame decode also use the heap. The module keeps the newest resource versions and no step history. A series keeps its newest 65,536 values in a fixed ring. The module reports `heapSize` and `heapUsed` in its counters.

Measured headless (SwiftShader) over 7 minutes of streaming at 60 snapshots per second, with all four views on one page and a 512 × 512 heatmap and image (46.6 GB received): the heap never grew from its 16 MB, and the bytes in use rose from 11.8 MB to 12.3 MB in the first two and a half minutes, while the series filled its plot buffers, and then stayed at exactly 12,287,952 bytes for the last four minutes. A page with only the heatmap view stayed at 5,320,656 bytes from the first sample to the last. The frame rate stayed at 60 per second throughout.

A data view change that changes the payload's shape makes the heatmap a new texture, and a resize makes the points a new render target. The graphics layer destroys a released texture once the last submit that used it completed, and releases its view at the end of the frame: the ImGui backend makes one bind group per texture view while it records a frame and releases them all when the frame is recorded, so nothing released outlives the frame. The module reports what the renderer holds as `textures`, `allocations` (the points' buffers and the layer's spare buffers) and `retiredViews`, and the layer's own buffers as `gfxBuffers` and `gfxSpare`. Measured headless on the Mac's GPU over 300 data view changes and 300 resizes after a warm-up (`test/wasm.spec.ts`), which made 300 new textures and 300 new render targets: the renderer held 2 textures and 7 buffers before and after (the points' buffers and the layer's spare buffers), no released view was left waiting, and the bytes in use of the two modules went from 2,667,152 to 2,666,952. The page's JavaScript heap, after a garbage collection, stayed between 3.0 and 3.35 MB over 900 changes, moving by no more than it moves in 18 seconds of streaming without changes.

## Size

Release build, as `python/caliper/_web` ships it:

| File | Raw | gzip -9 | Brotli 11 |
|---|---|---|---|
| `caliper-views.wasm` | 1,647,981 B | 706,006 B | 565,087 B |
| `caliper-views.js` | 34,335 B | 13,189 B | 11,497 B |
| `caliper-views-loader.js` | 7,616 B | 2,962 B | 2,486 B |

A page therefore transfers about 579 KB of the module with Brotli, once per page however many views it mounts. Drawing through the graphics layer added 130,764 bytes to the module (47,712 with gzip, 34,940 with Brotli) over the build of the commit before it, which was 1,488,484 bytes: the layer's core, with its validation and handle tables, and its WebGPU backend, and the self-test the browser tests start. The copy counters of the live views and the scene messages of wire protocol 3.2 in the codec add the other 28,733 bytes. The breakdown below was measured before that. The module holds about 0.97 MB of code and a 407 KB data segment, of which the two embedded faces take 321 KB (Source Code Pro 210 KB, IBM Plex Sans Condensed 111 KB). Attributed by origin in a relink at `-O1` that keeps function names, where little is inlined across origins, the code is:

| Origin | Share of code |
|---|---|
| Dear ImGui | 44% |
| Caliper | 29%: the wire codec and the source about half, the view layer and the hosted core a quarter, then the renderer, the tensor bridge and the page host |
| ImPlot | 12% |
| libc and compiler runtime | 8% |
| libc++ | 2.5% |
| emdawnwebgpu (the C side of `webgpu.h`) | 1.6% |
| stb_truetype | 1.4% |
| `imgui_impl_wgpu` | 0.6% |

Dear ImGui and ImPlot are more than half of the code, and the faces more than three quarters of the data.

## Frame time

`check.mjs` reports the module's mean work per frame on the browser's main thread: `pollMs` applies what arrived, `uiMs` builds the ImGui frame with the live views, `encodeMs` records and submits it, and `gpuMs` is the time from submit to the queue reporting the work done. With the synthetic producer's six views (a 256 × 192 heatmap, a series, a histogram, a 64 × 64 heatmap, 4,000 points in 3-D and a 256 × 192 image) on a 1280 × 800 canvas at 60 snapshots per second each:

| Adapter | Frames per second | CPU per frame (ms) | Poll (ms) | ImGui frame (ms) | Encode (ms) | GPU (ms) |
|---|---|---|---|---|---|---|
| Metal | 60 | 0.82 | 0.52 | 0.15 | 0.13 | 0.68 |
| SwiftShader | 15 | 0.74 | 0.40 | 0.20 | 0.14 | 255 |

The Metal row is the median of three rounds with the views drawing through the graphics layer, each measured beside the build before it on a quiet machine, which gave the same figures within their spread (CPU 0.82 ms, poll 0.52 ms, ImGui 0.16 ms, encode 0.15 ms, GPU 0.68 ms); 100,000 points orbited at 59.8 frames per second with both. The SwiftShader row was measured before the graphics layer. Sixty frames per second is the browser's animation-frame rate, so the Metal row is capped, not saturated. SwiftShader rasterises on the CPU, so its GPU column measures that emulation and says nothing about a real GPU. Large snapshots cost a second copy, from WebAssembly memory into the browser's GPU process. With a 2048 × 2048 heatmap updated every frame, 16 MB went into the GPU process per frame and the time from submit to done rose to about 70 ms on both adapters, because the queue backed up. That is one reason producers send data views rather than whole tensors.

## The stream

An instance reads its settings from `Module.caliperConfig`: `src` is the stream's URL, `ws://127.0.0.1:<port>/v3/stream?token=<token>`, `view` names the one view to show (every view of the session when empty) and `kind` the kind that view must have. The instance connects as a viewer: HELLO, WELCOME, the server's default attach, SUBSCRIBE for its views at 60 snapshots per second, then SNAPSHOT and SERIES messages, which the live views draw. It answers PING, pings after 10 seconds without input and closes after 30. `Module._caliper_views_set_data_view(json)` sends a VIEW_REQUEST for the shown view, and `Module.onDataView(view, dataView)` hears every data view the producer resolved. The frames go through the C++ codec in `src/host/net/wire.h`, which the native viewer uses too. Each snapshot's stamp is mapped into the page's clock (`src/host/live/live_clock.h`) before the live views see it, so their latency figure is the delay above the fastest recent message, which each view labels "rel lat"; the protocol has no clock exchange, so a constant transit time is not included.

## Checking

The check needs Node, a Python with `numpy` and `websockets`, and Playwright's Chromium headless shell in its browser cache.

```sh
cd web/wasm/test && npm install
CALIPER_PYTHON=<python> node check.mjs --build <build dir> --out <shot.png> --adapter swiftshader
CALIPER_PYTHON=<python> node check.mjs --build <build dir> --out <shot.png> --view loss --kind series --hover 400,150
```

The check starts `stream_server.py`, which serves the build directory and the stream on one port on 127.0.0.1, opens `demo.html` in headless Chromium, waits for frames and applied snapshots, prints the module's counters as JSON and saves screenshots. `swiftshader` uses Chromium's software WebGPU adapter and `metal` the Mac's GPU. The flags matter: Playwright's headless shell gives WebGPU a SwiftShader adapter with only `--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU`, but cannot present a WebGPU canvas then (`getCurrentTexture()` returns an invalid texture and the device is lost), in a page of plain JavaScript as well. `--use-angle=swiftshader` makes the software path present, and `--use-angle=metal` gives the Mac's GPU; the checks and the web package's tests pass these. The module also reads back each heatmap and image texture once and compares it, byte for byte, with what the CPU makes of the same snapshot (the reference colormap for a heatmap, the bytes themselves for an image); its counters list the results as `checks`. `--compat 1` (and `?compat=1` on `demo.html`, `compat: true` in the loader's options) asks for a WebGPU device in compatibility mode.

The web package's tests (`web/test/wasm.spec.ts`) check the graphics layer itself in a page, on both WebGPU projects and on a default and a compatibility-mode device. `Module._caliper_views_gfx_selftest()` (`src/gfx_selftest.cpp`) submits one frame through the layer: two compute dispatches into a storage texture with their parameters in the frame's uniform arena, a copy from a buffer with 256-byte rows into a texture, and a draw list with an RG32Uint ID target and a depth buffer that draws an indexed quad (with an index offset and a base vertex), an instanced draw fed by a vertex-step stream, instance-step streams at a 4-byte offset, a stride-0 constant and colours imported from host memory, depth-tested against the quad, and an alpha-blended instanced draw from a first instance under a scissor. Its draw program is released before the frame is submitted, which its pipelines and bind groups survive. It reads every target back through the layer's asynchronous readback and compares each texel exactly with the CPU's expectation and checks that the import reports `copied-from-host` with its bytes. Two more frames check the refusals: a draw that samples the colour target it renders into, which the layer's core refuses, and a dispatch that binds one buffer as read-only and read-write storage, which WebGPU refuses and the error scope around the submit reports. The report comes in `caliperStats.gfxSelftest`. `Module._caliper_views_colormap_keys(vmin, vmax)` puts in `Module.caliperColormapKeys` the colormap keys that the module derives at run time from its own build of the CPU reference, as its scene views do, and a test compares them with those of the case file that the native tool `caliper_colormap_cases` writes (`CALIPER_COLORMAP_CASES`), from which the browser's colormap rows take their keys and expected colours. Another test saves the fixed heatmaps and points of `mock_server.py --fixed` as pictures and, with `CALIPER_VIEWS_REFERENCE` naming a folder of pictures saved from an earlier build, requires every byte to be the same, which is how a migration step of the renderer core shows that the views' pixels did not move.

`web/test/scenes.spec.ts` runs scene fixtures on default and compatibility-mode devices.
`Module._caliper_views_scene_selftest(name)` decodes their protocol messages, applies them to a scene view and draws at 256 by 256 pixels.
The result must contain library pieces and nonclear pixels, with no view notes.
The report appears in `caliperStats.sceneSelftest`, and `Module.caliperScenePixels` holds the picture.

`web/test/draw_parity.spec.ts` compares these pieces with the app's output.
`Module._caliper_views_scene_parity(config)` draws the shared fixtures and reads color and ID targets.
`caliper_scene_parity_app` draws the same fixtures on Metal.
The browser test compares the pictures, piece counts and picks with the app's results.
The native scene tests separately compare with saved images from the renderer before graphics consolidation.
WebGPU copies the complete depth texture before it extracts a pick window.

## In CI

The Browser views workflow (`.github/workflows/web.yml`) runs on a pull request or a push to `main` that changes a source of the module or the web package. Its first job, on a macOS runner, installs the pinned Emscripten from the emsdk tag of the same name, caching the emsdk folder, whose `upstream/emscripten/cache` holds the downloaded `emdawnwebgpu` port. It builds the module with `scripts/build_wasm.py`, runs the checks above and the web package's tests, and checks that the checked-in TypeScript bundle matches a fresh build. The tests run in two steps. On SwiftShader, which needs no GPU, one test at a time runs the specs that draw with WebGPU: the WebAssembly views, the renderer choice and its fallback, the 3-D points, the colormap's shaders, linked selection and the specs that read pixels. On Canvas 2D every spec runs, among them the protocol and round-trip specs (codec, controls and e2e) and the frame-rate measurement, which in CI run there only (`web/playwright.config.ts`). It then uploads the three files as the artifact `caliper-views`. Its second job, on pushes to `main` and on request, puts that artifact into `python/caliper/_web/` and builds the wheel, which fails without it. The `webgpu` project of the web package's tests and the 100,000-point measurement need a GPU and are left out.

The job's commands, which also run on a Mac with Node and Python 3.12 or later:

```sh
# Emscripten, pinned
v=$(cat web/wasm/emscripten-version.txt)
git clone --depth 1 --branch "$v" https://github.com/emscripten-core/emsdk.git ~/emsdk
~/emsdk/emsdk install "$v" && ~/emsdk/emsdk activate "$v"

# The module, built and copied into python/caliper/_web/
EMSDK=~/emsdk python3 scripts/build_wasm.py

# The checks of the module (Playwright's Chromium headless shell, SwiftShader)
python3 -m venv /tmp/py && /tmp/py/bin/pip install numpy websockets pytest pyyaml
export CALIPER_PYTHON=/tmp/py/bin/python
(cd web/wasm/test && npm ci && npx playwright-core install chromium-headless-shell &&
  node check.mjs --build ../build --out /tmp/shots/views.png --adapter swiftshader &&
  node check.mjs --build ../build --out /tmp/shots/edges.png --view edges --kind heatmap &&
  node check.mjs --build ../build --out /tmp/shots/points.png --view embedding --kind points --dims 2 &&
  node multi.mjs --build ../build --out /tmp/shots/multi.png --destroy 1 &&
  node cross_origin.mjs --build ../build --out /tmp/shots/cross-origin.png)

# The element with these views and without them, the TypeScript bundle, and the packaged files
(cd web && npm ci && npx playwright install chromium-headless-shell && node build.mjs --test &&
  npx playwright test --project=swiftshader --project=canvas2d)
"$CALIPER_PYTHON" scripts/build_web.py --check --no-install
CALIPER_REQUIRE_WEB_VIEWS=1 PYTHONPATH=python "$CALIPER_PYTHON" -m pytest --noconftest \
  python/tests/test_assets.py python/tests/test_wasm_scripts.py
```

Each `check.mjs` run exits with 1 when the module reports an error or a check finds a differing texel or point.
