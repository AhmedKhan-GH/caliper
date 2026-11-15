# Caliper's views in the browser

This folder builds Caliper's own view layer (the live views, Dear ImGui and ImPlot) for the browser. It is compiled to WebAssembly with Emscripten and drawn by the WebGPU backend behind `HostRenderer` (`src/host/renderer/webgpu_renderer.*`). It is a separate Emscripten project, not part of the native CMake build.

## Layout

| Path | What it is |
|---|---|
| `CMakeLists.txt` | The Emscripten project: the view-layer library `caliper_view_layer` and the module `caliper-views` |
| `build.sh` | The one build command |
| `src/main.cpp` | The page host: the canvas, the ImGui and ImPlot contexts, the renderer and a hosted core that draws the views |
| `src/net_source.*` | The viewer side of wire protocol v3 on the browser's WebSocket, as a `caliper_live::Source` |
| `src/source_poll.*` | The per-frame poll of a `Source` into the live views, in the order `live_core.cpp` uses for shared memory |
| `src/web_input.*` | The canvas size, the device pixel ratio, pointer, wheel, keyboard and focus input, and the mouse cursor |
| `src/fonts.*` | The native app's two faces from `assets/fonts`, embedded in the module |
| `src/web_services.cpp` | The tensor bridge vending that `host_services.cpp` does natively |
| `caliper-views-loader.js` | The loader: `mountCaliperViews(container, options)`, an ES module with no dependencies |
| `demo.html` | A page that loads the module onto one canvas and connects it to a stream named in its URL |
| `test/stream_server.py` | A synthetic producer: one session with a heatmap, a series, a histogram and an image, served over wire protocol v3 with `python/caliper/_wire.py` |
| `test/check.mjs` | The headless check of one page (Playwright, headless Chromium only) |
| `test/multi.mjs`, `test/pages/multi.html` | The headless check of several mounts on one page through the loader |
| `test/cross_origin.mjs` | The headless check of the loader imported from another origin, as a notebook page does |

The view-layer library compiles these shared files unchanged: `src/host/live/live_views.cpp`, `live_views_draw.cpp` and `live_unsupported.cpp`, `src/host/embed/embed_views.cpp` (the embed v1.2 views and the hosted core, split out of `embed_core.cpp`), `src/host/tensor_bridge.cpp` and the WebGPU renderer. It does not compile `embed_core.cpp`, so nothing of the applet loader, the service registry, the job system or the app paths is referenced.

## Building

With an activated emsdk checkout, and cmake and ninja on the path:

```sh
EMSDK=<emsdk> web/wasm/build.sh <build dir>
```

The output is `<build dir>/caliper-views.wasm` and its JavaScript glue `<build dir>/caliper-views.js`, an ES module whose default export is `createCaliperViews(moduleArgs)`. `moduleArgs.canvas` is the canvas the instance draws on; every instance looks up only its own canvas, so several instances can share a page. The build was measured with Emscripten 6.0.13. Its first run downloads Emscripten's `emdawnwebgpu` port into the emsdk cache.

## Mounting the views on a page

`caliper-views-loader.js` is the one entry point for a page, the `<caliper-view>` element and the notebook widget:

```js
import { mountCaliperViews } from '<assetBase>caliper-views-loader.js';
const handle = await mountCaliperViews(container, {
  src,          // 'ws://127.0.0.1:<port>/v3/stream?token=<token>'
  view,         // the view to show; '' or absent: every view of the session
  kind,         // 'heatmap', 'image', 'points', 'series' or 'histogram'; optional
  height,       // CSS pixels; optional, the default fills the container
  assetBase,    // the folder of caliper-views.js and .wasm; default: the loader's own folder
  onSelect,     // (detail, view) => {}; the views emit no selections yet
  onDataView,   // (dataView, view) => {}; every data view the producer resolved
});
await handle.ready;                              // the first snapshot is drawn
handle.setDataView({ index: { layer: 2 } });     // a VIEW_REQUEST; false when it could not be sent
handle.stats;                                    // the module's counters
handle.destroy();                                // BYE, the GPU released, the canvas removed
```

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

Every listener sits on the instance's own canvas: pointer events with pointer capture for drags, the wheel, keys while the canvas has focus (a click gives it focus) and focus changes. The page keeps a wheel turn or a key only when ImGui does not want it. Each frame the canvas's backing store is sized to its CSS box times the device pixel ratio, so the canvas follows its element and text is rasterised at the screen's density. The views look as they do in the native app: the module embeds the app's faces (IBM Plex Sans Condensed and Source Code Pro, unchanged from `assets/fonts`, both under the SIL Open Font License 1.1) and applies its style with `sternbild::load_fonts_from_memory` and `sternbild::apply_style`.

## Memory

The heap is capped: it starts at 16 MB and may grow to 512 MB, which holds the protocol's largest frame (64 MB) about four times over, the most one snapshot needs while it is applied. Nothing grows with the length of a run. The latest snapshot of a view replaces the one before in two buffers that keep their capacity, a series keeps its newest 65,536 values in a fixed ring, and the socket's per-message buffer is freed after each message. The module reports `heapSize` and `heapUsed` (the bytes malloc has handed out) in its counters.

Measured headless (SwiftShader) over 7 minutes of streaming at 60 snapshots per second, with all four views on one page and a 512 × 512 heatmap and image (46.6 GB received): the heap never grew from its 16 MB, and the bytes in use rose from 11.8 MB to 12.3 MB in the first two and a half minutes, while the series filled its plot buffers, and then stayed at exactly 12,287,952 bytes for the last four minutes. A page with only the heatmap view stayed at 5,320,656 bytes from the first sample to the last. The frame rate stayed at 60 per second throughout.

## The stream

An instance reads its settings from `Module.caliperConfig`: `src` is the stream's URL, `ws://127.0.0.1:<port>/v3/stream?token=<token>`, `view` names the one view to show (every view of the session when empty) and `kind` the kind that view must have. The instance connects as a viewer: HELLO, WELCOME, the server's default attach, SUBSCRIBE for its views at 60 snapshots per second, then SNAPSHOT and SERIES messages, which the live views draw. It answers PING, pings after 10 seconds without input and closes after 30. `Module._caliper_views_set_data_view(json)` sends a VIEW_REQUEST for the shown view, and `Module.onDataView(view, dataView)` hears every data view the producer resolved. The frames go through the C++ codec in `src/host/net/wire.h`, which the native viewer uses too. Each snapshot's stamp is mapped into the page's clock (`src/host/live/live_clock.h`) before the live views see it, so their latency figure is the delay above the fastest recent message; the protocol has no clock exchange, so a constant transit time is not included.

## Checking

The check needs Node, a Python with `numpy` and `websockets`, and Playwright's Chromium headless shell in its browser cache.

```sh
cd web/wasm/test && npm install
CALIPER_PYTHON=<python> node check.mjs --build <build dir> --out <shot.png> --adapter swiftshader
CALIPER_PYTHON=<python> node check.mjs --build <build dir> --out <shot.png> --view loss --kind series --hover 400,150
```

The check starts `stream_server.py`, which serves the build directory and the stream on one port on 127.0.0.1, opens `demo.html` in headless Chromium, waits for frames and applied snapshots, prints the module's counters as JSON and saves screenshots. `swiftshader` uses Chromium's software WebGPU adapter and `metal` the Mac's GPU. When a heatmap is shown, the module also reads its texture back and counts the texels that differ from the CPU colormap.
