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
| `src/web_services.cpp` | The tensor bridge vending that `host_services.cpp` does natively |
| `demo.html` | A page that loads the module onto one canvas and connects it to a stream named in its URL |
| `test/stream_server.py` | A synthetic producer: one session with a heatmap, a series, a histogram and an image, served over wire protocol v3 with `python/caliper/_wire.py` |
| `test/check.mjs` | The headless check (Playwright, headless Chromium only) |

The view-layer library compiles these shared files unchanged: `src/host/live/live_views.cpp`, `live_views_draw.cpp` and `live_unsupported.cpp`, `src/host/embed/embed_views.cpp` (the embed v1.2 views and the hosted core, split out of `embed_core.cpp`), `src/host/tensor_bridge.cpp` and the WebGPU renderer. It does not compile `embed_core.cpp`, so nothing of the applet loader, the service registry, the job system or the app paths is referenced.

## Building

With an activated emsdk checkout, and cmake and ninja on the path:

```sh
EMSDK=<emsdk> web/wasm/build.sh <build dir>
```

The output is `<build dir>/caliper-views.wasm` and its JavaScript glue `<build dir>/caliper-views.js`, an ES module whose default export is `createCaliperViews(moduleArgs)`. `moduleArgs.canvas` is the canvas the instance draws on; every instance looks up only its own canvas, so several instances can share a page. The build was measured with Emscripten 6.0.13. Its first run downloads Emscripten's `emdawnwebgpu` port into the emsdk cache.

## The stream

An instance reads its settings from `Module.caliperConfig`: `src` is the stream's URL, `ws://127.0.0.1:<port>/v3/stream?token=<token>`, `view` names the one view to show (every view of the session when empty) and `kind` the kind that view must have. The instance connects as a viewer: HELLO, WELCOME, the server's default attach, SUBSCRIBE for its views at 60 snapshots per second, then SNAPSHOT and SERIES messages, which the live views draw. It answers PING, pings after 10 seconds without input and closes after 30. `Module._caliper_views_set_data_view(json)` sends a VIEW_REQUEST for the shown view, and `Module.onDataView(view, dataView)` hears every data view the producer resolved. The frames go through the C++ codec in `src/host/net/wire.h`, which the native viewer uses too.

## Checking

The check needs Node, a Python with `numpy` and `websockets`, and Playwright's Chromium headless shell in its browser cache.

```sh
cd web/wasm/test && npm install
CALIPER_PYTHON=<python> node check.mjs --build <build dir> --out <shot.png> --adapter swiftshader
CALIPER_PYTHON=<python> node check.mjs --build <build dir> --out <shot.png> --view loss --kind series --hover 400,150
```

The check starts `stream_server.py`, which serves the build directory and the stream on one port on 127.0.0.1, opens `demo.html` in headless Chromium, waits for frames and applied snapshots, prints the module's counters as JSON and saves screenshots. `swiftshader` uses Chromium's software WebGPU adapter and `metal` the Mac's GPU. When a heatmap is shown, the module also reads its texture back and counts the texels that differ from the CPU colormap.
