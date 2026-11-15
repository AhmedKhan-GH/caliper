# Caliper's views in the browser

This folder builds Caliper's own view layer (the live views, Dear ImGui and ImPlot) for the browser. It is compiled to WebAssembly with Emscripten and drawn by the WebGPU backend behind `HostRenderer` (`src/host/renderer/webgpu_renderer.*`). It is a separate Emscripten project, not part of the native CMake build.

## Layout

| Path | What it is |
|---|---|
| `CMakeLists.txt` | The Emscripten project: the view-layer library `caliper_view_layer` and the module `caliper-views` |
| `build.sh` | The one build command |
| `src/main.cpp` | The page host: the canvas, the ImGui and ImPlot contexts, the renderer and a hosted core that draws the views |
| `src/web_services.cpp` | The tensor bridge vending that `host_services.cpp` does natively |
| `demo.html` | A page that loads the module onto one canvas |
| `test/` | The headless check (Playwright, headless Chromium only) |

The view-layer library compiles these shared files unchanged: `src/host/live/live_views.cpp`, `live_views_draw.cpp` and `live_unsupported.cpp`, `src/host/embed/embed_views.cpp` (the embed v1.2 views and the hosted core, split out of `embed_core.cpp`), `src/host/tensor_bridge.cpp` and the WebGPU renderer. It does not compile `embed_core.cpp`, so nothing of the applet loader, the service registry, the job system or the app paths is referenced.

## Building

With an activated emsdk checkout, and cmake and ninja on the path:

```sh
EMSDK=<emsdk> web/wasm/build.sh <build dir>
```

The output is `<build dir>/caliper-views.wasm` and its JavaScript glue `<build dir>/caliper-views.js`, an ES module whose default export is `createCaliperViews(moduleArgs)`. `moduleArgs.canvas` is the canvas the instance draws on; every instance looks up only its own canvas, so several instances can share a page. The build was measured with Emscripten 6.0.13. Its first run downloads Emscripten's `emdawnwebgpu` port into the emsdk cache.

## Checking

```sh
cd web/wasm/test && npm install
node check.mjs <build dir> <screenshot.png> swiftshader 300
```

The check serves the build directory on 127.0.0.1, opens `demo.html` in headless Chromium, waits for the frames, prints the module's counters as JSON and saves screenshots. `swiftshader` uses Chromium's software WebGPU adapter and `metal` the Mac's GPU.
