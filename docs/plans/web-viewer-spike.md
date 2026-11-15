# Web viewer spike

This page reports a spike that asked one question. Can a WebGPU backend behind `HostRenderer`, together with an Emscripten build of Caliper's view layer (Dear ImGui, ImPlot and the live views), draw one heatmap and one series in a browser? The answer is yes. The prototype lives in `spikes/web-viewer/`, outside the main build, and this page records what it proved, what it cost and what the full [web viewer](streaming-and-embedding.md#the-web-viewer) still needs.

## The result

The real view layer runs in a browser. `live_views.cpp` and `live_views_draw.cpp` compiled for WebAssembly without a single change. The heatmap is colormapped on the GPU by a hand-written WGSL compute shader, and the series is drawn by ImPlot. A synthetic source compiled into the module stands in for the network transport: every frame it applies a heatmap snapshot (a moving 2-D pattern) and a series snapshot (a noisy decaying curve) through `caliper_core_live_apply`, exactly as the shared-memory transport does natively.

The following was verified in headless Chromium, driven by Playwright and never shown on screen:

- Both live views draw, as screenshots show, on two WebGPU adapters: SwiftShader (Chromium's software rasteriser) and the machine's GPU through Metal.
- The module reads the heatmap texture back and compares it with the CPU reference `map_f32_to_rgba8`. On SwiftShader every texel matched at every size tried, up to 2048 × 2048. On Metal every texel matched at 256 × 192 and 1024 × 1024, but at 1536 × 1536 and 2048 × 2048 a few texels (2 to 18 of several million) differed by one level in one channel. The [shader question](#the-shader-question) discusses why.
- Mouse input reaches ImGui. Moving the mouse over the series plot shows ImPlot's hover marker and the live view's tooltip (step and value).
- The frame loop holds the browser's 60 frames per second at the default heatmap size, and the view layer's own CPU work is about 0.3 ms per frame.

The following was not verified: keyboard and text input, canvas resizing, HiDPI (the test ran at a device scale factor of 1), browsers other than Chromium, Windows, long runs and memory growth, the points, image and histogram live views, and any network transport.

## How the prototype is built

- **Toolchain.** Emscripten 6.0.13 from a project-local emsdk. It was the newest 6.x release, four weeks old; 7.0.0 had been out for nine days and was skipped as too fresh for a major version.
- **WebGPU binding.** Emscripten's `emdawnwebgpu` port (`--use-port=emdawnwebgpu`, Dawn package v20260423.175430 as pinned by the SDK). This SDK no longer has the older `-sUSE_WEBGPU` binding, so `emdawnwebgpu` is the only choice. It provides the standard `webgpu.h`, which is also what Dawn offers natively, so the same backend code could later serve the native Dawn fallback that the [roadmap](roadmap.md#renderers) plans.
- **Window and input.** Emscripten's built-in GLFW (`-sUSE_GLFW=3`) with `GLFW_NO_API`, so the canvas gets a WebGPU context, and the unchanged `imgui_impl_glfw` for input. This keeps the native shape of `HostRenderer::init(GLFWwindow*)`.
- **Start-up.** A browser hands out a WebGPU device only through promises. The page host requests the adapter and the device with callbacks and starts the frame loop from the device callback, so the build needs neither Asyncify nor JSPI.
- **The page host** (`spikes/web-viewer/src/web_main.cpp`) plays the part the Caliper app plays natively. It owns the GLFW window, the ImGui and ImPlot contexts and the renderer, and draws the view layer through a hosted core (`caliper_core_create_hosted` and `caliper_core_hosted_frame`).
- **The renderer** (`spikes/web-viewer/src/webgpu_renderer.cpp`) implements the swapchain path (`init`, `new_frame`, `render`, `shutdown`) through `imgui_impl_wgpu`, RGBA8 textures, and the WGSL colormap. Everything else keeps the interface's default refusal: there are no device tensors, no imported allocations, no geometry and no embed canvas.
- **One additive seam.** The tensor bridge chooses its GPU path by the tensor's device, and in a browser every snapshot is host memory, so the bridge would always colormap on the CPU. The spike adds `HostRenderer::tex_update_mapped_from_host`, which defaults to `false`, and the bridge tries it before CPU staging. Native backends do not override it, so their behaviour is unchanged.
- **Release flags.** `-O3`, Closure over the JavaScript glue, `-sFILESYSTEM=0` (the viewer never opens a file) and `-sALLOW_MEMORY_GROWTH=1`.

To build and check it, with an activated emsdk and with cmake and ninja on the path:

```sh
EMSDK=<emsdk> spikes/web-viewer/build.sh <build dir> Release
cd spikes/web-viewer/test && npm install
node check.mjs <build dir> <screenshot.png> swiftshader 600
node check.mjs <build dir> <screenshot.png> metal 600 1024x1024
```

The check serves the build from a static server bound to 127.0.0.1, waits for the frames and the colormap check, prints the module's report as JSON, and saves a screenshot and a second screenshot with the mouse over the series plot.

## Which files compiled

| File | Status | Note |
|---|---|---|
| `src/host/live/live_views.cpp` | Unchanged | Its own non-Apple block stands in for `live_views_metal.mm` |
| `src/host/live/live_views_draw.cpp` | Unchanged | |
| `src/host/live/live_unsupported.cpp` | Unchanged | The existing stand-in for `live_core.cpp` and `live_attach.cpp` on platforms without the shared-memory session |
| `src/host/embed/embed_core.cpp` | Unchanged | Its create, shutdown and applet paths reference 22 native-only symbols (the applet loader, the service registry, the job system, the app paths and the core lifecycle). The hosted core never reaches them, so the linker drops them |
| `src/host/tensor_bridge.cpp` | Changed, 7 lines | Tries the new hook before CPU staging |
| `src/host/renderer/host_renderer.h` | Changed, 13 lines | Adds `tex_update_mapped_from_host` with a default of `false` |
| `src/host/host_services.cpp` | Replaced | `spikes/web-viewer/src/web_stubs.cpp` provides the three functions the views use (`services_set_renderer`, `services_renderer`, `services_bridge`); the real file pulls in DuckDB, the stores and the job system |
| `live_core.cpp`, `live_attach.cpp`, `live_client.mm`, `live_session.mm`, `live_activity.mm`, `live_views_metal.mm`, `viewer_main.mm` | Not compiled | The shared-memory transport and the macOS viewer shell |
| `metal_renderer.mm`, `vulkan_renderer.cpp`, `gl_renderer.cpp` | Not compiled | Replaced by `spikes/web-viewer/src/webgpu_renderer.cpp` |
| Dear ImGui, ImPlot, `imgui_impl_wgpu`, `imgui_impl_glfw` | Unchanged | ImPlot3D, the node editor and the text editor were not needed |

The plan says the web viewer has no tensor bridge. Today the heatmap and image live views are built on the embed v1.2 tensor views, which go through `TensorBridge`, so the spike kept the bridge. It is plain C++ and compiled with only the new hook added. Dropping it would mean giving the texture live views their own path to the renderer.

## The shader question

The renderers carry three copies of each shader today:

| Shader | Metal (MSL strings in `metal_renderer.mm`) | Vulkan (GLSL in `src/host/renderer/shaders`, compiled to SPIR-V by glslang) | CUDA (`kernels/cmap_direct.cu`, checked in as PTX) |
|---|---|---|---|
| Colormap, f32 to RGBA8 | `cmap_f32` compute kernel, 31 lines | `colormap.comp` compute, 43 lines | `caliper_cmap_f32` |
| u8 image copy | A blit, no shader | Buffer copy, no shader | `caliper_blit_rgba8` |
| Points | `points_vs`, `points_fs`, 47 lines | `points.vert`, `points.frag`, 52 lines | none |
| Geometry | `geom_vs`, `geom_vs_point`, `geom_fs`, `geom_tex_fs`, 190 lines | `geom.vert`, `geom.frag`, `geom_tex.frag`, 184 lines | none |

The colormap is the only compute shader. The CUDA kernels serve the Windows interop path and have no counterpart in a browser, so the web viewer needs the colormap, points and geometry shaders, seven entry points in all.

Translating them mechanically does not carry their design across. A Tint or Naga translation of the SPIR-V, or a Slang source with a WGSL target, meets three WebGPU limits that no translator removes:

- **Push constants do not exist in WebGPU.** The Vulkan colormap and geometry shaders take their parameters as push constants, so every one becomes a uniform buffer with its own bind group entry.
- **WebGPU points are always one pixel.** `points.vert` and `geom.vert` write `gl_PointSize` (`point_size` in MSL). The points live view must be redrawn as instanced quads, which is a change of design rather than of syntax.
- **Storage buffers in the vertex stage are capped.** `geom.vert` pulls its vertices from eight read-only storage buffers plus one uniform buffer, which is exactly WebGPU's default limit of eight per stage. WebGPU's compatibility mode allows none in the vertex stage.

Smaller differences matter for exact output. WGSL has no `isnan`, and an implementation may assume that NaN never reaches float arithmetic, so the WGSL colormap tests the bits instead. WGSL also allows a division error of 2.5 ULP. The one-level differences on Metal at large sizes fall within that freedom, while SwiftShader matched the CPU reference exactly. Caliper's byte-identical colormap contract therefore holds in the browser only on some adapters, and the full viewer must either accept a one-level tolerance there or find an exact formulation.

The hand-written WGSL colormap is 31 lines and produced exact output on SwiftShader. For the web viewer's three shaders, hand-written WGSL is the cheaper choice now. A single Slang source becomes worth its toolchain once the WebGPU backend also serves the native Dawn fallback and every shader exists three times, which is the evaluation the roadmap already lists.

## Bundle size

Release build, `-O3` with Closure and without the file system:

| File | Raw | gzip -9 | Brotli 11 |
|---|---|---|---|
| `caliper_web_spike.wasm` | 776,636 B | 340,313 B | 276,104 B |
| `caliper_web_spike.js` | 47,563 B | 17,772 B | 15,377 B |
| Total | 824,199 B | 358,085 B | 291,481 B |

A page therefore transfers about 290 KB with Brotli. Dropping Emscripten's file system saved 18 KB of JavaScript (65 KB with it).

The WebAssembly module holds 697 KB of code and a 77 KB data segment, of which ImGui's two embedded default fonts take 24 KB. To attribute the code by origin without the link-time inlining of the optimiser, the module was relinked at `-O1` with function names kept (838 KB of code). By origin:

| Origin | Share of code |
|---|---|
| Dear ImGui | 57% |
| ImPlot | 18% |
| Caliper (view layer, embed core, tensor bridge, renderer and page host) | 10% |
| libc and compiler runtime | 8% |
| stb_truetype (font rasterising) | 2.5% |
| emdawnwebgpu (the C side of `webgpu.h`) | 1.6% |
| libc++ | 1.5% |
| `imgui_impl_wgpu` and `imgui_impl_glfw` | 1.6% |

Dear ImGui and ImPlot are three quarters of the module. Caliper's own code is a tenth. In the JavaScript glue before minification, the WebGPU bindings and the GLFW library are the largest parts after the runtime.

## Frame time

Measured headless with the release build, 300 frames per run after 30 warm-up frames. The CPU columns are mean milliseconds per frame on the browser's main thread. "Synthetic" is the synthetic source computing its data, which a real viewer does not do. "Viewer" is the two `caliper_core_live_apply` calls, the ImGui frame with the live views' draw code, and the renderer's encoding and submit. "GPU" is the time from the frame's submit to the queue reporting the work done, which includes the colormap and the upload ahead of it in the queue.

| Adapter | Heatmap | Frames per second | Viewer CPU (ms) | Synthetic (ms) | GPU (ms) | Texels unlike the CPU reference |
|---|---|---|---|---|---|---|
| SwiftShader | 256 × 192 | 60 | 0.32 | 0.46 | 3.8 | 0 |
| SwiftShader | 1024 × 1024 | 60 | 0.32 | 8.5 | 9.3 | 0 |
| SwiftShader | 2048 × 2048 | 28.8 | 0.52 | 34.0 | 69.8 | 0 |
| Metal | 256 × 192 | 60 | 0.32 | 0.53 | 0.7 | 0 |
| Metal | 1024 × 1024 | 60 | 0.30 | 8.4 | 1.3 | 0 |
| Metal | 2048 × 2048 | 28.5 | 0.62 | 34.2 | 70.5 | 18, each one level |

SwiftShader is a software rasteriser, so its GPU column measures the CPU emulating a GPU and says nothing about a real GPU. Sixty frames per second is the browser's animation-frame rate, so the first four rows are capped, not saturated. At 2048 × 2048 the frame rate falls because the synthetic source takes 34 ms. The GPU time then rises to 70 ms on both adapters because every frame sends 16 MB from WebAssembly memory to the browser's GPU process, and the queue backs up. A browser viewer pays for every byte of a snapshot twice, once over the network and once into the GPU process, which supports the decided plan of sampling and data views rather than whole tensors.

## Findings in the view layer

- **The heatmap's path label is wrong in the browser.** `embed_core.cpp` labels every CPU payload "cpu-staged" without asking the renderer, so the view's info line says "cpu-staged" while the WGSL colormap ran. The renderer itself reports "webgpu-compute".
- **Heatmap hover shows no value.** `live_views_draw.cpp` keeps a CPU view of the payload only for an imported Metal buffer, so a CPU payload gets "value unavailable" in the tooltip. In a browser every payload is a CPU payload, so the viewer must keep the last snapshot's values for hover.
- **The latency figure needs a shared clock.** A snapshot's `stamp_ns` is the producer's monotonic clock, and the view computes latency against its own. In a browser that clock is unrelated to the producer's, so the wire protocol needs a clock offset or the browser must stop reporting latency.
- **The points live view is shaped around Metal.** `bind_buffer` in `live_views.cpp` accepts a Metal buffer, or stages a CPU payload into a new shared `MTLBuffer` on the renderer's Metal device. In a browser it refuses with "CPU payload needs a Metal renderer". Points therefore need a change in `live_views.cpp`, not only a renderer: a host-memory import through the renderer.
- **Releasing a texture can leave a stale bind group.** `imgui_impl_wgpu` caches one bind group per texture view and is never told that a view was released, so a later view at the same address would reuse it. The spike never releases a texture while drawing it; the full renderer must invalidate that cache.
- **The tests' `WAIT_GPU` path sleeps on the frame thread.** The browser's main thread must never block, so the points apply's bounded wait has to stay test-only or be removed for the web build.
- **Headless Chromium needs specific flags.** Playwright's headless Chromium (`chromium-headless-shell`, Chromium 158) gives WebGPU a SwiftShader adapter, but with only `--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU` it cannot present a WebGPU canvas: `getCurrentTexture()` returns an invalid texture and the device is lost with "A valid external Instance reference no longer exists". The same failure appeared in a page of plain JavaScript, so it is not the module's fault. Adding `--use-angle=swiftshader` makes the software path work, and `--use-angle=metal` gives the machine's GPU. The full Chrome for Testing binary in its new headless mode also worked with a plain JavaScript page, on Metal by default and on SwiftShader with `--use-webgpu-adapter=swiftshader`.

## What the full viewer needs

| Work item | Rough size | Notes |
|---|---|---|
| The wire protocol client | 1 to 2 weeks | A WebSocket client that decodes snapshots into WebAssembly memory and applies them, keeps the latest snapshot per live view, appends series deltas to a ring, and sends data view requests. It replaces the synthetic source. No threads are needed: the socket's callbacks run on the main thread between frames |
| A view-layer library without the native core | 3 to 5 days | Split the views and the hosted core out of `embed_core.cpp`, so the web build does not depend on the linker dropping its applet and service paths, and decide whether the texture live views keep the tensor bridge |
| Points and ImPlot3D | 1 to 2 weeks | A WebGPU geometry pipeline that draws points as instanced quads within the storage-buffer limits, a host-memory import for `bind_buffer`, and ImPlot3D compiled in. Camera and hover code can be reused |
| Histogram and image live views | 1 to 2 days | Both are compiled in already. The histogram is a CPU copy drawn by ImPlot, and the image goes through the RGBA8 upload; neither was exercised |
| Input | 2 to 3 days | Mouse and wheel already work through GLFW. Keyboard, text input, the clipboard, cursors and touch need checking |
| Canvas size and HiDPI | 2 to 3 days | Follow the element's size and `devicePixelRatio`, reconfigure the surface, and scale the style and fonts |
| Fonts | 1 to 2 days | Embed or fetch the Caliper fonts. Each TrueType file adds roughly its own size to the download |
| Hover values, path labels and latency | 1 to 2 days | The three view-layer findings above |
| Memory | 2 to 3 days | WebAssembly here is 32-bit, so the heap stops at 4 GB, and every snapshot is copied into the GPU process. Bound the snapshot sizes the viewer accepts and measure growth over long sessions |
| Packaging | 1 to 2 weeks | The `<caliper-view>` element, TypeScript types and the Jupyter widget around the module |
| CI | 2 to 3 days | Cache emsdk and the port download, build the module, and run the headless check with the SwiftShader flags on a macOS runner |
| Colormap exactness | 1 to 3 days | Accept a one-level tolerance in the browser, or find a WGSL formulation that is exact on every adapter |

## Risks

- **Browser support.** Chromium-based browsers ship WebGPU. Support in Safari and Firefox depends on the version and the platform, and the spike tested only Chromium. A WebGL2 fallback would be a second renderer for the browser.
- **API stability.** The core of `webgpu.h` is stable, but emdawnwebgpu's Emscripten-specific parts and `webgpu_cpp.h` are not, and the port is pinned by each Emscripten release.
- **Headless testing.** CI depends on SwiftShader and on flags that changed the outcome on this machine. A Chromium update can change them again.
- **Exact pixels.** WGSL does not promise the CPU reference's arithmetic, and the spike already saw one-level differences on Metal.
- **Download size.** About 290 KB with Brotli today. ImPlot3D, fonts and the wire client will add to it, and ImGui and ImPlot already dominate.
- **Data volume.** Large snapshots cost a copy into the GPU process on top of the network, so the viewer relies on the producer's sampling and data views.

## Recommendation

Build the web viewer on this path. The view layer compiled for WebAssembly without changes to the live views, a WebGPU backend behind `HostRenderer` took about 530 lines with its header, and both live views drew on a software and a hardware adapter with the view layer's own work at about 0.3 ms per frame and a download of about 290 KB. The next step is the wire protocol client in place of the synthetic source, together with splitting the view layer from the native embed core so the web build stops relying on the linker to drop the applet and service paths. Points and ImPlot3D, the largest remaining item, should follow as a WebGPU geometry pipeline with instanced quads and a host-memory import in `live_views.cpp`. Keep the WGSL shaders hand-written for now, decide on a one-level colormap tolerance for the browser, and run the headless check in CI with the SwiftShader flags from the start.
