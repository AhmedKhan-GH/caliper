# Testing

Caliper's tests use doctest and run through ctest. This page describes the layers that exist, what each proves, and how the tests that need a GPU report when there is none. The sources are in `tests/`, and `tests/CMakeLists.txt` registers them.

## The fast unit suite

`caliper_tests` is one executable that needs no window and no GPU. It holds the layers below, one source file per layer.

| Layer | What it proves |
|---|---|
| Manifest | The parser accepts a valid manifest, rejects a missing or malformed field, validates the applet id and ignores unknown keys. |
| Negotiation | The pre-load checks run in order: epoch, `min_host`, then required services, each with its refusal message. |
| Loader | The loader scans a folder, launches the `hello` applet through its full lifecycle, tears down an old instance before a relaunch, checks that the descriptor agrees with the manifest, shows a missing binary as a failed card, and refuses a path-escaping id. |
| Crash guard | A fault or an exception in an applet call is contained and reported, in the frame call and in a background job. The exception cases use `tests/fixtures/throwing_applet`, a hand-written descriptor that throws across the ABI, which the sugar would otherwise catch. |
| UI recovery | The host's UI stack is unwound after an applet fault left windows, plots and style pushes open. |
| Frame watchdog | Three consecutive slow frames latch the flag, a good frame resets the streak, and the flag stays until it is reset. |
| Services | Each service store and wrapper has its own tests: jobs, device, metrics, feed, artifacts, data and the tensor bridge. |
| Sugar | `CALIPER_APPLET`, the typed service wrappers, exception catching at the ABI edge and a run at a `dpi_scale` of 2. |
| ABI | `test_abi.cpp` asserts that the ABI structs are standard layout with `struct_size` first, and checks the epoch and every published table. |
| C compile check | `abi_c_check.c` includes every public header, newest first, in a C translation unit, so a header that is missing its own includes cannot hide behind another one. |

The same executable also covers the applet-side logic that needs neither torch nor a window, such as the instanced pose helper, the pulse ring and the export affordance.

The `hello` applet and `throwing_applet` are built first, because the loader tests load them from the build folder.

## The fixture host

`sdk/testing/caliper/fixture_host.h` is a headless fake `CaliperHost`. It serves `caliper.log.v1` and returns `NULL` for every other service, and a test can inject a table for any other service with `provide`. It lets an applet's logic and the sugar run without launching the UI. Only one fixture exists per process. The CMake target is `caliper::sdk_testing`; see [C++ sugar](../references/sugar.md) for its use in an applet's own tests.

## The pixel-exact graphics suite

`caliper_gfx_tests` (`tests/gfx/gfx_main.cpp`) checks the path from a tensor to a texture and the geometry path. It opens a hidden window, so it needs a graphical session. Every case reads the texture back from the GPU and compares it byte for byte with a CPU reference, which is the same reference the staged upload path uses.

The suite runs once per backend that the platform offers:

- OpenGL runs everywhere, as the fallback.
- Metal runs on macOS when a Metal device is present. It also covers the device paths, built from raw `MTLBuffer` allocations: the compute path for `f32` with a lookup table, the blit path for `u8`, stream-ordered updates, imported allocations and their gates, and the geometry service.
- Vulkan runs on Windows with the CPU-staged path. Cases that need CUDA hardware run only when a CUDA device is present.

The suite's Metal run also holds the rows of the renderer core. The `gfx/Metal layer` rows draw through the graphics layer (`src/host/gfx/`) on the renderer's own device and compare each readback byte for byte with a CPU reference. The `gfx/Metal colormap` rows run every boundary range of the colormap by keys through the layer, with the generated `colormap_texture` as a dispatch into an `rgba8unorm` texture and with the vertex stage of `scene_points` drawing one point a pixel wide on each pixel centre, each from a layer buffer and from an imported `MTLBuffer` reported in place with zero bytes copied, and compare every texel and pixel byte for byte with `map_f32_to_rgba8`. The `gfx/Metal scene` rows draw scenes that the scene core (`src/host/scene/`) compiles, through the layer's Metal backend and the generated programs. The one-entity row, square points on pixel centres, is byte-exact. The surface row, a 3-D scene at 512 x 512 with a colormapped height field of flat-shaded faces, a line strip, a disc marker and a box of segments, is compared with a CPU rasterisation of the same compiled draws under three rules instead, because the GPU lights a flat face from the screen-space derivatives of its position and rounds the interpolation in its own way:

- Away from edges, where the pixel and its eight neighbours show the same face, line or point in the reference, a flat colour and the clear colour are exact.
- A pixel next to an edge, the edges between a mesh's faces included, may take the colour that any primitive beside it gives at that pixel.
- A lit face may differ by 2 levels in each channel.

A pick row draws the same scene's pick pass and checks the entity, the primitive and the item that the picker decodes from the ID window. `caliper_scene_tests` (`tests/scene/`), which is not in the `gfx` label, checks the scene core headless and submits its draw lists to the layer's recording backend. [Rendering](rendering.md#the-scene-core) describes both suites.

The suite also has timing rows, which print numbers and prove nothing by themselves: on Vulkan with CUDA, the update cost by size and the cost of a create and release cycle against an update; on Metal, `gfx/metal geometry: timing — draws by count`, the call, completion and GPU microseconds of one `draw_primitives` frame of 1,000 to 4,000,000 imported points and of 1 to 256 draw records, which also checks that no draw counted a copy. Run them with `-s` to see their messages. The layer's and the scene core's timing rows run only when asked, with `CALIPER_GFX_LAYER_BENCH=1` and `CALIPER_GFX_SCENE_BENCH=1` ([Rendering](rendering.md#the-graphics-layer)). The renderer core's migration compares them before and after each step, with the benchmarks in `python/bench/`.

`caliper_shader_tests` (`tests/shaders/`), also under the `gfx` label, checks the shader programs of the graphics layer on Metal without a window: every generated MSL program compiles on the default device, every compute entry and checked pair of entries becomes a pipeline state, and the colormap decided by integer keys matches `map_f32_to_rgba8` byte for byte on its boundary cases, with the compiler's fast math on and off. `web/test/shaders.spec.ts` checks the same programs' WGSL in headless Chromium on a default and a compatibility-mode device. [Write a shader](../guides/write-a-shader.md) describes the programs and their checks.

The colormap by keys ([Rendering](rendering.md#the-colormap-by-keys)) is proved at three levels, on one set of boundary cases (`tests/shaders/colormap_cases.h`: fourteen ranges, and for each its keys and their neighbours, special values and 65,536 random bit patterns). On the CPU, `caliper_tests --test-case="colormap keys*"` compares the C++ decision of `src/host/gfx/colormap_keys.h` with `map_f32_to_rgba8` on those cases, and `caliper_colormap_exhaustive` does so for all 2<sup>32</sup> inputs of each range. On Metal, `caliper_colormap_gpu_exhaustive` runs the generated `colormap_texture` through the layer, compiled as the layer compiles it, on the same 2<sup>32</sup> inputs of each range, in 256 chunks of 4,096 x 4,096 texels that it reads back and compares with the reference in up to 6 CPU threads. Each makes 60 billion comparisons, about 16 and 30 seconds on an M-series Max, the first in under 10 MB and the second in under 500 MB of resident memory, so they are heavy jobs: their ctest entries carry the label `slow` and skip unless `CALIPER_SLOW_TESTS=1`:

```sh
CALIPER_SLOW_TESTS=1 ctest -L slow --output-on-failure
```

`CALIPER_EXHAUSTIVE_RANGES=<n>` and `CALIPER_GPU_EXHAUSTIVE_CHUNKS=<n>` run only the first n ranges or chunks and then fail, so that a small case can show a test's peak memory before a full run.

On the GPU, the `gfx/Metal colormap` rows above and the browser rows of `web/test/shaders.spec.ts` run the same cases through the generated programs. The browser rows, and `web/test/colormap.spec.ts`, which checks the Canvas 2D fallback's TypeScript colormap, compare with vectors generated by the C++: `caliper_colormap_cases <file>` writes each range's block, inputs and reference colours, and `CALIPER_COLORMAP_CASES=<file>` passes them to Playwright, which skips those tests without it. The web workflow builds the tool with one compiler command and runs them.

The browser's WebAssembly views draw through the graphics layer's WebGPU backend, and `web/test/wasm.spec.ts` checks it in headless Chromium on SwiftShader and on the machine's GPU, on a default and a compatibility-mode device: a frame of compute dispatches, a buffer-to-texture copy, an indexed quad and instanced draws with an ID target, depth and blending goes through the layer with its draw program already released, and every target read back through its asynchronous readback equals the CPU's expectation byte for byte. Two more frames check the refusals: the core refuses a draw that samples its own target, and a dispatch that WebGPU refuses is reported through the error scope of its submit. The same spec compares the fixed heatmaps and points, byte for byte, with a set of pictures saved from an earlier build, and the module itself compares every heatmap texture with `map_f32_to_rgba8`. `web/wasm/README.md` describes these checks.

`caliper_export_tests` checks `caliper.export.v1` in the same way: it drives the service as an embedder would and decodes the PNG it wrote. Its sidecar, atomic-write and sugar cases need no GPU.

`caliper_gfx_tests` and `caliper_export_tests` open their windows hidden and never show or focus them, and on macOS the test process shows no Dock icon and no menu bar while it runs. GLFW's own initialisation would give it both: unless the init hint `GLFW_COCOA_MENUBAR` is false, it builds a menu bar and sets the activation policy to `NSApplicationActivationPolicyRegular`. The suites start GLFW through `tests/glfw_background.h` instead, which sets that hint to false and sets the policy to `NSApplicationActivationPolicyProhibited` before and after `glfwInit()` (`src/host/background_process.h`). One case in each suite checks that the policy still holds once the window exists.

The app itself takes the same mechanism with `--hidden` or `CALIPER_HIDDEN=1` ([The hidden window](../references/cli.md#the-hidden-window)): the real app, with its renderer and its frame loop, and a window that is never shown or focused. The ctest entry `caliper_cli_app_hidden` (`tests/test_api.cpp`) starts it with `--api`, reads the token from the API file as a client does, uses every endpoint of [Caliper API](../references/api.md), checks that a capture is a PNG of the hidden window's framebuffer that changes when an applet is launched, completes a handshake on the stream, and closes the app with SIGTERM, leaving no process, file or socket. Without a Metal device it is reported as skipped. The window capture's read-back is also checked on its own in `caliper_export_tests`, on that suite's hidden window.

## Embedding and live streaming

`caliper_embed_tests` drives `libcaliper` as a third-party embedder would: it creates an offscreen core, loads the `hello` applet, pumps frames, reads pixels back and shuts down, and it checks the refusal gates. `caliper_live_tests` and `caliper_live_views_tests` cover live streaming, on macOS only, in one process over an offscreen core. See [Live streaming](live-streaming.md). Both check the copy counters of [Rendering](rendering.md#copy-counters-and-frame-statistics) on every route: an MPS-style heatmap through the shared-memory session, `MTLBuffer` payloads of every kind and their frames count no copy, a CPU heatmap, image and points payload count their uploads, a network snapshot counts its copy into the slot, series and histogram values count their copy at apply, and the browser's host import counts one copy per buffer on a stand-in renderer with no GPU. `caliper_live_views_tests` also checks that `CALIPER_METAL_STATS` records frame times, GPU times and updates by path, and `caliper_tests` checks the tensor bridge's counts with its stub renderer. The live page's controls of wire protocol 3.1 are checked at three levels: `caliper_live_tests` checks the page's state on a scripted source (who may control, the rate of `set`, pending values, actions, selections), the native stream against a fake script, and the run bar drawn on an offscreen canvas and clicked through ImGui's input queue (`src/host/live/live_ui_driver.h`). The ctest entry `caliper_app_live_controls` (`tests/test_net_hub.cpp`) starts the real app hidden with `--url` on a hub in the test process, clicks Pause, types a parameter value, runs an action and shift-drags across a heatmap with `CALIPER_TEST_UI`, and checks what the fake script received and what the page shows of its echo, once with the hub's `--allow-control` and once without.

## The command line

The entries named `caliper_cli_*` (label `cli`) run the `caliper` executable as a user runs it, through `tests/cli_run.cmake`: each checks the exit code and what the run printed. They run with `CALIPER_APPLET_PATH` unset, the user applets folder in the build tree, and `CALIPER_DATA_DIR` on a folder of their own that must still not exist after the run, which proves that the read-only subcommands create nothing in the app-data folder. `caliper_cli_new_builds` (labels `cli` and `sdk`) configures, builds and tests the project that `caliper new` writes against this build's installed SDK.

`caliper_cli_shot_temp` and `caliper_cli_shot_data_dir` (labels `cli` and `shot`) run [`caliper shot`](../references/cli.md#caliper-shot) on the `hello` fixture through `tests/shot_run.cmake`, and `caliper_png_check` checks that the image has the requested size and is not one flat colour. The first runs without `CALIPER_DATA_DIR` and checks that the temporary data folder is gone afterwards and that nothing was created under `HOME`; the second runs with it and checks that the applet's data folder is there and that no `imgui.ini` was written. Without a GPU both are reported as skipped, unless `CALIPER_REQUIRE_GPU=1` makes that a failure.

## Where the tests keep data

No test reads or writes the user's app-data folder. ctest runs every test binary that opens the stores (`caliper_tests`, `caliper_export_tests`, `caliper_embed_tests` and the live suites) with `CALIPER_DATA_DIR` on `tests/app-data/<binary>` in the build folder, and every command-line test with a `CALIPER_DATA_DIR` of its own, as described above. Because `CALIPER_DATA_DIR` moves the whole app-data folder, the stores, the applets' data folders, `imgui.ini`, the live session files and the default user applets folder all stay in the build folder. A test that needs another root, such as the embedding tests of `CaliperCoreDesc.data_dir`, sets its own.

## Applet and adapter tests

`caliper_torch_tests` tests the torch adapter in `adapters/torch.hpp`. Five small executables test the learning and simulation code of single applets: `caliper_sculpt_tests`, `caliper_mesh_tests`, `caliper_em_pic_tests`, `caliper_surface_tests` and `caliper_twin_tests`. They link libtorch and are built only when torch is enabled, which keeps the torch link cost out of `caliper_tests`.

## Package and boundary tests

`caliper_sdk_install` (label `sdk`) proves that an applet can build against the installed SDK alone, and that it cannot include `<caliper/embed.h>`. It installs the SDK component of the build into a work folder, copies `tests/sdk_install_probe` and `examples/hello` there, and builds each copy with only `-DCMAKE_PREFIX_PATH` pointing at the install. It then runs the probe, which draws a headless frame with the installed UI stack and calls the fixture host, checks that hello's library and its stamped manifest exist with the expected `[build]` lines, and fails if either build refers to a file in the source tree. `tests/sdk_install_probe/run.cmake` holds the steps, and `CALIPER_SDK_PROBE_DIR` sets the work folder, which CI points outside the checkout. See [SDK package](../references/sdk-package.md).

`caliper_sdk_install` also checks that the SDK holds none of the embedding headers, and builds an applet source that includes `caliper/embed.h` against every SDK target, which must fail because the header is not found. When the build has the embedding library, the test first installs the `caliper-embed` package into the same prefix, so the check holds with both packages installed together: the embedding headers live under an include root of their own.

`caliper_embed_install` (label `embed`) does the same for the embedding package. It installs the embed component alone into a work folder and checks the installed tree, and on macOS it checks that the library exports exactly the C functions that `libcaliper` defines. It then copies `examples/embed_host` there, builds the copy with only `-DCMAKE_PREFIX_PATH` pointing at the install, checks that the build names nothing in the source tree, and runs the copy in its offscreen mode on a copy of the `hello` fixture. Without a GPU the run is reported as skipped, unless `CALIPER_REQUIRE_GPU=1` makes that a failure. `tests/embed_install/run.cmake` holds the steps, and `CALIPER_EMBED_PROBE_DIR` sets the work folder, which CI points outside the checkout. See [Embedding Caliper](../references/embedding.md).

`caliper_host_names_no_applet` (label `boundary`) runs `scripts/check_host_names_no_applet.py`. The script collects the `[applet]` id and name of every manifest under `applets/` and `examples/`, and every window title that their sources pass to `ImGui::Begin()` as a string literal, and fails if any of them is a whole string literal in the code of `src/` or `include/`. Comments are skipped, so a header may cite an applet id as an example. The test is registered only when CMake finds Python 3.

## Tests that need a GPU

A suite that needs a GPU must not report success when it found none. `tests/gpu_skip.h` provides the convention that the gfx, export, embed, live and torch suites share:

- A case with no GPU counts as skipped, prints why, and returns.
- When any case was skipped for want of a GPU and none failed, the executable exits with code 77. `tests/CMakeLists.txt` registers 77 as `SKIP_RETURN_CODE`, so ctest shows the suite as skipped and not as passed.
- A case that can never apply on the platform, such as CUDA on macOS, is counted separately and does not change the exit code.
- With `CALIPER_REQUIRE_GPU=1` a skip for want of a GPU is a failure. A CI lane that must have a GPU sets it, so the lane cannot go green by accident.
- `CALIPER_TEST_FORCE_NO_GPU=1` makes the test environments report no device, so the skip path itself can be tried on a machine that has one.

## Running the tests

```bash
ctest --test-dir cmake-build-release --output-on-failure
```

A ctest label selects a group: `gfx` (`caliper_gfx_tests` and `caliper_shader_tests`), `export` (`caliper_export_tests`), `torch` (`caliper_torch_tests`), `cli` (the command-line tests), `shot` (the `caliper shot` runs), `sdk` and `embed` (the package tests) and `boundary` (`caliper_host_names_no_applet`). For example, `ctest --test-dir cmake-build-release -L gfx` runs only the pixel-exact graphics suite. CI runs the whole set on macOS and Windows; see [Continuous integration](../guides/ci.md).
