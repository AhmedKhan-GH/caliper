# Testing

Caliper's tests use doctest and run through ctest. The sources are in `tests/`, and `tests/CMakeLists.txt` registers them.

## Fast Unit Suite

`caliper_tests` is one executable that needs no window and no GPU. It holds the layers below, one source file per layer.

| Layer | What it proves |
|---|---|
| Manifest | The parser accepts a valid manifest, rejects a missing or malformed field, validates the applet id and ignores unknown keys. |
| Negotiation | The pre-load checks run in order: epoch, `min_host`, then required services, each with its refusal message. |
| Loader | The loader scans a folder, runs the `hello` applet through its full lifecycle, tears down an old instance before a relaunch, checks the descriptor against the manifest, shows a missing binary as a failed card, and refuses a path-escaping id. |
| Crash guard | A fault or an exception in an applet call is contained and reported, in the frame call and in a background job. The exception cases use `tests/fixtures/throwing_applet`, a hand-written descriptor that throws across the ABI, which the sugar would otherwise catch. |
| UI recovery | The host unwinds its UI stack after an applet fault leaves windows, plots and style pushes open. |
| Frame watchdog | Three consecutive slow frames latch the flag, a good frame resets the streak, and the flag stays until a reset. |
| Services | Each service store and wrapper has its own tests: jobs, device, metrics, feed, artifacts, data and the tensor bridge. |
| Sugar | `CALIPER_APPLET`, the typed service wrappers, exception catching at the ABI edge and a run at a `dpi_scale` of 2. |
| ABI | `test_abi.cpp` asserts that the ABI structs are standard layout with `struct_size` first, and checks the epoch and every published table. |
| C compile check | `abi_c_check.c` includes every public header, newest first, in a C translation unit. A header that lacks its own includes cannot hide behind another one. |

The same executable covers applet-side logic that needs neither torch nor a window, such as the pulse ring. The build makes the `hello` applet and `throwing_applet` first, because the loader tests load them from the build folder.

## Fixture Host

`sdk/testing/caliper/fixture_host.h` is a headless fake `CaliperHost`. It serves `caliper.log.v1` and returns `NULL` for every other service. A test can inject a table for any other service with `provide`. The fixture lets an applet's logic and the sugar run without the UI. Only one fixture exists per process. The CMake target is `caliper::sdk_testing`. See [C++ sugar](../applets/cpp-sugar.md) for its use in an applet's own tests.

## Pixel-Exact Graphics Suite

`caliper_gfx_tests` (`tests/gfx/gfx_main.cpp`) checks the path from a tensor to a texture and the scene core. It opens a hidden window, so it needs a graphical session. Every case reads the texture back from the GPU and compares it byte for byte with a CPU reference.

The suite runs on the backend that the platform offers:

- Metal runs on macOS when a Metal device is present. It also covers the device paths, built from raw `MTLBuffer` allocations: the compute path for `f32` with a lookup table, the blit path for `u8`, stream-ordered updates, and imported allocations.
- Vulkan runs on Windows with the CPU-staged path. Cases that need CUDA hardware run only when a CUDA device is present.

### Renderer Core Rows

The Metal rows check the graphics layer, colormap compute programs and library scene pieces.
The layer rows compare offscreen color, ID and depth targets with CPU expectations.
The colormap rows run `colormap_texture` from layer buffers and imported Metal buffers.
Each output texel must equal `map_f32_to_rgba8`.

The scene view rows draw protocol fixtures at 256 by 256 pixels.
The fixtures cover arrows, boxes, edges, lit and textured meshes, cameras, images and camera-space annotations.
Their expected images are saved PNGs in `tests/match_references/`.
The tests retain the pixel tolerances for edges, interpolation, transparency and lighting.
They do not execute the removed scene compiler.

The standalone suites compare `points@1`, `lines@1`, `arrows@1`, `path@1`, `boxes@1`, `mesh@1` and `surface@1` with saved images.
Their metadata records original camera matrices and pick identities where the fixture needs them.
A pick check reads the piece's ID target at the saved pixel and checks the expected item.
`tests/match_references/README.md` describes the reference source and update procedure.
The current renderer cannot write replacement references through these tests.

`web/test/scenes.spec.ts` checks that every scene fixture draws pieces, produces nonclear pixels and has no notes.
`web/test/draw_parity.spec.ts` compares the same scenes between the app and browser, including color, IDs and picks.
`caliper_scene_tests` checks the CPU scene model, camera, labels and frame-pick decode without a GPU.
[Scene Rendering](../applets/scene-rendering.md) describes the shared view code.

### Timing Rows

The suite also has a timing row on Vulkan with CUDA for the update cost by size and the cost of a create and release cycle against an update, which prints numbers and proves nothing by itself.

Run them with `-s` to see their messages. The layer timing rows run only when requested, with `CALIPER_GFX_LAYER_BENCH=1` ([Graphics Layer](../applets/graphics-layer.md)). The migration of the renderer core compares them before and after each step, with the benchmarks in `python/bench/`.

### Shader Tests

`caliper_shader_tests` (`tests/shaders/`), also under the `gfx` label, checks the shader programs of the graphics layer on Metal without a window:

- Every generated MSL program compiles on the default device.
- Every compute entry and checked pair of entries becomes a pipeline state.
- The colormap decided by integer keys matches `map_f32_to_rgba8` byte for byte on its boundary cases, with the compiler's fast math on and off.

`web/test/shaders.spec.ts` checks the WGSL of the same programs in headless Chromium on a default and a compatibility-mode device. [Shader Authoring](../applets/shader-authoring.md) describes the programs and their checks.

### Colormap Proof

The colormap by keys ([Colormap by Keys](../applets/graphics-layer.md#colormap-by-keys)) is proved at three levels, on one set of boundary cases. `tests/shaders/colormap_cases.h` holds fourteen ranges, and for each its keys and their neighbours, special values and 65,536 random bit patterns.

- On the CPU, `caliper_tests --test-case="colormap keys*"` compares the C++ decision of `src/host/gfx/colormap_keys.h` with `map_f32_to_rgba8` on those cases. `caliper_colormap_exhaustive` does so for all 2<sup>32</sup> inputs of each range.
- On Metal, `caliper_colormap_gpu_exhaustive` runs the generated `colormap_texture` through the layer, compiled as the layer compiles it, on the same 2<sup>32</sup> inputs of each range. It works in 256 chunks of 4,096 x 4,096 texels that it reads back and compares with the reference in up to 6 CPU threads.
- On the GPU, the `gfx/Metal colormap` rows above and the browser rows of `web/test/shaders.spec.ts` run the same cases through the generated programs.

Each exhaustive test makes 60 billion comparisons. The first takes about 16 seconds and the second about 30 seconds on an M-series Max, in under 10 MB and under 500 MB of resident memory. They are heavy jobs, so their ctest entries carry the label `slow` and skip unless `CALIPER_SLOW_TESTS=1`:

```sh
CALIPER_SLOW_TESTS=1 ctest -L slow --output-on-failure
```

`CALIPER_EXHAUSTIVE_RANGES=<n>` and `CALIPER_GPU_EXHAUSTIVE_CHUNKS=<n>` run only the first n ranges or chunks and then fail, so that a small case can show the peak memory of a test before a full run.

The browser rows, and `web/test/colormap.spec.ts`, which checks the TypeScript colormap of the Canvas 2D fallback, compare with vectors that the C++ generates. `caliper_colormap_cases <file>` writes the block, inputs and reference colors of each range, and `CALIPER_COLORMAP_CASES=<file>` passes them to Playwright, which skips those tests without it. The web workflow builds the tool with one compiler command and runs them.

### Views Suite

`caliper_gfx_views_tests` (`tests/gfx/gfx_points_view.mm`) draws the points view through an offscreen core.
It compares `points@1` with saved scene images in `tests/match_references/points_view/`.
The cases cover 2-D lattices, point sizes, subpixel positions, colormaps, categorical colors, selections and 3-D depth cues.
CPU, shared-storage and private-storage payloads use the same library piece.

A pixel with a uniform 3 by 3 reference neighborhood must keep that color.
At an edge, it can differ by one level from a color in that neighborhood.
Lattice center pixels also retain the CPU reference colors.
The interaction rows compare hover, click, rectangle and lasso results between CPU and device payloads.
They require a real pick and selection events, then compare the selected pictures.

A separate point-sprite test checks the edge bounds of the old Metal sprite rasterizer.
The optional `CALIPER_GFX_POINTS_BENCH=1` row measures redraw cost by point count.

### WebAssembly Views

The WebAssembly views of the browser draw through the WebGPU backend of the graphics layer. `web/test/wasm.spec.ts` checks it in headless Chromium on SwiftShader and on the machine's GPU, on a default and a compatibility-mode device.

- A frame of compute dispatches, a buffer-to-texture copy, an indexed quad and instanced draws with an ID target, depth and blending goes through the layer with its draw program already released. Every target read back through its asynchronous readback equals the CPU's expectation byte for byte.
- Two more frames check the refusals. The core refuses a draw that samples its own target, and a dispatch that WebGPU refuses is reported through the error scope of its submit.
- The same spec compares the fixed heatmaps and points, byte for byte, with a set of pictures saved from an earlier build, and the module itself compares every heatmap texture with `map_f32_to_rgba8`.

`web/wasm/README.md` describes these checks.

The Playwright tests of the web package run in three projects of headless Chromium: `webgpu` on the machine's GPU, `swiftshader` on the SwiftShader software adapter and `canvas2d` with WebGPU off. The WebAssembly views draw in the two WebGPU projects. The own views of the `<caliper-view>` element, written in TypeScript, draw with Canvas 2D in all three, because they have no WebGPU path ([Renderer Selection](../web/browser-elements.md#renderer-selection)).

- `web/test/views.spec.ts` checks every kind of the element's own views against known data, the color of every cell, pixel and point included.
- `web/test/edges.spec.ts` checks the color of every colormap edge case in both renderers.
- `web/test/points3d.spec.ts` checks the 3-D box, default camera, depth cues and rotate toggle in WebAssembly and Canvas 2D. Four WebGPU cases also compare the initial and replaced payload draws through `points@1`.
- `web/test/renderer.spec.ts` and `web/test/fallback.spec.ts` check which renderer draws a view, with a stub of the loader of the WebAssembly views and without the loader.

### Hidden Windows

`caliper_gfx_tests` opens its window hidden and never shows or focuses it. On macOS the test process also shows no Dock icon and no menu bar. The suite starts GLFW through `tests/glfw_background.h`, which sets `GLFW_COCOA_MENUBAR` to false and sets the activation policy to `NSApplicationActivationPolicyProhibited` around `glfwInit()` (`src/host/background_process.h`). One case in the suite checks that the policy still holds.

The app uses the same mechanism with `--hidden` or `CALIPER_HIDDEN=1` ([The hidden window](../reference/cli.md#hidden-window)). The ctest entry `caliper_cli_app_hidden` (`tests/test_api.cpp`) starts the real app this way with `--api`. It uses every endpoint of [Caliper API](../agents/http-api.md) and checks that a capture is a PNG of the hidden framebuffer. It then closes the app with SIGTERM and checks that no process, file or socket remains. Without a Metal device the entry is skipped.

## Embedding and Live Streaming

`caliper_embed_tests` drives `libcaliper` as a third-party embedder would. It creates an offscreen core, loads the `hello` applet, pumps frames, reads pixels back, shuts down and checks the refusal gates.

`caliper_live_tests` and `caliper_live_views_tests` cover live streaming on macOS only, in one process over an offscreen core. See [Live streaming](../python/live-streaming.md). Both check the copy counters of [Rendering](../applets/rendering.md#copy-counters-and-frame-statistics) on every route:

- An MPS-style heatmap through the shared-memory session, and `MTLBuffer` payloads of every kind, count no copy in their frames.
- A CPU heatmap, image and points payload count their uploads.
- A network snapshot counts its copy into the slot, and series and histogram values count their copy at apply.
- The host import of the browser counts one copy per buffer on a stand-in renderer with no GPU.

`caliper_live_views_tests` also checks that `CALIPER_METAL_STATS` records frame times, GPU times and updates by path, and `caliper_tests` checks the counts of the tensor bridge with its stub renderer. The live tests check the controls of wire protocol 3.1 at three levels:

- The state of the live page on a scripted source: who may control, the rate of `set`, pending values, actions and selections.
- The native stream against a fake script.
- The run bar, drawn on an offscreen canvas and clicked through the ImGui input queue (`src/host/live/live_ui_driver.h`).

The ctest entry `caliper_app_live_controls` (`tests/test_net_hub.cpp`) starts the real app hidden with `--url` on a hub in the test process. It checks the run bar while the fake script runs past the step of its last change of state. With `CALIPER_TEST_UI` it clicks Pause, types a parameter value, runs an action and shift-drags across a heatmap. It checks the run bar and what the script received, once with the hub's `--allow-control` and once without.

## Command Line

The entries named `caliper_cli_*` (label `cli`) run the `caliper` executable as a user runs it, through `tests/cli_run.cmake`. Each entry checks the exit code and the output. The entries run with `CALIPER_APPLET_PATH` unset and with `CALIPER_DATA_DIR` on a folder that must not exist after the run. This proves that the read-only subcommands create nothing in the app-data folder. `caliper_cli_new_builds` (labels `cli` and `sdk`) configures, builds and tests the project that `caliper new` writes, against the SDK that this build installs.

`caliper_cli_shot_temp` and `caliper_cli_shot_data_dir` (labels `cli` and `shot`) run [`caliper shot`](../reference/shot-and-serve-commands.md#caliper-shot) on the `hello` fixture through `tests/shot_run.cmake`. `caliper_png_check` checks that the image has the requested size and is not one flat color. The first entry runs without `CALIPER_DATA_DIR` and checks that the temporary data folder is gone and that nothing exists under `HOME`. The second entry runs with it and checks that the applet's data folder exists and that no `imgui.ini` was written. Without a GPU both entries are skipped, unless `CALIPER_REQUIRE_GPU=1` makes the skip a failure.

Caliper has no fallback renderer ([No Fallback Renderer](../applets/rendering.md#no-fallback-renderer)). Four entries check the refusals where the renderer cannot start, on any machine and with no GPU:

- `caliper_cli_app_no_renderer` and `caliper_cli_shot_no_renderer` run the hidden app and `caliper shot` with the test variable `CALIPER_TEST_RENDERER_FAIL=1`, which makes the start of the Metal or Vulkan renderer fail.
- `caliper_cli_app_renderer_gl` and `caliper_cli_shot_renderer_gl` run them with `CALIPER_RENDERER=gl`.

Each checks the exact message and exit code 1, and that the run left at most the empty data folder. The `embed/refusal` cases of `caliper_embed_tests` check the same for Caliper Embed. `caliper_core_attach_canvas` returns 0 with the message in `caliper_core_last_error`, and `caliper_core_create` refuses `CALIPER_RENDERER=gl` with the message in the log of the embedder.

## Test Data Location

No test reads or writes your own app-data folder. ctest runs every entry with `CALIPER_DATA_DIR` on `test-app-data/` in the build folder and with `CALIPER_TEST_RUN=1`. The file that `caliper_offline_tests()` writes (`src/builder/cmake/CaliperOfflineTests.cmake`) sets both, so an entry cannot forget them. Some entries replace that folder with their own:

- Every test binary that opens the stores uses `tests/app-data/<binary>`. These binaries are `caliper_tests`, `caliper_embed_tests`, the live suites and `caliper_catalog_providers_tests`.
- Every command-line test has its own `CALIPER_DATA_DIR`.
- Every test that starts the hidden app, `caliper shot` or a producer gives that process its own folder.

`CALIPER_DATA_DIR` moves the whole app-data folder, so the stores, the applets' data folders, `imgui.ini`, the live session files, the stream files, the API files and the default user applets folder all stay in the build folder. The tests of Caliper LIB (`python/tests/conftest.py`) set `CALIPER_TEST_RUN=1` and give every test a temporary `CALIPER_DATA_DIR`.

The app-data guard backs this up ([Folders and environment variables](../reference/folders-and-environment.md#app-data-folder)). A test or hidden run that would use your own app-data folder stops at once with exit code 1, before it creates anything there. The guard applies to `--hidden`, `CALIPER_HIDDEN=1`, any `CALIPER_TEST_*` variable and any test binary, because each binary links `tests/test_run_marker.cpp`. A test binary that you run by hand, from a shell or an IDE, therefore needs `CALIPER_DATA_DIR`, as ctest gives it.

`caliper_test_data_guard` (labels `cli` and `shot`, `tests/data_guard_run.cmake`) runs the typical launches with `HOME` on a stand-in folder: the hidden app, `caliper_embed_tests`, `caliper shot` and `caliper.serve()`. It checks that each refusal happens and that nothing appears in the stand-in's app-data folder. A test that needs another root, such as the embedding tests of `CaliperCoreDesc.data_dir`, sets its own.

## Applet and Adapter Tests

`caliper_torch_tests` tests the torch adapter in `adapters/torch.hpp`. It links libtorch and exists only when torch is enabled, which keeps the torch link cost out of `caliper_tests`.

## Package and Boundary Tests

`caliper_sdk_install` (label `sdk`) proves that an applet can build against the installed SDK alone and cannot include `<caliper/embed.h>`. It installs the SDK component into a work folder and builds `tests/sdk_install_probe` and `examples/hello` there with only `-DCMAKE_PREFIX_PATH` set to the install. It runs the probe, checks the library and stamped manifest of `hello`, and fails if either build refers to a file in the source tree. It also builds an applet that includes `caliper/embed.h` against every SDK target, and each build must fail. `tests/sdk_install_probe/run.cmake` holds the steps, and `CALIPER_SDK_PROBE_DIR` sets the work folder. See [SDK package](../applets/sdk-package.md).

`caliper_embed_install` (label `embed`) does the same for the embedding package. It installs the embed component alone and checks the installed tree. On macOS it checks that the library exports exactly the C functions that `libcaliper` defines. It builds `examples/embed_host` against the install and runs it offscreen on a copy of the `hello` fixture. Without a GPU the run is skipped, unless `CALIPER_REQUIRE_GPU=1` makes the skip a failure. `tests/embed_install/run.cmake` holds the steps, and `CALIPER_EMBED_PROBE_DIR` sets the work folder. See [Embedding Caliper](../web/embedding-api.md).

`caliper_theme_colors` (label `boundary`) runs `scripts/check_theme_colors.py`. The script fails if the theme sources `src/host/ui/sternbild_theme.h`, `sternbild_theme.cpp` and `sternbild_fonts.h` hold a color written by hand outside the lines marked `colours without a token`. The script finds `IM_COL32(...)`, `ImVec4` and `ImColor` constructions with parentheses or braces, declarations such as `ImVec4 c(1, 0, 0, 1)`, three or four numbers in braces when one of them is a float literal, and `0xRRGGBB` constants. The theme may hold only one marked region, and each marker must stand on a line of its own. The theme reads every other color from the generated header of the Sternbild design tokens. `caliper_theme_colors_script` runs the tests of the script.

`caliper_host_names_no_applet` (label `boundary`) runs `scripts/check_host_names_no_applet.py`. The script fails if the `[applet]` id or name of a manifest under `applets/` or `examples/`, or a window title that their sources pass to `ImGui::Begin()`, is a whole string literal in `src/` or `include/`. The script skips comments. CMake registers the test only when it finds Python 3.

## Builder Suites

With `CALIPER_WITH_BUILDER` on, the default except on Windows, the build also compiles the builder under `src/builder/`, and ctest runs its tests beside the host's tests. Each module has a doctest executable registered as `builder_<module>`, such as `builder_spec`, `builder_loaders` and `builder_session`. Other entries run `caliper-mcp`, `caliper-acp-harness` and `caliper-builder-host` as child processes against a stub agent. `builder_agent_py` runs the pytest suite of `caliper-agent`. The command `ctest --test-dir cmake-build-release -R '^builder_'` runs only these suites.

Two environment variables control the entries that need outside tools:

- `CALIPER_BUILDER_TEST_PYTHON` names an interpreter with numpy, pyarrow and the test dependencies of `caliper-agent`. Without it, the cases that run Python are skipped and say so.
- `CALIPER_BUILDER_TEST_APP` names a `caliper` executable. Without it, `builder_mcp_app` is skipped. With it, the entry runs `caliper-mcp` against that app.

The builder session in the app has its own entries. The entries that start the hidden app are skipped without a Metal device, and all of them run with `CALIPER_AGENT_OFFLINE=1`. The hidden app takes scripted input through `CALIPER_TEST_SPEC_UI`.

| Entry | What it checks |
|---|---|
| `caliper_app_session`, `caliper_app_session_hidden` | The session in `src/host/builder/app_session.h`. The first uses a fake frame loop. The second runs a command script ([The workspace](../reference/cli.md#workspace)) that imports a CSV file, builds and saves a spec applet and renders a panel, then checks the workspace and its journal. |
| `caliper_app_route` | The route of an outside agent against the real app, started with `--hidden --api --api-port 0 --workspace`. It imports a Parquet file and a safetensors file, builds, saves and opens the three-panel dashboard, and captures a non-blank frame through Caliper API. Then it runs `caliper-mcp` and `caliper shot --launch` against the app. |
| `caliper_app_live_entropy` and four `caliper_app_live_*` entries | The case "plot attention entropy per layer while it trains" (`tests/test_app_live.cpp`, macOS), and the landing screen and live page of the hidden app. `caliper_attention_producer` is a test tool with its own hub and no torch. The entries cover stream attach and detach, a second stream, Home from the live page, and dropped files. |
| `caliper_app_save_new`, `caliper_app_save_open`, `caliper_app_save_close` | The "Save changes to <name>?" question, through the `save` and `close` steps of the input driver. `app.launch` never asks. |
| `caliper_app_launch_mcp`, `caliper_app_launch_agent`, `caliper_app_launch_reply` | A launch from the origin `mcp` or `agent`, through `POST /v0/applets/{id}/launch` and `app.launch`. Over unsaved changes it waits on an approval card and answers `PENDING_APPROVAL` until the person decides. |
| `caliper_widget_surface`, `caliper_widget_surface_gpu`, `caliper_widget_surface_agent`, `caliper_widget_app` | The inline surface of the widget agent (`src/host/builder/widget_surface.h`), headless, offscreen, with the real `caliper-agent` on a scripted model, and in the hidden app. The agent entry is skipped without `CALIPER_BUILDER_TEST_PYTHON`. |
| `caliper_approval_cards`, `caliper_approval_cards_app` | The approval cards (`src/host/builder/approval_cards.h`): Approve, Deny (`command.denied`), timeout, conflict and destructive commands, then the same through the real `caliper-mcp` and `CALIPER_TEST_APPROVALS`. |
| `caliper_builder_pane`, `caliper_builder_pane_gpu`, `caliper_builder_app` | The builder pane (`src/host/builder/builder_pane.h`), headless, offscreen with `caliper-builder-shot`, and as a whole build in the hidden app. |

## Overhead Targets

A measured overhead share or throughput is a target, not a gate. The test prints the value with its target in one line, such as `overhead share 0.0187 (target 0.02): tower_arcs`, and adds a line that starts with `warning:` on a miss. It fails only above twice the target overhead, which is a share above twice its target or a throughput below 90 % for the target of 95 % (`python/tests/overhead_targets.py`).

## Instrument Acceptance

The ctest entry `caliper_instrument_acceptance` (`tests/instruments/driver.py`) is the instrument acceptance test. It runs `tests/instruments/train.py`, a script whose only Caliper line is `caliper.watch(model, optimizer)`. It starts the hidden app on the stream of the script and `caliper-mcp` as a child, and it sends `tests/instruments/transcript.jsonl` through the MCP tools. The agent creates a spec, a scene panel bound to `live:model` and a slider bound to `param:lr`, and it changes `lr` after an approval card. The test then waits for the line of the script that shows the new rate at its next step. Last, a call of `events_wait` for picks must be held, and it must return the pick with the panel id, the entity and the item index.

The test waits on output lines, events and published state, never for a fixed time. The entry needs a Python with torch, which `CALIPER_ACCEPTANCE_PYTHON` names (the default is the builder test Python). It also needs the native sender `caliper._netio`, which `scripts/build_netio.py` builds into `python/caliper/`. The driver puts that folder on `PYTHONPATH`. The entry exits with code 77, which ctest reports as skipped, when the app, Metal, torch or the sender is missing.

The hidden app takes the input of the test through `CALIPER_TEST_UI` and the decisions on cards through `CALIPER_TEST_APPROVALS`. The app reads both only with `--hidden`. The spec page adds three steps to the grammar of `CALIPER_TEST_UI`:

| Step | Meaning |
|---|---|
| `panel <id>` | Waits until the spec page draws the scene of the panel |
| `held [<kind>]` | Waits until a call of `events.wait` waits on the event ring, for that kind when it is given |
| `pick_panel <id> <fx> <fy>` | Clicks the panel at the fractions of its width and height, such as `0.5 0.5` |

The instrument check `python3 examples/python/instrument_check.py` runs the same transcript without a window, as one command with defaults. It finds `caliper` and `caliper-mcp` in `cmake-build-release`, then `cmake-build-notorch`, and it uses the Python that runs it when that Python has torch, else the `.venv` of the repository. It saves one capture of the app, taken after the learning rate changed, into a new folder under the build folder, and the arguments `--app`, `--mcp`, `--python` and a folder override these defaults. Then it prints one line for each acceptance point: the one Caliper line of the script, the scene panel, the slider, the learning rate that the script reports at its next step, and the pick event. It exits with a code other than 0 on any failure, and it stops after five minutes. In CI, the `macos` job builds a Python with torch and the native sender, passes it as `CALIPER_ACCEPTANCE_PYTHON`, and runs `caliper_instrument_acceptance` as its own step. That step fails when the test skips because torch or the sender is missing, and it accepts a skip because the runner has no Metal device. The same job builds the WGSL compiler library from source with Rust, builds `caliper._producer`, and runs `caliper_gpu_programs_acceptance` as its own step. That step fails for any skip except a missing Metal device.

## Hosted Runs Acceptance

The ctest entry `caliper_hosted_runs_acceptance` (`tests/hosted_runs/driver.py`) is the hosted runs acceptance test. It starts the hidden app with no script and `caliper-mcp` as a child. The driver copies `tests/hosted_runs/train.py` and `tests/hosted_runs/infer.py` into a throwaway workspace and sends `tests/hosted_runs/transcript.jsonl` through the MCP tools. The model runner of the app starts, forks and stops the scripts as managed runs.

The test waits on events and published state, never for a fixed time. `tests/CMakeLists.txt` defines it only when the platform is not Windows, `CALIPER_WITH_BUILDER` is on and the Python of the instrument test is set. It has the labels `hosted_runs` and `acceptance` and a timeout of 300 seconds. It runs with the Python that `CALIPER_ACCEPTANCE_PYTHON` names, and it exits with code 77, which ctest reports as skipped, when the app, Metal or torch is missing.

The check `python3 examples/python/hosted_runs_check.py` runs the same flow without a window, as one command with defaults. It writes the capture of the two loss curves into a folder, prints the states of the runs, and exits with 1 on a failure and with 2 when the check cannot run.

## GPU Programs Acceptance

The ctest entry `caliper_gpu_programs_acceptance` (`tests/gpu_programs/driver.py`) is the GPU programs acceptance test. It starts the hidden app with no script and `caliper-mcp` as a child, and it runs `tests/gpu_programs/train.py`, a script whose only Caliper line is `caliper.watch(model, optimizer)`, as a managed run. The WGSL sources of the programs are the files `grid.wgsl`, `heat.wgsl`, `writer.wgsl` and `slow.wgsl` beside it. The piece `grid_material` that the agent writes is the manifest `material_piece.json` with the files `material_derive.wgsl` and `material_draw.wgsl`. The file `tests/gpu_programs/transcript.jsonl` sends the steps through the MCP tools.

The agent adds a geometry program over the weight `0.weight` and a pixels program over `2.weight`. Then it writes the piece `grid_material`, whose derive half places a point for each of the first 512 values of `0.weight` where the geometry program places it, and whose draw half colors each point from orange to yellow by its value. Its first version has a typo in its draw half: `piece.update` must refuse it with `program.compile_failed` and a message that names `draw.wgsl`. The corrected version goes into the workspace, and `program.create` places it in the scene panel with `0.weight` bound. Both halves must compile, the derive half must run in the script and the draw half must draw a resource version. Both must stay inside their budgets: the instance must still run, with a cadence `every` of at least 1 and the memory of each half within its limit. The driver takes a capture of the empty scene, one with the geometry, one with the image panel and one with the piece, and it compares them by counts of pixels, not by eye. The geometry must change at least 300 pixels of the empty scene. The piece must give at least 50 yellow pixels in the lower left quarter of the frame, and at least ten times the count of the capture before it. A program that declares the model tensor writable must fail with `program.writes_input` and must not appear in `program.status`. A program over its `time_ms` must stop with a `program` event of change `stopped` and the code `program.over_time`, while the step of the run keeps growing. `program.status` must report a cadence `every` of at least 1, a last pass time and the memory in use. Then the test waits up to 90 seconds until the measured `overhead_share` of the geometry program reaches its target of 0.04, the budget of the cadence. A share above 0.04 after the wait gives a warning, and a share above 0.08 fails. The test does not assert a value of N.

A toggle bound to `intervene:0` is clicked through `CALIPER_TEST_SPEC_UI`, which waits for files that the driver writes. The mean of five losses while the toggle is on must be above three times the mean before it, and the mean after its release must be below half of the mean while it was on. The driver takes the SHA-256 of the app, of `caliper-mcp` and of each file of the Python package before and after, and the lists must be equal. Last, no process that the driver started may be alive, by its pid. The test waits on events and published state, never for a fixed time.

`tests/CMakeLists.txt` defines the entry only when the platform is not Windows, `CALIPER_WITH_BUILDER` is on and the Python of the instrument test is set. It has the labels `gpu_programs` and `acceptance` and a timeout of 300 seconds. It exits with code 77, which ctest reports as skipped, with the reason printed, when the app, Metal, torch, numpy, the native network sender `caliper._netio`, the producer module `caliper._producer` or the WGSL compiler library is missing. The producer module needs `CALIPER_BUILD_PYTHON=ON` and its file in `python/caliper/`, and the compiler library needs `CALIPER_WGSL_FROM_SOURCE=ON`, which needs Rust.

The check `python3 examples/python/gpu_program_check.py` runs the same flow without a window, as one command with defaults. It uses the app and `caliper-mcp` of the build folder with the newest app, and it writes the captures into a folder of that build folder. It prints the count of files with an unchanged SHA-256, the status of the geometry program, the losses around the toggle, the pixel counts and the capture paths. It exits with 1 on a failure and with 2 when the check cannot run.

## Programmable Graphics Tests

Pieces have tests at several levels. Each refusal that [Pieces](../agents/pieces.md#refusals) lists has a case at the level of the host that gives it.

| Level | Entry | What it checks |
|---|---|---|
| Builder | `builder_programs`, `builder_spec`, `builder_session`, `builder_commands`, `builder_mcp` | The names, files, imports, hash and layouts of a piece, the manifest checks with one case for each code, the composers of both halves, the codec and validator of the spec, the piece commands with their pins and undo steps, the command table, and the resources of `caliper://library/` (`src/builder/programs/tests/`, `src/builder/session/tests/test_piece_commands.cpp`) |
| Draw runtime | `caliper_gfx_draw_tests` | The draw half on the recording backend of the graphics layer: install, the order of the passes, a skipped pass, the checks before anything is made, a pass that does not compile, the layout of a version, the reset rules of the draw buffers, the stop after 30 slow frames, the late frames judged by their GPU time or three in a row, the heavy frames in a row, the frame whose time the backend no longer keeps, the stop after 5 s, the copy that a pick keeps of its frame's version, the draw memory, the decode of PROGRAM with the hash, and the draw host with its camera conflict (`src/host/draw/tests/test_draw_runtime.cpp`) |
| Metal rows | `caliper_gfx_tests` | The cases named `gfx/Metal pieces` (`tests/gfx/gfx_pieces.cpp`). The field draws as the CPU reference of its fragment, the arcs draw instanced and additive with an ID for each arc and the bloom of their glow, a slow pass stops its program while the other goes on, a compute pass of the draw half feeds an instanced draw with state, reset and depth, and the scene view draws its programs after its scene |
| Derive half | `caliper_producer_tests` | The derive half in the producer runtime on Metal: passes in order with the contents kept, clears and counters, read-only tensors, bounds, sizes, the watchdog and the timeout without a wait, a rebind and a texture |
| Library piece | `caliper_points_match` | The piece `points@1` against saved Metal images and original pick identities, including the live points view depth cues (`tests/points_match/`) |
| Host | `caliper_spec_runtime` | The `programs` cases: the item map gives the fields and the index of an item and refuses an item past the count, the text table gives a label at each valid anchor, and a piece in a scene panel is picked at its pixel, writes a select state and places its labels |
| Python | `python/tests/test_programs_pieces.py` | The facade of the script: an instance installs, the derive half runs and publishes RESOURCE, a draw half relays to a viewer, the backlog of RESOURCE frames stays bounded, a viewer gets the first chunk of a pass with the pass or skips the pass, a version above 8 MiB goes in parts that come back whole, each part only while the viewer's queue has room, a piece that publishes 64 MiB at every pass keeps its overhead share, and a pass above 128 MiB per buffer, 256 MiB per texture or 512 MiB in total stops the program with `program.resource_too_large` |

The wire frames have golden files and codec tests in all three languages, as [Wire Golden Frames](#wire-golden-frames) describes. The folder `tests/fixtures/wire_v3/v3_2` holds the frames of PROGRAM for an instance of a piece and of RESOURCE, and the invalid frames that each codec refuses.

### Piece Path

The ctest entry `caliper_piece_path_acceptance` (`tests/piece_path/driver.py`) follows one piece through the whole path. It starts the hidden app with no script and `caliper-mcp` as a child, and the model runner starts the training script of the GPU programs test, `tests/gpu_programs/train.py`, in a throwaway workspace. The driver writes the workspace piece `row_norms` with `piece_update`. Its derive half reads the weight `0.weight` and writes the norm of each row, and its draw half draws one bar for each row with an ID. The driver then creates an instance of it in a scene panel with `program_create`.

The driver waits for the event `compiled` of both halves and checks each step by state, not by eye:

- The status of the instance has the kind `piece`, and the hash that the script runs equals the hash of the pin.
- `program_read` gives the elements of the newest version, and each element has its row index and the norm of that row. A later version and the step history hold at least two versions, and a read at an older step gives the version that was newest then.
- The `draw` object of the status is `running` with a `resource_step`, and the panel gains at least 2,000 blue pixels against a capture of the empty panel.
- `selection_set` with `program` and `item` gives the fields of row 5 and a `pick` event. The linked selection then marks the bar, which adds magenta pixels.

The driver writes three captures, `empty`, `drawn` and `picked`, into `piece-path-captures` of the build folder. The entry has the labels `pieces` and `acceptance`, a timeout of 300 seconds, and the conditions of the GPU programs test. It exits with code 77 only when the machine has no Metal device or the Python modules `caliper._netio` and `caliper._producer` are not built, and its last line names the reason. Anything else that is missing fails it. The Build workflow runs it in a step of its own, which accepts only the skip for a missing Metal device.

### Programmable Graphics Acceptance

The ctest entry `caliper_programmable_graphics_acceptance` (`tests/programmable_graphics/driver.py`) is the acceptance test of programmable graphics. It starts the hidden app with no script and `caliper-mcp` as a child, and the model runner starts `examples/python/arc_tower_train.py` as a managed run. The driver sends the tool calls of an agent that has only the MCP tools, builds the Attention Arc Tower from the pieces of `demos/arc_tower`, and writes the calls it sent as `transcript.jsonl` into its work folder. The step `check` in this list names what the driver asserts.

1. It hashes the app, `caliper-mcp` and the Python package.
2. It opens a spec file of a foreign colleague with an inline piece. The program of that piece has the state `review`.
3. It starts the run and pauses it while the agent builds. `piece_list` of the library holds `points@1` and the four lib modules, and `piece_show` of `points` has a draw file.
4. `piece_check` accepts both pieces and `piece_update` writes them as version 1.
5. The agent creates the scene panels `tower` and `pattern`, the programs, the loss curves, the chart `head_scores` of the eight score series and a toggle, and sets the layout with `layout_set`. The spec holds no entity. After the run resumes, both halves of both programs give the event `compiled`.
6. `program_read` of the arcs gives elements with tokens below 32 and `from_token` below `to_token`. A capture of `tower` differs from the empty capture in at least 2,000 pixels.
7. At the bump, the newest `heads` version holds a score above 0.4, and the first kept version holds no score of 0.2 or more. The loss falls, and the in-context loss falls below 1.5. In `panel_data` of the chart, the scores of layer 1 stay below 0.2 up to the first kept step, and one line then passes 0.5. The step at which the strongest line of layer 1 passes 0.3 is at most 100 steps from the step at which the in-context loss falls below 2.
8. `selection_set` on the brightest arc of the best head gives the event `pick` with `layer`, `head`, `from_token` and `to_token`. The state `head` equals the picked head, and the capture of `pattern` changes.
9. After the in-context loss settles below 0.2, the driver pauses the run and ablates the head of layer 1 with the highest score in the newest `heads` version with `intervene_set`. The run then advances three single steps with `run_pause`, and each step publishes its in-context loss. The highest of them is at least 3 times the settled value. The other heads make up for the ablated head within tens of steps, and the views of a running script follow the shared cadence, thus the single steps make the check independent of the cadence and of the time at which the ablation reaches the script.
10. `run_fork` from the first kept step with the same head ablated gives a child whose score of that head is below 0.2 at 500 steps past the parent's step at the bump, while another head of layer 1 passes 0.4. The tower of the child differs from the tower of the parent.
11. A copy of `attention_arcs` whose bloom loops is written as a piece and placed in a panel of its own. The event `stopped` names the program, `program.draw_over_time` and a draw pass. The program `tower_arcs` still draws, and the step of the run still rises.
12. The `overhead_share` of each derive half is printed with its target of 0.02. A share above 0.02 gives a warning, and a share above 0.04 fails.
13. After both runs stop, no process that the test started is alive, and the SHA-256 of the files has not changed.

The driver waits on events and published state, never for a fixed time. The entry has the labels `programmable_graphics` and `acceptance` and a timeout of 900 seconds. It exits with code 77, which ctest reports as skipped, with the reason printed, when the app, Metal, torch, the native sender or the producer module is missing. The captures go to `tests/programmable_graphics-captures` in the build folder. The check `python3 examples/python/arc_tower_check.py` runs the same steps as one command with defaults. It prints each step as it passes and then the measured results, and it runs `caliper_points_match_tests`, the match of `points@1` with saved Metal references, when the build folder holds it.

### Draw Parity

The folder `tests/draw_parity/` compares the app and the browser module on the same frame. A fixture holds two instances, a field and a set of arcs, as PROGRAM frames and one RESOURCE frame for each, thus no training runs. `caliper_draw_parity_tests` checks what needs no GPU: the frames, the decode with the hash of the pin, the packed params, the refusals of the module, and an install of each program on the recording backend. `caliper_draw_parity_app` draws the fixture with the draw runtime of the app on Metal into a folder of the build. It exits with 77 and prints the reason when no Metal device exists.

The browser side is `web/test/draw_parity.spec.ts`. It draws the same frames with the browser module in headless Chromium and compares the color and the ID targets with the files of the app. An interior pixel, which has one ID in its 3 by 3 neighborhood in both images, differs by at most 2 levels in each channel and has the same ID. The bloom of the glow target may differ by 3 levels. The pixels that differ at an edge and take neither side stay below 1 percent of the image. A second case sends the draw halves over a connection as PROGRAM and RESOURCE and checks that the scene view of the module draws them and refuses a bad one. The test runs in the macOS lane of CI with the browser on SwiftShader. It waits on a written frame and never on a fixed time.

A second pixel test compares the whole Arc Tower dashboard, the app against the browser, on one recorded run in `tests/draw_parity/dashboard`. The folder holds the spec and the `panel.data` answer of every leaf, recorded once from a short seeded run, thus nothing trains in the test. The app's side is the tool `caliper_dashboard_parity_app` (`tests/draw_parity/dashboard_parity_app.cpp`), which ctest runs as `caliper_draw_parity_dashboard_app`. It draws the recording with the spec runtime of the app offscreen on Metal, and it exits with 77 without a Metal device. It also writes the rectangle of each scene panel and the labels that the panel drew over its programs, with their boxes. The browser side is `web/test/draw_parity_dashboard.spec.ts`, and it takes its picture at a frame at which every program draws the recorded step, in the same task as the frame of the module.

The test applies the rules to each panel, control and tab strip on the rectangle of the app, and to the rest of the area. An interior pixel, which has one color in its 3 by 3 neighborhood in both pictures, differs by at most 2 levels in each channel. An edge pixel counts only where it takes neither side. In a program panel, a control, a tab strip and the rest, such pixels stay at or below 1 percent of the region, the rule of the program pictures. In a plot panel they stay at or below 4 percent, because the plot lines come from two rasterizers, ImPlot in the app and Canvas 2D in the page, and most pixels of each line are edge pixels that differ. The test prints the numbers of each region. The text is compared apart. The box of every string that the page draws is masked in both pictures, and so is the box of every label over a scene, from either side. Every leaf is on the rectangle of the app, the tab labels are the same, each panel title is at the place of the app, and the legend names and the toggle text match. Each scene draws on the rectangle of the app, with the same label strings, each in a box within 1 pixel of the box of the app.

`web/test/spec-dashboard.spec.ts` checks the layout, the line names and colors and the control row of the Arc Tower against a mock of the app. `web/test/spec-programs.spec.ts` clicks an arc of the tower in the page. It checks the fields of the pick, the state that the `select` option of the panel writes in the page, the params that the grid gets, the highlight of the picked arc and that nothing goes to the app. It also checks that the programs give their labels, and that a `view_camera` of the panel starts the camera of the scene.

The test `caliper_dashboard_viewer_acceptance` (`tests/programmable_graphics/viewer_driver.py`) runs the browser's dashboard viewer end to end. The hidden app runs a live run of the Arc Tower's script, with a workspace piece in one scene panel and an inline piece in another. The real spec page opens by the viewer link of the app in headless Chromium, with WebGPU and with SwiftShader. In the page, in the same task as each frame of the module, the test counts the pixels of each program panel, and both programs must be running. Then the app opens another spec, and the page must say that the app does not have the dashboard open. The test exits with 77 without Metal, torch, Node or the WebAssembly views.

## Instrument Building Acceptance

The ctest entry `caliper_instrument_building_acceptance` (`tests/instrument_building/driver.py`) is the acceptance test of instrument building. It starts the hidden app with no script and `caliper-mcp` as a child, and it runs `tests/instrument_building/train.py`, a small transformer whose only Caliper line is `caliper.watch(model, optimizer)`, as a managed run started by `run_start`. The driver builds an instrument on that run with MCP tools only. It approves the cards through `CALIPER_TEST_APPROVALS` and moves the learning rate slider and the toggle through `CALIPER_TEST_SPEC_UI`, which waits for files that the driver writes.

The driver checks each part by state, events and panel data, not by eye:

- A scene panel holds a terrain from the `surface` operator over `live:blocks.0.attn.proj.act`. Its row count in `panel_data` rises while the run trains, and the loss is a line entity in the same scene.
- The entry of `live:blocks.0.attn.proj.act` in `catalog_list` has the warning of the shared cadence, which says how many steps apart the view updates and why.
- A pick on the terrain through `selection_set` gives a `pick` event with fields and writes the state `token`. The version and the values of the heatmap that reads the state through a reference change.
- A narrower range window leaves terrain rows whose `x` values stay inside the window.
- The transport holds the cursor at an earlier kept step of the activation source, which the terrain's rows show. The heatmap and the terrain state that step, and the series panels hold no later step. One step back, the cursor can stand at a step that only the loss keeps, and the heatmap and the terrain then show their newest value at or before it. Then the cursor is live again.
- The segmented group `measure` changes the values of the heatmap, and the group `palette` changes at least 500 pixels of a render of the terrain at a held step.
- The bound text is a readout control and a text panel. A text panel gives the bound value and not the formatted string, thus the driver checks the three inputs: the step of the cursor, the token and the value at that step.
- The spec holds a threshold line and a marker in a series panel, a band and a marker, and a layout that uses shares and a minimum. The data of a panel gives no rectangles of placed labels, so the driver checks that the annotations exist and that the threshold changes a render. It does not check that labels avoid each other.
- A person's drag changes `lr` in the run, and the toggle ablates `blocks.0.attn.proj` and releases it.
- A source from `live:head.act` with a `surface` step runs as the program `operator.head_surface.0` in the script, draws as an entity and is removed again.

The driver waits on events and published state, never for a fixed time. A terrain row comes with a step that the app keeps, at a pass of the shared cadence, thus the kept steps stand some steps apart. The driver takes that gap from the `x` values of the terrain where a check compares steps. It waits for 40 terrain rows before the checks that need a terrain. It prints the time from the first tool call to the first render of a terrain that has rows. It takes the SHA-256 of the app, of `caliper-mcp` and of each file of the Python package before and after, and the lists must be equal. Last, no process that the driver started may be alive, by its pid.

`tests/CMakeLists.txt` defines the entry only when the platform is not Windows, `CALIPER_WITH_BUILDER` is on and the Python of the instrument test is set. It has the labels `instrument_building` and `acceptance` and a timeout of 600 seconds. It exits with code 77, which ctest reports as skipped, when the app, Metal, torch, numpy, the native network sender `caliper._netio` or the producer module `caliper._producer` is missing. The macOS job of the workflow runs it in its own step.

The check `python3 examples/python/instrument_building_check.py` runs the same flow without a window, as one command with defaults. It uses the app and `caliper-mcp` of the build folder with the newest app, and it writes the captures into a folder of that build folder. It prints the time to the first frame, the row counts, the pick, the held step with its bound text, the changed pixels, the annotations, the count of files with an unchanged SHA-256 and the capture paths. It exits with 1 on a failure and with 2 when the check cannot run.

## GPU Tests

A suite that needs a GPU must not report success when it finds none. `tests/gpu_skip.h` provides the convention that the gfx, embed, live and torch suites share:

- A case with no GPU counts as skipped, prints why, and returns.
- When any case is skipped for want of a GPU and none fails, the executable exits with code 77. `tests/CMakeLists.txt` registers 77 as `SKIP_RETURN_CODE`, so ctest shows the suite as skipped and not as passed.
- A case that can never apply on the platform, such as CUDA on macOS, is counted separately and does not change the exit code.
- With `CALIPER_REQUIRE_GPU=1` a skip for want of a GPU is a failure. A CI lane that must have a GPU sets it, so the lane cannot go green by accident.
- `CALIPER_TEST_FORCE_NO_GPU=1` makes the test environments report no device, so you can try the skip path on a machine that has one.

## Test Runs

```bash
ctest --test-dir cmake-build-release --output-on-failure
```

A ctest label selects a group:

- `gfx`: `caliper_gfx_tests`, `caliper_gfx_views_tests` and `caliper_shader_tests`.
- `torch`: `caliper_torch_tests`.
- `cli`: the command-line tests.
- `shot`: the `caliper shot` runs.
- `sdk` and `embed`: the package tests.
- `boundary`: `caliper_host_names_no_applet`.

For example, `ctest --test-dir cmake-build-release -L gfx` runs only the pixel-exact graphics suite. CI runs the whole set on macOS and Windows. See [Continuous integration](ci.md).

## Chat Captures

`caliper-chat-shot` is a development tool, and the install does not include it. It draws a dashboard with the spec runtime beside the side chat on the offscreen canvas. It writes a PNG at each state of a scripted conversation with a stub agent.

```bash
caliper-chat-shot --workspace <dir> --agent <stub agent> --out-dir <dir> [--size WxH] [--prefix NAME]
                  [--frames N] [--review on|off] [--mcp PATH] [--inputs DIR]
```

The tool builds the dashboard in an empty workspace from the files of `--inputs`, or from small files that it writes itself: `run42_metrics.parquet` and `attn.safetensors`. It starts Caliper API on 127.0.0.1 and runs the agent as a custom command with the name "Stub agent".

The states are `1-empty`, `2-turn`, `3-permission` (`3-approval` with `--review on`), `4-done`, `5-exited` and `6-no-agent`. Each state goes to `<prefix>-<state>-<W>x<H>.png`. The tool prints one JSON object with the files, the conversation items, the panels of the open spec and the result of a check for the agent's `panel.create` in the journal.

## Windows Coverage

The socket tests of the WebSocket layer, the hub and Caliper API do not run in the Windows CI lane. The network source that `--url` attaches through exists only on macOS. The code of `serve`, `build`, `--api` and `--url` compiles on Windows, but no test runs it there.

## Wire Golden Frames

The script `generate.py` in `tests/fixtures/wire_v3` writes the golden frames of that folder with `python/caliper/_wire.py`. The test `python/tests/test_wire.py` makes sure that the writer gives the same bytes again.
