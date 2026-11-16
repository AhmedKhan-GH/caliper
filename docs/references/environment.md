# Folders and environment variables

This page lists where Caliper keeps its files, every environment variable the source reads, and the startup log lines that report which renderer and device path are in use.

## The app-data folder

The host keeps everything it writes under one folder, computed in `src/app_paths.cpp`.

| OS | Folder |
|---|---|
| macOS | `~/Library/Application Support/Caliper/` |
| Windows | `%APPDATA%\Caliper\` (the Roaming AppData known folder). If Windows cannot resolve it, `caliper-data/` under the current directory. |
| Linux | `$XDG_DATA_HOME/caliper/` when `XDG_DATA_HOME` is set and not empty, otherwise `~/.local/share/caliper/`. The folder name is lowercase. |

`CALIPER_DATA_DIR`, when it is set and not empty, moves the whole folder, with everything in the table below, for every Caliper process that reads it: the app, the Caliper CLI subcommands, Caliper Embed, the live viewer and Caliper LIB. It is read once, at the first lookup (`src/app_paths.cpp`). An embedding host moves the folder for one core with `CaliperCoreDesc.data_dir`, which wins over both, except for the live session files under `live/`; see [Embedding](embedding.md#per-core-settings-and-limits).

The host creates the folder on first use. The read-only subcommands `list`, `check`, `info`, `new` and `--help` create nothing in it, whether `CALIPER_DATA_DIR` is set or not (`check` only reads the data-schema stamps under `data/`), `caliper install` writes only the user applets folder, and `caliper shot` without `CALIPER_DATA_DIR` uses a temporary folder in its place; see [Caliper CLI](cli.md).

Inside it:

| Path | What it holds |
|---|---|
| `applets/` | The user applets folder: applet libraries and their `.caliper.toml` manifests, unless `CALIPER_USER_APPLETS_DIR` names another folder. The host searches it after the development folders and before the `applets/` folder next to the executable, and `caliper install` copies into it. See [Caliper CLI](cli.md#the-search-order). |
| `data/<id>/` | One folder per applet, named by the manifest `id`. The host creates it and passes it to the applet as `host.data_dir()`. For an applet that declares `[data].schema`, the host keeps the highest schema that has opened the folder in `.caliper-data.toml` inside it, and holds a lock on `.caliper-data.lock` beside it while it raises that value, so that two processes opening the folder at once cannot record the lower schema. |
| `hub.json` | The hub file, while `caliper serve` runs: the hub's port and session token, for producers on this machine (mode 0600). See [Caliper CLI](cli.md#caliper-serve). |
| `api.json` | The API file, while the app or `caliper shot` serves [Caliper API](api.md): its address and session token (mode 0600), removed when the process exits. `shot` writes it here also when it keeps its stores in a temporary folder. |
| `live/` | The live session folder. A script that calls `caliper.show` writes `<pid>.json` here, and the viewer lists these files; see [Attach to a running script](../guides/attach-to-a-running-script.md). |
| `imgui.ini` | The saved window layout. Delete it to return to the default layouts. |
| `fonts/SternbildDisplay-Regular.otf` | The default location of the display face, used when `CALIPER_DISPLAY_FONT` is not set. |
| `metrics.duckdb`, `data.duckdb`, `artifacts.duckdb` | The stores behind [`metrics.v1`](service-metrics-v1.md), [`data.v1`](service-data-v1.md) and [`artifacts.v1`](service-artifacts-v1.md). |
| `artifacts/` | The content-addressed files of `artifacts.v1`. |

## Host variables

The `caliper` executable reads these. The first six are development hooks for screenshots, recordings, headless checks and clean-exit runs.

| Variable | What it does | Read in |
|---|---|---|
| `CALIPER_AUTOLAUNCH=<id>` | Launches the applet with that manifest id as soon as the host starts, skipping the landing screen. | `src/main.cpp` |
| `CALIPER_AUTOATTACH=<pid>` | Attaches to that process's live session as soon as the landing screen lists it. | `src/main.cpp` |
| `CALIPER_EXIT_AFTER=<seconds>` | Closes the window normally after that many seconds, so the full teardown path runs. | `src/main.cpp` |
| `CALIPER_HOME_AFTER=<seconds>` | Returns to the landing screen once, that many seconds after an applet page or live page opened. | `src/main.cpp` |
| `CALIPER_WINDOW_SIZE=<W>x<H>` | Sets the window size in screen coordinates, for example `1440x900`. A malformed value is ignored and the default of 95% of the work area stands, or 1280 by 800 with `CALIPER_HIDDEN`. | `src/main.cpp` |
| `CALIPER_VIEWS_DEBUG=1` | Shows each view's diagnostics on a line under its header: the size, data type, device and renderer path, the sequence number, the step-to-screen latency, the update count and the drops. Off by default, in the app, `caliper-viewer`, `caliper shot` and Caliper Embed. | `src/host/live/live_views.cpp` |
| `CALIPER_TEST_UI=<steps>` | A testing aid, read only with `--hidden` or `CALIPER_HIDDEN=1`: plays mouse and key input into the live page through ImGui's input queue, so a test clicks the real widgets with nothing on screen. Steps are separated by `;`: `click <widget>` (`pause`, `resume`, `step`, `action:<name>`, `more`), `set <parameter> <value>` (a Cmd-click on macOS, a Ctrl-click elsewhere, on the slider or field, then the value typed and Enter), `toggle <parameter>`, `select <view>` (a shift-drag across the top quarter of the view's image or plot area), `leave` (the pointer leaves the window), `where <view>` (logs the view's data area and its own geometry text), `wait <frames>`, `state <run state>` and `bar [<text>]` (waits until the run bar's state starts with the text and logs what it shows). A picker's menus are widgets too: `click picker:<view>:<dim>`, then `click picker:<view>:<dim>=<entry>`. Each step and its outcome is printed to stderr as an `[ui]` line, and the page's controls print `[live]` lines. See `src/host/live/live_ui_driver.h`. | `src/main.cpp` |
| `CALIPER_HIDDEN=1` | A testing aid, as `--hidden`: runs the real app with its window never shown, no Dock icon and no menu bar, reads and writes no `imgui.ini`, and closes the window normally on SIGINT and SIGTERM. See [The hidden window](cli.md#the-hidden-window). | `src/main.cpp` |
| `CALIPER_CHROMELESS=1` | Hides the page bar, the status bar and the jobs tray on the applet page, so the applet's windows fill the whole window, for recordings. Jobs keep running, and going Home cancels them. Esc still returns to the landing screen, and the landing screen and the live page keep their bars. Any other value leaves the bars. [`caliper shot`](cli.md#caliper-shot) honours it too. | `src/main.cpp`, `src/host/shot.cpp` |
| `CALIPER_USER_APPLETS_DIR` | The user applets folder, in place of `applets/` under the app-data folder, when it is set and not empty. The host searches it after the development folders, and `caliper install` copies into it. Caliper Embed reads it too. | `src/app_paths.cpp` |
| `CALIPER_APPLET_PATH` | Folders to search for applets before the user applets folder and the shipped applets, separated by `:` (`;` on Windows), searched in the order given and after any `--applets` folders. A folder that holds no manifest but whose `applets/` subfolder does, such as an applet's build folder, stands for that subfolder (see [Build folders](cli.md#build-folders)). Caliper Embed reads it too, after its `applets_dir`. | `src/main.cpp`, `src/host/embed/embed_core.cpp` |
| `CALIPER_FONTS_DIR` | A folder that holds the UI fonts. It is searched before `assets/fonts` next to the executable and the source tree. | `src/host/ui/sternbild_fonts.cpp` |
| `CALIPER_DISPLAY_FONT` | The path of the display font file, in place of `fonts/SternbildDisplay-Regular.otf` in the app-data folder. | `src/host/ui/sternbild_fonts.cpp` |
| `CALIPER_HUB_FILE` | The path of the hub file that `caliper serve` writes, in place of `hub.json` in the app-data folder, when it is set and not empty. | `src/main.cpp` |
| `CALIPER_API=1` | Serves [Caliper API](api.md) as `--api` does, for the app, `caliper dev` and `caliper shot`. Any other value leaves it off. On Windows it is ignored, and a line on standard error says so ([Caliper CLI](cli.md#on-windows)). | `src/host/api/app_api.cpp` |
| `CALIPER_SDK_DIR` | The prefix of the installed SDK that [`caliper build`](cli.md#caliper-build) configures against, when it is set and not empty; a folder without the SDK is an error. Without it, `build` looks beside the executable and in `caliper-sdk` in the home folder. | `src/host/cli_build.cpp` |
| `CALIPER_CMAKE` | The cmake that `caliper build` runs, in place of `cmake` on `PATH`, when it is set and not empty. | `src/host/cli_build.cpp` |
| `CALIPER_DATA_DIR` | The app-data folder, in place of the one for the OS. It moves the whole folder: the three stores and `artifacts/`, `imgui.ini`, every applet's data folder under `data/`, the live session files under `live/`, the hub file `hub.json` of `caliper serve`, the API file `api.json`, the default location of the display face under `fonts/` and, unless `CALIPER_USER_APPLETS_DIR` is set, the user applets folder `applets/`. It is read once, at the first lookup. An embedder's `CaliperCoreDesc.data_dir` wins over it for everything but `live/`. The viewer sets it from its `--data-dir` flag, and the test suite sets it to a work folder so that no test opens the user's stores. | `src/app_paths.cpp`, `src/host/live/live_shm.h`, `src/host/live/viewer_main.mm` |
| `CALIPER_DUCKDB_THREADS`, `CALIPER_DUCKDB_MEMORY_LIMIT` | The thread count (an integer from 1 to 1024) and the memory limit (a DuckDB size string such as `4GB`) for all three stores. By default each store takes a share of the machine's RAM and cores. | `src/host/duckdb_sizing.h` |
| `CALIPER_DUCKDB_<STORE>_THREADS`, `CALIPER_DUCKDB_<STORE>_MEMORY_LIMIT` | The same limits for one store, where `<STORE>` is `METRICS`, `DATA` or `ARTIFACTS`. They win over the variables above. An unusable value is ignored and a `[duckdb]` line on stderr says so. | `src/host/duckdb_sizing.h`, `src/host/duckdb_config.h` |

On Linux and macOS the host also reads `HOME` (on Windows `USERPROFILE`) to shorten paths in the Activity log and to find the app-data folder.

## Renderer variables

| Variable | What it does | Read in |
|---|---|---|
| `CALIPER_RENDERER=gl` | Refused: the OpenGL renderer it chose was removed. The app and `caliper shot` exit with 1, and an embedding host that asks for the default renderer gets no core ([No fallback renderer](../explanations/rendering.md#no-fallback-renderer)). Any other value leaves the platform's renderer, Metal on macOS and Vulkan on Windows. | `src/host/core_lifecycle.cpp` |
| `CALIPER_VULKAN_NO_DIRECT` | Set to any value, it turns off the [direct path](../explanations/vulkan-cuda.md#the-direct-image-path), where a CUDA kernel writes the Vulkan image, so updates use the buffer path. | `src/host/renderer/vulkan_renderer.cpp` |
| `CALIPER_VULKAN_STATS` | Set to any value, the Vulkan renderer prints one summary line at shutdown with frame times and the count of direct and buffer updates. | `src/host/renderer/vulkan_renderer.cpp` |
| `CALIPER_METAL_STATS` | Set to any value, the Metal renderer records frame times, the GPU time of each update, view draw and frame, and the updates by path, and prints one summary line at shutdown with its copy count ([Copy counters and frame statistics](../explanations/rendering.md#copy-counters-and-frame-statistics)). `caliper-viewer --stats` sets it. | `src/host/renderer/metal_renderer.mm` |

## Applet and example variables

| Variable | What it does | Read in |
|---|---|---|
| `CALIPER_SINE_AUTOTRAIN` | SineScope starts training on its first frame. | `examples/sine_scope/sine_scope.cpp` |
| `CALIPER_EMBED_AUTOTRAIN` | EmbedScope starts training on its first frame. | `applets/embed_scope/embed_scope.cpp` |
| `CALIPER_GPT_AUTOTRAIN` | GPTScope starts training on its first frame. | `applets/gpt_scope/gpt_scope.cpp` |
| `CALIPER_EXPORT_SELFTEST` | MeshScope and TwinScope request one figure export and one recording after a few warm-up frames, to exercise `export.v1`. | `applets/mesh_scope/mesh_scope.cpp`, `applets/twin_scope/twin_scope.cpp` |
| `FS_FIELD=<n>` | FieldScope starts in field mode `n`. | `applets/field_scope/field_scope.cpp` |
| `CALIPER_HELLO_CRASH` | The hello applet faults in its frame function, to exercise quarantine. | `examples/hello/hello.cpp` |
| `CALIPER_HELLO_INIT_FAIL` | The hello applet's `on_init` returns false. | `examples/hello/hello.cpp` |
| `CALIPER_EMBED_APPLETS` | The applets folder the `embed_host` example passes in `CaliperCoreDesc.applets_dir`. | `examples/embed_host/main.mm`, `examples/embed_host/main_win32.cpp`, `examples/embed_host/embed_host.h` |
| `CALIPER_EMBED_EXIT_AFTER=<seconds>` | The `embed_host` example closes its window after that many seconds. In offscreen mode it draws frames for that many seconds, 1 by default. | `examples/embed_host/main.mm`, `examples/embed_host/main_win32.cpp`, `examples/embed_host/embed_host.h` |
| `CALIPER_EMBED_OFFSCREEN` | Any value other than empty or `0` runs the `embed_host` example in offscreen mode: it opens no window, draws the applet on a 1280 by 800 offscreen canvas at about 60 frames a second, reads the last frame back, prints how many frames it drew and how many pixels the applet drew, and exits with 0 only when the applet drew. | `examples/embed_host/embed_host.h` |

## Test variables

| Variable | What it does | Read in |
|---|---|---|
| `CALIPER_REQUIRE_GPU=1` | A skip for want of a GPU becomes a failure. | `tests/gpu_skip.h` |
| `CALIPER_TEST_FORCE_NO_GPU=1` | The device suites behave as if there were no GPU. | `tests/gpu_skip.h` |
| `CALIPER_TEST_RENDERER_FAIL=1` | The Metal or Vulkan renderer's start (`init` or `canvas_init`) fails before it touches a device, with the reason `CALIPER_TEST_RENDERER_FAIL=1 made its start fail`, so the refusals of the app, `caliper shot` and Caliper Embed can be tested on a machine whose GPU works. | `src/host/renderer/host_renderer.h` |
| `CALIPER_TEST_APPLET_DIR=<folder>` | Runs the headless acceptance case of `caliper_tests` on the applets in that folder: each is negotiated, opened, checked, created and destroyed with the capabilities the `caliper` executable advertises. Without it the case is skipped. | `tests/test_applet_acceptance.cpp` |
| `CALIPER_THROWING_WHERE` | Chooses where the test applet fails: `create`, `initialize`, `frame`, `frame-fault`, `frame-imgui`, `frame-imgui-fault`, `cleanup`, `job`, `job-fault` or `none`. The crash-guard and UI-recovery tests set it. | `tests/fixtures/throwing_applet/throwing_applet.cpp` |
| `CALIPER_GFX_CAPTURE_DIR=<folder>` | `caliper_gfx_tests` writes PNG captures of the graphics layer's rows, the colormap rows, the scene core's rows and the scene view's rows to that folder, and `caliper_gfx_views_tests` each points view case's readback as `points-view-<case>.png` (for example `layer_quad.png`, `colormap-keys-texture-ramp.png`, `first-scene-surface.png` and `scene-view-row.png`). Without it nothing is written. | `tests/gfx/gfx_layer.cpp`, `tests/gfx/gfx_colormap.cpp`, `tests/gfx/gfx_scene.cpp`, `tests/gfx/gfx_scene_view.cpp`, `tests/gfx/gfx_points_view.mm` |
| `CALIPER_GFX_RECORD=<file>` | `caliper_gfx_views_tests` records each case's line of the graphics suite's reference mode (its readback's hash, path label, copies and header) into that file instead of comparing; a migration step records its references with it at its first commit. | `tests/gfx/gfx_reference.h` |
| `CALIPER_GFX_REFERENCES=<file>` | `caliper_gfx_views_tests` compares with the references in that file instead of the committed `tests/gfx/references/points_view.txt`. | `tests/gfx/gfx_reference.h` |
| `CALIPER_GFX_REFERENCE_IMAGES=<folder>` | `caliper_gfx_views_tests` compares each case's readback pixel by pixel with its image in that folder, written by an earlier build with `CALIPER_GFX_CAPTURE_DIR`, and lists the differing pixels. Without it that row returns at once. | `tests/gfx/gfx_points_view.mm` |
| `CALIPER_GFX_POINTS_CASES=<text>` | `caliper_gfx_views_tests` runs only the points view's cases whose names contain the text. | `tests/gfx/gfx_points_view.mm` |
| `CALIPER_GFX_POINTS_BENCH=1` | `caliper_gfx_views_tests` runs the points view's timing row: the CPU time to compile and submit a redraw and the GPU time of the draw at 100,000, 1,000,000 and 4,000,000 points, with the load average. Without it the row returns at once. | `tests/gfx/gfx_points_view.mm` |
| `CALIPER_GFX_LAYER_BENCH=1` | `caliper_gfx_tests` runs the graphics layer's timing row: the CPU time to build and submit 1,000 draws a frame on the recording backend and on Metal, printed with the load average. Without it the row returns at once. | `tests/gfx/gfx_layer.cpp` |
| `CALIPER_GFX_SCENE_BENCH=1` | `caliper_gfx_tests` runs the scene core's timing row: the compile time and the layer's submit time of the surface scene and of a scene of 1,000 entities. Without it the row returns at once. | `tests/gfx/gfx_scene.cpp` |
| `CALIPER_GFX_SCENE_VIEW_BENCH=1` | `caliper_gfx_tests` runs the scene view's timing row: the view's compile, submit and GPU times for the demo scene at 512 x 512 and the time to apply one mesh update. Without it the row returns at once. | `tests/gfx/gfx_scene_view.cpp` |
| `CALIPER_GFX_CMAP_BENCH=1` | `caliper_gfx_tests` runs the colormap's timing row: the GPU time of the generated `colormap_texture` against the Metal renderer's float pass on 1,024 by 1,024 and 2,048 by 2,048 heatmaps, with the load average. `CALIPER_GFX_CMAP_BENCH_SAMPLES=<n>` sets the samples of each pass, 40 by default. Without it the row returns at once. | `tests/gfx/gfx_colormap.cpp` |
| `CALIPER_CMAP_BENCH=1` | `caliper_tests` runs the timing row of the colormap keys: the time to derive one range's keys and to find a block in the cache. Without it the row returns at once. | `tests/test_colormap_keys.cpp` |
| `CALIPER_SLOW_TESTS=1` | Runs the tests under the ctest label `slow`, `caliper_colormap_exhaustive` and `caliper_colormap_gpu_exhaustive`, which compare the colormap by keys with `map_f32_to_rgba8` for all 2<sup>32</sup> inputs. Without it they report themselves skipped. | `tests/test_colormap_exhaustive.cpp`, `tests/test_colormap_gpu_exhaustive.mm` |
| `CALIPER_TEST_THREADS=<n>` | The CPU threads of the two exhaustive colormap tests, 6 by default and at most 6. | `tests/test_colormap_exhaustive.cpp`, `tests/test_colormap_gpu_exhaustive.mm` |
| `CALIPER_EXHAUSTIVE_RANGES=<n>`, `CALIPER_GPU_EXHAUSTIVE_CHUNKS=<n>` | Limit `caliper_colormap_exhaustive` to its first n ranges and `caliper_colormap_gpu_exhaustive` to its first n of 256 chunks, the small cases whose peak memory is measured before a full run. A limited run fails, so it never passes as a full one. | `tests/test_colormap_exhaustive.cpp`, `tests/test_colormap_gpu_exhaustive.mm` |
| `CALIPER_METAL_STATS` | The renderer variable above. `caliper_live_views_tests` sets it to check the frame statistics, and `python/bench/bench_show.py` sets it for its window-mode rows on the offscreen canvas. | `src/host/renderer/metal_renderer.mm` |
| `CALIPER_SCENE_CAPTURE_DIR=<folder>` | `python/tests/test_scene_app.py` saves the two frames the app draws there, as `scene-view-app.png` and `scene-view-update.png`. Without it nothing is written. | `python/tests/test_scene_app.py` |
| `CALIPER_BENCH_KEEP_RAW` | Set to any value, `python/bench/bench_live.py` writes the viewer's log and latency samples of each row to `python/bench/results/S-raw-<tag>-*.txt`. Without it they go to the private data folder of the run. | `python/bench/bench_live.py` |
| `CARGO_TARGET_DIR` | The folder in which `tools/shaders/generate.py` builds the shader translator's driver, in place of `tools/shaders/naga-driver/target/`. | `tools/shaders/generate.py` |
| `CALIPER_VIEWS_REFERENCE=<folder>` | The web package's `web/test/wasm.spec.ts` compares each fixed view's picture with the file of the same name in that folder, saved from an earlier build, and fails on any differing byte of a view's data area. Without it the pictures are only saved. | `web/test/wasm.spec.ts` |
| `CALIPER_VIEWS_COMPAT=1` | `web/test/wasm.spec.ts` asks every mount of the WebAssembly views for a compatibility-mode WebGPU device. | `web/test/wasm.spec.ts` |
| `CALIPER_COLORMAP_CASES=<file>` | `web/test/shaders.spec.ts` and `web/test/colormap.spec.ts` take the colormap's cases from the file that `caliper_colormap_cases <file>` wrote from the C++: each range's block, inputs and reference colours. Without it their colormap tests are skipped. | `web/test/shaders.spec.ts`, `web/test/colormap.spec.ts` |

The metrics, data, loader, embed and live tests also set `CALIPER_DUCKDB_*`, `CALIPER_HELLO_*` and `CALIPER_DATA_DIR` themselves to exercise the code that reads them. ctest runs every test binary that opens the stores, and every command-line test, with `CALIPER_DATA_DIR` on a folder in the build tree, and runs the command-line tests with `CALIPER_APPLET_PATH` unset. See [Development basics](../tutorials/development-basics.md) for how the GPU skip variables behave.

## Python variables

| Variable | What it does | Read in |
|---|---|---|
| `CALIPER_SESSION=0` | Stops the first view call from opening a live session automatically. Calls then do nothing until `caliper.window()` or `caliper.offscreen()` is opened. | `python/caliper/__init__.py` |
| `CALIPER_VIEWER` | The path of the `caliper-viewer` executable, in place of the one inside the package. | `python/caliper/live.py` |
| `CALIPER_DATA_DIR` | The data folder an in-process window or offscreen canvas uses when no `data_dir` argument is given. Without either, a private temporary folder is used. | `python/caliper/_inproc.py` |
| `CALIPER_PY_LOG` | Any value other than `0` echoes the host's diagnostic lines from the in-process core to stderr. | `python/src/_native.mm` |
| `CALIPER_PY_LIVE_APPLY=0` | Sends heatmap and image snapshots through the embed view route instead of `caliper_core_live_apply`. | `python/src/_native.mm` |
| `CALIPER_TEST_NO_WINDOW=1` | Skips the Python tests that open a real window. | `python/tests/test_window_real.py` |
| `CALIPER_WEB_DIR` | The folder of the browser files that `serve()` and the widgets hand out, in place of the package's `caliper/_web/`. | `python/caliper/_assets.py` |
| `CALIPER_REQUIRE_WEB_VIEWS` | `1` makes a missing set of the WebAssembly views an error where it is otherwise a warning or a skip: in a wheel build and in `python/tests/test_assets.py`. A wheel build treats `CI=true` as `1` unless this is `0`. | `python/check_web_views.cmake`, `python/tests/test_assets.py` |
| `GITHUB_TOKEN`, `GH_TOKEN` | A token that `scripts/fetch_wasm.py` sends to the GitHub API, and only there, to download CI's build of the WebAssembly views. Without one it can use only a release's build. | `scripts/fetch_wasm.py` |

See [Caliper LIB](python-api.md) for the calls themselves.

## Startup log lines

The `caliper` executable writes these to stderr and, for the renderer, to the Activity pane.

| Line | Meaning |
|---|---|
| `[renderer] metal` or `[renderer] vulkan` | The backend that initialised. The Activity pane shows the same name as a `renderer` entry. |
| `caliper: the Vulkan renderer could not start (<reason>), and there is no fallback renderer until the WebGPU fallback through Dawn lands` | The platform's backend failed to start, for example when there is no Vulkan driver, and the app exits with 1 (Metal on macOS). [No fallback renderer](../explanations/rendering.md#no-fallback-renderer) gives the other refusals. |
| `[autolaunch] <id>` | `CALIPER_AUTOLAUNCH` matched an applet and launched it. |

On Windows the Vulkan renderer reports how each tensor update travelled. Each distinct message prints once, as `[vulkan] device path: <message>`:

| Message | Meaning |
|---|---|
| `CUDA interop OK — zero-copy (kernel reads the tensor, writes the image)` | The CUDA kernel wrote the update straight into the Vulkan image. |
| `import path OK — zero-copy (imported allocation in place)` | An imported allocation was used in place, with no copy. |
| `geometry path OK — points drawn from imported allocation in place` | `geometry.v1` drew points from an imported allocation. |
| `tensor not CUDA-resident`, `external-memory interop unavailable`, `dtype not f32/u8`, `CUDA context init failed` and similar | The update took the CPU-staged or buffer path, and the message gives the reason. The fallback is not an error; see [Platform rules](../explanations/architecture.md#platform-rules). |
| `direct image: exportable image failed, using the buffer path` | The direct path could not be set up, and updates use the buffer path. |

Other host lines carry a tag: `[jobs]` for failed jobs and jobs still running at shutdown, `[ui]` for a UI state unwound after an applet fault, `[metrics]`, `[artifacts]` and `[data]` when a store fails to open, `[duckdb]` for an ignored limit override and `[live]` for session events. A program that embeds the core through Caliper Embed receives its own `renderer: <name>` line through `log_fn`, and applet log lines tagged `[applet] `.

The device that `caliper.device.v1` reports is found at startup: Metal on Apple, and the CUDA driver API elsewhere. See [`caliper.device.v1`](service-device-v1.md).
