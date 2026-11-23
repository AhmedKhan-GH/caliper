# Folders and Environment

Caliper keeps its files in one app-data folder. It reads the environment variables below, and it writes startup log lines that name the renderer and device path.

## App-Data Folder

The host keeps everything it writes under one folder.

| OS | Folder |
|---|---|
| macOS | `~/Library/Application Support/Caliper/` |
| Windows | `%APPDATA%\Caliper\` (the Roaming AppData known folder). If Windows cannot resolve it, `caliper-data/` under the current directory. |
| Linux | `$XDG_DATA_HOME/caliper/` when `XDG_DATA_HOME` is set and not empty, otherwise `~/.local/share/caliper/`. The folder name is lowercase. |

`CALIPER_DATA_DIR` moves the whole folder when it is set and not empty. The app, the Caliper CLI subcommands, Caliper Embed, the live viewer and Caliper LIB all read it, once, at the first lookup. An embedding host moves the folder for one core with `CaliperCoreDesc.data_dir`. That setting wins over `CALIPER_DATA_DIR`, except for the files under `live/`. See [Embedding](../web/embedding-api.md#per-core-settings-and-limits).

A test or hidden run never uses the folder of the OS. These runs are the app with `--hidden` or `CALIPER_HIDDEN=1`, every process with a `CALIPER_TEST_*` variable that is set and not empty, and the test binaries. If such a run would use the folder of the OS, it prints `caliper: <the run> has no data folder of its own and would use your app-data folder <folder>; set CALIPER_DATA_DIR to a folder of its own` on standard error. It then exits with code 1 before it creates anything. Caliper LIB raises `caliper._net.AppDataRefused` with the same text from `caliper.serve()` and `caliper.connect()`. Thus a test never drives the app you are running. A folder given explicitly is used as given, even the folder of the OS.

A visible app, the core of an embedding host and `caliper shot` are not test runs by themselves. Without `CALIPER_DATA_DIR`, `shot` keeps its stores and API file in a temporary folder. It only reads the user applets folder, so you can take a shot of your own applets.

The host creates the folder on first use. The read-only subcommands `list`, `check`, `info`, `new` and `--help` create nothing in it. `caliper install` writes only the user applets folder. See [Caliper CLI](cli.md).

The folder holds these items:

| Path | What it holds |
|---|---|
| `applets/` | The user applets folder: applet libraries and their `.caliper.toml` manifests, unless `CALIPER_USER_APPLETS_DIR` names another folder. The host searches it after the development folders and before the `applets/` folder next to the executable, and `caliper install` copies into it. See [Caliper CLI](cli.md#search-order). |
| `data/<id>/` | One folder per applet, named by the manifest `id`. The host passes it to the applet as `host.data_dir()`. For an applet that declares `[data].schema`, `.caliper-data.toml` records the highest schema that has opened the folder. |
| `hub.json` | The hub file, while `caliper serve` runs: the hub's port and session token, for producers on this machine (mode 0600). See [Caliper CLI](shot-and-serve-commands.md#caliper-serve). |
| `api.json` | The API file, while the app or `caliper shot` serves [Caliper API](../agents/http-api.md): its address, the token of outside agents, whose commands are reviewed, and the link of its spec page (mode 0600). The process removes it on exit. Your own session token is not in it. Without `CALIPER_DATA_DIR`, `shot` writes the file into its temporary folder. |
| `api-agent-<pid>-<n>.json` | The API file of one agent the app runs, the side chat's agent or the widget agent's sidecar, while that agent's session lasts. It has the same keys as `api.json`, with a token of the agent's own whose commands are reviewed (mode 0600). See [Caliper API](../agents/http-api.md#app-startup). |
| `live/` | The live session folder. A script that calls `caliper.show` writes `<pid>.json` here, and the viewer lists these files. See [Attach to a running script](../python/script-attachment.md). |
| `streams/` | One `<pid>.json` (mode 0600) for each script whose `caliper.serve()` listens: its stream link with the token. The server removes it when it closes. The app's `F4 Attach to process` lists these streams. See [Attach to a stream by its link](../python/stream-links.md). |
| `imgui.ini` | The saved window layout. Delete it to return to the default layouts. |
| `fonts/SternbildDisplay-Regular.otf` | The default location of the display face, used when `CALIPER_DISPLAY_FONT` is not set. |
| `metrics.duckdb`, `data.duckdb`, `artifacts.duckdb` | The stores behind [`metrics.v1`](../applets/services/caliper-metrics-v1.md), [`data.v1`](../applets/services/caliper-data-v1.md) and [`artifacts.v1`](../applets/services/caliper-artifacts-v1.md). |
| `artifacts/` | The content-addressed files of `artifacts.v1`. |
| `workspace/` | The builder's default workspace: its loaders, specs, saved spec applets, the command journal `journal/commands.jsonl` and `workspace.duckdb`. The app creates it when the builder first needs it. `--workspace` or `CALIPER_WORKSPACE` names another folder in its place. See [The workspace](cli.md#workspace). |

## Host Variables

The `caliper` executable reads these. The first six are development hooks.

| Variable | What it does |
|---|---|
| `CALIPER_AUTOLAUNCH=<id>` | Launches the applet with that manifest id as soon as the host starts, and skips the landing screen. |
| `CALIPER_AUTOATTACH=<pid>` | Attaches to the live session of that process, or to its `caliper.serve()` stream, as soon as the landing screen lists it. It acts once. |
| `CALIPER_EXIT_AFTER=<seconds>` | Closes the window normally after that many seconds, so the full teardown path runs. |
| `CALIPER_HOME_AFTER=<seconds>` | Returns to the landing screen once for each kind of page, that many seconds after the first applet page, live page or spec page opens. From the live page the session stays attached, as with `esc Home`. |
| `CALIPER_WINDOW_SIZE=<W>x<H>` | Sets the window size in screen coordinates, for example `1440x900`. The host ignores a malformed value. The default is 95% of the work area, or 1280 by 800 with `CALIPER_HIDDEN`. |
| `CALIPER_VIEWS_DEBUG=1` | Shows diagnostics on a line under the header of each view: size, data type, device and renderer path, sequence number, latency, update count and drops. It is off by default. |
| `CALIPER_TEST_UI=<steps>` | A testing aid, read only with `--hidden` or `CALIPER_HIDDEN=1`. It plays mouse and key input into the live page, so a test clicks the real widgets with nothing on screen. Steps are separated by `;`. Two steps read the page back: `where <view>` logs the view's data area and its own geometry text, and `bar [<text>]` waits until the run bar's state starts with the text and logs what it shows. The spec page adds `panel <id>`, which waits until the page draws the scene of that panel, `held [<kind>]`, which waits until a call of `events.wait` waits on the event ring, and `pick_panel <id> <fx> <fy>`, which clicks the panel at fractions of its size. |
| `CALIPER_TEST_SPEC_UI=<steps>` | A testing aid, read only with `--hidden`. It plays mouse and key input into the landing screen, the spec page and the live page, and it reads named frames back. Steps are separated by `;`. |
| `CALIPER_TEST_APPROVALS=<file>` | A testing aid, read only with `--hidden`. It holds the decisions on the approval cards, one line each, such as `approve panel.create mcp` or `deny panel.create mcp`. The app reads lines as they are appended. |
| `CALIPER_HIDDEN=1` | A testing aid, the same as `--hidden`. It runs the real app with its window never shown, no Dock icon and no menu bar. It reads and writes no `imgui.ini`, and it closes the window normally on SIGINT and SIGTERM. See [The hidden window](cli.md#hidden-window). |
| `CALIPER_TEST_RUN=1` | Marks the process, and every process it starts, as a test run. Without `CALIPER_DATA_DIR`, a test run stops at the first lookup of the folder ([The app-data folder](#app-data-folder)). Any other `CALIPER_TEST_*` variable that is set and not empty has the same effect. |
| `CALIPER_WORKSPACE` | The builder's workspace folder, in place of `workspace/` under the app-data folder, when it is set and not empty and `--workspace` is not given. The app, `caliper dev` and `caliper shot` open it after they draw their first frame. See [The workspace](cli.md#workspace). |
| `CALIPER_COMMAND_SCRIPT=<file>` | A testing and demo aid: a file of JSON lines, one builder command each. The hidden app or `caliper shot` runs them once its workspace is open. A visible app ignores the file and says so, because a script may approve as the person does. See [The workspace](cli.md#workspace). |
| `CALIPER_AGENT_PYTHON=<interpreter>` | The Python interpreter, with the dependencies of `caliper-agent`, that starts the widget agent's sidecar for ⌘K on a spec page. Without it, the inline prompt offers only Ask in chat. The sidecar also needs Caliper API (`--api`) and a key in the system key store. |
| `CALIPER_CHROMELESS=1` | Hides the page bar, the status bar and the jobs tray on the applet page, so the applet fills the window, for recordings. Jobs keep running. Any other value leaves the bars. [`caliper shot`](shot-and-serve-commands.md#caliper-shot) honors it too. |
| `CALIPER_USER_APPLETS_DIR` | The user applets folder, in place of `applets/` under the app-data folder, when it is set and not empty. The host searches it after the development folders, and `caliper install` copies into it. Caliper Embed reads it too. |
| `CALIPER_APPLET_PATH` | Folders to search for applets before the user applets folder and the shipped applets, separated by `:` (`;` on Windows), after any `--applets` folders. A folder with no manifest whose `applets/` subfolder has one, such as a build folder, stands for that subfolder (see [Build folders](cli.md#build-folders)). Caliper Embed reads it too. |
| `CALIPER_FONTS_DIR` | A folder that holds the UI fonts. The host searches it before `assets/fonts` next to the executable and the source tree. |
| `CALIPER_DISPLAY_FONT` | The path of the display font file, in place of `fonts/SternbildDisplay-Regular.otf` in the app-data folder. |
| `CALIPER_HUB_FILE` | The path of the hub file that `caliper serve` writes, in place of `hub.json` in the app-data folder, when it is set and not empty. |
| `CALIPER_API_TOKEN` | Caliper does not read it. The examples of [Caliper API](../agents/http-api.md#worked-examples) use this name for your session token. You copy the token from the start line of the app into your own shell, so that your scripts run commands without review. Caliper never writes that token to a file. |
| `CALIPER_API=1` | Serves [Caliper API](../agents/http-api.md) as `--api` does, for the app, `caliper dev` and `caliper shot`. Any other value leaves it off. On Windows the host ignores it and prints a line on standard error ([Caliper CLI](cli.md#windows-limits)). |
| `CALIPER_SDK_DIR` | The prefix of the installed SDK that [`caliper build`](applet-commands.md#caliper-build) configures against, when it is set and not empty. A folder without the SDK is an error. Without this variable, `build` looks beside the executable and in `caliper-sdk` in the home folder. |
| `CALIPER_CMAKE` | The cmake that `caliper build` runs, in place of `cmake` on `PATH`, when it is set and not empty. |
| `CALIPER_DATA_DIR` | The app-data folder, in place of the one for the OS. It moves every item in [the folder table](#app-data-folder). The viewer sets it from its `--data-dir` flag. |
| `CALIPER_DUCKDB_THREADS`, `CALIPER_DUCKDB_MEMORY_LIMIT` | The thread count (an integer from 1 to 1024) and the memory limit (a DuckDB size string such as `4GB`) for all three stores. By default each store takes a share of the RAM and cores of the machine. |
| `CALIPER_DUCKDB_<STORE>_THREADS`, `CALIPER_DUCKDB_<STORE>_MEMORY_LIMIT` | The same limits for one store, where `<STORE>` is `METRICS`, `DATA` or `ARTIFACTS`. They win over the variables above. The host ignores an unusable value and a `[duckdb]` line on stderr says so. |

On Linux and macOS the host also reads `HOME` (on Windows `USERPROFILE`) to shorten paths in the Activity log and to find the app-data folder.

## Renderer Variables

| Variable | What it does |
|---|---|
| `CALIPER_RENDERER=gl` | Refused: the OpenGL renderer it chose was removed. The app and `caliper shot` exit with 1, and an embedding host that asks for the default renderer gets no core ([No Fallback Renderer](../applets/rendering.md#no-fallback-renderer)). Any other value leaves the platform's renderer, Metal on macOS and Vulkan on Windows. |
| `CALIPER_VULKAN_NO_DIRECT` | Set to any value, it turns off the [direct path](../development/vulkan-and-cuda.md#direct-image-path), where a CUDA kernel writes the Vulkan image, so updates use the buffer path. |
| `CALIPER_VULKAN_STATS` | Set to any value, the Vulkan renderer prints one summary line at shutdown with frame times and the count of direct and buffer updates. |
| `CALIPER_METAL_STATS` | Set to any value, the Metal renderer records frame times, the GPU time of each update, view draw and frame, and the updates by path, and prints one summary line at shutdown with its copy count ([Copy Counters and Frame Statistics](../applets/rendering.md#copy-counters-and-frame-statistics)). `caliper-viewer --stats` sets it. |

## Applet and Example Variables

| Variable | What it does |
|---|---|
| `CALIPER_SINE_AUTOTRAIN`, `CALIPER_EMBED_AUTOTRAIN`, `CALIPER_GPT_AUTOTRAIN` | SineScope, EmbedScope or GPTScope starts training on its first frame. |
| `CALIPER_EXPORT_SELFTEST` | MeshScope and TwinScope request one figure export and one recording, to exercise `export.v1`. |
| `FS_FIELD=<n>` | FieldScope starts in field mode `n`. |
| `CALIPER_HELLO_CRASH` | The hello applet faults in its frame function, to exercise quarantine. |
| `CALIPER_HELLO_INIT_FAIL` | The `on_init` function of the hello applet returns false. |
| `CALIPER_EMBED_APPLETS` | The applets folder the `embed_host` example passes in `CaliperCoreDesc.applets_dir`. |
| `CALIPER_EMBED_EXIT_AFTER=<seconds>` | The `embed_host` example closes its window after that many seconds. |
| `CALIPER_EMBED_OFFSCREEN` | Any value other than empty or `0` runs `embed_host` with no window. It exits with 0 only when the applet drew. |

## Test Variables

| Variable | What it does |
|---|---|
| `CALIPER_REQUIRE_GPU=1` | A skip for want of a GPU becomes a failure. |
| `CALIPER_TEST_FORCE_NO_GPU=1` | The device suites behave as if there were no GPU. |
| `CALIPER_TEST_RENDERER_FAIL=1` | The Metal or Vulkan renderer's start fails before it touches a device, with the reason `CALIPER_TEST_RENDERER_FAIL=1 made its start fail`, so the refusals of the app, `caliper shot` and Caliper Embed can be tested on a machine whose GPU works. |
| `CALIPER_TEST_APPLET_DIR=<folder>` | Runs the headless acceptance case of `caliper_tests` on the applets in that folder. Without it the case is skipped. |
| `CALIPER_GFX_CAPTURE_DIR=<folder>` | `caliper_gfx_tests` writes PNG captures of the graphics layer's rows, the colormap rows, the scene core's rows and the scene view's rows to that folder, and `caliper_gfx_views_tests` each points view case's readback as `points-view-<case>.png`. Without it nothing is written. |
| `CALIPER_GFX_RECORD=<file>` | `caliper_gfx_views_tests` records each case's line of the graphics suite's reference mode (its readback's hash, path label, copies and header) into that file instead of comparing. |
| `CALIPER_GFX_REFERENCES=<file>` | `caliper_gfx_views_tests` compares with the references in that file instead of the committed `tests/gfx/references/points_view.txt`. |
| `CALIPER_GFX_REFERENCE_IMAGES=<folder>` | `caliper_gfx_views_tests` compares each case's readback pixel by pixel with its image in that folder, written by an earlier build with `CALIPER_GFX_CAPTURE_DIR`, and lists the differing pixels. Without it that row returns at once. |
| `CALIPER_GFX_POINTS_CASES=<text>` | `caliper_gfx_views_tests` runs only the points view's cases whose names contain the text. |
| `CALIPER_GFX_POINTS_BENCH=1` | `caliper_gfx_views_tests` runs the points view's timing row: the CPU time to compile and submit a redraw and the GPU time of the draw at 100,000, 1,000,000 and 4,000,000 points, with the load average. Without it the row returns at once. |
| `CALIPER_TEST_SHOTS=<folder>` | `caliper_live_views_tests` writes PNG screenshots of its offscreen canvas to that folder: a 3-D points scene from three angles and the scene view with its header and scale bar in Y up, in Z up and in 2-D. Without it nothing is written. |
| `CALIPER_GFX_LAYER_BENCH=1` | `caliper_gfx_tests` runs the graphics layer's timing row: the CPU time to build and submit 1,000 draws a frame on the recording backend and on Metal, printed with the load average. Without it the row returns at once. |
| `CALIPER_GFX_SCENE_BENCH=1` | `caliper_gfx_tests` runs the scene core's timing row: the compile time and the layer's submit time of the surface scene and of a scene of 1,000 entities. Without it the row returns at once. |
| `CALIPER_GFX_SCENE_VIEW_BENCH=1` | `caliper_gfx_tests` runs the scene view's timing row: the view's compile, submit and GPU times for the demo scene at 512 x 512 and the time to apply one mesh update. Without it the row returns at once. |
| `CALIPER_GFX_CMAP_BENCH=1` | `caliper_gfx_tests` runs the colormap's timing row: the GPU time of the generated `colormap_texture` against the Metal renderer's float pass on 1,024 by 1,024 and 2,048 by 2,048 heatmaps, with the load average. `CALIPER_GFX_CMAP_BENCH_SAMPLES=<n>` sets the samples of each pass, 40 by default. Without it the row returns at once. |
| `CALIPER_CMAP_BENCH=1` | `caliper_tests` runs the timing row of the colormap keys: the time to derive one range's keys and to find a block in the cache. Without it the row returns at once. |
| `CALIPER_SLOW_TESTS=1` | Runs the tests under the ctest label `slow`, `caliper_colormap_exhaustive` and `caliper_colormap_gpu_exhaustive`, which compare the colormap by keys with `map_f32_to_rgba8` for all 2<sup>32</sup> inputs. Without it they report themselves skipped. |
| `CALIPER_TEST_THREADS=<n>` | The CPU threads of the two exhaustive colormap tests, 6 by default and at most 6. |
| `CALIPER_EXHAUSTIVE_RANGES=<n>`, `CALIPER_GPU_EXHAUSTIVE_CHUNKS=<n>` | Limit `caliper_colormap_exhaustive` to its first n ranges and `caliper_colormap_gpu_exhaustive` to its first n of 256 chunks, the small cases whose peak memory is measured before a full run. A limited run fails, so it never passes as a full one. |
| `CALIPER_METAL_STATS` | The renderer variable above. `caliper_live_views_tests` sets it to check the frame statistics, and `python/bench/bench_show.py` sets it for its window-mode rows on the offscreen canvas. |
| `CALIPER_SCENE_CAPTURE_DIR=<folder>` | `python/tests/test_scene_app.py` saves the two frames the app draws there, as `scene-view-app.png` and `scene-view-update.png`. Without it nothing is written. |
| `CALIPER_BENCH_KEEP_RAW` | Set to any value, `python/bench/bench_live.py` writes the viewer's log and latency samples of each row to `python/bench/results/S-raw-<tag>-*.txt`. Without it they go to the private data folder of the run. |
| `CARGO_TARGET_DIR` | The folder in which `tools/shaders/generate.py` builds the shader translator's driver, in place of `tools/shaders/naga-driver/target/`. |
| `CALIPER_VIEWS_REFERENCE=<folder>` | The web package's `web/test/wasm.spec.ts` compares each fixed view's picture with the file of the same name in that folder, saved from an earlier build, and fails on any differing byte of a view's data area. Without it the pictures are only saved. |
| `CALIPER_VIEWS_COMPAT=1` | `web/test/wasm.spec.ts` asks every mount of the WebAssembly views for a compatibility-mode WebGPU device. |
| `CALIPER_COLORMAP_CASES=<file>` | `web/test/shaders.spec.ts` and `web/test/colormap.spec.ts` take the colormap's cases from the file that `caliper_colormap_cases <file>` wrote from the C++: each range's block, inputs and reference colours. Without it their colormap tests are skipped. |

ctest runs every test binary that opens the stores, and every command-line test, with `CALIPER_DATA_DIR` on a folder in the build tree. See [Development basics](../development/development-basics.md) for how the GPU skip variables behave.

## Python Variables

| Variable | What it does |
|---|---|
| `CALIPER_SESSION=0` | Stops the first view call from opening a live session automatically. Calls then do nothing until `caliper.window()` or `caliper.offscreen()` is opened. |
| `CALIPER_VIEWER` | The path of the `caliper-viewer` executable, in place of the one inside the package. |
| `CALIPER_DATA_DIR` | The data folder that an in-process window or offscreen canvas uses when no `data_dir` argument is given. Without either, it uses a private temporary folder. |
| `CALIPER_PY_LOG` | Any value other than `0` echoes the diagnostic lines of the host from the in-process core to stderr. |
| `CALIPER_WEB_DIR` | The folder of the browser files that `serve()` and the widgets hand out, in place of `caliper/_web/` in the package. It is also the folder from which the app serves the WebAssembly views to the browser's dashboard viewer. |

See [Caliper LIB](../python/caliper-lib-api.md) for the calls themselves.

## Startup Log Lines

The `caliper` executable writes these to stderr and, for the renderer, to the Activity pane.

| Line | Meaning |
|---|---|
| `[renderer] metal` or `[renderer] vulkan` | The backend that initialized. |
| `caliper: the Vulkan renderer could not start (<reason>), and there is no fallback renderer until the WebGPU fallback through Dawn lands` | The platform's backend failed to start, for example when there is no Vulkan driver, and the app exits with 1 (Metal on macOS). [No Fallback Renderer](../applets/rendering.md#no-fallback-renderer) gives the other refusals. |
| `[autolaunch] <id>` | `CALIPER_AUTOLAUNCH` matched an applet and launched it. |

On Windows the Vulkan renderer reports how each tensor update travelled. Each distinct message prints once, as `[vulkan] device path: <message>`:

| Message | Meaning |
|---|---|
| `CUDA interop OK — zero-copy (kernel reads the tensor, writes the image)` | The CUDA kernel wrote the update straight into the Vulkan image. |
| `import path OK — zero-copy (imported allocation in place)` | The renderer used an imported allocation in place, with no copy. |
| `geometry path OK — points drawn from imported allocation in place` | `geometry.v1` drew points from an imported allocation. |
| `tensor not CUDA-resident`, `external-memory interop unavailable`, `dtype not f32/u8`, `CUDA context init failed` and similar | The update took the CPU-staged or buffer path, and the message gives the reason. The fallback is not an error. See [Platform rules](../introduction/architecture.md#design-decisions). |
| `direct image: exportable image failed, using the buffer path` | The direct path could not be set up, and updates use the buffer path. |

Other host lines carry a tag: `[jobs]`, `[ui]`, `[metrics]`, `[artifacts]`, `[data]`, `[duckdb]`, `[live]` and `[session]`. A program that embeds the core through Caliper Embed receives its own `renderer: <name>` line through `log_fn`, and applet log lines tagged `[applet] `.

The device that `caliper.device.v1` reports is found at startup: Metal on Apple, and the CUDA driver API elsewhere. See [`caliper.device.v1`](../applets/services/caliper-device-v1.md).

