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
| `api.json` | The API file, while the app or `caliper shot` serves [Caliper API](api.md): its address, its session token and the link of its spec page (mode 0600), removed when the process exits. `shot` writes it here also when it keeps its stores in a temporary folder. |
| `live/` | The live session folder. A script that calls `caliper.show` writes `<pid>.json` here, and the viewer lists these files; see [Attach to a running script](../guides/attach-to-a-running-script.md). |
| `imgui.ini` | The saved window layout. Delete it to return to the default layouts. |
| `fonts/SternbildDisplay-Regular.otf` | The default location of the display face, used when `CALIPER_DISPLAY_FONT` is not set. |
| `metrics.duckdb`, `data.duckdb`, `artifacts.duckdb` | The stores behind [`metrics.v1`](service-metrics-v1.md), [`data.v1`](service-data-v1.md) and [`artifacts.v1`](service-artifacts-v1.md). |
| `artifacts/` | The content-addressed files of `artifacts.v1`. |
| `workspace/` | The builder's default workspace: its loaders, specs, saved spec applets, the command journal `journal/commands.jsonl` and `workspace.duckdb`. The app creates it only when the builder first needs it, and `--workspace` or `CALIPER_WORKSPACE` names another folder in its place. See [The workspace](cli.md#the-workspace). |

## Host variables

The `caliper` executable reads these. The first six are development hooks for screenshots, recordings, headless checks and clean-exit runs.

| Variable | What it does | Read in |
|---|---|---|
| `CALIPER_AUTOLAUNCH=<id>` | Launches the applet with that manifest id as soon as the host starts, skipping the landing screen. | `src/main.cpp` |
| `CALIPER_AUTOATTACH=<pid>` | Attaches to that process's live session as soon as the landing screen lists it. | `src/main.cpp` |
| `CALIPER_EXIT_AFTER=<seconds>` | Closes the window normally after that many seconds, so the full teardown path runs. | `src/main.cpp` |
| `CALIPER_HOME_AFTER=<seconds>` | Returns to the landing screen once, that many seconds after an applet page, live page or spec page opened. | `src/main.cpp` |
| `CALIPER_WINDOW_SIZE=<W>x<H>` | Sets the window size in screen coordinates, for example `1440x900`. A malformed value is ignored and the default of 95% of the work area stands, or 1280 by 800 with `CALIPER_HIDDEN`. | `src/main.cpp` |
| `CALIPER_VIEWS_DEBUG=1` | Shows each view's diagnostics on a line under its header: the size, data type, device and renderer path, the sequence number, the step-to-screen latency, the update count and the drops. Off by default, in the app, `caliper-viewer`, `caliper shot` and Caliper Embed. | `src/host/live/live_views.cpp` |
| `CALIPER_TEST_UI=<steps>` | A testing aid, read only with `--hidden` or `CALIPER_HIDDEN=1`: plays mouse and key input into the live page through ImGui's input queue, so a test clicks the real widgets with nothing on screen. Steps are separated by `;`: `click <widget>` (`pause`, `resume`, `step`, `action:<name>`, `more`), `set <parameter> <value>` (a Cmd-click on macOS, a Ctrl-click elsewhere, on the slider or field, then the value typed and Enter), `toggle <parameter>`, `select <view>` (a shift-drag across the top quarter of the view's image or plot area), `leave` (the pointer leaves the window), `wait <frames>` and `state <run state>`. A picker's menus are widgets too: `click picker:<view>:<dim>`, then `click picker:<view>:<dim>=<entry>`. Each step and its outcome is printed to stderr as an `[ui]` line, and the page's controls print `[live]` lines. See `src/host/live/live_ui_driver.h`. | `src/main.cpp` |
| `CALIPER_TEST_SPEC_UI=<steps>` | A testing aid, read only with `--hidden`: plays mouse and key input into the spec page through ImGui's input queue and reads named frames back. Steps are separated by `;`: `wait <frames>`, `focus <panel>` (a click in the middle of a shown panel), `leave`, `key <chord>` (`cmd+k`, Ctrl+K off macOS, `escape` or `enter`), `type <text>`, `until <state>` (the inline prompt's state, such as `prompt`, `preview` or `applied`), `shot <name>` (the next frame as `<name>.png` in `CALIPER_TEST_SPEC_UI_SHOTS`, at the framebuffer's size) and `quit`. Each step is printed to stderr as an `[ui]` line; a shot's line gives the prompt's state, the focused panel and the open spec's undo steps. A hidden app given `CALIPER_AGENT_TEST_SCRIPT` runs the widget agent's sidecar on that script, with no key store, and names it Scripted. See `src/host/builder/spec_ui_driver.h`. | `src/main.cpp` |
| `CALIPER_TEST_SPEC_UI_SHOTS=<folder>` | Where `CALIPER_TEST_SPEC_UI` writes its shots; the working folder without it. | `src/main.cpp` |
| `CALIPER_HIDDEN=1` | A testing aid, as `--hidden`: runs the real app with its window never shown, no Dock icon and no menu bar, reads and writes no `imgui.ini`, and closes the window normally on SIGINT and SIGTERM. See [The hidden window](cli.md#the-hidden-window). | `src/main.cpp` |
| `CALIPER_WORKSPACE` | The builder's workspace folder, in place of `workspace/` under the app-data folder, when it is set and not empty and `--workspace` is not given. The app, `caliper dev` and `caliper shot` open it once they have drawn their first frame. See [The workspace](cli.md#the-workspace). | `src/host/builder/app_hooks.h` |
| `CALIPER_COMMAND_SCRIPT=<file>` | A testing and demo aid: a file of JSON lines, one builder command each, that the app, `caliper dev` or `caliper shot` runs through the command queue once its workspace is open, with the origin `ui` unless a line gives another. `shot` draws frames until the script is done. See [The workspace](cli.md#the-workspace). | `src/host/builder/app_hooks.h` |
| `CALIPER_COMMAND_RESULTS=<file>` | The file that receives the result of each command of `CALIPER_COMMAND_SCRIPT`, one JSON line each, then a line `{"done": true, "commands", "failed"}`. It is replaced at each run. | `src/host/builder/app_hooks.h` |
| `CALIPER_AGENT_PYTHON=<interpreter>` | The Python interpreter, with the dependencies of `caliper-agent`, with which the app starts the widget agent's sidecar for ⌘K on a spec page. Until `caliper-agent` is packaged with a Python runtime, the widget agent has no provider without it, and the inline prompt offers only Ask in chat. The sidecar also needs Caliper API (`--api`) and a key in the system key store. | `src/host/builder/widget_app.cpp` |
| `CALIPER_AGENT_SOURCE=<folder>` | The folder that holds `agent/caliper_agent/` and `commands/command_table.json`, in place of this build's `src/builder`. | `src/host/builder/widget_app.cpp` |
| `CALIPER_CHROMELESS=1` | Hides the page bar, the status bar and the jobs tray on the applet page, so the applet's windows fill the whole window, for recordings. Jobs keep running, and going Home cancels them. Esc still returns to the landing screen, and the landing screen and the live page keep their bars. Any other value leaves the bars. [`caliper shot`](cli.md#caliper-shot) honours it too. | `src/main.cpp`, `src/host/shot.cpp` |
| `CALIPER_USER_APPLETS_DIR` | The user applets folder, in place of `applets/` under the app-data folder, when it is set and not empty. The host searches it after the development folders, and `caliper install` copies into it. Caliper Embed reads it too. | `src/app_paths.cpp` |
| `CALIPER_APPLET_PATH` | Folders to search for applets before the user applets folder and the shipped applets, separated by `:` (`;` on Windows), searched in the order given and after any `--applets` folders. A folder that holds no manifest but whose `applets/` subfolder does, such as an applet's build folder, stands for that subfolder (see [Build folders](cli.md#build-folders)). Caliper Embed reads it too, after its `applets_dir`. | `src/main.cpp`, `src/host/embed/embed_core.cpp` |
| `CALIPER_FONTS_DIR` | A folder that holds the UI fonts. It is searched before `assets/fonts` next to the executable and the source tree. | `src/host/ui/sternbild_fonts.cpp` |
| `CALIPER_DISPLAY_FONT` | The path of the display font file, in place of `fonts/SternbildDisplay-Regular.otf` in the app-data folder. | `src/host/ui/sternbild_fonts.cpp` |
| `CALIPER_HUB_FILE` | The path of the hub file that `caliper serve` writes, in place of `hub.json` in the app-data folder, when it is set and not empty. | `src/main.cpp` |
| `CALIPER_API=1` | Serves [Caliper API](api.md) as `--api` does, for the app, `caliper dev` and `caliper shot`. Any other value leaves it off. On Windows it is ignored, and a line on standard error says so ([Caliper CLI](cli.md#on-windows)). | `src/host/api/app_api.cpp` |
| `CALIPER_SDK_DIR` | The prefix of the installed SDK that [`caliper build`](cli.md#caliper-build) configures against, when it is set and not empty; a folder without the SDK is an error. Without it, `build` looks beside the executable and in `caliper-sdk` in the home folder. | `src/host/cli_build.cpp` |
| `CALIPER_CMAKE` | The cmake that `caliper build` runs, in place of `cmake` on `PATH`, when it is set and not empty. | `src/host/cli_build.cpp` |
| `CALIPER_DATA_DIR` | The app-data folder, in place of the one for the OS. It moves the whole folder: the three stores and `artifacts/`, `imgui.ini`, every applet's data folder under `data/`, the live session files under `live/`, the hub file `hub.json` of `caliper serve`, the API file `api.json`, the default location of the display face under `fonts/`, the builder's default workspace `workspace/` and, unless `CALIPER_USER_APPLETS_DIR` is set, the user applets folder `applets/`. It is read once, at the first lookup. An embedder's `CaliperCoreDesc.data_dir` wins over it for everything but `live/`. The viewer sets it from its `--data-dir` flag, and the test suite sets it to a work folder so that no test opens the user's stores. | `src/app_paths.cpp`, `src/host/live/live_shm.h`, `src/host/live/viewer_main.mm` |
| `CALIPER_DUCKDB_THREADS`, `CALIPER_DUCKDB_MEMORY_LIMIT` | The thread count (an integer from 1 to 1024) and the memory limit (a DuckDB size string such as `4GB`) for all three stores. By default each store takes a share of the machine's RAM and cores. | `src/host/duckdb_sizing.h` |
| `CALIPER_DUCKDB_<STORE>_THREADS`, `CALIPER_DUCKDB_<STORE>_MEMORY_LIMIT` | The same limits for one store, where `<STORE>` is `METRICS`, `DATA` or `ARTIFACTS`. They win over the variables above. An unusable value is ignored and a `[duckdb]` line on stderr says so. | `src/host/duckdb_sizing.h`, `src/host/duckdb_config.h` |

On Linux and macOS the host also reads `HOME` (on Windows `USERPROFILE`) to shorten paths in the Activity log and to find the app-data folder.

## Renderer variables

| Variable | What it does | Read in |
|---|---|---|
| `CALIPER_RENDERER=gl` | Selects the OpenGL fallback instead of Metal or Vulkan. Any other value leaves the platform default. An embedding host that asks for the default renderer honours it too. | `src/host/core_lifecycle.cpp` |
| `CALIPER_VULKAN_NO_DIRECT` | Set to any value, it turns off the [direct path](../explanations/vulkan-cuda.md#the-direct-image-path), where a CUDA kernel writes the Vulkan image, so updates use the buffer path. | `src/host/renderer/vulkan_renderer.cpp` |
| `CALIPER_VULKAN_STATS` | Set to any value, the Vulkan renderer prints one summary line at shutdown with frame times and the count of direct and buffer updates. | `src/host/renderer/vulkan_renderer.cpp` |

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
| `CALIPER_TEST_APPLET_DIR=<folder>` | Runs the headless acceptance case of `caliper_tests` on the applets in that folder: each is negotiated, opened, checked, created and destroyed with the capabilities the `caliper` executable advertises. Without it the case is skipped. | `tests/test_applet_acceptance.cpp` |
| `CALIPER_THROWING_WHERE` | Chooses where the test applet fails: `create`, `initialize`, `frame`, `frame-fault`, `frame-imgui`, `frame-imgui-fault`, `cleanup`, `job`, `job-fault` or `none`. The crash-guard and UI-recovery tests set it. | `tests/fixtures/throwing_applet/throwing_applet.cpp` |

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
| `[renderer] metal`, `[renderer] vulkan` or `[renderer] gl` | The backend that initialised. The Activity pane shows the same name as a `renderer` entry. |
| `[renderer] <name> init failed; falling back to gl` | The default backend failed to initialise, for example when there is no Vulkan driver, and the host recreated its window for OpenGL. |
| `[autolaunch] <id>` | `CALIPER_AUTOLAUNCH` matched an applet and launched it. |

On Windows the Vulkan renderer reports how each tensor update travelled. Each distinct message prints once, as `[vulkan] device path: <message>`:

| Message | Meaning |
|---|---|
| `CUDA interop OK — zero-copy (kernel reads the tensor, writes the image)` | The CUDA kernel wrote the update straight into the Vulkan image. |
| `import path OK — zero-copy (imported allocation in place)` | An imported allocation was used in place, with no copy. |
| `geometry path OK — points drawn from imported allocation in place` | `geometry.v1` drew points from an imported allocation. |
| `tensor not CUDA-resident`, `external-memory interop unavailable`, `dtype not f32/u8`, `CUDA context init failed` and similar | The update took the CPU-staged or buffer path, and the message gives the reason. The fallback is not an error; see [Platform rules](../explanations/architecture.md#platform-rules). |
| `direct image: exportable image failed, using the buffer path` | The direct path could not be set up, and updates use the buffer path. |

Other host lines carry a tag: `[jobs]` for failed jobs and jobs still running at shutdown, `[ui]` for a UI state unwound after an applet fault, `[metrics]`, `[artifacts]` and `[data]` when a store fails to open, `[duckdb]` for an ignored limit override, `[live]` for session events, and `[session]` for the builder session: its workspace opened, not opened or closed, each command that changed the workspace, and the command script's failures and end. A program that embeds the core through Caliper Embed receives its own `renderer: <name>` line through `log_fn`, and applet log lines tagged `[applet] `.

The device that `caliper.device.v1` reports is found at startup: Metal on Apple, and the CUDA driver API elsewhere. See [`caliper.device.v1`](service-device-v1.md).
