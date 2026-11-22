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

The same executable covers applet-side logic that needs neither torch nor a window, such as the instanced pose helper, the pulse ring and the export affordance. The build makes the `hello` applet and `throwing_applet` first, because the loader tests load them from the build folder.

## Fixture Host

`sdk/testing/caliper/fixture_host.h` is a headless fake `CaliperHost`. It serves `caliper.log.v1` and returns `NULL` for every other service. A test can inject a table for any other service with `provide`. The fixture lets an applet's logic and the sugar run without the UI. Only one fixture exists per process. The CMake target is `caliper::sdk_testing`. See [C++ sugar](../applets/cpp-sugar.md) for its use in an applet's own tests.

## Pixel-Exact Graphics Suite

`caliper_gfx_tests` (`tests/gfx/gfx_main.cpp`) checks the path from a tensor to a texture and the geometry path. It opens a hidden window, so it needs a graphical session. Every case reads the texture back from the GPU and compares it byte for byte with a CPU reference.

The suite runs once for each backend that the platform offers:

- OpenGL runs everywhere, as the fallback.
- Metal runs on macOS when a Metal device is present. It also covers the device paths, built from raw `MTLBuffer` allocations: the compute path for `f32` with a lookup table, the blit path for `u8`, stream-ordered updates, imported allocations and their gates, and the geometry service.
- Vulkan runs on Windows with the CPU-staged path. Cases that need CUDA hardware run only when a CUDA device is present.

`caliper_export_tests` does the same for `caliper.export.v1`. It drives the service as an embedder would and decodes the PNG that the service wrote. Its sidecar, atomic-write and sugar cases need no GPU.

Both suites open their windows hidden and never show or focus them. On macOS the test process also shows no Dock icon and no menu bar. The suites start GLFW through `tests/glfw_background.h`, which sets `GLFW_COCOA_MENUBAR` to false and sets the activation policy to `NSApplicationActivationPolicyProhibited` around `glfwInit()` (`src/host/background_process.h`). One case in each suite checks that the policy still holds.

The app uses the same mechanism with `--hidden` or `CALIPER_HIDDEN=1` ([The hidden window](../reference/cli.md#hidden-window)). The ctest entry `caliper_cli_app_hidden` (`tests/test_api.cpp`) starts the real app this way with `--api`. It uses every endpoint of [Caliper API](../agents/http-api.md) and checks that a capture is a PNG of the hidden framebuffer. It then closes the app with SIGTERM and checks that no process, file or socket remains. Without a Metal device the entry is skipped.

## Embedding and Live Streaming

`caliper_embed_tests` drives `libcaliper` as a third-party embedder would. It creates an offscreen core, loads the `hello` applet, pumps frames, reads pixels back, shuts down and checks the refusal gates.

`caliper_live_tests` and `caliper_live_views_tests` cover live streaming on macOS only, in one process over an offscreen core. See [Live streaming](../python/live-streaming.md). They check the controls of wire protocol 3.1 at three levels:

- The state of the live page on a scripted source: who may control, the rate of `set`, pending values, actions and selections.
- The native stream against a fake script.
- The run bar, drawn on an offscreen canvas and clicked through the ImGui input queue (`src/host/live/live_ui_driver.h`).

The ctest entry `caliper_app_live_controls` (`tests/test_net_hub.cpp`) starts the real app hidden with `--url` on a hub in the test process. With `CALIPER_TEST_UI` it clicks Pause, types a parameter value, runs an action and shift-drags across a heatmap. It checks the run bar and what the script received, once with the hub's `--allow-control` and once without.

## Command Line

The entries named `caliper_cli_*` (label `cli`) run the `caliper` executable as a user runs it, through `tests/cli_run.cmake`. Each entry checks the exit code and the output. The entries run with `CALIPER_APPLET_PATH` unset and with `CALIPER_DATA_DIR` on a folder that must not exist after the run. This proves that the read-only subcommands create nothing in the app-data folder. `caliper_cli_new_builds` (labels `cli` and `sdk`) configures, builds and tests the project that `caliper new` writes, against the SDK that this build installs.

`caliper_cli_shot_temp` and `caliper_cli_shot_data_dir` (labels `cli` and `shot`) run [`caliper shot`](../reference/shot-and-serve-commands.md#caliper-shot) on the `hello` fixture through `tests/shot_run.cmake`. `caliper_png_check` checks that the image has the requested size and is not one flat color. The first entry runs without `CALIPER_DATA_DIR` and checks that the temporary data folder is gone and that nothing exists under `HOME`. The second entry runs with it and checks that the applet's data folder exists and that no `imgui.ini` was written. Without a GPU both entries are skipped, unless `CALIPER_REQUIRE_GPU=1` makes the skip a failure.

## Test Data Location

No test reads or writes your own app-data folder. ctest runs every entry with `CALIPER_DATA_DIR` on `test-app-data/` in the build folder and with `CALIPER_TEST_RUN=1`. The file that `caliper_offline_tests()` writes (`src/builder/cmake/CaliperOfflineTests.cmake`) sets both, so an entry cannot forget them. Some entries replace that folder with their own:

- Every test binary that opens the stores uses `tests/app-data/<binary>`. These binaries are `caliper_tests`, `caliper_export_tests`, `caliper_embed_tests`, the live suites and `caliper_catalog_providers_tests`.
- Every command-line test has its own `CALIPER_DATA_DIR`.
- Every test that starts the hidden app, `caliper shot` or a producer gives that process its own folder.

`CALIPER_DATA_DIR` moves the whole app-data folder, so the stores, the applets' data folders, `imgui.ini`, the live session files, the stream files, the API files and the default user applets folder all stay in the build folder. The tests of Caliper LIB (`python/tests/conftest.py`) set `CALIPER_TEST_RUN=1` and give every test a temporary `CALIPER_DATA_DIR`.

The app-data guard backs this up ([Folders and environment variables](../reference/folders-and-environment.md#app-data-folder)). A test or hidden run that would use your own app-data folder stops at once with exit code 1, before it creates anything there. The guard applies to `--hidden`, `CALIPER_HIDDEN=1`, any `CALIPER_TEST_*` variable and any test binary, because each binary links `tests/test_run_marker.cpp`. A test binary that you run by hand, from a shell or an IDE, therefore needs `CALIPER_DATA_DIR`, as ctest gives it.

`caliper_test_data_guard` (labels `cli` and `shot`, `tests/data_guard_run.cmake`) runs the typical launches with `HOME` on a stand-in folder: the hidden app, `caliper_embed_tests`, `caliper shot` and `caliper.serve()`. It checks that each refusal happens and that nothing appears in the stand-in's app-data folder. A test that needs another root, such as the embedding tests of `CaliperCoreDesc.data_dir`, sets its own.

## Applet and Adapter Tests

`caliper_torch_tests` tests the torch adapter in `adapters/torch.hpp`. Five small executables test the learning and simulation code of single applets: `caliper_sculpt_tests`, `caliper_mesh_tests`, `caliper_em_pic_tests`, `caliper_surface_tests` and `caliper_twin_tests`. They link libtorch and exist only when torch is enabled, which keeps the torch link cost out of `caliper_tests`.

## Package and Boundary Tests

`caliper_sdk_install` (label `sdk`) proves that an applet can build against the installed SDK alone and cannot include `<caliper/embed.h>`. It installs the SDK component into a work folder and builds `tests/sdk_install_probe` and `examples/hello` there with only `-DCMAKE_PREFIX_PATH` set to the install. It runs the probe, checks the library and stamped manifest of `hello`, and fails if either build refers to a file in the source tree. It also builds an applet that includes `caliper/embed.h` against every SDK target, and each build must fail. `tests/sdk_install_probe/run.cmake` holds the steps, and `CALIPER_SDK_PROBE_DIR` sets the work folder. See [SDK package](../applets/sdk-package.md).

`caliper_embed_install` (label `embed`) does the same for the embedding package. It installs the embed component alone and checks the installed tree. On macOS it checks that the library exports exactly the C functions that `libcaliper` defines. It builds `examples/embed_host` against the install and runs it offscreen on a copy of the `hello` fixture. Without a GPU the run is skipped, unless `CALIPER_REQUIRE_GPU=1` makes the skip a failure. `tests/embed_install/run.cmake` holds the steps, and `CALIPER_EMBED_PROBE_DIR` sets the work folder. See [Embedding Caliper](../web/embedding-api.md).

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

## GPU Tests

A suite that needs a GPU must not report success when it finds none. `tests/gpu_skip.h` provides the convention that the gfx, export, embed, live and torch suites share:

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

- `gfx`: `caliper_gfx_tests`.
- `export`: `caliper_export_tests`.
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
