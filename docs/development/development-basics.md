# Development Basics

An applet is one shared library that the running host loads. This page shows the sources of each library, the files of an applet and the edit-build-run loop. Read it before [Your first applet](../applets/first-applet.md).

```mermaid
flowchart LR
    core[Core build] -->|installs| sdk[SDK package]
    sdk -->|caliper new| project[Applet project]
    project -->|cmake build| lib[Library and manifest]
    lib -->|caliper check| ok[Checked applet]
    ok -->|caliper dev| run[Running host]
```

The loop builds the host once, installs the SDK, then builds, checks and runs your applet against that SDK.

## Mental Model

The host owns the window, the renderer, the ImGui contexts and the frame loop, as [Architecture](../introduction/architecture.md) shows. It also owns the job system, the metrics, artifacts and data stores, and the docked desktop. Persistence and crash containment are host work too. An applet never creates a window, a GL or Metal context or an ImGui context.

An applet gives three things to the host. Its UI is the ImGui, ImPlot and ImPlot3D calls in its frame function. Its compute is the work that it sends to `jobs.v1` on host worker threads. Its state is what it keeps from one frame to the next.

## Library Sources

The CMake build wires the project dependencies together. Install the platform toolchains first, as the root `README.md` describes.

| Library | Comes from | You link it via | Notes |
|---|---|---|---|
| SDK headers (ABI, services, sugar, adapters) | The installed `caliper-sdk` package (`include/caliper/`), from `sdk/include/` in this repository | `caliper::sdk` | Header-only INTERFACE target. Nothing compiles into the SDK itself. |
| ImGui (docking), ImPlot, ImPlot3D, FileDialog | The pinned sources in the SDK package, the same ones the host uses | `caliper::ui_stack` | Compiled in your applet's build, with no window or graphics backend. **Never bring your own copy.** The pin is part of the ABI ([Compatibility and Epochs](compatibility-and-epochs.md)), and your applet and the host must agree on the ImGui memory layout. |
| libtorch | The libtorch that the core build downloaded, which the SDK names | `caliper_add_applet(... TORCH)` | The applet links the host's libtorch and no other version, and the manifest records that version. The `caliper` executable and `libcaliper` do not link it ([Platform rules](../introduction/architecture.md#design-decisions)). |
| curl, zlib | The macOS system SDK | `CURL::libcurl`, `ZLIB::ZLIB` | Only if you download or decompress data. The host has no downloader. |
| DuckDB | Host-internal | Not possible | No DuckDB type crosses the ABI. Use `metrics.v1`, `artifacts.v1` and `data.v1` instead. |

[SDK Package](../applets/sdk-package.md) describes what the installed SDK holds.

## Applet Anatomy

An applet is three files in one folder:

```
examples/hello/
├── hello.caliper.toml    the manifest, checked before your code loads
├── hello.cpp             your applet class and the CALIPER_APPLET macro
└── CMakeLists.txt        find_package(caliper-sdk) and caliper_add_applet()
```

- **Manifest**: It holds the reverse-DNS `id`, the name, the version, a summary and the lists of required and optional services. The `id` and `version` must be byte-identical to the descriptor in the `CALIPER_APPLET` call. If they differ, the loader refuses the applet and lists it with the reason. See [Manifest](../applets/manifest.md).
- **Macro**: `CALIPER_APPLET(...)` at the bottom of the `.cpp` generates the epoch-2 C ABI glue. This is the descriptor and an exception wall on every entry point. You never write `extern "C"`.
- **Class**: It has three lifecycle methods. `on_init(Host&)` probes the services. Assert the required ones and let the optional ones degrade. `on_frame(const Frame&)` runs on the frame thread every frame. It submits ImGui windows and reads state that workers publish. `on_cleanup()` cancels jobs, waits for them with a bound, releases textures and returns.
- **CMakeLists**: Copy the one from `examples/hello/` for a UI-only applet. Copy the one from `examples/sine_scope/` for an ML applet, which adds `TORCH`. Edit only the project and target name, the source files and the manifest filename.

### Smallest Complete Applet

These listings are the files of `examples/hello/`, embedded verbatim. The folder builds alone against the installed SDK, and the core build uses it as the loader-test fixture. [Your first applet](../applets/first-applet.md) walks through the file line by line.

```cpp title="examples/hello/hello.cpp"
--8<-- "examples/hello/hello.cpp"
```

```toml title="examples/hello/hello.caliper.toml"
--8<-- "examples/hello/hello.caliper.toml"
```

```cmake title="examples/hello/CMakeLists.txt"
--8<-- "examples/hello/CMakeLists.txt"
```

The applet has no `main`, no window, no GL or Metal code, no ImGui context and no event loop. The macro and the host supply them. To add ML, add a `jobs.v1` training job and the bridge, as [Your first ML applet](../applets/first-ml-applet.md) shows.

## Development Loop

```bash
# 1. once, in your Caliper checkout: build the host and install the SDK
cmake --preset release
cmake --build --preset release
cmake --install cmake-build-release --prefix "$HOME/caliper-sdk" --component sdk
CALIPER="$PWD/cmake-build-release/caliper"

# 2. create: a project of your own, outside the checkout
"$CALIPER" new my-applet "$HOME/caliper-applets/my-applet"

# 3. build and test: the library and its stamped manifest land in build/applets/
cd "$HOME/caliper-applets/my-applet"
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/caliper-sdk" -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build

# 4. check, then run with the build folder searched first
"$CALIPER" check build/applets
"$CALIPER" dev build/applets
```

`caliper new my-applet` writes a project for the applet `dev.example.my-applet` with the stem `my_applet`. The project holds the manifest, the `CMakeLists.txt`, a source file with one window, and a test that runs the applet on the fixture host. Pass a full id such as `com.example.my-applet` to choose the id. [Caliper CLI](../reference/applet-commands.md#caliper-new) describes the template.

### Applet Loading

`caliper check` runs the checks that the host makes before it opens a library. `caliper dev` starts Caliper with your folder searched before every other applet folder. The host reads its applet folders at start and keeps a library loaded until it quits. Thus after each rebuild, quit Caliper and run `caliper dev` again.

`--applets <dir>` or `CALIPER_APPLET_PATH` adds development folders in the same way. `caliper install build/applets` copies the applet into the user applets folder to keep it installed. To skip the landing screen on every restart, set `CALIPER_AUTOLAUNCH`:

```bash
CALIPER_AUTOLAUNCH=dev.example.my-applet "$CALIPER" dev build/applets
```

`CALIPER_EXIT_AFTER=<sec>` makes a clean exit after that time, for CI. In CLion, open the applet's folder as a project of its own and add `-DCMAKE_PREFIX_PATH=<sdk prefix>` to its CMake options.

**Testing without a window:** The fixture host (`caliper::sdk_testing`, `<caliper/fixture_host.h>`) fakes `get_service` with tables that you inject. `tests/test_sugar_services.cpp` shows the pattern. Thus you can unit-test applet logic that reads services without a window.

**Your data folder:** The host creates `~/Library/Application Support/Caliper/data/<your-id>/` and hands it to you as `host.data_dir()`. Put downloads, caches and scratch files there and nowhere else. `CALIPER_DATA_DIR` moves the whole app-data folder, and your data folder with it. [Folders and Environment Variables](../reference/folders-and-environment.md) lists the folder on every platform.

## Landing Screen

`./cmake-build-release/caliper` opens on the landing screen. It has a command bar, four docked panes and a status bar. The status bar shows the backend, the GPU, memory and the numbers of installed and unavailable applets.

| Pane | Shows |
|---|---|
| **Projects** | The dashboards of the open workspace, one row each, and the ways in: `F2 Open folder` (disabled), `F3 New dashboard` ([Builder Pane](../builder/builder-pane.md)) and `F4 Attach to process` (see [Training Script Attachment](#training-script-attachment)). |
| **Instruments** | Every installed applet, with name, id, version, tag and status: `READY`, `ACTIVE`, `REFUSED`, `FAILED` or `QUARANTINED`. Only `READY` rows launch. Click a column header to sort. |
| **Inspector** | The manifest of the selected applet and the `⏎ Launch` button. If the applet cannot launch, the button is disabled and the reason is beside it. |
| **Activity** | Running jobs, then the host log, newest first. |

The command bar filters Instruments and the dashboards as you type. Every space-separated term must appear in the name, id or tag, in any case. If the filter hides the selected row, the selection moves to the top match, so Enter always launches the highlighted row.

| Key | Action |
|---|---|
| ⌘K | Focus the command bar. |
| ↑ / ↓ | Move the selection through the dashboards and then Instruments, as one list. |
| Esc | Clear the filter and focus Instruments. |
| Enter | Launch the selected applet. A double-click on a row does the same. |
| ⌘1–⌘4 | Focus Projects, Instruments, Inspector or Activity. |
| ⌘R | Unload and relaunch the selected applet. Offered only for an applet that a C++ exception quarantined. After a signal, restart Caliper. |
| F4 | List the live sessions and streams. In the list, Enter attaches and Esc returns. |
| F5 | Open the live page of the attached session. |
| F6 | Open the spec page of the dashboard that the builder has open. |

On Windows and Linux, use Ctrl in place of ⌘.

A launch replaces the landing screen with the page of the applet. The `esc Home` button returns to the landing screen on the row of the applet. Esc does the same, but only when no text field or popup uses it.

### Training Script Attachment

A Python script that calls `caliper.show` opens a live session. Caliper can attach to it while it runs (macOS, Metal renderer). Press F4 and Projects lists the sessions in `~/Library/Application Support/Caliper/live/` and the `caliper.serve()` streams, newest first. [Script Attachment](../python/script-attachment.md) covers the session files and the viewer.

| Column | Shows |
|---|---|
| **Script** | The name of the script |
| **Pid** | Its process id |
| **Started** | When the session opened |
| **State** | `running`, `starting` (attach waits), `closing`, `viewed by <pid>` (a session has one viewer), `protocol v<N>` (another Caliper version), `stream` or `attached` |
| **Views** | The names of the views that the script has shown |

Only `running`, `starting` and `stream` rows attach. A killed script leaves its session file behind, and the next listing removes it.

Attach opens a live page with the layout of an applet page. The script never waits for Caliper. `⌘D Detach` detaches the session, and the script keeps training. When the script ends, the views keep their last frames and the bar shows `SESSION ENDED`. `CALIPER_AUTOATTACH=<pid>` attaches as soon as the session is listed, and `CALIPER_HOME_AFTER=<seconds>` goes Home once. Both serve screenshots and headless checks.

### Jobs and Window Layout

The **Jobs** tray is a strip above the status bar on every page while a job runs or a failure is kept. It shows each running job with its progress and a `cancel` button, then the failed jobs with their reasons, then `clear finished`. A failed job keeps the strip until you press `clear finished`.

Caliper saves the window layout in `imgui.ini` in the app-data folder. The landing panes cannot be undocked, and only the dividers move. Delete the file to return to the default layouts. The window opens at 95% of the work area of the screen. `CALIPER_WINDOW_SIZE=<W>x<H>` sets the size, for example `1440x900`, and Caliper ignores a malformed value.

## Test Runs

The `release` and `debug` presets build the tests (`BUILD_TESTS=ON`):

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release            # prints the output of failing suites
```

Six suites need a GPU for some cases: `caliper_gfx_tests`, `caliper_export_tests`, `caliper_embed_tests`, `caliper_torch_tests`, `caliper_live_tests` and `caliper_live_views_tests`. A case that finds no device skips and is counted. The suite prints a summary when it ends:

```
[caliper-tests] skipped 49 of 49 test cases: 49 no GPU, 0 not applicable on this platform -> exit 77 (ctest: Skipped)
```

A suite that skipped a case for want of a GPU, and had no failures, exits with code 77. ctest reports it as `***Skipped`, not `Passed`, and its own exit status stays 0. Cases that can never run on the platform, such as the CUDA cases on macOS, count as *not applicable* and leave the suite `Passed`. ctest hides the output of suites that pass, so run it with `--verbose` to see the summary.

| Variable | Effect |
|---|---|
| `CALIPER_REQUIRE_GPU=1` | A skip for want of a GPU becomes a failure. Set it on a machine or CI lane that must have a GPU. |
| `CALIPER_TEST_FORCE_NO_GPU=1` | Test-only. The device suites behave as if there is no GPU. |

When you add a device test, skip through `tests/gpu_skip.h`:

- `CALIPER_SKIP_NO_GPU(why)`: Counts the skip, says why and returns.
- `CALIPER_SKIP_NOT_APPLICABLE(why)`: Is for a case that never applies on the platform.
- `CALIPER_NOTE_NO_GPU(why)`: Only counts, for a case that must tidy up before it returns.

A suite that uses them ends with `CALIPER_GPU_TEST_MAIN()` in place of `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`. A bare `MESSAGE` and `return` would report the suite as passed.

## Applets in This Repository

`caliper_add_applet()` is the same file in the core build and in the SDK package. The core build compiles no applets unless `CALIPER_BUILD_APPLETS=ON`. `examples/hello` builds with the tests as the loader-test fixture. `examples/sine_scope` builds with the tests in a build with libtorch.

Each applet folder under `applets/` and `examples/` also builds alone against the installed SDK. `scripts/build_applets_against_sdk.sh <core build folder> <work folder> [applet folder...]` installs the SDK and builds them that way, as CI does.

An older build folder can keep libraries from in-tree builds that predate `caliper_add_applet()`. Their manifests have no `[build]` table, so the host does not check them against the UI-stack hash. Delete the `applets/` folder of the build after you turn `CALIPER_BUILD_APPLETS` off, and when you reuse such a build folder.

## Test Hooks

Some environment variables exist only for tests, and the user page Folders and Environment does not list them:

- `CALIPER_TEST_UI` and `CALIPER_TEST_SPEC_UI` take step grammars, which `src/host/live/live_ui_driver.h` and `src/host/builder/spec_ui_driver.h` define.
- `CALIPER_TEST_APPROVALS` takes the line forms `approve`, `deny`, `session` and `always`, which `src/host/builder/approval_cards.h` defines. `CALIPER_TEST_OPEN_URL`, `CALIPER_TEST_SPEC_UI_SHOTS` and `CALIPER_AGENT_TEST_SCRIPT` are in the same file.
- The Python tests read `CALIPER_TEST_NO_WINDOW`, `CALIPER_REQUIRE_WEB_VIEWS`, `CALIPER_PY_LIVE_APPLY` and `CALIPER_THROWING_WHERE`.
- `scripts/fetch_wasm.py` reads `GITHUB_TOKEN` or `GH_TOKEN`.

## Docs Preview

The docs site renders the Python API reference from the docstrings of Caliper LIB. To see the docs while you change them, serve the site from the repository root with the versions of `requirements-docs.txt`:

```sh
uv run --no-project --with-requirements requirements-docs.txt mkdocs serve -a 127.0.0.1:8000
```

The server builds the site again when a page, a docstring under `python/caliper/` or a C header changes. `CONTRIBUTING.md` gives the strict build that the Docs workflow runs.

## Rules

- **No raw GL, Metal or Vulkan calls in applets** ([Platform rules](../introduction/architecture.md#design-decisions)). Draw through ImGui, ImPlot and ImPlot3D. Put tensors on screen through `tensor_bridge.v1`. Thus the applet runs the same on both backends.
- **Bridge and `data.v1` calls: frame thread only.** Torch ops: worker thread only.
- **Honor cancel within 100 ms** in every job loop. Check once for each batch and each download chunk.
- **Do not catch errors across the ABI.** The exception walls of the sugar already stop every throw into the host. Inside your own code, handle errors normally.
- **If a service is optional, degrade visibly.** A disabled panel that says "metrics: absent (ok)" is better than a silent absence.
