# Development Basics

What building on Caliper actually looks like: what code you write, what code
is already running, where every library comes from, and the edit-build-run
loop. Read this before [Your first applet](../applets/first-applet.md).

## Mental Model

When your applet runs, **the host is already a living program**: it owns the
window, the renderer (Metal on macOS, Vulkan on Windows, GL fallback), the ImGui/ImPlot/ImPlot3D
contexts, the job system, the metrics/artifacts/data stores, and the docked
desktop. You write **one shared library** that the host loads at runtime.
Your code contributes exactly three things:

1. **UI** — `ImGui::`/`ImPlot::`/`ImPlot3D::` calls inside your per-frame
   function, drawn into windows the host docks and composites.
2. **Compute** — work you submit to `jobs.v1`, running on host-owned worker
   threads.
3. **State** — whatever your applet remembers between frames.

Everything else — event loop, GPU device, rendering, persistence, crash
containment — is the host's job. You never initialize a window, a GL/Metal
context, or an ImGui context; if you find yourself wanting to, you're
fighting the model.

## Library Sources

The repository and CMake build wire the project dependencies together. Platform
toolchains and system prerequisites must still be installed; see the root
`README.md` before configuring a build.

| Library | Comes from | You link it via | Notes |
|---|---|---|---|
| SDK headers (ABI, services, sugar, adapters) | The installed `caliper-sdk` package (`include/caliper/`), from `sdk/include/` in this repo | `caliper::sdk` | Header-only INTERFACE target — nothing compiles into the SDK itself |
| ImGui (docking) + ImPlot + ImPlot3D + FileDialog | The pinned sources in the SDK package, the same ones the host was built from | `caliper::ui_stack` | Compiled inside your applet's build, with no window or graphics backend. **Never bring your own copy** — the pin is part of the ABI (see [Compatibility and epochs](compatibility-and-epochs.md)): your applet and the host must agree on ImGui's memory layout byte-for-byte |
| libtorch | The libtorch the core build downloaded, which the SDK names | `caliper_add_applet(... TORCH)` | It links the host's libtorch and no other version, and stamps that version into the manifest. The `caliper` executable and `libcaliper` do not link it ([Platform rules](../introduction/architecture.md#design-decisions)). |
| curl, zlib | macOS system SDK | `CURL::libcurl`, `ZLIB::ZLIB` | Only if you download/decompress data. Data acquisition is applet business — the host has no downloader |
| DuckDB | host-internal | **you can't** | Deliberately unreachable. You get its powers through services: `metrics.v1`, `artifacts.v1`, `data.v1`. No DuckDB type ever crosses the ABI |

The division is deliberate: what must be *shared* (UI stack, libtorch) is
pinned and provided; what is *yours* (curl) you link privately; what is
*persistent* (DuckDB) hides behind frozen service tables. [SDK package](../applets/sdk-package.md)
describes what the installed SDK holds.

## Applet Anatomy

Three files, one folder. This is the entire footprint:

```
examples/hello/
├── hello.caliper.toml    the manifest — checked BEFORE your code loads
├── hello.cpp             your applet class + the CALIPER_APPLET macro
└── CMakeLists.txt        find_package(caliper-sdk) + caliper_add_applet()
```

- **Manifest**: id (reverse-DNS), name, version, summary, required/optional
  service lists. The `id` and `version` must be **byte-identical** between
  the manifest and the descriptor in the `CALIPER_APPLET` macro call, or the loader refuses (the
  applet is listed with the reason, not a crash). See [manifest reference](../applets/manifest.md).
- **The macro**: the `CALIPER_APPLET(...)` at the bottom of your `.cpp`
  generates the epoch-2 C ABI glue — the descriptor plus exception walls on
  every entry point. You never write `extern "C"` yourself.
- **The class**: three lifecycle methods. `on_init(Host&)` — probe your
  services (required ones you assert; optional ones degrade). `on_frame(const Frame&)` —
  called every frame on the frame thread; submit ImGui windows, read
  worker-published state. `on_cleanup()` — cancel jobs, bounded-wait, release
  textures, return.
- **CMakeLists**: copy hello's verbatim for a UI-only applet, or SineScope's
  (`examples/sine_scope/`) for an ML applet, which adds `TORCH`. The only parts you
  edit: project and target name, source files, manifest filename.

### Smallest Complete Applet

This is everything — all three files. They live in this repository as
`examples/hello/`, which builds on its own against the installed SDK and inside
the core build as the fixture the loader tests load, and the listings
below are the actual files, embedded verbatim — they cannot drift from what
compiles. [Your first applet](../applets/first-applet.md) walks this same file line by
line:

```cpp title="examples/hello/hello.cpp"
--8<-- "examples/hello/hello.cpp"
```

```toml title="examples/hello/hello.caliper.toml"
--8<-- "examples/hello/hello.caliper.toml"
```

```cmake title="examples/hello/CMakeLists.txt"
--8<-- "examples/hello/CMakeLists.txt"
```

Note what is **absent**: no `main`, no window, no GL/Metal, no ImGui
context creation, no event loop, no `extern "C"`. The macro plus the host
supply all of it. Growing from here toward ML means adding a `jobs.v1`
training job and the bridge — that path is
[Your first ML applet](../applets/first-ml-applet.md).

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

`caliper new my-applet` writes a project for the applet `dev.example.my-applet`
with the stem `my_applet`: the manifest, the `CMakeLists.txt`, a source file with
one window, and a test that runs the applet on the fixture host. Pass a full id,
such as `com.example.my-applet`, to choose the id yourself.
[Caliper CLI](../reference/cli.md#caliper-new) describes the template.

### Applet Loading

`caliper check` runs the checks the host makes before it opens a library, and
`caliper dev` starts Caliper with the folder searched before every other applet
folder. The host reads its applet folders when it starts and keeps a library
loaded until it quits, so after each rebuild quit Caliper and run `caliper dev`
again. `--applets <dir>` on the command line, or `CALIPER_APPLET_PATH`, adds
development folders in the same way, and `caliper install build/applets` copies
the applet into the user applets folder to keep it installed.

```bash
# skip the landing screen on every restart:
CALIPER_AUTOLAUNCH=dev.example.my-applet "$CALIPER" dev build/applets
#    (CALIPER_EXIT_AFTER=<sec> exists too — clean-exit soak for CI;
#     CALIPER_WINDOW_SIZE=1440x900 fixes the window size for screenshots)
```

In CLion, open the applet's folder as a project of its own and add
`-DCMAKE_PREFIX_PATH=<sdk prefix>` to its CMake options.

**Testing without a window:** the fixture host
(`caliper::sdk_testing`, `<caliper/fixture_host.h>`) fakes `get_service`
with tables you inject — see `tests/test_sugar_services.cpp` for the
pattern. Applet logic that reads services can be unit-tested headlessly.

**Your data lives at** `~/Library/Application Support/Caliper/data/<your-id>/`
— the host creates it and hands it to you as `host.data_dir()`. Downloads,
caches, scratch files go there and nowhere else. [Folders and environment variables](../reference/folders-and-environment.md)
lists the folder on every platform; `CALIPER_DATA_DIR` moves the whole
app-data folder, and your data folder with it.

## Landing Screen

`./cmake-build-release/caliper` opens on the landing screen: a command bar along the top,
four docked panes, and a status bar along the bottom. It is drawn with ImGui,
so it looks the same on Metal, Vulkan and GL.

| Pane | Shows |
|---|---|
| **Projects** | Your own work: the dashboards of the open workspace, the spec applets you made with `F3 New dashboard` and saved, one row each with name, id and status, and under them the ways in. With no dashboard the header says `none yet`. `F2 Open folder` is disabled and says "not yet available" on hover. `F3 New dashboard` makes a new dashboard ([Builder pane](../builder/builder-pane.md)). `F4 Attach to process` lists the live sessions of training scripts running on this machine in the pane (see [Attaching to a training script](#training-script-attachment)); the pane header counts them, for example `1 dashboard · 1 live session`. |
| **Instruments** | Every installed applet the loader found, one row each: name, id, version, tag and status (`READY`, `ACTIVE`, `REFUSED`, `FAILED` or `QUARANTINED`). The workspace's dashboards are under Projects instead. Click a column header to sort by it; the default is tag, and the choice lasts for the session. Only `READY` rows launch. |
| **Inspector** | The selected applet's manifest: id, version, tag, ABI epoch, minimum host, required and optional services, binary path, status and summary. The `⏎ Launch` button sits at the bottom; when the applet cannot launch it is disabled, with the reason beside it. For an applet quarantined by a C++ exception, `⌘R Restart` takes the reason's place; after a signal it reads `restart Caliper to reload`. |
| **Activity** | Running jobs with their progress, then the host log, newest first: the renderer chosen, the applet folders scanned, launches, failed launches, failed jobs, quarantines, restarts, returns home, and live sessions attached, ended and detached, with any failed attach and its reason. |

The status bar shows the backend, the GPU, physical memory, the GPU
working-set budget, and how many applets are installed and unavailable.

The command bar filters the Instruments table and the dashboards under
Projects as you type: every space-separated term must appear, ignoring case,
in the applet's name, id or tag. When the filter hides the selected row, the
selection moves to the top match, so Enter always launches the highlighted
row. Each keyboard action also
has a button in the bar with its key printed on it:

| Key | Button | Action |
|---|---|---|
| ⌘K | `⌘K Command` | Focus the command bar. |
| ↑ / ↓ | `↑` `↓` | Move the selection through the dashboards under Projects and then Instruments, as one list, while either has focus or while you type in the command bar. The buttons are shown when the bar has room. |
| Esc | `esc Clear` | Clear the filter and move focus to Instruments. |
| Enter | `⏎ Launch` (Inspector) | Launch the selected applet. Double-clicking a row does the same. |
| ⌘1–⌘4 | the pane header | Focus Projects, Instruments, Inspector or Activity. Clicking a pane header does the same. |
| ⌘R | `⌘R Restart` (Inspector) | Unload and relaunch the selected applet. Offered only for an applet quarantined by a C++ exception. |
| F4 | `F4 Attach to process` (Projects) | List the live sessions and streams in Projects. In the list, ↑ / ↓ move the selection, Enter (`⏎ Attach`) attaches, and Esc (`esc Back`) returns to the ways in. |
| F5 | `F5 Live page` (command bar) | Open the live page of the attached session. Shown only while the app is attached. |
| F6 | `F6 Dashboard` (command bar) | Open the spec page of the dashboard the builder has open, beside its title: one left with Home, unsaved or not, or one an agent opened while an applet's page showed. Shown only while a dashboard is open. |

On Windows and Linux the keys are Ctrl K, Ctrl R and Ctrl 1–Ctrl 4, and the
buttons are labelled that way.

Launching an applet replaces the landing screen with the applet's page. The
bar at its top holds the `esc Home` button and the applet's name; the status
bar keeps the backend and GPU and adds the number of running jobs. Esc returns
to the landing screen only when no text field is being edited and no popup is
open; otherwise the field or popup gets Esc first. Home lands on the applet's
row, however it was launched (`CALIPER_AUTOLAUNCH` included); Home from a
dashboard lands on its row under Projects, also when it was just made and
saved.

### Training Script Attachment

A Python script that calls `caliper.show` (or any other view call) opens a
live session on its first call, and Caliper can attach to it while it runs
(macOS, Metal renderer; [Watch a PyTorch training run live](../introduction/python-quickstart.md)
sets one up). Press F4, or click `F4 Attach to process`, and Projects lists
the sessions found under `~/Library/Application Support/Caliper/live/`, and
the streams of `caliper.serve()` found under its `streams/` folder, newest
first, relisted every second:

| Column | Shows |
|---|---|
| **Script** | The script's name |
| **Pid** | Its process id |
| **Started** | When the session opened, as a clock time and how long ago |
| **State** | `running`, `starting` (attach waits for it), `closing`, `viewed by <pid>` (another viewer holds it; one viewer per session), `protocol v<N>` (another Caliper version), `stream` (a script that streams with `caliper.serve()`) or `attached` (the session this app is attached to; Enter opens its live page) |
| **Views** | The names of the views the script has shown so far |

Only `running`, `starting` and `stream` rows attach; the others but `attached` are faint, and the
`⏎ Attach` button under the table is disabled with the reason beside it. A
script that was killed leaves its session file behind; the next listing
removes it and its shared memory, so its row disappears instead of lingering.

Attaching opens the session's live page, laid out like an applet's page. Each
view is a window on the docked desktop, tiled on first appearance; move,
resize or dock them as on the applet page, and the layout is saved the same
way. The views show the script's newest GPU snapshots, and the page draws at
up to 60 fps whatever the display's refresh rate; the script never waits for
Caliper. The bar at the top holds `esc Home`, `⌘D Detach`, the script's name
and its state: a filled diamond and `live` while it runs. The status bar adds the
script's pid, the number of views, the frame rate and the step-to-screen
latency.

When the script exits, closes its session or is killed, the views keep their last
frames: the bar shows an outlined diamond and `SESSION ENDED`, the status bar
and the window title add `session ended`, and Activity logs `session ended`.
`esc Home` returns to the landing screen and keeps the session attached: its
views stay in the catalog as `live:` sources for a spec, and the command bar
shows the live diamond, the script's name and `F5 Live page`, which comes back
to the live page, as F5 does. `⌘D Detach` (⌘D) detaches: the script, if it
still runs, stops taking snapshots at once and keeps training, and the app
returns to the landing screen. Attaching another session detaches the
attached one first. Activity logs `attached` and `detached` with the script
and its pid.

`CALIPER_AUTOATTACH=<pid>` attaches to that process's session as soon as it is
listed, for screenshots and headless checks; `CALIPER_HOME_AFTER=<seconds>`
goes Home once, that long after the page opened (the applet page honours it
too).
[Attach to a running script](../python/script-attachment.md) covers
the session files, the standalone viewer and what happens when either side
goes away.

### Jobs and Window Layout

While a job runs or a failure is kept, the **Jobs** tray is a strip of its own
above the status bar, on every page. It takes its height from the area the
panes and the applet's windows fill, so it never covers them. On one line it
shows the running jobs, each with its progress and a `cancel` button, then the
failed jobs, each marked `FAILED` with the reason, then how many jobs finished
and `clear finished` at the right end. Jobs that do not fit are behind
`+N more`, which opens the whole list.

So that the panes do not jump, a job must run for half a second before the
strip appears, and the strip stays for two seconds after the last running job
ends, showing it as finished, so jobs that follow one another keep one strip.
A failed job keeps the strip, across Home, until `clear finished` is pressed,
and Activity logs it once as `job failed`; `clear finished` with nothing
running closes the strip at once.

The window layout is saved in `imgui.ini` in the app-data folder
(`~/Library/Application Support/Caliper/imgui.ini` on macOS): the dividers
between the landing panes, and where an applet's windows are docked. The
landing panes cannot be undocked or rearranged; only the dividers move. Delete
the file to return to the default layouts.

The window opens at 95% of the screen's work area.
`CALIPER_WINDOW_SIZE=<W>x<H>` (screen coordinates, e.g. `1440x900`) sets the
size exactly, so screenshots and layout checks are reproducible on any
display; a malformed value is ignored.

## Test Runs

The `release` and `debug` presets build the tests (`BUILD_TESTS=ON`):

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release            # prints the output of failing suites
```

Six suites need a GPU for some of their cases: `caliper_gfx_tests`,
`caliper_export_tests`, `caliper_embed_tests`, `caliper_torch_tests`,
`caliper_live_tests` and `caliper_live_views_tests`. A case
that finds no device skips and is counted, and the suite prints a summary when
it ends:

```
[caliper-tests] skipped 49 of 49 test cases: 49 no GPU, 0 not applicable on this platform -> exit 77 (ctest: Skipped)
```

A suite that skipped any case for want of a GPU, and had no failures, exits with
code 77. ctest reports it as `***Skipped`, not `Passed`, and lists it under "The
following tests did not run"; ctest's own exit status stays 0. Cases that can
never run on the platform, such as the CUDA cases on macOS, count as *not
applicable*: the summary reports them, but they leave the suite `Passed`. ctest
hides the output of suites that did not fail, so run it with `--verbose`, or run
the test binary directly, to see the summary line.

| Variable | Effect |
|---|---|
| `CALIPER_REQUIRE_GPU=1` | A skip for want of a GPU becomes a failure. Set it on a machine or CI lane that must have a GPU, so the device suites cannot pass without testing anything. Not-applicable skips never fail. |
| `CALIPER_TEST_FORCE_NO_GPU=1` | Test-only. The device suites behave as if there were no GPU, so the skip path can be checked on a machine that has one. |

When you add a device test, skip through `tests/gpu_skip.h`:
`CALIPER_SKIP_NO_GPU(why)` counts the skip, says why and returns;
`CALIPER_SKIP_NOT_APPLICABLE(why)` is for a case that never applies on the
platform; `CALIPER_NOTE_NO_GPU(why)` only counts, for a case that must tidy up
before it returns. A suite that uses them ends with `CALIPER_GPU_TEST_MAIN()` in
place of `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`. A bare `MESSAGE` and `return`
would report the suite as passed.

## Applets in This Repository

`caliper_add_applet()` is the same file in the core build and in the SDK package. The core build compiles no applets unless `CALIPER_BUILD_APPLETS=ON`. `examples/hello` builds with the tests as the loader-test fixture. `examples/sine_scope` builds with the tests in a build with libtorch.

Each applet folder under `applets/` and `examples/` also builds alone against the installed SDK. `scripts/build_applets_against_sdk.sh <core build folder> <work folder> [applet folder...]` installs the SDK and builds them that way, as CI does.

An older build folder can keep libraries from in-tree builds that predate `caliper_add_applet()`. Their manifests have no `[build]` table, so the host does not check them against the UI-stack hash. Delete the `applets/` folder of the build after you turn `CALIPER_BUILD_APPLETS` off, and when you reuse such a build folder.

## Rules

- **No raw GL/Metal/Vulkan calls in applets** ([Platform rules](../introduction/architecture.md#design-decisions)). Draw through
  ImGui/ImPlot/ImPlot3D; get tensors on screen through `tensor_bridge.v1`.
  This is what makes your applet run identically on both backends.
- **Bridge and data.v1 calls: frame thread only.** Torch ops: worker thread
  only.
- **Honor cancel within 100 ms** in every job loop (check per batch,
  per download chunk).
- **Don't catch your way across the ABI** — the sugar's exception walls
  already guarantee nothing throws into the host; inside your own code,
  handle errors normally.
- **If a service is optional, degrade visibly** — a disabled panel saying
  "metrics: absent (ok)" beats silent absence.

## Reading Order

1. This page — the lay of the land.
2. [Your first applet](../applets/first-applet.md) — walk hello line by line.
3. [Your first ML applet](../applets/first-ml-applet.md) — the staged climb: compute,
   then torch, then live plots, then the bridge — one capability per stage.
4. Per-service [reference pages](../applets/services/caliper-jobs-v1.md) — the
   contracts, when you need exact semantics.
