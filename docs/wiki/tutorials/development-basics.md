# Development basics

What building on Caliper actually looks like: what code you write, what code
is already running, where every library comes from, and the edit-build-run
loop. Read this before [Your first applet](first-applet.md); read the
[ML applet cookbook](../howto/ml-applet-cookbook.md) after it.

## The mental model

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

## Where every library comes from

The repository and CMake build wire the project dependencies together. Platform
toolchains and system prerequisites must still be installed; see the root
`README.md` before configuring a build.

| Library | Comes from | You link it via | Notes |
|---|---|---|---|
| SDK headers (ABI, services, sugar, adapters) | `sdk/include/` in this repo | `caliper::sdk` | Header-only INTERFACE target — nothing compiles into the SDK itself |
| ImGui (docking) + ImPlot + ImPlot3D + FileDialog | `third_party/` submodules, **pinned** | `caliper::ui_stack` | Compiled once in-tree. **Never bring your own copy** — the pin is part of the ABI (§9 of PLATFORM.md): your applet and the host must agree on ImGui's memory layout byte-for-byte |
| libtorch | `third_party/libtorch/` (vendored) | `"${TORCH_LIBRARIES}"` + the rpath block | The current root executable links the configured Torch libraries, and ML applets may link them as well. |
| curl, zlib | macOS system SDK | `CURL::libcurl`, `ZLIB::ZLIB` | Only if you download/decompress data. Data acquisition is applet business — the host has no downloader |
| DuckDB | host-internal | **you can't** | Deliberately unreachable. You get its powers through services: `metrics.v1`, `artifacts.v1`, `data.v1`. No DuckDB type ever crosses the ABI |

The division is deliberate: what must be *shared* (UI stack) is pinned and
provided; what is *yours* (torch, curl) you link privately; what is
*persistent* (DuckDB) hides behind frozen service tables.

## Anatomy of an applet

Three files, one folder. This is the entire footprint:

```
examples/hello/
├── hello.caliper.toml    the manifest — checked BEFORE your code loads
├── hello.cpp             your applet class + the CALIPER_APPLET macro
└── CMakeLists.txt        ~12 lines (hello) to ~40 (with torch)
```

- **Manifest**: id (reverse-DNS), name, version, summary, required/optional
  service lists. The `id` and `version` must be **byte-identical** between
  the manifest and the descriptor in plugin.cpp, or the loader refuses (the
  applet is listed with the reason, not a crash). See [manifest reference](../reference/manifest.md).
- **The macro**: the `CALIPER_APPLET(...)` at the bottom of your `.cpp`
  generates the epoch-2 C ABI glue — the descriptor plus exception walls on
  every entry point. You never write `extern "C"` yourself.
- **The class**: three lifecycle methods. `initialize(Host&)` — probe your
  services (required ones you assert; optional ones degrade). `draw_ui()` —
  called every frame on the frame thread; submit ImGui windows, read
  worker-published state. `cleanup()` — cancel jobs, bounded-wait, release
  textures, return.
- **CMakeLists**: copy hello's verbatim for a UI-only applet; copy the
  exemplar's for an ML applet. The only parts you edit: target name, source
  files, manifest filename.

### The smallest complete applet, in full

This is everything — all three files. It is **built in this repo** as
`examples/hello/` (the Hello row in the landing screen's Instruments
pane), and the listings
below are the actual files, embedded verbatim — they cannot drift from what
compiles. [Your first applet](first-applet.md) walks this same file line by
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
training job and the bridge — that path is the
[cookbook](../howto/ml-applet-cookbook.md), and its finished form is the
exemplar.

## The development loop

```bash
# 1. create — the applets/* glob auto-discovers it (CONFIGURE_DEPENDS);
#    no root CMake edits, just build:
mkdir applets/my_applet   # + the three files (copy examples/hello/ to start)

# 2. build — your dylib + manifest land in build/applets/
cmake --build build --target my_applet -j

# 3. run — it appears as a row in the landing screen's Instruments pane
./build/caliper

# 4. iterate fast — skip the landing screen every rebuild:
CALIPER_AUTOLAUNCH=dev.example.my-applet ./build/caliper
#    (CALIPER_EXIT_AFTER=<sec> exists too — clean-exit soak for CI;
#     CALIPER_WINDOW_SIZE=1440x900 fixes the window size for screenshots)
```

!!! warning "`--clean-first` on a single applet target prunes its siblings"
    The applet list is a `CONFIGURE_DEPENDS` glob, so **all** applets are
    configured but only the target you name gets built. A partial
    `cmake --build build --target my_applet --clean-first` cleans the whole
    build tree first, then rebuilds only `my_applet` — leaving the other
    applets' dylibs and manifests missing from `build/applets/` until a full
    `cmake --build build` restores them. If the Instruments pane suddenly
    lists only one applet, this is why: rebuild without `--clean-first`, or
    build the default target to bring every applet back.

In CLion the same works via the CMake tool window — reload the project once
after creating the folder, then the `my_applet` target exists.

**Testing without a window:** the fixture host
(`caliper::sdk_testing`, `<caliper/fixture_host.h>`) fakes `get_service`
with tables you inject — see `tests/test_sugar_services.cpp` for the
pattern. Applet logic that reads services can be unit-tested headlessly.

**Your data lives at** `~/Library/Application Support/Caliper/data/<your-id>/`
— the host creates it and hands it to you as `host.data_dir()`. Downloads,
caches, scratch files go there and nowhere else.

## The landing screen

`./build/caliper` opens on the landing screen: a command bar along the top,
four docked panes, and a status bar along the bottom. It is drawn with ImGui,
so it looks the same on Metal, Vulkan and GL.

| Pane | Shows |
|---|---|
| **Projects** | No projects yet. It lists two ways in. `F2 Open folder` is disabled and says "not yet available" on hover. `F4 Attach to process` lists the live sessions of training scripts running on this machine in the pane (see [Attaching to a training script](#attaching-to-a-training-script)); the pane header counts them, for example `none yet · 1 live session`. |
| **Instruments** | Every applet the loader found, one row each: name, id, version, tag and status (`READY`, `ACTIVE`, `REFUSED`, `FAILED` or `QUARANTINED`). Click a column header to sort by it; the default is tag, and the choice lasts for the session. Only `READY` rows launch. |
| **Inspector** | The selected applet's manifest: id, version, tag, ABI epoch, minimum host, required and optional services, binary path, status and summary. The `⏎ Launch` button sits at the bottom; when the applet cannot launch it is disabled, with the reason beside it. For an applet quarantined by a C++ exception, `⌘R Restart` takes the reason's place; after a signal it reads `restart Caliper to reload`. |
| **Activity** | Running jobs with their progress, then the host log, newest first: the renderer chosen, the applet folders scanned, launches, failed launches, failed jobs, quarantines, restarts, returns home, and live sessions attached, ended and detached, with any failed attach and its reason. |

The status bar shows the backend, the GPU, physical memory, the GPU
working-set budget, and how many applets are installed and unavailable.

The command bar filters the Instruments table as you type: every
space-separated term must appear, ignoring case, in the applet's name, id or
tag. When the filter hides the selected row, the selection moves to the top
match, so Enter always launches the highlighted row. Each keyboard action also
has a button in the bar with its key printed on it:

| Key | Button | Action |
|---|---|---|
| ⌘K | `⌘K Command` | Focus the command bar. |
| ↑ / ↓ | `↑` `↓` | Move the selection in Instruments, while Instruments has focus or while you type in the command bar. The buttons are shown when the bar has room. |
| Esc | `esc Clear` | Clear the filter and move focus to Instruments. |
| Enter | `⏎ Launch` (Inspector) | Launch the selected applet. Double-clicking a row does the same. |
| ⌘1–⌘4 | the pane header | Focus Projects, Instruments, Inspector or Activity. Clicking a pane header does the same. |
| ⌘R | `⌘R Restart` (Inspector) | Unload and relaunch the selected applet. Offered only for an applet quarantined by a C++ exception. |
| F4 | `F4 Attach to process` (Projects) | List the live sessions in Projects. In the list, ↑ / ↓ move the selection, Enter (`⏎ Attach`) attaches, and Esc (`esc Back`) returns to the ways in. |

On Windows and Linux the keys are Ctrl K, Ctrl R and Ctrl 1–Ctrl 4, and the
buttons are labelled that way.

Launching an applet replaces the landing screen with the applet's page. The
bar at its top holds the `esc Home` button and the applet's name; the status
bar keeps the backend and GPU and adds the number of running jobs. Esc returns
to the landing screen only when no text field is being edited and no popup is
open; otherwise the field or popup gets Esc first. Home lands on the applet's
row, however it was launched (`CALIPER_AUTOLAUNCH` included).

### Attaching to a training script

A Python script that calls `caliper.show` (or any other view call) opens a
live session on its first call, and Caliper can attach to it while it runs
(macOS, Metal renderer; [Watch a PyTorch training run live](live-pytorch.md)
sets one up). Press F4, or click `F4 Attach to process`, and Projects lists
the sessions found under `~/Library/Application Support/Caliper/live/`, newest
first, relisted every second:

| Column | Shows |
|---|---|
| **Script** | The script's name |
| **Pid** | Its process id |
| **Started** | When the session opened, as a clock time and how long ago |
| **State** | `running`, `starting` (attach waits for it), `closing`, `viewed by <pid>` (another viewer holds it; one viewer per session) or `protocol v<N>` (another Caliper version) |
| **Views** | The names of the views the script has shown so far |

Only `running` and `starting` rows attach; the others are faint, and the
`⏎ Attach` button under the table is disabled with the reason beside it. A
script that was killed leaves its session file behind; the next listing
removes it and its shared memory, so its row disappears instead of lingering.

Attaching opens the session's live page, laid out like an applet's page. Each
view is a window on the docked desktop, tiled on first appearance; move,
resize or dock them as on the applet page, and the layout is saved the same
way. The views show the script's newest GPU snapshots, and the page draws at
up to 60 fps whatever the display's refresh rate; the script never waits for
Caliper. The bar at the top holds `esc Home`, the script's name and
its state: a filled diamond and `live` while it runs. The status bar adds the
script's pid, the number of views, the frame rate and the step-to-screen
latency.

When the script exits, closes its session or is killed, the views keep their last
frames: the bar shows an outlined diamond and `SESSION ENDED`, the status bar
and the window title add `session ended`, and Activity logs `session ended`.
`esc Home` detaches (the script, if it still runs, stops taking snapshots at
once and keeps training) and returns to the session list. Activity logs
`attached` and `detached` with the script and its pid.

`CALIPER_AUTOATTACH=<pid>` attaches to that process's session as soon as it is
listed, for screenshots and headless checks; `CALIPER_HOME_AFTER=<seconds>`
goes Home once, that long after the page opened (the applet page honours it
too).
[Attach to a running script](../howto/attach-to-a-running-script.md) covers
the session files, the standalone viewer and what happens when either side
goes away.

### Jobs and window layout

While any job exists, a **Jobs** tray sits at the bottom right of every page:
running jobs with their progress and a `cancel` button, then failed jobs, marked
`FAILED` with the reason, then finished ones. A failed job stays in the tray,
across Home, until `clear finished` is pressed, and Activity logs it once as
`job failed`.

The window layout is saved in `imgui.ini` in the app-data folder
(`~/Library/Application Support/Caliper/imgui.ini` on macOS): the dividers
between the landing panes, and where an applet's windows are docked. The
landing panes cannot be undocked or rearranged; only the dividers move. Delete
the file to return to the default layouts.

The window opens at 95% of the screen's work area.
`CALIPER_WINDOW_SIZE=<W>x<H>` (screen coordinates, e.g. `1440x900`) sets the
size exactly, so screenshots and layout checks are reproducible on any
display; a malformed value is ignored.

## Running the tests

The `release` and `debug` presets build the tests (`BUILD_TESTS=ON`):

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release            # prints the output of failing suites
```

Four suites need a GPU for some of their cases: `caliper_gfx_tests`,
`caliper_export_tests`, `caliper_embed_tests` and `caliper_torch_tests`. A case
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

## The rules (each one guards something real)

- **No raw GL/Metal/Vulkan calls in applets** (§6c). Draw through
  ImGui/ImPlot/ImPlot3D; get tensors on screen through `tensor_bridge.v1`.
  This is what makes your applet run identically on both backends.
- **Bridge and data.v1 calls: frame thread only.** Torch ops: worker thread
  only. The [cookbook](../howto/ml-applet-cookbook.md) §1 has the full spine.
- **Honor cancel within 100 ms** in every job loop (check per batch,
  per download chunk).
- **Don't catch your way across the ABI** — the sugar's exception walls
  already guarantee nothing throws into the host; inside your own code,
  handle errors normally.
- **If a service is optional, degrade visibly** — a disabled panel saying
  "metrics: absent (ok)" beats silent absence. Every optional feature of the
  exemplar shows the idiom.

## Reading order

1. This page — the lay of the land.
2. [Your first applet](first-applet.md) — walk hello line by line.
3. [Your first ML applet](first-ml-applet.md) — the staged climb: compute,
   then torch, then live plots, then the bridge — one capability per stage.
4. `applets/embed_scope/` — the exemplar; every framework capability in one
   annotated file, structured to be copied.
5. [ML applet cookbook](../howto/ml-applet-cookbook.md) — the composition
   idioms (threading, cadences, the device-resident pull, viewport policy).
6. Per-service [reference pages](../reference/services/jobs-v1.md) — the
   contracts, when you need exact semantics.
