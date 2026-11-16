# Windows parity

Caliper is developed on macOS first. This page lists the work that brings Windows level with macOS once the current macOS sprints are finished: the applet platform, Caliper LIB's streaming and notebook widgets, the web viewer, Caliper API, Caliper Embed, the renderer core and its fallbacks, agent-built panels and loaders, and scenes and recordings. The [roadmap](roadmap.md) holds the order and status. Linux is not planned for now.

On Windows the fast path is Vulkan with CUDA interop, where macOS uses Metal with MPS. The toolchain contract is MSVC with the dynamic CRT (`/MD`), as Compatibility and epochs (`docs/explanations/compatibility.md`) describes.

## Where Windows stands

- **CI builds and tests the core on `windows-latest`.** The Windows lane of the Build workflow runs the same steps as the macOS lane: the core build without applets, the whole ctest suite with the Caliper CLI tests and the Caliper SDK and Caliper Embed package tests, the builds of `hello`, `sine_scope` and the scopes against the installed SDK, and the check that the host names no applet. It skips `embed_scope` and `gpt_scope`, which need the curl and zlib that only the core build provides on Windows. The runner has no GPU, so the GPU rows skip there. Several of these pieces have only run on macOS so far; Watch the first Windows CI run (`docs/guides/verify-on-windows.md`) lists what to look for.
- **The Vulkan backend, the CUDA interop paths and the gfx suite have been verified on Windows hardware before.** Verify on Windows hardware (`docs/guides/verify-on-windows.md`) describes the procedure.
- **The code already handles several Windows details.** The loader opens an applet with `LoadLibraryA` and looks for `<stem>.dll`, with or without the `lib` prefix. `CALIPER_APPLET_PATH` uses `;` as its separator, and the user applets folder is `%APPDATA%\Caliper\applets\`. The lock around the data-schema stamp uses `LockFileEx`. `caliper shot` draws into the Vulkan offscreen canvas and reads the frame back, with no window. The `caliper-embed` package installs `bin\caliper_embed.dll`, `lib\caliper_embed.lib`, the headers and the CMake package, without the `caliper_live_*` calls, which exist on macOS only, and `embed_host` copies the DLL beside itself.
- **No one has yet run on Windows hardware** applets loaded from outside the repository, the Caliper CLI subcommands, `caliper shot` or the Caliper Embed package.
- **The network stream, the hub, Caliper API, `caliper build`, the web viewer and the notebook widget are built and tested on macOS.** The first Windows CI run compiled the WebSocket layer, the hub, the live controls and the typed live views with MSVC, with the rest of `caliper_host_lib` and `libcaliper`. It stopped at `src/main.cpp`, which included the macOS-only shared-memory header `src/host/live/live_shm.h` without using it; that include is gone. Caliper API's server, the executable and the tests have not been compiled with MSVC yet. Before a push, `scripts/check_windows_includes.py` preprocesses on macOS every first-party translation unit that the Windows lane compiles, as clang does for MSVC and with no macOS header reachable, and fails on an include of a macOS-only or POSIX-only header and on identifiers and extensions that MSVC lacks. It cannot see type errors, overloads, link errors or anything the standard library or the Windows headers decide, so the Windows lane stays the check for those. The code was also checked by reading, by a syntax check of its Windows branches against the documented Winsock declarations, and by a check that the code the Windows lane compiles never turns a `std::filesystem::path` into a `std::string` implicitly. docs/guides/verify-on-windows.md, section "Watch the first Windows CI run", lists what to look for. On Windows they stand as follows.
    - **Compiled, not tested.** The WebSocket layer and the hub (`src/host/net/`) are portable C++ on Winsock and compiled in the Windows lane; the Caliper API server (`src/host/api/`) is portable in the same way and had not been reached when the first run stopped. They are untested on Windows: their socket tests (`test_net_ws.cpp`, `test_net_hub.cpp`, `test_api.cpp`) are left out of the Windows build of `caliper_tests`, and the `caliper_net_tests` entry reports them as skipped. Every feature built on them is refused on Windows. The codec of wire protocol v3 and the JSON reader, which the live controls use, are tested there through their `net/wire` cases, as are the unit cases of the command line, `caliper build`'s searches among them.
    - **Refused.** The executable refuses `caliper serve`, `caliper build`, `--api` and `--url` before anything starts: it prints `... is not available on Windows yet` and exits with 1 (`windows_unavailable` in `src/host/host_cli.h`), and it ignores `CALIPER_API`. The `caliper_cli_windows_*` tests check this in the lane, and docs/references/cli.md, section "On Windows", describes it.
    - **macOS only.** The app's network source, the live attach and the shared-memory sessions. On Windows `src/host/live/live_unsupported.cpp` stands in for them, and every attach, `caliper_core_live_attach_url` included, fails with a reason. The live controls and the typed live views compile on Windows.
    - **Not run on Windows.** The process-level tests of `caliper serve`, `caliper shot --api`, the hidden app, its run bar and `caliper build`, which are registered on macOS only; the network transport of Caliper LIB, `caliper.serve()` and the notebook widget; and the web viewer in a browser on Windows.
- **Agent-built panels, loaders and the command layer are built on macOS first,** on the sprint 3 branch, as libraries under `src/builder/` that no Windows build compiles yet. Starting a child process, the loopback HTTP client and the key store are stubs on Windows that refuse with a clear error.
- **The renderer core, scenes and recordings are designed and not built** ([Renderer core](renderer-core.md), [Scenes](scenes.md), [Recordings](recordings.md)). The Vulkan backend of the graphics layer is done with this page's steps, not with the macOS work. Because the user decided that the migration keeps no old path, sprint 4 reduces the Vulkan renderer to the window, the frame loop, the embed canvas and ImGui when the tensor bridge moves onto the layer. From then until step 7 of this page, Windows refuses bridge textures, geometry views, the applet scenes of the scene service that replaces the geometry service, PNG exports and embedding views, and draws them through Dawn with a copy once Dawn lands. The user has accepted that gap: Windows and Vulkan are deferred until the user moves to the Windows machine, so no Windows work is done before then.

## How the sprints add to this page

Each macOS sprint adds what Windows then needs. Under each step below, a table lists those items with the sprint that added them, what each depends on and an estimated size in days of one builder. Sprint 2 is streaming, the hub, Caliper API and the web viewer; sprint 3 is agent-built panels, loaders and the command layer; sprint 4 is the renderer core, scenes and recordings. The sizes are estimates made from reading the branches, and an item marked "to check" rests on reading the code, not on a run.

## The steps

Each step ends with a check that must pass on a Windows machine with an NVIDIA GPU, not only in CI.

### 1. Build and test baseline

- Run the full CI on `windows-latest` for every new package and subcommand. A test that skips on a GPU-less runner must also run on real hardware before the step counts as done.
- Run the embed suite on Windows hardware with zero skips. The gfx and export suites draw on Windows again only with the Vulkan backend of step 7, and their run with zero skips belongs to that step.
- **Done when** CI is green and the hardware run shows no skipped GPU rows.

| Item | Added by | Depends on | Builder days |
|---|---|---|---|
| Register on Windows the tests that CMake leaves out there today: `caliper_cli_serve`, `caliper_cli_api`, `caliper_cli_app_hidden`, `caliper_cli_build`, and the live and live network suites. The process-level ones need a way to start and stop a child without `waitpid` and `SIGTERM` | Sprint 2 | The items of steps 2 to 4 that make each one pass | 2 |
| Build and test the libraries under `src/builder/` in the Windows lane. No root build or CI job compiles them on any platform yet | Sprint 3 | Sprint 3's libraries wired into the host build | 2 |
| Add a web job on a Windows runner; none exists | Sprint 2 | Step 5 | 1 |

### 2. The applet platform

- **Loading.** The loader calls `LoadLibraryA` with the applet's full path. Windows then resolves the applet's own dependencies from the executable's folder, where the host carries libtorch's DLLs, but not from the applet's folder. Load with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`, or add each applet's folder with `AddDllDirectory`, so that DLLs an applet ships beside itself are found without editing `PATH`.
- **Search paths.** The user applets folder and the `;` separator are in place. Handle long paths and paths with spaces.
- **The libtorch pin.** Today the pin is the libtorch version alone, such as `2.5.1`, and the CUDA variant comes from the toolkit the core build found. The SDK should also name the host's CUDA libtorch build (for example `cu124`), and the loader should refuse a mismatch with a clear refusal message.
- **Installing and replacing applets.** Windows locks a loaded DLL. `caliper install` already copies each file to a temporary name and renames it over the destination, but the rename fails while the running app has the applet loaded, and `install` then reports only that it could not copy. It must say, clearly, that the running app has the applet loaded.
- **The CLI and the app's console.** `caliper.exe` is a console program today, so the subcommands print to the terminal that started them, and starting the app from Explorer also opens a console window. Make the app a GUI-subsystem program and ship a console entry point for the subcommands beside it, such as a small `caliper.com`, so `caliper list` works in PowerShell and `cmd`, as the `code` command does for Visual Studio Code.
- **`caliper shot`.** It is kept as it is: it draws into the Vulkan offscreen canvas and reads the frame back, and it has not been run on Windows hardware yet. Compare its PNG with a macOS shot of the same applet.
- **Dock roles** need nothing Windows-specific, but check them in a `shot` capture.
- **`caliper build`** runs CMake through `posix_spawn` and reads its output through a pipe. Give it a Windows process runner (`CreateProcess` with redirected output), then lift its refusal in `windows_unavailable` and run `tests/cli_build.cmake` on Windows.
- **Done when** a scope built outside the repository against the installed Caliper SDK loads through `caliper dev`, passes `caliper check`, and appears correctly in a `caliper shot` capture on Windows.

| Item | Added by | Depends on | Builder days |
|---|---|---|---|
| `caliper build` on Windows. It refuses today, because it starts CMake with `posix_spawn` | Sprint 2 | The Windows process helper of step 9 | 2 |
| The hidden app mode, `caliper --hidden`. It is untested on Windows: check that the window, its taskbar button and focus stay hidden, and register its test | Sprint 2 | Step 4 for the API it is tested through | 1 |

### 3. Caliper LIB

- **Wheels.** Build Windows wheels with MSVC and test them against current CUDA builds of PyTorch. The extensions link no libtorch, as on macOS, so they share a process with the torch the script imported.
- **The local session.** Today it uses POSIX shared memory and MPS. On Windows it needs named shared memory (`CreateFileMapping`) and CUDA tensors, with the snapshot copied on the device and never blocking training, matching the MPS design.
- **The network transport** is portable and should work first. Verify it, and measure the training overhead on CUDA as the benchmark does on MPS.
- **The in-process window** on Vulkan.
- **Done when** the example notebook runs on Windows against CUDA, and both the app and a browser show the live views.

| Item | Added by | Depends on | Builder days |
|---|---|---|---|
| The package's native modules in C++ for Windows. Both are Objective-C++ today, so no wheel can be built; the package metadata names macOS only and no wheel job exists | Before the sprints, extended by sprint 2 | Step 1 | 5 |
| `caliper.serve()` on Windows. The server is pure Python, so it should work once the package imports without the macOS modules; verify it with CUDA tensors | Sprint 2 | The item above | 1 |
| The native sender with zero-copy on CUDA: snapshots written on the device into exportable CUDA memory, which the app imports through Vulkan external memory, never blocking training | Sprint 2 for the sender, sprint 4 for import | The Vulkan backend of step 7; named shared memory | 8 |
| The scene layout of the shared-memory session (a directory segment and arena segments) on named shared memory | Sprint 4 | The local session of this step | 3 |
| The benchmarks on CUDA. `bench_live.py` and `bench_show.py` use MPS today, and the renderer core's acceptance thresholds need them on Windows | Sprint 2 and sprint 4 | The items above | 2 |
| The staging script of the Python build, a POSIX shell script, in a form that runs on Windows | Sprint 2 | None | 0.5 |
| Arrays from JAX and CuPy through DLPack, for scripts that do not use torch. Both are built for NVIDIA GPUs, so they come here rather than on macOS | Sprint 5 | The native modules item above | 3 |

### 4. Streaming, the hub and Caliper API

- The app's network source, the `caliper serve` hub and Caliper API v0 on Winsock. Build and run their socket tests on Windows first (`caliper_net_tests` stands in for them in the lane today). Port the network source, which is Objective-C++ for macOS today, and lift each refusal in `windows_unavailable` once its process-level test runs on Windows; those tests use POSIX processes and signals (`tests/test_proc.h`), so they need a Windows runner first.
- Bind to 127.0.0.1 only, so Windows Defender Firewall never prompts.
- Keep the API token file in `%APPDATA%\Caliper\`, readable only by the user.
- **Done when** a CUDA script streams to the app, to a browser and through the hub on one Windows machine, and every API v0 endpoint passes its tests.

| Item | Added by | Depends on | Builder days |
|---|---|---|---|
| The hub and API transport on Winsock. The socket layer has a Winsock branch whose own comment says it had not been compiled when written; run its tests and the hub on hardware | Sprint 2 | Step 1 | 1 |
| Owner-only access to the hub file, the API file and the token file. On Windows they are written with default access and read without a check, where macOS requires mode 600 | Sprint 2 | None | 1 |
| Stopping `caliper serve` cleanly. Only Ctrl-C reaches it on Windows; add a console control handler and the process-level test | Sprint 2 | None | 1 |
| The app's network source. It is refused on Windows today, because its slots are Metal buffers | Sprint 2 | The graphics layer's import and copy paths ([Renderer core](renderer-core.md)); the Vulkan backend of step 7, or the Dawn fallback with a reported copy | 4 |
| `GET /v0/capture.png` in the windowed app on Vulkan. It answers 501 today, since only the Metal renderer captures its window | Sprint 2 | Window capture in the Vulkan renderer, which keeps the window when sprint 4 reduces it | 2 |
| Recording in the hub, `caliper serve --record` and `--recordings`: appending, reading a file that is still growing (file sharing modes), recovery after a crash, and offsets past 4 GiB | Sprint 4 | The recording file library | 2 |

### 5. The web viewer and notebooks

- The browser side is platform-neutral, but check it in Chrome and Edge on Windows, where WebGPU runs on Direct3D 12.
- Check the notebook widget in JupyterLab and in Visual Studio Code on Windows.
- **Done when** the web viewer's headless checks pass on a Windows runner and the widget draws in both notebook front ends.
- Scenes reach a browser on Windows before they reach the native Windows app: the first increment of [Sprint 4](sprint-4.md), a loss-landscape demo in a notebook, draws in Chrome and Edge on Direct3D 12 through `caliper.serve()` as soon as Caliper LIB's network route runs on Windows.

| Item | Added by | Depends on | Builder days |
|---|---|---|---|
| The WebAssembly build of the web viewer on Windows. Its build script is a bash script with no Windows counterpart | Sprint 2 | None | 1 |
| A Direct3D project in the browser tests. The hardware project asks for Metal, and only the SwiftShader and Canvas 2D projects are portable | Sprint 2 | None | 1 |
| The scene parity tests of [Web geometry](web-geometry.md) in Chrome and Edge on Direct3D 12, on a default and on a compatibility device, and the loss-landscape demo of [Demos](demos.md) in a notebook there | Sprint 4 | Web geometry built; `caliper.serve()` on Windows, step 3 | 1 |

### 6. Caliper Embed

- Run the installed `caliper-embed` package on Windows hardware: the DLL, its import library, the headers and the CMake package, which the Windows lane installs and builds `embed_host` against.
- Run the Win32 `embed_host` against the installed package, in a window and in its offscreen mode.
- Add hosting examples for WPF (`HwndHost`), WinForms (a control's `Handle`) and Qt (`winId()`).
- **Decide on WinUI 3.** It expects a DXGI swap chain rather than a child window, so it needs Vulkan–DXGI interop or the Dawn path.
- **Done when** the installed package hosts an applet in a Win32 window and in a WPF window, and renders offscreen.

### 7. Renderer fallbacks

- The WebGPU backend through Dawn as the native fallback on Windows: Direct3D 12 and Direct3D 11 for machines without working Vulkan, and SwiftShader for machines without a usable GPU.
- The GLEW OpenGL fallback is already gone: sprint 4 retires it before Dawn, as the user decided, so until Dawn is verified here a Windows machine without working Vulkan refuses to start, with a message that says so.
- **Done when** the app runs on a machine with Vulkan disabled and says which fallback it is using.

| Item | Added by | Depends on | Builder days |
|---|---|---|---|
| The Vulkan backend of the graphics layer, with the CUDA direct path, the buffer path and the semaphores as private paths, and the texture path, the scene service and the points view on it. It was moved out of the macOS work to this page. Its code comes from the repository's history, because sprint 4 deletes the Vulkan renderer's texture, import, CUDA direct and buffer paths and their GLSL when the tensor bridge moves onto the layer, and the gfx/Vulkan and Vulkan+CUDA rows return with it, written against the layer. The Vulkan renderer's geometry paths do not come back, because sprint 4 deletes the geometry service | Sprint 4 | The layer and its Metal steps ([Renderer core](renderer-core.md), step 17 there); Windows hardware with CUDA | 8 |
| The single-version graphics services on Windows: the tensor bridge with CUDA import (`CALIPER_ALLOC_HANDLE_OPAQUE_WIN32`) through the layer's import, the scene service and export drawing applet scenes from imported CUDA blocks in place, the Windows refusals of sprint 4 removed, and the scopes, rewritten onto the scene service in sprint 4 and built against the SDK of epoch 3, run on the Vulkan backend with their captures compared with the macOS ones | Sprint 4 | The Vulkan backend | 1.5 |
| The Dawn fallback on Direct3D 12 and Direct3D 11, and on SwiftShader, verified on Windows | Sprint 4 | The Dawn step of the renderer core, built on macOS first | 2 |
| Scenes, bridge textures and embedding views on Windows before the Vulkan backend lands: they draw through Dawn, with a copy that the view's info line reports | Sprint 4 | The item above | 1 |
| The zero-copy measurement on Windows: the timing rows, `CALIPER_VULKAN_STATS` and the benchmarks within the renderer core's thresholds, comparing the Vulkan step's build with a build of the last commit that still had the Vulkan renderer's paths, run interleaved on one machine | Sprint 4 | The Vulkan backend; the CUDA benchmarks of step 3 | 2 |
| The CUDA direct kernel written again from the colormap's shared C++ header, so that it decides by integer keys, with its PTX generated and the boundary rows run on Vulkan with CUDA. Sprint 4 deletes the old kernel with the Vulkan renderer's paths | Sprint 4 | The colormap step of the renderer core; the Vulkan backend | 1 |

The Windows build loses its build-time glslang in sprint 4, together with the Vulkan renderer's GLSL shaders, so this step no longer carries that item. The Vulkan backend takes the SPIR-V that Naga generates from the WGSL sources, which is committed.

### 8. Signing, packaging and distribution

- Authenticode-sign the app, the CLI entry point and the first-party applet DLLs.
- Plan for SmartScreen reputation.
- **Decide the signing policy for third-party applet DLLs.** It is the Windows side of the open signing question on macOS.
- Package with an installer (MSIX or MSI) and publish through winget, putting the CLI on `PATH`.
- **Done when** a clean Windows machine installs Caliper from the package, runs `caliper info` in a terminal, and opens an applet without a SmartScreen block.

| Item | Added by | Depends on | Builder days |
|---|---|---|---|
| Sign and package `caliper-mcp` and the agent sidecar with the app | Sprint 3 | Step 9 | 0.5 |
| Offer the app as an opener of `.mcap` files, so that a recording can be opened from Explorer, without taking the extension from another program that already opens MCAP files | Sprint 4 | Step 10 | 0.5 |

### 9. Agent-built panels and loaders

Sprint 3 builds its libraries and tools under `src/builder/` on macOS, and guards what cannot build on Windows behind stubs that refuse with a clear error. This step replaces the stubs.

| Item | Added by | Depends on | Builder days |
|---|---|---|---|
| A Windows implementation of the process helper `Process` and `run_process`: `CreateProcess`, pipes, ending a whole child tree, and a search of `PATH` that honours `PATHEXT`, so that `npx.cmd` is found. Today it answers `process.unsupported` | Sprint 3 | None. It unblocks the agents, the Python loader and `caliper build` | 4 |
| The loopback HTTP client on Winsock. Today it answers `http.unsupported`, so every `caliper-mcp` tool that reaches the app fails | Sprint 3 | The socket layer of step 4 | 2 |
| `caliper-mcp` built and tested on Windows. Its stdio server already switches to binary mode there | Sprint 3 | The two items above | 1 |
| The agents over ACP on Windows: starting the adapters through `npx.cmd`, and the harness and client tests | Sprint 3 | The process helper | 2 |
| The key store on the Windows Credential Manager, in place of the Keychain. Today it answers `keystore.unsupported` | Sprint 3 | None | 2 |
| The Python loader on Windows: the interpreter's name, the `Scripts` folder of a virtual environment, and its tests | Sprint 3 | The process helper | 1 |
| Loader paths on Windows: a case-insensitive check that a path stays inside the workspace, drive letters, UNC and long-path prefixes, and junctions as well as symbolic links. To check | Sprint 3 | None | 2 |
| Tensor files mapped with `MapViewOfFile`. On Windows they are read whole into memory today | Sprint 3 | None | 1 |
| DuckDB on Windows: the import library and DLL of the standalone builder build, and how DuckDB compares allowed directories there. To check | Sprint 3 | None | 1 |
| The stores and the journal on Windows: replacing a file that is open, and the journal's append. To check | Sprint 3 | None | 1 |
| The stand-in command server, which runs on macOS and Linux only, or its removal once the app serves the commands itself | Sprint 3 | Sprint 3 wired into the app | 1 |

- **Done when** the three-panel dashboard of sprint 3's acceptance is built on Windows by hand, through `caliper-mcp`, and through an agent, with the key read from the Credential Manager.

### 10. Scenes and recordings

Scenes and recordings are designed in sprint 4. Most of the work is platform-neutral once the renderer steps above are done; these items are what remains for Windows.

| Item | Added by | Depends on | Builder days |
|---|---|---|---|
| The recording file library on Windows: mapping a file with `MapViewOfFile`, reading a file that another process is still writing, and recovery after a crash | Sprint 4 | The recording file library, built on macOS first | 2 |
| The app's recorder, the workspace tables and `caliper open <file>` on Windows | Sprint 4 | Step 9 for the workspace; step 2 for the console entry point | 1 |
| The scene view on Vulkan with CUDA: import of scene slots, the pick pass, the up-axis switch and the capture rows | Sprint 4 | The Vulkan backend of step 7; the scene layout of step 3 | 2 |
| The loss-landscape demo and the four demos on CUDA. They are written for MPS | Sprint 4 | Steps 3 and 7 | 2 |
| Slice planes and isosurfaces on CUDA: the device-side slice and the marching-cubes kernel | Sprint 5 | The scene view item above | 3 |

- **Done when** the training-run demo runs on a Windows machine with CUDA, its scene is drawn in place in the app, and its recording reopens and scrubs in the app and in a browser.

## Order

Steps 1 and 2 come first, because everything else depends on them. Steps 3 to 5 follow the macOS order of surfaces, with Caliper LIB and notebooks first. Caliper Embed, the renderer work and distribution come after them. Scenes reach Windows in three stages: first in a browser, through the network route; then in the app through the Dawn fallback, with a reported copy; and last in place, through the Vulkan backend of the graphics layer. Within step 7 the Dawn fallback therefore comes before the Vulkan backend. Step 9 needs steps 1 and 4 and can run beside steps 3 to 7, and its process helper comes early because step 2's `caliper build` needs it. Step 10 comes after steps 3, 7 and 9, and before distribution. Each step updates Verify on Windows hardware (`docs/guides/verify-on-windows.md`) with what it checked.
