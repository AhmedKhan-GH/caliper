# Windows parity

Caliper is developed on macOS first. This page lists the work that brings Windows level with macOS once the current macOS sprints are finished: the applet platform, Caliper LIB's streaming and notebook widgets, the web viewer, Caliper API, Caliper Embed and the renderer fallbacks. The [roadmap](roadmap.md) holds the order and status. Linux is not planned for now.

On Windows the fast path is Vulkan with CUDA interop, where macOS uses Metal with MPS. The toolchain contract is MSVC with the dynamic CRT (`/MD`), as Compatibility and epochs (`docs/explanations/compatibility.md`) describes.

## Where Windows stands

- **CI builds and tests the core on `windows-latest`.** The Windows lane of the Build workflow runs the same steps as the macOS lane: the core build without applets, the whole ctest suite with the Caliper CLI tests and the Caliper SDK and Caliper Embed package tests, the builds of `hello`, `sine_scope` and the scopes against the installed SDK, and the check that the host names no applet. It skips `embed_scope` and `gpt_scope`, which need the curl and zlib that only the core build provides on Windows. The runner has no GPU, so the GPU rows skip there. Several of these pieces have only run on macOS so far; Watch the first Windows CI run (`docs/guides/verify-on-windows.md`) lists what to look for.
- **The Vulkan backend, the CUDA interop paths and the gfx suite have been verified on Windows hardware before.** Verify on Windows hardware (`docs/guides/verify-on-windows.md`) describes the procedure.
- **The code already handles several Windows details.** The loader opens an applet with `LoadLibraryA` and looks for `<stem>.dll`, with or without the `lib` prefix. `CALIPER_APPLET_PATH` uses `;` as its separator, and the user applets folder is `%APPDATA%\Caliper\applets\`. The lock around the data-schema stamp uses `LockFileEx`. `caliper shot` draws into the Vulkan offscreen canvas and reads the frame back, with no window. The `caliper-embed` package installs `bin\caliper_embed.dll`, `lib\caliper_embed.lib`, the headers and the CMake package, without the `caliper_live_*` calls, which exist on macOS only, and `embed_host` copies the DLL beside itself.
- **No one has yet run on Windows hardware** applets loaded from outside the repository, the Caliper CLI subcommands, `caliper shot` or the Caliper Embed package.
- **The network stream, the hub, Caliper API, `caliper build`, the web viewer and the notebook widget are built and tested on macOS.** None of their code has been compiled with MSVC yet. It was checked by reading, by a syntax check of its Windows branches against the documented Winsock declarations, and by a check that the code the Windows lane compiles never turns a `std::filesystem::path` into a `std::string` implicitly. The first Windows CI run is their first compile; docs/guides/verify-on-windows.md, section "Watch the first Windows CI run", lists what to look for. On Windows they stand as follows.
    - **Compiled, not tested.** The WebSocket layer, the hub and the Caliper API server (`src/host/net/`, `src/host/api/`) are portable C++ on Winsock and compile in the Windows lane, but they are untested on Windows: their socket tests (`test_net_ws.cpp`, `test_net_hub.cpp`, `test_api.cpp`) are left out of the Windows build of `caliper_tests`, and the `caliper_net_tests` entry reports them as skipped. Every feature built on them is refused on Windows. The codec of wire protocol v3 and the JSON reader, which the live controls use, are tested there through their `net/wire` cases, as are the unit cases of the command line, `caliper build`'s searches among them.
    - **Refused.** The executable refuses `caliper serve`, `caliper build`, `--api` and `--url` before anything starts: it prints `... is not available on Windows yet` and exits with 1 (`windows_unavailable` in `src/host/host_cli.h`), and it ignores `CALIPER_API`. The `caliper_cli_windows_*` tests check this in the lane, and docs/references/cli.md, section "On Windows", describes it.
    - **macOS only.** The app's network source, the live attach and the shared-memory sessions. On Windows `src/host/live/live_unsupported.cpp` stands in for them, and every attach, `caliper_core_live_attach_url` included, fails with a reason. The live controls and the typed live views compile on Windows.
    - **Not run on Windows.** The process-level tests of `caliper serve`, `caliper shot --api`, the hidden app, its run bar and `caliper build`, which are registered on macOS only; the network transport of Caliper LIB, `caliper.serve()` and the notebook widget; and the web viewer in a browser on Windows.

## The steps

Each step ends with a check that must pass on a Windows machine with an NVIDIA GPU, not only in CI.

### 1. Build and test baseline

- Run the full CI on `windows-latest` for every new package and subcommand. A test that skips on a GPU-less runner must also run on real hardware before the step counts as done.
- Run the gfx, export and embed suites on Windows hardware with zero skips.
- **Done when** CI is green and the hardware run shows no skipped GPU rows.

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

### 3. Caliper LIB

- **Wheels.** Build Windows wheels with MSVC and test them against current CUDA builds of PyTorch. The extensions link no libtorch, as on macOS, so they share a process with the torch the script imported.
- **The local session.** Today it uses POSIX shared memory and MPS. On Windows it needs named shared memory (`CreateFileMapping`) and CUDA tensors, with the snapshot copied on the device and never blocking training, matching the MPS design.
- **The network transport** is portable and should work first. Verify it, and measure the training overhead on CUDA as the benchmark does on MPS.
- **The in-process window** on Vulkan.
- **Done when** the example notebook runs on Windows against CUDA, and both the app and a browser show the live views.

### 4. Streaming, the hub and Caliper API

- The app's network source, the `caliper serve` hub and Caliper API v0 on Winsock. Build and run their socket tests on Windows first (`caliper_net_tests` stands in for them in the lane today). Port the network source, which is Objective-C++ for macOS today, and lift each refusal in `windows_unavailable` once its process-level test runs on Windows; those tests use POSIX processes and signals (`tests/test_proc.h`), so they need a Windows runner first.
- Bind to 127.0.0.1 only, so Windows Defender Firewall never prompts.
- Keep the API token file in `%APPDATA%\Caliper\`, readable only by the user.
- **Done when** a CUDA script streams to the app, to a browser and through the hub on one Windows machine, and every API v0 endpoint passes its tests.

### 5. The web viewer and notebooks

- The browser side is platform-neutral, but check it in Chrome and Edge on Windows, where WebGPU runs on Direct3D 12.
- Check the notebook widget in JupyterLab and in Visual Studio Code on Windows.
- **Done when** the web viewer's headless checks pass on a Windows runner and the widget draws in both notebook front ends.

### 6. Caliper Embed

- Run the installed `caliper-embed` package on Windows hardware: the DLL, its import library, the headers and the CMake package, which the Windows lane installs and builds `embed_host` against.
- Run the Win32 `embed_host` against the installed package, in a window and in its offscreen mode.
- Add hosting examples for WPF (`HwndHost`), WinForms (a control's `Handle`) and Qt (`winId()`).
- **Decide on WinUI 3.** It expects a DXGI swap chain rather than a child window, so it needs Vulkan–DXGI interop or the Dawn path.
- **Done when** the installed package hosts an applet in a Win32 window and in a WPF window, and renders offscreen.

### 7. Renderer fallbacks

- The WebGPU backend through Dawn as the native fallback on Windows: Direct3D 12 and Direct3D 11 for machines without working Vulkan, and SwiftShader for machines without a usable GPU.
- Retire the GLEW OpenGL fallback once Dawn covers its cases.
- **Done when** the app runs on a machine with Vulkan disabled and says which fallback it is using.

### 8. Signing, packaging and distribution

- Authenticode-sign the app, the CLI entry point and the first-party applet DLLs.
- Plan for SmartScreen reputation.
- **Decide the signing policy for third-party applet DLLs.** It is the Windows side of the open signing question on macOS.
- Package with an installer (MSIX or MSI) and publish through winget, putting the CLI on `PATH`.
- **Done when** a clean Windows machine installs Caliper from the package, runs `caliper info` in a terminal, and opens an applet without a SmartScreen block.

## Order

Steps 1 and 2 come first, because everything else depends on them. Steps 3 to 5 follow the macOS order of surfaces, with Caliper LIB and notebooks first. Caliper Embed, the renderer fallbacks and distribution come last. Each step updates Verify on Windows hardware (`docs/guides/verify-on-windows.md`) with what it checked.
