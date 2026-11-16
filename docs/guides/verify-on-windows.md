# Verify on Windows hardware

This guide shows how to check a change on a Windows machine with an NVIDIA GPU. Do it for any change to the Vulkan
backend, the tensor bridge, the geometry service, the CUDA kernels or the torch adapters, because those paths are
compiled out on macOS and only run on this hardware. The method is the same for every such change: build, run the
tests until none are skipped, run an applet and read which path is live, then record what you saw.

## Prerequisites

- **Windows with an NVIDIA GPU and a current driver.** The driver provides both `vulkan-1.dll` and `nvcuda.dll`. The
  host loads each at run time, so neither the Vulkan SDK nor the CUDA toolkit is needed to build or run the host.
- **A desktop session on the machine.** The gfx suite creates hidden windows and cannot initialize Vulkan in some
  remote desktop sessions. When it cannot, its cases skip, as described below.
- **Visual Studio with the MSVC C++ tools, CMake 3.25 or later (the presets need it), Ninja and Git.** The presets use
  the Ninja generator, so configure from an x64 developer command prompt, where the MSVC environment is set.
- **A CUDA toolkit, if you want the torch CUDA paths.** The build chooses a CUDA libtorch by the toolkit it finds when
  it configures: CUDA 12.4 or newer selects `cu124`, 12.1 to 12.3 select `cu121`, and 11.8 to 12.0 select `cu118`. With no toolkit,
  or one older than 11.8, the build falls back to CPU libtorch, and the tests and applets that need CUDA tensors have
  nothing to run on. The host, the gfx suite and the direct-image kernels do not need the toolkit, because the kernels
  ship as checked-in PTX loaded through the driver.
- **One GPU visible to CUDA.** The device rows of the gfx suite run only when the Vulkan device pairs with the single
  CUDA device. On a hybrid laptop, make sure Caliper runs on the NVIDIA GPU (Windows graphics settings, high
  performance), so the Vulkan device and the CUDA device are the same one.

## Build

From the repository root:

```bat
git submodule update --init --recursive
cmake --preset release -DCALIPER_BUILD_APPLETS=ON
cmake --build --preset release
```

The `release` preset builds into `cmake-build-release` with tests enabled. The core build compiles no applets by
default; `CALIPER_BUILD_APPLETS=ON` builds the scopes into `cmake-build-release\applets\`, which the host scans. On
Windows this is how `embed_scope` and `gpt_scope` build, because they need the curl and zlib that the core build
provides. The first configure downloads libtorch
(about 2 GB) and fetches the Vulkan headers, volk and glslang, which is why no SDK is needed. The build compiles the
shaders to SPIR-V and embeds them, so a mistake in a shader or in the parameter block layout shows up as a build error.

## Run the test suites

Run everything first:

```bat
ctest --preset release --output-on-failure
```

Then run the gfx suite on its own, and read it closely:

```bat
ctest --preset release -L gfx --output-on-failure
```

The gfx suite has four environments, and the Vulkan ones are the point of this guide.

| Rows | What they check | Needs |
|---|---|---|
| `gfx/Vulkan` | The CPU-staged matrix on the Vulkan backend: viridis, magma and RdBu f32 mapping, u8 direct, update and invalid tensors, all byte-exact against the CPU reference | Any Vulkan driver |
| `gfx/Vulkan+CUDA` | The device paths against the same reference: `compute` at ragged sizes; the direct kernels on edge values and ranges, in u8 and in order with frames; direct, buffer and CPU-staged updates mixed on one texture; refusal of tensors the kernel cannot read safely; release straight after an update; burst updates ordered by the semaphore; `alloc_shared` in place; imported allocations at offsets 0 and 512, misaligned offsets, out-of-bounds windows and repeated import and release; the timing rows | A UUID-paired CUDA device, the only one in the machine |
| `gfx/geometry` | Points, primitives, textured meshes and instanced transforms drawn from imported allocations, byte-exact against shared CPU references (the lit rows allow two LSBs, the one tolerance), and every gate refusing the whole frame with pixels untouched | The portable gate rows need no CUDA. The drawing rows need imported CUDA blocks |
| `gfx` (OpenGL) | The CPU-staged matrix on OpenGL | A window with an OpenGL context |

Select rows by name with doctest, for example `caliper_gfx_tests -tc="gfx/Vulkan+CUDA*"`, which is useful when
iterating on one path.

### Reading skips

A case that finds no device counts a skip and says why. At the end the binary prints a line of the form
`[caliper-tests] skipped N of M test cases ...` and exits with code 77, which ctest reports as Skipped instead of
Passed. Skips that never apply on a platform are counted separately and do not change the exit code.

On this hardware the target is no skips. A skip with `no Vulkan ICD` means the Vulkan backend failed to initialize. One
with `no CUDA interop / not single-GPU` means the device rows did not run, and a green result then proves nothing about
them. Set `CALIPER_REQUIRE_GPU=1` to turn any such skip into a failure, which is the setting for a machine that must
have the GPU. Read the full log for a "skipping" line before you trust a pass.

If a byte-exact row fails, stop and find out why. Do not loosen the comparison. The CPU reference is the single source
of truth for every backend, and a mismatch is either a backend bug or a bug in the shared reference.

## Read the startup log

Run the application from a terminal so its standard error is visible:

```bat
.\cmake-build-release\caliper.exe
```

Open an applet that trains on CUDA. `embed_scope` starts training on its first frame when `CALIPER_EMBED_AUTOTRAIN=1`
is set, which is useful for repeatable runs. The host and the Vulkan backend print these lines.

| Line | Meaning |
|---|---|
| `[renderer] vulkan` | The Vulkan backend initialized. `[renderer] vulkan init failed; falling back to gl` means it did not, and everything after is on OpenGL |
| `[vulkan] device path: sync mode: pipelined (shared timeline semaphores)` | The Vulkan device paired with a CUDA device and exports timeline semaphores. The stream-ordered handoff is honored |
| `[vulkan] device path: sync mode: synchronous (timeline semaphores unavailable)` | Paired, but without pipelining. The direct path is unavailable and adapters keep draining |
| no `sync mode` line | Interop is not paired. Check that the Vulkan device is the NVIDIA GPU |
| `[vulkan] device path: CUDA interop OK — zero-copy (kernel reads the tensor, writes the image)` | A texture update took the direct path |
| `[vulkan] device path: CUDA interop OK — GPU-resident (no CPU staging)` | A texture update took the buffer path, which makes one copy inside VRAM |
| `[vulkan] device path: direct image: ... using the buffer path` | The direct path failed to set up for a texture. The line says why |
| `[vulkan] device path: import path OK — zero-copy (imported allocation in place)` | An update read an imported allocation |
| `[vulkan] device path: geometry path OK — primitives drawn from imported allocations in place` | The geometry path drew from imported memory |

Each line prints once per process, so a line that never appears means that path never ran. An applet's own status
line should name a zero-copy path only when one of these confirms it. See
[Vulkan and CUDA on Windows](../explanations/vulkan-cuda.md) for what each path does.

## Measure and compare

Set `CALIPER_VULKAN_STATS=1` before launching and quit the application normally. At exit the backend prints one line:

```text
[vulkan] stats: N frames, ms p50 .. p99 .. max ..; updates direct .. buffer ..; kernel waits on a frame ..; retire blocks .. (.. ms total)
```

The counts say how many updates took each path, how often a kernel waited on a frame, and how long the frame thread
blocked in `retire()`. Compare these with the measurements in
[the measured costs](../explanations/vulkan-cuda.md#measured-costs), which were taken on an RTX 500 Ada laptop GPU, and
only compare runs that are alike. Laptop thermal throttling changes training speed by 15 to 20 percent on its own, so
repeat each run and alternate the configurations.

Three configurations of the same binary isolate a change:

- **Default.** The direct path where it applies.
- **`CALIPER_VULKAN_NO_DIRECT=1`.** Every update takes the buffer path. Use it to confirm that the fallback rung still
  works and to compare costs.
- **`CALIPER_RENDERER=gl`.** The OpenGL backend. The applet should show an identical image and say that it is on its
  fallback, with no zero-copy claim.

### Validation layers

For a change that touches descriptors, pipelines or the parameter block, run once with the Vulkan validation layer. The
backend is built without the SDK, so install the Vulkan SDK separately and enable `VK_LAYER_KHRONOS_validation`, for
example with `VK_INSTANCE_LAYERS`. Run the gfx suite and a live applet that exercises the path, and read the output for
complaints in that area. Findings that predate your change belong in the record as such, not as blockers.

## Watch the first Windows CI run

The Windows lane of `.github/workflows/build.yml` runs the same steps as the macOS lane: the core build without
applets, the whole ctest suite with the command-line tests and `caliper_cli_new_builds`, the SDK and embedding package
tests, the out-of-tree builds of `hello`, `sine_scope` and the scopes, and the check that the host names no applet.
Several of these pieces have only run on macOS so far. When the lane first runs them, read its log for the following.

| Piece | What to look for |
|---|---|
| Command-line tests (`caliper_cli_*`) | The executable prints folders with `\` on Windows. The patterns of `check_build_folder` and `install_default` accept either separator; a new failure that quotes a folder is most likely another pattern that expects `/`. |
| `caliper_cli_new_builds` | The project that `caliper new` writes is configured and built with MSVC inside the test. Its test executable compiles the applet source, so it exports `caliper_applet_descriptor` and MSVC writes an import library beside it, which is expected. `caliper check` must then find the library and the manifest together in `build/applets`. |
| `caliper_sdk_install` | The check that the probe and `hello` builds name nothing in the source tree searches `CMakeCache.txt` and `build.ninja` for paths with `/`. A path that Ninja writes with `\` would pass unnoticed, so the check is weaker on Windows, not stricter. |
| `caliper_embed_install` | `embed_host` gets `caliper_embed.dll` beside it from its own post-build copy. An exit code of `0xc0000135` means the embedding library needs a DLL of its own that is not beside it or on `PATH`; `dumpbin /dependents caliper_embed.dll` names it. |
| `caliper_cli_shot_temp` and `caliper_cli_shot_data_dir` | Without a GPU both report `run skipped: no GPU`. On a machine with Vulkan, `caliper shot` draws into the Vulkan offscreen canvas, which has not drawn the applet page before: open the PNG and compare it with a macOS shot of the same applet. |
| The data-schema stamp | The `LockFileEx` branch of the stamp lock in `src/host/applet_loader.cpp` compiles for the first time. When the stamp is raised, the new file is renamed over the old one; if another process has the old one open at that moment, Windows refuses the rename and the launch fails with `could not record the data schema`. Watch the loader tests for that message. |
| `scripts/build_applets_against_sdk.sh` | The lane runs it in Git Bash (`shell: bash`). The script takes its own folder with `pwd -W`, so CMake receives `D:/a/...` paths; the work folder comes from `cygpath -m`. |
| `CALIPER_APPLET_PATH` | Folders are separated by `;` on Windows, so a drive letter's `:` stays in its folder. Set it as `C:\a;D:\b` from cmd or PowerShell. |
| The network cases of `caliper_tests` | The WebSocket layer, the codec, the hub and the Caliper API server (`src/host/net/`, `src/host/api/`) are compiled with MSVC and run on Winsock for the first time; they were checked only by reading and by syntax checks against the documented Winsock declarations. `WSAPoll` gets no entry for a connection that waits for its answer. A refused connect can take about two seconds on Windows, because Windows retries it, and its reason reads `socket error 10061`; the cases wait up to five seconds and look only for the word `connect`. The back-pressure case expects at least 16 MiB to stay queued for a client that never reads, so a loopback that buffers more than 48 MiB in the kernel would fail it. |
| `caliper_cli_windows_*` | The refusals of `caliper serve`, `caliper build`, `--api` and `--url`: each must exit with 1 and print `... is not available on Windows yet` before any window opens or any file is written. An exit code of 0, or a window, means that the refusal at the top of `main()` in `src/main.cpp` did not run. |
| `caliper_embed_install` and `caliper_embed.dll` | The module-definition file exports every C function that `embed.h`, `live.h` and `live_views.h` declare, except `caliper_live_*`. It now includes `caliper_core_live_attach_url`, which `src/host/live/live_unsupported.cpp` defines on Windows as a call that fails with a reason. An unresolved export at the link of `caliper_embed.dll` names a declared function that has no Windows definition. |
| Hidden windows | The gfx and export suites open hidden windows. On Windows a hidden window has no taskbar button, so the background helper that keeps the macOS test process out of the Dock does nothing there. |

## What to record

Write down enough that another person can repeat the check and compare numbers:

- the commit, the preset, and whether the tree was clean;
- the GPU model, the driver version, the Windows version, the CUDA toolkit version and the libtorch variant the build
  chose;
- the ctest result for each suite, with case and assertion counts for the gfx suite, and the `[caliper-tests]` skip
  summary line if one is printed;
- the `[vulkan]` and `[renderer]` log lines you saw, and any you expected and did not;
- the `CALIPER_VULKAN_STATS` line for each configuration you ran, and how many runs it came from;
- for each failure, the exact row name and the message, and what you changed in response.

State what you did not run. A check that skipped rows or ran only one configuration is still useful, provided the
record says so.
