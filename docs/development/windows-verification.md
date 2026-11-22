# Windows Verification

This procedure checks a change on a Windows machine with an NVIDIA GPU. Use it for any change to the Vulkan backend, the tensor bridge, the geometry service, the CUDA kernels or the torch adapters, because macOS compiles these paths out.

```mermaid
flowchart LR
    build[Build] -->|binaries| test[Run tests]
    test -->|no skips| run[Run an applet]
    run -->|log lines| record[Record results]
```

The flow runs from build to record, and each step needs the one before it.

## Requirements

- **Windows with an NVIDIA GPU and a current driver.** The driver provides `vulkan-1.dll` and `nvcuda.dll`. The host loads both at run time, so the build and the host need neither the Vulkan SDK nor the CUDA toolkit.
- **A desktop session on the machine.** The gfx suite creates hidden windows, and some remote desktop sessions cannot initialize Vulkan. In that case its cases skip.
- **Visual Studio with the MSVC C++ tools, CMake 3.25 or later, Ninja and Git.** The presets use the Ninja generator, so configure from an x64 developer command prompt.
- **A CUDA toolkit, for the torch CUDA paths.** The build picks a CUDA libtorch when it configures: CUDA 12.4 or newer selects `cu124`, 12.1 to 12.3 select `cu121`, and 11.8 to 12.0 select `cu118`. With no toolkit, or one older than 11.8, the build uses CPU libtorch, and tests and applets that need CUDA tensors have nothing to run on. The host, the gfx suite and the direct-image kernels do not need the toolkit, because the kernels ship as checked-in PTX that the driver loads.
- **One GPU visible to CUDA.** The device rows of the gfx suite run only when the Vulkan device pairs with the single CUDA device. On a hybrid laptop, set Caliper to the high-performance GPU in the Windows graphics settings.

## Build

From the repository root:

```bat
git submodule update --init --recursive
cmake --preset release -DCALIPER_BUILD_APPLETS=ON
cmake --build --preset release
```

The `release` preset builds into `cmake-build-release` with tests enabled. The core build compiles no applets by default. `CALIPER_BUILD_APPLETS=ON` builds the scopes into `cmake-build-release\applets\`, which the host scans. On Windows this is how `embed_scope` and `gpt_scope` build, because they need the curl and zlib that the core build provides. The first configure downloads libtorch and fetches the Vulkan headers, volk and glslang, which is why no SDK is needed. The build compiles the shaders to SPIR-V and embeds them, so a shader error or a parameter block layout error shows up as a build error.

## Test Runs

Run everything first:

```bat
ctest --preset release --output-on-failure
```

Then run the gfx suite on its own and read it closely:

```bat
ctest --preset release -L gfx --output-on-failure
```

The gfx suite has three environments, all on Vulkan, which are the point of this procedure.

| Rows | What they check | Needs |
|---|---|---|
| `gfx/Vulkan` | The CPU-staged matrix on the Vulkan backend, byte-exact against the CPU reference | Any Vulkan driver |
| `gfx/Vulkan+CUDA` | The device paths against the same reference: `compute`, the direct kernels, mixed update paths, refusal of unsafe tensors, burst updates ordered by the semaphore, `alloc_shared`, imported allocations and the timing rows | A UUID-paired CUDA device, the only one in the machine |
| `gfx/geometry` | Points, primitives, textured meshes and instanced transforms drawn from imported allocations, and every gate that refuses a whole frame. Results are byte-exact, except the lit rows, which allow two LSBs | The gate rows need no CUDA. The drawing rows need imported CUDA blocks |

To iterate on one path, select rows by name with doctest, for example `caliper_gfx_tests -tc="gfx/Vulkan+CUDA*"`.

### Skipped Tests

A case that finds no device counts as a skip and says why. At the end the binary prints a line of the form `[caliper-tests] skipped N of M test cases ...` and exits with code 77, which ctest reports as Skipped. Skips that never apply on a platform are counted separately and do not change the exit code.

On this hardware the target is no skips. A skip with `no Vulkan ICD` means the Vulkan backend failed to initialize. A skip with `no CUDA interop / not single-GPU` means the device rows did not run, and a green result proves nothing about them. Set `CALIPER_REQUIRE_GPU=1` to turn any such skip into a failure. Use this setting on a machine that must have the GPU. Read the full log for a "skipping" line before you trust a pass.

If a byte-exact row fails, find the cause and do not loosen the comparison. The CPU reference is the single source of truth for every backend, so a mismatch is a backend bug or a bug in the shared reference.

## Startup Log

Run the application from a terminal so its standard error is visible:

```bat
.\cmake-build-release\caliper.exe
```

Open an applet that trains on CUDA. For repeatable runs, set `CALIPER_EMBED_AUTOTRAIN=1`, and `embed_scope` starts training on its first frame. The host and the Vulkan backend print these lines.

| Line | Meaning |
|---|---|
| `[renderer] vulkan` | The Vulkan backend initialized. If it cannot start, the app prints `caliper: the Vulkan renderer could not start (<reason>), and there is no fallback renderer until the WebGPU fallback through Dawn lands` and exits with 1, because there is no fallback renderer |
| `[vulkan] device path: sync mode: pipelined (shared timeline semaphores)` | The Vulkan device paired with a CUDA device and exports timeline semaphores. The stream-ordered handoff is honored |
| `[vulkan] device path: sync mode: synchronous (timeline semaphores unavailable)` | Paired, but without pipelining. The direct path is unavailable and adapters keep draining |
| no `sync mode` line | Interop is not paired. Check that the Vulkan device is the NVIDIA GPU |
| `[vulkan] device path: CUDA interop OK — zero-copy (kernel reads the tensor, writes the image)` | A texture update took the direct path |
| `[vulkan] device path: CUDA interop OK — GPU-resident (no CPU staging)` | A texture update took the buffer path, which makes one copy inside VRAM |
| `[vulkan] device path: direct image: ... using the buffer path` | The direct path failed to set up for a texture. The line says why |
| `[vulkan] device path: import path OK — zero-copy (imported allocation in place)` | An update read an imported allocation |
| `[vulkan] device path: geometry path OK — primitives drawn from imported allocations in place` | The geometry path drew from imported memory |

Each line prints once per process, so a line that never appears means that path never ran. An applet's status line may name a zero-copy path only when one of these lines confirms it. [Vulkan and CUDA on Windows](vulkan-and-cuda.md) describes each path.

## Measurement and Comparison

Set `CALIPER_VULKAN_STATS=1` before launch and quit the application normally. At exit the backend prints one line:

```text
[vulkan] stats: N frames, ms p50 .. p99 .. max ..; updates direct .. buffer ..; kernel waits on a frame ..; retire blocks .. (.. ms total)
```

The counts show how many updates took each path, how often a kernel waited on a frame, and how long the frame thread blocked in `retire()`. Compare them with [the measured costs](performance.md#vulkan-and-cuda-costs), and compare only runs that are alike. Thermal throttling on a laptop changes training speed by itself, so repeat each run and alternate the configurations.

Two configurations of the same binary isolate a change:

- **Default.** The direct path where it applies.
- **`CALIPER_VULKAN_NO_DIRECT=1`.** Every update takes the buffer path. Use it to confirm that the fallback works and to compare costs.

### Validation Layers

For a change that touches descriptors, pipelines or the parameter block, run once with the Vulkan validation layer. The backend builds without the SDK, so install the Vulkan SDK separately and enable `VK_LAYER_KHRONOS_validation`, for example with `VK_INSTANCE_LAYERS`. Run the gfx suite and a live applet that uses the path, then read the output for complaints in that area. Record findings that predate your change as such, and do not treat them as blockers.

## Recorded Results

Write down enough that another person can repeat the check and compare numbers:

- The commit, the preset, and whether the tree was clean
- The GPU model, the driver version, the Windows version, the CUDA toolkit version and the libtorch variant the build chose
- The ctest result for each suite, with case and assertion counts for the gfx suite and the `[caliper-tests]` skip summary line if one prints
- The `[vulkan]` and `[renderer]` log lines you saw, and any you expected and did not see
- The `CALIPER_VULKAN_STATS` line for each configuration you ran, and how many runs it came from
- For each failure, the exact row name, the message, and what you changed in response

State what you did not run. A check that skipped rows or ran only one configuration is still useful, if the record says so.
