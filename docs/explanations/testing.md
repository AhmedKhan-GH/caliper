# Testing

Caliper's tests use doctest and run through ctest. This page describes the layers that exist, what each proves, and how the tests that need a GPU report when there is none. The sources are in `tests/`, and `tests/CMakeLists.txt` registers them.

## The fast unit suite

`caliper_tests` is one executable that needs no window and no GPU. It holds the layers below, one source file per layer.

| Layer | What it proves |
|---|---|
| Manifest | The parser accepts a valid manifest, rejects a missing or malformed field, validates the applet id and ignores unknown keys. |
| Negotiation | The pre-load checks run in order: epoch, `min_host`, then required services, each with its refusal message. |
| Loader | The loader scans a folder, launches the `hello` applet through its full lifecycle, tears down an old instance before a relaunch, checks that the descriptor agrees with the manifest, shows a missing binary as a failed card, and refuses a path-escaping id. |
| Crash guard | A fault or an exception in an applet call is contained and reported, in the frame call and in a background job. The exception cases use `tests/fixtures/throwing_applet`, a hand-written descriptor that throws across the ABI, which the sugar would otherwise catch. |
| UI recovery | The host's UI stack is unwound after an applet fault left windows, plots and style pushes open. |
| Frame watchdog | Three consecutive slow frames latch the flag, a good frame resets the streak, and the flag stays until it is reset. |
| Services | Each service store and wrapper has its own tests: jobs, device, metrics, feed, artifacts, data and the tensor bridge. |
| Sugar | `CALIPER_APPLET`, the typed service wrappers, exception catching at the ABI edge and a run at a `dpi_scale` of 2. |
| ABI | `test_abi.cpp` asserts that the ABI structs are standard layout with `struct_size` first, and checks the epoch and every published table. |
| C compile check | `abi_c_check.c` includes every public header, newest first, in a C translation unit, so a header that is missing its own includes cannot hide behind another one. |

The same executable also covers the applet-side logic that needs neither torch nor a window, such as the instanced pose helper, the pulse ring and the export affordance.

The `hello` applet and `throwing_applet` are built first, because the loader tests load them from the build folder.

## The fixture host

`sdk/testing/caliper/fixture_host.h` is a headless fake `CaliperHost`. It serves `caliper.log.v1` and returns `NULL` for every other service, and a test can inject a table for any other service with `provide`. It lets an applet's logic and the sugar run without launching the UI. Only one fixture exists per process. The CMake target is `caliper::sdk_testing`; see [C++ sugar](../references/sugar.md) for its use in an applet's own tests.

## The pixel-exact graphics suite

`caliper_gfx_tests` (`tests/gfx/gfx_main.cpp`) checks the path from a tensor to a texture and the geometry path. It opens a window, so it needs a graphical session. Every case reads the texture back from the GPU and compares it byte for byte with a CPU reference, which is the same reference the staged upload path uses.

The suite runs once per backend that the platform offers:

- OpenGL runs everywhere, as the fallback.
- Metal runs on macOS when a Metal device is present. It also covers the device paths, built from raw `MTLBuffer` allocations: the compute path for `f32` with a lookup table, the blit path for `u8`, stream-ordered updates, imported allocations and their gates, and the geometry service.
- Vulkan runs on Windows with the CPU-staged path. Cases that need CUDA hardware run only when a CUDA device is present.

`caliper_export_tests` does the same for `caliper.export.v1`: it drives the service as an embedder would and decodes the PNG it wrote. Its sidecar, atomic-write and sugar cases need no GPU.

## Embedding and live streaming

`caliper_embed_tests` drives `libcaliper` as a third-party embedder would: it creates an offscreen core, loads the `hello` applet, pumps frames, reads pixels back and shuts down, and it checks the refusal gates. `caliper_live_tests` and `caliper_live_views_tests` cover live streaming, on macOS only, in one process over an offscreen core. See [Live streaming](live-streaming.md).

## Applet and adapter tests

`caliper_torch_tests` tests the torch adapter in `adapters/torch.hpp`. Five small executables test the learning and simulation code of single applets: `caliper_sculpt_tests`, `caliper_mesh_tests`, `caliper_em_pic_tests`, `caliper_surface_tests` and `caliper_twin_tests`. They link libtorch and are built only when torch is enabled, which keeps the torch link cost out of `caliper_tests`.

## Package and boundary tests

`caliper_sdk_install` (label `sdk`) proves that an applet can build against the installed SDK alone. It installs the SDK component of the build into a work folder, copies `tests/sdk_install_probe` and `examples/hello` there, and builds each copy with only `-DCMAKE_PREFIX_PATH` pointing at the install. It then runs the probe, which draws a headless frame with the installed UI stack and calls the fixture host, checks that hello's library and its stamped manifest exist with the expected `[build]` lines, and fails if either build refers to a file in the source tree. `tests/sdk_install_probe/run.cmake` holds the steps, and `CALIPER_SDK_PROBE_DIR` sets the work folder, which CI points outside the checkout. See [SDK package](../references/sdk-package.md).

`caliper_host_names_no_applet` (label `boundary`) runs `scripts/check_host_names_no_applet.py`. The script collects the `[applet]` id and name of every manifest under `applets/` and `examples/`, and every window title that their sources pass to `ImGui::Begin()` as a string literal, and fails if any of them is a whole string literal in the code of `src/` or `include/`. Comments are skipped, so a header may cite an applet id as an example. The test is registered only when CMake finds Python 3.

## Tests that need a GPU

A suite that needs a GPU must not report success when it found none. `tests/gpu_skip.h` provides the convention that the gfx, export, embed, live and torch suites share:

- A case with no GPU counts as skipped, prints why, and returns.
- When any case was skipped for want of a GPU and none failed, the executable exits with code 77. `tests/CMakeLists.txt` registers 77 as `SKIP_RETURN_CODE`, so ctest shows the suite as skipped and not as passed.
- A case that can never apply on the platform, such as CUDA on macOS, is counted separately and does not change the exit code.
- With `CALIPER_REQUIRE_GPU=1` a skip for want of a GPU is a failure. A CI lane that must have a GPU sets it, so the lane cannot go green by accident.
- `CALIPER_TEST_FORCE_NO_GPU=1` makes the test environments report no device, so the skip path itself can be tried on a machine that has one.

## Running the tests

```bash
ctest --test-dir cmake-build-release --output-on-failure
```

Three ctest labels select a suite: `gfx` (`caliper_gfx_tests`), `export` (`caliper_export_tests`) and `torch` (`caliper_torch_tests`). For example, `ctest --test-dir cmake-build-release -L gfx` runs only the pixel-exact graphics suite. CI runs the whole set on macOS and Windows; see [Continuous integration](../guides/ci.md).
