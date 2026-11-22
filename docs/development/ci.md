# CI

Caliper builds and tests on GitHub Actions. Four workflows run, two lanes build the code, and two kinds of cache speed up the runs.

## Workflows

| Workflow | File | Runs on | Does |
|---|---|---|---|
| Build | `.github/workflows/build.yml` | pushes to `main` and every pull request, except changes only to `site/`, `docs/`, `design/`, `mkdocs.yml` or Markdown files; also manually | builds Caliper and runs the test suite on macOS and Windows |
| Browser views | `.github/workflows/web.yml` | pull requests and pushes to `main` that change a source of the WebAssembly views, the `web/` package, the Python package's browser files or the scripts that handle them; also manually | builds the WebAssembly views at the pinned Emscripten, checks them and the `<caliper-view>` element in headless Chromium (see [The Browser views tests](#browser-views-tests)), and uploads them as the artifact `caliper-views`. For `main` and on request, it also builds the wheel with them |
| Docs | `.github/workflows/docs.yml` | pull requests that touch `docs/`, `design/`, `docs-theme/`, `mkdocs.yml`, the files the docs embed, or the docstrings and headers of the API reference; also manually | builds the docs site and the design pages with `--strict`, so a broken link fails the pull request, checks that the API reference covers every public symbol, and publishes nothing |
| Deploy Pages | `.github/workflows/pages.yml` | pushes to `main` that touch `site/`, `docs/`, `design/`, `mkdocs.yml` or the files the docs embed | builds the docs with MkDocs and publishes them with the landing page, and checks that the design pages build, without publishing them |

A newer push to the same branch cancels a Build run still in progress.

## Two Lanes

| Lane | Runner | libtorch | Graphics and GPU | Builder |
|---|---|---|---|---|
| macOS | `macos-15` (Apple Silicon) | arm64 archive | Metal backend and MPS test cases run here | built, and its `builder_*` tests run |
| Windows | `windows-latest` with MSVC | CPU archive | GLEW prebuilt zip, Vulkan through volk with no SDK, and glslang fetched and built in the tree. `USE_CUDA` falls back when no CUDA toolkit is found | `CALIPER_WITH_BUILDER` is OFF by default, because the builder has not been compiled with MSVC yet |

The macOS builder lives under `src/builder/`. `scripts/check_macos_min.py` fails the macOS lane unless the `caliper` executable, the viewer, the embed library and everything built against the installed packages say macOS 14.0 as their minimum. The Browser views workflow checks the wheel's platform tag and the extensions in it the same way.

The Windows test executables that link libtorch need its DLLs beside them, because Windows has no rpath. `tests/CMakeLists.txt` copies them there.

Each lane runs these steps:
1. **Check out,** then run `./init-submodules.sh`. The script wipes `third_party/` and the build directories, then initializes the submodules. It falls back to the default branch for pins that the remotes cannot supply.
2. **Restore the caches.** This happens after the script, because the script would otherwise delete them.
3. **Configure** with `cmake --preset release -DCALIPER_WITH_SLANG=OFF` and sccache as the compiler launcher. slang is vendored but nothing links it. `CALIPER_SDK_PROBE_DIR` and `CALIPER_EMBED_PROBE_DIR` point the `caliper_sdk_install` and `caliper_embed_install` tests at the runner's temporary folder, outside the checkout.
4. **Build** with `cmake --build cmake-build-release --parallel`. The core build compiles no applets.
5. **Test** with `ctest --test-dir cmake-build-release --output-on-failure --timeout 600`.
   - The GPU suites (gfx, export, embed, torch) skip when the runner has no usable device, and ctest reports them as skipped.
   - Set `CALIPER_REQUIRE_GPU=1` in a lane that must have a GPU, to turn those skips into failures.
6. **On failure,** upload `Testing/Temporary/LastTest.log` as an artifact.
7. **Build the applets against the installed SDK** with `scripts/build_applets_against_sdk.sh`. The script installs the SDK from the core build. Then it builds `examples/hello`, `examples/sine_scope` and every scope in its own build folder, with only `-DCMAKE_PREFIX_PATH` pointing at the SDK, as a developer outside the repository would. The Windows lane skips `embed_scope` and `gpt_scope`, because they need curl and zlib, which Windows provides only inside the core build.
8. **Check that the host names no applet** with `scripts/check_host_names_no_applet.py`, on macOS only.
9. **Always** print `sccache --show-stats`.

To reproduce a lane locally, run the same configure, build and test commands. The README's [Quick start](https://github.com/AhmedKhan-GH/caliper#quick-start) covers the prerequisites.

## Browser Views Tests

The web package's Playwright tests run in two steps on a hosted macOS runner with three cores and no GPU.

The first step runs the `swiftshader` project, which uses WebGPU on the SwiftShader software adapter and runs one test at a time. It runs only the specs that draw with WebGPU. These are the WebAssembly views, the renderer choice and its fallback, the 3-D points, the colormap's shaders, linked selection and the specs that read pixels.

The second step runs the `canvas2d` project, with WebGPU off, on two workers and with every spec. The protocol and round-trip specs (`codec`, `controls` and `e2e`) and the frame-rate measurement run only here. Software rasterizing is slow on three cores, so specs that test the protocol rather than drawing do not wait on it.

`web/playwright.config.ts` sets this from `CI`. Without `CI`, every project runs every spec. The `webgpu` project needs a GPU and runs only locally. A failed step keeps Playwright's HTML report and the traces of the failed tests in the artifact `browser-views-report`.

## Caches

**libtorch and GLEW.** The `actions/cache` action stores them. The key includes `LIBTORCH_VERSION`, set at the top of `build.yml`. That version must match `PYTORCH_VERSION` in `cmake/Dependencies.cmake`. If one changes without the other, the cache misses and the run does not reuse a stale archive.

**Compiled objects.** Every compile runs through [sccache](https://github.com/mozilla/sccache), which stores each object in the GitHub Actions cache (`SCCACHE_GHA_ENABLED`). A run recompiles only the translation units whose preprocessed source, flags or compiler changed.

**Cache scope.** A pull request reads its own entries and the entries of `main`, but `main` reads only its own. As a result, the first `main` run after a merge builds cold once. After that, the entries of `main` warm every pull request.

**Size and eviction.** GitHub caps the caches of a repository at 10 GB. Beyond that, it evicts entries least-recently-used. It also evicts any entry unused for 7 days.

**What invalidates sccache entries:**
- A runner image with a new MSVC, Xcode or AppleClang: every entry for that lane misses, because sccache hashes the compiler.
- A submodule bump: the objects of that library miss.
- A change to a widely included header: everything that includes it misses.
- A change to a global compile flag: every compile it touches misses.
- An sccache upgrade: it can invalidate everything.

### `sccache` Statistics

Two values in the statistics need a check:
- **Cache hits and misses:** a warm run is close to all hits.
- **Non-cacheable compilations:** sccache lists the reason for each. The known cases are below.
  - **MSVC compiles that write a PDB (`/Zi`).** Release has none, because the top-level `cmake_minimum_required` makes policy CMP0141 NEW. Thus debug-info flags apply only to Debug and RelWithDebInfo.
  - **Precompiled headers.** Only the glslang `MachineIndependent` target uses them, so the Windows configure sets `CMAKE_DISABLE_PRECOMPILE_HEADERS=ON`. Local builds are not affected.
  - **CMake's C++20 module scanning** (`-interface`). This is one object, and it is never cached.

An object that fails to upload (a cache write error) misses once on the next run, then sccache stores it.

## Run Failures

**Downloads.** The configure step downloads sources from GitHub without authentication:
- Release tarballs for zlib, curl, Vulkan-Headers, volk and glslang (on Windows).
- Git clones of tomlplusplus and doctest (both lanes).

CMake retries each download several times, with a longer wait each time. During GitHub incidents, the downloads can still fail with timeouts or 504 errors and the message "Each download failed!". This is not a code failure. Re-run only the failed job:

```sh
gh run rerun <run-id> --failed
```

**Windows tests that hang until the 600 s timeout.** This usually means an executable could not load a DLL. The Windows loader shows an error dialog on the desktop session of the runner and waits for a person to dismiss it. Run the executable directly from bash to see the `0xC0000135` exit code immediately.

**A stack overflow test that kills `caliper_tests` with no output on Windows.** The crash guard must re-arm the guard page of the thread after a contained overflow (`_resetstkoflw` in `src/host/crash_guard.cpp`). Without that, a second overflow on the same thread ends the process.
