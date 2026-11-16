# Continuous integration

This page describes how Caliper is built and tested on GitHub Actions: what runs, what is cached, and what to do when a run fails.

## Workflows

| Workflow | File | Runs on | Does |
|---|---|---|---|
| Build | `.github/workflows/build.yml` | pushes to `main` and every pull request, except changes only to `site/`, `docs/`, `design/`, `mkdocs.yml` or Markdown files; also manually | builds Caliper and runs the test suite on macOS and Windows |
| Browser views | `.github/workflows/web.yml` | pull requests and pushes to `main` that change a source of the WebAssembly views, the `web/` package, the Python package's browser files or the scripts that handle them; also manually | builds the WebAssembly views at the pinned Emscripten, checks them and the `<caliper-view>` element in headless Chromium on SwiftShader, and uploads them as the artifact `caliper-views`; for `main` and on request, it also builds the wheel with them |
| Shaders | `.github/workflows/shaders.yml` | pull requests and pushes to `main` that change the shader sources, their generated outputs, the graphics layer's header or `tools/shaders/`; also manually | builds the pinned Naga driver (cached), regenerates every shader output and fails on a difference from the committed files or a source that does not compile, tests the generator, validates the SPIR-V with `spirv-val` and compiles the generated headers as C++17 ([Write a shader](write-a-shader.md)) |
| Docs | `.github/workflows/docs.yml` | pull requests that touch `docs/`, `design/`, `docs-theme/`, `mkdocs.yml` or the files the docs embed; also manually | builds the docs site and the design pages with `--strict`, so a broken link fails the pull request, and publishes nothing |
| Deploy Pages | `.github/workflows/pages.yml` | pushes to `main` that touch `site/`, `docs/`, `design/`, `mkdocs.yml` or the files the docs embed | builds the docs with MkDocs and publishes them with the landing page, and checks that the design pages build, without publishing them |

A newer push to the same branch cancels a Build run still in progress.

## The two lanes

**macOS** runs on `macos-15` (Apple Silicon). It uses the arm64 libtorch archive and is the lane where the Metal backend and the MPS test cases are exercised.

**Windows** runs on `windows-latest` with MSVC. The setup is CPU only:
- libtorch is the CPU archive, and GLEW the prebuilt zip;
- Vulkan comes through volk, with no SDK, and glslang is fetched and built in the tree;
- `USE_CUDA` falls back when no CUDA toolkit is found.

The test executables that link libtorch need its DLLs beside them, because Windows has no rpath. `tests/CMakeLists.txt` copies them there.

Each lane runs these steps:
1. **Check out,** then `./init-submodules.sh`. The script wipes `third_party/` and the build directories, then initializes the submodules. Some pins can't be fetched from their remotes, and the script falls back to the default branch for those.
2. **Restore the caches.** This happens after the script, because the script would otherwise delete them.
3. **Configure** with `cmake --preset release -DCALIPER_WITH_SLANG=OFF` and sccache as the compiler launcher. slang is vendored but nothing links it. `CALIPER_SDK_PROBE_DIR` and `CALIPER_EMBED_PROBE_DIR` point the `caliper_sdk_install` and `caliper_embed_install` tests at the runner's temporary folder, outside the checkout.
4. **Build** with `cmake --build cmake-build-release --parallel`. The core build compiles no applets.
5. **Test** with `ctest --test-dir cmake-build-release --output-on-failure --timeout 600`.
   - The GPU suites (gfx, export, embed, torch) skip when the runner has no usable device, and ctest reports them as skipped.
   - Set `CALIPER_REQUIRE_GPU=1` in a lane that must have a GPU, to turn those skips into failures.
6. **On failure,** upload `Testing/Temporary/LastTest.log` as an artifact.
7. **Build the applets against the installed SDK** with `scripts/build_applets_against_sdk.sh`, which installs the SDK from the core build and builds `examples/hello`, `examples/sine_scope` and every scope in its own build folder, with only `-DCMAKE_PREFIX_PATH` pointing at the SDK, as a developer outside the repository would. The Windows lane skips `embed_scope` and `gpt_scope`, because they need curl and zlib, which Windows provides only inside the core build.
8. **Check that the host names no applet** with `scripts/check_host_names_no_applet.py`, on macOS only.
9. **Always** print `sccache --show-stats`.

To reproduce a lane locally, run the same configure, build and test commands. The README's [Quick start](https://github.com/AhmedKhan-GH/caliper#quick-start) covers the prerequisites.

## Caches

**libtorch and GLEW.** Cached with `actions/cache`. The key includes `LIBTORCH_VERSION`, set at the top of `build.yml`. That version must match `PYTORCH_VERSION` in `cmake/Dependencies.cmake`, so bumping one without the other misses the cache instead of reusing a stale archive.

**Compiled objects.** Every compile runs through [sccache](https://github.com/mozilla/sccache), which stores each object in the GitHub Actions cache (`SCCACHE_GHA_ENABLED`). A run recompiles only translation units whose preprocessed source, flags or compiler changed.

| Lane | Build step, cold | Build step, warm |
|---|---|---|
| macOS | about 13 min | about 1 min |
| Windows | about 26 min | about 3 min |

**Cache scope.** A pull request reads its own entries and `main`'s; `main` reads only its own. So the first `main` run after a merge builds cold once, and after that `main`'s entries warm every pull request.

**Size and eviction.** GitHub caps a repository's caches at 10 GB. Beyond that, entries are evicted least-recently-used, and any entry unused for 7 days is evicted regardless. The sccache entries for both lanes total about 130 MB.

**What invalidates sccache entries:**
- a runner image with a new MSVC, Xcode or AppleClang: every entry for that lane misses, because sccache hashes the compiler;
- a submodule bump: that library's objects miss;
- a change to a widely included header: everything that includes it misses;
- a change to a global compile flag: every compile it touches misses;
- an sccache upgrade: it may invalidate everything.

### Reading `sccache --show-stats`

- **Cache hits and misses:** a warm run should be close to all hits.
- **Non-cacheable compilations:** sccache lists the reason for each. The known cases:
  - **MSVC compiles that write a PDB (`/Zi`).** Release has none: the top-level `cmake_minimum_required` makes policy CMP0141 NEW, so debug-info flags apply only to Debug and RelWithDebInfo.
  - **Precompiled headers.** Only glslang's `MachineIndependent` target uses them, so the Windows configure sets `CMAKE_DISABLE_PRECOMPILE_HEADERS=ON`. Local builds are unaffected.
  - **CMake's C++20 module scanning** (`-interface`). One object, never cached.
- **Cache write errors:** an object that failed to upload misses once on the next run, then gets stored.

## When a run fails

**Downloads.** The configure step downloads sources from GitHub without authentication:
- release tarballs for zlib, curl, Vulkan-Headers, volk and glslang (on Windows);
- git clones of tomlplusplus and doctest (both lanes).

CMake retries each about six times, waiting 5 to 90 seconds. During GitHub incidents these can still fail with timeouts or 504s and "Each download failed!". That isn't a code failure. Re-run only the failed job:

```sh
gh run rerun <run-id> --failed
```

**Windows tests that hang until the 600 s timeout.** This usually means an executable couldn't load a DLL. The Windows loader shows an error dialog on the runner's desktop session and waits for it to be dismissed. Run the executable directly from bash to see the `0xC0000135` exit code immediately.

**A stack overflow test that kills `caliper_tests` with no output on Windows.** The crash guard must re-arm the thread's guard page after a contained overflow (`_resetstkoflw` in `src/host/crash_guard.cpp`). Without that, a second overflow on the same thread ends the process.
