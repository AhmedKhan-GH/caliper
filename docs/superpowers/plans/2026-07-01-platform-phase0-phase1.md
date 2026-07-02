# Caliper Platform — Phase 0 (SDK Extraction) + Phase 1 (ABI Epoch 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute PLATFORM.md §17 Phase 0 and Phase 1: extract an installable in-tree SDK package, then replace the six-dlsym v1 applet ABI with the epoch-2 contract (single descriptor export + `get_service` registry + manifest-gated loading + crash guard), porting all three applets and deleting the v1 loader.

**Architecture:** The SDK becomes `sdk/include/caliper/` exposed as an installable CMake package (`caliper::sdk`); the UI stack stays a monorepo target (`caliper::ui_stack`) until Phase 3. The host gains a `src/host/` library (manifest parser, negotiation, crash guard, watchdog, loader v2) that is unit-testable without OpenGL, plus service tables (`caliper.ui.v1`, `caliper.log.v1`) wired in `main.cpp`. Applets are rewritten to a header-only sugar layer (`caliper.hpp` + `CALIPER_APPLET` macro) and each ships a `<name>.caliper.toml` manifest checked **before** `dlopen`.

**Tech Stack:** C++20 (SDK consumers + host lib; root default stays C++17), CMake ≥3.18, toml++ v3.4.0 (FetchContent), doctest 2.4.11 (FetchContent), existing third_party submodules (imgui/implot/implot3d/glfw/glew), macOS arm64 as the verification platform.

## Global Constraints

- **Reference spec:** `PLATFORM.md` §6 (contract), §7.1 (log), §6d (ui), §10.3 (manifest), §14 (negotiation order), §15 (crash guard/watchdog), §16 (testing), §17 (phase definitions). Where this plan deviates, the deviation is listed in "Spec Deviations" at the end — do not silently re-deviate.
- **TDD:** every `src/host/` unit and the sugar layer get a failing test before implementation. UI glue (`main.cpp`, card text) and CMake plumbing are verified by build + scripted checks + a manual demo checklist instead (per the standing rule: no change-detector tests, no tests for glue).
- **Every task ends green:** `cmake --build build` succeeds AND `ctest --test-dir build --output-on-failure` passes AND (where stated) the app launches. Never commit red.
- **Docs ride along (docs-as-code):** the wiki lives in `docs/wiki/` (MkDocs Material — see the "Documentation Track" section near the end of this plan). Every task listed in that section's mapping table updates its page(s) **in the same commit as the code**. From the docs scaffold onward, `mkdocs build --strict` must pass whenever docs change. Contract text (headers, manifests, refusal strings) is embedded from the real source files via snippets, never hand-copied.
- **Build directory:** use `build/` (not CLion's `cmake-build-debug/`). First configure: `cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON`. libtorch/submodules are already present locally, so configure is fast.
- **Branches:** Phase 0 on `platform/phase-0` (Tasks 1–4), then merge to `main`; Phase 1 on `platform/phase-1` (Tasks 5–17), merge at the end. Both branch from `main` (tip `a290ced` or later).
- **Commits:** conventional style matching repo history (`feat(sdk): …`, `test(host): …`, `refactor(applets): …`). Every commit message body ends with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- **Versions (fixed):** ABI epoch = `2`; host version = `0.6.0` = `(0<<16)|(6<<8)|0`; SDK package version = `0.1.0`; service ids `"caliper.ui.v1"`, `"caliper.log.v1"`; manifest filename pattern `<stem>.caliper.toml`; descriptor export symbol `caliper_applet_descriptor`.
- **Applet ids (fixed):** `dev.caliper.hello`, `dev.ahmed.circuitnet`, `dev.ahmed.opengllama`, `dev.ahmed.repnet-demo`.
- **Do not touch:** `third_party/` submodules, `intro_screen.cpp/.h`, applet internal logic (only `plugin.cpp`, manifests, and CMake change in ports), `PLATFORM.md` (except nothing), `demos/`, `dump/`.

## File Map (what exists when done)

```
sdk/
  include/caliper/abi_v1.h            T1: moved v1 header (deleted in T17)
  include/caliper/abi.h               T5: epoch-2 frozen ABI
  include/caliper/services/ui_v1.h    T5
  include/caliper/services/log_v1.h   T5
  include/caliper/caliper.hpp         T10: header-only sugar
  testing/caliper/fixture_host.h      T10: headless fake host (tests)
  cmake/caliper-sdk-config.cmake      T2: package config
src/
  applet_api.h                        T1: compat shim (deleted in T17)
  applet_host.h / applet_host.cpp     unchanged until deleted in T17
  host/applet_manifest.h/.cpp         T6
  host/negotiation.h/.cpp             T7
  host/crash_guard.h/.cpp             T8
  host/frame_watchdog.h               T9
  host/applet_loader.h/.cpp           T12
  host/host_services.h/.cpp           T13 (ui/log service tables; needs ImGui)
  host/host_version.h                 T5
  main.cpp                            T13: loader v2 + services + FrameInfo
examples/hello/CMakeLists.txt         T11
examples/hello/hello.cpp              T11
examples/hello/hello.caliper.toml     T11
examples/signal_scope/*               already authored (exemplar; see its README) — wired into the build at T11
tests/CMakeLists.txt                  T5 (grows through T12)
tests/test_abi.cpp  tests/abi_c_check.c            T5
tests/test_manifest.cpp               T6
tests/test_negotiation.cpp            T7
tests/test_crash_guard.cpp            T8
tests/test_watchdog.cpp               T9
tests/test_sugar.cpp                  T10
tests/test_loader.cpp                 T12
tests/sdk_install_probe/CMakeLists.txt + probe.cpp  T4
scripts/test-sdk-install.sh           T4
applets/{circuitnet,opengllama,repnet_demo}/plugin.cpp        rewritten T14–T16
applets/{circuitnet,opengllama,repnet_demo}/<name>.caliper.toml  new T14–T16
applets/*/CMakeLists.txt              edited T3 and T14–T16
CMakeLists.txt                        edited T2, T3, T5, T11, T13, T17
```

---

# PHASE 0 — SDK extraction in-tree (Tasks 1–4)

### Task 1: Move the ABI header into `sdk/`, leave a shim

**Files:**
- Create: `sdk/include/caliper/abi_v1.h` (git mv from `src/applet_api.h`)
- Create: `src/applet_api.h` (new shim)

**Interfaces:**
- Produces: `<caliper/abi_v1.h>` include path (used by T3 applet edits); shim keeps `"applet_api.h"` working for host sources until T17.

- [ ] **Step 1: Create the branch**

```bash
cd /Users/ahmed/CLionProjects/caliper
git checkout main && git checkout -b platform/phase-0
```

- [ ] **Step 2: Move the header with history**

```bash
mkdir -p sdk/include/caliper
git mv src/applet_api.h sdk/include/caliper/abi_v1.h
```

- [ ] **Step 3: Create the shim at the old path**

Create `src/applet_api.h` with exactly:

```cpp
#pragma once
// COMPAT SHIM — the v1 ABI now lives in the SDK. This file exists only so
// host sources keep compiling until the Phase 1 port completes (PLATFORM.md
// §17 Phase 0/1). Deleted at the end of Phase 1. Do not add anything here.
#include "../sdk/include/caliper/abi_v1.h"
```

- [ ] **Step 4: Verify the build is unchanged**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j
```
Expected: builds clean; `build/applets/` contains `libcircuitnet.dylib`, `libopengllama.dylib`, `librepnet_demo.dylib`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor(sdk): move applet_api.h to sdk/include/caliper/abi_v1.h with compat shim"
```

---

### Task 2: `caliper::sdk` + `caliper::ui_stack` + `caliper_app_paths` targets, install rules

**Files:**
- Modify: `CMakeLists.txt:74-94` (replace the `caliper_applet_sdk` block; keep the old target as an alias-consumer until T3)
- Create: `sdk/cmake/caliper-sdk-config.cmake` (name must match `find_package(caliper-sdk)` lookup rules: `<name>-config.cmake`)

**Interfaces:**
- Produces: CMake targets `caliper::sdk` (INTERFACE, sdk headers), `caliper::ui_stack` (INTERFACE, pinned UI includes+libs), `caliper_app_paths` (STATIC, `src/app_paths.cpp`); install component `sdk`; package `caliper-sdk` version `0.1.0`. Consumed by T3, T4, T5, T10, T11, T13–T16.

- [ ] **Step 1: Replace the SDK section of the root CMakeLists**

Replace lines 74–94 (the `Applet SDK` section defining `caliper_applet_sdk`) with:

```cmake
# ============================================================================
# Caliper SDK (PLATFORM.md §17 Phase 0: installable package, in-tree source)
# ============================================================================

set(CALIPER_SDK_VERSION 0.1.0)

add_library(caliper_sdk INTERFACE)
add_library(caliper::sdk ALIAS caliper_sdk)
target_include_directories(caliper_sdk INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/sdk/include>
    $<INSTALL_INTERFACE:include>
)

# UI stack: pinned imgui/implot/implot3d/ImGuiFileDialog from the monorepo.
# Moves into the SDK repo at Phase 3; a separate target so the seam is visible.
add_library(caliper_ui_stack INTERFACE)
add_library(caliper::ui_stack ALIAS caliper_ui_stack)
target_include_directories(caliper_ui_stack INTERFACE
    ${THIRD_PARTY_DIR}/imgui
    ${THIRD_PARTY_DIR}/imgui/backends
    ${THIRD_PARTY_DIR}/implot
    ${THIRD_PARTY_DIR}/implot3d
    ${THIRD_PARTY_DIR}/ImGuiFileDialog
)
target_link_libraries(caliper_ui_stack INTERFACE
    imgui implot implot3d ImGuiFileDialog
)
target_compile_definitions(caliper_ui_stack INTERFACE
    IMGUI_IMPL_OPENGL_LOADER_GLEW
)

# Host-owned app-paths utility. Applets that still need the shared data dir
# link this TARGET instead of compiling src/app_paths.cpp by path.
# Dies when applets move to CaliperHost.applet_data_dir (Phase 1+).
add_library(caliper_app_paths STATIC src/app_paths.cpp)
target_include_directories(caliper_app_paths PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(caliper_app_paths PROPERTIES POSITION_INDEPENDENT_CODE ON)

# TRANSITIONAL (deleted in Task 3): old monolithic SDK target.
add_library(caliper_applet_sdk INTERFACE)
target_link_libraries(caliper_applet_sdk INTERFACE caliper::sdk caliper::ui_stack)
target_include_directories(caliper_applet_sdk INTERFACE
    ${CMAKE_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}/duckdb/src/include
)
target_compile_definitions(caliper_applet_sdk INTERFACE DUCKDB_BUILD_LIBRARY)

# ---- SDK install rules (component: sdk) ----
include(CMakePackageConfigHelpers)
install(DIRECTORY sdk/include/ DESTINATION include COMPONENT sdk)
# Without EXPORT_NAME the installed export would be caliper::caliper_sdk —
# the alias only exists in the build tree (caught by the Task 4 exit proof).
set_target_properties(caliper_sdk PROPERTIES EXPORT_NAME sdk)
install(TARGETS caliper_sdk EXPORT CaliperSDKTargets COMPONENT sdk)
install(EXPORT CaliperSDKTargets NAMESPACE caliper::
        DESTINATION lib/cmake/caliper-sdk COMPONENT sdk)
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/caliper-sdk-config-version.cmake"
    VERSION ${CALIPER_SDK_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
    "${CMAKE_SOURCE_DIR}/sdk/cmake/caliper-sdk-config.cmake"
    "${CMAKE_BINARY_DIR}/caliper-sdk-config-version.cmake"
    DESTINATION lib/cmake/caliper-sdk COMPONENT sdk)
```

- [ ] **Step 2: Create `sdk/cmake/caliper-sdk-config.cmake`**

```cmake
# caliper-sdk CMake package entry point.
# Phase 0 scope: ABI headers only. The UI stack joins the package at Phase 3
# (PLATFORM.md §17); until then applets in this repo use caliper::ui_stack.
include("${CMAKE_CURRENT_LIST_DIR}/CaliperSDKTargets.cmake")
```

(For package name `caliper-sdk`, CMake searches for `caliper-sdk-config.cmake` or `caliper-sdkConfig.cmake` — the lowercase-dashed form above is the conventional one; the version file `caliper-sdk-config-version.cmake` pairs with it.)

- [ ] **Step 3: Build + run check**

```bash
cmake -B build && cmake --build build -j && ./build/caliper &
sleep 5 && kill %1
```
Expected: build clean, app opens with 3 applet cards (visual check), no behavior change.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "build(sdk): caliper::sdk + caliper::ui_stack targets with install rules and package config"
```

---

### Task 3: Repoint all three applets; delete the transitional target

**Files:**
- Modify: `applets/circuitnet/CMakeLists.txt`, `applets/circuitnet/plugin.cpp:1`
- Modify: `applets/opengllama/CMakeLists.txt`, `applets/opengllama/plugin.cpp:1`
- Modify: `applets/repnet_demo/CMakeLists.txt`, `applets/repnet_demo/plugin.cpp:1`
- Modify: `CMakeLists.txt` (remove the TRANSITIONAL `caliper_applet_sdk` block from Task 2)

**Interfaces:**
- Consumes: `caliper::sdk`, `caliper::ui_stack`, `caliper_app_paths` (Task 2).
- Produces: applet CMake with zero `${CMAKE_SOURCE_DIR}/src` references — the Phase 0 exit criterion.

- [ ] **Step 1: circuitnet CMake**

In `applets/circuitnet/CMakeLists.txt`: remove `${CMAKE_SOURCE_DIR}/src/app_paths.cpp` from sources; replace the include/link sections:

```cmake
add_library(circuitnet SHARED
    plugin.cpp
    circuitnet.cpp
    verilog_parser.cpp
    circuit_db.cpp
    circuit_viz.cpp
)

target_include_directories(circuitnet PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(circuitnet PRIVATE
    caliper::sdk
    caliper::ui_stack
    caliper_app_paths
    imgui-node-editor
    duckdb_static
    duckdb_generated_extension_loader
    parquet_extension
    core_functions_extension
)
```
(keep the `CALIPER_APPLET_EXPORT` definition and `set_target_properties` block as-is).

- [ ] **Step 2: opengllama CMake**

Same pattern: remove `${CMAKE_SOURCE_DIR}/src/app_paths.cpp` from sources and `${CMAKE_SOURCE_DIR}/src` from include dirs; link list becomes `caliper::sdk caliper::ui_stack caliper_app_paths llama libglew_static cpp-httplib`. The two `third_party/llama.cpp/...` include paths **stay** (heavy-dep decoupling is Phase 4, not Phase 0).

- [ ] **Step 3: repnet_demo CMake**

Remove `${CMAKE_SOURCE_DIR}/src` from `target_include_directories` (keep `${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor`); in `target_link_libraries` replace `caliper_applet_sdk` with `caliper::sdk caliper::ui_stack`. (repnet_demo compiles its own local `app_paths.cpp` copy — leave it; it's applet-internal.)

- [ ] **Step 4: Switch the three plugin.cpp includes**

In each of `applets/{circuitnet,opengllama,repnet_demo}/plugin.cpp` line 1:
`#include "applet_api.h"` → `#include <caliper/abi_v1.h>`

Then sweep for any other reference:
```bash
grep -rn 'applet_api.h' applets/ || echo CLEAN
```
Expected: `CLEAN` (if any applet header includes it, apply the same swap there).

- [ ] **Step 5: Delete the transitional target**

Remove the `# TRANSITIONAL (deleted in Task 3)` block (the whole `caliper_applet_sdk` definition) from the root `CMakeLists.txt`.

- [ ] **Step 6: Verify the exit criterion**

```bash
grep -rn 'CMAKE_SOURCE_DIR}/src' applets/*/CMakeLists.txt || echo "PHASE0-EXIT-OK"
cmake -B build && cmake --build build -j
```
Expected: `PHASE0-EXIT-OK`; clean build; app still runs with 3 cards.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "refactor(applets): consume caliper::sdk/ui_stack targets; no src/ path reach-ins"
```

---

### Task 4: Installed-prefix proof (`find_package` probe)

**Files:**
- Create: `tests/sdk_install_probe/CMakeLists.txt`, `tests/sdk_install_probe/probe.cpp`
- Create: `scripts/test-sdk-install.sh` (chmod +x)

**Interfaces:**
- Consumes: install component `sdk` (Task 2).
- Produces: the scripted Phase 0 exit proof; rerun after any SDK CMake change.

- [ ] **Step 1: Probe project**

`tests/sdk_install_probe/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.18)
project(sdk_install_probe CXX)
find_package(caliper-sdk 0.1 CONFIG REQUIRED)
add_library(probe STATIC probe.cpp)
target_link_libraries(probe PRIVATE caliper::sdk)
set_target_properties(probe PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
```

`tests/sdk_install_probe/probe.cpp`:
```cpp
// Compile-only proof that the installed caliper-sdk package is self-contained.
#include <caliper/abi_v1.h>
static_assert(CALIPER_APPLET_ABI == 1, "v1 header reachable from installed prefix");
```

- [ ] **Step 2: The script**

`scripts/test-sdk-install.sh`:
```bash
#!/usr/bin/env bash
# Phase 0 exit proof (PLATFORM.md §17): the SDK installs to a prefix and a
# standalone consumer builds against it via find_package — no monorepo paths.
set -euo pipefail
BUILD_DIR="${1:-build}"
PREFIX="$(mktemp -d)"
trap 'rm -rf "$PREFIX"' EXIT
cmake --install "$BUILD_DIR" --component sdk --prefix "$PREFIX" >/dev/null
cmake -S tests/sdk_install_probe -B "$PREFIX/probe-build" \
      -DCMAKE_PREFIX_PATH="$PREFIX" >/dev/null
cmake --build "$PREFIX/probe-build" >/dev/null
echo "sdk-install-probe: OK (prefix consumable via find_package(caliper-sdk))"
```

- [ ] **Step 3: Run it**

```bash
chmod +x scripts/test-sdk-install.sh && ./scripts/test-sdk-install.sh build
```
Expected: `sdk-install-probe: OK …`.

- [ ] **Step 4: Commit, merge Phase 0**

```bash
git add -A && git commit -m "test(sdk): installed-prefix find_package probe — Phase 0 exit proof"
git checkout main && git merge --no-ff platform/phase-0 -m "Phase 0: SDK extracted in-tree (PLATFORM.md §17)"
```

---

# PHASE 1 — ABI epoch 2 (Tasks 5–17)

### Task 5: Epoch-2 ABI headers + test infrastructure

**Files:**
- Create: `sdk/include/caliper/abi.h`, `sdk/include/caliper/services/ui_v1.h`, `sdk/include/caliper/services/log_v1.h`, `src/host/host_version.h`
- Create: `tests/CMakeLists.txt`, `tests/test_main.cpp`, `tests/test_abi.cpp`, `tests/abi_c_check.c`
- Modify: `CMakeLists.txt` (tests hookup)

**Interfaces:**
- Produces: `CaliperHost`, `CaliperFrameInfo`, `CaliperAppletAPI`, `CaliperAppletDescriptor`, `CALIPER_ABI_EPOCH`, `CALIPER_DESCRIPTOR_SYMBOL`, `CaliperUiV1`/`CALIPER_UI_V1`, `CaliperLogV1`/`CALIPER_LOG_V1`/`CaliperLogLevel`, `caliper_host::kHostVersionU32`/`kHostVersionStr`. Test target `caliper_tests`. Consumed by every later task.

- [ ] **Step 1: Branch; write the failing test first**

```bash
git checkout -b platform/phase-1
```

`tests/test_main.cpp`:
```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
```

`tests/test_abi.cpp`:
```cpp
#include <doctest/doctest.h>
#include <caliper/abi.h>
#include <caliper/services/ui_v1.h>
#include <caliper/services/log_v1.h>
#include <cstddef>
#include <type_traits>

// ABI hygiene (PLATFORM.md §6c): POD, struct_size-prefixed, C-safe.
static_assert(std::is_standard_layout_v<CaliperHost>);
static_assert(std::is_standard_layout_v<CaliperFrameInfo>);
static_assert(std::is_standard_layout_v<CaliperAppletAPI>);
static_assert(std::is_standard_layout_v<CaliperAppletDescriptor>);
static_assert(std::is_standard_layout_v<CaliperUiV1>);
static_assert(std::is_standard_layout_v<CaliperLogV1>);
static_assert(offsetof(CaliperHost, struct_size) == 0);
static_assert(offsetof(CaliperFrameInfo, struct_size) == 0);
static_assert(offsetof(CaliperAppletAPI, struct_size) == 0);
static_assert(offsetof(CaliperAppletDescriptor, struct_size) == 0);
static_assert(CALIPER_ABI_EPOCH == 2);

TEST_CASE("abi: descriptor symbol name is fixed") {
    CHECK(std::string(CALIPER_DESCRIPTOR_SYMBOL) == "caliper_applet_descriptor");
}
```

`tests/abi_c_check.c` (catches C++-isms leaking into the C ABI):
```c
#include <caliper/abi.h>
#include <caliper/services/ui_v1.h>
#include <caliper/services/log_v1.h>
int caliper_abi_c_check_anchor(void) { return CALIPER_ABI_EPOCH; }
```

`tests/CMakeLists.txt`:
```cmake
include(FetchContent)
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(doctest)

add_executable(caliper_tests
    test_main.cpp
    test_abi.cpp
    abi_c_check.c
)
target_link_libraries(caliper_tests PRIVATE doctest::doctest caliper::sdk)
set_target_properties(caliper_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_test(NAME caliper_tests COMMAND caliper_tests)
```

Root `CMakeLists.txt` — replace the bare `option(BUILD_TESTS "Build tests" OFF)` line's downstream (there is none today) by appending at the END of the file:
```cmake
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

- [ ] **Step 2: Run to verify it fails**

```bash
cmake -B build -DBUILD_TESTS=ON && cmake --build build --target caliper_tests
```
Expected: FAIL — `caliper/abi.h: No such file or directory`.

- [ ] **Step 3: Write the headers**

`sdk/include/caliper/abi.h` — the frozen contract (PLATFORM.md §6a/§6b), exactly:
```c
#pragma once
/* Caliper ABI — epoch 2. FROZEN: any change here is an epoch bump (§14).
 * C types only; no STL, exceptions, or third-party types (§6c). */
#include <stdint.h>
#include <stdbool.h>

#define CALIPER_ABI_EPOCH 2
#define CALIPER_DESCRIPTOR_SYMBOL "caliper_applet_descriptor"

#if defined(_WIN32)
  #define CALIPER_EXPORT __declspec(dllexport)
#else
  #define CALIPER_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The host is a service registry, not a struct of fields (§6b). */
typedef struct CaliperHost {
    uint32_t    struct_size;      /* sizeof(CaliperHost) as the host built it */
    uint32_t    abi_epoch;        /* epoch this host speaks */
    uint32_t    host_version;     /* (major<<16)|(minor<<8)|patch */
    const char* applet_data_dir;  /* per-applet sandbox dir, UTF-8, host-owned */
    /* Returns a service table or NULL; pointer valid for the applet's
     * lifetime. Unknown ids return NULL — never UB. */
    const void* (*get_service)(const struct CaliperHost* host,
                               const char* service_id);
} CaliperHost;

/* Pixel-space contract (§6a): fb_* are PHYSICAL framebuffer pixels; ImGui
 * coordinates are logical units; physical = logical * dpi_scale. */
typedef struct CaliperFrameInfo {
    uint32_t struct_size;
    int32_t  fb_width;
    int32_t  fb_height;
    float    dpi_scale;
    double   time_sec;
    double   delta_sec;
} CaliperFrameInfo;

typedef struct CaliperAppletAPI {
    uint32_t struct_size;
    void* (*create)(void);
    void  (*destroy)(void* self);
    bool  (*initialize)(void* self, const CaliperHost* host);
    void  (*frame)(void* self, const CaliperFrameInfo* info);
    void  (*cleanup)(void* self);
    /* future entry points are APPENDED here, guarded by struct_size */
} CaliperAppletAPI;

typedef struct CaliperAppletDescriptor {
    uint32_t struct_size;
    uint32_t abi_epoch;                   /* must equal a host-supported epoch */
    const char* id;                       /* reverse-DNS, matches manifest */
    const char* version;                  /* applet semver, matches manifest */
    const char* name;
    const char* summary;
    const char* tag;
    const char* const* required_services; /* NULL-terminated; may be NULL */
    CaliperAppletAPI api;
} CaliperAppletDescriptor;

/* Every epoch-2 applet exports exactly one symbol:
 *   const CaliperAppletDescriptor* caliper_applet_descriptor(void);
 * (generated by CALIPER_APPLET in caliper.hpp, or hand-written in C). */

#ifdef __cplusplus
}
#endif
```

`sdk/include/caliper/services/log_v1.h`:
```c
#pragma once
/* caliper.log.v1 — structured logs into the host console (PLATFORM.md §7.1).
 * IMMUTABLE once published: new capability = log_v2, alongside. */
#include <stdint.h>

#define CALIPER_LOG_V1 "caliper.log.v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CaliperLogLevel {
    CALIPER_LOG_DEBUG = 0,
    CALIPER_LOG_INFO  = 1,
    CALIPER_LOG_WARN  = 2,
    CALIPER_LOG_ERROR = 3
} CaliperLogLevel;

typedef struct CaliperLogV1 {
    uint32_t struct_size;
    void (*log)(CaliperLogLevel level, const char* message_utf8); /* pre-formatted */
} CaliperLogV1;

#ifdef __cplusplus
}
#endif
```

`sdk/include/caliper/services/ui_v1.h`:
```c
#pragma once
/* caliper.ui.v1 — ImGui/ImPlot/ImPlot3D contexts + allocators (§6d).
 * The allocator handoff is what makes context-sharing across the DLL
 * boundary sound (Dear ImGui's own DLL guidance). IMMUTABLE once published.
 * Function-pointer typedefs mirror ImGuiMemAllocFunc/ImGuiMemFreeFunc
 * layout-exactly, without pulling imgui.h into the C ABI (§6c). */
#include <stdint.h>
#include <stddef.h>

#define CALIPER_UI_V1 "caliper.ui.v1"

#ifdef __cplusplus
extern "C" {
#endif

struct ImGuiContext;
struct ImPlotContext;
struct ImPlot3DContext;

typedef void* (*CaliperImGuiAllocFn)(size_t sz, void* user_data);
typedef void  (*CaliperImGuiFreeFn)(void* ptr, void* user_data);

typedef struct CaliperUiV1 {
    uint32_t struct_size;
    struct ImGuiContext*    (*imgui_context)(void);
    struct ImPlotContext*   (*implot_context)(void);
    struct ImPlot3DContext* (*implot3d_context)(void);
    /* Host's allocator pair — the applet side MUST install these into its
     * copy of ImGui's globals so every allocation lands on the host heap. */
    void (*imgui_allocators)(CaliperImGuiAllocFn* out_alloc,
                             CaliperImGuiFreeFn*  out_free,
                             void** out_user_data);
} CaliperUiV1;

#ifdef __cplusplus
}
#endif
```

`src/host/host_version.h`:
```cpp
#pragma once
#include <cstdint>
namespace caliper_host {
inline constexpr uint32_t kHostVersionU32 = (0u << 16) | (6u << 8) | 0u; // 0.6.0
inline constexpr const char* kHostVersionStr = "0.6.0";
}
```

- [ ] **Step 4: Run to verify it passes**

```bash
cmake --build build --target caliper_tests && ctest --test-dir build --output-on-failure
```
Expected: `100% tests passed, 0 tests failed out of 1`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(sdk): ABI epoch 2 headers (descriptor, service registry, ui/log v1) + test infra"
```

---

### Task 6: Manifest parser (`caliper.toml`) — TDD

**Files:**
- Create: `src/host/applet_manifest.h`, `src/host/applet_manifest.cpp`
- Create: `tests/test_manifest.cpp`
- Modify: `CMakeLists.txt` (toml++ FetchContent + `caliper_host_lib`), `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `caliper_host::AppletManifest{id,name,version,summary,tag,abi_epoch,min_host,required_services,optional_services}`, `ManifestResult{ok,manifest,error}`, `parse_manifest_text(const std::string&)`, `parse_manifest_file(const std::string&)`, `is_valid_semver(const std::string&)`. CMake target `caliper_host_lib` (STATIC, C++20) that T7/T8/T12 grow. Consumed by negotiation (T7), loader (T12), main (T13).

- [ ] **Step 1: Write the failing tests**

`tests/test_manifest.cpp`:
```cpp
#include <doctest/doctest.h>
#include "applet_manifest.h"
using namespace caliper_host;

static const char* kGolden = R"([applet]
id = "dev.ahmed.circuitnet"
name = "CircuitNet 3.0"
version = "1.0.0"
summary = "Gate-level circuit explorer"
tag = "EDA"

[compat]
abi_epoch = 2
min_host = "0.6.0"

[services]
required = ["caliper.ui.v1"]
optional = ["caliper.log.v1"]
)";

TEST_CASE("manifest: golden parses fully") {
    auto r = parse_manifest_text(kGolden);
    REQUIRE(r.ok);
    CHECK(r.manifest.id == "dev.ahmed.circuitnet");
    CHECK(r.manifest.name == "CircuitNet 3.0");
    CHECK(r.manifest.version == "1.0.0");
    CHECK(r.manifest.summary == "Gate-level circuit explorer");
    CHECK(r.manifest.tag == "EDA");
    CHECK(r.manifest.abi_epoch == 2);
    CHECK(r.manifest.min_host == "0.6.0");
    REQUIRE(r.manifest.required_services.size() == 1);
    CHECK(r.manifest.required_services[0] == "caliper.ui.v1");
    REQUIRE(r.manifest.optional_services.size() == 1);
    CHECK(r.manifest.optional_services[0] == "caliper.log.v1");
}

TEST_CASE("manifest: minimal — only id/name/version/epoch required") {
    auto r = parse_manifest_text(
        "[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n"
        "[compat]\nabi_epoch=2\n");
    REQUIRE(r.ok);
    CHECK(r.manifest.min_host.empty());
    CHECK(r.manifest.required_services.empty());
    CHECK(r.manifest.summary.empty());
}

TEST_CASE("manifest: adversarial inputs refuse with a reason") {
    struct Case { const char* toml; const char* needle; };
    const Case cases[] = {
        {"", "missing"},                                              // empty
        {"not toml {{{", "parse"},                                    // syntax
        {"[applet]\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=2\n", "id"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\n[compat]\nabi_epoch=2\n", "version"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"1.0\"\n[compat]\nabi_epoch=2\n", "semver"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n", "abi_epoch"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=\"two\"\n", "abi_epoch"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=2\nmin_host=\"soon\"\n", "semver"},
    };
    for (auto& c : cases) {
        auto r = parse_manifest_text(c.toml);
        CAPTURE(c.toml);
        CHECK_FALSE(r.ok);
        CHECK(r.error.find(c.needle) != std::string::npos);
    }
}

TEST_CASE("manifest: unknown tables/keys are ignored (forward compat)") {
    auto r = parse_manifest_text(
        "[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\nauthors=[\"x\"]\n"
        "[compat]\nabi_epoch=2\n[future]\nx=1\n");
    CHECK(r.ok);
}

TEST_CASE("semver validation") {
    CHECK(is_valid_semver("0.6.0"));
    CHECK(is_valid_semver("10.20.30"));
    CHECK_FALSE(is_valid_semver("1.0"));
    CHECK_FALSE(is_valid_semver("v1.0.0"));
    CHECK_FALSE(is_valid_semver(""));
}
```

CMake — root `CMakeLists.txt`, after the SDK section add:
```cmake
# Host library: loader/negotiation logic, unit-testable without OpenGL.
include(FetchContent)
FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.4.0 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(tomlplusplus)

add_library(caliper_host_lib STATIC
    src/host/applet_manifest.cpp
)
target_include_directories(caliper_host_lib PUBLIC ${CMAKE_SOURCE_DIR}/src/host)
target_link_libraries(caliper_host_lib
    PUBLIC caliper::sdk
    PRIVATE tomlplusplus::tomlplusplus)
set_target_properties(caliper_host_lib PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
```
`tests/CMakeLists.txt`: add `test_manifest.cpp` to the executable and `caliper_host_lib` to its link list.

- [ ] **Step 2: Run to verify failure**

```bash
cmake -B build -DBUILD_TESTS=ON && cmake --build build --target caliper_tests
```
Expected: FAIL — `applet_manifest.h` not found.

- [ ] **Step 3: Implement**

`src/host/applet_manifest.h`:
```cpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace caliper_host {

// Parsed caliper.toml (PLATFORM.md §10.3). Unknown keys/tables are ignored
// for forward compatibility; missing required fields are errors.
struct AppletManifest {
    std::string id;        // reverse-DNS
    std::string name;
    std::string version;   // strict x.y.z
    std::string summary;
    std::string tag;
    uint32_t    abi_epoch = 0;
    std::string min_host;  // "" = no floor; else strict x.y.z
    std::vector<std::string> required_services;
    std::vector<std::string> optional_services;
};

struct ManifestResult {
    bool ok = false;
    AppletManifest manifest;
    std::string error;     // human-readable, shown on the failure card
};

ManifestResult parse_manifest_text(const std::string& toml_text);
ManifestResult parse_manifest_file(const std::string& path);
bool is_valid_semver(const std::string& v);

} // namespace caliper_host
```

`src/host/applet_manifest.cpp`:
```cpp
#include "applet_manifest.h"
#include <toml++/toml.hpp>
#include <cctype>

namespace caliper_host {

bool is_valid_semver(const std::string& v) {
    int part = 0, digits = 0;
    for (char c : v) {
        if (std::isdigit((unsigned char)c)) { digits++; continue; }
        if (c == '.') {
            if (digits == 0) return false;
            part++; digits = 0; continue;
        }
        return false;
    }
    return part == 2 && digits > 0;
}

namespace {
ManifestResult fail(std::string msg) {
    ManifestResult r; r.error = std::move(msg); return r;
}
std::vector<std::string> read_array(const toml::table& t, const char* key) {
    std::vector<std::string> out;
    if (auto* arr = t[key].as_array())
        for (auto& e : *arr)
            if (auto s = e.value<std::string>()) out.push_back(*s);
    return out;
}
} // namespace

ManifestResult parse_manifest_text(const std::string& toml_text) {
    toml::table root;
    try {
        root = toml::parse(toml_text);
    } catch (const toml::parse_error& e) {
        return fail(std::string("manifest parse error: ") + e.what());
    }

    ManifestResult r;
    auto* applet = root["applet"].as_table();
    if (!applet) return fail("manifest missing [applet] table");

    auto req_str = [&](const char* key, std::string& dst) -> bool {
        if (auto v = (*applet)[key].value<std::string>()) { dst = *v; return true; }
        return false;
    };
    if (!req_str("id", r.manifest.id) || r.manifest.id.empty())
        return fail("manifest missing applet.id");
    if (!req_str("name", r.manifest.name) || r.manifest.name.empty())
        return fail("manifest missing applet.name");
    if (!req_str("version", r.manifest.version))
        return fail("manifest missing applet.version");
    if (!is_valid_semver(r.manifest.version))
        return fail("applet.version is not strict semver x.y.z: " + r.manifest.version);
    req_str("summary", r.manifest.summary);
    req_str("tag", r.manifest.tag);

    auto* compat = root["compat"].as_table();
    if (!compat) return fail("manifest missing [compat].abi_epoch");
    if (auto e = (*compat)["abi_epoch"].value<int64_t>(); e && *e >= 1)
        r.manifest.abi_epoch = (uint32_t)*e;
    else
        return fail("manifest missing or invalid [compat].abi_epoch (integer >= 1)");
    if (auto mh = (*compat)["min_host"].value<std::string>()) {
        if (!is_valid_semver(*mh))
            return fail("compat.min_host is not strict semver x.y.z: " + *mh);
        r.manifest.min_host = *mh;
    }

    if (auto* services = root["services"].as_table()) {
        r.manifest.required_services = read_array(*services, "required");
        r.manifest.optional_services = read_array(*services, "optional");
    }

    r.ok = true;
    return r;
}

ManifestResult parse_manifest_file(const std::string& path) {
    toml::table root;
    try {
        root = toml::parse_file(path);
    } catch (const toml::parse_error& e) {
        return fail(std::string("manifest parse error: ") + e.what());
    } catch (...) {
        return fail("manifest unreadable: " + path);
    }
    std::ostringstream oss; oss << toml::toml_formatter(root);
    return parse_manifest_text(oss.str());
}

} // namespace caliper_host
```
(Note the `parse_manifest_file` round-trip through a formatter keeps one validation path; include `<sstream>`.)

- [ ] **Step 4: Run to verify pass**

```bash
cmake --build build --target caliper_tests && ./build/tests/caliper_tests --test-case="manifest*,semver*"
```
Expected: all manifest/semver cases pass; full `ctest --test-dir build` green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(host): caliper.toml manifest parser with golden+adversarial tests"
```

---

### Task 7: Negotiation — TDD

**Files:**
- Create: `src/host/negotiation.h`, `src/host/negotiation.cpp`
- Create: `tests/test_negotiation.cpp`
- Modify: root `CMakeLists.txt` (add `negotiation.cpp` to `caliper_host_lib`), `tests/CMakeLists.txt` (add test file)

**Interfaces:**
- Consumes: `AppletManifest` (T6).
- Produces: `caliper_host::HostCaps{abi_epoch,version,services}`, `Negotiation{ok,reason}`, `negotiate(const AppletManifest&, const HostCaps&)`, `semver_cmp(a,b)→int`. Consumed by loader (T12) and main (T13). **Reason strings below are contractual** — the loader tests and cards reuse them.

- [ ] **Step 1: Write the failing tests**

`tests/test_negotiation.cpp`:
```cpp
#include <doctest/doctest.h>
#include "negotiation.h"
using namespace caliper_host;

static AppletManifest base() {
    AppletManifest m;
    m.id = "a.b"; m.name = "A"; m.version = "1.0.0";
    m.abi_epoch = 2; m.min_host = "0.6.0";
    m.required_services = {"caliper.ui.v1"};
    return m;
}
static HostCaps host() {
    return HostCaps{2, "0.6.0", {"caliper.ui.v1", "caliper.log.v1"}};
}

TEST_CASE("negotiate: compatible applet passes") {
    auto n = negotiate(base(), host());
    CHECK(n.ok);
    CHECK(n.reason.empty());
}

TEST_CASE("negotiate: epoch mismatch → friendly reason") {
    auto m = base(); m.abi_epoch = 1;
    auto n = negotiate(m, host());
    CHECK_FALSE(n.ok);
    CHECK(n.reason ==
        "Built for ABI epoch 1; this host speaks 2 — check for an applet update.");
}

TEST_CASE("negotiate: min_host newer than host → refuse") {
    auto m = base(); m.min_host = "9.9.9";
    auto n = negotiate(m, host());
    CHECK_FALSE(n.ok);
    CHECK(n.reason == "Requires host 9.9.9 or newer; this host is 0.6.0.");
}

TEST_CASE("negotiate: missing required service → refuse, first missing named") {
    auto m = base();
    m.required_services = {"caliper.ui.v1", "caliper.jobs.v1", "caliper.metrics.v1"};
    auto n = negotiate(m, host());
    CHECK_FALSE(n.ok);
    CHECK(n.reason ==
        "Requires a capability this host doesn't have: caliper.jobs.v1.");
}

TEST_CASE("negotiate: empty min_host means no floor") {
    auto m = base(); m.min_host.clear();
    CHECK(negotiate(m, host()).ok);
}

TEST_CASE("semver_cmp is numeric, not lexical") {
    CHECK(semver_cmp("0.6.0",  "0.6.0")  == 0);
    CHECK(semver_cmp("0.6.0",  "0.10.0") <  0);   // lexical would say >
    CHECK(semver_cmp("1.0.0",  "0.9.9")  >  0);
    CHECK(semver_cmp("0.6.1",  "0.6.0")  >  0);
}
```

- [ ] **Step 2: Run to verify failure** — `cmake --build build --target caliper_tests` → FAIL (`negotiation.h` missing).

- [ ] **Step 3: Implement**

`src/host/negotiation.h`:
```cpp
#pragma once
#include "applet_manifest.h"
#include <set>
#include <string>

namespace caliper_host {

struct HostCaps {
    uint32_t abi_epoch;
    std::string version;              // host semver, e.g. "0.6.0"
    std::set<std::string> services;   // ids this host can vend
};

struct Negotiation {
    bool ok = false;
    std::string reason;               // friendly card text when !ok
};

// PLATFORM.md §14 order (Phase-1 subset — packs/platform checks arrive
// Phase 4): epoch supported → min_host satisfied → required services present.
Negotiation negotiate(const AppletManifest& m, const HostCaps& caps);

int semver_cmp(const std::string& a, const std::string& b); // <0, 0, >0
}
```

`src/host/negotiation.cpp`:
```cpp
#include "negotiation.h"
#include <cstdio>

namespace caliper_host {

int semver_cmp(const std::string& a, const std::string& b) {
    int av[3] = {0,0,0}, bv[3] = {0,0,0};
    std::sscanf(a.c_str(), "%d.%d.%d", &av[0], &av[1], &av[2]);
    std::sscanf(b.c_str(), "%d.%d.%d", &bv[0], &bv[1], &bv[2]);
    for (int i = 0; i < 3; i++)
        if (av[i] != bv[i]) return av[i] < bv[i] ? -1 : 1;
    return 0;
}

Negotiation negotiate(const AppletManifest& m, const HostCaps& caps) {
    Negotiation n;
    if (m.abi_epoch != caps.abi_epoch) {
        n.reason = "Built for ABI epoch " + std::to_string(m.abi_epoch) +
                   "; this host speaks " + std::to_string(caps.abi_epoch) +
                   " — check for an applet update.";
        return n;
    }
    if (!m.min_host.empty() && semver_cmp(caps.version, m.min_host) < 0) {
        n.reason = "Requires host " + m.min_host + " or newer; this host is " +
                   caps.version + ".";
        return n;
    }
    for (const auto& svc : m.required_services) {
        if (!caps.services.count(svc)) {
            n.reason = "Requires a capability this host doesn't have: " + svc + ".";
            return n;
        }
    }
    n.ok = true;
    return n;
}

} // namespace caliper_host
```
Add `src/host/negotiation.cpp` to `caliper_host_lib` sources; add the test file to `caliper_tests`.

- [ ] **Step 4: Run to verify pass** — `ctest --test-dir build --output-on-failure` → all green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(host): pre-dlopen negotiation with contractual refusal reasons"
```

---

### Task 8: Crash guard — TDD

**Files:**
- Create: `src/host/crash_guard.h`, `src/host/crash_guard.cpp`
- Create: `tests/test_crash_guard.cpp`
- Modify: root `CMakeLists.txt` (`caliper_host_lib` sources), `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `caliper_host::GuardResult{ok,fault}`, `guarded_call(const std::function<void()>&)`. Consumed by loader (T12) and main (T13). POSIX path verified here; the Windows SEH path compiles-only until a Windows session (§ Global Constraints).

- [ ] **Step 1: Write the failing tests**

`tests/test_crash_guard.cpp`:
```cpp
#include <doctest/doctest.h>
#include "crash_guard.h"
using namespace caliper_host;

TEST_CASE("guard: normal call passes through, side effects run") {
    int x = 0;
    auto r = guarded_call([&] { x = 42; });
    CHECK(r.ok);
    CHECK(r.fault.empty());
    CHECK(x == 42);
}

TEST_CASE("guard: null write is contained and named") {
    auto r = guarded_call([] {
        volatile int* p = nullptr;
        *p = 1;
    });
    CHECK_FALSE(r.ok);
    // macOS arm64 reports EXC_BAD_ACCESS as SIGSEGV or SIGBUS — accept either.
    CHECK(r.fault.find("SIG") != std::string::npos);
}

TEST_CASE("guard: handlers restore — ok call after a crash works") {
    (void)guarded_call([] { volatile int* p = nullptr; *p = 1; });
    int x = 0;
    auto r = guarded_call([&] { x = 7; });
    CHECK(r.ok);
    CHECK(x == 7);
}
```

- [ ] **Step 2: Run to verify failure** — build fails, `crash_guard.h` missing.

- [ ] **Step 3: Implement**

`src/host/crash_guard.h`:
```cpp
#pragma once
#include <functional>
#include <string>

namespace caliper_host {

// Best-effort containment, not a sandbox (PLATFORM.md §15): after a fault the
// process memory is suspect; callers must quarantine the applet, not retry it.
struct GuardResult {
    bool ok = true;
    std::string fault;   // e.g. "SIGSEGV (invalid memory access)"; "" when ok
};

GuardResult guarded_call(const std::function<void()>& fn);

} // namespace caliper_host
```

`src/host/crash_guard.cpp`:
```cpp
#include "crash_guard.h"

#ifdef _WIN32
#include <windows.h>
#include <cstdio>

namespace caliper_host {
namespace {
// SEH needs a frame without C++ objects requiring unwinding.
int seh_invoke(const std::function<void()>* fn, unsigned long* code) {
    __try {
        (*fn)();
        return 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *code = GetExceptionCode();
        return 1;
    }
}
} // namespace

GuardResult guarded_call(const std::function<void()>& fn) {
    GuardResult r;
    unsigned long code = 0;
    if (seh_invoke(&fn, &code)) {
        r.ok = false;
        char buf[64];
        snprintf(buf, sizeof buf, "SEH exception 0x%08lX", code);
        r.fault = buf;
    }
    return r;
}
} // namespace caliper_host

#else // POSIX

#include <csetjmp>
#include <csignal>

namespace caliper_host {
namespace {

thread_local sigjmp_buf t_jmp;
thread_local volatile sig_atomic_t t_active = 0;
thread_local volatile int t_signal = 0;

void fault_handler(int sig) {
    if (t_active) {
        t_signal = sig;
        siglongjmp(t_jmp, 1);
    }
    // Fault outside a guarded region: restore default and re-raise.
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

constexpr int kSignals[] = {SIGSEGV, SIGBUS, SIGFPE, SIGILL};
constexpr int kNumSignals = 4;

const char* describe(int sig) {
    switch (sig) {
        case SIGSEGV: return "SIGSEGV (invalid memory access)";
        case SIGBUS:  return "SIGBUS (bad memory alignment/mapping)";
        case SIGFPE:  return "SIGFPE (arithmetic fault)";
        case SIGILL:  return "SIGILL (illegal instruction)";
        default:      return "signal";
    }
}

} // namespace

GuardResult guarded_call(const std::function<void()>& fn) {
    struct sigaction sa {}, saved[kNumSignals];
    sa.sa_handler = fault_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_NODEFER;
    for (int i = 0; i < kNumSignals; i++)
        sigaction(kSignals[i], &sa, &saved[i]);

    GuardResult r;
    t_active = 1;
    if (sigsetjmp(t_jmp, 1) == 0) {
        fn();
    } else {
        r.ok = false;
        r.fault = describe(t_signal);
    }
    t_active = 0;

    for (int i = 0; i < kNumSignals; i++)
        sigaction(kSignals[i], &saved[i], nullptr);
    return r;
}

} // namespace caliper_host
#endif
```
Add `src/host/crash_guard.cpp` to `caliper_host_lib`; add test file to `caliper_tests`.

- [ ] **Step 4: Run to verify pass** — `ctest --test-dir build --output-on-failure` green (the null-write test must NOT kill the test binary — that's the whole point).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(host): signal-trampoline crash guard (POSIX) + SEH path (Windows, unverified)"
```

---

### Task 9: Frame watchdog — TDD

**Files:**
- Create: `src/host/frame_watchdog.h` (header-only)
- Create: `tests/test_watchdog.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `caliper_host::FrameWatchdog{feed(ms), flagged(), reset()}` — 250 ms budget, 3 consecutive overruns latch the flag (PLATFORM.md §15). Consumed by main (T13).

- [ ] **Step 1: Write the failing tests**

`tests/test_watchdog.cpp`:
```cpp
#include <doctest/doctest.h>
#include "frame_watchdog.h"
using caliper_host::FrameWatchdog;

TEST_CASE("watchdog: three consecutive overruns latch the flag") {
    FrameWatchdog w;                       // 250 ms budget, threshold 3
    w.feed(300); w.feed(300);
    CHECK_FALSE(w.flagged());
    w.feed(300);
    CHECK(w.flagged());
}

TEST_CASE("watchdog: a good frame resets the streak") {
    FrameWatchdog w;
    w.feed(300); w.feed(300); w.feed(10); w.feed(300); w.feed(300);
    CHECK_FALSE(w.flagged());
}

TEST_CASE("watchdog: flag latches until reset()") {
    FrameWatchdog w;
    w.feed(300); w.feed(300); w.feed(300);
    w.feed(1);                              // fast frame does NOT clear it
    CHECK(w.flagged());
    w.reset();
    CHECK_FALSE(w.flagged());
}
```

- [ ] **Step 2: Run to verify failure** — `frame_watchdog.h` missing.

- [ ] **Step 3: Implement**

`src/host/frame_watchdog.h`:
```cpp
#pragma once

namespace caliper_host {

// Makes the platform threading rule observable (PLATFORM.md §15): frame()
// exceeding budget repeatedly flags the applet — "long work belongs in
// caliper.jobs". Latches until reset (applet relaunch).
class FrameWatchdog {
public:
    explicit FrameWatchdog(double budget_ms = 250.0, int threshold = 3)
        : budget_ms_(budget_ms), threshold_(threshold) {}

    void feed(double frame_ms) {
        if (flagged_) return;
        if (frame_ms > budget_ms_) {
            if (++over_ >= threshold_) flagged_ = true;
        } else {
            over_ = 0;
        }
    }
    bool flagged() const { return flagged_; }
    void reset() { over_ = 0; flagged_ = false; }

private:
    double budget_ms_;
    int threshold_;
    int over_ = 0;
    bool flagged_ = false;
};

} // namespace caliper_host
```
Add `tests/test_watchdog.cpp` to `caliper_tests` (header-only — include path already comes from `caliper_host_lib`).

- [ ] **Step 4: Run to verify pass** — `ctest --test-dir build` green.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(host): frame watchdog (250ms x3, latching)"
```

---

### Task 10: Sugar layer (`caliper.hpp`) + fixture host — TDD

**Files:**
- Create: `sdk/include/caliper/caliper.hpp`
- Create: `sdk/testing/caliper/fixture_host.h`
- Create: `tests/test_sugar.cpp`
- Modify: root `CMakeLists.txt` (add `caliper_sdk_testing` INTERFACE target), `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `abi.h`, `ui_v1.h`, `log_v1.h` (T5).
- Produces: `caliper::Applet` (virtuals `on_init(Host&)`, `on_frame(const Frame&)`, `on_cleanup()`), `caliper::Host{raw(),service(id),data_dir(),log_info(),log_error()}`, `caliper::Frame{fb_width,fb_height,dpi_scale,time_sec,delta_sec; static from(CaliperFrameInfo)}`, `caliper::AppletMeta`, macro `CALIPER_APPLET(CLASS, .id=…, .version=…, .name=…, .summary=…, .tag=…, .services={…})` (fields in exactly that order — C++20 designated initializers), `caliper::ui::connect(const CaliperHost*)`, and `caliper::testing::FixtureHost{host(),log_lines(),log_contains()}`. Consumed by hello (T11), loader tests (T12), ports (T14–16).

- [ ] **Step 1: Write the failing tests**

`tests/test_sugar.cpp` (one `CALIPER_APPLET` per binary — keep all sugar cases in this single TU):
```cpp
#include <doctest/doctest.h>
#include <caliper/caliper.hpp>
#include <caliper/fixture_host.h>
#include <string>

namespace {
bool g_throw_in_frame = false;

class TinyApplet final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        host_ = &host;
        host.log_info("tiny.on_init");
        return true;
    }
    void on_frame(const caliper::Frame& f) override {
        if (g_throw_in_frame) throw std::runtime_error("boom");
        last_w_ = f.fb_width;
    }
    void on_cleanup() override { if (host_) host_->log_info("tiny.on_cleanup"); }
    int last_w_ = 0;
private:
    caliper::Host* host_ = nullptr;
};
} // namespace

CALIPER_APPLET(TinyApplet,
    .id       = "dev.caliper.tiny",
    .version  = "0.1.0",
    .name     = "Tiny",
    .summary  = "sugar test applet",
    .tag      = "Test",
    .services = {CALIPER_LOG_V1})

TEST_CASE("sugar: macro-generated descriptor matches meta") {
    const CaliperAppletDescriptor* d = caliper_applet_descriptor();
    REQUIRE(d != nullptr);
    CHECK(d->struct_size == sizeof(CaliperAppletDescriptor));
    CHECK(d->abi_epoch == CALIPER_ABI_EPOCH);
    CHECK(std::string(d->id) == "dev.caliper.tiny");
    CHECK(std::string(d->version) == "0.1.0");
    CHECK(std::string(d->name) == "Tiny");
    CHECK(std::string(d->tag) == "Test");
    REQUIRE(d->required_services != nullptr);
    CHECK(std::string(d->required_services[0]) == "caliper.log.v1");
    CHECK(d->required_services[1] == nullptr);          // NULL-terminated
    CHECK(d->api.struct_size == sizeof(CaliperAppletAPI));
    REQUIRE(d->api.create); REQUIRE(d->api.destroy); REQUIRE(d->api.initialize);
    REQUIRE(d->api.frame);  REQUIRE(d->api.cleanup);
}

TEST_CASE("sugar: lifecycle bridges to the class through the C table") {
    caliper::testing::FixtureHost fx;
    const auto* d = caliper_applet_descriptor();
    void* self = d->api.create();
    REQUIRE(self != nullptr);
    CHECK(d->api.initialize(self, fx.host()));
    CHECK(fx.log_contains("tiny.on_init"));

    CaliperFrameInfo fi{};
    fi.struct_size = sizeof fi; fi.fb_width = 640; fi.fb_height = 480;
    fi.dpi_scale = 2.0f;
    d->api.frame(self, &fi);

    d->api.cleanup(self);
    CHECK(fx.log_contains("tiny.on_cleanup"));
    d->api.destroy(self);
}

TEST_CASE("sugar: exceptions never cross the C boundary") {
    caliper::testing::FixtureHost fx;
    const auto* d = caliper_applet_descriptor();
    void* self = d->api.create();
    REQUIRE(d->api.initialize(self, fx.host()));
    g_throw_in_frame = true;
    CaliperFrameInfo fi{}; fi.struct_size = sizeof fi;
    d->api.frame(self, &fi);                    // must not terminate/propagate
    g_throw_in_frame = false;
    CHECK(fx.log_contains("unhandled exception in on_frame"));
    d->api.cleanup(self);
    d->api.destroy(self);
}
```

CMake — root, after the SDK install rules:
```cmake
# Fixture host for TDD of applets/sugar (ships in the SDK at Phase 3).
add_library(caliper_sdk_testing INTERFACE)
add_library(caliper::sdk_testing ALIAS caliper_sdk_testing)
target_include_directories(caliper_sdk_testing INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/sdk/testing>)
target_link_libraries(caliper_sdk_testing INTERFACE caliper::sdk)
```
`tests/CMakeLists.txt`: add `test_sugar.cpp`; link `caliper::sdk_testing caliper::ui_stack` to `caliper_tests` (sugar's `ui::connect` references ImGui symbols; the static libs are already built and need no GL context at link or in these tests).

- [ ] **Step 2: Run to verify failure** — `caliper/caliper.hpp` missing.

- [ ] **Step 3: Implement the sugar header**

`sdk/include/caliper/caliper.hpp`:
```cpp
#pragma once
// Caliper C++ sugar (PLATFORM.md §8). Header-only, optional by design: a C
// applet can implement abi.h by hand. Requires C++20 (designated inits).
#include <caliper/abi.h>
#include <caliper/services/log_v1.h>
#include <caliper/services/ui_v1.h>

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

namespace caliper {

struct Frame {
    int32_t fb_width = 0, fb_height = 0;   // PHYSICAL pixels (§6a)
    float   dpi_scale = 1.0f;
    double  time_sec = 0.0, delta_sec = 0.0;
    static Frame from(const CaliperFrameInfo& fi) {
        Frame f;
        f.fb_width = fi.fb_width;  f.fb_height = fi.fb_height;
        f.dpi_scale = fi.dpi_scale;
        f.time_sec = fi.time_sec;  f.delta_sec = fi.delta_sec;
        return f;
    }
};

class Host {
public:
    Host() = default;
    explicit Host(const CaliperHost* raw) : raw_(raw) {
        if (raw_ && raw_->get_service)
            log_ = static_cast<const CaliperLogV1*>(
                raw_->get_service(raw_, CALIPER_LOG_V1));
    }
    const CaliperHost* raw() const { return raw_; }
    const void* service(const char* id) const {
        return (raw_ && raw_->get_service) ? raw_->get_service(raw_, id) : nullptr;
    }
    const char* data_dir() const {
        return (raw_ && raw_->applet_data_dir) ? raw_->applet_data_dir : "";
    }
    void log(CaliperLogLevel lv, const char* msg) const {
        if (log_ && log_->log) log_->log(lv, msg);
    }
    void log_info(const char* m) const  { log(CALIPER_LOG_INFO, m); }
    void log_error(const char* m) const { log(CALIPER_LOG_ERROR, m); }

private:
    const CaliperHost* raw_ = nullptr;
    const CaliperLogV1* log_ = nullptr;
};

class Applet {
public:
    virtual ~Applet() = default;
    virtual bool on_init(Host& host) = 0;
    virtual void on_frame(const Frame& frame) = 0;
    virtual void on_cleanup() {}
};

namespace ui {
// SetAllocatorFunctions + SetCurrentContext x3, in one call authors cannot
// get wrong (§6d). Returns false when the host has no ui.v1 (headless).
inline bool connect(const CaliperHost* h) {
    if (!h || !h->get_service) return false;
    auto* ui = static_cast<const CaliperUiV1*>(h->get_service(h, CALIPER_UI_V1));
    if (!ui) return false;
    CaliperImGuiAllocFn alloc = nullptr;
    CaliperImGuiFreeFn  free_fn = nullptr;
    void* user = nullptr;
    ui->imgui_allocators(&alloc, &free_fn, &user);
    if (alloc && free_fn)
        ImGui::SetAllocatorFunctions(reinterpret_cast<ImGuiMemAllocFunc>(alloc),
                                     reinterpret_cast<ImGuiMemFreeFunc>(free_fn),
                                     user);
    ImGui::SetCurrentContext(ui->imgui_context());
    ImPlot::SetCurrentContext(ui->implot_context());
    ImPlot3D::SetCurrentContext(ui->implot3d_context());
    return true;
}
} // namespace ui

struct AppletMeta {
    const char* id;
    const char* version;
    const char* name;
    const char* summary;
    const char* tag;
    const char* services[15];   // NULL-terminated by aggregate zero-init
};

} // namespace caliper

// Generates: descriptor + the five C bridge functions + the single export.
// Field order is fixed: id, version, name, summary, tag, services.
#define CALIPER_APPLET(CLASS, ...)                                             \
    namespace caliper_applet_gen {                                             \
    static const ::caliper::AppletMeta kMeta{__VA_ARGS__};                     \
    struct Holder {                                                            \
        CLASS obj;                                                             \
        ::caliper::Host host;                                                  \
    };                                                                         \
    static void* cal_create(void) {                                            \
        try { return new Holder(); } catch (...) { return nullptr; }           \
    }                                                                          \
    static void cal_destroy(void* s) { delete static_cast<Holder*>(s); }       \
    static bool cal_initialize(void* s, const CaliperHost* h) {                \
        auto* hold = static_cast<Holder*>(s);                                  \
        hold->host = ::caliper::Host(h);                                       \
        ::caliper::ui::connect(h);                                             \
        try { return hold->obj.on_init(hold->host); }                          \
        catch (...) {                                                          \
            hold->host.log_error("unhandled exception in on_init");            \
            return false;                                                      \
        }                                                                      \
    }                                                                          \
    static void cal_frame(void* s, const CaliperFrameInfo* fi) {               \
        auto* hold = static_cast<Holder*>(s);                                  \
        try { hold->obj.on_frame(::caliper::Frame::from(*fi)); }               \
        catch (...) {                                                          \
            hold->host.log_error("unhandled exception in on_frame");           \
        }                                                                      \
    }                                                                          \
    static void cal_cleanup(void* s) {                                         \
        auto* hold = static_cast<Holder*>(s);                                  \
        try { hold->obj.on_cleanup(); }                                        \
        catch (...) {                                                          \
            hold->host.log_error("unhandled exception in on_cleanup");         \
        }                                                                      \
    }                                                                          \
    } /* namespace caliper_applet_gen */                                       \
    extern "C" CALIPER_EXPORT const CaliperAppletDescriptor*                   \
    caliper_applet_descriptor(void) {                                          \
        static const CaliperAppletDescriptor kDesc = {                         \
            (uint32_t)sizeof(CaliperAppletDescriptor),                         \
            CALIPER_ABI_EPOCH,                                                 \
            ::caliper_applet_gen::kMeta.id,                                    \
            ::caliper_applet_gen::kMeta.version,                               \
            ::caliper_applet_gen::kMeta.name,                                  \
            ::caliper_applet_gen::kMeta.summary,                               \
            ::caliper_applet_gen::kMeta.tag,                                   \
            ::caliper_applet_gen::kMeta.services,                              \
            { (uint32_t)sizeof(CaliperAppletAPI),                              \
              &::caliper_applet_gen::cal_create,                               \
              &::caliper_applet_gen::cal_destroy,                              \
              &::caliper_applet_gen::cal_initialize,                           \
              &::caliper_applet_gen::cal_frame,                                \
              &::caliper_applet_gen::cal_cleanup } };                          \
        return &kDesc;                                                         \
    }
```

`sdk/testing/caliper/fixture_host.h`:
```cpp
#pragma once
// Headless fake CaliperHost (PLATFORM.md §16 "fixture host"): TDD applets and
// sugar without launching UI. Vends log.v1 only; get_service returns NULL for
// everything else. ONE live fixture per process (C tables carry no user data,
// so the thunks route through a static active pointer).
#include <caliper/abi.h>
#include <caliper/services/log_v1.h>
#include <string>
#include <vector>

namespace caliper::testing {

class FixtureHost {
public:
    FixtureHost() {
        active_ = this;
        log_table_.struct_size = sizeof(CaliperLogV1);
        log_table_.log = &FixtureHost::log_thunk;
        host_.struct_size = sizeof(CaliperHost);
        host_.abi_epoch = CALIPER_ABI_EPOCH;
        host_.host_version = (0u << 16) | (6u << 8) | 0u;
        host_.applet_data_dir = data_dir_.c_str();
        host_.get_service = &FixtureHost::get_service_thunk;
    }
    ~FixtureHost() { if (active_ == this) active_ = nullptr; }

    const CaliperHost* host() const { return &host_; }
    const std::vector<std::string>& log_lines() const { return lines_; }
    bool log_contains(const std::string& needle) const {
        for (const auto& l : lines_)
            if (l.find(needle) != std::string::npos) return true;
        return false;
    }

private:
    static void log_thunk(CaliperLogLevel, const char* msg) {
        if (active_ && msg) active_->lines_.emplace_back(msg);
    }
    static const void* get_service_thunk(const CaliperHost*, const char* id) {
        if (active_ && id && std::string(id) == CALIPER_LOG_V1)
            return &active_->log_table_;
        return nullptr;
    }
    inline static FixtureHost* active_ = nullptr;
    CaliperHost host_{};
    CaliperLogV1 log_table_{};
    std::string data_dir_ = "/tmp/caliper-fixture-data";
    std::vector<std::string> lines_;
};

} // namespace caliper::testing
```

- [ ] **Step 4: Run to verify pass**

```bash
cmake -B build -DBUILD_TESTS=ON && cmake --build build --target caliper_tests && ctest --test-dir build --output-on-failure
```
Expected: all green, including the exception-containment case.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(sdk): header-only sugar (CALIPER_APPLET, ui::connect, Host/Frame) + fixture host"
```

---

### Task 11: `examples/hello` fixture applet

**Files:**
- Create: `examples/hello/CMakeLists.txt`, `examples/hello/hello.cpp`, `examples/hello/hello.caliper.toml`
- Wire (already authored, do NOT rewrite): `examples/signal_scope/` — the exemplar applet (`signal_scope.cpp`, manifest, CMakeLists exist; this task only adds it to the root build). If it fails to compile against the T5/T10 headers, fix the *exemplar* to match the built SDK, not vice versa, and note the divergence in the commit message.
- Modify: root `CMakeLists.txt` (add examples subdirectory)

**Interfaces:**
- Consumes: sugar (T10), `caliper::sdk`/`caliper::ui_stack` (T2).
- Produces: `build/applets/libhello.dylib` + `build/applets/hello.caliper.toml` — the loader-test substrate (T12) and the first epoch-2 applet visible in the app (T13). Env var `CALIPER_HELLO_CRASH=1` makes `on_frame` fault **before** touching ImGui (quarantine test hook).

- [ ] **Step 1: The applet**

`examples/hello/hello.cpp`:
```cpp
// Epoch-2 fixture applet (PLATFORM.md §13.1): loader-test substrate and the
// "hello world" of the sugar layer. Kept deliberately tiny.
#include <caliper/caliper.hpp>
#include <cmath>
#include <cstdlib>
#include <vector>

class HelloApplet final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        host_ = &host;
        crash_on_frame_ = std::getenv("CALIPER_HELLO_CRASH") != nullptr;
        host.log_info("hello.on_init");
        return true;
    }

    void on_frame(const caliper::Frame& f) override {
        if (crash_on_frame_) {           // test hook: fault before any ImGui call
            volatile int* p = nullptr;
            *p = 1;
        }
        ImGui::SetNextWindowPos({40, 60}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({520, 360}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Hello, Caliper");
        ImGui::Text("ABI epoch %d applet via CALIPER_APPLET macro", CALIPER_ABI_EPOCH);
        ImGui::Text("framebuffer: %d x %d px   dpi_scale: %.1f",
                    f.fb_width, f.fb_height, f.dpi_scale);
        if (ImPlot::BeginPlot("sine", {-1, 220})) {
            static std::vector<float> xs(256), ys(256);
            for (int i = 0; i < 256; i++) {
                xs[i] = i / 255.0f * 6.28318f;
                ys[i] = std::sin(xs[i] + (float)f.time_sec);
            }
            ImPlot::PlotLine("sin", xs.data(), ys.data(), 256);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    void on_cleanup() override {
        if (host_) host_->log_info("hello.on_cleanup");
    }

private:
    caliper::Host* host_ = nullptr;
    bool crash_on_frame_ = false;
};

CALIPER_APPLET(HelloApplet,
    .id       = "dev.caliper.hello",
    .version  = "0.1.0",
    .name     = "Hello",
    .summary  = "Epoch-2 fixture applet: sugar demo + loader-test substrate.",
    .tag      = "Demo",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1})
```

`examples/hello/hello.caliper.toml`:
```toml
[applet]
id      = "dev.caliper.hello"
name    = "Hello"
version = "0.1.0"
summary = "Epoch-2 fixture applet: sugar demo + loader-test substrate."
tag     = "Demo"

[compat]
abi_epoch = 2
min_host  = "0.6.0"

[services]
required = ["caliper.ui.v1", "caliper.log.v1"]
```

`examples/hello/CMakeLists.txt`:
```cmake
add_library(hello_applet SHARED hello.cpp)
target_link_libraries(hello_applet PRIVATE caliper::sdk caliper::ui_stack)
target_compile_definitions(hello_applet PRIVATE CALIPER_APPLET_EXPORT)
set_target_properties(hello_applet PROPERTIES
    OUTPUT_NAME hello
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_custom_command(TARGET hello_applet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/hello.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/hello.caliper.toml)
```

Root `CMakeLists.txt` — placement matters: the stale-applet cleanup loop (currently lines 162–170) deletes any `lib*.dylib` in `build/applets/` that isn't in `_active_applet_libs`, so hello must be **registered in that list before the cleanup runs**. Insert this block BETWEEN the applet `foreach(...)` loop and the `# Remove stale applet libraries` block:
```cmake
option(CALIPER_BUILD_EXAMPLES "Build example applets (hello fixture + SignalScope exemplar)" ON)
if(CALIPER_BUILD_EXAMPLES)
    add_subdirectory(examples/hello)
    add_subdirectory(examples/signal_scope)
    add_dependencies(caliper hello_applet signal_scope_applet)
    list(APPEND _active_applet_libs
        "${CMAKE_BINARY_DIR}/applets/${CMAKE_SHARED_LIBRARY_PREFIX}hello${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${CMAKE_BINARY_DIR}/applets/${CMAKE_SHARED_LIBRARY_PREFIX}signal_scope${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()
```

- [ ] **Step 2: Build and verify artifacts**

```bash
cmake -B build -DBUILD_TESTS=ON && cmake --build build --target hello_applet signal_scope_applet
ls build/applets/libhello.dylib build/applets/hello.caliper.toml \
   build/applets/libsignal_scope.dylib build/applets/signal_scope.caliper.toml
nm -gU build/applets/libhello.dylib | grep caliper_applet_descriptor
```
Expected: both files listed; the symbol `_caliper_applet_descriptor` exported.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat(examples): hello fixture applet on epoch 2 (sugar + manifest + crash hook)"
```

---

### Task 12: Loader v2 — TDD

**Files:**
- Create: `src/host/applet_loader.h`, `src/host/applet_loader.cpp`
- Create: `tests/test_loader.cpp`
- Modify: root `CMakeLists.txt` (`caliper_host_lib` sources), `tests/CMakeLists.txt` (test file + applets-dir define + hello dependency)

**Interfaces:**
- Consumes: manifest (T6), negotiation (T7), crash guard (T8), abi.h (T5), hello artifacts (T11).
- Produces: `caliper_host::AppletStatus{Ready,Refused,Failed,Active,Quarantined}`, `AppletEntry{manifest,dylib_path,data_dir,status,status_text,handle,desc,instance}`, `AppletLoader{AppletLoader(HostCaps,data_root), scan(dir), count(), at(i), launch(i,CaliperHost), frame(i,const CaliperFrameInfo&)→bool, teardown(i), close_all()}`. Consumed by main (T13). **Negotiation order per PLATFORM.md §14:** manifest parse → binary present → negotiate → (launch:) dlopen → descriptor sanity → create/initialize.

- [ ] **Step 1: Write the failing tests**

`tests/test_loader.cpp`:
```cpp
#include <doctest/doctest.h>
#include "applet_loader.h"
#include <caliper/fixture_host.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;
using namespace caliper_host;

// CALIPER_TEST_APPLETS_DIR + CALIPER_TEST_DATA_ROOT are compile definitions.
static HostCaps caps() {
    return HostCaps{2, "0.6.0", {"caliper.ui.v1", "caliper.log.v1"}};
}
static int find_by_id(AppletLoader& L, const std::string& id) {
    for (int i = 0; i < L.count(); i++)
        if (L.at(i).manifest.id == id) return i;
    return -1;
}
static int count_log(const caliper::testing::FixtureHost& fx, const std::string& s) {
    int n = 0;
    for (auto& l : fx.log_lines()) if (l.find(s) != std::string::npos) n++;
    return n;
}

TEST_CASE("loader: scan finds hello via manifest, status Ready") {
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(CALIPER_TEST_APPLETS_DIR);
    int i = find_by_id(L, "dev.caliper.hello");
    REQUIRE(i >= 0);
    CHECK(L.at(i).status == AppletStatus::Ready);
    CHECK_FALSE(L.at(i).dylib_path.empty());
}

TEST_CASE("loader: full lifecycle, hooks called exactly once") {
    caliper::testing::FixtureHost fx;
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(CALIPER_TEST_APPLETS_DIR);
    int i = find_by_id(L, "dev.caliper.hello");
    REQUIRE(i >= 0);
    REQUIRE(L.launch(i, *fx.host()));
    CHECK(L.at(i).status == AppletStatus::Active);
    CHECK(count_log(fx, "hello.on_init") == 1);
    L.teardown(i);
    CHECK(L.at(i).status == AppletStatus::Ready);
    CHECK(count_log(fx, "hello.on_cleanup") == 1);
    L.close_all();
}

TEST_CASE("loader: relaunch tears down the old instance first") {
    caliper::testing::FixtureHost fx;
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(CALIPER_TEST_APPLETS_DIR);
    int i = find_by_id(L, "dev.caliper.hello");
    REQUIRE(L.launch(i, *fx.host()));
    REQUIRE(L.launch(i, *fx.host()));
    CHECK(count_log(fx, "hello.on_init") == 2);
    CHECK(count_log(fx, "hello.on_cleanup") == 1);
    L.close_all();
}

TEST_CASE("loader: descriptor/manifest agreement is enforced") {
    // Manifest lies about the version -> launch must fail with a reason.
    caliper::testing::FixtureHost fx;
    fs::path dir = fs::temp_directory_path() / "caliper-liar";
    fs::create_directories(dir);
    fs::copy_file(fs::path(CALIPER_TEST_APPLETS_DIR) / "libhello.dylib",
                  dir / "libhello.dylib", fs::copy_options::overwrite_existing);
    std::ofstream(dir / "hello.caliper.toml") <<
        "[applet]\nid=\"dev.caliper.hello\"\nname=\"Hello\"\nversion=\"9.9.9\"\n"
        "[compat]\nabi_epoch=2\n[services]\nrequired=[\"caliper.ui.v1\"]\n";
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(dir.string());
    REQUIRE(L.count() == 1);
    CHECK(L.at(0).status == AppletStatus::Ready);      // pre-dlopen checks pass
    CHECK_FALSE(L.launch(0, *fx.host()));              // descriptor sanity fails
    CHECK(L.at(0).status == AppletStatus::Failed);
    CHECK(L.at(0).status_text.find("descriptor") != std::string::npos);
    fs::remove_all(dir);
}

TEST_CASE("loader: epoch mismatch refused before any dlopen") {
    fs::path dir = fs::temp_directory_path() / "caliper-epoch99";
    fs::create_directories(dir);
    std::ofstream(dir / "fake.caliper.toml") <<
        "[applet]\nid=\"x.fake\"\nname=\"Fake\"\nversion=\"1.0.0\"\n"
        "[compat]\nabi_epoch=99\n";
    std::ofstream(dir / "libfake.dylib") << "not a real dylib";  // never opened
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(dir.string());
    REQUIRE(L.count() == 1);
    CHECK(L.at(0).status == AppletStatus::Refused);
    CHECK(L.at(0).status_text.find("epoch 99") != std::string::npos);
    fs::remove_all(dir);
}

TEST_CASE("loader: missing binary is a Failed card, not a crash") {
    fs::path dir = fs::temp_directory_path() / "caliper-nobin";
    fs::create_directories(dir);
    std::ofstream(dir / "ghost.caliper.toml") <<
        "[applet]\nid=\"x.ghost\"\nname=\"Ghost\"\nversion=\"1.0.0\"\n"
        "[compat]\nabi_epoch=2\n";
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(dir.string());
    REQUIRE(L.count() == 1);
    CHECK(L.at(0).status == AppletStatus::Failed);
    CHECK(L.at(0).status_text.find("not found") != std::string::npos);
    fs::remove_all(dir);
}

TEST_CASE("loader: fault in frame() quarantines, host survives") {
    caliper::testing::FixtureHost fx;
    setenv("CALIPER_HELLO_CRASH", "1", 1);
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(CALIPER_TEST_APPLETS_DIR);
    int i = find_by_id(L, "dev.caliper.hello");
    REQUIRE(L.launch(i, *fx.host()));
    CaliperFrameInfo fi{}; fi.struct_size = sizeof fi;
    CHECK_FALSE(L.frame(i, fi));                       // fault -> quarantined
    CHECK(L.at(i).status == AppletStatus::Quarantined);
    CHECK(L.at(i).status_text.find("SIG") != std::string::npos);
    unsetenv("CALIPER_HELLO_CRASH");
    // The host (this test process) is alive to assert all of the above.
}
```

`tests/CMakeLists.txt` additions:
```cmake
target_sources(caliper_tests PRIVATE test_loader.cpp)
add_dependencies(caliper_tests hello_applet)
target_compile_definitions(caliper_tests PRIVATE
    CALIPER_TEST_APPLETS_DIR="${CMAKE_BINARY_DIR}/applets"
    CALIPER_TEST_DATA_ROOT="${CMAKE_BINARY_DIR}/test-data")
```

- [ ] **Step 2: Run to verify failure** — `applet_loader.h` missing.

- [ ] **Step 3: Implement**

`src/host/applet_loader.h`:
```cpp
#pragma once
#include "applet_manifest.h"
#include "negotiation.h"
#include <caliper/abi.h>
#include <string>
#include <vector>

namespace caliper_host {

enum class AppletStatus {
    Ready,        // negotiated, will dlopen on launch
    Refused,      // pre-dlopen negotiation refusal (friendly reason)
    Failed,       // broken: parse error, missing binary, bad descriptor, init false
    Active,       // instance running
    Quarantined,  // faulted; never called again this session (§15)
};

struct AppletEntry {
    AppletManifest manifest;
    std::string dylib_path;    // "" when binary missing
    std::string data_dir;      // per-applet sandbox; storage for the ABI pointer
    AppletStatus status = AppletStatus::Failed;
    std::string status_text;   // card text for Refused/Failed/Quarantined

    void* handle = nullptr;
    const CaliperAppletDescriptor* desc = nullptr;
    void* instance = nullptr;
};

// Manifest-first loader (PLATFORM.md §14): scan() never dlopens; launch()
// performs dlopen -> descriptor sanity -> guarded create/initialize.
class AppletLoader {
public:
    AppletLoader(HostCaps caps, std::string data_root);
    ~AppletLoader() { close_all(); }

    int scan(const std::string& dir);            // returns entries added
    int count() const { return (int)entries_.size(); }
    AppletEntry&       at(int i)       { return entries_[i]; }
    const AppletEntry& at(int i) const { return entries_[i]; }

    // host_proto: filled CaliperHost except applet_data_dir, which the loader
    // points at this entry's sandbox dir before initialize().
    bool launch(int idx, CaliperHost host_proto);
    bool frame(int idx, const CaliperFrameInfo& info);  // false => just quarantined
    void teardown(int idx);
    void close_all();

private:
    HostCaps caps_;
    std::string data_root_;
    std::vector<AppletEntry> entries_;
    std::vector<CaliperHost> host_blocks_;  // stable storage per entry
};

} // namespace caliper_host
```

`src/host/applet_loader.cpp`:
```cpp
#include "applet_loader.h"
#include "crash_guard.h"
#include <algorithm>
#include <filesystem>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

namespace fs = std::filesystem;

namespace caliper_host {
namespace {

void* lib_open(const char* path) {
#ifdef _WIN32
    return (void*)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}
void lib_close(void* h) {
#ifdef _WIN32
    FreeLibrary((HMODULE)h);
#else
    dlclose(h);
#endif
}
void* lib_sym(void* h, const char* name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)h, name);
#else
    return dlsym(h, name);
#endif
}
std::string lib_error() {
#ifdef _WIN32
    char buf[256] = {};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(),
                   0, buf, sizeof(buf), nullptr);
    return buf;
#else
    const char* e = dlerror();
    return e ? e : "unknown dlopen error";
#endif
}

#ifdef _WIN32
  constexpr const char* kExt = ".dll";
  constexpr const char* kPrefix = "";
#elif __APPLE__
  constexpr const char* kExt = ".dylib";
  constexpr const char* kPrefix = "lib";
#else
  constexpr const char* kExt = ".so";
  constexpr const char* kPrefix = "lib";
#endif

constexpr const char* kManifestSuffix = ".caliper.toml";

// <stem>.caliper.toml -> sibling (lib)?<stem>.<ext>, or "".
std::string find_binary(const fs::path& dir, const std::string& stem) {
    for (const std::string& name : {stem + kExt, kPrefix + stem + kExt}) {
        std::error_code ec;
        if (fs::is_regular_file(dir / name, ec)) return (dir / name).string();
    }
    return {};
}

} // namespace

AppletLoader::AppletLoader(HostCaps caps, std::string data_root)
    : caps_(std::move(caps)), data_root_(std::move(data_root)) {}

int AppletLoader::scan(const std::string& dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return 0;

    int added = 0;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        const std::string fname = e.path().filename().string();
        if (fname.size() <= std::string(kManifestSuffix).size() ||
            fname.substr(fname.size() - std::string(kManifestSuffix).size())
                != kManifestSuffix)
            continue;

        AppletEntry entry;
        auto parsed = parse_manifest_file(e.path().string());
        if (!parsed.ok) {
            entry.manifest.name = fname;
            entry.status = AppletStatus::Failed;
            entry.status_text = parsed.error;
            entries_.push_back(std::move(entry));
            added++;
            continue;
        }
        entry.manifest = std::move(parsed.manifest);
        entry.data_dir = data_root_ + "/" + entry.manifest.id;

        const std::string stem =
            fname.substr(0, fname.size() - std::string(kManifestSuffix).size());
        entry.dylib_path = find_binary(e.path().parent_path(), stem);
        if (entry.dylib_path.empty()) {
            entry.status = AppletStatus::Failed;
            entry.status_text = "applet binary not found next to " + fname;
        } else if (auto n = negotiate(entry.manifest, caps_); !n.ok) {
            entry.status = AppletStatus::Refused;
            entry.status_text = n.reason;
        } else {
            entry.status = AppletStatus::Ready;
        }
        entries_.push_back(std::move(entry));
        added++;
    }
    std::sort(entries_.begin(), entries_.end(),
              [](const AppletEntry& a, const AppletEntry& b) {
                  return a.manifest.name < b.manifest.name;
              });
    // Rescanning reallocates host_blocks_, which active applets hold pointers
    // into — scan() must only run before any launch (true for both the app
    // and the tests; enforce it if a rescan feature ever appears).
    host_blocks_.assign(entries_.size(), CaliperHost{});
    return added;
}

bool AppletLoader::launch(int idx, CaliperHost host_proto) {
    if (idx < 0 || idx >= count()) return false;
    AppletEntry& a = entries_[idx];
    if (a.status == AppletStatus::Active) teardown(idx);
    if (a.status != AppletStatus::Ready) return false;

    auto fail = [&](std::string why) {
        a.status = AppletStatus::Failed;
        a.status_text = std::move(why);
        return false;
    };

    if (!a.handle) {
        a.handle = lib_open(a.dylib_path.c_str());
        if (!a.handle) return fail("load failed: " + lib_error());
        auto get_desc = (const CaliperAppletDescriptor* (*)(void))
            lib_sym(a.handle, CALIPER_DESCRIPTOR_SYMBOL);
        if (!get_desc)
            return fail(std::string("missing export ") + CALIPER_DESCRIPTOR_SYMBOL);
        a.desc = get_desc();
    }

    // Descriptor sanity: the binary must agree with its manifest (§14).
    const auto* d = a.desc;
    if (!d || d->struct_size < sizeof(CaliperAppletDescriptor))
        return fail("descriptor missing or truncated");
    if (d->abi_epoch != caps_.abi_epoch)
        return fail("descriptor ABI epoch disagrees with manifest");
    if (!d->id || a.manifest.id != d->id)
        return fail("descriptor id disagrees with manifest");
    if (!d->version || a.manifest.version != d->version)
        return fail("descriptor version disagrees with manifest");
    if (!d->api.create || !d->api.destroy || !d->api.initialize ||
        !d->api.frame || !d->api.cleanup)
        return fail("descriptor function table incomplete");

    std::error_code ec;
    fs::create_directories(a.data_dir, ec);
    host_blocks_[idx] = host_proto;
    host_blocks_[idx].applet_data_dir = a.data_dir.c_str();

    void* instance = nullptr;
    auto cr = guarded_call([&] { instance = d->api.create(); });
    if (!cr.ok) { a.status = AppletStatus::Quarantined;
                  a.status_text = "crashed in create(): " + cr.fault; return false; }
    if (!instance) return fail("create() returned null");

    bool init_ok = false;
    auto ir = guarded_call([&] {
        init_ok = d->api.initialize(instance, &host_blocks_[idx]);
    });
    if (!ir.ok) { a.status = AppletStatus::Quarantined;
                  a.status_text = "crashed in initialize(): " + ir.fault; return false; }
    if (!init_ok) {
        guarded_call([&] { d->api.destroy(instance); });
        return fail("initialize() returned false");
    }

    a.instance = instance;
    a.status = AppletStatus::Active;
    a.status_text.clear();
    return true;
}

bool AppletLoader::frame(int idx, const CaliperFrameInfo& info) {
    if (idx < 0 || idx >= count()) return false;
    AppletEntry& a = entries_[idx];
    if (a.status != AppletStatus::Active || !a.instance) return false;

    auto r = guarded_call([&] { a.desc->api.frame(a.instance, &info); });
    if (!r.ok) {
        // Memory is suspect after a fault: abandon the instance, never call
        // cleanup/destroy/dlclose on it (§15 honesty).
        a.status = AppletStatus::Quarantined;
        a.status_text = "crashed in frame(): " + r.fault;
        a.instance = nullptr;
        return false;
    }
    return true;
}

void AppletLoader::teardown(int idx) {
    if (idx < 0 || idx >= count()) return;
    AppletEntry& a = entries_[idx];
    if (a.status != AppletStatus::Active || !a.instance) return;

    auto cl = guarded_call([&] { a.desc->api.cleanup(a.instance); });
    auto de = cl.ok
        ? guarded_call([&] { a.desc->api.destroy(a.instance); })
        : cl;
    a.instance = nullptr;
    if (!cl.ok || !de.ok) {
        a.status = AppletStatus::Quarantined;
        a.status_text = "crashed during teardown: " + (cl.ok ? de : cl).fault;
        return;
    }
    a.status = AppletStatus::Ready;
}

void AppletLoader::close_all() {
    for (int i = 0; i < count(); i++) teardown(i);
    for (auto& a : entries_) {
        // Quarantined dylibs are left mapped: running static destructors in a
        // corrupted image is worse than a small leak at shutdown.
        if (a.handle && a.status != AppletStatus::Quarantined) lib_close(a.handle);
        a.handle = nullptr;
        a.desc = nullptr;
    }
    entries_.clear();
    host_blocks_.clear();
}

} // namespace caliper_host
```
Add `src/host/applet_loader.cpp` to `caliper_host_lib` sources.

- [ ] **Step 4: Run to verify pass**

```bash
cmake --build build --target caliper_tests && ctest --test-dir build --output-on-failure
```
Expected: all loader cases green, including quarantine-with-survival.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(host): manifest-first loader v2 with negotiation gate, descriptor sanity, quarantine"
```

---

### Task 13: Host integration — services, FrameInfo, cards, `/MD`

**Files:**
- Create: `src/host/host_services.h`, `src/host/host_services.cpp`
- Modify: `src/main.cpp` (replace `AppletHost` usage with `AppletLoader`)
- Modify: `CMakeLists.txt` (host sources + `/MD` option)

**Interfaces:**
- Consumes: loader (T12), services headers (T5), watchdog (T9), `host_version.h` (T5).
- Produces: the running epoch-2 host. `caliper_host::services_init()` (call once, after ImGui contexts exist), `services_get(id)`, `service_ids()`. This is UI/CMake glue — verified by build + the manual demo checklist below, not unit tests.

- [ ] **Step 1: Service tables**

`src/host/host_services.h`:
```cpp
#pragma once
#include <set>
#include <string>

namespace caliper_host {
// Host-side service registry (PLATFORM.md §6b). Call services_init() once
// after the ImGui/ImPlot/ImPlot3D contexts exist; tables are static and live
// for the process lifetime (the ABI's pointer-validity guarantee).
void services_init();
const void* services_get(const char* service_id);   // NULL for unknown ids
const std::set<std::string>& service_ids();
}
```

`src/host/host_services.cpp`:
```cpp
#include "host_services.h"
#include <caliper/services/log_v1.h>
#include <caliper/services/ui_v1.h>
#include <imgui.h>
#include <implot.h>
#include <implot3d.h>
#include <cstdio>
#include <cstring>
#include <ctime>

namespace caliper_host {
namespace {

// --- caliper.log.v1: timestamped console lines (console panel = later) ---
void log_impl(CaliperLogLevel level, const char* msg) {
    static const char* kTag[] = {"DEBUG", "INFO ", "WARN ", "ERROR"};
    int idx = (level >= 0 && level <= 3) ? (int)level : 1;
    std::time_t t = std::time(nullptr);
    char ts[16];
    std::strftime(ts, sizeof ts, "%H:%M:%S", std::localtime(&t));
    std::fprintf(stderr, "[%s] [%s] %s\n", ts, kTag[idx], msg ? msg : "");
}
const CaliperLogV1 kLog = {sizeof(CaliperLogV1), &log_impl};

// --- caliper.ui.v1: contexts + allocator handoff (§6d) ---
ImGuiContext*    ui_imgui()    { return ImGui::GetCurrentContext(); }
ImPlotContext*   ui_implot()   { return ImPlot::GetCurrentContext(); }
ImPlot3DContext* ui_implot3d() { return ImPlot3D::GetCurrentContext(); }
void ui_allocators(CaliperImGuiAllocFn* out_alloc, CaliperImGuiFreeFn* out_free,
                   void** out_user) {
    ImGuiMemAllocFunc a = nullptr; ImGuiMemFreeFunc f = nullptr; void* u = nullptr;
    ImGui::GetAllocatorFunctions(&a, &f, &u);
    *out_alloc = reinterpret_cast<CaliperImGuiAllocFn>(a);
    *out_free  = reinterpret_cast<CaliperImGuiFreeFn>(f);
    *out_user  = u;
}
const CaliperUiV1 kUi = {sizeof(CaliperUiV1), &ui_imgui, &ui_implot,
                         &ui_implot3d, &ui_allocators};

const std::set<std::string> kIds = {CALIPER_UI_V1, CALIPER_LOG_V1};

} // namespace

void services_init() { /* tables are static; hook kept for later services */ }

const void* services_get(const char* id) {
    if (!id) return nullptr;
    if (std::strcmp(id, CALIPER_UI_V1) == 0)  return &kUi;
    if (std::strcmp(id, CALIPER_LOG_V1) == 0) return &kLog;
    return nullptr;   // unknown ids: NULL, never UB (§6b)
}

const std::set<std::string>& service_ids() { return kIds; }

} // namespace caliper_host
```

- [ ] **Step 2: Rewire `main.cpp`**

Apply these exact changes to `src/main.cpp`:

(a) Replace `#include "applet_host.h"` with:
```cpp
#include "host/applet_loader.h"
#include "host/host_services.h"
#include "host/host_version.h"
#include "host/frame_watchdog.h"
```

(b) Replace the members `AppletHost host_;` and `int active_applet_ = -1;` with:
```cpp
    caliper_host::AppletLoader loader_{
        caliper_host::HostCaps{2, caliper_host::kHostVersionStr,
                               caliper_host::service_ids()},
        caliper::app_data_path("data")};
    caliper_host::FrameWatchdog watchdog_;
    int active_applet_ = -1;
    double last_frame_time_ = 0.0;
```
(`HostCaps.services` is a `std::set<std::string>` — `service_ids()` returns exactly that type.)

(c) In `initialize()`, replace the two `host_.scan(...)` calls with `loader_.scan(...)` (same directories), add `caliper_host::services_init();` right after `ImGui_ImplOpenGL3_Init(...)`, and replace the card-population loop with:
```cpp
        std::vector<AppletCard> cards;
        for (int i = 0; i < loader_.count(); i++) {
            const auto& e = loader_.at(i);
            std::string desc = e.manifest.summary;
            if (e.status != caliper_host::AppletStatus::Ready &&
                e.status != caliper_host::AppletStatus::Active)
                desc = "[unavailable] " + e.status_text + "\n\n" + desc;
            cards.push_back({e.manifest.name, e.manifest.summary, desc,
                             e.manifest.tag});
        }
        intro_.set_applets(std::move(cards));
```

(d) Replace the launch block (the `CaliperHostContext ctx{...}; host_.launch(...)` section) with:
```cpp
                    if (idx >= 0 && idx < loader_.count()) {
                        CaliperHost proto{};
                        proto.struct_size  = sizeof(CaliperHost);
                        proto.abi_epoch    = 2;
                        proto.host_version = caliper_host::kHostVersionU32;
                        proto.applet_data_dir = nullptr;   // loader fills per-applet
                        proto.get_service = [](const CaliperHost*, const char* id) {
                            return caliper_host::services_get(id);
                        };
                        if (loader_.launch(idx, proto)) {
                            active_applet_ = idx;
                            watchdog_.reset();
                            last_frame_time_ = glfwGetTime();
                            page_ = AppPage::Applet;
                            glfwSetWindowTitle(window_,
                                ("Caliper - " + loader_.at(idx).manifest.name).c_str());
                        }
                    }
```

(e) Replace the applet-page block (`host_.draw(...)` + menu bar name) with:
```cpp
                bool go_back = glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS;

                if (ImGui::BeginMainMenuBar()) {
                    if (ImGui::MenuItem("< Home")) go_back = true;
                    ImGui::Separator();
                    ImGui::TextDisabled("%s",
                        loader_.at(active_applet_).manifest.name.c_str());
                    if (watchdog_.flagged()) {
                        ImGui::Separator();
                        ImGui::TextColored({1.0f, 0.6f, 0.2f, 1.0f},
                            "slow: long work belongs in background jobs");
                    }
                    ImGui::EndMainMenuBar();
                }

                double now = glfwGetTime();
                int ww = 0, wh = 0;
                glfwGetWindowSize(window_, &ww, &wh);
                CaliperFrameInfo fi{};
                fi.struct_size = sizeof fi;
                fi.fb_width = dw; fi.fb_height = dh;              // physical px
                fi.dpi_scale = (ww > 0) ? (float)dw / (float)ww : 1.0f;
                fi.time_sec = now;
                fi.delta_sec = now - last_frame_time_;
                last_frame_time_ = now;

                double t0 = glfwGetTime();
                bool alive = loader_.frame(active_applet_, fi);
                watchdog_.feed((glfwGetTime() - t0) * 1000.0);

                if (!alive) go_back = true;   // quarantined mid-frame

                if (go_back) {
                    loader_.teardown(active_applet_);
                    active_applet_ = -1;
                    page_ = AppPage::Landing;
                    glfwSetWindowTitle(window_, "Caliper");
                    // refresh cards so refusal/quarantine text shows up
                    std::vector<AppletCard> cards;
                    for (int i = 0; i < loader_.count(); i++) {
                        const auto& e = loader_.at(i);
                        std::string desc = e.manifest.summary;
                        if (e.status != caliper_host::AppletStatus::Ready)
                            desc = "[unavailable] " + e.status_text + "\n\n" + desc;
                        cards.push_back({e.manifest.name, e.manifest.summary,
                                         desc, e.manifest.tag});
                    }
                    intro_.set_applets(std::move(cards));
                }
```

(f) In `cleanup()`: `host_.close_all();` → `loader_.close_all();`

- [ ] **Step 3: CMake — host sources + `/MD`**

Root `CMakeLists.txt`: add to the `caliper` executable sources: `src/host/host_services.cpp`; link `caliper_host_lib caliper::sdk` into `caliper` (`target_link_libraries(caliper PRIVATE caliper_host_lib caliper::sdk ${CALIPER_DEPENDENCY_LIBS})`). Keep `src/applet_host.cpp` in the build (dead but compiling) until T17.

Replace line 23 (`set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")`) with:
```cmake
# PLATFORM.md D7: dynamic CRT so host + applet DLLs share one heap. DuckDB
# under /MD is unverified until the first Windows session — CALIPER_STATIC_CRT
# is the escape hatch if it breaks.
option(CALIPER_STATIC_CRT "Use static MSVC runtime (/MT) — pre-D7 behavior" OFF)
if(CALIPER_STATIC_CRT)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
```

- [ ] **Step 4: Build + manual demo checklist (the Phase 1 demo)**

```bash
cmake -B build -DBUILD_TESTS=ON && cmake --build build -j && ctest --test-dir build --output-on-failure
./build/caliper
```
Verify by hand — every line must hold before committing:
1. Landing page shows the **Hello** and **SignalScope** cards (the three v1 applets have no manifests yet, so they correctly do NOT appear — expected shrinkage until the T14–16 ports land).
2. Launch Hello → sine plot animates; menu bar shows "Hello"; ESC returns to landing.
3. `CALIPER_HELLO_CRASH=1 ./build/caliper` → launching Hello bounces back to the landing page; Hello's card shows `[unavailable] crashed in frame(): SIGSEGV …`; the app keeps running.
4. Edit `build/applets/hello.caliper.toml` → `abi_epoch = 1`, relaunch app → card shows `[unavailable] Built for ABI epoch 1; this host speaks 2 — check for an applet update.` and won't launch. Restore the file (`cmake --build build --target hello_applet` re-copies it).
5. Launch SignalScope → three leads scroll; pause/speed/leads controls work; under "Anti-patterns" check "block the frame thread" → within ~1 s the menu bar shows the watchdog warning; uncheck → warning persists (latching) until you ESC out and relaunch the applet; relaunch also restores the speed slider from the saved settings (data-dir persistence).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(host): epoch-2 host — service registry, loader v2 wiring, FrameInfo, watchdog, /MD option"
```

---

### Task 14: Port circuitnet to epoch 2

**Files:**
- Rewrite: `applets/circuitnet/plugin.cpp`
- Create: `applets/circuitnet/circuitnet.caliper.toml`
- Modify: `applets/circuitnet/CMakeLists.txt` (manifest copy + C++20 already set)

**Interfaces:**
- Consumes: sugar (T10), running host (T13). `CircuitNetApplet` (in `circuitnet.h`) keeps its `initialize()/draw_ui(w,h)/cleanup()` methods — the port is entry-boilerplate only.

- [ ] **Step 1: Replace `plugin.cpp` entirely with**

```cpp
#include <caliper/caliper.hpp>
#include "circuitnet.h"

// Epoch-2 entry (PLATFORM.md §6a). All applet logic stays in CircuitNetApplet;
// this file is only the bridge. ui::connect is handled by the macro.
class CircuitNetPlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        (void)host;
        return impl_.initialize();
    }
    void on_frame(const caliper::Frame& f) override {
        impl_.draw_ui(f.fb_width, f.fb_height);   // physical px, same as v1
    }
    void on_cleanup() override { impl_.cleanup(); }

private:
    CircuitNetApplet impl_;
};

CALIPER_APPLET(CircuitNetPlugin,
    .id       = "dev.ahmed.circuitnet",
    .version  = "1.0.0",
    .name     = "CircuitNet 3.0",
    .summary  = "Gate-level circuit architecture explorer with DuckDB-powered "
                "querying, Verilog netlist parsing, and interactive graph "
                "visualization.",
    .tag      = "EDA",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1})
```

- [ ] **Step 2: Create `applets/circuitnet/circuitnet.caliper.toml`**

```toml
[applet]
id      = "dev.ahmed.circuitnet"
name    = "CircuitNet 3.0"
version = "1.0.0"
summary = "Gate-level circuit architecture explorer with DuckDB-powered querying, Verilog netlist parsing, and interactive graph visualization."
tag     = "EDA"

[compat]
abi_epoch = 2
min_host  = "0.6.0"

[services]
required = ["caliper.ui.v1", "caliper.log.v1"]
```

- [ ] **Step 3: Manifest copy in `applets/circuitnet/CMakeLists.txt`** (append)

```cmake
add_custom_command(TARGET circuitnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/circuitnet.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/circuitnet.caliper.toml)
```

- [ ] **Step 4: Build, test, smoke**

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure && ./build/caliper
```
Expected: tests green; CircuitNet card appears; launches; parses/queries as before; ESC returns cleanly.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(circuitnet): port to ABI epoch 2 (CALIPER_APPLET + manifest)"
```

---

### Task 15: Port opengllama to epoch 2

**Files:** rewrite `applets/opengllama/plugin.cpp`, create `applets/opengllama/opengllama.caliper.toml`, append manifest copy to its CMakeLists — same three steps as Task 14 with these exact contents:

plugin.cpp:
```cpp
#include <caliper/caliper.hpp>
#include "opengllama.h"

// NOTE: this applet still issues raw GL calls for its heatmaps — a known §6c
// violation, grandfathered until caliper.tensor_bridge.v1 lands in Phase 2.
class OpenGllamaPlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        (void)host;
        return impl_.initialize();
    }
    void on_frame(const caliper::Frame& f) override {
        impl_.draw_ui(f.fb_width, f.fb_height);
    }
    void on_cleanup() override { impl_.cleanup(); }

private:
    OpenGllamaApplet impl_;
};

CALIPER_APPLET(OpenGllamaPlugin,
    .id       = "dev.ahmed.opengllama",
    .version  = "0.1.0",
    .name     = "OpenGllama",
    .summary  = "Load GGUF models via llama.cpp and visualize layer activations "
                "with OpenGL-rendered heatmaps on Metal/CUDA backends.",
    .tag      = "LLM",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1})
```

opengllama.caliper.toml:
```toml
[applet]
id      = "dev.ahmed.opengllama"
name    = "OpenGllama"
version = "0.1.0"
summary = "Load GGUF models via llama.cpp and visualize layer activations with OpenGL-rendered heatmaps on Metal/CUDA backends."
tag     = "LLM"

[compat]
abi_epoch = 2
min_host  = "0.6.0"

[services]
required = ["caliper.ui.v1", "caliper.log.v1"]
```

In `applets/opengllama/CMakeLists.txt`: change `CXX_STANDARD 17` to `CXX_STANDARD 20` in `set_target_properties` (the sugar requires C++20), and append:
```cmake
add_custom_command(TARGET opengllama POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/opengllama.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/opengllama.caliper.toml)
```

- [ ] Build + ctest + smoke (card appears, launches, ESC works) → commit:
```bash
git add -A && git commit -m "feat(opengllama): port to ABI epoch 2 (CALIPER_APPLET + manifest)"
```

---

### Task 16: Port repnet_demo to epoch 2

**Files:** rewrite `applets/repnet_demo/plugin.cpp`, create `applets/repnet_demo/repnet_demo.caliper.toml`, append manifest copy + `CXX_STANDARD 20` to its CMakeLists — same shape as Task 15:

plugin.cpp:
```cpp
#include <caliper/caliper.hpp>
#include "repnet_demo.h"

class RepNetDemoPlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        (void)host;
        return impl_.initialize();
    }
    void on_frame(const caliper::Frame& f) override {
        impl_.draw_ui(f.fb_width, f.fb_height);
    }
    void on_cleanup() override { impl_.cleanup(); }

private:
    RepNetDemoApplet impl_;
};

CALIPER_APPLET(RepNetDemoPlugin,
    .id       = "dev.ahmed.repnet-demo",
    .version  = "1.0.0",
    .name     = "RepNet Demo",
    .summary  = "Load and visualize the UCDH Senior Design dataset, run signal "
                "preprocessing, inspect raw data via DuckDB, and run model "
                "inference on ECG recordings.",
    .tag      = "ECG",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1})
```

repnet_demo.caliper.toml:
```toml
[applet]
id      = "dev.ahmed.repnet-demo"
name    = "RepNet Demo"
version = "1.0.0"
summary = "Load and visualize the UCDH Senior Design dataset, run signal preprocessing, inspect raw data via DuckDB, and run model inference on ECG recordings."
tag     = "ECG"

[compat]
abi_epoch = 2
min_host  = "0.6.0"

[services]
required = ["caliper.ui.v1", "caliper.log.v1"]
```

In `applets/repnet_demo/CMakeLists.txt`: change `CXX_STANDARD 17` to `CXX_STANDARD 20` in `set_target_properties`, and append:
```cmake
add_custom_command(TARGET repnet_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/repnet_demo.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/repnet_demo.caliper.toml)
```

- [ ] Build + ctest + smoke — **including the Training Lab tab**: start a short training run, confirm live loss/AUROC plots still update and cancel works (this exercises `draw_ui` under real load, the riskiest port). Commit:
```bash
git add -A && git commit -m "feat(repnet_demo): port to ABI epoch 2 (CALIPER_APPLET + manifest)"
```

---

### Task 17: Delete the v1 loader — Phase 1 exit

**Files:**
- Delete: `src/applet_host.h`, `src/applet_host.cpp`, `src/applet_api.h` (shim), `sdk/include/caliper/abi_v1.h`
- Modify: `CMakeLists.txt` (remove `src/applet_host.cpp` from the `caliper` sources), `tests/sdk_install_probe/probe.cpp` (point at epoch 2), `APPLETS.md` (banner)

**Interfaces:** none new — this task removes the v1 surface. After it, `CALIPER_APPLET_ABI`, `CaliperHostContext`, `applet_info`/`applet_create`/… no longer exist anywhere.

- [ ] **Step 1: Delete + de-reference**

```bash
git rm src/applet_host.h src/applet_host.cpp src/applet_api.h sdk/include/caliper/abi_v1.h
```
Remove `src/applet_host.cpp` from the `add_executable(caliper …)` list in the root CMakeLists.

Replace `tests/sdk_install_probe/probe.cpp` with:
```cpp
// Compile-only proof that the installed caliper-sdk package is self-contained.
#include <caliper/abi.h>
#include <caliper/services/ui_v1.h>
#include <caliper/services/log_v1.h>
static_assert(CALIPER_ABI_EPOCH == 2, "epoch-2 SDK reachable from installed prefix");
```

Prepend to `APPLETS.md`:
```markdown
> **Superseded (Phase 1, PLATFORM.md §17):** applets now use ABI epoch 2 —
> `caliper_applet_descriptor()` + `CALIPER_APPLET` macro + `<name>.caliper.toml`
> manifest. See `examples/hello/` for the canonical minimal applet. The v1
> `applet_info`/six-function ABI described below no longer exists.
```

- [ ] **Step 2: Prove nothing references v1**

```bash
grep -rnE "applet_api\.h|CaliperHostContext|applet_info|CALIPER_APPLET_ABI" \
    src/ applets/ examples/ sdk/ tests/ CMakeLists.txt || echo "V1-GONE"
```
Expected: `V1-GONE`.

- [ ] **Step 3: Full verification — the Phase 1 exit**

```bash
rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build -j && ctest --test-dir build --output-on-failure
./scripts/test-sdk-install.sh build
./build/caliper
```
Expected: clean configure from scratch; all tests green; install probe OK; app shows **4 cards** (Hello, CircuitNet 3.0, OpenGllama, RepNet Demo), all launch and return.

- [ ] **Step 4: Commit + merge**

```bash
git add -A && git commit -m "refactor(host)!: delete v1 loader and ABI — epoch 2 only (Phase 1 exit)"
git checkout main && git merge --no-ff platform/phase-1 -m "Phase 1: ABI epoch 2 (PLATFORM.md §17)"
```

---

## Documentation Track (wiki alongside development)

Maintainability comes from three mechanisms, not from effort: (1) doc updates land in the same commit as the change (mapping table below — part of each task's definition of done); (2) reference pages **embed the real files** via `pymdownx.snippets` with `check_paths: true`, so a moved/renamed header breaks the docs build instead of silently orphaning the page; (3) `mkdocs build --strict` fails on broken links/anchors.

### Docs scaffold (execute between Task 4 and Task 5, on `platform/phase-1`)

- [ ] Create `mkdocs.yml` at the repo root:

```yaml
site_name: Caliper Platform
docs_dir: docs/wiki
site_dir: build/wiki-site
theme:
  name: material
  features:
    - navigation.sections
    - content.code.copy
    - search.suggest
markdown_extensions:
  - admonition
  - toc:
      permalink: true
  - pymdownx.superfences
  - pymdownx.snippets:
      base_path: ['.']
      check_paths: true
nav:
  - Home: index.md
  - Tutorials:
      - Your first applet: tutorials/first-applet.md
  - How-to:
      - Port a v1 applet: howto/port-v1-applet.md
      - Debug an applet: howto/debug-an-applet.md
  - Reference:
      - ABI (epoch 2): reference/abi.md
      - Manifest (caliper.toml): reference/manifest.md
      - Refusal messages: reference/refusals.md
      - Services:
          - caliper.ui.v1: reference/services/ui-v1.md
          - caliper.log.v1: reference/services/log-v1.md
      - C++ sugar: reference/sugar.md
  - Explanation:
      - Architecture: explanation/architecture.md
      - Compatibility & epochs: explanation/compatibility.md
      - Trust model: explanation/trust-model.md
  - Decisions: decisions/index.md
```

- [ ] Create the page skeleton: every file in the nav above, each starting with an H1 and a one-line "status: written at Task N" note so `--strict` passes from day one. Seed content: `index.md` = what Caliper is + links; `explanation/architecture.md` = adapted from PLATFORM.md §5.1–5.3 (layers diagram, frame loop); `explanation/compatibility.md` = adapted from §6b growth rules + §14 table; `decisions/index.md` = the §18 decision log table, copied with a pointer back to PLATFORM.md as source of truth until ADR files start.
- [ ] Verify: `pip install mkdocs-material && mkdocs build --strict` → exit 0. `mkdocs serve` renders with working nav/search.
- [ ] Commit: `docs(wiki): MkDocs scaffold (Diátaxis layout, strict build, snippet embedding)`

Example of the embed-don't-paste rule (`reference/abi.md` body):

````markdown
# ABI — epoch 2

The frozen contract. This page embeds the header verbatim; the build fails
if the file moves.

```c
--8<-- "sdk/include/caliper/abi.h"
```
````

### Task → page mapping (same-commit rule)

| Task | Page(s) due in that task's commit |
|---|---|
| T5 (ABI headers) | `reference/abi.md` (embed abi.h), stub service pages embed their headers |
| T6 (manifest) | `reference/manifest.md` — schema table + embedded golden example |
| T7 (negotiation) | `reference/refusals.md` — every contractual refusal string, verbatim, with "what the user should do" |
| T8/T9 (guard/watchdog) | `explanation/trust-model.md` — quarantine semantics (§15 honesty), watchdog rule |
| T10 (sugar) | `reference/sugar.md` — `CALIPER_APPLET` field order, `ui::connect`, fixture host usage |
| T11 (hello) | `tutorials/first-applet.md` — walk hello.cpp end to end (embed it); `howto/debug-an-applet.md` (attach LLDB, log tail) |
| T13 (host integration) | `reference/services/ui-v1.md`, `reference/services/log-v1.md` — semantics, lifetimes, threading notes |
| T14 (first port) | `howto/port-v1-applet.md` — the v1→v2 recipe, written while doing it (T15/T16 refine if they surprise) |
| T17 (v1 deletion) | sweep: remove v1 references from all pages; `index.md` links APPLETS.md successor content; final `mkdocs build --strict` |

Publishing (GitHub Pages CI job) and doc versioning (`mike`, per SDK release) are deliberately deferred to Phase 3 — until the SDK has external consumers, the wiki's audience is local (`mkdocs serve`) and GitHub's markdown rendering.

Generated API reference (PLATFORM.md D15): adopt `mkdocs-cxxdox` (libclang-based MkDocs plugin; alpha) when the Phase-2 service catalog lands — one `mkdocs.yml` block, additive to this site and trivially droppable. At adoption time, promote SDK header doc comments to Doxygen `/** */` style in a mechanical sweep; until then the embedded-header pages are the reference.

## Exit-Criteria Mapping (spec → proof)

| PLATFORM.md requirement | Proven by |
|---|---|
| Phase 0: applets compile against SDK, not `${CMAKE_SOURCE_DIR}` paths | T3 grep + T4 `find_package` probe script |
| Phase 1: descriptor + `get_service` + frame info (§6a/6b) | T5 headers + `test_abi.cpp` + `abi_c_check.c` |
| `ui_v1` with allocators (§6d) | T13 `host_services.cpp` + sugar `ui::connect` (T10) |
| `log_v1` (§7.1) | T13 impl; exercised by fixture + loader tests |
| Sugar + `CALIPER_APPLET` (§8) | T10 + `test_sugar.cpp` (descriptor, lifecycle, exception edge) |
| `caliper.toml` parsing (§10.3) | T6 golden + adversarial tests |
| Loader v2, negotiation manifest-first (§14 order) | T7 + T12 tests (refusal pre-dlopen, descriptor sanity) |
| Friendly failure cards | negotiation reason strings (T7, contractual) + card text (T13 demo checklist items 3–4) |
| Crash guard + watchdog (§15) | T8/T9 tests + T13 demo item 3 |
| Windows `/MD` (D7) | T13 CMake option — **compile-side only; DuckDB-under-/MD verification deferred to first Windows session** |
| All applets on epoch 2; v1 loader deleted | T14–16 ports + T17 grep + from-scratch build |

## Spec Deviations (deliberate, do not "fix" silently)

1. **Manifest filename** is `<stem>.caliper.toml` beside the dylib, not `caliper.toml` inside a bundle — bundles are Phase 4 (§10.4); the stem links manifest↔binary in a flat directory.
2. **`SHARED` not `MODULE`** for applet libraries — `caliper_add_applet()` owns that switch in Phase 3/4 (§10.2); changing it now would churn the root glob's stale-cleanup logic for no Phase-1 benefit.
3. **Failure cards reuse the existing card UI** (reason text prefixed into the description; `intro_screen.*` untouched). Dedicated card styling is polish for later phases.
4. **C++20 for SDK consumers** (sugar needs designated initializers), while §10.2 sketches C++17. Root default stays 17; consuming targets set 20 explicitly.
5. **`CaliperImGuiAllocFn/FreeFn` typedefs** replace §6d's direct use of `ImGuiMemAllocFunc` in the C header — §6c (no third-party types in the ABI) wins over the §6d sketch.
6. **`platforms = [...]`** from §10.3 is not parsed — Phase 1 has no bundle/platform selection; the sibling-dylib check plays that role. Add in Phase 4.
7. **UI stack stays a monorepo target** (`caliper::ui_stack`) until Phase 3 splits the SDK repo — the Phase 0 install probe therefore proves header/package consumption, not a full standalone applet build (that's Phase 3's golden-applet wall).
8. **Quarantined dylibs are never `dlclose`d** — running static destructors in a faulted image risks more than the leak costs (§15 honesty).

## Risks / Environment Notes

- **FetchContent needs network** on first configure (doctest, toml++). If offline, vendor the two single-header amalgams under `third_party/` and swap the CMake — do not skip the pins.
- **arm64 signal semantics:** null-deref may raise SIGBUS instead of SIGSEGV (tests accept both); integer div-by-zero does NOT raise SIGFPE on arm64 — don't "improve" the crash tests with one.
- **`parse_manifest_file` round-trips through the formatter** to keep one validation path; if toml++'s formatter output ever diverges (comments are dropped — fine), parsing behavior is unchanged.
- **CLion:** the plan builds in `build/`; CLion keeps using `cmake-build-debug/` independently. Reload CMake in the IDE after T2/T5 so targets appear.
- **Windows paths in this plan compile blind** (SEH guard, `/MD`, `AddDllDirectory` comes later) — first Windows session must run the whole ctest suite before trusting them.

## End Result (what exists after Task 17)

Same app, new spine: launching `./build/caliper` shows four cards (Hello + the three real applets), every applet loads through manifest-gated negotiation with friendly refusal cards, a crashing applet quarantines with the fault named on its card while the host keeps running, slow frames get flagged by the watchdog, all applet code talks to the host exclusively through `caliper_applet_descriptor` / `get_service` / `caliper.ui.v1` / `caliper.log.v1`, the SDK is an installable `find_package`-able package proven by script, ~40 unit/integration tests run in ctest, and the v1 ABI no longer exists in the tree. This is the launchpad for Phase 2 (services extraction: jobs/metrics/tensor-bridge from repnet_demo).

