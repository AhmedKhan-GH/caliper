# SDK Package

The SDK installs as the `caliper-sdk` CMake package, version 0.2.0. An applet builds against it from a folder of its own with `find_package(caliper-sdk)` and `caliper_add_applet()`, and needs nothing else from the Caliper source tree. On macOS the host runs on macOS 14 (Sonoma) or later, and an applet built against the package is built for the same minimum unless its project chooses another. How the SDK version relates to the ABI epoch is in [The SDK version](../development/compatibility-and-epochs.md#sdk-version).

## Installation

The package is installed from a core build folder:

```sh
cmake --preset release
cmake --build --preset release
cmake --install cmake-build-release --prefix <sdk prefix> --component sdk
```

The package records the version and the absolute path of the libtorch of the build that installed it. An SDK used on another machine, or after that libtorch moves, needs `-DCALIPER_TORCH_DIR=<libtorch root>` pointing at a copy of the same version (see [Variables](#variables)). An SDK installed from a build with `CALIPER_WITH_TORCH=OFF`, such as the `release-notorch` preset, names no libtorch and cannot build applets that link it.

## Installed Files

| Path under the prefix | Contents |
|---|---|
| `include/caliper/` | The ABI, the interchange types, the service headers, the C++ sugar and the adapters |
| `include/caliper-testing/` | The fixture host (`caliper/fixture_host.h`) and doctest (`doctest/doctest.h`) |
| `share/caliper-sdk/ui_stack/` | The pinned sources of ImGui, ImPlot, ImPlot3D and ImGuiFileDialog with their licences, and `ui_stack.txt`, the inputs of the [UI-stack hash](#ui-stack-hash) |
| `share/caliper-sdk/licenses/` | The licence of doctest |
| `lib/cmake/caliper-sdk/` | The package files, including `caliper_add_applet()` |

## Package Usage

An applet's `CMakeLists.txt` finds the package and calls `caliper_add_applet()`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_applet LANGUAGES CXX)
find_package(caliper-sdk 0.2 CONFIG REQUIRED)

caliper_add_applet(my_applet
    SOURCES my_applet.cpp
    MANIFEST my_applet.caliper.toml)
```

It is configured with the prefix on `CMAKE_PREFIX_PATH`:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
cmake --build build
```

`find_package(caliper-sdk 0.2)` accepts an installed SDK with major version 0 at or above 0.2, and `find_package(caliper-sdk)` with no version accepts any installed SDK. While the major version is 0, that includes a later 0.x SDK built for another ABI epoch; `caliper_add_applet()` then stops the configure, because the manifest's `abi_epoch` must equal the SDK's epoch.

### Targets

| Target | What it gives |
|---|---|
| `caliper::sdk` | The headers under `include/caliper/`. It compiles nothing. |
| `caliper::ui_stack` | A static library compiled inside the applet's build from the pinned sources, with position-independent code and the host's UI-stack definitions. It holds no GLFW, GLEW or ImGui backend, because an applet draws into the host's ImGui context and never talks to a window or a graphics API. |
| `caliper::sdk_testing` | The fixture host and doctest, for an applet's own tests without a window. See [C++ sugar](cpp-sugar.md). |

The UI stack ships as sources rather than prebuilt libraries because a prebuilt static library would tie every applet to the compiler, standard library, build type and runtime library of the SDK build: on Windows a Release `/MD` library does not link into a Debug applet, and ImGuiFileDialog uses the STL. Compiling the sources with the applet's own toolchain avoids that, and the UI-stack hash names the commits, the `imconfig.h` and the definitions they were compiled from. The cost is a few seconds of compile time in each applet build folder.

### Variables

| Variable | Value |
|---|---|
| `CALIPER_SDK_VERSION` | The SDK version, `0.2.0` |
| `CALIPER_SDK_ABI_EPOCH` | The ABI epoch the headers speak, `2` |
| `CALIPER_UI_STACK_HASH` | The [UI-stack hash](#ui-stack-hash) |
| `CALIPER_TORCH_VERSION` | The version of the host's libtorch, or empty when the SDK was built without libtorch |
| `CALIPER_MACOS_MIN` | The host's minimum macOS, `14.0` by default. On macOS it becomes `CMAKE_OSX_DEPLOYMENT_TARGET` unless the project, `-DCMAKE_OSX_DEPLOYMENT_TARGET` or the environment variable `MACOSX_DEPLOYMENT_TARGET` chose one. |
| `CALIPER_TORCH_DIR` | Where that libtorch is. It defaults to the absolute path of the libtorch of the build that installed the SDK, recorded in the package. A cache variable, so `-DCALIPER_TORCH_DIR=<libtorch root>` points it at another copy of the same version. |

## `caliper_add_applet()`

```cmake
caliper_add_applet(<target>
    SOURCES <source>...
    MANIFEST <stem>.caliper.toml
    [OUTPUT_NAME <stem>]
    [OUTPUT_DIRECTORY <dir>]
    [TORCH])
```

| Argument | Meaning |
|---|---|
| `SOURCES` | The applet's source files. Required. |
| `MANIFEST` | The manifest the author writes. Required. Its file name must be `<stem>.caliper.toml`, because the loader pairs a manifest with the library of the same stem. |
| `OUTPUT_NAME` | The stem of the library and the manifest. The default is `<target>`. |
| `OUTPUT_DIRECTORY` | Where the library and the stamped manifest go. The default is `applets/` in the top build folder. A relative path is taken from the current build folder. |
| `TORCH` | Link the libtorch that the SDK names. See [Linking libtorch](#libtorch-linking). |

It builds the shared library `lib<stem>.dylib` on macOS or `<stem>.dll` on Windows as C++20, links it privately to `caliper::sdk` and `caliper::ui_stack`, and defines `CALIPER_APPLET_EXPORT`. The library and the stamped manifest always share one folder, also under multi-configuration generators.

### Configure-Time Checks

The configure step fails, naming the manifest and the problem, when:

- `[applet]` lacks `id`, `name` or `version`, or one of them is not a string;
- `id` is not a valid [applet id](manifest.md#applet-ids), `name` is empty, or `version` is not strict semver;
- `[compat]` lacks `abi_epoch`, or `abi_epoch` is not an integer of at least 1, or it differs from the SDK's epoch;
- `[compat].min_host` is present and not strict semver;
- the manifest has a `[build]` table;
- `[compat].torch` is present without `TORCH`, or differs from the SDK's libtorch version.

The check reads only the scalar keys of `[applet]` and `[compat]`, from `[table]` headers and single-line `key = value` pairs with string or integer values.

### Stamped Manifest

Beside the library, `caliper_add_applet()` writes a copy of the manifest with a `[build]` table appended:

```toml
# Written by caliper_add_applet() when the applet was built.
[build]
sdk = "0.2.0"
ui_stack = "<the UI-stack hash>"
```

With `TORCH`, the copy also gets `torch = "<version>"` in its `[compat]` table, unless the source already states it. The copy is written into the build folder at configure time and copied beside the library after every link. Under the Ninja and Makefile generators a changed copy also relinks the library, through `LINK_DEPENDS`, so the manifest beside the library never falls behind. The Xcode and Visual Studio generators ignore `LINK_DEPENDS`: after a change to the manifest alone, the copy beside the library stays stale until the next link, so delete the library or rebuild with `--clean-first`. The keys are described in [Manifest](manifest.md#schema), and how the host uses them in [Compatibility and epochs](../development/compatibility-and-epochs.md#versioning).

### `libtorch` Linking

With `TORCH`, `caliper_add_applet()`:

- fails at configure time when the SDK was built without libtorch;
- finds libtorch at `CALIPER_TORCH_DIR`, and fails unless its version equals `CALIPER_TORCH_VERSION`;
- links it, and on macOS sets the build rpath to its `lib` folder and links `libobjc`, which the torch adapter needs;
- copies no libtorch DLL on Windows, because the host executable carries them and an applet never brings a second copy.

A process holds one libtorch, so an applet links the host's version or none (see [Embedding](../introduction/architecture.md#embedding)).

## UI-Stack Hash

The hash is the first 16 hexadecimal digits of the SHA-256 of these seven lines, each ending in a newline:

```text
imgui_version=<IMGUI_VERSION_NUM from imgui.h>
imgui=<commit>
implot=<commit>
implot3d=<commit>
ImGuiFileDialog=<commit>
imconfig=<SHA-256 of imgui/imconfig.h>
definitions=<the UI-stack definitions, sorted and joined with ",">
```

A commit is the checked-out commit of the library's own git checkout. Without one it is the commit the superproject pins, and without that it is `sha256:` followed by the SHA-256 of the library's main source file. The hash therefore covers the ImGui version number, the committed state of each library, the content of `imconfig.h` and the definitions. An uncommitted edit to the sources of ImGui, ImPlot, ImPlot3D or ImGuiFileDialog does not change it, so a pin that is changed on purpose is committed first. The core build computes the hash in `cmake/CaliperUiStack.cmake`, compiles it into the host, and installs the lines as `share/caliper-sdk/ui_stack/ui_stack.txt`.

## In-Tree Applets

`caliper_add_applet()` is the same file in the core build and in the package. The core build compiles no applets unless `CALIPER_BUILD_APPLETS=ON`; `examples/hello` builds with the tests as the loader-test fixture, and `examples/sine_scope` builds with the tests in a build with libtorch. Every applet folder under `applets/` and `examples/` also builds on its own against the installed SDK, and `scripts/build_applets_against_sdk.sh <core build folder> <work folder> [applet folder...]` installs the SDK and builds them that way, as CI does. The core build no longer deletes applet libraries it did not build, so an older build folder can keep libraries from in-tree builds that predate `caliper_add_applet()`; their manifests have no `[build]` table, so the host does not check them against the UI-stack hash. Delete the build's `applets/` folder after turning `CALIPER_BUILD_APPLETS` off, and when reusing such a build folder.
