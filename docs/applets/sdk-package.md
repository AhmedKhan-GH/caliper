# SDK Package

The SDK installs as the `caliper-sdk` CMake package, version 0.2.0. An applet builds against it from its own folder with `find_package(caliper-sdk)` and `caliper_add_applet()`. It needs nothing else from the Caliper source tree. The host runs on macOS 14 (Sonoma) or later, and an applet built against the package targets the same minimum unless its project chooses another. [The SDK version](../development/compatibility-and-epochs.md#sdk-version) describes how the SDK version relates to the ABI epoch.

## Installation

Install the package from a core build folder:

```sh
cmake --preset release
cmake --build --preset release
cmake --install cmake-build-release --prefix <sdk prefix> --component sdk
```

The package records the version and the absolute path of the libtorch of the build that installed it. If you use the SDK on another machine, or if that libtorch moves, set `-DCALIPER_TORCH_DIR=<libtorch root>` to a copy of the same version (see [Variables](#variables)). An SDK installed from a build with `CALIPER_WITH_TORCH=OFF`, such as the `release-notorch` preset, names no libtorch and cannot build applets that link it.

## Installed Files

| Path under the prefix | Contents |
|---|---|
| `include/caliper/` | The ABI, the interchange types, the service headers, the C++ sugar and the adapters |
| `include/caliper-testing/` | The fixture host (`caliper/fixture_host.h`) and doctest (`doctest/doctest.h`) |
| `share/caliper-sdk/ui_stack/` | The pinned sources of ImGui, ImPlot, ImPlot3D and ImGuiFileDialog with their licenses, and `ui_stack.txt`, the inputs of the [UI-stack hash](#ui-stack-hash) |
| `share/caliper-sdk/licenses/` | The license of doctest |
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

Configure the applet with the prefix on `CMAKE_PREFIX_PATH`:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
cmake --build build
```

`find_package(caliper-sdk 0.2)` accepts an installed SDK with major version 0 at or above 0.2. `find_package(caliper-sdk)` with no version accepts any installed SDK. While the major version is 0, that includes a later 0.x SDK built for another ABI epoch. In that case `caliper_add_applet()` stops the configure step, because the `abi_epoch` of the manifest must equal the epoch of the SDK.

### Targets

| Target | What it gives |
|---|---|
| `caliper::sdk` | The headers under `include/caliper/`. It compiles nothing. |
| `caliper::ui_stack` | A static library compiled inside the build of the applet from the pinned sources, with position-independent code and the UI-stack definitions of the host. It holds no GLFW or ImGui backend, because an applet draws into the ImGui context of the host and never talks to a window or a graphics API. |
| `caliper::sdk_testing` | The fixture host and doctest, for the tests of an applet without a window. See [C++ sugar](cpp-sugar.md). |

The UI stack ships as sources because a prebuilt static library would tie every applet to the compiler, standard library, build type and runtime library of the SDK build. For example, on Windows a Release `/MD` library does not link into a Debug applet, and ImGuiFileDialog uses the STL. The applet compiles the sources with its own toolchain, which avoids these problems. The UI-stack hash names the commits, the `imconfig.h` and the definitions that the compile used.

### Variables

| Variable | Value |
|---|---|
| `CALIPER_SDK_VERSION` | The SDK version, `0.2.0` |
| `CALIPER_SDK_ABI_EPOCH` | The ABI epoch the headers speak, `3` |
| `CALIPER_UI_STACK_HASH` | The [UI-stack hash](#ui-stack-hash) |
| `CALIPER_TORCH_VERSION` | The version of the libtorch of the host, or empty when the SDK was built without libtorch |
| `CALIPER_MACOS_MIN` | The minimum macOS of the host, `14.0` by default. On macOS it becomes `CMAKE_OSX_DEPLOYMENT_TARGET` unless the project, `-DCMAKE_OSX_DEPLOYMENT_TARGET` or the environment variable `MACOSX_DEPLOYMENT_TARGET` chose one. |
| `CALIPER_TORCH_DIR` | Where that libtorch is. It defaults to the absolute path of the libtorch of the build that installed the SDK, recorded in the package. It is a cache variable, so `-DCALIPER_TORCH_DIR=<libtorch root>` points it at another copy of the same version. |

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
| `SOURCES` | The source files of the applet. Required. |
| `MANIFEST` | The manifest the author writes. Required. Its file name must be `<stem>.caliper.toml`, because the loader pairs a manifest with the library of the same stem. |
| `OUTPUT_NAME` | The stem of the library and the manifest. The default is `<target>`. |
| `OUTPUT_DIRECTORY` | Where the library and the stamped manifest go. The default is `applets/` in the top build folder. A relative path starts at the current build folder. |
| `TORCH` | Link the libtorch that the SDK names. See [Linking libtorch](#libtorch-linking). |

The function builds the shared library `lib<stem>.dylib` on macOS or `<stem>.dll` on Windows as C++20. It links the library privately to `caliper::sdk` and `caliper::ui_stack`, and it defines `CALIPER_APPLET_EXPORT`. The library and the stamped manifest always share one folder, also under multi-configuration generators.

### Configure-Time Checks

The configure step fails, and names the manifest and the problem, when one of these is true:

- `[applet]` lacks `id`, `name` or `version`, or one of them is not a string.
- `id` is not a valid [applet id](manifest.md#applet-ids), `name` is empty, or `version` is not strict semver.
- `[compat]` lacks `abi_epoch`, `abi_epoch` is not an integer of at least 1, or it differs from the epoch of the SDK.
- `[compat].min_host` is present and is not strict semver.
- The manifest has a `[build]` table.
- `[compat].torch` is present without `TORCH`, or it differs from the libtorch version of the SDK.

The check reads only the scalar keys of `[applet]` and `[compat]`. It reads them from `[table]` headers and single-line `key = value` pairs with string or integer values.

### Stamped Manifest

Beside the library, `caliper_add_applet()` writes a copy of the manifest with a `[build]` table appended:

```toml
# Written by caliper_add_applet() when the applet was built.
[build]
sdk = "0.2.0"
ui_stack = "<the UI-stack hash>"
```

With `TORCH`, the copy also gets `torch = "<version>"` in its `[compat]` table, unless the source already states it. The function writes the copy into the build folder at configure time and copies it beside the library after every link.

Under the Ninja and Makefile generators, a changed copy also relinks the library through `LINK_DEPENDS`. Thus the manifest beside the library never falls behind. The Xcode and Visual Studio generators ignore `LINK_DEPENDS`. After a change to the manifest alone, the copy beside the library stays stale until the next link, so delete the library or rebuild with `--clean-first`.

[Manifest](manifest.md#schema) describes the keys. [Compatibility and epochs](../development/compatibility-and-epochs.md#versioning) describes how the host uses them.

### `libtorch` Linking

With `TORCH`, `caliper_add_applet()` does these steps:

- It fails at configure time when the SDK was built without libtorch.
- It finds libtorch at `CALIPER_TORCH_DIR`, and it fails unless the version equals `CALIPER_TORCH_VERSION`.
- It links libtorch. On macOS it also sets the build rpath to the `lib` folder of libtorch and links `libobjc`, which the torch adapter needs.
- It copies no libtorch DLL on Windows, because the host executable carries them and an applet never brings a second copy.

A process holds one libtorch, so an applet links the version of the host or none (see [Embedding](../introduction/architecture.md#embedding)).

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

A commit is the checked-out commit of the git checkout of the library. Without one, it is the commit that the superproject pins. Without that, it is `sha256:` followed by the SHA-256 of the main source file of the library.

Thus the hash covers the ImGui version number, the committed state of each library, the content of `imconfig.h` and the definitions. An uncommitted edit to the sources of ImGui, ImPlot, ImPlot3D or ImGuiFileDialog does not change it, so commit a pin first if you change it on purpose. The core build compiles the hash into the host and installs the lines as `share/caliper-sdk/ui_stack/ui_stack.txt`.
