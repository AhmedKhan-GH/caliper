---
kind: reference
---

# Applet Commands

This page describes the `caliper` subcommands that list, check, install, create and build applets, and that print host information.

## `caliper list`

Lists every applet in the search folders, without opening any library. Applets are ordered by the position of their folder in the [search order](cli.md#search-order), then by id, then from the highest version down. Each applet takes one line, with its status, id, version, kind and folder in columns. The kind is `library` for a compiled applet and `spec` for a [spec applet](../applets/manifest.md#spec-applets), whose folder is the applet's own folder. A refused, failed or shadowed applet has its reason on the next line, indented under the id:

```text
READY     dev.caliper.mnist        0.1.0  library  <development folder>
SHADOWED  dev.caliper.mnist        0.1.0  library  <user applets folder>
          Version 0.1.0 in <development folder> is used instead.
READY     local.loss-view          0.1.0  spec     <user applets folder>/loss_view
READY     dev.caliper.hello-world  0.1.0  library  <applets folder beside the executable>
```

`--workspace <dir>` names the workspace whose spec applets `list` shows. Without it, `list` shows those of `CALIPER_WORKSPACE` or of the default workspace, as the app does.

The status is `READY`, `REFUSED`, `FAILED` or `SHADOWED`, and the reasons are those in [Refusal messages](refusal-messages.md). Folders print as absolute paths. `list` makes only the checks of the listing. It reads a spec applet's manifest and the first lines of its spec, and it does not validate the spec. When no folder holds an applet, `list` prints `no applets found`. It always exits with 0.

## `caliper check`

Runs the checks that the host makes before it opens a library on every applet in one folder, and only that folder. It uses this host's capabilities (see `caliper info`) and opens no library. The data-schema check reads the stamps in the app-data folder. The output is the same lines as `list`, then a summary:

```text
READY  dev.caliper.mnist  0.1.0  library  <folder>
1 of 1 applets pass the checks
```

When `<folder>` is a build folder whose applets are in its `applets/` subfolder (see [Build Folders](cli.md#build-folders)), `check` uses that subfolder and says so first:

```text
using the applets in <folder>/applets
READY  dev.caliper.mnist  0.1.0  library  <folder>/applets
1 of 1 applets pass the checks
```

`check` validates each [spec applet](../applets/manifest.md#spec-applets) that passes the listing checks in full, with the builder's spec library, as a launch does. Under each applet it prints every error and warning of the validator, with the spec's file name, its line and column, its severity and its stable code. A spec with an error fails, and its first error is its reason. Warnings do not fail it. `<folder>` may be the folder of one spec applet, or a folder that holds spec applets one folder down, such as a workspace's `applets/` folder:

```text
FAILED  local.broken     0.1.0  spec  <folder>/broken
        broken.spec.toml:9:14: error spec.unknown_source: The binding "nope.loss" names "nope", which is not a source of this spec.
READY   local.loss-view  0.1.0  spec  <folder>/loss_view
        loss_view.spec.toml:6:1: warning spec.unused_source: No panel binds the source "metrics", directly or through from.
1 of 2 applets pass the checks
```

A host built without the builder lists every spec applet as `REFUSED`, so none passes there.

An applet passes when it is `READY` or `SHADOWED`. `check` exits with 0 when every applet passes. It exits with 1 in these cases:

- Any applet does not pass.
- `<folder>` is not a folder (`caliper: not a folder: <folder>`).
- Neither `<folder>` nor its `applets/` subfolder holds an applet (`caliper: no applets in <folder>`).

## `caliper install`

Runs the same checks on a local folder. When every applet passes, it copies each applet into the user applets folder: its library, then its asset folder `assets/<stem>/` beside the library if there is one, then its manifest. An applet that reads files from beside its library keeps them in that folder, and its own build stages them there. An installed asset folder replaces the old one as a whole. Other files beside the library are not copied. A build folder is resolved as for `check`, with the same `using the applets in` line.

The install is all or nothing for the whole folder. Every file and asset folder is first copied to a hidden temporary name beside its destination. Then each one is renamed into place, each manifest after its library and assets, so the host never sees half a library or a manifest without its library. If a step fails, `install` removes what it placed, puts back what it replaced, and leaves the user applets folder as it was. The output is the lines of `check`, then one line per applet:

```text
installed dev.caliper.mnist 0.1.0 in <user applets folder>
```

If any applet fails a check, `install` copies nothing, prints `caliper: nothing installed; N of M applets do not pass the checks`, and exits with 1. `install` does not take spec applets, which live in a workspace. When the folder holds one, it copies nothing, prints `caliper: nothing installed; <id> is a spec applet, which install does not take`, and exits with 1.

`install` also exits with 1 when `<folder>` is the user applets folder itself, when that folder cannot be created, or when a copy fails. It reports these as `caliper: nothing installed; <reason>`. A URL in place of a folder is a usage error (`caliper: install takes a local folder, not <url>`), with exit code 2.

## `caliper new`

Creates an applet project from a template built into the executable. `<name>` is either a short name or a full applet id:

- A short name is lowercase letters, digits and `-`. It starts with a letter, does not end with `-`, and has at most 64 characters. The applet's id is `dev.example.<name>`.
- The stem names the CMake targets `<stem>` and `<stem>_tests`. Thus the stem may not be a name that CMake or one of its generators reserves (`all`, `all_build`, `clean`, `depend`, `edit_cache`, `help`, `install`, `list_install_components`, `package`, `package_source`, `preinstall`, `rebuild_cache`, `run_tests`, `test` and `zero_check`). It may not start with `caliper_`, which the SDK uses. `new` refuses such a name with `caliper: not a usable applet name: <name> (the CMake target "<target>" is reserved; choose another name)`.
- A full id, such as `com.example.my-viewer`, is used as given, and its last part is the short name.

From the short name, `new` derives the stem (hyphens become underscores: `my_viewer`), the display name (`My Viewer`) and the class name (`MyViewerApplet`). The project goes into `<dir>`, or into a folder named after the short name in the current folder. `new` writes nothing and exits with 1 when the folder already exists. It exits with 2 for a name that breaks the rules above.

| File | Contents |
|---|---|
| `CMakeLists.txt` | Finds `caliper-sdk` with this host's SDK version as the minimum, builds the applet with `caliper_add_applet()`, and builds and registers the test |
| `<stem>.caliper.toml` | The manifest: the id, the display name, version `0.1.0`, this host's ABI epoch, this host's version as `min_host`, and `caliper.ui.v1` and `caliper.log.v1` as required services |
| `<stem>.cpp` | One window, docked in the center through a dock hint, with a Play and Pause button and a sine plot, and the `CALIPER_APPLET` descriptor |
| `tests/test_<stem>.cpp` | A doctest case on the SDK's fixture host that checks the descriptor against the manifest and runs the lifecycle without a window |

```text
created <dir> for applet dev.example.my-viewer
  CMakeLists.txt
  my_viewer.caliper.toml
  my_viewer.cpp
  tests/test_my_viewer.cpp
```

The comment at the top of the generated `CMakeLists.txt` gives the next steps: configure with `-DCMAKE_PREFIX_PATH=<sdk prefix>`, build, run `ctest --test-dir build`, and run `caliper dev build/applets`. [caliper build](#caliper-build) does the first two and the checks in one step.

## `caliper build`

Configures and builds an applet's folder against the installed SDK with CMake, then runs the checks of [caliper check](#caliper-check) on the result. The folder must hold a `CMakeLists.txt` that finds `caliper-sdk`, such as one that `caliper new` writes. The build goes into `<folder>/build`, in the Release configuration. The library and its manifest land in `<folder>/build/applets`, where `caliper dev` and `caliper install` take them. It runs these steps and prints what CMake prints as it comes:

```sh
cmake -S <folder> -B <folder>/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<sdk prefix>
cmake --build <folder>/build --config Release --parallel <cores>
caliper check <folder>/build
```

```text
caliper build my-viewer
using the Caliper SDK in /home/me/caliper-sdk
configuring /home/me/my-viewer/build with /usr/local/bin/cmake
...
using the applets in /home/me/my-viewer/build/applets
READY  dev.example.my-viewer  0.1.0  /home/me/my-viewer/build/applets
1 of 1 applets pass the checks
```

`build` finds the SDK's prefix without being told. It takes the first of these that holds `lib/cmake/caliper-sdk/caliper-sdk-config.cmake`:

1. `CALIPER_SDK_DIR`, when it is set and not empty. A folder without the SDK is an error.
2. The prefix the executable is installed in: the folder above the executable's folder, then the executable's folder.
3. `caliper-sdk` in the home folder, where the [tutorials](../development/development-basics.md#development-loop) install the SDK.

When none of them holds the SDK, `build` says where it looked and lets CMake search its own places, such as `CMAKE_PREFIX_PATH` in the environment and the system prefixes.

`build` runs `CALIPER_CMAKE` when it is set and not empty. Otherwise it runs `cmake` from `PATH`, and on macOS, as a last choice, the command of CMake.app in `/Applications`. CMake chooses the generator and the compiler as usual, so `CMAKE_GENERATOR` and `CXX` in the environment apply. A folder that was built before is configured and built again incrementally.

`build` exits with 0 when the build passes the checks, and with 2 for a wrong command line. It exits with 1 in these cases:

- `<folder>` is not a folder or holds no `CMakeLists.txt`.
- No cmake is found (`caliper: cmake was not found: install CMake 3.18 or later and put it on PATH, or set CALIPER_CMAKE to its path`), or `CALIPER_CMAKE` names something that is not a program.
- `CALIPER_SDK_DIR` names a folder without the SDK.
- The configure fails. When CMake found no working C++ compiler, no build tool for its generator, or no SDK, a last line says which and how to install it. An example is `caliper: no Caliper SDK was found. Install it from a Caliper build with cmake --install <build folder> --prefix <prefix> --component sdk, then set CALIPER_SDK_DIR=<prefix> or use the prefix ~/caliper-sdk`.
- The build fails, or an applet in the build folder fails the checks.

`build` is built and tested on macOS. The Windows build refuses it (see [Windows Limits](cli.md#windows-limits)). On Windows, configure and build with CMake as [SDK package](../applets/sdk-package.md#package-usage) describes.

## `caliper info`

Prints the host's version, its ABI epoch, the SDK version it was built with, its libtorch version, its UI-stack hash, and one line for each service it serves, in sorted order. A value that the build does not define prints as `unknown` for the SDK and `none` for libtorch and the UI-stack hash. This output comes from a release build:

```text
host 0.6.0
abi_epoch 3
sdk 0.2.0
libtorch 2.5.1
ui_stack 03c557182adc5cd9
service caliper.artifacts.v1
service caliper.data.v1
...
service caliper.ui.v1_1
```
