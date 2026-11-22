# First Applet

*(New here? [Development basics](../development/development-basics.md) explains what you write and what the host provides, and where every library comes from.)*

This page walks through the **hello** applet, the smallest complete Caliper applet on ABI epoch 2. It has one manifest, one macro, and three lifecycle methods. For a small ML example, continue to `examples/sine_scope/`. No single example covers the whole service catalog.

```mermaid
flowchart LR
    src[Source and manifest] -->|cmake build| lib[Applet library]
    lib -->|caliper check| ready[READY]
    ready -->|caliper dev| run[Running applet]
```

The applet builds into a library, passes the host checks, and then runs in Caliper.

The whole applet lives in `examples/hello/`. It builds the same way in a folder of your own outside the repository:

```
examples/hello/
├── hello.caliper.toml   # the manifest — identity + what the host must provide
├── hello.cpp            # the applet — one class + one macro
└── CMakeLists.txt       # the build — find_package(caliper-sdk) + caliper_add_applet()
```

## Manifest

The host reads `hello.caliper.toml` **before it loads any of your code**. The manifest declares who the applet is, which ABI epoch it uses, and which framework services it needs:

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

The `required` services are a gate. If the host cannot provide `caliper.ui.v1` or `caliper.log.v1`, it refuses to load the applet. The `id`, `version`, and `services` **must agree** with the fields you pass to the `CALIPER_APPLET` macro. The loader checks both and rejects the applet if they differ. See [reference/manifest.md](manifest.md) for the full schema.

## Macro

One include and one class make the whole C++ surface. `#include <caliper/caliper.hpp>` pulls in the sugar layer (the `Applet`, `Host`, and `Frame` types) and the pinned ImGui and ImPlot stack. You program raw ImGui, with no wrapper to learn:

```cpp
--8<-- "examples/hello/hello.cpp"
```

The `CALIPER_APPLET(HelloApplet, ...)` macro at the bottom is the whole ABI boundary of the dylib. It generates the descriptor the loader looks for (`caliper_applet_descriptor`) and the exception-safe C bridge functions. It also generates the `ui::connect()` call, which shares the host's ImGui and ImPlot contexts and allocators with your dylib. As a result, `ImGui::` and `ImPlot::` calls land in the single UI world of the host. The field order is fixed: `id`, `version`, `name`, `summary`, `tag`, `services`.

## Applet Lifecycle

Your class overrides three methods from `caliper::Applet`:

- **`on_init(Host& host)`** runs once when the applet opens. The `Host&` stays valid for the whole life of the applet, so keep the pointer. Do setup here. Log through the host (`host.log_info(...)`) and never use `printf` (see [howto/debug-an-applet.md](applet-debugging.md) for why). Return `false` to abort loading.
- **`on_frame(const Frame& f)`** runs every frame. Everything visible happens here, and nothing slow, because you share the frame thread with the host and every other applet. `f.fb_width` and `f.fb_height` are **physical** pixels. `f.dpi_scale` converts them to the logical units that ImGui uses. Drive animation from `f.time_sec` and `f.delta_sec`, never from a wall-clock sleep.
- **`on_cleanup()`** runs when the applet closes. It mirrors `on_init`: persist, release, and log. After it returns, the host destroys your object, so do not touch host services afterwards.

Hello also reads `CALIPER_HELLO_CRASH` in `on_init`. When it is set, the applet faults in `on_frame` before any ImGui call. The crash-quarantine tests of the loader use this hook, and your own applets do not need it.

## Input Buttons

The button above the plot teaches three ideas of ImGui input that you will use in every applet:

- **A widget call both draws and reports.** `ImGui::Button(...)` draws the button and returns `true` on the single frame of the click. There is no callback and no event queue:
  ```cpp
  if (ImGui::Button(playing_ ? "Pause" : "Play")) playing_ = !playing_;
  ```
- **The label comes from state, every frame.** There is one button, and its label is an expression over `playing_`. The click flips the same `bool`, so the button relabels itself on the next frame.
- **The state lives in your applet, not in ImGui.** `playing_` is a member of `HelloApplet`, and ImGui only owns the pixels and the click. Thus the state is a `bool` on the class, initialized in the header, and not a `static` inside `on_frame`.

This is *immediate mode*: the UI is a function of your state, and you issue it again every frame.

A pausable animation needs its own clock. The value `f.time_sec` is the monotonic wall clock of the host, and it advances no matter what the applet does. Hello thus accumulates its own `phase_`, and it advances `phase_` by `f.delta_sec` **only while playing**:

```cpp
if (playing_) phase_ += (float)f.delta_sec;   // ...then draw sin(x + phase_)
```

Derive what you draw from state you control, and let the widgets edit that state. The same shape works for sliders (`SliderFloat` returns an edited value), checkboxes, and the live training controls of the ML applets.

## Build

An applet builds against the installed SDK, in a folder of its own outside the Caliper repository. First build Caliper and install the SDK from your checkout:

```bash
cmake --preset release
cmake --build --preset release
cmake --install cmake-build-release --prefix "$HOME/caliper-sdk" --component sdk
CALIPER="$PWD/cmake-build-release/caliper"   # the host you just built
```

Then copy hello to a folder of your own and build it there:

```bash
mkdir -p "$HOME/caliper-applets"
cp -R examples/hello "$HOME/caliper-applets/hello"
cd "$HOME/caliper-applets/hello"
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/caliper-sdk" -DCMAKE_BUILD_TYPE=Release
cmake --build build
ls build/applets/   # libhello.dylib + hello.caliper.toml
```

The `CMakeLists.txt` is short:

```cmake title="examples/hello/CMakeLists.txt"
--8<-- "examples/hello/CMakeLists.txt"
```

When you build the folder on its own, the `if` block declares the project and finds the SDK. Inside the Caliper repository, the core build also includes the folder as the fixture that the loader tests load. The core build then provides `caliper_add_applet()` itself. An applet of your own can drop the `if` and keep its three lines. `caliper_add_applet()` checks the manifest when you configure. It builds the library against `caliper::sdk` and `caliper::ui_stack`. It also writes a copy of the manifest beside the library, with a `[build]` table that records the SDK version and the UI-stack hash. [SDK package](sdk-package.md) describes the function and the package.

This tutorial copies hello so that the walk-through matches the code you build. To start an applet of your own, `caliper new <name>` writes a project of the same shape, as [Development basics](../development/development-basics.md#development-loop) shows.

## Applet Loading

First check the build. `caliper check` runs the checks that the host makes before it opens a library, without a window:

```bash
"$CALIPER" check build/applets
```

```text
READY  dev.caliper.hello  0.1.0  <your folder>/build/applets
1 of 1 applets pass the checks
```

Then start Caliper with that folder searched first:

```bash
"$CALIPER" dev build/applets
```

On Windows the executable is `cmake-build-release\caliper.exe`. The `caliper dev` command puts the folder ahead of every other applet folder. The release preset also builds its own copy of Hello beside the executable for the loader tests. At equal versions, the host loads the copy it finds first, which is yours, and lists the other copy as `SHADOWED`. The host keeps a library loaded until it quits. After each rebuild, quit Caliper and run `caliper dev` again.

To keep the applet installed, `"$CALIPER" install build/applets` runs the same checks. It then copies the library and its manifest into the user applets folder, which the host searches at every start. [Caliper CLI](../reference/cli.md) describes every subcommand.

## App Visibility

Hello appears as a row in the **Instruments** pane of the landing screen. Select the row and press Enter, or double-click it, to open the applet. The loader finds every applet through two signals: its `<stem>.caliper.toml` manifest and the `caliper_applet_descriptor` export. If the row is missing, run `caliper check` on the folder. If its status is `REFUSED`, `FAILED`, or `QUARANTINED` and not `READY`, the **Inspector** shows the reason of the loader under Status (see the [refusal reference](../reference/refusal-messages.md)). Esc, or the `esc Home` button, returns to the landing screen. [Development basics](../development/development-basics.md#landing-screen) describes the panes and keys.
