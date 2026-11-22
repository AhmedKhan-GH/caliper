# First Applet

*(New here? [Development basics](../development/development-basics.md) explains what you write vs. what the host provides, and where every library comes from.)*

This walks the **hello** applet — the smallest complete Caliper applet on ABI
epoch 2. It is the canonical starting point: one manifest, one macro, three
lifecycle methods. Continue to `examples/sine_scope/` for a small ML-oriented
example. The in-tree applets demonstrate larger combinations of services; no
single exemplar should be treated as covering the entire service catalog.

The whole applet lives in `examples/hello/`, and it builds the same way in a
folder of your own outside the repository:

```
examples/hello/
├── hello.caliper.toml   # the manifest — identity + what the host must provide
├── hello.cpp            # the applet — one class + one macro
└── CMakeLists.txt       # the build — find_package(caliper-sdk) + caliper_add_applet()
```

## Manifest

The host reads `hello.caliper.toml` **before it loads any of your code**. It
declares who the applet is, which ABI epoch it was compiled against, and which
framework services it needs to run:

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

`required` services are a gate: if the host cannot provide `caliper.ui.v1` or
`caliper.log.v1`, it refuses to load the applet rather than handing you a
half-working `Host`. The `id`, `version`, and `services` here **must agree** with
the fields you pass to the `CALIPER_APPLET` macro (below) — the loader verifies
the two and rejects the applet if they drift. See
[reference/manifest.md](manifest.md) for the full schema.

## Macro

One include and one class is the entire C++ surface. `#include
<caliper/caliper.hpp>` pulls in the sugar layer (the `Applet`/`Host`/`Frame`
types) *and* the pinned ImGui/ImPlot stack — there is no wrapper to learn, you
program raw ImGui:

```cpp
--8<-- "examples/hello/hello.cpp"
```

The `CALIPER_APPLET(HelloApplet, ...)` macro at the bottom is the whole ABI
boundary of the dylib. It generates the descriptor the loader looks for
(`caliper_applet_descriptor`), the five exception-safe C bridge functions, and
the `ui::connect()` call that shares the host's ImGui/ImPlot contexts and
allocators with your dylib — so `ImGui::` and `ImPlot::` calls land in the host's
single UI world. Field order is fixed: `id`, `version`, `name`, `summary`,
`tag`, `services`.

## Applet Lifecycle

Your class overrides three methods from `caliper::Applet`:

- **`on_init(Host& host)`** runs once when the applet is opened. The `Host&` is
  valid for your whole lifetime — keep the pointer. Do setup here, and log
  through the host (`host.log_info(...)`), never `printf` (see
  [howto/debug-an-applet.md](applet-debugging.md) for why). Return
  `false` to abort loading.

- **`on_frame(const Frame& f)`** runs every frame. Everything visible happens
  here, and nothing slow — you share the frame thread with the host and every
  other applet. `f.fb_width`/`f.fb_height` are **physical** pixels;
  `f.dpi_scale` converts to the logical units ImGui sizes in; drive animation
  from `f.time_sec`/`f.delta_sec`, never a wall-clock sleep (§3a shows why a
  *pausable* animation accumulates its own phase from `f.delta_sec` rather than
  reading `f.time_sec` directly).

- **`on_cleanup()`** runs when the applet closes — symmetric with `on_init`:
  persist, release, log. After it returns the host destroys your object; do not
  touch host services afterwards.

Hello also reads `CALIPER_HELLO_CRASH` in `on_init` and, when set, faults inside
`on_frame` *before* any ImGui call. That is a deliberate test hook the loader's
crash-quarantine tests use — not something your own applets need.

## Input Buttons

The button above the plot is the smallest complete lesson in ImGui **IO**, and
it turns on three ideas you will use in every applet:

- **A widget call both draws and reports.** `ImGui::Button(...)` draws the
  button *and* returns `true` on the single frame it was clicked. There is no
  callback and no event queue — you check the return value inline:
  ```cpp
  if (ImGui::Button(playing_ ? "Pause" : "Play")) playing_ = !playing_;
  ```
  This is *immediate mode*: the UI is a function of your state, re-issued every
  frame, and input comes back as the return value of the call that drew it.

- **The label is derived from state, every frame.** There is one button, not a
  Play button and a Pause button — its *label* is an expression over
  `playing_`, recomputed on every `on_frame`. Because the whole UI is rebuilt
  each frame from your state, a widget that reflects state costs nothing extra:
  you just compute what to show. The click flips the same `bool` the label
  reads, so the button relabels itself the very next frame.

- **The state lives in your applet, not in ImGui.** ImGui does not remember
  "paused" for you — `playing_` is a member of `HelloApplet`. The widget reads
  and writes your field; ImGui only owns pixels and the click. That is why the
  state is a `bool` on the class, initialised in the header, not a `static`
  inside `on_frame`.

The subtle part is *why the animation can be paused at all*. Earlier the sine
was drawn from `f.time_sec` — the host's monotonic wall-clock, which keeps
advancing no matter what the applet does, so there is nothing you could freeze.
Pause only becomes possible once the applet **owns the time**: Hello accumulates
its own `phase_`, advanced by `f.delta_sec` **only while playing**:

```cpp
if (playing_) phase_ += (float)f.delta_sec;   // ...then draw sin(x + phase_)
```

That is the general shape of interactive animation in a Caliper applet — derive
what you draw from state you control, and let the widgets edit that state. The
same three ideas scale straight up to sliders (`SliderFloat` returns an edited
value), checkboxes, and the live training controls the ML applets use.

## Build

An applet builds against the installed SDK, in a folder of its own outside the
Caliper repository. First build Caliper and install the SDK from your checkout:

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

When the folder is built on its own, as here, the `if` block declares the
project and finds the SDK. Inside the Caliper repository the same folder is also
part of the core build, as the fixture the loader tests load, and the core build
provides `caliper_add_applet()` itself; an applet of your own can drop the `if`
and keep its three lines. `caliper_add_applet()` checks the manifest when you
configure, builds the library against `caliper::sdk` and `caliper::ui_stack`,
and writes a copy of the manifest beside it with a `[build]` table that records
the SDK version and the UI-stack hash. [SDK package](sdk-package.md)
describes the function and the package.

This tutorial copies hello so that the walk above matches the code you build.
To start an applet of your own, `caliper new <name>` writes a project of the
same shape, as [Development basics](../development/development-basics.md#the-development-loop)
shows.

## Applet Loading

Check the build first. `caliper check` runs the checks the host makes before it
opens a library, without opening a window:

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

On Windows the executable is `cmake-build-release\caliper.exe`. `caliper dev`
puts the folder ahead of every other applet folder. The release preset also
builds its own copy of Hello beside the executable, because the loader tests
load it; at equal versions the host loads the copy found first, which is yours,
and lists the other one as `SHADOWED`. The host keeps a library loaded until it
quits, so after each rebuild quit Caliper and run `caliper dev` again.

To keep the applet installed, `"$CALIPER" install build/applets` runs the same
checks and copies the library and its manifest into the user applets folder,
which the host searches every time it starts. [Caliper CLI](../reference/cli.md)
describes every subcommand.

## App Visibility

Hello appears as a row in the landing screen's **Instruments** pane; select it
and press Enter (or double-click the row) to open it. The host's epoch-2 loader
discovers it by reading its `<stem>.caliper.toml` manifest and the
`caliper_applet_descriptor` export — the two signals every applet you build this
way must ship. If the row is missing, run `caliper check` on the folder. If its
status is `REFUSED`, `FAILED` or `QUARANTINED` instead of `READY`, the **Inspector** shows the
loader's reason under Status (see the
[refusal reference](../reference/refusal-messages.md)). Esc, or the `esc Home` button,
returns to the landing screen. [Development basics](../development/development-basics.md#the-landing-screen)
describes the panes and keys.
