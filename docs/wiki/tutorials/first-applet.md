# Your first applet

*(New here? [Development basics](development-basics.md) explains what you write vs. what the host provides, and where every library comes from.)*

This walks the **hello** applet — the smallest complete Caliper applet on ABI
epoch 2. It is the canonical starting point: one manifest, one macro, three
lifecycle methods. Once it makes sense, open `applets/embed_scope/` — **the
exemplar** — and copy from there. EmbedScope demonstrates every idiom the
platform has: probing optional services (the same binary runs with or without
each), training off the frame thread via
[`caliper.jobs.v1`](../reference/services/jobs-v1.md), the host-negotiated
device via [`caliper.device.v1`](../reference/services/device-v1.md), streaming
scalars to [`caliper.metrics.v1`](../reference/services/metrics-v1.md) (which
inherits the Runs dashboard for free), GPU-resident visualization across
[`caliper.tensor_bridge.v1`](../reference/services/tensor-bridge-v1.md) via the
[torch adapter](../reference/adapters.md), checkpoints through
[`caliper.artifacts.v1`](../reference/services/artifacts-v1.md), and live SQL
over [`caliper.data.v1`](../reference/services/data-v1.md) — all eight
services in one applet, with an ImPlot3D embedding cloud as the centerpiece.

Earlier exemplars that drove the platform's architecture (SignalScope,
MLScope, GPTScope) are archived under `applets/legacy-dev/` — not built or
loaded, kept for history and code reference.

Of the archived exemplars, **GPTScope** (`applets/legacy-dev/gpt_scope/`) is
worth a read even unbuilt — a char-level mini-GPT trained live on
TinyShakespeare, built entirely on public services, which served as the
Phase-2 flagship proof.

For the live applet that exercises **every** service, see **EmbedScope**
(`applets/embed_scope/`) — a small MNIST net with a learned 3-D embedding
bottleneck, drawn as a live ImPlot3D scatter that splits one blob into ten
colored lobes as it trains. It is the reference consumer of
[`caliper.artifacts.v1`](../reference/services/artifacts-v1.md) (Save/Load a
checkpoint — Load skips training) and
[`caliper.data.v1`](../reference/services/data-v1.md) (SQL over the live embedding
table for class centroids and misclassified counts), on top of the same jobs /
device / metrics / bridge stack. Its 3-D acceptance checks join the same
[demo checklist](../reference/services/tensor-bridge-v1.md#demo-checklist-human).

The whole applet lives in `examples/hello/`:

```
examples/hello/
├── hello.caliper.toml   # the manifest — identity + what the host must provide
├── hello.cpp            # the applet — one class + one macro
└── CMakeLists.txt       # the build — links caliper::sdk + caliper::ui_stack
```

## 1. The manifest

The host reads `hello.caliper.toml` **before it loads any of your code**. It
declares who the applet is, which ABI epoch it was compiled against, and which
platform services it needs to run:

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
[reference/manifest.md](../reference/manifest.md) for the full schema.

## 2. The macro

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

## 3. The lifecycle: `on_init` → `on_frame` → `on_cleanup`

Your class overrides three methods from `caliper::Applet`:

- **`on_init(Host& host)`** runs once when the applet is opened. The `Host&` is
  valid for your whole lifetime — keep the pointer. Do setup here, and log
  through the host (`host.log_info(...)`), never `printf` (see
  [howto/debug-an-applet.md](../howto/debug-an-applet.md) for why). Return
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

## 3a. Input: the Play/Pause button

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

## 4. Build it

The `CMakeLists.txt` links the two SDK targets and drops the dylib plus a copy
of the manifest into `build/applets/`, where the host scans for applets:

```bash
cmake -B build
cmake --build build --target hello_applet
ls build/applets/            # libhello.dylib + hello.caliper.toml
```

In-tree the build links `caliper::sdk` and `caliper::ui_stack` directly; an
out-of-tree applet swaps those two lines for a `find_package`/CPM fetch of a
tagged SDK release under the *same* target names — nothing else changes.

## 5. See it in the app

Hello appears on the launcher landing page as a card you can open. The host's
epoch-2 loader discovers it by reading its `<stem>.caliper.toml` manifest and the
`caliper_applet_descriptor` export — the two signals every applet you build this
way must ship. If the card is missing, check that both are present (the dylib and
its manifest sit side by side in `build/applets/`); if the card shows an
`[unavailable]` line instead, that line is the loader's refusal reason (see the
[refusal reference](../reference/refusals.md)).
