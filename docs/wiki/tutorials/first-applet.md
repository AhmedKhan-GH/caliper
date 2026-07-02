# Your first applet

This walks the **hello** applet — the smallest complete Caliper applet on ABI
epoch 2. It is the canonical starting point: one manifest, one macro, three
lifecycle methods. Once it makes sense, open `examples/signal_scope/` in the
repo — the exemplar with every idiom (probing optional services, persisting to
the data dir, the watchdog anti-pattern) — and copy from there. For ML work, the
**ML exemplar** `examples/ml_scope/` shows the idioms that matter on a GPU:
training off the frame thread via [`caliper.jobs.v1`](../reference/services/jobs-v1.md),
the host-negotiated device via [`caliper.device.v1`](../reference/services/device-v1.md),
and a live loss curve — the pattern to copy for any applet that computes. It is
also the exemplar for streaming training metrics to
[`caliper.metrics.v1`](../reference/services/metrics-v1.md) (probed optionally, so
the same binary runs with or without it) — every applet that logs a scalar that
way inherits the Runs dashboard. MLScope is also the exemplar for GPU-resident
visualization — its live conv-kernel grid crosses
[`caliper.tensor_bridge.v1`](../reference/services/tensor-bridge-v1.md) via the
[torch adapter](../reference/adapters.md), zero-copy on the Metal renderer.

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
  from `f.time_sec`/`f.delta_sec`, never a wall-clock sleep.

- **`on_cleanup()`** runs when the applet closes — symmetric with `on_init`:
  persist, release, log. After it returns the host destroys your object; do not
  touch host services afterwards.

Hello also reads `CALIPER_HELLO_CRASH` in `on_init` and, when set, faults inside
`on_frame` *before* any ImGui call. That is a deliberate test hook the loader's
crash-quarantine tests use — not something your own applets need.

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
