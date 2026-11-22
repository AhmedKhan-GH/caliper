# First ML Applet

The ladder from an empty window to a live machine-learning demonstration,
one capability per stage. We build **SineScope**: a tiny MLP that learns
`y = sin(x)` from noisy samples in front of you — live loss curve, the model's
curve settling through the scattered points in real time, and its weight matrix rendered as a
heatmap. Synthetic data keeps it self-contained (no downloads); the last
two stages show where real datasets and the remaining services attach.

**SineScope lives in this repo** — `examples/sine_scope/`. Every code block
below is embedded **verbatim from that source file**; the docs build fails if
they drift. The release preset builds a copy beside the executable, because the
embedding tests load it, so you can run the finished thing first if you like:

```bash
cmake --preset release
cmake --build --preset release
CALIPER_AUTOLAUNCH=dev.example.sine-scope ./cmake-build-release/caliper
```

Prereqs: [Development basics](../development/development-basics.md) (the mental model),
[Your first applet](first-applet.md) (the hello walkthrough).

## Stage 0 — the build and the manifest

The manifest requires what training needs and marks visualization optional,
so the applet still runs without the bridge:

```toml title="examples/sine_scope/sine_scope.caliper.toml"
--8<-- "examples/sine_scope/sine_scope.caliper.toml"
```

The CMake file is hello's plus one word, `TORCH` — this is the *entire* ML
build delta. `TORCH` links the libtorch the host was built with, which the SDK
names, and stamps its version into the built manifest, so the host refuses the
applet if it ever runs beside another libtorch:

```cmake title="examples/sine_scope/CMakeLists.txt"
--8<-- "examples/sine_scope/CMakeLists.txt"
```

### Build and load your copy

Build your copy in a folder of its own against the installed SDK, as
[Your first applet](first-applet.md#4-build-it) does for hello, with `CALIPER`
set to the host you built. The SDK must come from a core build with libtorch,
which the release preset is:

```bash
mkdir -p "$HOME/caliper-applets"
cp -R examples/sine_scope "$HOME/caliper-applets/sine_scope"
cd "$HOME/caliper-applets/sine_scope"
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/caliper-sdk" -DCMAKE_BUILD_TYPE=Release
cmake --build build
"$CALIPER" check build/applets
CALIPER_AUTOLAUNCH=dev.example.sine-scope "$CALIPER" dev build/applets
```

`caliper dev` searches your build folder first, so at equal versions the host
loads your copy rather than the one the release preset built beside the
executable. The host keeps a library loaded until it quits, so after each
rebuild quit Caliper and run `caliper dev` again.

## Stage 1 — the model and the state spine

Two layers are enough to bend a line into a sine:

```cpp
--8<-- "examples/sine_scope/sine_scope.cpp:model"
```

Every live applet shares the same skeleton state — service wrappers, one
mutex, published copies, a generation counter:

```cpp
--8<-- "examples/sine_scope/sine_scope.cpp:state"
```

`on_init` probes services. The manifest already guaranteed the required
ones exist — no null checks needed for jobs. The optional bridge is
**falsy-inert**: callable unconditionally, no-ops when absent — but good
demos *show* the degradation (Stage 4):

```cpp
--8<-- "examples/sine_scope/sine_scope.cpp:init"
```

## Stage 2 — background compute: the training job

Never compute on the frame thread. A job is a plain function the host runs
on a worker thread — note the per-step cancel check (the ≤100 ms contract)
and the progress reports that light up the host's jobs tray:

```cpp
--8<-- "examples/sine_scope/sine_scope.cpp:job"
```

The frame side submits on click, shows progress, offers cancel:

```cpp
--8<-- "examples/sine_scope/sine_scope.cpp:controls"
```

## Stage 3 — publish and plot: watching it learn

The worker publishes owned copies (plot data) and a tensor handle (the
weight display) under the mutex, bumping the generation:

```cpp
--8<-- "examples/sine_scope/sine_scope.cpp:publish"
```

The frame consumes when the generation moves, then draws. The prediction
curve settling through the noisy samples is the "it's alive" moment — and it's
a `PlotScatter` and a `PlotLine`, plus the follow-toggle idiom on the loss curve:

```cpp
--8<-- "examples/sine_scope/sine_scope.cpp:plots"
```

## Stage 4 — the bridge: a tensor as pixels

The weight matrix as a colormapped texture. Frame thread only, gen-gated,
released in cleanup — and when the bridge is absent, the panel says so
politely instead of failing:

```cpp
--8<-- "examples/sine_scope/sine_scope.cpp:bridge"
```

(The tutorial stages the weights to CPU for simplicity;
[`tensor_bridge.v1`](services/caliper-tensor-bridge-v1.md) also draws device
tensors without them leaving the GPU.)

Cleanup grows its symmetric duties — cancel, bounded wait, release:

```cpp
--8<-- "examples/sine_scope/sine_scope.cpp:cleanup"
```

**Run it** (`CALIPER_AUTOLAUNCH=dev.example.sine-scope "$CALIPER" dev build/applets`):
you should see the flat line snap into a sine within seconds while the
heatmap's blocks reorganize.

## Stage 5 — real data instead of synthetic

Everything above holds; only acquisition changes. The rules:
fetch **inside the job**, cache in `host.data_dir()`, write atomically
(`.tmp` + rename), self-heal corrupt caches, make the transfer cancellable
via curl's progress callback, and add `CURL::libcurl`/`ZLIB::ZLIB` to the
CMake links, found with `find_package(CURL)` and `find_package(ZLIB)`.

## Stage 6 — the rest of the framework, one line each

Each remaining service is a small delta from here:

- **`metrics.v1`** — persistence for two lines:
  `run = metrics.begin_run("sine", "mlp32")` once, then
  `metrics.scalar(run, "train/loss", step, loss)` in the loop. Your run now
  survives restarts in the host's metrics store. The host has no view of stored runs.
- **`artifacts.v1`** — Save/Load buttons so a trained model outlives the
  process.
  Load-then-eval *without retraining* is the demo magic.
- **`data.v1`** — when your published state is genuinely tabular, register
  it and ask SQL questions.
