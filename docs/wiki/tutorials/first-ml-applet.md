# Your first ML applet

The ladder from an empty window to a live machine-learning demonstration,
one capability per stage. We build **SineScope**: a tiny MLP that learns
`y = sin(x)` in front of you — live loss curve, the model's prediction
bending toward the target in real time, and its weight matrix rendered as a
heatmap. Synthetic data keeps it self-contained (no downloads); the last
two stages show where real datasets and the remaining services attach.

**SineScope is built in this repo** — `examples/sine_scope/`, the SineScope
card in your launcher. Every code block below is embedded **verbatim from
that source file**; the docs build fails if they drift. Run the finished
thing first if you like:

```bash
cmake --build build --target sine_scope
CALIPER_AUTOLAUNCH=dev.example.sine-scope ./build/caliper
```

Prereqs: [Development basics](development-basics.md) (the mental model),
[Your first applet](first-applet.md) (the hello walkthrough). The finished
staircase — same patterns, full scale — is `applets/embed_scope/`, with the
[cookbook](../howto/ml-applet-cookbook.md) as its field guide.

## Stage 0 — the build and the manifest

The manifest requires what training needs and marks visualization optional,
so the applet still runs without the bridge:

```toml title="examples/sine_scope/sine_scope.caliper.toml"
--8<-- "examples/sine_scope/sine_scope.caliper.toml"
```

The CMake file is hello's plus the torch lines — this is the *entire* ML
build delta:

```cmake title="examples/sine_scope/CMakeLists.txt"
--8<-- "examples/sine_scope/CMakeLists.txt"
```

## Stage 1 — the model and the state spine

Two layers are enough to bend a line into a sine:

```cpp
--8<-- "examples/sine_scope/sine_scope.cpp:model"
```

Every live applet shares the same skeleton state — service wrappers, one
mutex, published copies, a generation counter
([cookbook §1](../howto/ml-applet-cookbook.md#1-the-threading-spine)):

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
curve bending toward the target is the "it's alive" moment — and it's just
two `PlotLine`s, plus the follow-toggle idiom on the loss curve
([cookbook §6](../howto/ml-applet-cookbook.md#6-viewport-policy-who-owns-the-camera)):

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

(The tutorial stages the weights to CPU for simplicity; the exemplar's
Tensors panel shows the full **zero-copy device pull** — weights that never
leave the GPU — in [cookbook §3](../howto/ml-applet-cookbook.md#3-the-device-resident-pull-the-usp-pattern).)

Cleanup grows its symmetric duties — cancel, bounded wait, release:

```cpp
--8<-- "examples/sine_scope/sine_scope.cpp:cleanup"
```

**Run it** (`CALIPER_AUTOLAUNCH=dev.example.sine-scope ./build/caliper`):
you should see the flat line snap into a sine within seconds while the
heatmap's blocks reorganize.

## Stage 5 — real data instead of synthetic

Everything above holds; only acquisition changes. The rules
([cookbook §8](../howto/ml-applet-cookbook.md#8-data-acquisition-the-download-recipe)):
fetch **inside the job**, cache in `host.data_dir()`, write atomically
(`.tmp` + rename), self-heal corrupt caches, make the transfer cancellable
via curl's progress callback, and add `CURL::libcurl`/`ZLIB::ZLIB` to the
CMake links. The exemplar's `ensure_dataset` + `mnist_path` are the
copy-paste source — including the sibling-cache trick (reuse another
applet's MNIST download rather than duplicating 11 MB).

## Stage 6 — the rest of the platform, one line each

Each remaining service is a small delta from here, and the exemplar shows
all of them finished:

- **`metrics.v1`** — persistence + the Runs dashboard for two lines:
  `run = metrics.begin_run("sine", "mlp32")` once, then
  `metrics.scalar(run, "train/loss", step, loss)` in the loop. Your run now
  survives restarts and plots in the host's Runs window.
- **`artifacts.v1`** — Save/Load buttons so a trained model outlives the
  process ([cookbook §9](../howto/ml-applet-cookbook.md#9-checkpoints-via-artifactsv1)).
  Load-then-eval *without retraining* is the demo magic.
- **`data.v1`** — when your published state is genuinely tabular, register
  it and ask SQL questions
  ([cookbook §10](../howto/ml-applet-cookbook.md#10-sql-over-live-data-datav1)).

When all of these feel natural, read `applets/embed_scope/` end to end —
it is exactly this tutorial's patterns at full scale: a real dataset, a 3-D
learned embedding, per-step device pulls, and all eight services in ~900
annotated lines.
