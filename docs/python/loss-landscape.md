---
kind: procedure
---

# Loss Landscape

The notebook `demos/loss_landscape/loss_landscape.ipynb` trains the small transformer of [Notebook Training Runs](notebook-training-runs.md) and streams, while it trains, the loss surface around the model's current weights as a 3-D scene.

## Requirements

The setup of [Notebook Training Runs](notebook-training-runs.md#wheel-installation) installs Caliper LIB with the extra `caliper[notebook]`, PyTorch and JupyterLab. The model trains on the GPU through MPS on an Apple Silicon Mac, and on the CPU, more slowly, elsewhere. The notebook downloads nothing.

## Steps

### Notebook Start

From the repository root:

```sh
.venv/bin/jupyter lab demos/loss_landscape/loss_landscape.ipynb
```

Run the cells in order. The first cell starts `caliper.serve()` and prints the
link of the viewer page, which shows the same scene and controls in a browser
tab. `CALIPER_NOTEBOOK_STEPS` sets the number of training steps, 8,000 by
default. The notebook repeats the model, data and training loop of
[Notebook Training Runs](notebook-training-runs.md), so the run is the same run, and `landscape.py` beside it
computes the landscape.

### Controls

`caliper.controls()` above the scene widget shows the run's Pause, Resume and
Step, three parameters and one action:

| Control | Range | Does |
|---|---|---|
| `every` | 10 to 5,000 steps, 2,000 by default | The steps between refits |
| `span` | 0.05 to 4, on a log scale, 1 by default | What the box's half-width covers, read per mode as above |
| `directions` | `path` or `random` | The plane |
| `surface now` | An action | Refits at once, also while the run is paused |

A new `span` or `directions` takes effect at the next refit. The notebook's
last cells switch to random directions from Python and train a little longer.

## Result

The scene shows the loss of a fixed batch on a 41 by 41 grid over a plane through the current weights, the optimizer's path in that plane as a line, the current point as a marker, and a box with a floor grid. Every 2,000 steps the plane and the surface are computed again on the GPU, and every 10th step the path gets a new point. By default the plane is the one in which the weights have moved most so far, so the picture shows how the optimizer moves across the surface, and a control switches it to two random directions. The viewer page shows the same scene and controls in a browser tab.

### Scene Contents

The scene `landscape` is logged with `caliper.scene("landscape", up="+z", units="")`
and the scene calls of [Scene Calls](scene-calls.md):

| Entity | Call | Sent | Holds |
|---|---|---|---|
| the root, `""` | `scene.lines(..., static=True)` | Once | The 12 edges of the box and 9 floor lines each way, as segments |
| `surface` | `scene.mesh` | At each refit | 1,681 vertices and 3,200 triangles, coloured by log loss through `viridis` from -4 to 2 |
| `path` | `scene.lines` with `strip_lengths` | When the path has a new point, every 10th step, at most 10 times a second, and at once after each refit | The optimizer's path as one strip |
| `current` | `scene.points` | With the path | The current point, a red marker, which is the path's last point; between points it stays on it |

The series `plane share` receives one value at the end of every training
step, the share of the last refit, so its index is the training step: the
curve is flat between refits and steps where a refit changes it. Before the
first plane, at step 40, and in random directions, which have no plane, the
values are NaN, which the series view draws as a gap. After its loop,
`train()` calls `caliper.flush()`, so that the values of the last steps reach
the viewers at once rather than with the next call.

Height and colour are both the log of the loss. A vertex stands at half the
log10 of the loss of the landscape batch, the first 128 of the notebook's 512
evaluation sequences, clipped to the decades from 0.0001 to 100, so the
box's floor is a loss of 0.0001 and its top a loss of 100; the scene view has
no axis labels. The box spans -1 to 1 across and -2 to 1 in height in both
modes, so the camera never has to move. The surface is the loss on the plane
through the current weights: its centre vertex is the current point at each
refit. Each point of the path stands at its own loss at its coordinates on
that plane, and the plane holds about 95 % of the path's variance on this
model, so the path runs close to the surface and not exactly on it. A `span`
below 1 zooms in on the surface around the current weights; the earlier part
of the path then lies outside the box.

### Plane Modes

The landscape moves the 102,098 entries of the weight matrices and of the
token and position embeddings. The biases and the LayerNorm gains and
offsets, about 1,800 parameters, keep their current values.

| | The path's plane, the default | Random directions |
|---|---|---|
| The directions | The top two principal components of up to 256 snapshots of the weights, taken every 20 steps around the current weights, matched in order and sign to the previous refit's | Two directions drawn once with seed 0, each row of each weight matrix and embedding scaled to the norm of the same row of the weights at step 0 |
| The first refit | At step 40, when three snapshots exist | At step 0 |
| The box's half-width | 1.5 times `span` times the farthest snapshot's distance from the centre on either axis | `span` units of the directions |
| The path | One point per kept snapshot, then one point every 10th step until the next refit, where it is drawn again in the new plane | One point every 10th step, never drawn again; it barely moves and reads as an almost vertical line at the centre |
| `plane share` | The share of the kept snapshots' variance that the plane of the last refit holds | NaN at every step, a gap |

## Landscape Class

`Landscape(model, x, y, *, grid=41, seed=0, point_every=10, snapshot_every=20, keep=256, loss=F.cross_entropy)`
in `landscape.py` holds everything on the model's device and reads nothing
back to the CPU:

| Member | Does |
|---|---|
| `point(model, step)` | After every training step. At every `point_every`-th step: the path's point from one forward pass of the batch without gradients, a snapshot every `snapshot_every` steps, a multiple of `point_every` (every second one is dropped and the interval doubles when `keep` are held), and the point appended to the path once a plane exists. At the other steps it returns `None` |
| `due(step, every, directions)` | Whether to refit at this step |
| `refit(model, span, directions)` | The plane, then the surface: returns the mesh's `positions` (1681, 3) and `values` (1681,) and the plane share, or `None` in random directions |
| `can_fit(directions)` | Whether a refit can run now |
| `frame`, `triangles` | The box and floor grid as segments, and the mesh's triangles |
| `path`, `path_length`, `current` | The path in box units, its length, and its last point |
| `method` | `"vmap"` or `"loop"`: how the grid is evaluated, decided at the first surface |

The grid is evaluated with `torch.func.functional_call` under `vmap`, one row
of 41 points in one batched forward pass, under `torch.no_grad()`. Where an
operation of the model has no batching rule on the device, the first surface
says so and the grid is evaluated with one forward pass per point instead.

## Cost

At the defaults the landscape makes the notebook's 8,000 steps about 1.2
times as long as training without it. `every` is the control that trades the
surface's freshness for speed: each surface costs about 0.63 s of GPU time, so
lowering `every` while the model trains shows the surface more often and
makes the run longer.

`python/bench/bench_landscape.py --steps 8000 --repeats 3` measures this on
MPS. It runs the notebook's own cells in fresh processes, with a viewer
attached in another process, with and without the landscape, interleaved, and
stops each run's clock after `torch.mps.synchronize()`, so that every kernel
the loop queued is counted. On an Apple M5 Max with torch 2.15 and the Python
sender, at a 1-minute load average of 4.5 to 6.1 for the first pair and up to
11.7 for the others:

| Row | Measured |
|---|---|
| 8,000 steps with the landscape, three runs, with five surfaces because the 20 warm-up steps put step 8,000 inside the timed run | 18.84 to 19.32 s |
| 8,000 steps without it, its work committed to the device after every step, three runs | 14.18 to 16.28 s |
| The ratio, median and range | 1.20, from 1.16 to 1.36 |
| One surface, synchronised, under `vmap` | 0.62 to 0.64 s, in either mode |
| One surface with one forward pass per point | 1.18 to 1.23 s |
| A surface's peak memory above what training holds | 575 MB |
| The buffer of 256 snapshots | 105 MB, allocated at the first snapshot |
| `scene.lines` and `scene.points` of the path and marker, per call | 42 to 45 µs median, 0.4 to 0.5 ms p99 |

The baseline matters. On MPS, a training loop that never waits for the device
and records no event leaves its work queued until it ends: the notebook's
loop without the landscape spends 14.2 s on the host for 8,000 steps, and the
device then needs 14.4 s more, one after the other. The landscape's calls
record events, so with it the work reaches the device as the loop runs, and a
wall-clock comparison with the plain loop would measure that drain rather
than the landscape. The baseline is therefore the same loop without the
landscape with an MPS event recorded after every step, which hands its work to
the device as it runs; `--diagnose` adds the plain loop and the device's own
times to the report.

## Tests

```sh
PYTHONPATH=python .venv/bin/python -m pytest tests/demos
```

`tests/demos/test_loss_landscape.py` runs the notebook's code cells on the CPU
with fixed seeds and deterministic algorithms, through a real loopback
connection to a test viewer that decodes every scene batch, in under a
minute: the path's plane for 300 steps, random directions for 100 steps, the
controls, and two fixtures of `Landscape` alone for the order and sign of the
matched directions and the thinning of the snapshots.

Its capture row, `test_capture`, runs where `CALIPER_TEST_CALIPER_EXE` names
the Caliper app and Metal draws it, and skips elsewhere. It runs the same 300
steps, forces a surface at step 299, attaches the hidden app
(`caliper --hidden --url`) from the start so that its camera fits the box,
and again when the scene view's area narrows as the view of the plane share
joins it, takes the window's frame from Caliper API once two frames agree, finds the
marker at the current point's pixel, and compares the scene view's area with
the reference `tests/demos/reference/loss_landscape_app_step299.png` under
the parity rules of the scene rows: a flat colour exact away from edges, a
lit pixel of the surface within 2 levels, and a pixel within one pixel of an
edge either side's colour. The scale bar in the area's bottom-right corner,
an overlay with tests of its own, is left out of the comparison; the app
reports where it drew it. Without the reference it writes the capture for a
person to review and skips; `CALIPER_DEMO0_WRITE_REFERENCE=1` writes it as
the reference. `CALIPER_SCENE_CAPTURE_DIR` keeps the frame and the area.

Its browser row, `test_capture_browser`, draws the same scene on the viewer
page of `caliper.serve()` with Caliper's WebAssembly views, in headless
Chromium through Playwright (`tests/demos/viewer_page.mjs`), on the SwiftShader
adapter, which needs no GPU, or on the Mac's GPU with
`CALIPER_DEMO0_WEB_ADAPTER=metal`. It runs where the WebAssembly views are
built into `python/caliper/_web/` (`scripts/build_wasm.py`), Node is installed
and `npm ci` has been run in `web/wasm/test` with Playwright's Chromium headless
shell in its cache, and skips elsewhere; it downloads nothing. CI's Browser
views workflow has no PyTorch, so the row runs on a developer's Mac. It opens
the page before training, runs the same 300 steps and the surface at step 299,
waits until the scene's module has received every message the server sent on
its connection and has drawn 30 more frames, checks that the path's, the
marker's and the frame's colours appear exactly, and compares the scene's
drawing area with `tests/demos/reference/loss_landscape_web_swiftshader_step299.png`
under the same parity rules, leaving the scale bar out as the app's row does;
the module publishes where it drew it. Without the reference it writes the capture and
skips; `CALIPER_DEMO0_WRITE_WEB_REFERENCE=1` writes it as the reference.
`CALIPER_SCENE_CAPTURE_DIR` keeps the area and the page's report of where the
module drew it, with the scale bar's band.
