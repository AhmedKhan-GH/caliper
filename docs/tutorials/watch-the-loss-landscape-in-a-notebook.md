# Watch the loss landscape in a notebook

The notebook `demos/loss_landscape/loss_landscape.ipynb` trains the small
transformer of [Watch a training run in a notebook](watch-a-training-run-in-a-notebook.md)
and streams, while it trains, the loss surface around the model's current
weights as a 3-D scene: the loss of a fixed batch on a 41 by 41 grid over a
plane through the current weights, the optimizer's path in that plane as a
line, the current point as a marker, and a box with a floor grid. Every 500
steps the plane and the surface are computed again on the GPU. By default the
plane is the one in which the weights have moved most so far, so the picture
shows how the optimizer moves across the surface, and a control switches it to
two random directions. The notebook downloads nothing.

Prereqs: the setup of [Watch a training run in a notebook](watch-a-training-run-in-a-notebook.md#install),
which installs Caliper LIB with the extra `caliper[notebook]`, PyTorch and
JupyterLab. The model trains on the GPU through MPS on an Apple Silicon Mac,
and on the CPU, more slowly, elsewhere.

## Run it

From the repository root:

```sh
.venv/bin/jupyter lab demos/loss_landscape/loss_landscape.ipynb
```

Run the cells in order. The first cell starts `caliper.serve()` and prints the
link of the viewer page, which shows the same scene and controls in a browser
tab. `CALIPER_NOTEBOOK_STEPS` sets the number of training steps, 8,000 by
default. The notebook repeats the tutorial's model, data and training loop,
so the run is the same run as the tutorial's, and `landscape.py` beside it
computes the landscape.

## What the scene holds

The scene `landscape` is logged with `caliper.scene("landscape", up="+z", units="")`
and the scene calls of [Caliper LIB](../references/python-api.md#scenes):

| Entity | Call | Sent | Holds |
|---|---|---|---|
| the root, `""` | `scene.lines(..., static=True)` | Once | The 12 edges of the box and 9 floor lines each way, as segments |
| `surface` | `scene.mesh` | At each refit | 1,681 vertices and 3,200 triangles, coloured by log loss through `viridis` from -4 to 2 |
| `path` | `scene.lines` with `strip_lengths` | Every step, at most 10 times a second, and at once after each refit | The optimizer's path as one strip |
| `current` | `scene.points` | With the path | The current point, a red marker, which is the path's last point |

The series `plane share` receives one value at each refit in the path's plane.

Height and colour are both the log of the loss. A vertex stands at half the
log10 of the loss of the landscape batch, the first 128 of the notebook's 512
evaluation sequences, clipped to the decades from 0.0001 to 100. The box spans
-1 to 1 across and -2 to 1 in height in both modes, so the camera never has
to move. The surface is the slice through the current weights: its centre
vertex is the current point at each refit, and earlier points of the path
need not lie on it.

## The two modes

The landscape moves the 102,098 entries of the weight matrices and of the
token and position embeddings. The biases and the LayerNorm gains and
offsets, about 1,800 parameters, keep their current values.

| | The path's plane, the default | Random directions |
|---|---|---|
| The directions | The top two principal components of up to 256 snapshots of the weights, taken every 25 steps around the current weights, matched in order and sign to the previous refit's | Two directions drawn once with seed 0, each row of each weight matrix and embedding scaled to the norm of the same row of the weights at step 0 |
| The first refit | At step 50, when three snapshots exist | At step 0 |
| The box's half-width | 1.5 times `span` times the farthest snapshot's distance from the centre on either axis | `span` units of the directions |
| The path | One point per kept snapshot, then one point per step until the next refit, where it is drawn again in the new plane | One point per step, never drawn again; it barely moves and reads as an almost vertical line at the centre |
| `plane share` | The share of the kept snapshots' variance that the plane holds | Not sent |

## Controls

`caliper.controls()` above the scene widget shows the run's Pause, Resume and
Step, three parameters and one action:

| Control | Range | Does |
|---|---|---|
| `every` | 10 to 5,000 steps, 500 by default | The steps between refits |
| `span` | 0.05 to 4, on a log scale, 1 by default | What the box's half-width covers, read per mode as above |
| `directions` | `path` or `random` | The plane |
| `surface now` | An action | Refits at once, also while the run is paused |

A new `span` or `directions` takes effect at the next refit. The notebook's
last cells switch to random directions from Python and train a little longer.

## Landscape

`Landscape(model, x, y, *, grid=41, seed=0, snapshot_every=25, keep=256, loss=F.cross_entropy)`
in `landscape.py` holds everything on the model's device and reads nothing
back to the CPU:

| Member | Does |
|---|---|
| `point(model, step)` | After a training step: the path's point from one forward pass of the batch without gradients, a snapshot every `snapshot_every` steps (every second one is dropped and the interval doubles when `keep` are held), and the point appended to the path once a plane exists |
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

`python/bench/bench_landscape.py --steps 8000 --repeats 3 --diagnose`
measures the cost on MPS. It runs the notebook's own cells in fresh
processes, with a viewer attached in another process, with and without the
landscape, interleaved. Each run's clock stops after `torch.mps.synchronize()`,
so it includes every kernel the loop queued. On an Apple M5 Max with torch
2.15, at a 1-minute load average of 1.4 to 2.9, with `every` at 500 and the
Python sender:

| Row | Measured |
|---|---|
| 8,000 steps with the landscape, 17 surfaces, three runs | 28.02 to 29.19 s |
| 8,000 steps without it, as the notebook's loop is written, three runs | 28.52 to 28.76 s |
| 8,000 steps without it, its work committed to the device after every step | 14.38 s |
| One surface, synchronised, under `vmap` | 0.61 to 0.66 s, in either mode |
| One surface with one forward pass per point | 1.18 to 1.20 s |
| A surface's peak memory above what training holds | 575 MB |
| The buffer of 256 snapshots | 105 MB, allocated at the first snapshot |
| The device's time a step: the training step, then with `land.point()` | 1.77 ms, then 2.17 ms |
| `scene.lines` and `scene.points` of the path and marker, per step | 17 to 18 µs median, 0.2 ms p99 |

Without the landscape, nothing in the loop waits for the device or records an
event until the loop ends, and the device does almost none of the queued work
until then: the host finishes its 8,000 steps in 14.2 s and the device needs
14.4 s more, one after the other. The landscape's calls record events, so with
it the work reaches the device as the loop runs, and host and device work at
the same time. The same loop without the landscape, with an event recorded
after every step, takes 14.38 s. Against that run the landscape makes training
1.95 to 2.03 times as long. The 17 surfaces account for 10.7 s of that time,
and the extra forward pass of `land.point()` for about 3.2 s. A surface call
also holds the host for about 0.28 s while it queues the 41 rows, and the path
and marker calls right after a refit take about 45 ms.

## Tests

```sh
PYTHONPATH=python .venv/bin/python -m pytest tests/demos
```

`tests/demos/test_loss_landscape.py` runs the notebook's code cells on the CPU
with fixed seeds, through a real loopback connection to a test viewer that
decodes every scene batch, in about a minute: the path's plane for 300 steps,
random directions for 100 steps, the controls, and two fixtures of
`Landscape` alone for the order and sign of the matched directions and the
thinning of the snapshots. `python/bench/bench_landscape.py` measures the cost
on MPS.
