# Watch a PyTorch training run live

This tutorial takes an ordinary PyTorch training loop and adds five kinds of
live view: a weight matrix as a heatmap, a convolution's activations as a
channel mosaic, a hidden layer's embedding as a point cloud coloured by class,
the loss as a curve and a weight histogram. The script keeps training at full
speed; Caliper samples it.

The worked example is `examples/python/train_show.py`: a tiny CNN learning to
tell which quadrant of a 28x28 image holds a Gaussian blob, on synthetic data
(no downloads). The excerpts below are shortened from it; run the file itself
for the full version.

Prereqs: Apple Silicon, macOS 14 or later, Python 3.12, PyTorch with MPS, and
CMake plus Ninja on `PATH`. [Live streaming](../explanation/live-streaming.md)
explains the design; this page only uses it.

## Build and install the wheel

The package is built from this tree; there is no published wheel and the sdist
is not usable. From the repo root, in a Python 3.12 virtual environment that
has `torch`, `numpy`, `scikit-build-core` and `nanobind` installed:

```sh
MACOSX_DEPLOYMENT_TARGET=14.0 uv build --wheel --no-build-isolation --python .venv/bin/python --out-dir dist
uv pip install --python .venv/bin/python --reinstall --no-deps dist/caliper-*.whl
```

With plain pip, `pip install dist/caliper-0.1.0-cp312-cp312-macosx_14_0_arm64.whl`
does the same. The first build compiles libcaliper and its dependencies (DuckDB
among them) from source and takes minutes; later builds reuse `build/<wheel tag>`.
The build is torch-free (see [the torch-free build](../reference/embedding.md#the-torch-free-build)),
so the extension shares the process with your own torch without a clash.

The wheel contains the `caliper` package, its two native modules and the
viewer process, installed as `caliper/bin/caliper-viewer` inside the package.
To print its path:

```sh
.venv/bin/python -c "from caliper import live; print(live.viewer_executable())"
```

Check the install with the example in its no-viewer mode, which trains for a
few seconds and prints throughput and per-call costs:

```sh
.venv/bin/python examples/python/train_show.py --mode none --seconds 5
```

## The calls in the training loop

The model and optimiser are plain PyTorch. Two forward hooks keep the latest
outputs of `conv1` and `fc1`, and a fixed random projection turns the 64-wide
hidden layer into 3-D points:

```python
model = TinyCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
acts = {}
model.conv1.register_forward_hook(lambda m, i, o: acts.__setitem__("conv1", o))
model.fc1.register_forward_hook(lambda m, i, o: acts.__setitem__("hidden", o))
proj = (torch.randn(64, 3, generator=torch.Generator().manual_seed(1)) / 8.0).to(device)
```

The training step is unchanged; the Caliper calls follow it:

```python
import caliper

for step in range(steps):
    x, y = make_batch(64, device)
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    caliper.show(model.fc1.weight, "fc1.weight", cmap="rdbu")     # (64, 784) heatmap
    caliper.show(acts["conv1"], "conv1.act")                       # (64, 8, 28, 28) -> mosaic
    caliper.points(acts["hidden"] @ proj, y, "embedding", classes=4)
    caliper.series(loss.detach(), "loss")
    caliper.histogram(model.fc1.weight, "fc1.weight.hist")
```

What each call does with its tensor:

| Call | Tensor | View |
|---|---|---|
| `show(model.fc1.weight, ...)` | `(64, 784)` float32 on MPS | Heatmap with the red-blue colormap, autoscaled on the GPU |
| `show(acts["conv1"], ...)` | `(64, 8, 28, 28)` | Item 0 of the batch, its 8 channels tiled as a 3x3 mosaic (84x84, the empty cell padded) |
| `points(z, y, ..., classes=4)` | `(64, 3)` positions, `(64,)` labels | 3-D scatter; `classes=4` colours labels 0..3 with a categorical palette |
| `series(loss.detach(), ...)` | 0-d tensor on MPS | One value appended to a ring of the last 4096 |
| `histogram(model.fc1.weight, ...)` | every element | 64 bins over [min, max], binned on the device |

None of these calls reads a value back to the CPU. `series` takes the loss
tensor itself, not `loss.item()`, so the loop never waits for the GPU; the value
is copied on the device into the series ring. Each call returns at once with
`True` if it took a snapshot and `False` if it was skipped.

Instead of registering hooks by hand, `caliper.watch(model)` shows every leaf
module's output as a view named `<module>.act`, and `gradients=True` adds each
weight's gradient as `<module>.<param>.grad`:

```python
handle = caliper.watch(model, gradients=True, include=["conv*", "fc1"])
...
handle.remove()          # takes the hooks off again
```

## Attach the viewer

With no other setup, the script's first call opens a **live session**: shared
memory plus a session file under
`~/Library/Application Support/Caliper/live/<pid>.json`. Until a viewer
attaches, every call returns after a check, with no snapshot and no GPU work.

Start the viewer from another terminal while the script runs:

```sh
"$(.venv/bin/python -c 'from caliper import live; print(live.viewer_executable())')"
```

It opens a "Live sessions" list, refreshed every second, with one row per
running script (script, PID, start time, views, state) and an **Attach** button.
Click it and the view windows appear. To have the script start the viewer
itself, open the session explicitly before training:

```python
sess = caliper.session(auto_launch=True)     # spawns caliper-viewer --session <token>
sess.wait_attached(15.0)                     # optional: wait for the first heartbeat
```

This is what `train_show.py --mode attach` does:

```sh
.venv/bin/python examples/python/train_show.py --mode attach --seconds 20
```

The viewer is a separate process: it stays responsive whatever the training
loop does, it keeps the last frames when the script exits, and closing it does
not stop training. [Attach to a running script](../howto/attach-to-a-running-script.md)
covers listing, detaching and what happens when either side goes away.

## The own window

`caliper.window()` opens a Caliper window inside the training process instead.
A render thread repaints it at 60 Hz whatever the loop does, but macOS delivers
mouse, keyboard, resize and the close button only to the main thread, which is
the training thread. During a long step the window repaints but does not
respond. `caliper.run(train)` fixes that by running the training function on a
worker thread and giving the main thread to the window:

```python
caliper.window(1280, 800, title="train_show")
result = caliper.run(train, model, opt)      # train(...) runs on a worker thread
caliper.spin()                               # keep the window open until it is closed
```

`run` returns `train`'s result and re-raises its exception; Ctrl-C reaches the
worker as `KeyboardInterrupt`. If you close the window while training runs, the
calls become no-ops and training continues. A crash of the script takes the
window with it, which the attach mode avoids. `train_show.py --mode window` (the
default) runs this way.

When `caliper.window()` is opened before the first call, no session is opened;
call `caliper.session()` as well to have both.

## What you see

Each view gets its own window, tiled on first appearance and titled with the
view's name. Every window starts with a **Pause** button and a line with the
view's sequence number, step-to-screen latency, update count and the number of
snapshots dropped while paused.

| View | Interaction and hover |
|---|---|
| Heatmap, image | Hover shows the row, column and value (RGBA for images). Small tensors are magnified with nearest sampling, so each cell stays a cell. |
| Points | 3-D: left or right drag orbits, middle drag or shift+drag pans, the wheel zooms, **Fit** or the F key re-fits. 2-D: drag pans, the wheel zooms about the cursor. Hover marks the nearest point and shows its index, position and label. |
| Series | A line over the append index; hover shows step and value; the latest value is printed. |
| Histogram | Bars at bin centres; hover shows the bin range and count. |

Views sample the training; they do not record every step. Each view takes at
most 60 snapshots per second (the default `rate`), and the viewer draws the
newest completed one at up to 60 fps, so most steps are skipped by design. In
the benchmark behind these numbers (the same TinyCNN plus a 1024x1024 weight
shown every step, attach mode, MPS), 468 of 9,460 calls in 8 s took a snapshot,
training ran at 98.1 % of the commit-matched control (see
[the commit effect](../explanation/live-streaming.md#sync)) and the median step
to screen was 23.53 ms. A call that took a snapshot cost 35.5 µs median, a
skipped one 2.0 µs.

Two exceptions to "every call is a sample": `series` appends every value it is
given and only redraws at the rate, so the loss curve has every step from the
moment a viewer was attached; and a heatmap's hover value is read from the
shared slot, so it can come from a newer snapshot than the pixels under the
cursor.

**Pause** freezes the display of every live view in that window or viewer.
Snapshots keep arriving and are dropped, so pausing never slows training.
**Resume** shows the newest one again.

In attach mode the viewer also has a status window: the script, PID and session
token, "running" or "SESSION ENDED", frame rate, updates, skipped snapshots,
the step-to-screen median and p95, one line per view, and **Detach**.

## Next

- [Python API](../reference/python-api.md) lists every call, argument and
  accepted layout.
- [Attach to a running script](../howto/attach-to-a-running-script.md) for the
  session workflow.
- [Live streaming](../explanation/live-streaming.md) for how snapshots, the
  slot ring and the two modes work, with the measurements.
