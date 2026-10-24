# Python API (caliper)

`import caliper` gives a PyTorch script live views of its tensors. This page
lists every public call of the `caliper` package (version 0.1.0, Python 3.12,
macOS arm64) and of `caliper.live`. The source of truth is
`python/caliper/__init__.py`; the design is in
[Live streaming](../explanation/live-streaming.md), and
[Watch a PyTorch training run live](../tutorials/live-pytorch.md) walks through
a script.

Importing the package does not touch GLFW, AppKit, Metal or torch; the native
extension loads on first use. Tensor arguments are torch tensors (anything else
goes through `torch.as_tensor`) and are detached before use.

## Routing

A call is offered to every **route** that is open, in this order:

1. The **live session** (attach to app). The first view call of a script opens
   it implicitly when no route is open yet, unless `CALIPER_SESSION=0`. If the
   build has no session transport, that first call warns once and calls do
   nothing until `window()` or `offscreen()` opens a route.
2. The **in-process core**: the window of `caliper.window()` or the canvas of
   `caliper.offscreen()`.

Each route that is **attached** takes its own snapshot of the call: the session
while a viewer's heartbeat is younger than 1 s, the in-process core while it is
open. The device-side preparation is built once and shared. With a viewer
attached and a window open, both receive every snapshot. With nothing attached,
the call returns `False` after the check.

Opening `caliper.window()` before the first view call means no session is
opened; call `caliper.session()` as well to have both.

View calls return `True` if at least one route took a snapshot, `False` when
every route skipped it (rate limit, all slots busy, nothing attached), and a
[`ShowStats`](#constants-and-types) in the synchronous offscreen mode.

Name, colormap and range arguments are checked on every call. Checks on the
tensor itself (shape, label count, bins) run when a snapshot is built, so a bad
shape raises only once something is attached.

## Cost of a call

With nothing attached a call is one atomic load and a clock read, with no
snapshot, no slot and no GPU work (`S-noviewer.txt`, 100,000 calls per row,
each timed singly):

| Call, nothing attached | Median | p99 |
|---|---|---|
| `caliper.show`, 1024x1024 MPS or 256x256 CPU | 0.167 µs | 0.291 to 0.292 µs |
| `caliper.series(1.0)` | 0.125 µs | 0.209 µs |

Attached, in `python/bench/bench_show.py` (the TinyCNN of `train_show.py` on
64-image batches plus a square f32 weight updated in place and shown every
step, autoscaled; 256², 1024² and 2048²; 8 s per row after a 1 s warm-up;
torch 2.14.1, 18 cores, load average about 3):

| MPS | Own window | Attach to app |
|---|---|---|
| Throughput at 60 Hz, % of the commit-matched control | 98.3 to 99.5 % | 98.1 to 99.4 % |
| Throughput with `rate=0`, % of the control | 94.3 to 97.6 % | 97.5 to 97.9 % |
| Call that took a snapshot, median (60 Hz) | 37.7 to 48.1 µs | 34.4 to 39.8 µs |
| Call skipped by the rate limit, median | 2.17 to 2.21 µs | 2.0 to 2.04 µs |
| Call skipped with all slots busy (`rate=0`), median | 7.5 to 7.83 µs | 8.17 to 8.33 µs |
| Step to screen, median, 60 Hz | 23.06 to 23.12 ms | 22.81 to 24.08 ms |
| Step to screen, median, `rate=0` | 8.39 to 9.9 ms | 9.23 to 9.75 ms |

The control is the same loop recording a `torch.mps.Event` at 60 Hz without
snapshots; [the commit effect](../explanation/live-streaming.md#sync) explains
why. `torch.mps.current_allocated_memory()` changed by 0.0 MB in every row. A
CPU 256x256 snapshot cost 80.0 µs (window) and 66.1 µs (attach) median.

## Rate limit

Each view has a rate limit per route, set by the `rate` argument of the call:

| `rate` | Meaning |
|---|---|
| `None` (default) | The route's default: 60 snapshots/s (`DEFAULT_RATE_HZ`) for the session and the window (`window(rate=...)` changes the window's), every call for `offscreen()` |
| `> 0` | At most that many snapshots per second for this view |
| `0` | A snapshot on every call, whenever a slot is free |

The interval runs from the view's last snapshot. A call inside it returns
`False`. A view other than a series has 3 slots; when all are busy (the GPU has not finished the
earlier snapshots, or the viewer still reads them) the call takes no snapshot,
returns `False` and the interval does not restart. A negative or NaN `rate`
raises `ValueError`.

`series` is the exception: every call appends its values (on every attached
route), and the view is republished at most `rate` times per second, one
pending snapshot at a time. `watch(every=N)` drops N-1 of N hook calls before
the rate limit applies.

## View calls

### show

```python
caliper.show(t, name="tensor", cmap="viridis", vmin=None, vmax=None, rate=None)
```

Show or update view `name` with tensor `t` as a heatmap or an RGBA image.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `t` | tensor | | See the layout table below |
| `name` | `str` | `"tensor"` | View id; the same name updates in place. Non-empty, no `##` |
| `cmap` | `str` or `int` | `"viridis"` | `"viridis"`, `"magma"`, `"rdbu"` (case-insensitive) or 0, 1, 2 |
| `vmin`, `vmax` | `float` or `None` | `None` | Fixed ends of the colormap range; `None` takes that end from a GPU `aminmax` of the snapshot. Each end is independent; both must be finite |
| `rate` | `float` or `None` | `None` | See [Rate limit](#rate-limit) |

| Input | Shown as |
|---|---|
| 0-D or 1-D | A 1xN heatmap strip |
| 2-D `(H, W)` | Heatmap (cast to float32 in the copy kernel) |
| `(H, W, 3)` or `(H, W, 4)` | RGBA image |
| `(1, H, W)` | Heatmap |
| `(3, H, W)` or `(4, H, W)` | RGBA image |
| `(C, H, W)`, any other C | Channel mosaic heatmap: ceil(sqrt(C)) columns, empty cells NaN, range over all channels |
| `(N, C, H, W)` | Item 0, then as above |
| More than 4 dims | Leading dims stacked as channels, `(-1, H, W)` |

A 3-D tensor whose last dimension is 3 or 4 is read as `(H, W, C)` when it is
uint8 or its first dimension is not 1, 3 or 4, and as `(C, H, W)` otherwise.
Image values: uint8 as is, floating point clamped to [0, 1] and scaled to 255,
other integers clamped to 0..255; a missing alpha channel is 255. Booleans
become float32 (uint8 for an image), complex tensors their magnitude. With
autoscale, a non-finite range keeps the last finite one and a flat range widens
by ±0.5. NaN values draw as colormap index 0.

### points

```python
caliper.points(z, labels=None, name="points", classes=None, cmap="viridis",
               vmin=None, vmax=None, rate=None)
```

A scatter of `z`: `(N, 2)` (pan and zoom) or `(N, 3)` (orbit), any dtype, cast
to float32 on its device; N must be at least 1.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `labels` | tensor `(N,)` or `None` | `None` | Per-point labels or values, on `z`'s device or on the CPU (uploaded host to device, never the other way). `None` draws one flat colour |
| `classes` | `int` or `None` | `None` | K categorical labels 0..K-1, drawn with a fixed 16-colour palette (label L gets colour L mod 16). Needs `labels` |
| `cmap`, `vmin`, `vmax` | | | Without `classes`, labels are colormapped over [vmin, vmax], autoscaled like `show` |
| `name`, `rate` | | `"points"`, `None` | As for `show` |

The bounding box for auto-fit is computed on the device. Wrong shapes, a label
count other than N, labels on a device other than `z`'s or the CPU,
`classes < 1` and `classes` without `labels` raise `ValueError`.

### series

```python
caliper.series(value, name="series", rate=None)
```

Append `value` to the series `name`: a Python `int` or `float`, a 0-d tensor
(pass `loss.detach()`, not `loss.item()`, to avoid a GPU sync), or a tensor of
several values (flattened; more than 4096 keeps the last 4096). The series is a
ring of the last `SERIES_CAPACITY` (4096) float32 values on the value's device;
a CPU series that later receives device values moves to the device with one
host-to-device copy of its history. The x axis is the append index. Values
offered while nothing is attached are not recorded. An empty tensor raises
`ValueError`.

### histogram

```python
caliper.histogram(t, name="histogram", bins=64, rate=None)
```

Bin counts of every element of `t` (flattened, cast to float32, at least one
element) over its [min, max], with `bins` in 1..65536. The range comes from a
GPU `aminmax`; constant data uses [v - 1, v + 1] as `torch.histc` does, and NaN
values fall into bin 0. On MPS the binning runs on the device (`index_add_`),
because `torch.histc` on MPS reads its range back to the host, a full drain of
the queue measured at 15 to 53 ms. On the CPU it is `torch.histc`.

### watch

```python
handle = caliper.watch(model, activations=True, gradients=False, every=1,
                       include=None, rate=None)
```

Hooks that call `show` for a module's outputs and gradients. Returns a
`WatchHandle`.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `model` | `torch.nn.Module` | | Anything else raises `TypeError` |
| `activations` | `bool` | `True` | A forward hook per selected module shows its output as `<module>.act` (the first tensor of a tuple output; floating-point, non-empty only). `(N, C, H, W)` outputs show item 0 as a channel mosaic, `(N, F)` a heatmap |
| `gradients` | `bool` | `False` | Each selected module's own parameters with 2 or more dims and `requires_grad` show their gradient as `<module>.<param>.grad`, colormap rdbu, higher-rank weights reshaped to `(out, -1)` |
| `every` | `int` | 1 | Show 1 of every N hook calls (per hook); must be at least 1 |
| `include` | `None`, `str`, list of `str`, or callable | `None` | `None`: leaf modules. Strings are `fnmatch` patterns over `named_modules()` names. A callable is `predicate(name, module) -> bool` |
| `rate` | | `None` | Passed to `show` |

The root module is labelled with its class name in lower case. `WatchHandle`
has `remove()` (takes every hook off), `active`, `views` (the view names),
`modules` (the hooked module names), `len()`, and works as a context manager.

## No host copy for MPS

A snapshot is a few torch operations queued on the tensor's own stream, writing
into a slot that the route allocated as CPU-readable shared memory (a shared-
memory segment for the session, a Caliper-made shared `MTLBuffer` for the
window) and handed to torch through DLPack. The snapshot code calls no
`.item()`, `.cpu()`, `.numpy()`, `.tolist()` and no `synchronize()`
(`python/tests/test_routing.py` checks this); completion is detected by polling
a `torch.mps.Event`. `synchronize()` appears only in `flush()` and `close()`.

A CPU tensor is copied once into the slot. Tensors on any other device (CUDA)
have no zero-copy path on macOS and are moved to the CPU first, one host copy.
Labels for `points` may move host to device, never device to host.

## Routes and lifetime

| Call | Returns | Behaviour |
|---|---|---|
| `session(auto_launch=False, viewer_path=None, script_name=None)` | `SessionTransport` | Open the live session, or return the open one; attach to it from the Caliper app (F4 Attach to process). `auto_launch=True` also spawns `caliper-viewer --session <token>` unless a spawned viewer is running or a viewer is attached. `viewer_path` overrides the viewer binary; `script_name` the name viewers show (default: the basename of `sys.argv[0]`) |
| `window(width=1280, height=800, title="caliper", *, offscreen=False, fps=60.0, rate=None, activate=True, heartbeat=False)` | `Window` | Open the in-process window and route every call to it; an open window of the same kind is returned as is and any other canvas is replaced. Main thread only |
| `offscreen(width=256, height=256)` | `Window` | The synchronous offscreen canvas; replaces an open window. Main thread only |
| `run(fn, *args, **kwargs)` | `fn`'s result | With a window open, run `fn` on a worker thread while the main thread keeps the window interactive; otherwise call `fn` |
| `spin(seconds=None)` | `None` | Flush, then keep the view alive: with a window, run its event loop until it is closed or `seconds` pass (main thread); with a session, wait until the viewer exits or `seconds` pass (at once when no viewer is attached and none was spawned) |
| `flush(timeout=5.0)` | `int` | Publish every pending snapshot, polling their GPU events for up to `timeout` s; returns how many are still pending. Not for the training loop |
| `close()` | `None` | Close every route. The next view call opens a new session. Also runs at interpreter exit |
| `remove(name)` | `bool` | Remove view `name` from every route; `True` if the in-process core had it |

`window()` arguments: `offscreen=True` drives an offscreen canvas from the same
render thread (benchmarks, CI, `render()`); `fps` is the render thread's frame
rate; `rate` the route's default snapshot rate; `activate=False` shows the window
without taking focus from the terminal; `heartbeat=True` adds a view named
`caliper.render` that scrolls one texel per rendered frame, so a screenshot
shows whether the canvas is repainting. The window's core keeps its data under
`$CALIPER_DATA_DIR`, or a private temporary directory removed on close.

The window's limits: macOS delivers input only to the main thread, so input
waits during long steps unless `run()` gives the main thread to Caliper
(outside `run()`, a view call on the main thread pumps events at most 60 times
a second). A crash of the script takes the window down, and a hung window
hangs the script. One core per process: one window or offscreen canvas at a
time. While open, the window holds an `NSProcessInfo` user-initiated,
latency-critical activity, because macOS App Nap otherwise throttles an
occluded window's process; measured before that fix, the render thread ran at
9 fps and training at 56 to 60 % of the control.

`run()` details: the worker is a daemon thread named `caliper.train`; `run`
re-raises its exception; Ctrl-C is forwarded to it as `KeyboardInterrupt`. If
the user closes the window, `fn` keeps running, its view calls become no-ops,
and the window closes when `fn` returns.

`Window` methods: `run(fn, *args, **kwargs)` (as above), `stats(reset=False)`,
`render()`, `close()`, the `closed` property, and use as a context manager.

**`offscreen()`** is for tests and notebooks. There is no render thread: each
call waits for its snapshot's GPU event, applies it on the calling thread and
renders a frame (at most 60 per second), so `view_info`, `view_readback` and
`render()` see the call at once. The default rate is every call. In a notebook,
`caliper.render()` returns the composited frame as an `(H, W, 4)` uint8 array
to display.

## Inspection and test helpers

These read the in-process core and raise `RuntimeError` without a `window()` or
`offscreen()`, except where noted. Arrays are numpy.

| Call | Returns |
|---|---|
| `stats()` | `dict` per route: `"session"` has `ring` counters and `transport` (`SessionTransport.stats()`); `"window"` has `ring` plus the render stats below. Works without a core |
| `frame()` | Render one frame now; its wall time in ms. Main thread for a real window |
| `render()` | `(H, W, 4)` uint8 of a freshly composited frame (offscreen canvases) |
| `read_pixels()` | `(H, W, 4)` uint8 copy of the last composited frame |
| `view_info(name)` | Heatmap/image view: `path`, `width`, `height`, `device` (`"metal"`, `"cpu"`), `updates`, `last_update_ms`, `imported`, `creates` |
| `view_readback(name)` | `(H, W, 4)` uint8 of a heatmap/image view's texture |
| `view_remove(name)` | Same as `remove(name)` |
| `view_count()` | Number of heatmap/image views; 0 without a core |
| `live_view_info(name)` | Any typed view: `kind` (1 heatmap, 2 image, 3 points, 4 series, 5 histogram), `path`, `width`, `height`, `updates`, `creates`, `dropped`, `seq`, `stamp_ns`, `last_apply_ms`, `latency_ms`, `imported`, `read_done`, `count` |
| `live_view_values(name)` | float32 array: series values oldest first, or histogram counts |
| `live_view_readback(name)` | `(H, W, 4)` uint8 of a heatmap, image or points view |
| `live_pause(paused=True)` | Freeze the display of every typed view; snapshots keep flowing and are dropped |
| `live_paused()` | `bool`; `False` without a core |
| `colormap_lut(name="viridis")` | `(256,)` uint32 LUT, each entry `R | G<<8 | B<<16 | A<<24`. Needs no core |
| `log_lines()` | The core's log lines; `[]` without a core |

Ring counters: `taken`, `published`, `skipped` (rate limit), `ring_full`,
`superseded` (a newer snapshot completed first), `appended` (series values),
`allocs`, `recreated`. Window render stats: `running`, `fps`, `frames`,
`published`, `superseded`, `drawn`, `deferred_takes`, `errors`, `last_error`,
`stats_unreadable`, `nonfinite_range`, `queued`, `max_frame_gap_ms`,
`events_applied`, `heatmap_live_apply`, and the sample lists `latency_ms`
(snapshot to first frame drawn with it), `tick_ms` and `frame_gap_ms`;
`Window.stats(reset=True)` clears them.

## caliper.live

`from caliper import live` (not imported by `import caliper`).

| Call | Returns |
|---|---|
| `live.list_sessions(data_root=None)` | Live sessions under `<data_root>/live`, newest first, as dicts: `pid`, `alive`, `view_count`, `protocol`, `compatible`, `start_unix_ms`, `viewer_pid`, `session`, `script`, `path`. Stale sessions are reclaimed, not listed |
| `live.data_root()` | The default data root: `$CALIPER_DATA_DIR`, else Caliper's app-data folder |
| `live.viewer_executable(path=None)` | `path`, else `$CALIPER_VIEWER`, else the viewer in the package (`caliper/bin/caliper-viewer`); `FileNotFoundError` if missing |

`SessionTransport` (what `caliper.session()` returns):

| Member | Meaning |
|---|---|
| `session` | The token (`clp-<pid>-<6 hex>`), what `caliper-viewer --session` takes |
| `session_file` | The session file's path; `""` once closed |
| `closed` | `True` after `close()` |
| `attached()` | `True` while a viewer heartbeat is younger than 1 s; safe from any thread |
| `launch_viewer(exit_after=None, args=(), log=None)` | Spawn `caliper-viewer --session <token>` in its own process session; `log=None` discards its stderr, `"-"` inherits it, a path writes it there. Returns the `Popen` |
| `viewer` | The spawned viewer's `Popen`, or `None` |
| `viewer_alive()` | The spawned viewer runs; without one, whether a viewer is attached |
| `wait_attached(timeout=10.0)` | Wait for a viewer heartbeat; `True` if attached |
| `wait_viewer(timeout=None)` | Wait until the viewer is gone; `True` if it is |
| `stats()` | Session, file, script, attached, viewer pid and heartbeat age, and per view the publish and claim counters |
| `close()` | Remove the session file and unlink the shared memory, never waiting for the viewer |

## Constants and types

| Name | Value |
|---|---|
| `CMAPS` | `{"viridis": 0, "magma": 1, "rdbu": 2}` |
| `SERIES_CAPACITY` | 4096 |
| `DEFAULT_RATE_HZ` | 60.0 |
| `ShowStats` | Returned in the offscreen mode, always truthy: `name`, `kind` (`"heatmap"`, ...), `path` (renderer path, e.g. `"compute-imported"`), `imported`, `total_ms` |
| `HAVE_VIEWS`, `HAVE_LIVE_VIEWS`, `VIEW_FLAG_RAW`, `VIEW_FLAG_WAIT_GPU`, `EMBED_API_VERSION`, `HEATMAP_IMAGE_VIA_LIVE_APPLY` | Native module attributes; reading one loads the extension |

View names in a session are 1 to 63 bytes, and a session holds at most 64
views. Reusing a name with another kind rebuilds the view; a change of shape,
dtype or device reallocates its slots.

## Environment variables

| Variable | Effect |
|---|---|
| `CALIPER_SESSION` | `0` stops the first view call from opening a session (default `1`) |
| `CALIPER_DATA_DIR` | Root of the session files (`<root>/live`) for the script, the Caliper app's session list and `caliper-viewer`; also the in-process core's data directory |
| `CALIPER_VIEWER` | Path of `caliper-viewer` for `auto_launch` and `viewer_executable()` |
| `CALIPER_PY_LOG` | Non-empty and not `0`: echo the in-process core's log lines to stderr |
| `CALIPER_PY_LIVE_APPLY` | `0`: in-process heatmaps and images use the embed v1.2 view path (`view_tensor` with `WAIT_GPU` on the frame thread) instead of the typed live views |

## Threading

View calls (`show`, `points`, `series`, `histogram`, and `watch`'s hooks),
`flush`, `remove` and `stats` may be called from any thread; a module lock
serialises them. `window()` and `offscreen()` must be called from the main
thread, as must `run()`, `frame()` and `spin()` while a real window is open;
they raise `RuntimeError` otherwise.
