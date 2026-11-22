# Caliper LIB

Caliper LIB is the `caliper` Python package: `import caliper` gives a PyTorch script live views of its tensors. This page
lists every public call of the `caliper` package (version 0.1.0, Python 3.12,
macOS arm64) and of `caliper.live`. The package needs macOS 14 (Sonoma) or
later on Apple silicon, and its wheel is tagged `macosx_14_0_arm64`. The source of truth is
`python/caliper/__init__.py`; the design is in
[Live streaming](../explanations/live-streaming.md), and
[Watch a PyTorch training run live](../introduction/python-quickstart.md) walks through
a script.

The package is pure Python on top of two nanobind extensions. `caliper._native`
binds the embed C ABI (`caliper/embed.h`, including the view calls) of
Caliper Embed built without libtorch, so the core lives next to the torch the
script already imported. `caliper._live` is the client of the shared-memory
live session protocol. `caliper.__version__` is `"0.1.0"`.

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
snapshot, no slot and no GPU work (100,000 calls per row, each timed
singly):

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
snapshots; [the commit effect](../explanations/live-streaming.md#sync) explains
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
caliper.show(t, name="tensor", cmap="viridis", vmin=None, vmax=None, rate=None, *,
             dims=None, data_view=None)
```

Show or update view `name` with tensor `t` as a heatmap or an RGBA image.
With `dims` or `data_view`, the view shows a data view of `t` instead of the
layouts below; see [Data views](#data-views).

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
               vmin=None, vmax=None, rate=None, *, dims=None, data_view=None)
```

A scatter of `z`: `(N, 2)` (pan and zoom) or `(N, 3)` (orbit), any dtype, cast
to float32 on its device; N must be at least 1.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `labels` | tensor `(N,)` or `None` | `None` | Per-point labels or values, on `z`'s device or on the CPU. CPU labels for MPS points are copied to the device without waiting for its queue, from a private copy kept until the device has read it, so the caller may reuse its tensor at once; nothing is copied the other way. `None` draws one flat colour |
| `classes` | `int` or `None` | `None` | K categorical labels 0..K-1, drawn with a fixed 16-colour palette (label L gets colour L mod 16). Needs `labels` |
| `cmap`, `vmin`, `vmax` | | | Without `classes`, labels are colormapped over [vmin, vmax], autoscaled like `show` |
| `name`, `rate` | | `"points"`, `None` | As for `show` |

The bounding box for auto-fit is computed on the device. Wrong shapes, a label
count other than N, labels on a device other than `z`'s or the CPU,
`classes < 1` and `classes` without `labels` raise `ValueError`. With `dims`
or `data_view`, `z` is a source of data views and `labels` follows its kept
dimension; see [Data views](#data-views).

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
caliper.histogram(t, name="histogram", bins=64, rate=None, *, dims=None, data_view=None)
```

Bin counts of every element of `t` (flattened, cast to float32, at least one
element) over its [min, max], with `bins` in 1..65536. The range comes from a
GPU `aminmax`; constant data uses [v - 1, v + 1] as `torch.histc` does, and NaN
values fall into bin 0. On MPS the binning runs on the device (`index_add_`),
because `torch.histc` on MPS reads its range back to the host, a full drain of
the queue measured at 15 to 53 ms. On the CPU it is `torch.histc`. With `dims`
or `data_view`, the histogram bins the data view of `t`; see
[Data views](#data-views).

### watch

```python
handle = caliper.watch(model, activations=True, gradients=False, every=1,
                       include=None, rate=None, groups=True)
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
| `groups` | `bool` or `str` | `True` | Also offer the grouped views below; a string prefixes their names with `<string>.`, and `False` offers none |

The root module is labelled with its class name in lower case. `WatchHandle`
has `remove()` (takes every hook off and removes the grouped views), `active`,
`views` (the view names), `modules` (the hooked module names), `groups` (the
grouped view names), `len()`, and works as a context manager.

The grouped views are heatmaps with [data views](#data-views) whose sources are
the selected modules or their parameters, so one widget can switch between
layers. The same hooks feed them, so `watch()` adds no hook for them.

| View | Sources | Dims | Default data view |
|---|---|---|---|
| `weights` | `<module>.<param>`, every parameter with 2 or more dims | `out`, `in` (higher-rank weights reshaped to `(out, -1)`) | The first source, whole |
| `activations` | `<module>`, with `activations=True` | From the rank: `batch`, `feature`; `batch`, `position`, `feature`; `batch`, `channel`, `height`, `width`; `batch`, `channel`, `length` for a 1-D convolution | The first module, its leading dims at index 0 |
| `gradients` | `<module>.<param>`, with `gradients=True`, as the `.grad` views | As `weights` | The first source, whole |

Because the weights and gradients have the same two dims, `{"source":
"fc2.weight"}` alone switches between them. Activations of different ranks have
different dims, so a request for another module names its own `index` (and
`{}` for a 2-D output). `weights` takes a snapshot when its module runs forward
(with `activations=False`, after its gradient accumulates) and on
`caliper.update()`. A source does tensor work only when a route needs it: the
window and the session need the default data view's source, a server the
sources its subscriptions select; the hook of any other source records the
shape and returns. A view has at most 64 sources, so a group keeps the first
64, and sources whose names are longer than 63 bytes are left out, each with a
`RuntimeWarning`.

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
shows whether the canvas is repainting. The window's core keeps its data, which is its
stores, the applets' data folders and its `imgui.ini`, in the `data_dir` given,
else in `$CALIPER_DATA_DIR`, else in a private temporary directory removed on
close.

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

## Network streaming

`caliper.serve()` serves the script's live views over a WebSocket on the
loopback interface, `caliper.widget()` shows one of them in a notebook cell,
and data views let each viewer choose which part of a tensor travels.
[Network streaming](../explanations/network-streaming.md) explains the design,
the [Wire protocol](wire-protocol.md) reference defines the messages, and
[Watch a training run in a notebook](../tutorials/watch-a-training-run-in-a-notebook.md)
uses all three. `caliper.connect()` streams the same views to a `caliper serve`
hub instead, or as well. `websockets` is a dependency of the package. The widget needs
the extra `caliper[notebook]`, which adds anywidget.

The server's sender is native when `caliper._netio` is built: a C++ thread that
never takes the GIL encodes and sends. `python scripts/build_netio.py`, run with
the script's Python, builds it and copies it into the installed package.
Without it, `serve()` uses a sender written in Python and says so once on
stderr, and `connect()` raises `RuntimeError`. `CALIPER_NETIO=0` chooses the
sender written in Python.

The browser files that the server and the widgets hand out live in the
package's `caliper/_web/` folder. The viewer page and the modules of the
element and the widgets are always there. Caliper's own views compiled to
WebAssembly, `caliper-views-loader.js`, `caliper-views.js` and
`caliper-views.wasm`, are optional: a wheel built in CI ships them, and in a
checkout `python scripts/build_wasm.py` builds them and
`python scripts/fetch_wasm.py` downloads CI's build. Without them every view
draws with the views written in TypeScript, and `serve()` says so once on
stderr, naming those two commands. `CALIPER_WEB_DIR` points the package at
another folder of these files.

### serve

```python
server = caliper.serve(port=0, *, control=True)
```

Start the stream server of this process, print its link,
`http://127.0.0.1:<port>/?token=<token>`, and return it, a
`caliper._net.NetServer`. A second call while it runs returns the same server.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `port` | `int` | 0 | TCP port on 127.0.0.1; 0 lets the system choose a free one |
| `control` | `bool` | `True` | Whether viewers may pause the run, change parameters and run actions ([Controls](#controls)); `False` shows them read-only |

The server listens on 127.0.0.1 only and makes a new random session token.
While it listens, it writes the stream file `streams/<pid>.json` in
Caliper's app-data folder (`$CALIPER_DATA_DIR` when set, see
[Environment](environment.md)), with mode 0600 in a folder of mode 0700: its
pid, the machine's name, the session id, the script's name, the start time,
the port, the stream link with its token, `control` and the names of its
views, which the server rewrites when the views change. The Caliper app lists
the stream from it in `F4 Attach to process`, so that it attaches without the
link being pasted. `close()` removes the file; a killed script's file is
removed by the app's next listing. A file that cannot be written leaves the
server running, and the printed link still attaches. It is one more route: every view call is offered to it, and a view takes a
snapshot for it only while a viewer subscribes to that view, once for each
data view that its subscriptions ask for. Snapshots for the server are computed
on the device and copied into pinned host memory without waiting; the call
still returns at once.

| Address | Serves | Token |
|---|---|---|
| `http://127.0.0.1:<port>/?token=<token>` | The viewer page: every view of the stream in a grid | Not needed; the page reads it from its own link |
| `http://127.0.0.1:<port>/<file>` | The page's modules, the WebAssembly views and the fonts, from the package | Not needed; the files hold no data |
| `ws://127.0.0.1:<port>/v3/stream?token=<token>` | The stream, as the [Wire protocol](wire-protocol.md) reference defines it | Required |

Every request must name the server in its `Host` header, `127.0.0.1:<port>` or
`localhost:<port>`. The stream also refuses a wrong token and an `Origin`
header other than a loopback page (http or https on 127.0.0.1, localhost or
[::1], any port, such as a JupyterLab page), `null` or a VS Code webview. The
files carry `Access-Control-Allow-Origin` for such a page, so a notebook page
can load the widget's modules. The stream carries snapshots, series,
subscriptions, data view requests and viewer events, and never host commands.

The server keeps the last 1,048,576 values of each series whether or not a
viewer watches it, and sends them first to a viewer that subscribes late.

This implementation limits the replies and other control messages waiting to be
sent on one connection to 65,536 messages and 16 MiB, and keeps one PONG
waiting, to the latest PING. Only a viewer that keeps sending requests while it
reads nothing reaches the limit; the server then closes its connection with code
1008, as the hub does with a send queue that overflows. The training loop is not
affected. These are limits of Caliper LIB's server, not rules of the protocol.

| Member | Meaning |
|---|---|
| `url` | The viewer page's link, with the token |
| `stream_url` | The stream's `ws://` URL, with the token |
| `port` | The port the server listens on |
| `token` | The session token |
| `session` | The session id, `clp-<pid>-<6 hex>`, that viewers see |
| `control` | Whether viewers may control the run, as `serve()` was told |
| `closed` | `True` once the server is stopped |
| `stats()` | A dict: `url`, `port`, `session`, `closed`, `attached` (connected viewers), `connections`, `views` (per view its `id`, `kind`, `groups` of subscriptions with one data view, and `seq`), `series` (values retained per series) and counters |
| `on_event(fn)` | Call `fn(event)` for every viewer event from now on; returns `fn`, so it also works as a decorator |
| `remove_event_handler(fn)` | Stop calling `fn`; `True` if it was registered |
| `events()` | The events received since the last call, oldest first, and clears them; never blocks |
| `close()` | Deliver what is pending (for about a second), end every connection and stop the server |

An event is a dict with `view`, `type` (`"select"`, `"pick"` or `"hover"`),
`detail` (in source indices, as the [Wire protocol](wire-protocol.md#events)
reference lists for each kind), `sub`, `seq`, `data_view`, `connection` and
`time`. `on_event` functions run on a thread of their own, `caliper.events`,
never on the training thread; an exception in one becomes a `RuntimeWarning`
and does not stop the others. The server keeps at most 256 unread events for
`events()` and drops the oldest. Events are data for Python code to read;
nothing in them runs as a command.

`close()` may be called more than once. It also runs at `caliper.close()` and at
interpreter exit, and the server works as a context manager that closes it.

### connect

```python
hub = caliper.connect(url=None)
```

Stream every view of this process to a `caliper serve` hub on this machine, as
a producer, and return the connection, a `caliper._net_native.HubConnection`.
[Run a hub](../guides/run-a-hub.md) explains the hub.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `url` | `str` or `None` | `None` | The hub's stream URL with its token, `ws://127.0.0.1:<port>/v3/stream?token=<token>`, or the viewer link it prints. `None` reads `CALIPER_HUB_URL`, then the hub file: `CALIPER_HUB_FILE`, or `hub.json` in the app-data folder (`~/Library/Application Support/Caliper/hub.json` on macOS) |

`connect()` returns at once and never blocks the training loop: the native
sender connects in the background. When the hub cannot be reached or the
connection drops, it tries again after 0.5 seconds, doubling the wait up to 30
seconds, and proposes the session id the hub assigned, so that the hub resumes
the session. It stops when the hub ends the connection with BYE `replaced`.
The views reach the hub through the same calls as for `serve()`: a view takes a
snapshot for the hub only while a viewer of the hub watches it. Caliper LIB
keeps one producer per process, so `serve()` and `connect()` in one script
share it, and every view call reaches both.

It raises `ValueError` for a URL that has no token or names a host other than
127.0.0.1 or localhost, and `RuntimeError` when no URL is given and there is no
hub file, or when the native sender is not built.

| Member | Meaning |
|---|---|
| `url` | The hub's stream URL, without the token |
| `session` | The session id the hub assigned; the proposed `clp-<pid>-<6 hex>` until it answers |
| `state` | `connecting`, `connected`, `waiting` (to try again), `replaced` (another connection of this session took over) or `closed` |
| `connected` | `True` while the connection is open |
| `stats()` | A dict with `state`, `connected`, `session`, `error` (the last failure, or empty), `attempts` and `connects` |
| `wait_connected(timeout=10.0)` | Wait, polling, until the connection is open; `True` if it is. For tests and scripts, not for the training loop |
| `close()` | End the connection to the hub; a running `serve()` keeps serving |

`caliper.close()` ends the connection as well.

### widget

```python
w = caliper.widget(name, kind=None, *, height=320)
```

An anywidget widget that shows the live view `name` in a notebook cell. It
reads the server's WebSocket stream directly, not the notebook's message
channel, so it keeps updating while a cell runs. It is an ipywidgets widget, so
`HBox` and `VBox` lay several out together.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | | The view to show. The widget may be created before the view's first call |
| `kind` | `str` or `None` | `None` | `"series"`, `"heatmap"`, `"image"`, `"points"` or `"histogram"`; `None` takes the kind from the stream |
| `height` | `int` | 320 | Height of the whole widget, header included, in CSS pixels |

When no server runs, `widget()` starts one with `serve()`. An unknown kind
raises `ValueError`. Without anywidget installed, the call raises `ImportError`
and names the extra.

Its traits, synced through anywidget:

| Trait | Type | Meaning |
|---|---|---|
| `src` | `str` | The stream's `ws://` URL, with the token |
| `view` | `str` | The view's name |
| `kind` | `str` or `None` | The live view's kind |
| `height` | `int` | Height in CSS pixels |
| `data_view` | `dict` | The data view of the widget's own subscription, both directions: the browser writes the data view the user picked, and setting it from Python makes the widget request that data view |
| `selection` | `dict` | Written by the browser: the latest event it reported, as `{"view", "type", "detail"}` |

The `detail` of an event is given in source indices; for a `select` in a
points view it holds `indices`, the positions in the points that were shown,
and `row_ids`, their source indices. The [Wire protocol](wire-protocol.md)
reference lists the detail of every kind. `w.on_select(callback)` calls
`callback(selection)` each time the browser reports a new selection, and
`w.on_select(callback, remove=True)` stops it.

Traits travel over the notebook's message channel, which the kernel reads only
while no cell runs. A selection made during a long cell reaches `selection`
when the cell ends, and a `data_view` set from Python reaches the widget at
once. A data view chosen in the widget's own picker goes to the producer over
the stream and takes effect during the cell.

### Data views

A data view decides which data of a tensor travels: a slice, selection,
projection or aggregation that the producer computes on the tensor's device.
[Network streaming](../explanations/network-streaming.md#data-views) explains
the idea, and the [Wire protocol](wire-protocol.md#data-views) reference
defines the rules exactly. A view offers data views when its tensors have
named dimensions:

```python
caliper.show(t, name="tensor", ..., *, dims=None, data_view=None)
caliper.points(z, labels=None, name="points", ..., *, dims=None, data_view=None)
caliper.histogram(t, name="histogram", ..., *, dims=None, data_view=None)
caliper.offer(name, sources, kind="heatmap", dims=None, *, data_view=None, rate=None, **options)
caliper.watch(model, ..., groups=True)       # the grouped views of watch
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `dims` | sequence of `str` or `None` | `None` | One name for each dimension of the tensor, 1 to 63 bytes each, at most 8. An image (a tensor whose last three dimensions hold RGB or RGBA, by the rule of [show](#show)) names every dimension but the colour one. Points name every dimension but the last, which holds the 2 or 3 coordinates, and end with the point dimension. With `data_view` and no `dims` the names are `dim0`, `dim1`, ... |
| `data_view` | `dict` or `None` | `None` | The view's default data view, in the form below |

A call with neither keeps the layouts of `show`, `points` and `histogram`. With
either, the view has one source, the call's tensor, named after the view. A
default without an `index` gets one: the leading dimensions that it does not
reduce or tile are indexed at 0 until the dimensions that remain fit the live
view, so a batch shows item 0 as `show` does. A `data_view` that does not fit
the tensor raises `DataViewError` at the call.

A data view is a dict in which every field is optional:

| Field | Form | Meaning |
|---|---|---|
| `source` | `str` | The source to read |
| `index` | `{"<dim>": i}`, `{"<dim>": [start, stop]}` or `{"<dim>": [start, stop, step]}` | An integer selects one position and drops the dimension; a range keeps the dimension |
| `reduce` | `{"<dim>": op}`, with `op` one of `"mean"`, `"sum"`, `"max"`, `"min"` and `"norm"` | Reduces over the dimension and removes it; `"norm"` is the square root of the sum of squares |
| `tile` | `"<dim>"` | Heatmaps only: lays out a kept dimension as a mosaic of cells |
| `budget` | `{"max_extent": n, "pool": op, "max_points": n}` | The largest heatmap or image extent (default 512, at most 4096), how a heatmap is pooled to meet it (`"mean"`, `"max"` or `"min"`; an image always by mean), and the largest number of points (default 100,000, at most 1,000,000) |

Indices count from 0, and a negative one counts from the end, as in Python. A
field that a request names replaces the default's field as a whole, except
`budget`, whose keys are taken one by one, so `{}` is the default and
`{"index": {}}` the default without its indexing. The dimensions that remain
must fit the live view: two for a heatmap (three with `tile`) or an image, one
for points (the point dimension), any number for a histogram. A budget value
under 1 is refused, and one over the limit is taken as the limit. The producer
applies the index, then the reductions one dimension at a time in the source's
order, then the tile layout and the budget, all on the tensor's device. Pooling
covers each cell with the window of torch's adaptive pooling, and points are
sampled with a stride, point i being position floor(i × count / size), so the
sample stays the same while the size does. A pooled or tiled heatmap reports
the range of its data before pooling, the empty cells left out, for autoscale.
A data view that breaks a rule is refused with the error `bad_data_view`
(`DataViewError` in Python), and the view keeps the data view it had. So is one
whose payload would not fit one frame of the protocol, such as a heatmap or image
of 4096 by 4096; its message names the largest `max_extent` that fits.

**Default and subscriptions.** The view's default data view is what the window
and the live session draw, and what a viewer's subscription starts from. Each
subscription of a viewer, such as a widget or a browser tab, has its own data
view and changes it with a request over the stream, so two viewers can show
different heads of one view at once. The producer computes each distinct data
view once per snapshot for every subscription that shares it, and a source
that no route needs costs nothing. A subscription whose data view no longer
fits, because the source's sizes changed, receives no snapshots until it fits
again; nothing raises in the training loop. A default that no longer fits falls
back to the default the view was registered with, then to the one the rule
above gives, with a `RuntimeWarning`.

| Call | Returns | Meaning |
|---|---|---|
| `data_view(name)` | `dict` or `None` | The view's default data view, resolved: `source`, `index`, `reduce`, `budget`, and `tile` when the heatmap tiles. `None` when the view offers no data views or has no data yet |
| `set_data_view(name, request=None)` | `dict` or `None` | Set the view's default data view from `request` (a dict or its JSON text). Its fields replace the current default's as above, so `{}` changes nothing, and `None` returns to the default the view was registered with. Returns the resolved data view and raises `DataViewError` when it does not fit; the next VIEWS message carries the new default, and a subscription with its own data view keeps it. A request for a view that does not exist yet, or has no data yet, is kept, checked at its first snapshot, and `None` is returned. Safe from any thread |
| `offer(name, sources, kind="heatmap", dims=None, *, data_view=None, rate=None, **options)` | `dict` or `None` | Register view `name` with several sources, for tensors no view call shows, such as attention maps or a layer's output on a fixed probe batch. Returns its default data view, resolved, or `None` while the default source has no tensor |
| `update(name=None)` | `bool` | Take a snapshot of every offered view that is due, and of `watch()`'s `weights`, or of view `name` only; call it in the training loop. Same return values as `show` |
| `DataViewError` | | A `ValueError` whose message names the broken rule |

`offer()` arguments: `sources` maps source names to tensors or to callables
that return the current tensor or `None`; a single tensor or callable is one
source named `name`. A tensor is read at each snapshot, so one updated in
place, such as a parameter, stays current. A callable runs under
`torch.no_grad()`, once at the call to learn the shape and then only when a
route needs that source and a snapshot is due. `dims` is one tuple for every
source or a dict of tuples per source, by default `dim0`, `dim1`, ...; `kind`
is `"heatmap"`, `"image"`, `"points"` or `"histogram"` (a series has no
sources); `options` are the matching call's arguments: `cmap`, `vmin` and
`vmax` (heatmap, points), `labels` (a tensor or a callable, along the kept
dimension) and `classes` (points), and `bins` (histogram). An unknown option
raises `TypeError`. A later `offer()` of the same name replaces the view.

Over a server, a view called without `dims` offers one source named `data`
over its payload: a heatmap or image with dimensions `row` and `col`, points
with `point`, a histogram none. A viewer can range or pool it, which is how it
zooms in on a plain heatmap, and its default is the whole payload within the
budget.

```python
caliper.show(attention, "attention", dims=("layer", "sequence", "head", "query", "key"),
             data_view={"index": {"layer": 1, "sequence": 0, "head": 0}})
widget.data_view = {"index": {"layer": 0, "sequence": 0}, "reduce": {"head": "max"}}
caliper.set_data_view("attention", {"index": {"layer": 0, "sequence": 2, "head": 1}})
```

## Controls

Controls let viewers reach back into the running script: pause, resume and
single-step its training loop, change the parameters it registered, and run
the actions it registered. Linked selection makes a selection in one view
highlight the same items in the other views. Both travel on the stream as
messages of wire protocol 3.1 ([Controls](wire-protocol.md#controls) and
[Linked selection](wire-protocol.md#linked-selection)), so they work while a
notebook cell runs, when the kernel cannot read widget messages. The viewers
of `caliper.serve()` may control unless it was started with `control=False`;
the viewers of a hub that `connect()` streams to may control only when the hub
was started with `--allow-control`. Both senders carry them.
[Control a running script](../guides/control-a-running-script.md) uses them in
a training loop. Every call also works without a server: parameters keep their
values, `step()` returns at once, and an action can be called from Python.

| Call | Returns | Does |
|---|---|---|
| `step(step=None)` | `None` | The checkpoint of the training loop: sets changed parameter values, runs requested actions, waits while the run is paused |
| `pause()` | `bool` | Pauses the run at the next `step()` |
| `param(name, value, *, min=None, max=None, log=False, choices=None, help=None)` | `Param` | A value that viewers can change |
| `action(name, fn=None, *, help=None)` | `Action` | A function that viewers can run; also a decorator |
| `items(space, details=None)` | `ItemSpace` | An item space for linked selection, with optional details |
| `controls(*, height=160)` | widget | The notebook panel of run controls, parameters and actions |

### step

```python
caliper.step(step=None)
```

The checkpoint of the training loop. Call it once per step, on the training
thread, between two steps. It is the only place where the controls change
the script: there it sets the parameter values that viewers changed, runs the
actions they asked for in the order they asked, and waits while a viewer has
paused the run. `step` is the step's number, which viewers show where the run
paused; `None` counts the calls.

A waiting `step()` returns when a viewer resumes the run or asks for a number
of single steps, when the server closes, or when no connection that may
control has been open for 10 seconds, which a page reload stays well within.
While it waits, it still sets values and runs actions as they arrive. Ctrl-C,
or a notebook's interrupt, ends the wait with `KeyboardInterrupt`, and the run
is no longer paused afterwards. Right before it waits, it sends the snapshots
and series values of the step it paused at, and the run state goes to each
viewer after them, so a view that reads `paused` already shows that step
([What a viewer may rely on](wire-protocol.md#what-a-viewer-may-rely-on-when-a-run-pauses)).
With nothing waiting, a call costs about 22 ns on an Apple silicon Mac, with a
viewer attached or not: it counts the step and reads one flag.

Before the script's first `step()`, viewers can change parameters, which then
take effect at once, but cannot pause the run, and actions wait for the first
checkpoint.

### pause

```python
caliper.pause()
```

Pause the run at the next `step()`, as a viewer's Pause button does, for
example when the loss stops being finite. A viewer resumes it. Returns `True`.
When no stream server runs, no viewer could resume the run, so it returns
`False` with a `RuntimeWarning` and does nothing. When a server runs but no
connection may control it, it warns that the run resumes after 10 seconds.

### param

```python
lr = caliper.param("lr", 1e-3, min=1e-5, max=1e-1, log=True, help="learning rate")
for batch in loader:
    caliper.step()
    for group in opt.param_groups:
        group["lr"] = lr.value
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | | 1 to 63 bytes. Registering a name again replaces the parameter |
| `value` | see below | | The first value; its type sets the parameter's type |
| `min`, `max` | number | `None` | The range of a number or an integer; a value from a viewer is clamped to it |
| `log` | `bool` | `False` | Draw the slider on a log scale; needs `min` greater than 0 and `max` |
| `choices` | list of `str` | `None` | The choices of a `str` value, 1 to 64 |
| `help` | `str` | `None` | A tooltip, at most 1,024 bytes |

| `value` | Type | A viewer shows |
|---|---|---|
| `bool` | `boolean` | A toggle |
| `int` | `integer` | A slider or a field |
| `float` | `number` | A slider or a field |
| `str`, with `choices` | `choice` | A menu |

A value from a viewer is checked and clamped: a number or an integer to `min`
and `max`, an integer must be a whole number, and a choice must be one of
`choices`. It reaches `value` at the next `step()`, or at once before the
script's first `step()`. When a viewer sends several values before a `step()`,
the last one wins. A script has at most 64 parameters. `param()` returns a
`caliper.Param`:

| Member | Meaning |
|---|---|
| `value` | The current value, to read in the loop. It changes only in `step()`, or at once before the first `step()` |
| `pending` | A value from a viewer that `value` takes at the next `step()`, or `None` |
| `set(v)` | Set the value from the script, at once and for every viewer, checked and clamped like a viewer's value; returns the value set, or raises `ValueError` |
| `remove()` | Take the parameter away from the viewers; `value` keeps its last value |
| `name`, `type`, `min`, `max`, `log`, `choices`, `help` | The registration |

### action

```python
@caliper.action("save checkpoint", help="saves the model's weights")
def save_checkpoint():
    torch.save(model.state_dict(), "checkpoint.pt")
    return "saved checkpoint.pt"
```

`caliper.action(name, fn)` registers `fn`, a function without arguments, and
`caliper.action(name)` is the same as a decorator. A viewer's request runs it
at the next `step()`, on the training thread, after the parameter values are
set. Requests run one at a time in the order they arrived, and a request for
an action that already waits joins that run. The text of what `fn` returns
goes back to the viewer that asked, cut to 1,024 bytes. An exception goes back
to that viewer with its message, appears in the script as a `RuntimeWarning`,
and does not stop the loop. A request that no `step()` has run within 10
minutes fails with a time-out. A script has at most 64 actions, and
registering a name again replaces the action.

`action()` returns a `caliper.Action`. Calling it runs `fn` at once on the
caller's thread, and `remove()` takes it away from the viewers.

### items

```python
caliper.items("eval", details={"class": eval_labels})
caliper.points(z, eval_labels, "embedding", classes=10, items="eval")
caliper.show(probabilities, "predictions", items="eval")
```

An item space is a set of items numbered from 0 that several views index,
such as the samples of an evaluation set. A view names the item space of its
item axis with `items=`, a keyword of `show`, `points`, `series` and
`offer`:

| Call | `items="eval"` names the space of |
|---|---|
| `show` | Its rows: the first dimension, or the first of `dims` |
| `points` | Its points: the point dimension |
| `series` | Its index, the number of values appended before each value |
| `offer` | The first dimension of each source, or the point dimension of points |

`items={"sample": "eval"}` names the dimensions of `dims` instead. A histogram
has no item axis. A selection made on an axis with an item space highlights
the same items in every view of the session that names the space: the
points, the rows of a heatmap, markers on a series. Without `items=`, a call
costs what it did.

`details` gives a viewer a few fields of the selected items, shown in a small
table under the view in which the selection was made. It is a mapping
`{field: sequence}` indexed by item id, copied to the host once, now; or a
function `fn(ids)` that returns `{field: values}` in the order of `ids`, or
one dict per item. The function runs on the thread `caliper.details`, never
on the training thread, and has 2 seconds, so it must read only data that the
loop does not change in place, such as labels or arrays the script computed
before. Details cover the first 32 selected items and at most 16 fields, and
each value is a number, a string of at most 256 bytes, a boolean or `None`.

`items()` returns a `caliper.ItemSpace`. Its `selection` is the list of
selected item ids, ascending, and it is current even while a cell runs.
`select(ids)` sets the selection for every viewer, and `clear()` clears it. A
session has at most 16 item spaces. Python code still receives each select
event as before, through `server.on_event`, `server.events()`,
`widget.selection` and `on_select`, and its `detail` now has `space` and
`items` too.

### controls

```python
caliper.controls(*, height=160)
```

A notebook widget, like `widget()`, with the run state and the Pause, Resume
and Step buttons, every parameter and every action of the script, and the
result of the last action it ran. It reads the stream directly, so it works
while a cell runs, and it starts `serve()` when no server runs. Its traits are
`src` and `height`. The header of every view widget also shows the run state
and the Pause, Resume and Step buttons. A build whose browser assets lack the
panel's module, `caliper-controls-widget.js`, warns and shows a note in its
place.

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
| `CALIPER_DATA_DIR` | Root of the session files (`<root>/live`) for the script, the Caliper app's session list and `caliper-viewer`; also the in-process core's data directory when no `data_dir` is given. For the Caliper app it moves the whole app-data folder ([Folders and environment variables](environment.md#the-app-data-folder)) |
| `CALIPER_VIEWER` | Path of `caliper-viewer` for `auto_launch` and `viewer_executable()` |
| `CALIPER_PY_LOG` | Non-empty and not `0`: echo the in-process core's log lines to stderr |
| `CALIPER_PY_LIVE_APPLY` | `0`: in-process heatmaps and images use the embed v1.2 view path (`view_tensor` with `WAIT_GPU` on the frame thread) instead of the typed live views |

## Threading

View calls (`show`, `points`, `series`, `histogram`, and `watch`'s hooks),
`flush`, `remove` and `stats` may be called from any thread; a module lock
serialises them. `window()` and `offscreen()` must be called from the main
thread, as must `run()`, `frame()` and `spin()` while a real window is open;
they raise `RuntimeError` otherwise.
