---
kind: reference
figure: none
---

# Routes and Inspection

The `caliper` package sends each view call to the routes that are open, and it offers calls to open and close routes and to inspect the in-process core.

## Routing

A call is offered to every **route** that is open, in this order:

1. The **live session** (attach to app). The first view call of a script opens it when no route is open yet, unless `CALIPER_SESSION=0`. If the build has no session transport, that first call warns once. Then calls do nothing until `window()` or `offscreen()` opens a route.
2. The **in-process core**: the window of `caliper.window()` or the canvas of `caliper.offscreen()`.

Each route that is **attached** takes its own snapshot of the call. The session is attached while the heartbeat of a viewer is younger than 1 s, and the in-process core is attached while it is open. The device-side preparation is built once and shared. With a viewer attached and a window open, both receive every snapshot. With nothing attached, the call returns `False` after the check.

If `caliper.window()` opens before the first view call, no session opens. Call `caliper.session()` as well to have both.

View calls return `True` if at least one route took a snapshot. They return `False` when every route skipped it (rate limit, all slots busy, nothing attached). In the synchronous offscreen mode they return a [`ShowStats`](caliper-lib-api.md#constants-and-types).

Name, colormap and range arguments are checked on every call. Checks on the tensor itself (shape, label count, bins) run when a snapshot is built. Thus a bad shape raises only when something is attached.

## Routes and Lifetime

| Call | Returns | Behavior |
|---|---|---|
| `session(auto_launch=False, viewer_path=None, script_name=None)` | `SessionTransport` | Open the live session, or return the open one. Attach to it from the Caliper app (F4 Attach to process). `auto_launch=True` also spawns `caliper-viewer --session <token>` unless a spawned viewer is running or a viewer is attached. `viewer_path` overrides the viewer binary. `script_name` sets the name that viewers show (default: the basename of `sys.argv[0]`) |
| `window(width=1280, height=800, title="caliper", *, offscreen=False, fps=60.0, rate=None, activate=True, heartbeat=False)` | `Window` | Open the in-process window and route every call to it. An open window of the same kind is returned as is, and any other canvas is replaced. Main thread only |
| `offscreen(width=256, height=256)` | `Window` | The synchronous offscreen canvas. It replaces an open window. Main thread only |
| `run(fn, *args, **kwargs)` | result of `fn` | With a window open, run `fn` on a worker thread while the main thread keeps the window interactive. Otherwise call `fn` |
| `spin(seconds=None)` | `None` | Flush, then keep the view alive. With a window, run its event loop until it closes or `seconds` pass (main thread). With a session, wait until the viewer exits or `seconds` pass. The wait ends at once when no viewer is attached and none was spawned |
| `flush(timeout=5.0)` | `int` | Publish every pending snapshot and poll their GPU events for up to `timeout` s. Returns how many are still pending. Not for the training loop |
| `close()` | `None` | Close every route. The next view call opens a new session. Also runs at interpreter exit |
| `remove(name)` | `bool` | Remove view `name` from every route. `True` if the in-process core had it |

`window()` takes these arguments:

- `offscreen=True` drives an offscreen canvas from the same render thread (CI, `render()`).
- `fps` is the frame rate of the render thread.
- `rate` is the default snapshot rate of the route.
- `activate=False` shows the window without taking focus from the terminal.
- `heartbeat=True` adds a view named `caliper.render` that scrolls one texel per rendered frame. A screenshot then shows whether the canvas repaints.

The core of the window keeps its data in the `data_dir` that the call gives, else in `$CALIPER_DATA_DIR`, else in a private temporary directory that closes with the window. The data are its stores, the data folders of the applets and its `imgui.ini`.

The window has these limits:

- macOS delivers input only to the main thread. Input waits during long steps unless `run()` gives the main thread to Caliper. Outside `run()`, a view call on the main thread pumps events at most 60 times a second.
- A crash of the script takes the window down, and a hung window hangs the script.
- A process has one core, so only one window or offscreen canvas exists at a time.

While open, the window holds an `NSProcessInfo` user-initiated, latency-critical activity. Without it, macOS App Nap throttles the process of an occluded window.

`run()` starts a daemon thread named `caliper.train` and re-raises its exception. Ctrl-C reaches the thread as `KeyboardInterrupt`. If the user closes the window, `fn` keeps running and its view calls become no-ops. The window closes when `fn` returns.

`Window` has these members: `run(fn, *args, **kwargs)` (as above), `stats(reset=False)`, `render()`, `close()` and the `closed` property. It also works as a context manager.

`offscreen()` is for tests and notebooks. It has no render thread. Each call waits for the GPU event of its snapshot, applies it on the calling thread and renders a frame (at most 60 per second). Thus `view_info`, `view_readback` and `render()` see the call at once. The default rate is every call. In a notebook, `caliper.render()` returns the composited frame as an `(H, W, 4)` uint8 array to display.

## Inspection and Test Helpers

These calls read the in-process core. They raise `RuntimeError` without a `window()` or `offscreen()`, except where noted. Arrays are numpy.

| Call | Returns |
|---|---|
| `stats()` | `dict` per route. `"session"` has `ring` counters and `transport` (`SessionTransport.stats()`). `"window"` has `ring` plus the render stats below. Works without a core |
| `frame()` | Render one frame now. Returns its wall time in ms. Main thread for a real window |
| `render()` | `(H, W, 4)` uint8 of a freshly composited frame (offscreen canvases) |
| `read_pixels()` | `(H, W, 4)` uint8 copy of the last composited frame |
| `view_info(name)` | Heatmap or image view: `path`, `width`, `height`, `device` (`"metal"`, `"cpu"`), `updates`, `last_update_ms`, `imported`, `creates` |
| `view_readback(name)` | `(H, W, 4)` uint8 of the texture of a heatmap or image view |
| `view_remove(name)` | Same as `remove(name)` |
| `view_count()` | Number of heatmap and image views. 0 without a core |
| `live_view_info(name)` | Any typed view: `kind` (1 heatmap, 2 image, 3 points, 4 series, 5 histogram), `path`, `width`, `height`, `updates`, `creates`, `dropped`, `seq`, `stamp_ns`, `last_apply_ms`, `latency_ms`, `imported`, `read_done`, `count` |
| `live_view_values(name)` | float32 array: series values oldest first, or histogram counts |
| `live_view_readback(name)` | `(H, W, 4)` uint8 of a heatmap, image or points view |
| `live_pause(paused=True)` | Freeze the display of every typed view. Snapshots keep flowing and are dropped |
| `live_paused()` | `bool`. `False` without a core |
| `colormap_lut(name="viridis")` | `(256,)` uint32 LUT, each entry `R | G<<8 | B<<16 | A<<24`. Needs no core |
| `log_lines()` | The log lines of the core. `[]` without a core |

The ring counters are `taken`, `published`, `skipped` (rate limit), `ring_full`, `superseded` (a newer snapshot completed first), `appended` (series values), `allocs` and `recreated`.

The window render stats are `running`, `fps`, `frames`, `published`, `superseded`, `drawn`, `deferred_takes`, `errors`, `last_error`, `stats_unreadable`, `nonfinite_range`, `queued`, `max_frame_gap_ms`, `events_applied` and `heatmap_live_apply`. The sample lists are `latency_ms` (snapshot to first frame drawn with it), `tick_ms` and `frame_gap_ms`. `Window.stats(reset=True)` clears them.

## `caliper.live`

`from caliper import live` imports the module. `import caliper` does not import it.

| Call | Returns |
|---|---|
| `live.list_sessions(data_root=None)` | Live sessions under `<data_root>/live`, newest first, as dicts: `pid`, `alive`, `view_count`, `protocol`, `compatible`, `start_unix_ms`, `viewer_pid`, `session`, `script`, `path`. Stale sessions are reclaimed, not listed |
| `live.data_root()` | The default data root: `$CALIPER_DATA_DIR`, else the app-data folder of Caliper |
| `live.viewer_executable(path=None)` | `path`, else `$CALIPER_VIEWER`, else the viewer in the package. Raises `FileNotFoundError` if missing |

`caliper.session()` returns a `SessionTransport`:

| Member | Meaning |
|---|---|
| `session` | The token (`clp-<pid>-<6 hex>`), which `caliper-viewer --session` takes |
| `session_file` | The path of the session file. `""` once closed |
| `closed` | `True` after `close()` |
| `attached()` | `True` while a viewer heartbeat is younger than 1 s. Safe from any thread |
| `launch_viewer(exit_after=None, args=(), log=None)` | Spawn `caliper-viewer --session <token>` in its own process session. `log=None` discards its stderr, `"-"` inherits it, and a path writes it there. Returns the `Popen` |
| `viewer` | The `Popen` of the spawned viewer, or `None` |
| `viewer_alive()` | `True` if the spawned viewer runs. Without one, whether a viewer is attached |
| `wait_attached(timeout=10.0)` | Wait for a viewer heartbeat. `True` if attached |
| `wait_viewer(timeout=None)` | Wait until the viewer is gone. `True` if it is |
| `stats()` | Session, file, script, attached, viewer pid and heartbeat age, and per view the publish and claim counters |
| `close()` | Remove the session file and unlink the shared memory, never waiting for the viewer |

## Related Pages

- [Caliper LIB API](caliper-lib-api.md)
- [View Calls](view-calls.md)
- [Script Attachment](script-attachment.md)
- [Live Streaming](live-streaming.md)
