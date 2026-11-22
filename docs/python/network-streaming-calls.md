---
kind: reference
figure: none
---

# Network Streaming Calls

The calls `caliper.serve()`, `caliper.connect()` and `caliper.widget()` stream the live views of a script to browsers, notebooks and a hub.

## Network Streaming

`caliper.serve()` serves the live views of the script over a WebSocket on the loopback interface. `caliper.widget()` shows one of them in a notebook cell. `caliper.connect()` streams the same views to a `caliper serve` hub instead, or as well. Data views let each viewer choose which part of a tensor travels. See [Data View Calls](data-view-calls.md).

[Network Streaming](network-streaming.md) explains the design, and the [Wire Protocol](../reference/wire-protocol.md) reference defines the messages. [Watch a training run in a notebook](notebook-training-runs.md) uses all three calls. `websockets` is a dependency of the package. The widget needs the extra `caliper[notebook]`, which adds anywidget.

The sender of the server is native when `caliper._netio` is built. It is a C++ thread that encodes and sends and never takes the GIL. Run `python scripts/build_netio.py` with the Python of the script to build it and copy it into the installed package. Without it, `serve()` uses a sender written in Python and says so once on stderr, and `connect()` raises `RuntimeError`. `CALIPER_NETIO=0` chooses the sender written in Python.

The viewer page and the widget modules always come with the package. The WebAssembly build of the Caliper views is optional. A wheel built in CI includes it. In a checkout, `python scripts/build_wasm.py` builds it and `python scripts/fetch_wasm.py` downloads the CI build. Without it, every view draws with the views written in TypeScript. Then `serve()` says so once on stderr and names those two commands. `CALIPER_WEB_DIR` points the package at another folder of these files.

### `serve`

```python
server = caliper.serve(port=0, *, control=True)
```

Start the stream server of this process, print its link, `http://127.0.0.1:<port>/?token=<token>`, and return the server, a `caliper._net.NetServer`. A second call while the server runs returns the same server.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `port` | `int` | 0 | TCP port on 127.0.0.1. 0 lets the system choose a free port |
| `control` | `bool` | `True` | Whether viewers may pause the run, change parameters and run actions ([Controls](run-control-calls.md#controls)). `False` shows them read-only |

The server listens on 127.0.0.1 only and makes a new random session token. While it listens, it writes the stream file `streams/<pid>.json` in the app-data folder of Caliper (`$CALIPER_DATA_DIR` when set, see [Environment](../reference/folders-and-environment.md)). The file has mode 0600 in a folder of mode 0700. It holds the pid, the machine name, the session id, the script name, the start time, the port, the stream link with its token, `control` and the view names. The server rewrites the file when the views change.

The Caliper app lists the stream from this file in `F4 Attach to process`, so it attaches without a pasted link. `close()` removes the file, and the next listing of the app removes the file of a killed script. A file that cannot be written leaves the server running, and the printed link still attaches.

The server is one more route. Every view call is offered to it. A view takes a snapshot for it only while a viewer subscribes to that view, once for each data view that its subscriptions ask for. The snapshots are computed on the device and copied into pinned host memory without waiting, so the call still returns at once.

| Address | Serves | Token |
|---|---|---|
| `http://127.0.0.1:<port>/?token=<token>` | The viewer page: every view of the stream in a grid | Not needed. The page reads it from its own link |
| `http://127.0.0.1:<port>/<file>` | The modules of the page, the WebAssembly views and the fonts, from the package | Not needed. The files hold no data |
| `ws://127.0.0.1:<port>/v3/stream?token=<token>` | The stream, as the [Wire Protocol](../reference/wire-protocol.md) reference defines it | Required |

Every request must name the server in its `Host` header, `127.0.0.1:<port>` or `localhost:<port>`. The stream also refuses a wrong token. It refuses an `Origin` header other than a loopback page (http or https on 127.0.0.1, localhost or [::1], any port, such as a JupyterLab page), `null` or a VS Code webview. The files carry `Access-Control-Allow-Origin` for such a page, so a notebook page can load the modules of the widget. The stream carries snapshots, series, subscriptions, data view requests and viewer events, and never host commands.

The server keeps the last 1,048,576 values of each series whether or not a viewer watches it. It sends them first to a viewer that subscribes late.

The server limits the replies and other control messages that wait to be sent on one connection to 65,536 messages and 16 MiB. It keeps one PONG waiting, for the latest PING. Only a viewer that keeps sending requests while it reads nothing reaches the limit. The server then closes its connection with code 1008, as the hub does with a send queue that overflows. The training loop is not affected. These are limits of the server of Caliper LIB, not rules of the protocol.

| Member | Meaning |
|---|---|
| `url` | The link of the viewer page, with the token |
| `stream_url` | The `ws://` URL of the stream, with the token |
| `port` | The port the server listens on |
| `token` | The session token |
| `session` | The session id that viewers see, `clp-<pid>-<6 hex>` |
| `control` | Whether viewers may control the run, as `serve()` was told |
| `closed` | `True` once the server is stopped |
| `stats()` | A dict: `url`, `port`, `session`, `closed`, `attached` (connected viewers), `connections`, `views` (per view its `id`, `kind`, `groups` of subscriptions with one data view, and `seq`), `series` (values retained per series) and counters |
| `on_event(fn)` | Call `fn(event)` for every viewer event from now on. Returns `fn`, so it also works as a decorator |
| `remove_event_handler(fn)` | Stop calls to `fn`. `True` if it was registered |
| `events()` | The events received since the last call, oldest first. It clears them and never blocks |
| `close()` | Deliver what is pending (for about a second), end every connection and stop the server |

An event is a dict with `view`, `type` (`"select"`, `"pick"` or `"hover"`), `detail`, `sub`, `seq`, `data_view`, `connection` and `time`. The `detail` is in source indices, as the [Wire Protocol](../reference/data-messages.md#events) reference lists for each kind. Events are data for Python code to read, and nothing in them runs as a command.

The `on_event` functions run on a thread named `caliper.events`, never on the training thread. An exception in one becomes a `RuntimeWarning` and does not stop the others. The server keeps at most 256 unread events for `events()` and drops the oldest.

`close()` may be called more than once. It also runs at `caliper.close()` and at interpreter exit, and the server works as a context manager that closes it.

### `connect`

```python
hub = caliper.connect(url=None)
```

Stream every view of this process, as a producer, to a `caliper serve` hub on this machine. The call returns the connection, a `caliper._net_native.HubConnection`. [Hub Setup](hub-setup.md) explains the hub.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `url` | `str` or `None` | `None` | The stream URL of the hub with its token, `ws://127.0.0.1:<port>/v3/stream?token=<token>`, or the viewer link that the hub prints. `None` reads `CALIPER_HUB_URL`, then the hub file: `CALIPER_HUB_FILE`, or `hub.json` in the app-data folder (`~/Library/Application Support/Caliper/hub.json` on macOS) |

`connect()` returns at once and never blocks the training loop, because the native sender connects in the background. When the hub cannot be reached or the connection drops, the sender tries again after 0.5 seconds and doubles the wait up to 30 seconds. It proposes the session id that the hub assigned, so the hub resumes the session. It stops when the hub ends the connection with BYE `replaced`.

The views reach the hub through the same calls as for `serve()`. A view takes a snapshot for the hub only while a viewer of the hub watches it. Caliper LIB keeps one producer per process, so `serve()` and `connect()` in one script share it, and every view call reaches both.

`connect()` raises `ValueError` for a URL that has no token or names a host other than 127.0.0.1 or localhost. It raises `RuntimeError` when no URL is given and there is no hub file, or when the native sender is not built.

| Member | Meaning |
|---|---|
| `url` | The stream URL of the hub, without the token |
| `session` | The session id that the hub assigned. Until the hub answers, it is the proposed `clp-<pid>-<6 hex>` |
| `state` | `connecting`, `connected`, `waiting` (to try again), `replaced` (another connection of this session took over) or `closed` |
| `connected` | `True` while the connection is open |
| `stats()` | A dict with `state`, `connected`, `session`, `error` (the last failure, or empty), `attempts` and `connects` |
| `wait_connected(timeout=10.0)` | Wait, with polls, until the connection is open. `True` if it is. For tests and scripts, not for the training loop |
| `close()` | End the connection to the hub. A running `serve()` keeps serving |

`caliper.close()` ends the connection as well.

### `widget`

```python
w = caliper.widget(name, kind=None, *, height=320)
```

An anywidget widget that shows the live view `name` in a notebook cell. It reads the WebSocket stream of the server directly, not the message channel of the notebook, so it keeps updating while a cell runs. It is an ipywidgets widget, so `HBox` and `VBox` lay several out together.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | | The view to show. The widget may be created before the first call of the view |
| `kind` | `str` or `None` | `None` | `"series"`, `"heatmap"`, `"image"`, `"points"` or `"histogram"`. `None` takes the kind from the stream |
| `height` | `int` | 320 | Height of the whole widget, header included, in CSS pixels |

When no server runs, `widget()` starts one with `serve()`. An unknown kind raises `ValueError`. Without anywidget installed, the call raises `ImportError` and names the extra.

The widget has these traits, synced through anywidget:

| Trait | Type | Meaning |
|---|---|---|
| `src` | `str` | The `ws://` URL of the stream, with the token |
| `view` | `str` | The name of the view |
| `kind` | `str` or `None` | The kind of the live view |
| `height` | `int` | Height in CSS pixels |
| `data_view` | `dict` | The data view of the own subscription of the widget, in both directions. The browser writes the data view that the user picked. Setting it from Python makes the widget request that data view |
| `selection` | `dict` | Written by the browser: the latest event it reported, as `{"view", "type", "detail"}` |

The `detail` of an event is in source indices. For a `select` in a points view it holds `indices`, the positions in the points that were shown, and `row_ids`, their source indices. The [Wire Protocol](../reference/wire-protocol.md) reference lists the detail of every kind. `w.on_select(callback)` calls `callback(selection)` each time the browser reports a new selection, and `w.on_select(callback, remove=True)` stops it.

Traits travel over the message channel of the notebook, which the kernel reads only while no cell runs. A selection made during a long cell reaches `selection` when the cell ends. A `data_view` set from Python reaches the widget at once. A data view chosen in the picker of the widget goes to the producer over the stream and takes effect during the cell.
