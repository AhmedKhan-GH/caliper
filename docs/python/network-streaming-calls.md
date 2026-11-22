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

<!-- The empty divs keep the anchors of the former hand-written entries, which the roadmap registry links to. -->
<div id="serve"></div>

::: caliper.serve

::: caliper._net.NetServer
    options:
      members: [control, session, stats, on_event, remove_event_handler, events, close]

#### Server Addresses

| Address | Serves | Token |
|---|---|---|
| `http://127.0.0.1:<port>/?token=<token>` | The viewer page: every view of the stream in a grid | Not needed. The page reads it from its own link |
| `http://127.0.0.1:<port>/<file>` | The modules of the page, the WebAssembly views and the fonts, from the package | Not needed. The files hold no data |
| `ws://127.0.0.1:<port>/v3/stream?token=<token>` | The stream, as the [Wire Protocol](../reference/wire-protocol.md) reference defines it | Required |

Every request must name the server in its `Host` header, `127.0.0.1:<port>` or `localhost:<port>`. The stream also refuses a wrong token. It refuses an `Origin` header other than a loopback page (http or https on 127.0.0.1, localhost or [::1], any port, such as a JupyterLab page), `null` or a VS Code webview. The files carry `Access-Control-Allow-Origin` for such a page, so a notebook page can load the modules of the widget. The stream carries snapshots, series, subscriptions, data view requests and viewer events, and never host commands.

#### Server Limits

The server keeps the last 1,048,576 values of each series whether or not a viewer watches it. It sends them first to a viewer that subscribes late.

The server limits the replies and other control messages that wait to be sent on one connection to 65,536 messages and 16 MiB. It keeps one PONG waiting, for the latest PING. Only a viewer that keeps sending requests while it reads nothing reaches the limit. The server then closes its connection with code 1008, as the hub does with a send queue that overflows. The training loop is not affected. These are limits of the server of Caliper LIB, not rules of the protocol.

<div id="connect"></div>

::: caliper.connect

::: caliper._net_native.HubConnection
    options:
      members: [state, connected, session, stats, wait_connected, close]

<div id="widget"></div>

::: caliper.widget

::: caliper._widget.CaliperWidget
    options:
      members: [on_select]
