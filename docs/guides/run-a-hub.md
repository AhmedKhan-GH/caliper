# Run a hub

The hub is a separate process, `caliper serve`, that holds the live sessions of several training scripts at once. Each script connects out to it as a producer, and viewers connect to it, list the sessions and attach to one. This page covers starting the hub, connecting scripts and viewers to it, what it keeps after a script ends, and its limits. The protocol it speaks is [wire protocol v3](../references/wire-protocol.md), and every option is in [Caliper CLI](../references/cli.md#caliper-serve).

## When to use a hub

`caliper.serve()` serves the views of one script from inside that script, and its viewers lose the stream when the script ends. Use the hub instead when:

- several scripts stream at the same time and you want them at one address;
- a viewer should move from one run to the next, for example after a notebook kernel restarts;
- you want to look at a run after its script has exited.

A viewer written for `caliper.serve()` works against the hub unchanged: when the hub holds exactly one live session, it attaches a new viewer to it.

## Start the hub

```sh
caliper serve
```

```text
viewers   http://127.0.0.1:7340/?token=<token>
producers ws://127.0.0.1:7340/v3/stream (address and token in <hub file>)
stop with Ctrl-C
```

The hub listens on 127.0.0.1 only, on port 7340 unless `--port` names another (`--port 0` takes any free port). It makes a random session token for each run. To keep the same viewer link across restarts, put a token in a file that only you can read and pass the file:

```sh
python -c "import secrets; print(secrets.token_urlsafe(24))" > ~/.caliper-hub-token
chmod 600 ~/.caliper-hub-token
caliper serve --token-file ~/.caliper-hub-token
```

The token is never given on the command line, because other users can read a process's arguments.

To open the stream in a browser, the hub must also serve the viewer page. Caliper LIB ships it in its `caliper/_web/` folder; pass that folder:

```sh
caliper serve --assets "$(python -c 'import caliper, os; print(os.path.join(os.path.dirname(caliper.__file__), "_web"))')"
```

Without `--assets`, the link opens a short note that says where the page is. The stream itself works either way.

## Connect a script

A script connects to `ws://127.0.0.1:<port>/v3/stream?token=<token>` as a producer, as [The producer connection to a hub](../references/wire-protocol.md#the-producer-connection-to-a-hub) describes. It sends HELLO with the role `producer` and its session identity: an id, the script's name, its pid and the time it started streaming. The hub answers WELCOME with the id it chose. It keeps the proposed id unless a live session of another process holds it, in which case it appends `-2`, `-3` and so on. The script then sends its list of views, and from then on the hub is its one viewer: it subscribes to the views that viewers ask for, at the rate they ask for, and forwards their events.

Scripts on the same machine find the hub through the **hub file**, which the hub writes when it starts and removes when it stops. It is `hub.json` in the app-data folder (`~/Library/Application Support/Caliper/hub.json` on macOS), or the path in `CALIPER_HUB_FILE`. Only its owner can read it, because it holds the token along with the port.

!!! note "Python specifics pending"
    The Python call that connects a script to a hub, `caliper.connect(url=None)`, is being built on the native sender. How it takes the address, where it looks for the token, and how it reconnects will be described here and in the [Caliper LIB](../references/python-api.md) reference when it lands.

A script that loses its connection and connects again with the same id, pid and start time **resumes** its session. The hub closes the old connection, if it is still open, with BYE `replaced`, keeps the session's viewers attached and subscribes again to every view they watch.

## Open a viewer

Any viewer of protocol v3 connects to the hub with the viewer link or the stream link that `caliper serve` prints:

- **A browser tab** opens the viewer link, `http://127.0.0.1:<port>/?token=<token>`, when the hub serves the page (`--assets`).
- **A notebook widget or a `<caliper-view>` element** takes the stream link, `ws://127.0.0.1:<port>/v3/stream?token=<token>`, as its source.
- **The Caliper app and caliper-viewer** take either link, as [Attach to a stream by its link](attach-to-a-running-script.md#attach-to-a-stream-by-its-link) describes. With several sessions, add `&session=<session id>` to choose one:

```sh
caliper-viewer --headless --url "http://127.0.0.1:7340/?token=$TOKEN&session=clp-41222-9f03ab" --exit-after 5 --stats
```

## Choose a session

Right after a viewer connects, the hub sends it the list of sessions, and sends the list again whenever it changes. Each entry has the session's id, script, pid, start time, number of views and state, `live` or `ended`.

A viewer that has not chosen a session follows the hub's default: whenever it is not attached to a live session and the hub holds exactly one live session, the hub attaches it to that session. So a viewer that stays open moves to the next run when one script ends and another starts. A viewer that attaches to a session by its id chooses for itself from then on, and attaching to another session ends its subscriptions to the previous one.

A new script that proposes the id of an ended session of another process takes that id over: the ended session is forgotten, and its viewers move to the new session. This is what happens when a notebook kernel restarts and the notebook streams under the same id.

## When a script ends

When a script disconnects, its session stays in the list with the state `ended`. Viewers that are attached keep it, and a viewer can still attach to it later and subscribe. A subscription to a data view that a viewer watched while the script ran receives its last snapshot at once, and a subscription to a series view receives the values the hub retained. A data view that nobody watched cannot be computed any more, since only the script could compute it, so the hub answers with the data view of that view that was watched last, and the reply names it. A view that nobody watched at all is answered with the error `unknown_view`.

An ended session is kept for 24 hours, and at most 32 ended sessions are kept, the oldest going first. When either limit forgets a session, its viewers are detached, or attached to the one live session if there is one.

## What the hub keeps

The hub keeps data so that a viewer that arrives late sees something at once, and so that a session can be looked at after its script ended. It keeps:

- **The latest snapshot of every producer subscription.** Viewers that ask for the same view under equal resolved data views share one subscription at the script and one stored snapshot. When the last viewer of a data view leaves, the hub ends the subscription at the script, so the script does no more work for it, but keeps the last snapshot. At most 64 such idle data views are kept per session, the least recently used going first.
- **The values of every series view.** The hub subscribes to each series view as soon as the script lists it, whether a viewer watches it or not, so the whole curve is there for a late viewer. It keeps the newest 1,048,576 values of each series view (4 MiB).
- **The controls and selections of every session.** The latest CONTROLS of each session, which holds the run state, the parameters and the actions its script registered, and the latest SELECTION of each item space, at most 16 spaces per session. They are kept after the session ends, so a viewer still sees how the run stood. See [Controls and selections](#controls-and-selections).
- **Nothing else.** It decodes the metadata of each message to route it, never copies a payload per viewer, and never interprets the data beyond what forwarding, the stored snapshots and series backfill need.

All of this together is bounded by a budget of 1 GiB. Over it, the hub forgets the oldest ended sessions first, then the stored snapshots of the data views nobody watches, least recently used first. If live sessions alone fill the budget, series stop growing and keep their newest values, at least 4,096 per series view.

Sending takes memory too, but little. On each viewer connection the hub hands one data message at a time to the socket, and keeps at most one pending snapshot per subscription, which is the stored snapshot itself rather than a copy. A connection whose send queue passes 160 MiB, which only a viewer that stopped reading while the hub sent it a great many control messages could reach, is closed with code 1008.

## How the hub forwards

The hub applies the protocol's [flow control](../references/wire-protocol.md#flow-control) on every viewer connection, so one slow viewer slows only itself:

- A newer snapshot replaces a pending one, and a viewer that falls behind receives the newest snapshot when it catches up, never a backlog.
- Series values are sent from the hub's retained values by index. They are coalesced into as few messages as possible and are not dropped while the hub retains them.
- Control messages go first, series before snapshots, and the subscriptions of one connection take turns.
- Each viewer subscription receives at most its `max_rate` messages per second on average, in bursts of at most two, so that a script that sends at exactly that rate, with some jitter in its timing, passes through without waiting.
- The subscription at the script runs at the largest `max_rate` among the viewers that share it.
- The hub keeps reading every script whatever its viewers do, so a slow viewer never holds a script back.

On an Apple silicon Mac, a 512 × 512 heatmap at 60 Hz reaches four viewers through the hub with a median latency of about 1.4 ms and a 99th percentile under 3 ms, from the script's send to the viewer's receive, at 240 MiB/s in total. The hub's tests measure it with `caliper_tests -tc="net/hub: relay*" --no-skip`.

## Controls and selections

From [protocol 3.1](../references/wire-protocol.md#controls-through-a-hub), a viewer can pause, resume and step a script's training loop, change the parameters the script registered and run its actions, and a selection in one view highlights the same items in the session's other views. The hub announces both capabilities, `controls` and `selection`, to viewers and to scripts.

**Controls are off unless you allow them.** Without `--allow-control` the hub forwards no control: it tells every viewer in its WELCOME that it may not control, and answers every CONTROL itself with the error `forbidden`. Viewers still show the run state, the parameters and the actions, without letting them be changed. Start the hub with the option to forward controls:

```sh
caliper serve --allow-control
```

Then the hub forwards a viewer's CONTROL only to the script of the session that viewer is attached to. It numbers the request itself at the script, so two viewers that use the same request id never get each other's answers, and it routes each CONTROL_REPLY back to the viewer that asked, the two answers of a queued action included. It answers a CONTROL itself with `not_attached` when the viewer is not attached, and with `unsupported` when the session has ended or its script does not take controls. When a script disconnects with requests still waiting, the hub answers them `unsupported`, and a request the script never answers is answered `timeout` after a minute, or after the protocol's ten minutes for a queued action. At most 64 requests of one viewer, and 1,024 of one session, wait at once; one more is answered `limit`.

**Selections need no permission.** Selecting is not a control: a viewer's select events reach the script whether or not the hub allows control, and the script's SELECTION reaches every viewer of the session that announced `selection`. A viewer that attaches late gets the session's latest CONTROLS and every item space's selection that is not empty right after its VIEWS. A viewer that did not announce a capability gets none of its messages, and VIEWS reach it without the item spaces; a script that did not announce `selection` gets select events without them. So viewers and scripts of protocol 3.0 work with this hub unchanged.

An ended session keeps its last controls and selections, which a viewer shows with the controls disabled, and the hub answers its controls `unsupported`.

## Limits

| Limit | Value | Beyond it |
|---|---|---|
| Viewer connections | 256 | ERROR `limit`, then close code 4005 |
| Live sessions | 64 | ERROR `limit`, then close code 4005 |
| TCP connections, plain HTTP included | 512 | Closed at once |
| Subscriptions on one viewer connection | 256 | VIEW_REPLY with the error `limit` |
| Subscriptions of the hub at one script | 256 | VIEW_REPLY with the error `limit` |
| Ended sessions | 32, for 24 hours each | The oldest is forgotten |
| Idle data views per session | 64 | The least recently used is forgotten |
| Series values per series view | 1,048,576 | The oldest values go |
| Stored snapshots, series values, controls and selections | 1 GiB | See [What the hub keeps](#what-the-hub-keeps) |
| Item spaces whose selection is kept, per session | 16 | A new space's SELECTION is dropped |
| Controls waiting for a script, per viewer connection | 64 | CONTROL_REPLY with the error `limit` |
| Controls waiting for a script, per session | 1,024 | CONTROL_REPLY with the error `limit` |
| Send queue per connection | 160 MiB | Closed with code 1008 |

The timers are the protocol's: a connection that sends no HELLO within 10 seconds is closed with 4003, a peer silent for 10 seconds gets PING, and one silent for 30 seconds is closed with 4004.

## Security

The hub follows the protocol's [security](../references/wire-protocol.md#security) rules. It listens on 127.0.0.1 only. Every request must name 127.0.0.1 or localhost with the hub's port in its Host header, and a stream connection must come from no origin, a loopback origin or a VS Code webview, and must carry the session token. The viewer page and its files need no token: they are public code and hold no data.

The hub forwards data, viewer events and, only when it was started with `--allow-control`, controls that name a parameter or an action the script registered, to the script of the viewer's own session. It never runs anything a message names, and the stream never carries host commands. Whoever holds the session token of a hub started with `--allow-control` can pause the scripts connected to it, change their registered parameters and run their registered actions, so give that option only to a hub whose token you keep to yourself. The token appears in the viewer link, so it can reach a browser's history; the hub file that holds it can be read by its owner only.

## Stop the hub

Press Ctrl-C, or send SIGTERM. The hub sends BYE `shutdown` on every connection, closes them with code 1001, removes the hub file and exits with 0. Scripts connected to it keep training; they lose only the connection.
