# Attach to a running script

A script that calls `caliper.show` (or any other view call) opens a **live
session** on its first call. A viewer can attach to it at any point while the
script runs, detach, and attach again; the training loop never waits for it.
This page covers the viewer side: finding a session, attaching from the
Caliper app (the usual way) or with the standalone `caliper-viewer`, what each
shows, detaching, and what happens when either process goes away. The design
is in [Live streaming](../explanations/live-streaming.md).

Everything here is macOS / Metal, local, same user.

## Find the session

Each session is a file `<data_root>/live/<pid>.json`, written when the session
opens and removed when it closes. `data_root` is `$CALIPER_DATA_DIR` when set,
otherwise Caliper's app-data folder:

```sh
ls ~/Library/Application\ Support/Caliper/live/
```

The file is one JSON object with the protocol version, the pid and its start
time, the session token (`clp-<pid>-<6 hex>` by default), the script name, the
start time, the view names and the shared-memory segment names. If the script
sets `CALIPER_DATA_DIR`, start the viewer with the same root (`--data-dir`, or
the same `CALIPER_DATA_DIR` for the Caliper app).

From inside the script, `caliper.session()` returns the open session:
`.session` is the token and `.session_file` the file's path. From any other
Python process:

```python
from caliper import live
for s in live.list_sessions():
    print(s["pid"], s["script"], s["session"], s["view_count"], s["viewer_pid"])
```

Each entry also has `alive`, `protocol`, `compatible`, `start_unix_ms` and
`path`. Listing reclaims stale sessions as a side effect (see below).

## Attach from the Caliper app

On the landing screen, press F4 or click `F4 Attach to process` in the Projects
pane. Projects lists the sessions, newest first, relisted every second, with
the script, PID, start time (clock time and how long ago), state and view
names of each. The states and which ones attach are those of the table below;
a row that cannot attach is faint, and the `⏎ Attach` button under the table
is disabled with the reason beside it. Select a row with ↑ / ↓ and press Enter,
double-click it, or click `⏎ Attach`; Esc or `esc Back` returns to the ways in.

The session's live page opens:

| Part | Shows |
|---|---|
| Bar | `esc Home`, the script's name, and its state: a filled, breathing diamond and `live` while it runs, `waiting for the session to start`, or an outlined diamond and `SESSION ENDED` once it has gone |
| Run bar | For a stream only, under the bar: the script's run state, Pause or Resume, Step, its parameters and its actions (see [Control the run and share a selection](#control-the-run-and-share-a-selection)) |
| Views | One window per view on a grid that fills the page and lays itself out again when the page narrows, each with a header of its name, a summary, its tools and a status dot (see [What you see](../tutorials/live-pytorch.md#what-you-see)); a window you move, resize or dock keeps its place, and that layout is kept in `imgui.ini` |
| Status bar | The backend and GPU, then the script's pid, the number of views, and the frame rate and step-to-screen latency over the last second |

The window title reads `Caliper - <script> (pid <pid>)`, with
` - session ended` appended once the script has gone. The page draws at up to
60 fps, as `caliper-viewer` does, whatever the display's refresh rate. Activity on the
landing logs `attached`, `session ended` and `detached` with the script and
its pid, and `attach failed` with the reason when a session refuses (another
viewer took it, it closed, or the renderer is not Metal).

For automation, `CALIPER_AUTOATTACH=<pid>` attaches to that process's session
as soon as the landing lists it, and `CALIPER_HOME_AFTER=<seconds>` detaches
that long after the page opened:

```sh
CALIPER_AUTOATTACH=41222 CALIPER_WINDOW_SIZE=1440x900 cmake-build-release/caliper
```

## Attach with caliper-viewer

`caliper-viewer` ships in the wheel as `caliper/bin/caliper-viewer`
(`live.viewer_executable()` prints the path; `$CALIPER_VIEWER` overrides it for
`auto_launch`).

**From the session list.** Without `--session` the viewer opens a "Live
sessions" window, refreshed every second, with one row per session: script,
PID, start time, view count, state and an **Attach** button.

| State | Meaning | Attach |
|---|---|---|
| running | The script is alive and no viewer is attached | Enabled |
| starting | The session file exists but the shared memory is not initialised yet | Enabled; the viewer waits for it |
| viewed by `<pid>` | Another viewer holds the session | Disabled |
| closing | The script is closing the session | Disabled |
| protocol v`N` | Written by an incompatible Caliper version | Disabled |

**By token.** `--session <token>` attaches at once, tolerating a session that
is still starting. If no such session exists the viewer exits with status 2
before opening a window.

```sh
caliper-viewer --session clp-41222-9f03ab
```

**By pid.** The viewer has no pid flag. Pick the row with that PID in the list,
or read the token from the session file:

```sh
caliper-viewer --session "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["session"])' \
    ~/Library/Application\ Support/Caliper/live/41222.json)"
```

An embed host attaches by pid directly with `caliper_core_live_attach_pid`
(below).

**From the script.** `caliper.session(auto_launch=True)` spawns
`caliper-viewer --session <token>` in its own process session, so a Ctrl-C in
the script's terminal does not close it and it outlives the script.

| Flag | Effect |
|---|---|
| `--session <token>` | Attach to this session at once |
| `--url <link>` | Attach to a network stream by its link (see [Attach to a stream by its link](#attach-to-a-stream-by-its-link)) |
| `--data-dir <root>` | Session-file root (default `$CALIPER_DATA_DIR`, else the app-data folder) |
| `--exit-after <s>` | Close the window after `s` seconds |
| `--size <w>x<h>` | Window size in points (default 1280x800) |
| `--stats` | Print a stats line to stderr every 2 s: fps, updates, step-to-screen median/p95/max, skipped and client-dropped counts, each view's renderer path, RSS and footprint |
| `--latency-log <file>` | Write every latency sample (`t_ns lat_ms` per line); implies `--stats` |
| `--headless` | No window, an offscreen canvas (CI, tests). Needs `--session` or `--url`, and prints one `received` line per view when it exits |
| `--dump-dir <dir>` | With `--headless`: write each view's pixels or values to `<dir>` whenever its sequence number changes |

## Attach to a stream by its link

A script that streams with `caliper.serve()` sends its views over a WebSocket on
127.0.0.1 ([wire protocol v3](../references/wire-protocol.md)) instead of shared
memory, so it has no session file and does not appear in the session lists
above. Attach to it with the link it prints. Two forms are accepted, with
`localhost` in place of `127.0.0.1`:

- the page link, `http://127.0.0.1:<port>/?token=<token>`;
- the stream link, `ws://127.0.0.1:<port>/v3/stream?token=<token>`.

A server that holds several sessions, such as the `caliper serve` hub, needs
the one to show: add `&session=<session id>` to the link. With one session the
server attaches the viewer to it.

The token in the link is the stream's only secret. The viewer never prints it,
and messages show the link without its query. Only links to 127.0.0.1 or
`localhost` are accepted.

**In the Caliper app**, press F4 and paste the link into the field under the
session list, then press Enter or click `Attach link`. A refusal stays on the
landing, and Activity says why: nothing listens on that port, or the server
refused the token (HTTP 403). To open the app on the stream directly, give
the link on the command line:

```sh
caliper --url "http://127.0.0.1:8765/?token=$TOKEN"
```

**With caliper-viewer**, pass `--url`:

```sh
caliper-viewer --url "http://127.0.0.1:8765/?token=$TOKEN"
```

Without a window, the viewer runs for a while and prints what it received,
which is how a script that streams is checked in a terminal or CI:

```sh
caliper-viewer --headless --url "ws://127.0.0.1:8765/v3/stream?token=$TOKEN" --exit-after 5 --stats
```

```text
[caliper-viewer] received session=clp-41222-9f03ab script=train.py pid=41222 alive=1 views=3 updates=381
[caliper-viewer] received view=weights kind=heatmap updates=127 seq=135 series=0
[caliper-viewer] received view=loss kind=series updates=127 seq=135 series=135
[caliper-viewer] received view=acts kind=histogram updates=127 seq=135 series=0
```

`updates` counts the snapshots the viewer drew, `seq` is the last one's number,
and `series` is the number of values a series holds. The exit status is 2 when
the attach fails.

The viewer subscribes to every view of the session with the view's default
data view and draws it like a shared-memory view. When the connection drops,
the views keep their last pixels, the status says the session ended, and the
viewer reconnects; a series continues from the values the producer retained.
When the script ends, the views keep their last pixels.

## Choose what a view shows

A view of a stream may offer more than it draws by default: several sources,
such as the layers of a model, and dimensions beyond the ones its kind
draws, such as the heads of an attention map
([Data views](../references/wire-protocol.md#data-views)). In the app and in
`caliper-viewer` such a view has a picker under its header, as in the
browser: a menu of its sources when it has several, and a menu per
dimension that keeps the dimension (`all`), reduces it (`mean`, `sum`,
`max`, `min` or `norm`), or picks one index (a field past 256 positions). A
heatmap that keeps three dimensions shows the first as a mosaic of tiles. A
view with nothing to choose has no picker.

A choice goes to the script at once; the script computes the new data view
on its device and only the result travels. The menus then show the data view
the script resolved, so a choice it refused or clamped returns to what it
shows. A shared-memory session offers no choices.

## Control the run and share a selection

A script on protocol 3.1 can take controls from its viewers: it calls a
checkpoint once per training step (`caliper.step()`), and it may register
parameters and actions and declare which axes of its views index the same
items, such as the samples of a test set ([Controls](../references/wire-protocol.md#controls)
and [Item spaces](../references/wire-protocol.md#item-spaces)). The app's live
page and `caliper-viewer` show these for a stream. A shared-memory session has
no channel back to the script, so its page has no run bar.

The run bar sits under the page bar, above the views:

| Part | Shows and does |
|---|---|
| Run state | `running`, `pausing`, `paused at step N`, `stepping, N left` or `no checkpoint yet`. Only a paused run names a step, the one at which the loop waits: the script sends its run state when the state changes, not at every step, so the bar does not know the current step of a running loop |
| Pause, Resume | Pause holds the training loop at its next checkpoint; Resume releases it |
| Step | Runs one step and pauses again |
| Parameters | One control per registered parameter, by its type: a slider for a number or an integer with a range (on a log scale when the script asks for one) with a field beside it that shows and takes the exact value, a field for one without a range, a box for a boolean, a menu for a choice |
| Actions | One button per registered action, then `queued` or `running` while it waits or runs, and its result for 15 s after it finishes, or `FAILED:` and the reason |
| Right edge | The last refused request and why, for 10 s, or why the controls are off |

A value you set shows at once, marked `pending`, until the script confirms
it: the script reads a new value at its next checkpoint, and the mark stays
until a CONTROLS sent after the script's reply carries the value. A value out
of range comes back clamped, and a refused value returns to the script's own
with the reason at the right edge. While a slider is dragged, at most 30
values a second go to the script, and the value where the drag stops always
goes last.

**Who may control.** The run bar shows the run state, the parameters and the
actions to every viewer, but changes them only when all of these hold;
otherwise every control is off and the bar says why:

- the server lets this viewer control: `caliper.serve()` of a script does
  unless it was told to refuse controls, and a hub does only when it was
  started with `--allow-control` (`caliper serve --allow-control`, or the app's
  `--api` stream with `--allow-control`; see [Run a hub](run-a-hub.md#controls-and-selections));
- the script takes controls (it announced them);
- the session is live: the controls of an ended session stay visible and off.

Before the script's first checkpoint, Pause, Step and the actions are off
with `the script has called no checkpoint yet`; a parameter can already be
set, and the script reads it at once. The stream carries only these requests,
never a command to the app or the machine.

**Selecting.** Shift-drag across a view to select: rows of a heatmap or an
image, a range of steps of a series, points inside the lasso of an embedding.
When the selected axis indexes an item space, the selection is sent to the
script with the item ids, and the script shares it with every viewer of the
session. Every view whose axis indexes the same space then shows it, as the
browser views do: the selected points over the dimmed rest with their count,
the selected rows or columns of a heatmap or an image outlined in bronze (in
every cell of a mosaic), and markers on the selected steps of a series. So
selecting units in one weight matrix marks the same units in the next and in
an embedding of them. A selection is not a control: every viewer may select,
with or without `--allow-control`.

**Details.** When the script provides details for the space, such as a few
fields of each selected sample, a Selection pane opens at the right of the
page with one row per selected item, for the first 32, under the item space's
name. It closes when the selection is cleared. When the details failed, the
pane says why.

In `caliper-viewer` the same run bar and details are in the status window, in
ImGui's own style.

The page logs what changes to standard error, one `[live]` line each, for
example `[live] run paused at step 165`, `[live] param lr = 0.02`,
`[live] save: saved at step 165` or `[live] controls off: the server does not
let this viewer control (a hub needs --allow-control)`.

## What caliper-viewer shows

The window title reads `caliper: <script> (<pid>)` while attached, with
` - starting` or ` - session ended` appended when that applies. One window per
view shows the live data (see [What you see](../tutorials/live-pytorch.md#what-you-see)),
and a status window in the bottom-right corner shows:

| Line | Content |
|---|---|
| Header | Script, pid and session token |
| State | "running", "waiting for the session to start", or "SESSION ENDED: the script is gone; showing the last frames received" |
| Frames | Frame count, fps, number of views |
| Counters | Updates applied, snapshots skipped (a newer one arrived first), refused updates |
| Latency | Step-to-screen median and p95 over the recent samples |
| Per view | Name, kind, last sequence number, latency, `dropped@client` (snapshots the script dropped for want of a free slot), and an error if the last update was refused |
| Controls | For a stream: the run bar and the selection's details (see [Control the run and share a selection](#control-the-run-and-share-a-selection)) |
| Buttons | **Detach** and **Pause** / **Resume** |

Pause here freezes the display only; snapshots keep arriving and are dropped,
so training is not slowed. To hold the training loop itself, use Pause in the
run bar.

## Detach

In the Caliper app, `esc Home` (or Esc, when no field or popup takes it) drops
the views, releases the claims on the shared slots and clears the heartbeat,
then returns to the session list. In `caliper-viewer`, **Detach** in the status
window does the same. The script stops taking snapshots at once and keeps
training, and you can attach again from the list. Closing either window
detaches the same way. A series
keeps its values in the session while nobody is attached, so after a re-attach
the loss curve shows the values from before the detach, followed by the new
ones; values offered while no viewer was attached are not recorded.

## When the script ends

A clean exit, `caliper.close()` or interpreter shutdown closes the session:
the session file is removed and the shared memory unlinked, without waiting for
the viewer. The viewer keeps its last frames on screen, adds "session ended" to
the title and shows "SESSION ENDED" in the status window until you detach or
close it. The Caliper app does the same on its live page: `SESSION ENDED` in
the bar, `session ended` in the status bar and the title, and a `session ended`
row in Activity; `esc Home` then returns to the list.

A killed script (SIGKILL, a crash) looks the same in the viewer and in the app;
in the measurement, the viewer kept drawing its last frames at 60 fps after the
trainer was SIGKILLed. The session file and shared memory are left behind until
the next listing reclaims them, which on the app's landing is at most a second
after you return to it.

## When the viewer dies

The script's `attached()` check is "a viewer heartbeat younger than 1 s". When
the viewer exits or crashes, the heartbeat stops and within a second
`attached()` turns false and snapshots stop. Nothing in the script blocks, and
another viewer can attach once the old viewer's process is gone or its
heartbeat is stale.

## Cost to the script when nobody is attached

With no viewer, a call returns after one atomic load and a clock read: no
snapshot, no slot, no GPU work. Measured (torch 2.14.1,
100,000 calls per row, each timed singly):

| Call, nothing attached | Median | p99 |
|---|---|---|
| `SessionTransport.attached()` | 0.041 µs | 0.084 µs |
| `caliper.show(W)`, 1024x1024 MPS | 0.167 µs | 0.291 µs |
| `caliper.show(W)`, 256x256 CPU | 0.167 µs | 0.292 µs |
| `caliper.series(1.0)` | 0.125 µs | 0.209 µs |

In a training loop (MPS, 1024x1024 weight shown every step, 10 s after a 1 s
warm-up), the unattached script ran at 100.1 % and 103.1 % of the no-Caliper
baseline in two repeats.

## One viewer per session

The viewer's claims on the shared slots are single-reader words, so a session
takes one viewer at a time. A second attach is refused with "session is being
viewed by pid `<pid>` (one viewer per session)", and the session list shows the
row as "viewed by `<pid>`" with **Attach** disabled. Detach the first viewer, or
let it exit, to attach from another.

## Stale session cleanup

Every listing checks each session file's pid and start time. A file whose
process is gone, or whose pid now belongs to a different process, is stale: the
shared-memory names under that session's own `/<token>-` prefix are unlinked,
the file is removed, and the session is not listed. Any listing does this: the
Caliper app's landing (every second while it shows), the viewer's session list
(every second), `caliper.live.list_sessions()` and
`caliper_live_list_sessions`. A killed script therefore leaks nothing past the
next listing.

## From an embed host

The Caliper app's "Attach to process" drives the same live core as
`caliper-viewer`. The app owns its window and renderer, so it attaches through
a hosted core that borrows them (`caliper_core_create_hosted`,
`src/host/embed/embed_core.h`; the session list and the attachment are in
`src/host/live/live_attach.h`), and the session-row rules are shared with the
viewer's list. An embed host uses the C calls of `include/caliper/live.h`.
Listing works from any thread; the `caliper_core_live_*` calls need an embed
core with an attached canvas and run on its frame thread:

```c
#include <caliper/live.h>

/* List sessions (any thread). struct_size must be set on every entry. */
CaliperLiveSessionInfo s[16];
for (int i = 0; i < 16; ++i) s[i].struct_size = sizeof s[i];
int n = caliper_live_list_sessions(s, 16);      /* newest first; -1 + caliper_live_last_error() */

/* Attach: by token, by pid, by a stream link, or show the built-in session list. */
caliper_core_live_attach(core, s[0].session);
caliper_core_live_attach_pid(core, pid, NULL);  /* NULL = the default data root */
caliper_core_live_attach_url(core, "ws://127.0.0.1:8765/v3/stream?token=...");
caliper_core_live_picker(core, 1, NULL);        /* list with Attach buttons; Detach returns to it */

/* Every frame, before caliper_core_frame. */
caliper_core_live_poll(core);                   /* claim newest snapshots, write the heartbeat */
caliper_core_frame(core);

caliper_core_live_detach(core);                 /* release claims, clear the heartbeat */
```

| Call | Purpose |
|---|---|
| `caliper_live_data_root()` | The resolved default data root |
| `caliper_live_list_sessions(out, cap)` / `_in(root, out, cap)` | List sessions newest first, reclaiming stale ones; returns the count |
| `caliper_core_live_attach(core, token)` | Attach by token; refuses an unknown token, another protocol version and a session another live viewer holds |
| `caliper_core_live_attach_pid(core, pid, root)` | Attach to the session of process `pid` |
| `caliper_core_live_attach_url(core, link)` | Attach to a stream by its link; waits up to 2 s for the server and fails with the reason when nothing listens or the server refuses |
| `caliper_core_live_picker(core, enable, root)` | The session list drawn by the core while not attached |
| `caliper_core_live_poll(core)` | Once per frame: apply the newest completed snapshot of every view, write the heartbeat; -1 when not attached |
| `caliper_core_live_detach(core)` | Drop the views and claims, clear the heartbeat |
| `caliper_core_live_status` / `_view_status` | The counters the status window shows |

A host whose window may be hidden must keep its frame timer running:
`caliper-viewer` holds an `NSProcessInfo` activity
(`NSActivityUserInitiatedAllowingIdleSystemSleep | NSActivityLatencyCritical`)
for that reason, and the Caliper app holds the same one while a session is
attached (`src/host/live/live_activity.h`). With the user-initiated option alone an occluded viewer
measured 5 to 11 fps for 10 s while the trainer ran, and a throttled viewer's
heartbeat can go stale, which stops the script's snapshots while the user is
still attached.
