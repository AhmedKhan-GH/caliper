# Attach to a running script

A script that calls `caliper.show` (or any other view call) opens a **live
session** on its first call. A viewer can attach to it at any point while the
script runs, detach, and attach again; the training loop never waits for it.
This page covers the viewer side: finding a session, attaching from the
Caliper app (the usual way) or with the standalone `caliper-viewer`, what each
shows, detaching, and what happens when either process goes away. The design
is in [Live streaming](../explanation/live-streaming.md).

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
| Views | One window per view (see [What you see](../tutorials/live-pytorch.md#what-you-see)), tiled into the page on first appearance; move, resize or dock them, and the layout is kept in `imgui.ini` |
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
| `--data-dir <root>` | Session-file root (default `$CALIPER_DATA_DIR`, else the app-data folder) |
| `--exit-after <s>` | Close the window after `s` seconds |
| `--size <w>x<h>` | Window size in points (default 1280x800) |
| `--stats` | Print a stats line to stderr every 2 s: fps, updates, step-to-screen median/p95/max, skipped and client-dropped counts, each view's renderer path, RSS and footprint |
| `--latency-log <file>` | Write every latency sample (`t_ns lat_ms` per line); implies `--stats` |
| `--headless` | No window, an offscreen canvas (CI, tests). Needs `--session` |
| `--dump-dir <dir>` | With `--headless`: write each view's pixels or values to `<dir>` whenever its sequence number changes |

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
| Buttons | **Detach** and **Pause** / **Resume** |

Pause freezes the display only; snapshots keep arriving and are dropped, so
training is not slowed.

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
snapshot, no slot, no GPU work. Measured (`S-noviewer.txt`, torch 2.14.1,
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

/* Attach: by token, by pid, or show the built-in session list. */
caliper_core_live_attach(core, s[0].session);
caliper_core_live_attach_pid(core, pid, NULL);  /* NULL = the default data root */
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
