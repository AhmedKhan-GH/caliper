---
kind: reference
figure: none
---

# Viewer

The Caliper app and `caliper-viewer` attach to a live session or a stream and show its views, with the options, states and displays below.

## Options

`caliper-viewer` takes these options. An attach that fails ends with exit status 2.

| Option | Effect |
|---|---|
| `--session <token>` | Attach to this session at once |
| `--url <link>` | Attach to a network stream by its link ([Stream Links](stream-links.md)) |
| `--data-dir <root>` | Session-file root, `$CALIPER_DATA_DIR` by default, then the app-data folder |
| `--exit-after <s>` | Close the window after `s` seconds |
| `--size <w>x<h>` | Window size in points, 1280x800 by default |
| `--stats` | Print a stats line to stderr each 2 s: fps, updates, step-to-screen median, p95 and maximum, skipped and client-dropped counts, renderer path of each view, RSS and footprint |
| `--latency-log <file>` | Write each latency sample as `t_ns lat_ms` on one line, and imply `--stats` |
| `--headless` | Use an offscreen canvas and no window, for CI and tests. It needs `--session` or `--url`, and prints one `received` line for each view when it exits |
| `--dump-dir <dir>` | With `--headless`, write the pixels or values of each view to `<dir>` when its sequence number changes |

## Session List

The Projects pane of the Caliper app and the "Live sessions" window of `caliper-viewer` list the sessions and relist them each second. The app lists the newest first, with the script, the PID, the start time (clock time and how long ago), the state and the view names of each. The viewer window shows the script, the PID, the start time, the number of views, the state and an **Attach** button. The app also lists the streams of this computer with the state `stream`, as [Stream Links](stream-links.md#stream-listing) shows. A row that cannot attach is faint in the app, and `⏎ Attach` is off with the reason beside it.

| State | Meaning | Attach |
|---|---|---|
| running | The script runs and no viewer is attached | On |
| starting | The session file exists but the shared memory is not ready | On, and the viewer waits |
| viewed by `<pid>` | One more viewer holds the session | Off |
| closing | The script closes the session | Off |
| protocol v`N` | A Caliper version that is not compatible wrote it | Off |
| attached | The app is attached, and Enter opens its live page | Not applicable |
| stream | A script that streams listens on this computer | On |

## Display

### Live Page

The live page of the Caliper app has four parts.

| Part | Shows |
|---|---|
| Bar | `esc Home`, `⌘D Detach`, the script name and its state |
| Run bar | For a stream only: run state, Pause or Resume, Step, parameters and actions |
| Views | One window for each view on a grid |
| Status bar | Backend and GPU, pid of the script, number of views, frame rate and step-to-screen latency of the last second |

The state in the bar is a filled, breathing diamond and `live` while the script runs. It is `waiting for the session to start` before that, and an outlined diamond and `SESSION ENDED` after the script stops. The grid fills the page and lays itself out again when the page narrows. The layout of the windows that you move, resize or dock is kept in `imgui.ini`.

The window title is `Caliper - <script> (pid <pid>)`, with ` - session ended` added after the script stops. The page draws at up to 60 fps, as `caliper-viewer` does, for each display refresh rate. Activity on the landing screen logs `attached`, `session ended` and `detached` with the script and its pid. It logs `attach failed` with the reason when a session refuses the app, because one more viewer holds it, it closed, or the renderer is not Metal.

### Status Window

The window title of `caliper-viewer` is `caliper: <script> (<pid>)` while it is attached, with ` - starting` or ` - session ended` added when that applies. One window for each view shows the live data. A status window in the bottom-right corner has these lines.

| Line | Content |
|---|---|
| Header | Script, pid and session token |
| State | "running", "waiting for the session to start", or "SESSION ENDED: the script is gone; showing the last frames received" |
| Frames | Frame count, fps and number of views |
| Counters | Updates applied, snapshots skipped because a newer one came first, and refused updates |
| Latency | Step-to-screen median and p95 for the recent samples |
| Per view | Name, kind, last sequence number, latency, `dropped@client` and the error of a refused update |
| Controls | For a stream: the run bar and the details of the selection |
| Buttons | **Detach** and **Pause** or **Resume** |

The value `dropped@client` counts the snapshots that the script dropped because it had no free slot. The **Pause** button freezes the display only. The snapshots continue to come and the viewer drops them, and thus the training does not slow down. To hold the training loop, use Pause in the run bar.

### View Windows

Each view has its own window on a grid that fills the page. The windows come in the order in which the views appear. The grid lays itself out again when the page changes size or when a view is added. It gives each view at least 340 by 240 points when the page has room. A window that you move, resize or dock keeps its place.

The header of a window shows the name of the view on the left and a short summary on the right. The summaries are `64×32  -0.53 … 0.57` for a heatmap, `64 points  2-D  values` for points, `457 steps  last 0.0263` for a series and `32 bins  -0.53 … 0.57  n 2048` for a histogram. The header also has tools: `pause` for all views, `log` for a series or a histogram, `follow` for a series and `rotate` for 3-D points. A dot shows the state of the stream. It is bronze while the stream is live and the view has data, an outline while the view waits for its first data, and grey after the script stops.

With `CALIPER_VIEWS_DEBUG=1`, a line under each header also shows the diagnostics of the view. The line has the size, the renderer path, the sequence number, the step-to-screen latency, the update count and the number of snapshots that Caliper skipped while the view was paused.

| View | Display | Input | Hover |
|---|---|---|---|
| Heatmap | Fills its plot area, with a colorbar beside it | Drag pans, wheel zooms, double click fits | Row, column and value |
| Image | Keeps equal aspect | Same as a heatmap | Row, column and RGBA value |
| Points, 3-D | Points in a box with tick labels on three edges and a faint floor grid. Nearer points are larger. Points behind the target of the camera are dimmer. | Left or right drag orbits. Middle drag or shift+drag pans. Wheel zooms. Double click or the F key fits the default three-quarter view. `rotate` turns the view until a drag. | Nearest point: index, position and label |
| Points, 2-D | Points on a plane | Drag pans. Wheel zooms about the cursor. | Same as 3-D |
| Series | A line over the append index, with the latest value printed | None | Step and value |
| Histogram | Bars at the bin centers | None | Bin range and count |

A heatmap and an image have axes of the source row and column indices. The axes carry the names of the dims as titles, and they follow a pan or a zoom. A heatmap cell uses nearest sampling when it covers a pixel, and thus each cell stays a cell.

The hover value of a heatmap comes from the shared slot. As a result, it can come from a newer snapshot than the pixels under the cursor. The `pause` tool reads `paused` in bronze until a click ends the pause. When the display resumes, the newest snapshot shows again.

## Stream Controls

### View Pickers

A view of a stream can offer more than it draws by default. It can have several sources, such as the layers of a model. It can also have dimensions that its kind does not draw, such as the heads of an attention map ([Data Views](../reference/wire-protocol.md#data-views)). In the app and in `caliper-viewer`, such a view has a picker under its header, as in the browser. The picker has a menu of the sources when the view has several. It has a menu for each dimension, which keeps the dimension (`all`), reduces it (`mean`, `sum`, `max`, `min` or `norm`), or picks one index. A field of more than 256 positions uses the index pick.

A heatmap that keeps three dimensions shows the first one as a mosaic of tiles. A view with nothing to choose has no picker. A choice goes to the script at once. The script calculates the new data view on its device, and only the result travels. The menus then show the data view that the script resolved, and thus a choice that the script refused or clamped returns to what the script shows. A shared-memory session offers no choices.

### Run Bar

A script on protocol 3.1 can take controls from its viewers. It calls a checkpoint one time for each training step with `caliper.step()`. It can register parameters and actions, and it can declare which axes of its views index the same items, such as the samples of a test set ([Controls](../reference/wire-protocol.md#controls) and [Item Spaces](../reference/wire-protocol.md#item-spaces)). The live page of the app and `caliper-viewer` show these for a stream. A shared-memory session has no channel back to the script, and thus its page has no run bar.

The run bar is under the page bar and above the views.

| Part | Shows and does |
|---|---|
| Run state | `running`, `pausing`, `paused at step N`, `stepping, N left` or `no checkpoint yet` |
| Pause, Resume | Pause holds the training loop at its next checkpoint, and Resume releases it |
| Step | Runs one step and pauses again |
| Parameters | One control for each registered parameter, by its type |
| Actions | One button for each registered action, then `queued` or `running`, and the result for 15 s after it ends, or `FAILED:` and the reason |
| Right edge | The last refused request and its reason for 10 s, or the reason why the controls are off |

Only a paused run names a step, the one at which the loop waits. The script sends its run state when the state changes, not at each step, and thus the bar does not know the current step of a running loop. A parameter with a number or an integer with a range is a slider, on a log scale if the script asks for one, with a field beside it that shows and takes the exact value. A parameter without a range is a field, a boolean is a box and a choice is a menu.

A value that you set shows at once with the mark `pending`. The script reads a new value at its next checkpoint. The mark stays until a CONTROLS message that the script sent after its reply carries the value. A value out of range comes back clamped. A refused value returns to the value of the script, with the reason at the right edge. While you drag a slider, at most 30 values each second go to the script, and the value where the drag stops always goes last.

The page logs each change to standard error, one `[live]` line each, in `caliper-viewer` as well as in the app:

```text
[live] run paused at step 165
[live] param lr = 0.02
[live] save: saved at step 165
[live] controls off: the server does not let this viewer control (a hub needs --allow-control)
```

In `caliper-viewer`, the same run bar and details are in the status window, in the style of ImGui.

### Selection

Shift-drag across a view to select. A selection is rows of a heatmap or an image, a range of steps of a series, or the points inside the lasso of an embedding. When the selected axis indexes an item space, the viewer sends the selection to the script with the item ids. The script shares it with all viewers of the session.

Each view whose axis indexes the same space then shows the selection. The selected points show over the dimmed rest, with their count. The selected rows or columns of a heatmap or an image have a bronze outline, in each cell of a mosaic. A series has markers on the selected steps. Thus a selection of units in one weight matrix marks the same units in the next matrix and in an embedding of them. A selection is not a control, and every viewer can select, with or without `--allow-control`.

When the script gives details for the space, such as a few fields of each selected sample, a Selection pane opens at the right of the page. It has one row for each selected item, for the first 32, under the name of the item space. It closes when the selection is clear. If the details failed, the pane gives the reason.

## Session End

The viewer keeps its last frames on the screen when the script stops. In `caliper-viewer`, the title gets "session ended", and the status window shows "SESSION ENDED" until you detach or close it. The Caliper app shows `SESSION ENDED` in the bar of its live page, `session ended` in its status bar and its title, and a `session ended` row in Activity. A killed script, with SIGKILL or a crash, looks the same in the viewer and in the app.

The app stays attached to the ended session on any page until you detach or attach a different session. Then the landing command bar shows an outlined diamond and `SESSION ENDED` beside the script. The `live:` sources of the session keep their last data, and thus a spec over them shows the last values of the run.

## Security

The run bar shows the run state, the parameters and the actions to all viewers. It changes them only when all of these conditions are true. If one is not true, all controls are off and the bar gives the reason:

- The server lets this viewer control. The `caliper.serve()` of a script does, unless it refuses controls. A hub does only when it starts with `--allow-control`, either as `caliper serve --allow-control` or as the `--api` stream of the app with `--allow-control` ([Hub Setup](hub-setup.md)).
- The script takes controls, because it announced them.
- The session is live. The controls of an ended session stay visible and are off.

Before the first checkpoint of the script, Pause, Step and the actions are off with `the script has called no checkpoint yet`. A parameter can already change, and the script reads it at once. The stream carries only these requests and never a command to the app or the computer.

## Related Pages

- [Script Attachment](script-attachment.md)
- [Stream Links](stream-links.md)
- [Script Control](script-control.md)
- [Wire Protocol](../reference/wire-protocol.md)
- [Python Quickstart](../introduction/python-quickstart.md)
