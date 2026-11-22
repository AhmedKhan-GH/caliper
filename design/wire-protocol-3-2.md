# Wire Protocol 3.2 Draft

This page keeps what remains of the draft of version 3.2 of wire protocol v3: what producers, the hub and servers do with timelines, text logs, bar charts and recordings, the scene data views beyond the default, and the numbers that the owners of the protocol still have to confirm. [Scenes](scenes.md) plans the scene model, and the recordings plan, which git history keeps as `design/recordings.md`, planned the timelines, the recording file and the scrubber that these messages serve.

The rest of the draft is built. The three codecs, `src/host/net/wire.h`, `python/caliper/_wire.py` and `web/src/protocol/codec.ts`, encode and check every message, key, section, dtype, limit and error code of the draft, with golden frames under `tests/fixtures/wire_v3/v3_2`, and Wire Protocol 3.2 (`docs/reference/wire-protocol-3-2.md`) specifies them. Producers, the hub and viewers use the scene messages, with the hub's forwarding and the scene's state for a late viewer, and the scene view sends pick, hover and select events. No producer sends a stamp, a text log or a bar chart yet, and no server holds a recording.

## Timelines and Stamps

A producer with `timelines` declares them in VIEWS when the script declares one, and it sends the whole VIEWS again. Timelines are never removed or reordered during a session, so a timeline's id is its position in the list. A message with `static` true carries no stamp. A producer that streams scenes without timelines announces `scenes` alone, and its scene messages carry no stamp, which is what producers do today. A recorder of a session without `timelines` stamps each message with its own receive time on `time` and leaves it absent on `step`.

The formats of the declaration, the stamp sections and stamp text are in Wire Protocol 3.2 (`docs/reference/wire-protocol-3-2.md`), Timelines and Stamps.

## New Kinds in VIEWS

A producer lists a view of a new kind only to a peer that announced its capability. The producer and the hub do so for `scene` today and not yet for `text` and `bars`, which no producer sends.

**Data views.** A scene's data view has the keys `entities` and `budget` of [Scene Data Views](scenes.md#scene-data-views). `entities` is `{"include": [...], "exclude": [...]}`, at most 16 filters each, every filter an entity path, `<path>/**` or `**`; a filter outside the path grammar is `bad_data_view`. The budget keys are `max_points` (default 100,000, limit 1,000,000), `max_items` (100,000, 1,048,576), `max_vertices` (1,000,000, 4,194,304) and `max_extent` (1,024, 8,192), resolved key by key as for the other kinds. A subscription takes only the default form today, and any other is answered with `bad_data_view`. When a subscription's data view changes, the sender sends the scene's state again.

## Message Types

Wire Protocol 3.2 (`docs/reference/wire-protocol-3-2.md`) gives the format of every message; these subsections keep what senders and servers do with them.

### `text`

A line is the text of one `caliper.text` call and may itself hold line breaks. A producer cuts a line longer than the limit and marks the cut. On subscribe the sender first sends the lines it retains, at least the last 4,096, under `max_backfill`, and then the new ones. A viewer stores lines by index, as series values. Lines are never dropped silently: a viewer so far behind that its next lines left the sender's retention receives a `start` past what it holds, with `dropped`.

### Events on the New Kinds

| Kind | `pick` and `hover` detail | `select` detail |
|---|---|---|
| Bars | `bar`, the payload index; `row_id`, its source index; `value` | `bars`: a `[start, stop]` range of source indices |
| Text | none | none |

With an item space, a select also carries `space` and `items`, as in 3.1. The scene's pick, hover and select events are built, and Caliper LIB still refuses `items=`, so a scene select carries no `space` and `items` yet.

### `recordings`

The server sends it right after SESSIONS to a viewer that announced `recordings`, and again whenever the list changes.

A session object in SESSIONS may carry `recording`, `{"id", "seekable", "complete"}`: a session opened from a recording, or a live session of which the server holds a recording. A viewer may send CURSOR only for a session whose `seekable` is true.

### `recording_open`

The server opens the recording as an ended session, unless it is already open, and attaches the viewer to it as ATTACH would. It answers with SESSIONS when the list changed, then VIEWS, which confirms the attachment, CONTROLS as recorded at the end, and TIME_STATE with the `request_id`, the cursor at the end of the recording. A failure is answered with ERROR `unknown_recording` or `bad_recording` and leaves the viewer as it was. The session's id is the recording's id, with `-2`, `-3` and so on when that id is taken. An opened recording stays in the list while a viewer is attached and for 10 minutes after the last one leaves.

### `cursor`

A cursor belongs to one viewer connection and the session it is attached to; attaching to another session drops it. `pause` holds the cursor where it is, `step` moves it to the next or previous position with an item of a view the viewer subscribes to, and `follow` returns a live session's viewer to the live edge. Playback follows the recordings plan, which git history keeps as `design/recordings.md`. A viewer sends at most 30 CURSOR messages a second, for example while a person drags the scrubber, ending with the position where the drag stops, and a server may skip a `seek` it has not started when a newer one arrives. The server answers each CURSOR with TIME_STATE, or with ERROR `not_seekable` or `bad_cursor`: an unknown timeline, malformed stamp text, a speed or count out of range, or `follow` on an ended session.

While the cursor does not follow the live edge, the server sends each of the viewer's subscriptions to that session what lies at the cursor, each data message with the key `cursor`, the generation of the TIME_STATE it answers:

- For a heatmap, image, points, histogram or bars view, the latest-at snapshot under the subscription's data view, or the recorded data view the reply named, when it differs from the one last sent; or a SNAPSHOT with `empty` true and no section when nothing lies at or before the cursor.
- For a series or a text log, nothing more: the backfill of a recorded session holds every recorded value with its stamp, and the viewer draws the cursor itself. A recorded series that was decimated arrives with the section `indices` i64 (M,), the absolute index of each value, and the key `decimated`, the bucket size.
- For a scene, during continuous playback the scene messages between the previous position and the new one, in order; after any other move, the scene's state at the new cursor.

Live data for those subscriptions stops until `follow`, which first sends the newest snapshot of each and each scene's state.

### `time_state`

A server sends it as the answer to CURSOR and RECORDING_OPEN, at most 30 times a second while playing, and at most once a second when the ranges of a live session grow. A viewer drops a data message whose `cursor` generation is older than the newest TIME_STATE it has received.

## Flow Control

The rules of 3.1 hold, with these additions:

- **Text lines never drop silently.** TEXT coalesces into one pending message per subscription and is never dropped while the sender retains its lines; beyond that, `dropped` says how many were lost.
- **Order.** Text goes with series, before scene messages and snapshots.
- **TIME_STATE and RECORDINGS** are control messages: they go in order and before pending data, and a newer one replaces an unsent older one.
- **Playback** obeys `max_rate` per subscription and the one-pending-snapshot rule, so a slow viewer sees fewer frames during playback, never older ones.

## Hub Forwarding

A 3.2 hub announces the capabilities it implements. It removes the additions from what it forwards to a viewer that lacks their capabilities: views of unknown kinds and the key `timelines` leave VIEWS, stamp sections and new keys leave SNAPSHOT, SERIES and CONTROLS, and the payload sections stay shared, since only the metadata is encoded again, as it already is to rewrite `sub`. The hub does this for `scenes` today, and not yet for `timelines`, `text`, `bars` and `recordings`.

| Message | Forwarded | Kept for a late viewer |
|---|---|---|
| VIEWS with `timelines` | To every viewer of the session, filtered as above | The latest, as in 3.1 |
| SNAPSHOT, bars included | As in 3.1 | The latest of each producer subscription |
| SERIES with stamps | As in 3.1 | The newest 1,048,576 values of each series view with their stamps, 8 × T more bytes per value |
| TEXT | As SERIES | The newest 65,536 lines or 16 MiB of each text view |
| CONTROLS with a stamp | As in 3.1 | The latest |
| RECORDINGS, TIME_STATE | Made by the hub | |
| RECORDING_OPEN, CURSOR | Answered by the hub | |

The hub answers CURSOR only for sessions of which it holds a recording: a recording opened from its `--recordings` folder, or a session it records with `--record`. It never forwards CURSOR to a producer, so a viewer of a live session that the hub does not record scrubs its own buffer. With `--record`, the hub subscribes to every view at its recording rate, as its own recorder.

## Versioning and Compatibility

A 3.2 peer decides by capabilities, never by the minor alone.

| Viewer | Server | What happens |
|---|---|---|
| 3.2 | `caliper.serve()` of 3.1 | WELCOME announces none of them. The viewer draws what it gets and scrubs its own buffer, placed by its receive time on `time` |
| 3.2 | A 3.2 hub with a 3.1 producer | The session lists no `timelines`. The hub forwards 3.1 frames; with `--record` it stamps them with its receive time on `time` |
| 3.1 | A 3.2 hub serving recordings | The viewer sees an opened recording as an ended session and gets its state at the end. It receives no RECORDINGS and sends no CURSOR |
| 3.2 with `timelines` but without `scenes` | A 3.2 producer | The scene views are left out of VIEWS; the other views carry stamps |

## Worked Example

A viewer of a hub started with `--recordings` opens a recording of a training run with a series `loss`, a scene `world` and a text log `log`, seeks to step 5,000 and plays.

| # | From | Message | Content |
|---|---|---|---|
| 1 | V, H | HELLO, WELCOME | `capabilities` `["bars", "controls", "recordings", "scenes", "selection", "text", "timelines"]` both ways; `control` false |
| 2 | H | SESSIONS, RECORDINGS | The live sessions; `train-20261115-120314`, complete, `step` from `"0"` to `"19999"` |
| 3 | V | RECORDING_OPEN | `request_id` 1, `recording` `train-20261115-120314` |
| 4 | H | SESSIONS | The new ended session, with `recording` `{"seekable": true, "complete": true}` |
| 5 | H | VIEWS | `timelines` `step` and `time`; `loss` (4), `world` (6) and `log` (7) |
| 6 | H | CONTROLS | As recorded at the end, with its `stamp` |
| 7 | H | TIME_STATE | `request_id` 1, `generation` 1, `paused`, `step` at `"19999"` |
| 8 | V, H | SUBSCRIBE, VIEW_REPLY | `sub` 1 `loss`, `sub` 2 `world`, `sub` 3 `log`; three replies with the recorded data views |
| 9 | H | SERIES, TEXT | Every recorded value and line, with their `stamps` |
| 10 | H | Scene state | SCENE_CLEAR `reset`, SCENE_CLASSES, SCENE_TRANSFORMS, SCENE_UPDATE…, each with `state` true and `cursor` 1 |
| 11 | V | CURSOR | `request_id` 2, `seek`, `step`, `value` `"5000"` |
| 12 | H | TIME_STATE, scene state | `request_id` 2, `generation` 2, `paused` at `"5000"`; then the scene at step 5,000 with `cursor` 2 |
| 13 | V | CURSOR | `request_id` 3, `play`, `step`, `speed` 2 |
| 14 | H | TIME_STATE, scene messages | `request_id` 3, `generation` 3, `playing`; then the scene messages after step 5,000 at twice the recorded pace, each with `cursor` 3, and TIME_STATE at most 30 times a second |
| 15 | V | CURSOR | `request_id` 4, `pause`; answered by TIME_STATE `paused` at the step reached |

## Numbers to Confirm

The numbers of 3.2 are proposals until the owners of the protocol confirm them. They sit in a separate block at the end of `include/caliper/live_net.h` and of each codec, and the senders write minor version 1 in their frames until 3.2 is confirmed.

1. The minor version 2; the kinds `scene` 6, `text` 7 and `bars` 8, beside the table kind that 3.0 reserves without an id; the message types `0x0070` to `0x0074` and `0x0078` to `0x007B`.
2. `CALIPER_DT_U16 = 6` in `CaliperDType`, which the SDK owns and which has no `u16` yet, and `i64` as a dtype that producers send for stamps.
3. The capabilities `timelines`, `scenes`, `text`, `bars` and `recordings`, the error codes and the limits of Wire Protocol 3.2 (`docs/reference/wire-protocol-3-2.md`).
4. The hub's duties that are not built: filtering `timelines`, `text`, `bars` and `recordings` for older viewers, keeping text, serving and recording sessions, and answering CURSOR.

## Decisions

| Decision | Reason |
|---|---|
| A cursor per viewer connection, with generations | Several viewers of one recording scrub independently, and late data of an earlier cursor is dropped |
| The hub never forwards CURSOR to a producer | A producer serves one shared stream; seeking stays where a recording is held |
| A recorded session is an ended session to a 3.1 viewer | Older viewers still open recordings, at their end |

## Open Decisions

- **Server-side playback.** Options: the server plays, as drafted, or the viewer sends a seek per frame. Recommendation: the server plays, which takes one message per view per frame off the viewer's link.
- **Time windows.** Options: latest-at only, or a subscription key that asks for every item within a window before the cursor, for trails. Recommendation: a key `window` in a later minor version.
