---
kind: reference
---

# Data Messages

This page lists the messages of [Wire Protocol](wire-protocol.md) v3 that describe views, carry their data, and report viewer events.

Each message gives the keys of its metadata. Keys marked optional may be absent. The header of every message follows the [frame layout](wire-frames.md#frame-layout). The [message types](wire-protocol.md#message-types) table gives the type id of each message.

## `views`

VIEWS has no section.

| Key | Type | Meaning |
|---|---|---|
| `session` | string or null | The session id. Required on a viewer connection, where null means not attached. A producer may omit it on its connection to a hub |
| `views` | array of objects | Every view of the session, at most 256 |

The producer sends the full list again whenever it changes. A change is a view added or removed, or a source that changes its dims. A newer list replaces an unsent older one. Each view object has these keys:

| Key | Type | Meaning |
|---|---|---|
| `id` | integer | Never reused within the session. A view re-created with the same name gets a new id |
| `name` | string | 1 to 63 bytes, unique in the session |
| `kind` | integer | The live view: 1 heatmap, 2 image, 3 points, 4 series, 5 histogram (the values of `CaliperLiveKind`) |
| `sources` | array of objects | What the data views of the view can draw from, as [View Sources](data-view-messages.md#view-sources) describes. Empty for a series |
| `default` | object, optional | The default data view of the view. Absent means `{}` |
| `items` | string, optional | 3.1, series only: the [item space](linked-selection-messages.md#item-spaces) that the absolute index numbers of the series |

When a view disappears from the list or its id changes, the producer ends the subscriptions that named it. A viewer subscribes again if it still wants the view.

## `snapshot`

SNAPSHOT has one section, `data`, which holds the payload by kind as the second table below shows. With a second section, `aux`, it has two sections. Only points use `aux`, as `f32` `(N,)`, one label or value per point.

| Key | Type | Meaning |
|---|---|---|
| `sub` | integer | The subscription |
| `view` | string | The name of the view |
| `seq` | integer | The snapshot counter of the producer for the view. It grows with every snapshot of the view under any subscription, so a subscription sees it increase with gaps |
| `stamp_ns` | integer | The monotonic clock of the producer in nanoseconds at the call that made the snapshot (Python's `time.monotonic_ns()`) |
| `kind` | integer | The kind of the view |
| `data_view` | object | The resolved data view that produced the snapshot |
| `axes` | array of objects | How the payload maps onto the source, as [Payload Axes](data-view-messages.md#payload-axes) describes. Empty for a histogram |
| `tile` | object, optional | For a tiled heatmap, `{"dim", "rows", "cols"}` |
| `colormap` | integer, optional | 0 viridis, 1 magma, 2 rdbu (the values of `CaliperColormap`). Heatmaps, and points with continuous values. Default 0 |
| `flags` | integer, optional | 1: the low end of the range is `stats[0]`. 2: the high end is `stats[1]`. 4: draw a heatmap or image one texel per pixel at the origin (tests). The values of `CALIPER_LIVE_SNAP_*`. Default 0 |
| `vmin`, `vmax` | numbers, optional | The fixed ends of the range, for an end whose flag is not set. Default 0 and 1 |
| `stats` | array, optional | At most 64 numbers computed on the device of the producer, by kind below. A non-finite value is `null` |
| `points_dims` | integer | Points only: 2 or 3, equal to the second dimension of `data` |
| `label_count` | integer | Points only: more than 0 when `aux` holds categorical labels 0 to `label_count` − 1, and 0 when `aux` holds values to colormap |

| Kind | `data` | `stats` |
|---|---|---|
| Heatmap | `f32` `(H, W)` | `[min, max]` of the payload |
| Image | `u8` `(H, W, 4)`, RGBA | Empty |
| Points | `f32` `(N, D)`, D = `points_dims` | `[aux min, aux max, x min, x max, y min, y max, z min, z max]` over the points sent. z is 0 for 2-D points |
| Histogram | `f32` `(B,)`, bin counts | `[lo, hi]`, the range the bins cover |

A viewer draws a range end from `stats` when its flag is set. As in protocol v2, a non-finite range from `stats` keeps the last finite one, a flat range widens by 0.5 on each side, and a NaN value draws as colormap index 0. Categorical labels draw with the fixed 16-color palette, with label L in color L mod 16. Unlike protocol v2, 2-D points travel as `(N, 2)` and not as `(N, 3)` with padding.

## `series`

SERIES has one section, `values`, as `f32` `(M,)` with 1 ≤ M ≤ 262,144.

| Key | Type | Meaning |
|---|---|---|
| `sub` | integer | The subscription |
| `view` | string | The name of the view |
| `start` | integer | The absolute index of the first value: the number of values appended to the series before it |
| `total` | integer | The number of values appended so far. `start` + M ≤ `total` |
| `stamp_ns` | integer | The monotonic clock of the producer at the newest append |

A series travels only as SERIES messages, never as snapshots. On subscribe, the producer first sends the values it retains, the backfill, in as many messages as needed. Then it sends the new values. The x axis of a series is its absolute index. A viewer stores values by index. A message that starts past the end the viewer holds leaves a gap, and a message that overlaps values the viewer holds replaces them. A series view that is re-created gets a new view id, and its indices start at 0 again.

## `event`

EVENT has no section.

| Key | Type | Meaning |
|---|---|---|
| `view` | string | The name of the view |
| `type` | string | `select`, `pick` or `hover` |
| `detail` | object | What the viewer selected, picked or hovered, by kind as [Events](#events) describes |
| `sub` | integer, optional | The subscription the viewer was drawing |
| `seq` | integer, optional | The `seq` of the snapshot the viewer was showing |
| `data_view` | object, optional | The data view of that snapshot |

## Events

An EVENT reports what a person did in a viewer: a selection, a pick or a hover. Python code in the producer reads events. Events are data and never commands, and no event makes a producer run code. A producer that does not read events lacks the capability `events` and ignores them.

Coordinates in `detail` are source indices, computed from the [payload axes](data-view-messages.md#payload-axes) of the snapshot. Ranges are half-open, `[start, stop]`.

| Kind | `pick` and `hover` detail | `select` detail |
|---|---|---|
| Heatmap | `row`, `col`: source indices of the first position of the cell. `tile`: the source index of the tile dim, for a tiled heatmap. `value`, optional: the payload value | `rows`, `cols`: `[start, stop]` ranges of source indices. `tile` as for pick |
| Image | `row`, `col` as for a heatmap | `rows`, `cols` as for a heatmap |
| Points | `index`: the index of the point in the sent sample. `row_id`: its source index along the point dim | `indices`: indices in the sent sample. `row_ids`: their source indices, in the same order. At most 32,768 each. A viewer that cuts a larger selection sets `truncated` true |
| Series | `step`: the absolute index. `value`, optional | `steps`: a `[start, stop]` range of absolute indices |
| Histogram | `bin`: the bin index. `lo`, `hi`: the value range of the bin. `count`, optional | `range`: a `[lo, hi]` value range |

A `select` whose `detail` is `{}` clears the selection. A viewer sends at most 30 hover events per second per view, and it drops a hover and does not queue it. The producer keeps at most 256 unread events per connection and drops the oldest.

In 3.1, the `detail` of a `select` also carries `space` and `items` when the viewer selects on an axis with an item space. [Selection Flow](linked-selection-messages.md#selection-flow) describes them.
