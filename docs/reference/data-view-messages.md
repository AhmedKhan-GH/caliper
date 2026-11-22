---
kind: reference
---

# Data View Messages

This page lists the messages with which a viewer requests a data view in [Wire Protocol](wire-protocol.md) v3, and the rules for data views.

Each message gives the keys of its metadata. Keys marked optional may be absent. The header of every message follows the [frame layout](wire-frames.md#frame-layout). The [message types](wire-protocol.md#message-types) table gives the type id of each message. All the messages on this page have no section.

## `subscribe`

| Key | Type | Meaning |
|---|---|---|
| `sub` | integer (uint32) | Chosen by the viewer. A `sub` that is already open is replaced, as if the viewer unsubscribed first |
| `view` | string | The name of the view |
| `request_id` | integer (uint32) | Chosen by the viewer, echoed by VIEW_REPLY |
| `max_rate` | number, optional | The most messages per second for this subscription. Default 60. The producer treats a value over 240 as 240. It must be positive |
| `data_view` | object, optional | The requested [data view](#data-views). Absent or `{}` means the default of the view |

The producer answers with VIEW_REPLY. On success, the subscription is open and VIEW_REPLY carries the resolved data view. On failure, VIEW_REPLY carries an error and no subscription exists. A connection holds at most 256 subscriptions, and the producer answers one more with the error `limit`.

## `unsubscribe`

| Key | Type | Meaning |
|---|---|---|
| `sub` | integer (uint32) | The subscription to end. The producer ignores an unknown `sub` |

The producer discards the pending snapshot and series values of the subscription.

## `view_request`

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer (uint32) | Chosen by the viewer, echoed by VIEW_REPLY |
| `sub` | integer (uint32) | An open subscription |
| `view` | string | The name of its view |
| `data_view` | object | The new data view |

A zoom, a pan at another resolution, or a choice of layer, head or slice is a view request. On success, the subscription switches to the new data view. On failure, it keeps the data view it had. Snapshots made under the old data view may still arrive after the reply. Each snapshot names its data view, so the viewer can tell them apart.

## `view_reply`

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer | The `request_id` of the SUBSCRIBE or VIEW_REQUEST it answers |
| `sub` | integer | The subscription |
| `view` | string | The name of its view |
| `data_view` | object | On success, the resolved data view |
| `error` | object | On failure, `{"code", "message"}`, with a code from the [error codes](wire-protocol.md#error-codes): `unknown_view`, `unknown_subscription`, `bad_data_view` or `limit` |

A reply carries exactly one of `data_view` and `error`.

## Data Views

A data view is a slice, selection, projection or aggregation of a source that the producer computes on its device. Only the result travels. A viewer requests a data view with SUBSCRIBE or VIEW_REQUEST. The producer resolves it and answers with the resolved form, and every snapshot names the resolved data view that produced it. The schema is small: it names a source, indexes dimensions, reduces dimensions, and sets a size budget.

### View Sources

Each view in VIEWS lists its sources, so a viewer can build a picker for layers, heads and slices.

| Key | Type | Meaning |
|---|---|---|
| `name` | string | The name of the source, 1 to 63 bytes, unique in the view, such as a layer or a parameter |
| `dims` | array of objects | The dimensions of the source in order, at most 8. Each is `{"name", "size"}`, with a name of 1 to 63 bytes, unique in the source. In 3.1 a dim may also have `items`, the [item space](linked-selection-messages.md#item-spaces) whose items it indexes |
| `dtype` | string, optional | The element type of the source tensor, for display: `f32`, `f16`, `bf16`, `f64`, `i64`, `i32`, `i16`, `i8`, `u8` or `bool` |

A view has at most 64 sources. The dims of an image source are its height and width, and the RGBA channels are not a dim. The dims of a points source end with the point dim, and the coordinates are not a dim. A series view has no sources.

### Data View Object

| Key | Type | Meaning |
|---|---|---|
| `source` | string | The name of the source. Default: the first source of the view |
| `index` | object | Per dim: an integer, which selects one position and drops the dim, or `[start, stop]` or `[start, stop, step]`, which selects a range and keeps the dim |
| `reduce` | object | Per dim: `mean`, `sum`, `max`, `min` or `norm` (the square root of the sum of squares), which reduces the dim away |
| `tile` | string | Heatmaps only: a kept dim to lay out as a mosaic of cells |
| `budget` | object | The size budget, by kind below |

An index counts from 0. A negative index or bound counts from the end of the dim, as in Python. In a range, `start` and `stop` may be `null`, which means the start and the end of the dim. The producer clamps the bounds to the dim, and `step` is at least 1.

| Budget key | Kinds | Default | Limit | Meaning |
|---|---|---|---|---|
| `max_extent` | Heatmap, image | 512 | 4096 | The most payload rows and the most payload columns |
| `pool` | Heatmap | `mean` | | How cells pool to meet `max_extent`: `mean`, `max` or `min`. Images always pool by `mean` |
| `max_points` | Points | 100,000 | 1,000,000 | The most points in the payload |
| `max_backfill` | Series | All retained values | The retention of the producer | The most retained values sent on subscribe. 0 sends none |

A budget value under 1 (under 0 for `max_backfill`) is an error. The producer treats a value over the limit as the limit.

### Data View Resolution

The producer resolves a requested data view in these steps:

1. Members the request leaves out come from the `default` of the view. A member in the request replaces the member of the default as a whole. The exception is `budget`, whose keys come one by one from the request, then from the default, then from the table above. Thus `{}` means the default of the view, and `{"index": {}}` means the default without its indexing.
2. The source must exist, and every dim named in `index`, `reduce` and `tile` must be one of its dims.
3. An integer index must lie inside its dim, and a range must select at least one position.
4. A dim indexed by an integer cannot be reduced, and the tile dim must be kept.
5. The dims that remain, the **kept dims**, must fit the kind: 2 for a heatmap (3 with `tile`), 2 for an image, 1 for points, and any number for a histogram. A series takes no `index`, `reduce` or `tile`.
6. The payload must not exceed 4096 rows or columns. It must also fit in one SNAPSHOT frame, at most `CALIPER_NET_MAX_FRAME_BYTES`, with room for the largest head a frame may have. Thus the payload is at most 66,060,176 bytes. Only a heatmap or an image, at 4 bytes a cell, can exceed this, above 16,515,044 cells.

The resolved data view has the same members as a request, with every default filled in:

- `source`.
- `index`, with integers made non-negative and every range written as `[start, stop, step]` with non-negative clamped bounds.
- `reduce`.
- `budget`, with the keys of its kind: `max_extent` and `pool` for a heatmap, `max_extent` for an image, `max_points` for points, and none for a histogram.
- `tile`, when the heatmap tiles.

A series resolves to `{"budget": {"max_backfill": n}}`. If a data view breaks a rule, the producer answers with the error `bad_data_view` and a message that names the rule.

The producer computes the data view on its device in this order. It takes the source and applies `index`. It applies `reduce` one dim at a time in the dim order of the source. It lays out the tile mosaic and then meets the budget by pooling (heatmap and image) or sampling (points). Points sample with a stride, so the sample stays the same while the size of the source does not change.

### Payload Axes

`axes` in a snapshot lists the kept dims in payload order, one object each. For a tiled heatmap, the tile dim comes first, then the rows and columns of the cell. For other kinds, the kept dims keep their source order.

| Key | Meaning |
|---|---|
| `dim` | The source dim |
| `start`, `step` | The selected range of the dim: position p is source index `start + p × step` |
| `count` | The number of selected positions |
| `size` | The payload extent along this axis (for the tile axis, the number of cells) |

Payload index i stands for the selected positions from `floor(i × count / size)` up to, but not including, `ceil((i + 1) × count / size)`. A pooled axis aggregates those positions, and a sampled axis (the point dim) takes the first of them. This is the rule of the adaptive pooling of torch, so a viewer can map a hovered texel or a picked point back to source indices.

A tiled heatmap of C cells has `cols = ceil(sqrt(C))` and `rows = ceil(C / cols)`. Each cell pools to `min(H, max(1, floor(max_extent / rows)))` rows and `min(W, max(1, floor(max_extent / cols)))` columns. Cell t is at row `t div cols` and column `t mod cols`, and the cells left over hold NaN. This is the channel mosaic of `caliper.show()`.

### Data View Kinds

| Kind | Kept dims | Payload | Budget |
|---|---|---|---|
| Heatmap | 2 (rows, columns), or 3 with `tile` | `f32` `(H, W)` after pooling | `max_extent`, `pool` |
| Image | 2 (height, width) | `u8` `(H, W, 4)` after mean pooling | `max_extent` |
| Points | 1 (the point dim) | `f32` `(N, D)` after stride sampling, with `aux` sampled alike | `max_points` |
| Histogram | Any | The counts of every element of the result over its range, binned on the device | none |
| Series | No sources | SERIES messages | `max_backfill` |

## Related Pages

- [Wire Protocol](wire-protocol.md)
- [Wire Frames](wire-frames.md)
- [Data Messages](data-messages.md)
- [Data Views](../python/data-views.md)
- [Data view calls](../python/data-view-calls.md)
