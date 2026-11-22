---
kind: reference
---

# Linked Selection Messages

This page defines the message and rules for linked selection in [Wire Protocol](wire-protocol.md) 3.1.

## Linked Selection

A selection in one view highlights the matching items in the other views of the same session. The items can be the points of an embedding, the rows of a heatmap or markers on a series. Linked selection is the capability `selection` of 3.1.

### Item Spaces

An **item space** is a named set of items numbered from 0, such as the samples of a test set. A producer declares in VIEWS which axes index the items of a space:

- A dim of a source with `items`: source index i along that dim is item i of the space.
- A series view with `items`: absolute index i of the series is item i.

The point dim of an embedding and the `sample` dim of a heatmap with the same space show the same items. A viewer maps a payload position to a source index with the [payload axes](data-view-messages.md#payload-axes) of the snapshot, and the source index is the item id. The `row_id` of points already does this for a sample of points. A pooled heatmap row or column stands for several items, and the viewer highlights it when any of them is selected. A session uses at most 16 item spaces. A producer adds `items` to VIEWS only for a peer that announced `selection`.

### Selection Flow

A viewer that selects on an axis with an item space adds two keys to the `detail` of its select EVENT, besides the keys of its kind:

- `space`: the item space.
- `items`: the selected item ids, at most 32,768, with `truncated` set when the viewer cut them.

For points, the items are the `row_ids`. For a heatmap or an image, they are the source indices of the selected rows, or of the selected columns when only the columns have a space. For a series, they are the indices in `steps`. Empty `items` clears the selection of the space. A viewer adds these keys only when the server announced `selection`.

The producer keeps one current selection for each item space. On a select event with `space`, it replaces the selection of that space with the items of the event, made ascending and without repeats. It then sends SELECTION to every viewer, including the one that selected. Every viewer then draws the same highlight in every view that maps the space. The script can set a selection too, which reaches the viewers the same way, without `view`. Python code reads the select event as before.

### Item Details

The script may provide details for an item space: a few fields of each item, from a static table or from a function it registered. For each new selection, the producer looks up the details of its first 32 items and sends them in SELECTION as `details`:

| Key | Type | Meaning |
|---|---|---|
| `fields` | array of strings | The field names, at most 16, each 1 to 63 bytes |
| `items` | array of integers | The item ids, at most 32 |
| `values` | array of arrays | One row per item, with one value per field: a number, a string of at most 256 bytes, a boolean or null |

The function runs neither on the training thread nor on the network thread, and it has 2 seconds (`CALIPER_NET_DETAILS_TIMEOUT_MS`). A producer may send a selection at once and send it again with its details. When the details fail or time out, the selection goes without them and `details_error` says why. A viewer shows the details in a small table beside the view in which the selection was made. It never sends code or names of fields to compute.

## `selection`

SELECTION has no section. The [message types](wire-protocol.md#message-types) table gives its type id, and the header follows the [frame layout](wire-frames.md#frame-layout). Keys marked optional may be absent.

| Key | Type | Meaning |
|---|---|---|
| `space` | string | The [item space](#item-spaces), 1 to 63 bytes |
| `items` | array of integers | The selected item ids, ascending and without repeats, at most 32,768. Empty when nothing is selected |
| `truncated` | boolean, optional | True when the selection was cut to its first 32,768 items. Default false |
| `view` | string, optional | The view in which a viewer made the selection. Absent when the script set it |
| `details` | object, optional | The fields of the script for the first selected items, as [Item Details](#item-details) describes |
| `details_error` | string, optional | Why `details` is missing although the script provides details for the space, at most 1,024 bytes |

After the CONTROLS of an attachment, or after its VIEWS for a viewer without `controls`, the producer sends one SELECTION for each item space whose selection is not empty. It sends SELECTION to every viewer whenever the selection or the details of a space change.

## Related Pages

- [Wire Protocol](wire-protocol.md)
- [Wire Frames](wire-frames.md)
- [Data Messages](data-messages.md)
- [Run Control Messages](run-control-messages.md)
