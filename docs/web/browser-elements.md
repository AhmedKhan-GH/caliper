# Browser Elements

The `web/` package defines two custom elements, `<caliper-view>` and `<caliper-controls>`. The viewer page that `caliper.serve()` serves and the notebook widgets of Caliper LIB are built from them. Both read a [wire protocol v3](../reference/wire-protocol.md) stream, and elements with the same `src` on one page share one WebSocket. The module `caliper-view.js` registers both elements and exports `connect(src)`, the shared stream client. The `web/` package is licensed under Apache 2.0, like the core.

```html
<script type="module" src="http://127.0.0.1:7341/caliper-view.js"></script>
<caliper-controls src="ws://127.0.0.1:7341/v3/stream?token=..."></caliper-controls>
<caliper-view src="ws://127.0.0.1:7341/v3/stream?token=..." view="embedding" link details></caliper-view>
```

## `<caliper-view>`

One live view of the stream, with a header (the view's name, its data view picker, a line about the view, the run state, its toggles and a status dot) and a drawing area. Caliper's own views compiled to WebAssembly draw the view where WebGPU gives an adapter, and the element's views written in TypeScript draw it otherwise. The header, the gestures, the events and the highlights are the same with both.

### Attributes

| Attribute | Meaning |
|---|---|
| `src` | The stream's `ws://` URL, with the session token. `session=<id>` in its query picks a session of a hub |
| `view` | The view's name. Several series share one plate when the names are separated by commas |
| `kind` | `heatmap`, `image`, `points`, `series` or `histogram`. Without it, the kind comes from the stream |
| `height` | The element's height in CSS pixels, header included. Default 320 |
| `renderer` | `auto` (the default) uses the WebAssembly views for the kinds their loader lists when WebGPU gives an adapter; `wasm` uses them whatever the adapter; `ts` uses the TypeScript views |
| `run-controls` | The header shows Pause or Resume and Step beside the run state, when the producer has [controls](../reference/wire-protocol.md#controls) |
| `link` | [Linked selection](../reference/wire-protocol.md#linked-selection): a selection is sent with its item space and items, and the producer's selections are drawn in this view when one of its axes carries their space |
| `details` | With `link`, the details of a selection made in this view show under it, one row per item |

The header shows the run state, such as `running` or `paused at 1200`, whenever the producer announces one other than `none`, with or without `run-controls`. Pause, Resume and Step are disabled, with the reason as their tooltip, when the server does not let this viewer control the run, when the session has ended, and when the script has not reached a checkpoint yet.

A linked selection draws the selected points over the others, which are dimmed; marks the rows or columns of a heatmap or image that stand for a selected item with a band, an outline and a bar at the image's edge; and puts markers on a series at the selected steps.

The header's toggles are `log` for a series or a histogram, `follow` for a series, and `rotate` for 3-D points. `rotate` is off until pressed; it turns the view slowly about its vertical axis, and any drag in the view stops it.

3-D points draw inside a box around the data, as both renderers draw it. The box's edges are a wireframe with tick labels and the axis's name on one edge of each axis, and a faint grid lies on the floor, the face across the vertical axis that is behind the data. The back edges and the grid are drawn behind the points and the other edges and the labels in front of them. Nearer points are drawn larger, within a factor of 0.5 to 2 of their size at the camera's target, and points behind the target are dimmed in proportion to their distance, to half their brightness at the far side of the data, with their hue kept. Selected points keep their full colour. The camera starts at the same three-quarter view in both renderers, fitted to the data, and a double click returns to it. 2-D points have no box, cues or `rotate`.

### Properties and Methods

| Member | Meaning |
|---|---|
| `source` | A stream client to read from instead of the socket named by `src` |
| `dataView` | The resolved data view of the view; setting it asks the producer for another one, as the picker does |
| `renderer` | What draws the view: `wasm`, `webgpu` or `canvas2d` |
| `pause()`, `resume()`, `step(n = 1)` | Send the control and resolve with the producer's reply, or with null when this viewer may not control |
| `settled()` | Resolves once pending drawing is done |
| `debugState()` | Read-only state for tests |

### Events

| Event | `detail` |
|---|---|
| `caliper-select` | A selection made in this view: `{"view", "type": "select", "detail"}`, the detail the producer receives |
| `caliper-dataview` | The data view the user picked in the header |
| `caliper-selection` | A SELECTION from the producer for an item space this view carries, or one made in this view: the message's `space`, `items`, `truncated`, `view`, `details` and `details_error` |
| `caliper-reply` | A CONTROL_REPLY to a control this element sent |

## `<caliper-controls>`

The controls of the session: the run state with Pause or Resume and Step, every parameter the script registered and every action. A `number` or `integer` parameter with a range shows a slider, on a log scale when the parameter says so, and a field for an exact value; one without a range shows the field only. A `boolean` shows a toggle and a `choice` a menu. An action shows a button, marked busy while it is queued or running, and its last result. The controls travel on the stream's own WebSocket, so they work while a notebook cell runs.

A change is sent as CONTROL. The control then shows the value it asked for, marked `sending`, until the producer's CONTROLS shows it pending (marked `at the next step`) or applied, so every viewer shows what the producer holds. While a slider is dragged, the value goes out at most 30 times a second and ends with the value where the drag stops. Every control is disabled, with the reason as its tooltip, when this viewer may not control the run.

| Attribute | Meaning |
|---|---|
| `src` | The stream's `ws://` URL, with the session token |
| `height` | The panel's largest height in CSS pixels; it scrolls beyond it |

| Member | Meaning |
|---|---|
| `source` | A stream client to read from instead of the socket named by `src` |
| `controls` | The latest CONTROLS of the session, or null |
| `setParam(name, value)` | Sets a parameter; resolves with the reply, the value after clamping included, or with null |
| `runAction(name)` | Runs an action; resolves with its result after the reply `queued`, or with null |
| `pause()`, `resume()`, `step(n = 1)` | As on `caliper-view` |
| `debugState()` | Read-only state for tests |

| Event | `detail` |
|---|---|
| `caliper-controls` | The run state, a parameter or an action changed: the CONTROLS message, or null when the session has none |
| `caliper-reply` | A CONTROL_REPLY to a control this element sent, `queued` included |

## Viewer Page and Widgets

The viewer page shows a `caliper-controls` panel above its grid when the producer announces controls and the script has a run state, a parameter or an action, and gives every view `link` and `details`. The notebook widget of `caliper.widget()` (`caliper-widget.js`) is a `caliper-view` with `link`, `details` and `run-controls`. The module `caliper-controls-widget.js` is the anywidget module of a controls panel, with the traits `src` and `height`.
