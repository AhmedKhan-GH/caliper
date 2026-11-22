# Browser Elements

The `web/` package defines two custom elements, `<caliper-view>` and `<caliper-controls>`. The viewer page that `caliper.serve()` serves and the notebook widgets of Caliper LIB use them. Both elements read a [wire protocol v3](../reference/wire-protocol.md) stream. Elements with the same `src` on one page share one WebSocket. The module `caliper-view.js` registers both elements and exports `connect(src)`, the shared stream client. The `web/` package uses the Apache 2.0 license, like the core.

```html
<script type="module" src="http://127.0.0.1:7341/caliper-view.js"></script>
<caliper-controls src="ws://127.0.0.1:7341/v3/stream?token=..."></caliper-controls>
<caliper-view src="ws://127.0.0.1:7341/v3/stream?token=..." view="embedding" link details></caliper-view>
```

## `<caliper-view>`

One live view of the stream. It has a header and a drawing area. The header shows the name of the view, its data view picker, a line about the view, the run state, its toggles and a status dot. Where WebGPU gives an adapter, views compiled to WebAssembly draw the data. Otherwise, views written in TypeScript draw it. The header, the gestures, the events and the highlights are the same with both.

### Attributes

| Attribute | Meaning |
|---|---|
| `src` | The stream's `ws://` URL, with the session token. `session=<id>` in its query picks a session of a hub |
| `view` | The view's name. Several series share one plate when the names are separated by commas |
| `kind` | `heatmap`, `image`, `points`, `series` or `histogram`. Without it, the kind comes from the stream |
| `height` | The element's height in CSS pixels, header included. Default 320 |
| `renderer` | `auto` (the default) uses the WebAssembly views for the kinds their loader lists when WebGPU gives an adapter. `wasm` uses them with any adapter. `ts` uses the TypeScript views |
| `run-controls` | The header shows Pause or Resume and Step beside the run state if the producer has [controls](../reference/run-control-messages.md#controls) |
| `link` | [Linked selection](../reference/linked-selection-messages.md#linked-selection): The view sends a selection with its item space and items. It also draws the producer's selections when one of its axes carries their space |
| `details` | With `link`, the details of a selection made in this view show under it, with one row per item |

The header shows the run state, such as `running` or `paused at 1200`, when the producer announces a state other than `none`. This is true with or without `run-controls`. Pause, Resume and Step are disabled, with the reason as their tooltip, in three cases: the server does not let this viewer control the run, the session has ended, or the script has not reached a checkpoint yet.

A linked selection draws the selected points over the other points, which are dimmed. In a heatmap or image, it marks the rows or columns of a selected item with a band, an outline and a bar at the edge of the image. In a series, it puts markers on the selected steps.

The header has three toggles: `log` for a series or a histogram, `follow` for a series, and `rotate` for 3-D points. `rotate` is off until the user presses it. It turns the view slowly about its vertical axis, and any drag in the view stops it.

Both renderers draw 3-D points inside a box around the data. The edges of the box are a wireframe with tick labels. One edge of each axis also shows the name of the axis. A faint grid lies on the floor, the face across the vertical axis that is behind the data. The renderer draws the back edges and the grid behind the points, and the other edges and the labels in front of them. Nearer points look larger, within a factor of 0.5 to 2 of their size at the target of the camera. Points behind the target look dimmer in proportion to their distance, down to half their brightness at the far side of the data, and keep their hue. Selected points keep their full color. The camera starts at the same three-quarter view in both renderers, fitted to the data, and a double click returns to it. 2-D points have no box, cues or `rotate`.

### Properties and Methods

| Member | Meaning |
|---|---|
| `source` | A stream client to read from instead of the socket named by `src` |
| `dataView` | The resolved data view of the view. If you set it, the view asks the producer for another data view, as the picker does |
| `renderer` | What draws the view: `wasm`, `webgpu` or `canvas2d` |
| `pause()`, `resume()`, `step(n = 1)` | Send the control and resolve with the reply of the producer, or with null if this viewer may not control the run |
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

The controls of the session: the run state with Pause or Resume and Step, every parameter the script registered and every action. A `number` or `integer` parameter with a range shows a slider and a field for an exact value. The slider uses a log scale if the parameter says so. A parameter without a range shows the field only. A `boolean` shows a toggle and a `choice` shows a menu. An action shows a button and its last result. The button is marked busy while the action is queued or running. The controls travel on the WebSocket of the stream, so they work while a notebook cell runs.

A change is sent as CONTROL. The control then shows the value it asked for, marked `sending`, until the CONTROLS of the producer shows the value as pending (marked `at the next step`) or applied. As a result, every viewer shows what the producer holds. While the user drags a slider, the value goes out at most 30 times a second and ends with the value where the drag stops. Every control is disabled, with the reason as its tooltip, when this viewer may not control the run.

### Attributes

| Attribute | Meaning |
|---|---|
| `src` | The stream's `ws://` URL, with the session token |
| `height` | The largest height of the panel in CSS pixels. The panel scrolls beyond it |

### Properties and Methods

| Member | Meaning |
|---|---|
| `source` | A stream client to read from instead of the socket named by `src` |
| `controls` | The latest CONTROLS of the session, or null |
| `setParam(name, value)` | Sets a parameter and resolves with the reply, which includes the value after clamping, or with null |
| `runAction(name)` | Runs an action and resolves with its result after the reply `queued`, or with null |
| `pause()`, `resume()`, `step(n = 1)` | As on `caliper-view` |
| `debugState()` | Read-only state for tests |

### Events

| Event | `detail` |
|---|---|
| `caliper-controls` | The run state, a parameter or an action changed. The value is the CONTROLS message, or null if the session has none |
| `caliper-reply` | A CONTROL_REPLY to a control this element sent, `queued` included |

## Viewer Page and Widgets

The viewer page shows a `caliper-controls` panel above its grid when the producer announces controls and the script has a run state, a parameter or an action. The page gives every view `link` and `details`. The notebook widget of `caliper.widget()` (`caliper-widget.js`) is a `caliper-view` with `link`, `details` and `run-controls`. The module `caliper-controls-widget.js` is the anywidget module of a controls panel, with the traits `src` and `height`.
