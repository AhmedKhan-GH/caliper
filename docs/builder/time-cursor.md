# Time Cursor

The time cursor moves over the recent steps that Caliper keeps. Every panel that reads kept steps shows the value at the cursor. The cursor changes only the view, and the run goes on.

## Step History

The app keeps the recent values of a source in its own memory. The script keeps nothing for it. The key `history` of a source sets how many steps the app keeps.

```toml
[[source]]
id = "act"
catalog = "live:blocks.0.attn.proj.act"
history = 2048
transform = [
  { op = "slice", axis = 0, start = 0, stop = 64 },
  { op = "reduce", fn = "norm", axis = 1 },
]
```

The app keeps the value after the longest run of steps from the start that read no reference. A source without references thus keeps its final value. It keeps one value for each step, and a later value of a step replaces an earlier one. The step comes from the stamp of the snapshot. A run that is faster than the app reads it keeps steps with gaps between them.

| Rule | Value |
|---|---|
| Depth | `history` is from 0 to 65,536 steps. A source whose root is a live tensor and that has no `from` keeps 64 steps by default. Other sources keep none by default. |
| Budget | All kept steps of the open spec share 512 MiB. One source keeps at most 128 MiB of it. |
| Reset | A new attachment, a changed chain and the opening of a spec clear the kept steps of the source. |

When the requested steps need more than the budget, the app shortens the largest rings first. `cursor.get` gives the depth that each source keeps. A source that keeps fewer steps than requested has a `note` that names the source and the limits and suggests a reduce step before the kept value. `history` on a source whose root is not a live tensor is `spec.history_not_allowed`, and a value outside the range is `spec.out_of_range`.

A live series keeps its own ring of values but no steps. A source over a live series that states `history` keeps the newest value of the series at each step. A window over that source then gives the values with their steps. A series without `history` keeps nothing, and the series panel draws its ring as before.

## Window Step

The step `window` stacks the kept steps of another source. It is the first step of a source and reads the kept steps of the source that `from` names. That source must keep all of its steps, or the validator gives `spec.no_history`.

| Key | Type | Meaning |
|---|---|---|
| `start`, `stop` | Integer step numbers or references | The first and the last step, both included |
| `steps` | Integer from 1 to 65,536, or a reference | The last N kept steps up to the cursor |

`steps` excludes `start` and `stop`, and a window with both is `spec.bad_transform`. Without a key, the window holds every kept step. A window never holds a step after the cursor. A window over a tensor source gives a tensor with shape (W, ...) in step order. A window binds as the columns `step` and `value`, such as `x = "loss_window.step"`.

## Cursor

The cursor is the built-in state `cursor`. A reference reads its fields. Only the transport, the cursor commands and new steps change them.

| Field | Type | Meaning |
|---|---|---|
| `step` | Integer | The step that the panels show |
| `first` | Integer | The oldest step that any source keeps |
| `latest` | Integer | The newest step that the app received |
| `live` | Boolean | Whether the cursor follows the newest step |
| `playing` | Boolean | Whether the cursor moves toward the newest step |
| `speed` | Number | The steps for each second of play, from 0.1 to 1,000 |

The cursor is live when a spec opens, and it moves to each new step. While it plays, it moves through the kept steps at its speed, and at the newest step it becomes live. A held cursor stays at its step. A step back or forward moves it to the previous or the next step that any source keeps. A move to a step that no source keeps goes to the nearest kept step, and the answer gives the step that the cursor took.

## Latest at the Cursor

Each panel shows, for each of its sources, the newest kept value at or before the cursor. A source with `from` evaluates its own steps on that value. A source with no kept steps shows its newest value, and the info line of the panel says "Live only".

| Panel | At the cursor |
|---|---|
| `series` | The points up to the cursor, and a vertical line at the cursor when it is not live |
| `heatmap`, `image`, `histogram`, `points`, `table` | The value of each source at the cursor |
| `text` and readouts | Each field at the cursor |
| `scene` | Each entity at the cursor. A live scene from the script stays live, and the info line says so. |

`panel.data` states the step that a panel shows in `data.cursor`, with the keys `step`, `live_only` and `window`. A panel that reads one value has the step. A panel that reads a window has the window and no step.

## Transport Control

The control kind `transport` binds `cursor`. It shows, from left to right, a step back, Play or Hold, a step forward, Live, a speed menu and the step as text. The text reads "Live" or "Held" beside the step number. The option `speed` sets the speed at open, with a default of 10. With `keys = true`, the default, Space plays or holds, the arrow keys step, Home goes to the first kept step and End goes live.

## Run Pause and Cursor

The pause of the run and the cursor are two things. A pause stops the script at a step boundary. The cursor chooses which kept step the panels show. A hold never pauses the run, and a pause never moves the cursor. The run has its own button, a control bound to `run:pause`. While the run is paused, the transport shows the chip "Run Paused".

## Cursor Events

The event kind `cursor` has the keys `change`, `step`, `first`, `latest`, `playing` and `speed`. The change is `live`, `held`, `playing`, `moved` or `clamped`. A move by a command, a step or a key adds `moved`. A move by play or by a new live step adds no event, so that the ring of 1,024 events keeps room for decisions. [Instrument Tools](../agents/instrument-tools.md#cursor-commands) lists the commands.
