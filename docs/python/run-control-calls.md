---
kind: reference
figure: none
---

# Run Control Calls

The run control calls let viewers pause a training loop, change its parameters, run its actions and link selections across views.

## Controls

Controls let viewers reach back into the running script. They pause, resume and single-step its training loop, change the parameters it registered and run the actions it registered. Linked selection makes a selection in one view highlight the same items in the other views.

Both travel on the stream as messages of wire protocol 3.1 ([Controls](../reference/run-control-messages.md#controls) and [Linked selection](../reference/linked-selection-messages.md#linked-selection)). Thus they work while a notebook cell runs, when the kernel cannot read widget messages. The viewers of `caliper.serve()` may control unless it started with `control=False`. The viewers of a hub that `connect()` streams to may control only when the hub started with `--allow-control`. Both senders carry them.

[Script Control](script-control.md) uses them in a training loop. Every call also works without a server. Parameters keep their values, `step()` returns at once, and Python code can call an action.

| Call | Returns | Does |
|---|---|---|
| `step(step=None)` | `None` | The checkpoint of the training loop: sets changed parameter values, runs requested actions, waits while the run is paused |
| `pause()` | `bool` | Pauses the run at the next `step()` |
| `param(name, value, *, min=None, max=None, log=False, choices=None, help=None)` | `Param` | A value that viewers can change |
| `action(name, fn=None, *, help=None)` | `Action` | A function that viewers can run. Also a decorator |
| `items(space, details=None)` | `ItemSpace` | An item space for linked selection, with optional details |
| `controls(*, height=160)` | widget | The notebook panel of run controls, parameters and actions |

### `step`

```python
caliper.step(step=None)
```

The checkpoint of the training loop. Call it once per step, on the training thread, between two steps. It is the only place where the controls change the script. There it sets the parameter values that viewers changed, runs the actions they asked for in the order they asked, and waits while a viewer has paused the run. `step` is the number of the step, which viewers show where the run paused. `None` counts the calls.

A waiting `step()` returns in these cases:

- A viewer resumes the run or asks for a number of single steps.
- The server closes.
- No connection that may control has been open for 10 seconds. A page reload stays well within this time.

While it waits, it still sets values and runs actions as they arrive. Ctrl-C, or the interrupt of a notebook, ends the wait with `KeyboardInterrupt`, and the run is no longer paused afterward. Right before it waits, it sends the snapshots and series values of the step where it paused. The run state goes to each viewer after them, so a view that reads `paused` already shows that step ([What a viewer may rely on](../reference/run-control-messages.md#what-a-viewer-may-rely-on-when-a-run-pauses)).

With nothing waiting, a call only counts the step and reads one flag. It costs the same with a viewer attached or not.

Before the first `step()` of the script, viewers can change parameters, which then take effect at once. They cannot pause the run, and actions wait for the first checkpoint.

### `pause`

```python
caliper.pause()
```

Pause the run at the next `step()`, as the Pause button of a viewer does, for example when the loss stops being finite. A viewer resumes it. The call returns `True`. When no stream server runs, no viewer could resume the run. The call then returns `False` with a `RuntimeWarning` and does nothing. When a server runs but no connection may control it, the call warns that the run resumes after 10 seconds.

### `param`

```python
lr = caliper.param("lr", 1e-3, min=1e-5, max=1e-1, log=True, help="learning rate")
for batch in loader:
    caliper.step()
    for group in opt.param_groups:
        group["lr"] = lr.value
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | | 1 to 63 bytes. Registering a name again replaces the parameter |
| `value` | see below | | The first value. Its type sets the type of the parameter |
| `min`, `max` | number | `None` | The range of a number or an integer. A value from a viewer is clamped to it |
| `log` | `bool` | `False` | Draw the slider on a log scale. Needs `min` greater than 0 and `max` |
| `choices` | list of `str` | `None` | The choices of a `str` value, 1 to 64 |
| `help` | `str` | `None` | A tooltip, at most 1,024 bytes |

| `value` | Type | A viewer shows |
|---|---|---|
| `bool` | `boolean` | A toggle |
| `int` | `integer` | A slider or a field |
| `float` | `number` | A slider or a field |
| `str`, with `choices` | `choice` | A menu |

A value from a viewer is checked and clamped. A number or an integer is clamped to `min` and `max`, an integer must be a whole number, and a choice must be one of `choices`. The value reaches `value` at the next `step()`, or at once before the first `step()` of the script. When a viewer sends several values before a `step()`, the last one wins. A script has at most 64 parameters. `param()` returns a `caliper.Param`:

| Member | Meaning |
|---|---|
| `value` | The current value, to read in the loop. It changes only in `step()`, or at once before the first `step()` |
| `pending` | A value from a viewer that `value` takes at the next `step()`, or `None` |
| `set(v)` | Set the value from the script, at once and for every viewer, checked and clamped like a value of a viewer. Returns the value set, or raises `ValueError` |
| `remove()` | Take the parameter away from the viewers. `value` keeps its last value |
| `name`, `type`, `min`, `max`, `log`, `choices`, `help` | The registration |

### `action`

```python
@caliper.action("save checkpoint", help="saves the model's weights")
def save_checkpoint():
    torch.save(model.state_dict(), "checkpoint.pt")
    return "saved checkpoint.pt"
```

`caliper.action(name, fn)` registers `fn`, a function without arguments. `caliper.action(name)` does the same as a decorator. A request of a viewer runs the function at the next `step()`, on the training thread, after the parameter values are set. Requests run one at a time in the order they arrived. A request for an action that already waits joins that run.

The text that `fn` returns goes back to the viewer that asked, cut to 1,024 bytes. An exception goes back to that viewer with its message and appears in the script as a `RuntimeWarning`. It does not stop the loop. A request that no `step()` has run within 10 minutes fails with a time-out. A script has at most 64 actions, and registering a name again replaces the action.

`action()` returns a `caliper.Action`. Calling it runs `fn` at once on the thread of the caller, and `remove()` takes it away from the viewers.

### `items`

```python
caliper.items("eval", details={"class": eval_labels})
caliper.points(z, eval_labels, "embedding", classes=10, items="eval")
caliper.show(probabilities, "predictions", items="eval")
```

An item space is a set of items numbered from 0 that several views index, such as the samples of an evaluation set. A view names the item space of its item axis with `items=`, a keyword of `show`, `points`, `series` and `offer`:

| Call | `items="eval"` names the space of |
|---|---|
| `show` | Its rows: the first dimension, or the first of `dims` |
| `points` | Its points: the point dimension |
| `series` | Its index, the number of values appended before each value |
| `offer` | The first dimension of each source, or the point dimension of points |

`items={"sample": "eval"}` names the dimensions of `dims` instead. A histogram has no item axis. A selection made on an axis with an item space highlights the same items in every view of the session that names the space: the points, the rows of a heatmap and markers on a series. A call without `items=` has no extra cost.

`details` gives a viewer a few fields of the selected items, shown in a small table under the view where the selection was made. It is one of two forms:

- A mapping `{field: sequence}` indexed by item id, copied to the host once, at the call.
- A function `fn(ids)` that returns `{field: values}` in the order of `ids`, or one dict per item.

The function runs on the thread `caliper.details`, never on the training thread, and has 2 seconds. It must read only data that the loop does not change in place, such as labels or arrays that the script computed before. Details cover the first 32 selected items and at most 16 fields. Each value is a number, a string of at most 256 bytes, a boolean or `None`.

`items()` returns a `caliper.ItemSpace`. Its `selection` is the list of selected item ids, ascending, and it is current even while a cell runs. `select(ids)` sets the selection for every viewer, and `clear()` clears it. A session has at most 16 item spaces. Python code also receives each select event through `server.on_event`, `server.events()`, `widget.selection` and `on_select`. Its `detail` has `space` and `items` too.

### `controls`

```python
caliper.controls(*, height=160)
```

A notebook widget, like `widget()`, with the run state, the Pause, Resume and Step buttons, every parameter and every action of the script, and the result of the last action it ran. It reads the stream directly, so it works while a cell runs, and it starts `serve()` when no server runs. Its traits are `src` and `height`. The header of every view widget also shows the run state and the Pause, Resume and Step buttons. A build whose browser assets lack the module of the panel, `caliper-controls-widget.js`, warns and shows a note in its place.
