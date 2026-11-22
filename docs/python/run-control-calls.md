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

[](){#caliper.step}
### `step`

::: caliper.step
    options:
      show_root_heading: false
      show_root_toc_entry: false

[](){#caliper.pause}
### `pause`

::: caliper.pause
    options:
      show_root_heading: false
      show_root_toc_entry: false

[](){#caliper.param}
### `param`

::: caliper.param
    options:
      show_root_heading: false
      show_root_toc_entry: false

::: caliper.Param
    options:
      members: [value, pending, set, remove]

[](){#caliper.action}
### `action`

::: caliper.action
    options:
      show_root_heading: false
      show_root_toc_entry: false

::: caliper.Action
    options:
      members: [__call__, remove]

[](){#caliper.items}
### `items`

::: caliper.items
    options:
      show_root_heading: false
      show_root_toc_entry: false

::: caliper.ItemSpace
    options:
      members: [selection, select, clear]

[](){#caliper.controls}
### `controls`

::: caliper.controls
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Watch With an Optimizer

`caliper.watch(model, optimizer)` is the one line that puts a training script under run control. Its signature is `caliper.watch(model, optimizer=None, *, activations=True, gradients=False, every=1, include=None, rate=None, groups=True, scene=True)`, and every argument after `optimizer` is a keyword. Without an optimizer, the call behaves as before. With one, it starts `caliper.serve()` when no server runs, and it keeps the shared memory session closed. Thus a view does not appear twice in an app that attaches with `--url`.

The call makes `optimizer.step()` the checkpoint of the run with a step pre-hook. At each checkpoint the hook publishes the loss, calls `caliper.step()` and writes the parameter `lr` into the parameter groups. Pending parameters apply, queued actions run and a pause waits there, before the weights change. A script that passes an optimizer must not also call `caliper.step()`.

The call registers the parameter `lr` on a log scale, from the start rate divided by 1,000 to the start rate multiplied by 10. It registers no parameter when parameter group 0 has no positive rate, and it warns instead. A new value of `lr` is the rate of group 0 and moves every other group by the same factor. It applies to the same optimizer step.

The call also publishes these names:

| Name | Kind | Content |
|---|---|---|
| `loss` | Series | The last 0-d tensor that `backward()` ran on before the step, also for a closure that `optimizer.step(closure)` runs |
| `lr` | Series | The learning rate of group 0 that each step used |
| `model` | Scene | One entity, `modules`, with one box for each watched module along +y |

The size of a box shows the parameter count of its module, and its color shows the mean absolute value of the last output of the module. The label is the module name. The row id of a box is the index of its module in `WatchHandle.modules`, thus a pick on a box names the module. The scene is computed only while a viewer watches it. The argument `scene=False` leaves it out.

Without a scheduler, the new rates stay in `optimizer.param_groups`. Under a torch learning rate scheduler, a new `lr` scales the schedule from the step where it applies. The hook multiplies the scheduled rates by a factor before each step and takes it off after the step. The scheduler thus neither undoes the change nor compounds it. `param_groups` and `get_last_lr()` show the scheduled rates, and the series `lr` shows the rate that the step used. `WatchHandle.remove()` also removes `lr` and the loss tap, and it leaves the series and the scene.
