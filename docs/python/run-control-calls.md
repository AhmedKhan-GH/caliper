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

`caliper.watch(model, optimizer)` is the one line that puts a training script under run control. It starts `caliper.serve()` when no server runs, and it makes `optimizer.step()` the checkpoint of the run with a step pre-hook that calls `caliper.step()`. A script that passes an optimizer must not also call `caliper.step()`.

The call registers the parameter `lr` on a log scale, from the start rate divided by 1,000 to the start rate multiplied by 10. A new value of `lr` scales every parameter group by the same factor and applies to the same optimizer step. The call also publishes the series `loss` and `lr` and the scene `model`. The scene has one entity, `modules`, with one box for each watched module along +y. The row id of a box is the index of its module in `WatchHandle.modules`, thus a pick on a box names the module. The argument `scene=False` leaves the scene out.

Under a torch learning rate scheduler, a new `lr` scales the scheduled rate from the step where it applies. Without a scheduler, `optimizer.param_groups` holds the new rates after that step.
