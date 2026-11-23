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

## Programs in the Script

Under `caliper.watch(model, optimizer)`, the script announces the capabilities `programs` and `interventions` to a viewer. The app then sends the GPU programs of its open spec as PROGRAM messages ([GPU Programs](../agents/gpu-programs.md)). The script needs no other line. A script that does not call `caliper.watch(model, optimizer)` announces neither capability.

The module `caliper._producer` loads when the first program arrives, thus a script that only streams tensors loads nothing new. It compiles the program through the sandbox, imports each input tensor on the GPU in place and read-only, and never copies a training tensor. `caliper.step()` runs the programs that are due after the work of the step is done. A program runs every N steps, as the cadence of the program rules, and its GPU time has a limit. A program that is refused or stopped is an error of its panel and never raises in the script.

An input is `live:<name>`. The name is a parameter name of the watched model, a parameter name with `.grad` added for its gradient, or a module name with `.act` added for the last output of the module. The script gives a geometry output to the publish path of `caliper.scene`, as a batch of the target entity. It gives a pixels output to `caliper.show`, as the live view `program.<id>`. A pick, a selection and a recording then work on the output as on any other. The script answers each PROGRAM with a PROGRAM_STATE, and it sends the status of each program again after a change and at most twice a second while programs run.

The script also lists each module of the model as an intervention point by its dotted name, and each head of an attention module that has a head count and an output projection as `<module>#<h>`. A CONTROL with the op `intervene` sets or clears an `ablate` of a point, and the script applies it at the next step with a forward hook. The wire messages are in the [wire protocol reference](../reference/wire-protocol-3-2.md#gpu-programs-and-interventions).

## Watch in a Managed Run

The app's model runner starts a script as a managed run and sets `CALIPER_RUN_DIR`, with the other variables of [Managed Runs](../agents/managed-runs.md). When that variable is set, `caliper.watch(model, optimizer)` and `caliper.watch(model)` also do the following, and the script needs no other line.

1. It starts the server when none runs.
2. It loads the checkpoint that `CALIPER_RUN_RESUME` names into the model and the optimizer, and the step count continues from the step of the checkpoint. A run that resumes redoes that step with its own parameter values.
3. It applies the parameter values of the checkpoint, and then those of `CALIPER_RUN_PARAMS`, when each parameter is registered. A value of the wrong type causes a warning, and a number out of range is clamped.
4. It registers the actions `caliper.checkpoint` and `caliper.stop`.
5. It writes the link file that `CALIPER_RUN_LINK_FILE` names, with the stream link and no newline. The file comes last, thus the app that attaches finds the parameters and the actions.

The action `caliper.checkpoint` writes `checkpoint-<step>.pt` in the run folder at the next checkpoint and returns its path. The file holds the model state, the optimizer state, the step and the parameter values. The action `caliper.stop` makes the script end with `SystemExit(0)` at its next checkpoint, after it sends what is pending and before the weights change. It also ends a pause.

Without an optimizer, each forward pass of the model is the checkpoint, thus an inference loop with `caliper.watch(model)` takes the actions too. Such a script must not also call `caliper.step()`. Outside a managed run, `caliper.watch` does none of this.
