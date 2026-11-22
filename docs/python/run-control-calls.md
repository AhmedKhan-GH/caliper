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

<!-- The empty divs keep the anchors of the former hand-written entries, which the roadmap registry links to. -->
<div id="step"></div>

::: caliper.step

<div id="pause"></div>

::: caliper.pause

<div id="param"></div>

::: caliper.param

::: caliper.Param
    options:
      members: [value, pending, set, remove]

<div id="action"></div>

::: caliper.action

::: caliper.Action
    options:
      members: [__call__, remove]

<div id="items"></div>

::: caliper.items

::: caliper.ItemSpace
    options:
      members: [selection, select, clear]

<div id="controls_1"></div>

::: caliper.controls
