# Run Control and Events

An agent that uses `caliper-mcp` can add a control to the open spec, change a parameter of the script that the app is attached to, and read what the person does in the app. This page describes the control commands, the run commands and the `events_wait` tool.

## Controls

A control is a widget of the open spec. M1 builds the kinds `slider` and `button`. The commands `control_create`, `control_patch` and `control_remove` edit the open spec as the panel commands do. Each one has a dry run, a diff, one undo step and a journal entry.

A control has one target in its `bind` string. The string `param:<name>` names a parameter that the script registered with `caliper.param`. The string `action:<name>` names a registered action. The string `run:pause` pauses and resumes the run. The string `value` names a value that the spec holds. A slider takes the range and the scale of its parameter when it gives none of its own. The script clamps each value to its own range.

A spec that has a control is a version 1 spec. The first scene panel or control that you add to a version 0 spec raises its version in the same undo step.

## Run Commands

The run commands act on the script that the app is attached to. They go to the script as protocol 3.1 CONTROL messages, through the connection that the app already has.

| Tool | What it does |
|---|---|
| `run_state` | Returns the run state, the parameters and the actions. |
| `run_set_param` | Sets one parameter. The value applies at the next step of the script. |
| `run_action` | Queues one action for the next step. |
| `run_pause` | Pauses, resumes or steps the run. |

A run command asks for your approval when the agent is in the mode Ask or Accept edits. The card names the change and the script. A rule that you save allows one parameter or one action, such as `run.set_param:lr`. The mode Read only refuses a run command.

A run command fails with `run.not_attached` when no script is attached. It fails with `run.control_not_allowed` when the connection takes no control. A script allows control when it starts its server with `caliper.serve(control=True)`, which is the default. A hub allows control when you start it with `--allow-control`. Thus you decide at the start of the run, and the agent cannot change it.

## Events

The app keeps the last 1,024 events in a ring. The tool `events_wait` returns the events after a cursor. A cursor of 0 reads from the oldest event that the ring keeps. The result holds `events`, the new `cursor`, `lost` and `more`. When no matching event waits, the call holds the request until one comes or `timeout_s` ends. The default is 10 seconds and the maximum is 60 seconds.

Each event has the keys `seq`, `time`, `kind`, `panel`, `view` and `origin`, and the kind adds its own keys.

| Kind | Cause |
|---|---|
| `pick` | A click on an item of a scene |
| `select` | A selection in a scene, made or cleared |
| `control` | A control of the spec changed its value |
| `run` | The attached script changed: its state, a parameter, an action or its attachment |
| `spec` | A command changed the open spec |
| `approval` | A card asked or was decided |

A pick in a scene panel has the panel id in `panel` and the name of the scene in `view`. It also gives `entity`, `primitive`, `index`, `row_id` and `position`. Hovers never enter the ring.
