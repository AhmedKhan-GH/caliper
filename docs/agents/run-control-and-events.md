# Run Control and Events

An agent that uses `caliper-mcp` can add a control to the open spec, change a parameter of the script that the app is attached to, and read what happens in the app. This page describes the run commands, their approval rules and errors, and the command `events.wait`. The control commands are on [Spec Version 1](../builder/spec-version-1.md).

Each command is one MCP tool with the first dot of its name replaced by `_`, such as `run_set_param` and `events_wait`.

## Run Commands

The run commands act on the script that the app is attached to. The app sends each request as a CONTROL message of wire protocol 3.1 through the viewer connection that it already has. The agent never talks to the script. No request waits for the script. A value or an action reaches the script at its next step, and a `run` event reports what the script did.

| Command | Arguments | Result |
|---|---|---|
| `run.state` | None | `attached`, `script`, `control`, `why_not`, `state`, `step`, `steps_left`, `params`, `actions` |
| `run.set_param` | `name`, `value` | `name`, `value`, `pending` |
| `run.action` | `name` | `name`, `queued` |
| `run.pause` | `op` (`pause`, `resume` or `step`), `count` (1 to 1,000,000, for `step`) | `op`, `state` |

`run.state` changes nothing. Its `state` is `none` before the first step of the script, then `running`, `pausing`, `paused` or `stepping`. The key `step` is the count of `caliper.step()` calls, or null before the first one. The key `steps_left` counts the steps that remain while the run steps, and it is null otherwise. Each entry of `params` has `name`, `type` (`number`, `integer`, `boolean` or `choice`), `value`, `pending`, and when the script gave them, `min`, `max`, `log`, `choices` and `help`. Each entry of `actions` has `name`, `help`, `running` and `queued`. The keys `script` and `why_not` are absent when they do not apply.

`run.set_param` answers `pending: true`, because the value applies at the next step of the script. The script clamps the value to the range of the parameter. The `run` event of change `param` then carries the value after the clamp. Under `caliper.watch(model, optimizer)`, the next step is the next `optimizer.step()`.

`run.action` answers `queued: true`. The text that the action returns, or its error, comes back as a `run` event of change `action`. `run.pause` gives the state at the time of the request. A script pauses only at a step, thus a pause or a step before the first step fails with `run.no_step`.

## Approval Rules

A run command is of the effect class `run`. It changes the running script and nothing in the workspace. It has no undo, and the journal does not record it. The origins `agent` and `mcp` need your approval. The origins `ui`, `builder` and `api` run it at once.

| Mode | A `run` command |
|---|---|
| Ask | Asks for a card |
| Accept edits | Asks for a card |
| Read only | Refused with `command.denied` |
| Allow all | Runs |

The card states the change and the script, such as "Set parameter lr to 0.005 in train.py. The change applies at the next step." Its buttons are Allow, Allow for this session, Always allow and Deny. For `run.set_param` and `run.action`, the two rule buttons name the target, such as "Always allow changes of lr". The rule is stored with the key `run.set_param:lr` in `agent-permissions.json`. Thus a rule allows one parameter or one action and not the whole command. The key of `run.pause` is the command name. The slider and the button of your own spec send their values directly, so they show no card.

## Run Command Errors

Each error has a code and one plain sentence.

| Code | Cause |
|---|---|
| `command.not_available` | The session has no run control, because the app does not run |
| `run.not_attached` | No script is attached to the app |
| `run.control_not_allowed` | The connection takes no control. The message gives the reason. |
| `run.unknown_param` | The script has no parameter of this name |
| `run.unknown_action` | The script has no action of this name |
| `run.bad_value` | The value has the wrong type, or it is not one of the choices |
| `run.no_step` | A pause or a step before the first step of the script |
| `run.refused` | The script or the hub refused the request, with its message |

A script allows control when it starts its server with `caliper.serve(control=True)`, which is the default. A hub allows control only when you start it with `--allow-control`. Thus you decide at the start of the run, and the agent cannot change it. The shared memory session of `caliper.show` takes no control. The run control needs the network connection that `caliper.watch` starts.

## Intervention Commands

An intervention changes a module of the model at a step boundary while the run trains. The first intervention is `ablate`, which sets the output of a module to zero. For an attention module that gives heads, it sets the slice of one head. The names `scale`, `steer` and `patch` are reserved, and `intervene.set` refuses them with `intervene.op_reserved`. A script offers interventions only under `caliper.watch(model, optimizer)`.

| Command | Arguments | Result |
|---|---|---|
| `intervene.points` | `match`, a glob over the point names | `points`, each with `point`, `kind` (`module` or `head`), `module`, `head`, `type` and `shape` |
| `intervene.set` | `point`, `op`, `from_step`, `until_step` | `point`, `op`, `pending: true` |
| `intervene.clear` | `point`, or `all: true` | `cleared` (the points) and `pending: true` |
| `intervene.list` | None | `interventions`, each with `point`, `op`, `from_step`, `until_step`, `since_step` and `state` (`pending`, `active` or `ended`) |

Each module of `model.named_modules()` is a point, with its dotted name, such as `blocks.0.attn`. A module gives heads when it has an integer attribute `num_heads`, `n_heads` or `n_head`, and one child `nn.Linear` named `out_proj`, `o_proj`, `c_proj` or `proj` whose `in_features` divides by the head count. Each head is then a point `<module>#<h>`, such as `blocks.0.attn#3`. A module of the class `nn.MultiheadAttention` gives no heads, because it never calls its `out_proj`. A point holds at most one intervention of each op.

`intervene.set` answers at once, and the script applies the intervention at its next step. A `run` event of change `intervene` then reports each applied change, with the `point`, the `op` and the `step`. The `op` is null when the interventions of the point were cleared. The commands have the effect class `run`, thus they follow the Approval Rules above. The cards read "Ablate `<point>` in `<script>` from the next step.", "Clear the interventions on `<point>` in `<script>` at the next step." and "Clear every intervention in `<script>` at the next step.". A rule allows one point, with the key `intervene.set:<point>` or `intervene.clear:<point>`, and clearing all interventions always asks.

| Code | Cause |
|---|---|
| `intervene.unknown_point` | The model has no point of this name |
| `intervene.op_reserved` | The op is `scale`, `steer` or `patch` |
| `intervene.bad_range` | `until_step` is not above `from_step`, or a step is in the past |
| `intervene.not_supported` | The attached script did not announce `interventions` |

The run command errors above also apply, such as `run.not_attached`.

## Toggle Control

A control of the kind `toggle` bound to `intervene:<point>` sets the ablation of the point while it is on and clears it when it is off. A toggle bound to `param:<name>` of a boolean parameter sets that parameter. The click of the person has the origin `ui`, thus it runs at once and shows no card. A toggle is disabled, with the reason beside it, when no script is attached, when the script takes no interventions, or when the model has no such point. `control.create` takes the bind `intervene:<point>`, as [Spec Version 1](../builder/spec-version-1.md) describes.

## Events

The app keeps the last 1,024 events in a ring. The command `events.wait` returns the events after a cursor.

| Argument | Meaning |
|---|---|
| `cursor` | The cursor of the previous result. The default 0 reads from the oldest event that the ring keeps. |
| `kinds` | Only these kinds. Every kind when absent. |
| `panels` | Only the events of these panels or controls. An event without a panel never matches. |
| `limit` | The most events returned, from 1 to 256. The default is 64. |
| `timeout_s` | How long to wait for a first event, from 0 to 60 seconds. The default is 10. The value 0 answers at once. |

The result holds `events`, `cursor`, `lost` and `more`. The new `cursor` is past every event that the call looked at, also the events that did not match, thus the next call does not read them again. The key `lost` counts the events after the given cursor that the ring dropped before the call. The key `more` is true when more matching events wait beyond `limit`.

When matching events wait, the call answers at once. Otherwise it holds the request until one comes or `timeout_s` ends. Other commands run while the call waits. `caliper-mcp` waits for the answer up to `timeout_s` plus the margin of a call of Caliper API. Hovers never enter the ring.

Each event has the keys `seq`, `time`, `kind`, `panel`, `view` and `origin`. The key `seq` grows by one with each event, starting at 1. The key `time` is UTC ISO 8601 with milliseconds. The keys `panel` and `view` are null when they do not apply. The key `origin` is `ui` for the person, `script` for the running script, and otherwise the origin of the command. The kind adds its own keys.

| Kind | Cause | Keys of the kind |
|---|---|---|
| `pick` | A click on an item of a scene, or `selection.set` | `entity`, `primitive`, `index`, `row_id`, `position`, `fields`, `annotation`, and `pixel` for an image |
| `select` | A selection in a scene, made or cleared | `entities`, `truncated` |
| `control` | The person finished an edit of a control, or `control.set` set one | `control`, `bind`, `value`, and `key` when a shortcut made the change |
| `state` | A state variable changed | `state`, `value`, `previous`, `writer` |
| `cursor` | The time cursor changed | `change`, `step`, `first`, `latest`, `playing`, `speed` |
| `run` | The attached script changed | `change`, then the keys of the change |
| `spec` | A command changed the open spec | `command`, `event`, `applet` |
| `approval` | A card asked or was decided | `approval`, `command`, `state`, `target` |
| `program` | A program or a stage changed its state ([GPU Programs and Stages](gpu-programs.md#program-events)) | `program`, `change`, `host`, `code`, `message`, `line`, `column` |

A pick in a scene panel has the panel id in `panel` and the name of the scene in `view`. A pick or a selection in a live view outside a spec has `panel` null. The key `index` is the index of the item in the batch that the script sent. The key `row_id` is its row in the source tensor of the script, or null. The key `position` is the point that the click hit in scene coordinates, or null without depth. Each entry of `entities` in a selection has `entity`, `primitive`, `indices` and `row_ids`, and an empty list means that the selection was cleared. The key `truncated` is true when the selection held more than 32,768 items.

The key `fields` of a `pick` event holds the pick fields of the item, such as `row`, `column`, `x`, `y`, `z`, `value` and `step` for a surface ([Pick Fields](../builder/scene-entities.md#pick-fields)). The key `annotation` is set when the pick hit a label or a marker. The `writer` of a `state` event is a control id, a panel id or the command name. The `change` of a `cursor` event is `live`, `held`, `playing`, `moved` or `clamped`. A move by play or by a new live step adds no event.

A `control` event is one event for each finished edit. A slider sends each value to the script while you drag it, but the ring gets only the last value, at the end of the drag or typed entry. A button gives one event for each click.

| Change of `run` | Keys | Cause |
|---|---|---|
| `state` | `state`, `step` | The run state changed |
| `param` | `name`, `value`, `step` | The script applied a parameter at a step |
| `action` | `name`, `status`, `text` | An action ended with `ok` or `error` |
| `attach` | `attached`, `script` | A script was attached or detached |
| `intervene` | `point`, `op`, `step` | The script applied or ended an intervention at a step |

A `spec` event comes from the journal. Its `event` is `do`, `undo` or `redo`, and `applet` is the id of the spec that is open after the command. An `approval` event has the `state` `pending`, `approved`, `denied` or `expired`. It also comes for a run command that you deny or that expires. For a run command, `target` names the parameter or the action.

The `step` of a `run` event is the step of the script when it changed, or null before the first step.
