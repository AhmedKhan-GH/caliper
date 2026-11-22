# Instruments

This page defines the interfaces of the first vertical slice of the agent-built playground. In this slice, an agent that uses only `caliper-mcp` adds a live 3-D scene and a slider to a spec, changes a parameter of a training script, and receives a pick from the scene. The training script has one Caliper line, `caliper.watch(model, optimizer)`.

The page states the spec format version 1, the new commands, the event format of `events.wait`, the run control of the app and the extension of `caliper.watch`. Entities from spec sources, binding references, programs, the model runner and interventions come in later versions. This page leaves room for them and does not define them.

## Spec Version 1

Version 1 is version 0 with two additions: the primitive `scene` and the top-level array `control`. Binding references, scene entities and programs join version 1 later, before version 1 goes into `main`. Until then, the reader refuses their keys as `spec.unknown_key`.

### Version Rules

A version 0 document stays valid, and the reader reads it as before. The reader knows the versions 0 and 1, and it refuses any other version with `spec.unsupported_version`. A version 0 document that uses a key of version 1 gets the errors of version 0, `spec.unknown_primitive` or `spec.unknown_key`.

The migration from version 0 to version 1 changes only the number, because version 1 is a strict superset. Thus a version 0 document with `caliper_spec = 1` is a valid version 1 document. The canonical writers write the version that the `Spec` holds. They do not raise it.

An edit that adds a scene panel or a control to a version 0 spec raises the version to 1 in the same command. The edit and the new number are one undo step. The fixtures of version 0 and their canonical text thus stay unchanged.

The JSON Schema of version 1 is `src/builder/spec/schema/caliper-spec-v1.schema.json`, beside the schema of version 0. Its `$id` is `urn:caliper:schema:spec:v1`.

### Scene Primitive

A panel with `primitive = "scene"` shows a live scene of the attached script. Its one slot, `scene`, is required. The slot holds `"live:<scene name>"`, the scene as `catalog.list` names it, and not a source id. The panel draws the scene as the live scene view draws it, inside the panel's content rectangle.

```toml
[[panel]]
id = "world"
primitive = "scene"
title = "Model"
bind = { scene = "live:model" }
options = { up = "+z", axes = true }
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `up` | `+x`, `-x`, `+y`, `-y`, `+z` or `-z` | The scene's own | The axis that the view keeps upright |
| `view_camera` | Orbit table or entity path | The scene's own | The camera that the panel opens on, as wire protocol 3.2 defines it |
| `include` | 1 to 64 entity path patterns | Every entity | The entities that the panel draws |
| `exclude` | 1 to 64 entity path patterns | None | The entities that the panel does not draw, after `include` |
| `labels` | Boolean | `true` | Whether the panel draws the labels of the entities |
| `axes` | Boolean | `true` | Whether the panel draws the axes and the ground grid |
| `scale_bar` | Boolean | `false` | Whether the panel draws a scale bar in the scene's units |
| `x` | Table | None | Extensions |

An orbit table has the keys `target`, `distance`, `azimuth` and `elevation`. In a pattern, `*` matches one path segment and `**` matches any number of segments. A scene that the script does not send makes the state "Source offline", as a `live:` source does in version 0. The validator gives the warning `spec.unresolved_source` for it, because a live scene can come later.

### Control Table

The array `control` holds the controls of the spec. A control is a widget that the person and the agent can use. It acts on one target.

| Key | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `id` | Id | Yes | | The control's id. Panels and controls share one set of ids. |
| `kind` | `slider`, `button`, `range`, `toggle`, `picker` or `text` | Yes | | The widget |
| `title` | Text, at most 256 characters | No | The `id` | The label of the widget |
| `bind` | Target string | Yes | | The one target of the control |
| `options` | Table | No | None | The options of the kind |
| `x` | Table | No | None | Extensions |

The `bind` string has one of four forms:

- `param:<name>` is a parameter that the script registered with `caliper.param`.
- `action:<name>` is an action that the script registered with `caliper.action`.
- `run:pause` is the pause and the resume of the run.
- `value` is a value that the spec holds. Binding references read it later.

A name has 1 to 63 bytes of UTF-8 and no control characters, as `caliper.param` requires. The prefix `intervene:` is reserved for interventions. The first version builds the kinds `slider` and `button`, and the GPU programs work builds `toggle` (see the GPU programs plan, Toggle Control). The kinds `range`, `picker` and `text` are part of the format, but the validator refuses them until they are built.

| Kind | Targets | Options |
|---|---|---|
| `slider` | `param:<name>`, `value` | `min`, `max`, `step`, `log`, `value` |
| `button` | `action:<name>`, `run:pause` | None |

A slider bound to a parameter takes the parameter's own range and scale when `min`, `max` or `log` is absent. The script clamps every value to its own range. Thus a slider can show a narrower range, but not a wider one. A slider bound to `value` starts at its option `value`, or at `min`. A button bound to `run:pause` pauses a run that runs and resumes a run that is paused.

### Control Layout

A control is a leaf of the layout, as a panel is. A layout leaf is an id, and the id names a panel or a control. The commands `layout.place` and `layout.set` accept control ids as well as panel ids. In a vertical split, a control leaf takes the height of one control row, and the other children share the rest of the height.

A spec with no `layout` places its panels as version 0 does. When the spec has controls, the default root is a vertical split of two children: the panels, then one horizontal split of the controls in declaration order. A single control is a leaf, not a split. The validator warns with `spec.unplaced_panel` for a control that the layout does not place.

### Validator Codes

Version 1 adds three error codes. The other errors of controls and scene panels use the codes of version 0.

| Code | Stage | Cause |
|---|---|---|
| `spec.unknown_control_kind` | Parse | A `kind` that is not one of the six kinds |
| `spec.unsupported_control` | Validate | A kind of the format that this Caliper does not draw yet |
| `spec.control_mismatch` | Validate | A target that the kind cannot take, or an option of another kind |

| Existing Code | New Use |
|---|---|
| `spec.bad_binding` | A `bind` that does not match the target grammar, or a scene slot that is not `live:<name>` |
| `spec.duplicate_id` | A control id that is also a panel id |
| `spec.unknown_panel` | A layout leaf that names no panel and no control |
| `spec.out_of_range` | A slider with `min` not below `max`, a `step` not above 0, or `log` with `min` not above 0 |
| `spec.too_many` | More than 64 controls, or more than 64 patterns |

## Commands

Eight commands join the command table. Each one becomes an MCP tool with the dot replaced by `_`, such as `events_wait`. The descriptors in `src/builder/commands/command_table.cpp` hold the argument schemas, the result schemas and one example for each command.

| Command | Effect | Mutates | Frame Thread | Function |
|---|---|---|---|---|
| `control.create` | `edit` | Yes | Yes | Adds a control to the open spec and places it in the layout |
| `control.patch` | `edit` | Yes | Yes | Changes a control with a JSON merge patch |
| `control.remove` | `edit` | Yes | Yes | Removes a control from the spec and the layout |
| `run.state` | `read` | No | Yes | Returns the run state, the parameters and the actions |
| `run.set_param` | `run` | No | Yes | Sets one parameter, applied at the next step |
| `run.action` | `run` | No | Yes | Queues one action for the next step |
| `run.pause` | `run` | No | Yes | Pauses, resumes or steps the run |
| `events.wait` | `read` | No | No | Returns the events after a cursor, and waits for them when none wait |

The commands `panel.create` and `panel.patch` accept the primitive `scene`, the slot `scene` and the scene options. The commands `layout.place` and `layout.set` accept control ids in their descriptions and their checks. The argument schemas of these commands did not change otherwise.

### Control Commands

The control commands are edits of the open spec. They follow the rules of the panel commands: a dry run, a diff, review for the origins that have review on, one undo step and a journal entry. `apply_spec_edit()` gives them to `apply_control_edit()` in `src/builder/session/control_edit.cpp`. Their own errors are `control.exists` and `control.not_found`.

`control.create` takes `id`, `kind`, `bind`, and optionally `title`, `options` and `place`. The argument `place` works as in `panel.create`, and its target can be a panel or a control. Without `place`, the control goes into the row of controls under the whole layout. The result holds `id`, `warnings`, `diff` and `applied`.

`control.patch` takes `id` and a merge patch of `kind`, `title`, `bind` and `options`. Its result holds the control as it is after the patch. `control.remove` takes `id` and also removes the control from the layout.

### Run Commands

The run commands act on the script that the app is attached to. They call the `RunControl` interface of `run_control.h`. A session without a `RunControl` answers them with `command.not_available`. This is the case for the stand-in host of the tests and for `caliper-mcp` without the app.

`run.state` returns `attached`, `script`, `control`, `why_not`, `state`, `step`, `steps_left`, `params` and `actions`. The state is `none`, `running`, `pausing`, `paused` or `stepping`, as the CONTROLS message gives it. Each parameter has its type, value, pending value, range, scale, choices and help.

`run.set_param` takes `name` and `value`. The value applies at the script's next step, and the result says `pending: true`. When the script applies the value, a `run` event of change `param` carries the value after the clamp.

`run.action` takes `name`. It answers at once with `queued: true`, because the command runs on the frame thread and cannot wait for the script. The text that the action returns comes back as a `run` event of change `action`.

`run.pause` takes `op`, which is `pause`, `resume` or `step`, and `count` for `step`. The result gives the run state at the time of the request.

### Run Effect Class

The effect `run` is a new effect class beside `read`, `edit` and `change`. A `run` command acts on the running script. It changes no workspace state, it has no undo, and the journal does not record it. The descriptor of a `run` command thus has `mutates` and `destructive` false. The MCP tool table gives it `readOnlyHint` false, because it changes the run.

The session's policy asks the person about a `run` command from the origins `agent` and `mcp`. The origins `ui`, `builder` and `api` run it at once. The agent modes of `permissions.h` then apply as follows:

| Mode | `read` | `edit` | `run` | `change` |
|---|---|---|---|---|
| Ask | Runs | Asks when review is on | Asks | Asks when destructive |
| Accept edits | Runs | Runs | Asks | Asks when destructive |
| Read only | Runs | Dry runs and previews only | Refused | Refused |
| Allow all (side chat only) | Runs | Runs | Runs | Runs |

A rule of the session or a saved rule allows a `run` command without a card. The key of a rule is `permission_rule_key()`. For `run.set_param` and `run.action`, the key is the command and its target, such as `run.set_param:lr`. Thus a rule allows one parameter or one action, and not the whole command. For `run.pause`, the key is the command name.

### Approval Card

The card of a `run` command states the change in one line and the running script that it acts on. Its buttons are Allow, Allow for this session, Always allow and Deny. The two rule buttons name the target, such as "Always allow changes of lr".

| Command | Card Text |
|---|---|
| `run.set_param` | Set parameter `<name>` to `<value>` in `<script>`. The change applies at the next step. |
| `run.action` | Run action `<name>` in `<script>` at the next step. |
| `run.pause` with `pause` | Pause `<script>` at the next step. |
| `run.pause` with `resume` | Resume `<script>`. |
| `run.pause` with `step` | Run `<count>` steps of `<script>`, then pause. |

### Run Command Errors

Each error has a code and one plain sentence. The codes are constants in `run_control.h`.

| Code | Cause |
|---|---|
| `command.not_available` | The session has no run control, because the app does not run |
| `run.not_attached` | No script is attached to the app |
| `run.control_not_allowed` | The viewer connection takes no control. The message gives the reason that the connection gives. |
| `run.unknown_param` | The script has no parameter of this name |
| `run.unknown_action` | The script has no action of this name |
| `run.bad_value` | The value has the wrong type, or it is not one of the choices |
| `run.no_step` | A pause or a step before the script's first step |
| `run.refused` | The script or the hub refused the request, with its message |

## Run Control in App

The app sends each run request as a protocol 3.1 CONTROL message through its own viewer connection. It does not open a second connection, and the agent never talks to the script. The request uses `caliper_live::RunControls` of the attachment, as the run bar does. The fields are those of `docs/reference/run-control-messages.md`:

| Command | CONTROL |
|---|---|
| `run.set_param` | `{op: "set", name, value}` |
| `run.action` | `{op: "run", name}` |
| `run.pause` | `{op: "pause"}`, `{op: "resume"}` or `{op: "step", count}` |

The app may send CONTROL only when its connection allows it. The stream must announce the capability `controls`, WELCOME must carry `control: true`, and the session must list `controls`. A script allows control when it starts its server with `caliper.serve(control=True)`, which is the default. A hub allows control when the person starts it with `--allow-control`. Thus the person decides at the start of the run, and the agent cannot change it.

When no script is attached, each run command fails with `run.not_attached`. When the connection allows no control, each run command fails with `run.control_not_allowed`. Its message is the reason that `RunControls::why_not()` gives. The shared memory session of `caliper.show` takes no control (`live_source.h`). Thus the run control needs the network connection that `caliper.watch` starts.

A control widget of the spec sends its value through `RunControl` directly, and not through `run.set_param`. Thus the person's own slider shows no approval card.

The app implements `RunControl` in a class of its own that wraps the attachment's `RunControls`. It gives the class to `Session::set_run_control()` when the session opens, and it takes it back with null before the attachment closes.

## Events

Events go to the agent through one read command, `events.wait`. The app records each event in a ring, and the command reads the ring after a cursor.

### Event Ring

The ring of `events.h` keeps the last 1,024 events. Each event has a number `seq`, which grows by one with each event, from 1. A push is safe from any thread. Hovers never enter the ring.

`events.wait` takes `cursor`, `kinds`, `panels`, `timeout_s` and `limit`. The cursor is the `seq` of the last event that the agent looked at. A cursor of 0 reads from the oldest event that the ring keeps. The filter `kinds` keeps the events of those kinds. The filter `panels` keeps the events of those panels and controls. An event without a panel never matches a filter of panels.

The result holds `events`, `cursor`, `lost` and `more`. The new cursor is past every event that the call looked at, also the events that did not match. Thus the next call does not read them again. `lost` counts the events after the given cursor that the ring dropped before the call, of any kind. `more` is true when more matching events wait beyond `limit`.

### Long Poll

The call answers at once when matching events wait. Otherwise it holds the request until a matching event comes or `timeout_s` ends. The default of `timeout_s` is 10 seconds, the maximum is 60 seconds, and 0 answers at once.

The handler never waits, because the command queue runs it on the frame thread. Instead, `wait_for_events()` holds the thread of the caller. It submits the call to the queue, and it waits on the ring when the answer holds no event. When a matching event comes, it submits the call again with `timeout_s` 0. Caliper API serves `POST /v0/commands/events.wait` through this function, on a thread of its own for each call. `Session::close()` waits until each such thread has left the function. The HTTP client of `caliper-mcp` thus waits up to 60 seconds plus a margin for this command.

At most 8 calls of `events.wait` wait at once, because each held call takes one of the 32 command slots of Caliper API for up to 60 seconds. Thus 24 slots stay free for other commands. When 8 calls wait already, a call that finds no matching event fails at once with the code `events.too_many_waits`. A call with `timeout_s` 0 still answers. The bound is `kMaxEventWaits` in `events.h`.

A long poll works with each MCP client and each protocol revision that `caliper-mcp` serves. The resource `caliper://events` with `resources/subscribe` is a later, additive path.

### Event Format

Each event is one JSON object. Six keys are common to all kinds, and the kind adds its own keys. The schema `event` of the command table lists every key.

| Key | Type | Meaning |
|---|---|---|
| `seq` | Integer | The event's number |
| `time` | String | The time of the push, UTC ISO 8601 with milliseconds |
| `kind` | String | `pick`, `select`, `control`, `run`, `spec` or `approval` |
| `panel` | String or null | The spec panel or control of the event |
| `view` | String or null | The live view of a pick or a selection, such as the scene's name |
| `origin` | String | `ui` for the person, `script` for the running script, else the origin of the command |

### Event Kinds

| Kind | Cause | Keys |
|---|---|---|
| `pick` | A click on an item of a scene | `entity`, `primitive`, `index`, `row_id`, `position`, and `pixel` for an image |
| `select` | A selection in a scene, made or cleared | `entities` (each with `entity`, `primitive`, `indices`, `row_ids`), `truncated` |
| `control` | The person finished an edit of a control of the spec | `control`, `bind`, `value` |
| `run` | The attached script changed | `change`, then the keys of the change |
| `spec` | A command changed the open spec | `command`, `event`, `applet` |
| `approval` | A card asked or was decided | `approval`, `command`, `state`, `target` |

The keys of a pick and a selection are those of the scene view's events in `src/host/live/live_scene.h`. A pick in a scene panel has the panel id in `panel` and the scene name in `view`. A pick in a live view outside a spec has `panel` null.

| Change of `run` | Keys | Cause |
|---|---|---|
| `state` | `state`, `step` | The run state changed |
| `param` | `name`, `value`, `step` | The script applied a parameter at a step |
| `action` | `name`, `status`, `text` | An action ended with `ok` or `error` |
| `attach` | `attached`, `script` | A script was attached or detached |

The state of an `approval` event is `pending`, `approved`, `denied` or `expired`. The key `target` names the parameter or the action of a `run` command.

### Event Feeds

Each kind has one producer, and each producer pushes through the `EventSink` interface:

- The scene panel of the spec runtime pushes `pick` and `select` with its panel id.
- The live views outside a spec push `pick` and `select` without a panel.
- The control widgets push one `control` event for each finished edit: at the end of a slider's drag or typed entry, and at a button's click. A slider sends each new value to the script while the person drags it, but the ring gets only the last value.
- The app's run control pushes `run` when the CONTROLS message or a CONTROL_REPLY shows a change.
- The session pushes `spec` from its journal through `note_journal_event()`.
- The session pushes `approval` from the approval observer of the command registry: when the registry parks a call, when the person decides and when the wait ends. This includes the `run` commands, which the journal does not record.

The ring lives in the session, so that `events.wait` works the same way in the app and in a headless session. The spec runtime receives the ring and the run control through `SpecInstrumentHost` in `spec_instruments.h`.

## Watch Hook

The acceptance test allows one Caliper line in the training script. The line is `caliper.watch(model, optimizer)`.

### Current Watch

Today, `caliper.watch(model, activations=True, gradients=False, every=1, include=None, rate=None, groups=True)` takes no optimizer. It is in `python/caliper/_watch.py` and returns a `WatchHandle`. It adds a forward hook to each selected module, which publishes the view `<label>.act`. With `gradients=True`, it publishes `<label>.<param>.grad`. With `groups=True`, it offers the grouped views `activations`, `weights` and `gradients`. It publishes no series, no loss, no step and no scene, and it registers no parameter and no action.

### Watch Extension

The signature becomes `caliper.watch(model, optimizer=None, *, activations=True, gradients=False, every=1, include=None, rate=None, groups=True, scene=True)`. Without an optimizer, the function behaves as today. With an optimizer, it also does these things:

1. It starts the stream server with `caliper.serve()` when no server runs, so that the app can attach with control.
2. It adds a step pre-hook to the optimizer (`register_step_pre_hook`), which is the checkpoint of the run.
3. It registers the parameter `lr` and publishes the series `loss` and `lr` and the scene `model`.

At the checkpoint, the hook first calls `caliper.step()`. Pending parameters then apply, queued actions run, and a pause waits there. Then the hook writes the parameter `lr` into the optimizer's parameter groups. Thus a new learning rate applies to the same optimizer step, and a pause holds the run before the weights change. A script that passes an optimizer must not also call `caliper.step()`.

### Published Names

| Name | Kind | Content |
|---|---|---|
| `loss` | Series | The loss of each optimizer step |
| `lr` | Series | The learning rate of parameter group 0 at each step |
| `lr` | Parameter | A number on a log scale, from the start value divided by 1,000 to the start value multiplied by 10 |
| `model` | Scene | One box for each watched module, in the order of `WatchHandle.modules` |
| `<label>.act` | Tensor view | The output of each watched module, as today |
| `activations`, `weights`, `gradients` | Grouped views | As today |

The parameter `lr` is the learning rate of parameter group 0. A new value scales every group by the same factor, so that the ratios between the groups stay. The loss is the last 0-d tensor that `backward()` ran on before the step. When the hook sees no such tensor, it uses the value that `optimizer.step(closure)` returns.

The scene `model` has one entity, `modules`, of boxes. The boxes stand along +y in module order. The size of a box shows the parameter count of its module, and its color shows the mean absolute value of the module's last output. The row id of a box is the index of its module in `WatchHandle.modules`. Thus a pick on a box names the module.

## Acceptance Test

The instrument acceptance test starts a script whose only Caliper line is `caliper.watch(model, optimizer)`. It then starts the hidden app with `--url` and the script's stream link, and it drives `caliper-mcp` as a child process over stdio. Through the MCP tools, it creates a scene panel bound to `live:model` and a slider bound to `param:lr`. It changes the parameter with `run_set_param` and reads the new value from the script's output at its next step. A pick in the scene panel, made through the app's test seam, comes back from `events_wait`.

## Open Points

- A learning rate scheduler sets the rate again at each of its steps. It is open whether `lr` scales the scheduler's base rates or the current rates.
- It is open whether the shared memory session also carries the views when the server runs, so that no view appears twice in the app.
