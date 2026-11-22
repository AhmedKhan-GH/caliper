# Instrument Interface

This page defines the interface layer of an instrument that an agent builds through `caliper-mcp`. A scene panel draws entities from the spec's own sources, and two built-in operators, `surface` and `path`, turn tensors into geometry. Named state values, which controls and picks write, steer what the panels show through binding references. One time cursor moves several panels over the recent steps that the app keeps. Bound texts, annotations and markers explain the values, and a layout of shares with minimums holds the panels and the controls.

The page states the spec additions, the validator codes, the commands, the events, the place of each piece in the app or in the script, the tests and what this work leaves out. It extends `design/instruments.md`, which defines spec version 1, the controls and the events, and `design/gpu-programs.md`, which defines programs, the sandbox, the cadence and the wire path. The scene model is in `design/scenes.md`, and the runtime in the script's process is in `design/producer-runtime.md`.

## Status

The features are built and joined. State and references, the step history with the cursor and the transport, the operators with their programs in the script, the scene entities with the data box, and the annotations, bound texts and layout shares work together in the app, through the eighteen commands and `caliper-mcp`. The validator still refuses a few keys with `spec.not_built`, because the code that would read them is later work: annotations on heatmap and image panels, the option `legend` set to `true` or `false` outside a series panel, the option `scale_bar` outside a scene, and the option `choices_from` of a picker.

## Spec Additions

All additions join spec version 1, as the programs did, because version 1 has not reached `main` yet. Each addition raises the required version to 1. The table lists the new keys and where they stand.

| Key | Place | Content |
|---|---|---|
| `state` | Top level | An array of state variables, at most 64 |
| `history` | Source | The number of recent steps that the app keeps for the source |
| `window`, `surface`, `path` | Transform `op` | A step over the kept steps, and the two operators |
| `entity` | Scene panel | An array of entities that the panel draws from sources, at most 64 |
| `annotation` | Panel | An array of labels, thresholds, bands and markers, at most 64 |
| `select`, `legend`, `box`, `views` | Panel options | A pick that writes state, a legend, the data box of a scene and its saved views |
| `format`, `fields` | Text panel and readout options | A bound text and its values |
| `segmented`, `readout`, `transport` | Control `kind` | New control kinds beside the built `range`, `picker` and `text` |
| `state:<id>`, `cursor` | Control `bind` | New control targets |
| `shares`, `min` | Layout | The shares of a split and the minimum of a leaf |
| `params` | Program and stage | Named numbers that a program or a stage reads in a uniform |
| `tensor` | Program output `kind` | An output that the script publishes as a live tensor |

A binding reference can stand at some of these keys and at some keys that exist. The section Binding References lists the places.

## Binding References

A binding reference is a small table that stands where the spec takes a literal value. The app evaluates it, and the script never sees it. A reference thus changes what a panel shows at once, also while the run is paused.

### Reference Forms

The three forms share one grammar. Exactly one of the keys `control`, `selection` and `state` names the input.

```text
{ control = "<control id>" [, field = "lo" | "hi"] [, default = <literal>] }
{ selection = "<panel id>", field = "<pick field>" [, entity = "<entity path>"] [, default = <literal>] }
{ state = "<state id>" [, field = "<field>"] [, default = <literal>] }
```

| Form | Reads |
|---|---|
| `control` | The current value of a control. A range gives `[lo, hi]`, or one end with `field`. A control bound to `param:<name>` gives the last value that the script reported. |
| `selection` | One field of the last pick in the panel, as the section Pick Fields lists them. With `entity`, only a pick on that entity counts. |
| `state` | The current value of a state variable. A range state takes `field` with `lo` or `hi`, and the built-in state `cursor` takes the fields of the section Cursor. |

A reference without a value, such as a selection before the first pick, gives its `default`. Without a `default`, a step waits and its panel shows the state "Waiting for input", an option keeps its own default, and a bound text shows a dash. A `default` is a literal of the type that the place takes, and a reference never holds another reference.

### Reference Places

A reference stands only where its value changes data and never changes the shape of the spec. Thus the validator checks the spec once, and a change of a value never makes a spec invalid.

| Place | Keys | Type |
|---|---|---|
| `slice` step | `index`, `start`, `stop`, `step` | Integer |
| `window` step | `start`, `stop`, `steps` | Integer, a step number or a count |
| Panel options | `range`, `x_range`, `y_range` | A pair, or a range reference |
| Panel and entity options | `colormap` | Text, from a state whose choices are colormap names |
| Entity options | `vmin`, `vmax`, `highlight`, `visible` | Number, item or boolean |
| Slot binding | Any slot | Text, from a state whose choices are source ids |
| Annotation | `value`, `range`, `at` | Number, pair or position |
| Bound text | Each entry of `fields` | Any |
| Control options | `min`, `max` of a slider or a range | Number |
| Program and stage | Each entry of `params` | Number |

Every other key takes a literal: ids, kinds, `from`, `catalog`, the axes of a step, the parameters of an operator, `history`, poses and the layout. A reference there is `spec.reference_not_allowed`.

### Re-Evaluation

The app builds a dependency index when a spec opens and after each edit. The inputs of the index are the value of each control, each state variable, the last pick of each panel, the cursor and the generation of each source. Its readers are steps, options, slots, entities, annotations, bound texts and the `params` of programs and stages.

A change of an input marks its readers. At the next frame, each marked reader evaluates again, at most once. A slider drag thus costs one evaluation for each frame, and the last value wins. The cache key of a source chain holds the values of the references that the chain reads, and the cursor step when the chain reads kept steps. A chain that no change touched thus hits the cache. In a chain that a change touched, the steps before the first step that reads the changed input keep their cached results.

A changed program parameter goes to the script in the message `PROGRAM` with `op = "params"`, and it applies at the next pass. A changed stage parameter goes to the uniform of the stage in the viewer at once.

### Reference Rules

The validator checks each reference against the spec. It refuses a reference with these codes:

| Code | Meaning |
|---|---|
| `spec.bad_reference` | No input key or two of them, an unknown key, a nested reference, or a `default` of the wrong type |
| `spec.reference_not_allowed` | A reference at a key that takes only a literal |
| `spec.unknown_control` | The control id is not a control of the spec |
| `spec.unknown_state` | The state id is not a state variable of the spec, and not `cursor` |
| `spec.unknown_panel` | The panel of a selection is not a panel of the spec |
| `spec.unknown_field` | The field does not exist for that control kind, state type or pick |
| `spec.reference_type` | The type of the value does not fit the place |
| `spec.reference_cycle` | A control's options read, directly or through state, the value that the control writes |
| `spec.reference_in_history` | A `window` reads a source whose own steps read a reference |

Evaluation writes a value in one case only: a range control with `follow` moves its value with its bounds. Thus the validator builds a graph with an edge from each input to its readers and from each control to the value that it writes. A cycle in this graph is `spec.reference_cycle`, and its message names each item on the cycle. A pick writes state only on a person's input or a command, never during evaluation, and so a pick adds no edge.

## Shared State

A state variable is a named value of the spec. Controls and picks write it, and panels read it through references to filter, highlight, recolor or retarget. The declaration is spec content and the spec saves it. The current value is session state: it starts at the default when the spec opens, and the spec never saves it.

### State Table

```toml
[[state]]
id = "token"
title = "Token"
type = "integer"
default = 0
min = 0
max = 63

[[state]]
id = "measure"
type = "text"
default = "norm"
choices = ["norm", "mean"]
```

| Key | Type | Required | Meaning |
|---|---|---|---|
| `id` | Id | Yes | Unique among state variables. The id `cursor` is reserved. |
| `title` | Text | No | The label that a control or a legend shows. The id when absent. |
| `type` | `number`, `integer`, `boolean`, `text`, `range` or `list` | Yes | The type of the value. A `list` holds texts. |
| `default` | Value of the type | No | 0, `false`, the first choice or an empty text, `[min, max]`, or an empty list |
| `min`, `max` | Number | No | The bounds of a number, an integer or a range |
| `choices` | 1 to 1,024 values | No | The allowed values of a text, an integer or the items of a list |
| `x` | Table | No | Extensions |

A text value has at most 1,024 bytes, and a list has at most 256 items. A default outside the bounds or the choices is `spec.out_of_range`. Two state variables with one id are `spec.duplicate_id`.

### State Writers

Three things write a state variable:

- A control with `bind = "state:<id>"`. The section Control Kinds lists which kinds write which types, and a kind that does not fit the type is `spec.control_mismatch`.
- A pick in a panel with the option `select = { state = "<id>", field = "<pick field>" [, entity = "<path>"] }`. Each pick writes that field of the pick.
- The command `state.set`.

Two controls that write one state variable give the warning `spec.state_writers`, because one setting has one control. A pick and a control may write one state variable, because a pick is a selection and every view shows the same selection.

### State Readers

A reader names the state variable in a reference. The four uses are these:

- **Filter.** A `slice` step reads `{ state = "token" }` as its index.
- **Highlight.** An entity option `highlight = { state = "token", along = "column" }` draws the items with that index in the annotation color.
- **Recolor.** An option `colormap = { state = "palette" }` reads a text state whose choices are colormap names.
- **Retarget.** A slot reads a text state whose choices are source ids, so that the panel shows the chosen source.

```toml
[[panel]]
id = "detail"
primitive = "heatmap"
bind = { values = { state = "measure" } }
```

A retarget state whose choices are not all source ids is `spec.reference_type`.

### State Events

A changed value adds an event of the kind `state` with the keys `state`, `value`, `previous` and `writer`. The writer is a control id, a panel id or the command name. A slider drag adds one event when the edit ends, as a control event does.

## Scene Entities

A scene panel draws entities that it builds from spec sources. Each entity is a batch of the scene model, written by the app into a scene that belongs to the panel. The script draws nothing for it, and the spec builds a terrain from a live tensor with no scene call in the script.

### Entity Table

```toml
[[panel]]
id = "world"
primitive = "scene"
title = "Activation Terrain"
options = { up = "+z", box = [3.0, 1.0, 0.8], x_label = "Step", y_label = "Token", z_label = "Norm", select = { state = "token", field = "column", entity = "terrain" } }

[[panel.entity]]
path = "terrain"
draw = "surface"
bind = "terrain"
options = { colormap = "viridis", highlight = { state = "token", along = "column" } }

[[panel.entity]]
path = "loss"
draw = "lines"
title = "Loss"
bind = { x = "loss_window.step", z = "loss_window.value" }
options = { mode = "strip", y = "max", own = ["z"], width_px = 2.0 }
```

| Key | Type | Required | Meaning |
|---|---|---|---|
| `path` | 1 to 8 segments of `[a-z0-9_-]`, joined by `/` | Yes | Unique in the panel |
| `draw` | `points`, `lines` or `surface` | Yes | The draw kind |
| `title` | Text | No | The name in the legend and the tooltip. The path when absent. |
| `bind` | Source id, or a table of slots | Yes | A source id binds each named array of a geometry source to the slot of the same name |
| `options` | Table | No | The options of the draw kind |
| `pose` | `{ position, rotation, scale }` | No | A translation, a quaternion `[x, y, z, w]` and a scale, as literals |
| `x` | Table | No | Extensions |

The entities stand under the reserved root `spec/` of the panel's scene, so that a stage names one as `spec/<path>`. A scene panel with at least one entity needs no `scene` slot. When it also binds a live scene, it draws both in one view with one camera.

### Draw Kinds

This work delivers three draw kinds. `points` and `lines` exist in the scene model, and `surface` is new.

| Draw | Slots | Options |
|---|---|---|
| `points` | `positions` (N, 3), or `x`, `y`, `z` (N) each; `values` (N); `colors` (N, 4) bytes or (N) packed; `radii` (N) | `colormap`, `vmin`, `vmax`, `size_px`, `shape`, `x`, `y`, `z`, `own`, `depth_offset`, `highlight`, `visible` |
| `lines` | As `points`, without `radii` | As `points`, plus `mode` (`strip` or `segments`) and `width_px`, without `size_px` and `shape` |
| `surface` | `height` (R, C), required; `values` (R, C); `x` (R); `y` (C) | `colormap`, `vmin`, `vmax`, `depth_offset`, `highlight`, `visible` |

Positions need either `positions` or at least two of `x`, `y` and `z`. An axis without a slot takes the option of the same name: a number in data units, or `min` or `max` of the data box. Thus `y = "max"` puts the loss line on the back wall of the box. The option `own` lists axes that map the entity's own data range onto the box, and the data box draws their ticks on the wall that the entity stands on.

The `surface` draw kind is a height field. The viewer builds the triangles on its own GPU from the grid: a fixed index buffer for R by C cells, and a vertex stage that reads `height`, `x` and `y`. A cell with a NaN height is a hole. The color comes from `values`, else from `height`, through the colormap. A material stage reads the color value as `value`, as for other kinds.

### Array Resolution

The app resolves the arrays of an entity in four steps:

1. It resolves the source chain to its value at the cursor, as the section Latest at the Cursor states.
2. It checks the shape of each array against its slot. Per-item slots must have one length. The app converts a float16, bfloat16, float64 or integer array to float32.
3. It writes one batch for the entity into the panel's scene through the batch path of `caliper.scene`. In process, the batch keeps the zero-copy route of the scene core where one exists. Otherwise it costs one copy to the device.
4. It writes the batch again when the source has a new generation, when the cursor moves or when a reference that the chain reads changes. It writes at most once a frame, and the last value wins.

A wrong shape or length does not stop the panel. The entity shows nothing, the panel shows the state "Wrong shape" with the entity's path, and `panel.data` gives the code `entity.shape` or `entity.length`.

### Pick Fields

A pick on an entity gives a set of fields. The event `pick` carries them in its new key `fields`, a `selection` reference reads one of them, and the option `select` writes one into state.

| Field | Meaning | Draw kinds |
|---|---|---|
| `entity` | The entity's path | All |
| `index` | The item, or the vertex nearest the pick on a line | `points`, `lines`, live entities |
| `row_id` | The row id of the item, when its source has one | `points`, `lines`, live entities |
| `row`, `column` | The cell of the grid, as indices of the input before pooling | `surface` |
| `x`, `y`, `z` | The picked position in data units | All |
| `value` | The color value at the pick | All with values |
| `step` | The step of the row, when `x` came from a `window` | `surface`, `lines` |

### Entity Limits

| Limit | Value |
|---|---|
| Entities in one panel | 64 |
| Items of one `points` or `lines` entity | 1,048,576 |
| Cells of one `surface` entity | 262,144, and at most 4,096 on one side |
| Arrays of one panel | 64 MiB |

An entity over a limit shows the state "Over limit" and draws nothing.

## Operators

An operator is a step of a source, like the shipped steps `slice`, `reduce` and `normalise`. Its input is a tensor, and its output is geometry: named arrays that the slots of an entity bind. Its parameters are literals. An operator thus composes after the shipped steps and after a `window`, and an entity binds its result. A shipped step after an operator is `spec.bad_transform`, because geometry is not a tensor.

```toml
[[source]]
id = "terrain"
from = "act"
transform = [
  { op = "window", start = { control = "window", field = "lo" }, stop = { control = "window", field = "hi" } },
  { op = "surface", max_cells = 32768, pool = "mean" },
]
```

### Surface Operator

`surface` makes a height field from a tensor of rank 2, (R, C). Its parameters are `max_cells`, from 1 to 262,144 with a default of 32,768, and `pool`, one of `mean`, `max` and `min` with a default of `mean`.

When R times C is above `max_cells`, the operator pools blocks of f by f cells. The factor f is the smallest integer for which the pooled grid fits. A pooled cell skips NaN values, and a block of NaN values gives NaN. The output arrays are these:

- `height`, (R', C'): The pooled values.
- `x`, (R'): The mean coordinate of the pooled rows. After a `window` it is the step, else the row index.
- `y`, (C'): The mean column index of the pooled columns.

A tensor of another rank is `spec.operator_shape` when the catalog gives its shape, and the panel state "Wrong shape" otherwise.

### Path Operator

`path` makes a polyline through the rows of a tensor (R, D), with D at least 2. Its parameters are `columns`, two or three column indices with a default of the first three, and `max_points`, from 2 to 1,048,576 with a default of 65,536.

When R is above `max_points`, the operator keeps every s-th row, with the smallest s that fits. The output arrays are `positions`, (P, 3), and `values`, (P). Two columns give a `z` of 0. The value of a row is its step after a `window`, else its row index. A `lines` entity with `mode = "strip"` draws the path, and a `points` entity draws its vertices.

### Operator Placement

A source chain runs in two places. An operator runs in the script when it is the first step of a source whose root is a live tensor. Every other step runs in the app. In this work an operator takes a tensor and gives geometry, and so the part in the script is at most one operator.

In the script, the operator is a built-in program on the producer runtime. It reads the live tensor on the script's device through the same import as an agent's program, under the same sandbox, the same shared cadence and the same watchdog. Only its pooled result travels. In the app, an operator runs its CPU reference. The inputs there are small, because they come after a reduction, a slice or a window, and the app's own GPU compute of steps is later work.

Each operator has one WGSL source and one CPU reference in the builder library. The tests compare the device result with the reference.

### Operator Programs

The app turns a placed operator into a program declaration and sends it in the message `PROGRAM`, as `design/gpu-programs.md` defines. The script runs it like any program. The table states what differs from an agent's program and what stays the same.

| Aspect | Agent program | Operator |
|---|---|---|
| Spec | A `[[program]]` table with WGSL, inputs, an output and limits | A step `{ op = "surface", ... }` of a source, with no WGSL |
| Id | The program's id | `operator.<source>.<step index>`, which no program id can match |
| WGSL | From the spec | From Caliper's built-in file, the same in every spec |
| `PROGRAM` metadata | The declaration | The declaration, plus `builtin` with the operator's name and version |
| Output | `geometry` to an entity, or `pixels` to an image view | `tensor`, published as the live view of its id, which the app reads as the source's value |
| Review | A program of a foreign spec waits for a review of its hash | Never, because the app knows the hashes of its built-in sources |
| Edits | The program commands | The source commands |
| Sandbox, cadence, watchdog | As designed | The same |
| Status and events | `program.status` and the event kind `program` | The same, with `kind = "operator"` and the source's id |

The app computes the output shape and the workgroups from the shape that the catalog gives. When the catalog shape changes, the app sends the operator again.

### Tensor Output

A program gains the output kind `tensor`, for operators and for agents. The declaration is `output = { kind = "tensor", shape = [16] }`, with 1 to 4 axes and at most 1,000,000 elements. The WGSL declares `var<storage, read_write> out: array<f32>`. After each pass, the script publishes the first elements of `out` as a snapshot of the live view `program.<id>`, with the step's stamp. An agent's program can thus write one row of a statistic at each pass into a source that keeps its steps.

### Program Parameters

A program and a stage gain the key `params`: at most 16 names, each a number or a reference.

```toml
[[stage]]
panel = "world"
entity = "spec/terrain"
kind = "material"
params = { level = { control = "level" } }
wgsl = """
fn material(in: MaterialIn) -> vec4<f32> {
  if (abs(in.value - params.level) < 0.01) { return vec4<f32>(1.0, 0.6, 0.2, 1.0); }
  return in.color;
}
"""
```

A program declares `var<uniform> params: Params`, and `Params` is a struct with one `f32`, `i32` or `u32` field for each name. The viewer adds the same declaration to the prelude of a stage. The sandbox refuses a struct whose fields differ from the names with `program.params_mismatch` or `stage.params_mismatch`. A changed value needs no compile and no new review, because the hash covers only the WGSL. The script applies new values at the next pass, and the viewer applies them at the next frame.

## Controls

This work builds the control kinds that version 1 reserved, and three new kinds. The control commands of `design/instruments.md` edit them all.

### Control Kinds

| Kind | Bind | Options | Value |
|---|---|---|---|
| `slider` | `param:`, `value`, `state:` of a number or an integer | `min`, `max`, `step`, `log`, `value` | Number |
| `range` | `value`, `state:` of a range | `min`, `max`, `step`, `value`, `min_span`, `follow` | `[lo, hi]` |
| `toggle` | `intervene:`, `param:`, `value`, `state:` of a boolean | `value` | Boolean |
| `button` | `action:`, `run:pause` | None | None |
| `picker` | `value`, `state:` of a text or an integer | `choices` or `choices_from`, `value` | Text or integer |
| `text` | `value`, `state:` of a text | `placeholder`, `max_length`, `value` | Text |
| `segmented` | `value`, `state:` of a text, an integer or a list | `choices`, `multiple`, `value` | Text, integer or list |
| `readout` | None | `format`, `fields` | None |
| `transport` | `cursor` | `speed`, `keys` | The cursor |

A choice is a value, or a table `{ value, label, key }`. `choices_from = "<source>.<column>"` reads the distinct values of a column, at most 1,024. A text control applies its value when the person presses Return or leaves the field. A range with `follow = true` moves with its upper bound while its upper end equals it, and keeps its span. A range thus follows the newest step until the person moves it. A control bound to a state shows the state's value, so that `state.set` moves it too.

```toml
[[control]]
id = "window"
kind = "range"
title = "Steps"
bind = "value"
options = { min = { state = "cursor", field = "first" }, max = { state = "cursor", field = "latest" }, value = [0, 2048], min_span = 16, follow = true }
```

### Segmented Groups

A segmented group is a row of 2 to 16 buttons. With `multiple = false`, one button is on, and the value is its choice. With `multiple = true`, any number of buttons are on, and the value is the list of their choices. A segmented group draws in the house style, with square corners and the accent color on the buttons that are on.

```toml
[[control]]
id = "measure"
kind = "segmented"
bind = "state:measure"
options = { choices = [{ value = "norm", label = "Norm", key = "n" }, { value = "mean", label = "Mean", key = "m" }] }
```

### Keyboard Shortcuts

A choice of a segmented group may have a key. The key selects the choice, or turns it on or off in a group with `multiple`. The tooltip of the button shows the key.

```text
key = [ "shift+" ] ( "a" .. "z" | "0" .. "9" )
```

The shortcuts of an instrument act only while its window has focus and no text field holds the keyboard. The app's own shortcuts always win. The grammar refuses chords with Command, Control or Option. The validator refuses a key that the app reserves with `spec.reserved_shortcut`. The reserved keys are Space, the arrow keys, Home, End, Escape, Tab, Return and Delete, and each single key that a view of the app uses. One list in the builder library holds them, and a test compares it with the keys that the app registers. Two choices with one key are `spec.duplicate_shortcut`.

### Control Values

The command `control.set` sets the value of a control bound to `value` or to a state, as a person's edit does. It adds the same `control` event, with the origin of the caller. A control bound to a parameter takes `run.set_param`, and `control.set` refuses it with `control.not_settable`. The event `control` gains the key `key` when a shortcut made the change.

## Time Transport

The time transport moves one cursor over the recent steps that the app keeps. Every panel that reads kept steps shows the value at the cursor. The cursor changes only the view, and the run goes on.

### Step History

The app keeps the recent values of a source in its own memory, in the builder session. The script keeps nothing for it.

```toml
[[source]]
id = "act"
catalog = "live:blocks.0.attn.proj.act"
history = 2048
transform = [
  { op = "reduce", fn = "norm", axis = 2 },
  { op = "reduce", fn = "mean", axis = 0 },
]
```

- **Value.** The app keeps the value after the longest run of the source's steps from the start that read no reference. A source without references thus keeps its final value.
- **Steps.** The app keeps each value that it receives, with the step of the stamp of the snapshot, or else the run's step at arrival. It keeps one value for each step, and a later value of a step replaces the earlier one. A program's cadence of N steps gives values N steps apart. A watched view publishes at most 60 times a second, and only at the passes of the shared cadence that keeps everything viewers ask of the run inside 4 % of its time (`design/gpu-programs.md`, Views in the Cadence), so a fast run keeps steps with gaps between them. A live series keeps every step, because the script moves the values of all steps at each pass.
- **Stamps.** No snapshot carries a step stamp today: the script's sender writes a sequence number and a time only, although the wire's `timelines` capability carries stamps. This work makes the sender stamp each snapshot with the run's step, and the live catalog keep the step of each view's newest snapshot.
- **Depth.** `history` is from 0 to 65,536 steps. A source whose root is a live tensor and that has no `from` keeps 64 steps by default. Other sources keep none by default. A series already keeps its own ring and needs no `history`.
- **Budget.** All kept steps of the open spec share 512 MiB, and one source keeps at most a quarter of it, 128 MiB. When the requested steps need more, the app shortens the largest rings first, and `cursor.get` gives the depth that each source keeps. A source that keeps fewer steps than it requests has a `note` in `cursor.get`, which names the source, the size of one step and the two limits, and suggests a reduce step before the kept value. A raw activation of 512 by 4096 values thus keeps 16 steps and leaves the rest of the budget to the other sources.
- **Reset.** A new attachment, a change of the source's chain and the opening of a spec clear the source's kept steps.

`history` on a source whose root is not a live tensor is `spec.history_not_allowed`, and a value outside its range is `spec.out_of_range`.

### Window Step

The step `window` stacks the kept steps of another source. It is the first step of a source, and it reads the kept steps of the source that `from` names. That source keeps all of its steps, or the validator gives `spec.no_history`. A window may also read a live series, whose ring holds its steps.

| Key | Type | Meaning |
|---|---|---|
| `start`, `stop` | Integer step numbers or references | The first and last step, both included |
| `steps` | Integer from 1 to 65,536, or a reference | The last N kept steps up to the cursor |

`steps` excludes `start` and `stop`, and both together are `spec.bad_transform`. Without a key, the window holds every kept step. The window never holds a step after the cursor. A window over a tensor source gives a tensor (W, ...) in step order and the array `step`, (W). Its columns `step` and `value` bind the slots of an entity, such as `x = "loss_window.step"` and `z = "loss_window.value"`.

A live series keeps its own ring of values, but the ring holds no steps. Thus a source over a live series that states `history` keeps the newest value of the series at each step, as a value of no axes, and a window over that source gives the values (W) with their steps. A source over a series without `history` keeps nothing, and the series panel draws the ring as before.

### Cursor

The cursor is the built-in state variable `cursor`. A reference reads its fields, and nothing writes them except the transport, the cursor commands and new steps.

| Field | Type | Meaning |
|---|---|---|
| `step` | Integer | The step that the panels show |
| `first` | Integer | The oldest step that any source keeps |
| `latest` | Integer | The newest step that the app received |
| `live` | Boolean | Whether the cursor follows the newest step |
| `playing` | Boolean | Whether the cursor moves toward the newest step |
| `speed` | Number | The steps for each second of play, from 0.1 to 1,000 |

The cursor is live when a spec opens. When it is live, it moves to each new step. When it plays, it moves through the kept steps at its speed, and at the newest step it becomes live. When it is held, it stays at its step. A step back or forward moves it to the previous or next step that any source keeps. When its step falls out of every source, it moves to `first`.

### Latest at the Cursor

Each panel shows, for each of its sources, the newest kept value at or before the cursor. A source with `from` evaluates its own steps on that value of its nearest source that keeps steps. A source with no kept steps shows its newest value, and the panel's info line says "Live only".

| Panel | At the cursor |
|---|---|
| `series` | The points up to the cursor, and a vertical line at the cursor when it is not live |
| `heatmap`, `image`, `histogram`, `points`, `table` | The value of each source at the cursor |
| `text` and readouts | Each field at the cursor |
| `scene` | Each entity at the cursor. A live scene from the script stays live, and the info line says so. |
| A geometry output of an agent's program | The newest output |

### Transport Control

The control kind `transport` binds `cursor`, and a spec has at most one, or the validator gives `spec.too_many`. It draws, from left to right, a step back, play or hold, a step forward, a live button, a speed menu and the step as text. The text reads "Live" or "Held" beside the step number. The option `speed` sets the speed at open, with a default of 10. With `keys = true`, the default, Space plays or holds, the left and right arrows step, Home goes to `first` and End goes live.

### Run Pause and Cursor

The run's pause and the cursor are different things, and the interface keeps them apart. A pause stops the training script at a step boundary. The cursor only chooses which kept step the panels show. A hold never pauses the run, and a pause never moves the cursor.

The transport's button is "Hold", and its tooltip is "Hold view". The run's button is a separate control with the title "Pause Run". While the run is paused, the transport shows the chip "Run Paused". While the view is held and the run trains, the transport text shows the held step and the newest step.

### Cursor Commands

The commands `cursor.get`, `cursor.set` and `cursor.step` read and move the cursor, as the section Commands lists. A move to a step outside the kept steps goes to the nearest kept step, and the result gives the step that the cursor took.

### Cursor Events

The event kind `cursor` has the keys `change`, `step`, `first`, `latest`, `playing` and `speed`. The change is `live`, `held`, `playing`, `moved` or `clamped`. A move by a command, a step or a key adds `moved`. A move by play or by a new live step adds no event, so that the event ring of 1,024 events keeps room for decisions.

## Bound Text

A bound text formats live values and state into a short text. Its language has named fields, number formats and literal units, and no expressions.

### Format Language

```text
text   = { literal | "{{" | "}}" | field }
field  = "{" name [ ":" format ] "}"
name   = a key of the fields table
format = [ "," ] [ "." digits ] ( "d" | "f" | "e" | "g" | "%" | "s" ) | "si" [ "." digits ]
```

```toml
[[control]]
id = "now"
kind = "readout"
options = { format = "Step {step:,d}  Loss {loss:.3f}  Token {token:d}", fields = { step = { state = "cursor", field = "step" }, loss = "loss", token = { state = "token" } } }
```

| Format | Output for 12345.678 |
|---|---|
| `d`, `,d` | `12346`, `12,346` |
| `.2f` | `12345.68` |
| `.2e` | `1.23e+04` |
| `.3g` | `1.23e+04` |
| `.1%` | `1234567.8%` |
| `si.3` | `12.3k` |
| `s` | The value as text |

A field binds a source or a reference. A source gives its value at the cursor: a 0-d tensor, a tensor of one element, or the last point of a series. A larger tensor shows its shape, such as `[16]`. A field without a value shows a dash. A format that does not fit the value's type is `spec.bad_format` when the validator knows the type, and shows `?` otherwise. An unknown field name, an open brace and a nested brace are `spec.bad_format`.

### Text Places

A bound text stands in three places:

- **Text panel.** The options `format` and `fields`. They exclude the static option `text`, and both together are `spec.bad_value`. A bound `value` slot is also the field `value`.
- **Readout.** A control of the kind `readout` in a control row, with the same two options.
- **Annotation.** The keys `text` and `fields` of each annotation kind.

### Text Limits

A format has at most 1,024 characters and at most 16 fields. The output has at most 1,024 characters, and a longer output ends with an ellipsis. A bound text updates at most 10 times a second. Digits use the tabular figures of the font, so that a changing number does not move its neighbors.

## Annotations and Markers

An annotation marks a value in a panel: a label with a leader line, a threshold line, a highlighted band or a marker on the current item. It is a spec item of its panel, because it binds a value. A legend and a scale bar bind no value, and so they are panel options.

### Annotation Table

```toml
[[panel.annotation]]
id = "window"
kind = "band"
axis = "x"
range = { control = "window" }
text = "Window"

[[panel.annotation]]
id = "picked"
kind = "marker"
at = { selection = "world", entity = "terrain" }
text = "Token {token:d}"
fields = { token = { state = "token" } }
```

| Kind | Keys | 2-D panel | Scene panel |
|---|---|---|---|
| `label` | `at`, `text`, `fields`, `leader` | A text at a point in data units | A text at a point in data units, at an item `{ entity, index }` or at a pick |
| `threshold` | `axis`, `value`, `text` | A line across the plot | The outline of the plane on the data box, with a faint fill |
| `band` | `axis`, `range`, `text` | A shaded span | The outline of the slab on the data box |
| `marker` | `at`, `text`, `fields` | A ring at a point | A ring at an item or a pick, always on top |

Every kind also takes `id`, which is unique in the panel, and `emphasis`, `accent` or `muted`. An annotation uses the annotation color of the theme, never a data color, so that the color key of the data stays clear. A threshold or a band in a scene takes the axis `x`, `y` or `z`, and in a 2-D panel `x` or `y`. A key that the kind does not take is `spec.unknown_option`, and a missing key is `spec.missing_key`.

### Legends and Scale Bars

The option `legend` is `auto`, `true` or `false`, with a default of `auto`. It applies to `series`, `heatmap`, `points`, `histogram`, `image` and `scene` panels. With `auto`, a panel shows a legend when it has two series or more, or a colormap. A scene's legend has a color bar for each entity with a colormap and a key entry for each entity with a flat color. The scale bar of a scene exists, and the option `scale_bar` also joins `image` and `points` panels, in the data units of the x axis.

### Label Placement

The scene view has no overlap avoidance today. One placement function, without ImGui, places the labels of annotations and the item labels of a scene. It takes rectangles and gives rectangles, and so tests check it without a window.

The function places labels in order: labels at a pick or a state first, then annotation labels in spec order, then item labels nearest first. For each label it tries eight positions around the anchor, then the same eight at three times the distance with a leader line. It takes the first position that overlaps no placed label and no marker and stays inside the panel. A label with no free position shows only its anchor dot, and a hover shows its text. The function runs again only when an anchor moves or the panel's size changes, and the same input always gives the same placement. A panel places at most 512 labels.

### Annotation Picking

Labels and markers take picks before the data under them. A pick on one adds a `pick` event with the key `annotation` and the fields of its anchor, and it does not change the panel's selection. Thresholds and bands take no picks in this work.

## Layout and Viewport

The layout gains shares with minimums, and a scene panel gains a depth offset, more colormaps, its starting camera, saved views and a data box.

### Layout Shares

A split gains the key `shares`: one positive number for each child. It replaces `ratio`, and a split with both is `spec.bad_layout`. A leaf may be a table `{ panel = "<id>", min = [columns, lines] }`, whose minimum is in characters and lines of the text scale, from 4 to 400 each.

```toml
[layout]
split = "vertical"
shares = [6, 3, 1]
children = [
  { split = "horizontal", shares = [3, 2], children = [{ panel = "world", min = [48, 20] }, "detail"] },
  "loss",
  { split = "horizontal", children = ["window", "measure", "transport", "now"] },
]
```

Each child first gets its minimum. The rest of the space goes to the children in proportion to their shares, and a child whose share falls below its minimum keeps the minimum while the others share the rest again. When the minimums do not fit, the children share the space in proportion to their minimums, as today. The function gives whole pixels and runs without ImGui. A control leaf keeps its fixed height.

### Depth Offset

An entity option `depth_offset`, an integer from -8 to 8, moves the entity's depth toward the camera by that many units. The default is 1 for `points` and `lines` and 0 for `surface`. Lines thus win over the surface that they lie on. The vertex stage applies the offset in clip space, and so it works the same on Metal and in WebGPU. This is the depth offset of the scene model additions.

### Colormaps

Twelve colormaps join the four of version 0: `cividis`, `inferno`, `plasma`, `turbo`, `twilight`, `coolwarm`, `batlow`, `vik`, `roma`, `oslo`, `lajolla` and `berlin`. One table in the builder library holds the sixteen names, and one table in the graphics layer holds their colors. Spec panels, entities, program outputs and the scene core read the same tables. The exact color tests check each colormap at fixed values.

### Starting Camera and Saved Views

A scene panel applies `view_camera` when it opens: an orbit table, a camera entity's path or the name of a saved view. A person's camera move stays for the session, and the reset of the view returns to `view_camera`.

The option `views` holds at most 16 saved views. Each is `{ name, target, distance, azimuth, elevation }`, and a name is unique in the panel. The header of a scene panel has a menu of its views. `camera.save` writes the current camera into `views` as an edit, with an undo step.

### Data Box

A wide flat scene, such as a terrain of 2,048 steps by 16 rows, is a thin strip when the camera fits its true proportions. A scene panel with entities thus scales its data box on each axis. The option `box` is three positive numbers, the proportions of the drawn box, or `auto`, the default. With `auto`, no side of the box is more than four times another side.

The scale applies only to the spec's entities. A live scene from the script keeps its own units. Picks report data units, and the camera's fit frames the box. With `axes = true`, the panel draws the three back walls of the box with ticks in data units and the titles `x_label`, `y_label` and `z_label`.

## Commands

Eighteen commands join the command layer and the MCP tool table. The `edit` commands follow the rules of `design/instruments.md`: a dry run, a diff, an undo step, a journal entry and the review of each origin. The `change` commands here are not destructive. They have no undo step and no journal entry, because they change session state and not the workspace. Each mode except read-only runs them at once.

| Command | Arguments | Result | Effect |
|---|---|---|---|
| `state.create` | `id`, `type`, and the other keys of a state variable | The diff | `edit` |
| `state.patch` | `id`, `set` | The diff | `edit` |
| `state.remove` | `id` | The diff | `edit` |
| `state.set` | `id`, `value` | `value`, `previous` | `change` |
| `state.get` | `ids`, optional | Each state's id, type, value, default, writers and readers | `read` |
| `control.set` | `id`, `value` | `value` | `change` |
| `selection.set` | `panel`, then `entity` and `index`, or `clear` | The pick fields | `change` |
| `selection.get` | `panel` | The pick fields, or null | `read` |
| `entity.set` | `panel`, `entity` | The diff | `edit` |
| `entity.remove` | `panel`, `path` | The diff | `edit` |
| `annotation.set` | `panel`, `annotation` | The diff | `edit` |
| `annotation.remove` | `panel`, `id` | The diff | `edit` |
| `cursor.get` | None | The cursor, and each source's kept steps, requested steps and bytes | `read` |
| `cursor.set` | `step`, `live`, `playing`, `speed`, each optional | The cursor | `change` |
| `cursor.step` | `count`, from -1,000 to 1,000 and not 0 | The cursor | `change` |
| `camera.get` | `panel` | The orbit and the saved views | `read` |
| `camera.set` | `panel`, then `view` or `orbit` | The orbit | `change` |
| `camera.save` | `panel`, `name` | The diff | `edit` |

`camera.get` and `camera.set` need the app's view, and the app answers them. `camera.save` asks the app for the current orbit first and then edits the spec, and the journal records that orbit with the arguments. `entity.set` creates an entity or replaces the entity with the same path, and `annotation.set` does the same by id. `panel.create` and `panel.patch` also accept the arrays `entity` and `annotation`. `selection.set` makes the same pick event, field values and state writes as a person's pick, with the origin of the caller.

### Command Errors

| Code | Meaning |
|---|---|
| `state.exists` | A state variable with the id exists |
| `state.not_found` | No state variable has the id |
| `state.in_use` | A reference, a control or a pick reads or writes the state variable |
| `state.bad_value` | The value does not fit the type, the bounds or the choices |
| `state.read_only` | The state variable is `cursor` |
| `control.not_settable` | The control binds a parameter, an action, an intervention or the run |
| `selection.bad_item` | The entity or the item does not exist in the panel |
| `entity.not_scene_panel` | The panel is not a scene panel |
| `entity.not_found` | The panel has no entity with the path |
| `annotation.not_found` | The panel has no annotation with the id |
| `cursor.no_history` | No source keeps steps |
| `camera.not_scene_panel` | The panel is not a scene panel |
| `camera.unknown_view` | The panel has no saved view with the name |

## Validator Codes

The codes below join the validator. The sections above state when each applies.

| Code | Severity |
|---|---|
| `spec.bad_reference`, `spec.reference_not_allowed`, `spec.unknown_control`, `spec.unknown_state`, `spec.unknown_field`, `spec.reference_type`, `spec.reference_cycle`, `spec.reference_in_history` | Error |
| `spec.history_not_allowed`, `spec.no_history` | Error |
| `spec.operator_shape` | Error |
| `spec.duplicate_shortcut`, `spec.reserved_shortcut` | Error |
| `spec.bad_format` | Error |
| `program.params_mismatch`, `stage.params_mismatch` | Error |
| `spec.state_writers` | Warning |
| `spec.not_built` | Error, until the feature that reads the key is built |

Existing codes cover the rest: `spec.duplicate_id`, `spec.out_of_range`, `spec.bad_value`, `spec.bad_transform`, `spec.bad_layout`, `spec.unknown_slot`, `spec.missing_slot`, `spec.slot_mismatch`, `spec.unknown_option`, `spec.missing_key`, `spec.control_mismatch` and `spec.too_many`. The control kinds `range`, `picker` and `text` stop giving `spec.unsupported_control`.

## Events

Two event kinds are new, and two kinds gain keys. Each new key joins the list of keys of its kind in the event ring.

| Kind | Keys |
|---|---|
| `state` | `state`, `value`, `previous`, `writer` |
| `cursor` | `change`, `step`, `first`, `latest`, `playing`, `speed` |
| `pick` | Adds `fields` and `annotation` |
| `control` | Adds `key`. A range gives `[lo, hi]`, and a segmented group with `multiple` gives a list. |

## Work Placement

| Piece | App | Script |
|---|---|---|
| References, state, selections and re-evaluation | All | Nothing |
| Entities | Resolution, upload and drawing | Nothing |
| Operators | Steps after a shipped step or a window, through the CPU reference | The first operator of a live tensor source, on the producer runtime |
| Program parameters | Evaluation, and stage uniforms | New values at the next pass |
| Controls and shortcuts | All, and parameters as today | New parameter values, as today |
| Step history and cursor | All | Step stamps on its snapshots, which this work adds |
| Bound text, annotations and layout | All | Nothing |

## Browser Position

The browser is designed for and built later, with browser parity. The reference evaluator, the state, the bound text, the label placement and the layout shares live in libraries without ImGui or platform code, so that the browser's spec runtime can build them to WebAssembly. The browser will keep its own steps of the derived values that it receives, with a smaller budget. The `surface` draw kind needs its pipeline in the web viewer, and the shortcuts follow the focus of the page. The script's operators stay as they are, because the browser receives derived data only.

## Tests

The tests prove each piece without a window:

- **Spec.** The spec tests read and write every new key in TOML and JSON, keep the canonical order and give each new validator code at its line and column.
- **References.** The session tests count evaluations: a slider drag of 100 values over 10 frames gives at most 10 evaluations, and an unrelated source evaluates none. A cycle gives `spec.reference_cycle`.
- **State.** The session tests check each state command, the writes of each control kind, the `select` writes and the `state` events.
- **Entities.** Offscreen captures of the spec runtime show points, a line strip on a wall and a surface with a hole. A pick at a known pixel gives the expected row and column.
- **Operators.** The operator tests compare the CPU references with fixed grids. The producer tests compare the device result with the reference, check the `tensor` output and the `params` op, and check that an operator shares the cadence.
- **Controls.** The spec runtime tests draw each kind, and the session tests check the shortcut codes and that a key changes a segmented group only while no text field holds the keyboard.
- **Time transport.** The session tests fill a history with stamped values and check the depth, the budget, the window, the clamp, each cursor command and the events.
- **Bound text.** The text tests check each format and each refusal of the grammar.
- **Annotations.** The placement tests check rectangles: no overlap, the priority order and the same placement for the same input. Captures show each kind in a series panel and a scene panel.
- **Layout and viewport.** The layout tests check shares and minimums at two window sizes. The graphics suite checks the depth offset and the new colormaps byte for byte, and a capture shows a wide terrain in its data box.
- **Commands.** The MCP tests check that the tool table holds the eighteen commands and that the served schema holds the new keys.

## Acceptance Test

The acceptance test drives `caliper-mcp` against the hidden app with `CALIPER_AGENT_OFFLINE=1`. Its driver and transcript are in `tests/instrument_building/`, and its check script is `examples/python/instrument_building_check.py`. The run is a small transformer whose only Caliper line is `caliper.watch(model, optimizer)`. The test builds the instrument of the training debugger demo without the agent's own programs, from the outputs of the output projections of the attention blocks. It does these steps in this order:

1. It records the SHA-256 list of the app bundle, `caliper-mcp` and the installed Python package.
2. It starts the managed run with `run.start` and reads the catalog.
3. It adds the state variables `token` and `measure`, and the source `act` with `history = 2048`.
4. It adds the range control `window`, the source `terrain` with a window and the `surface` operator, and the scene panel `world` with the terrain and the loss line on its back wall.
5. It adds a heatmap panel whose slice reads `{ state = "token" }`, a segmented group for `measure`, a readout, a transport, a Pause Run button and an `lr` slider.
6. It adds a band on the loss panel and a marker at the pick, and sets a layout with shares.
7. It renders `world` twice, some steps apart, and checks that the terrain grew.
8. It moves the range with `control.set` and checks that the terrain's window follows.
9. It picks a cell with `selection.set`, checks the `pick` and `state` events, and checks with `panel.data` that the heatmap shows that token.
10. It holds the cursor 100 steps back, steps back once, checks that the heatmap and the terrain changed, and goes live.
11. It adds a `surface` operator over a live weight matrix, checks in `program.status` that the operator runs in the script, and removes it.
12. It prints the time from the first tool call to the first complete frame of `world`.
13. It records the SHA-256 list again and checks that it did not change.

## Left Out

This work leaves these points out:

- A computed state variable, and references to a rectangle or lasso selection.
- Recordings to files, and kept steps of live scenes and of program geometry.
- The draw kinds `mesh`, `arrows`, `boxes` and `image` for entities, instanced meshes and additive blending.
- The other operators of the catalog, and GPU compute of the app's steps.
- References in operator parameters, poses and the layout.
- Pickers and text controls bound to script parameters.
- Dragged thresholds, bands and labels.
- The tree table, small multiples, scene helpers and the guided tour, which come with the node editor.
- The browser build of every piece.

## Decisions

Each decision below has its reason in one line.

- **Version 1.** The additions join spec version 1, because version 1 has not reached `main`.
- **App evaluates references.** References run only in the app, because a view must answer a control at once, also while the run is paused.
- **Value places only.** A reference never stands where it can change a shape or the graph, because the validator then checks a spec once.
- **One grammar.** Controls, selections and state share one reference table with `default`, because an agent then learns one form.
- **Cursor as state.** The cursor is the built-in state `cursor`, because every reference place then reads it with no new form.
- **No computed state.** A state variable has no formula, because only a control with `follow` then writes during evaluation and cycles stay rare.
- **Pick writes no edge.** A pick adds no edge to the cycle graph, because evaluation never makes a pick.
- **History in the app.** The app keeps the steps, because the script keeps its memory and 95 percent of its speed, and the cursor works while the run is paused.
- **History point.** A source keeps its value after its longest reference-free prefix, because the kept values then never depend on a later control.
- **Default depth.** A live tensor source keeps 64 steps, because the cursor then works without any setting, and the budget bounds the cost.
- **Window step.** The step window is a new step `window` with step numbers, not a `slice`, because a slice takes positions and the ring moves under them.
- **Ring as history.** The ring of recent rows of `surface` is the source's history and a window, because `path` and the cursor need the same ring.
- **Surface draw kind.** The viewer builds the surface's triangles from a grid, because a grid of heights is a quarter of a mesh's bytes and needs no index output.
- **One pool factor.** The surface pools with one factor on both axes, because the cells then stay square and the rule fits one line.
- **Operator placement.** Only the first operator of a live tensor source runs in the script, because the shipped steps run in the app today.
- **CPU in the app.** Operators in the app run their CPU reference, because their inputs there are small and GPU steps in the app are later work.
- **Operator ids.** An operator's wire id is `operator.<source>.<index>`, because no program id has a dot and the two never collide.
- **Tensor output.** Operators publish a `tensor` output that agents may use too, because one output kind serves both and feeds a source's history.
- **Params uniform.** Parameters reach WGSL as a uniform struct by name, because inputs already bind by name and a new value needs no compile.
- **Picker and text targets.** Pickers and text fields bind `value` or state only, because script parameters have no text or choice kind yet.
- **Shortcut grammar.** A shortcut is one key with an optional Shift, because the app owns every chord.
- **Hold, not pause.** The transport says "Hold" and the run says "Pause Run", because the two must never look like one control.
- **Few cursor events.** Play and live moves add no events, because they would fill the ring of 1,024 events.
- **Annotations as items.** Labels, thresholds, bands and markers are spec items, because they bind values, and legends and scale bars are options.
- **Annotation color.** Annotations use the theme's annotation color only, because one color key holds across the app.
- **Placement function.** Label placement is a pure function over rectangles, because tests then check it without a window.
- **Shares on splits.** Shares extend the split, not a new grid node, because nested splits express every layout that the demos need.
- **Data box.** A scene with entities scales its data box on each axis, because a terrain of 2,048 steps by 16 rows is a strip otherwise.
- **Session commands.** State, cursor, selection and camera moves use the effect `change` without undo, because they change the view and not the workspace.
- **Orbit in the journal.** The journal and the redo record of `camera.save` keep the orbit that the app gave, because a replay then rebuilds the saved view without the app.
- **Share of one source.** One source keeps at most a quarter of the budget, because a raw activation otherwise takes the steps of every other source.
- **Series steps by history.** A window over a live series reads the newest values that the history keeps with their steps, because the ring of a series holds no steps.

## Open Points

- **Attention weights.** The watch hook publishes the output of each leaf module as `<module>.act`, the first tensor of a tuple output, and no attention weights. An attention block that holds other modules is not a leaf, and so the acceptance test reads a leaf inside it, such as its output projection `blocks.0.attn.proj.act` of shape (B, T, D).
- **Snapshot stamps.** Resolved: no watched view carries a stamp today, and the live catalog keeps no step. The step history adds both, and until then a value gets the run's step at arrival, which can be one step late.
- **Budget.** The 512 MiB of kept steps was measured with one raw hidden activation of the medium model, 512 by 4096 values: at 64 steps it fills the whole budget, with 551.6 MB of peak memory in all. The share of one source, 128 MiB, keeps such a source from taking the budget of the others. Whether a quarter is the right share needs a longer use of real instruments.
- **Tile map of a channel mosaic.** The watch hook shows a module output of rank 3 as a channel mosaic, and its snapshots carry no tile map, so the live catalog serves it as one picture: an output of (8, 64, 64) lists as (192, 192). The acceptance script thus feeds its attention projection the tokens flattened to (512, 64). The sender could send the tile map of its default mosaic, as a data view with `tile` does, so that the catalog serves the tensor whole.
- **Registry entries.** State variables, the time transport, bound text, annotations and segmented groups have no registry ids of their own. The coverage file maps them to the nearest entries.
