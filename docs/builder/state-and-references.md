# State and References

A state variable is a named value of the spec. Controls and picks write it, and panels read it through references to filter, highlight, recolor or retarget. A binding reference is a small table that stands where the spec takes a literal. Caliper evaluates it, and the script never sees it. A reference thus changes what a panel shows at once, also while the run is paused.

## State Table

The declaration of a state variable is part of the spec, and the spec saves it. The current value belongs to the session. It starts at the default when the spec opens, and the spec never saves it.

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
default = "raw"
choices = ["raw", "zscore"]
```

| Key | Type | Required | Meaning |
|---|---|---|---|
| `id` | Id | Yes | The id is unique among state variables. The id `cursor` is reserved. |
| `title` | Text | No | The label that a control or a legend shows. The default is the id. |
| `type` | `number`, `integer`, `boolean`, `text`, `range` or `list` | Yes | The type of the value. A `list` holds texts. |
| `default` | Value of the type | No | The default is 0, `false`, the first choice, an empty text, `[min, max]` or an empty list. |
| `min`, `max` | Number | No | The bounds of a number, an integer or a range |
| `choices` | 1 to 1,024 values | No | The allowed values of a text or an integer, or the items of a list |
| `x` | Table | No | Extensions |

A spec holds at most 64 state variables. A text value has at most 1,024 bytes and a list has at most 256 items. A default outside the bounds or the choices is `spec.out_of_range`. Two state variables with one id are `spec.duplicate_id`, and more than 64 are `spec.too_many`.

## State Writers

Three things write a state variable:

- A control with `bind = "state:<id>"`. A control kind that does not fit the type is `spec.control_mismatch`.
- A pick in a panel with the option `select = { state = "<id>", field = "<pick field>", entity = "<path>" }`. Each pick writes that field of the pick. The key `entity` is optional.
- The command `state.set`.

Two controls that write one state variable give the warning `spec.state_writers`. A pick and a control may write one variable, because every view shows the same selection.

Each change of a value adds an event of the kind `state` with the keys `state`, `value`, `previous` and `writer`. The writer is a control id, a panel id or the command name.

## State Readers

A reader names the state variable in a reference. These are the four uses:

- **Filter.** A `slice` step reads `{ state = "token" }` as its index or its start.
- **Highlight.** An entity option `highlight = { state = "token", along = "column" }` draws the items with that index in the annotation color.
- **Recolor.** An option `colormap = { state = "palette" }` reads a text state whose choices are colormap names.
- **Retarget.** A slot reads a text state whose choices are source ids, and the panel shows the chosen source.

```toml
[[panel]]
id = "detail"
primitive = "heatmap"
bind = { values = { state = "measure" } }
```

A retarget state whose choices are not all source ids is `spec.reference_type`.

## Reference Forms

Exactly one of the keys `control`, `selection` and `state` names the input.

```text
{ control = "<control id>" [, field = "lo" | "hi"] [, default = <literal>] }
{ selection = "<panel id>", field = "<pick field>" [, entity = "<entity path>"] [, default = <literal>] }
{ state = "<state id>" [, field = "<field>"] [, default = <literal>] }
```

| Form | Reads |
|---|---|
| `control` | The current value of a control. A range gives `[lo, hi]`, or one end with `field`. A control bound to `param:<name>` gives the last value that the script reported. |
| `selection` | One field of the last pick in the panel, as [Pick Fields](scene-entities.md#pick-fields) lists them. With `entity`, only a pick on that entity counts. |
| `state` | The current value of a state variable. A range state takes `field` with `lo` or `hi`. The built-in state `cursor` takes the fields of the [time cursor](time-cursor.md#cursor). |

A reference without a value, such as a selection before the first pick, gives its `default`. Without a `default`, a step waits and its panel shows "Waiting for input". An option keeps its own default, and a bound text shows a dash. A `default` is a literal of the type that the place takes, and a reference never holds another reference.

## Reference Places

A reference stands only where its value changes data and never changes the shape of the spec. Thus the validator checks the spec once, and a new value never makes a spec invalid.

| Place | Keys | Type |
|---|---|---|
| `slice` step | `index`, `start`, `stop`, `step` | Integer |
| `window` step | `start`, `stop`, `steps` | Integer |
| Panel options | `range`, `x_range`, `y_range` | A pair, or a range reference |
| Panel and entity options | `colormap` | Text, from a state whose choices are colormap names |
| Entity options | `vmin`, `vmax`, `highlight`, `visible` | Number, item or boolean |
| Slot binding | Any slot | Text, from a state whose choices are source ids |
| Annotation | `value`, `range`, `at` | Number, pair or position |
| Bound text | Each entry of `fields` | Any |
| Control options | `min`, `max` of a slider or a range | Number |
| Program and stage | Each entry of `params` | Number |

Every other key takes a literal: ids, kinds, `from`, `catalog`, the axes of a step, the parameters of an operator, `history`, poses and the layout. A reference there is `spec.reference_not_allowed`.

## Re-Evaluation

Caliper builds a dependency index when a spec opens and after each edit. Its inputs are the value of each control and state variable, the last pick of each panel, the cursor and the generation of each source. A change of an input marks the readers of that input. At the next frame, each marked reader evaluates again, at most once. A slider drag thus costs one evaluation for each frame, and the last value wins. The steps of a chain before the first step that reads the changed input keep their cached results.

A changed program parameter goes to the script in the message `PROGRAM` with the op `params`, and it applies at the next pass. A changed stage parameter goes to the uniform of the stage at the next frame.

## Reference Errors

| Code | Meaning |
|---|---|
| `spec.bad_reference` | No input key or two of them, an unknown key, a nested reference, or a `default` of the wrong type |
| `spec.reference_not_allowed` | A reference at a key that takes only a literal |
| `spec.unknown_control` | The control id is not a control of the spec |
| `spec.unknown_state` | The state id is not a state variable of the spec and not `cursor` |
| `spec.unknown_panel` | The panel of a selection is not a panel of the spec |
| `spec.unknown_field` | The field does not exist for that control kind, state type or pick |
| `spec.reference_type` | The type of the value does not fit the place |
| `spec.reference_cycle` | The options of a control read, directly or through state, the value that the control writes |
| `spec.reference_in_history` | A `window` reads a source whose own steps read a reference |

Evaluation writes a value in one case only: a range control with `follow` moves its value with its bounds. A cycle through such a control is `spec.reference_cycle`, and its message names each item on the cycle. A pick writes state only on a person's input or a command, never during evaluation.
