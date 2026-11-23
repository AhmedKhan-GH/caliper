# Spec Version 1

Spec version 1 is spec version 0 with additions: the panel primitive `scene`, the top-level arrays `control` and `state`, and the keys that instruments need. The [Instrument Keys](#instrument-keys) section lists them. A version 0 document stays valid and is read as before. A document of version 1 starts with `caliper_spec = 1`. The reader knows the versions 0 and 1 and refuses any other with `spec.unsupported_version`. The JSON Schema of version 1 is `caliper-spec-v1.schema.json`, and `caliper-mcp` serves it beside the schema of version 0.

## Instrument Keys

An instrument is a spec that reacts to a run: its panels read live tensors, kept steps and the values of controls. All keys of an instrument belong to version 1, and each one raises the required version to 1. This table lists the keys and the page that describes each one.

| Key | Place | Content | Page |
|---|---|---|---|
| `state` | Top level | An array of state variables, at most 64 | [State and References](state-and-references.md) |
| `history` | Source | The number of recent steps that Caliper keeps | [Time Cursor](time-cursor.md) |
| `window` | Transform `op` | A step over the kept steps | [Time Cursor](time-cursor.md#window-step) |
| `surface`, `path` | Transform `op` | The two operators | [Operators](operators.md) |
| `entity` | Scene panel | An array of entities, at most 64 | [Scene Entities](scene-entities.md) |
| `annotation` | Panel | An array of labels, thresholds, bands and markers, at most 64 | [Annotations](annotations.md) |
| `select`, `box`, `views` | Scene panel options | A pick that writes state, the data box and the saved views | [Scene Entities](scene-entities.md) |
| `legend` | Series panel option | The legend of a series panel | [Annotations](annotations.md#legends-and-scale-bars) |
| `format`, `fields` | Text panel and readout options | A bound text and its values | [Bound Text](bound-text.md) |
| `range`, `picker`, `text`, `segmented`, `readout`, `transport` | Control `kind` | The control kinds | [Controls](controls.md) |
| `state:<id>`, `cursor` | Control `bind` | The state and the cursor as targets | [Controls](controls.md) |
| `shares`, `min` | Layout | The shares of a split and the minimum of a leaf | [Layout Shares](#layout-shares) |
| `params` | Program | Named numbers that the program reads | [Operators](operators.md#program-parameters) |
| `tensor` | Program output `kind` | An output that the script publishes as a live tensor | [Operators](operators.md#tensor-output) |
| `piece` | Top level | An array of inline pieces | [Pieces](#pieces) |
| `piece`, `bind`, `panel` | Program | The instance of a piece: its pin, its bindings and its scene panel | [Pieces](#pieces) |

A binding reference can stand at some of these keys. [State and References](state-and-references.md#reference-places) lists the places.

## Version Rules

A version 0 document that uses a key of version 1 gets the errors of version 0, `spec.unknown_primitive` or `spec.unknown_key`. A version 0 document with the number changed to 1 is a valid version 1 document, because version 1 is a strict superset. The canonical writers write the version that the spec holds and do not raise it.

An edit that adds a scene panel, a control, a state variable or another key of version 1 to a version 0 spec raises the version to 1 in the same command. The edit and the new number are one undo step. This holds for `panel.create` of a scene panel, for the first `control.create` and for the first `state.create`.

A saved spec applet of format 1 loads from the workspace. The host refuses only a format above the newest one or below 0.

## Scene Panel

A panel with `primitive = "scene"` shows a live scene of the attached script. Its one slot, `scene`, is required. The slot holds `live:<scene name>`, the name that `catalog.list` gives for the scene, and not a source id. A scene panel with at least one [entity](scene-entities.md) needs no `scene` slot.

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
| `view_camera` | Orbit table or entity path | The scene's own | The camera that the panel opens on |
| `include` | 1 to 64 entity path patterns | Every entity | The entities that the panel draws |
| `exclude` | 1 to 64 entity path patterns | None | The entities that the panel does not draw, after `include` |
| `labels` | Boolean | `true` | Whether the panel draws the entity labels |
| `axes` | Boolean | `true` | Whether the panel draws the axes and the ground grid |
| `scale_bar` | Boolean | `false` | Whether the panel draws a scale bar in the units of the scene |
| `box` | `auto` or three positive numbers | `auto` | The proportions of the data box of the entities |
| `views` | At most 16 tables | None | The saved views of the camera |
| `select` | `{ state, field, entity }` | None | The state that a pick writes |
| `x_label`, `y_label`, `z_label` | Text | None | The titles of the axes of the data box |

An orbit table has the keys `target`, `distance`, `azimuth` and `elevation`. In a pattern, `*` matches one path segment and `**` matches any number of segments. A scene that the script does not send shows "Source offline", and the validator warns with `spec.unresolved_source`, because a live scene can come later.

In the app, the spec page draws the panel from the live scene of the attached script. A drag orbits the view, the wheel zooms, a click picks and a shift-drag selects. The panel sends its picks and selections to the event ring as `pick` and `select` events that carry its panel id. [Run Control and Events](../agents/run-control-and-events.md) lists their keys. The command `panel.data` of a scene panel gives `{"scene": "<name>"}` and, for each entity, the shape, the minimum, the maximum and the values of its arrays.

## Controls

The array `control` holds the controls of the spec. A control acts on one target.

| Key | Type | Required | Meaning |
|---|---|---|---|
| `id` | Id | Yes | The id of the control. Panels and controls share one set of ids. |
| `kind` | `slider`, `button`, `range`, `toggle`, `picker`, `text`, `segmented`, `readout` or `transport` | Yes | The widget |
| `title` | Text, at most 256 characters | No | The label. The default is the `id`. |
| `bind` | Target string | Yes, except for a `readout` | The one target of the control |
| `options` | Table | No | The options of the kind |

The `bind` string has one of these forms:

- `param:<name>` is a parameter that the script registered with `caliper.param`.
- `action:<name>` is an action that the script registered with `caliper.action`.
- `run:pause` is the pause and the resume of the run.
- `value` is a value that the spec holds.
- `state:<id>` is a [state variable](state-and-references.md). A `readout` has no target and a `transport` binds `cursor`.

Version 1 builds all nine kinds: `slider`, `button`, `range`, `toggle`, `picker`, `text`, `segmented`, `readout` and `transport`. [Controls](controls.md) describes each kind. A toggle binds `intervene:<point>`, which it ablates while it is on, or `param:<name>` of a boolean parameter.

| Kind | Targets | Options |
|---|---|---|
| `slider` | `param:<name>`, `value` | `min`, `max`, `step`, `log`, `value` |
| `button` | `action:<name>`, `run:pause` | None |

A slider bound to a parameter takes the range and the scale of the parameter when `min`, `max` or `log` is absent, and it sets the parameter while you drag it. The script clamps every value to its own range, thus a slider can show a narrower range but not a wider one. A slider bound to `value` holds its value, which starts at the option `value` or at `min`. A button bound to an action runs it. A button bound to `run:pause` pauses a run that runs and resumes a run that is paused. A control sends its value to the attached script directly, so it shows no approval card. Each finished edit gives one `control` event with the origin `ui`.

## Programs

The array `program` holds at most 16 GPU programs. A program is WGSL source with a declaration of its inputs, its output and its limits. The keys are `id` (at most 55 characters), `title`, `wgsl` (at most 64 KiB), `entry`, `inputs` (at most 8, each `{var, source}` with a source `live:<name>`), `output`, `limits` (`time_ms`, `memory_mb` and `every`) and `workgroups`. The output has the `kind` `geometry` or `pixels`. A geometry output names a scene `panel`, and optionally an `entity`, a `part` of `points` or `lines`, `max_count`, the `arrays` and a `style`. A pixels output has `width`, `height` and `format`. The validator holds each program to the limits of the sandbox, and it reports a target that is not a scene panel of the spec as `spec.unknown_panel`. [GPU Programs](../agents/gpu-programs.md) describes the keys, the limits and what the script does with a program. A program that holds the key `piece` has another form, which the next section describes.

## Pieces

A program is either a one-pass program or an instance of a piece. An instance holds the keys `id`, `title`, `piece`, `bind`, `panel` and `params`. It holds none of the keys `wgsl`, `entry`, `inputs`, `output`, `limits` and `workgroups`, and a mix of the two forms is `program.form_mixed`. The key `piece` is a pin with `name`, `version`, `hash` and `from`. The key `from` is `library` (the default), `workspace` or `inline`. The key `bind` maps each input of the piece to `live:<tensor>` or to `program:<id>/<resource>`. The key `panel` names a scene panel, and a program of a piece that draws needs it. The key `params` gives each parameter a number or a binding reference, at most 16 names.

The top-level array `piece` holds inline pieces. Each one has the keys of a `piece.toml` manifest, and its table `files` holds the WGSL under `common`, `derive` and `draw`. Two inline pieces may not share a name and a version. The validator checks each inline piece on its own, and a program that pins an inline piece draws only when the piece is whole. A pin that names a piece that its source lacks, or a hash that differs, does not make the spec invalid. The spec opens, and the state of the program is `refused` with `piece.missing` or `piece.hash_mismatch`. [Pieces](../agents/pieces.md) describes the manifest, the halves, the hash and the commands.

## Old Stage Tables

Library pieces replace the old `stage` array. A spec with a `[[stage]]` table opens without that stage. The report gives one `spec.stage_replaced` warning per table at `stage[<index>]`. Each message identifies the stage's kind, entity, and panel, and explains the replacement piece. The next save omits the stage and its review hash. Review hashes for programs stay in the spec.

## Provenance

The table `provenance` marks a spec from someone else. It has `foreign` (true), `from` (a text that names the origin) and `reviewed`, a list of `{target, sha256}`. A target is `program:<id>`, and the hash is the SHA-256 of its WGSL. In a foreign spec, a program runs only when its hash is in `reviewed`, and else its state is `review`. A spec without `provenance` is the person's own. No command writes `provenance`, and `spec.set_text` refuses a text whose `provenance` differs from the open spec's with `spec.provenance_locked`.

## Control Layout

A control is a leaf of the layout, as a panel is. A leaf is an id, and the id names a panel or a control. The commands `layout.place` and `layout.set` accept control ids. In a vertical split, a control leaf takes the height of one control row.

A spec with no `layout` places its panels as version 0 does. When the spec has controls, the default root is a vertical split of two children: the panels, then one horizontal split of the controls in declaration order. A single control is a leaf and not a split. The validator warns with `spec.unplaced_panel` for a control that the layout does not place.

## Layout Shares

A split takes the key `shares`: one positive number for each child. It replaces `ratio`, and a split with both is `spec.bad_layout`. A leaf may be a table `{ panel = "<id>", min = [columns, lines] }`, whose minimum is in characters and lines of the text scale, from 4 to 400 each.

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

Each child first gets its minimum. The rest of the space goes to the children in proportion to their shares. A child whose share falls below its minimum keeps the minimum, and the other children share the rest again. When the minimums do not fit, the children share the space in proportion to their minimums. The result has whole pixels. A control leaf keeps its height. The command `layout.set` takes the nodes with `shares` and `mins`.

## Control Commands

The commands `control.create`, `control.patch` and `control.remove` edit the open spec as the panel commands do. Each one has a dry run, a diff, review for the origins that have it on, one undo step and a journal entry.

`control.create` takes `id`, `kind` and `bind`, and optionally `title`, `options`, `place` and `dry_run`. The argument `place` takes `target`, `where` (`beside`, `above`, `below` or `tabbed`) and `ratio`, and its target can be a panel or a control. Without `place`, the control joins the row of controls under the layout. `control.patch` takes `id` and a merge patch of `kind`, `title`, `bind` and `options`. A key set to null is removed, and the id cannot change. `control.remove` takes `id` and also removes the control from the layout.

The result of `control.create` and `control.remove` holds `id`, `warnings`, `diff` and `applied`. The result of `control.patch` holds `control` instead of `id`. A control id that exists fails with `control.exists`, and an unknown one fails with `control.not_found`.

## Validator Codes

Version 1 adds three error codes for controls. The first one is given for an unknown kind. A spec with more than 16 programs fails with `spec.too_many`. The other errors of controls and scene panels use the codes of version 0.

| Code | Stage | Cause |
|---|---|---|
| `spec.unknown_control_kind` | Parse | A `kind` that is not one of the nine kinds |
| `spec.unsupported_control` | Validate | A kind of the format that this Caliper does not draw yet |
| `spec.control_mismatch` | Validate | A target that the kind cannot take, or an option of another kind |

| Existing Code | Use in Version 1 |
|---|---|
| `spec.bad_binding` | A `bind` that does not match the target forms, or a scene slot that is not `live:<name>` |
| `spec.duplicate_id` | A control id that is also a panel id |
| `spec.unknown_panel` | A layout leaf that names no panel and no control |
| `spec.out_of_range` | A slider with `min` not below `max`, a `step` not above 0, or `log` with `min` not above 0 |
| `spec.too_many` | More than 64 controls or more than 64 patterns |

## Instrument Codes

The keys of an instrument add these codes. The pages named above state when each one applies.

| Code | Severity | Cause |
|---|---|---|
| `spec.bad_reference`, `spec.reference_not_allowed`, `spec.unknown_control`, `spec.unknown_state`, `spec.unknown_panel`, `spec.unknown_field`, `spec.reference_type`, `spec.reference_cycle`, `spec.reference_in_history` | Error | A binding reference that is wrong ([State and References](state-and-references.md#reference-errors)) |
| `spec.history_not_allowed`, `spec.no_history` | Error | `history` on a source that is not a live tensor, and a `window` over a source that keeps no steps |
| `spec.operator_shape` | Error | An operator input of another rank, or a column past the last |
| `spec.duplicate_shortcut`, `spec.reserved_shortcut` | Error | A key of a segmented choice that is used twice, or that the app keeps |
| `spec.bad_format` | Error | A bound text that does not parse |
| `program.params_mismatch` | Error | A `Params` struct whose fields differ from the names |
| `spec.state_writers` | Warning | Two controls write one state variable |
| `spec.not_built` | Error | A key that is part of the format but not built: annotations on heatmaps and images, `legend` set to `true` or `false` outside a series panel, `scale_bar` outside a scene, and `choices_from` of a picker |

## Piece Codes

The keys of pieces add these codes to the validator.

| Code | Cause |
|---|---|
| `program.form_mixed` | A program with a `piece` pin that also holds a key of the one-pass form |
| `spec.missing_key` | A program that pins an inline piece with a draw half and names no `panel` |
| `spec.bad_value` | A pin hash that is not `sha256:` and 64 lowercase hex digits, or a name of a parameter that is not a WGSL name |
| `spec.too_many` | A program with more than 16 `params` |
| `spec.duplicate_id` | Two inline pieces with the same name and version |
| `piece.bad_name`, `piece.bad_manifest`, `piece.unknown_import` | An inline piece that breaks the rules of its name, its manifest or its imports |
| `piece.bad_binding`, `piece.cycle` | A binding that is not `live:<tensor>` or `program:<id>/<resource>`, names a program that the spec lacks, or names a resource of its own program |
| `spec.unknown_panel`, `program.bad_target` | A `panel` that is no panel of the spec, and a `panel` that is not a scene panel |

The code `spec.unsupported_control` is no longer given, because Caliper draws all nine control kinds.
