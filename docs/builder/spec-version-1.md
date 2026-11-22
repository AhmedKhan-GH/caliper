# Spec Version 1

Spec version 1 is spec version 0 with two additions: the panel primitive `scene` and the top-level array `control`. A version 0 document stays valid and is read as before. A document of version 1 starts with `caliper_spec = 1`. The reader knows the versions 0 and 1 and refuses any other with `spec.unsupported_version`. The JSON Schema of version 1 is `caliper-spec-v1.schema.json`, and `caliper-mcp` serves it beside the schema of version 0.

## Version Rules

A version 0 document that uses a key of version 1 gets the errors of version 0, `spec.unknown_primitive` or `spec.unknown_key`. A version 0 document with the number changed to 1 is a valid version 1 document, because version 1 is a strict superset. The canonical writers write the version that the spec holds and do not raise it.

An edit that adds a scene panel or a control to a version 0 spec raises the version to 1 in the same command. The edit and the new number are one undo step. This holds for `panel.create` of a scene panel and for the first `control.create`.

A saved spec applet of format 1 loads from the workspace. The host refuses only a format above the newest one or below 0.

## Scene Panel

A panel with `primitive = "scene"` shows a live scene of the attached script. Its one slot, `scene`, is required. The slot holds `live:<scene name>`, the name that `catalog.list` gives for the scene, and not a source id.

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

An orbit table has the keys `target`, `distance`, `azimuth` and `elevation`. In a pattern, `*` matches one path segment and `**` matches any number of segments. A scene that the script does not send shows "Source offline", and the validator warns with `spec.unresolved_source`, because a live scene can come later.

In the app, the spec page draws the panel from the live scene of the attached script. A drag orbits the view, the wheel zooms, a click picks and a shift-drag selects. The panel sends its picks and selections to the event ring as `pick` and `select` events that carry its panel id. [Run Control and Events](../agents/run-control-and-events.md) lists their keys. The command `panel.data` of a scene panel gives `{"scene": "<name>"}`.

## Controls

The array `control` holds the controls of the spec. A control acts on one target.

| Key | Type | Required | Meaning |
|---|---|---|---|
| `id` | Id | Yes | The id of the control. Panels and controls share one set of ids. |
| `kind` | `slider`, `button`, `range`, `toggle`, `picker` or `text` | Yes | The widget |
| `title` | Text, at most 256 characters | No | The label. The default is the `id`. |
| `bind` | Target string | Yes | The one target of the control |
| `options` | Table | No | The options of the kind |

The `bind` string has one of four forms:

- `param:<name>` is a parameter that the script registered with `caliper.param`.
- `action:<name>` is an action that the script registered with `caliper.action`.
- `run:pause` is the pause and the resume of the run.
- `value` is a value that the spec holds.

Version 1 builds the kinds `slider`, `button` and `toggle`. A toggle binds `intervene:<point>`, which it ablates while it is on, or `param:<name>` of a boolean parameter. The other three kinds are part of the format, but the validator refuses them with `spec.unsupported_control`.

| Kind | Targets | Options |
|---|---|---|
| `slider` | `param:<name>`, `value` | `min`, `max`, `step`, `log`, `value` |
| `button` | `action:<name>`, `run:pause` | None |

A slider bound to a parameter takes the range and the scale of the parameter when `min`, `max` or `log` is absent, and it sets the parameter while you drag it. The script clamps every value to its own range, thus a slider can show a narrower range but not a wider one. A slider bound to `value` holds its value, which starts at the option `value` or at `min`. A button bound to an action runs it. A button bound to `run:pause` pauses a run that runs and resumes a run that is paused. A control sends its value to the attached script directly, so it shows no approval card. Each finished edit gives one `control` event with the origin `ui`.

## Programs

The array `program` holds at most 16 GPU programs. A program is WGSL source with a declaration of its inputs, its output and its limits. The keys are `id` (at most 55 characters), `title`, `wgsl` (at most 64 KiB), `entry`, `inputs` (at most 8, each `{var, source}` with a source `live:<name>`), `output`, `limits` (`time_ms`, `memory_mb` and `every`) and `workgroups`. The output has the `kind` `geometry` or `pixels`. A geometry output names a scene `panel`, and optionally an `entity`, a `part` of `points` or `lines`, `max_count`, the `arrays` and a `style`. A pixels output has `width`, `height` and `format`. The validator holds each program to the limits of the sandbox, and it reports a target that is not a scene panel of the spec as `spec.unknown_panel`. [GPU Programs and Stages](../agents/gpu-programs.md) describes the keys, the limits and what the script does with a program.

## Stages

The array `stage` holds at most 32 stages. A stage has a `panel` (a scene panel), an `entity` (an entity path), a `kind` (`material` or `vertex`) and `wgsl` (at most 16 KiB). The panel, the entity and the kind are unique among the stages. A stage replaces one function in the draw of the entity, and its source holds functions and constants only. When a command removes a panel, the same undo step removes the stages of that panel.

## Provenance

The table `provenance` marks a spec from someone else. It has `foreign` (true), `from` (a text that names the origin) and `reviewed`, a list of `{target, sha256}`. A target is `program:<id>` or `stage:<panel>/<entity>/<kind>`, and the hash is the SHA-256 of its WGSL. In a foreign spec, a program or a stage runs only when its hash is in `reviewed`, and else its state is `review`. A spec without `provenance` is the person's own. No command writes `provenance`, and `spec.set_text` refuses a text whose `provenance` differs from the open spec's with `spec.provenance_locked`.

## Control Layout

A control is a leaf of the layout, as a panel is. A leaf is an id, and the id names a panel or a control. The commands `layout.place` and `layout.set` accept control ids. In a vertical split, a control leaf takes the height of one control row.

A spec with no `layout` places its panels as version 0 does. When the spec has controls, the default root is a vertical split of two children: the panels, then one horizontal split of the controls in declaration order. A single control is a leaf and not a split. The validator warns with `spec.unplaced_panel` for a control that the layout does not place.

## Control Commands

The commands `control.create`, `control.patch` and `control.remove` edit the open spec as the panel commands do. Each one has a dry run, a diff, review for the origins that have it on, one undo step and a journal entry.

`control.create` takes `id`, `kind` and `bind`, and optionally `title`, `options`, `place` and `dry_run`. The argument `place` takes `target`, `where` (`beside`, `above`, `below` or `tabbed`) and `ratio`, and its target can be a panel or a control. Without `place`, the control joins the row of controls under the layout. `control.patch` takes `id` and a merge patch of `kind`, `title`, `bind` and `options`. A key set to null is removed, and the id cannot change. `control.remove` takes `id` and also removes the control from the layout.

The result of `control.create` and `control.remove` holds `id`, `warnings`, `diff` and `applied`. The result of `control.patch` holds `control` instead of `id`. A control id that exists fails with `control.exists`, and an unknown one fails with `control.not_found`.

## Validator Codes

Version 1 adds three error codes for controls. A spec with more than 16 programs or 32 stages fails with `spec.too_many`. The other errors of controls and scene panels use the codes of version 0.

| Code | Stage | Cause |
|---|---|---|
| `spec.unknown_control_kind` | Parse | A `kind` that is not one of the six kinds |
| `spec.unsupported_control` | Validate | A kind of the format that this Caliper does not draw yet |
| `spec.control_mismatch` | Validate | A target that the kind cannot take, or an option of another kind |

| Existing Code | Use in Version 1 |
|---|---|
| `spec.bad_binding` | A `bind` that does not match the target forms, or a scene slot that is not `live:<name>` |
| `spec.duplicate_id` | A control id that is also a panel id |
| `spec.unknown_panel` | A layout leaf that names no panel and no control |
| `spec.out_of_range` | A slider with `min` not below `max`, a `step` not above 0, or `log` with `min` not above 0 |
| `spec.too_many` | More than 64 controls or more than 64 patterns |
