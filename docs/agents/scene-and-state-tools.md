# Scene and State Tools

An agent edits scenes, cameras, state variables, selections, controls, the time cursor, annotations and layout locks in the open spec with the tools on this page and the tools of [MCP Server](mcp-server.md). The builder pages describe what each key does. This page lists the tools, their arguments and their answers. The exact schemas are in the command table, and `tools/list` serves them.

## Tool Kinds

A tool is an edit, a change or a read. An edit changes the open spec. It has a dry run, a diff, an undo step, a journal entry and the review of its origin, like `panel.create`. A change moves state of the session, such as a value or the cursor. It has no undo step and no journal entry, and each mode except read only runs it at once. A read changes nothing.

| Tool | Kind | Purpose |
|---|---|---|
| `entity_set`, `entity_remove` | Edit | Add, replace or remove an entity of a scene panel |
| `camera_get`, `camera_set` | Read, change | Read or move the camera of a scene panel |
| `camera_save` | Edit | Save the current camera as a named view |
| `state_create`, `state_patch`, `state_remove` | Edit | Declare, change or remove a state variable |
| `state_set`, `state_get` | Change, read | Set a state value, or read the variables with their writers and readers |
| `selection_set`, `selection_get` | Change, read | Pick an item as a person does, or read the last pick |
| `control_set` | Change | Set the value of a control |
| `cursor_get`, `cursor_set`, `cursor_step` | Read, change | Read or move the time cursor |
| `annotation_set`, `annotation_remove` | Edit | Add, replace or remove an annotation of a panel |
| `layout_lock` | Edit | Lock or unlock the whole layout of the open spec |

## Entity Commands

`entity_set` takes `panel`, `entity` and `dry_run`. The entity is a table with a `path` and the other keys of an [entity](../builder/scene-entities.md#entity-table). The command creates the entity or replaces the entity with the same path. `entity_remove` takes `panel`, `path` and `dry_run`. Both answer with `panel`, `diff`, `warnings` and `applied`. A panel that is not a scene panel fails with `entity.not_scene_panel`, and an unknown path fails with `entity.not_found`.

`panel_data` of a scene panel gives, for each entity, the arrays as slots with `shape`, `min`, `max` and values. The `x` and `y` slots of a surface hold the axes. A wrong shape or length shows as the code `entity.shape` or `entity.length`, and an entity without data as `entity.no_data`. `panel_render` of a scene panel draws the entities and the anchors of markers, but not the texts of annotations. `capture_frame` shows the texts.

## Camera Commands

`camera_get` takes `panel` and answers with the `orbit` and the saved `views`, at most 16. `camera_set` takes `panel` and either `view`, the name of a saved view, or `orbit`, a table with `target`, `distance`, `azimuth` and `elevation`, and answers with the orbit after the move. `camera_save` takes `panel`, `name` and `dry_run`. It reads the current orbit from the app, writes it into the option `views` and records it in the journal. The commands fail with `camera.not_scene_panel` for another panel, and `camera_set` fails with `camera.unknown_view` for an unknown name. `camera_save` fails with `camera.no_orbit` when no app view gives an orbit. All three need the app.

## State Commands

`state_create` takes `id` and `type`, and optionally `title`, `default`, `min`, `max`, `choices`, `x` and `dry_run`. `state_patch` takes `id` and `set`, an object with the keys to change. `state_remove` takes `id`. The three answer with `id`, `diff`, `warnings` and `applied`.

`state_set` takes `id` and `value` and answers with `value` and `previous`. It makes the same `state` event as a control does. `state_get` takes `ids` or none and answers with `states`: for each one the `id`, the `type`, the `value`, the `default`, the `writers` and the `readers`. A writer or a reader is the place in the spec, such as `panel[0].options.select`.

| Code | Cause |
|---|---|
| `state.exists` | A state variable with the id exists |
| `state.not_found` | No state variable has the id |
| `state.in_use` | A reference, a control or a pick reads or writes the variable |
| `state.bad_value` | The value does not fit the type, the bounds or the choices |
| `state.read_only` | The state variable is `cursor` |

## Selection Commands

`selection_set` takes `panel` and one of three forms: `entity` with `index`, `program` with `item`, or `clear`. The form with `program` picks the element `item` of the item map of a program that draws a piece in the panel, at the cursor. Its answer holds `program`, `item`, `step`, `fields` and `index`, as [Pieces](pieces.md#selection-and-picking) describes. It makes the same `pick` event, the same fields and the same state writes as a person's pick, with the origin of the caller. For a surface, `index` is `row * columns + column` in the grid of the entity. The answer holds `fields`, the [pick fields](../builder/scene-entities.md#pick-fields), or null. `selection_get` takes `panel` and answers with the fields of the last pick. A missing item fails with `selection.bad_item`. A `program` that the spec lacks fails with `program.not_found`, and a piece without an item map fails with `program.bad_pick`.

## Control Set

`control_set` takes `id` and `value`. The value fits the kind of the control: a number, a boolean, a text, `[lo, hi]` for a range, or a list for a group with `multiple`. The tool works on a control that is bound to `value` or to a state, and it makes the same `control` event as a person's edit. It answers with the value that the control holds now. A control that is bound to a parameter, an action, an intervention or the run fails with `control.not_settable`. Those targets have their own tools: `run_set_param`, `run_action`, `run_pause` and `intervene_set`.

## Cursor Commands

`cursor_get` answers with `cursor` and `sources`. The cursor has `step`, `first`, `latest`, `live`, `playing` and `speed`. Each source that keeps steps has an `id`, the `first` and `last` step it keeps, the number of `steps` it keeps, the number it `requested`, its `bytes` and a `note`, which is null unless the budget shortened the source. A spec that keeps no steps fails with `cursor.no_history`, and so does a move that holds the cursor.

`cursor_set` takes any of `step`, `live`, `playing` and `speed`. A step outside the kept steps goes to the nearest kept step, and the answer gives the step that the cursor took. `cursor_step` takes `count`, from -1,000 to 1,000 and not 0, and moves the cursor over that many kept steps. Both answer with `cursor`. A held cursor changes `panel_data`: its `data.cursor` states the step or the window, and the `version` changes.

## Annotation Commands

`annotation_set` takes `panel`, `annotation` and `dry_run`. The annotation is a table with an `id`, a `kind` and the keys of that [kind](../builder/annotations.md#annotation-table). The command creates it or replaces the annotation with the same id. `annotation_remove` takes `panel`, `id` and `dry_run`. Both answer with `panel`, `diff`, `warnings` and `applied`. An unknown id fails with `annotation.not_found`.

## Layout Lock

`layout_lock` takes `locked` (`true` or `false`) and optional `dry_run`. It answers with `locked`, `warnings`, `diff` and `applied`. When `locked` is `true`, every panel and control becomes non-closable, and `panel_create`, `panel_remove`, `control_create`, `control_remove`, `layout_place` and `layout_set` fail with `layout.locked`.

`panel_create`, `panel_patch`, `control_create` and `control_patch` also accept `closable`. Set `closable` to `false` to protect a single component even when the layout is unlocked. A remove command on a component with `closable = false` fails with `panel.not_closable` or `control.not_closable`.

## Events

The tools make the events that [Run Control and Events](run-control-and-events.md#events) lists. A wait with `events_wait` on the kinds `pick`, `state`, `control` and `cursor` is the way to know that a change reached the instrument. An agent reads `panel_data` after the event, because the version of the data changes when its inputs change.
