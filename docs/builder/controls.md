# Controls

A control is a widget in the spec that acts on one target. [Spec Version 1](spec-version-1.md#controls) lists the keys of the `control` table and the forms of `bind`. This page lists the nine kinds, the segmented group and the keyboard shortcuts.

## Control Kinds

| Kind | Bind | Options | Value |
|---|---|---|---|
| `slider` | `param:<name>`, `value`, or `state:<id>` of a number or an integer | `min`, `max`, `step`, `log`, `value` | Number |
| `range` | `value`, or `state:<id>` of a range | `min`, `max`, `step`, `value`, `min_span`, `follow` | `[lo, hi]` |
| `toggle` | `intervene:<point>`, `param:<name>`, `value`, or `state:<id>` of a boolean | `value` | Boolean |
| `button` | `action:<name>` or `run:pause` | None | None |
| `picker` | `value`, or `state:<id>` of a text or an integer | `choices`, `value` | Text or integer |
| `text` | `value`, or `state:<id>` of a text | `placeholder`, `max_length`, `value` | Text |
| `segmented` | `value`, or `state:<id>` of a text, an integer or a list | `choices`, `multiple`, `value` | Text, integer or list |
| `readout` | None | `format`, `fields` | None |
| `transport` | `cursor` | `speed`, `keys` | The cursor |

A choice is a value, or a table `{ value, label, key }`. The option `choices_from` of a picker is not built, and the validator refuses it with `spec.not_built`. A text control applies its value when the person presses Return or leaves the field. A control bound to a state shows the value of the state, thus `state.set` moves it too. Each `[[control]]` table accepts `closable = false` to prevent close commands (`control.not_closable`). When `[applet]` sets `layout_locked = true`, every control stays non-closable until the layout unlocks (`layout.locked`).

The `min` and `max` of a slider or a range may be [references](state-and-references.md#reference-places). The control `window` in the example below reads the first and the newest step of the cursor:

```toml
[[control]]
id = "window"
kind = "range"
title = "Steps"
bind = "value"
options = { min = { state = "cursor", field = "first" }, max = { state = "cursor", field = "latest" }, value = [0, 2048], min_span = 16, follow = true }
```

A range with `follow = true` moves with its upper bound while its upper end equals that bound, and it keeps its span. A range thus follows the newest step until the person moves it. A value that `control.set` writes counts as a move. The option `min_span` is the smallest span that the range keeps.

## Segmented Groups

A segmented group is a row of 2 to 16 buttons. With `multiple = false`, one button is on and the value is its choice. With `multiple = true`, any number of buttons are on and the value is the list of their choices. A button that is on takes the accent color.

```toml
[[control]]
id = "measure"
kind = "segmented"
bind = "state:measure"
options = { choices = [{ value = "raw", label = "Raw", key = "r" }, { value = "zscore", label = "Z-Score", key = "z" }] }
```

## Keyboard Shortcuts

A choice of a segmented group may have a key. The key selects the choice, or turns it on or off in a group with `multiple`. The tooltip of the button shows the key.

```text
key = [ "shift+" ] ( "a" .. "z" | "0" .. "9" )
```

A shortcut acts only while the window has focus and no text field holds the keyboard. The grammar refuses chords with Command, Control or Option. The keys of the app win. The validator refuses a reserved key with `spec.reserved_shortcut`: Space, the arrow keys, Home, End, Escape, Tab, Return, Delete, and each single key that a view of the app uses, such as `F` and `U`. Two choices with one key are `spec.duplicate_shortcut`. A `control` event gets the key `key` when a shortcut made the change.

## Control Values

The command `control.set` sets the value of a control that is bound to `value` or to a state, as a person's edit does. It adds the same `control` event with the origin of the caller. A control bound to a parameter, an action, an intervention or the run takes its own command, and `control.set` refuses it with `control.not_settable`. A slider drag adds one event when the drag ends. A range event holds `[lo, hi]`, and a group with `multiple` gives a list.

## Readouts and Transport

A `readout` control shows a [bound text](bound-text.md) in a control row. A `transport` control moves the [time cursor](time-cursor.md#transport-control). A spec holds at most one transport, and a second one is `spec.too_many`.
