# Bound Text

A bound text formats live values and state into a short text. Its language has named fields, number formats and literal units, and no expressions.

## Format Language

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
options = { format = "Step {step:,d}  Token {token:d}  Value {value:.3f}", fields = { step = { state = "cursor", field = "step" }, token = { state = "token" }, value = { selection = "world", entity = "terrain", field = "value" } } }
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

The doubled braces `{{` and `}}` write a literal brace. A field binds a source, a column in the form `<source>.<column>`, or a [reference](state-and-references.md#reference-forms). A source gives its value at the cursor: a tensor of one element, or the last point of a series. A larger tensor shows its shape, such as `[16]`. A field without a value shows a dash. A format that does not fit the type of its value shows `?`.

An unknown field name, an open brace and a nested brace are `spec.bad_format`. So is a format that does not fit a type that the validator knows. A format has at most 1,024 characters and at most 16 fields. The output has at most 1,024 characters, and a longer output ends with an ellipsis. A bound text updates when one of its values changes, at most 10 times a second.

## Text Places

A bound text stands in three places:

- **Text panel.** The options `format` and `fields`. They exclude the static option `text`, and both together are `spec.bad_value`. A bound `value` slot is also the field `value`, and its source must hold one element.
- **Readout.** A control of the kind `readout` in a control row, with the same two options. A field that names a source shows its kept value at the cursor.
- **Annotation.** The keys `text` and `fields` of each [annotation](annotations.md) kind.

`panel.data` of a text panel gives the bound value of the slot `value`, and not the formatted text. The formatted text shows in `capture.frame` and in `panel.render`.
