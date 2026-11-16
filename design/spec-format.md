# Spec format

A spec is a text file that describes an applet built from data: the sources it reads, the panels that draw them and the layout of those panels. This page defines version 0 of the format, its canonical TOML and JSON forms, the rules that check a spec and the manifest of a saved spec applet. It elaborates section 6 of the sprint 3 contracts, and the parts of sections 3 and 4 that a spec uses: the catalog ids that sources name and the tensor transforms that sources carry.

## What is built and what is planned

This branch builds the format and its library, and nothing that draws a spec:

- the library `caliper_builder_spec` in `src/builder/spec/`, with its headers in `src/builder/include/caliper_builder/spec/`, which parses a spec, validates it, writes it back in canonical TOML and JSON, and applies a JSON merge patch to one panel or source;
- the JSON Schema of the canonical JSON form, `src/builder/spec/schema/caliper-spec-v0.schema.json`;
- the tests in `src/builder/spec/tests/`.

These are planned and not built here: the spec runtime that draws a spec in the app, the loader's knowledge of the `kind = "spec"` manifest, the command handlers that edit a spec (`spec.*`, `source.*`, `panel.*`, `layout.*`), and the runtime in the web viewer that reads the canonical JSON. The library is not wired into the app until sprint 2 merges.

## Two examples

The example of the contracts, embedded from its test fixture:

```toml
--8<-- "src/builder/spec/tests/fixtures/attention-entropy.spec.toml"
```

A dashboard of three panels over one table source and one tensor source, also a test fixture:

```toml
--8<-- "src/builder/spec/tests/fixtures/dashboard.spec.toml"
```

## The document

A spec is one TOML table. Its keys, at the top level:

| Key | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `caliper_spec` | integer | Yes | | The format version. This page defines version `0`. |
| `applet` | table | Yes | | The applet's identity, in [The applet](#the-applet). |
| `source` | array of tables | No | none | The sources, in [Sources](#sources). |
| `panel` | array of tables | No | none | The panels, in [Panels](#panels). |
| `layout` | layout node | No | a vertical split of every panel | The arrangement of the panels, in [Layout](#layout). |
| `x` | table | No | none | An extension table, in [Extensions](#extensions). |

A spec with no panels is valid: `spec.new` creates one, and panels are added to it one by one.

Every key that this page does not name is an error, `spec.unknown_key`, so that a misspelt key never passes silently. The only exception is a table named `x`, which the format reserves for extensions.

### The applet

| Key | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `id` | id | Yes | | The applet's id in the workspace. It names the folder `applets/<id>/` and the files in it. |
| `title` | text, at most 256 characters, not empty | No | the `id` | The name shown for the applet. |
| `description` | long text, at most 4,096 characters | No | none | Prose about the applet. Comments are not kept by the canonical writer, so this is the place for notes. |
| `x` | table | No | none | Extensions. |

An id matches `[a-z][a-z0-9_-]{0,63}`: a lower-case letter, then at most 63 lower-case letters, digits, `_` or `-`. The applet, every source and every panel have ids of this form. Sources and panels have separate sets of ids, so a source and a panel may share one, as `loss` does in the example; two sources, or two panels, may not.

"Text" is a string with no control characters (U+0000 to U+001F and U+007F). "Long text" may also hold tabs, line feeds and carriage returns. Lengths count Unicode code points, as JSON Schema's `maxLength` does.

### Sources

A source is data that panels bind to. It has an id and exactly one of three forms; a source with none or more than one is `spec.source_form`.

| Key | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `id` | id | Yes | | The source's id, which bindings name. |
| `catalog` | catalog id | One of three | | A catalog source, named by its id `<kind>:<name>`. |
| `sql` | long text, at most 65,536 characters, not empty | One of three | | One read-only `SELECT` over catalog tables, which it names by table name. |
| `from` | id | One of three | | Another source of this spec, which this one derives from. |
| `transform` | array of transform tables | No | none | Tensor transforms, applied in order, in [Tensor transforms](#tensor-transforms). |
| `x` | table | No | none | Extensions. |

The three forms:

- **`catalog`** names a catalog entry. A catalog id is `<kind>:<name>`, where the kind is `table`, `tensor`, `live`, `feed` or `metric` and the name is text of at least one character; the whole id has at most 256 characters. The kind fixes the shape of the data: `table`, `feed` and `metric` sources are tabular, with columns, and `tensor` and `live` sources are tensors, with a dtype and a shape. The catalog module owns the grammar of names; the spec only checks the kind and that the name is not empty.
- **`sql`** is DuckDB SQL over the catalog's tables, such as `SELECT step, loss FROM run42_metrics ORDER BY step`, where `run42_metrics` is the catalog source `table:run42_metrics`. Its result is tabular. The spec library does not parse SQL: the data layer runs it on a read-only connection with a row limit and a time limit, and refuses anything that is not one `SELECT`.
- **`from`** names another source of the same spec, which may itself be derived. A chain of `from` must end at a `catalog` or `sql` source; a cycle is `spec.cycle`, and a name that is not a source of the spec is `spec.unknown_source`. A derived source has the shape of its origin after the origin's transforms, and then applies its own.

Transforms apply only to tensors. A transform on a tabular source, whether `catalog`, `sql` or derived from one, is `spec.bad_transform`. Tabular data is shaped with SQL instead.

### Tensor transforms

The transforms are the fixed, ordered list of operations of the contracts' section 4. A loader and a spec use the same records, and the browser runtime implements the same three. The spec library parses and checks them and works out their result shapes; the loaders module holds the reference implementation.

| `op` | Key | Type | Required | Default | Meaning |
|---|---|---|---|---|---|
| `slice` | `axis` | integer | Yes | | The axis to select along; negative counts from the end. |
| | `index` | integer | No | | Selects one position and drops the axis. Negative counts from the end. |
| | `start` | integer | No | the axis start | With `stop` and `step`, selects a range and keeps the axis, with Python's semantics. |
| | `stop` | integer | No | the axis end | The end of the range, exclusive. |
| | `step` | integer, at least 1 | No | `1` | The stride of the range. |
| `reduce` | `axis` | integer, or a list of 1 to 8 integers | Yes | | The axes to reduce over. |
| | `fn` | one of `mean sum min max std var norm argmax argmin entropy` | Yes | | The reduction. `entropy` treats the axis as a probability distribution, in nats. `argmax` and `argmin` take one axis only. |
| | `keepdims` | boolean | No | `false` | Keeps each reduced axis with size 1. |
| `normalise` | `mode` | one of `minmax zscore softmax l2` | Yes | | The rescaling. |
| | `axis` | integer | No | the whole tensor | The axis to rescale along. |

Every transform table may also hold `x`. A key that belongs to another `op` is `spec.unknown_key`, a missing required key is `spec.missing_key`, and an `op` outside the three is `spec.unknown_transform`. A slice has either `index` or a range (`start`, `stop`, `step`, each optional), never both; a slice with neither selects the whole axis. An axis is an integer from −8 to 7, because a tensor has at most 8 dimensions, and `index`, `start` and `stop` are integers within ±(2^53 − 1).

Each transform changes the dtype and the shape as the contracts say:

| `op` | Result shape | Result dtype |
|---|---|---|
| `slice` with `index` | the axis is removed | unchanged |
| `slice` with a range | the axis has the length of `range(start, stop, step)` over its size | unchanged |
| `reduce` | each reduced axis is removed, or kept with size 1 with `keepdims` | `f32`; `f64` for `f64` input; `i64` for `argmax` and `argmin` |
| `normalise` | unchanged | `f32`; `f64` for `f64` input |

A `reduce` that lists the same axis twice, `argmax` or `argmin` over several axes, and a slice with both `index` and a range are `spec.bad_transform` in any case. When the shape of the input is known, through the [source resolver](#the-source-resolver), the validator also checks each step against it: an axis out of range for the rank, an `index` out of range for the axis's size, and two axes that name the same dimension, such as 1 and −3 of a rank-4 tensor, are `spec.bad_transform`. When the shape is not known, these checks wait until it is.

### Panels

A panel is one primitive with its slots bound to sources.

| Key | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `id` | id | Yes | | The panel's id, which the layout names. |
| `primitive` | one of `series heatmap points histogram table image text` | Yes | | What the panel draws; an unknown name is `spec.unknown_primitive`. |
| `title` | text, at most 256 characters, not empty | No | the `id` | The panel's title. |
| `bind` | table of slot to binding | No | none | The data of each slot, in [Bindings](#bindings). |
| `options` | table | No | none | The primitive's options, in [Options](#options). |
| `x` | table | No | none | Extensions. |

A slot that the primitive does not have is `spec.unknown_slot`, and an option that it does not take is `spec.unknown_option`. A required slot that is not bound is `spec.missing_slot`. The `bind` table holds slots only; it has no `x`, because `x` is a slot name.

### Bindings

A binding is a string that names data of the spec's sources:

```text
binding  = source-id [ "." column ]
source-id = [a-z][a-z0-9_-]{0,63}
column    = one or more characters, at most 255, no control characters
```

- `"<source>"` names a whole source: a tensor, or a whole table.
- `"<source>.<column>"` names one column of a tabular source. The source id ends at the first `.`, and everything after it is the column's name, which may itself contain `.` or spaces, as DuckDB column names can.

A binding has at most 320 characters (`spec.string_too_long`). A binding whose source is not a source of the spec is `spec.unknown_source`. A binding that does not match the grammar, or that names a column of a tensor source, is `spec.bad_binding`. A column that the source does not have, when its columns are known, is `spec.unknown_column`. Every slot takes one binding, except `series.y`, which takes one binding or a list of 1 to 16; a list in any other slot is `spec.wrong_type`, and an empty list is `spec.missing_slot`.

### The primitives

The primitives are the live view types of Caliper LIB plus a table and text, so each has a wire form. In the rules below, a **column** is a binding `"<source>.<column>"` of a tabular source, a **vector** is a column or a rank-1 tensor, and a **numeric column** is a column whose DuckDB type is a number or a boolean. A column of a text, binary, nested or time type does not fit a numeric slot, and a type name the validator does not recognise is let through. Each rule is checked only when the shapes or columns it needs are known; a slot that breaks one is `spec.slot_mismatch`.

#### series

Lines over an index or over an x column.

| Slot | Required | Takes |
|---|---|---|
| `x` | No | A vector: a numeric or time column, or a rank-1 tensor. Without it, the x axis is the index 0, 1, 2 and so on. |
| `y` | Yes | One binding or a list of 1 to 16. Each is a numeric column, a rank-1 tensor (one line) or a rank-2 tensor (one line per row, along the last axis). |

When `x` is a column, every `y` is a column of the same source, so that rows line up. When `x` is a tensor of length n, every `y` is a tensor whose last axis has length n.

Options: `x_label`, `y_label`, `x_range`, `y_range`, `x_log`, `y_log`.

#### heatmap

A colour-mapped grid.

| Slot | Required | Takes |
|---|---|---|
| `values` | Yes | A rank-2 tensor `[rows, columns]`. |

Options: `x_label`, `y_label`, `colormap`, `range`.

#### points

A 2-D or 3-D point cloud, from columns or from one tensor of positions.

| Slot | Required | Takes |
|---|---|---|
| `x`, `y` | Yes, unless `positions` is bound | Vectors of the same length. |
| `z` | No | A vector; it makes the cloud 3-D. |
| `positions` | Yes, unless `x` and `y` are bound | A tensor `[n, 2]` or `[n, 3]`. |
| `color` | No | A vector of n values, mapped through `colormap`. |
| `size` | No | A vector of n values, the size of each point. |

`positions` excludes `x`, `y` and `z`. With columns, every slot binds columns of one source. With tensors, every vector has the same length n, which is the first axis of `positions`.

Options: `x_label`, `y_label`, `z_label`, `colormap`.

#### histogram

Counts of values in bins.

| Slot | Required | Takes |
|---|---|---|
| `values` | Yes | A numeric column, or a tensor of any rank, whose elements are all counted. |

Options: `x_label`, `y_label`, `bins`, `range`, `y_log`.

#### table

A table of rows.

| Slot | Required | Takes |
|---|---|---|
| `rows` | Yes | A whole tabular source, `"<source>"`. |

Options: `columns`. Each name in `columns` is a column of the source, when its columns are known, and no name appears twice.

#### image

An image.

| Slot | Required | Takes |
|---|---|---|
| `values` | Yes | A tensor `[H, W]` (one channel), `[H, W, 3]` (RGB) or `[H, W, 4]` (RGBA). |

Integer values are read in 0 to 255 and floating-point values in 0 to 1, unless `range` says otherwise. `colormap` applies to a one-channel image only and is ignored for colour images.

Options: `colormap`, `range`.

#### text

Static text, a bound value, or both.

| Slot | Required | Takes |
|---|---|---|
| `value` | No | A column, shown as its last value, or a tensor with exactly one element, shown as a number. |

Options: `text`. A text panel needs `value` or `options.text`, or both; with neither it is `spec.missing_slot`. With both, the text is shown above the value.

### Options

Options are small and concrete. Each primitive takes only the options its section lists.

| Option | Type | Default | Meaning | Primitives |
|---|---|---|---|---|
| `x_label`, `y_label`, `z_label` | text, at most 256 characters | no label | Axis labels. | series, heatmap, points (`z_label` only here), histogram |
| `x_range`, `y_range` | `[min, max]`, two finite numbers, `min < max` | fitted to the data | Fixed axis ranges. | series |
| `x_log`, `y_log` | boolean | `false` | A log scale on that axis. | series (both), histogram (`y_log`, the counts) |
| `range` | `[min, max]`, two finite numbers, `min < max` | the data's minimum and maximum; 0 to 255 or 0 to 1 for an image | The value range: the ends of the colormap for a heatmap or an image, and the binned range for a histogram. | heatmap, image, histogram |
| `colormap` | one of `viridis magma rdbu greys` | `viridis`; `greys` for a one-channel image | The colormap. | heatmap, points, image |
| `bins` | integer, 1 to 65,536 | `64` | The number of bins. | histogram |
| `columns` | list of 1 to 256 column names | every column, in the source's order | The columns shown, in this order. | table |
| `text` | long text, at most 16,384 characters | none | The static text. | text |
| `x` | table | none | Extensions. | all |

The four colormaps are the three that Caliper LIB's `show` offers, `viridis`, `magma` and `rdbu`, plus `greys` for one-channel images. A colormap outside the list is `spec.bad_value`, and a range whose `min` is not below its `max` is `spec.out_of_range`.

### Layout

The layout is a tree of nodes. A node is one of three things:

- a **panel id**, a string, which places that panel;
- a **split**, a table `{ split = "horizontal" | "vertical", ratio = <number>, children = [<node>, ...] }`, which divides its space among its children;
- **tabs**, a table `{ tabs = [<node>, ...] }`, which shows one child at a time.

| Node | Key | Type | Required | Default | Meaning |
|---|---|---|---|---|---|
| split | `split` | `horizontal` or `vertical` | Yes | | `horizontal` puts the children side by side, left to right; `vertical` stacks them top to bottom. |
| | `ratio` | number, 0.05 to 0.95 | No | `0.5` | The share of the first child. Only a split with exactly two children takes it; a split with more children shares its space equally. |
| | `children` | 2 to 64 nodes | Yes | | The children, in order. |
| | `title` | text, at most 256 characters, not empty | No | | The tab label, when the split is a child of tabs. |
| | `x` | table | No | none | Extensions. |
| tabs | `tabs` | 1 to 64 nodes | Yes | | The tabs, in order. A panel's tab label is its title. |
| | `title` | text, at most 256 characters, not empty | No | | The tab label, when these tabs are themselves a child of tabs. |
| | `x` | table | No | none | Extensions. |

The root is the `layout` key, so it is usually a `[layout]` table, but `layout = "<panel>"` is also a valid root. A table with both `split` and `tabs`, or neither, a split with fewer than two children, tabs with none and a `ratio` on a split of more than two children are `spec.bad_layout`. A panel id that is not a panel of the spec is `spec.unknown_panel`, and a panel placed twice is `spec.duplicate_placement`. A panel that the layout does not place is not drawn, and the validator warns with `spec.unplaced_panel`. A path from the root to a panel passes through at most 8 nodes, the root and the panel included (`spec.too_deep`), and the tree has at most 256 nodes (`spec.too_many`).

A spec with no `layout` places its panels in declaration order: one panel fills the applet, and several are stacked as one vertical split with equal shares.

### Extensions

A table named `x` may appear in the document's root, in `applet`, in each source, transform and panel, in `options` and in each layout table. Validation ignores its content, the canonical writers keep it, and the spec runtime never reads it. It is the place for a tool's own data, such as an editor's state. Its content follows the rules of [Values](#values) and counts toward the limits of the whole document. An empty `x` is dropped.

### Reserved names

Two names are reserved for work that is planned and not built, so that nothing else takes them:

- The primitive name `scene` is reserved for 3-D scenes. Version 0 refuses it as `spec.unknown_primitive`, like any other name outside the seven, and a later version of the format adds it with its slots.
- The catalog kind `recording` is reserved for recorded sessions, which the local database will store beside the loaders' tables. Version 0 refuses a `catalog` id of that kind as `spec.bad_value`, like any kind outside the five, and the kind is added when the catalog lists recordings.

## Values

A spec holds only values that the canonical JSON form can carry unchanged to a browser:

- tables (JSON objects), arrays, strings, booleans, integers and finite floating-point numbers;
- no dates or times, which TOML has and JSON lacks (`spec.wrong_type`);
- no JSON `null` (`spec.wrong_type`);
- no infinite or NaN numbers (`spec.out_of_range`);
- integers within ±(2^53 − 1), the integers that JavaScript reads exactly (`spec.out_of_range`);
- strings in valid UTF-8 (`spec.bad_value`).

A key that takes an integer accepts a floating-point number with no fractional part, such as `3.0`, because a JSON reader in a browser cannot tell `3.0` from `3`. A key that takes a number accepts an integer.

These rules apply everywhere, including inside `x`.

## The canonical forms

A spec has one canonical TOML text and one canonical JSON form. Both are a deterministic function of the parsed `Spec` value: writing, parsing and writing again gives the same text, and parsing either form gives the same `Spec`.

### Canonical TOML

The canonical TOML writer:

- writes `caliper_spec`, then, when they are present, a root `layout` that is a panel id and the root `x`, then the tables `[applet]`, each `[[source]]` followed directly by its `[[source.transform]]` tables indented by two spaces, each `[[panel]]` and `[layout]`, with a blank line before each table except the transforms;
- writes the keys of each table in a fixed order: `id`, `title`, `description`, `x` in `[applet]`; `id`, `catalog`, `sql`, `from`, `x` in a source; `op`, `axis`, `index`, `start`, `stop`, `step`, `fn`, `keepdims`, `mode`, `x` in a transform; `id`, `primitive`, `title`, `bind`, `options`, `x` in a panel; and `split` or `tabs`, `ratio`, `title`, `children`, `x` in a layout node;
- writes `bind`, `options`, `x` and nested layout nodes as inline tables on one line, with the keys of `bind`, `options` and `x` in byte order;
- writes a key only when the `Spec` value holds it, so an explicit default such as `keepdims = false` stays as written;
- writes strings as basic strings with escapes, or as multi-line basic strings when they hold a line feed; integers in decimal; and floating-point numbers in a form that reads back to the same value and always has a `.` or an exponent, so a range written `[0, 255]` is written back as `[0.0, 255.0]`;
- writes `series.y` as a single string or as a list, as it was given.

The writer does not keep comments, blank lines or the author's key order. That is a recorded limit of version 0: `description` is the place for prose.

### Canonical JSON

The canonical JSON form is the document's tables as JSON objects, under the same keys, so the arrays of tables are the arrays `"source"`, `"panel"` and `"transform"`. Keys are sorted in byte order, the text is compact UTF-8, and the empty arrays and tables that the TOML writer leaves out are left out. The JSON Schema `src/builder/spec/schema/caliper-spec-v0.schema.json` (draft 2020-12) describes this form. The web viewer reads it, so it holds nothing that is specific to TOML.

The schema checks the structure, the types, the keys of each table, the enumerations, the ranges, the slots and options of each primitive and the three source forms. It does not check what needs the whole document or the catalog: references, cycles, layout placement, shapes and columns. A document that the schema accepts may still fail validation; a document that validates always passes the schema.

The schema, embedded from its file so that this page cannot drift from it:

```json
--8<-- "src/builder/spec/schema/caliper-spec-v0.schema.json"
```

## Versioning

`caliper_spec` is the version of the format. Version 0 promises that a reader of version 0 reads every version 0 document the same way: the keys, their types, their defaults and their meanings on this page do not change. Because parsing is strict, any new key, primitive, option, colormap or transform, and any change of meaning, makes a new version. Data that does not change the meaning of a spec goes in `x` and needs no new version.

A reader refuses a version it does not know with `spec.unsupported_version`, whose message names the document's version and the versions the reader knows, and asks for a newer Caliper when the document is newer. It reports no other error for that document, because the keys of a later version would only add noise. A value of `caliper_spec` that is not an integer is `spec.wrong_type`.

A later reader keeps reading version 0. It migrates an older document in its JSON form, one version at a time, before decoding it, and the canonical writer then writes the newest version, so a saved spec moves forward only when it is saved. Version 0 has no migrations.

## Validation

Checking a spec has two stages, which the library keeps apart:

1. **Parsing** (`parse_toml`, `from_json`) reads the text and decodes it into a `Spec`. It checks the syntax, the [values](#values), the limits of the whole document, the types, the keys of each table, the required keys and the enumerations. Its errors in TOML carry a line and a column.
2. **Validation** (`validate`) checks a `Spec`, however it was made: ids, lengths, counts and ranges, references between sources, panels and the layout, cycles, the slot rules, and the shapes of transforms and slots when a [source resolver](#the-source-resolver) knows them. It reports errors and warnings.

A spec is valid when parsing succeeds and validation reports no error. Warnings do not make a spec invalid. Every diagnostic has a severity, a stable code, one plain sentence and a document path such as `panel[2].bind.y` or `source[1].transform[0].axis`, with indices from 0. For a spec parsed from TOML, the library maps each path to the line and column where that value is written, so validation errors point into the text too.

### Error codes

| Code | Severity | Stage | When |
|---|---|---|---|
| `spec.too_large` | error | parse | The text is longer than 512 KiB. |
| `spec.syntax` | error | parse | The text is not TOML, or not JSON, or a JSON object repeats a key. |
| `spec.too_deep` | error | parse, validate | Tables and arrays nest more than 32 levels deep, a dotted TOML key has more than 32 parts, or the layout is more than 8 nodes deep. |
| `spec.too_many` | error | parse, validate | A count is over its limit: the values of the document, sources, panels, transforms, `y` bindings, `columns`, the axes of a `reduce`, layout nodes or children. |
| `spec.string_too_long` | error | parse, validate | A string is longer than its limit. |
| `spec.wrong_type` | error | parse, validate | A value has the wrong type: a date or time, a `null`, a list in a slot that takes one binding, or any other mismatch. |
| `spec.out_of_range` | error | parse, validate | A number is outside its range, not finite, not a safe integer, or a `[min, max]` pair has `min >= max`. |
| `spec.bad_value` | error | parse, validate | A string is not one of its allowed values, holds control characters, is empty where it may not be, is not valid UTF-8, or repeats in `columns`; or a catalog id is not `<kind>:<name>`. |
| `spec.unknown_key` | error | parse, validate | A key the format does not have at that place. |
| `spec.missing_key` | error | parse, validate | A required key is missing. |
| `spec.unsupported_version` | error | parse, validate | `caliper_spec` is not a version this reader knows. |
| `spec.unknown_primitive` | error | parse | `primitive` is not one of the seven. |
| `spec.unknown_transform` | error | parse | `op` is not one of the three. |
| `spec.unknown_slot` | error | parse, validate | `bind` names a slot the primitive does not have. |
| `spec.unknown_option` | error | parse, validate | `options` holds an option the primitive does not take. |
| `spec.bad_id` | error | validate | An id does not match `[a-z][a-z0-9_-]{0,63}`. |
| `spec.duplicate_id` | error | validate | Two sources, or two panels, share an id. |
| `spec.source_form` | error | validate | A source has none, or more than one, of `catalog`, `sql` and `from`. |
| `spec.unknown_source` | error | validate | `from` or a binding names a source that the spec does not have. |
| `spec.cycle` | error | validate | A chain of `from` returns to a source it has passed. |
| `spec.bad_binding` | error | validate | A binding does not match the grammar, or names a column of a tensor. |
| `spec.unknown_column` | error | validate | A binding or `columns` names a column the source does not have. |
| `spec.missing_slot` | error | validate | A required slot is not bound. |
| `spec.slot_mismatch` | error | validate | A slot's data does not fit it: a tensor where a column is needed or the reverse, the wrong rank, lengths that differ, columns of different sources, or a column that is not numeric. |
| `spec.bad_transform` | error | validate | A transform does not apply: on a tabular source, an axis or index out of range, a repeated axis, `index` together with a range, or several axes for `argmax` or `argmin`. |
| `spec.bad_layout` | error | parse, validate | A layout node has the wrong form. |
| `spec.unknown_panel` | error | validate | The layout names a panel that the spec does not have. |
| `spec.duplicate_placement` | error | validate | The layout places a panel twice. |
| `spec.not_found` | error | patch | The panel or source to patch does not exist. |
| `spec.bad_patch` | error | patch | The patch is not a JSON object, nests too deeply, or would change or remove the id. |
| `spec.internal` | error | any | The library failed unexpectedly, for example because the source resolver threw an exception or memory ran out. |
| `spec.not_portable` | warning | validate | A bound source has no wire form, so the spec runs in the app but not in the browser. |
| `spec.unresolved_source` | warning | validate | The resolver does not know a catalog source. |
| `spec.unused_source` | warning | validate | No panel binds a source, directly or through `from`. |
| `spec.unplaced_panel` | warning | validate | The layout does not place a panel. |

### The source resolver

Validation takes an optional `SourceResolver`, a callback that maps a catalog id to what the catalog knows of it, as a small struct of the spec module: the kind, the columns with their DuckDB types, the dtype, the shape (with −1 for a size that is not fixed) and the `wire` flag. The spec module does not depend on the catalog module; the command layer passes a resolver that asks the catalog.

Without a resolver, validation still knows each catalog source's kind from its id, and knows that `sql` sources are tabular, so it checks every rule that needs only the kind. With a resolver it also checks columns, ranks, sizes and portability, and it reports a catalog id that the resolver does not know with the warning `spec.unresolved_source`: a live session may not be attached yet, and a loader may not have run yet, so the spec stays valid.

Validation also returns what it learned of each source: its kind and, when known, its columns, dtype and shape after the transforms. The builder shows these, and an agent reads them through `spec.validate`.

## The browser

The web viewer reads the canonical JSON and draws a spec with the same primitives. The portability rule: a spec runs in the browser when every catalog source that a panel binds, directly or through `from`, has a wire form, which its catalog entry says with `wire = true`. The validator reports each bound catalog source with `wire = false` as the warning `spec.not_portable`, at that source; the spec stays valid and runs in the app.

- The primitives are the live view types plus table and text, which have wire forms.
- The browser runtime implements the three tensor transforms; a producer may also compute them as a data view and send only the result.
- A `sql` source has two routes to the browser. In the first, the query runs where the data lives, in the producer, the app or a hub, and its result travels as a table. This is the default of version 0. In the second, the query runs in the page with DuckDB-WASM, the official WebAssembly build of DuckDB, over data loaded into the page. That is the later path, and the format needs no change for it, because a spec says what to query and not where. A page that `caliper.serve()` serves from a script runs a `sql` source only once one of the two routes exists for it.
- The JSON form holds no dates, no `null`, no non-finite numbers and no integers that JavaScript cannot read exactly, and its keys outside `x` are ASCII.

## The manifest of a spec applet

A saved spec applet is a folder in the workspace, `applets/<id>/`, with two files: the manifest `<id>.caliper.toml` and the spec `<id>.spec.toml`. It has no library. The manifest follows [Manifest](../docs/references/manifest.md) with these differences:

```toml
[applet]
id = "local.attention-entropy"        # the manifest's own grammar, lower-case reverse DNS
name = "Attention entropy"
version = "0.1.0"
summary = "Entropy of each attention head while the model trains."
kind = "spec"
spec = "attention-entropy.spec.toml"
```

| Key | Type | Required | Meaning |
|---|---|---|---|
| `applet.kind` | string | No | `"library"` (the default, today's compiled applets) or `"spec"`. |
| `applet.spec` | string | Yes, for `kind = "spec"` | The spec's file name, `<id>.spec.toml`, in the same folder. It is a file name only, with no `/`, `\` or `..`. |

- `[compat].abi_epoch` is not required for a spec applet and is not written, because there is no binary to check. `[compat].torch`, `[build]` and `[services]` do not apply either. `[compat].min_host` still applies.
- The manifest's `applet.id` keeps the manifest's grammar (`[a-z0-9.-]`, dot-separated), which has no `_`. `applet.save` writes `local.<id>`, with each `_` of the workspace id replaced by `-`, unless the caller names another id; it refuses an id that another applet of the workspace already has.
- `applet.save` writes `name` from the spec's `applet.title` (or its id), `summary` from `applet.description` when that is one line of at most 200 characters, and `version = "0.1.0"` for a new applet.
- The loader learns this kind after sprint 2 merges. Until then, a host lists such a folder as a failed applet, because it finds no library.

## Limits

Every size is bounded, and each limit has its error code:

| What | Limit | Code |
|---|---|---|
| The text of a spec, TOML or JSON | 512 KiB, half of Caliper API's 1 MiB request limit | `spec.too_large` |
| Values in the document, counting every table, array and scalar, `x` included | 65,536 | `spec.too_many` |
| Nesting of tables and arrays | 32 levels | `spec.too_deep` |
| Sources | 64 | `spec.too_many` |
| Panels | 64 | `spec.too_many` |
| Transforms of one source | 16 | `spec.too_many` |
| Bindings in `series.y` | 16 | `spec.too_many` |
| Names in `options.columns` | 256 | `spec.too_many` |
| Axes of one `reduce` | 8 | `spec.too_many` |
| Layout depth | 8 nodes | `spec.too_deep` |
| Layout nodes | 256 | `spec.too_many` |
| Children of one split or tabs node | 64 | `spec.too_many` |
| An id | 64 characters, by its grammar | `spec.bad_id` |
| A title, a label or a layout title | 256 characters | `spec.string_too_long` |
| `applet.description` | 4,096 characters | `spec.string_too_long` |
| A catalog id | 256 characters | `spec.string_too_long` |
| A column name, in a binding or in `columns` | 255 characters | `spec.string_too_long` |
| A binding | 320 characters | `spec.string_too_long` |
| `sql` | 65,536 characters | `spec.string_too_long` |
| `options.text` | 16,384 characters | `spec.string_too_long` |
| Any other string, such as one inside `x` | 65,536 characters | `spec.string_too_long` |
| An integer | ±(2^53 − 1) | `spec.out_of_range` |
| Diagnostics in one report | 256; errors are kept before warnings, and the rest are dropped | |

The parser checks the text size and the length of dotted TOML keys before it parses, and the value count and nesting before it decodes, so a hostile document costs a bounded amount of work. TOML's dotted keys need the early check because each part is a nested table, which the TOML library would otherwise build and free recursively. Validation follows chains of `from` without recursion and stops walking the layout at its limits.

### Recorded limits of version 0

- The canonical writer keeps no comments; `description` carries prose.
- Validation does not know the columns of a `sql` source's result, so bindings to them are checked only against the grammar. The command layer may later describe a query with DuckDB and check them too.
- A spec draws data and does not edit it: there are no controls, such as sliders or selections, no linked selection between panels and no time windows over live sources.
- A series has no per-line labels or colours; each line is labelled by its binding.
- A split of more than two children shares its space equally.
- A `sql` source cannot run in a page that `caliper.serve()` serves.

## The library

`caliper_builder_spec` follows the rules of the builder libraries: C++17, the namespace `caliper_builder::spec`, and no exceptions across its headers. Its headers are in `caliper_builder/spec/`:

- `spec.h` holds the value types (`Spec`, `Applet`, `Source`, `Transform`, `Panel`, `Options`, `LayoutNode`), the limits as constants, `Diagnostic` and its JSON form for command results.
- `format.h` holds `parse_toml`, `from_json`, `from_json_text`, `to_toml`, `to_json`, `to_json_text`, the per-item conversions that `panel.create`, `source.add` and `layout.set` need (`panel_from_json` and the like), and `locate`, which maps diagnostics to lines and columns.
- `validate.h` holds `SourceResolver`, `ResolvedSource`, `validate`, which also returns a `SourceInfo` for each source, and `check_toml`, which parses and validates in one call.
- `patch.h` holds `patch_panel` and `patch_source`, which apply an RFC 7386 JSON merge patch to one panel or source and revalidate the spec. A patch that would change the item's id, or that does not decode, is refused and leaves the spec unchanged; otherwise the result holds the patched spec and the full validation, and the command layer decides whether errors elsewhere refuse the change.

## Tests

The tests in `src/builder/spec/tests/` run as the ctest test `builder_spec`. They check:

- the fixtures in `tests/fixtures/`: the contracts' example, the dashboard above, a spec that uses every primitive, option, transform and extension place, and a minimal spec;
- round trips: TOML to `Spec` to TOML to `Spec` gives an equal `Spec` and the same text, and `Spec` to JSON to `Spec` gives an equal `Spec`, for every fixture and for a spec built in code;
- the canonical TOML of the contracts' example, line by line;
- shape propagation through `from` and each transform, and the slot, option and layout rules;
- every code of the table above at least once, and no code outside it;
- hostile input: a text over 512 KiB, a 1 MiB string, inline arrays nested 400 deep, a dotted key of 100,000 parts, JSON nested 100,000 deep, 70,000 values, 2,000 unknown keys, cycles in `from`, duplicate ids and a resolver that throws;
- merge patches, against the examples of RFC 7386 and the refusal rules;
- the JSON Schema: each valid fixture's canonical JSON and further valid documents pass it, and 41 invalid documents fail both the schema and the library.

The schema test reads the schema through a test-only checker, `tests/schema_check.h`, which implements exactly the keywords the schema uses and refuses a schema that uses any other. The test also writes each document it checks into its build folder, and during development those 49 documents were checked with the reference Python validator, jsonschema 4.26 with draft 2020-12, which agreed on every one.

## Decisions

- **Separate id sets for sources and panels.** The contracts' example has a source and a panel both named `loss`, and the layout names only panels, so the two sets never meet.
- **No `x` table in `bind`.** `x` is a slot of `series` and `points`, so in `bind` it is a slot. The contracts allow `x` "at any level"; this is the one exception, and the smallest fix.
- **`horizontal` means side by side.** A horizontal split arranges its children along the horizontal axis, as a horizontal box layout does.
- **`ratio` only with two children.** The contracts give one `ratio` per split; a split of more children shares its space equally, which keeps one number per split.
- **A default layout.** A spec with no `layout` stacks its panels, so that a new spec draws without one.
- **Four colormaps.** The three of Caliper LIB's `show`, so a spec and a live view look the same, plus `greys` for one-channel images.
- **`bins` from 1 to 65,536 with a default of 64,** as in Caliper LIB's `histogram`.
- **Ranges as `[min, max]` pairs.** One key per range is smaller than separate ends, and a fixed range needs both.
- **Strict parsing, with the schema as a second, partial check.** The library is the authority; the schema lets a browser or an outside tool check structure without it.
- **A test-only schema checker rather than a fetched validator.** The C++ validators for nlohmann/json under the licences this repository accepts, pboettch/json-schema-validator and valijson, implement draft 7, not 2020-12, so they would judge this schema by other rules; jsoncons implements 2020-12 but brings its own JSON type under the Boost licence. A checker of exactly the keywords the schema uses, which refuses any other keyword and is itself tested against the draft's rules, cannot skip a rule silently, and adds no dependency. The reference Python validator agreed with it on every test document.
- **Duplicate keys in JSON are a syntax error,** as they are in TOML, so that a repeated key cannot hide a typo.
- **Lengths in code points,** so the limits and JSON Schema's `maxLength` agree.
- **Safe integers only,** so that the JSON form reads the same in a browser.
- **No dates anywhere, `x` included,** because the JSON form has none.
- **A text limit of 512 KiB,** so that a spec fits one Caliper API request with room for its envelope.
- **The resolver describes catalog ids only,** which keeps the spec module free of the catalog and of DuckDB; the columns of a `sql` result are left to the command layer.
- **A patch cannot change an id,** because bindings and the layout name it; a rename is a separate command, not in v0.
- **The spec applet's manifest omits `abi_epoch`,** because a spec has no binary, and its id keeps the manifest's grammar, which has no `_`.

## Defects in the contracts

- Section 6 allows a table named `x` "at any level", but `x` is also a slot of `series` and `points`. The fix: `bind` has no extension table.
- Section 6 lets a split hold several children with one `ratio`, without saying what the ratio means for more than two. The fix: `ratio` is allowed only with exactly two children.
- Section 2 gives workspace applet ids an `_`, which a manifest id may not hold. The fix: `applet.save` maps `_` to `-` and refuses a clash.
- The manifest's `[compat].abi_epoch` is required today, which a spec applet cannot meet. The fix: the loader will not require it for `kind = "spec"`.
- A `sql` source runs in the browser by one of two routes: the query runs where the data lives and the result is streamed, which is the default of version 0, or it runs in the page with DuckDB-WASM, which is the later path. Neither route is built yet.
