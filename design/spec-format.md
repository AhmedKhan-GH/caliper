# Spec format

A spec is a text file that describes an applet built from data: the sources it reads, the panels that draw them and the layout of those panels. This page defines version 0 of the format, its canonical TOML and JSON forms, the rules that check a spec and the manifest of a saved spec applet. It elaborates section 6 of the sprint 3 contracts, and the parts of sections 3 and 4 that a spec uses: the catalog ids that sources name and the tensor transforms that sources carry.

## What is built and what is planned

Built:

- the library `caliper_builder_spec` in `src/builder/spec/`, with its headers in `src/builder/include/caliper_builder/spec/`, which parses a spec, validates it, writes it back in canonical TOML and JSON, and applies a JSON merge patch to one panel or source;
- the JSON Schema of the canonical JSON form, `src/builder/spec/schema/caliper-spec-v0.schema.json`;
- the tests in `src/builder/spec/tests/`;
- the command handlers that edit a spec (`spec.*`, `source.*`, `panel.*`, `layout.*`), in the builder session ([Builder session](builder-session.md));
- the first slice of the spec runtime in `src/host/spec_runtime/`, which draws a spec's panels natively from the data the session resolves, with its development harness `caliper-spec-shot` and the renderer behind `panel.render`, described in [The spec runtime](#the-spec-runtime);
- the first slice of the browser viewer: a saved spec applet opens in a browser tab from the running app, laid out and drawn as the app draws it from the data the app resolves, described in [What is built of the browser viewer](#what-is-built-of-the-browser-viewer).

The loader knows the spec applet's manifest (item 5 of the integration order in [Builder overview](builder-overview.md)): the manifest parser reads `kind = "spec"` without the builder, the loader lists spec applets from the search folders, their subfolders and the open workspace's `applets/` folder, and with the builder it validates a spec applet's spec through this library ([The manifest of a spec applet](#the-manifest-of-a-spec-applet)). The command handlers that edit a spec are built in the session ([Builder session](builder-session.md)).

Planned: the page of the app that shows a spec applet, which calls the runtime; the data of `live:`, `feed:` and `metric:` sources in the runtime; and the rest of the browser viewer: its data from a hub or a page that `caliper.serve()` serves, the live sources over the stream, and the tensor transforms in the page.

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
| `color` | No | A vector of n values, mapped through `colormap`; class labels take the categorical palette instead (below). |
| `size` | No | A vector of n values, the size of each point. |

`positions` excludes `x`, `y` and `z`. With columns, every slot binds columns of one source. With tensors, every vector has the same length n, which is the first axis of `positions`.

**Class labels.** A `color` whose data has an integer type (an integer or boolean tensor, or a column of an integer or boolean DuckDB type) and holds only whole numbers from 0 to 255, with at most 16 distinct values, is a vector of class labels. It is drawn with the categorical palette of Caliper LIB's points view, label L in its colour L mod 16, as `points(..., classes=K)` draws classes, with a legend of the labels in place of the colormap's scale, and `colormap` does not apply to it. Any other `color`, a float vector of whole numbers included, is a value mapped through `colormap`. The rule is inferred from the data rather than written as an option, because an option would be a new key, which the strict parser of version 0 refuses (see [Versioning](#versioning)), and because the dtype already tells labels from values: a loader writes labels as integers and measurements as floats. A browser runtime applies the same rule.

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

### What is built of the browser viewer

The first slice opens a saved spec applet in a browser tab from the running app. Caliper API (`docs/agents/http-api.md`, GET /v0/spec) and Caliper CLI (`docs/reference/cli.md`, The spec page) are its reference.

- **The app resolves, the browser draws.** The page computes no transform: it asks the app for each panel's data through the read-only command `panel.data`, which resolves the panel as the native runtime does (`Session::resolve()` over the store and the served sources, transforms included, so a `sql` source runs where the data lives, the first route above) and sends it compact and bounded with the native runtime's own bounds (`src/builder/include/caliper_builder/session/panel_data.h`): a line of more than 8,192 points as the min/max envelope of 2,048 buckets, a grid or image sampled at a stride to at most 2,048 on a side, at most 100,000 points, a histogram's bin counts with the runtime's range and bin rule rather than its values, at most 1,000 rows of a table with a text cell cut at 256 characters, and a text panel's one value. One implementation of the transforms and of the bounds is kept, and a browser never receives more than a panel shows.
- **The encoding.** JSON for tables, cells and counts; arrays of numbers as `{"dtype": "f32" | "f64", "shape", "base64"}`, the little-endian bytes in base64 (f64 for a series' points, f32 for grids, images and clouds, as the runtime keeps them). A binary body would be smaller, but every command answers one JSON envelope; the bounds keep a heatmap's answer under 23 MB, and the largest answer is a full RGBA image of 2,048 by 2,048, about 89 MB, for which bytes rather than f32 values are the next step.
- **Refresh.** Each answer carries a `version`, a short hash of the spec's canonical JSON, the panel's id, the catalog's revision and the panel's generation. The page sends the version it has as `known`, and the app then answers `unchanged` from the cheap probe of the generation without resolving anything. The page asks once a second for the spec (`spec.get` with `applet`, which reads a saved applet's files without opening it) and for every panel it shows; a live panel's data therefore follows its view at most once a second, through these polls rather than the stream, and costs the app one probe a second while nothing changed and one resolve per new snapshot.
- **The page** is a new entry of the web package beside the viewer page, `web/src/spec/`, built by `web/build.mjs` into `spec.html`, one file with its script and its faces inside it, whose checked-in copy `python/caliper/_web/spec.html` the executable embeds (`cmake/embed_spec_page.cmake`), so the app serves it without Node and without reading a file. It reads the token and the applet's workspace id from its address, and without an applet it lists the workspace's saved spec applets.
- **The layout** is a port of `compute_spec_layout()` to TypeScript (`web/src/spec/layout.ts`), with the titles and binding names of `spec_layout.cpp`; `web/test/spec-layout.spec.ts` runs the cases of the native tests against it, and they agree to the pixel.
- **The look** is the native runtime's, drawn on a canvas: the app's colours, the two faces at the app's sizes, the title strip with the live views' info string, the plots laid out by ImPlot's rules (its padding, tick locator, minor ticks and grid, log ticks of 1, 2 and 5, and the legend in the north-east corner), heatmaps and images over index axes with a heatmap's colorbar, the colormaps of the web package (the live views' own tables, plus greys), points as squares of the live views' sizes with their colormap's scale or, under the class-label rule of [points](#points), the categorical palette and a legend, the house table with whole rows, and the one colour key of the spec. The states are the runtime's noun phrases with the source's catalog id under them. The web package's live views are not reused as views: they draw wire snapshots with their own toggles, gestures and selections, so the page reuses their parts, the colormap tables, the palette and the number formats.
- **Opening it.** With `--api` the app prints the page's link, `http://127.0.0.1:<port>/v0/spec?token=<viewer token>&applet=<id>`, and writes it without the applet to the API file as `spec_url`.
- **Security.** The link carries a viewer token, a second random token of the run, which opens only the page and the three commands it reads with, `spec.get`, `applet.list` and `panel.data`, always from a new origin, `viewer`, from which the command layer runs only commands that change nothing (`command.not_allowed` otherwise). The loopback, `Host` and `Origin` checks hold as for every request; the page answers with a content security policy that lets it reach only its own origin, and without a referrer.
- **Tests.** `builder_session` (`test_panel_data.cpp`) checks the data of the dashboard's three panels from a saved applet, the version and `known`, every primitive's bounds, the states and the viewer origin; `builder_commands` the table; `caliper_api_commands` and `caliper_tests` the viewer token over HTTP and the routes; and the web package's Playwright tests check the layout port, every primitive, state, tab and request against a mock of the app, and the page end to end against `caliper --hidden --api` after a command script imported a Parquet and a safetensors file and saved the dashboard (`web/test/spec-e2e.spec.ts`, which needs `CALIPER_BIN`).

Recorded limits of this slice: the page polls rather than following the stream; it draws no tooltips, no selection and no zoom; a 3-D cloud is drawn in its turned box with its far walls but without the walls' grid and tick labels; a series' log ticks and a colormap's scale use the web faces, so text widths can differ by a pixel from ImGui's; the class-label rule runs on the points the app sends, which past 100,000 points are a sample of them, so a cloud of more than 100,000 points can show as classes in the browser where its full data has more than 16 labels; and the page reads the saved files of an applet, so the edits of the open spec show in the browser once it is saved.

## The spec runtime

The spec runtime draws the panels of a spec in the app's own look. It is built in `src/host/spec_runtime/` as the library `caliper_spec_runtime`, only with the CMake option `CALIPER_WITH_BUILDER`, and the page that shows a spec applet will be a thin caller of it.

- **Layout.** `compute_spec_layout()` lays the layout tree out as whole-pixel rectangles, without ImGui: a split gives its first child `ratio` of its length (or equal shares to more children), tabs show one child under a strip of labels, and a 1 px rule separates the children of a split. Each panel has a minimum of 12 by 8 lines of the UI face, which grows with the text scale; a split keeps every child at its minimum while the space allows and otherwise shares the space in proportion to the minimums, so no rectangle leaves the area. The tab shown in each tabs node is the runtime's state, by the node's path of child indices.
- **Data.** A `SpecPanelSource` gives a panel's resolved data; `SessionPanelSource` asks `Session::resolve()` for the open spec and `resolve_panel()` over the session's store for any other spec. The runtime resolves a panel when its fragment, the spec's sources or the source's revision (the catalog's) change, never once a frame, and only while it is shown: a panel in a hidden tab is resolved when its tab is chosen. `prepare_panel()` then turns the slots into arrays to draw, within bounds that are recorded on the panel: a series line of more than 8,192 values becomes a min/max envelope of 2,048 buckets, as a live series does; a heatmap or image wider or taller than 2,048 is sampled at a stride; a cloud of more than 100,000 points is sampled at a stride; a table shows at most 1,000 rows; a cell or value is cut at 256 characters.
- **Drawing.** `draw()` draws each panel in its own child window: a title strip with the title on the left and the live views' info string in mono on the right ("4,000 steps  2 lines", "48×48  0 … 1", "1,000 of 4,000 rows"), then the primitive. A series and a histogram are ImPlot plots in the house style; a heatmap and an image are textures over index axes in the mono face, as the live views draw them, with a colorbar and its values beside a heatmap; points are squares of the live views' sizes, in ImPlot for 2-D and ImPlot3D for 3-D, with the colormap's scale as their key, or a legend of the classes over the top right corner for class labels, and a 3-D box turned and scaled to fill its panel on opening and on resize; a table is the house table, its columns as wide as their content and together at the left, its figures right-aligned in one format per column, ending on the last whole row that fits; a text panel shows its text over its value. A histogram's counts are on a linear axis, and on a log axis only with `y_log`, which starts under 1 so that a bin of one value shows. The colormaps are the live views' own tables, plus greys. A tab has no strip of its own: its tab carries the title and the tab bar the info string.
- **One colour key.** Every name that a series draws gets the next colour of the data palette, the house's class pigments and then the live views' categorical palette, in the order of the spec's panels, and keeps it in every panel: a table column of the same name carries its colour as a swatch beside its head.
- **Titles.** A panel without a title goes by what it draws, never by its id: the y columns of a series ("loss, accuracy"), "value by step" for points over two columns, the catalog name of a whole source ("run42_metrics"), the table a `sql` source reads first, or the primitive's name.
- **States.** A panel whose data is not there shows a short noun phrase and, under it in mono, the catalog id of its source: "Source offline" for a `live:`, `feed:` or `metric:` source the session cannot read yet, "No data", "Mismatched data", "Data too large", "Query error", "Missing source", "Missing column", "Missing data", "Transform error", or "Data error". Nothing that a slot holds can crash the drawing.
- **Textures** of heatmaps and images are ImGui user textures in RGBA8, which the renderer's ImGui backend creates and destroys with the font atlas, so the runtime needs no renderer of its own and the colours are byte for byte those of `map_f32_to_rgba8`, the CPU reference of the live views' colormaps.
- **Seams for the pages.** `draw()` takes any area of a window on the frame thread; `set_preview()` swaps one panel's fragment and data for a preview's until `clear_preview()`; `set_overlay()` is called after each panel's content with the panel's rectangles, for the widget agent's prompt row and diff card.
- **Offscreen.** `SpecOffscreen` draws into the renderer's offscreen canvas, as `caliper shot` does, with ImGui, ImPlot and ImPlot3D contexts of its own in the house style; it shares the font atlas of the context that is current when it is made, and puts the caller's contexts back after each call. It needs nothing of the app's page machinery: the seam is the renderer's canvas. `OffscreenPanelRenderer` draws one panel over a canvas and encodes it as PNG, and `panel_render_output()` makes the result of `panel.render` from it, so the app's `SessionHost::render_panel()` is two calls.
- **`caliper-spec-shot`**, a development tool that is not installed, draws a saved spec applet of a workspace (`--workspace <dir> --applet <id>`) or a spec file (`--spec <file>`) at `--size WxH` into `--out <file.png>`, with `--tab <path>=<index>` to show another tab. It reads the applet through its manifest and runs no command, so the workspace is not changed.
- **Tests.** The ctest entry `caliper_spec_runtime` checks the layout of the fixtures at two sizes and two text scales, tabs, the shares of a split and the titles, and the prepared data of every primitive, its options, states and bounds, headless. `caliper_spec_runtime_gpu` draws each primitive into an offscreen canvas and checks image properties that do not depend on exact pixels (a line's key colour rising to the right, a heatmap's corner cells in the colormap's end colours, an image's rows in order, the number of a table's lines of text, a state with no plot), the runtime over the dashboard with its rules, preview and overlay, the textures' lifetimes, `panel.render` through a fake `SessionHost` bound to a session over a real workspace, and `caliper-spec-shot` as a process over a workspace whose Parquet and safetensors files were imported, and whose dashboard was built through commands and saved.

Recorded limits of this slice: the runtime resolves on the frame thread, so a slow query holds a frame once when the data changes; a series' x column of a time type is not read; class labels written as floats are drawn as values; and without a current ImGui context only one offscreen canvas may exist at a time, because the house faces it loads are global.

## The manifest of a spec applet

A saved spec applet is a folder in the workspace, `applets/<id>/`, with two files: the manifest `<id>.caliper.toml` and the spec `<id>.spec.toml`. It has no library. The manifest follows Manifest (`docs/applets/manifest.md`) with these differences:

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
| `applet.kind` | string | No | `"library"` (the default, today's compiled applets) or `"spec"`. Any other value is a manifest error. |
| `applet.spec` | string | Yes, for `kind = "spec"` | The spec's file name, `<id>.spec.toml`, in the same folder. It is a plain file name: not empty, not `.` or `..`, at most 255 bytes, with no `/`, `\`, `:` or control character. |

- `[compat].abi_epoch` is not required for a spec applet and is not written, because there is no binary to check. The loader refuses it as a manifest error, and likewise `[compat].torch`, `[build]`, `[services]` and `[data]`, so that a library's manifest with `kind = "spec"` added by mistake is never taken for a spec applet. `[compat].min_host` still applies. `[data].schema` has no meaning for a spec applet, which keeps no data of its own: the host gives it no data folder.
- The manifest's `applet.id` keeps the manifest's grammar (`[a-z0-9.-]`, dot-separated), which has no `_`. `applet.save` writes `local.<id>`, with each `_` of the workspace id replaced by `-`, unless the caller names another id; it refuses an id that another applet of the workspace already has. The loader never compares the folder's name with the manifest id.
- `applet.save` writes `name` from the spec's `applet.title` (or its id), `summary` from `applet.description` when that is one line of at most 200 characters, and `version = "0.1.0"` for a new applet.
- The spec names the format version it needs with `caliper_spec`. When it lists the applet, the loader reads that line without parsing the TOML and refuses a version it does not read.

### The spec applet in the loader

Built in `src/host/applet_manifest.*`, `src/host/applet_loader.*` and `src/host/applet_spec.*`, with Manifest, Refusal messages and Caliper CLI (`docs/applets/manifest.md`, `docs/reference/refusal-messages.md`, `docs/reference/cli.md`) as the reference:

- The manifest parser reads `kind` and builds without the builder. The loader lists a spec applet with the statuses of a compiled one and chooses one copy per manifest id by the same rule, whatever the kind. It never maps or calls anything for it, and gives it no data folder.
- A spec applet is found in the folders of the search order, as a manifest directly in a searched folder or one folder down (`<folder>/<name>/<stem>.caliper.toml`, as a workspace keeps them), and in the open workspace's `applets/` folder, which the loader is told with `AppletLoader::set_workspace` and which `applet_search_dirs` places after the development folders and before the user applets folder. The workspace's folder holds spec applets only: a compiled applet's manifest there is refused and its library never opened.
- Listing is cheap: the manifest (at most 64 KiB and no dotted key of more than 32 parts, one folder down or in the workspace), the spec file (a regular file in the manifest's folder, not a symbolic link, at most 512 KiB), its `caliper_spec` line, `min_host`, the spec runtime and the format version.
- With the builder (`CALIPER_WITH_BUILDER`), `caliper_host_lib` links `caliper_builder_spec`, and `AppletLoader::check_spec` reads the spec again, parses it with `parse_toml`, validates it with `validate` and locates every diagnostic; the first error becomes the applet's reason. `read_spec_file` in `applet_spec.h` returns the parsed spec for the app. Without the builder every spec applet is refused with the reason that the build has no spec runtime.
- `launch()` on a spec applet validates it and returns false, leaving a valid one `READY` with a recognisable reason until the app opens its page instead.
- `caliper list` and `caliper check` print each applet's kind, `caliper check` validates a spec applet fully and prints the validator's diagnostics with their lines and columns, `caliper install` refuses a spec applet, and `GET /v0/applets` has a `kind` field.
- A test saves an applet through a `Session` and lists it through the loader, which accepts what `applet.save` writes unchanged.

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
- **The keys of a library are errors in a spec applet's manifest,** not ignored, so that a mistaken `kind` never makes a library's manifest a spec applet.
- **The loader reads the version line, not the TOML, when it lists a spec applet,** so listing never parses a spec; the full parse and validation run at launch and in `caliper check`.
- **No symbolic link for a spec,** wherever it points: a plain file name keeps the spec in its folder, and refusing every link is simpler than resolving one.
- **A workspace holds spec applets only,** because a workspace is data that anyone may hand to a person; compiled applets run only from folders a person installs into or names.
- **The workspace's folder comes after the development folders and before the user applets folder,** so a development build named on the command line wins a tie, and the open workspace's copy wins a tie with an installed one.

## Defects in the contracts

- Section 6 allows a table named `x` "at any level", but `x` is also a slot of `series` and `points`. The fix: `bind` has no extension table.
- Section 6 lets a split hold several children with one `ratio`, without saying what the ratio means for more than two. The fix: `ratio` is allowed only with exactly two children.
- Section 2 gives workspace applet ids an `_`, which a manifest id may not hold. The fix: `applet.save` maps `_` to `-` and refuses a clash.
- The manifest's `[compat].abi_epoch` was required, which a spec applet cannot meet. The fix, built: the loader does not require it for `kind = "spec"`, and refuses it there.
- A `sql` source runs in the browser by one of two routes: the query runs where the data lives and the result is streamed, which is the default of version 0, or it runs in the page with DuckDB-WASM, which is the later path. The first is built for the app's spec page, through `panel.data`; neither is built for a hub or a page that `caliper.serve()` serves.
