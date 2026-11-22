# Annotations

An annotation marks a value in a panel: a label with a leader line, a threshold line, a highlighted band or a marker on the current item. It is an item of its panel because it binds a value. A legend and a scale bar bind no value, and so they are panel options.

## Annotation Table

Each annotation is a table in the array `panel.annotation`. A panel holds at most 64 annotations. The commands `annotation.set` and `annotation.remove` edit one, and `panel.create` and `panel.patch` also take the array.

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
| `label` | `at`, `text`, `fields`, `leader` | A text at a point in data units | A text at a point, at an item `{ entity, index }` or at a pick |
| `threshold` | `axis`, `value`, `text` | A line across the plot | The outline of the plane on the data box, with a faint fill |
| `band` | `axis`, `range`, `text` | A shaded span | The outline of the slab on the data box |
| `marker` | `at`, `text`, `fields` | A ring at a point | A ring at an item or a pick, always on top |

Every kind also takes `id`, which is unique in the panel, and the optional key `emphasis`, which is `accent` or `muted`. An annotation uses the annotation color of the theme and never a data color. A threshold or a band in a scene takes the axis `x`, `y` or `z`, and in a 2-D panel `x` or `y`. A key that the kind does not take is `spec.unknown_option`, and a missing key is `spec.missing_key`. The values `value`, `range` and `at` may be [references](state-and-references.md#reference-places), and `text` is a [bound text](bound-text.md) with its `fields`.

Annotations draw in `series`, `histogram`, 2-D `points` and `scene` panels. Heatmaps and images refuse them with `spec.not_built`.

## Legends and Scale Bars

The option `legend` is `auto`, `true` or `false`, with `auto` as the default. A series panel takes all three values. Another panel takes `auto` only, and `true` or `false` there is `spec.not_built`. The option `scale_bar` works in scene panels, and outside a scene it is `spec.not_built`.

## Label Placement

One placement function without a window places the labels of annotations and the item labels of a scene. It takes rectangles and gives rectangles, thus the same input always gives the same placement. It places labels at a pick or a state first, then annotation labels in spec order, then item labels nearest first. For each label it tries eight positions around the anchor, then the same eight at three times the distance with a leader line. It takes the first position that overlaps no placed label and no marker and stays inside the panel. A label with no free position shows only its anchor dot, and a hover shows its text. The function runs again only when an anchor moves or the panel changes size. A panel places at most 512 labels.

The commands do not give the rectangles of the placed labels. `spec.get` shows that an annotation exists, and `panel.render` and `capture.frame` show where its label stands.

## Annotation Picking

A label or a marker takes a pick before the data under it. The pick adds a `pick` event with the key `annotation` and the fields of its anchor, and it does not change the selection of the panel. A threshold and a band take no picks.
