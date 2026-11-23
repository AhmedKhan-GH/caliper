# Scene Entities

A scene panel draws entities that Caliper builds from the sources of the spec. The script draws nothing for them. A spec thus builds a terrain from a live tensor with no scene call in the script. This page lists the keys of an entity, its draw kinds, the picks and the data box.

## Entity Table

Each entity is a table in the array `panel.entity` of a scene panel. A panel holds at most 64 entities. A scene panel with at least one entity needs no `scene` slot. When it also binds a live scene, it draws both in one view with one camera.

```toml
[[panel]]
id = "world"
primitive = "scene"
title = "Activation Terrain"
options = { up = "+z", box = [3.0, 1.0, 0.8], select = { state = "token", field = "column", entity = "terrain" } }

[[panel.entity]]
path = "terrain"
draw = "surface"
bind = "terrain"
options = { colormap = { state = "palette" } }

[[panel.entity]]
path = "loss"
draw = "lines"
title = "Loss"
bind = { x = "loss_window.step", z = "loss_window.value" }
options = { mode = "strip", y = "max", own = ["z"], width_px = 2.0 }
```

| Key | Type | Required | Meaning |
|---|---|---|---|
| `path` | 1 to 8 segments of `[a-z0-9_-]`, joined by `/` | Yes | The name of the entity. It is unique in the panel. |
| `draw` | `points`, `lines` or `surface` | Yes | The draw kind |
| `title` | Text | No | The name in the legend and the tooltip. The default is the path. |
| `bind` | Source id, or a table of slots | Yes | A source id binds each array of a geometry source to the slot of the same name. |
| `options` | Table | No | The options of the draw kind |
| `pose` | `{ position, rotation, scale }` | No | A translation, a quaternion `[x, y, z, w]` and a scale, all literals |

The entities stand under the root `spec/` of the scene of the panel, thus the entity `terrain` has the path `spec/terrain` there. The commands `entity.set` and `entity.remove` edit one entity. `panel.create` and `panel.patch` also take the array `entity`.

## Draw Kinds

| Draw | Slots | Options |
|---|---|---|
| `points` | `positions` (N, 3), or `x`, `y` and `z` (N) each. Also `values` (N), `colors` and `radii` (N). | `colormap`, `vmin`, `vmax`, `size_px`, `shape`, `x`, `y`, `z`, `own`, `depth_offset`, `highlight`, `visible` |
| `lines` | The slots of `points` without `radii` | The options of `points` without `size_px` and `shape`, plus `mode` (`strip` or `segments`) and `width_px` |
| `surface` | `height` (R, C), which is required. Also `values` (R, C), `x` (R) and `y` (C). | `colormap`, `vmin`, `vmax`, `depth_offset`, `highlight`, `visible` |

Positions need either `positions` or at least two of `x`, `y` and `z`. An axis without a slot takes the option of the same name. The option holds a number in data units, or `min` or `max` of the data box. Thus `y = "max"` puts the loss line on the back wall of the box. The option `own` lists the axes that map the data range of the entity onto the box.

A `surface` is a height field. Caliper builds its triangles on the GPU from the grid, and a cell with a NaN height is a hole. The color comes from `values`, else from `height`, through the colormap. A stage on a surface entity is not built, because stages take points and lines only.

The option `depth_offset` is an integer from -8 to 8. It moves the entity toward the camera by that many units. The default is 1 for `points` and `lines`, and 0 for `surface`. A line thus wins over the surface that it lies on.

## Array Resolution

Caliper resolves the arrays of an entity at the cursor of the [time cursor](time-cursor.md). It checks the shape of each array against its slot and converts float16, bfloat16, float64 and integer arrays to float32. It writes the entity again when its source has new data, when the cursor moves or when a reference that the chain reads changes. This happens at most once a frame.

A wrong shape or length does not stop the panel. The entity draws nothing, the panel shows "Wrong shape" with the path of the entity, and `panel.data` gives the code `entity.shape` or `entity.length`. An entity over a limit shows "Over limit". The limits are 1,048,576 items for a `points` or `lines` entity, 262,144 cells for a `surface` entity with at most 4,096 on one side, and 64 MiB of arrays in one panel.

## Pick Fields

A pick on an entity gives a set of fields. The `pick` event carries them in its key `fields`, a `selection` reference reads one of them, and the panel option `select` writes one into a [state variable](state-and-references.md).

| Field | Meaning | Draw kinds |
|---|---|---|
| `entity` | The path of the entity | All |
| `index` | The item, or the vertex nearest to the pick on a line | `points`, `lines` |
| `row_id` | The row id of the item, when its source has one | `points`, `lines` |
| `row`, `column` | The cell of the grid, as indices of the input before pooling | `surface` |
| `x`, `y`, `z` | The picked position in data units | All |
| `value` | The color value at the pick | All with values |
| `step` | The step of the row, when `x` came from a `window` | `surface`, `lines` |

## Data Box

A wide flat scene, such as a terrain of 2,048 steps by 16 rows, is a thin strip at its true proportions. A scene panel with entities thus scales its data box on each axis. The option `box` is three positive numbers, which are the proportions of the drawn box, or `auto`. With `auto`, no side of the box is more than four times another side.

The scale applies to the entities of the spec only. A live scene from the script keeps its own units. Picks report data units. With `axes = true`, the panel draws the three back walls of the box with ticks in data units and the titles `x_label`, `y_label` and `z_label`.

## Camera and Saved Views

A scene panel applies `view_camera` when it opens. The value is an orbit table, the path of a camera entity or the name of a saved view. The view keeps fitting the scene while its parts arrive, until the person moves the camera. A move stays for the session, and the reset of the view returns to `view_camera`. This holds for every scene view, also for the scene of a script.

The option `views` holds at most 16 saved views. Each view is `{ name, target, distance, azimuth, elevation }`, with the angles in degrees. A name is unique in the panel. The header of the panel has a menu of its views. The command `camera.save` writes the current camera into `views` as an edit, and `camera.set` moves the camera to a view or an orbit. [Instrument Tools](../agents/instrument-tools.md) lists the camera commands.

## Colormaps

Sixteen colormaps are available for panels and entities: `viridis`, `magma`, `rdbu`, `greys`, `cividis`, `inferno`, `plasma`, `turbo`, `twilight`, `coolwarm`, `batlow`, `vik`, `roma`, `oslo`, `lajolla` and `berlin`. A colormap option may read a text state whose choices are colormap names. In a panel option, the command schema accepts only the literal names.

## Label Placement

Spec scene panels place the labels of annotations and the item labels without overlap. A label takes the first free place of eight around its anchor, then of eight at three times the distance with a leader line. A label with no free place shows only its anchor dot, and a hover shows its text. Labels at a pick or a state come first, then annotation labels in spec order, then item labels nearest first. A panel places at most 512 labels. [Annotations](annotations.md) describes the kinds.
