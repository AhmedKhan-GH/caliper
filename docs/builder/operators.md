# Operators

An operator is a step of a source, like the steps `slice`, `reduce` and `normalise`. Its input is a tensor and its output is geometry: named arrays that the slots of an [entity](scene-entities.md) bind. An operator composes after the other steps and after a `window`. A step after an operator is `spec.bad_transform`, because geometry is not a tensor.

```toml
[[source]]
id = "terrain"
from = "act"
transform = [
  { op = "window", start = { control = "window", field = "lo" }, stop = { control = "window", field = "hi" } },
  { op = "surface", max_cells = 32768, pool = "mean" },
]
```

## Surface Operator

`surface` makes a height field from a tensor of rank 2 with shape (R, C). Its parameters are `max_cells`, from 1 to 262,144 with a default of 32,768, and `pool`, one of `mean`, `max` and `min` with a default of `mean`.

When R times C is above `max_cells`, the operator pools blocks of f by f cells. The factor f is the smallest integer for which the pooled grid fits. A pooled cell skips NaN values, and a block of NaN values gives NaN. The output arrays are these:

- `height` (R', C') holds the pooled values.
- `x` (R') holds the mean coordinate of the pooled rows. After a `window` it is the step, else the row index.
- `y` (C') holds the mean column index of the pooled columns.

When the catalog gives the shape, a tensor of another rank is `spec.operator_shape`. Without the shape, the panel shows "Wrong shape".

## Path Operator

`path` makes a polyline through the rows of a tensor with shape (R, D) and D at least 2. Its parameters are `columns`, two or three column indices with the first three as default, and `max_points`, from 2 to 1,048,576 with a default of 65,536. A column past D is `spec.operator_shape`.

When R is above `max_points`, the operator keeps every s-th row, with the smallest s that fits. The output arrays are `positions` (P, 3) and `values` (P). Two columns give a `z` of 0. The value of a row is its step after a `window`, else its row index. A `lines` entity with `mode = "strip"` draws the path, and a `points` entity draws its vertices.

## Operator Placement

A source chain runs in two places. An operator that is the first step of a source whose root is a live tensor runs in the script. Every other step runs in the app, where an operator uses its CPU reference.

In the script, the operator is a built-in program with the id `operator.<source>.<step index>`. It reads the live tensor on the device of the script, under the same sandbox, the same cadence and the same time watchdog as the program of an agent. Only its pooled result travels to the app. The app sends the program again when the shape of the input changes. The built-in program is never reviewed, because Caliper knows its hash.

`program.status` lists the operator with `kind` set to `operator` and the id of its source. The `program` events report it as they report any program. [GPU Programs and Stages](../agents/gpu-programs.md) describes status and events.

## Tensor Output

A program may declare the output kind `tensor`: `output = { kind = "tensor", shape = [16] }`, with 1 to 4 axes and at most 1,000,000 elements. The WGSL declares `var<storage, read_write> out: array<f32>`. After each pass, the script publishes the first elements of `out` as a snapshot of the live view `program.<id>`, with the stamp of the step. An operator uses this output for its result, and an agent can use it to write one row of a statistic at each pass into a source that keeps its steps.

## Program Parameters

A program and a stage take the key `params`: at most 16 names, each a number or a [reference](state-and-references.md#reference-places).

An instance of a piece takes the params of its manifest by name. This instance of the library piece `points@1` draws a cloud in the panel `world` and takes the upper end of its colormap from the control `level`:

```toml
[[program]]
id = "cloud"
piece = { name = "points", version = 1, from = "library", hash = "sha256:..." }
panel = "world"
bind = { positions = "live:cloud.positions", values = "live:cloud.values" }
params = { vmax = { control = "level" } }
```

A one-pass program declares `var<uniform> params: Params`, where `Params` is a struct with one `f32`, `i32` or `u32` field for each name. The viewer adds the same declaration for a stage, with `f32` fields in the order of the names. A struct whose fields differ from the names is `program.params_mismatch` or `stage.params_mismatch`. A new value needs no compile and no new review, because the hash covers the WGSL only. The script applies it at the next pass of that program, and the viewer applies it at the next frame.

A stage works on the entities of the draw kinds `points` and `lines`. A stage on a `surface` entity is not built.
