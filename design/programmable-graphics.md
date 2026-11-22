# Programmable Graphics

This page defines programmable graphics. An agent that uses only `caliper-mcp` writes the graphics of an instrument itself: buffers, compute passes, render passes, blending, depth and state that persists between frames, over live tensors of a training run. The unit of this work is the piece. A piece is one folder with a manifest and WGSL files, and it has two halves. Its derive half runs compute passes in the training script at `caliper.step()` and publishes named, typed resources. Its draw half runs compute and render passes in the viewer every frame, reads those resources, keeps buffers between frames and writes the host's color, ID and depth targets. A program of a spec is an instance of a piece, which binds the piece's inputs, sets its parameters and names its panel.

The page states the piece format, the program instances, the resources and their wire message, the prelude and the generated header, the render graph, the draw state, the sandbox of both halves, the safety of draws, picking, text, the camera, the commands, the library pieces, the pixel comparison between the app and the browser, the Attention Arc Tower and the second graphic, the acceptance test and the work items. It extends `design/gpu-programs.md`, which defines the one-pass programs, the sandbox, the cadence and the wire path, and `design/instrument-interface.md`, which defines state, params that follow controls, the step history and the cursor. For new work, pieces replace the output kinds and the stages of `design/gpu-programs.md`. Those stay unchanged, with the built-in primitives, until Graphics Consolidation moves every shape onto library pieces and removes them.

## Status

Not built. The interfaces are in the tree, and every function behind them answers `program.not_built`:

- `src/builder/include/caliper_builder/spec/piece.h`: the value types of pieces and their pins, which `Program` and `Spec` hold;
- `src/builder/include/caliper_builder/programs/pieces.h`: names and paths, imports, the hash, the composer and the checks, with every code of this page;
- `src/builder/programs/draw_prelude.wgsl` and the lib modules in `src/builder/programs/library/lib/`, which the build embeds into the app, the script's module and the browser module from the same files;
- `src/host/draw/draw_runtime.h`: the render graph of the draw half on the graphics layer, which the app and the browser module compile;
- `src/host/producer/caliper_producer_derive.h`: the producer runtime's C interface for the derive half;
- the capability `resources`, the sections and the limits of the wire in `include/caliper/live_net.h` and both codecs, with the type 0x0082 of RESOURCE reserved;
- the commands `piece.check`, `piece.list`, `piece.show`, `piece.save`, `piece.copy`, `piece.update`, `piece.remove` and `program.read`, with their schemas.

The spec codec refuses the key `piece` and the program keys `piece`, `bind` and `panel` with `spec.not_built`.

## Pieces

A piece is a reusable, self-describing module of graphics. Library pieces, workspace pieces and the agent's inventions are the same kind of thing, in one format. An instrument uses a piece through a program that pins it, and a piece can feed another piece through a published resource.

### Piece Files

A piece is a folder with the file `piece.toml`, which is its manifest, and up to three WGSL files: `common.wgsl`, which both halves include, `derive.wgsl` and `draw.wgsl`. The optional folder `sample/` holds one file `sample/<input>.bin` for each input that the manifest's `sample` names. No other file belongs to a piece, and the host reads no other name.

A workspace piece lives in `<workspace>/pieces/<name>/<version>/`. A name is `[a-z][a-z0-9_]{0,31}` and a version is an integer from 1 to 1,000,000, thus no path that the host builds can leave `pieces/`. The host refuses a link anywhere in the folder, and it opens each file only after its real path is inside `<workspace>/pieces/`, as the model runner checks a script's path. Each WGSL file holds at most 64 KiB, and the samples of a piece hold at most 1 MiB together. A breach is `piece.bad_path`.

An inline piece lives in one spec, in the top-level array `piece`, for a quick experiment. It has the same keys as `piece.toml`, and its table `files` holds the WGSL under the keys `common`, `derive` and `draw`. `piece.save` extracts it into the workspace.

### Piece Manifest

| Key | Value |
|---|---|
| `name` | The piece's name |
| `version` | The piece's version, 1 when absent; a saved version never changes |
| `description` | One line that says what the piece draws or publishes |
| `inputs` | At most 8 typed inputs, which a program binds |
| `params` | At most 16 typed parameters with defaults and ranges, which become controls |
| `resources` | At most 8 resources that the derive half writes; the published ones are the piece's outputs |
| `imports` | At most 8 lib modules by name and version, such as `colormap@1` |
| `depends` | At most 8 pieces, by name and version, whose outputs feed this piece's buffer inputs |
| `derive` | The derive half's passes and limits; absent for a piece that only draws |
| `draw` | The draw half's camera, buffers, targets, passes, item map, text and limits; absent for a piece that only publishes |
| `sample` | A small input for each named input, for the preview |
| `hash` | `sha256:` and 64 hex digits; the host computes it, and a given value must match |

### Input and Parameter Keys

An input of kind `tensor` binds a tensor of the training script, which the derive half reads read-only and in place. An input of kind `buffer` binds a published buffer resource of another program, which the draw half reads.

| Input Key | Value |
|---|---|
| `name` | The WGSL name of the generated array |
| `kind` | `tensor` or `buffer` |
| `dtype` | `tensor`: `f32` (default), `i32` or `u32`; the bound tensor must match it |
| `rank_min`, `rank_max` | `tensor`: the fewest and the most axes, 1 and 8 when absent |
| `element`, or `struct` with `fields` | `buffer`: the layout that the bound resource must have |
| `title` | The input's label in the builder |

A parameter is a field of the generated struct `Params`, which both halves read as `params.<name>`. A program gives it a number or a binding reference, as Program Parameters of `design/instrument-interface.md` defines. A parameter without a value in the program gets a range control from `min` to `max`, placed by the app, and without a range it keeps its default.

| Parameter Key | Value |
|---|---|
| `name` | The field's name |
| `type` | `f32` (default), `i32` or `u32` |
| `default` | The value when the program gives none, 0 when absent |
| `min`, `max` | The range of the control |
| `title` | The control's label |

### Resource Keys

| Key | Value |
|---|---|
| `id` | A WGSL name, unique among the piece's inputs, resources and draw buffers |
| `kind` | `buffer` or `texture` |
| `element`, or `struct` with `fields` | `buffer`: one plain type, or a struct whose fields are written `name: type` in order |
| `count` | `buffer`: the elements, at most 4,194,304 |
| `counted` | `buffer`: the derive half also gets the counter `<id>_count`, and the valid elements are the first min(counter, count) |
| `width`, `height` | `texture`: the size in pixels, at most 4,096 each |
| `publish` | `false` for a private resource of the derive half; `true` when absent |
| `clear` | `true` zeroes the resource before each run of the derive passes; `false` when absent |
| `keep` | The versions that the app's step history keeps, from 1 to 4,096, and 256 when absent |

A field type is `f32`, `i32`, `u32`, `vec2<f32>`, `vec4<f32>`, `vec2<u32>` or `vec4<u32>`. The composer lays out a struct by the storage rules of WGSL, and one element holds at most 256 bytes. One resource holds at most 64 MiB.

### Derive Keys

The table `derive` holds `passes`, from 1 to 8, and the limits `time_ms`, `memory_mb` and `every` of `design/gpu-programs.md`, for one run of all passes. A pass has `entry`, a compute entry point of `derive.wgsl` that is never `main`, `workgroups`, which is `[1, 1, 1]` when absent, and `writes`, the resources that the pass changes. The passes run in their order in one submit at a pass of the shared cadence, with a barrier between two passes. After a run, the runtime publishes each published resource that a pass of the run names in `writes`. A published resource that no pass names is static, and the runtime publishes it after the first run only.

### Draw Keys

| Key | Value |
|---|---|
| `camera` | `orbit` (default), `pan_zoom` or `none` |
| `buffers` | At most 4 buffers that the draw half keeps between frames: `id`, a layout, `count` and `reset` |
| `targets` | At most 4 offscreen targets: `id`, `scale` (1, 0.5 or 0.25) and `clear` |
| `passes` | 1 to 16 passes of one frame, in order |
| `pick` | The item map: `resource` and its `fields` that a pick event carries |
| `text` | The text table: `anchors`, `position`, `string`, `priority` and `strings` |
| `frame_ms` | The most GPU time of the draw half in one frame, 4 when absent and at most 16 |
| `memory_mb` | The budget of the resources, buffer inputs, draw buffers and targets in the viewer, 64 when absent and at most 256 |

### Pass Keys

| Key | Kind | Value |
|---|---|---|
| `kind` | Both | `compute` or `render` |
| `entry`, `workgroups`, `writes` | `compute` | The entry point, the workgroup counts, and the draw buffers that the pass changes |
| `vertex` | `render` | The vertex entry point; absent for a full-screen triangle |
| `fragment` | `render` | The fragment entry point |
| `target` | `render` | `host` (default) or an offscreen target |
| `reads` | Both | Offscreen targets that the pass samples |
| `instances` | `render` | A count, or the id of a buffer resource or buffer input whose valid count at the cursor is the count; 1 when absent |
| `vertices` | `render` | The vertices of one instance; 3 for a full-screen triangle, and necessary with `vertex` |
| `topology` | `render` | `triangle_list` (default), `triangle_strip`, `line_list`, `line_strip` or `point_list` |
| `blend` | `render` | `opaque`, `alpha` or `additive`; opaque into `host` and additive into a target when absent |
| `depth` | `render` | `none`, `test` or `write`; `write` with a vertex entry and `none` for a full-screen triangle when absent |
| `pick` | `render` | The fragment returns `CaliperPicked` and writes the ID at location 1 |

### Lib Modules

A lib module is a WGSL file of helpers that pieces import by name and version: `colormap@1`, `pick@1`, `camera@1` and `easing@1` now. The built-in set lives in `src/builder/programs/library/lib/<name>/<version>.wgsl`. A released version never changes, and a change makes the next version. A module may use the prelude's bindings and structs, and it declares bindings of its own never.

| Module | Items |
|---|---|
| `colormap@1` | `sample(map, t)`, `viridis(t)`, `magma(t)`, and the rows `VIRIDIS`, `MAGMA`, `RDBU` and `GREYS` of the host's colormaps |
| `pick@1` | `id(item)`, the ID of a pick pass; `selected(item)` and `selected_index(index)`, the linked selection |
| `camera@1` | `clip(world)`, `pixel_direction(a, b)` and `offset_pixels(clip, pixels)` |
| `easing@1` | `smooth(t)` and `out_cubic(t)` |

### Imports

A piece calls an item of an imported module as `<module>::<item>`, for example `colormap::magma(t)`. The host resolves the imports before the compiler runs, and the sandbox checks the resolved text. The resolution checks that each `<module>::<item>` names an imported module and an item of it, else it gives `piece.unknown_import`. It then rewrites each `::` to `__`, and it puts the modules' text, in the order of `imports`, before `common.wgsl`. A module declares its items with the prefix `<module>__` already, and a piece's own files may not declare a name with `__`, thus no name of a module collides with a name of the piece.

The rewrite keeps the length of each line, thus every line and column of the author's files stays where it was. The composed module records its segments: the prelude, the generated header, each module, `common.wgsl` and the half's file, each with its first line. A compiler message on line 212 of the composed text thus maps back to its file and to the line in that file, with the same column, and the messages of `piece.check` and of the status name the file.

### Piece Hash

The hash identifies what runs. It is the SHA-256 of these bytes, in lowercase hex after `sha256:`:

1. the line `caliper-piece/1`;
2. the manifest as RFC 8785 canonical JSON, without `hash` and without the files: keys sorted, no white space, numbers in their shortest form;
3. for each imported module in the order of `imports`, and then for each of `common.wgsl`, `derive.wgsl` and `draw.wgsl` that the piece has, a line `--- <part> <bytes>` and the text with LF line ends.

The hash covers the modules' text because they run, and a released module never changes, thus the hash is stable across releases. It does not cover the samples. The canonical JSON and the LF line ends make it the same on every platform and in every language that computes it: the editor, the script and the browser module.

### Piece Sources

A program names the source of its piece with `from`:

- **`library`**, the default: the built-in library, which ships read-only with Caliper. This work fills it with the lib modules and `points@1`. Graphics Consolidation adds a piece for every shape.
- **`workspace`**: `<workspace>/pieces/`, which the person and the agent write through the piece commands.
- **`inline`**: the array `piece` of the open spec.

A workspace piece with the name of a library piece replaces it only in a program that says `from = "workspace"`. A piece from a foreign spec, inline or pinned, runs only after a review of its hash, as Provenance of `design/gpu-programs.md` states. Pieces that the person or the person's agent made run at once.

### Versions and Pins

A program pins its piece by name, version and hash. `piece.update` writes the next version of a workspace piece and never changes a saved one, thus a program keeps its version until `program.patch` pins another. When the pinned piece is missing from its source, the program's state is `refused` with `piece.missing`. When the piece's hash differs from the pin, the state is `refused` with `piece.hash_mismatch`. In both cases the spec opens, the panel's info line names the piece and the reason, and the app never pins another version on its own. The person or the agent restores the folder, or pins an existing version with `program.patch`. An inline piece without `hash` gets it from the editor, and an inline piece with a wrong one is `piece.hash_mismatch` too.

### Program Instances

A program of a spec is an instance of a piece. It holds no WGSL and none of the one-pass keys `wgsl`, `entry`, `inputs`, `output`, `limits` and `workgroups`; a mix is `program.form_mixed`.

| Key | Value |
|---|---|
| `id` | The program's id, as in `design/gpu-programs.md` |
| `title` | An optional label |
| `piece` | The pin: `name`, `version`, `hash` and `from` |
| `bind` | Each input of the piece: `live:<tensor>` for a tensor input, `program:<id>/<resource>` for a buffer input |
| `params` | A number or a binding reference for each parameter that does not keep its default or get a control |
| `panel` | The scene panel that the draw half writes; necessary when the piece draws |

The editor checks a binding against the catalog: a tensor's dtype and rank, or the layout of the other program's published resource (`piece.bad_binding`). `program.create` takes these keys, and with `inline`, a whole piece that it adds to the array `piece` in the same undo step.

### Piece Graph

A buffer input that binds `program:<id>/<resource>` makes the programs a graph, the graph that the node editor will show. In this work only the draw half reads a buffer input. The viewer gives it the version of the other program's resource at the cursor, as it gives the piece's own resources. A derive half that reads another derive half's output needs an order of the passes in the script, and it comes with the derivation catalog. A program whose binding names a resource that its piece does not publish is `piece.bad_binding`.

### Field Example

The piece `attention_field` draws the attention pattern of one head as a smooth field with contour lines, with a fragment-only draw. Its derive half copies the first batch row of two attention tensors into one published buffer, and its draw half samples that buffer by hand with bilinear weights. The piece has no item map. This is its `piece.toml`:

```toml
name = "attention_field"
version = 1
description = "The attention pattern of one head of a two-layer model, as a field with contour lines"
imports = ["colormap@1"]

[[inputs]]
name = "attn0"
kind = "tensor"
rank_min = 4
rank_max = 4

[[inputs]]
name = "attn1"
kind = "tensor"
rank_min = 4
rank_max = 4

[[params]]
name = "layer"
type = "i32"
min = 0
max = 1

[[params]]
name = "head"
type = "i32"
min = 0
max = 3

[[params]]
name = "gain"
default = 1.0
min = 0.1
max = 10.0

[[resources]]
id = "pattern"
kind = "buffer"
element = "f32"
count = 8192      # 2 layers, 4 heads, 32 by 32 tokens

[derive]
passes = [{ entry = "copy_rows", workgroups = [16, 1, 1], writes = ["pattern"] }]

[draw]
camera = "none"
passes = [{ kind = "render", fragment = "field" }]
```

`common.wgsl` holds the sizes, `derive.wgsl` the copy and `draw.wgsl` the field:

```wgsl
// common.wgsl
const H: u32 = 4u;    // heads
const T: u32 = 32u;   // tokens of the fixed sentence
```

```wgsl
// derive.wgsl: batch row 0 of each (B, H, T, T) tensor is its first H * T * T values.
@compute @workgroup_size(256)
fn copy_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = H * T * T;
    if (id.x >= n) {
        return;
    }
    pattern[id.x] = attn0[id.x];
    pattern[n + id.x] = attn1[id.x];
}
```

```wgsl
// draw.wgsl: rows are the query token, columns the key token.
fn value(layer: u32, head: u32, i: u32, j: u32) -> f32 {
    return pattern[((layer * H + head) * T + i) * T + j];
}

@fragment
fn field(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    let layer = u32(clamp(params.layer, 0, 1));
    let head = u32(clamp(params.head, 0, i32(H) - 1));
    let p = in.uv * f32(T) - 0.5;
    let a = vec2<u32>(clamp(floor(p), vec2<f32>(0.0), vec2<f32>(f32(T - 1u))));
    let b = min(a + 1u, vec2<u32>(T - 1u));
    let f = clamp(p - floor(p), vec2<f32>(0.0), vec2<f32>(1.0));
    let v = mix(mix(value(layer, head, a.y, a.x), value(layer, head, a.y, b.x), f.x),
                mix(value(layer, head, b.y, a.x), value(layer, head, b.y, b.x), f.x), f.y);
    let t = clamp(v * params.gain, 0.0, 1.0);
    var color = colormap::magma(t);
    if (abs(fract(t * 8.0) - 0.5) < 0.04) {
        color = mix(color, vec4<f32>(1.0), 0.35);
    }
    return color;
}
```

A program places it in the panel `pattern` and follows the head that a pick selects, through two state variables:

```toml
[[program]]
id = "head_pattern"
piece = { name = "attention_field", version = 1, from = "workspace", hash = "sha256:…" }
panel = "pattern"
bind = { attn0 = "live:blocks.0.attn.weights.act", attn1 = "live:blocks.1.attn.weights.act" }
params = { layer = { state = "layer" }, head = { state = "head" }, gain = 2.0 }
```

### Arc Example

The piece `attention_arcs` is the standard path: compute into a storage buffer, then an instanced draw. Its derive half scores each head for induction, then appends one arc for each attention weight above a threshold, then places one label anchor for each token. Its draw half draws the arcs as instanced triangle strips with additive blending and an ID, draws them again into a half-size offscreen target, and adds a blur of that target as a glow.

```toml
name = "attention_arcs"
version = 1
description = "Arcs between the tokens of one sentence, one ring for each layer, from the attention weights"
imports = ["colormap@1", "pick@1", "camera@1"]

[[inputs]]
name = "attn0"
kind = "tensor"
rank_min = 4
rank_max = 4

[[inputs]]
name = "attn1"
kind = "tensor"
rank_min = 4
rank_max = 4

[[params]]
name = "threshold"
default = 0.05
min = 0.0
max = 0.5
title = "Weight Threshold"

[[params]]
name = "glow"
default = 1.0
min = 0.0
max = 4.0

[[resources]]
id = "heads"
kind = "buffer"
element = "f32"
count = 8

[[resources]]
id = "arcs"
kind = "buffer"
struct = "Arc"
fields = ["from_token: u32", "to_token: u32", "layer: u32", "head: u32", "weight: f32", "score: f32"]
count = 4096
counted = true

[[resources]]
id = "labels"
kind = "buffer"
struct = "Label"
fields = ["position: vec4<f32>", "string: u32"]
count = 32

[derive]

[[derive.passes]]
entry = "scores"
writes = ["heads"]

[[derive.passes]]
entry = "build"
workgroups = [128, 1, 1]
writes = ["arcs"]

[[derive.passes]]
entry = "place_labels"

[draw]
camera = "orbit"

[[draw.targets]]
id = "glow"
scale = 0.5

[[draw.passes]]
kind = "render"
vertex = "arc_vs"
fragment = "arc_fs"
instances = "arcs"
vertices = 64
topology = "triangle_strip"
blend = "additive"
depth = "none"
pick = true

[[draw.passes]]
kind = "render"
vertex = "arc_vs"
fragment = "arc_glow"
instances = "arcs"
vertices = 64
topology = "triangle_strip"
target = "glow"

[[draw.passes]]
kind = "render"
fragment = "bloom"
reads = ["glow"]
blend = "additive"

[draw.pick]
resource = "arcs"
fields = ["layer", "head", "from_token", "to_token", "weight", "score"]

[draw.text]
anchors = "labels"
strings = [
  "the", "old", "map", "shows", "red", "fox", "jumps", "over", "lazy", "dogs", "near", "green",
  "hills", "under", "blue", "skies", "red", "fox", "jumps", "over", "lazy", "dogs", "near", "green",
  "hills", "under", "blue", "skies", "and", "then", "it", "ends",
]
```

`place_labels` names no resource in `writes`, thus `labels` is static and travels once. The tokens 4 to 15 repeat at 16 to 27, so that an induction head attends from token t in the second copy to token t − 11, the token after the previous occurrence.

```wgsl
// common.wgsl
const L: u32 = 2u;     // layers
const H: u32 = 4u;     // heads
const T: u32 = 32u;    // tokens of the fixed sentence
const P: u32 = 12u;    // length of the repeated segment
const S: u32 = 4u;     // first token of its first copy

fn ring_point(layer: u32, token: u32) -> vec3<f32> {
    let a = 6.2831853 * f32(token) / f32(T);
    return vec3<f32>(cos(a), f32(layer) * 0.6, sin(a));
}
```

```wgsl
// derive.wgsl
fn attn(layer: u32, head: u32, i: u32, j: u32) -> f32 {
    let k = (head * T + i) * T + j;     // batch row 0
    if (layer == 0u) {
        return attn0[k];
    }
    return attn1[k];
}

// The induction score of each head: its mean weight from a token of the
// second copy to the token after the same token's first occurrence.
@compute @workgroup_size(8)
fn scores(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= L * H) {
        return;
    }
    var sum = 0.0;
    for (var t = S + P; t < S + 2u * P; t = t + 1u) {
        sum = sum + attn(id.x / H, id.x % H, t, t - P + 1u);
    }
    heads[id.x] = sum / f32(P);
}

@compute @workgroup_size(64)
fn build(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= L * H * T * T) {
        return;
    }
    let j = id.x % T;
    let i = (id.x / T) % T;
    let head = (id.x / (T * T)) % H;
    let layer = id.x / (T * T * H);
    if (j >= i) {
        return;                            // causal, and no arc to itself
    }
    let w = attn(layer, head, i, j);
    if (w < params.threshold) {
        return;
    }
    let slot = atomicAdd(&arcs_count, 1u);
    if (slot >= arrayLength(&arcs)) {
        return;
    }
    arcs[slot] = Arc(j, i, layer, head, w, heads[layer * H + head]);
}

@compute @workgroup_size(32)
fn place_labels(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= T) {
        return;
    }
    labels[id.x] = Label(vec4<f32>(ring_point(0u, id.x) * 1.15, 1.0), id.x);
}
```

```wgsl
// draw.wgsl
const SEG: u32 = 32u;    // segments of one arc: 64 vertices of a strip

struct ArcOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) item: u32,
}

fn arc_point(arc: Arc, t: f32) -> vec3<f32> {
    let p0 = ring_point(arc.layer, arc.from_token);
    let p2 = ring_point(arc.layer, arc.to_token);
    let p1 = 0.5 * (p0 + p2) + vec3<f32>(0.0, 0.1 + 0.5 * arc.weight, 0.0);
    let u = 1.0 - t;
    return u * u * p0 + 2.0 * u * t * p1 + t * t * p2;
}

@vertex
fn arc_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> ArcOut {
    let arc = arcs[i];
    let t = f32(v / 2u) / f32(SEG - 1u);
    let e = 0.5 / f32(SEG - 1u);
    let c = camera::clip(arc_point(arc, t));
    let a = camera::clip(arc_point(arc, max(t - e, 0.0)));
    let b = camera::clip(arc_point(arc, min(t + e, 1.0)));
    let d = camera::pixel_direction(a, b);
    let side = f32(v % 2u) - 0.5;
    let width = (1.0 + 5.0 * arc.weight) * caliper_view.time.z;
    var out: ArcOut;
    out.position = camera::offset_pixels(c, vec2<f32>(-d.y, d.x) * side * width);
    let rgb = colormap::viridis(f32(arc.head) / f32(H - 1u)).rgb * (0.3 + 0.7 * arc.weight);
    out.color = vec4<f32>(rgb * arc.weight, arc.weight);
    out.item = i;
    if (pick::selected(i)) {
        out.color = vec4<f32>(1.0, 0.9, 0.6, 1.0);
    }
    return out;
}

@fragment
fn arc_fs(in: ArcOut) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = in.color;
    out.id = pick::id(in.item);
    return out;
}

@fragment
fn arc_glow(in: ArcOut) -> @location(0) vec4<f32> {
    return in.color;
}

@fragment
fn bloom(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(glow));
    var sum = vec4<f32>(0.0);
    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            let w = exp(-f32(x * x + y * y) / 4.0);
            let uv = in.uv + vec2<f32>(f32(x), f32(y)) * texel * 2.0;
            sum = sum + w * textureSampleLevel(glow, caliper_linear, uv, 0.0);
        }
    }
    return sum * 0.12 * params.glow;
}
```

The program of the tower binds the same two tensors and writes the picked head into the state that `head_pattern` reads:

```toml
[[program]]
id = "tower_arcs"
piece = { name = "attention_arcs", version = 1, from = "workspace", hash = "sha256:…" }
panel = "tower"
bind = { attn0 = "live:blocks.0.attn.weights.act", attn1 = "live:blocks.1.attn.weights.act" }
params = { threshold = 0.05 }

[[panel]]
id = "tower"
primitive = "scene"
options = { select = { state = "head", field = "head" } }
```

## Resources

A resource is a named, typed array that the derive half writes and the viewer reads. Resources replace the output shapes of `design/gpu-programs.md` for new work: a piece states what it publishes, and the draw half decides how it looks.

### Resource Kinds

A `buffer` is `count` elements of a plain type or a generated struct. A `texture` is an `rgba8unorm` image that the derive half writes as a packed `array<u32>`, `width` times `height` texels in row order, and that the viewer copies into a texture for the draw half to sample. The derive half never writes a texture directly, because storage textures are cut from this work. A counted buffer's valid elements are the first min(counter, count), and a viewer receives only those.

### Allocation and Lifetime

The producer runtime allocates each resource of a program at its install, inside `derive.memory_mb`, and keeps it until the program is removed. Its contents stay from one run to the next unless it has `clear`, thus the derive half can keep state, such as a running mean. A counter returns to zero before each run. A private resource never leaves the script.

The viewer allocates its copy of a resource on its first version and grows it when a larger version arrives, inside `draw.memory_mb`. It allocates draw buffers and targets at the install and targets again at a resize of the panel. It releases all of them when the program is removed or the panel closes.

### Resource Message

RESOURCE (0x0082, reserved) carries one version of one published resource from the script to a viewer or a hub that announced the capability `resources`. Its metadata holds `program`, `resource`, `step` (the step of the run), `count` (valid elements), `stride` (bytes of one element, 4 for a texture), and for a texture `width` and `height`. The section `resource` holds the valid bytes, at most 64 MiB (`CALIPER_NET_RESOURCE_MAX_BYTES`). One message carries one resource, thus a large resource does not hold a small one back, and the viewer keeps each in its own ring. A viewer that falls behind gets the newest version of each resource and loses the older ones, as with snapshots.

PROGRAM of a piece's instance carries the metadata `request_id`, `op`, `id`, `piece` (name, version and hash), `manifest` (the manifest as JSON), `bind`, `params`, `panel` and `segments`, and the sections `common`, `derive` and `draw` with the imports resolved: `common` holds the lib modules and `common.wgsl`, and `segments` gives their lines. The script runs the derive half. It also sends PROGRAM, without the section `derive`, to each viewer that announced `resources`, when the viewer attaches and after each install, so that a browser viewer of the run gets each draw half from the same place as its resources. A hub forwards RESOURCE as it forwards snapshots, and PROGRAM as it forwards CONTROL.

### Kept Versions

The app keeps the versions of each published resource in the step history of the builder session, under `<program>/<resource>`, beside the kept steps of the sources. A resource keeps at most `keep` versions, 256 when absent. When its ring is full, the history drops every second kept version and from then on keeps a version only when its step is at least twice the former stride after the newest one. Thus the kept versions always span the whole run, evenly, and the step cursor scrubs the whole training. The versions share the 512 MiB of kept steps, and one resource keeps at most 128 MiB, as Step History of `design/instrument-interface.md` states.

The viewer gives the draw half the newest kept version at or before the cursor. A version at an earlier step than the newest one of the same attachment starts a new epoch, as for snapshots. The browser module keeps no history in this work and draws the newest version.

### Reopened Instrument

The spec saves the pins, the bindings, the params and the inline pieces, never a resource. When an instrument opens, the app resolves each pin, checks each piece and sends each program to the attached script, which publishes again from its next pass. The kept versions start empty, thus the cursor scrubs only the steps since the open. Without an attached run, the program's state is `waiting`, and the panel's info line reads "Waiting for the run's resources".

## Prelude

The draw prelude is one fixed WGSL text, the same in the app and in the browser module, which the host puts first in every pass of a draw half. It declares group 0 with five bindings: the frame's view, the linked selection, two samplers and the host's colormaps. The helpers that read them are the lib modules.

### Prelude Text

The file `src/builder/programs/draw_prelude.wgsl` is the prelude. The build embeds it into the builder's programs library and the browser module from that one file.

```wgsl
--8<-- "src/builder/programs/draw_prelude.wgsl"
```

### Generated Header

The composer writes the header of each module from the manifest, in a fixed order, so that the app, the script and the browser bind the same names at the same places without reflection. A piece's files declare no binding: `@group` or `@binding` in a file is `program.binding_not_allowed`.

| Half | Group | Bindings |
|---|---|---|
| Derive | 0 | 0 `caliper: CaliperFrame` (the frame uniform of `design/gpu-programs.md`), 1 `params: Params`, 2 `shapes: CaliperShapes` |
| Derive | 1 | One `var<storage, read>` array for each tensor input, at the input's index |
| Derive | 2 | Each resource at 2 × its index, `read_write`, and its counter `<id>_count: atomic<u32>` at 2 × its index + 1 |
| Draw | 0 | The prelude's 0 to 4, then 5 `params: Params` |
| Draw | 1 | 0 `counts: CaliperCounts`, then each published resource and each buffer input, read-only, from 1 in the manifest's order |
| Draw | 2 | Each draw buffer at its index, `read_write` only in a compute pass that writes it; each offscreen target that the pass reads at 4 + its index |

`CaliperShapes` holds one `vec4<u32>` for each tensor input, named as the input: the last four axes of its shape, padded with 1 at the front. `CaliperCounts` holds one `u32` for each published resource and buffer input: its valid count in the version of the frame. `Params` holds the parameters in the manifest's order. The composer generates the struct of each layout in both halves. For pass 0 of `attention_arcs`, the draw header is this:

```wgsl
struct Params { threshold: f32, glow: f32 }
@group(0) @binding(5) var<uniform> params: Params;
struct CaliperCounts { heads: u32, arcs: u32, labels: u32 }
struct Arc { from_token: u32, to_token: u32, layer: u32, head: u32, weight: f32, score: f32 }
struct Label { position: vec4<f32>, string: u32 }
@group(1) @binding(0) var<uniform> counts: CaliperCounts;
@group(1) @binding(1) var<storage, read> heads: array<f32>;
@group(1) @binding(2) var<storage, read> arcs: array<Arc>;
@group(1) @binding(3) var<storage, read> labels: array<Label>;
```

The bloom pass adds `@group(2) @binding(4) var glow: texture_2d<f32>;`, because it reads `glow`, and the two passes that write `glow` do not declare it.

### Params

`Params` comes from the manifest, and a program gives each value as a number or a binding reference to a control, a selection or a state, as Program Parameters of `design/instrument-interface.md` defines. A piece reads no state variable by name, thus a piece stays independent of the spec that uses it. The app evaluates the references, packs `Params`, and gives the bytes to the viewer each frame and to the script at its next pass. A change of a value needs no compile.

### Linked Selection

A pick in a scene panel with the option `select` writes a state variable from a field of the pick, as State Writers of `design/instrument-interface.md` states, and the field names of a piece's pick are those of its item map. The host also gives every draw half of the spec the selection as `caliper_selection`: up to 256 entries, each with the program's pick id, the item and the first four integer fields of the item map as tensor indices. `pick::selected(item)` and `pick::selected_index(index)` read it, thus one program can highlight the head that a pick in another program selected.

## Render Graph

The render graph of a draw half is the list of its passes. The host allocates the resources, the buffers and the targets, records the passes in their order, and owns the camera, the text and the pick readback.

### Pass Order

The passes of one frame run in the declared order, after the scene's own draws and before its selection highlight. A compute pass dispatches over the draw buffers. A render pass with `vertex` draws `instances` instances of `vertices` vertices; without `vertex`, it draws the prelude's `caliper_fullscreen` with 3 vertices, as Shadertoy does. A pass whose `instances` names a resource with no version yet, or with a valid count of zero, is skipped.

### Targets

The host target is the scene panel's color (`rgba8unorm`), ID (`rg32uint`) and depth (`depth32float`), which the panel already renders offscreen. An offscreen target is `rgba8unorm` at `scale` of the panel's size, cleared to `clear` at the start of each frame. A pass may sample a target that an earlier pass of the frame wrote, and never its own target, else `program.target_cycle`. No pass reads the host target.

### Pass Values

| Value | Allowed | Default |
|---|---|---|
| `topology` | `triangle_list`, `triangle_strip`, `line_list`, `line_strip`, `point_list` | `triangle_list` |
| `blend` | `opaque`, `alpha` (premultiplied), `additive` | `opaque` into `host`, `additive` into a target |
| `depth` | `none`, `test` (less-or-equal, no write), `write` (less-or-equal and write) | `write` with a vertex entry, `none` for a full-screen triangle; always `none` into a target |
| Culling | None | None |

The ID never blends. A pass with `pick` writes it at location 1, and in a pass with depth `none` the last fragment at a pixel wins.

### Graph Cuts

The graph has no custom blend factors, no target wider than eight bits a channel, no indirect draw or dispatch, no storage texture, no multisampling and no node materials. A draw half's compute pass never changes an instance count, because the count comes from the CPU copy of a resource.

## Draw State

A draw buffer keeps its contents from frame to frame, thus motion comes from the draw half and not from the derive half, which runs only at its share of step time. `caliper_view.frame.x` counts the frames since the last reset, and `caliper_view.frame.y` is 1 on the first frame after a reset. The host zeroes every draw buffer at an install, at a new attachment and at a new epoch. A buffer with `reset = "cursor"` also returns to zero when the cursor moves by a command, a key or the transport, and one with `reset = "resource"` also when a resource of the frame changes version. A change of a parameter never resets a buffer.

## Sandbox

The sandbox checks each piece on its own, before any program runs it, in the editor, in the script and in the viewer. The app's checks with the compiler stand in front of every other host.

### Check Order

1. **Name and files.** The name, the version and each file pass `check_piece_name()` and `check_piece_file()`.
2. **Manifest.** `check_piece_manifest()`: kinds, layouts, counts, sizes, limits, the passes' entries and values, the order of the targets, the item map, the text table, the params' ranges and the limits of WebGPU (`check_web_limits()`).
3. **Provenance.** A piece from a foreign spec has its hash in `provenance.reviewed`.
4. **Files.** `check_piece_source()` on each file: its size, no directive, no `@group` or `@binding`, and no declaration of a `caliper_`, `Caliper` or `CALIPER_` name or of a name with `__`.
5. **Imports and hash.** `resolve_imports()` and `piece_hash()`; a pinned hash must match.
6. **Compose and compile.** `compose_derive()` and `compose_draw()` for each pass, then caliper-wgsl with the bounds checks of `design/gpu-programs.md`, in the app and in the script.
7. **Entry points and interfaces.** `check_piece_reflection()`: each entry point with its stage, at most 256 invocations in a workgroup and at most 256, 256 and 64 on its axes, the vertex outputs against the fragment inputs, a pick pass's fragment returning `CaliperPicked`, a target pass's fragment returning one `vec4<f32>`, at most 16 inter-stage variables, and no entry named `main`.
8. **Host.** The read-only imports of the derive half, both memory budgets, and the frame validation of the graphics layer.

### Check Codes

The codes are constants of `programs/pieces.h`, beside those of `programs/sandbox.h`.

| Code | Cause |
|---|---|
| `piece.bad_name` | A name or a version outside its form |
| `piece.bad_manifest` | A key, a count or a value of the manifest |
| `piece.bad_path` | A file outside its folder, a link, or a file name that a piece cannot have |
| `piece.unknown_import` | No such lib module, or `m::item` names nothing |
| `piece.not_found` | A command names a piece that its source lacks |
| `piece.missing` | A program pins a piece that its source lacks |
| `piece.hash_mismatch` | A pinned or given hash differs from the piece's |
| `piece.exists` | A workspace version of that name exists |
| `piece.read_only` | A command would change a library piece |
| `piece.in_use` | A program of the open spec pins that version |
| `piece.bad_binding` | An input unbound, or bound to another dtype, rank or layout |
| `piece.cycle` | Buffer inputs form a cycle among derive halves |
| `program.form_mixed` | A program has keys of the one-pass form and a pin |
| `program.bad_resource` | A resource's kind, count or size, or a duplicate id |
| `program.bad_layout` | A field's name or type, or a stride over 256 bytes |
| `program.unknown_resource` | A pass, a pick or a text names no resource |
| `program.bad_pass` | A pass's kind, entry, target or values |
| `program.target_cycle` | A pass reads its own target, or a target before a pass writes it |
| `program.bad_pick` | The item map names no field, or the piece has no pick pass |
| `program.bad_text` | The anchors lack their fields, or the strings their limits |
| `program.camera_conflict` | Two programs of a panel select two cameras |
| `program.binding_not_allowed` | A file declares a binding |
| `program.reserved_name` | A file declares a host name or a name with `__` |
| `program.entry_stage` | An entry point of the wrong stage, or none |
| `program.interface_mismatch` | Stage outputs and inputs disagree, or a fragment's outputs and its target |
| `program.over_web_limits` | A pass over a default limit of WebGPU |

### Browser Checks

The browser module has no compiler. It runs checks 1, 2, 4 and 5 with the same code, compiled from `pieces.cpp`, and it computes the hash of what PROGRAM brought: a hash that differs from the pin refuses the program. It then gives each composed module to WebGPU, and a compilation message or a refused pipeline refuses the program with `program.browser_refused`, with the line mapped to its file. WebGPU's robust buffer access keeps a draw inside its bindings, thus the browser needs no bounds checks of its own.

The app's check stands in front of a browser viewer in two ways. A browser viewer gets a draw half only from the script, which got it from the app after checks 1 to 7 with the compiler, or from a spec that the app saved. The hash then proves that the browser runs what the app checked. And check 2 holds every pass to the default limits of WebGPU, thus a piece that runs in the app also runs in a browser: a pass binds at most 8 storage buffers in one stage, so that the published resources, the buffer inputs and the draw buffers of a piece number at most 7 beside the selection.

## Safety of Draws

A draw half writes only offscreen targets: the panel's own color, ID and depth, and its own targets. The host records each program's passes in a submit of its own after the scene's frame, and the layer's GPU time of that submit is the program's frame time. Without timestamps, as in a browser without the timestamp query, the host takes the time from the submit until the work is done, which is an upper bound.

The host cannot stop a pass that runs on the GPU, thus it removes a draw that goes over its budget and reports it:

- A program over `frame_ms` on 30 frames in a row stops with `program.draw_over_time`.
- A frame of a program that is not done after 1 s stops the program at once with `program.draw_timeout`. The GPU runs that frame until the driver ends it.
- The programs of one panel together over 16.7 ms on 30 frames in a row stop the costliest one with `program.draw_over_time`.
- A resource, a draw buffer or a target that would pass `draw.memory_mb` stops the program with `program.draw_over_memory`.

The report names the object. The status item's `error.where` names the pass, such as `draw.passes[1]`, the event `program` carries the program, the pass's index and its entry point, and the panel's info line reads, for example, "Stopped tower_arcs: draw pass 1 (arc_glow) took 21.3 ms a frame, above its 4 ms". The other programs of the panel keep drawing. The derive half keeps the watchdog, the timeout and the budget of `design/gpu-programs.md`, for one run of all its passes.

## Picking

A pass with `pick` writes `pick::id(item)` at location 1 of the host target, in every frame: the program's pick id, which the scene view gives out above the ids of its entities, and the item. The scene view reads the 16 by 16 window around the pointer from the newest completed frame's ID and depth, as `pick.h` reads its own pass, and takes the texel nearest the center. A hit of a built-in entity's pick pass wins when its depth is nearer.

The item is an element index of the item map's resource. The host reads that element from the CPU copy of the version that the frame showed and gives the pick event, of the kind `pick`, these keys beside the common ones:

| Key | Value |
|---|---|
| `program` | The program's id |
| `item` | The element index |
| `step` | The step of the version |
| `fields` | The item map's fields of the element, by name, as numbers |
| `index` | The first four integer fields as tensor indices, −1 where there are fewer |

A pick of an arc of `tower_arcs` thus gives `layer`, `head`, `from_token`, `to_token`, `weight` and `score`, and its `index` is the layer, the head and the token pair.

## Text

The host owns text. A piece publishes the anchors of its labels in a buffer resource, and its text table names the anchors, the fields `position` and `string`, an optional `priority` and at most 4,096 strings of at most 64 bytes. The scene view reads the anchors of the version it shows, projects them with the camera, and places at most 512 labels with `place_labels()` of `spec_runtime/label_layout.h`, with the overlap rule and the leader lines of the panel's other labels. A label of a higher priority wins an overlap. The anchors come from a derive half only, because only the derive half's output reaches the CPU.

## Camera

The host owns the camera. A piece selects `orbit` for the panel's 3-D orbit camera, `pan_zoom` for its 2-D camera, or `none`, which gives identity matrices and takes no camera input, for a piece that draws in clip space. The programs of one panel must agree, else `program.camera_conflict`. The panel's saved views, its starting camera and `camera.get`, `camera.set` and `camera.save` work as for entities.

## Commands

Every command is an MCP tool with the dot replaced by `_`, and Caliper API and the builder pane reach it through the command layer. The descriptors with their schemas are in `src/builder/commands/command_table.cpp`.

### Piece Commands

| Command | Effect | Function |
|---|---|---|
| `piece.check` | `read` | Checks a piece on its own and returns the messages by file, the hash and the generated headers |
| `piece.list` | `read` | Lists the library, workspace and inline pieces with their inputs, params, outputs, hash and pins |
| `piece.show` | `read` | Returns one piece with its manifest, files, source, folder and pins |
| `piece.save` | `change` | Extracts an inline piece into the workspace and pins its programs to it, in one undo step of the spec |
| `piece.copy` | `change` | Forks a library piece into the workspace, as version 1 |
| `piece.update` | `change` | Writes the next version of a workspace piece, or version 1 of a new name, after the checks |
| `piece.remove` | `change` | Removes one workspace version that no program of the open spec pins |

`piece.remove` is destructive and asks for the person's card from an agent. A piece can carry a sample input. `piece.show` with a sample gives a preview: the app runs both halves on the sample in its own process and captures 256 by 256 pixels.

### Program Commands

`program.create` and `program.patch` take the keys of Program Instances, and `program.create` also takes `inline`, a whole piece that joins the array `piece`. `program.remove` is unchanged. `program.status` gives a program of a piece the kind `piece`, the pin, and a `draw` object with `state`, `frame_ms`, `frame_limit_ms`, `memory_bytes`, `memory_limit_bytes`, `resource_step` and `error`, beside the derive half's fields of `design/gpu-programs.md`. `program.read` reads elements of a published resource at a step, from the step history, so that an agent can check what its derive half wrote.

### Stage Commands

Pieces replace stages for new work. `stage.set`, `stage.remove` and `stage.check` keep working unchanged, and they gain nothing. Graphics Consolidation removes them with the stage hooks, and a spec with a `stage` key then fails with `spec.unknown_key`.

### Status and Events

The event `program` gains the keys `half` (`derive` or `draw`) and `pass` (the pass's index, or null), and the changes `stopped` and `refused` carry the codes of this page. The event `pick` gains the keys of Picking. A program pinned to a missing or changed piece gives the event `refused` with `piece.missing` or `piece.hash_mismatch` when the spec opens.

## Library Pieces

The library is `src/builder/programs/library/`: the lib modules in `lib/`, and the pieces in `pieces/<name>/<version>/` in the format of Piece Files. The build embeds it, `piece.list` and `piece.show` read it, and `caliper-mcp` serves each file as a read-only resource under `caliper://library/`, so that an agent can read a piece before it pins, copies or imitates it. This work fills the library with what the demo needs: the four lib modules and `points@1`.

`points@1` draws a cloud as the built-in `points` primitive draws it: round sprites of a radius in pixels, colored by a packed RGBA or by a value through a colormap, with depth and an ID for each point. Its derive half reads an (N, 3) position tensor and an optional (N) value tensor and publishes one element for each point. It matches the built-in draw when the same points, camera and options give the same pixels within the tolerances of `tests/scene/draw_reference.h`: 0 levels for a looked-up color, 1 for a blended pixel, and an edge pixel that may take either side. Picks at 16 chosen pixels must also return the same items. A test in the app's offscreen mode compares the two.

## Pixel Comparison

A new test compares the app and the browser module on the same frame. A fixture holds two instances, `attention_field` and `attention_arcs`, their pieces, and one recorded version of each resource, thus no training runs. The app draws the fixture in its offscreen mode on its Metal device, and the browser module draws it in headless Chromium through Playwright, at 512 by 512 pixels with a fixed camera. The test then compares the color and the ID targets:

- An interior pixel, whose 3 by 3 neighborhood has one ID in both images, differs by at most 2 levels in each channel, and by at most 3 in the bloom of the glow target.
- An edge pixel may differ in any way, but the edge pixels are at most 1 % of the image.
- The ID of each interior pixel is the same.

CI runs it on the macOS lane, with the browser on SwiftShader, as the browser suite runs. It waits on a written frame, never on a fixed time.

## Attention Arc Tower

The demo answers one question: when and where does the model learn to copy, and is that what drops the loss? An agent builds it through `caliper-mcp` alone, on a managed run whose script's only Caliper line is `caliper.watch(model, optimizer)`. No built-in primitive is in its frames: the scene panels `tower` and `pattern` hold only programs, and the loss curves are 2-D panels, which are interface components.

### Training Script

`examples/python/arc_tower_train.py` trains a two-layer transformer with four heads, a width of 64 and a vocabulary of 64 words, on sequences of 32 tokens in which a random segment of 12 tokens repeats, so that induction heads form at a visible bump of the loss. Row 0 of every batch is the fixed sentence of the arc example, and the loss leaves it out, thus the shown attention is always of the same sentence and never trains on it. The attention module calls a leaf `weights = nn.Identity()` on the softmax of the scores, so that the watch publishes `live:blocks.<i>.attn.weights.act`. It has the integer `n_head` and the output projection `proj`, so that `intervene.points` lists its heads. A leaf `token_loss = nn.Identity()` takes the per-token loss of all rows, so that a source can slice row 0's second copy into the in-context loss. The script fixes its seed, and its size puts the bump within a few minutes on this machine.

### Agent Steps

The agent starts the run with `run.start`, reads `catalog.list`, writes `attention_arcs` and `attention_field` as workspace pieces with `piece.check` and `piece.update`, and places them with `program.create` in the panels `tower` and `pattern`. It waits for the event `compiled` of both halves, checks the arcs with `program.read`, and adds the loss series, the in-context loss, the time transport and a readout of `layer`, `head` and `score` from the state that a pick writes. The person scrubs through the training and sees the arcs from each token of the second copy back to the token after its first occurrence form at the bump. A click on an arc opens that head's pattern in `pattern` and fills the readout. The agent then places a toggle on `intervene:blocks.1.attn#<head>`, and the in-context loss rises while it is on. Last, the agent forks the run from a step before the bump with the head ablated from the child's first step, and the child never forms that head's arcs.

### Demo Gaps

| Gap | Plan |
|---|---|
| Attention weights in the catalog | The script's leaf `weights`; no change of Caliper |
| Head points | The script's `n_head` and `proj`; no change of Caliper |
| Fork from a past step | The parent keeps checkpoints, thinned as kept versions are; `run.fork` takes `step` and starts from the newest checkpoint at or before it |
| Ablation from the child's first step | `run.fork` takes `interventions`, which the child applies before its first step |
| Scrub of the whole training | Kept versions thinned over the run (Kept Versions) |
| Step stamps from the native sender | Joined with the budget of an attached instrument, before this work |
| In-context loss | The script's leaf `token_loss`, and a source with a slice and a reduce |
| Training length | Measured by the demo's work and pinned in the script with its seed |

### Run Budget

The derive halves run in the shared cadence of `design/gpu-programs.md`, which keeps the passes and the views of an attached viewer at 2 % of the run's wall clock, and the hooks and the publish path inside the rest of the 5 %. The arcs piece runs 8,200 invocations a pass and publishes at most 4,096 arcs of 24 bytes, the field piece 32 KiB, thus a pass costs mostly its sync point, about 1 ms on the small model, and the cadence sets N from it. The acceptance test checks a measured `overhead_share` of at most 0.02 while the viewer is attached.

## Second Graphic

`attention_field` is the second graphic on the same contract, unlike the tower: a fragment-only field with no geometry, the camera `none`, contour lines, and parameters that follow two state variables. It reads a live tensor through its derive half and shows linked selection between two pieces, because a pick in `tower` retargets it.

## Acceptance Test

The test drives `caliper-mcp` against the hidden app with `CALIPER_AGENT_OFFLINE=1`, as `tests/gpu_programs/` does. Its driver is `tests/programmable_graphics/driver.py` with the steps in `tests/programmable_graphics/transcript.jsonl`, its ctest entry is `caliper_programmable_graphics_acceptance`, and its check script is `examples/python/arc_tower_check.py`. The steps, in order:

1. Record the SHA-256 list of the app, `caliper-mcp` and the installed package.
2. Start `examples/python/arc_tower_train.py` with `run.start`.
3. `piece.list` lists `points@1` and the four lib modules. `piece.show` of `points` returns its files.
4. `piece.check` of both pieces is ok, and `piece.update` writes both as version 1.
5. Create the panels `tower` and `pattern` without entities, and both programs. Wait for the events `compiled` of both halves of both.
6. `program.read` of `tower_arcs/arcs` gives at least one element, and each element's tokens are below 32.
7. Capture `tower`: at least 2,000 pixels differ from the empty panel, and `panel.get` lists no entity.
8. Wait until the newest `heads` version holds a score above 0.4, then move the cursor to the first kept version: there every score is below 0.2. Capture both.
9. Pick at the projected middle of the brightest arc of the best head: the event has `program`, `layer`, `head`, `from_token` and `to_token`, and the state `head` equals the picked head. Capture `pattern` before and after: they differ.
10. Turn on a toggle on `intervene:blocks.1.attn#<head>`: within 200 steps the in-context loss rises above twice its value before. Turn it off.
11. `run.fork` from the step of the first kept version with the head ablated: 500 steps past the parent's step of step 8, the child's score of that head stays below 0.2.
12. Create a program of a broken copy of `attention_arcs` whose bloom loops for 50 ms: the event `stopped` with `program.draw_over_time` names the program and pass 2, and `tower_arcs` still draws.
13. `program.status` shows `overhead_share` of at most 0.02 for the derive halves.
14. Stop the runs, check that no process that the test started is alive, and check that the SHA-256 list did not change.

## Work Items

| Work Item | Size in Builder Days | Depends On |
|---|---|---|
| Pieces: the codec of pieces and instances with the schema, the workspace and library stores with path safety, imports, the canonical manifest and the hash, the seven piece commands, the library's MCP resources, and every schema change of this page in the command table | 5, and 1 more for the preview | The interface |
| Wire and resources: RESOURCE and the instance form of PROGRAM in the three codecs with golden frames, the relay to viewers and the hub, the app's resource store, kept versions with thinning, `program.read` | 5.5 | Pieces |
| Derive half: the derive composer and layouts, several passes, resources, counters and clears in the producer runtime, the facade's install and the RESOURCE send in both senders, the status | 5.5 | Wire and resources |
| Draw half: the draw composer and checks, the render graph on the graphics layer, the layer's additions, the scene view's frame, draw state, timing, removal and memory | 8 | Pieces |
| Browser viewer: the same runtime in the module, PROGRAM and RESOURCE in its network source, the browser checks, the pixel comparison | 4.5 | Draw half |
| Picking, text and the fixes of the scene panels | 5 | Draw half |
| Demo: the training script, the fork from a past step with interventions, the pieces as an agent writes them, the second graphic, `points@1` and its match test | 6.5 | All of the above |
| Acceptance test, check script and docs | 3 | All of the above |

The items come to 43 builder days, and 44 with the preview, against about 40 for this work with pieces. The first cuts, in this order, are the preview (1 day), the offscreen targets and the bloom (1 day, and the arcs keep their additive glow), the scene options of a live scene panel (1 day), and the ellipsis of cut text (0.5 day).

## Decisions

- **Piece as the unit.** A program of a spec is an instance of a piece, because one format for library shapes, workspace pieces and inline experiments lets an instrument, the node editor and Graphics Consolidation use one thing.
- **Pins with hashes.** A program pins name, version and hash, because a saved instrument must draw what it drew when it was saved, and a hash proves it in every host.
- **No silent repin.** A missing piece or a changed hash refuses the program, because a different piece in place of the pinned one would show different data as if it were the same.
- **Hash over resolved text.** The hash covers the lib modules' text, because it runs, and released modules never change, thus the hash stays stable.
- **Canonical JSON.** The manifest is hashed as RFC 8785 JSON, because the editor, the script and the browser compute the hash in three languages.
- **Same-length rewrite.** Imports rewrite `::` to `__`, because a rewrite of the same length keeps every line and column of the author's files.
- **Prefixed modules.** A lib module declares its items with `<module>__`, because then resolution rewrites only the piece's calls and never parses a module.
- **Fixed file names.** A piece has only `piece.toml`, three WGSL files and `sample/`, because a fixed set of names keeps every path inside `<workspace>/pieces/`.
- **Generated bindings.** The composer writes every binding of both halves, because the browser has no reflection and must bind by the manifest alone.
- **One composer, three builds.** The composer and the checks are one source without dependencies, because the app, the script and the browser must compose the same text.
- **Module per pass.** The draw half composes each pass on its own, because WebGPU forbids a writable storage buffer in a vertex stage that the same buffer's compute pass needs.
- **Params, not state.** A piece reads params only, and a program binds a param to a state, because a piece must not depend on the names of one spec.
- **Packed textures.** A texture resource is written as a packed buffer, because storage textures are cut and the layer copies a buffer into a texture.
- **Eight-bit targets.** Offscreen targets are `rgba8unorm`, because wider targets are cut and the layer has no 16-bit float format.
- **Counts on the CPU.** An instance count comes from a resource's valid count, because indirect draws are cut and the count travels with the resource.
- **ID in every frame.** A pick pass writes the ID in the frame, because a separate pick pass would run the piece's vertex work twice and need a second pipeline for each pass.
- **Submit per program.** Each program's passes go in a submit of their own, because the layer measures GPU time for each submit.
- **Removal like stages.** A draw over its budget for 30 frames, or a frame that hangs 1 s, is removed, because the host cannot stop a GPU pass and stages already use these limits.
- **One resource a message.** RESOURCE carries one resource, because the history keeps resources apart and a large one must not hold a small one back.
- **Reserved type.** The type 0x0082 stays a comment until its codecs exist, because a defined type needs a golden frame and three decoders.
- **Relay by the script.** The script sends each draw half to its viewers, because a browser viewer of a run then gets the graph from the same place as the resources.
- **Even thinning.** Kept versions thin evenly by doubling their stride, because the cursor must scrub the whole run and the bump can come at any step.
- **No history in the browser.** The browser draws the newest version, because the step history lives in the builder session, which the module does not have.
- **Resources outside the catalog.** A resource is no catalog source in this work, because programs and `program.read` read it, and catalog sources of library pieces come with the derivation catalog.
- **Text from the derive half.** Label anchors come from published resources only, because the host places text on the CPU and only those reach it.
- **Web limits in the app.** The app refuses what a browser would refuse, because a piece that runs in the app must run in a browser.
- **Schemas with the codec.** The spec schema gains pieces with the codec, because the codec refuses their keys until then.
- **Command table with the pieces work.** One work item writes every schema change of this page, because the table and its pinned tests are one file each, and one owner keeps them consistent.

## Left Out

- Custom blend factors, targets wider than eight bits, indirect draws, storage textures, multisampling and node materials, by the contract.
- A derive half that reads another derive half's output; it comes with the derivation catalog.
- Kept versions in the browser.
- Resources as catalog sources.
- Recordings of resources.
- The producer runtime on Vulkan; it stays Metal only.
- The compiler in the wheel, unless the demo needs it.

## Open Points

- **Other heads.** In the fork, another head may form induction arcs in place of the ablated one. The acceptance test therefore asks only that the ablated head's arcs never form. The demo's work measures it, and the agent may ablate every head of layer 1.
- **Interface reflection.** Check 7 needs the entry points' inputs and outputs, which the reflection of caliper-wgsl does not give yet. Until it does, the app relies on the layer's pipeline creation for those checks.
- **Thin arcs.** Without multisampling, arcs of one pixel alias. A width of at least one pixel times the pixel ratio keeps them visible.
- **Timestamps in a browser.** The timestamp query is optional in WebGPU, thus a browser's frame time can be the slower upper bound.
