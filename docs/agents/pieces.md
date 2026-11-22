# Pieces

An agent that uses only `caliper-mcp` can write the graphics of an instrument. The unit of this work is the piece. A piece is a folder with a manifest and WGSL files, and it has two halves. The derive half runs compute passes in the training script and publishes named, typed resources. The draw half runs in the viewer every frame, reads those resources and draws into a scene panel. A program of the open spec is an instance of a piece. It binds the inputs of the piece, sets its parameters and names its panel.

The sandbox checks both halves before anything runs. Every access of a piece to a buffer stays bounds-checked. A piece never writes a model tensor, and it never adds a message per training step. [GPU Programs and Stages](gpu-programs.md) describes the one-pass programs, the sandbox and the shared cadence that the derive half uses.

## Piece Folder

A piece is a folder with the manifest `piece.toml` and up to three WGSL files. The file `common.wgsl` holds structs, constants and functions that both halves include. The file `derive.wgsl` holds the entry points of the derive half, and `draw.wgsl` holds the entry points of the draw half. The optional folder `sample/` holds one file `sample/<input>.bin` for each input that the manifest names in `sample`. The host reads no other file name.

A workspace piece lives in `<workspace>/pieces/<name>/<version>/`. A name matches `[a-z][a-z0-9_]{0,31}`, and a version is a whole number from 1 to 1,000,000. Thus no path that the host builds can leave the folder `pieces/`. The host refuses a link anywhere in a piece folder, and it opens a file only when its real path lies inside `pieces/`. Each WGSL file holds at most 64 KiB. The samples of one piece hold at most 1 MiB together. A breach gives `piece.bad_path`.

An inline piece lives in one spec, in the top-level array `piece`. It has the keys of `piece.toml`, and its table `files` holds the WGSL under the keys `common`, `derive` and `draw`. A spec holds an inline piece for a quick experiment, and `piece.save` moves it into the workspace.

## Piece Manifest

The manifest names the piece, its inputs, its parameters, its resources, its imports and the passes of each half. The files of a piece declare no `@group` and no `@binding`. The host writes every binding from the manifest, as the section Generated Header shows.

| Key | Value |
|---|---|
| `name` | The name of the piece, at most 32 characters |
| `version` | The version, 1 when absent. A saved version never changes. |
| `description` | One line that says what the piece draws or publishes, at most 256 characters |
| `inputs` | At most 8 typed inputs that a program binds |
| `params` | At most 16 typed parameters with defaults and ranges |
| `resources` | At most 8 resources that the derive half writes |
| `imports` | At most 8 lib modules by name and version, such as `colormap@1` |
| `depends` | At most 8 pieces, by `name` and `version`, whose outputs feed the buffer inputs of this piece |
| `derive` | The passes and limits of the derive half. It is absent for a piece that only draws. |
| `draw` | The camera, buffers, targets, passes, item map, text table and limits of the draw half. It is absent for a piece that only publishes. |
| `sample` | A small input for each named input, as `input` and `shape` |
| `hash` | `sha256:` and 64 lowercase hex digits. The host computes it, and a given value must match. |

### Inputs and Parameters

An input of kind `tensor` binds a tensor of the training script. The derive half reads it read-only and in place. An input of kind `buffer` binds a published buffer resource of another program, and the draw half reads it.

| Input Key | Value |
|---|---|
| `name` | The WGSL name of the generated array |
| `kind` | `tensor` or `buffer` |
| `dtype` | For a tensor, `f32` (default), `i32` or `u32`. The bound tensor must have this type. |
| `rank_min`, `rank_max` | For a tensor, the fewest and the most axes, 1 and 8 when absent |
| `element`, or `struct` with `fields` | For a buffer, the layout that the bound resource must have |
| `title` | The label of the input in the builder |

A parameter is a field of the generated struct `Params`, and both halves read it as `params.<name>`. A parameter without a value in the program gets a range control from `min` to `max`. Without a range it keeps its default.

| Parameter Key | Value |
|---|---|
| `name` | The name of the field |
| `type` | `f32` (default), `i32` or `u32` |
| `default` | The value when the program gives none, 0 when absent |
| `min`, `max` | The range of the control |
| `title` | The label of the control |

### Resources

A resource is a named, typed array that the derive half writes and the viewer reads. A `buffer` holds `count` elements of one plain type or of a struct. A `texture` is an `rgba8unorm` image that the derive half writes as a packed `array<u32>` in row order. The viewer copies it into a texture that the draw half samples.

| Resource Key | Value |
|---|---|
| `id` | A WGSL name, unique among the inputs, resources and draw buffers of the piece |
| `kind` | `buffer` or `texture` |
| `element`, or `struct` with `fields` | For a buffer, one plain type, or a struct whose fields are written `name: type` in order |
| `count` | For a buffer, the elements, at most 4,194,304 |
| `counted` | For a buffer, the derive half also gets the counter `<id>_count`. The valid elements are the first min(counter, count). |
| `width`, `height` | For a texture, the size in pixels, at most 4,096 each |
| `publish` | `false` for a private resource that never leaves the script. `true` when absent. |
| `clear` | `true` zeroes the resource before each run. `false` when absent, and the contents stay from the run before. |
| `keep` | The versions that the step history keeps, from 1 to 4,096. 256 when absent. |

A field type is `f32`, `i32`, `u32`, `vec2<f32>`, `vec4<f32>`, `vec2<u32>` or `vec4<u32>`. The composer lays out a struct by the storage rules of WGSL, and one element holds at most 256 bytes. A struct has at most 16 fields. A published resource holds at most 67,043,328 bytes, because then one version fits one wire frame. A private resource holds at most 64 MiB.

## Piece Halves

The two halves share one manifest and the file `common.wgsl`. They run in different processes and at different times.

### Derive Half

The derive half runs at a pass of the shared cadence, inside the budget of the run. The table `derive` holds `passes` (1 to 8), `time_ms` (4 by default, at most 50), `memory_mb` (64 by default, at most 512) and `every` (1 to 10,000). A pass has `entry`, a compute entry point of `derive.wgsl` that is never named `main`, `workgroups` (`[1, 1, 1]` when absent) and `writes`, the resources that the pass changes.

The passes run in their order in one submit, with a barrier between two passes. After a run, the runtime publishes each published resource that a pass names in `writes`. A published resource that no pass names is static, and the runtime publishes it after the first run only. A resource keeps its contents from one run to the next, thus the derive half can keep state such as a running mean. A counter returns to zero before each run.

The derive half reads tensors read-only and without a copy. A pass that writes a tensor input is refused. The watchdog, the timeout and the budget of the shared cadence apply to one run of all passes.

### Draw Half

The draw half runs in the viewer every frame, after the draws of the scene. The table `draw` holds the camera, `buffers`, `targets`, `passes`, `pick`, `text`, `frame_ms` and `memory_mb`.

| Draw Key | Value |
|---|---|
| `camera` | `orbit` (default), `pan_zoom` or `none` |
| `buffers` | At most 4 buffers that the draw half keeps between frames, each with `id`, a layout, `count` and `reset` |
| `targets` | At most 4 offscreen targets, each with `id`, `scale` (1, 0.5 or 0.25) and `clear` |
| `passes` | 1 to 16 passes of one frame, in order |
| `pick` | The item map, with `resource` and `fields` |
| `text` | The text table, with `anchors`, `position`, `string`, `priority` and `strings` |
| `frame_ms` | The most GPU time of one frame in milliseconds, 4 when absent and at most 16 |
| `memory_mb` | The budget of the resources, buffer inputs, draw buffers and targets in the viewer, 64 when absent and at most 256 |

A pass has these keys. A compute pass uses `entry`, `workgroups` and `writes`. A render pass uses the other keys.

| Pass Key | Value |
|---|---|
| `kind` | `compute` or `render` |
| `entry`, `workgroups`, `writes` | The compute entry point, its workgroup counts and the draw buffers that it changes |
| `vertex` | The vertex entry point. Without it, the pass draws the full-screen triangle `caliper_fullscreen` with 3 vertices. |
| `fragment` | The fragment entry point |
| `target` | `host` (default), the color, ID and depth of the panel, or the id of an offscreen target |
| `reads` | The offscreen targets that the pass samples |
| `instances` | A count, or the id of a buffer resource or buffer input whose valid count at the cursor is the count. 1 when absent. |
| `vertices` | The vertices of one instance, at most 65,536. It is necessary with `vertex`. |
| `topology` | `triangle_list` (default), `triangle_strip`, `line_list`, `line_strip` or `point_list` |
| `blend` | `opaque`, `alpha` or `additive`. The default is opaque into `host` and additive into a target. |
| `depth` | `none`, `test` or `write`. The default is `write` with a vertex entry and `none` for a full-screen triangle. |
| `pick` | The fragment returns `CaliperPicked` and writes the item ID at location 1. Only a pass into `host` may do this. |

The passes of one frame run in their declared order. A pass whose `instances` names a resource with no version yet, or with a valid count of zero, is skipped. An offscreen target is `rgba8unorm` at `scale` of the panel size, and it clears to `clear` at the start of each frame. A pass samples a target that an earlier pass of the frame wrote, and never its own target. No pass reads the host target. The ID never blends.

A draw buffer keeps its contents from frame to frame, thus motion comes from the draw half. `caliper_view.frame.x` counts the frames since the last reset, and `caliper_view.frame.y` is 1 on the first frame after a reset. The host zeroes every draw buffer at an install, at a new attachment and at a new epoch. A buffer with `reset = "cursor"` also returns to zero when the cursor moves by a command, a key or the transport. A buffer with `reset = "resource"` also returns to zero when a resource of the frame changes its version. A change of a parameter never resets a buffer.

The render graph has no custom blend factors, no culling, no storage texture, no multisampling and no indirect draw or dispatch. A compute pass of the draw half never changes an instance count, because the count comes from the CPU copy of a resource.

## Draw Prelude

The host puts one fixed WGSL text, the draw prelude, before every pass of a draw half. It is the same in the app and in the browser module. It declares these names, which all start with `caliper_` or `Caliper`.

| Name | Meaning |
|---|---|
| `caliper_view` | The uniform `CaliperView` of the frame: the matrices `clip_from_world`, `world_from_clip` and `view_from_world`, the eye, the viewport, the pointer, the time, the step cursor and the frame counters |
| `caliper_selection` | A read-only array of at most 256 entries of the linked selection. Each entry holds the pick ID of a program, the item and the first four integer fields of its item map as tensor indices. |
| `caliper_linear`, `caliper_nearest` | Two samplers |
| `caliper_colormaps` | The colormaps of the host as one texture |
| `CaliperFullscreen`, `CaliperPicked` | The input of a full-screen fragment and the output of a pass with `pick` |

Inside `caliper_view`, `step` holds the step of the cursor, the newest step, the step of the resources that the frame shows and 1 when the cursor is live. `time` holds the seconds since the install, the seconds since the previous frame and the pixel ratio. `pointer` holds the pointer position in pixels, its buttons and 1 when the pointer is over the panel. A piece reads no state variable by name, thus a piece stays independent of the spec that uses it. A parameter carries any value that comes from a state.

### Generated Header

The composer writes the header of each module from the manifest in a fixed order. Thus the app, the script and the browser bind the same names at the same places. A file that declares `@group` or `@binding` gives `program.binding_not_allowed`.

| Half | Group | Bindings |
|---|---|---|
| Derive | 0 | 0 `caliper: CaliperFrame`, 1 `params: Params`, 2 `shapes: CaliperShapes` |
| Derive | 1 | One `var<storage, read>` array for each tensor input, at the index of the input |
| Derive | 2 | Each resource at 2 times its index, `read_write`, and its counter `<id>_count: atomic<u32>` at 2 times its index plus 1 |
| Draw | 0 | The five bindings of the prelude, then 5 `params: Params` |
| Draw | 1 | 0 `counts: CaliperCounts`, then each published resource and each buffer input, read-only, from 1 in the order of the manifest |
| Draw | 2 | Each draw buffer at its index, and each offscreen target that the pass reads at 4 plus its index |

`CaliperShapes` holds one `vec4<u32>` for each tensor input, named as the input. It holds the last four axes of the shape, padded with 1 at the front. `CaliperCounts` holds one `u32` for each published resource and buffer input, which is its valid count in the version of the frame. `Params` holds the parameters in the order of the manifest. The composer also writes the struct of each layout in both halves.

A piece may not name a resource, input or draw buffer `params`, `counts`, `shapes` or `caliper`, and it may not declare a name that starts with `caliper_` or `Caliper`. A name with a double underscore is also reserved. A breach gives `program.reserved_name`.

## Shader Imports

A lib module is a WGSL file of helpers that pieces import by name and version. The built-in modules live in `src/builder/programs/library/lib/<name>/<version>.wgsl`. A released version never changes, and a change makes the next version. A module may use the bindings of the prelude, and it declares no binding of its own.

| Module | Items |
|---|---|
| `colormap@1` | `sample(map, t)`, `viridis(t)`, `magma(t)` and the rows `VIRIDIS`, `MAGMA`, `RDBU` and `GREYS` of the host colormaps |
| `pick@1` | `id(item)`, the ID of a pick pass, and `selected(item)` and `selected_index(index)` for the linked selection |
| `camera@1` | `clip(world)`, `pixel_direction(a, b)` and `offset_pixels(clip, pixels)` |
| `easing@1` | `smooth(t)` and `out_cubic(t)` |

A piece lists a module in `imports` and calls its items as `<module>::<item>`, for example `colormap::magma(t)`. The host resolves the imports before the compiler runs. It checks that each `<module>::<item>` names an imported module and one of its items, and else it gives `piece.unknown_import`. It then rewrites each `::` to `__` and puts the text of the modules, in the order of `imports`, before `common.wgsl`.

The rewrite keeps the length of every line. Thus every line and column of the author files stay where they were. The composed module records its segments, which are the prelude, the generated header, each module, `common.wgsl` and the file of the half. A message of the compiler on a line of the composed text maps back to its file and its line, with the same column. The messages of `piece.check` and the status name that file.

The derive half has no prelude. A module that reads the prelude, which are `colormap@1`, `pick@1` and `camera@1`, stays out of the derive half. A call of one of them from `common.wgsl` or `derive.wgsl` gives `piece.unknown_import`.

## Hashes and Pins

The hash identifies what runs. It is the SHA-256 of these bytes, in lowercase hex after `sha256:`:

1. The line `caliper-piece/1`.
2. The manifest as RFC 8785 canonical JSON, without `hash` and without the files.
3. For each imported module in the order of `imports`, and then for each of `common.wgsl`, `derive.wgsl` and `draw.wgsl` that the piece has, a line `--- <part> <bytes>` and the text with LF line ends.

The hash covers the text of the imported modules, because they run. It does not cover the samples. The canonical JSON and the LF line ends make the hash the same on every platform and in every language that computes it. The editor, the script and the browser module all compute it.

A program pins its piece by name, version, hash and source. The pin is `piece = { name = "...", version = 1, hash = "sha256:...", from = "workspace" }`. The editor writes the hash when the pin has none. A program keeps its pinned version until `program.patch` pins another one, and `piece.update` never changes a saved version.

When the source of a pin lacks the piece, the state of the program is `refused` with `piece.missing`. When the piece has another hash than the pin, the state is `refused` with `piece.hash_mismatch`. In both cases the spec still opens, and the app never pins another version on its own. The person or the agent restores the folder, or pins an existing version with `program.patch`. A script and a browser viewer compute the hash again from what they receive. A hash that differs from the pin refuses the program.

## Piece Sources

A program names the source of its piece with `from`.

- `library` is the default. It is the built-in library of the build, which is read-only. It holds the lib modules, and its pieces live in `pieces/<name>/<version>/` in the format of a piece folder.
- `workspace` is the folder `<workspace>/pieces/`, which the piece commands write.
- `inline` is the array `piece` of the open spec.

A workspace piece with the name of a library piece replaces it only in a program that says `from = "workspace"`. `caliper-mcp` serves each file of the library as a read-only resource under `caliper://library/`. Thus an agent can read a piece before it pins, copies or imitates it.

## Program Instances

A program of a spec is an instance of a piece. It holds no WGSL, and none of the one-pass keys `wgsl`, `entry`, `inputs`, `output`, `limits` and `workgroups`. A mix of the two forms gives `program.form_mixed`.

| Key | Value |
|---|---|
| `id` | The id of the program |
| `title` | An optional label |
| `piece` | The pin, with `name`, `version`, `hash` and `from` |
| `bind` | One entry for each input: `live:<tensor>` for a tensor input, and `program:<id>/<resource>` for a buffer input |
| `params` | A number, or a binding reference to a control, a selection or a state, for each parameter |
| `panel` | The scene panel that the draw half writes. It is necessary when the piece draws. |

The editor checks each binding against the catalog. It checks the dtype and rank of a tensor, and the layout of the published resource of another program. An input that is unbound, or bound to another dtype, rank or layout, gives `piece.bad_binding`. Buffer inputs that form a cycle give `piece.cycle`. A binding that names a resource that the other piece does not publish also gives `piece.bad_binding`.

A buffer input makes the programs of a spec a graph. Only the draw half reads a buffer input. The viewer gives it the version of the other resource at the cursor.

This example is the piece `attention_field`. It draws the attention pattern of one head as a field with contour lines. Its derive half copies the first batch row of two attention tensors into one resource, and its draw half has one full-screen pass.

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
count = 8192

[derive]
passes = [{ entry = "copy_rows", workgroups = [16, 1, 1], writes = ["pattern"] }]

[draw]
camera = "none"
passes = [{ kind = "render", fragment = "field" }]
```

A program then places the piece in the panel `pattern`, and it follows the head that a pick selects through two state variables.

```toml
[[program]]
id = "head_pattern"
piece = { name = "attention_field", version = 1, from = "workspace", hash = "sha256:..." }
panel = "pattern"
bind = { attn0 = "live:blocks.0.attn.weights.act", attn1 = "live:blocks.1.attn.weights.act" }
params = { layer = { state = "layer" }, head = { state = "head" }, gain = 2.0 }
```

## Piece Commands

Each command is one MCP tool, with the first dot of its name replaced by `_`. The commands `piece.check`, `piece.list`, `piece.show` and `program.read` read. The commands `piece.save`, `piece.copy`, `piece.update` and `piece.remove` change the workspace or the open spec.

| Command | Arguments | Function |
|---|---|---|
| `piece.check` | `piece`, `headers` | Checks a piece without saving it. It returns `ok`, the messages with file, line and column, the hash, one item for each pass and the generated header of each half. |
| `piece.list` | `from`, `match` | Lists the library, workspace and inline pieces. Each entry has its inputs, params, outputs, hash, halves and the programs that pin it. It also lists the lib modules. |
| `piece.show` | `name`, `version`, `from` | Returns one piece with its manifest, its files, its folder and its pins. The newest version of the first source that has the name is the default. |
| `piece.save` | `name`, `version`, `dry_run` | Moves an inline piece into the workspace and pins its programs to it, in one undo step of the spec. |
| `piece.copy` | `name`, `version`, `as` | Forks a library piece into the workspace as version 1, under its own name or under `as`. |
| `piece.update` | `piece` | Writes a new version of a workspace piece after the checks. The version must be above every saved version, and it is the next one when absent. |
| `piece.remove` | `name`, `version` | Removes one workspace version and its folder. |
| `program.read` | `id`, `resource`, `step`, `first`, `count` | Reads elements of a published resource at a step from the step history. |

`piece.check` composes each pass with the prelude and the generated header, and it compiles it as the script and the viewer compile it. The messages of the compiler name the file and the line in that file. `piece.update` and `piece.save` run the same checks first. `piece.update` returns `name`, `version`, `hash`, `path` and `messages`, and `piece.save` also returns the programs that now pin the piece.

`piece.remove` is destructive and asks for the card of the person when an agent sends it. A version that a program of the open spec pins gives `piece.in_use`. A library piece gives `piece.read_only`. `piece.save` and `piece.update` on a version that exists give `piece.exists`. A command that names a piece that the source lacks gives `piece.not_found`.

`program.create` takes the keys of Program Instances. With `inline`, it adds a whole piece to the inline pieces of the spec in the same undo step. `program.patch` changes `piece`, `bind`, `params`, `panel` and `title`. `program.remove` removes an instance as it removes any program.

`program.read` returns `step`, `count`, `elements` and `kept`. `count` is the number of valid elements of the version at or before `step`, and the cursor step is the default. Each element is the list of the numbers of its fields in order. A vector field gives one number for each component. `count` is at most 1,024 and 16 by default. `kept` holds `versions`, `first` and `last` of the versions that the history holds.

## Program Status and Events

`program.status` gives an instance the kind `piece` and its pin. The derive half reports as for any program, with `host`, `state`, `every`, `memory_bytes` and `error`. An instance that draws also has a `draw` object with `state` (`waiting`, `compiled`, `running`, `refused` or `stopped`), `frame_ms`, `frame_limit_ms`, `memory_bytes`, `memory_limit_bytes`, `resource_step` and `error`. The `error.where` of the draw half names the pass, such as `draw.passes[1]`.

The event `program` carries the keys `half` (`derive` or `draw`) and `pass`, the index of the pass or null. An agent waits for the event `compiled` of both halves of an instance before it reads. The changes `stopped` and `refused` carry the codes of this page. The event `pick` carries the keys of the next section. [Run Control and Events](run-control-and-events.md) lists all events.

## Resources and Step History

The script sends each version of a published resource to a viewer as one RESOURCE message. The message needs the capability `resources` on both ends. A script that does not announce it refuses an instance with `program.resources_unavailable`. One message carries one resource, thus a large resource never holds a small one back. A viewer that falls behind gets the newest version of each resource and loses the older ones.

The app keeps the versions of each published resource in the step history under `<program>/<resource>`, beside the kept steps of the sources. A resource keeps at most `keep` versions. When its ring is full, the history drops every second kept version. From then on it keeps a version only when its step lies far enough after the newest one. Thus the kept versions span the whole run evenly, and the step cursor scrubs the whole training. The viewer gives the draw half the newest kept version at or before the cursor.

The spec saves the pins, the bindings, the params and the inline pieces, and it never saves a resource. When an instrument opens, the app resolves each pin, checks each piece and sends each program to the attached script. The script publishes again from its next pass. Without an attached run, the state of the program is `waiting`.

## Selection and Picking

A pass with `pick = true` writes `pick::id(item)` at location 1 of the host target in every frame. The ID holds the pick ID of the program and the item. The pick ID of a program starts above the IDs of the entities of the scene. The scene view reads a 16 by 16 window around the pointer from the newest completed frame, with its ID and depth. A hit of a built-in entity wins only when its depth is nearer.

The item is the index of an element of the resource that the item map names. The item map lists 1 to 8 fields of that element, and each field is an `f32`, `i32` or `u32` field. The host reads the element from the CPU copy of the version that the frame showed. It then makes a `pick` event with these keys beside the common ones.

| Key | Value |
|---|---|
| `program` | The id of the program |
| `item` | The element index |
| `step` | The step of the version |
| `fields` | The fields of the item map for this element, by name, as numbers |
| `index` | The first four integer fields as tensor indices, and -1 where there are fewer |

A pick in a panel with the option `select` writes a state variable from a field. The field names of a piece are those of its item map. Every draw half of the spec also reads the pick through `caliper_selection`. Thus `pick::selected(item)` and `pick::selected_index(index)` let one program mark the head that a pick in another program chose.

`selection.set` makes the same pick without a pointer. With `panel`, `program` and `item`, it picks the element `item` of the item map of that program at the cursor. The program must draw a piece with an item map in that panel. The result is `fields` with the keys of the table above, and the command makes the same event, state writes and linked selection as a click. `clear` removes the selection. The command changes the session and not the spec, thus it has no undo step.

| Code | Cause |
|---|---|
| `selection.bad_item` | The program draws no piece in the panel, `item` is not a whole number from 0 to 4,294,967,295, the program has no version at the cursor yet, or `item` lies beyond the valid count |
| `program.not_found` | The open spec has no program of that id |
| `program.bad_pick` | The piece has no item map, or the item map names no buffer |

## Program Text

The host owns text. A piece publishes the anchors of its labels in a buffer resource, and its text table names them. The table has `anchors`, the resource, and `position`, a `vec4<f32>` field whose xyz is the world position of the label. It has `string`, a `u32` field that indexes `strings`, and an optional `priority`, an `f32` field. The list `strings` holds at most 4,096 strings of at most 64 bytes each.

The scene view reads the anchors of the version that the frame shows and projects them with the camera. It places at most 512 labels with the overlap rule and the leader lines of the other labels of the panel. A label of a higher priority wins an overlap. When more than 512 anchors are valid, the anchors of the highest priority stay. A position or a priority that is not a finite number leaves the anchor out. A string index outside the table leaves it out too. The anchors come from a derive half only, because only the derive half output reaches the CPU. A table that breaks its limits gives `program.bad_text`.

## Camera

The host owns the camera. A piece selects `orbit` for the 3-D orbit camera of the panel, `pan_zoom` for its 2-D camera, or `none`. The camera `none` gives identity matrices, for a piece that draws in clip space. The programs of one panel must agree. The first program of the panel sets the camera, and a program that selects another one is refused with `program.camera_conflict`. The saved views of the panel and the commands `camera.get`, `camera.set` and `camera.save` work as they do for entities.

## Draw Budget and Removal

A draw half writes only the color, ID and depth of its panel and its own targets. The host records the passes of each program in a submit of its own after the frame of the scene. The GPU time of that submit is the frame time of the program. Where the backend has no timestamps, the host measures the time from the submit until the work is done, which is an upper bound.

The host cannot stop a pass that runs on the GPU. Thus it removes a draw that goes over its budget and reports it. The other programs of the panel keep drawing, and the derive half keeps its own limits.

| Condition | Result |
|---|---|
| A program is over `frame_ms` on 30 frames in a row | `program.draw_over_time`, naming the costliest pass |
| A frame of a program is not done after 1 s | `program.draw_timeout`, at once |
| The programs of one panel together are over 16.7 ms on 30 frames in a row | `program.draw_over_time` for the costliest program |
| A resource, a draw buffer or a target would pass `memory_mb` | `program.draw_over_memory` |
| The graphics layer refuses a submitted frame | `program.layer_refused` |
| A version is larger than its declared resource, or does not match its layout and count | `program.resource_too_large` or `program.bad_resource` |

The `draw` object of the status shows `stopped` with the code and the `where` of the pass, such as `draw.passes[2]`. The message names the program and the pass, for example "Stopped tower_arcs: draw pass 2 (bloom) took 50.0 ms a frame, above its 4 ms". A removed program frees its buffers and targets. A change of the program installs it again.

## Refusals

The editor, the script and the viewer refuse a piece with a stable code before anything runs. The codes of pieces and instances are in this table. The code `program.compile_failed` and the other codes of GPU programs keep their meaning for both halves.

| Code | Cause |
|---|---|
| `piece.bad_name` | A name or a version outside its form |
| `piece.bad_manifest` | A key, a count or a value of the manifest |
| `piece.bad_path` | A file outside its folder, a link, or a file name that a piece cannot have |
| `piece.unknown_import` | No such lib module, or `<module>::<item>` names nothing |
| `piece.not_found` | A command names a piece that its source lacks |
| `piece.missing` | A program pins a piece that its source lacks |
| `piece.hash_mismatch` | A pinned or given hash differs from the hash of the piece |
| `piece.exists` | That name and version exist in the workspace |
| `piece.read_only` | A command would change a library piece |
| `piece.in_use` | A program of the open spec pins that version |
| `piece.bad_binding` | An input that is unbound, or bound to another dtype, rank or layout |
| `piece.cycle` | A piece depends on itself, or a program binds a resource of its own |
| `program.form_mixed` | A program has keys of the one-pass form and a pin |
| `program.bad_resource` | A kind, count or size of a resource, or a duplicate id |
| `program.bad_layout` | A field name or type, or a stride over 256 bytes |
| `program.unknown_resource` | A pass, an item map or a text table names no resource |
| `program.bad_pass` | The kind, entry, target or values of a pass |
| `program.target_cycle` | A pass reads its own target, or a target before a pass writes it |
| `program.bad_pick` | The item map names no field, or the piece has no pass with `pick` |
| `program.bad_text` | The anchors lack their fields, or the strings break their limits |
| `program.camera_conflict` | Two programs of a panel select two cameras |
| `program.binding_not_allowed` | A file declares a binding, or a pass writes a tensor input |
| `program.reserved_name` | A file declares a name of the host or a name with a double underscore |
| `program.entry_stage` | An entry point of the wrong stage, none, or the name `main` |
| `program.over_web_limits` | A pass goes over a default limit of WebGPU |
| `program.resources_unavailable` | The attached script takes no pieces |
| `program.browser_refused` | WebGPU refused a module or a pipeline of the browser viewer |

The manifest check holds every pass to the default limits of WebGPU, thus a piece that runs in the app also runs in a browser. One stage binds at most 8 storage buffers. As a result, the published resources, buffer inputs and draw buffers of a piece number at most 7 beside the selection.

## Browser Viewer

The browser module runs the same draw half from the same WGSL and the same declaration. It receives each draw half as PROGRAM from the script, and the resources as RESOURCE. It checks the names, the files, the imports and the hash with the same code that the app compiles. A hash that differs from the pin refuses the program. It then gives each composed module to WebGPU. A compilation message or a refused pipeline refuses the program with `program.browser_refused`, and the line maps back to its file.

The browser has no compiler of its own, because WebGPU keeps each draw inside its bindings. The browser module keeps no step history and draws the newest version of each resource. Without timestamps, the frame time is the time from the submit until the work is done.

## Tests

[Testing](../development/testing.md) lists the tests of pieces. The test `tests/piece_path/driver.py` follows one piece from its command to a pick. The test `tests/draw_parity/` compares the draw of the app and of the browser module on one recorded frame. Each refusal of this page has a test in `src/builder/programs/tests/test_pieces.cpp` or in the suite of the host that gives it.
