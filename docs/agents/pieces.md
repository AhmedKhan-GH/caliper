# Pieces

An agent that uses only `caliper-mcp` can write the graphics of an instrument. The unit of this work is the piece. A piece is a folder with a manifest and WGSL files, and it has two halves. The derive half runs compute passes in the training script and publishes named, typed resources. The draw half runs in the viewer every frame, reads those resources and draws into a scene panel. A program of the open spec is an instance of a piece. It binds the inputs of the piece, sets its parameters and names its panel.

The sandbox checks both halves before anything runs. Every access of a piece to a buffer stays bounds-checked. A piece never writes a model tensor, and it never adds a message per training step. [GPU Programs](gpu-programs.md) describes the one-pass programs, the sandbox and the shared cadence that the derive half uses.

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
| `dtype` | For a tensor, `f32` (default), `i32`, `u32`, `u8` or `u16`, or two to five of them joined by `\|`, such as `f32\|u8`. The bound tensor must have one of these types. |
| `rank_min`, `rank_max` | For a tensor, the fewest and the most axes, 1 and 8 when absent |
| `element`, or `struct` with `fields` | For a buffer, the layout that the bound resource must have |
| `title` | The label of the input in the builder |
| `optional` | `true` when a program may leave the input unbound. `false` when absent. |

A tensor input of one 32-bit type binds an array of that type. A tensor input whose dtype names `u8` or `u16`, or several types, is typed by its binding. The header binds the bytes of its tensor in place as `array<u32>` under the name of the input, and it declares two names more. The constant `<name>_type` holds the bound type as `CALIPER_F32`, `CALIPER_I32`, `CALIPER_U32`, `CALIPER_U8` or `CALIPER_U16`. The accessor `<name>_at(i: u32)` returns element `i` of the tensor. It returns `f32` when the dtype names `f32`, else `i32` when it names `i32`, else `u32`. The accessor returns zero at or past the element count of the tensor, which the field `<name>__len` of `shapes` holds. The script composes the header for the bound type, thus the same piece reads a u8 image or an f32 image without a copy. A rebind to a tensor of another type is refused, because the program must compile again. The raw words stay readable within the binding, which holds the bytes of the tensor rounded up to a whole word.

A program may leave an optional input unbound, and the piece then sees a declared default instead of a tensor that it does not need. An unbound tensor input has the shape (0, 0, 0, 0) in `shapes`, and every read of its array gives zero. An unbound buffer input has the count 0 in `counts`, and a pass whose `instances` names it is skipped. A bound tensor never has an axis of 0, because the script imports no tensor without elements. Thus `shapes.<name>.w == 0u` tells a piece that the input is unbound.

A parameter is a field of the generated struct `Params`, and both halves read it as `params.<name>`. A parameter without a value in the program gets a range control from `min` to `max`. Without a range it keeps its default.

| Parameter Key | Value |
|---|---|
| `name` | The name of the field |
| `type` | `f32` (default), `i32` or `u32` |
| `default` | The value when the program gives none, 0 when absent |
| `min`, `max` | The range of the control |
| `title` | The label of the control |

### Resources

A resource is a named, typed array that the derive half writes and the viewer reads. A `buffer` holds `count` elements of one plain type or of a struct. A buffer of the narrow element `u8` or `u16` packs four or two elements into each 32-bit word, from the low bits up, and both halves bind it as `array<u32>`. The draw half reads it through the accessor `<id>_at(i: u32)`, which returns zero at or past the valid count in `counts`. A narrow buffer holds at most 134,217,728 `u8` or 67,108,864 `u16` elements, which is 33,554,432 words.

A `texture` has a format and a size for each version. The native derive half writes its texels as a packed `array<u32>` in row order, and it sets the size of the version in `<id>_size`, a `vec2<u32>`. The runtime sets `<id>_size` to the largest sides before each run, thus a piece that never writes it publishes the largest size. A size outside the largest sides, or with more texels than `texels`, gives a version without texels. The viewer checks the format and the size of each version against the manifest, and it makes a texture of that size, which the draw half samples. An `rgba8unorm` or `r8unorm` texture is a `texture_2d<f32>` that the samplers filter.

An `r32uint` texture is a `texture_2d<u32>` that the draw half reads with `textureLoad`. Its optional `fields` declares one pick field, such as `depth: f32` or `class_id: u32`. The field type is `f32`, `i32`, or `u32`. It controls pick decoding and leaves the raw texture bits unchanged. Without a field declaration, the pick field is `value: u32`.

| Resource Key | Value |
|---|---|
| `id` | A WGSL name, unique among the inputs, resources and draw buffers of the piece |
| `kind` | `buffer` or `texture` |
| `element`, or `struct` with `fields` | For a buffer, one plain type or the narrow `u8` or `u16`, or a struct whose fields are written `name: type` in order |
| `count` | For a buffer, the elements, at most 33,554,432 words |
| `counted` | For a buffer, the derive half also gets the counter `<id>_count`. The valid elements are the first min(counter, count). |
| `format` | For a texture, `rgba8unorm` (default, one word a texel), `r8unorm` (four texels a word) or `r32uint` (one word a texel) |
| `width`, `height` | For a texture, the largest sides of a version in pixels, at most 8,192 each |
| `texels` | For a texture, the most texels of a version, `width` times `height` when absent |
| `publish` | `false` for a private resource that never leaves the script. `true` when absent. |
| `clear` | `true` zeroes the resource before each run. `false` when absent, and the contents stay from the run before. |
| `keep` | The versions that the step history keeps, from 1 to 4,096. 256 when absent. |

A field type is `f32`, `i32`, `u32`, `vec2<f32>`, `vec4<f32>`, `vec2<u32>` or `vec4<u32>`. The composer follows the WGSL storage rules. An element holds at most 256 bytes, and a struct has at most 16 fields.

A published buffer holds at most 128 MiB because each viewer binds it whole. This limit matches the default WebGPU storage limit. A published texture holds at most 256 MiB, enough for 8,192 by 8,192 RGBA8 or R32 texels.

The published resources of a piece hold at most 512 MiB together. The history reserves that amount for resources within its 768 MiB total budget. When resources use the full budget, sources share the other 256 MiB. A single source keeps at most 128 MiB.

The `lines@1` piece holds 4,194,304 segments in separate 32-byte endpoint rows and 8-byte pick rows. The `mesh@1` piece holds 4,194,304 vertices in separate 32-byte vertex rows and 12-byte attribute rows. Its 4,194,304 triangles, 256 MiB texture, and flag word bring the resource maximum to 480 MiB plus 4 bytes. The split retains scalar values, colors, normals, UVs, and pick indices.

A texture with `texels` 67,108,864 holds versions of up to 8,192 pixels per side and 256 MiB. An `r8unorm` texture of 8,192 by 8,192 pixels holds 64 MiB. A private resource holds at most 64 MiB.

## Piece Halves

The two halves share one manifest and the file `common.wgsl`. They run in different processes and at different times.

### Derive Half

The derive half runs at a pass of the shared cadence, inside the budget of the run. The table `derive` holds `passes` (1 to 8), `time_ms` (4 by default, at most 50), `memory_mb` (64 by default, at most 512) and `every` (1 to 10,000). A pass has `entry`, a compute entry point of `derive.wgsl` that is never named `main`, `workgroups` (`[1, 1, 1]` when absent) and `writes`, the resources that the pass changes.

The passes run in their order in one submit, with a barrier between two passes. After a run, the runtime publishes each published resource that a pass names in `writes`. A published resource that no pass names is static, and the runtime publishes it after the first run only. A resource keeps its contents from one run to the next, thus the derive half can keep state such as a running mean. A counter returns to zero before each run.

The derive half reads tensors read-only and without a copy. A pass that writes a tensor input is refused. The watchdog, the timeout and the budget of the shared cadence apply to one run of all passes.

Each derive pass is a program of its entry point alone. It binds only the tensor inputs, resources and counters that its entry point uses, and one pass binds at most 8 storage buffers, the default of WebGPU. A piece can thus hold more storage buffers than one pass, such as a mesh with a texture, when it gives the work to several passes. A pass over 8 gives `program.over_web_limits`, which names the pass.

No compute pass of either half may use `var<workgroup>`. The Metal hosts give a compute pass no threadgroup memory, thus such a variable would read zero in the app and hold its values in a browser. `piece.check`, the script and the app refuse a pass whose entry point uses workgroup memory with `program.workgroup_memory`. A piece keeps partial results in a private resource or a draw buffer instead.

### Draw Half

The draw half runs in the viewer every frame, after the draws of the scene. The table `draw` holds the camera, `buffers`, `targets`, `passes`, `pick`, `text`, `frame_ms` and `memory_mb`.

| Draw Key | Value |
|---|---|
| `camera` | `orbit` (default), `pan_zoom` or `none` |
| `buffers` | At most 4 buffers that the draw half keeps between frames, each with `id`, a layout, `count` and `reset` |
| `targets` | At most 4 offscreen targets, each with `id`, `scale` (1, 0.5 or 0.25) and `clear` |
| `passes` | 1 to 16 passes of one frame, in order |
| `pick` | The item map, with `resource` and `fields` |
| `text` | The text table, with `anchors`, `position`, `string`, `priority`, `strings` or `strings_from`, and `align` |
| `frame_ms` | The most GPU time of one frame in milliseconds, 4 when absent and at most 16 |
| `memory_mb` | The budget of the resources, buffer inputs, draw buffers and targets in the viewer, 64 when absent and at most 512 |

A pass has these keys. A compute pass uses `entry`, `workgroups` and `writes`. A render pass uses the other keys.

| Pass Key | Value |
|---|---|
| `kind` | `compute` or `render` |
| `entry`, `workgroups`, `writes` | The compute entry point, its workgroup counts and the draw buffers that it changes |
| `vertex` | The vertex entry point. Without it, the pass draws the full-screen triangle `caliper_fullscreen` with 3 vertices. |
| `fragment` | The fragment entry point |
| `target` | `host` (default), the color, ID and depth of the panel, or the id of an offscreen target |
| `reads` | The offscreen targets that the pass samples |
| `instances` | A count, a published resource, or a buffer input. Resources use their valid element count, or their texel count for textures. 1 when absent. |
| `vertices` | The vertices of one instance, at most 65,536. It is necessary with `vertex` and without `indices`. |
| `indices` | A published buffer resource or buffer input of `u32` elements. The pass draws its valid elements as an index buffer, and each instance draws all of them. |
| `topology` | `triangle_list` (default), `triangle_strip`, `line_list`, `line_strip` or `point_list` |
| `blend` | `opaque`, `alpha` or `additive`. The default is opaque into `host` and additive into a target. |
| `depth` | `none`, `test` or `write`. The default is `write` with a vertex entry and `none` for a full-screen triangle. |
| `pick` | The fragment returns `CaliperPicked` and writes the item ID at location 1. Only a pass into `host` may do this. |

The passes of one frame run in their declared order. A pass whose `instances` names a resource with no version yet, or with a valid count of zero, is skipped. An offscreen target is `rgba8unorm` at `scale` of the panel size, and it clears to `clear` at the start of each frame. A pass samples a target that an earlier pass of the frame wrote, and never its own target. No pass reads the host target. The ID never blends.

A draw buffer keeps its contents from frame to frame, thus motion comes from the draw half. `caliper_view.frame.x` counts the frames since the last reset, and `caliper_view.frame.y` is 1 on the first frame after a reset. The host zeroes every draw buffer at an install, at a new attachment and at a new epoch. A buffer with `reset = "cursor"` also returns to zero when the cursor moves by a command, a key or the transport. A buffer with `reset = "resource"` also returns to zero when a resource of the frame changes its version. A change of a parameter never resets a buffer.

An indexed pass has a vertex entry and the topology `triangle_list`, `line_list` or `point_list`, because a strip would read the index 4,294,967,295 as a restart. Each index becomes the `vertex_index` of the vertex entry. The render graph binds no vertex buffer, thus an index is only a number that the vertex entry reads, and every read of a buffer is bounds-checked. An index past the valid elements of a resource therefore reads zero or an element of the binding, and never memory outside it. The host checks no index on the CPU, which costs nothing per index. It only holds the index count to the elements of the version, which the buffer holds. A pass whose index resource has no version, or no valid element, is skipped.

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
| Derive | 2 | Each resource at 2 times its index, `read_write`, and its counter `<id>_count: atomic<u32>` or a texture's size `<id>_size: vec2<u32>` at 2 times its index plus 1 |
| Draw | 0 | The five bindings of the prelude, then 5 `params: Params` |
| Draw | 1 | 0 `counts: CaliperCounts`, then each published resource and each buffer input, read-only, from 1 in the order of the manifest |
| Draw | 2 | Each draw buffer at its index, and each offscreen target that the pass reads at 4 plus its index |

`CaliperShapes` holds one `vec4<u32>` for each tensor input, named as the input. It holds the last four axes of the shape, padded with 1 at the front. After them, it holds the element count `<name>__len: u32` of each typed input. `CaliperCounts` holds one `u32` for each published resource and buffer input, which is its valid count in the version of the frame: the elements of a buffer, also of a narrow one, and the texels of a texture. `Params` holds the parameters in the order of the manifest. The composer also writes the struct of each layout in both halves.

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

The hash covers the text of the imported modules, because they run. It does not cover the samples. The canonical JSON and the LF line ends make the hash the same on every platform and in every language that computes it. The editor, the app and the browser module compute it.

A program pins its piece by name, version, hash and source. The pin is `piece = { name = "...", version = 1, hash = "sha256:...", from = "workspace" }`. `program.create` and `program.patch` write the hash when the pin has none. A pin without a hash in a spec file, written by hand or with `spec.set_text`, refuses its program with `spec.missing_key`, and `program.patch` with the same pin writes the hash. A program keeps its pinned version until `program.patch` pins another one, and `piece.update` never changes a saved version.

When the source of a pin lacks the piece, the state of the program is `refused` with `piece.missing`. When the piece has another hash than the pin, the state is `refused` with `piece.hash_mismatch`. In both cases the spec still opens, and the app never pins another version on its own. The person or the agent restores the folder, or pins an existing version with `program.patch`. The app computes the hash from the files that it sends, and it sends a program to the script only when that hash equals the pin. The script checks only the form of the pin's hash, because the derive half comes with its imports resolved and without the manifest. A browser viewer computes the hash again from the sections that it receives, and a hash that differs from the pin refuses the program.

### Foreign Specs

A spec that came from outside the workspace has a table `provenance` with `foreign = true`. A program of such a spec runs only when `provenance.reviewed` holds the hash of its source. For an instance of a piece, the target is `program:<id>` and the hash is the 64 hex digits of the piece hash. Until then, the state of the program is `review`, and nothing runs, neither the derive half nor the draw half. The `draw` object of the status shows `review` too. No command writes `provenance`, and the app has no review card for pieces yet. Thus a person adds the entry to the spec file. A spec of the person's own workspace has no `provenance`, and its pieces run at once.

## Piece Sources

A program names the source of its piece with `from`.

- `library` is the default. It is the built-in library of the build, which is read-only. It holds the four lib modules and the piece `points@1`, and its pieces live in `pieces/<name>/<version>/` in the format of a piece folder.
- `workspace` is the folder `<workspace>/pieces/`, which the piece commands write.
- `inline` is the array `piece` of the open spec.

The piece `points@1` draws a cloud of round or square points of a size in pixels, or with `radius_mode` 1 of a radius in scene units. It colors them by one color, by a value through a colormap, or by a packed RGBA value. Its derive half reads a tensor of positions with rank 2, a tensor of values and an optional tensor of radii with rank 1, one radius for each point or one for all. The values are an `f32` value or packed RGBA bits for each point, a `u32` packed RGBA, a `u8` color row of 3 or 4 channels, or an `i32`, `u16` or `u8` class id, which takes the class color of the scene, entry k mod 16. Color mode 2 draws the rows, the packed values and the class colors. A radius becomes `radius * radius_scale * (proj_x, proj_y)` in clip units, with a minimum of half a pixel. The param `radius_scale` is the scale of the model-view, and `proj_x` and `proj_y` are the scales of the projection. Saved renderer images and original pick identities check its pixels and picks within the scene test tolerances.

The params `att_ref`, `fade_w0` and `fade_w1` of `points@1` give the depth cues of the live points view, and at 0 they are off. With `att_ref` above 0, a point is its size times clamp(`att_ref` / w, 0.5, 2) wide, where w is its clip-space w. Past `fade_w0`, its color falls linearly toward half at `fade_w1`, and its hue and alpha stay.

The piece `surface@1` draws a height field as a lit surface in a data box.
Its derive half pools a rank-2 tensor into at most `max_cells` cells, as the `surface` operator does.
Its draw half renders this grid, with saved reference images and original pick identities as test expectations.
The params `x_lo` to `z_hi`, `box_x` to `box_z`, `vmin` and `vmax` set the data box and value range.
Each bit of `fit` takes one of these ranges from the data.

The piece `lines@1` reads positions, values, and a topology tensor. It draws segments, strips with their lengths, or edges from index pairs. The piece `arrows@1` reads origins, vectors, and values. Both pieces support one color, colormaps, packed RGBA, width, depth offset, and translucent colors. The piece `path@1` keeps every s-th row of a tensor at the chosen columns and draws a strip through those rows. Each pick returns the original item index.

The pieces `image@1`, `depth@1` and `segmentation@1` draw color images, depth maps and class masks. Their inputs use `u8` or `f32` for images, `u16` or `f32` for depth, and `i32`, `u16` or `u8` for class IDs. Each piece supports up to 8,192 by 8,192 pixels in a texture of at most 256 MiB. The color texture uses `rgba8unorm` with bilinear sampling. Depth and class textures use `r32uint` with exact texel reads. Depth texels contain float bits, and the image map applies its colormap and optional cloud.

Each image lies flat at a pose or on a camera image plane. Without the optional `pose` input, the piece uses the identity pose. The optional segmentation colors cover class IDs from 0 through 65,535. Classes without a color use the default palette. A pick returns the pixel index and its color, depth or class ID.

The piece `boxes@1` draws up to 1,048,576 boxes with optional rotations, colors, labels, and translucent faces. It supports `u8` and `f32` colors, plus `i32`, `u16`, and `u8` class IDs. Its labels contain UTF-8 bytes in 16 `u32` words per row. The piece selects at most 4,096 labels at box centers through its published `names` resource.

The piece `mesh@1` draws up to 4,194,304 vertices and 4,194,304 triangles. It supports normals, flat faces, one color, RGBA, and scalar values through a colormap. Its `faces` resource holds triangle indices. Without indices, each group of three vertices forms a triangle. Without colors, the one color applies. Optional UVs and a texture add bilinear texture samples, multiplied by vertex RGB with texel alpha. The texture supports up to 8,192 pixels per side and 67,108,864 texels.

The piece `frustum@1` draws the frustum of a pinhole camera.
Its derive half reads a pose tensor and a pinhole tensor with seven numbers.
These are fx, fy, cx, cy, image width, image height and image-plane distance.
It draws four edges from the camera center to the image corners and draws the image outline.
The params set one color and a width in pixels.
Each edge returns the camera's pick, element 0.

The scene view draws scene parts and spec entities with library pieces.
Current senders supply resources in the [Piece Form](../reference/wire-protocol-3-2.md#piece-form).
A read-only decoder preserves per-shape parts in saved recordings.
The view binds entity poses, depth offsets, class colors, camera intrinsics and panel data boxes to the pieces.
A surface entity uses `surface@1`, and a lines entity with a `path` source step uses `path@1`.

A pinhole view draws pieces within the image rectangle and preserves the letterbox.
Selection highlights use pieces too.
`image@1` dims the picture, and extra `points@1` or `boxes@1` instances draw selected rows without depth.
The view reports unsupported parts and piece errors through its notes.

The live points view uses `points@1` in the app and browser module.
Each instance holds at most 4,194,304 points, with selection instances after the base points.
The view supplies positions, packed colors, camera matrices, point sizes and depth-cue params.
Private GPU payloads pass through textures and asynchronous readback before the view draws them.
Hover, picks and selections use CPU-visible positions with the same camera.
The view's own points have no agent draw budget.
The browser's Canvas 2D points view remains available without WebGPU.

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

The editor checks each binding against the catalog. It checks the dtype and rank of a tensor, and the layout of the published resource of another program. An input that is unbound and not optional, or an input bound to another dtype, rank or layout, gives `piece.bad_binding`. Buffer inputs that form a cycle give `piece.cycle`. A binding that names a resource that the other piece does not publish also gives `piece.bad_binding`.

A buffer input makes the programs of a spec a graph. Only the draw half reads a buffer input. The viewer gives it the version of the other resource at the cursor.

This example is the piece `attention_grid` of `demos/arc_tower/`. It draws the attention weights of one head as a grid with one cell for each pair of tokens. Its derive half copies the first batch row of two attention tensors into one resource. It also places one label anchor for each token at the left of its row. Its draw half has one full-screen pass, and its text table holds the words of the sentence.

```toml
name = "attention_grid"
version = 1
description = "The attention grid of one head of a two-layer model, one cell for each pair of tokens, with the words of the rows"
imports = ["colormap@1"]

[[inputs]]
name = "attn0"
kind = "tensor"
rank_min = 2
rank_max = 2
title = "Attention Weights of Layer 0"

[[inputs]]
name = "attn1"
kind = "tensor"
rank_min = 2
rank_max = 2
title = "Attention Weights of Layer 1"

[[params]]
name = "layer"
type = "i32"
min = 0
max = 1
title = "Layer"

[[params]]
name = "head"
type = "i32"
min = 0
max = 3
title = "Head"

[[params]]
name = "gain"
default = 1.0
min = 0.1
max = 10.0
title = "Gain"

[[resources]]
id = "pattern"
kind = "buffer"
element = "f32"
count = 8192      # 2 layers, 4 heads, 32 by 32 tokens

[[resources]]
id = "labels"
kind = "buffer"
struct = "Label"
fields = ["position: vec4<f32>", "string: u32"]
count = 32

[[derive.passes]]
entry = "copy_rows"
workgroups = [16, 1, 1]
writes = ["pattern"]

[[derive.passes]]
entry = "place_labels"

[draw]
camera = "none"

[draw.text]
anchors = "labels"
strings = [
  "the", "old", "map", "shows", "red", "fox", "jumps", "over", "lazy", "dogs", "near", "green",
  "hills", "under", "blue", "skies", "red", "fox", "jumps", "over", "lazy", "dogs", "near", "green",
  "hills", "under", "blue", "skies", "and", "then", "it", "ends",
]

[[draw.passes]]
kind = "render"
fragment = "grid"
```

A program then places the piece in the panel `pattern`, and it follows the head that a pick selects through two state variables.

```toml
[[program]]
id = "head_pattern"
piece = { name = "attention_grid", version = 1, from = "workspace", hash = "sha256:..." }
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

`program.status` gives an instance the kind `piece` and its pin. The derive half reports as for any program, with `host`, `state`, `every`, `memory_bytes` and `error`. An instance that draws also has a `draw` object with `state` (`waiting`, `review`, `compiled`, `running`, `refused` or `stopped`), `frame_ms`, `frame_limit_ms`, `memory_bytes`, `memory_limit_bytes`, `resource_step` and `error`. The `error.where` of the draw half names the pass, such as `draw.passes[1]`.

The event `program` carries the keys `half` (`derive` or `draw`) and `pass`, the index of the pass or null. An agent waits for the event `compiled` of both halves of an instance before it reads. The changes `stopped` and `refused` carry the codes of this page. The event `pick` carries the keys of the next section. [Run Control and Events](run-control-and-events.md) lists all events.

## Resources and Step History

The script sends each version of a published resource to a viewer as one RESOURCE message. The message needs the capability `resources` on both ends. A script that does not announce it refuses an instance with `program.resources_unavailable`. One message carries one resource, thus a small resource never waits for a large one. A version of more than 8 MiB goes as RESOURCE parts of 8 MiB, which the viewer puts together and shows only whole ([Wire Protocol 3.2](../reference/wire-protocol-3-2.md#resource-parts)).

The script sends the frames of a pass to a viewer in chunks of at most 8 MiB and 64 KiB, which is half of the 16 MiB control queue of a connection. A chunk goes only when the viewer's queue then holds at most that much. This count also holds the frames that other programs gave to that viewer in the same pass and that the script has not queued yet. The first chunk goes with the pass or not at all, thus a viewer that falls behind skips whole passes and gets a later pass when it catches up. The other chunks of a pass wait for the viewer, and each step sends the chunks that then fit. A viewer takes no other pass until they went, and a pass makes no copy of its resources while every viewer waits. Two programs that publish 8 MiB each do not fit in one pass of a viewer. The program that the viewer skipped goes first at the next pass, thus the programs reach the viewer in turn. The time of the copies and of the chunks counts toward the overhead of the passes, thus the cadence stretches for a piece that publishes large versions. A pass that would send more than 128 MiB of one buffer, 256 MiB of one texture, or 512 MiB together, from a manifest that did not pass the app's check, stops the program with `program.resource_too_large`.

The app keeps the versions of each published resource in the step history under `<program>/<resource>`, beside the kept steps of the sources. A resource keeps at most `keep` versions. When its ring is full, the history drops every second kept version. From then on it keeps a version only when its step lies far enough after the newest one. Thus the kept versions span the whole run evenly, and the step cursor scrubs the whole training. The viewer gives the draw half the newest kept version at or before the cursor. A version that replaces another at the same step, as after a fork from that step, takes its place in the history, in the draw and in picks and labels.

The app keeps only a version that the pinned piece declares. The program must be in the open spec, the piece must publish a resource of that name, and the version must have the declared stride, at most the declared count and count times stride bytes. A texture version must have the declared width and height. The app refuses every other version and keeps nothing of it. These limits hold for the resources in the step history:

| Limit | Value |
|---|---|
| Resources of one program | 8, as many as a piece declares |
| Resources of all programs | 128, 8 for each of the 16 programs of a script |
| Bytes of one resource version | 128 MiB for buffers, 256 MiB for textures |
| Bytes of all resources | 512 MiB within the history budget of 768 MiB |

Over the bytes of all resources, the largest rings drop every second version first. Then the largest rings drop their oldest versions, down to one version each. A version that does not fit beside one version of each other resource is not kept. The history can keep two versions of a texture of 256 MiB alone. Other resources share the same budget. Of a piece that publishes 512 MiB at each pass, it keeps one version of each resource, the newest, and the cursor finds no older pass of that piece.

The spec saves the pins, the bindings, the params and the inline pieces, and it never saves a resource. When an instrument opens, the app resolves each pin, checks each piece and sends each program to the attached script. The script publishes again from its next pass. Without an attached run, the state of the program is `waiting`.

## Selection and Picking

A pass with `pick = true` writes `pick::id(item)` at location 1 of the host target in every frame. The ID holds the pick ID of the program and the item. The pick ID of a program starts above the IDs of the entities of the scene. The scene view reads a 16 by 16 window around the pointer from the newest completed frame, with its ID and depth. A hit of a built-in entity wins only when its depth is nearer.

The item is the index of an element of the resource that the item map names. The item map lists 1 to 8 fields of that element, and each field is an `f32`, `i32` or `u32` field. The item map of a texture names its one field: `rgba` for an `rgba8unorm` texel or the declared scalar field for an `r32uint` texel, or `value` when absent. Its item is a texel of the version in row order. An `r8unorm` texture and a narrow buffer have no item map. The host reads the element from the CPU copy of the version that the frame showed. It then makes a `pick` event with these keys beside the common ones.

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

The strings can also come from the run. With `strings_from`, the table names a published buffer resource or buffer input in place of `strings`, and each of its valid elements is one string. The layout of that buffer holds only `u32`, `vec2<u32>` and `vec4<u32>` fields, at most 64 bytes, and a resource of it holds at most 4,096 elements. The bytes of an element are UTF-8 text, and the first zero byte ends the string before the end of the element. The host reads the version of that buffer that the frame shows, and it checks each string again. A string that is not valid UTF-8 leaves its labels out. Thus the strings of a table are at most 4,096, each of at most 64 bytes and 256 KiB together, whichever source they come from. A derive half writes the strings of a run, such as the label of each box, from a tensor of packed bytes.

The key `align` says where the host puts each label. With `placed`, the default, the placer chooses one of eight places around the anchor, as for the other labels of the panel. With `left`, `right` or `center`, the host pins each label at its anchor, with the start, the end or the middle of its text there and the text centered on the anchor vertically. A pinned label does not move away from a label that it overlaps, and the labels that the placer places after it keep clear of it. Axis labels use it.

The scene view reads the anchors of the version that the frame shows and projects them with the camera. It places at most 512 labels with the overlap rule and the leader lines of the other labels of the panel. A label of a higher priority wins an overlap. When more than 512 anchors are valid, the anchors of the highest priority stay. A position or a priority that is not a finite number leaves the anchor out. A string index outside the table leaves it out too. The anchors come from a derive half only, because only the derive half output reaches the CPU. A table that breaks its limits gives `program.bad_text`.

### Hover Labels

The optional `text.hover` table maps a local pointer hit to one label.
For example, `hover = { anchor = 33, string_offset = 32 }` uses anchor 33 and string `32 + item`.
The anchor retains its published position, priority, and alignment.
The host hides this anchor when the pointer has no current hit in that program.
The anchor's stored string index does not control a hover label.

The `anchor` field is required, and `string_offset` defaults to zero.
The anchor must fit its resource and have an index below 512.
A hover label requires a buffer item map.
The string offset plus the declared item count must fit the string table.
Strings can come from the manifest or a published resource with a declared count.
A buffer input has no declared count and cannot supply hover strings.

The app and browser use the program ID target for the local hit.
Pointer exit, camera motion, a drag, or a node resource change clears the previous hit.
The text resources and item map must show the same step.
The hover does not write selection state, send a command, or request a derive pass.
All existing string and label limits apply, and a piece without `text.hover` retains its text behavior.

### Selected Labels

The optional `text.selected` table selects one label from the current draw parameter.
For example, `selected = { anchor = 32, param = "selected_node", string_offset = 32 }` uses string `32 + selected_node`.
The app and browser use the same parameter bytes as the draw frame.
A selection change needs no new derive pass, resource version, or model step.
The anchor retains its published position, priority, and alignment.
Its stored string index does not control the selected label.

The `anchor` and `param` fields are required, and `string_offset` defaults to zero.
The parameter must have type `i32` or `u32` and name an item in the shown buffer item map.
A negative value, an absent item, or an invalid string hides the label.
The anchor, item map, and text resource must show the same step.
The selected anchor must differ from the hover anchor.
The anchor and string limits for hover labels also apply to selected labels.

## Camera

The host owns the camera. A piece selects `orbit` for the 3-D orbit camera of the panel, `pan_zoom` for its 2-D camera, or `none`. The camera `none` gives identity matrices, for a piece that draws in clip space. The programs of one panel must agree. The first program of the panel sets the camera. `program.create` and `program.patch` refuse a program that selects another one with `program.camera_conflict`, and `spec.validate` names each later program of a spec that holds such a panel. The viewer also refuses the later program with the same code. The saved views of the panel and the commands `camera.get`, `camera.set` and `camera.save` work as they do for entities. A panel that holds programs alone frames their box, -1 to 1 on each axis, and opens at its `view_camera`: the angles of an orbit with the fit of the box at those angles, and its target and distance when it gives them.

## Draw Budget and Removal

A draw half writes only the color, ID and depth of its panel and its own targets. The host records the passes of each program in a submit of its own after the frame of the scene. The GPU time of that submit is the frame time of the program. Where the backend has no timestamps, the host measures the time from the submit until the work is done, which is an upper bound.

A training run on the same GPU can delay a correct frame by a second or more, thus the GPU time of a frame decides. A frame that is not done 1 s after its submit is late, and the program submits no new frame until it is done. A frame over 250 ms of GPU time is heavy, which is more than 15 times the largest `frame_ms`. Heavy frames in a row remove the program when their GPU time passes 1 s together. Thus one frame over 1 s of GPU time removes it at once. Where the backend has no timestamps, three late frames in a row remove it.

The host cannot stop a pass that runs on the GPU. Thus it removes a draw that goes over its budget and reports it. The other programs of the panel keep drawing, and the derive half keeps its own limits.

| Condition | Result |
|---|---|
| A program is over `frame_ms` on 30 frames in a row | `program.draw_over_time`, naming the costliest pass |
| Heavy frames in a row took more than 1 s of GPU time together | `program.draw_timeout`, at once |
| The driver ended a late frame | `program.draw_timeout` |
| Three frames in a row are late, on a backend without timestamps | `program.draw_timeout` |
| A frame of a program is not done after 5 s | `program.draw_timeout`, at once |
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
| `piece.bad_binding` | An input that is unbound and not optional, or bound to another dtype, rank or layout |
| `piece.cycle` | A piece depends on itself, or a program binds a resource of its own |
| `program.form_mixed` | A program has keys of the one-pass form and a pin |
| `program.bad_resource` | A kind, count or size of a resource, a published buffer above 128 MiB, a texture above 256 MiB, published resources above 512 MiB together, or a duplicate id |
| `program.bad_layout` | A field name or type, or a stride over 256 bytes |
| `program.unknown_resource` | A pass, an item map or a text table names no resource |
| `program.bad_pass` | The kind, entry, target or values of a pass |
| `program.target_cycle` | A pass reads its own target, or a target before a pass writes it |
| `program.bad_pick` | The item map names no field, or the piece has no pass with `pick` |
| `program.bad_text` | The anchors lack their fields, the strings break their limits, both `strings` and `strings_from` are given, or the buffer of `strings_from` has another layout or more than 4,096 elements |
| `program.camera_conflict` | Two programs of a panel select two cameras. The editor, `spec.validate` and the viewer give it. |
| `program.binding_not_allowed` | A file declares a binding, or a pass writes a tensor input |
| `program.reserved_name` | A file declares a name of the host or a name with a double underscore |
| `program.entry_stage` | An entry point of the wrong stage, none, or the name `main` |
| `program.interface_mismatch` | A fragment entry reads a location that the vertex entry does not write, or with another type or interpolation, or a vertex entry reads a vertex attribute |
| `program.over_web_limits` | A pass goes over a default limit of WebGPU, such as a derive pass that binds more than 8 storage buffers |
| `program.workgroup_memory` | An entry point of a compute pass uses `var<workgroup>` |
| `program.resources_unavailable` | The attached script takes no pieces |
| `program.browser_refused` | WebGPU refused a module or a pipeline of the browser viewer |

The manifest check holds every pass to the default limits of WebGPU, thus a piece that runs in the app also runs in a browser. One stage binds at most 8 storage buffers. As a result, the published resources, buffer inputs and draw buffers of a piece number at most 7 beside the selection. The derive half counts the storage buffers of each pass after the compile, because a pass binds only what its entry point uses.

## Browser Viewer

Library scene pieces require vertex storage.
On a WebGPU device in compatibility mode without vertex storage, the viewer reports `program.browser_refused` for each affected entity.

The browser module runs the same draw half from the same WGSL and the same declaration. It receives each draw half as PROGRAM from the script, and the resources as RESOURCE. It checks the names, the files, the imports and the hash with the same code that the app compiles. A hash that differs from the pin refuses the program. It then gives each composed module to WebGPU. A compilation message or a refused pipeline refuses the program with `program.browser_refused`, and the line maps back to its file.

The browser has no compiler of its own, because WebGPU keeps each draw inside its bindings. The browser's dashboard viewer gets the instances and the newest resources of a scene panel through `panel.data` and draws them with the same module. A program of a foreign spec stays in review there until the app reviews it. The browser module keeps no step history and draws the newest version of each resource. It keeps a version only of a resource that the piece of its program publishes, with at most the declared count and bytes, and it keeps at most 16 programs of a script. All its versions hold at most 512 MiB together. A version above this limit is not kept. Without timestamps, the frame time is unknown. The time from the submit until the work is seen done also holds the polling of the page, once a frame, thus it cannot judge a limit of a few milliseconds. Only 30 frames in a row over 250 ms of that time stop a program with `program.draw_over_time`, and the rules of late frames apply as in the app.

## Builder Pane

The palette of the builder pane has a group Pieces. It lists one row for each entry of `piece.list`, named `<name>@<version>`, with the source of the piece and its description. The pane refreshes the list when a piece command changes the workspace.

A person drags a piece row onto the middle of a scene panel, or focuses a scene panel and uses Add. The pane then sends `program.create` with the pin of the piece and that panel. The pane binds no input. A piece with inputs is refused with `piece.bad_binding`, and the message says that the program must be written in the spec text or by the agent. A drop that would make a new panel is refused with `program.bad_target`, because a piece makes no panel. A piece that only publishes needs no panel.

## Attention Arc Tower

The Attention Arc Tower answers one question. When and where does the model learn to copy, and is that what drops the loss? An agent builds it through `caliper-mcp` alone, on a managed run of a script that calls `caliper.watch(model, optimizer)` and `caliper.series`. The chart of the induction score of each head is the headline view. The tower of arcs is the overview, and the attention grid is the inspector of one head. No built-in primitive is in the frames of the tower and the grid. The scene panels `tower` and `pattern` hold programs only, and the chart and the loss curves are series panels.

The script `examples/python/arc_tower_train.py` trains an attention-only transformer with two layers of four heads, a width of 64 and a vocabulary of 64 words. Each sequence holds 32 random tokens, and a segment of 12 of them repeats. The loss stays near ln 64 until heads of layer 1 learn to copy, and then it drops at a bump, which comes between step 600 and step 1,000 on an Apple GPU. Row 0 of every batch is a fixed sentence that the loss leaves out. Thus the attention that a viewer shows is always that of the same sentence.

The watch publishes the attention weights of each layer as `live:blocks.<i>.attn.weights.act`, as rows of batch, head and query token against the key tokens. It publishes `live:token_loss.act`, the loss of each token, and one intervention point `blocks.<i>.attn#<h>` for each head. The in-context loss is the mean of the loss of row 0 over the second copy, which only a copy from the first occurrence predicts. After each optimizer step, the script publishes the induction score of each head as the series `induction_l<i>_h<h>`. The score is the mean weight of row 0 from a token of the second copy to the token after the first occurrence of the same token. The calls pass 0-d tensors on the device, thus a step does not wait for the GPU.

The pieces are in `demos/arc_tower/`. The piece `attention_arcs` scores each head for induction in its derive half and publishes `heads`, one score for each head. It also publishes the arcs, one for each attention weight above the threshold, with `from_token`, `to_token`, `layer`, `head`, `weight` and `score`. Its draw half draws the arcs instanced and additive with an ID for each arc, a glow target and a bloom pass, and it labels the tokens. Its item map holds the six fields of an arc. The piece `attention_grid` is the second graphic. It is fragment-only and has the camera `none`. It draws the weights of one head as flat cells in magma, with the words of the rows in a left margin. It leaves the half above the diagonal empty, because a token never attends to a later token. It outlines the cells that an induction head fills, and it lights the row and the column under the pointer. Its parameters follow the state `head` and the layer of the last pick in `tower`.

The agent starts the run, writes both pieces with `piece_check` and `piece_update`, and places them with `program_create`. It adds the loss series, the in-context loss, and a window source for each score series. The series panel `head_scores` shows one line for each head, with a legend. A toggle is bound to `intervene:blocks.1.attn#0`. In the layout, the left column holds the tower, the chart below it, and the in-context loss and the loss as tabs below the chart. The attention grid fills the right column. After the bump, `program_read` shows the head scores rise from below 0.2 at the first kept version to above 0.4. In the chart, the strongest line of layer 1 starts below 0.2 and passes 0.5, at the steps where the in-context loss falls. A pick of an arc with `selection_set` gives its layer, head and token pair, and the attention grid follows the head. The toggle raises the in-context loss at once. A fork from the first kept step with the same head ablated through `run_fork` never forms that head.

## Limits

These parts are not built, and the pieces do not depend on them.

- Custom blend factors, targets wider than eight bits, indirect draws, storage textures, multisampling and node materials.
- A derive half that reads the output of another derive half.
- Kept versions in the browser viewer, which draws the newest version of each resource.
- Resources as catalog sources, and recordings of resources.
- The derive half on a platform other than Metal.
- A preview of a piece from its sample.

Three points need care when a piece is written. In a fork, another head may form induction arcs in place of the ablated one. Without multisampling, an arc of one pixel aliases, thus a width of at least one pixel times the pixel ratio keeps it visible. The timestamp query is optional in WebGPU, thus a browser does not know the frame time, and it stops a draw only when its frames take far longer than the limit.

## Tests

[Testing](../development/testing.md) lists the tests of pieces. The test `tests/piece_path/driver.py` follows one piece from its command to a pick. The test `tests/draw_parity/` compares the draw of the app and of the browser module on one recorded frame. The test `caliper_programmable_graphics_acceptance` builds the Attention Arc Tower through `caliper-mcp`, and `python3 examples/python/arc_tower_check.py` runs the same steps and prints each check. Each refusal of this page has a test in `src/builder/programs/tests/test_pieces.cpp` or in the suite of the host that gives it.
