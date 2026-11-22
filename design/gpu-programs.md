# GPU Programs

This page defines the interfaces of GPU programs. An agent that uses only `caliper-mcp` writes a WGSL compute program over a live training tensor, and the program draws new geometry in a scene panel or new pixels in an image panel. The agent also changes how a scene entity draws with a material stage or a vertex stage. The same work adds the first intervention: a toggle that ablates a module of the model while the run trains.

The page states the spec additions, the eleven new commands, the program declaration, the sandbox, the cadence of a program, the wire path between the app and the script, the status of a program, the program events and the intervention commands. The design of the runtime in the script's process and its measurements are in `design/producer-runtime.md`. The conventions of commands, events and the effect `run` are in `design/instruments.md`.

## Status

The interfaces of this page are in the code: the command descriptors with handlers that answer `command.not_implemented`, the value types of the spec, the sandbox interface, the program hosts of the session, the C interface of the producer runtime, the stage interface of the scene compiler and the final signatures of the framework adapter. The features come next, in parallel work that fills these interfaces.

## Spec Additions

Programs and stages join spec version 1, as `design/instruments.md` planned. A program and a stage are spec content: WGSL source with a declaration. They have a diff, an undo step and a journal entry like each other edit. Spec version 1 gains three top-level keys:

- **`program`** is an array of program declarations, at most 16.
- **`stage`** is an array of stages, at most 32.
- **`provenance`** is an object that marks a spec from someone else.

The value types are in `src/builder/include/caliper_builder/spec/program.h`, and `Spec` holds them as `programs`, `stages` and `provenance`. The reader refuses the three keys with `spec.unknown_key` until the codec reads them. The argument schemas of `program.create` and `stage.set` in `src/builder/commands/command_table.cpp` are the reference for the JSON form. The schema of version 1 copies them as `$defs/program`, `$defs/stage` and `$defs/provenance`.

### Program Declaration

A program declaration holds the WGSL source and a declaration of its inputs, its output and its limits. A program has one compute pass with one dispatch. It never writes a tensor of the model.

| Key | Value |
|---|---|
| `id` | The program's id, unique among programs |
| `title` | An optional label |
| `wgsl` | The WGSL source, at most 64 KiB |
| `entry` | The compute entry point; the one compute entry point when absent |
| `inputs` | At most 8 inputs, each `{var, source}`: a WGSL variable and a live tensor `live:<name>` |
| `output` | One output of kind `geometry` or `pixels` |
| `limits` | `time_ms`, `memory_mb` and `every`, each optional |
| `workgroups` | Three workgroup counts; derived from the output when absent |

An input binds a tensor of the script by its name, read-only. The names are the parameter names of `model.named_parameters()`, such as `layers.0.weight`, a parameter's gradient as `<parameter>.grad`, and a module's last output as `<module>.act`. The WGSL declares each input as `var<storage, read>` with the name of `var`, in any group and binding.

The limits have these defaults and maximums:

| Limit | Default | Maximum |
|---|---|---|
| `time_ms`: GPU time of one pass | 4 | 50 |
| `memory_mb`: output buffers and bound tensors | 64 | 512 |
| `every`: the fewest steps between two passes | 1 | 10000 |

### Geometry Output

A geometry output is a scene batch on one entity of a scene panel. Its keys are `kind` (`geometry`), `panel` (a scene panel's id), `entity` (an entity path, `programs/<id>` when absent), `part` (`points` or `lines`), `max_count` (at most 1,000,000) and `arrays`. The optional `style` holds `colormap`, `vmin` and `vmax` for the values.

The WGSL declares each array of `arrays` as `var<storage, read_write>` with the array's name, and a counter `var<storage, read_write> count: atomic<u32>`. The runtime allocates each array for `max_count` items and sets `count` to zero before each pass. After the pass, the first `min(count, max_count)` items are valid.

| Array | WGSL Type | Items |
|---|---|---|
| `positions` | `array<f32>` | 3 floats for each item: x, y and z |
| `colors` | `array<u32>` | 1 for each item: RGBA8, packed with `pack4x8unorm` |
| `values` | `array<f32>` | 1 for each item, mapped through `style.colormap` |
| `radii` | `array<f32>` | 1 for each item; `points` only |

The array `positions` is necessary. For `points`, an item is a point. For `lines`, an item is a vertex, and each two vertices make one segment. The part `mesh` is reserved.

### Pixels Output

A pixels output is an image of a stated size and format. Its keys are `kind` (`pixels`), `width` and `height` (each at most 2048) and `format` (`rgba8unorm`, the only format now). The WGSL declares `var<storage, read_write> pixels: array<u32>` with `width × height` items in row order, each one RGBA8 packed with `pack4x8unorm`.

The script publishes the image as the live view `program.<id>`, of kind image. An image panel shows it through a source whose catalog id is `live:program.<id>`. Thus the output uses the image view, the shared-memory slot and the network snapshot as they are.

### Frame Uniform

A program may declare one uniform buffer with the name `caliper`. The runtime fills it before each pass with a struct of 32 bytes:

```wgsl
struct CaliperFrame {
    step: u32,        // the script's step
    pass_index: u32,  // the count of earlier passes of this program
    max_count: u32,   // geometry: max_count; pixels: 0
    width: u32,       // pixels: width; geometry: 0
    height: u32,      // pixels: height; geometry: 0
    time_s: f32,      // seconds since the program was installed
    every: u32,       // the cadence N of this pass
    pad: u32,
}
```

### Stage Declaration

A stage replaces one function of the draw of one scene entity in the viewer. A stage item holds `panel` (a scene panel's id), `entity` (an entity path), `kind` (`material` or `vertex`) and `wgsl` (at most 16 KiB). The triple of panel, entity and kind is unique. A panel removal also removes the stages of that panel in the same undo step.

The source of a stage holds functions and constants only. It declares no binding, no entry point and no directive. The viewer puts a fixed prelude before the source and its own entry points after it, and the entry points call the stage's function. Thus the scene keeps its camera, depth, pick pass and selection.

| Kind | Function |
|---|---|
| `material` | `fn material(in: MaterialIn) -> vec4<f32>` gives the color of a fragment |
| `vertex` | `fn vertex(in: VertexIn) -> VertexOut` gives the position and the radius of an item |

The prelude declares these structs:

```wgsl
struct MaterialIn { color: vec4<f32>, value: f32, item: u32, position: vec3<f32>, normal: vec3<f32>, time_s: f32, step: u32 }
struct VertexIn { position: vec3<f32>, value: f32, radius: f32, item: u32, time_s: f32, step: u32 }
struct VertexOut { position: vec3<f32>, radius: f32 }
```

The positions are in the coordinates of the scene. A vertex stage also runs in the pick pass, thus a moved item picks where it draws. The kind `compute` of earlier plans is a program with a geometry output.

### Provenance

The object `provenance` marks a spec from someone else, such as a shared or downloaded spec. Its keys are `foreign` (true), `from` (a free text that names the origin) and `reviewed`, a list of `{target, sha256}`. A target is `program:<id>` or `stage:<panel>/<entity>/<kind>`, and the hash is the SHA-256 of its WGSL.

No command writes `provenance`. The app writes it when it opens a spec from outside the workspace, and the edits of a command keep it. In a foreign spec, a program or a stage runs only when its hash is in `reviewed`. Else its state is `review`, and it does not run. The review card that adds a hash comes in a later task. A spec without `provenance` is the person's own, and its programs run at once.

## Commands

Eleven commands join the command table. Each one becomes an MCP tool with the dot replaced by `_`, such as `program_create`. The descriptors in `src/builder/commands/command_table.cpp` hold the argument schemas, the result schemas and one example for each command.

| Command | Effect | Mutates | Frame Thread | Function |
|---|---|---|---|---|
| `stage.set` | `edit` | Yes | Yes | Sets the material or vertex stage of one entity of a scene panel |
| `stage.remove` | `edit` | Yes | Yes | Removes one stage |
| `stage.check` | `read` | No | No | Compiles a stage or a program source and returns the compiler's messages |
| `program.create` | `edit` | Yes | Yes | Adds a program to the open spec |
| `program.patch` | `edit` | Yes | Yes | Changes a program with a JSON merge patch |
| `program.remove` | `edit` | Yes | Yes | Removes a program |
| `program.status` | `read` | No | Yes | Returns the state of each program and stage |
| `intervene.points` | `read` | No | Yes | Lists the intervention points of the attached model |
| `intervene.set` | `run` | No | Yes | Sets one intervention, applied at the next step |
| `intervene.clear` | `run` | No | Yes | Clears one point's interventions or all of them |
| `intervene.list` | `read` | No | Yes | Lists the interventions and their states |

The `events.wait` kinds gain `program`, and its event schema gains the program keys of Program Events.

### Stage Commands

The stage commands are edits of the open spec. They follow the rules of the panel commands: a dry run, a diff, review for the origins that have review on, one undo step and a journal entry. `apply_spec_edit()` gives them to `apply_program_edit()` in `src/builder/session/program_edit.cpp`.

`stage.set` takes `panel`, `entity`, `kind` and `wgsl`, and optionally `dry_run`. It replaces the stage of the same panel, entity and kind. Before the edit, it runs the sandbox checks of the editor on the stage. The result holds `panel`, `entity`, `kind`, `messages`, `warnings`, `diff` and `applied`. `stage.remove` takes `panel`, `entity` and `kind`.

`stage.check` takes `kind` (`material`, `vertex` or `compute`) and `wgsl`, and optionally `entry`. It changes nothing. It compiles the source as the viewer or the runtime does and returns `ok`, `messages`, `entry_points` and `bindings`. Each message has `severity`, `code`, `message`, `line` and `column`, with the line and column in the given source.

### Program Commands

`program.create` takes the keys of the program declaration and optionally `dry_run`. It checks the declaration and the source, then adds the program. The result holds `id`, `messages`, `warnings`, `diff` and `applied`. A program with errors is not added, and the error carries the first message.

`program.patch` takes `id` and a merge patch of `title`, `wgsl`, `entry`, `inputs`, `output`, `limits` and `workgroups`, and optionally `dry_run`. It checks the program again as `program.create` does. Its result holds the program after the patch. `program.remove` takes `id`.

`program.status` takes an optional `id`. It returns `programs` and `stages`, each a list of the status items of Program Status. It reads the state that the program hosts of the app report, and it answers with `host` set to `none` for an item that no host runs.

The person's own agent needs no review card for a program. The sandbox is always on, and the panel shows the source. The review settings of spec edits apply as for each other edit.

### Command Errors

Each error has a code and one plain sentence. The codes are constants in `src/builder/include/caliper_builder/programs/sandbox.h`.

| Code | Cause |
|---|---|
| `program.exists` | A program with this id is in the spec |
| `program.not_found` | The spec has no program with this id |
| `stage.not_found` | The panel has no stage of this kind on this entity |
| `stage.not_scene_panel` | The panel is not a scene panel |
| `program.bad_target` | The output names a panel that is not a scene panel |
| `program.too_large` | The source is above its size limit |
| `program.directive_not_allowed` | The source has an `enable`, `requires` or `diagnostic` directive |
| `program.compile_failed` | The compiler refused the source; the message has the line and the column |
| `program.no_entry` | No compute entry point, or not the named one |
| `program.binding_undeclared` | A binding of the source is not an input, an output array, `count`, `pixels` or `caliper` |
| `program.writes_input` | An input is not `var<storage, read>` |
| `program.output_mismatch` | An output binding has the wrong kind or access, or a necessary one is missing |
| `program.over_limits` | A count, a size, a workgroup or a limit is above its maximum |
| `program.compiler_unavailable` | The build has no WGSL compiler; a dry run cannot check the source |
| `stage.binding_not_allowed` | A stage declares a binding |
| `stage.entry_not_allowed` | A stage declares an entry point |
| `stage.compile_failed` | The viewer's composed shader does not compile; the line and the column are in the stage's source |

Without the compiler library, `program.create` and `stage.set` apply the edit with the warning `program.not_checked`. The hosts check each program and each stage again before a backend sees it, thus the editor's check gives early messages and is not the guard.

## Sandbox

The sandbox is automatic and always on. The same checks run in the editor before an edit, in the producer runtime before a program runs, and in the viewer before a stage draws. The interface is `src/builder/include/caliper_builder/programs/sandbox.h`: a declaration and the compiler's reflection go in, and a verdict comes out. The library `caliper_builder_programs` holds it, without dependencies, thus the app, the session and the native module of the script link the same code.

### Check Order

The checks run in this order, and the first failure stops the program with its code:

1. **Declaration.** The ids, the counts, the sizes and the limits are inside their maximums (`check_declaration()`).
2. **Provenance.** A program or a stage of a foreign spec has its hash in `reviewed`, else its state is `review`.
3. **Source.** The source is inside its size limit, and it has no directive (`check_source()`).
4. **Compile.** `caliper-wgsl` parses and validates the source (`compile_and_reflect()`). Naga refuses a store to a `var<storage, read>` binding.
5. **Entry point.** A program has the compute entry point, with at most 256 invocations in a workgroup. A stage has no entry point.
6. **Bindings.** Each binding of the reflection is declared, each input is a storage buffer with read access only, and each output has its kind and access (`check_program()` or `check_stage()`).
7. **Import.** The runtime imports each input in place on its device with the layer's `read_only` flag, and never copies a training tensor.
8. **Memory.** The output buffers and the bound tensors are inside `memory_mb`.
9. **Layer.** The frame validation of the graphics layer checks the bind groups and the dispatch. The layer refuses a read-only import in a `StorageReadWrite` binding.

A program has no file access, no network access and no access to a tensor that it does not declare. A failed program is an error of its panel, never of the training script or the app.

### Time Watchdog

The runtime measures the GPU time of each program's dispatch, with one submit for each due program. A program over `time_ms` on 3 passes in a row stops with `program.over_time`. A wait of a pass that lasts longer than 10 times the sum of the due programs' limits, and at least 100 ms, stops every program with `program.timeout`, and the runtime releases its imports. Training continues without the programs.

The viewer measures the GPU time of each frame of a scene panel with stages. When it is above 16.7 ms for 30 frames in a row, and above the same panel's time without its stages, the panel draws without its stages, and each one stops with `stage.over_time`.

### Host Errors

The hosts report these codes in a status item and in a `program` event. The constants are in `sandbox.h` beside the codes of the editor.

| Code | Cause |
|---|---|
| `program.unknown_input` | The script has no tensor of an input's name |
| `program.import_refused` | A tensor is on another device, is not contiguous, or its buffer is too short |
| `program.over_memory` | The outputs and the bound tensors are above `memory_mb` |
| `program.layer_refused` | The frame validation of the graphics layer refused the pass |
| `program.over_time` | The program went over `time_ms` on 3 passes in a row |
| `program.timeout` | A wait of a pass timed out, and every program stopped |
| `program.device_lost` | The device was lost, and every program stopped |
| `program.unavailable` | The script cannot load the runtime, such as a torch without DLPack export on MPS |
| `stage.over_time` | The panel's frame time went over its limit with the stages |

## Cadence

A program never runs on each step by default. It runs every N steps, and the runtime sets N from the measured cost of a pass against the measured step time. Thus training keeps at least 95 % of its speed, and the runtime uses at most 4 % of it, as `design/producer-runtime.md` states in Budget per Step.

The facade measures two times on the training thread. `T` is the median time between two `caliper.step()` calls over the last 32 steps without a pass. `C` is the median added time of the last 8 steps with a pass, from the start of the first sync point to the end of the third. The rate of the budget is then:

```text
N_budget = ceil(C / (0.04 × T))
N        = max(N_budget, every, 1)
```

All due programs share one pass step and its sync points, thus `C` is the cost of the pass with all of them. A program with a larger `every` runs on each pass step that is a multiple of its own `max(N, every)`, rounded up to a multiple of `N`. After a program is installed, the facade measures `T` for 8 steps and runs the first pass. It then computes N after each pass, and it changes N only when the new value differs by more than 25 %. N is at most 10000. The declaration can ask for a slower cadence with `every`, never for a faster one.

## Wire Path

A program runs in the process of the script, and its declaration goes there on the viewer connection that the app attaches. The outputs come back as messages that exist now. Only the declaration and the status need new messages.

| Direction | Message | Use |
|---|---|---|
| App to script | `PROGRAM` (0x0080), new | Installs, replaces or removes one program |
| Script to app | `PROGRAM_STATE` (0x0081), new | Answers a `PROGRAM`, and sends the status of each program |
| Script to app | `SCENE_UPDATE` (0x0070) | A geometry output, as a batch of the target scene |
| Script to app | `SNAPSHOT` (0x0030) of kind image | A pixels output, the live view `program.<id>` |
| Script to app | `VIEWS` (0x0020) | Announces the view `program.<id>` |
| App to script | `CONTROL` (0x0051) with op `intervene` | Sets or clears an intervention |
| Script to app | `CONTROLS` (0x0050) with the key `interventions` | The points and the interventions |

The data view request (`SUBSCRIBE` and `VIEW_REQUEST` with a `data_view`) does not carry a program, because a data view names a reduction of the catalog and holds no code.

### Program Messages

The capability `programs` guards both new messages. The script announces it in its `HELLO` only under `caliper.watch(model, optimizer)`, which also announces `controls`. The app sends a `PROGRAM` only to a script that announced it.

`PROGRAM` holds JSON metadata and one section. The metadata keys are `request_id`, `op` (`set` or `remove`) and `id`. For `set`, it also holds `declaration` (the program declaration without `wgsl`), `scene` (the scene name of the target panel, for a geometry output) and `sha256`. The section `wgsl` holds the source as UTF-8. A `set` replaces a program with the same id.

`PROGRAM_STATE` holds the metadata keys `request_id` (absent when the script sends it on its own), `step`, `step_ms` and `programs`, a list of status items. The script sends one after each `PROGRAM`, after each change of a state, and at most twice a second while programs run.

The security rule of the wire protocol says that no message carries code. WGSL is code for the GPU only. The runtime runs it in the sandbox, never on the CPU, and only in a script that announced `programs`. The text of the rule in `include/caliper/live_net.h` and the reference page gains this exception.

### Attachment

The app keeps the programs of the open spec and the programs of the attached script equal. The session calls `ProgramHost::reconcile()` after each change of the open spec, and the app's host compares the programs by id and hash. It sends `set` for a new or changed program and `remove` for a removed one. After an attachment, it sends each program again. Without an attached script, a program's host is `none` and its state is `waiting`.

### Output Route

The facade in the script gives each output to the publish path of a `caliper.scene` call or a `caliper.show` call, as `design/producer-runtime.md` states in Output Path. A geometry output goes to the batch of `entity` in the scene `scene`. A pixels output goes to the live view `program.<id>`. Picks, selection and recording then work on an output without change.

## Program Status

`program.status` and `PROGRAM_STATE` carry the same status item. The app adds `host`.

| Key | Value |
|---|---|
| `id` | The program's id, or the stage as `<panel>/<entity>/<kind>` |
| `kind` | `geometry`, `pixels`, `material` or `vertex` |
| `host` | `script`, `viewer` or `none` |
| `state` | `waiting`, `review`, `compiled`, `running`, `refused` or `stopped` |
| `every` | The cadence N; null for a stage or before the first pass |
| `every_min` | The declared `every`; null for a stage |
| `last_pass_ms` | The GPU time of the last pass, or null |
| `last_cost_ms` | The added time of the last pass step on the training thread, or null |
| `step_ms` | The median step time `T`, or null |
| `last_step` | The step of the last pass, or null |
| `runs` | The count of passes |
| `memory_bytes` | The bytes of the output buffers and the bound tensors |
| `memory_limit_bytes` | The budget from `memory_mb` |
| `error` | `{code, message, line, column}` of the last error, or null |
| `sha256` | The SHA-256 of the source the host runs |

The interfaces of the hosts are in `src/builder/include/caliper_builder/session/program_host.h`. The app gives the session a `ProgramHost` for the programs in the script and a `StageHost` for the stages in the viewer, as it gives its `RunControl`.

## Program Events

Programs add the event kind `program` to the ring of `events.wait`. The hosts push it. Its keys are `program` (the id of Program Status), `change`, `host`, `code`, `message`, `line` and `column`. The common key `panel` holds the target scene panel of a geometry output or a stage, else null.

| Change | When |
|---|---|
| `compiled` | A host compiled the program or the stage, and it runs from the next pass or frame |
| `refused` | A check refused it; `code`, `message`, `line` and `column` say why |
| `stopped` | The watchdog, the memory budget or a lost device stopped it |

## Producer Runtime Interface

The native module `caliper._producer` calls the C interface `src/host/producer/caliper_producer.h`, which is internal and not part of the public C API. The interface creates a runtime on the device of the first bound tensor, imports tensors read-only, compiles a program after the sandbox checks, runs a pass of the due programs and waits for it, gives each output buffer, returns the status of a program, and destroys the runtime. The library `caliper_producer` holds it.

The Python side has two files with final signatures. `python/caliper/_programs.py` is the facade that the wire path calls: `install()`, `remove()`, `status()` and `at_step()`. `python/caliper/_adapter.py` holds the framework adapter: `export()`, `framework_done()`, `before_write()`, `hooks()` and `probe()` for the runtime, and `points()` and `ablate()` for the interventions.

## Viewer Stage Interface

The scene compiler takes the stages through `src/host/scene/stages.h`. The spec runtime compiles each stage of a scene panel after the sandbox checks, with the prelude of `stage_prelude()` and the entry points of `stage_entries()`, and gives the result as a `StageProgram`. The compiler asks a `StageLookup` for each entity's program and uses it in place of the built-in program of the primitive, in the draw pass and in the pick pass.

## Interventions

The first intervention is `ablate`. It sets the output of a module to zero, or one head of an attention module where the module gives heads. It applies at a step boundary through the module hooks of the framework adapter. The names `scale`, `steer` and `patch` are reserved, and `intervene.set` refuses them with `intervene.op_reserved`.

### Intervention Points

Each module of `model.named_modules()` is a point, with its dotted name, such as `blocks.0.attn`. A module gives heads when it has an integer attribute `num_heads`, `n_heads` or `n_head`, and one child `nn.Linear` with the name `out_proj`, `o_proj`, `c_proj` or `proj` whose `in_features` divides by the head count. Each head is then a point `<module>#<h>`, such as `blocks.0.attn#3`. The ablation of a head sets that head's slice of the child's input to zero, with a forward pre-hook on the child.

### Intervention Commands

`intervene.points` takes an optional `match`, a glob over the point names. It returns `points`, each with `point`, `kind` (`module` or `head`), `module`, `head`, `type` (the module's class) and `shape` (the last output's shape, or null before the first step).

`intervene.set` takes `point` and `op`, and optionally `from_step` and `until_step`. It answers at once with `pending: true`, and the script applies it at its next step. A point holds at most one intervention of each op. `intervene.clear` takes `point`, or `all: true`. `intervene.list` returns `interventions`, each with `point`, `op`, `from_step`, `until_step`, `since_step` and `state` (`pending`, `active` or `ended`).

The commands act through five functions of `RunControl` in `run_control.h`: `interventions_supported()`, `intervention_points()`, `interventions()`, `intervene_set()` and `intervene_clear()`. Their defaults give no points and answer `command.not_available`. A `run` event of change `intervene` reports each applied change.

### Intervention Effect

`intervene.set` and `intervene.clear` have the effect `run`, with its card and its rules (`design/instruments.md`, Run Effect Class). The key of a rule is `intervene.set:<point>` or `intervene.clear:<point>`, thus a rule allows one point.

| Command | Card Text |
|---|---|
| `intervene.set` with `ablate` | Ablate `<point>` in `<script>` from the next step. |
| `intervene.clear` | Clear the interventions on `<point>` in `<script>` at the next step. |
| `intervene.clear` with `all` | Clear every intervention in `<script>` at the next step. |

### Toggle Control

A control of kind `toggle` binds `intervene:<point>`. On, it sets the ablation of the point, and off, it clears it. The person's toggle has the origin `ui` and runs at once. The toggle kind is built in this work, because the acceptance test needs it. It also binds `param:<name>` of a boolean parameter.

### Intervention Wire

The script announces the capability `interventions` under `caliper.watch(model, optimizer)`. Its `CONTROLS` message then holds the key `interventions` with `points` and `active`. The app sends `CONTROL` with op `intervene` and the key `intervene`, which holds `{set: {point, op, from_step, until_step}}` or `{clear: <point>}` or `{clear_all: true}`. The `CONTROL_REPLY` answers as for the other ops.

### Intervention Errors

| Code | Cause |
|---|---|
| `intervene.unknown_point` | The model has no point of this name |
| `intervene.op_reserved` | The op is `scale`, `steer` or `patch`, which come later |
| `intervene.bad_range` | `until_step` is not above `from_step`, or a step is in the past |
| `intervene.not_supported` | The attached script did not announce `interventions` |

The run command errors of `design/instruments.md` also apply, such as `run.not_attached`.

## Acceptance Test

The acceptance test drives `caliper-mcp` against the hidden app with `CALIPER_AGENT_OFFLINE=1`. Its check script is `examples/python/gpu_program_check.py`. The test does these steps in this order:

1. It records the SHA-256 list of the app bundle, `caliper-mcp` and the installed Python package.
2. It starts a managed training run with `run.start`, and the run's script has only `caliper.watch(model, optimizer)`.
3. It adds a geometry program over a live weight tensor, such as one segment for each row of a weight matrix in the space of its first three principal axes, which no primitive call draws. The capture shows it.
4. It adds a pixels program that writes an image of a tensor, and an image panel shows `live:program.<id>`.
5. It sets a material stage on the program's entity and sees the colors change in the capture.
6. It sends a program that declares an input `read_write`, and sees `program.writes_input` with no pass of that program.
7. It sends a program with a loop that goes over `time_ms`, and sees the event `stopped` with `program.over_time`.
8. It reads `program.status` and sees `every` above 1 for the medium model.
9. It measures training throughput as the median steps per second of five short runs with the same seed, with and without the programs, and checks at least 95 %.
10. It places a toggle bound to `intervene:<point>`, turns it on, and sees the loss change at the next steps.
11. It records the SHA-256 list again and checks that it did not change.

## Decisions

Each decision below has its reason in one line.

- **Stage storage.** Stages are the top-level array `stage`, because then only the stage commands edit their code past the compiler, and the node editor gets one node kind for each array.
- **Stage kinds.** `stage.set` takes `material` and `vertex` only, because compute over a training tensor has one path, a program in the script.
- **Stage source.** A stage holds functions only, and the viewer adds the prelude and the entry points, because the scene then keeps its camera, pick pass and selection.
- **Pixels buffer.** A pixels output is a packed `array<u32>`, not a storage texture, because the layer exports buffers through DLPack and fills an image from an RGBA8 buffer, but has no copy from a texture to a buffer.
- **Pixels view.** A pixels output is the live view `program.<id>`, because the image view, the shared-memory slot and the network snapshot then carry it unchanged.
- **Binding by name.** Inputs and outputs bind by their WGSL variable names in any group, because the reflection gives the names and the agent writes plain WGSL.
- **Flat arrays.** Positions are `array<f32>` with 3 floats for each item, because an array of `vec3<f32>` has a stride of 16 bytes and the scene message packs 12.
- **Two parts.** Geometry has `points` and `lines` now, because `mesh` needs a second counter for its indices.
- **New messages.** `PROGRAM` and `PROGRAM_STATE` are new under the capability `programs`, because `CONTROL` carries no code and no arguments, and a data view names a reduction.
- **Intervention wire.** Interventions use `CONTROL` and `CONTROLS` under the capability `interventions`, because an intervention applies at the same step boundary as a parameter.
- **Head names.** A head's point is `<module>#<h>`, because a torch module name never holds `#`, and the head rule covers `nn.MultiheadAttention` and the usual GPT layouts.
- **Toggle now.** The toggle control is built with the interventions, because the acceptance test needs it. The picker and the text control stay for the interface layer.
- **Sandbox library.** The sandbox is the builder module `caliper_builder_programs` without dependencies, because the builder configures on its own and all three hosts must run the same checks.
- **Unchecked edits.** Without the compiler library, an edit applies with `program.not_checked`, because the hosts check again before any run.
- **One submit each.** The runtime submits each due program on its own, because the layer gives the GPU time of a submit, and the watchdog needs each program's time.
- **Shared cadence.** All programs share the pass steps of one N, because the sync points cost more than a pass, and a shared step pays them once.
- **Version 1.** Programs join spec version 1, because `design/instruments.md` planned them there before version 1 reaches `main`.
- **Provenance by the app.** No command writes `provenance`, because a mark that an agent can clear protects nothing.
- **Internal C interface.** The runtime's C header is in `src/host/producer`, because the native module is its only caller.
- **Schema with the codec.** The schema of version 1 gains its three keys with the codec, because the reader refuses the keys until the codec reads them.
- **Wire constants with the wire.** The constants of the new messages go into `include/caliper/live_net.h`, the codec of Caliper LIB and the reference page together, because a test binds the three.

## Open Points

- **Mesh output.** The part `mesh` needs a second counter for indices. It comes with the multi-pass programs.
- **Browser stages.** The browser gives the WGSL of a stage to WebGPU directly, and the prelude must match. This comes with browser parity.
- **Deferred wait.** The producer runtime keeps the wait at once, as `design/producer-runtime.md` decides.
- **Review card.** The card that adds a hash to `provenance.reviewed` comes with saved instruments.
