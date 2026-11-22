# GPU Programs and Stages

An agent that uses `caliper-mcp` can write a WGSL program that runs on the GPU over a live tensor of a training script. The program draws new geometry in a scene panel or new pixels in an image panel. The agent can also change how one entity of a scene panel draws, with a material stage or a vertex stage. The script runs a program in its own process, on the tensor in place, and the program never writes the tensor. A program runs only in a build of Caliper that has the WGSL compiler library, which a build from source gets with `CALIPER_WGSL_FROM_SOURCE=ON`. [Run Control and Events](run-control-and-events.md) describes the intervention tools and the toggle control that came with this work.

## Tools

Each command is one MCP tool with the first dot of its name replaced by `_`, such as `program_create`. The first four commands edit the open spec and give a diff, one undo step and a journal entry, as the panel commands do.

| Command | Arguments | Function |
|---|---|---|
| `program.create` | The keys of a program declaration, `dry_run` | Adds a program to the open spec |
| `program.patch` | `id`, a merge patch, `dry_run` | Changes a program and checks it again |
| `program.remove` | `id` | Removes a program |
| `stage.set` | `panel`, `entity`, `kind`, `wgsl`, `dry_run` | Sets the material or vertex stage of one entity of a scene panel |
| `stage.remove` | `panel`, `entity`, `kind` | Removes one stage |
| `stage.check` | `kind`, `wgsl`, `entry` | Compiles a source and returns the messages of the compiler, and changes nothing |
| `program.status` | `id` | Returns the state of each program and stage |

The app sends each program of the open spec to the script that it is attached to. It sends them again after each new attachment. Without an attached script, the state of a program is `waiting`. The person's own agent needs no review card for a program, because the sandbox is always on and the panel shows the source. The review settings of the other spec edits apply to these edits.

## Programs

A program has one compute pass with one dispatch. Its declaration holds the WGSL source, its inputs, one output and its limits.

| Key | Meaning |
|---|---|
| `id` | The id of the program, at most 55 characters |
| `wgsl` | The WGSL source, at most 64 KiB |
| `entry` | The compute entry point. It is the one compute entry point when absent, and it is never named `main`. |
| `inputs` | At most 8 pairs of `var` and `source`, where `source` is a live tensor `live:<name>` |
| `output` | One output of kind `geometry` or `pixels` |
| `limits` | `time_ms`, `memory_mb` and `every`, each optional |

An input names a tensor of the script. The name is a parameter name of `model.named_parameters()`, such as `0.weight`. A parameter name with `.grad` added is its gradient, and a module name with `.act` added is the last output of the module. The program declares each input as `var<storage, read>`. A gradient or an output that does not exist yet keeps the program in the state `waiting`.

A geometry output draws as one entity of a scene panel. It has the keys `panel`, `entity` (`programs/<id>` when absent), `part` (`points` or `lines`), `max_count` (at most 1,000,000), `arrays` and an optional `style` with `colormap`, `vmin` and `vmax`. The program declares each array of `arrays` as `var<storage, read_write>`, and also a counter `count: atomic<u32>`. The runtime sets the counter to zero before each pass, and the first `count` items of each array are valid after it. The array `positions` is necessary, with three floats for each item. The arrays `colors` (one packed RGBA8 value for each item), `values` (one float for each item) and `radii` (points only) are optional.

A pixels output has the keys `width` and `height` (each at most 2048) and `format` (`rgba8unorm`). The program declares `pixels: array<u32>` with one packed RGBA8 value for each pixel in row order. The script publishes the image as the live view `program.<id>`, and an image panel shows it through a source whose catalog id is `live:program.<id>`. Because of this name, a program id has at most 55 characters.

A program can also declare one uniform buffer with the name `caliper`. The runtime fills it before each pass with the step, the count of earlier passes, the sizes of the output, the seconds since the install, and the cadence of the pass.

| Limit | Default | Maximum |
|---|---|---|
| `time_ms`, the GPU time of one pass | 4 | 50 |
| `memory_mb`, the output buffers and the bound tensors | 64 | 512 |
| `every`, the fewest steps between two passes | 1 | 10000 |

## Stages

A stage replaces one function in the draw of one entity of a scene panel. The panel, the entity and the kind identify a stage, and the source holds at most 16 KiB. The source holds functions and constants only. It declares no binding, no entry point and no directive. The viewer puts a fixed prelude before the source and adds its own entry points. Thus the scene keeps its camera, its depth, its pick pass and its selection.

| Kind | Function |
|---|---|
| `material` | `fn material(in: MaterialIn) -> vec4<f32>` gives the color of a fragment |
| `vertex` | `fn vertex(in: VertexIn) -> VertexOut` gives the position and the radius of an item |

`MaterialIn` has `color`, `value`, `item`, `position`, `normal`, `time_s` and `step`. `VertexIn` has `position`, `value`, `radius`, `item`, `time_s` and `step`, and `VertexOut` has `position` and `radius`. A vertex stage also runs in the pick pass, thus a moved item picks where it draws. The id of a stage in `program.status` is `<panel>/<entity>/<kind>`. A refused or stopped stage shows one line over its scene panel with its code, line, column and message.

## Sandbox Rules

The sandbox is always on. The same checks run in the editor before an edit, in the script before a program runs, and in the viewer before a stage draws. A failed program is an error of its panel and never of the training script or the app. The checks run in this order, and the first failure stops the program with its code:

1. The ids, the counts, the sizes and the limits are inside their maximums.
2. The source is inside its size limit and has no `enable`, `requires` or `diagnostic` directive.
3. The compiler parses and validates the source. It refuses a store to a `var<storage, read>` binding.
4. A program has its compute entry point, with at most 256 invocations in a workgroup. A stage has no entry point.
5. Each binding is declared, each input is a read-only storage buffer, and each output has its kind and its access.
6. The script imports each input in place on its device and never copies a training tensor. The import is read-only.
7. The output buffers and the bound tensors are inside `memory_mb`.
8. The frame validation of the graphics layer checks the bind groups and the dispatch.

A program has no file access, no network access and no access to a tensor that it does not declare. The time watchdog measures the GPU time of each pass. A program over `time_ms` on three passes in a row stops with `program.over_time`, and training goes on. A wait of a pass that lasts longer than ten times the sum of the limits of the due programs, and at least 100 ms, stops every program with `program.timeout`. The viewer measures the frame time of a scene panel with stages. When it stays above 16.7 ms for 30 frames and above the time of the panel without its stages, the panel draws without them, and each stage stops with `stage.over_time`.

## Errors

Each error has a code and one plain sentence. A message that comes from the compiler also has a line and a column in the source that the agent sent.

| Code | Cause |
|---|---|
| `program.exists`, `program.not_found`, `stage.not_found` | The id or the stage is already there, or is not there |
| `stage.not_scene_panel`, `program.bad_target` | The panel is not a scene panel, or the spec has no such panel |
| `program.too_large`, `program.over_limits` | A size, a count or a limit is above its maximum |
| `program.directive_not_allowed` | The source has a directive |
| `program.compile_failed`, `stage.compile_failed` | The compiler refused the source, or the entry point is named `main` |
| `program.no_entry` | The source has no compute entry point, or not the named one |
| `program.binding_undeclared`, `program.output_mismatch` | A binding is not declared, or an output has the wrong kind or access |
| `program.writes_input` | An input is not `var<storage, read>` |
| `stage.binding_not_allowed`, `stage.entry_not_allowed` | A stage declares a binding or an entry point |
| `program.compiler_unavailable` | The build has no WGSL compiler library |

Without the compiler library, `program.create` and `stage.set` apply the edit with the warning `program.not_checked`, and no program runs. A host reports these codes in the status item and in a `program` event: `program.unknown_input`, `program.import_refused`, `program.over_memory`, `program.layer_refused`, `program.over_time`, `program.over_budget`, `program.timeout`, `program.device_lost`, `program.unavailable` and `stage.over_time`.

## Program Status

`program.status` returns `programs` and `stages`, and each item has the same keys. The key `host` is `script` for a program, `viewer` for a stage, and `none` for an item that no host runs.

| Key | Value |
|---|---|
| `id`, `kind`, `host`, `state` | The id, the kind (`geometry`, `pixels`, `material` or `vertex`), the host and the state (`waiting`, `review`, `compiled`, `running`, `refused` or `stopped`) |
| `every`, `every_min` | The steps between two passes of the program, which is the shared cadence N or a multiple of N for a larger declared `every`, and the `every` of the declaration |
| `last_pass_ms`, `last_cost_ms`, `step_ms` | The GPU time of the last pass, the overhead `D` of a pass, which is the time that a pass adds to the run's wall clock and is shared by all programs, and the base step time `b`, which is the time of a training step far from a pass |
| `overhead_share`, `passes_per_s` | The measured share `D / (N * b)` of the training time that the passes take, and the passes of the program a second on the run's wall clock over its last 8 passes. Both are null for a stage and before they are measured, and `passes_per_s` is also null while the program does not run |
| `last_step`, `runs` | The step of the last pass, and the count of passes |
| `memory_bytes`, `memory_limit_bytes` | The memory in use and its budget |
| `error`, `sha256` | The last error as `code`, `message`, `line` and `column`, and the SHA-256 of the source that the host runs |

The state `review` belongs to a spec from someone else. The object `provenance` of the spec marks such a spec, and a program or a stage in it runs only when its SHA-256 is in the list `reviewed`. No command writes `provenance`. The review card that adds a hash is not built yet.

## Program Events

The tool `events_wait` takes the kind `program`. The keys of the event are `program` (the id of the status item), `change`, `host`, `code`, `message`, `line` and `column`, and the common key `panel` holds the target panel. The change `compiled` means that a host compiled the program or the stage. The change `refused` means that a check refused it, and the change `stopped` means that the watchdog, the memory budget, the budget of the cadence or a lost device stopped it. An agent waits for the event instead of a fixed time.

## Cadence

A program does not run on each step. It runs every N steps, and the script sets N by feedback from the time that the passes add to the run's wall clock. The aim is that the passes take at most 4 % of the training time. The rest of a 5 % loss stays for the hooks of `caliper.watch(model, optimizer)` and the publish path.

While a program is installed, the script records a `torch.mps.Event` at each step. The record gives the queued work of torch to the GPU without a wait. The script reads with `query()` which marks completed, and it takes the completion times from these reads. A step without a pass never waits.

From the completion times, it measures two times for each cycle between two passes. The base step time `b` is the median of the last 9 cycles. The overhead `D` is the mean of the last 8 cycles without the largest and the smallest. It is the time that a pass adds to the run's wall clock.

Then N follows from this rule:

```text
N_want = ceil(D / (0.04 * b))
N_max  = max(3, min(10000, ceil(10 s / b)))
target = min(max(N_want, 3), N_max)
```

N moves up by at most a factor of 2 for each update. It moves down by at most a factor of 1.5, and only when it is more than 25 % above the target. N is at least 3, and it is 3 before the first measured pass. The first pass of a program runs 8 steps after its install.

All programs share the pass steps, and one pass runs every due program. A declared `every` asks for a slower cadence and never for a faster one. When `N_want` stays above `N_max` for 3 updates in a row, and the smallest declared `every` is below `N_want` too, every program stops with `program.over_budget`. Training goes on, and the next install measures again. A loop that already waits for the GPU at each step, for example with `loss.item()`, gets a mark only every 8 steps, before a pass and at the pass.

`program.status` shows N as `every`, `D` as `last_cost_ms`, `b` as `step_ms`, the share of the training time as `overhead_share` and the rate of passes as `passes_per_s`.

With one geometry program under `caliper.watch(model, optimizer)`, a small model with steps of about 0.4 ms kept 95.9 % of its speed in a quiet run. A medium model with steps of about 11 ms kept 96.8 %. Loops that read the loss at each step kept 97.4 % and 98.7 %.

On the small model, the marks themselves cost about 1 % to 3 % of its short steps, thus its loss is larger than the measured share. On a loaded machine, the small model kept 93.6 % to 94.4 %. Each number is the median of five runs on one Apple machine with torch on MPS. Thus a model with steps below 1 ms can keep less than 95 % of its speed when the machine is busy.
