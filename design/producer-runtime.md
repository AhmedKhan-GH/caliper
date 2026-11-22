# Producer Runtime

This page plans the producer runtime. The producer runtime is the graphics layer without a window, loaded as a native module into the process of a PyTorch training script. Caliper's own WGSL compute then reads training tensors in place, and only the results travel to a viewer.

A design spike measured the plan on one machine before the work starts: an Apple M5 Max with macOS, Metal and the MPS backend of torch. The spike code is not part of the repository. It linked a small library against the graphics layer (`src/host/gfx/`) and called it from Python. The measurements and their method are in [Measurements](#measurements).

## Design

### Components

The runtime has three parts, and each part has one owner:

- **Native module.** `caliper._producer` is a nanobind module, built as `caliper._native` is built, that links the graphics layer with the Metal backend (the Vulkan backend on Windows and Linux) and the Naga library of the shader compiler. It holds one `caliper_gfx::Backend`, the imports of the bound tensors, the programs and their output buffers.
- **Framework adapter.** `caliper._adapters.torch` is the only part that knows torch. It gives a tensor as a DLPack capsule, it gives the two sync points, and it installs the hooks. A later framework needs only a new adapter.
- **Python facade.** The facade belongs to `caliper.watch(model, optimizer)`. It runs the programs at `caliper.step()`, keeps the budget, and gives each output to the publish path of a `caliper.scene` call.

### Device and Queue

The runtime makes its backend on the `MTLDevice` of the first tensor that a spec binds. The device comes from the tensor's own buffer (`id<MTLBuffer>.device`), as `caliper._live` already does for the shared-memory segments. Thus the runtime and torch always use one device. The spike found the same registry ID for torch's device and for `MTLCreateSystemDefaultDevice()`. Thus an in-process viewer on the default device also shares that device.

The runtime makes its own `MTLCommandQueue`, because torch does not give its queue to Python. The torch C++ API gives the queue (`at::mps::getCurrentMPSStream()`), but a module that uses it must be built for each torch version. The runtime stays free of the torch ABI, and the price is the CPU sync points below.

Torch allocates its MPS buffers in shared storage with untracked hazards (`MTLHazardTrackingModeUntracked`, measured). Thus Metal orders no access between torch and Caliper, on two queues or on one queue. Each order comes from an explicit sync point.

Both sides allocate from one unified memory and one device. An import allocates nothing: the spike measured no change of `MTLDevice.currentAllocatedSize` after an import. The runtime allocates only its output buffers and its uniform ring. These allocations show in `torch.mps.driver_allocated_memory()`, which is the device-wide counter, but not in `torch.mps.current_allocated_memory()`, which counts torch's allocator only. The memory budget of the sandbox counts each output buffer, and it also counts each bound tensor that torch would otherwise free, because the import keeps that storage alive.

### Framework Adapter Interface

The adapter has these functions. The spike used the first two in the form shown, and `before_write` only for the deferred wait of the open decisions.

```python
class TorchAdapter:
    def export(self, tensor) -> "PyCapsule":
        """tensor.__dlpack__(): kDLMetal on MPS, data is the storage's id<MTLBuffer>,
        byte_offset is storage_offset * itemsize. Refuses a tensor that is not contiguous."""

    def framework_done(self) -> None:
        """torch's work on every bound tensor is complete:
        torch.mps.Event().record(), then .synchronize() (CPU wait). Off MPS only."""

    def step_marks(self) -> "StepMarks | None":
        """A ring of torch.mps.Event marks: record(step) gives the queued work to the GPU
        without a wait, poll() returns the marks that completed, wait(step) is sync point 1."""

    def before_write(self, callback) -> None:
        """Calls callback before the framework writes a bound tensor again:
        optimizer.register_step_pre_hook for parameters."""

    def hooks(self, model, optimizer, on_outputs, on_grads, on_step) -> None: ...
    def probe(self, model, inputs) -> "PyCapsule": ...                 # for a later probe feature
    def state(self, model, optimizer) -> bytes: ...                    # for a later state feature
```

The native module has this interface. Each call runs on the thread that calls `caliper.step()`.

```python
class Runtime:
    def __init__(self, capsule): ...                         # device from the tensor, own queue
    def bind(self, name, capsule) -> int: ...                # in-place import, read-only, never a copy
    def program(self, wgsl, declaration) -> int: ...         # Naga, then the read-only rule, then a pipeline
    def run(self, program, params) -> int: ...               # one submit; returns its serial
    def wait(self, serial, timeout_ms) -> bool: ...          # Caliper's work complete
    def output(self, program, name) -> "PyCapsule": ...      # kDLMetal capsule over the output buffer
    def stats(self) -> dict: ...                             # GPU time per program, imports, bytes copied
```

### Threads

The runtime adds no thread. The graphics layer is used from one thread (decision 17 of `src/host/gfx/gfx.h`), and that thread is the training thread, at `caliper.step()`. Metal reports completion on its own threads, and the layer only records it there. The publish path keeps its network thread and its snapshot ring as they are.

### Sync Points

Each step that runs programs has three sync points, in this order:

1. **Framework work complete.** The facade waits until the step's mark completed. The mark is a `torch.mps.Event` that the facade records at the checkpoint, before `optimizer.step()`, and the facade looks at it and at the earlier marks with `query()` until it completed. Torch's writes to the bound tensors are then complete.
2. **Caliper's pass.** The runtime submits one frame with the dispatches of the due programs on its own queue.
3. **Caliper's work complete.** The runtime waits for the serial of that frame at once, before `caliper.step()` returns. Torch then writes no bound tensor while the pass reads it.

A step without a due program has no sync point at all. It only records a mark, which gives torch's queued work to the GPU without a wait. The spike also measured a deferred wait, in the optimizer step pre-hook of the next step, which lets the pass run while torch runs the next forward and backward passes. It helped the medium model and hurt the small model, thus it is an open decision. The import holds the capsule, and the capsule keeps the storage alive. Thus torch cannot give a bound storage to another tensor while the runtime reads it.

### Read-Only Rule

A program can never write a training tensor. Three checks enforce this, and the first two run before any backend sees the program:

1. **WGSL validation.** Naga refuses a store to a `var<storage, read>` binding. The spike's program with `w[base] = 0.0;` failed with "writing to this location is not permitted".
2. **Binder rule.** The runtime refuses a program whose declaration binds a training tensor to a binding that is not `var<storage, read>` (`BindingType::Storage` in the reflection). The spike refused a program that declares the tensor binding `read_write`, before `create_program`.
3. **Layer rule.** `ExternalBuffer` gets a `read_only` field. The core refuses a bind group that puts a read-only import in a `StorageReadWrite` binding. The spike showed that the core accepts such a bind group today. Thus this field is part of the work. An import already has no `kCopyDst` usage, and `write_buffer` already refuses a buffer without it.

Naga's MSL also declares the binding `const device`, and the Metal compiler refuses a write to it.

### Budget per Step

Check 5 of the specification allows 5 % of training throughput for the whole instrument. The runtime gets at most 4 % of it, and the rest stays for the hooks and the publish path. The runtime measures the time of each step between two `caliper.step()` calls and the cost of its own sync points and passes. It then runs the programs every N steps, with the smallest N that keeps its cost under 4 % of the step time. A program over its time limit of the sandbox is skipped for the next steps, and the view says why.

The built facade sets N by feedback from the measured overhead of a pass, as Cadence in `design/gpu-programs.md` states, because the time between two `caliper.step()` calls is often only the time to queue a step. Cadence Measurement below gives the reasons.

The measured costs set N. On this machine, a step with all three sync points and the pass added 0.42 ms on the small model and 0.90 ms on the medium model at each step. It added 0.8 ms to 1.6 ms when it ran every 10 or 20 steps, probably because torch's queue then held more work at the wait. The first sync point alone added 0.23 ms, which is 23.6 % of the throughput of the small model with its step of 0.75 ms. Thus a program over the weights of the small model runs about every 27 steps, and the medium model, with its step of 12.2 ms, allows a program every 2 to 4 steps. With a pass every 20 steps, the small model kept 94.6 % and the medium model 99.6 % of its throughput. The measured loops never waited for the GPU. A loop that already waits at each step, for example with `loss.item()`, pays less for the first sync point.

### Cadence Measurement

Torch's MPS queue gives its work to the GPU only at a sync point, such as `torch.mps.synchronize()`, `loss.item()` or the record of a `torch.mps.Event`. Thus a loop without a sync point queues its steps on the CPU, and the GPU idles until the end. The spike's small model queued 3,000 steps in 0.91 s, at 0.30 ms a step, and the GPU then ran them in 1.26 s more, at 0.42 ms a step. The loop gave 1,380 steps per second. With an event record at each step, which gives the queued work to the GPU without a wait, the same loop gave 2,387 steps per second.

Under `caliper.watch(model, optimizer)`, the small model gave 2,363 to 2,372 steps per second, because the hooks of the watch give the queue to the GPU at times. A CPU wait every 40 steps then gave only 1,429 steps per second. The GPU idled while the training thread queued the 40 steps, and the thread idled while the GPU ran them. With a record at each step, the same wait gave 2,296 steps per second, which is 97.0 %. Thus the loss of a pass came from the queue that torch held back, not from the wait itself. The facade therefore records a mark at each step while a program is installed.

The training thread can run ahead of the GPU without a bound. Without a record, it was 1.26 s ahead of the GPU at the end of the small model's loop. Thus the time between two `caliper.step()` calls is often only the time to queue a step. With a record at each step, Metal's limit of command buffers in flight held the lead to 26 ms on the small model and to 0.70 s on the medium model. The facade therefore times the steps by the completion of their marks, as Cadence in `design/gpu-programs.md` states.

A mark's completion shows only through `query()`, at the next look. The `elapsed_time()` of timing events would give exact intervals, but it waits for the device. It took 109 ms with 20 steps of the medium model in the queue. Thus the facade reads plain events with `query()` at each step, and it waits only at a pass. That wait looks at the marks until the step's mark completed, thus each mark that completes during the wait gets an exact time.

In a loop that reads `loss.item()` at each step, torch's queue is empty at each checkpoint. A record there adds a command buffer of its own, which the GPU runs before the next step's work. On the small model with steps of 0.94 ms, this cost 4.2 %: 1,020 instead of 1,065 steps per second, in the median of three runs. A record every 8 steps cost 1.2 %. Thus the facade marks sparsely in such a loop.

A sync on the GPU would also remove the wait at a pass. The pass would wait on the GPU for an event that torch's queue signals after the step's work, and torch's next work would wait for an event that the pass signals. The stream-ordered handoff of the tensor bridge does the first half with a command buffer on torch's queue. Torch's C++ API gives that queue through `torch::mps::get_command_buffer()` and `get_dispatch_queue()`. The second half also needs the queue, because torch's buffers have untracked hazards, and Metal orders nothing between two queues. Python has no access to the queue, and a module that uses torch's C++ API must be built for each torch version, which Own Command Queue in Decisions excludes. `torch.mps.compile_shader()` runs Metal source on torch's own queue, and torch then orders the pass in both directions. But the pass then runs outside the graphics layer, its read-only rule and its GPU time of each submit. With the marks, a pass costs about 1 ms on the small model, which a pass every 60 to 70 steps keeps under 4 %. Thus both routes stay in Open Decisions.

### Output Path

A program's output buffers are the runtime's own, in shared storage, and the declaration names them with a geometry schema. After Caliper's work is complete, the runtime reads the valid count from the header buffer and gives each output to torch as a `kDLMetal` capsule over the buffer, as `caliper._native` already does for its `SharedBuffer`. The facade then calls the same publish path as a `caliper.scene` call. Thus the three routes reuse existing calls:

| Route | Copies | Calls Reused |
|---|---|---|
| In process | Zero | The in-process renderer imports the output's `id<MTLBuffer>` with `import_buffer`, as the tensor bridge's `import_allocation` does for `CALIPER_ALLOC_HANDLE_MTLBUFFER`. A scene batch has no in-process route yet (`python/caliper/_scene.py` sends to the network only), thus this route is an M2 task. |
| Shared-memory slot | One, on the GPU | `SnapshotRing.take` and the plan's `copy_` on torch's stream into the region that `Segment.region` gives as a `kDLMetal` capsule, then the `torch.mps.Event` and `poll` of `python/caliper/_ring.py`. Scenes through shared memory are planned in Scenes and not built. Until then, a scene output takes the network route on one machine too. |
| Network | One, to host memory | `_net_ring.py` stages the tensor into pinned memory with `copy_(non_blocking=True)` and a `torch.mps.Event`. `_scene.py`'s `Batch`, `_net_scene.py`'s `finish` and the snapshot copy of `python/src/_netio.cpp` then send it. |

On unified memory the output buffer is already readable by the CPU. Thus the network route can skip the pinned stage later, and `mapped()` gives the bytes after Caliper's work is complete.

### On-Demand Load

`import caliper` loads no native module today. It loads five Python modules in 9 ms to 12 ms, and the maximum resident set grows from 16.5 MB to 23.3 MB. The runtime keeps this unchanged. `caliper._producer` loads when a spec first binds a stage, a program or a WGSL operator to a tensor of the script, through the lazy `__getattr__` that already loads `caliper._native`. A script that only streams tensors loads nothing new, and a test checks that `caliper._producer` is not in `sys.modules` after `import caliper` and after `caliper.watch`.

The spike measured the first use, from the load of its library to the end of the first pass. It cost 12 ms to 13 ms and 1.3 MiB of resident memory in later processes, probably because Metal's shader cache then held the program. It cost 183 ms in the first process on this machine, and the Metal device and the pipeline took most of that time. Without a use, the spike loaded no module and added no memory and no thread.

### Failure Behavior

- **No DLPack export.** Torch 2.6.0 refuses `__dlpack__` on an MPS tensor ("Cannot pack tensors on mps:0"). The runtime refuses to load on such a version, the panel says why, and the shipped views keep their paths that copy.
- **Refused import.** A tensor on another device, a tensor that is not contiguous, or a buffer shorter than the tensor refuses the bind. The runtime never copies a training tensor in place of an import.
- **Refused program.** A Naga error, a breach of the read-only rule or a failed pipeline is an error of the panel, never of the training script. The runtime catches each one at the native boundary.
- **Over budget.** The runtime lowers the rate first, then skips the program, and the view shows the reason.
- **Lost device or a timeout of the wait.** The runtime stops all programs, releases its imports, and reports the reason. Training continues without the instrument.

## Measurements

### Method

The spike translated one WGSL program with the pinned Naga driver (`tools/shaders/naga-driver`, Naga 30.0.2), because the Naga library of the shader compiler was not ready. The program computes the norm of each row of one bound weight, one invocation per row, and writes one float per row. A Python script gave the tensor through `__dlpack__()`, and the spike library made the backend with `make_metal_backend` on the tensor's device and imported the buffer with `import_buffer`.

The training loop is a multilayer perceptron with AdamW and mean squared error, on fixed random batches with seed 0, in two sizes. The small model has layers 784, 1024, 1024 and 10 with a batch of 256, and 1,500 timed steps. The medium model has layers 2048, 4096, 4096 and 2048 with a batch of 512, and 250 timed steps. Each run has 30 steps of warm-up. Each number is the median steps per second of five runs, one process per run, with the modes in a different order in each repetition. The first series ran the modes with a pass every N steps with a sync point on each step, which was a fault of the spike. Thus a second series ran them again, with the sync points only on the pass steps, beside its own base. A run took less than 6 seconds, and the peak memory footprint was 0.47 GB for the small model and 1.6 GB for the medium model under `/usr/bin/time -l`. The runs held one heavy slot while the other slots were free, and the 1-minute load average before each run is in the table.

The modes are:

- **Base.** No instrument.
- **Sync.** Only the first sync point, at each step.
- **Pass each step.** All three sync points at each step, with the wait at once after the pass.
- **Pass every 10 steps.** The same every tenth step.
- **Deferred wait.** All three sync points at each step, with the wait in the next optimizer step pre-hook.

### Results

| Model and Mode | Steps per Second | Share of Base | Added Time per Step | 1-Minute Load |
|---|---|---|---|---|
| Small, base of the first series | 1,330.3 | 100 % | None | 2.27 to 3.47 |
| Small, sync each step | 1,016.5 | 76.4 % | 0.23 ms | 2.27 to 3.47 |
| Small, pass each step | 852.5 | 64.1 % | 0.42 ms | 2.49 to 3.51 |
| Small, deferred wait each step | 546.8 | 41.1 % | 1.08 ms | 2.69 to 3.42 |
| Small, base of the second series | 1,310.6 | 100 % | None | 2.90 to 3.84 |
| Small, pass every 10 steps | 1,190.9 | 90.9 % | 0.08 ms | 2.70 to 3.84 |
| Small, pass every 20 steps | 1,239.9 | 94.6 % | 0.04 ms | 2.70 to 3.77 |
| Medium, base of the first series | 83.42 | 100 % | None | 2.65 to 4.29 |
| Medium, sync each step | 81.61 | 97.8 % | 0.27 ms | 2.68 to 4.29 |
| Medium, pass each step | 77.61 | 93.0 % | 0.90 ms | 2.80 to 4.03 |
| Medium, deferred wait each step | 79.91 | 95.8 % | 0.53 ms | 2.74 to 3.54 |
| Medium, base of the second series | 81.91 | 100 % | None | 2.74 to 3.77 |
| Medium, pass every 10 steps | 80.88 | 98.7 % | 0.16 ms | 2.92 to 3.87 |
| Medium, pass every 20 steps | 81.61 | 99.6 % | 0.04 ms | 2.85 to 3.72 |

The share compares each median with the base of its own series. The CPU time of the pass, from encode to the end of the wait, had a median of 0.24 ms on the small model and 0.58 ms on the medium model, which reads 64 MiB.

The functional checks passed on torch 2.14.1 and 2.15.0:

- The import took the in-place route with 0 bytes copied, and it did not change `MTLDevice.currentAllocatedSize`.
- The norms matched torch's `w.norm(dim=1)` within 5.4e-7. After an optimizer step changed the weight in place, the same import gave the new norms within 4.8e-7, with no new import.
- A view with a storage offset of 784 elements came as the storage's buffer with a byte offset of 3,136 bytes. The program read it through an element offset in its uniform block and matched within 2.3e-5. The DLPack `data` field is the storage's `id<MTLBuffer>`, while `tensor.data_ptr()` of the view is that pointer plus the offset, which is no object.

### Runtime Results

The built runtime ran the spike's two models under `caliper.watch(model, optimizer)` with its defaults, as a managed run's script does, without an explicit `caliper.step()`. One geometry program was installed through `caliper._programs.install`, and its outputs went through the real publish path of `caliper.scene`. The program computes the norm of each row of the bound weight and writes one point for each row. Each number is the median of five runs, one process per run of 6 s to 9 s, with the two modes in alternating order. The runs held all three heavy slots in a quiet window. The peak memory footprint was 0.49 GB for the small model.

| Model | Loop | Base Steps per Second | With the Program | Share of Base | Cadence N | `D` | `b` | Measured Share | Passes per Second | 1-Minute Load |
|---|---|---|---|---|---|---|---|---|---|---|
| Small | Plain | 2,400.0 | 2,302.1 | 95.9 % | 64 to 102 | 1.20 ms | 0.417 ms | 3.70 % | 32.7 | 1.30 to 2.49 |
| Medium | Plain | 87.63 | 84.80 | 96.8 % | 3 to 5 | 1.25 ms | 11.39 ms | 3.54 % | 26.9 | 1.21 to 2.65 |
| Small | `loss.item()` | 996.6 | 971.0 | 97.4 % | 5 to 10 | 0.29 ms | 0.937 ms | 3.68 % | 112.3 | 1.28 to 2.53 |
| Medium | `loss.item()` | 82.80 | 81.71 | 98.7 % | 3 | 0.64 ms | 12.04 ms | 1.78 % | 27.2 | 1.33 to 2.37 |

The cadence before the step marks kept 57.2 % on the small model and 94.9 % on the medium model in the same setup. On the small model, the loss is larger than the measured share, because the marks themselves cost about 1 % to 3 %: a program that ran no pass after its warm-up gave 2,326 steps per second against 2,392 to 2,396 without it. The base runs of the small model with `loss.item()` spread from 893 to 1,066 steps per second, thus that row has a larger error than the others. Under a 1-minute load of 4 to 7, with other builds on the machine, the small model kept 93.6 % to 94.4 % in the same measurement.

The counters of the runtime showed the in-place route in each run: one import in place, no import refused, 0 bytes copied, and each output given as a DLPack capsule over the runtime's own buffer with 0 bytes copied. The publish path then makes the copy of its route, which the table of Output Path states.

### Documented and Undocumented Behavior

DLPack documents the device type `kDLMetal` and says that `data` can be opaque on some device types, with `byte_offset` to the first element. Torch documents `__dlpack__` and `torch.mps.Event`. Torch does not document that `data` is the storage's `id<MTLBuffer>` on MPS, that its MPS buffers use shared storage and untracked hazards, or the first version that exports MPS tensors. The runtime depends on the first point, and it checks it at run time: it refuses a `data` value that is not an `MTLBuffer` of the device.

## Platforms Without Measurements

Vulkan with CUDA on Windows and Linux is expected to follow the same design: torch exports `kDLCUDA`, and the runtime imports CUDA memory through Vulkan external memory, as the tensor bridge's exportable pool does today. Memory from torch's default caching allocator cannot be exported. Thus a bound tensor on CUDA needs the exportable pool, or one GPU-side copy. The sync points use a CUDA event and a shared timeline semaphore. None of this is tested on this machine.

## Risks

- **Small steps.** The first sync point is a CPU wait for the whole queue of torch. Its cost is fixed, thus it is a large share of a short step. A pass every 20 steps still took 5.4 % of the small model's throughput, thus the budget lowers its rate to about one pass in 27 steps.
- **Undocumented buffer handle.** A torch release can change what `data` holds on MPS. The run-time check turns such a change into a refusal, not a crash.
- **Torch versions.** Old versions cannot export MPS tensors. The supported range starts at a version between 2.6.0 and 2.14.1 that the spike did not find.
- **Shared device under memory pressure.** Torch's allocator limits itself by its own watermark and does not see the runtime's buffers. A large output budget can push the device past its working set.
- **Compiler in the training process.** A Naga or driver crash ends the training script. The source size limit and the validation of the sandbox reduce this risk, and they do not remove it.

## Work Items

| Work item | Size in days | Depends on |
|---|---|---|
| `caliper._producer`: the native module, its build in the wheel, and the Python facade | 2 | The Naga library of the shader compiler |
| Torch adapter: export, the two sync points, the step pre-hook and the version check | 1 | None |
| Read-only imports in the layer, with the binder rule and its tests | 1 | None |
| Budget per step: step time, the rate N, skip and the reason in the view | 1 | The adapter |
| Outputs as scene batches through the network route, with the header count | 1.5 | The facade |
| In-process route of a scene output, imported in place by the in-process renderer | 1.5 | The facade |
| Load-on-demand test and the counters of REN-31 for the three routes | 0.5 | The facade |
| Failure paths: refusals, timeouts and a lost device | 0.5 | The facade |

These items come to 9 days. The shared-memory route waits for scenes through shared memory.

## Decisions

- **Own command queue.** The runtime does not use torch's queue, and so it stays free of the torch ABI. The cost is the CPU wait of the first sync point.
- **Wait at once.** The runtime waits for Caliper's work at once after its pass. The deferred wait was not stable across the two models, and the wait at once costs less than a millisecond.
- **Sync points only on pass steps.** A step without a due program waits for nothing, thus the rate N lowers the cost of the sync points as well as the cost of the pass. It records a mark, which gives torch's queued work to the GPU without a wait.
- **Marks at each step.** While a program is installed, the facade records a mark at each step, because torch's MPS queue otherwise holds its work until the pass, and the wait then costs that work's whole GPU time.
- **Device from the tensor.** The runtime takes the device from the first bound tensor's buffer and refuses a tensor on another device.
- **Read-only in three places.** Naga, the binder and the layer each refuse a write to a training tensor, and the layer field keeps the rule when a later caller skips the binder.

## Open Decisions

- **Deferred wait.** A wait in the next optimizer step pre-hook gave 95.8 % instead of 93.0 % on the medium model, but 41.1 % instead of 64.1 % on the small model, with runs from 397 to 643 steps per second. The cause is not known. Recommendation: keep the wait at once in the first version, and measure the deferred wait again with the GPU time per pass before it becomes an option.
- **GPU-side sync with torch's queue.** A module built against the torch C++ API could make torch's queue signal and wait on a shared event and remove the CPU wait at a pass. `torch.mps.compile_shader()` could instead run the pass on torch's own queue, outside the graphics layer. Recommendation: not in the first version, because the step marks cut the cost of a pass on the small model from about 13 ms to about 1 ms. Revisit if the budget lowers the rate of the demo models below one pass per second.
