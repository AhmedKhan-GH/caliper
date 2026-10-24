# Live streaming

The goal is a normal PyTorch script that keeps training at full speed while
Caliper shows its tensors live, without the script waiting for the GPU or the
display and without the window freezing, whatever the loop is doing.

The first Python entry point did not meet that. `caliper.show` drew in-process
through the [embed v1.2 views](../reference/embedding.md#built-in-tensor-views-embed-v12),
so the window was only live while the script called `show()`, and every call on
MPS drained torch's queue: 3.6 of the 4.1 ms a show-step cost, taking training
from 331 to 159 steps/s with two shows per step. Live streaming is the second
stage: `show()` waits for nothing and the viewer samples at its own pace.

To use it, start with [Watch a PyTorch training run live](../tutorials/live-pytorch.md);
every call is in the [Python API](../reference/python-api.md) reference.

!!! note "Status"
    macOS / Metal (Apple Silicon) only, local only: protocol v2
    (`include/caliper/live.h`) and embed v1.3 typed views
    (`include/caliper/live_views.h`). The numbers come from `SPIKE.md` §8, on
    one machine (M5 Max, 18 cores, 64 GB, torch 2.14.1); raw logs are in
    `python/bench/results/`.

## One view layer, two transports

Every call (`show()` and its siblings for points, series and histograms)
produces a **snapshot**. It reaches the embed core by one of two transports,
and both end in `caliper_core_live_apply()`, so the drawing code never knows
where a snapshot came from:

```text
training process                                         viewer process
show(t) ─► SnapshotRing ─┬─► SessionTransport ─► shm segments ─► ShmSource ─► live_core ─┐
                         │                                                               ├─► caliper_core_live_apply
                         └─► WindowTransport ─► render thread (same process) ────────────┘
```

The top path is **attach to app** (the viewer is `caliper-viewer` or the
Caliper app); the bottom one is the **own window**. One `SnapshotRing`
(`python/caliper/_ring.py`) owns slot choice, the rate limit and event polling
for both; a transport only allocates slots and receives completed snapshots. On
the viewer side, `live_core.cpp` reads a `Source` (`src/host/live/live_source.h`);
today the only `Source` is the shared-memory session.

## Sessions

The training process is the **client**. Its first call opens a local live
session: POSIX shared memory plus a **session file**
`<data_root>/live/<pid>.json`, where `data_root` is the client's `data_dir`,
else `$CALIPER_DATA_DIR`, else Caliper's app-data folder (macOS default
`~/Library/Application Support/Caliper`). The file is written atomically at open
and on every view change, removed at close, and names the pid and its start
time (against pid reuse), the session token, the script, the views and the
segment names.

The shared memory is page-aligned segments the client creates and unlinks:

- `/<session>-ctl`, the **control table**: a session header and up to 64 view
  rows, guarded by a `generation` seqlock. `magic` is stored last, so 0 means
  "still starting".
- `/<session>-v<id>`, one per view: a header page with the sync words, then 3
  slots (1 for a series), each a payload, an optional aux region and a 256-byte
  stats region.

Each process wraps its mapping of a segment as one shared-storage `MTLBuffer`
(`newBufferWithBytesNoCopy`), so the same physical pages are the trainer's torch
tensor and the viewer's draw source. Names and files are 0600: same user, no
network.

A viewer lists the session files, attaches to one, and on every poll writes a
heartbeat into the control table. The client's "is anyone watching" check is
one atomic load and a clock read: a heartbeat younger than 1 s. With no viewer,
`show()` returns after that check, with no snapshot and no GPU work. The claim
words are single-reader, so a session has **one viewer**; attach refuses a
session whose viewer is still attached.

Neither side ever waits for the other:

| Event | What happens |
|---|---|
| Viewer detaches or dies | The heartbeat is cleared (or ages past 1 s) and snapshots stop. Training is untouched; another viewer can attach. |
| Script closes | Session marked closed, names unlinked, session file removed, without waiting for a viewer. |
| Script is killed | The viewer keeps drawing the last frames and shows "session ended" (measured: 60 fps after a SIGKILL). The next session listing sees the dead pid, unlinks that session's names and removes the file. |

Every shared struct carries magic, version and `struct_size`; a reader refuses a
mismatch.

## Snapshots

A snapshot is **one GPU-to-GPU copy on torch's own stream** into CPU-readable
shared memory. The transport hands torch each slot through DLPack (for MPS, a
`kDLMetal` capsule over the segment's `MTLBuffer` at the slot's offset) and the
snapshot is a queued `slot.copy_(src)`. Autoscale ranges are computed on the GPU
by `torch.aminmax(src, out=...)` into the slot's stats region (points add a
bounding box; histograms are binned on the device). Nothing reads a value back:
no `.item()`, no `.cpu()`, no `synchronize()`. The viewer reads the stats from
the CPU side of the same pages.

For MPS tensors there is no host copy anywhere. A CPU tensor is copied once
into the shared pages; tensors on other devices are moved to the CPU first.

A probe established this was possible (`python/bench/probe_foreign_mtlbuffer.py`):
torch 2.14.1 accepts `torch.from_dlpack` over a `kDLMetal` capsule whose
`MTLBuffer` torch did not allocate (`newBufferWithBytesNoCopy` over a
page-aligned `MAP_SHARED` mmap), `copy_` and `aminmax(out=)` write into it on
torch's stream, and `torch.mps.Event.record()` commits torch's command buffer so
`Event.query()` turns true with no `synchronize()`. The whole path costs
0.057 ms CPU at 1024x1024, against the 3.6 ms drain it replaces.

## Sampling

Each view has a rate limit, 60 Hz by default; a call inside the interval
returns after a clock check. The viewer draws the **newest completed** snapshot
of each view at up to 60 fps, and older ones that completed in between are
superseded. Skipped training steps are by design: a display shows no more than
its frame rate, and every snapshot costs GPU work.

`show()` has **no back-pressure**. It never waits for a slot, the GPU or the
viewer; with no free slot the new snapshot is dropped and counted. In a GPU loop
with no sync point at all, torch runs ahead, a snapshot completes only when the
GPU reaches it, and latency grows with the run-ahead until the loop hits a sync
(`loss.item()`, logging). A benchmark loop without any sync queued 228k 16 MB
kernels per second, and torch's own encoded-command queue grew RSS at 85 MB/s
with Caliper and 250 MB/s without it.

## Sync

**Producer side.** After queueing a snapshot the ring records a
`torch.mps.Event` and returns. Completed snapshots are published oldest first
by polling `Event.query()`, from later calls and from a publisher thread that
polls every 2 ms while snapshots are pending, so a snapshot taken before a long
step still arrives.

**The 3-slot ring.** The client writes a slot that is neither the last
published one, nor claimed by the viewer, nor pending; with none free it drops
the new snapshot. Overwriting an in-flight slot instead was tried first and
starved publishing (20 published of about 1180 under a GPU backlog); dropping
guarantees the oldest pending snapshot completes and publishes. Publish and
claim are lock-free: both sides store with seq_cst, so either the client sees
the viewer's claim and avoids the slot, or the viewer sees the newer publish.
The viewer's GPU reads are never waited on either;
`caliper_core_live_view_read_done` polls the renderer's completion handles.

**Claims per kind.** How long the viewer holds a slot depends on how the kind
reads it:

| Kind | Payload read | Slot free again |
|---|---|---|
| Heatmap, image | Once per apply, into the view's texture | When every apply's device update has completed, the last one included |
| Points | Redrawn on the GPU when payload, camera or window size changes | The current payload stays claimed; a superseded one is free once the last frame that drew it retired |
| Series | One append-only ring slot; the viewer copies the window it draws | Never claimed (an overwrite during the copy is a one-sample glitch at the old end) |
| Histogram | Bin counts copied at apply | At once |

A tearing detector in the in-process prototype saw 0 bad frames in about 450
with the real ring, and 18 in 441 with a deliberately unsafe control.

**The commit effect.** `torch.mps.Event.record()` commits torch's command
buffer, and on the benchmark's small model that alone raises throughput by 40
to 75 %: torch's adaptive commit holds many small kernels until a sync, and a
periodic commit lets the GPU start early. A plain "no show" baseline therefore
flatters `show()`, so the benchmarks compare against a **commit-matched
control**, the same loop recording an Event at 60 Hz and taking no snapshot.

## Two modes

| | Attach to app | Own window (`caliper.window()`) |
|---|---|---|
| Window | A separate viewer process | The training process; a render thread repaints it |
| During a long step | Repaints and takes input | Repaints; input waits for the main thread unless `run(train)` is used |
| Script ends or crashes | The viewer keeps the last frames | The window goes with it |
| Window crashes or hangs | Training continues | The script crashes or hangs with it |
| Limit | One viewer per session | One window per process |

Attach to app survives the script, never freezes, and is what the Caliper app's
landing will attach to. The own window needs no second process, but AppKit
delivers input only to the main thread, which is the training thread. A 60 Hz
`SIGALRM` pump handled input within 16 ms (5 s late during a long C call) and
was disqualified: `DataLoader(num_workers=2)` hangs with any 60 Hz `setitimer`,
reproduced without Caliper. `run(train)` runs training on a worker thread and
gives the main thread to Caliper, which fixes input but changes how the script
is run.

Liveness with a 5 s `time.sleep` in the training loop:

| | Spike (v1.2 `show`) | Own window | Own window + `run(train)` | Attach to app |
|---|---|---|---|---|
| Canvas repaints | No | Yes, 300 frames | Yes | Yes, 59.5 to 60 fps |
| Input handled | After the sleep (4.5 s late) | After the sleep | 0 to 0.3 ms | At once |
| Close | After the sleep | After the sleep | 103 ms | At once |

Hence the decision: both modes behind one view layer, attach to app first as
the primary mode, the own window second with these limits documented.

## Measured numbers

These are the Phase 1 prototypes the decision was made on. The final code's
numbers (98.3 to 99.5 % of the control in the own window, 98.1 to 99.4 % in
attach mode, at 60 Hz) are in [Cost of a call](../reference/python-api.md#cost-of-a-call).

The TinyCNN of `train_show.py` plus a synthetic SxS f32 weight rewritten in
place and shown every step; 10 s after a 1 s warm-up; two repeats (run 1 /
run 2); machine otherwise idle. A is the in-process prototype, B the separate
process; "off" means no rate limit.

| MPS throughput, % of commit-matched control | 256² | 1024² | 2048² |
|---|---|---|---|
| A, 60 Hz | 96 / 99 | 97 / 98 | 95 / 97 |
| A, off | 92 / 99 | 91 / 96 | 83 / 95 |
| B, 60 Hz | 95.7 | 97.2 / 95.5 | 96.9 / 94.0 |
| B, off | 96.5 | 97.1 / 97.1 | 94.9 / 94.1 |

Against the plain baseline the same rows read 136 to 175 %, the commit effect.
`train_show.py`'s two views, which the spike's `show()` ran at 48 % of
baseline, run under B at 137 % (96.8 % of the control). CPU tensors at 256²: A
98 / 97 % of baseline with the rate limit, B 98.1 %.

| Per call, median | A | B |
|---|---|---|
| `show()` taking a snapshot, MPS | 35 to 54 µs, flat 256² to 2048² | 50 to 57 µs, flat |
| `show()` skipped by the rate limit | 1.7 to 2.2 µs | 2.0 µs |
| `show()` taking a snapshot, CPU 256² | 65 to 86 µs | 72 µs |
| Step to screen, 60 Hz (median / p95) | 28 to 40 / 31 to 47 ms | 19 / 29 ms |
| Step to screen, off | 2 to 22 ms | 4.3 to 5.7 / 7 to 8 ms |
| Viewer frame rate during training | 60 Hz | 59.9 to 60 fps |

Latency is mostly the snapshot waiting behind torch's queued work (B measured
13 of its 19 ms there; with a sync every step it drops to 9.9 ms), then the
wait for the next 60 Hz tick.

No host copy in either prototype: snapshot cost is flat with size; over 200
snapshots at every size `torch.mps.current_allocated_memory()` changed by
0.000 MB, trainer RSS by at most +0.45 MB (A) / +0.12 MB (B), viewer RSS by at
most +0.66 MB; every MPS view reports `compute-imported` with `imported = 1`
(`blit-imported` for RGBA).
