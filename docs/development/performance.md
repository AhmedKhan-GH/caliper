---
kind: reference
profile: measurements
figure: none
---

# Performance

Caliper's measured costs, with the machine, the scripts and the method behind each number.

## Live Streaming

### Method

| Item | Detail |
|---|---|
| Machine | M5 Max, 18 cores, 64 GB, torch 2.14.1; otherwise idle |
| Scripts | `python/bench/` |
| Model | The TinyCNN of `train_show.py` plus a synthetic SxS f32 weight rewritten in place and shown every step |
| Duration | 10 s after a 1 s warm-up |
| Repeats | Two (run 1 / run 2) |
| Prototype A | The in-process prototype |
| Prototype B | The separate-process prototype |
| Off | No rate limit |
| Control | Commit-matched: the same loop recording an Event at 60 Hz and taking no snapshot |

### Commit Effect

`torch.mps.Event.record()` commits torch's command buffer. On the benchmark's small model that alone raises throughput by 40 to 75 %: torch's adaptive commit holds many small kernels until a sync, and a periodic commit lets the GPU start early. A plain "no show" baseline therefore flatters `show()`, which is why every comparison uses the commit-matched control.

### Final Code

The final code runs at 98.3 to 99.5 % of the control in the own window and 98.1 to 99.4 % in attach mode, at 60 Hz. Its per-call figures are in [Caliper LIB API](../python/caliper-lib-api.md#call-cost). The sections below hold the prototype measurements the design rests on.

### Throughput

MPS throughput as a percentage of the commit-matched control, with run 1 and run 2 where both were recorded.

| Prototype and Rate | 256² | 1024² | 2048² |
|---|---|---|---|
| A, 60 Hz | 96 / 99 | 97 / 98 | 95 / 97 |
| A, off | 92 / 99 | 91 / 96 | 83 / 95 |
| B, 60 Hz | 95.7 | 97.2 / 95.5 | 96.9 / 94.0 |
| B, off | 96.5 | 97.1 / 97.1 | 94.9 / 94.1 |

| Comparison | Result |
|---|---|
| Same rows against the plain baseline | 136 to 175 %, the commit effect |
| `train_show.py`'s two views, synchronous `show()` | 48 % of baseline |
| `train_show.py`'s two views, under B | 137 % of baseline (96.8 % of the control) |
| CPU tensors at 256², A, with the rate limit | 98 / 97 % of baseline |
| CPU tensors at 256², B | 98.1 % of baseline |

### Call Cost

Median cost per call, and the viewer's frame rate during training.

| Measure | A | B |
|---|---|---|
| `show()` taking a snapshot, MPS | 35 to 54 µs, flat 256² to 2048² | 50 to 57 µs, flat |
| `show()` skipped by the rate limit | 1.7 to 2.2 µs | 2.0 µs |
| `show()` taking a snapshot, CPU 256² | 65 to 86 µs | 72 µs |
| Viewer frame rate during training | 60 Hz | 59.9 to 60 fps |

### Latency

Step-to-screen latency, from a training step to the frame that shows it.

| Rate | A | B |
|---|---|---|
| 60 Hz, median / p95 | 28 to 40 / 31 to 47 ms | 19 / 29 ms |
| Off | 2 to 22 ms | 4.3 to 5.7 / 7 to 8 ms |

Latency is mostly the snapshot waiting behind torch's queued work, then the wait for the next 60 Hz tick. B measured 13 of its 19 ms in the queue. With a sync every step it drops to 9.9 ms.

### Memory

There is no host copy in either prototype, so snapshot cost is flat with size. Over 200 snapshots at every size, the changes were these:

| Quantity | Change |
|---|---|
| `torch.mps.current_allocated_memory()` | 0.000 MB |
| Trainer RSS | At most +0.45 MB (A), +0.12 MB (B) |
| Viewer RSS | At most +0.66 MB |

Every MPS view reports `compute-imported` with `imported = 1` (`blit-imported` for RGBA).

### Window Liveness

The training loop sleeps for 5 s with `time.sleep`. The first column is the in-process `show` without a snapshot ring, the baseline the snapshot ring replaced.

| Observation | In-Process Baseline | Own Window | Own Window with `run(train)` | Attach to App |
|---|---|---|---|---|
| Canvas repaints | No | Yes, 300 frames | Yes | Yes, 59.5 to 60 fps |
| Input handled | After the sleep (4.5 s late) | After the sleep | 0 to 0.3 ms | At once |
| Close | After the sleep | After the sleep | 103 ms | At once |

### Session End

After a SIGKILL of the script, the viewer kept drawing at 60 fps.

### Tearing Detector

| Ring | Bad frames |
|---|---|
| The real ring | 0 in about 450 |
| A deliberately unsafe control | 18 in 441 |

The detector ran in the in-process prototype.

### Probe

`python/bench/probe_foreign_mtlbuffer.py` established that the snapshot path works on torch 2.14.1:

- `torch.from_dlpack` accepts a `kDLMetal` capsule whose `MTLBuffer` torch did not allocate (`newBufferWithBytesNoCopy` over a page-aligned `MAP_SHARED` mmap).
- `copy_` and `aminmax(out=)` write into it on torch's stream.
- `torch.mps.Event.record()` commits torch's command buffer, so `Event.query()` turns true with no `synchronize()`.

| Path | CPU cost |
|---|---|
| Snapshot path of `probe_foreign_mtlbuffer.py`, 1024x1024 | 0.057 ms |
| The synchronous drain it replaces | 3.6 ms |

### Unsynchronized Loop

A benchmark loop without any sync queued 228k 16 MB kernels per second. Torch's own encoded-command queue grew RSS at 85 MB/s with Caliper and 250 MB/s without it.

## Related Pages

- [Live Streaming](../python/live-streaming.md)
- [Live Session Layout](live-session-layout.md)
- [Caliper LIB API](../python/caliper-lib-api.md)
- [Testing](testing.md)
