---
kind: reference
profile: measurements
figure: none
---

# Performance

Each Caliper measurement here has its computer, its scripts and its method.

## Live Streaming

### Method

The tests operated on an M5 Max with 18 cores and 64 GB of memory, with torch 2.14.1 and no other load. The scripts are in `python/bench/`. The model is the TinyCNN of `train_show.py` plus a synthetic SxS f32 weight. The script changes this weight in place and shows it at each step. Each test operated for 10 s after a 1 s warm-up, and the tests operated each configuration two times. Thus the tables give run 1 and run 2 where the two runs have values.

The tests measured two prototypes. Prototype A is the in-process prototype, and prototype B is the prototype with a second process. In the tables, "off" is the value for no rate limit on `show()`. Each percentage compares with a commit-matched control, which is the same loop when it records an Event at 60 Hz and makes no snapshot.

### Commit Effect

`torch.mps.Event.record()` commits the command buffer of torch. On the small model of the benchmark, this commit increases throughput by 40 to 75 % without a snapshot. The adaptive commit of torch keeps many small kernels until a sync, and a commit at regular intervals lets the GPU start before the sync. Thus a "no show" baseline without an Event gives `show()` a result that is too good. For this reason, the tests compare each result with the commit-matched control.

### Final Code

At 60 Hz, the last version of the code operates at 98.3 to 99.5 % of the control in the own window. In attach mode, it operates at 98.1 to 99.4 %. [Caliper LIB API](../python/view-calls.md#call-cost) gives its cost for each call. The sections that follow give the prototype measurements that the design uses.

### Throughput

The table gives MPS throughput as a percentage of the commit-matched control. Where the two runs have values, the table gives run 1 and run 2.

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

The table gives the median cost of each call and the frame rate of the viewer during the training.

| Measure | A | B |
|---|---|---|
| `show()` taking a snapshot, MPS | 35 to 54 µs, flat 256² to 2048² | 50 to 57 µs, flat |
| `show()` skipped by the rate limit | 1.7 to 2.2 µs | 2.0 µs |
| `show()` taking a snapshot, CPU 256² | 65 to 86 µs | 72 µs |
| Viewer frame rate during training | 60 Hz | 59.9 to 60 fps |

### Latency

The table gives the latency from a training step to the frame that shows it.

| Rate | A | B |
|---|---|---|
| 60 Hz, median / p95 | 28 to 40 / 31 to 47 ms | 19 / 29 ms |
| Off | 2 to 22 ms | 4.3 to 5.7 / 7 to 8 ms |

The largest part of the latency is the time that the snapshot waits behind the queued work of torch. The remaining time is the time until the next 60 Hz tick. Prototype B had 13 of its 19 ms in the queue. With a sync at each step, its latency decreases to 9.9 ms.

### Memory

The two prototypes make no host copy, and thus the cost of a snapshot does not change with its size. The table gives the changes for 200 snapshots at each size.

| Quantity | Change |
|---|---|
| `torch.mps.current_allocated_memory()` | 0.000 MB |
| Trainer RSS | At most +0.45 MB (A), +0.12 MB (B) |
| Viewer RSS | At most +0.66 MB |

Each MPS view shows `compute-imported` with `imported = 1`, and an RGBA view shows `blit-imported`.

### Window Liveness

In this test, the training loop stops for 5 s in `time.sleep`. The first column is the in-process `show` without a snapshot ring, which is the baseline that the snapshot ring replaced.

| Observation | In-Process Baseline | Own Window | Own Window with `run(train)` | Attach to App |
|---|---|---|---|---|
| Canvas repaints | No | Yes, 300 frames | Yes | Yes, 59.5 to 60 fps |
| Input handled | After the sleep (4.5 s late) | After the sleep | 0 to 0.3 ms | At once |
| Close | After the sleep | After the sleep | 103 ms | At once |

### Session End

After a SIGKILL stopped the script, the viewer continued to draw at 60 fps.

### Tearing Detector

The detector operated in the in-process prototype, and it counted the bad frames for two rings.

| Ring | Bad frames |
|---|---|
| The real ring | 0 in about 450 |
| A deliberately unsafe control | 18 in 441 |

### Probe

`python/bench/probe_foreign_mtlbuffer.py` showed that the snapshot path operates on torch 2.14.1. `torch.from_dlpack` accepts a `kDLMetal` capsule for an `MTLBuffer` that torch did not allocate (`newBufferWithBytesNoCopy` on a page-aligned `MAP_SHARED` mmap). Then `copy_` and `aminmax(out=)` write into this buffer on the stream of torch. Because `torch.mps.Event.record()` commits the command buffer of torch, `Event.query()` becomes true with no `synchronize()`.

The table compares the CPU cost of the snapshot path with the cost of the synchronous drain that it replaces.

| Path | CPU cost |
|---|---|
| Snapshot path of `probe_foreign_mtlbuffer.py`, 1024x1024 | 0.057 ms |
| The synchronous drain it replaces | 3.6 ms |

### Unsynchronized Loop

A benchmark loop with no sync queued 228k kernels of 16 MB each second. The encoded-command queue of torch itself increased RSS at 85 MB/s with Caliper and at 250 MB/s without Caliper.

### Example Script

A further measurement of the same attach mode uses the `TinyCNN` of `train_show.py` plus a 1024x1024 weight. The script shows that weight at each step, on MPS. In 8 s, 468 of 9,460 calls took a snapshot. Training ran at 98.1 % of the commit-matched control, which [Commit Effect](#commit-effect) explains. The median step to screen was 23.53 ms. A call that took a snapshot cost 35.5 µs at the median, and a skipped call cost 2.0 µs.

## Python Calls

With no viewer attached, 100,000 calls per row give these costs. A `caliper.show` call takes 0.167 µs at the median, with a p99 of 0.291 to 0.292 µs. A `caliper.series(1.0)` call takes 0.125 µs at the median, with a p99 of 0.209 µs. A `caliper.step()` call with no command in the queue only counts the step, and it takes about 22 ns on Apple silicon.

The `histogram` call bins with `index_add_` and not with `torch.histc`, because `torch.histc` on MPS drains the queue. The measured drain was 15 to 53 ms.

## Hub Relay

On an Apple silicon Mac, the hub sends a 512 × 512 heatmap at 60 Hz to four viewers. The median latency from script send to viewer receive is about 1.4 ms, and the 99th percentile is less than 3 ms. The total rate is 240 MiB/s. The test `caliper_tests -tc="net/hub: relay*" --no-skip` makes this measurement.

## Tensor Handoff

On an RTX 500 Ada, `embed_scope` gave approximately the same steps per second with the drained handoff and with the stream-ordered handoff. The cause is the `loss.item()` call at each step, which already synchronizes. The stream-ordered handoff stops the stalls of the frame thread on the training queue. A texture create costs about 1 ms, and an update costs a fraction of that.

A stress test in `tests/test_torch_adapter.cpp` makes 500 CUDA handoffs on a pool stream while a second thread trains. The largest measured stall of the frame thread came from a database table that the UI thread rebuilt, not from tensors.

## Related Pages

- [Live Streaming](../python/live-streaming.md)
- [Live Session Layout](live-session-layout.md)
- [Caliper LIB API](../python/caliper-lib-api.md)
- [Testing](testing.md)
