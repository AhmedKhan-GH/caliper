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

With no viewer attached, 100,000 calls per row give these costs. A `caliper.show` call takes 0.167 µs at the median, with a p99 of 0.291 to 0.292 µs. A `caliper.series(1.0)` call takes 0.125 µs at the median, with a p99 of 0.209 µs. A `caliper.step()` call with no command in the queue only counts the step, and it takes about 22 ns on Apple silicon. A scene call also runs its argument checks, which read host metadata only. With no viewer of the scene, `scene.mesh` (1,681 vertices, 3,200 triangles, values), `scene.lines` (a strip of 200) and `scene.points` (one point) take 2.33 to 2.83 µs at the median, with a p99 of 3.08 to 3.63 µs. These rows come from `python/bench/bench_call_cost.py` with MPS tensors and a server without a viewer, at a load average of 1.1, where the existing calls measured as in the rows above.

The `histogram` call bins with `index_add_` and not with `torch.histc`, because `torch.histc` on MPS drains the queue. The measured drain was 15 to 53 ms.

## Hub Relay

On an Apple silicon Mac, the hub sends a 512 × 512 heatmap at 60 Hz to four viewers. The median latency from script send to viewer receive is about 1.4 ms, and the 99th percentile is less than 3 ms. The total rate is 240 MiB/s. The test `caliper_tests -tc="net/hub: relay*" --no-skip` makes this measurement.

## Tensor Handoff

On an RTX 500 Ada, an MNIST training instrument gave approximately the same steps per second with the drained handoff and with the stream-ordered handoff. The cause is the `loss.item()` call at each step, which already synchronizes. The stream-ordered handoff stops the stalls of the frame thread on the training queue. A texture create costs about 1 ms, and an update costs a fraction of that.

A stress test in `tests/test_torch_adapter.cpp` makes 500 CUDA handoffs on a pool stream while a second thread trains. The largest measured stall of the frame thread came from a database table that the UI thread rebuilt, not from tensors.

## Embed Tensor Views

These numbers come from `caliper_embed_tests` and `bench_show.py` on Apple silicon. A 1024x1024 f32 view update, as the mean of 20 updates, takes 0.97 ms with CPU staging and 0.34 ms with METAL zero-copy and a GPU wait.

The native call, as the median of 50 calls in an offscreen 1280x800 window, gives these times:

| Size | CPU Staging | METAL with GPU Wait |
|---|---|---|
| 256² | 0.047 ms | 0.169 ms |
| 1024² | 0.680 ms | 0.174 ms |
| 2048² | 2.991 ms | 0.270 ms |

A 64x48 f32 METAL update takes 0.129 ms with `WAIT_GPU` and 0.009 ms without it. The first METAL f32 view compiles the colormap shader one time. This takes 1.6 ms with a warm cache and about 40 ms with a cold cache.

## Torch-Free Build

Slang takes most of a 12.7 s configure. A fresh torch-free configure takes 6.5 s. A cold torch-free build of all targets has 462 steps and takes 120 s at `-j 8`. Then `libcaliper.a` is 276 KB, in six objects. A cold `libcaliper` build with torch on takes 123 s. An offscreen Caliper Embed run of `hello_world` draws 43 frames in 1.02 s.

## Widget Agent Models

With a local model, `qwen3.8:27b` on an Apple M5 Max took 5 to 30 seconds for each ⌘K request in tests. The first request takes longer while Ollama loads the model. The model has a context length of 262,144 tokens, and its context cache is about 18 GB at full length. The cache is about 1 GB at 16,384 tokens for ⌘K and about 2 GB at 32,768 tokens for the side chat.

The first request of the widget agent has about 7,000 tokens before the reply of the model. It holds ten tools, the context bundle and the render, and the default `num_ctx` is 16,384. The first request of the general role has every command of the table as a tool. It has about 11,000 tokens and grows with each tool result, and the default `num_ctx` is 32,768. The OpenAI-compatible endpoint of Ollama ignores `options.num_ctx`.

## HTTP API Threads

One thread runs the loop of the hub and answers every request, and it never waits for the app. A request that needs app state waits for the frame thread, which takes it at the start of a frame. A launch runs there, as `CALIPER_AUTOLAUNCH` runs before the first frame.

A command goes to the queue of the command layer, which the frame thread drains. The request then has no thread of its own, and the frame thread and the hub thread both look for the answer without a wait. A capture copies the presented frame on the GPU, and a later frame takes the copy. A separate thread encodes the PNG, and thus the frame loop never waits for the encoder.

The windowless mode runs at about 60 frames per second. The first command opens the default workspace in a few milliseconds. The code is in `src/host/api/api_server.h`, `src/host/api/app_api.h`, `src/host/builder/app_session.h` and `src/host/api/session_gateway.h`. The spec page is in `web/src/spec/`, and the build makes it from `python/caliper/_web/spec.html`.

## Data Store Rebuilds

A table that `caliper.data.v1` rebuilt with `CREATE OR REPLACE TABLE` several times per second on a persistent store became slow. Over one session on Windows, each rebuild went from 80 ms to 500 ms. The cause is the growth of the write-ahead log and the catalog. `TEMP` tables do not have this problem.

## Vulkan and CUDA Costs

Measured on one RTX 500 Ada laptop GPU, driver 596.47, Release build. Small samples, so the numbers describe this hardware only.

- Training (MNIST instrument, 20 s per run, three runs per path): training steps are level within noise. The direct path removes the 2.5 to 3.5 s frame-thread block in `retire()` and lowers frame time p99 from 19.8 to 22.5 ms down to 12.8 to 19.3 ms. Frame time p50 stays at about 8.3 ms because of vsync.
- With a single image per texture, about 97 percent of kernels waited on a frame, and 30-second training dropped to 880 to 897 steps against 1407 on the buffer path. The second image is required.
- One update by size (call / done): 256 by 256 is 57 / 265 µs buffer and 30 / 65 µs direct. 4096 by 4096 is 100 / 1971 µs buffer and 51 / 1391 µs direct. Source: gfx test `Vulkan+CUDA: timing — update cost by size`.
- Texture create and release costs about 1 ms at instrument sizes. Release still waits for the queue to go idle.
- Eliding the adapter drain through stream ordering did not change training steps per second within 15 to 20 percent thermal variance, because `loss.item()` already synchronizes. The gain is that the frame thread stops stalling.

## CI Build Cache

With sccache, the CI build step takes about 13 min cold and about 1 min warm on macOS. On Windows, it takes about 26 min cold and about 3 min warm. The sccache entries of both lanes total about 130 MB.
