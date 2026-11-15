# Network streaming benchmark

`bench_net.py` measures what streaming snapshots over a WebSocket costs a
training loop. It reports throughput against a commit-matched control, the
cost of each view call, the snapshots and bytes per second that reach a viewer,
and the step-to-receipt latency at that viewer. Each viewer is `net_client.py`
running in its own process, as a browser tab or the Caliper app would, so the
measurement includes a real second process and a real socket. Neither file
opens a window.

## Running it

Make a local environment in the checkout and run from its root:

```sh
uv venv && uv pip install torch numpy websockets
export PYTHONPATH=python CALIPER_SESSION=0
```

`PYTHONPATH=python` imports the package from the checkout. `CALIPER_SESSION=0`
stops the first view call from opening the shared-memory live session, so
no native extension is needed.

The product's transport (`caliper.serve()`), one heatmap size per invocation:

```sh
for s in 256 1024 2048; do
  .venv/bin/python python/bench/bench_net.py --transport serve --size $s
done
```

The stand-in transport, every arm with its recommended way of getting a
snapshot to the host, and then the ways compared side by side:

```sh
.venv/bin/python python/bench/bench_net.py --transport standin --size 1024 --repeat 5
.venv/bin/python python/bench/bench_net.py --transport standin --size 1024 --repeat 5 \
    --host-copy nonblocking,direct,event-locked --arms baseline,control,heatmap,all
```

A smoke test of about ten seconds, which writes a `-quick` result file:

```sh
.venv/bin/python python/bench/bench_net.py --transport standin --quick
```

| Option | Default | Meaning |
|---|---|---|
| `--transport` | `serve` | `serve` streams through `caliper.serve()`; `standin` through the stand-in transport in `bench_net.py` |
| `--host-copy` | `nonblocking` | Stand-in only: how a reduced snapshot reaches the host (see below); with several, comma-separated, every arm with a viewer runs once per method, named `<arm>/<method>` |
| `--device` | `mps` | `mps` or `cpu` |
| `--size` | 1024 | The heatmap's weight is size x size |
| `--points` | `10000,100000` | Point counts of the points views |
| `--viewers` | 3 | Viewers in the `viewers-K` arm; below 2 there is no such arm |
| `--arms` | all | Comma-separated arm names or prefixes, such as `baseline,control,heatmap` |
| `--repeat` | 3 | Times every arm runs; the arms are interleaved within each repeat |
| `--seconds`, `--warmup` | 8, 1 | Measured and warm-up time per arm |
| `--rate` | 60 | The stand-in's snapshot rate and the control's event rate, in Hz |
| `--quick` | off | One repeat of 0.5 s arms, at most 2 viewers |

A run of all arms with three repeats takes about five minutes. The result goes
to `python/bench/results/N-<transport>-<device>-<size>.txt`, and for the
stand-in to `N-standin-<method>-<device>-<size>.txt` (`hostcopy` in place of the
method when several are compared), as well as to stdout.

## The workload

The loop is the one of `bench_show.py`: the TinyCNN of
`examples/python/train_show.py` on 64-image batches with Adam, plus a weight
`W` (size x size, float32) updated in place every step. The points views show a
static embedding `E` of 100,000 x 3 float32 values with 4-class labels; the
10,000-point view is its first 10,000 rows, a contiguous slice, so the points
add no work to the loop. The series view is the loss. The loop has no sync
point of its own, as in `bench_show.py`; elapsed time includes the device sync
at the end of each arm.

## The arms

| Arm | View calls every step | Viewers |
|---|---|---|
| `baseline` | None | 0 |
| `control` | None; `torch.mps.Event.record()` at 60 Hz (MPS only) | 0 |
| `noviewer` | Every view below | 0 |
| `heatmap-<S>` | `show(W, "W")` | 1 |
| `series` | `series(loss, "loss")` | 1 |
| `points-10k`, `points-100k` | `points(E[:N], labels[:N], "emb10k", classes=4)`, or `"emb100k"` | 1 |
| `all` | All four views | 1, subscribed to all four |
| `viewers-K` | All four views | K, each subscribed to all four |

The stream is open in every arm, so `baseline` and `control` run beside an idle
server. In `noviewer` nothing is connected, which measures a call when no viewer
subscribes. Every arm runs once per repeat, the arms in the order above, and
every ratio is taken between arms of the same repeat, which removes slow drift
in the machine's load.

## Why the control is commit-matched

On MPS, `torch.mps.Event.record()` commits torch's command buffer. On this small
model a periodic commit by itself raises throughput by 40 to 75 %: torch holds
many small kernels until a sync, and a commit lets the GPU start on them early
([the commit effect](../../docs/explanations/live-streaming.md#sync)). Every
snapshot path records an event or syncs, so comparing it only with the plain
baseline flatters it. The `control` arm is the same loop recording an event at
the snapshot rate and taking no snapshot, and "% of control" is the number to
read. "% of baseline" is printed too; on MPS it is well above 100 % for any
transport that commits.

## Reading the output

The file starts with the command, the torch version, the workload and the load
average. One `RESULT` line of JSON follows per arm and repeat, with every number
of that run. Three tables close the file.

The first table has one row per arm, the median over the repeats with the
range across them in parentheses:

| Column | Meaning |
|---|---|
| steps/s | Training steps per second over the measured window, including the final device sync |
| % of control, % of baseline | Steps/s divided by the same repeat's `control` or `baseline` |
| snapshots taken/s | View calls that took a snapshot (returned a true value), per second, over all views |
| received/s | `snapshot` and `series` messages that reached the viewers, per second, summed over viewers |
| MB/s on the socket | Bytes of every WebSocket message the viewers received, summed over viewers (frame headers of 2 to 14 bytes excluded) |
| latency ms | Viewer receipt time minus the snapshot's `stamp_ns`, the producer's `time.monotonic_ns()` at the view call; median and p95 over every message whose stamp falls in the measured window, pooled over the repeats |
| step µs | Wall time of one training step with its view calls: p50, p99 and maximum, pooled |

Received/s can be lower than taken/s because the latest snapshot wins: a newer
snapshot replaces one that has not been sent. Latency includes the time the
snapshot waits behind training work already queued on the GPU, then encoding,
the socket and the viewer's event loop. On macOS both processes read
`mach_absolute_time`, so no clock offset applies.

The second table gives the cost of each view call in microseconds, median and
p99, split into calls that took a snapshot and calls that did not. In
`noviewer` every call is one that nobody subscribed to. Elsewhere a skipped call
is one inside the rate limit. The `control` row is the cost of recording the
event.

The third table appears only for the stand-in and describes its background
thread: how often it called `.cpu()`, how long each call took, and the step
times of training steps that overlapped such a call against all other steps.
A training thread stalled by the background thread shows up there as a heavier
tail than in the `control` row of the first table.

The numbers are only as quiet as the machine. With other work running, repeat
every arm (the default is 3), read the ranges, and compare arms within one
file rather than across files.

## The stand-in transport

`--transport standin` uses a deliberately simple transport written inside
`bench_net.py`. It is a measuring stick that proves the harness without
anything from the sprint, not the product's transport, and its numbers are not
the product's numbers. Its result files say `standin` in their names.

For each view call at up to 60 Hz it reduces the tensor on the device (a
heatmap is average-pooled to at most 512 x 512, points are strided to at most
100,000), computes the value range on the device, and queues the result. A
background thread moves it to the host, encodes it, and hands it to a
`websockets` server on 127.0.0.1 in a second thread. The latest snapshot wins
per view and viewer, series values travel as deltas, and a view that nobody
subscribes to costs a dictionary lookup. The framing (`standin_encode` in
`net_client.py`) has the shape of wire protocol v3 but is not v3.

`--host-copy` chooses how the reduced snapshot reaches the host without the
training thread waiting for the GPU. This is the design question the
stand-in answers for the transport.

| `--host-copy` | Training thread | Background thread |
|---|---|---|
| `event` (a) | Queues the reduced copy and records a `torch.mps.Event` | Polls the event, then calls `.cpu()` on the copy |
| `event-locked` (a) | As (a), holding a lock around each step and its view calls | As (a), holding the same lock around `.cpu()` |
| `direct` (b) | Calls `.cpu()` on the reduced tensor, which drains torch's queue | Encodes and sends |
| `nonblocking` (c) | Queues `copy_(non_blocking=True)` into a pinned host buffer and records an event | Polls the event, then reads the buffer; it never calls torch to encode GPU work |

### What the stand-in found

These numbers were measured on an M5 Max (18 cores, torch 2.14.1) while other
work ran on the machine, at load averages of 7 to 10. Between repeats, the
control alone moved by a factor of up to 3 (356 to 1082 steps/s at 256²), so
a single "% of control" is uncertain by tens of percent. The comparison
between methods is steadier, because their arms run next to each other in each
repeat. The order (c), (a), (b) held in all 5 repeats at 256² and 1024², for
the heatmap and for all views together. At 2048², (c) was fastest in 4 of 5
repeats and the full order held in 3.

Approach (a) as specified is not usable with torch 2.14.1. Calling `.cpu()`
from a background thread while the training thread queues work kills the
process within seconds, with the Metal assertion
`_status < MTLCommandBufferStatusCommitted` in
`-[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]` or with a segmentation
fault. It did so in 6 of 7 smoke runs of the harness (0.5-second arms), in the
full run, and in 3 of 3 runs of a separate 8-second training loop. `--host-copy event` reproduces it, and
`N-standin-event-mps-1024.txt` ends where the process died. Torch's MPS stream
is not safe to drive from two threads. A lock makes it safe, but the product cannot
put a lock around the user's own training code, and the measurement shows
what the lock costs: the background `.cpu()` waits for everything torch has
queued (5.3 to 6.8 ms median, p99 23 to 27 ms), and the training thread
waits behind it.

Throughput with one heatmap viewer at 60 Hz, median over 5 repeats with the
range across them (files `N-standin-hostcopy-mps-<size>.txt`):

| | 256² | 1024² | 2048² |
|---|---|---|---|
| (c) `nonblocking`, % of control | 90 (73 to 185) | 91 (70 to 142) | 91 (89 to 101) |
| (a) `event-locked`, % of control | 68 (54 to 172) | 61 (52 to 78) | 79 (46 to 98) |
| (b) `direct`, % of control | 62 (41 to 102) | 49 (38 to 57) | 58 (36 to 67) |
| (a) `event-locked`, % of (c) in the same repeat | 75 (71 to 93) | 64 (55 to 87) | 86 (45 to 104) |
| (b) `direct`, % of (c) in the same repeat | 59 (55 to 69) | 46 (40 to 74) | 63 (40 to 72) |
| Step time p99, control / (c) / (a) / (b), ms | 4.7 / 4.7 / 18.3 / 21.1 | 3.4 / 4.0 / 18.1 / 20.9 | 3.7 / 4.1 / 12.7 / 16.9 |
| Call that took a snapshot, median, (c) / (a) / (b), µs | 107 / 66 / 6210 | 116 / 85 / 7181 | 122 / 90 / 6406 |
| Latency median / p95, (c), ms | 11.7 / 16.8 | 14.5 / 17.2 | 9.9 / 16.0 |

The step-time distribution shows the stall that the means hide. With (a)
behind a lock, training steps that overlapped a background `.cpu()` had a p99
of 19.6 to 24.1 ms, against 2.6 to 3.0 ms for the other steps of the same arms.
With (b) the drain is in the view call itself: a call that took a snapshot cost
6.2 to 7.2 ms however small the snapshot, because `.cpu()` waits for all queued
work, not just for the copy. With (c) the step-time p99 stays within 0.6 ms
of the control's, and no torch call that encodes GPU work leaves the training
thread.

Approach (c) is the one that keeps training unblocked, and it is the default.
Its call that takes a snapshot costs 107 to 122 µs here (pooling, `aminmax`,
`cat`, the queued copy and the event); the product's shared-memory path costs
35 to 50 µs per snapshot. With all four views streaming at 159 MB/s through the
stand-in's Python `websockets` server, (c) kept 90 % of the control at 1024²
(72 to 139), close to the single heatmap. The remaining cost in this stand-in
is probably the Python threads that encode and send while the training thread
needs the interpreter, but this machine was too busy to measure an effect of a
few percent.

Every arm with (c) at 1024² is in `N-standin-nonblocking-mps-1024.txt`, measured
at load averages of 5.6 to 20.5. By median every arm with a viewer ran at 96
to 107 % of the control, with ranges of tens of percent, so at this load the
stand-in's cost with (c) is below what the machine resolves. Three viewers
received 674 messages and 469 MB per second in total, with a latency of
8.6 ms median and 17.9 ms p95. A call that nobody subscribed to cost 0.2 to
0.8 µs median.

## Fitting the harness to the real transport

Two adapter functions at the top of each file are the only code that knows the
sprint's transport and wire codec. In `bench_net.py`, `open_stream` starts
`caliper.serve(port=0)` and returns the server with `caliper.show`,
`caliper.series` and `caliper.points`, and `stream_url` builds the viewer's
endpoint `ws://127.0.0.1:<port>/v3/stream?token=<token>` from the server's
`port` and `token`. In `net_client.py`, `wire_codec` maps the client's few
messages to `caliper._wire` and back, and `handshake` performs the v3 opening:
HELLO, WELCOME, then a SUBSCRIBE for each view once VIEWS lists it, until every
VIEW_REPLY has confirmed. The client answers PING with PONG and reads
`stamp_ns` from every SNAPSHOT and SERIES. It was checked against a v3
producer built from the protocol branch's codec. Before that codec is merged,
`--wire-py` loads a copy of `_wire.py` from a file.

The harness assumes three things of the product. `caliper.serve()` returns an
object with `port`, `token` and `close()`. A view call returns a true value
when it took a snapshot and a false one otherwise; without that, the per-call
table cannot split the two. A view appears in VIEWS once the script has
called it, so the harness keeps training and calling the arm's views while
the viewers subscribe.
