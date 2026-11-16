# Network streaming

Network streaming sends a script's live views over a WebSocket, so that a
notebook widget, a browser tab or the hub can show them. It keeps the view calls
and the snapshots of [Live streaming](live-streaming.md), adds a transport that
copies each snapshot from the device into a socket, and adds data views, which
decide which part of a large tensor travels at all.

To use it, start with [Watch a training run in a notebook](../tutorials/watch-a-training-run-in-a-notebook.md).
The calls are in the [Caliper LIB](../references/python-api.md#serve) reference,
and the [Wire protocol](../references/wire-protocol.md) reference defines every
message on the stream. This page explains the design and does not repeat the
protocol.

!!! note "Status"
    macOS, on the loopback interface only, with a session token. The wire
    protocol is version 3, defined by `include/caliper/live_net.h`.

## One set of calls, three transports

A script calls `show()`, `points()`, `series()` and `histogram()` the same way
whichever viewer watches it. Each call is offered to every open route, and each
route that a viewer is attached to takes its own snapshot:

```text
view call ─┬─► live session ─► shared memory ─► Caliper app or caliper-viewer       same machine, zero-copy
           ├─► own window ─► render thread in the same process                       same process, zero-copy
           └─► caliper.serve() ─► WebSocket ─► notebook widget, browser tab, hub      copies
```

The first two transports are described in [Live streaming](live-streaming.md).
The third is new. The **producer** is the process that makes the calls, such as
a training script or a notebook's kernel, and `caliper.serve()` starts its
server. A **viewer** is anything that connects to the stream and draws, such as
a notebook widget or a browser tab with the viewer page. The **hub**,
`caliper serve`, is a subcommand of the `caliper` binary that stands between
the two: any number of producers stream to it, any number of viewers watch it,
and it outlives the producers. It holds one session for each producer, and a
viewer chooses one. When only one session is live, the hub attaches a new
viewer to it, so a viewer written for `caliper.serve()` works against the hub
unchanged.

The kinds of live view are the same on every transport: heatmap, image, points,
series and histogram. So are the names, the arguments, the rate limit and the
device-side snapshot, which is why the same training code shows in the Caliper
app, a widget and a browser tab at once.

What differs is where a snapshot ends. The two local transports give the viewer
the memory the snapshot was written to, without a copy
([Zero-copy tensor visualization](zero-copy.md)). A socket needs bytes, so the
network transport copies each snapshot off the device and encodes it as one
binary message. The zero-copy claim therefore applies only to viewing on the
same machine through a live session or the own window; a viewer on the network
receives copies.

The viewers in the browser live in the `web/` package: the `<caliper-view>`
element, the notebook widget built on it, and the viewer page. The element
decides per view what draws it. Where WebGPU gives an adapter, Caliper's own
view layer compiled to WebAssembly (`web/wasm`, drawn by the
[WebGPU backend](rendering.md#the-webgpu-backend)) draws the
kinds its loader lists, which are now all five: heatmaps, images, points,
series and histograms. A kind the loader does not list is drawn by small views
written for the browser in TypeScript. Without an adapter the TypeScript views
draw every kind, with Canvas 2D where they need it. The element's header, with the view's name, its status, the data view
picker and a line about the view, is the same either way. Both kinds of
views select, pick and hover with the same gestures and send the producer the
same events, in source indices, so a widget's `selection` and the server's
`events()` do not depend on which views drew the view.

## Snapshots reduced on the device

A tensor in training can be far larger than anything a viewer can show. A 4096
by 4096 float32 weight is 64 MiB, so at the default 60 snapshots per second it
would be 3.75 GiB/s on the socket, for a widget a few hundred pixels wide. The
network transport therefore reduces a snapshot on the device before anything is
copied, and only the result leaves the GPU. With the default budget, a heatmap
snapshot is pooled to at most 512 by 512 values, which is 1 MiB.

The reduction runs where the data lives because the alternative defeats its
purpose. Reading the whole tensor back to the host to reduce it there costs the
transfer that the reduction is meant to avoid, and on MPS a read-back waits for
torch's queue, which stalls training ([Live streaming](live-streaming.md)
measured that wait). The viewer, in turn, never needs the whole tensor, so no
computation runs in the browser, and torch stays native in the producer.

## Data views

A **data view** decides which data travels; a **live view** decides how the
viewer draws it. A data view is a slice, selection, projection or aggregation
of a tensor, such as one layer, one head or a mean over the batch. The viewer
asks for it, the producer computes it on the device, and only the result is
streamed. The idea is the same as a view of a tensor in torch or a view in a
database.

A producer offers a data view by naming the dimensions of a tensor, such as
layer, sequence, head, query and key. The view then advertises its sources,
their named dimensions with their sizes, and a default data view, and a viewer
builds its picker from that. A request can index a dimension, either selecting
one position, which drops the dimension, or keeping a range of it. It can
reduce a dimension with a mean, a sum, a maximum, a minimum or a norm, lay out
one dimension of a heatmap as a mosaic, and set a budget. The producer applies
the index, then the reduction, then the mosaic and the budget, which pools a
heatmap or samples points down to the viewer's limit. A request that does not
fit the live view, such as one that leaves three dimensions for a heatmap, is
refused, and the view keeps the data view it had. The reply to a request that
fits carries the resolved data view,
with every default filled in, and so does every snapshot, so a viewer always
knows what it is drawing. A choice of layer or head in a picker, and a zoom
that needs another resolution, are requests for a different data view.

Each subscription has its own data view, so two viewers can show different
heads of one view at the same time. The producer computes each distinct data
view once per snapshot for every subscription that shares it, and a source
that no subscription selects costs nothing. The script itself sets the view's
default data view, with the `data_view` argument of the call or with
`caliper.set_data_view()`; the window and the live session draw that default,
and a new subscription starts from it. A view whose tensor has no named
dimensions still offers its payload as one source, so a viewer can range it to
zoom in, or pool it to a smaller budget.

Data views and sampling divide the work. Sampling in time, by the rate limit,
decides how often a view travels. Data views and the budget decide how much of
it travels each time.

## Latest wins, and series as deltas

A snapshot carries the newest state of one view, with a sequence number, and
the latest snapshot wins. The producer hands one data message at a time to a
connection's socket and keeps at most one pending snapshot for each
subscription; a newer snapshot replaces the pending one. A viewer that cannot
keep up therefore sees fewer frames rather than older ones, and the producer's
memory does not grow with the viewer's backlog.

A series is the exception, because a curve needs every value. A series message
is a delta: the absolute index of its first value and the values that are new
since the last message. Values that are not yet sent are coalesced into one
pending message and never dropped. The producer retains the last 1,048,576
values of each series whether or not anyone watches (the protocol asks for at
least 4096), and when a viewer
subscribes, it first sends that history and then the deltas. Because every
delta carries an absolute index, a viewer that joins late draws the same curve
as one that watched from the start.

A viewer subscribes to the views it shows, and apart from retaining series
values, a producer does no work for a view that no viewer subscribes to. A
notebook widget that shows one view costs the producer that one view and no
other.

## Training is never blocked

A view call queues its snapshot on the device and returns, as it does for the
local transports. Encoding and sending happen outside the training loop, and the
loop never waits for a viewer, for the socket or for the browser. A viewer that
falls behind loses intermediate snapshots, a viewer that disconnects loses its
subscriptions, and neither changes how fast the script trains. The calls never
read a value back to the host, so the advice of the other pages holds unchanged:
pass `loss.detach()`, not `loss.item()`.

## Cost of streaming

Streaming costs a training loop the work it adds to the thread that makes the
view calls, and the time other threads of the same process hold Python's global
interpreter lock (the GIL), which the loop needs between its torch calls. The
server therefore splits the work so that as little as possible of it runs in
Python:

| Thread | What it does | The GIL |
|---|---|---|
| The training thread | For each subscribed view that is due: reduce the snapshot on the device (pool a heatmap, sample points) and queue a non-blocking copy into pinned host memory. The views that are due take their snapshots together, in a **tick** at the session's rate of 60 a second, with the series values gathered since the last tick, and one `torch.mps.Event` recorded when the tick closes covers all of their copies | Held: per tick, about 140 µs for a 1024 by 1024 heatmap, 120 µs for 100,000 points and 100 µs for a series' values. A call between ticks returns after a clock check, in 1 to 2 µs for a series, which keeps its value, and under 1 µs for the others |
| The pump (`caliper.net`, Python) | Waits in native code. Wakes for a request from a viewer, when a tick closes, or every 4 ms while a tick's copies are in flight; polls the tick's event, which needs the GIL; hands each completed snapshot to the native sender | Held about 60 µs per snapshot. The native sender copies the snapshot with the GIL released |
| The native sender (C++, `caliper._netio`) | Listens on 127.0.0.1, checks upgrades, serves the viewer page, runs the handshake, pings and time-outs, keeps the latest snapshot per subscription and the retained series, computes heatmap and points stats from the payload (with NEON on arm64), encodes and sends | Never |

A tick opens at the first view call after it is due and closes as soon as
every subscribed view called in the last second has its snapshot, which is
right after the step's last view call. It also closes when a view it already
holds is called again, because the next step began, and at the first call
after it has been open for a second. So the view calls of one step share a tick
in any order and however far apart in the step they are, with series calls
between them. Recording an event commits torch's command buffer, so the tick
keeps the commits at 60 a second however many views a viewer watches. A view
called with a lower rate, or a subscription with a lower `max_rate`, joins only
the ticks its rate allows, every second one at 30 a second, and never exceeds
that rate.

SUBSCRIBE, UNSUBSCRIBE, VIEW_REQUEST and EVENT go to Python, because the data
view engine and the event handlers live there. They are rare, and the native
sender queues at most 256 of them per connection. In the other direction, a
connection's replies and other control messages wait in a queue of at most
65,536 messages and 16 MiB on both senders; a viewer that keeps sending requests
while it reads nothing fills it, and its connection is then closed with 1008.

When the native sender is not built, `caliper.serve()` uses a sender written in
Python, which encodes and sends on an asyncio thread that holds the GIL, and it
says so once. `python scripts/build_netio.py` builds the native sender in a few
seconds from `python/src/_netio.cpp` and `src/host/net`.

### What it costs

The table gives throughput as a percentage of a reference, for the workload of
`python/bench/bench_net.py` on MPS: a small CNN at about 2,000 steps per second
that calls its views every step, with a 1024 by 1024 weight, the loss and
100,000 3-D points with 4-class labels, at 60 snapshots per second unless the
row says otherwise. With viewers the reference is the commit-matched control
(see [Live streaming](live-streaming.md#sync)); with no viewer it is the plain
loop. Each figure is the median of 24 interleaved pairs of one-second segments
from three runs of `python/bench/bench_net_pairs.py`, with the runs' medians in
brackets and the step-time p99 against the reference's.
The runs were made with no other heavy job on the machine, at 1-minute load
averages from 1.8 to 7.8; a few pairs of the first run ran at up to 15 when
another job started, and the all-three row has only the first two runs, because
another build started during the third run's all-three arms. Each viewer is a
process of its own.

| Views called every step | No viewer | One viewer | Two viewers |
|---|---|---|---|
| 1024 by 1024 heatmap | 99.7 % (99.7, 99.6, 99.4) | 98.2 % (98.7, 97.0, 97.6), p99 690 µs against 600 | 98.1 % (98.1, 98.4, 95.5), p99 693 µs against 628 |
| Series of the loss | 192 % (172, 192, 194) | 99.0 % (100.0, 98.8, 99.0), p99 604 µs against 594 | 98.9 % (97.5, 99.5, 98.1), p99 600 µs against 586 |
| 100,000 points | 100.0 % (100.3, 100.0, 100.0) | 98.0 % (97.9, 98.4, 98.0), p99 679 µs against 637 | 97.5 % (96.3, 98.0, 96.5), p99 658 µs against 618 |
| 100,000 points at `rate=30` | 100.0 % (103.6, 99.8, 100.3) | 96.7 % (95.2, 97.1, 96.9), p99 632 µs against 590 | 96.2 % (95.5, 96.5, 96.5), p99 649 µs against 608 |
| All three | 191 % (184, 191, 191) | 97.0 % (96.5, 97.0), p99 705 µs against 581 | 95.4 % (95.4, 95.5), p99 747 µs against 700 |

So on a quiet machine one view costs this loop 1 to 2.5 % of its throughput and
all three together 3 to 4.6 %, with one viewer or two. The training thread's own
work, timed below, is 0.6 to 0.8 % of the loop's time for one view and 1.4 to
1.6 % for all three; the rest is the work of the pump, the native sender and the
viewer processes on the same machine.

The series and all-three rows with no viewer run faster than the plain loop. A
series keeps its values for viewers that come later and moves them to the host
every 64 values with an event, about 25 times a second here, and each event
commits torch's command buffer, which this workload needs to run at full speed
(the control shows the same effect). Leaving out the plain loop's long stalls
(the mean of the steps shorter than 3 ms), the series costs nothing measurable
(100.0 %) and all three views 1 % without a viewer. A heatmap or points call
with no viewer costs what it costs without a server, 0.1 µs.

A lower rate does not make the points cheaper here. At `rate=30` the points
take half the snapshots, but their ticks are then the only commits, 31 a second
against the control's 60, and this workload runs slower with fewer commits. The
default rate of the points therefore stays the session's 60 a second.

### Where the time goes

Timed part by part on the training thread with one viewer, the ticks aligned,
per tick (60 times a second), with the events recorded against the control's.
These timings come from the same benchmark in 8 to 24 interleaved rounds at
load averages of 8 or less, taken before the last two fixes to the pump and to
the ticks, which do not change this path:

| | Heatmap 1024 by 1024 | 100,000 points | Series |
|---|---|---|---|
| Reduction on the device | 45 to 51 µs: the pooling to 512 by 512 and the range of the whole tensor | 15 to 19 µs: nothing runs on the device, since the default budget is 100,000 points and nothing is sampled | |
| Copy into pinned memory | 60 to 64 µs: the payload and its range | 50 to 65 µs: the points, and the labels cast to float32 in the same copy | 83 µs: the step's values stacked and copied |
| The event and the enqueue | 16 to 18 µs | 19 to 24 µs | 21 µs |
| All of it | 140 to 156 µs | 112 to 131 µs | 104 µs per tick, and 1 to 2 µs per call between ticks |
| Events per second, against the control's | 58 against 58 | 58 against 58 | 58 against 58 |
| The pump's thread time per snapshot | 73 to 79 µs | 117 µs (331 µs before the native stats used NEON) | |

Nothing waits for the device: the only `torch.mps.synchronize` in a measured
segment is the benchmark's own at its end. To find what the points cost
beyond the training thread's share, they were measured with each part removed
in turn, before the pump's lost wake was fixed, when the full path measured
89.9 to 98.6 % (it now measures 97.5 to 98 %). With the labels already float32,
or with no labels, nothing changed. Without the copies into pinned memory the
loop measured 98.6 to 99.9 %, and without handing snapshots to the native sender
99.6 to 99.8 %, in the same rounds. So the rest is the delivery: copying 1.6 MB per snapshot out of pinned
memory, its stats, encoding, sending and decoding in the viewer process, about
96 MB a second, on other threads and processes of the same machine. The native
sender's stats were the largest part of the pump's work for the points and now
use NEON, which cut it from 331 to 117 µs per snapshot. The same variants for
the heatmap and the series stayed within the noise.

Not pursued, as later work: handing the pinned buffer itself to the native
sender, which would copy it on its own thread and free it after sending, instead
of the pump copying it; recording completed snapshots from the training thread
at the next tick, so that the pump no longer takes the GIL to poll events; a
default rate that follows a view's payload size rather than its kind; the same
ticks for the shared-memory route, whose `SnapshotRing` still records one event
per snapshot; and the sender written in Python, which was not measured again
after the ticks (before them it measured 92.7 % for the heatmap and 90.7 % for
the points with one viewer).

## Security

The server listens only on the loopback interface, 127.0.0.1, so only programs
on the same machine can connect to it. Each server makes a random **session
token**, and the stream requires it. The viewer page and its files hold no
data, so they are served without the token; the page reads the token from its
own link and passes it to the stream. The link that `caliper.serve()` prints
carries the token, so anyone who has the link can watch the views while the
server runs; it should be kept like a password. The server also checks the
`Host` header of every request and the `Origin` header of every stream
connection, so that a web page from another site, opened in a browser on the
same machine, cannot reach the stream, even by DNS rebinding.

The stream carries data and nothing else. Snapshots and series travel to the
viewer. Subscriptions, data view requests and viewer events (`select`, `pick`
and `hover`) travel back. A data view request can only name a slice, selection
or reduction of a source that the producer offered, and an event is data for
the producer's Python code to read, through `server.on_event()` or
`server.events()`. The stream never carries host commands, and nothing a viewer
sends runs as code.

This position holds until a dedicated security review, which comes before any
session leaves the machine.
Until then the server cannot listen on another interface, and the stream is not
encrypted, which is acceptable only because it never leaves the loopback
interface.

## Known limits

- **Same machine only.** The browser that shows a widget or a tab must run on
  the machine of the producer. A notebook on a remote Jupyter server cannot
  show the widgets, because the browser connects to its own 127.0.0.1.
- **Copies, not zero-copy.** Every snapshot on the network is a copy off the
  device. Data views and the budget keep it small; they do not remove it.
- **No computation and no applets in the browser.** The browser draws what a
  producer sends and computes nothing; data views are computed by the producer
  on its own device. The browser runs no applet. Applets reach it only after
  each applet is split into a computation that runs natively and an interface
  compiled to WebAssembly, which is planned.
- **A bounded heap.** WebAssembly in a browser is 32-bit, so the views can never
  hold more than 4 GB, and the module caps its heap at 512 MB
  (`web/wasm/README.md` gives the reasons). It bounds the size of the snapshots
  the views accept.
- **A second copy in the browser.** The browser copies each snapshot from
  WebAssembly memory into its GPU process. A 2048 by 2048 heatmap updated every
  frame sent 16 MB into the GPU process per frame, and the GPU queue backed up.
  Sampling and data views keep snapshots small for this reason as well.
- **ImGui is not a web page.** Caliper's views draw into one canvas. The
  pointer, the wheel, keys and text reach them while the canvas has focus, but
  the clipboard and multi-touch gestures do not. The canvas gives the browser's
  accessibility tools nothing to read, so a screen reader cannot follow it.
- **Most browsers are untested.** The browser views are tested in headless
  Chromium only, on SwiftShader and on the Mac's GPU, at device pixel ratios of
  1 and 2. Chromium-based browsers ship WebGPU, while support in Safari and
  Firefox depends on the version and the platform, and neither is tested, nor
  is Windows. Where WebGPU is absent, the views written for the browser draw
  instead.
- **Widget traits wait for an idle kernel.** A widget's `selection` and
  `data_view` traits travel over the notebook's message channel, which the
  kernel reads only while no cell runs. The picker in a widget works during
  training, because its requests go over the stream.
- **One stream over TCP.** A WebSocket delivers messages in order, so a large
  snapshot delays the messages behind it. WebTransport is planned as a later
  transport.
- **macOS only.** Caliper LIB, and with it the server, is built and
  tested only on macOS.
