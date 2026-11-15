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
view layer compiled to WebAssembly (`web/wasm`, the
[web viewer](../plans/streaming-and-embedding.md#the-web-viewer)) draws the
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
| The training thread | For each subscribed view that is due: reduce the snapshot on the device (pool a heatmap, sample points), queue a non-blocking copy into pinned host memory, record a `torch.mps.Event`. For a series: keep each value, and move gathered values to the host 30 times a second while a viewer watches | Held: about 65 µs per snapshot and 1.5 µs per series call. A call between due times returns after a clock check |
| The pump (`caliper.net`, Python) | Waits in native code. Wakes for a request from a viewer, for the training thread, or every 4 ms while snapshots are in flight; polls their events, which needs the GIL; hands each completed snapshot to the native sender | Held about 20 µs per snapshot. The native sender copies the snapshot with the GIL released |
| The native sender (C++, `caliper._netio`) | Listens on 127.0.0.1, checks upgrades, serves the viewer page, runs the handshake, pings and time-outs, keeps the latest snapshot per subscription and the retained series, computes heatmap and points stats from the payload, encodes and sends | Never |

SUBSCRIBE, UNSUBSCRIBE, VIEW_REQUEST and EVENT go to Python, because the data
view engine and the event handlers live there. They are rare, and the native
sender queues at most 256 of them per connection.

When the native sender is not built, `caliper.serve()` uses a sender written in
Python, which encodes and sends on an asyncio thread that holds the GIL, and it
says so once. `python scripts/build_netio.py` builds the native sender in a few
seconds from `python/src/_netio.cpp` and `src/host/net`.

The table gives throughput as a percentage of the commit-matched control (see
[Live streaming](live-streaming.md#sync)), for `bench_net.py`'s workload on MPS
at 60 snapshots per second: the median of three runs of ten interleaved pairs,
with the runs' medians in brackets. Each viewer is a process of its own. The
machine was shared with other builds, at load averages from 3 to 15.

| Views called every step | One viewer | Two viewers |
|---|---|---|
| 1024 by 1024 heatmap | 96.4 % (93.3 to 97.0) | 96.0 % (93.8 to 96.1) |
| Series of the loss | 96.8 % (88.9 to 98.6) | 97.6 % (97.1 to 97.7) |
| 100,000 points | 95.2 % (94.4 to 97.5) | 96.3 % (94.7 to 97.3) |
| All three | 94.3 % (94.2 to 95.7) | 92.9 % (91.3 to 93.5) |

The sender written in Python measured 92.7 % for the heatmap, 90.7 % for the
points and 85.6 % for all three with one viewer, in one run at load averages
from 8 to 15. With no viewer, a heatmap or points call costs what it costs
without a server, 0.13 µs, and the loop runs at 99.5 to 106 % of the same loop
without calls. A series call costs 0.2 µs for a number and 0.5 µs for a tensor,
because its values are kept for viewers that come later.

Most of the remaining cost has two measured parts. The training thread's own
work for the snapshots and series values takes about 2 % of the loop's time for
all three views at 1,700 steps per second, timed call by call. Each snapshot
also commits torch's command buffer with its event, about 150 commits per
second for all three views against the control's 60, and events recorded at
300 Hz instead of 60 Hz cost the control 3 % by themselves. Both grow with the
snapshots per second. The pump's wake-ups were no measurable part: polling
every 4, 8 or 12 ms gave the same throughput.

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
session leaves the machine ([Security](../plans/streaming-and-embedding.md#security)).
Until then the server cannot listen on another interface, and the stream is not
encrypted, which is acceptable only because it never leaves the loopback
interface.

## Known limits

- **Same machine only.** The browser that shows a widget or a tab must run on
  the machine of the producer. A notebook on a remote Jupyter server cannot
  show the widgets, because the browser connects to its own 127.0.0.1.
- **Copies, not zero-copy.** Every snapshot on the network is a copy off the
  device. Data views and the budget keep it small; they do not remove it.
- **No applets in the browser yet.** The browser draws views and runs no
  applet. Applets reach the browser only after the split into computation and
  interface that Phase 2 settles
  ([Applets in the browser](../plans/streaming-and-embedding.md#applets-in-the-browser)).
  [Limits in the browser](../plans/streaming-and-embedding.md#limits-in-the-browser)
  lists the limits of the web viewer itself.
- **Widget traits wait for an idle kernel.** A widget's `selection` and
  `data_view` traits travel over the notebook's message channel, which the
  kernel reads only while no cell runs. The picker in a widget works during
  training, because its requests go over the stream.
- **One stream over TCP.** A WebSocket delivers messages in order, so a large
  snapshot delays the messages behind it. WebTransport is planned as a later
  transport.
- **macOS only.** Caliper LIB, and with it the server, is built and
  tested only on macOS.
