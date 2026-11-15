# Network streaming

Network streaming sends a script's live views over a WebSocket, so that a
notebook widget, a browser tab or the hub can show them. It keeps the view calls
and the snapshots of [Live streaming](live-streaming.md), adds a transport that
copies each snapshot from the device into a socket, and adds data views, which
decide which part of a large tensor travels at all.

To use it, start with [Watch a training run in a notebook](../tutorials/watch-a-training-run-in-a-notebook.md).
The calls are in the [Python API](../references/python-api.md#serve) reference,
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
element, the notebook widget built on it, and the viewer page. Where WebGPU is
available, they draw with Caliper's own view layer compiled to WebAssembly, the
[web viewer](../plans/streaming-and-embedding.md#the-web-viewer). Where it is
absent, small views written for the browser in TypeScript draw with Canvas 2D.

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
pending message and never dropped. The producer retains at least the last 4096
values of each series whether or not anyone watches, and when a viewer
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

## Security

The server listens only on the loopback interface, 127.0.0.1, so only programs
on the same machine can connect to it. Each server makes a random **session
token**, and both the viewer page and the stream require it. The link that
`caliper.serve()` prints carries the token, so anyone who has the link can
watch the views while the server runs; it should be kept like a password. The
server also checks the `Host` and `Origin` headers of every connection, so that
a web page from another site, opened in a browser on the same machine, cannot
reach the stream, even by DNS rebinding.

The stream carries data and nothing else. Snapshots and series travel to the
viewer. Subscriptions, data view requests and viewer events (`select`, `pick`
and `hover`) travel back. A data view request can only name a slice, selection
or reduction of a source that the producer offered, and an event is data for
the producer's Python code to read. The stream never carries host commands, and
nothing a viewer sends runs as code.

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
- **macOS only.** The Python package, and with it the server, is built and
  tested only on macOS.
