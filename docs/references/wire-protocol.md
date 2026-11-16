# Wire protocol v3

Protocol v3 carries live data from a training process to viewers over one WebSocket: to a browser tab, a Jupyter widget, the hub and the native Caliper app. From version 3.1 it also carries a viewer's controls back to the script. This page specifies it completely. The header `include/caliper/live_net.h` defines its constants and frame header, and is [embedded at the end of this page](#the-header-verbatim). The folder `tests/fixtures/wire_v3` holds [golden frames](#golden-frames) that every implementation decodes in its tests.

!!! note "Status"
    Version 3.1, which adds [controls](#controls) and [linked selection](#linked-selection) to 3.0. It is used on the loopback interface only, with a session token, until a dedicated security review. [Network streaming](../explanations/network-streaming.md) explains the design decisions behind it.

Protocol v2 ([`caliper/live.h`](../explanations/live-streaming.md), shared memory) is unchanged. v2 stays the zero-copy path between two processes on the same machine. v3 sends copies of reduced snapshots over a socket. A producer reduces each snapshot on its own device before it sends it, so the copy is small.

## Roles and terms

| Role | What it is | What it does |
|---|---|---|
| Producer | The training process, through `caliper.serve()` in Python | Owns the tensors, computes every data view on its own device, and sends snapshots and series values |
| Viewer | A browser element, the Jupyter widget, or the native app's `NetSource` | Draws what it receives. It computes nothing |
| Hub | `caliper serve`, a subcommand of the `caliper` binary | Accepts producers and viewers, holds several sessions and forwards between them |

The side that opens a connection is its **client**, and the side that accepts it is its **server**. There are three kinds of connection:

| Connection | Client | Server |
|---|---|---|
| Viewer to producer | A viewer | `caliper.serve()` |
| Viewer to hub | A viewer | The hub |
| Producer to hub | A producer | The hub |

`caliper.serve(port=0)` listens on 127.0.0.1 and prints `http://127.0.0.1:<port>/?token=<token>`. The viewer page reads the token from its own address and connects to `ws://<same host>/v3/stream?token=<token>`. A viewer written for `caliper.serve()` works against the hub unchanged, because a server that holds exactly one session attaches a new viewer to it.

These pages use the following terms with one meaning each.

- A **session** is the stream of one producer process. A **session id** such as `clp-4242-a1b2c3` names it in lists. The session id is not secret. (Protocol v2 calls the same string the session token.)
- The **session token** is the secret of one server. A client must present it to open a connection.
- A **view** is one named stream of a producer, such as `fc1.weight` or `loss`, of one kind.
- A **live view** decides how the viewer draws a view: heatmap, image, points, series or histogram. The kind of a view names its live view.
- A **data view** decides which data travels: a source, an index selection, a reduction and a size budget. [Data views](#data-views) defines it.
- A **subscription** is a viewer's request for one view under one data view. A viewer chooses its number, `sub`.
- A **snapshot** is the reduced payload of a view at one moment, sent for one subscription.

## Opening a connection

### The upgrade request

A client opens a connection with a WebSocket upgrade (RFC 6455) of `GET /v3/stream?token=<session token>`. The session token travels as the query parameter `token` because a browser cannot set headers on a WebSocket. The server makes these checks before it upgrades, in this order:

| Check | Rule | Answer when it fails |
|---|---|---|
| Path | The path is exactly `/v3/stream` | HTTP 404 |
| Host | The `Host` header is `127.0.0.1:<port>` or `localhost:<port>`, where `<port>` is the port the server listens on and the host name is compared ASCII case-insensitively. When the server listens on port 80, the forms without a port are accepted too. Anything else could be a DNS rebinding attack | HTTP 403 |
| Origin | The `Origin` header is absent or accepted by the rule below | HTTP 403 |
| Token | The `token` parameter is present and equal to the server's session token, compared in constant time | HTTP 403 |

The `Origin` header is accepted when its value is one of these:

1. The literal `null`, which sandboxed frames and some notebook front ends send.
2. An origin `<scheme>://<host>` or `<scheme>://<host>:<port>` whose scheme is `http` or `https` and whose host is `127.0.0.1`, `localhost` or `[::1]`, compared ASCII case-insensitively, with any port. This covers the server's own page at `http://127.0.0.1:<port>` and notebook front ends on loopback, such as JupyterLab at `http://localhost:8888`.
3. `vscode-webview://` followed by any authority, the origin of the webviews in which VS Code shows notebook output.

Any other value, including an origin with a path or user information, is refused. A missing `Origin` means a client that is not a browser, such as the Python producer or the native app. When the origin is `null` or absent, the session token is the only protection.

The server answers a refused upgrade with the status above and closes the TCP connection. It never logs the token. The upgrade uses no subprotocol: a client sends no `Sec-WebSocket-Protocol` header, and the server declines `permessage-deflate`. Every message after the upgrade is binary. A text message makes the receiver close with code 1003.

The server may serve plain HTTP on the same port, such as the viewer page at `/`. Those requests are outside this specification, but they get the same Host check.

A session token is 22 to 128 characters of `A-Z`, `a-z`, `0-9`, `_` and `-`, made from at least 128 random bits, for example by Python's `secrets.token_urlsafe(16)`. The viewer page should remove the token from its address bar once it has read it.

### Hello and welcome

After the upgrade the client sends HELLO, and the server answers WELCOME. Neither side sends anything else before them, except BYE. A server closes with code 4003 when no HELLO arrives within 10 seconds, and a client closes with 4003 when no WELCOME arrives within 10 seconds of its HELLO. Each side learns the other's minor version from the header of HELLO or WELCOME.

| Server | Accepts | Refuses |
|---|---|---|
| `caliper.serve()` | Viewers | Producers, with close code 4002 |
| The hub | Viewers and producers | Nothing by role |

A server that has reached its connection limit closes with 4005 after the HELLO. Either side may send an ERROR with `fatal` true just before such a close, to explain it.

### Choosing a session

Right after WELCOME on a viewer connection, the server sends SESSIONS, the list of the sessions it holds, and sends it again whenever that list changes. `caliper.serve()` lists exactly one session, its own.

A viewer that has not sent ATTACH follows the server's default: whenever it is not attached to a live session and the server holds exactly one live session, the server attaches it to that session. This is how a viewer of `caliper.serve()` is attached right after WELCOME, and how a viewer of the hub follows a notebook kernel that restarts. A viewer that sends ATTACH chooses for itself from then on.

ATTACH names a session id. The server answers with VIEWS for that session, which confirms the attachment, or with ERROR `unknown_session`, which leaves the viewer as it was. ATTACH with `session` null detaches: the server ends the viewer's subscriptions and sends VIEWS with `session` null and no views. Attaching to another session ends every subscription of the previous one.

When an attached session ends, the hub keeps it in the list with state `ended`, and the viewer keeps the last frames it drew. When the producer of `caliper.serve()` ends, it closes every connection with BYE `shutdown` and close code 1001.

### The producer connection to a hub

A producer that connects to a hub sends HELLO with role `producer` and its session identity: `id`, `script`, `pid` and `start_unix_ms`. The hub keeps the proposed id unless a live session of another process holds it. In that case it appends `-2`, `-3` and so on. WELCOME carries the identity with the id the hub chose. A producer that reconnects with the same id, pid and start time resumes its session: the hub closes the old connection, if it is still open, with BYE `replaced` and code 1000, and its viewers stay attached.

After WELCOME the producer sends VIEWS. On this connection the hub acts as the producer's one viewer: it sends SUBSCRIBE, UNSUBSCRIBE, VIEW_REQUEST and EVENT, and the producer answers with VIEW_REPLY, SNAPSHOT and SERIES. The hub:

- rewrites `sub` and `request_id` between its viewer connections and its producer connection;
- shares one producer subscription among viewers whose resolved data views are equal, at the largest `max_rate` among them;
- keeps the latest snapshot of every producer subscription and the retained series values, so that a viewer that subscribes late gets them at once;
- forwards every viewer's events to the producer;
- from 3.1, forwards the controls of the viewers it allows to the producer and the producer's CONTROLS and SELECTION to the session's viewers, as [Controls through a hub](#controls-through-a-hub) describes;
- resubscribes after a producer resumes its session.

### Staying connected and closing

A peer that has received nothing for 10 seconds sends PING, and the other side answers with PONG carrying the same nonce. A peer that has received nothing for 30 seconds closes with code 4004. Browsers cannot send WebSocket control pings, which is why the protocol has its own.

To close, a peer sends BYE with a reason and then closes the WebSocket with the [close code](#close-codes) that matches it: 1000 for `normal` and `replaced`, 1001 for `shutdown`, and the code of the error for `error`.

## Frames

Every WebSocket message is one binary **frame**. All integers in it are little-endian.

### The frame layout

| Offset | Size | Type | Field | Rule |
|---|---|---|---|---|
| 0 | 4 | u32 | `magic` | `0x574C4143`, the bytes `C A L W` |
| 4 | 2 | u16 | `major` | 3 |
| 6 | 2 | u16 | `minor` | The sender's minor version: 1 in 3.1, 0 in 3.0 |
| 8 | 2 | u16 | `type` | The [message type](#message-types) |
| 10 | 2 | u16 | `flags` | Frame flags. 3.1 defines none, so senders write 0 |
| 12 | 4 | u32 | `frame_len` | Bytes of the whole frame, this header included |
| 16 | 4 | u32 | `meta_len` | Bytes of the metadata |
| 20 | 2 | u16 | `section_count` | Payload sections, 0 to 8 |
| 22 | 2 | u16 | `reserved` | 0 when sent, ignored when received |
| 24 | 4 × `section_count` | u32 each | `section_len[i]` | Bytes of section i |
| 24 + 4 × `section_count` | `meta_len` | UTF-8 | metadata | One JSON object |
| See below | `section_len[i]` | bytes | section i | A raw array |

Section i starts at the first offset that is a multiple of 8, counted from the first byte of the frame, at or after the end of the part before it. The part before section 0 is the metadata. The sender writes zero bytes in the gaps, and the receiver ignores them. `frame_len` is the end of the last section, or the end of the metadata when there is no section, and no bytes follow. Because every section starts at a multiple of 8, a browser that receives the frame as an `ArrayBuffer` can wrap a section in a `Float32Array` or `Uint8Array` without a copy.

### Metadata

The metadata is one JSON object (RFC 8259) in UTF-8, at least 2 bytes long, with no byte order mark and with objects and arrays nested at most 16 deep. It holds no `NaN` or `Infinity`, which JSON cannot express; a value that may be non-finite is written as `null`. A sender never repeats a key in an object. An optional key whose value is `null` means the same as an absent key.

Integers are exact in every implementation up to 2<sup>53</sup>. The only field that can exceed that is `stamp_ns`, a monotonic clock in nanoseconds, which passes 2<sup>53</sup> after about 104 days of uptime. A JavaScript reader then loses a few nanoseconds of it, which only affects latency display.

A receiver accepts keys in any order and any whitespace. The Python encoder and the golden frames write the canonical form: no whitespace, keys sorted, and non-ASCII characters as raw UTF-8. Encoders in other languages need not produce the same bytes.

### Sections

When `section_count` is not 0, the metadata key `sections` is an array with one descriptor per section, in order. The key `sections` is reserved in every message.

| Key | Type | Meaning |
|---|---|---|
| `name` | string | The section's name, unique in the frame. Each message names its sections |
| `dtype` | integer | A dtype id from the table below |
| `shape` | array of integers | The array's shape, row-major |
| `encoding` | integer, optional | 0, a raw array (the default). 1, Arrow IPC, is reserved for a future table kind and not used in 3.0 |

A raw section holds the product of `shape` times the item size bytes, little-endian, row-major.

| Id | Name | Item size | numpy | TypedArray |
|---|---|---|---|---|
| 0 | `f32` | 4 | `<f4` | `Float32Array` |
| 1 | `f16` | 2 | `<f2` | `Float16Array` where available |
| 2 | `bf16` | 2 | none | `Uint16Array` (raw bits) |
| 3 | `i64` | 8 | `<i8` | `BigInt64Array` |
| 4 | `i32` | 4 | `<i4` | `Int32Array` |
| 5 | `u8` | 1 | `u1` | `Uint8Array` |

The ids are the values of `CaliperDType` (`caliper/tensor.h`). Producers of 3.1 send only `f32` and `u8` payloads. The other ids are defined so that a receiver can name what it refuses.

### Decoding a frame

A receiver applies these checks in this order, so that every implementation rejects a bad frame with the same close code:

| Step | Check | Close code |
|---|---|---|
| 1 | Fewer than 8 bytes | 4000 |
| 2 | `magic` is not `0x574C4143` | 4000 |
| 3 | `major` is not 3 | 4001 |
| 4 | Fewer than 24 bytes | 4000 |
| 5 | `frame_len` over 64 MiB, `meta_len` over 1 MiB, or `section_count` over 8 | 1009 |
| 6 | The message length is not `frame_len` | 4000 |
| 7 | The layout does not end exactly at `frame_len` | 4000 |
| 8 | The metadata is not a JSON object as described above | 4000 |
| 9 | `sections` is missing or has another count, a descriptor is malformed, or a raw section of a known dtype has a length that disagrees with its shape | 4000 |

`minor`, `flags`, `reserved` and the padding bytes are never grounds to reject a frame. A frame that passes every check is **well formed**. A well-formed frame whose message is invalid, such as one with a missing field, an unknown view or a bad data view, is answered with an ERROR message, and the connection stays open. A section with an encoding or dtype the receiver does not support makes it ignore the message and answer ERROR `unsupported`.

## Message types

| Id | Name | Direction | Sections | Answer |
|---|---|---|---|---|
| `0x0001` | `hello` | client to server | none | `welcome` |
| `0x0002` | `welcome` | server to client | none | |
| `0x0003` | `bye` | either | none | |
| `0x0004` | `error` | either | none | |
| `0x0005` | `ping` | either | none | `pong` |
| `0x0006` | `pong` | either | none | |
| `0x0010` | `sessions` | server to viewer | none | |
| `0x0011` | `attach` | viewer to server | none | `views` or `error` |
| `0x0020` | `views` | producer to viewer or hub, hub to viewer | none | |
| `0x0021` | `subscribe` | viewer to producer or hub, hub to producer | none | `view_reply` |
| `0x0022` | `unsubscribe` | viewer to producer or hub, hub to producer | none | |
| `0x0023` | `view_request` | viewer to producer or hub, hub to producer | none | `view_reply` |
| `0x0024` | `view_reply` | producer to viewer or hub, hub to viewer | none | |
| `0x0030` | `snapshot` | producer to viewer or hub, hub to viewer | `data`, then optionally `aux` | |
| `0x0031` | `series` | producer to viewer or hub, hub to viewer | `values` | |
| `0x0040` | `event` | viewer to producer or hub, hub to producer | none | |
| `0x0050` | `controls` | producer to viewer or hub, hub to viewer | none | |
| `0x0051` | `control` | viewer to producer or hub, hub to producer | none | `control_reply` |
| `0x0052` | `control_reply` | producer to viewer or hub, hub to viewer | none | |
| `0x0060` | `selection` | producer to viewer or hub, hub to viewer | none | |

The types from `0x0050` are additions of 3.1. A sender sends CONTROLS, CONTROL and CONTROL_REPLY only to a peer that announced the capability `controls`, and SELECTION only to a peer that announced `selection` ([Versioning](#versioning)). Type 0 is never sent. A known message sent in the wrong direction, or before HELLO and WELCOME, is a protocol error (4000). A message of an unknown type is ignored.

Each message below gives the header fields that depend on it and the keys of its metadata. Keys marked optional may be absent. Bytes 0 to 7 and 10 to 23 of the header follow [the frame layout](#the-frame-layout) in every message.

### hello

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0001` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `role` | string | `viewer` or `producer` |
| `capabilities` | array of strings, optional | The [capabilities](#versioning) the client has. Default: none. A 3.1 viewer that shows controls announces `controls`, and one that draws linked selections announces `selection` |
| `software` | object, optional | `{"name", "version"}` of the client, for display and logs |
| `session` | object, required for a producer | The producer's session identity: `id` (1 to 63 characters of `A-Z`, `a-z`, `0-9`, `.`, `_`, `-`), `script` (the script's name), `pid`, and `start_unix_ms` (wall-clock milliseconds when the process started streaming) |

### welcome

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0002` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `role` | string | `producer` (for `caliper.serve()`) or `hub` |
| `capabilities` | array of strings, optional | The server's capabilities. A producer that reads viewer events announces `events`. A 3.1 producer or hub announces `controls` and `selection` for what it implements |
| `software` | object, optional | `{"name", "version"}` of the server |
| `session` | object, optional | From a producer, its own session identity. From a hub to a producer, the identity with the id the hub chose. Absent from a hub to a viewer |
| `control` | boolean, optional | 3.1, sent to a client that announced `controls`. To a viewer: true when the server accepts CONTROL from this connection. From a hub to a producer: true when the hub forwards its viewers' controls. Absent means false |

### bye

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0003` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `reason` | string | `normal` (done), `shutdown` (the process is exiting), `replaced` (a producer resumed its session on a new connection) or `error` (after a fatal error) |
| `message` | string, optional | Human-readable text |

The sender closes the WebSocket right after BYE.

### error

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0004` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `code` | string | An [error code](#error-codes) |
| `message` | string, optional | Human-readable text |
| `fatal` | boolean, optional | True when the sender closes the connection next. Default false |
| `in_reply_to` | integer, optional | The type of the message that caused the error |
| `request_id`, `sub`, `view`, `session` | optional | The fields of that message that identify what failed |

### ping and pong

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0005` for ping, `0x0006` for pong |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `nonce` | integer (uint32) | Chosen by the sender of PING and echoed by PONG |

### sessions

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0010` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `sessions` | array of objects | Every session the server holds |

Each session object has the identity keys `id`, `script`, `pid` and `start_unix_ms`, and also:

| Key | Type | Meaning |
|---|---|---|
| `state` | string | `live`, or `ended` once its producer disconnected without resuming |
| `views` | integer | The number of views the session has |
| `capabilities` | array of strings, optional | The producer's capabilities |

### attach

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0011` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `session` | string or null | The session id to attach to, or null to detach. The key is required |

### views

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0020` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `session` | string or null | The session id. Required on a viewer connection, where null means not attached. A producer may omit it on its connection to a hub |
| `views` | array of objects | Every view of the session, at most 256 |

The producer sends the full list again whenever it changes: a view is added or removed, or a source changes its dims. A newer list replaces an unsent older one. Each view object:

| Key | Type | Meaning |
|---|---|---|
| `id` | integer | Never reused within the session. A view re-created with the same name gets a new id |
| `name` | string | 1 to 63 bytes, unique in the session |
| `kind` | integer | The live view: 1 heatmap, 2 image, 3 points, 4 series, 5 histogram (the values of `CaliperLiveKind`) |
| `sources` | array of objects | What the view's data views can draw from: see [What a view offers](#what-a-view-offers). Empty for a series |
| `default` | object, optional | The view's default data view. Absent means `{}` |
| `items` | string, optional | 3.1, series only: the [item space](#item-spaces) that the series' absolute index numbers |

When a view disappears from the list or its id changes, the producer ends the subscriptions that named it. A viewer subscribes again if it still wants the view.

### subscribe

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0021` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `sub` | integer (uint32) | Chosen by the viewer. A `sub` that is already open is replaced, as if unsubscribed first |
| `view` | string | The view's name |
| `request_id` | integer (uint32) | Chosen by the viewer, echoed by VIEW_REPLY |
| `max_rate` | number, optional | Messages per second at most for this subscription. Default 60. Values over 240 are treated as 240. It must be positive |
| `data_view` | object, optional | The requested [data view](#data-views). Absent or `{}` means the view's default |

The producer answers with VIEW_REPLY. On success the subscription is open and VIEW_REPLY carries the resolved data view. On failure VIEW_REPLY carries an error and no subscription exists. A connection holds at most 256 subscriptions; one more is answered with the error `limit`.

### unsubscribe

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0022` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `sub` | integer (uint32) | The subscription to end. An unknown `sub` is ignored |

The producer discards the pending snapshot and series values of the subscription.

### view_request

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0023` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer (uint32) | Chosen by the viewer, echoed by VIEW_REPLY |
| `sub` | integer (uint32) | An open subscription |
| `view` | string | Its view's name |
| `data_view` | object | The new data view |

A zoom, a pan at another resolution, or a choice of layer, head or slice is a view request. On success the subscription switches to the new data view. On failure it keeps the data view it had. Snapshots made under the old data view may still arrive after the reply. Each snapshot names its data view, so the viewer can tell them apart.

### view_reply

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0024` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer | The `request_id` of the SUBSCRIBE or VIEW_REQUEST it answers |
| `sub` | integer | The subscription |
| `view` | string | Its view's name |
| `data_view` | object | On success, the resolved data view |
| `error` | object | On failure, `{"code", "message"}`, with a code from the [error codes](#error-codes): `unknown_view`, `unknown_subscription`, `bad_data_view` or `limit` |

A reply carries exactly one of `data_view` and `error`.

### snapshot

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0030` |
| Bytes 20 to 21, `section_count` | 1, or 2 with `aux` |
| Section `data` | The payload, by kind in the table below |
| Section `aux` | Points only, optional: `f32` `(N,)`, one label or value per point |

| Key | Type | Meaning |
|---|---|---|
| `sub` | integer | The subscription |
| `view` | string | The view's name |
| `seq` | integer | The producer's snapshot counter of the view. It grows with every snapshot of the view under any subscription, so a subscription sees it increase with gaps |
| `stamp_ns` | integer | The producer's monotonic clock in nanoseconds at the call that made the snapshot (Python's `time.monotonic_ns()`) |
| `kind` | integer | The view's kind |
| `data_view` | object | The resolved data view that produced the snapshot |
| `axes` | array of objects | How the payload maps onto the source: see [Payload axes](#payload-axes). Empty for a histogram |
| `tile` | object, optional | For a tiled heatmap, `{"dim", "rows", "cols"}` |
| `colormap` | integer, optional | 0 viridis, 1 magma, 2 rdbu (the values of `CaliperColormap`). Heatmaps, and points with continuous values. Default 0 |
| `flags` | integer, optional | 1: the low end of the range is `stats[0]`. 2: the high end is `stats[1]`. 4: draw a heatmap or image one texel per pixel at the origin (tests). The values of `CALIPER_LIVE_SNAP_*`. Default 0 |
| `vmin`, `vmax` | numbers, optional | The fixed ends of the range, used for an end whose flag is not set. Default 0 and 1 |
| `stats` | array, optional | At most 64 numbers computed on the producer's device, by kind below. A non-finite value is `null` |
| `points_dims` | integer | Points only: 2 or 3, equal to the second dimension of `data` |
| `label_count` | integer | Points only: more than 0 when `aux` holds categorical labels 0 to `label_count` − 1, 0 when `aux` holds values to colormap |

| Kind | `data` | `stats` |
|---|---|---|
| Heatmap | `f32` `(H, W)` | `[min, max]` of the payload |
| Image | `u8` `(H, W, 4)`, RGBA | Empty |
| Points | `f32` `(N, D)`, D = `points_dims` | `[aux min, aux max, x min, x max, y min, y max, z min, z max]` over the points sent; z is 0 for 2-D points |
| Histogram | `f32` `(B,)`, bin counts | `[lo, hi]`, the range the bins cover |

A viewer draws a range end from `stats` when its flag is set. As in protocol v2, a non-finite range from `stats` keeps the last finite one, a flat range widens by 0.5 on each side, and a NaN value draws as colormap index 0. Categorical labels draw with the fixed 16-colour palette, label L in colour L mod 16. Unlike protocol v2, 2-D points travel as `(N, 2)` rather than padded to `(N, 3)`.

### series

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0031` |
| Bytes 20 to 21, `section_count` | 1 |
| Section `values` | `f32` `(M,)`, 1 ≤ M ≤ 262,144 |

| Key | Type | Meaning |
|---|---|---|
| `sub` | integer | The subscription |
| `view` | string | The view's name |
| `start` | integer | The absolute index of the first value: the number of values appended to the series before it |
| `total` | integer | The number of values appended so far. `start` + M ≤ `total` |
| `stamp_ns` | integer | The producer's monotonic clock at the newest append |

A series travels only as SERIES messages, never as snapshots. On subscribe the producer first sends the values it retains, the backfill, in as many messages as needed, and from then on the new values. The x axis of a series is its absolute index. A viewer stores values by index: a message that starts past the end it holds leaves a gap, and one that overlaps values it holds replaces them. A series view that is re-created gets a new view id, and its indices start at 0 again.

### event

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0040` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `view` | string | The view's name |
| `type` | string | `select`, `pick` or `hover` |
| `detail` | object | What was selected, picked or hovered, by kind in [Events](#events) |
| `sub` | integer, optional | The subscription the viewer was drawing |
| `seq` | integer, optional | The `seq` of the snapshot the viewer was showing |
| `data_view` | object, optional | That snapshot's data view |

### controls

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0050` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `run` | object | The run state, below |
| `params` | array of objects | Every parameter the script registered, at most 64, in the order of registration |
| `actions` | array of objects | Every action the script registered, at most 64, in the order of registration |

The producer sends CONTROLS right after the VIEWS that confirms an attachment, and sends it again, whole, whenever the run state, a parameter or an action changes. A script that registered nothing and called no checkpoint still sends one, with the state `none` and empty lists. [Controls](#controls) gives the rules behind every field.

| `run` key | Type | Meaning |
|---|---|---|
| `state` | string | `none`, `running`, `pausing`, `paused` or `stepping`: see [Pause, resume and step](#pause-resume-and-step) |
| `step` | integer, optional | The script's step number at the checkpoint where the state last changed |
| `steps_left` | integer, optional | While `stepping`, the steps the loop still runs before it pauses |

| Parameter key | Type | Meaning |
|---|---|---|
| `name` | string | 1 to 63 bytes, unique among the parameters |
| `type` | string | `number`, `integer`, `boolean` or `choice` |
| `value` | number, boolean or string | The value the script reads now |
| `pending` | as `value`, optional | A value accepted from a viewer that the script reads from the next checkpoint on |
| `min`, `max` | numbers, optional | `number` and `integer`: the range a value is clamped to; integers for an `integer` |
| `log` | boolean, optional | `number` and `integer`: draw the control on a log scale. Requires `min` greater than 0 and `max` |
| `choices` | array of strings | `choice` only: 1 to 64 distinct strings of 1 to 63 bytes |
| `help` | string, optional | At most 1,024 bytes, for a tooltip |

| Action key | Type | Meaning |
|---|---|---|
| `name` | string | 1 to 63 bytes, unique among the actions |
| `running` | boolean | The action runs now |
| `queued` | boolean | A run of the action waits for a checkpoint |
| `help` | string, optional | At most 1,024 bytes, for a tooltip |

### control

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0051` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer (uint32) | Chosen by the viewer, echoed by CONTROL_REPLY |
| `op` | string | `pause`, `resume`, `step`, `set` or `run` |
| `name` | string | `set`: the parameter's name. `run`: the action's name. Matched exactly, byte for byte |
| `value` | number, boolean or string | `set` only: the new value |
| `count` | integer (uint32), optional | `step` only: how many steps the loop runs before it pauses, at least 1. Default 1 |

A receiver ignores the keys that do not belong to the op. The key `args` is reserved for typed action arguments in a later version: a viewer sends it only to a producer that announced a capability for it, and 3.1 defines none. A viewer sends CONTROL only as [Who may control](#who-may-control) allows, and at most 30 `set` requests per second for one parameter, for example while a slider is dragged, ending with the value where the drag stops. A producer or a hub answers a CONTROL from a peer that did not announce `controls` in its HELLO with an ERROR `bad_request`, which names the request in `in_reply_to` and `request_id`, and keeps the connection.

### control_reply

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0052` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer | The `request_id` of the CONTROL it answers |
| `op` | string | That CONTROL's `op` |
| `status` | string | `ok`, `queued` or `error` |
| `name` | string, optional | That CONTROL's `name` |
| `value` | number, boolean or string, optional | `set` with `ok`: the value the producer accepted, after clamping |
| `message` | string, optional | `run` with `ok`: the action's result text, at most 1,024 bytes |
| `error` | object | With the status `error` only: `{"code", "message"}`, with a code from the [error codes](#error-codes) |

The producer answers each CONTROL with one CONTROL_REPLY, to its sender only. The exception is an accepted `run`: it is answered `queued` at once and later once more, `ok` when the action returned, or `error` with `action_failed`, `timeout` or `unknown_control`. A refused CONTROL is answered `error` at once and changes nothing. A malformed CONTROL, such as one without `request_id`, is answered with an ERROR `bad_request`, as any malformed message is.

### selection

| Part | Content |
|---|---|
| Bytes 8 to 9, `type` | `0x0060` |
| Bytes 20 to 21, `section_count` | 0 |

| Key | Type | Meaning |
|---|---|---|
| `space` | string | The [item space](#item-spaces), 1 to 63 bytes |
| `items` | array of integers | The selected item ids, ascending and without repeats, at most 32,768. Empty when nothing is selected |
| `truncated` | boolean, optional | True when the selection was cut to its first 32,768 items. Default false |
| `view` | string, optional | The view in which a viewer made the selection. Absent when the script set it |
| `details` | object, optional | The script's fields of the first selected items: see [Item details](#item-details) |
| `details_error` | string, optional | Why `details` is missing although the script provides details for the space, at most 1,024 bytes |

After the CONTROLS of an attachment, or after its VIEWS for a viewer without `controls`, the producer sends one SELECTION for each item space whose selection is not empty. It sends SELECTION to every viewer whenever a space's selection or its details change.

## Data views

A data view is a slice, selection, projection or aggregation of a source that the producer computes on its device. Only the result travels. A viewer requests a data view with SUBSCRIBE or VIEW_REQUEST, the producer resolves it and answers with the resolved form, and every snapshot names the resolved data view that produced it. The schema is deliberately small: it names a source, indexes dimensions, reduces dimensions, and sets a size budget.

### What a view offers

Each view in VIEWS lists its sources, so that a viewer can build a picker for layers, heads and slices without guessing.

| Key | Type | Meaning |
|---|---|---|
| `name` | string | The source's name, 1 to 63 bytes, unique in the view, such as a layer or a parameter |
| `dims` | array of objects | The source's dimensions in order, at most 8. Each is `{"name", "size"}`, with a name of 1 to 63 bytes, unique in the source. In 3.1 a dim may also have `items`, the [item space](#item-spaces) whose items it indexes |
| `dtype` | string, optional | The source tensor's element type, for display: `f32`, `f16`, `bf16`, `f64`, `i64`, `i32`, `i16`, `i8`, `u8` or `bool` |

A view has at most 64 sources. The dims of an image source are its height and width; the RGBA channels are not a dim. The dims of a points source end with the point dim; the coordinates are not a dim. A series view has no sources.

### The data view object

| Key | Type | Meaning |
|---|---|---|
| `source` | string | The source's name. Default: the view's first source |
| `index` | object | Per dim: an integer, which selects one position and drops the dim, or `[start, stop]` or `[start, stop, step]`, which selects a range and keeps the dim |
| `reduce` | object | Per dim: `mean`, `sum`, `max`, `min` or `norm` (the square root of the sum of squares), which reduces the dim away |
| `tile` | string | Heatmaps only: a kept dim to lay out as a mosaic of cells |
| `budget` | object | The size budget, by kind below |

An index counts from 0. A negative index or bound counts from the end of the dim, as in Python. In a range, `start` and `stop` may be `null`, meaning the start and the end of the dim, the bounds are clamped to the dim, and `step` is at least 1.

| Budget key | Kinds | Default | Limit | Meaning |
|---|---|---|---|---|
| `max_extent` | Heatmap, image | 512 | 4096 | The most payload rows and the most payload columns |
| `pool` | Heatmap | `mean` | | How cells are pooled to meet `max_extent`: `mean`, `max` or `min`. Images are always pooled by `mean` |
| `max_points` | Points | 100,000 | 1,000,000 | The most points in the payload |
| `max_backfill` | Series | All retained values | The producer's retention | The most retained values sent on subscribe; 0 sends none |

A budget value under 1 (under 0 for `max_backfill`) is an error, and a value over the limit is treated as the limit.

### Resolving a data view

The producer resolves a requested data view as follows.

1. Members the request leaves out come from the view's `default`. A member present in the request replaces the default's member as a whole, except `budget`, whose keys are taken one by one from the request, then from the default, then from the table above. So `{}` means the view's default, and `{"index": {}}` means the default without its indexing.
2. The source must exist, and every dim named in `index`, `reduce` and `tile` must be one of its dims.
3. An integer index must lie inside its dim, and a range must select at least one position.
4. A dim indexed by an integer cannot be reduced, and the tile dim must be kept.
5. The dims that remain, the **kept dims**, must fit the kind: 2 for a heatmap (3 with `tile`), 2 for an image, 1 for points, any number for a histogram. A series takes no `index`, `reduce` or `tile`.
6. The payload must not exceed 4096 rows or columns.

The resolved data view has the same members as a request, with every default filled in: `source`; `index` with integers made non-negative and every range written as `[start, stop, step]` with non-negative clamped bounds; `reduce`; `budget` with the keys of its kind (heatmap `max_extent` and `pool`, image `max_extent`, points `max_points`, histogram none); and `tile` when the heatmap tiles. A series resolves to `{"budget": {"max_backfill": n}}`. A data view that breaks a rule is answered with the error `bad_data_view` and a message that names the rule.

The producer computes the data view on its device in this order: it takes the source, applies `index`, applies `reduce` one dim at a time in the source's dim order, lays out the tile mosaic, and then meets the budget by pooling (heatmap and image) or sampling (points). Points are sampled with a stride, so the sample stays the same while the source's size does not change.

### Payload axes

`axes` in a snapshot lists the kept dims in payload order, one object each. For a tiled heatmap the tile dim comes first, then the cell's rows and columns. For other kinds the kept dims keep their source order.

| Key | Meaning |
|---|---|
| `dim` | The source dim |
| `start`, `step` | The selected range of the dim: position p is source index `start + p × step` |
| `count` | The number of selected positions |
| `size` | The payload extent along this axis (for the tile axis, the number of cells) |

Payload index i stands for the selected positions from `floor(i × count / size)` up to, but not including, `ceil((i + 1) × count / size)`. A pooled axis aggregates those positions, and a sampled axis (the point dim) takes the first of them. This is the rule of torch's adaptive pooling, so a viewer can map a hovered texel or a picked point back to source indices.

A tiled heatmap of C cells has `cols = ceil(sqrt(C))` and `rows = ceil(C / cols)`. Each cell is pooled to `min(H, max(1, floor(max_extent / rows)))` rows and `min(W, max(1, floor(max_extent / cols)))` columns. Cell t is at row `t div cols` and column `t mod cols`, and the cells left over are filled with NaN. This is the channel mosaic of `caliper.show()`.

### Data views of each kind

| Kind | Kept dims | Payload | Budget |
|---|---|---|---|
| Heatmap | 2 (rows, columns), or 3 with `tile` | `f32` `(H, W)` after pooling | `max_extent`, `pool` |
| Image | 2 (height, width) | `u8` `(H, W, 4)` after mean pooling | `max_extent` |
| Points | 1 (the point dim) | `f32` `(N, D)` after stride sampling, with `aux` sampled alike | `max_points` |
| Histogram | Any | The counts of every element of the result over its range, binned on the device | none |
| Series | No sources | SERIES messages | `max_backfill` |

## Events

An EVENT reports what a person did in a viewer: a selection, a pick or a hover. Python code in the producer reads events. Events are data, never commands: no event makes a producer run code. The stream never carries host commands or the command layer. A producer that does not read events lacks the capability `events` and ignores them.

Coordinates in `detail` are source indices, computed from the snapshot's [payload axes](#payload-axes). Ranges are half-open, `[start, stop]`.

| Kind | `pick` and `hover` detail | `select` detail |
|---|---|---|
| Heatmap | `row`, `col`: source indices of the first position of the cell. `tile`: the tile dim's source index, for a tiled heatmap. `value`, optional: the payload value | `rows`, `cols`: `[start, stop]` ranges of source indices. `tile` as for pick |
| Image | `row`, `col` as for a heatmap | `rows`, `cols` as for a heatmap |
| Points | `index`: the point's index in the sent sample. `row_id`: its source index along the point dim | `indices`: indices in the sent sample. `row_ids`: their source indices, in the same order. At most 32,768 each; a viewer that cuts a larger selection sets `truncated` true |
| Series | `step`: the absolute index. `value`, optional | `steps`: a `[start, stop]` range of absolute indices |
| Histogram | `bin`: the bin index. `lo`, `hi`: the bin's value range. `count`, optional | `range`: a `[lo, hi]` value range |

A `select` whose `detail` is `{}` clears the selection. A viewer sends at most 30 hover events per second per view, and drops a hover rather than queueing it. The producer keeps at most 256 unread events per connection and drops the oldest.

In 3.1 the `detail` of a `select` also carries `space` and `items` when the selection is made on an axis with an item space: see [Selecting](#selecting).

## Controls

Controls let a viewer reach back into the running script: pause, resume and single-step its training loop, change parameters it registered, and run actions it registered. They travel on the stream, so they work while a notebook cell runs and the kernel cannot read widget messages. Controls are the capability `controls` of 3.1. Interventions on a model, such as ablating a head or steering along a direction, are not part of 3.1; they will come in a later version as new ops with typed arguments in the reserved key `args`, behind a capability of their own.

### The checkpoint

A script opts in by calling a checkpoint in its training loop, once per step, on its training thread (`caliper.step()` in Caliper LIB, the Python package). The checkpoint is the only safe point: only there does the producer apply parameter values, run actions or hold the loop. At each checkpoint the training thread, in this order:

1. sets every pending parameter value, in the order the parameters were registered;
2. runs the queued actions one at a time, in the order they were accepted;
3. waits, when the run pauses here, and while it waits sets parameter values and runs actions as they arrive, until a `resume` or a `step` releases it.

The producer keeps a waiting checkpoint interruptible, so that the person running the script can still stop it, for example with Ctrl-C or a notebook's interrupt; after such an interrupt the run is no longer paused. A paused run also resumes when its server closes, or when no connection that may control it has been open for 10 seconds (`CALIPER_NET_RELEASE_GRACE_MS`), which leaves a viewer time to reload its page.

### Pause, resume and step

| State | Meaning |
|---|---|
| `none` | The script has called no checkpoint yet. `pause` and `step` are answered with the error `unsupported` |
| `running` | The loop runs |
| `pausing` | A pause was asked for and the training thread has not reached a checkpoint yet |
| `paused` | The training thread waits in a checkpoint |
| `stepping` | The loop runs `steps_left` more steps and then pauses |

`pause` makes a running or stepping run `pausing`, and `paused` at its next checkpoint. `resume` makes the run `running` and releases a waiting checkpoint. `step` with `count` n lets the loop run n steps and pause at the n-th checkpoint after the one it waits in, or, from `running`, after n more checkpoints. A `pause` of a run that is already pausing or paused, and a `resume` of a running one, change nothing. All three are answered `ok` at once, and the new state reaches every viewer in CONTROLS.

#### What a viewer may rely on when a run pauses

The CONTROLS that says a run is `paused`, after a `pause` or after the steps of a `step`, follows the data of the step at which the run stopped. Through `caliper.serve()`, and on a producer's connection to a hub, a run state follows the data sent before it on the same connection: before its checkpoint waits, the producer sends the snapshots and series values that the run's view calls took, and it sends that CONTROLS once none of its connections has such data left to send, or after `CALIPER_NET_ORDER_WAIT_MS` (2 seconds). So when a viewer reads `paused` on a connection, every SERIES message of the step at which the run stopped has already arrived on it, and so has the last SNAPSHOT of each view that it subscribes to there, unless sending them took longer than those 2 seconds, as it can for a viewer that has stopped reading. A hub delivers a run state to a viewer after the data it received before it, or after 2 seconds (`CALIPER_NET_ORDER_WAIT_MS`), so a viewer of a hub may rely on the same order within the same bound.

- A series view then holds every value up to that step. A snapshot view holds the last snapshot the view took, which its rate limit (60 per second by default) can place a few steps before the one the run stopped at.
- Separate connections have no order between them. A viewer that draws a view from one connection and shows the run state from another, as the WebAssembly views of the element do, can read `paused` a moment before that view has drawn its last data; it holds the same data once that has arrived.
- A view that a viewer subscribes to while the run is paused receives the retained series values at once, but the snapshot of another kind of view only when the run takes one again, after it resumes. The producer computes snapshots in the script's view calls, which do not run while it waits.

### Parameters

A parameter is a named value that the script registered with a type and, for a number, a range. The script reads it, and a viewer shows it as a slider or a field (`number`, `integer`), a toggle (`boolean`) or a menu (`choice`). On `set` the producer:

1. finds the parameter by its exact name, or answers `unknown_control`;
2. checks the value: a finite number for `number`; a whole number, such as 64 or 64.0, of magnitude at most 2<sup>53</sup> for `integer`; `true` or `false` for `boolean`; one of `choices` for `choice`. Any other value is answered `bad_value`;
3. clamps a `number` or an `integer` to `min` and `max`;
4. answers `ok` with the clamped value;
5. sets the value at once before the script's first checkpoint, and otherwise records it as `pending`, replacing an earlier pending value, and sets it at the next checkpoint, which for a paused run is the one it waits in;
6. sends CONTROLS to every viewer, with the pending value and again with the new value once it is set.

When the script changes a value itself, the producer announces it the same way.

### Actions

An action is a function that the script registered under a name. A viewer can only ask for it to run: it sends no code and no argument. On `run` the producer finds the action by its exact name, or answers `unknown_control`. It then queues a run unless a run of the action is already queued, answers `queued`, and announces `queued` true in CONTROLS. A request for an action whose run is already queued joins that run and receives its result too.

Queued runs start at checkpoints, one at a time, in the order they were accepted, and CONTROLS shows each as `running` while it runs. While the state is `none`, runs stay queued, and a viewer shows that they wait for a checkpoint. A run that has not started 10 minutes after it was queued (`CALIPER_NET_ACTION_WAIT_MS`) is dropped and answered `timeout`. When the action returns, each requester gets `ok` with the action's result text; when it raises, `action_failed` with the error's text. A running action is never interrupted. A run of an action that the script removes before it starts is answered `unknown_control`.

### Ordering

A producer handles the CONTROL messages of one connection in the order they arrive, and those of several connections in the order it receives them. For a parameter the latest accepted value wins. Actions run one at a time in the order they were accepted, and at most one run of each action waits, so at most 64 runs wait. CONTROLS is coalesced like VIEWS: a newer one replaces an unsent older one.

### Who may control

A viewer sends CONTROL only when all three hold: the WELCOME of the server it is connected to announced `controls` and set `control` true, and the session it is attached to lists `controls` among its capabilities. A viewer that may not control still shows the run state, the parameters and the actions, without letting them be changed. A selection is not a control: every viewer may select.

- **`caliper.serve()`** of Caliper LIB accepts CONTROL from every connection that holds the session token. A server option makes it refuse every CONTROL; its WELCOME then sets `control` false, and a CONTROL is answered `forbidden`.
- **The hub**, `caliper serve` of Caliper CLI, forwards no control unless it was started with `--allow-control`, which is off by default. Without it, its WELCOME sets `control` false, to viewers and to producers, and it answers every CONTROL itself with `forbidden`.

### Controls through a hub

A hub that implements 3.1 announces `controls` and `selection` in its WELCOME. It then:

- forwards a viewer's CONTROL only to the producer of the session that the viewer is attached to, rewriting `request_id` in both directions as for SUBSCRIBE, and routes each CONTROL_REPLY to the viewer that asked, keeping the route of a `queued` reply until the final one;
- answers a CONTROL itself with `forbidden` without `--allow-control`, `not_attached` when the viewer is not attached, and `unsupported` when the session's producer lacks `controls` or the session has ended. When a producer disconnects with requests still waiting, the hub answers them `unsupported`. A CONTROL that the producer does not answer within 60 seconds (`CALIPER_NET_REPLY_WAIT_MS`) the hub answers `timeout` itself, and a `run` that the producer answered `queued` gets its final reply from the hub as `timeout` 10 minutes and 60 seconds after it was queued (`CALIPER_NET_ACTION_WAIT_MS` plus `CALIPER_NET_REPLY_WAIT_MS`);
- forwards the producer's CONTROLS and SELECTION to every viewer of the session that announced the capability, coalescing them like the producer;
- keeps the latest CONTROLS of each session and its latest SELECTION of each item space, sends them to a viewer right after the VIEWS of an attachment, and keeps them after the session ends. A viewer disables the controls of an ended session.

A viewer drops its controls and selections when VIEWS names another session or none.

## Linked selection

A selection in one view highlights the matching items in the other views of the same session: the points of an embedding, the rows of a heatmap, markers on a series. Linked selection is the capability `selection` of 3.1.

### Item spaces

An **item space** is a named set of items numbered from 0, such as the samples of a test set. A producer declares in VIEWS which axes index the items of a space:

- a dim of a source with `items`: source index i along that dim is item i of the space;
- a series view with `items`: absolute index i of the series is item i.

The point dim of an embedding and the `sample` dim of a heatmap with the same space show the same items. A viewer maps a payload position to a source index with the snapshot's [payload axes](#payload-axes), which the points' `row_id` already does for a sample of points, and the source index is the item id. A pooled heatmap row or column stands for several items and is highlighted when any of them is selected. A session uses at most 16 item spaces. A producer adds `items` to VIEWS only for a peer that announced `selection`.

### Selecting

A viewer that selects on an axis with an item space adds two keys to the `detail` of its select EVENT, besides the keys of its kind: `space`, the item space, and `items`, the selected item ids, at most 32,768, with `truncated` set when it cut them. For points they are the `row_ids`; for a heatmap or an image, the source indices of the selected rows, or of the selected columns when only the columns have a space; for a series, the indices in `steps`. `items` empty clears the space's selection. A viewer adds them only when the server announced `selection`.

The producer keeps one current selection for each item space. On a select event with `space`, it replaces that space's selection with the event's items, made ascending and without repeats, and sends SELECTION to every viewer, the one that selected included. Every viewer then draws the same highlight in every view that maps the space. The script can set a selection too, which reaches the viewers the same way, without `view`. Python code reads the select event as it always did.

### Item details

The script may provide details for an item space: a few fields of each item, from a static table or from a function it registered. For each new selection, the producer looks up the details of its first 32 items and sends them in SELECTION as `details`:

| Key | Type | Meaning |
|---|---|---|
| `fields` | array of strings | The field names, at most 16, each 1 to 63 bytes |
| `items` | array of integers | The item ids, at most 32 |
| `values` | array of arrays | One row per item, with one value per field: a number, a string of at most 256 bytes, a boolean or null |

The function runs neither on the training thread nor on the network thread, and has 2 seconds (`CALIPER_NET_DETAILS_TIMEOUT_MS`). A producer may send a selection at once and send it again with its details. When the details fail or time out, the selection goes without them and `details_error` says why. A viewer shows them in a small table beside the view in which the selection was made. It never sends code or names fields to compute.

## Flow control

Nothing in the protocol lets a slow viewer block a producer, and training is never blocked by a connection.

- A producer takes no snapshot of a view that no subscription names. With no subscription at all, a call costs one check, as in protocol v2.
- Series are the one exception. A producer retains appended series values whether or not anyone subscribes, without synchronising with its device, so that a late viewer receives the whole curve as backfill. It retains at least 4096 values per series.
- A sender hands one data message (SNAPSHOT or SERIES) at a time to its socket and chooses the next only when the previous one has drained. A message that has not been handed over is pending.
- Per connection, a sender keeps at most one pending snapshot per subscription, and a newer snapshot replaces it. For a viewer with one subscription per view this is one snapshot per view.
- Series values are coalesced into one pending SERIES message per subscription and never dropped. A message that would exceed 262,144 values is split.
- Control messages (every type but SNAPSHOT and SERIES) are sent in order and before pending data. Among pending data, series go before snapshots, and subscriptions take turns.
- A newer CONTROLS replaces an unsent older one, and a newer SELECTION of an item space replaces an unsent older one of the same space.
- A producer sends at most `max_rate` messages per second for each subscription.
- A hub applies the same rules on each viewer connection, so one slow viewer does not slow the others or the producer.

## Limits

| Constant | Value | Meaning |
|---|---|---|
| `CALIPER_NET_MAX_FRAME_BYTES` | 67,108,864 (64 MiB) | The largest frame |
| `CALIPER_NET_MAX_META_BYTES` | 1,048,576 (1 MiB) | The largest metadata |
| `CALIPER_NET_MAX_SECTIONS` | 8 | Sections in a frame |
| `CALIPER_NET_MAX_JSON_DEPTH` | 16 | Nesting of the metadata |
| `CALIPER_NET_NAME_MAX` | 63 | Bytes of a view, source or dim name |
| `CALIPER_NET_SESSION_ID_MAX` | 63 | Bytes of a session id |
| `CALIPER_NET_TOKEN_MIN` | 22 | Characters of a session token, at least |
| `CALIPER_NET_TOKEN_MAX` | 128 | Characters of a session token, at most |
| `CALIPER_NET_MAX_VIEWS` | 256 | Views in a session |
| `CALIPER_NET_MAX_SOURCES` | 64 | Sources of a view |
| `CALIPER_NET_MAX_DIMS` | 8 | Dims of a source |
| `CALIPER_NET_MAX_SUBSCRIPTIONS` | 256 | Subscriptions on a connection |
| `CALIPER_NET_MAX_STATS` | 64 | Entries of `stats` |
| `CALIPER_NET_DEFAULT_MAX_EXTENT` | 512 | Default `max_extent` |
| `CALIPER_NET_LIMIT_MAX_EXTENT` | 4096 | Largest `max_extent` |
| `CALIPER_NET_DEFAULT_MAX_POINTS` | 100,000 | Default `max_points` |
| `CALIPER_NET_LIMIT_MAX_POINTS` | 1,000,000 | Largest `max_points` |
| `CALIPER_NET_MAX_BINS` | 65,536 | Bins of a histogram |
| `CALIPER_NET_SERIES_RETAIN_MIN` | 4096 | Series values a producer retains, at least |
| `CALIPER_NET_SERIES_MAX_VALUES` | 262,144 | Values in one SERIES message |
| `CALIPER_NET_MAX_EVENT_INDICES` | 32,768 | Entries of an event's index list |
| `CALIPER_NET_DEFAULT_RATE_HZ` | 60 | Default `max_rate` |
| `CALIPER_NET_LIMIT_RATE_HZ` | 240 | Largest `max_rate` |
| `CALIPER_NET_MAX_QUEUED_EVENTS` | 256 | Unread events a producer keeps per connection |
| `CALIPER_NET_HANDSHAKE_TIMEOUT_MS` | 10,000 | Wait for HELLO or WELCOME |
| `CALIPER_NET_PING_INTERVAL_MS` | 10,000 | Silence before a PING |
| `CALIPER_NET_IDLE_TIMEOUT_MS` | 30,000 | Silence before closing |
| `CALIPER_NET_MAX_PARAMS` | 64 | Parameters of a session (3.1) |
| `CALIPER_NET_MAX_ACTIONS` | 64 | Actions of a session (3.1) |
| `CALIPER_NET_MAX_CHOICES` | 64 | Choices of a `choice` parameter (3.1) |
| `CALIPER_NET_TEXT_MAX` | 1,024 | Bytes of a `help` text, a reply's `message` or `details_error` (3.1) |
| `CALIPER_NET_CONTROL_RATE_HZ` | 30 | `set` requests a viewer sends per parameter per second, at most (3.1) |
| `CALIPER_NET_ACTION_WAIT_MS` | 600,000 | A queued action that has not started by then is answered `timeout` (3.1) |
| `CALIPER_NET_MAX_ITEM_SPACES` | 16 | Item spaces of a session (3.1) |
| `CALIPER_NET_MAX_DETAIL_ITEMS` | 32 | Items in a selection's details (3.1) |
| `CALIPER_NET_MAX_DETAIL_FIELDS` | 16 | Fields in a selection's details (3.1) |
| `CALIPER_NET_DETAIL_TEXT_MAX` | 256 | Bytes of a string in a selection's details (3.1) |
| `CALIPER_NET_DETAILS_TIMEOUT_MS` | 2,000 | The time the script's details function has (3.1) |
| `CALIPER_NET_REPLY_WAIT_MS` | 60,000 | A hub answers a CONTROL `timeout` when the producer has not answered it by then (3.1) |
| `CALIPER_NET_RELEASE_GRACE_MS` | 10,000 | A paused run resumes when no connection that may control it has been open this long (3.1) |
| `CALIPER_NET_ORDER_WAIT_MS` | 2,000 | The longest the run state of a pause waits for the data of its step to be sent (3.1) |

With the defaults, a heatmap snapshot is at most 1 MiB, an image snapshot 1 MiB, and a 2-D points snapshot with values 1.2 MB.

## Versioning

The major version is 3 and the minor version is 1. Both are in every frame's header.

- A receiver that sees another major closes with code 4001 and sends nothing else. The close reason may say which major it speaks.
- Within major 3, a receiver ignores message types, metadata keys, capabilities, frame flags and sections it does not know. A minor version only adds such things and never changes the meaning of existing ones.
- A sender uses an addition only when the peer's minor version, read from its HELLO or WELCOME, or one of the peer's capabilities says the peer knows it. A flag that changes how a frame decodes is only sent after a capability announced it.
- Capabilities are strings in HELLO and WELCOME, and in the session list. 3.0 defines `events` (a producer that reads viewer events) and reserves `arrow_ipc` for the future table kind. 3.1 defines `controls` and `selection`. A viewer and a hub announce what they understand, and a producer and a hub what they implement.
- Protocol v2 is unchanged and keeps its own version, `CALIPER_LIVE_PROTOCOL_VERSION` 2.

3.1 adds [controls](#controls), under the capability `controls`: the messages CONTROLS, CONTROL and CONTROL_REPLY, and the key `control` of WELCOME. It adds [linked selection](#linked-selection), under the capability `selection`: the message SELECTION, the key `items` of a source dim and of a series view in VIEWS, and the keys `space` and `items` in the detail of a select event. It adds the error codes `forbidden`, `unknown_control`, `bad_value`, `timeout` and `action_failed`, which only CONTROL_REPLY carries. Each addition, the new keys of 3.0 messages included, is used only with a peer that announced its capability. A peer decides by capabilities and never by the minor version alone, so an implementation may report minor 1 before it implements both capabilities. Every 3.0 message keeps its meaning and its encoding, so a 3.0 peer and a 3.1 peer work together in both directions without the additions:

| Viewer | Server | What happens |
|---|---|---|
| 3.0 | `caliper.serve()` of 3.1 | The viewer announces neither capability, so it receives no CONTROLS or SELECTION and no `items` in VIEWS |
| 3.1 | `caliper.serve()` of 3.0 | WELCOME announces neither capability, so the viewer shows no controls and sends no item ids |
| 3.1 | A 3.0 hub with a 3.1 producer | The hub's WELCOME announces neither capability, so neither the viewer nor the producer uses the additions. The session list may name the producer's capabilities, but a viewer goes by the WELCOME of the server it is connected to |

## Close codes

| Code | Name | When |
|---|---|---|
| 1000 | `CALIPER_NET_CLOSE_NORMAL` | After BYE `normal` or `replaced` |
| 1001 | `CALIPER_NET_CLOSE_GOING_AWAY` | The process is exiting, after BYE `shutdown` |
| 1003 | `CALIPER_NET_CLOSE_UNSUPPORTED_DATA` | A text message arrived |
| 1009 | `CALIPER_NET_CLOSE_TOO_BIG` | Decoding step 5 |
| 4000 | `CALIPER_NET_CLOSE_PROTOCOL_ERROR` | A frame or message that breaks this specification |
| 4001 | `CALIPER_NET_CLOSE_UNSUPPORTED_MAJOR` | Decoding step 3 |
| 4002 | `CALIPER_NET_CLOSE_ROLE_REFUSED` | The server does not take the client's role |
| 4003 | `CALIPER_NET_CLOSE_HANDSHAKE_TIMEOUT` | No HELLO or WELCOME in time |
| 4004 | `CALIPER_NET_CLOSE_IDLE_TIMEOUT` | Nothing received for 30 seconds |
| 4005 | `CALIPER_NET_CLOSE_SERVER_FULL` | The server's connection limit |

## Error codes

| Code | Meaning |
|---|---|
| `bad_request` | A field is missing or has the wrong type |
| `unknown_session` | ATTACH named a session the server does not hold |
| `unknown_view` | A message named a view the session does not have |
| `unknown_subscription` | VIEW_REQUEST named a `sub` that is not open |
| `bad_data_view` | A data view did not resolve |
| `not_attached` | A viewer of a hub subscribed, or sent CONTROL, before it was attached |
| `unsupported` | A known feature the receiver does not support, such as a section encoding, a pause of a script without a checkpoint, or a CONTROL for a session whose producer lacks `controls` or has ended |
| `limit` | A limit was reached, such as the number of subscriptions |
| `internal` | The producer failed while computing a data view |
| `forbidden` | The connection may not control: a hub without `--allow-control`, or a server that refuses controls |
| `unknown_control` | A CONTROL named a parameter or an action that the script has not registered |
| `bad_value` | A `set` value of the wrong type, a fraction for an `integer`, or a choice that is not listed |
| `timeout` | A queued action did not start within `CALIPER_NET_ACTION_WAIT_MS` |
| `action_failed` | The action raised an error; the message gives its text |

## Security

Network streaming stays on the local machine, on the loopback interface and with the session token, until a dedicated security review. That review happens before any session leaves the machine. A server listens on 127.0.0.1 only and checks the Host header, the Origin header and the session token before every upgrade. The token appears in a URL, so it can reach a browser's history; it protects against other pages and other users on the machine, not against someone who can read this user's files. Controls need nothing more than the stream: whoever holds the token of `caliper.serve()` may control, unless the script turns controls off.

The stream carries data, viewer events and, from 3.1, control requests. It never carries host commands or the command layer. No message carries code. A CONTROL can only name a parameter or an action that the script registered, matched exactly; the producer checks and clamps every value; and an action takes no argument from the viewer. The only code that a message makes a producer run is such a registered action, at a checkpoint on the training thread. A hub forwards controls only when it was started with `--allow-control`, and only to the producer of the viewer's own session.

## Worked example

This is the golden frame `tests/fixtures/wire_v3/series.bin`: three values 2.5, 1.25 and 0.75 of the series `loss`, the first three values ever appended, for subscription 3. It is the frame of a 3.0 sender, so its minor version is 0; a 3.1 sender writes 1 there.

```text
0000  43 41 4c 57 03 00 00 00 31 00 00 00 a4 00 00 00
0010  7a 00 00 00 01 00 00 00 0c 00 00 00 7b 22 73 65
0020  63 74 69 6f 6e 73 22 3a 5b 7b 22 64 74 79 70 65
0030  22 3a 30 2c 22 6e 61 6d 65 22 3a 22 76 61 6c 75
0040  65 73 22 2c 22 73 68 61 70 65 22 3a 5b 33 5d 7d
0050  5d 2c 22 73 74 61 6d 70 5f 6e 73 22 3a 38 31 32
0060  33 34 35 36 37 38 39 31 31 32 33 2c 22 73 74 61
0070  72 74 22 3a 30 2c 22 73 75 62 22 3a 33 2c 22 74
0080  6f 74 61 6c 22 3a 33 2c 22 76 69 65 77 22 3a 22
0090  6c 6f 73 73 22 7d 00 00 00 00 20 40 00 00 a0 3f
00a0  00 00 40 3f
```

| Offset | Bytes | Field | Value |
|---|---|---|---|
| 0 | `43 41 4c 57` | `magic` | `CALW` |
| 4 | `03 00` | `major` | 3 |
| 6 | `00 00` | `minor` | 0 |
| 8 | `31 00` | `type` | `0x0031`, series |
| 10 | `00 00` | `flags` | 0 |
| 12 | `a4 00 00 00` | `frame_len` | 164 |
| 16 | `7a 00 00 00` | `meta_len` | 122 |
| 20 | `01 00` | `section_count` | 1 |
| 22 | `00 00` | `reserved` | 0 |
| 24 | `0c 00 00 00` | `section_len[0]` | 12 |
| 28 | `7b 22` to `22 7d` | metadata | 122 bytes, ending at offset 150 |
| 150 | `00 00` | padding | up to 152, the next multiple of 8 |
| 152 | `00 00 20 40` | `values[0]` | 2.5 |
| 156 | `00 00 a0 3f` | `values[1]` | 1.25 |
| 160 | `00 00 40 3f` | `values[2]` | 0.75 |

The metadata, in canonical form:

```json
{"sections":[{"dtype":0,"name":"values","shape":[3]}],"stamp_ns":81234567891123,"start":0,"sub":3,"total":3,"view":"loss"}
```

The values section starts at offset 152, so a browser reads it as `new Float32Array(buffer, 152, 3)`.

## Golden frames

`tests/fixtures/wire_v3` holds one frame per message type, a few more to cover sections and options, and a set of frames to reject. `generate.py` in that folder writes them with `python/caliper/_wire.py`, and `python/tests/test_wire.py` checks that it reproduces every byte.

| File | Contents |
|---|---|
| `<name>.bin` | The bytes of one frame of a 3.0 sender, with minor 0 in its header |
| `<name>.json` | What a decoder must find: the header fields, `section_lengths`, `meta_offset`, the metadata object, and per section its name, dtype, shape, offset, length and values |
| `ping_future.bin`, `unknown_type.bin` | Frames from a later minor version, with unknown flags, keys and type, which a 3.0 decoder accepts and ignores in part |
| `v3_1/<name>.bin`, `v3_1/<name>.json` | The frames that 3.1 adds, with minor 1: one for each new message type and op, and 3.0 messages with the keys 3.1 adds |
| `invalid/<name>.bin` | A frame every decoder rejects |
| `invalid/expect.json` | For each invalid frame, the reason and the close code |

Every implementation of the protocol, the Python codec, the native `NetSource`, the hub and the TypeScript viewer, decodes these files in its own tests. The 3.0 frames keep minor 0, as a 3.0 sender wrote them, so an implementation that knows only 3.0 keeps passing. An implementation that encodes a decoded golden frame again to compare bytes encodes it at the frame's own minor. It reads `v3_1/` once it implements 3.1; a 3.0 decoder finds those frames well formed, with the new types unknown to it.

## The header, verbatim

This page embeds the real header; the docs build fails if the file moves.

```c
--8<-- "include/caliper/live_net.h"
```
