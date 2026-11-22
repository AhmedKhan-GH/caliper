# Wire Protocol

Protocol v3 carries live data from a training process to viewers over one WebSocket. The viewers are a browser tab, a Jupyter widget, the hub and the native Caliper app. From version 3.1, the stream also carries the controls of a viewer back to the script. The [Header Listing](#header-listing) embeds the header `include/caliper/live_net.h`, which defines the constants and the frame header. The [golden frames](#golden-frames) give test data for every implementation.

!!! note "Status"
    Version 3.1 adds [controls](run-control-messages.md#controls) and [linked selection](linked-selection-messages.md#linked-selection) to 3.0. The stream works on the loopback interface only, with a session token. [Network streaming](../python/network-streaming.md) explains the design decisions behind it.

Protocol v2 ([`caliper/live.h`](../development/live-session-layout.md#versions), shared memory) is a separate zero-copy path between two processes on the same machine. Protocol v3 sends copies of reduced snapshots over a socket. A producer reduces each snapshot on its own device before it sends it, so the copy is small.

## Roles and Terms

| Role | What it is | What it does |
|---|---|---|
| Producer | The training process, through `caliper.serve()` in Python | Owns the tensors, computes every data view on its own device, and sends snapshots and series values |
| Viewer | A browser element, the Jupyter widget, or the native app's `NetSource` | Draws what it receives. It computes nothing |
| Hub | `caliper serve`, a subcommand of the `caliper` binary | Accepts producers and viewers, holds several sessions and forwards between them |

The side that opens a connection is its **client**. The side that accepts it is its **server**. A connection has one of three kinds:

| Connection | Client | Server |
|---|---|---|
| Viewer to producer | A viewer | `caliper.serve()` |
| Viewer to hub | A viewer | The hub |
| Producer to hub | A producer | The hub |

`caliper.serve(port=0)` listens on 127.0.0.1 and prints `http://127.0.0.1:<port>/?token=<token>`. The viewer page reads the token from its own address. It then connects to `ws://<same host>/v3/stream?token=<token>`. A viewer written for `caliper.serve()` works against the hub unchanged, because a server that holds exactly one session attaches a new viewer to it.

The pages use these terms with one meaning each:

- A **session** is the stream of one producer process. A **session id** such as `clp-4242-a1b2c3` names it in lists. The session id is not secret. (Protocol v2 calls the same string the session token.)
- The **session token** is the secret of one server. A client must present it to open a connection.
- A **view** is one named stream of a producer, such as `fc1.weight` or `loss`, of one kind.
- A **live view** decides how the viewer draws a view: heatmap, image, points, series or histogram. The kind of a view names its live view.
- A **data view** decides which data travels: a source, an index selection, a reduction and a size budget. [Data Views](data-view-messages.md#data-views) defines it.
- A **subscription** is a request of a viewer for one view under one data view. The viewer chooses its number, `sub`.
- A **snapshot** is the reduced payload of a view at one moment, sent for one subscription.

## Connection Setup

### Upgrade Request

A client opens a connection with a WebSocket upgrade (RFC 6455) of `GET /v3/stream?token=<session token>`. The token is a query parameter because a browser cannot set headers on a WebSocket. Before the upgrade, the server makes these checks in this order:

| Check | Rule | Answer when it fails |
|---|---|---|
| Path | The path is exactly `/v3/stream` | HTTP 404 |
| Host | The `Host` header is `127.0.0.1:<port>` or `localhost:<port>`, where `<port>` is the port the server listens on. The host name is not case-sensitive. When the server listens on port 80, the forms without a port are also accepted. This stops DNS rebinding attacks | HTTP 403 |
| Origin | The `Origin` header is absent or accepted by the rule below | HTTP 403 |
| Token | The `token` parameter is present and equal to the session token of the server, compared in constant time. A server may accept more tokens of its own, as the hub of Caliper API does ([Control Permissions](run-control-messages.md#control-permissions)) | HTTP 403 |

The server accepts the `Origin` header when its value is one of these:

1. The literal `null`, which sandboxed frames and some notebook front ends send.
2. An origin `<scheme>://<host>` or `<scheme>://<host>:<port>` with the scheme `http` or `https`, any port, and the host `127.0.0.1`, `localhost` or `[::1]`. The host name is not case-sensitive. This covers the page of the server itself and notebook front ends on loopback, such as JupyterLab at `http://localhost:8888`.
3. `vscode-webview://` followed by any authority, the origin of the webviews that show notebook output in VS Code.

The server refuses any other value, including an origin with a path or user information. A missing `Origin` means a client that is not a browser, such as the Python producer or the native app. When the origin is `null` or absent, the session token is the only protection.

The server answers a refused upgrade with the status above and closes the TCP connection. It never logs the token. The upgrade uses no subprotocol. A client sends no `Sec-WebSocket-Protocol` header, and the server declines `permessage-deflate`. Every message after the upgrade is binary. A receiver closes with code 1003 when it gets a text message.

The server may serve plain HTTP on the same port, such as the viewer page at `/`. This specification does not cover those requests, but they get the same Host check.

A session token has 22 to 128 characters of `A-Z`, `a-z`, `0-9`, `_` and `-`. It comes from at least 128 random bits, for example from `secrets.token_urlsafe(16)`. The viewer page should remove the token from its address bar after it reads it.

### Handshake

After the upgrade, the client sends HELLO and the server answers WELCOME. Neither side sends anything else before them, except BYE. A server closes with code 4003 when no HELLO arrives within 10 seconds. A client closes with 4003 when no WELCOME arrives within 10 seconds of its HELLO. Each side learns the minor version of the other from the header of HELLO or WELCOME.

| Server | Accepts | Refuses |
|---|---|---|
| `caliper.serve()` | Viewers | Producers, with close code 4002 |
| The hub | Viewers and producers | Nothing by role |
| The hub of Caliper API | Viewers, and producers with the session token | Producers on a token that may only watch ([Control Permissions](run-control-messages.md#control-permissions)), with close code 4002 |

A server at its connection limit closes with 4005 after the HELLO. Either side may send an ERROR with `fatal` true just before such a close, to explain it.

### Session Choice

Right after WELCOME on a viewer connection, the server sends SESSIONS, the list of the sessions it holds. It sends the list again whenever the list changes. `caliper.serve()` lists exactly one session, its own.

A viewer that has not sent ATTACH follows the default of the server. Whenever the viewer is not attached to a live session and the server holds exactly one live session, the server attaches the viewer to it. Thus a viewer of `caliper.serve()` is attached right after WELCOME, and a viewer of the hub follows a notebook kernel that restarts. A viewer that sends ATTACH chooses for itself from then on.

ATTACH names a session id. The server answers with VIEWS for that session, which confirms the attachment. If the session is unknown, it answers with ERROR `unknown_session` and the viewer stays as it was. ATTACH with `session` null detaches the viewer. The server then ends the subscriptions of the viewer and sends VIEWS with `session` null and no views. Attaching to another session ends every subscription of the previous one.

When an attached session ends, the hub keeps it in the list with the state `ended`, and the viewer keeps the last frames it drew. When the producer of `caliper.serve()` ends, it closes every connection with BYE `shutdown` and close code 1001.

### Producer Connection

A producer that connects to a hub sends HELLO with the role `producer` and its session identity: `id`, `script`, `pid` and `start_unix_ms`. The hub keeps the proposed id unless a live session of another process holds it. In that case, the hub appends `-2`, `-3` and so on. WELCOME carries the identity with the id the hub chose.

A producer that reconnects with the same id, pid and start time resumes its session. The hub closes the old connection, if it is still open, with BYE `replaced` and code 1000. The viewers stay attached.

After WELCOME, the producer sends VIEWS. On this connection the hub acts as the one viewer of the producer. It sends SUBSCRIBE, UNSUBSCRIBE, VIEW_REQUEST and EVENT, and the producer answers with VIEW_REPLY, SNAPSHOT and SERIES. The hub does these things:

- It rewrites `sub` and `request_id` between its viewer connections and its producer connection.
- It shares one producer subscription among viewers whose resolved data views are equal, at the largest `max_rate` among them.
- It keeps the latest snapshot of every producer subscription and the retained series values, so a viewer that subscribes late gets them at once.
- It forwards every event of the viewers to the producer.
- From 3.1, it forwards the controls of the viewers it allows to the producer. It forwards the CONTROLS and SELECTION of the producer to the viewers of the session ([Hub Forwarding](run-control-messages.md#hub-forwarding)).
- It resubscribes after a producer resumes its session.

### Keepalive and Closing

A peer that has received nothing for 10 seconds sends PING, and the other side answers with PONG that carries the same nonce. A peer that has received nothing for 30 seconds closes with code 4004. The protocol has its own ping because a browser cannot send WebSocket control pings.

To close, a peer sends BYE with a reason. It then closes the WebSocket with the [close code](#close-codes) that matches the reason. The code is 1000 for `normal` and `replaced`, 1001 for `shutdown`, and the code of the error for `error`.

## Message Types

Every message is one frame, and [Wire Frames](wire-frames.md) defines the frame.

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

The types from `0x0050` are additions of 3.1. A sender sends CONTROLS, CONTROL and CONTROL_REPLY only to a peer that announced the capability `controls`. It sends SELECTION only to a peer that announced `selection` ([Versioning](#versioning)). Type 0 is never sent. A known message sent in the wrong direction, or before HELLO and WELCOME, is a protocol error (4000). A receiver ignores a message of an unknown type.

These pages define the messages:

- [Session Messages](session-messages.md): `hello`, `welcome`, `bye`, `error`, `ping`, `pong`, `sessions` and `attach`.
- [Data Messages](data-messages.md): `views`, `snapshot`, `series` and `event`, and the rules for events.
- [Data View Messages](data-view-messages.md): `subscribe`, `unsubscribe`, `view_request` and `view_reply`, and the rules for data views.
- [Run Control Messages](run-control-messages.md): `controls`, `control` and `control_reply`, and the rules for controls.
- [Linked Selection Messages](linked-selection-messages.md): `selection` and the rules for item spaces.

## Flow Control

A slow viewer cannot block a producer, and a connection never blocks training. These rules apply:

- A producer takes no snapshot of a view that no subscription names. With no subscription at all, a call costs one check, as in protocol v2.
- A producer retains appended series values whether or not anyone subscribes, so a late viewer receives the whole curve as backfill. It does not synchronize with its device to do this. It retains at least 4096 values per series.
- A sender hands one data message (SNAPSHOT or SERIES) at a time to its socket. It chooses the next message only after the previous one drains. A message that is not yet handed over is pending.
- Per connection, a sender keeps at most one pending snapshot per subscription, and a newer snapshot replaces it. A viewer with one subscription per view has one pending snapshot per view.
- Series values merge into one pending SERIES message per subscription, and no value is dropped. A message that would exceed 262,144 values is split.
- Control messages, which are every type but SNAPSHOT and SERIES, go in order and before pending data. Among pending data, series go before snapshots, and subscriptions take turns.
- A newer CONTROLS replaces an unsent older one. A newer SELECTION of an item space replaces an unsent older one of the same space.
- A producer sends at most `max_rate` messages per second for each subscription.
- A hub applies the same rules on each viewer connection, so one slow viewer does not slow the others or the producer.

## Resource Limits

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
| `CALIPER_NET_DETAILS_TIMEOUT_MS` | 2,000 | The time the details function of the script has (3.1) |
| `CALIPER_NET_REPLY_WAIT_MS` | 60,000 | A hub answers a CONTROL `timeout` when the producer has not answered it by then (3.1) |
| `CALIPER_NET_RELEASE_GRACE_MS` | 10,000 | A paused run resumes when no connection that may control it has been open this long (3.1) |
| `CALIPER_NET_ORDER_WAIT_MS` | 2,000 | The longest the run state of a pause waits for the data of its step to go out (3.1) |

With the defaults, a heatmap snapshot is at most 1 MiB, an image snapshot is at most 1 MiB, and a 2-D points snapshot with values is at most 1.2 MB.

## Versioning

The major version is 3 and the minor version is 1. Both are in the header of every frame. These rules apply:

- A receiver that sees another major version closes with code 4001 and sends nothing else. The close reason may say which major version it speaks.
- Within major 3, a receiver ignores message types, metadata keys, capabilities, frame flags and sections it does not know. A minor version only adds such things and never changes the meaning of existing ones.
- A sender uses an addition only when the peer knows it. The peer says so with its minor version, read from its HELLO or WELCOME, or with one of its capabilities. A flag that changes how a frame decodes goes out only after a capability announced it.
- Capabilities are strings in HELLO, WELCOME and the session list. Version 3.0 defines `events` (a producer that reads viewer events) and reserves `arrow_ipc` for a future table kind. Version 3.1 defines `controls` and `selection`.
- A viewer and a hub announce what they understand. A producer and a hub announce what they implement.

Version 3.1 adds two groups of features:

- [Controls](run-control-messages.md#controls), under the capability `controls`: the messages CONTROLS, CONTROL and CONTROL_REPLY, and the key `control` of WELCOME.
- [Linked selection](linked-selection-messages.md#linked-selection), under the capability `selection`: the message SELECTION, the key `items` of a source dim and of a series view in VIEWS, and the keys `space` and `items` in the detail of a select event.

It also adds the error codes `forbidden`, `unknown_control`, `bad_value`, `timeout` and `action_failed`, which only CONTROL_REPLY carries. A peer uses each addition, including the new keys of 3.0 messages, only with a peer that announced its capability. A peer decides by capabilities and never by the minor version alone. Thus an implementation may report minor 1 before it implements both capabilities.

Every 3.0 message keeps its meaning and its encoding. As a result, a 3.0 peer and a 3.1 peer work together in both directions without the additions:

| Viewer | Server | What happens |
|---|---|---|
| 3.0 | `caliper.serve()` of 3.1 | The viewer announces neither capability, so it receives no CONTROLS or SELECTION and no `items` in VIEWS |
| 3.1 | `caliper.serve()` of 3.0 | WELCOME announces neither capability, so the viewer shows no controls and sends no item ids |
| 3.1 | A 3.0 hub with a 3.1 producer | The WELCOME of the hub announces neither capability, so neither the viewer nor the producer uses the additions. The session list may name the capabilities of the producer, but a viewer goes by the WELCOME of the server it is connected to |

## Close Codes

| Code | Name | When |
|---|---|---|
| 1000 | `CALIPER_NET_CLOSE_NORMAL` | After BYE `normal` or `replaced` |
| 1001 | `CALIPER_NET_CLOSE_GOING_AWAY` | The process is exiting, after BYE `shutdown` |
| 1003 | `CALIPER_NET_CLOSE_UNSUPPORTED_DATA` | A text message arrived |
| 1009 | `CALIPER_NET_CLOSE_TOO_BIG` | [Frame decoding](wire-frames.md#frame-decoding) step 5 |
| 4000 | `CALIPER_NET_CLOSE_PROTOCOL_ERROR` | A frame or message that breaks this specification |
| 4001 | `CALIPER_NET_CLOSE_UNSUPPORTED_MAJOR` | Frame decoding step 3 |
| 4002 | `CALIPER_NET_CLOSE_ROLE_REFUSED` | The server does not take the role of the client |
| 4003 | `CALIPER_NET_CLOSE_HANDSHAKE_TIMEOUT` | No HELLO or WELCOME in time |
| 4004 | `CALIPER_NET_CLOSE_IDLE_TIMEOUT` | Nothing received for 30 seconds |
| 4005 | `CALIPER_NET_CLOSE_SERVER_FULL` | The connection limit of the server |

## Error Codes

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
| `internal` | The producer failed while it computed a data view |
| `forbidden` | The connection may not control: a hub without `--allow-control`, or a server that refuses controls |
| `unknown_control` | A CONTROL named a parameter or an action that the script has not registered |
| `bad_value` | A `set` value of the wrong type, a fraction for an `integer`, or a choice that is not listed |
| `timeout` | A queued action did not start within `CALIPER_NET_ACTION_WAIT_MS` |
| `action_failed` | The action raised an error, and the message gives its text |

## Security

Network streaming stays on the local machine, on the loopback interface and with the session token. A server listens on 127.0.0.1 only. Before every upgrade, it checks the Host header, the Origin header and the session token. The token appears in a URL, so it can reach the history of a browser. It protects against other pages and other users on the machine, but not against someone who can read the files of the user. Controls need nothing more than the stream. Whoever holds the token of `caliper.serve()` may control, unless the script turns controls off.

The stream carries data, viewer events and, from 3.1, control requests. It never carries host commands or the command layer, and no message carries code. A CONTROL can only name a parameter or an action that the script registered, matched exactly. The producer checks and clamps every value, and an action takes no argument from the viewer. The only code that a message makes a producer run is such a registered action, at a checkpoint on the training thread. A hub forwards controls only when it starts with `--allow-control`, and only to the producer of the session of the viewer.

## Golden Frames

The folder `tests/fixtures/wire_v3` holds test data that every implementation decodes in its own tests. These implementations are the Python codec, the native `NetSource`, the hub and the TypeScript viewer.

| File | Contents |
|---|---|
| `<name>.bin` | The bytes of one frame of a 3.0 sender, with minor 0 in its header |
| `<name>.json` | What a decoder must find: the header fields, `section_lengths`, `meta_offset`, the metadata object, and for each section its name, dtype, shape, offset, length and values |
| `ping_future.bin`, `unknown_type.bin` | Frames from a later minor version, with unknown flags, keys and type, which a 3.0 decoder accepts and partly ignores |
| `v3_1/<name>.bin`, `v3_1/<name>.json` | The frames that 3.1 adds, with minor 1: one for each new message type and op, and 3.0 messages with the keys that 3.1 adds |
| `invalid/<name>.bin` | A frame every decoder rejects |
| `invalid/expect.json` | For each invalid frame, the reason and the close code |

The folder has one frame per message type, a few more for sections and options, and a set of frames to reject. The 3.0 frames keep minor 0, as a 3.0 sender wrote them, so an implementation that knows only 3.0 keeps passing. An implementation that encodes a decoded frame again to compare bytes uses the minor version of the frame. It reads `v3_1/` after it implements 3.1. A 3.0 decoder finds those frames well formed, with the new types unknown to it. The [worked example](wire-frames.md#worked-example) decodes one of these frames by hand.

## Header Listing

```c
--8<-- "include/caliper/live_net.h"
```
