---
kind: reference
---

# Session Messages

This page lists the keys of the messages that open, choose, keep alive and close a connection of [Wire Protocol](wire-protocol.md) v3.

Each message gives the keys of its metadata. Keys marked optional may be absent. The header of every message follows the [frame layout](wire-frames.md#frame-layout). The [message types](wire-protocol.md#message-types) table gives the type id of each message. All the messages on this page have no section.

## `hello`

| Key | Type | Meaning |
|---|---|---|
| `role` | string | `viewer` or `producer` |
| `capabilities` | array of strings, optional | The [capabilities](wire-protocol.md#versioning) the client has. Default: none. A 3.1 viewer that shows controls announces `controls`, and one that draws linked selections announces `selection` |
| `software` | object, optional | `{"name", "version"}` of the client, for display and logs |
| `session` | object, required for a producer | The session identity of the producer: `id` (1 to 63 characters of `A-Z`, `a-z`, `0-9`, `.`, `_`, `-`), `script` (the name of the script), `pid`, and `start_unix_ms` (wall-clock milliseconds when the process started to stream) |

## `welcome`

| Key | Type | Meaning |
|---|---|---|
| `role` | string | `producer` (for `caliper.serve()`) or `hub` |
| `capabilities` | array of strings, optional | The capabilities of the server. A producer that reads viewer events announces `events`. A 3.1 producer or hub announces `controls` and `selection` for what it implements |
| `software` | object, optional | `{"name", "version"}` of the server |
| `session` | object, optional | From a producer, its own session identity. From a hub to a producer, the identity with the id the hub chose. Absent from a hub to a viewer |
| `control` | boolean, optional | 3.1, sent to a client that announced `controls`. To a viewer: true when the server accepts CONTROL from this connection. From a hub to a producer: true when the hub forwards the controls of its viewers. Absent means false |

## `bye`

| Key | Type | Meaning |
|---|---|---|
| `reason` | string | `normal` (done), `shutdown` (the process is exiting), `replaced` (a producer resumed its session on a new connection) or `error` (after a fatal error) |
| `message` | string, optional | Human-readable text |

The sender closes the WebSocket right after BYE.

## `error`

| Key | Type | Meaning |
|---|---|---|
| `code` | string | An [error code](wire-protocol.md#error-codes) |
| `message` | string, optional | Human-readable text |
| `fatal` | boolean, optional | True when the sender closes the connection next. Default false |
| `in_reply_to` | integer, optional | The type of the message that caused the error |
| `request_id`, `sub`, `view`, `session` | optional | The fields of that message that identify what failed |

## `ping` and `pong`

| Key | Type | Meaning |
|---|---|---|
| `nonce` | integer (uint32) | Chosen by the sender of PING and echoed by PONG |

## `sessions`

| Key | Type | Meaning |
|---|---|---|
| `sessions` | array of objects | Every session the server holds |

Each session object has the identity keys `id`, `script`, `pid` and `start_unix_ms`, and these keys:

| Key | Type | Meaning |
|---|---|---|
| `state` | string | `live`, or `ended` after its producer disconnected without a resume |
| `views` | integer | The number of views the session has |
| `capabilities` | array of strings, optional | The capabilities of the producer |

## `attach`

| Key | Type | Meaning |
|---|---|---|
| `session` | string or null | The session id to attach to, or null to detach. The key is required |
