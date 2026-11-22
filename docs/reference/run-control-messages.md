---
kind: reference
---

# Run Control Messages

This page defines the messages and rules with which a viewer controls a running script in [Wire Protocol](wire-protocol.md) 3.1.

## Controls

Controls let a viewer reach back into the running script. A viewer can pause, resume and single-step the training loop, change the parameters the script registered, and run the actions the script registered. They travel on the stream, so they work while a notebook cell runs and the kernel cannot read widget messages. Controls are the capability `controls` of 3.1. Interventions on a model, such as an ablation of a head or steering along a direction, are not part of 3.1. A later version may add them as new ops with typed arguments in the reserved key `args`, behind a capability of their own.

### Checkpoint

A script opts in when it calls a checkpoint in its training loop, once per step, on its training thread (`caliper.step()` in Caliper LIB, the Python package). The checkpoint is the only safe point, because only there does the producer apply parameter values, run actions or hold the loop. At each checkpoint, the training thread does these things in this order:

1. It sets every pending parameter value, in the order of registration of the parameters.
2. It runs the queued actions one at a time, in the order the producer accepted them.
3. It waits, when the run pauses here. While it waits, it sets parameter values and runs actions as they arrive, until a `resume` or a `step` releases it.

A waiting checkpoint stays interruptible, so the person who runs the script can still stop it, for example with Ctrl-C or the interrupt of a notebook. After such an interrupt, the run is no longer paused. A paused run also resumes when its server closes, or when no connection that may control it has been open for 10 seconds (`CALIPER_NET_RELEASE_GRACE_MS`). This leaves a viewer time to reload its page.

### Pause, Resume and Step

| State | Meaning |
|---|---|
| `none` | The script has called no checkpoint yet. `pause` and `step` get the error `unsupported` |
| `running` | The loop runs |
| `pausing` | A pause was requested and the training thread has not reached a checkpoint yet |
| `paused` | The training thread waits in a checkpoint |
| `stepping` | The loop runs `steps_left` more steps and then pauses |

`pause` makes a running or stepping run `pausing`, and `paused` at its next checkpoint. `resume` makes the run `running` and releases a waiting checkpoint. `step` with `count` n lets the loop run n steps. The loop pauses at the n-th checkpoint after the one it waits in, or, from `running`, after n more checkpoints. A `pause` of a run that is already pausing or paused changes nothing, and neither does a `resume` of a running one. The producer answers all three `ok` at once, and the new state reaches every viewer in CONTROLS.

#### What a viewer may rely on when a run pauses

The CONTROLS that reports a run as `paused` follows the data of the step at which the run stopped. This holds after a `pause` and after the steps of a `step`. It holds through `caliper.serve()` and on the connection of a producer to a hub. A run state follows the data sent before it on the same connection, as these steps show:

1. Before its checkpoint waits, the producer sends the snapshots and series values that the view calls of the run took.
2. It sends that CONTROLS when none of its connections has such data left to send, or after `CALIPER_NET_ORDER_WAIT_MS` (2 seconds).

Thus when a viewer reads `paused` on a connection, every SERIES message of the step at which the run stopped has already arrived on it. The last SNAPSHOT of each view that the viewer subscribes to there has also arrived. The exception is sending that took longer than 2 seconds, as it can for a viewer that has stopped reading.

A hub delivers a run state to a viewer after the data it received before it, or after 2 seconds (`CALIPER_NET_ORDER_WAIT_MS`). Thus a viewer of a hub may rely on the same order within the same bound. The hub holds only the run state for this. A CONTROL_REPLY, a SELECTION or a VIEWS message that reaches the hub after a held run state can arrive at the viewer before it. When a newer snapshot replaces one the hub has not yet sent, the run state follows that newer snapshot.

- A series view then holds every value up to that step. A snapshot view holds the last snapshot the view took. The rate limit (60 per second by default) can place that snapshot a few steps before the step at which the run stopped.
- Separate connections have no order between them. A viewer may draw a view from one connection and show the run state from another, as the WebAssembly views of the element do. It can read `paused` a moment before that view has drawn its last data, and it holds the same data after that data arrives.
- A view that a viewer subscribes to while the run is paused receives the retained series values at once. The snapshot of another kind of view comes only when the run takes one again, after it resumes. The producer computes snapshots in the view calls of the script, which do not run while it waits.

### Parameters

A parameter is a named value that the script registered with a type and, for a number, a range. The script reads it. A viewer shows it as a slider or a field (`number`, `integer`), a toggle (`boolean`) or a menu (`choice`). On `set`, the producer does these steps:

1. It finds the parameter by its exact name, or answers `unknown_control`.
2. It checks the value. A `number` must be a finite number. An `integer` must be a whole number, such as 64 or 64.0, of magnitude at most 2<sup>53</sup>. A `boolean` must be `true` or `false`. A `choice` must be one of `choices`. The producer answers any other value `bad_value`.
3. It clamps a `number` or an `integer` to `min` and `max`.
4. It answers `ok` with the clamped value.
5. Before the first checkpoint of the script, it sets the value at once. After that, it records the value as `pending`, replaces an earlier pending value, and sets it at the next checkpoint. For a paused run, that is the checkpoint it waits in.
6. It sends CONTROLS to every viewer, with the pending value and again with the new value after it is set.

When the script changes a value itself, the producer announces it the same way.

### Actions

An action is a function that the script registered under a name. A viewer can only request that it runs. It sends no code and no argument. On `run`, the producer finds the action by its exact name, or answers `unknown_control`. It then queues a run unless a run of the action is already queued. It answers `queued` and announces `queued` true in CONTROLS. A request for an action whose run is already queued joins that run and receives its result too.

Queued runs start at checkpoints, one at a time, in the order the producer accepted them. CONTROLS shows each as `running` while it runs. While the state is `none`, runs stay queued, and a viewer shows that they wait for a checkpoint. A run that has not started 10 minutes after the producer queued it (`CALIPER_NET_ACTION_WAIT_MS`) is dropped and answered `timeout`. When the action returns, each requester gets `ok` with the result text of the action. When it raises an error, each requester gets `action_failed` with the text of the error. The producer never interrupts a running action. A run of an action that the script removes before it starts is answered `unknown_control`.

### Ordering

A producer handles the CONTROL messages of one connection in the order they arrive, and those of several connections in the order it receives them. For a parameter, the latest accepted value wins. Actions run one at a time in the order the producer accepted them. At most one run of each action waits, so at most 64 runs wait. CONTROLS merges like VIEWS, and a newer one replaces an unsent older one.

### Control Permissions

A viewer sends CONTROL only when all three of these hold:

- The WELCOME of the server it is connected to announced `controls`.
- That WELCOME set `control` true.
- The session it is attached to lists `controls` among its capabilities.

A viewer that may not control still shows the run state, the parameters and the actions, but it does not let a person change them. A selection is not a control, and every viewer may select. These rules apply to each server:

- **`caliper.serve()`** of Caliper LIB accepts CONTROL from every connection that holds the session token. A server option makes it refuse every CONTROL. Its WELCOME then sets `control` false, and it answers a CONTROL `forbidden`.
- **The hub**, `caliper serve` of Caliper CLI, forwards no control unless it starts with `--allow-control`, which is off by default. Without the option, its WELCOME sets `control` false to viewers and to producers, and the hub answers every CONTROL itself with `forbidden`.
- **The hub of Caliper API**, in the app, takes the session token of the person and also the tokens it gives to agents ([Caliper API](../agents/http-api.md#command-origin)). A viewer connected with the token of an agent only watches. Its WELCOME sets `control` false, and the hub answers its CONTROL with `forbidden`, even with `--allow-control`. The hub refuses a HELLO of a producer on such a token, as `caliper.serve()` refuses producers. A viewer connected with the session token may control as with `caliper serve`.

### Hub Forwarding

A hub that implements 3.1 announces `controls` and `selection` in its WELCOME. It then does these things:

- It forwards the CONTROL of a viewer only to the producer of the session the viewer is attached to. It rewrites `request_id` in both directions as for SUBSCRIBE. It routes each CONTROL_REPLY to the viewer that asked, and it keeps the route of a `queued` reply until the final one.
- It answers a CONTROL itself with `forbidden` without `--allow-control`. It answers `not_attached` when the viewer is not attached. It answers `unsupported` when the producer of the session lacks `controls` or the session has ended. When a producer disconnects with requests still waiting, the hub answers them `unsupported`.
- It answers `timeout` itself to a CONTROL that the producer does not answer within 60 seconds (`CALIPER_NET_REPLY_WAIT_MS`). A `run` that the producer answered `queued` gets its final `timeout` reply from the hub 10 minutes and 60 seconds after the producer queued it (`CALIPER_NET_ACTION_WAIT_MS` plus `CALIPER_NET_REPLY_WAIT_MS`).
- It forwards the CONTROLS and SELECTION of the producer to every viewer of the session that announced the capability, and it merges them like the producer.
- It keeps the latest CONTROLS of each session and its latest SELECTION of each item space. It sends them to a viewer right after the VIEWS of an attachment, and it keeps them after the session ends. A viewer disables the controls of an ended session.

A viewer drops its controls and selections when VIEWS names another session or none.

## `controls`

CONTROLS has no section, and the [message types](wire-protocol.md#message-types) table gives its type id. Each message gives the keys of its metadata. Keys marked optional may be absent. The header of every message follows the [frame layout](wire-frames.md#frame-layout).

| Key | Type | Meaning |
|---|---|---|
| `run` | object | The run state, below |
| `params` | array of objects | Every parameter the script registered, at most 64, in the order of registration |
| `actions` | array of objects | Every action the script registered, at most 64, in the order of registration |

The producer sends CONTROLS right after the VIEWS that confirms an attachment. It sends CONTROLS again, whole, whenever the run state, a parameter or an action changes. A script that registered nothing and called no checkpoint still sends one, with the state `none` and empty lists. The rules for every field are in [Controls](#controls).

| `run` key | Type | Meaning |
|---|---|---|
| `state` | string | `none`, `running`, `pausing`, `paused` or `stepping`, as [Pause, Resume and Step](#pause-resume-and-step) describes |
| `step` | integer, optional | The step number of the script at its latest checkpoint. While the run runs, the producer sends CONTROLS again at most one time each second, so this number can be up to one second old. A paused run reports the step where it paused |
| `steps_left` | integer, optional | While `stepping`, the steps the loop still runs before it pauses |

| Parameter key | Type | Meaning |
|---|---|---|
| `name` | string | 1 to 63 bytes, unique among the parameters |
| `type` | string | `number`, `integer`, `boolean` or `choice` |
| `value` | number, boolean or string | The value the script reads now |
| `pending` | as `value`, optional | A value accepted from a viewer that the script reads from the next checkpoint on |
| `min`, `max` | numbers, optional | `number` and `integer`: the range a value is clamped to. Integers for an `integer` |
| `log` | boolean, optional | `number` and `integer`: draw the control on a log scale. Requires `min` greater than 0 and `max` |
| `choices` | array of strings | `choice` only: 1 to 64 distinct strings of 1 to 63 bytes |
| `help` | string, optional | At most 1,024 bytes, for a tooltip |

| Action key | Type | Meaning |
|---|---|---|
| `name` | string | 1 to 63 bytes, unique among the actions |
| `running` | boolean | The action runs now |
| `queued` | boolean | A run of the action waits for a checkpoint |
| `help` | string, optional | At most 1,024 bytes, for a tooltip |

## `control`

CONTROL has no section.

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer (uint32) | Chosen by the viewer, echoed by CONTROL_REPLY |
| `op` | string | `pause`, `resume`, `step`, `set` or `run` |
| `name` | string | `set`: the name of the parameter. `run`: the name of the action. Matched exactly, byte for byte |
| `value` | number, boolean or string | `set` only: the new value |
| `count` | integer (uint32), optional | `step` only: how many steps the loop runs before it pauses, at least 1. Default 1 |

A receiver ignores the keys that do not belong to the op. The key `args` is reserved for typed action arguments in a later version. A viewer sends it only to a producer that announced a capability for it, and 3.1 defines none. A viewer sends CONTROL only as [Control Permissions](#control-permissions) allows. It sends at most 30 `set` requests per second for one parameter, for example while a person drags a slider, and it ends with the value where the drag stops.

A producer or a hub answers a CONTROL from a peer that did not announce `controls` in its HELLO with an ERROR `bad_request`. The ERROR names the request in `in_reply_to` and `request_id`, and the connection stays open.

## `control_reply`

CONTROL_REPLY has no section.

| Key | Type | Meaning |
|---|---|---|
| `request_id` | integer | The `request_id` of the CONTROL it answers |
| `op` | string | The `op` of that CONTROL |
| `status` | string | `ok`, `queued` or `error` |
| `name` | string, optional | The `name` of that CONTROL |
| `value` | number, boolean or string, optional | `set` with `ok`: the value the producer accepted, after clamping |
| `message` | string, optional | `run` with `ok`: the result text of the action, at most 1,024 bytes |
| `error` | object | With the status `error` only: `{"code", "message"}`, with a code from the [error codes](wire-protocol.md#error-codes) |

The producer answers each CONTROL with one CONTROL_REPLY, to its sender only. The exception is an accepted `run`. The producer answers it `queued` at once and answers it again later: `ok` when the action returned, or `error` with `action_failed`, `timeout` or `unknown_control`. A refused CONTROL gets `error` at once and changes nothing. A malformed CONTROL, such as one without `request_id`, gets an ERROR `bad_request`, as any malformed message does.
