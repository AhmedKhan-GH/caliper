# Command layer

The command layer is the one surface through which everything acts on a workspace: the app's UI, the WYSIWYG builder, the agent pane over ACP, outside agents over MCP and programs over Caliper API. Each action is a typed command with a JSON Schema for its arguments and one for its result, so every client gets the same validation, the same undo history, the same journal and the same approval rules. This page elaborates sections 7 and 8 of the sprint 3 contracts and records the decisions taken while building the library. The [Applet builder](applet-builder.md) page gives the background, and the [Trust model](../explanations/trust-model.md) explains why everything an agent builds is data.

## What is built and what is planned

Built on this branch, in the library `caliper_builder_commands` (`src/builder/commands/`, headers in `src/builder/include/caliper_builder/commands/`):

- descriptors, invocations and results, with their JSON forms;
- the registry and the built-in schema validator;
- the undo history, with groups, merging, the redo rule and its bounds;
- the journal, its reader and its replay;
- the approval policy and the pending-approval flow;
- the queue, with `submit()` from any thread and `drain()` on the caller's thread;
- the descriptor table of every command of v0, as data, and its checked-in JSON form `src/builder/commands/command_table.json`.

Planned:

- The handlers of the catalog, data, loader, spec, source, panel, layout and applet commands are bound by the [builder session](builder-session.md), which is built and tested but not yet wired into the app. Until a handler is bound, its command answers `command.not_bound`.
- The handlers that need the running app (`app.info`, `app.applets`, `app.launch`, `capture.frame`, `panel.render` and the live catalog provider) are bound when the libraries are wired into the app, after sprint 2 merges.
- The two endpoints of Caliper API described [below](#caliper-api) come with that wiring.

## The command model

### Descriptors

A descriptor describes one command. The registry holds one per command name, and the descriptor table holds the descriptors of v0 as data, with no handlers.

| Field | Meaning |
|---|---|
| `name` | `group.verb`. The group matches `[a-z][a-z0-9]*` and has no underscore; the verb matches `[a-z][a-z0-9_]*`; the whole name has at most 64 characters. |
| `summary` | What the command does, in plain sentences written for an agent that has never seen Caliper. |
| `args` | The JSON Schema of the arguments. It is always an object schema. |
| `result` | The JSON Schema of the result object. |
| `examples` | One or more argument objects that pass `args`. |
| `mutates` | The command changes workspace state, so it is journaled and can be undone. |
| `destructive` | The command deletes or overwrites saved data (a file of the workspace or a database table), or runs a Python loader. It needs approval when an agent sends it. |
| `frame_thread` | The command must run on the app's frame thread, because it reads or changes what the frame thread draws. |

`destructive` is about saved data. An edit of the open spec in memory is not destructive, even when it removes a panel, because undo reverses it exactly. Saving an applet, which can overwrite its files, is.

### The schema subset

The registry validates every argument object against its descriptor's `args` schema before the handler runs, with a small built-in validator. It accepts these keywords and refuses a schema with any other:

| Keyword | Rule |
|---|---|
| `type` | A type name or a list of them: `object`, `array`, `string`, `integer`, `number`, `boolean`, `null`. An `integer` is a number with no fractional part. |
| `properties`, `required` | As in JSON Schema. Properties are checked in name order, so the first error is deterministic. |
| `additionalProperties` | `false`, `true` or a schema for the values of the other properties. |
| `items` | One schema for every element. |
| `minItems`, `maxItems` | Bounds on the length of an array. |
| `enum` | A list of allowed values, compared as JSON values. |
| `minimum`, `maximum` | Inclusive bounds on a number. |
| `minLength`, `maxLength` | Bounds on a string's length in Unicode code points. |
| `pattern` | An ECMAScript regular expression that must match somewhere in the string. It is applied only to strings of at most 4096 bytes; a longer string fails with a message that says so. |
| `description`, `title`, `default` | Annotations. They are not checked. |

The validator bounds its own work, because arguments come from agents. Before it walks a value it measures it without recursion and refuses one nested deeper than 64 levels or holding more than 1,048,576 values, with `command.invalid_args`. A schema may nest at most 32 levels. The first error is returned, with `where` set to the path of the value inside the arguments, for example `args.bind.y[1]`.

The descriptors in the table follow stricter rules, so that the Claude API can use them as strict tools and MCP clients get closed schemas:

- Every object schema in `args` states `"additionalProperties": false`, lists its `required` properties (possibly none), and gives every property a `description`.
- No schema is recursive. The layout tree is therefore passed as a flat list of nodes (see `layout.set`).
- Result schemas follow the same rules, with one exception: `spec.get` returns the spec's canonical JSON in `json`, whose schema is the spec format's own JSON Schema, so `json` is an open object.

A test of the table enforces these rules.

### Invocations

An invocation asks for one command to run. Its JSON form:

```json
{"id": "c-41", "name": "panel.patch", "args": {"id": "heat", "patch": {"title": "Entropy"}},
 "origin": "agent", "group": "turn-7", "merge_key": null}
```

| Field | Rule |
|---|---|
| `id` | Up to 128 characters of `A-Z a-z 0-9 . _ : -`. When it is empty or absent, the registry assigns `c-<n>` from a counter. |
| `name` | A command name. |
| `args` | An object; absent or null means `{}`. |
| `origin` | `ui` (the app's own UI), `builder` (the WYSIWYG builder), `agent` (the agent pane), `mcp` (an outside agent through `caliper-mcp`) or `api` (a program through Caliper API). |
| `group` | Optional. Invocations that share a group undo as one step. |
| `merge_key` | Optional. Consecutive invocations with the same merge key merge into one step. |

`group` and `merge_key` match `[A-Za-z0-9._:/-]{1,128}`. An invocation with an unknown field is refused with `command.invalid_invocation`, so a typo does not pass silently.

### Results

A result is `{"id": "...", "ok": true, "result": {...}}` or `{"id": "...", "ok": false, "error": {"code": "...", "message": "...", "where": "..."}}`. The `id` is the invocation's. `where` is always present and empty when there is nothing to locate. When the code is `command.needs_approval`, the error also carries `"approval": "<approval id>"`.

No handler throws across the registry. An exception that escapes a handler becomes `command.handler_failed`, with the exception's text in the message.

### Errors common to every command

The registry checks an invocation in this order and answers the first fault: the invocation itself, the command's name, the arguments, whether a handler is bound, the thread, and then the approval policy. The arguments come before the handler, so that a caller learns that its call is malformed even from an app that has no handler for the command yet.

| Code | When |
|---|---|
| `command.invalid_invocation` | The invocation itself is malformed: a missing name, a bad id, group or merge key, an unknown origin or an unknown field. |
| `command.unknown` | No command has that name. |
| `command.invalid_args` | The arguments fail the schema, or are nested too deeply or hold too many values. `where` names the value. |
| `command.not_bound` | The command is in the table but no handler is bound yet. |
| `command.wrong_thread` | A `frame_thread` command was invoked directly on a thread other than the frame thread. |
| `command.reentrant` | A handler invoked a command. Handlers call library functions, never commands. |
| `command.needs_approval` | The approval policy asked; the error carries the approval id. |
| `command.denied` | The approval policy or the person denied the invocation. |
| `command.too_many_pending` | 64 invocations already wait for approval. |
| `command.handler_failed` | The handler threw. |
| `command.invalid_result` | The handler's result failed the result schema. The registry checks results only when it is asked to, which the tests do. |
| `command.queue_full` | 1024 invocations already wait in the queue. |
| `command.shutdown` | The queue shut down before the invocation ran or before its approval was decided. |
| `command.approval_timeout` | Nobody decided within the time the caller allowed, so the approval was withdrawn. |
| `command.unknown_approval` | `approve` or `deny` was given an id under which nothing waits. |

## The commands of v0

The flags of each command, and what undo does for the ones that mutate:

| Command | Mutates | Destructive | Frame thread | Undo |
|---|---|---|---|---|
| `app.info` | | | | |
| `app.applets` | | | yes | |
| `app.launch` | | | yes | |
| `catalog.list`, `catalog.describe`, `catalog.sample` | | | | |
| `data.query` | | | | |
| `data.import` | yes | | | Deletes the loaders it created and their outputs. |
| `loader.list`, `loader.describe` | | | | |
| `loader.create` | yes | | | Deletes the loader and its outputs. |
| `loader.update` | yes | yes | | Restores the previous recipe. The outputs stay as the last run made them. |
| `loader.run` | yes | yes | | Cannot be undone. |
| `loader.delete` | yes | yes | | Restores the recipe. Its outputs come back when it runs again. |
| `spec.new`, `spec.open` | yes | | yes | Restores the spec that was open before, with its unsaved changes. |
| `spec.get`, `spec.validate` | | | yes | |
| `spec.set_text` | yes | | yes | Restores the previous text. |
| `source.add`, `source.patch`, `source.remove` | yes | | yes | Restores the previous sources. |
| `panel.create`, `panel.patch`, `panel.remove` | yes | | yes | Restores the previous panels and layout. |
| `panel.render` | | | yes | |
| `layout.place`, `layout.set` | yes | | yes | Restores the previous layout. |
| `applet.save` | yes | yes | yes | Cannot be undone. |
| `applet.open` | yes | | yes | Restores the spec that was open before. |
| `applet.list` | | | | |
| `capture.frame` | | | yes | |
| `history.undo`, `history.redo` | yes | | yes | Each reverses the other. |
| `history.list` | | | yes | |

`app.launch` changes the page the app shows, not the workspace, so it does not mutate in this sense and is not journaled; Caliper API v0 already offers it without approval. `history.undo` and `history.redo` mutate but are special: they are not journaled as `do` events and do not become steps of their own.

A "diagnostic" below is an object `{"code", "message", "where"}`, as the spec validator gives them. Every edit of the open spec validates the result and refuses an edit that leaves the spec with an error, with the validator's own code (for example `spec.unknown_primitive`), so the open spec is valid after every command and the preview can always draw it. Warnings, such as `spec.not_portable`, come back in the result.

### app

| Command | Arguments | Result | Errors |
|---|---|---|---|
| `app.info` | none | `abi_epoch`, `host`, `libtorch`, `sdk`, `services`, `ui_stack`, as `GET /v0/info` | |
| `app.applets` | none | `applets`: a list of `folder`, `id`, `reason`, `status`, `version`, as `GET /v0/applets` | |
| `app.launch` | `id`: the manifest id of an applet that `app.applets` lists | `already_open`, `id`, `status`, `version` | `app.applet_not_found`, `app.launch_refused`, `app.launch_failed` |

### catalog

The [Catalog](catalog.md) page gives the entry, the sample and the limits.

| Command | Arguments | Result | Errors |
|---|---|---|---|
| `catalog.list` | `kind`, `prefix` (of the name), `limit` (1 to 1000, default 100) | `entries`, `truncated`, `warnings` | |
| `catalog.describe` | `id` | `entry` | `catalog.invalid_id`, `catalog.not_found` |
| `catalog.sample` | `id`, `limit` (rows, 1 to 1000, default 20), `start` and `count` (per axis of a tensor) | the sample | `catalog.invalid_id`, `catalog.not_found`, `catalog.no_data`, `catalog.bad_request`, `catalog.provider_failed` |

### data

| Command | Arguments | Result | Errors |
|---|---|---|---|
| `data.query` | `sql` (one read-only SELECT), `limit` (1 to 100,000 rows, default 1000), `timeout_ms` (100 to 30,000, default 5000) | `columns`, `rows`, `truncated`, `elapsed_ms` | `data.not_select`, `data.sql_error`, `data.timeout`, `data.unavailable` |
| `data.import` | `paths` (1 to 256), `run` (default true) | `loaders`: `id`, `path`, `reader`, `outputs`, `ran` for each; `refused`: `path`, `code`, `message` for each path refused | |

`data.import` refuses a path one by one, with the loader store's codes (`loader.pickle_refused`, `loader.no_reader`, `loader.reader_mismatch`, `loader.io`) and `loader.path_refused`, and does not fail as a whole. A path outside the workspace is accepted only from the `ui` and `builder` origins, which stand for a file the person dropped or picked, and is then granted; from `agent`, `mcp` and `api` it is refused with `loader.path_refused` unless an earlier grant of the person already covers it, so an agent cannot read an arbitrary file of the machine through a loader. The same rule holds for the input path of `loader.create` and `loader.update`. A Python loader is created but never run by an import.

`loader.describe` gives `last_run.elapsed_ms` as null when the session did not make that run itself, because the store does not record a run's duration.

### loader

| Command | Arguments | Result | Errors |
|---|---|---|---|
| `loader.list` | none | `loaders`: `id`, `title`, `reader`, `path`, `status`, `outputs`, `updated_at` for each | |
| `loader.describe` | `id` | `id`, `title`, `text` (the TOML recipe), `reader`, `path`, `status`, `last_run`, `outputs` | `loader.not_found` |
| `loader.create` | `text`, `run` (default false) | `id`, `outputs` | `loader.invalid`, `loader.exists`, `loader.path_refused`, `loader.unknown_reader`, and the codes of `loader.run` when it runs |
| `loader.update` | `id`, `text`, `run` (default false) | `id`, `outputs` | `loader.not_found`, `loader.invalid`, `loader.id_mismatch`, `loader.path_refused` |
| `loader.run` | `id` | `id`, `outputs` (`id`, `rows`, `shape` for each), `elapsed_ms` | `loader.not_found`, `loader.read_failed`, `loader.pickle_refused`, `loader.sql_refused`, `loader.python_not_available`, `loader.python_needs_approval` |
| `loader.delete` | `id`, `keep_outputs` (default false) | `id`, `removed` (catalog ids) | `loader.not_found`, `loader.unsupported` (for `keep_outputs`, in this phase) |

`loader.run` is destructive because a run replaces the tables its previous run made, and because it may run a Python loader. An app may give a narrower policy that allows native readers without asking; the default follows the flag. Running a Python loader's script, though, always asks every origin except `ui` and `builder`, whatever the flags say; see [Approval](#approval).

### spec, source, panel and layout

These commands edit the open spec, the one spec the builder shows. There is at most one open spec per workspace.

| Command | Arguments | Result | Errors |
|---|---|---|---|
| `spec.new` | `id`, `title`, `description` | `id`, `text` | `spec.invalid_id` |
| `spec.open` | `path` (relative to the workspace, ending in `.spec.toml`) | `id`, `path`, `warnings` | `spec.not_found`, `spec.path_refused`, validator codes |
| `spec.get` | none | `id`, `path`, `dirty`, `text`, `json` | `spec.no_open_spec` |
| `spec.set_text` | `text` (at most 256 KiB), `dry_run` | `id`, `warnings`, `diff`, `applied` | `spec.no_open_spec`, validator codes |
| `spec.validate` | `text` (optional; the open spec when absent) | `valid`, `errors`, `warnings` | `spec.no_open_spec` |
| `source.add` | `id`, one of `catalog`, `sql`, `from`, and `transform`; `dry_run` | `id`, `warnings`, `diff`, `applied` | `spec.no_open_spec`, `source.exists`, validator codes |
| `source.patch` | `id`, `patch`, `dry_run` | `source`, `warnings`, `diff`, `applied` | `source.not_found`, `source.id_immutable`, validator codes |
| `source.remove` | `id`, `dry_run` | `id`, `warnings`, `diff`, `applied` | `source.not_found`, `source.in_use` |
| `panel.create` | `id`, `primitive`, `title`, `bind`, `options`, `place`, `dry_run` | `id`, `warnings`, `diff`, `applied` | `spec.no_open_spec`, `panel.exists`, `panel.not_found` (the place target), validator codes |
| `panel.patch` | `id`, `patch` (a JSON merge patch, RFC 7386, of the panel), `dry_run` | `panel`, `warnings`, `diff`, `applied` | `panel.not_found`, `panel.id_immutable`, validator codes |
| `panel.remove` | `id`, `dry_run` | `id`, `warnings`, `diff`, `applied` | `panel.not_found` |
| `panel.render` | `id`, `width`, `height` (64 to 4096; 800 by 500 by default) | `png_base64`, `width`, `height` | `panel.not_found`, `panel.render_failed` |
| `layout.place` | `panel`, `target`, `where` (`beside`, `above`, `below` or `tabbed`), `ratio`, `dry_run` | `layout`, `warnings`, `diff`, `applied` | `panel.not_found`, `layout.same_panel` |
| `layout.set` | `layout`, `dry_run` | `layout`, `warnings`, `diff`, `applied` | `layout.invalid`, `panel.not_found` |

- `spec.validate` reports an invalid spec in its result; it fails only when there is nothing to validate.
- `source.remove` refuses a source that a panel or another source still uses, and names the first user in `where`.
- `panel.create` without `place` puts the new panel to the right of the whole layout, with half the width.
- `layout.place` moves a panel that is already in the layout. `beside` puts it to the right of the target. `ratio` is the share of the space that the placed panel gets, from 0.05 to 0.95, 0.5 by default; the handler turns it into the spec's `ratio`, which is the share of the first child.
- The bindings and options that `panel.create` and `panel.patch` accept are every slot and option of the [spec format](spec-format.md), with its enumerations and bounds where the schema subset can state them; a test keeps the table in line with the format's fixtures. Extension tables (`x`) are open objects, which closed tool schemas cannot describe, so only `spec.set_text` sets them; the patches keep an item's `x`. `panel.create` needs no `bind`, since a text panel may have only `options.text`.
- Every command of this table that edits the open spec returns the change as a unified diff of the spec's canonical TOML (`diff`) and whether it was applied (`applied`), and takes `dry_run`, which checks the change and returns its diff and warnings without applying it, so that the change can be shown for approval first. A dry run has no effect at all: its handler reports `no_effect`, so the registry keeps the redo stack, files no step and writes no journal line, as for a read (see [Steps and records](#steps-and-records)). An edit that leaves the spec as it was reports the same.

The layout in `layout.set` and in the results is a flat list of nodes, because tool schemas cannot be recursive: `{"root": "<id>", "nodes": [{"id": "n1", "split": "horizontal", "ratio": 0.6, "children": ["heat", "n2"]}, ...]}`. `split` is `horizontal`, `vertical` or `tabs`; `ratio` is the share of the first child, or null for tabs. A child is a node id or a panel id; node ids must not be panel ids, and a node is used once. `root` names a node, or the one panel when the layout is a single panel. The handler turns this into the spec's nested `[layout]` table and back.

### applet, capture and history

| Command | Arguments | Result | Errors |
|---|---|---|---|
| `applet.save` | `id` (default: the spec's applet id), `overwrite` (default false) | `id`, `manifest`, `spec` (paths relative to the workspace) | `spec.no_open_spec`, `applet.exists`, `applet.write_failed`, validator codes |
| `applet.open` | `id` | `id`, `title`, `warnings` | `applet.not_found`, `applet.not_spec`, validator codes |
| `applet.list` | none | `applets`: `id`, `title`, `path`, `updated_at` for each | |
| `capture.frame` | none | `png_base64`, `width`, `height` | `capture.unavailable` |
| `history.undo` | `steps` (1 to 100, default 1) | `steps` (one summary for each step undone), `can_undo`, `can_redo` | `history.nothing_to_undo`, `history.undo_failed`, `history.changed`, `command.needs_approval` |
| `history.redo` | `steps` (1 to 100, default 1) | `steps`, `can_undo`, `can_redo` | `history.nothing_to_redo`, `history.redo_failed`, `history.changed`, `command.needs_approval` |
| `history.list` | `limit` (1 to 200, default 50) | `undo`, `redo` (step summaries, newest first), `can_undo`, `can_redo` | |

A step summary is `{"id", "label", "count", "group", "origin", "at"}`: the step's number, the name of its first command, how many commands it holds, its group or null, the origin of its first command and the UTC time of its last one.

## Undo

### Steps and records

The history is one undo stack and one redo stack per workspace, held by the registry. A **step** is one entry on a stack: one or more **records** that undo and redo together. A mutating command that succeeds returns its result and one undo record, which is one of:

- **a closure pair**, `undo` and `redo`, two functions that each return an error or nothing;
- **an inverse invocation**, a command name and arguments that reverse the command; redo runs the original invocation's handler again;
- **none**, for a command that cannot be undone.

A handler may instead report that its command changed nothing, with `no_effect` on its output, whether it succeeded or failed: a dry run, or an edit that left everything as it was. The registry then treats the invocation as a read: it files no step and ignores any undo record, keeps the redo stack and an open merge, and writes no journal line unless the journal records read-only commands. A preview by an agent therefore never costs the person their redo.

Undoing a step runs its records' `undo` in reverse order; redoing it runs their `redo` in order. An inverse invocation runs through the registry's handler table, with its arguments validated, but it is not journaled as a `do` event, not recorded as a step and not sent to the approval policy on its own (see [Undo, redo and approval](#undo-redo-and-approval)). The inverse kept is always the one from the first time the command ran: an inverse must restore the state from before, given the state after, and a handler that makes ids must make them from the state, so that a redo makes the same ids again.

### Groups

An invocation with a group joins the step on top of the undo stack when that step has the same group; otherwise it begins a new step that carries the group. An agent's proposal of several edits, sent with one group, is therefore one step. The history stays a stack: if another command lands between two invocations of a group, the rest of the group begins a second step, because joining a step that is not on top would undo changes out of order.

### Merging

A slider drag sends many commands that should undo as one. An invocation with a merge key merges into the last record of the top step when all of these hold:

- that record has the same merge key and the same origin;
- the invocation's group is the step's group (both may be absent);
- the step is still open: no undo or redo happened since its last command;
- the last command of the record ran less than 1000 ms ago (configurable);
- the record holds fewer than 4096 merged commands.

A merged record keeps the `undo` of its first command and takes the `redo` of its newest one, so a long drag costs one record, not one per frame. This is correct only when every command of the run sets the same thing absolutely, which is what a merge key promises; a command that changes something relatively (adds 0.1) must not carry one. For the same reason, the check that an undo makes against changes from outside (see below) must, for a command meant to merge, test only what the whole run leaves unchanged, such as that the panel still exists, and not the value its own command set, because the kept undo belongs to the first command of the run.

### Redo

Undo moves a step from the undo stack to the redo stack, and redo moves it back. Any mutating command that succeeds clears the redo stack, including one that cannot be undone and excepting one whose handler reports no effect, because the redo steps were recorded against a state that no longer holds. A command that fails, a read-only command and a refused invocation leave both stacks as they were.

### Commands that cannot be undone

A command whose handler returns no undo record (`loader.run`, `applet.save`) records no step. It clears the redo stack, keeps the undo stack and closes merging. Undo then passes it by: after a save, undo reverts the open spec and leaves the saved files as they are, as in a text editor, and after a run, undoing an earlier `loader.update` restores the recipe and leaves the tables as the run made them. The descriptor table says in each such command's summary that it cannot be undone, and its row in the table above says so too.

### When the workspace changed underneath

Changes made outside the command layer, such as a loader file edited in another program, are not in the history. An undo record must check that the state it is about to replace is the state its command left, for example by keeping the SHA-256 of the text it wrote, and must fail with `history.conflict` when it is not, rather than overwrite a change it never saw. When the app reloads a changed file into the open spec, it does so through a command, which becomes a step of its own.

When the `undo` of a record fails, the history first redoes the records of the step that it had already undone, so the state is as it was before the attempt. It then drops that step and every older step from the undo stack, because older steps assume this one was undone first, and keeps the redo stack. The command answers `history.undo_failed`, naming the command whose undo failed and its reason. A failing `redo` is handled the same way: the partly redone step is rolled back, the step and every step above it on the redo stack are dropped, and the command answers `history.redo_failed`. If the rollback fails as well, the state is unknown: the history clears both stacks and the command answers `history.inconsistent`.

`history.undo` with `steps` greater than 1 undoes one step at a time and stops at the first failure; the error's message says how many steps were undone before it.

### Undo, redo and approval

An agent must not be able to do through the history what it could not do directly. Before it undoes or redoes a step, the history asks the approval policy about each destructive command in that step, with the origin of the `history.undo` or `history.redo` invocation: for an undo, the inverse invocations and the commands whose undo is a closure; for a redo, the commands themselves. A deny denies the whole command; an ask parks the history invocation for approval, with the ids of the steps it would touch. When it is approved, the history checks that those steps are still the ones on top, and answers `history.changed` if they are not.

### Bounds

| Bound | Default | What happens beyond it |
|---|---|---|
| Steps on the undo stack | 200 | The oldest step is dropped. |
| Memory held by both stacks | 64 MiB | The oldest undo steps are dropped until the total fits. The newest step is always kept, so the last command can always be undone. |
| Commands merged into one record | 4096 | The next command begins a new record. |
| Merge window | 1000 ms | The next command begins a new record. |

The memory of a step is the sum of its records' costs. A closure pair costs what its handler declares (at least 256 bytes); an inverse invocation costs the size of its arguments and of the original arguments as JSON text, plus 128 bytes.

## The journal

### Format

The journal is append-only JSON Lines at `journal/commands.jsonl` in the workspace, one event per line:

```json
{"seq": 12, "ts": "2026-11-15T12:00:00.125Z", "origin": "agent", "name": "panel.patch", "args": {"id": "heat", "patch": {"title": "Entropy"}}, "ok": true, "group": "turn-7", "event": "do"}
{"seq": 13, "ts": "2026-11-15T12:00:04.500Z", "origin": "ui", "name": "history.undo", "args": {}, "ok": true, "group": "turn-7", "event": "undo", "seqs": [10, 11, 12]}
```

- `seq` is a positive integer that increases by one with each line and continues across reopening and rotation.
- `ts` is the UTC time in ISO 8601 with milliseconds.
- `group` is the invocation's group, or null.
- A failed event carries `"ok": false` and `"error": "<code>"`.
- An `undo` or `redo` event is written for each step that `history.undo` or `history.redo` undid or redid. Its `name` is that command, its `args` are `{}`, its `group` is the step's group, and `seqs` lists the `seq` of every `do` event of the step. A failed undo or redo is written with `ok: false` and the error code.

The journal records every invocation of a mutating command that passed argument validation: the ones that succeeded, the ones whose handler failed, and the ones that were denied, by the policy or by the person. An invocation whose handler reports no effect is a read and is not recorded. An invocation refused earlier (unknown command, invalid arguments, not bound, wrong thread) ran nothing and is not written, since its arguments may be malformed. An invocation waiting for approval is written once, when it is decided or withdrawn. Read-only commands are written only when the journal is opened with `include_read_only`, as `do` events. Results are never written.

Each line is written and flushed before the command's result is returned, so a crash loses no line that a client saw acknowledged, apart from what the operating system had not yet written to disk; the journal does not call `fsync`. A journal that cannot be written does not fail the command: the registry keeps the error and reports it to the app through `journal_error()`.

### What is redacted

- **Large values.** Before a line is written, every value inside `args` is measured, innermost first. A string longer than 64 KiB is replaced by `{"$redacted": {"sha256": "<hex of its UTF-8 bytes>", "bytes": <its length>}}`. An array or object whose JSON text is still longer than 64 KiB after its members were redacted is replaced the same way, with the SHA-256 of its compact JSON text. The `args` object itself can be replaced, so no line holds more than about 64 KiB of arguments.
- **Secrets.** No command takes a token or a key, and a test of the table checks that no property is named like one. As a second line of defence, the journal replaces the value of any key named `token`, `secret`, `password`, `api_key`, `apikey` or `authorization`, or ending in `_token`, `_secret`, `_password` or `_api_key`, by `{"$redacted": "secret"}`. The session token of Caliper API travels in a header and never reaches a command.

### Rotation and sequence numbers

When the next line would take the file past 16 MiB, the file is renamed `commands.1.jsonl`, older files shift up to `commands.4.jsonl`, the oldest is deleted, and a new `commands.jsonl` begins. Both bounds are options. When the journal is opened, it reads the end of the newest file that has a line (the current file, else `commands.1.jsonl`) to find the last `seq`, and continues from it. If the current file does not end with a newline, because a write was cut short, it first writes one, so the cut line stays a damaged line of its own and the next line is whole.

### Reading and replay

The reader reads the rotated files from oldest to newest and then the current file. It refuses a line longer than 1 MiB and skips a line that is not a valid event; both are reported with the file and line number, as are a `seq` that does not increase and a gap in the sequence.

Replay rebuilds a state from the events, given a function that applies one `do` event. It does not run any undo record. It first works out which `do` events are in effect at the end: an `undo` event takes the `do` events in its `seqs` out of effect, and a `redo` event puts them back; failed events change nothing. It then applies the events still in effect, in `seq` order, to the initial state. Because the history is a stack, the steps in effect at any moment are exactly the done steps in their original order, so this gives the same state as the session did, provided that each command's effect is fully described by its arguments. Replay stops with an error in these cases:

- `journal.incomplete`: the first `seq` is not 1, because the files that held the start were rotated away;
- `journal.redacted`: an event in effect has redacted arguments;
- `journal.unreplayable`: an event with the code `history.inconsistent`.

Commands that cannot be undone changed things outside the state that a replay rebuilds, such as files and tables, and those effects happened once. An apply function that rebuilds the open spec skips them.

## Approval

An `ApprovalPolicy` is a function that is given a descriptor and an invocation and answers `allow`, `deny` or `ask`. The default policy allows every invocation from `ui` and `builder`, allows every command that is not destructive, and asks for a destructive command from `agent`, `mcp` and `api`.

On `ask`, the registry parks the invocation under a new approval id and answers `command.needs_approval` with that id in `error.approval`. The app shows the pending approvals (`pending_approvals()` lists them with the descriptor's summary and the arguments) and calls `approve(id)` or `deny(id)`. `approve` runs the parked invocation as if the policy had allowed it and returns its result; `deny` completes it with `command.denied`. Both are journaled then. At most 64 invocations wait at once; more are refused with `command.too_many_pending`. The registry sets no time limit of its own, because the person may be away; a caller that cannot wait for ever gives the queue a limit (below), after which the approval is withdrawn with `command.approval_timeout` and journaled like a denial. Pending approvals also end, with `command.shutdown`, when the queue shuts down.

The builder session installs a policy that adds two rules, decided per invocation because a descriptor's flags are fixed per command:

- `loader.run` of a Python loader, and `loader.create` or `loader.update` of one with `run` set, ask every origin except `ui` and `builder`, each time, because running a Python loader runs the user's code. The handler passes the store's permission to run the script (`RunOptions::allow_python`) only for `ui` and `builder`, or for an invocation that the policy asked about and the person approved, so an app policy that allowed everything would still get `loader.python_needs_approval`. A redo never runs a script.
- `applet.save` from `agent`, `mcp` or `api` asks only when it would overwrite a saved applet; writing a new one destroys nothing.

There is no command and no endpoint that approves or denies. Only the app's own UI calls `approve` and `deny`, so an agent cannot approve its own request, and approval ids need not be secret.

## Threads and the queue

- `CommandQueue::submit()` may be called from any thread. It returns a `std::future` of the result, and refuses with `command.queue_full` when 1024 invocations already wait.
- `drain()` runs the waiting invocations in submission order on the calling thread, optionally at most a given number per call. The app calls it on the frame thread between two frames.
- When the policy asks, the future does not complete with `command.needs_approval` by default: it waits for the decision. The queue's `approve(id)` and `deny(id)` may be called from any thread; the decision runs at the next `drain()` and completes both the waiting future and the future that `approve` or `deny` returned. A submit may set `approval_timeout`; when the person has not decided by then, the next `drain()` withdraws the approval and completes the future with `command.approval_timeout`. A caller that must answer at once submits with `wait_for_approval = false`, gets `command.needs_approval` straight away, and later reads the outcome with `approval_outcome(id)`, which the queue keeps for the last 64 decisions.
- `shutdown()` completes every waiting future, and every future waiting for an approval, with `command.shutdown`. A later `submit()` returns a future that is already complete with that error. The destructor shuts down.
- The registry runs one invocation at a time, under one lock that also covers the history, the journal and the approvals. When the app names its frame thread to the registry, a `frame_thread` command invoked directly on another thread is refused with `command.wrong_thread`. In this phase every command runs in `drain()`; a later phase may drain the commands that do not need the frame thread on a worker, so that a long `loader.run` does not hold a frame.

## Caliper API

After sprint 2 merges, Caliper API gains these endpoints, under the same loopback, `Host`, `Origin` and session-token checks as API v0:

- **`GET /v0/commands`** returns the descriptor table, the output of `command_table_to_json()`: `{"version": 0, "commands": [{"name", "summary", "args", "result", "examples", "mutates", "destructive", "frame_thread"}]}`, in the table's order.
- **`POST /v0/commands/{name}`** takes the `args` object as a JSON body of at most 1 MiB with `Content-Type: application/json`, and returns the result envelope with HTTP 200 whether `ok` is true or false. The server assigns the invocation id. The origin is `api`, or `mcp` or `agent` when the request carries `X-Caliper-Origin: mcp` or `X-Caliper-Origin: agent`; any other value is refused with 400, so a request never claims `ui` or `builder`. An optional `X-Caliper-Group` header gives the group. A PNG result is base64 in `result.png_base64`, with `result.width` and `result.height`.
- **Approval over the API** follows contracts section 15. A call with the origin `mcp` or `agent` that needs approval is held while the person decides in the app, for up to 120 seconds: the server submits it with `approval_timeout` set to 120 seconds and waits for the future. Approved, the call answers with the command's result; denied, with `command.denied`; after the time limit, with `command.approval_timeout`, and the approval is withdrawn. `caliper-mcp` and `caliper-agent` allow at least 130 seconds for a call. A call with the origin `api` submits with `wait_for_approval = false` and gets `command.needs_approval` with the approval id at once.
- **`GET /v0/approvals/{id}`** returns `{"approval": "<id>", "state": "pending" | "decided" | "unknown", "result": <envelope or null>}`, from `approval_outcome(id)` and the registry's pending list, so that an `api` client can learn the outcome of a request that needed approval. This endpoint is an elaboration of this page, not of the contracts.

Transport faults keep their HTTP status: 401 and 403 from the checks, 413 for a body over 1 MiB, 415 for another content type, 400 for a body that is not JSON, and 503 and 504 as in API v0. A body that is JSON but not an object is answered 200 with `command.invalid_args`. A command name that does not exist is answered 200 with `command.unknown`, not 404, so that a 404 means only that the app has no command layer, which is how `caliper-mcp` tells an older app apart. The 5-second bound of API v0 applies to the wait before a command starts on the frame thread; once it has started, the request waits for its result, and a call held for approval waits as described above. The server never logs a request's token, and it builds the invocation from the path, the body and the two headers only, so no header value reaches the journal.

## caliper-mcp and the Python sidecar

The descriptor table is the one source of tool definitions. `caliper-mcp` links the C++ table, and the BYOK sidecar `caliper-agent` loads its JSON form, `src/builder/commands/command_table.json`, which is the output of `command_table_to_json()` printed with two-space indents, sorted keys and a final newline. A test fails when the file and the C++ table differ. After a change to the table, regenerate the file with the tool target of the commands module:

```sh
cmake --build <build> --target caliper_builder_command_table
<build>/commands/caliper_builder_command_table --write src/builder/commands/command_table.json
```

Both clients name a tool after its command, with `_` for `.`: `panel_create` for `panel.create`, `spec_set_text` for `spec.set_text`. Since a group has no underscore, the first underscore of a tool name marks the dot, and the mapping is reversible. A tool's description is the command's summary; its input schema is the command's `args`. `caliper-mcp` may mark a tool read-only when `mutates` is false and destructive when `destructive` is true. For strict tools, the sidecar removes the keywords that the Claude API does not take in a strict schema (`minimum`, `maximum`, `minLength`, `maxLength`, `minItems`, `maxItems`, `pattern`, `default`) from what it sends; nothing is lost, because the command layer checks every argument against the full schema.

## Changes to the contracts

These are additions to, or fixes of, sprint 3 contracts section 7, each the smallest that worked:

- **`merge_key` on the invocation.** Section 7 names merge keys but no envelope carries one. The invocation gains an optional `merge_key`, because the source of a gesture (a slider) knows when commands belong together and a handler cannot tell a drag from two deliberate edits.
- **`approval` on the error.** The approval id of `command.needs_approval` needs a field; it is `error.approval`.
- **`seqs` on `undo` and `redo` events, and `error` on failed events.** The journal line of section 7 cannot be replayed across a merged step, because merge keys are not journaled. Naming the `seq` of each `do` event of the step makes replay exact without journaling the history's internals.
- **`examples` on the descriptor**, kept outside the schema, so that strict schemas stay plain and clients can show sample calls.
- **`minItems` and `maxItems`** join the schema subset, so that the schemas state the bounds of their lists.
- **The flat layout** of `layout.set`, and the one open result object, `spec.get`'s `json`, follow from closed, non-recursive tool schemas.

## Decisions

- **Destructive means saved data.** An edit in memory that undo reverses exactly is not destructive, so the builder and agents edit the open spec without a prompt, and approval is asked when work leaves memory: overwriting a loader, running one, deleting one, saving an applet.
- **Groups join only the top step.** It keeps the history a stack, so every undo reverses the newest changes and the state stays consistent.
- **A failed undo drops the older steps.** Older steps assume the failed one was undone first; keeping them would let a later undo apply to a state it was not recorded against.
- **Commands that cannot be undone are passed by, not barriers.** This is how editors treat a save, and it keeps undo useful after a run or a save.
- **Replay applies the steps in effect instead of running undo records.** It needs nothing but the journal and the apply function, and it is exact because the history is a stack.
- **A command meant to merge checks only what a whole run leaves unchanged.** The history keeps the first undo of a merged run, so a check against the value the first command set would report a conflict after any drag.
- **Journaled invocations are those that passed validation.** A refused invocation ran nothing, and its arguments may be malformed or hostile.
- **One lock for the registry in this phase.** It makes ordering and the history simple; running long read-only commands off the frame thread is left for the wiring.
- **No approval through commands or endpoints.** Only the app's UI decides, so an agent cannot approve itself.
- **Paths outside the workspace only from `ui` and `builder`.** Loaders would otherwise let an agent read any file the app can read.
- **The table's summaries are written for an agent that has never seen Caliper**: each says what the command does, what its ids are and which command gives them.

## Open questions

- Whether a later phase runs the commands that do not need the frame thread on a worker, and how the registry's lock is split then.
- Whether the history should be saved with the workspace, so that undo survives a restart; in this phase it lives in memory and the journal is the lasting record.
