# Command layer

The command layer is the one surface through which everything acts on a workspace: the app's UI, the WYSIWYG builder, the side chat and the widget agent, outside agents over MCP and programs over Caliper API. Each action is a typed command with a JSON Schema for its arguments and one for its result, so every client gets the same validation, the same undo history, the same journal and the same approval rules. This page elaborates sections 7 and 8 of the sprint 3 contracts and records the decisions taken while building the library. The [Instrument builder](instrument-builder.md) page gives the background, and the Trust model (`docs/instruments/trust-model.md`) explains why everything an agent builds is data.

## What is built and what is planned

Built on this branch, in the library `caliper_builder_commands` (`src/builder/commands/`, headers in `src/builder/include/caliper_builder/commands/`):

- descriptors, invocations and results, with their JSON forms;
- the registry and the built-in schema validator;
- the undo history, with groups, merging, the redo rule and its bounds;
- the journal, its reader and its replay;
- the approval policy and the pending-approval flow, with a review that a pending approval carries (the diff of a spec edit);
- follow-up invocations, which a handler returns and the registry runs as one undo step (the commit of a stacked preview);
- the queue, with `submit()` from any thread, `drain()` on the caller's thread and `peek()`, which tells the draining side what runs next;
- an observer on the journal (`JournalOptions::on_append`), through which the app shows each journaled command in its Activity pane;
- the descriptor table of every command of v0, as data, and its checked-in JSON form `src/builder/commands/command_table.json`;
- agent permissions: what each command does as data (`effect`), the modes of the side chat's agent and of `caliper-mcp`'s clients, rules for a session and saved ones, and `allowed_by` in the journal ([Agent permissions](#agent-permissions); `permissions.h`).

Built in the host, in `src/host/api/`: the command endpoints of Caliper API described [below](#caliper-api), served by the app's own API server through a small interface, `CommandGateway` (`command_gateway.h`), and the gateway over the builder session, `caliper_api_session` (`session_gateway.h`, built with `CALIPER_WITH_BUILDER`). They are tested with a real session behind the server and with `caliper-mcp` against it. The app serves them once it sets the gateway on its server, which is one call when the session is in the app; until then they answer 404. The reference page is `docs/agents/http-api.md`.

The [builder session](builder-session.md) builds two things on these: the review of spec edits for the origins `agent` and `mcp` ([Approval](#approval)) and stacked previews ([preview](#preview)).

Planned:

- The handlers of the catalog, data, loader, spec, source, panel, layout and instrument commands are bound by the [builder session](builder-session.md), which the app now holds for its workspace. Until a handler is bound, its command answers `command.not_bound`.
- The app binds `app.info`, `app.instruments`, `app.launch` and `capture.frame`, and binds `panel.render` to `command.not_available` until the spec runtime draws panels ([Builder session](builder-session.md#the-session-in-the-app)). The live catalog providers come later.
- The app sets the command gateway on its API server when it owns a session: `api.set_command_gateway(caliper_api::make_session_gateway(session))`, and `nullptr` before the session closes.

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
| `agents` | An agent may run the command: the origins `agent` and `mcp`. It is false only for `history.undo` and `history.redo`, the person's own tools; the registry refuses them to agents with `command.not_allowed`, and the clients that build an agent's tools from the table leave them out. |
| `effect` | What the command does, by which an agent's mode decides it: `read` (it reads, or proposes without changing the workspace), `edit` (it edits the open spec in memory) or `change` (anything else that changes something). See [Agent permissions](#agent-permissions). |

`destructive` is about saved data. An edit of the open spec in memory is not destructive, even when it removes a panel, because undo reverses it exactly. Saving an instrument, which can overwrite its files, is.

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

**A null on an optional argument is the argument left out.** Before it validates, the registry removes from the arguments every property whose value is null when the property is not required and its own schema does not admit null (by its type or its enum), following the schema into nested objects and the items of arrays (`drop_null_optional_args()` in `schema.h`). The policy, the review on the approval card, the journal and the handler then see the call as it is understood, without the nulls. A null on a required argument is still refused with `command.invalid_args`, and a null that the schema admits stays a value: in a merge patch such as `panel.patch`, `"title": null` takes the title away. The values of `additionalProperties` are left alone. **Decision:** taken after a local model sent `source.add` with `sql`, `from` and `transform` null, which every client refused, and the model had to try each call again; `caliper-mcp` and the `caliper-agent` sidecar, which check arguments before they send them, apply the same rule.

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
| `origin` | `ui` (the app's own UI), `builder` (the WYSIWYG builder), `agent` (the side chat or the widget agent), `mcp` (an outside agent through `caliper-mcp`) or `api` (a program through Caliper API). |
| `group` | Optional. Invocations that share a group undo as one step. |
| `merge_key` | Optional. Consecutive invocations with the same merge key merge into one step. |
| `client` | Optional, and never from a request. Which agent the app started sent it: `chat` for the side chat's agent, `widget` for the widget agent's sidecar. The app's API server sets it from the token it issued for that agent, so an agent cannot choose it; an agent's mode applies by it ([Agent permissions](#agent-permissions)). 0 to 32 characters of `a-z`. |
| `preview` | Optional. The id of a stacked preview: an edit of the open spec applies to the preview instead, and `spec.get`, `spec.validate` and `panel.render` read it ([preview](#preview)). Commands that neither read nor edit the open spec ignore it, and any other command that mutates is refused with `preview.not_previewable`. Over Caliper API it is the header `X-Caliper-Preview`. |

`group`, `merge_key` and `preview` match `[A-Za-z0-9._:/-]{1,128}`. An invocation with an unknown field is refused with `command.invalid_invocation`, so a typo does not pass silently.

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
| `command.not_allowed` | The origin may not run the command: a command that mutates from the `viewer` origin, or a command whose `agents` is false (`history.undo`, `history.redo`) from `agent` or `mcp`. It comes before any policy or history work and is not journaled. |
| `command.not_bound` | The command is in the table but no handler is bound yet. |
| `command.not_available` | The handler is bound but what it needs is not in this app yet, such as `panel.render` before the spec runtime, or the app's information outside the app. |
| `command.wrong_thread` | A `frame_thread` command was invoked directly on a thread other than the frame thread. |
| `command.reentrant` | A handler invoked a command. Handlers call library functions, never commands. |
| `command.needs_approval` | The approval policy asked; the error carries the approval id. |
| `command.denied` | The approval policy or the person denied the invocation. |
| `command.too_many_pending` | 64 invocations already wait for approval. |
| `command.handler_failed` | The handler threw, or returned a follow-up invocation that cannot run (see [Follow-ups](#follow-ups)). |
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
| `app.instruments` | | | yes | |
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
| `instrument.save` | yes | yes | yes | Cannot be undone. |
| `instrument.open` | yes | | yes | Restores the spec that was open before. |
| `instrument.list` | | | | |
| `preview.begin`, `preview.diff`, `preview.discard` | | | yes | |
| `preview.commit` | yes | | yes | One step reverts every edit of the preview. |
| `review.status` | | | | |
| `capture.frame` | | | yes | |
| `history.undo`, `history.redo` | yes | | yes | Each reverses the other. |
| `history.list` | | | yes | |

The table holds 40 commands. `app.launch` changes the page the app shows, not the workspace, so it does not mutate in this sense and is not journaled; Caliper API v0 already offers it without approval. `history.undo` and `history.redo` mutate but are special: they are not journaled as `do` events and do not become steps of their own, and they are the person's tools, which the origins `agent` and `mcp` may not run.

A "diagnostic" below is an object `{"code", "message", "where"}`, as the spec validator gives them. Every edit of the open spec validates the result and refuses an edit that leaves the spec with an error, with the validator's own code (for example `spec.unknown_primitive`), so the open spec is valid after every command and the preview can always draw it. Warnings, such as `spec.not_portable`, come back in the result.

### app

| Command | Arguments | Result | Errors |
|---|---|---|---|
| `app.info` | none | `abi_epoch`, `host`, `libtorch`, `sdk`, `services`, `ui_stack`, as `GET /v0/info` | |
| `app.instruments` | none | `instruments`: a list of `folder`, `id`, `reason`, `status`, `version`, as `GET /v0/instruments` | |
| `app.launch` | `id`: the manifest id of an instrument that `app.instruments` lists | `already_open`, `id`, `status`, `version` | `app.instrument_not_found`, `app.launch_refused`, `app.launch_failed` |

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
| `spec.get` | `instrument` (optional; the open spec when absent) | `open`, then `id`, `path`, `dirty`, `text`, `json`; with no spec open and no `instrument`, `open` false alone, which is a read and not an error | `instrument.not_found`; `spec.no_open_spec` only when it names a preview |
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

### preview

A stacked preview is a scratch copy of the open spec on which edits stack without changing the open spec, the history or the journal, so that edits that build on each other, such as a derived source and a panel that binds it, can be shown together and committed as one step. The widget agent proposes every change this way ([Widget agent](widget-agent.md#preview-and-apply-on-the-command-layer)).

| Command | Arguments | Result | Errors |
|---|---|---|---|
| `preview.begin` | `id` (optional; `preview-<n>` is made when absent), `panel` or `source` (optional, the scope) | `id`, `panel`, `source` | `spec.no_open_spec`, `preview.exists`, `preview.too_many`, `preview.bad_scope`, `panel.not_found`, `source.not_found` |
| `preview.diff` | `id` | `id`, `edits`, `diff`, `warnings`, `panel`, `source` | `preview.not_found`, `preview.expired`, `history.conflict` |
| `preview.commit` | `id` | `id`, `edits`, `diff`, `warnings` | `preview.not_allowed`, `preview.not_found`, `preview.expired`, `history.conflict` |
| `preview.discard` | `id` | `id`, `edits` | `preview.not_found`, `preview.expired` |

- **Edits.** An edit of the open spec whose invocation names the preview applies to the preview's copy, validated as a real edit, and returns its diff against the copy as it was before the edit, with `applied` true. Its handler reports `no_effect`, so it writes no journal line and files no step. With `dry_run` it is only checked against the copy. The preview records the edits that changed its copy, in order, without `dry_run`; consecutive edits with one merge key keep only the newest.
- **Reads.** `spec.get`, `spec.validate` without `text`, `panel.render` and `Session::resolve()` read the preview's copy when they name it, so the agent can check its work and the app can draw the preview in place.
- **The open spec changing underneath.** A preview is always the open spec with its edits applied. When the open spec changed since the copy was computed, the edits are applied again to it before the preview is read, edited or committed; when one no longer applies, or the preview was begun on a spec that is no longer open, the call answers `history.conflict`.
- **Commit.** `preview.commit` is the person's Accept, so only the origins `ui` and `builder` may send it; any other origin gets `preview.not_allowed`, journaled as a failure, and the preview stays. The handler applies the edits again to the open spec, refuses with `history.conflict` if one no longer applies, and otherwise returns them as [follow-ups](#follow-ups) with the preview's id as their group, so they run as one step that one undo reverts. The journal holds the commit's own line from the person and then each edit, under the origin that sent it and the preview's group. A preview with no edits commits nothing and files no step.
- **Discard.** `preview.discard` drops the preview. Nothing was journaled or filed, so nothing remains.
- **Scope.** A preview begun with `panel` or `source` takes new sources and panels, and edits only of its target, of the sources a target panel binds (following `from`) and of the items it added; `spec.set_text` and `layout.set` are refused. Anything else is `preview.out_of_scope`.
- **Bounds.** At most 16 previews are open in a session (`preview.too_many`), each holds at most 64 edits (`preview.too_many_edits`) and 4 MiB of their arguments (`preview.too_large`), and a preview unused for 30 minutes is dropped; a later call names it as `preview.expired`.

### review

| Command | Arguments | Result | Errors |
|---|---|---|---|
| `review.status` | none | `origin`, `spec_edits`, `destructive` | |

`review.status` tells the caller which of its commands wait for the person: `spec_edits` when review of spec edits is on for its origin ([Approval](#approval)), and `destructive` for every origin but `ui` and `builder`. `caliper-agent` reads it so that it does not ask the person again for what Caliper already asks.

### instrument, capture and history

| Command | Arguments | Result | Errors |
|---|---|---|---|
| `instrument.save` | `id` (default: the spec's instrument id), `manifest_id` (default: the saved manifest's id for an overwrite, `local.<id>` otherwise), `overwrite` (default false) | `id`, `manifest`, `spec` (paths relative to the workspace) | `spec.no_open_spec`, `instrument.exists`, `instrument.write_failed`, validator codes |
| `instrument.open` | `id` | `id`, `title`, `warnings` | `instrument.not_found`, `instrument.not_spec`, validator codes |
| `instrument.list` | none | `instruments`: `id`, `title`, `path`, `updated_at` for each | |
| `capture.frame` | none | `png_base64`, `width`, `height` | `capture.unavailable` |
| `history.undo` | `steps` (1 to 100, default 1) | `steps` (one summary for each step undone), `can_undo`, `can_redo` | `history.nothing_to_undo`, `history.undo_failed`, `history.changed`, `command.needs_approval`, `command.not_allowed` |
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

### Follow-ups

A handler may return follow-up invocations with its result (`CommandOutput::follow_ups`); `preview.commit` returns the edits of the preview it commits. The registry checks every follow-up before any runs: they share one non-empty group, and each names a bound command that mutates, is not destructive and is not a history command, with arguments that pass its schema; otherwise the invocation answers `command.handler_failed`. They then run in order, with the ids `<id>.1`, `<id>.2` and so on, without asking the approval policy, because the handler's own invocation was the approval. When one fails, those that ran are undone, newest first, and the invocation fails with that error. When all succeed, the handler's own invocation is journaled, then each follow-up that changed something, and their records form one new step, even when the step on top has the same group, because the history seals its top step first.

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

A command whose handler returns no undo record (`loader.run`, `instrument.save`) records no step. It clears the redo stack, keeps the undo stack and closes merging. Undo then passes it by: after a save, undo reverts the open spec and leaves the saved files as they are, as in a text editor, and after a run, undoing an earlier `loader.update` restores the recipe and leaves the tables as the run made them. The descriptor table says in each such command's summary that it cannot be undone, and its row in the table above says so too.

### When the workspace changed underneath

Changes made outside the command layer, such as a loader file edited in another program, are not in the history. An undo record must check that the state it is about to replace is the state its command left, for example by keeping the SHA-256 of the text it wrote, and must fail with `history.conflict` when it is not, rather than overwrite a change it never saw. When the app reloads a changed file into the open spec, it does so through a command, which becomes a step of its own.

When the `undo` of a record fails, the history first redoes the records of the step that it had already undone, so the state is as it was before the attempt. It then drops that step and every older step from the undo stack, because older steps assume this one was undone first, and keeps the redo stack. The command answers `history.undo_failed`, naming the command whose undo failed and its reason. A failing `redo` is handled the same way: the partly redone step is rolled back, the step and every step above it on the redo stack are dropped, and the command answers `history.redo_failed`. If the rollback fails as well, the state is unknown: the history clears both stacks and the command answers `history.inconsistent`.

`history.undo` with `steps` greater than 1 undoes one step at a time and stops at the first failure; the error's message says how many steps were undone before it.

### Undo, redo and approval

Undo and redo are the person's tools. The registry refuses `history.undo` and `history.redo` from the origins `agent` and `mcp` with `command.not_allowed` before it asks any policy or looks at the history, so an agent can neither take back the person's change nor restore one the person took back; an agent that wants to revert something sends an edit that reverts it, which is reviewed like any edit. `history.list` stays readable for every origin. For the other origins, a program over Caliper API must not be able to do through the history what it could not do directly. Before it undoes or redoes a step, the history asks the approval policy about each destructive command in that step, with the origin of the `history.undo` or `history.redo` invocation: for an undo, the inverse invocations and the commands whose undo is a closure; for a redo, the commands themselves. A deny denies the whole command; an ask parks the history invocation for approval, with the ids of the steps it would touch. When it is approved, the history checks that those steps are still the ones on top, and answers `history.changed` if they are not.

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
- A `do` event of a command that ran without the person's card because an agent's mode or a rule allowed it carries `"allowed_by"`: `accept_edits`, `allow_all`, `session_rule` or `saved_rule`. It is still a step of its own that the person undoes like any other; the app shows the side chat's agent's ones in the chat as APPLIED rows with Undo.
- An `undo` or `redo` event is written for each step that `history.undo` or `history.redo` undid or redid. Its `name` is that command, its `args` are `{}`, its `group` is the step's group, and `seqs` lists the `seq` of every `do` event of the step. A failed undo or redo is written with `ok: false` and the error code.

The journal records every invocation of a mutating command that passed argument validation, and the follow-ups of a handler after its own line: the ones that succeeded, the ones whose handler failed, and the ones that were denied, by the policy or by the person. An invocation whose handler reports no effect is a read and is not recorded. An invocation refused earlier (unknown command, invalid arguments, not bound, wrong thread) ran nothing and is not written, since its arguments may be malformed. An invocation waiting for approval is written once, when it is decided or withdrawn. Read-only commands are written only when the journal is opened with `include_read_only`, as `do` events. Results are never written.

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

On `ask`, the registry parks the invocation under a new approval id and answers `command.needs_approval` with that id in `error.approval`. When the registry has an approval review (`set_approval_review()`), it asks it once for the invocation it parks and keeps what it returns. The app shows the pending approvals (`pending_approvals()` lists them with the descriptor's summary, the arguments and the review; their JSON form is `{"approval", "invocation", "summary", "requested_at", "review"}`) and calls `approve(id)` or `deny(id)`. `approve` runs the parked invocation as if the policy had allowed it and returns its result; `deny` completes it with `command.denied`. Both are journaled then. At most 64 invocations wait at once; more are refused with `command.too_many_pending`. The registry sets no time limit of its own, because the person may be away; a caller that cannot wait for ever gives the queue a limit (below), after which the approval is withdrawn with `command.approval_timeout` and journaled like a denial. Pending approvals also end, with `command.shutdown`, when the queue shuts down.

The builder session installs a policy that adds two rules, decided per invocation because a descriptor's flags are fixed per command:

- `loader.run` of a Python loader, and `loader.create` or `loader.update` of one with `run` set, ask every origin except `ui` and `builder`, each time, because running a Python loader runs the user's code. The handler passes the store's permission to run the script (`RunOptions::allow_python`) only for `ui` and `builder`, or for an invocation that the policy asked about and the person approved, so an app policy that allowed everything would still get `loader.python_needs_approval`. A redo never runs a script.
- `instrument.save` from `agent`, `mcp` or `api` asks only when it would overwrite a saved instrument; writing a new one destroys nothing. The approval is bound to content: the pending approval carries the review of the overwrite, the diff of the saved file against the open spec it would be written with, and the session keeps the document and a digest of both texts.
- **Replacing an open spec with unsaved changes.** `spec.new`, `spec.open` and `instrument.open` from an origin under review ask when the open spec has unsaved changes, with the diff of that spec against the one that would replace it, bound to content as an overwrite is. Over a saved spec they run at once, since its file and the history keep it. A launch of a workspace's spec-based instrument, by `app.launch` or by Caliper API's `POST /v0/instruments/{id}/launch`, opens it with an `instrument.open` that the app sends from the origin of who asked: the `app.launch` invocation's origin, which the session hands to the app (`SessionHost::app_launch(id, origin)`), or the origin of the request's token. It used to be sent from `ui` whatever the caller, so an agent's launch replaced unsaved changes with no card; now it is reviewed here as the agent's own `instrument.open` would be. Only the person's launches from the landing screen (and `CALIPER_AUTOLAUNCH` and `caliper shot --launch`, the person's own settings) send it from `ui`, and the app asks them "Save changes to <name>?" instead of a card.
- **Review of spec edits.** When review is on for an origin (a session setting, `ReviewSettings`, on by default for `agent` and `mcp` and off for `ui`, `builder` and `api`, which the app sets per origin), an edit of the open spec from that origin that would change it asks. The session computes the dry run itself and the pending approval carries its review, `{"diff", "warnings"}`: the unified diff of the canonical TOML before and after, and the validator's warnings. The call is then held or answered `command.needs_approval` exactly as a destructive command's is. A dry run, an edit that names a preview, an edit that would fail validation and one that would change nothing run at once, so an agent's own `dry_run` still works and has no effect.

**When the open spec changed between the parking and the approval**, the approved edit is made again on the open spec as it is then. It runs only when the change it makes now is the change that was shown, compared as the lines it removes and adds without line numbers or context, so a change elsewhere in the spec does not stop it. When the change differs, the edit no longer applies, or another spec was opened, it answers `history.conflict`, is journaled as a failure and files no step.

**A group of edits from one agent turn** is reviewed edit by edit. A held call answers before the agent sends its next edit, so each diff is against the spec that the earlier approvals left, and the pending approvals carry their group, so the pane can stack the cards of one turn. A review of several edits as one stacked diff is a [preview](#preview), which the widget agent uses and the person accepts with one Accept.

**An approval bound to content holds only while that content does.** Before an approved invocation runs, the registry asks the approval check (`set_approval_check()`). The session's check compares the open document and the digest of what the invocation would write or open now with what the card showed. When they differ, because the open spec, the saved file or the document changed while the card waited, the approval is void: a caller that waits for the outcome (a held call) keeps waiting on a new card, which the policy and the review make for the new content, and the `approve` future answers `command.needs_approval` with the new approval id; a caller that does not wait (an `api` client that polls) gets `history.conflict` for the approval it holds, journaled like a denial, and sends the call again. Nothing is written that the person did not see.

There is no command and no endpoint that approves or denies, or that sets an agent's mode or rules ([Agent permissions](#agent-permissions)), and only the `ui` and `builder` origins commit a preview. Only the app's own UI calls `approve` and `deny`, so an agent cannot approve its own request or accept its own preview, and approval ids need not be secret.

## Agent permissions

The person chooses how much the side chat's agent and outside agents may do without asking, as coding agents commonly offer. The command layer applies it, never the agent: the modes and rules are decided in the session's policy (`Session`, `permissions.h`), after the rules above.

### Who they govern

- The side chat's agent: the origin `agent` with the client `chat`. The app issues the chat's agent its own token with that client (`AppApi::issue_agent_file(why, "chat")`, `Server::issue_token("agent", "chat")`), and the API server sets the invocation's `client` from the token, never from the request.
- `caliper-mcp`'s clients: the origin `mcp`, through `api.json`'s token.

Nothing else is governed. The widget agent's sidecar has the client `widget` and keeps its preview with Accept, whatever the chat's mode: that is how the two kinds of `agent` session are told apart, and a test sends the same commands with both clients under Allow all and Read only. The person's own UI (`ui`, `builder`) never asks, the viewer only reads, and the person's `api` token behaves as before.

### Modes

Each governed principal has a mode; both start at Ask.

| Mode | What it does |
|---|---|
| Ask | The session's policy as before: a card for what asks. |
| Accept edits | An edit of the open spec (`effect` `edit`) that would ask runs at once, journaled with `allowed_by` and undone like any step. Everything else that asks still asks: deleting or overwriting saved data, running a loader, replacing the open spec. |
| Read only | Reads and proposals run: commands whose `effect` is `read`, a dry run, an edit into a stacked preview. Every other command is refused with `command.denied` and a plain message that says the agent is in Read only mode. |
| Allow all | What would ask runs without a card, except a data-loss card ([Allow all](#allow-all)). It is saved like the other modes, for the side chat's agent and for `mcp` clients each, applies at launch and stays on until the person sets another mode. |

Ask is the policy as it was, so a mode never makes an agent stricter than before except Read only, and only Read only can refuse what the policy allowed. The side chat's header shows the mode as a word and Shift-Tab cycles Ask, Accept edits and Read only; Allow all has its own item in the word's menu, and a banner with End, which sets Ask, while it is on. The settings (⌘,) set the mode of each principal, Allow all included, as a row of four buttons per principal, which matches the file's one mode per principal.

### Allow All

Allow all is one of the four values of a principal's saved mode, not a mode of a session. The person sets it in the settings or in the side chat's mode menu, Caliper saves it in `agent-permissions.json` and reads it at launch, and it stays on until the person sets another mode. The end of the chat's agent or of the app does not end it. While either principal is in Allow all, every page's status bar says so in plain words in the accent: "allow all for side chat", "allow all for MCP clients", or both. The session also says it on standard error when it opens ("[session] allow all for ..."), for the hidden app and `caliper shot`, which show no status bar. The chromeless instrument page (`CALIPER_CHROMELESS=1`) hides the status bar and with it the indicator, and it shows no other mark. This replaced the earlier Allow all of the chat's session, which was never saved. One saved value per principal is one rule with one indicator, and a session's mode above a saved one needed a second rule for which of the two holds.

Under Allow all, everything that asks runs without a card, except a data-loss card: an approval bound to the content it would lose, which only a card decides. These still ask under Allow all:

- `instrument.save` that overwrites a saved instrument or dashboard.
- `spec.new`, `spec.open` and `instrument.open` over an open spec with unsaved changes, from an origin under review, and the `instrument.open` of a launch by `app.launch` or `POST /v0/instruments/{id}/launch` in that case.

These run without a card under Allow all: an edit that review would show, `data.import`, `app.launch`, `loader.create`, `loader.update` and `loader.delete`, the run of a Python loader (`loader.run`, or `loader.create` and `loader.update` with `run`), `run.start` and `run.fork`, the other `run` commands (`run.set_param`, `run.action`, `run.pause`, `run.stop`, `intervene.set`, `intervene.clear`), `instrument.save` of a new instrument, and the replacement of a saved open spec. A command that deletes or overwrites saved work keeps its card under Allow all when the session binds its approval to that content (`content_of()` and `bind_content()` in `session.cpp`), as an overwrite of an instrument is bound. A destructive command declared later, such as `piece.remove`, keeps its card by the same binding, and `decide_agent_permission()` needs no change for it.

### Rules

A card of a governed agent offers three answers besides Deny: Allow once (the approval as before), Allow always for this session (that command, for that principal, until the chat's agent ends, or for `mcp` until the app quits), and Always allow `<command>` (saved, per agent, until the person revokes it in the settings). A rule names the command that the command layer runs, from the parked invocation, so no argument, title or `_meta` of an agent selects one.

Saved rules and the two saved modes live in `agent-permissions.json` in the app-data folder, keyed by agent: the side chat's agent id (`claude`, `codex`, `custom`, ...) or `mcp`:

```json
{"version": 1, "modes": {"chat": "allow_all", "mcp": "accept_edits"},
 "rules": [{"agent": "claude", "command": "panel.patch"}]}
```

The file is written owner-only (0600) through a temporary file. Reading it ignores and reports a file that is not version 1 or not JSON, an unknown field, a mode that is not `ask`, `accept_edits`, `read_only` or `allow_all`, and a rule for an id that is not an agent id or for a command that is not an agent's (`rule_command_allowed()`: in the table, `agents` true, not a read). Allow all is read only from a file without any such problem: an unknown key or a value of the wrong type leaves a mode of `allow_all` at Ask, and a malformed file is Ask with no rules. On POSIX a file that others may write, or that another user owns, grants nothing. A missing file is Ask with no rules.

**Saving.** Each save takes a lock between apps (the sibling file `agent-permissions.json.lock`), reads the file again and writes it through a temporary file of its own. Two rules decide what is written and what the app then holds. First, the mode or the rule that the person chose in this app is always written, also when the app held it already, and a choice whose save failed is written by the next save. Second, a re-read only lowers: a mode in memory takes the file's mode only where that mode is stricter, a rule that the file no longer holds leaves memory, and nothing that the file adds enters memory, so Allow all turns on only by the person's choice in this app, after its own save. Before a decision that Allow all would make without a card, the app also compares the file's size, modification time and identity with its last read or write. When the file changed and no longer holds Allow all for that principal, the mode drops to the file's at once. A file that cannot be read fails the save, and a file that reads with a problem is replaced with what the app holds. Allow all takes effect only after its save succeeds. Another mode takes effect at once, also when its save fails. Until a later save succeeds, the app then says that the file still holds the old mode:

- The Permissions section shows why the save failed, which mode the file still holds, and that this mode returns at the next launch.
- The side chat's banner stays, with "Still saved", while the file holds the chat's Allow all. Its End saves the chosen mode again, never a looser one.
- The status bar says "allow all in the saved file for side chat" (or for MCP clients). It says the same when a second app turned Allow all on that this app does not take over.

Choosing the mode again saves it.

**Protected files.** No command writes, replaces or removes the settings that set what an agent may do. These files are protected: `agent-permissions.json` (the modes and the saved rules), `side-chat.json` (the side chat's agent and its command line, a program that the app runs), `widget-agent.json` (the widget agent's provider, server and model) and `api.json` (where `caliper-mcp`'s clients find the app). The agent token files `api-agent-<pid>-<n>.json` are not protected: the API server keeps every token in memory, so a changed file gives no token more rights. The check is per write, in the functions that write, replace or remove a file for a command (`touches_protected_file()` in `loaders/paths.h`: the instrument files, the loader files, the derived tensors, the outputs of a Python loader, their removal, the workspace database and the command journal). It compares the resolved path of each write with each protected file: links (also a link whose target does not exist yet) and `..` are followed, the case of letters is ignored, since the file systems of macOS and Windows ignore it, and a file that is the same file as a protected one (a hard link) counts. A folder that holds a protected file is never removed. No command takes a path to write. Every write is a fixed name, or a checked plain name, in a folder of the workspace: `instruments/<id>/<id>.spec.toml` and `.caliper.toml`, `loaders/<id>.loader.toml`, `inputs.toml` (from `ui` and `builder` only), `tensors/<name>.npy`, `loader-outputs/<id>/` with `manifest.json`, `loader_host.py` and the plain file names that a Python loader's manifest names, `runs/<id>/link.txt`, `output.log` and `output.log.1`, `journal/commands.jsonl` with its rotated files, and the database `workspace.duckdb` with its folder `workspace.duckdb.tmp`. Thus a workspace inside the app-data folder, or the app-data folder itself as a workspace, works as any other. The model runner's writes have no check of their own: a managed run's folder `runs/<id>/` must be new, so a run writes only files of its own. The hidden app's input driver (`CALIPER_TEST_SPEC_UI`) plays clicks as the person's own in a hidden run. The guard of `app_paths.h` keeps a test from using the person's data folder by accident, but it is no boundary against a process of the person's account: a run started with another `HOME`, or with another spelling of the folder, passes it and can set the person's modes.

**What this does not protect against.** Any process of the person's account can write the file, and Caliper cannot prevent that. An agent that has its own access to the person's files, such as a coding agent run with file-system tools, can turn Allow all on by writing the file, as it can edit any of the person's files. A process can also start Caliper with `CALIPER_DATA_DIR` set to a folder that holds a file of its own. The limit of the rule is the person's own code: a Python loader or a managed run runs as the person, and it can write this file as it can write any other. Allow all lets such a run start without a card, so under Allow all an agent can start code that turns Allow all on, for itself and for the other principal. A file written that way takes effect at the next launch. A running app does not take it over, because a re-read only lowers (Saving, above). The file stops nothing that the person's own account can do.

### What no mode or rule changes

- Only the person's own UI sets a mode or adds or revokes a rule: there is no command and no endpoint for them, so an agent, an `mcp` client, the viewer and the `api` token cannot, by any command, endpoint, MCP tool, sidecar tool, follow-up, group or preview. No environment variable, command-line flag or spec sets a mode either. The only source besides the UI is the saved file. The app calls `Session::set_agent_mode()`, `allow_always()` and `revoke_agent_rule()` from the chat's header, the cards and the settings only. A test sends `permissions.set` and similar names with every token and gets `command.unknown`, and a request to `/v0/permissions` or `/v0/settings` gets 404.
- The ceilings stand: the registry refuses a viewer's change and `history.undo` or `history.redo` from an agent before any policy runs, in every mode; an agent cannot act as `ui` or choose its origin or client.
- An approval bound to content always asks: the overwrite of a saved instrument and the replacement of an open spec with unsaved changes, whatever the mode or rule, Allow all included, since the card shows what is lost. Running Python code (a Python loader, `run.start`, `run.fork`) is never covered by a rule; only a card, or Allow all, lets it run.
- Read only beats every rule.

### Pending approvals and the end of a session

A mode changed while an approval waits leaves it waiting: a stricter mode does not deny it and a looser one does not approve it; the person decides it on its card. When the chat's agent ends, the chat's session rules go and its mode stays, Allow all included. Its pending approvals stay on their cards until they are decided or their wait runs out, as before.

### What each command does

The `effect` of every command is data in the table (`kEffects` in `command_table.cpp`) and in `command_table.json`; a command missing from it is a fault the table's test reports. When in doubt a command is `change`, which asks.

| Effect | Commands |
|---|---|
| `read` | `app.info`, `app.instruments`, `catalog.*`, `data.query`, `loader.list`, `loader.describe`, `spec.get`, `spec.validate`, `panel.render`, `panel.data`, `instrument.list`, `review.status`, `capture.frame`, `history.list`, `run.state`, `events.wait`, and the proposals `preview.begin`, `preview.diff`, `preview.discard` |
| `edit` | `spec.set_text`, `source.add`, `source.patch`, `source.remove`, `panel.create`, `panel.patch`, `panel.remove`, `layout.place`, `layout.set`, `control.create`, `control.patch`, `control.remove`, `preview.commit` |
| `change` | `app.launch` (it changes the page and may replace the open spec), `data.import`, `loader.create`, `loader.update`, `loader.run`, `loader.delete`, `spec.new`, `spec.open`, `instrument.open` (they replace the open spec), `instrument.save`, `history.undo`, `history.redo` |
| `run` | `run.set_param`, `run.action`, `run.pause`: they act on the running script, change no workspace state and have no undo. The session asks about them for the origins `agent` and `mcp`, and Accept edits does not apply them. The page [Instruments](instruments.md) defines the class. |

## Threads and the queue

- `CommandQueue::submit()` may be called from any thread. It returns a `std::future` of the result, and refuses with `command.queue_full` when 1024 invocations already wait.
- `drain()` runs the waiting invocations in submission order on the calling thread, optionally at most a given number per call. The app calls it on the frame thread between two frames.
- When the policy asks, the future does not complete with `command.needs_approval` by default: it waits for the decision. The queue's `approve(id)` and `deny(id)` may be called from any thread; the decision runs at the next `drain()` and completes both the waiting future and the future that `approve` or `deny` returned. A submit may set `approval_timeout`; when the person has not decided by then, the next `drain()` withdraws the approval and completes the future with `command.approval_timeout`. A caller that must answer at once submits with `wait_for_approval = false`, gets `command.needs_approval` straight away, and later reads the outcome with `approval_outcome(id)`, which the queue keeps for the last 64 decisions.
- `shutdown()` completes every waiting future, and every future waiting for an approval, with `command.shutdown`. A later `submit()` returns a future that is already complete with that error. The destructor shuts down.
- The registry runs one invocation at a time, under one lock that also covers the history, the journal and the approvals. When the app names its frame thread to the registry, a `frame_thread` command invoked directly on another thread is refused with `command.wrong_thread`. The app keeps that one lock and still never holds a frame on a loader: before it drains, it looks at the next item with `peek()`, and runs `data.import`, `data.query`, `loader.run`, a `loader.create` or `loader.update` with `run`, or the approval of one of them, with `drain(1)` on a worker, while its frame thread leaves the queue, the registry and the store alone until the worker is done ([Builder session](builder-session.md#long-commands-and-the-registrys-lock)).

## Caliper API

Caliper API serves these endpoints, under the same loopback, `Host`, `Origin` and session-token checks as API v0, while a `CommandGateway` is set on its server (`src/host/api/api_server.h`, `src/host/api/command_gateway.h`):

- **`GET /v0/commands`** returns the descriptor table, the output of `command_table_to_json()`: `{"version": 0, "commands": [{"name", "summary", "args", "result", "examples", "mutates", "destructive", "frame_thread"}]}`, in the table's order.
- **`POST /v0/commands/{name}`** takes the `args` object as a JSON body of at most 1 MiB with `Content-Type: application/json`, and returns the result envelope with HTTP 200 whether `ok` is true or false. The server assigns the invocation id. The origin is `api`, or `mcp` or `agent` when the request carries `X-Caliper-Origin: mcp` or `X-Caliper-Origin: agent`; any other value is refused with 400, so a request never claims `ui` or `builder`. An optional `X-Caliper-Group` header gives the group, and an optional `X-Caliper-Preview` header the stacked preview that the call's edits apply to; the registry checks both. A PNG result is base64 in `result.png_base64`, with `result.width` and `result.height`.
- **Approval over the API** follows contracts section 15. A call with the origin `mcp` or `agent` that needs approval is held while the person decides in the app, for up to 120 seconds: the gateway submits it with `approval_timeout` set to 120 seconds, and the server holds the request until the future completes. Approved, the call answers with the command's result; denied, with `command.denied`; after the time limit, with `command.approval_timeout`, and the approval is withdrawn. `caliper-mcp` and `caliper-agent` allow at least 130 seconds for a call. A call with the origin `api` submits with `wait_for_approval = false` and gets `command.needs_approval` with the approval id at once.
- **`GET /v0/approvals/{id}`** returns `{"approval": "<id>", "state": "pending" | "decided" | "unknown", "result": <envelope or null>}`, with the pending approval's `review` while it waits, from `approval_outcome(id)` and the registry's pending list, so that an `api` client can learn the outcome of a request that needed approval. This endpoint is an elaboration of this page, not of the contracts.

Transport faults keep their HTTP status: 401 and 403 from the checks, 411 without `Content-Length`, 413 for a body over 1 MiB, 415 for another content type, 400 for a body that is not JSON, a repeated header or another origin, 501 for `Transfer-Encoding`, 408 for a body that does not arrive within 10 seconds, and 503 and 504 as in API v0. A body that is JSON but not an object, or nests deeper than 64 levels, is answered 200 with `command.invalid_args`. A command name that does not exist is answered 200 with `command.unknown`, not 404, so that a 404 means only that the app has no command layer, which is how `caliper-mcp` tells an older app apart. The server never logs a request's token, and it builds the invocation from the path, the body and the three headers only, so no other header value reaches the journal.

How the server holds a request, as built. The server has one thread, the hub's, which also carries the stream, and it must never wait. A command request waits for the frame thread like every call of API v0: `Server::take()`, which the app calls once per frame, hands it to the gateway there, and the gateway submits it to the session's queue, which the frame thread drains. From then on the request is held as an entry of the server's table, with no thread of its own; `take()` and the hub's thread ask the gateway's `PendingCommand` whether the future is ready, never waiting, and answer it when it is. The 5-second bound of API v0 is therefore the wait before the frame thread takes the command; once taken, the request waits for its result, a call held for approval waits as described above, and a held request with no result within 600 seconds gets 504. At most 32 command requests are held or wait for the frame thread at once; one more gets 503 with `Retry-After: 1`, which also keeps them below the registry's 64 pending approvals. `GET /v0/approvals/{id}` is answered on the frame thread too, because the registry's lock may be held by a long command, and `GET /v0/commands` from the table the gateway made when it was created, so it never waits.

### Which origin a call has

The registry takes the origin from its caller and nothing else. Over HTTP the origin belongs to the token a request carries, not to the request: the stand-in's server takes it from its options (`caliper-builder-host --origin mcp|agent|api`, `mcp` by default), refuses a server for `ui`, `builder` or `viewer`, and refuses with 400 an `X-Caliper-Origin` header that names another origin, so a request cannot raise its origin by leaving the header out or by naming another. The app's API server fixes the origin by the token in the same way, with a token of its own for the side chat's agent and for the sidecar, so no agent holds a token whose origin is `api`. Review of spec edits therefore stays off by default for `api`, the person's own scripts, and on for `agent` and `mcp`.

## caliper-mcp and the Python sidecar

The descriptor table is the one source of tool definitions. `caliper-mcp` links the C++ table, and the widget agent sidecar `caliper-agent` loads its JSON form, `src/builder/commands/command_table.json`, which is the output of `command_table_to_json()` printed with two-space indents, sorted keys and a final newline. A test fails when the file and the C++ table differ. After a change to the table, regenerate the file with the tool target of the commands module:

```sh
cmake --build <build> --target caliper_builder_command_table
<build>/commands/caliper_builder_command_table --write src/builder/commands/command_table.json
```

`caliper-mcp` leaves out every command whose `agents` is false, `history.undo` and `history.redo`, from its tools, and a forced call of either is refused as an unknown tool; the stand-in's `GET /v0/commands` leaves them out for an agent's or mcp's token, as the app's should. `caliper-agent` offers its model neither the `preview` commands nor `review.status`, which belong to the app and to the sidecar itself; it does not read `agents` yet, so its model may still call the two, which the command layer refuses. Both clients name a tool after its command, with `_` for `.`: `panel_create` for `panel.create`, `spec_set_text` for `spec.set_text`. Since a group has no underscore, the first underscore of a tool name marks the dot, and the mapping is reversible. A tool's description is the command's summary; its input schema is the command's `args`. `caliper-mcp` may mark a tool read-only when `mutates` is false and destructive when `destructive` is true. For strict tools, the sidecar removes the keywords that the Claude API does not take in a strict schema (`minimum`, `maximum`, `minLength`, `maxLength`, `minItems`, `maxItems`, `pattern`, `default`) from what it sends; nothing is lost, because the command layer checks every argument against the full schema.

## Changes to the contracts

These are additions to, or fixes of, sprint 3 contracts section 7, each the smallest that worked:

- **`merge_key` on the invocation.** Section 7 names merge keys but no envelope carries one. The invocation gains an optional `merge_key`, because the source of a gesture (a slider) knows when commands belong together and a handler cannot tell a drag from two deliberate edits.
- **`approval` on the error.** The approval id of `command.needs_approval` needs a field; it is `error.approval`.
- **`seqs` on `undo` and `redo` events, and `error` on failed events.** The journal line of section 7 cannot be replayed across a merged step, because merge keys are not journaled. Naming the `seq` of each `do` event of the step makes replay exact without journaling the history's internals.
- **`examples` on the descriptor**, kept outside the schema, so that strict schemas stay plain and clients can show sample calls.
- **`minItems` and `maxItems`** join the schema subset, so that the schemas state the bounds of their lists.
- **The flat layout** of `layout.set`, and the one open result object, `spec.get`'s `json`, follow from closed, non-recursive tool schemas.
- **`preview` on the invocation**, with the header `X-Caliper-Preview`, so that the sidecar can send a widget session's edits into its preview without the model setting anything, and `review` on the pending approval.

## Decisions

- **Destructive means saved data.** An edit in memory that undo reverses exactly is not destructive, so the builder and agents edit the open spec without a prompt, and approval is asked when work leaves memory: overwriting a loader, running one, deleting one, saving an instrument.
- **Groups join only the top step.** It keeps the history a stack, so every undo reverses the newest changes and the state stays consistent.
- **A failed undo drops the older steps.** Older steps assume the failed one was undone first; keeping them would let a later undo apply to a state it was not recorded against.
- **Commands that cannot be undone are passed by, not barriers.** This is how editors treat a save, and it keeps undo useful after a run or a save.
- **Replay applies the steps in effect instead of running undo records.** It needs nothing but the journal and the apply function, and it is exact because the history is a stack.
- **A command meant to merge checks only what a whole run leaves unchanged.** The history keeps the first undo of a merged run, so a check against the value the first command set would report a conflict after any drag.
- **Journaled invocations are those that passed validation.** A refused invocation ran nothing, and its arguments may be malformed or hostile.
- **One lock for the registry.** It makes ordering and the history simple. The app runs the long commands on a worker and does not drain while one runs, so a frame never waits on the lock and the order of the queue holds.
- **No approval through commands or endpoints.** Only the app's UI decides, so an agent cannot approve itself.
- **Undo and redo are the person's.** An agent reverts with an edit, which is reviewed; letting an agent move the history would let it take back the person's work, or restore what the person took back, without an edit to show. A flag on the descriptor (`agents`), not a list of names, tells the registry and the clients.
- **Approvals of overwrites and replacements are bound to the content shown,** and a held call gets a new card when that content changes, so the person never approves one file and gets another.
- **The origin belongs to the token.** A header that a caller sets cannot be trusted to lower its own rights.
- **Modes and rules in the command layer, keyed by the token.** The side chat's agent and `mcp` clients get a mode each; the token's client, which the app sets, tells the chat's agent from the widget agent, so nothing a request carries selects a mode or a rule. Ask is the policy as it was; Accept edits and the rules only turn an ask into an allow; Read only only denies.
- **No rule for a Python run, and no mode or rule over content.** A rule allows a command by name, and `loader.run` of a CSV reader and of a Python script share it; running the person's code needs a card or Allow all. An approval bound to content always asks, since the card shows what is lost.
- **Protected files, checked per write.** A check of the workspace's folders refused every agent change in the default workspace, which lies in the app-data folder. Each write compares its own resolved path with the protected files instead, so only a write that would change one of them is refused.
- **Allow all is a saved mode of each principal.** It reverses the earlier rule (the chat's only, for its session, never saved). One saved mode per principal is one rule and one status bar indicator. Data-loss cards still ask, and only the person's UI sets it.
- **Review in the command layer, not in each agent.** An agent that asks before an edit is polite; one that does not, such as an outside agent over MCP, is still reviewed, because the session computes the diff itself and the policy parks the edit.
- **An approved edit runs only as it was shown.** Recomputing it and comparing the lines it removes and adds lets an unrelated change pass and refuses one that would make the approved change mean something else.
- **Edits are reviewed one at a time; a stacked review is a preview.** A held call makes an agent's edits sequential, so each card shows a change against the spec the person has seen.
- **A preview commit is a command that only `ui` and `builder` may send**, so the queue, the journal and the history treat it like every other change, and no agent can accept its own proposal.
- **Follow-ups for a commit, not a handler that calls commands.** Handlers never invoke commands, and a commit journaled as one opaque line could not be replayed; follow-ups give each edit its own journal line and still one undo step.
- **Paths outside the workspace only from `ui` and `builder`.** Loaders would otherwise let an agent read any file the app can read.
- **The table's summaries are written for an agent that has never seen Caliper**: each says what the command does, what its ids are and which command gives them.
- **Caliper API reaches the command layer through an interface of JSON text.** The server builds without the builder, and the app wires the session in with one call. The interface's submit and approval lookup run on the frame thread inside `take()`, so the bound of API v0 still means "the frame thread took the request".
- **A held request costs a table entry, not a thread.** The futures are polled without waiting from `take()` and the hub's tick, so a frame or at most 50 ms passes between a result and its answer.
- **A command's body is read only after its head passed the Host, Origin and token checks with a length within 1 MiB.** Nothing is buffered for a request without the token; a refusal of the content type or the claimed origin comes after the body is read, so that the client receives it instead of a reset. The WebSocket layer reads a body only when its owner allows it for that head (`Options::body_limit` in `src/host/net/ws.h`), so every other request still carries none.
- **`Transfer-Encoding` on the command endpoint is 501**, as in the stand-in, while the endpoints of API v0 keep 413 for any body.

## Open items

From agent permissions (sprint 3.1):

- **Deny rules and finer scopes.** Only allow rules exist, for the side chat's agent and for all `mcp` clients together; the person asked for nothing finer yet.
- **Data-loss cards for loaders.** Under Allow all, `loader.update` and `loader.delete` run without a card, because their approval is not bound to content. Binding them, with a card that shows the loader's text before and after, would keep their card as an overwritten instrument's is.
- **Reads of the settings files.** The protected files are protected from writes only. A workspace that is or contains the app-data folder, such as the home folder, lets an agent's loader read `api.json` and the agent token files, and so use the token of the other principal and its mode.
- **An `mcp` session.** A rule "for this session" of an `mcp` client lasts until the app quits, because `api.json`'s token is shared by every outside agent and has no session of its own.

These were found in the review of sprint 3's first pull request and left for a later round:

- **An expectation on `preview.commit`.** The commit applies whatever the preview holds when the person clicks Accept. An optional expectation (the number of edits and a digest of the preview's canonical text, which `preview.diff` would report) would let the app refuse with `history.conflict` a preview that changed after its card was drawn.
- **Approvals from `api` never expire.** An `api` caller does not wait, so its approvals stay until the person decides, and many of them could fill the 64 pending slots so that agents get `command.too_many_pending`. A long expiry for approvals nobody waits for, and a share of the slots for each origin, would bound it.
- **A command can join another's undo step.** A group joins the top step whatever its origin and whether it is still open, so an agent that sends a preview's id as its group after an Accept appends its command to the person's step. A group should join only the open top step of the same origin, and the step of a commit should close after its follow-ups.
- **A launch that is held as one command.** A launch of a workspace's spec-based instrument by an origin under review, over an open spec with unsaved changes, answers at once with `status` `PENDING_APPROVAL` and a `note` (the app reads the open spec and the origin's review when it answers, and says `PENDING` when the session's worker runs a command and it cannot read them), while its `instrument.open` waits on a card of its own; the answer carries no approval id, since the handler cannot know it, and the client checks `spec.get`. Holding the launch itself, as every other held command is held, would need five things: a review rule for `app.launch` of a workspace spec-based instrument from an origin under review over unsaved changes, with `content_of` and `approval_holds` taught `app.launch` and bound to the diff of the open spec against that instrument's spec (the session learning the manifest id's workspace id from a `SessionHost` hook); once approved, `app.launch` returning its `instrument.open` as a follow-up, so that it runs without a second card, the journal showing the two as one step; `POST /v0/instruments/{id}/launch` routed through the command gateway as `app.launch` for the `mcp` and `agent` tokens, so that it holds and polls the same way; `caliper-mcp`'s `app_launch` then getting held, denied and timed-out answers like any held tool call; and the table's description of `app.launch`, this page and the API reference. It was left because it changes the review rules and adds a follow-up that skips the policy.
- **Two launches in one frame share one pending open.** The app keeps one pending `instrument.open` for a launch of a workspace's spec-based instrument (`pending_spec_open_` and its origin in `src/main.cpp`), so a launch from an agent and one from the person in the same frame, before the first is sent, leave only the later, with its own origin. A queue of pending opens, each with its origin, would keep both.
- **A timing test of the stand-in's server.** The connection-bound test of `test_command_server.cpp` still waits 300 ms for the first connection to take the slot. The approval test of that file now decides its held calls when they are parked, rather than inside a 400 ms window.

## Open questions

- Whether a later phase splits the registry's lock, so that short commands can run on the frame thread while a loader runs on the worker; the app holds them in the queue until then.
- Whether the history should be saved with the workspace, so that undo survives a restart; in this phase it lives in memory and the journal is the lasting record.
