# Builder session

The builder session is the headless object that binds the four builder libraries for one workspace: the [spec format](spec-format.md), the [catalog](catalog.md), the [command layer](command-layer.md) and the [loaders](loaders.md). It owns everything a person or an agent changes while building a visual, and it answers every command that needs no window. The app will own one session per open workspace once the libraries are wired in; until then, a small stand-in tool serves the session's commands over HTTP so that `caliper-mcp` and `caliper-agent` can be tested end to end. This page describes the session, what it binds, what it leaves to the app, how it resolves the data of a panel, and the stand-in.

## What is built and what is planned

Built on this branch, in `src/builder/session/` with headers in `src/builder/include/caliper_builder/session/`:

- the library `caliper_builder_session`: `Session`, the `SessionHost` interface, the spec edits as pure functions with the conversions between the spec and the command table's forms, a line diff, the catalog provider for loader outputs, and the resolved data of a panel;
- the library `caliper_builder_session_host`: a strict HTTP/1.1 server for the two command endpoints of Caliper API, with the API file and token helpers;
- the tool `caliper-builder-host`, the development stand-in for the app's command endpoints;
- the command table brought in line with the spec format, with a test that keeps it so;
- the tests in `src/builder/session/tests/`, including the two acceptance tests of this phase.

Planned, at the wiring after sprint 2 merges:

- the app owns a `Session`, calls `drain()` on its frame thread, implements `SessionHost`, and adds the live, feed and metric catalog providers;
- the app's own API server serves `GET /v0/commands` and `POST /v0/commands/{name}` and replaces `caliper-builder-host`;
- the spec runtime draws a panel from the data `Session::resolve()` gives.

## The session

`Session::open(workspace)` opens a workspace folder, creating it and its layout when missing (sprint 3 contracts, section 2). It holds:

| Part | What it is |
|---|---|
| `LoaderStore` | The loaders, their runs, the grants and `workspace.duckdb`. |
| `Catalog` | One provider, `loaders` at priority 10: `LoaderCatalogProvider`, a thin adapter over the store's sources. |
| The open spec | At most one spec, the one the builder shows, with the file it came from and the text it had when it was last opened or saved. |
| `CommandJournal` | `journal/commands.jsonl`. |
| `CommandRegistry` | The whole descriptor table, the session's handlers and approval policy, and the undo history. |
| `CommandQueue` | Brings invocations from any thread to the thread that drains it. |

The app reaches each part through an accessor (`store()`, `catalog()`, `registry()`, `queue()`), runs a command with `invoke()` on the calling thread or through the queue, reads the open spec with `open_spec()` (its value, its path, whether it is dirty and its canonical text) and resolves a panel's data with `resolve()`. `close()` shuts the queue down, which answers every waiting call with `command.shutdown`, and closes the journal.

### The commands it binds

| Group | Commands | Notes |
|---|---|---|
| catalog | `catalog.list`, `catalog.describe`, `catalog.sample` | Through `Catalog`, which checks every entry and bounds every sample. |
| data | `data.query`, `data.import` | `data.query` runs the store's read-only query and cuts its rows at 8 MiB of JSON. |
| loader | `loader.list`, `loader.describe`, `loader.create`, `loader.update`, `loader.run`, `loader.delete` | |
| spec | `spec.new`, `spec.open`, `spec.get`, `spec.set_text`, `spec.validate` | |
| source, panel, layout | `source.add`, `source.patch`, `source.remove`, `panel.create`, `panel.patch`, `panel.remove`, `layout.place`, `layout.set` | |
| applet | `applet.save`, `applet.open`, `applet.list` | |
| history | `history.undo`, `history.redo`, `history.list` | Built into the registry. |

### What SessionHost leaves to the app

Five commands need the running app: `app.info`, `app.applets`, `app.launch`, `panel.render` and `capture.frame`. They stay unbound, and answer `command.not_bound`, until the app gives the session an implementation of `SessionHost` with `set_host()`. The interface has one function per command, each returning the command's result or its error; `set_host(nullptr)` unbinds them again. For `panel.render` the session does the part that needs no window: it finds the panel in the open spec (`panel.not_found`) and resolves its data, and the host only draws that data off screen and returns the PNG. A test binds a fake host and checks each command and that `panel.render` hands the host the resolved data.

## Editing the open spec

Every command that edits the open spec goes through one function, `apply_spec_edit()`, which applies the command to a copy of the spec and validates the result with a `SourceResolver` that asks the catalog. The resolver gives the spec validator what the catalog knows of each catalog id: its kind, its columns with their DuckDB types, its dtype and shape and its wire flag. A catalog id that nothing knows yet is the warning `spec.unresolved_source`, so a spec can name a source before its loader runs.

- An edit whose own rule fails is refused with the command's code: `source.exists`, `source.not_found`, `source.in_use` (with `where` naming the first panel binding or the first `from` that uses it), `panel.exists`, `panel.not_found`, `layout.same_panel`, `layout.invalid`.
- An edit that leaves the spec with any error is refused with the validator's own code and the document path in `where`, so the open spec is valid after every command and a preview can always draw it.
- The result carries the validator's warnings, such as `spec.not_portable`, and the change as a unified diff of the spec's canonical TOML (`diff`, empty when the text does not change), with `applied`.
- Each edit takes `dry_run`. A dry run checks the change and returns its warnings and its diff without applying it, so that the agent pane can show the change for approval before it happens, as contracts section 12 asks of the BYOK sidecar.

`panel.create` without `place` puts the new panel to the right of the whole layout; the first panel of a spec becomes the whole layout. `place` and `layout.place` wrap the target's leaf in a split, or add a tab next to a target that is already a tab. `panel.remove` and `layout.place` take a panel's leaf out and collapse a split that is left with one child.

### Undo

Every edit returns a closure pair. The undo restores the spec as it was before the command, after checking that the open spec is still the same document and still holds what the command left; otherwise it fails with `history.conflict`. The redo makes the same checks the other way. A command sent with a merge key, such as a drag of a slider, checks only that the document is the same, because the history keeps the first undo of a merged run and the newest redo. `spec.new`, `spec.open` and `applet.open` replace the whole document; their undo puts back the document that was open before, unsaved changes included, and their redo puts back the one it replaced. Documents carry an id, so an undo can tell the open spec from another spec opened since.

### Diffs

`unified_line_diff()` writes the diff in the format of `diff -u`, with three lines of context, hunks merged when they are close, and the marker for a text without a final newline. It sets aside the lines both texts share at their start and end, then compares the rest with a longest common subsequence when the product of the two line counts is at most 2^22, and shows the middle as removed and added as a whole beyond that. A spec is at most 512 KiB, so the work is bounded either way.

## Loaders, paths and approval

The loader commands call the store and record what each change needs for its undo:

| Command | Undo | Redo |
|---|---|---|
| `data.import`, `loader.create` | Deletes the loaders the command created, if their recipes are unchanged. | Saves them again and runs those that ran, except a Python loader. |
| `loader.update` | Saves the previous recipe back, if the current one is the command's. | Saves the new recipe again. |
| `loader.delete` | Saves the recipe back; the outputs return when it runs again. | Deletes it again. |
| `loader.run` | None: a run cannot be undone. | |

`loader.create` with `run` keeps nothing when the first run fails, and `loader.update` with `run` puts the previous recipe back when the run fails, so a failed command changes nothing. `loader.delete` with `keep_outputs` answers `loader.unsupported` in this phase, because the store deletes a loader's outputs with it.

**Paths outside the workspace.** `data.import`, `loader.create` and `loader.update` accept a path outside the workspace from the `ui` and `builder` origins, which stand for a file the person dropped or named, and the session grants it through the store (a glob grants its fixed folder). From `agent`, `mcp` and `api` such a path is refused with `loader.path_refused`, unless an earlier grant of the person already covers it. An agent can therefore build loaders over what the person has shared, and cannot widen what loaders may read.

**The approval policy.** The session's policy is the default policy of the command layer, with two rules that look at the invocation itself, because a descriptor's `destructive` flag is fixed per command:

- Running a Python loader runs the user's code. `loader.run` of a Python loader, and `loader.create` or `loader.update` of one with `run` set, ask for every origin except `ui` and `builder`, each time. The policy records each invocation it asked about, and the handler passes `RunOptions::allow_python` to the store only for `ui` and `builder`, or for an invocation it finds in that record, which can reach its handler only after the person approved it. An app that replaced the policy with one that allows everything still gets `loader.python_needs_approval` from the store, so the rule fails closed. A redo never runs a script: it puts the recipe back, and a run needs its own approval. `data.import` creates a Python loader but never runs it.
- `applet.save` from `agent`, `mcp` or `api` asks only when it would overwrite a saved applet. Writing a new applet destroys nothing, so an agent can save its work without a prompt, and overwriting asks.

## Applets

`applet.save` writes `applets/<id>/<id>.spec.toml` and then the manifest `applets/<id>/<id>.caliper.toml`, each through a temporary file in the same folder and a rename, so a reader sees the old file or the new one and never a mix. The manifest follows the spec format page: `id = "local.<id>"` with `_` mapped to `-`, `name` from the title, `version = "0.1.0"` for a new applet (an overwrite keeps the old version), `summary` from a one-line description of at most 200 characters, `kind = "spec"` and `spec`. An applet that is already saved needs `overwrite`, and an id whose manifest id another applet of the workspace has is refused with `applet.exists`. Saving under the spec's own id makes the files the open spec's home and clears its dirty flag; saving under another id writes a copy and leaves the open spec as it was.

`applet.open` reads the manifest (at most 64 KiB), requires `kind = "spec"` and a plain file name in `spec`, and opens the spec after parsing and validating it. `spec.open` takes a path relative to the workspace and refuses an absolute path, a `..` segment and a symbolic link that leads out of the workspace with `spec.path_refused`. `applet.list` lists the folders of `applets/` whose manifests are spec applets, at most 4096 folders.

## Resolving a panel's data

`Session::resolve(panel)` returns, for each bound slot of a panel of the open spec, the data the spec runtime will draw. It is the data path of the runtime and of `panel.render`, tested here without drawing.

1. A binding names a spec source. The chain of `from` is followed to its root, a `catalog` or `sql` source, at most 64 hops, so a cycle stops with `resolve.cycle`.
2. A tensor root, a `tensor:` catalog source, is mapped from its file through the store. The transforms of the root, then of each derived source down the chain, are applied in order with the loaders library's `tensor_ops`. The size of each result is computed with `transform_result_shape()` before it is made, and a result past 256 MiB is refused with `resolve.too_large`.
3. A tabular root, a `table:` catalog source or a `sql` source, is read once per call through the store's bounded read-only query. A table selects only the columns the panel binds, or all of them when a binding names the whole source; a `sql` source runs as written, so its `ORDER BY` holds. The rows are cut at 100,000 and at 2^22 cells, and `truncated` says so.
4. A column binding gives one column as a list of cells. A whole tabular binding gives a table; for `table.rows`, its columns are those of `options.columns`, in that order. A tensor binding gives the tensor.

Sources that the app serves, `live:`, `feed:` and `metric:`, answer `resolve.unavailable` until the wiring gives the session their data. The bounds are `ResolveLimits`, which an app may change.

## The catalog provider

`LoaderCatalogProvider` turns each `LoadedSource` of the store into a `CatalogEntry`. `origin` holds the loader, the file and, for a tensor, its name in the file; an origin value is cut to 1024 bytes and the warnings to 16 of 1024 bytes, so an entry from a long path passes the catalog's strict check instead of being dropped. A table's sample is its first rows through the store's query, with the non-finite values of floating-point columns written `nan`, `inf` and `-inf` as the catalog spells them; a tensor's sample comes from `sample_tensor()` over its mapped bytes. The store's sources change only through loader commands, so the session increments the provider's revision after each of them.

## The command table and the spec format

The `source.*`, `panel.*` and `layout.*` schemas of the command table now accept exactly what the spec format accepts wherever the schema subset can say it: every slot of every primitive; the binding grammar with its 320-character bound; every option, with `z_label`, the ranges `x_range`, `y_range` and `range` as pairs of numbers, the log flags, the four colormaps as an enum, `bins` from 1 to 65,536, `columns` of 1 to 256 names and `text` of at most 16,384 characters; at most 16 transforms with axes from −8 to 7, safe integers for `index`, `start` and `stop`, and `step` of at least 1; a catalog id of at most 256 characters; and the layout's titles. `panel.create` no longer requires `bind`, since a text panel may have only `options.text`.

The test `test_table_conformance.cpp` keeps them in line: every source, panel and layout of every valid spec fixture, turned into the arguments of `source.add`, `panel.create`, `panel.patch`, `source.patch` and `layout.set`, passes the table's schemas and decodes back to the same item; the result forms pass the result schemas; and values that the spec refuses and the subset can express (a fifth colormap, 65,537 bins, a range of three numbers, an axis of 8, 17 transforms) are refused by the table too. What the subset cannot express, such as `index` together with a range in a slice or the slots a primitive takes, the spec validator refuses after the arguments pass.

## The stand-in host

`caliper-builder-host` is a development tool, not part of the product. It stands in for the command endpoints of the app's Caliper API until the wiring, and the app's own API server replaces it then.

```sh
caliper-builder-host --workspace <dir> --api-dir <dir> [--port <n>] [--approve allow|deny|ask]
```

It opens the workspace as a `Session`, listens on 127.0.0.1 on a free port (or `--port`), and writes `<api-dir>/api.json` with the shape of the app's API file: `version` 1, `pid`, `port`, `token` and `url`, `http://127.0.0.1:<port>/v0`. The file has mode 0600 and is removed on exit; it has no `stream_url`, because the stand-in serves no stream. The main thread drains the command queue, as the app's frame thread will. SIGTERM or SIGINT stops it: waiting calls are answered with `command.shutdown`, the API file goes, and it exits with 0. It never prints the session token.

It serves `GET /v0/commands` and `POST /v0/commands/{name}` exactly as contracts sections 8 and 15 and the command layer page describe, and nothing else:

- the checks of API v0, in order: `Host` must be `127.0.0.1:<port>` or `localhost:<port>` (403); an `Origin` must be `null`, an http or https origin on 127.0.0.1, `localhost` or `[::1]`, or `vscode-webview://` (403); the path and method (404, 405 with `Allow`); the token in `Authorization: Bearer`, or for a GET in the query parameter `token`, compared in constant time (401 with `WWW-Authenticate: Bearer`);
- a strict, minimal HTTP/1.1: one request per connection; a head of at most 16 KiB (431) read within 5 seconds (408); CRLF line ends, no folded headers, no control characters and no repeated header that the checks read (400); no `Transfer-Encoding` (501); a POST with `Content-Length` (411) of at most 1 MiB (413) and the type `application/json` (415), read within 10 seconds (408), that is JSON (400); a GET with no body (413);
- a body that is JSON but not an object, or nests deeper than 64 levels, is answered 200 with `command.invalid_args`, and a name that is not a command with `command.unknown`, so that a 404 means only that there is no command layer;
- the origin is `api`, or `mcp` or `agent` from `X-Caliper-Origin`; any other value is refused with 400, so a request never claims `ui` or `builder`; `X-Caliper-Group` gives the group;
- a call from `mcp` or `agent` that needs approval is held for up to 120 seconds; one from `api` gets `command.needs_approval` with its approval id at once;
- at most 32 connections are served at once, and one more gets 503 with `Retry-After: 1`; every response carries `Connection: close`, `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.

`--approve` decides what a pending approval gets in this tool: `allow` approves it, `deny` denies it, and `ask`, the default, prints one JSON line per pending approval on standard output, `{"approval", "name", "origin", "args"}`, and reads `allow <id>` or `deny <id>` lines on standard input. The HTTP acceptance test drives it that way.

### How the wiring replaces it

After sprint 2 merges, the app opens a `Session` for its workspace, names its frame thread to the registry, calls `queue().drain()` between frames, implements `SessionHost` on its own window, and gives the session the live, feed and metric catalog providers. Its API server, which already holds the loopback, `Host`, `Origin` and token checks of API v0, gains the two command routes; each submits to the session's queue as `CommandServer::handle()` does here, with the same origins, approval waits and envelope, and the approvals come from the app's own approval UI rather than from standard input. `caliper-builder-host` and `caliper_builder_session_host` are then removed, and the tests of `caliper-mcp` and `caliper-agent` move to `caliper --hidden --api`.

## The acceptance tests

Both run headless in `test_acceptance.cpp`.

1. **In process.** A workspace is created; a Parquet file written through DuckDB and a safetensors file are put in a folder outside it; `data.import` imports both from the `ui` origin and `loader.run` runs their loaders; the three-panel dashboard of `dashboard.spec.toml` is built command by command (`spec.new`, `source.add`, `panel.create`, `layout.set`), and the open spec equals the fixture; two steps are undone and redone; every panel's data is resolved and checked against the files (the series' columns, the heatmap's mean over heads after its minmax, the table's chosen columns); `applet.save` writes the applet; the session closes; a new session over the workspace finds it with `applet.list`, opens it with `applet.open`, and gets the same spec and the same resolved data; and a replay of the journal gives the same spec.
2. **Over HTTP.** `caliper-builder-host` runs as a child process with `--approve ask`. From the `mcp` origin, `data.import` takes the files inside the workspace and refuses one outside it; each `loader.run` waits until the test answers the host's approval line with `allow`, and a `loader.delete` is denied; the dashboard is built with a dry run before each panel, undone and redone in part; `applet.save` writes a new applet without approval and waits for approval to overwrite it; SIGTERM stops the host with exit code 0 and removes its API file; and the workspace it leaves opens in process to the same spec and data, and its journal replays to the same spec.

## Decisions

- **`dry_run` and `diff` on every spec edit.** The agent pane must show a change before it applies (contracts section 12), and only the session can compute the canonical text after an edit. A dry run is a mutating command that records no undo step, so, like any such command, it clears the redo stack; the journal keeps it, and replay passes it by.
- **Extension tables are not carried by the edit commands.** `x` is an open, recursive object, which a closed, strict tool schema cannot describe. `panel.patch` and `source.patch` keep an item's `x`, `spec.set_text` carries `x` anywhere, and `layout.set` replaces the layout without it.
- **`panel.create` takes no required `bind`.** A text panel may have only `options.text`.
- **Outside paths from agents are accepted when the person has already granted them.** It keeps the rule's purpose, that an agent cannot widen what loaders read, and lets an agent build a second loader over a file the person dropped.
- **Python runs are approved per invocation, and the handler checks that the approval happened.** The policy alone cannot tell the handler, and an app's own policy could allow everything; recording the asked invocations makes the permission fail closed.
- **`applet.save` asks an agent only before an overwrite.** Saving a new applet loses nothing.
- **Saving under another id writes a copy.** Changing the open spec's applet id as part of a save that cannot be undone would leave the undo history describing another document.
- **Resolved tabular data selects only the bound columns of a table but runs a `sql` source as written.** Wrapping a query in a subquery would not keep its order in DuckDB.
- **The stand-in accepts `X-Caliper-Origin: agent`.** The BYOK sidecar calls Caliper API on behalf of the agent pane, and `agent` has the same approval rules as `mcp`.
- **`loader.describe` reports `last_run.elapsed_ms` as null when it is not known.** The store does not record a run's duration, so the session knows it only for the runs it made since it opened.

## Changes to the contracts and to the earlier pages

- Contracts section 8 names `api` and `mcp` as the origins of an HTTP call; the stand-in also accepts `agent`, for the sidecar. The app's API server should do the same.
- The command layer page listed `loader.unknown_reader` and `loader.read_failed` among `data.import`'s refusals; the store's own codes (`loader.no_reader`, `loader.reader_mismatch`, `loader.pickle_refused`, `loader.io`) are what the refusals carry, plus `loader.path_refused` for an agent's outside path.
- The results of the spec edits gained `diff` and `applied`, their arguments `dry_run`, and `loader.describe`'s `last_run.elapsed_ms` became nullable. The command layer page records these.

## Limits

- The session reads the data of loader outputs only; live views, feed channels and metrics wait for the wiring.
- `loader.delete` with `keep_outputs` is not available, because the store deletes a loader's outputs with it. The loaders library would need `LoaderStore::remove(id, keep_outputs)`.
- Undo does not reach files outside the session: a saved applet stays saved, and a loader run's tables stay as the run made them.
- Replay rebuilds the open spec, not the loaders; `spec.open` and `applet.open` read their files as they are when the journal is replayed.
- The stand-in serves no stream, no `/v0/info`, `/v0/applets` or capture, and runs on macOS and Linux only; on Windows it says so and exits.
