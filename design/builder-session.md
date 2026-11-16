# Builder session

The builder session is the headless object that binds the four builder libraries for one workspace: the [spec format](spec-format.md), the [catalog](catalog.md), the [command layer](command-layer.md) and the [loaders](loaders.md). It owns everything a person or an agent changes while building a visual, and it answers every command that needs no window. The app owns one session for its workspace and drains its commands between two frames; until the app's API server serves the command endpoints, a small stand-in tool serves the session's commands over HTTP so that `caliper-mcp` and `caliper-agent` can be tested end to end. This page describes the session, what it binds, what it leaves to the app, the session in the app, how it resolves the data of a panel, and the stand-in.

## What is built and what is planned

Built on this branch, in `src/builder/session/` with headers in `src/builder/include/caliper_builder/session/`:

- the library `caliper_builder_session`: `Session`, the `SessionHost` interface, the spec edits as pure functions with the conversions between the spec and the command table's forms, a line diff, the catalog provider for loader outputs, the resolved data of a panel, [stacked previews](#stacked-previews), the [review of spec edits](#review-of-spec-edits), and the widget agent's context bundle (`context_bundle.h`, described on the [Widget agent](widget-agent.md#the-context-bundle) page);
- the library `caliper_builder_session_host`: a strict HTTP/1.1 server for the two command endpoints of Caliper API, with the API file and token helpers;
- the tool `caliper-builder-host`, the development stand-in for the app's command endpoints;
- the command table brought in line with the spec format, with a test that keeps it so;
- the seam for the sources the app serves: `CatalogDataSource`, which gives the current tensor or table of a catalog id with its generation, `DataSources`, where the app registers one per kind with `Session::data_sources()`, the generation of a panel with its cheap probe `panel_generation()`, and the cache of transformed tensors;
- the tests in `src/builder/session/tests/`, including the two acceptance tests of this phase;
- the session in the app, in `src/host/builder/` ([below](#the-session-in-the-app)): the app, `caliper dev` and `caliper shot` own one session for the workspace of `--workspace`, `CALIPER_WORKSPACE` or `workspace/` under the app-data folder, drain its queue on the frame thread within a time budget, run the long commands on the host's job system, implement `SessionHost`, show each journaled command in the Activity pane, and close the session before the host's stores; with the command script, a testing and demo aid, and the tests `caliper_app_session` and `caliper_app_session_hidden`;
- the spec page, which draws the open spec with the spec runtime after `applet.open`, `panel.render` through the runtime's offscreen renderer, and the app's API server serving the command endpoints through the session (`AppSession::command_gateway()`), with the process test `caliper_app_route`.

Built in the host, in `src/host/catalog_providers/`: the catalog providers and data sources for live views, feed channels and metrics, and `register_catalog_providers()`, which gives them to a session ([Catalog](catalog.md)).

Planned:

- `caliper-builder-host` is removed and the tests of `caliper-mcp` and `caliper-agent` move to `caliper --hidden --api`;
- the live, feed and metric catalog providers;
- the approval cards, so that the person sees and decides what an agent's command waits for.

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

The app reaches each part through an accessor (`store()`, `catalog()`, `registry()`, `queue()`), runs a command with `invoke()` on the calling thread or through the queue, reads the open spec with `open_spec()` (its value, its path, whether it is dirty and its canonical text), reads a preview with `preview(id)`, resolves a panel's data with `resolve()`, and turns the review of spec edits on or off per origin with `set_review()`. `close()` shuts the queue down, which answers every waiting call with `command.shutdown`, and closes the journal.

### The commands it binds

| Group | Commands | Notes |
|---|---|---|
| catalog | `catalog.list`, `catalog.describe`, `catalog.sample` | Through `Catalog`, which checks every entry and bounds every sample. |
| data | `data.query`, `data.import` | `data.query` runs the store's read-only query and cuts its rows at 8 MiB of JSON. |
| loader | `loader.list`, `loader.describe`, `loader.create`, `loader.update`, `loader.run`, `loader.delete` | |
| spec | `spec.new`, `spec.open`, `spec.get`, `spec.set_text`, `spec.validate` | |
| source, panel, layout | `source.add`, `source.patch`, `source.remove`, `panel.create`, `panel.patch`, `panel.remove`, `layout.place`, `layout.set` | |
| preview | `preview.begin`, `preview.diff`, `preview.commit`, `preview.discard` | [Stacked previews](#stacked-previews). |
| applet | `applet.save`, `applet.open`, `applet.list` | |
| review | `review.status` | Which of the caller's commands wait for the person. |
| history | `history.undo`, `history.redo`, `history.list` | Built into the registry. |

### What SessionHost leaves to the app

Five commands need the running app: `app.info`, `app.applets`, `app.launch`, `panel.render` and `capture.frame`. They stay unbound, and answer `command.not_bound`, until the app gives the session an implementation of `SessionHost` with `set_host()`. The interface has one function per command, each returning the command's result or its error; `set_host(nullptr)` unbinds them again. For `panel.render` the session does the part that needs no window: it finds the panel in the open spec (`panel.not_found`) and resolves its data, and the host only draws that data off screen and returns the PNG. A test binds a fake host and checks each command and that `panel.render` hands the host the resolved data.

## Editing the open spec

Every command that edits the open spec goes through one function, `apply_spec_edit()`, which applies the command to a copy of the spec and validates the result with a `SourceResolver` that asks the catalog. The resolver gives the spec validator what the catalog knows of each catalog id: its kind, its columns with their DuckDB types, its dtype and shape and its wire flag. A catalog id that nothing knows yet is the warning `spec.unresolved_source`, so a spec can name a source before its loader runs.

- An edit whose own rule fails is refused with the command's code: `source.exists`, `source.not_found`, `source.in_use` (with `where` naming the first panel binding or the first `from` that uses it), `panel.exists`, `panel.not_found`, `layout.same_panel`, `layout.invalid`.
- An edit that leaves the spec with any error is refused with the validator's own code and the document path in `where`, so the open spec is valid after every command and a preview can always draw it.
- The result carries the validator's warnings, such as `spec.not_portable`, and the change as a unified diff of the spec's canonical TOML (`diff`, empty when the text does not change), with `applied`.
- Each edit takes `dry_run`. A dry run checks the change and returns its warnings and its diff without applying it, so that the side chat and the widget agent can show the change for approval before it happens, as contracts section 12 asks of the BYOK sidecar.
- A dry run has no effect at all, whether it succeeds or is refused, and neither has an edit that leaves the spec as it was. The handler reports `no_effect` on its output, and the registry treats the invocation as a read: no undo step, the redo stack and an open merge kept, and no journal line. An agent that previews a change never costs the person their redo.

`panel.create` without `place` puts the new panel to the right of the whole layout; the first panel of a spec becomes the whole layout. `place` and `layout.place` wrap the target's leaf in a split, or add a tab next to a target that is already a tab. `panel.remove` and `layout.place` take a panel's leaf out and collapse a split that is left with one child.

### Undo

Every edit returns a closure pair. The undo restores the spec as it was before the command, after checking that the open spec is still the same document and still holds what the command left; otherwise it fails with `history.conflict`. The redo makes the same checks the other way. A command sent with a merge key, such as a drag of a slider, checks only that the document is the same, because the history keeps the first undo of a merged run and the newest redo. `spec.new`, `spec.open` and `applet.open` replace the whole document; their undo puts back the document that was open before, unsaved changes included, and their redo puts back the one it replaced. Documents carry an id, so an undo can tell the open spec from another spec opened since.

### Diffs

`unified_line_diff()` writes the diff in the format of `diff -u`, with three lines of context, hunks merged when they are close, and the marker for a text without a final newline. It sets aside the lines both texts share at their start and end, then compares the rest with a longest common subsequence when the product of the two line counts is at most 2^22, and shows the middle as removed and added as a whole beyond that. A spec is at most 512 KiB, so the work is bounded either way.

## Stacked previews

A preview lets edits that build on each other be shown together and committed as one step ([Command layer](command-layer.md#preview) gives the commands, [Widget agent](widget-agent.md#preview-and-apply-on-the-command-layer) the role that uses them). The session keeps each preview under its document lock: its id, the document it was begun on, the open spec its copy was last computed from, the copy itself, the edits that changed the copy, their size, its scope and when it was last used.

- An invocation names a preview with its `preview` field, which the stand-in's server and the app's API server fill from `X-Caliper-Preview`. `spec_edit` then applies the edit to the copy with `apply_spec_edit()`, as a real edit is applied, and reports `no_effect`. Every other command that mutates is refused with `preview.not_previewable` before its handler runs, also with no effect.
- Before a preview is read or edited, it is refreshed: when the open spec is no longer the spec its copy was computed from, its edits are applied again to the open spec, and an edit that no longer applies answers `history.conflict`. The copy is therefore always the open spec with the preview's edits applied.
- `preview.commit` refreshes the preview, ends it and returns its edits as follow-ups with the preview's id as their group; the registry runs them through `spec_edit` against the open spec, with the origin that sent each, as one step. They cannot fail at that point, because the refresh just applied them to the same spec under the same lock.
- `Session::resolve(panel, preview)` resolves a panel of a preview, and `Session::preview(id)` gives the preview's spec, its number of edits and its scope, so the app can draw the preview in place. Reading a preview counts as using it.
- `PreviewLimits` bounds the previews open at once (16), the edits of one (64) and their arguments (4 MiB), and the idle time after which a preview is dropped (30 minutes); the ids of the last 64 dropped previews are kept, so that a late call learns `preview.expired` rather than `preview.not_found`.

## Review of spec edits

The session's policy enforces the review of spec edits for every agent, whether or not the agent asks on its own ([Command layer](command-layer.md#approval) has the rule). `ReviewSettings` in `SessionOptions` says which origins are reviewed, `agent` and `mcp` by default, and `set_review(origin, on)` changes it while the session runs; the setting is an atomic per origin, read by the policy on the frame thread and written by the app's settings.

- The policy asks for an edit of the open spec from a reviewed origin when the edit would change the spec. It computes the edit on the open spec itself; an edit that fails or changes nothing, a dry run and an edit of a preview are allowed and run at once.
- When it asks, the registry asks the session's approval review, which hands over what the policy just computed: `{"diff", "warnings"}`, with the diff of the canonical TOML labelled with the spec's file. The pending approval carries it, so `pending_approvals()`, the stand-in's approval lines and the app's approval endpoint show the same card for the side chat's agent, an outside agent over MCP and a program.
- The session also records, for each parked edit, the document and the lines the shown change removes and adds. When the person approves, `spec_edit` finds the record, makes the edit on the open spec as it is then and runs it only when the document is the same and the change is the same; otherwise it answers `history.conflict`.

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

**The approval policy.** The session's policy is the default policy of the command layer, with the [review of spec edits](#review-of-spec-edits) and two rules that look at the invocation itself, because a descriptor's `destructive` flag is fixed per command:

- Running a Python loader runs the user's code. `loader.run` of a Python loader, and `loader.create` or `loader.update` of one with `run` set, ask for every origin except `ui` and `builder`, each time. The policy records each invocation it asked about, and the handler passes `RunOptions::allow_python` to the store only for `ui` and `builder`, or for an invocation it finds in that record, which can reach its handler only after the person approved it. An app that replaced the policy with one that allows everything still gets `loader.python_needs_approval` from the store, so the rule fails closed. A redo never runs a script: it puts the recipe back, and a run needs its own approval. `data.import` creates a Python loader but never runs it.
- `applet.save` from `agent`, `mcp` or `api` asks only when it would overwrite a saved applet. Writing a new applet destroys nothing, so an agent can save its work without a prompt, and overwriting asks.

## Applets

`applet.save` writes `applets/<id>/<id>.spec.toml` and then the manifest `applets/<id>/<id>.caliper.toml`, each through a temporary file in the same folder and a rename, so a reader sees the old file or the new one and never a mix. The manifest follows the spec format page: `id = "local.<id>"` with `_` mapped to `-`, `name` from the title, `version = "0.1.0"` for a new applet (an overwrite keeps the old version), `summary` from a one-line description of at most 200 characters, `kind = "spec"` and `spec`. An applet that is already saved needs `overwrite`, and an id whose manifest id another applet of the workspace has is refused with `applet.exists`. Saving under the spec's own id makes the files the open spec's home and clears its dirty flag; saving under another id writes a copy and leaves the open spec as it was.

`applet.open` reads the manifest (at most 64 KiB), requires `kind = "spec"` and a plain file name in `spec`, and opens the spec after parsing and validating it. `spec.open` takes a path relative to the workspace and refuses an absolute path, a `..` segment and a symbolic link that leads out of the workspace with `spec.path_refused`. `applet.list` lists the folders of `applets/` whose manifests are spec applets, at most 4096 folders.

## Resolving a panel's data

`Session::resolve(panel)` returns, for each bound slot of a panel of the open spec, the data the spec runtime will draw; `Session::resolve(panel, preview)` does the same for a panel of a preview. It is the data path of the runtime and of `panel.render`, tested here without drawing.

1. A binding names a spec source. The chain of `from` is followed to its root, a `catalog` or `sql` source, at most 64 hops, so a cycle stops with `resolve.cycle`.
2. A tensor root, a `tensor:` catalog source, is mapped from its file through the store. The transforms of the root, then of each derived source down the chain, are applied in order with the loaders library's `tensor_ops`. The size of each result is computed with `transform_result_shape()` before it is made, and a result past 256 MiB is refused with `resolve.too_large`.
3. A tabular root, a `table:` catalog source or a `sql` source, is read once per call through the store's bounded read-only query. A table selects only the columns the panel binds, or all of them when a binding names the whole source; a `sql` source runs as written, so its `ORDER BY` holds. The rows are cut at 100,000 and at 2^22 cells, and `truncated` says so.
4. A column binding gives one column as a list of cells. A whole tabular binding gives a table; for `table.rows`, its columns are those of `options.columns`, in that order. A tensor binding gives the tensor.

5. A root that the app serves, a `live:`, `feed:` or `metric:` catalog source, is read through the `CatalogDataSource` registered for its kind. A live view gives its newest snapshot as a tensor, a copy that the source never changes, on which the transforms of step 2 run as on a mapped tensor, with the same bound; a feed channel and a metric give columns, cut at the same rows and cells as a table. Without a registered source such a root answers `resolve.unavailable`, and a source that went away (a detached session, a deleted run) answers it with the reason.

The bounds are `ResolveLimits`, which an app may change.

### Refresh

A served source changes many times a second, so a panel needs to learn cheaply whether its data changed.

- **The generation.** Every served source has a generation that grows whenever its data changes: a live view's grows with each copy of a new snapshot, a feed channel's is its newest sequence number, and a metric's moves when its series gains points. `PanelData::generation` mixes the generations of the served roots that a panel binds, in binding order, and is 0 for a panel that binds none.
- **The probe.** `panel_generation(spec, panel, data_sources())` computes the same number from the sources' `generation()` alone: a walk of the panel's chains and one call per served root, each a map lookup under a short mutex, about a microsecond in all, with no data read. The spec runtime calls it once per frame for each panel it shows and resolves a panel again only when the number differs from the generation of the data it drew.
- **The cache.** The result of a chain of transforms over a served tensor is kept in `DataSources` by the chain (the root's id and every transform down the chain) and the generation it was computed from, at most 64 results and 256 MiB, the least recently used going first. A resolve whose root has not changed reads the generation and takes the cached tensor, so the transforms run at most once per new snapshot and chain, however many panels show it and however many frames ask. Tables are not cached: a panel that binds a feed or a metric is resolved again only when the probe changes.
- **The cost per frame.** With nothing new, a live panel costs the probe. A resolve with nothing new costs a copy of the open spec and the cached tensor; a resolve after a new snapshot costs the transforms once. The benchmark `caliper_catalog_providers_bench` (`src/host/catalog_providers/bench/`, built on demand) measures these costs for the acceptance case; [Catalog](catalog.md#the-cost-of-a-live-panel-measured) has the figures, an upper bound taken on a busy machine. In short: with nothing new a live panel costs under a microsecond for the probe, and after a new snapshot the entropy and the mean over a `[12, 12, 128, 128]` attention tensor take about 11 ms, too much for 60 snapshots a second on the frame thread, while the same panel over a tiled `[12, 128, 128]` payload, with the heads reduced on the producer, takes 0.9 ms.

## The catalog provider

`LoaderCatalogProvider` turns each `LoadedSource` of the store into a `CatalogEntry`. `origin` holds the loader, the file and, for a tensor, its name in the file; an origin value is cut to 1024 bytes and the warnings to 16 of 1024 bytes, so an entry from a long path passes the catalog's strict check instead of being dropped. A table's sample is its first rows through the store's query, with the non-finite values of floating-point columns written `nan`, `inf` and `-inf` as the catalog spells them; a tensor's sample comes from `sample_tensor()` over its mapped bytes. The store's sources change only through loader commands, so the session increments the provider's revision after each of them.

## The command table and the spec format

The `source.*`, `panel.*` and `layout.*` schemas of the command table now accept exactly what the spec format accepts wherever the schema subset can say it: every slot of every primitive; the binding grammar with its 320-character bound; every option, with `z_label`, the ranges `x_range`, `y_range` and `range` as pairs of numbers, the log flags, the four colormaps as an enum, `bins` from 1 to 65,536, `columns` of 1 to 256 names and `text` of at most 16,384 characters; at most 16 transforms with axes from −8 to 7, safe integers for `index`, `start` and `stop`, and `step` of at least 1; a catalog id of at most 256 characters; and the layout's titles. `panel.create` no longer requires `bind`, since a text panel may have only `options.text`.

The test `test_table_conformance.cpp` keeps them in line: every source, panel and layout of every valid spec fixture, turned into the arguments of `source.add`, `panel.create`, `panel.patch`, `source.patch` and `layout.set`, passes the table's schemas and decodes back to the same item; the result forms pass the result schemas; and values that the spec refuses and the subset can express (a fifth colormap, 65,537 bins, a range of three numbers, an axis of 8, 17 transforms) are refused by the table too. What the subset cannot express, such as `index` together with a range in a slice or the slots a primitive takes, the spec validator refuses after the arguments pass.

## The session in the app

The app holds an `AppSession` (`src/host/builder/app_session.h`), and so do `caliper dev` and `caliper shot`. It owns the `Session` of one workspace, drains its queue on the frame thread and implements `SessionHost`. It is built with `CALIPER_WITH_BUILDER`; without it a stub of the same interface opens nothing and says, for a workspace named on the command line, that the builder is not in the build. `main.cpp` and `shot.cpp` gain a few lines: they start the session, call `frame()` where they take Caliper API's calls, hand it the read-back of a capture, and close it.

### The workspace

The workspace is the folder of `--workspace <dir>`, else of `CALIPER_WORKSPACE` when it is set and not empty, else `workspace/` under the app-data folder, which `CALIPER_DATA_DIR` moves with the rest (`workspace_choice()` in `src/host/host_cli.h`, `caliper::default_workspace_dir()` in `src/app_paths.h`). For `caliper shot` without `CALIPER_DATA_DIR` the default is under its temporary data folder, so a shot still writes nothing in the app-data folder. The reference pages Caliper CLI (`docs/references/cli.md`, "The workspace") and Folders and environment variables (`docs/references/environment.md`) describe them.

### Opening

`start()` opens nothing. A workspace that the command line chose is opened on a worker once the first frame has been drawn; the default workspace is opened only when something first needs it, through `ensure_open()`, `wait_open()`, a command script or the first command of Caliper API. So the app without `--workspace` starts as before and creates no folder. The frame thread adopts the opened session at its next `frame()`: it names itself to the registry as the frame thread and binds the host. A workspace that cannot be opened, such as a path that names a file, leaves the app without the builder; standard error and the Activity pane say why, `wait_open()` returns null and `open_error()` gives the reason.

### The frame loop

`frame()` runs once per frame where the app takes Caliper API's calls, between two frames. It first withdraws the approvals whose wait ran out (`drain(0)`), then drains one item at a time while the frame's budget lasts, 4 ms by default, and at least one item a frame, so a long queue is spread over frames instead of holding one. Before each item it looks at it with `CommandQueue::peek()`:

- `data.import`, `data.query`, `loader.run`, `loader.create` or `loader.update` with `run`, and the approval of one of them, run on a worker, a job of the host's job system labelled with the command and its loader, which the jobs tray shows. The worker calls `drain(1)`, so the command runs through the registry as every command does and its result reaches its caller through the queue's future.
- `capture.frame` asks the app for a read-back of the frame drawn next and drains nothing further until the app hands the pixels over or says it could not; then the command runs and answers at once.
- Every other command runs on the frame thread, where the spec, source, panel, layout, applet and history commands must run.

### Long commands and the registry's lock

The registry runs every command under one lock, and the store under another. A loader that runs on a worker holds both, so a frame that called the registry, the store or the catalog would wait for it. The app keeps the one lock and keeps the frame thread off it instead: while the worker runs, `frame()` does not touch the queue, the registry or the store, and later commands wait in the queue, in their order, until the worker is done. `worker_busy()` says so to other code of the frame thread, such as the spec runtime, which must not read the store then either. Splitting the registry's lock would let short commands pass a running loader, but it would also let them run against a workspace that the loader is changing, and it would change a library that the other builders use; the queue's order is what agents and the journal rely on.

### Caliper API

`AppSession::command_gateway()` is the gateway that the app and `caliper shot` set on their API server once it listens, and clear before the session closes. It serves the command table at once, without a workspace. A command that arrives before the workspace is open opens it, the default workspace on its first use, and goes to the queue as soon as the frame thread has adopted the session, from whichever of the frame thread and the server's thread first asks whether it is ready; a workspace that cannot be opened answers `command.not_available` with the reason, and a command held after the session closed answers `command.shutdown`. The gateway converts requests and answers approval lookups with the pieces of `src/host/api/session_gateway.h`, but `GET /v0/approvals/{id}` reads the pending approvals from a copy that each frame without a worker refreshes, because the registry's lock is held while a loader runs and the lookup runs on the frame thread. A pending approval's state carries the command's name, its origin and its review, the diff and the warnings of a spec edit, for the approval cards to come.

### SessionHost

- `app.info` and `app.applets` answer what `GET /v0/info` and `GET /v0/applets` answer, from the same functions (`info_json()` and `applets_json()` in `host_cli.h`).
- `app.launch` goes through the app's navigation, which `POST /v0/applets/{id}/launch` now shares (`launch_by_id()` in `main.cpp`): the page shown goes first and the applet's page opens. A 404 is `app.applet_not_found`, an applet that is not ready `app.launch_refused`, and one that fails to launch `app.launch_failed`, each with the API's message.
- `capture.frame` reads back the frame the window presents next, on Metal, through the app's `WindowCaptures` under a ticket of its own beside the API's; in `caliper shot` the offscreen canvas at the end of the next frame. Where the app cannot read its window back, it answers `capture.unavailable` with the reason Caliper API gives with its 501. A read-back that does not arrive within 5 seconds also answers `capture.unavailable`. The result is `png_base64`, `width` and `height` at the framebuffer's size.
- `panel.render` draws with the spec runtime: the app and `caliper shot` give `AppSession::set_panel_renderer()` a `PanelRenderSeam` (`src/host/builder/panel_renderer.h`, renamed from `PanelRenderer`, which is the runtime's own interface) over the runtime's `OffscreenPanelRenderer`, whose canvas is made on the first call between two frames and shares the app's font atlas. Without one it answers `command.not_available`. The session finds the panel and resolves its data before it asks, and hands a resolve error to the renderer with the data, so an offline panel is drawn in its state rather than refused.

### The spec page

The app shows the session's open spec on a page of its own beside the applet page and the live page (`src/host/builder/spec_page.h`). Every `applet.open` that succeeds, from any origin, opens it: the journal's observer counts those calls (`AppSession::applets_opened()`), and the app leaves the page it shows, as Home does, and enters the spec page between two frames. The trigger is the command, so that every route ends in one path: a launch of a spec applet from the landing screen's list, `CALIPER_AUTOLAUNCH`, `app.launch`, `POST /v0/applets/{id}/launch` or `caliper shot --launch` validates its spec and, for an applet of the workspace, sends `applet.open` from the `ui` origin through the queue with `SpecPage::request_open()`; an applet found in another search folder is drawn from its file without becoming the open spec. The app gives the loader the session's workspace before it scans, so the landing screen lists the workspace's spec applets, marks the one the page shows `ACTIVE`, lists the folder again after each `applet.save` (`AppletLoader::rescan_workspace()`, `AppSession::applets_saved()`), and shows a spec applet's format, panels and sources in the Inspector. The page bar holds Home and the spec's title where an applet's name goes, the status bar the machine, the workspace's folder name, the numbers of panels and sources and `unsaved` while the open spec is dirty, and the jobs tray keeps its strip; `SpecRuntime::draw()` fills the work area those strips leave. The page reads `open_spec()` again only when the journal has recorded a command since it last looked (`AppSession::journaled()`), and the runtime resolves only panels whose fragment changed, and every panel again when the catalog's revision grows. While a long command runs on the worker, the page reads neither the session nor its catalog: it draws what it has, a panel that must be resolved meanwhile shows a busy state, and every panel is resolved again afterwards. `caliper shot` draws the same page for `--launch <workspace applet id>` or after an `applet.open`. The app links `caliper_sternbild_ui`, now defined at the root beside the executable, so the runtime and the app share one copy of the faces and one font atlas. Without the builder the page is a stub that never opens.

### Activity

The session's journal has an observer (`JournalOptions::on_append`). Every command it records, that is every mutating command that ran, failed or was denied, but no dry run and no read, becomes one Activity line at the next frame: the event `<origin> command` (`ui`, `builder`, `agent`, `mcp` or `api`), the command's name as the object, and its error code as the detail when it failed. The origin leads because the object column is as wide as its widest path. Each line is also written to standard error as `[session] command <name> (<origin>)`. The workspace's opening is a `workspace` line, or `workspace not opened` with the reason.

### Shutdown

`close()` runs in the app's teardown after the API stops and before the host's job system and stores. It shuts the queue down, so waiting calls and approvals get `command.shutdown`; waits for a command running on the worker, which holds the registry, and for the command script; then unbinds the host and closes the journal and the workspace's database, all before `main()` returns, since DuckDB aborts when it is left to static destructors.

### The command script

`CALIPER_COMMAND_SCRIPT` names a file of JSON lines, each an invocation (`origin` defaults to `ui`; empty lines and lines starting with `#` are skipped). Once the workspace is open, a thread of its own submits them one at a time through the queue, as an outside client does, and writes each result to `CALIPER_COMMAND_RESULTS` as a JSON line, then `{"done": true, "commands", "failed"}`. It served the hidden app's process test before the API's command endpoints were connected, and it serves demos and screenshots: `caliper shot` keeps drawing frames until the script is done. A line whose command asks for approval never holds the script: it answers `command.needs_approval` with its approval id, which stays pending, unless the line carries `"approve": true`, in which case the script approves it as the person would.

### The start-up cost

Nothing of the session runs before the first frame: `start()` only stores its options, and the workspace opens on a worker afterwards. Measured on an Apple M5 Max with `caliper --hidden` and `CALIPER_EXIT_AFTER=1`, ten runs each, medians, while two other builds ran on the machine (load average 7.3 before, 7.3 to 11.3 during), so the figures are rough: the time to the end of initialisation was 92 ms without a workspace, 73 ms with a new one and 75 ms with an existing one, all before any session code runs; a whole run with its teardown took 1132, 1120 and 1127 ms; and opening the workspace took 6 ms for a new folder and 5 ms for an existing one, on the worker. The session adds no cost that these runs can tell apart from noise.

### Tests

`caliper_app_session` (`tests/test_app_session.cpp`) drives `AppSession` with a fake frame loop on a thread of its own and a job system of its own: commands from another thread run on the frame thread in order, at most two of 3 ms each within a budget of 4 ms; a loader that runs 600 ms on the job system lets frames go on, none longer than 100 ms, and the command after it waits; closing while a loader runs answers the waiting commands with `command.shutdown`, finishes the loader, and leaves a journal and a database that open again; the host's commands, the capture and its refusal, and `panel.render` with and without a renderer; the Activity lines; the lazy opening of the default and of a chosen workspace; a workspace that cannot be opened; and the command script. `caliper_app_session_hidden` starts `caliper --hidden --workspace` with its own `CALIPER_DATA_DIR` and a command script that imports a CSV file, runs its loader, builds a two-panel spec, saves the applet, captures the frame and asks for `panel.render`; it stops the app with SIGTERM and checks the results, the workspace's files, the journal, a replay of it and the saved applet.

## The stand-in host

`caliper-builder-host` is a development tool, not part of the product. It stands in for the command endpoints of the app's Caliper API until the wiring, and the app's own API server replaces it then.

```sh
caliper-builder-host --workspace <dir> --api-dir <dir> [--port <n>] [--approve allow|deny|ask]
                     [--review none|<origin>,<origin>...]
```

It opens the workspace as a `Session`, listens on 127.0.0.1 on a free port (or `--port`), and writes `<api-dir>/api.json` with the shape of the app's API file: `version` 1, `pid`, `port`, `token` and `url`, `http://127.0.0.1:<port>/v0`. The file has mode 0600 and is removed on exit. The app's file also has `stream_url`; the stand-in leaves it out, because it serves no stream, so a client that needs the stream must run against the app. The main thread drains the command queue, as the app's frame thread will. SIGTERM or SIGINT stops it: waiting calls are answered with `command.shutdown`, the API file goes, and it exits with 0. It never prints the session token.

It serves `GET /v0/commands` and `POST /v0/commands/{name}` exactly as contracts sections 8 and 15 and the command layer page describe, and nothing else:

- the checks of API v0, in order: `Host` must be `127.0.0.1:<port>` or `localhost:<port>` (403); an `Origin` must be `null`, an http or https origin on 127.0.0.1, `localhost` or `[::1]`, or `vscode-webview://` (403); the path and method (404, 405 with `Allow`); the token in `Authorization: Bearer`, or for a GET in the query parameter `token`, compared in constant time (401 with `WWW-Authenticate: Bearer`);
- a strict, minimal HTTP/1.1: one request per connection; a head of at most 16 KiB (431) read within 5 seconds (408); CRLF line ends, no folded headers, no control characters and no repeated header that the checks read (400); no `Transfer-Encoding` (501); a POST with `Content-Length` (411) of at most 1 MiB (413) and the type `application/json` (415), read within 10 seconds (408), that is JSON (400); a GET with no body (413);
- a body that is JSON but not an object, or nests deeper than 64 levels, is answered 200 with `command.invalid_args`, and a name that is not a command with `command.unknown`, so that a 404 means only that there is no command layer;
- the origin is `api`, or `mcp` or `agent` from `X-Caliper-Origin`; any other value is refused with 400, so a request never claims `ui` or `builder`; `X-Caliper-Group` gives the group and `X-Caliper-Preview` the preview;
- a call from `mcp` or `agent` that needs approval is held for up to 120 seconds; one from `api` gets `command.needs_approval` with its approval id at once;
- at most 32 connections are served at once, and one more gets 503 with `Retry-After: 1`; every response carries `Connection: close`, `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.

`--review` names the origins whose spec edits are reviewed, such as `agent,mcp`, which is what the app has by default; it is `none` unless given, so that the tests written before the review keep their flows. The host prints one JSON line per pending approval on standard output, `{"approval", "name", "origin", "group", "args", "review"}`, with the review of a spec edit or null. `--approve` decides what the approval gets: `allow` approves it and `deny` denies it, and the line then also carries `"decided"`; `ask`, the default, reads `allow <id>` or `deny <id>` lines on standard input. The HTTP acceptance test drives it with `ask`, the review tests with `--review mcp`, and the sidecar's dashboard test with `--review agent,mcp --approve allow`.

### How the wiring replaces it

The app now opens a `Session` for its workspace, names its frame thread to the registry, drains the queue between frames and implements `SessionHost` on its own window ([above](#the-session-in-the-app)); the live, feed and metric catalog providers are built in `src/host/catalog_providers/` and are given to the session with `caliper_host::register_catalog_providers()`, with the live catalog's `update()` called once per frame after the live poll. Its API server, which already holds the loopback, `Host`, `Origin` and token checks of API v0, gains the two command routes; each submits to the session's queue as `CommandServer::handle()` does here, with the same origins, approval waits and envelope, and the approvals come from the app's own approval UI rather than from standard input. `caliper-builder-host` and `caliper_builder_session_host` are then removed, and the tests of `caliper-mcp` and `caliper-agent` move to `caliper --hidden --api`.

## The acceptance tests

Both run headless in `test_acceptance.cpp`.

1. **In process.** A workspace is created; a Parquet file written through DuckDB and a safetensors file are put in a folder outside it; `data.import` imports both from the `ui` origin and `loader.run` runs their loaders; the three-panel dashboard of `dashboard.spec.toml` is built command by command (`spec.new`, `source.add`, `panel.create`, `layout.set`), and the open spec equals the fixture; two steps are undone and redone; every panel's data is resolved and checked against the files (the series' columns, the heatmap's mean over heads after its minmax, the table's chosen columns); `applet.save` writes the applet; the session closes; a new session over the workspace finds it with `applet.list`, opens it with `applet.open`, and gets the same spec and the same resolved data; and a replay of the journal gives the same spec.
2. **Over HTTP.** `caliper-builder-host` runs as a child process with `--approve ask`. From the `mcp` origin, `data.import` takes the files inside the workspace and refuses one outside it; each `loader.run` waits until the test answers the host's approval line with `allow`, and a `loader.delete` is denied; the dashboard is built with a dry run before each panel, undone and redone in part; `applet.save` writes a new applet without approval and waits for approval to overwrite it; SIGTERM stops the host with exit code 0 and removes its API file; and the workspace it leaves opens in process to the same spec and data, and its journal replays to the same spec.

## Decisions

- **`dry_run` and `diff` on every spec edit.** An agent's change must be shown before it applies (contracts section 12), and only the session can compute the canonical text after an edit.
- **A command that changed nothing is a read, decided by its handler.** A descriptor's `mutates` is fixed per command, but whether one invocation changed anything is known only to its handler, so `CommandOutput` carries `no_effect` and the registry, not each handler, keeps the history and the journal untouched. An earlier version let a dry run clear the redo stack and write a journal line.
- **Extension tables are not carried by the edit commands.** `x` is an open, recursive object, which a closed, strict tool schema cannot describe. `panel.patch` and `source.patch` keep an item's `x`, `spec.set_text` carries `x` anywhere, and `layout.set` replaces the layout without it.
- **`panel.create` takes no required `bind`.** A text panel may have only `options.text`.
- **Outside paths from agents are accepted when the person has already granted them.** It keeps the rule's purpose, that an agent cannot widen what loaders read, and lets an agent build a second loader over a file the person dropped.
- **Python runs are approved per invocation, and the handler checks that the approval happened.** The policy alone cannot tell the handler, and an app's own policy could allow everything; recording the asked invocations makes the permission fail closed.
- **`applet.save` asks an agent only before an overwrite.** Saving a new applet loses nothing.
- **Saving under another id writes a copy.** Changing the open spec's applet id as part of a save that cannot be undone would leave the undo history describing another document.
- **Resolved tabular data selects only the bound columns of a table but runs a `sql` source as written.** Wrapping a query in a subquery would not keep its order in DuckDB.
- **The stand-in accepts `X-Caliper-Origin: agent`.** The widget agent sidecar calls Caliper API, and `agent` has the same approval rules as `mcp`.
- **A preview is the open spec with its edits applied, refreshed on every use.** A copy taken once would go stale when the side chat or the person changes the open spec; applying the edits again keeps a preview drawable and makes a commit apply exactly what the preview shows, or refuse with `history.conflict`.
- **A preview needs an open spec.** For a dropped file with no open spec, the app runs `spec.new` from the `ui` origin before `preview.begin`.
- **The stand-in reviews no origin unless asked.** The app's default is `agent` and `mcp`, which the stand-in takes with `--review agent,mcp`; its own default keeps the flows of the tests that came before the review.
- **`loader.describe` reports `last_run.elapsed_ms` as null when it is not known.** The store does not record a run's duration, so the session knows it only for the runs it made since it opened.
- **The default workspace is `workspace/` under the app-data folder and opens on first use.** `CALIPER_DATA_DIR` then moves it with everything else, and the app without `--workspace` starts as before and creates nothing until the builder is used. A workspace that the command line chose opens after the first frame, on a worker, so a wrong path is reported at once without delaying the window.
- **Long commands run on a worker, and the frame thread waits for them by not draining.** The registry keeps its one lock and the queue its order. The app looks at the next item with `CommandQueue::peek()`, a small addition to the queue, and runs `data.import`, `data.query` and the commands that run a loader with `drain(1)` on the host's job system. Every `data.import` goes there, whatever the size of its files, because a size check would read the files on the frame thread.
- **`capture.frame` holds the queue until the next frame is read back.** A command answers on the thread that runs it, and the read-back of a presented frame completes a frame or two later; holding the queue keeps the order of the commands and never waits on the frame thread.
- **The Activity pane takes the journal's events.** An observer on the journal (`JournalOptions::on_append`) sees exactly the commands that changed the workspace or were denied, and none of the dry runs and reads, from whatever thread ran them.
- **The test path of the app's session is a command script.** Until the API's command endpoints are connected, `CALIPER_COMMAND_SCRIPT` drives the real queue from a thread of its own, as a client does, and its results file is what a process test reads; it also serves demos and screenshots.
- **The served sources reach the session through an interface it owns.** `CatalogDataSource` names no host type, so the session library still builds and tests without the host, and the host implements it beside its providers.
- **A panel's generation is a mix, not a sum.** A source that goes away drops its generation to 0, so a sum could come back to an earlier value; a mix of the generations changes whenever one of them does.
- **The cache is keyed by generation, not by time.** A transform's result is valid exactly as long as its snapshot is, so no time limit can make it stale or recompute it early.

## Changes to the contracts and to the earlier pages

- Contracts section 8 names `api` and `mcp` as the origins of an HTTP call; the stand-in also accepts `agent`, for the sidecar. The app's API server should do the same.
- The command layer page listed `loader.unknown_reader` and `loader.read_failed` among `data.import`'s refusals; the store's own codes (`loader.no_reader`, `loader.reader_mismatch`, `loader.pickle_refused`, `loader.io`) are what the refusals carry, plus `loader.path_refused` for an agent's outside path.
- The results of the spec edits gained `diff` and `applied`, their arguments `dry_run`, and `loader.describe`'s `last_run.elapsed_ms` became nullable. The command layer page records these.
- The invocation gained `preview`, the pending approval `review`, and the table `preview.begin`, `preview.diff`, `preview.commit`, `preview.discard` and `review.status`. The widget agent page had planned `Session::commit_preview()` and `Session::drop_preview()` as calls of the app and a `preview.stale` refusal; they became `preview.commit`, which only `ui` and `builder` may send, `preview.discard`, and `history.conflict` when the edits no longer apply to the open spec.

## Limits

- A preview's edits are kept in memory only; a restart drops every preview, as it drops the undo history.
- The session reads live views, feed channels and metrics only through the data sources the app registers; in the app that happens once the session builder connects `register_catalog_providers()`.
- `loader.delete` with `keep_outputs` is not available, because the store deletes a loader's outputs with it. The loaders library would need `LoaderStore::remove(id, keep_outputs)`.
- Undo does not reach files outside the session: a saved applet stays saved, and a loader run's tables stay as the run made them.
- Replay rebuilds the open spec, not the loaders; `spec.open` and `applet.open` read their files as they are when the journal is replayed.
- The stand-in serves no stream, no `/v0/info`, `/v0/applets` or capture, and runs on macOS and Linux only; on Windows it says so and exits.
- In the app, every command waits while a loader runs on the worker, reads included, because they all pass through the one queue in order.
- `history.redo` of an import runs its loaders again on the frame thread, since the history commands must run there.
- Going Home from an applet joins every job of the host's job system, so it waits for a running loader, up to the job system's 5-second limit.
- The app has no approval UI yet: a command from `agent` or `mcp` that needs approval, which includes their spec edits while review is on, waits until its caller's time limit, and the person sees nothing of it; the command script approves with `"approve": true`.
