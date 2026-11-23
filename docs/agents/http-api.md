# HTTP API

Caliper API is an HTTP interface to the running app. It lists the applets, launches one, takes data streams over [wire protocol v3](../reference/wire-protocol.md), and returns a capture of the frame the app draws. It also runs the commands of the command layer, which builds specs, loaders and sources with undo, a journal and approvals, and it serves the browser viewer of spec applets. The API listens on 127.0.0.1 only, and tokens protect it. It is off unless the app starts with `--api` or the [side chat or Open in browser starts it](#app-startup).

A capture works in the app's window on Metal and in the windowless mode, `caliper shot --api`. The command endpoints exist in the app, `caliper dev` and `caliper shot` when built with `CALIPER_WITH_BUILDER`. A build without the builder answers 404 on them, and `caliper-mcp` reads that as an app with no command layer. The Windows build refuses `--api`, as [Caliper CLI](../reference/cli.md#windows-limits) describes.

| Method and path | What it does |
|---|---|
| `GET /v0/info` | The host's version, ABI epoch, services and SDK |
| `GET /v0/applets` | The applets the app lists, as `caliper list` shows them |
| `POST /v0/applets/{id}/launch` | Opens that applet's page |
| `GET /v0/capture.png` | The frame the app draws, as a PNG |
| `GET /v3/stream` | The WebSocket of wire protocol v3, with the app as the hub, at the protocol's own path |
| `GET /v0/stream` | The same WebSocket, under the API's prefix |
| `GET /v0/commands` | The command layer's descriptor table |
| `POST /v0/commands/{name}` | Runs one command, with its arguments as a JSON body, and answers its result |
| `GET /v0/approvals/{id}` | What became of a command that needed the person's approval |
| `GET /v0/spec` | The browser viewer of spec applets, a page |
| `GET /v0/spec/{file}` | One of the three files of the WebAssembly views that the page loads |

## Startup

`--api` turns the API on for the app, for `caliper dev` and for `caliper shot`. The environment variable `CALIPER_API=1` does the same. `--api-port <n>` chooses the port, from 0 to 65535, and needs `--api`. Without it, the API listens on port 7341, and 0 takes any free port. [Caliper CLI](../reference/cli.md#caliper-api) lists the options.

```sh
caliper --api
caliper dev build --api --api-port 0
caliper shot --applets build --launch com.example.my-viewer --frames 216000 --out page.png --api
```

When the API listens, the process prints the address and **your session token**. It also writes the address and a token for outside agents in the **API file**:

```text
[api] Caliper API on http://127.0.0.1:7341/v0 with the session token <token> (agents read <API file>, whose commands are reviewed)
[api] Spec applets in a browser at http://127.0.0.1:7341/v0/spec?token=<viewer token>&applet=<id>
```

The session token on the first line is yours. Commands sent with it run as the origin `api`, without review, and on the stream it may control a run when the app allows control. Caliper writes it to no file. A program that should act without review takes it from this line, for example through `CALIPER_API_TOKEN` in the [examples](#worked-examples). The second line is the link of the [spec page](#get-v0spec), with the viewer token. It opens that page and nothing else.

**Commands sent with the API file's token are reviewed.** An outside agent reads the API file, through [`caliper-mcp`](mcp-server.md) or otherwise. Its token runs commands as the origin `mcp`. Each change to the open spec waits for your review in the app, and each command that deletes or overwrites saved data waits for your approval ([Who sends a command](#command-origin)).

The API file is `api.json` in the app-data folder ([Folders and environment variables](../reference/folders-and-environment.md)). The process writes it with mode 0600, so only its owner can read it. The process removes it when it exits, as `caliper serve` does with its hub file. It holds one JSON object:

| Key | Value |
|---|---|
| `version` | 1 |
| `pid` | The process id |
| `port` | The port the API listens on |
| `token` | The token of outside agents, not your session token. Its commands run as `mcp` and are reviewed. On the stream it only watches |
| `url` | `http://127.0.0.1:<port>/v0` |
| `stream_url` | `ws://127.0.0.1:<port>/v3/stream`, without the token |
| `spec_url` | `http://127.0.0.1:<port>/v0/spec?token=<viewer token>`, the spec page. Add `&applet=<id>` to open a saved spec applet |

Every token is new for each run. Each has 32 characters of `A-Z`, `a-z`, `0-9`, `_` and `-`, made from 24 random bytes. When the port is taken, the app starts without the API and says so on standard error and in the Activity pane, and `caliper shot` exits with 1. While the API runs, the status bar of every page shows its address, such as `api 127.0.0.1:7341`.

### App Startup

Two things in the app start the API when the app started without `--api`. The first is the [side chat](side-chat.md). The second is ⌘O Open in browser on the spec page ([Builder pane](../builder/builder-pane.md#browser-preview)), which opens a saved dashboard in the browser viewer. In both cases the API follows the rules of `--api`, and no setting can make it listen on another address or a chosen port. The app reports the start in Activity and on standard error, never with a token. An API that the person started with `--api` stays as it is, and neither feature starts a second one or stops it.

**For the side chat.** The side chat starts the API at the person's first message. The agent does not read `api.json`. Each agent session gets its own API file, `api-agent-<pid>-<n>.json` in the app-data folder, with the same keys and mode 0600. Its token runs commands as the origin `agent`, so the agent's changes to the open spec wait for your review. The app revokes that token and removes the file when the agent's session ends. The widget agent's sidecar gets its own file. When the chat's agent ends, the side chat stops the API it started and the app removes the API file. If Open in browser gave out a link of that API, the API keeps running until the app quits.

**For Open in browser.** ⌘O starts the API if it does not run. Then it opens the dashboard's link in the default browser:

```text
http://127.0.0.1:<port>/v0/spec?token=<viewer token>&applet=<id>
```

The link carries the viewer token, never your session token or the API file's token, and the app refuses to open it otherwise. The app also refuses, before it starts the API, in a build without the browser viewer and for a dashboard id that is not a workspace's applet id. That id is a lower-case letter, then at most 63 lower-case letters, digits, `_` or `-`. If the browser does not start, an API started for this request alone stops again.

Anyone who has the link can read what the [spec page](#get-v0spec) reads, from the origin `viewer`: the list of applets, the saved spec of every spec applet of the open workspace, and the panel data and the three files of the views. The link opens no other endpoint, no command that changes anything and not the stream. It works only on this machine, but any program or person that has it can read that data, and a browser may keep it in its history.

The viewer token ends when the API that made it stops, and the API answers an old link with 401. An API started for Open in browser runs until the app quits, because a page open in a browser keeps reading.

A hidden app opens no browser. A test gives it `CALIPER_TEST_OPEN_URL`, a file to which the app appends the link instead ([Host variables](../reference/folders-and-environment.md#host-variables)). Without it, the hidden app refuses and starts no API.

## Windowless Mode

`caliper shot` draws the app's applet page into an offscreen canvas and shows nothing on screen ([caliper shot](../reference/shot-and-serve-commands.md#caliper-shot)). With `--api`, it serves the API while it draws its frames, so it is the app without a window. `--frames` sets how long it runs, and SIGINT (Ctrl-C) or SIGTERM ends the frames early. It then writes its image, stops the API, removes the API file and exits with 0. A capture reads the offscreen canvas at the end of a frame.

## Security

Every request gets the checks of the stream ([The upgrade request](../reference/wire-protocol.md#upgrade-request)):

- The server listens on 127.0.0.1 only.
- The `Host` header must be `127.0.0.1:<port>` or `localhost:<port>`, with the API's port. The API refuses anything else with 403, as a guard against DNS rebinding.
- An `Origin` header, when the request has one, must be one that the protocol accepts: `null`, an `http` or `https` origin on `127.0.0.1`, `localhost` or `[::1]`, or `vscode-webview://`. The API refuses any other origin with 403.
- A token must come with the request: your session token, the API file's token, an agent's own token, or for the spec page the viewer token.

A request carries the token in an `Authorization: Bearer <token>` header. A GET may carry it as the query parameter `token` instead. The WebSocket and a link opened in a browser use that form. A POST takes the header only, so a state-changing request never has the token in its address. The API compares the token in constant time with every token it holds. It refuses a missing, wrong or revoked token with 401 and `WWW-Authenticate: Bearer`. On the stream it answers 403, as the protocol says, and the viewer token does not open the stream.

The API serves no file that a request names. It serves the spec page, which is built into the executable, and the three fixed files of the views, and it cannot run code. The commands of the command layer change specs, loaders and sources, which are data. A command that deletes or overwrites saved data, or runs a Python loader, waits for the person's approval in the app, and no endpoint approves or denies.

Only `POST /v0/commands/{name}` takes a body, and the API refuses any other request with a body with 413. Launch takes only an applet id in its path that the app's loader lists. A command request is built from its path, its body, the origin of its token and the headers `X-Caliper-Group` and `X-Caliper-Preview`. No other header value reaches the command layer or its journal, and the API never logs a token. Every response carries `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`, and the connection closes after it.

**The viewer token.** A request with the viewer token may only `GET /v0/spec`, `GET` the three files of the views, and `POST /v0/commands/{name}` for the three commands the page reads with: `spec.get`, `applet.list` and `panel.data`. The API refuses everything else with 403, including the stream. Its commands run from the origin `viewer`, and the API refuses an `X-Caliper-Origin` other than `viewer` with 400. The command layer runs only commands that change nothing from that origin, and it refuses any other with `command.not_allowed`.

### Command Origin

The origin of a command decides what the app reviews. It comes from the token that the request carries, never from a header:

| Token | Where it is | Origin | Review | The stream |
|---|---|---|---|---|
| Your session token | The start line only | `api` | None for spec edits. A destructive command answers `command.needs_approval` at once | Watches, streams a run in, and controls a run when the app allows control |
| The API file's token | `api.json` | `mcp` | Spec edits wait for your review, and destructive commands for your approval | Watches only |
| An agent's own token | `api-agent-<pid>-<n>.json` | `agent` | As `mcp` | Watches only |
| The viewer token | The spec page's link | `viewer` | Reads only | Refused |

The API ignores `X-Caliper-Origin` for every token but the viewer token, so a request cannot claim an origin that its token does not have.

**Agent permissions.** The side chat's agent and the clients of the API file's token each have a mode that you set in the app:

- Ask: the review described above.
- Accept edits: an edit of the open spec applies at once, and stays journaled and undoable.
- Read only: reads, dry runs and edits of a preview run. Every other change answers `command.denied` with a message that says so.
- Allow all: a command that would wait for your card runs at once. An overwrite of a saved applet and the replacement of a spec with unsaved changes still wait for your card. The app saves the mode, and it stays on until you choose another mode.

A mode does not govern the widget agent, your session token or the viewer token. No endpoint and no command reads or sets a mode or a rule. In every mode, an overwrite of a saved applet and the replacement of a spec with unsaved changes still wait for your card, and a rule never lets a Python loader run.

## Endpoints

### `GET /v0/info`

What `caliper info` prints, as JSON. `sdk`, `libtorch` and `ui_stack` are `null` when they are unknown or absent.

```json
{"abi_epoch":2,"host":"0.6.0","libtorch":"2.5.1","sdk":"0.2.0","services":["caliper.artifacts.v1","caliper.data.v1","caliper.log.v1","caliper.ui.v1"],"ui_stack":"03c557182adc5cd9"}
```

### `GET /v0/applets`

What `caliper list` prints for the applets the running app found, in the same order: the search order, then the id, then the highest version first. The fields have these values:

- `status` is the word that `caliper list` prints, such as `READY`, `ACTIVE` for the applet whose page is open, `REFUSED` or `SHADOWED`.
- `kind` is `library` for a compiled applet or `spec` for a [spec applet](../applets/manifest.md#spec-applets).
- `reason` is the text that `caliper list` prints under the applet. It is empty for a `READY` applet.

```json
{"applets":[{"folder":"/path/to/build/applets","id":"com.example.my-viewer","kind":"library","reason":"","status":"ACTIVE","version":"0.1.0"}]}
```

### `POST /v0/applets/{id}/launch`

Opens the page of the applet with that manifest id, as a click on its row on the landing screen does. The app tears down an open applet first, and an attached live session stays attached. If the new applet fails to launch, the app stays on the landing screen with the reason in the Activity pane.

For a [spec applet](../applets/manifest.md#spec-applets) the page is the spec page ([Caliper CLI](../reference/cli.md)). The answer comes once the app validated the spec, and the page opens a frame or two later with an `applet.open` from the origin of the request's token. If the open spec has unsaved changes, a launch with the token of `mcp` or `agent` waits on an approval card in the app. The answer then comes at once, with `status` `PENDING_APPROVAL` instead of `ACTIVE` and a `note` that explains the wait. A denial leaves the open spec as it was.

```json
{"already_open":false,"id":"local.attention-entropy","note":"Accepted, not open yet: opening it would replace the open spec's unsaved changes, so it waits for the person's approval in the app. spec.get names the open spec once it has opened; if the person denies it, the open spec stays as it is.","status":"PENDING_APPROVAL","version":"0.1.0"}
```

If a long command blocks the app from reading the open spec, the status is `PENDING`. The open then waits for that command and, if needed, for the person. Every other launch answers `ACTIVE`. A launch with your own token opens at once, and ⌘Z brings the unsaved spec back.

```json
{"already_open":false,"id":"com.example.my-viewer","status":"ACTIVE","version":"0.1.0"}
```

| Status | When |
|---|---|
| 200 | The applet's page is open. `already_open` is true when it was open before the request. |
| 400 | The id is not 1 to 128 characters of `A-Z`, `a-z`, `0-9`, `.`, `_` and `-`. |
| 404 | No folder in the search order holds an applet with that id. |
| 409 | The applet cannot be launched, with the status and reason that `caliper list` shows, or it failed to launch, with the loader's reason. |

### `GET /v0/capture.png`

The frame the app draws, as an opaque PNG. The app reads it back at the end of the next frame and draws no extra frame. The result depends on the mode:

- In the app's window, the PNG is the frame the window presents, at the window's framebuffer size in pixels. A capture works with the window hidden too ([`--hidden`](../reference/cli.md#hidden-window)), which is how tests and agents drive the whole app with nothing on screen. On the GL and Vulkan renderers, the app answers a capture with 501 and the reason.
- In the windowless mode, the PNG is the offscreen canvas. Its size is the one that `caliper shot --size` gives, 1600 by 1000 by default.

A capture answers 504 when no frame draws within 5 seconds.

### `GET /v3/stream`

The WebSocket of [wire protocol v3](../reference/wire-protocol.md), with the app in the hub role. It is the hub of [`caliper serve`](../python/hub-setup.md) on the API's port. Producers connect to it and each becomes a session, and viewers list the sessions and attach to one. Everything the protocol states for a hub applies.

The stream is at the protocol's own path, so every client of the protocol attaches to it unchanged: the app's NetSource, `caliper-viewer --url`, the link field of the app's attach pane and the `<caliper-view>` element. The client gives the token in the address. Your session token, the API file's token and an agent's own token open the stream. The viewer token does not.

```text
ws://127.0.0.1:7341/v3/stream?token=<token>
```

A connection opened with the API file's token or an agent's token only watches. It gets the data and the controls' state, its WELCOME sets `control` false, and the hub answers a CONTROL from it with `forbidden`. Thus an agent cannot pause, step, change a parameter of or run an action of a training run. It cannot publish either. The hub answers a producer's HELLO with the error `unsupported` and closes it with code 4002.

Only a connection opened with your session token may control. As for `caliper serve`, the hub forwards no control unless the app started with `--allow-control` next to `--api`. Without it, the hub answers every control with `forbidden` ([Controls and selections](../python/hub-setup.md#controls-and-selections)). `GET /v0/stream` is an alias with the same checks, for clients that keep every address under the API's prefix. The stream carries data, viewer events and, from protocol 3.1, controls. It never carries host commands.

### `GET /v0/commands`

The command layer's descriptor table. Each entry gives the command's summary, examples, and the JSON Schema of its arguments and result. It also says whether the command mutates, is destructive, runs on the frame thread, or an agent may send it (`agents`). Its `effect` decides how an agent's mode treats it: `read` for a read or a proposal, `edit` for an edit of the open spec, and `change` for anything else that changes something.

`caliper-mcp` and `caliper-agent` turn this table into tools. For the API file's token or an agent's own, the table leaves out the commands with `agents` false, which are `history.undo` and `history.redo`. The command layer refuses them to those tokens with `command.not_allowed`, and an agent reverts a change with an edit that you review.

```json
{"commands":[{"agents":true,"args":{"additionalProperties":false,"properties":{},"required":[],"type":"object"},"destructive":false,"effect":"read","examples":[{}],"frame_thread":false,"mutates":false,"name":"applet.list","result":{"...":"..."},"summary":"..."}],"version":0}
```

### `POST /v0/commands/{name}`

Runs the command `{name}`, such as `spec.new` or `panel.patch`. The body is the command's arguments, a JSON object of at most 1 MiB, with `Content-Type: application/json` and a `Content-Length`. Three headers shape the invocation:

| Header | Meaning |
|---|---|
| `X-Caliper-Origin` | Ignored. The origin is the token's ([Who sends a command](#command-origin)). With the viewer token it must be `viewer` when it is given. |
| `X-Caliper-Group` | Optional. Commands that share a group undo as one step. |
| `X-Caliper-Preview` | Optional. The id of a stacked preview, from `preview.begin`. An edit of the open spec applies to the preview instead, and `spec.get` reads it. |

The answer is the command layer's result envelope, with 200 whether the command succeeded or not. The server gives each invocation an id, `api-<n>`.

The commands act on the builder session's workspace. That is the folder of `--workspace` or `CALIPER_WORKSPACE`, else `workspace/` in the app-data folder ([Caliper CLI](../reference/cli.md#workspace)). The first command opens the default workspace, and a command that arrives meanwhile waits for it. If the workspace cannot open, the command answers `command.not_available` with the reason. The table of `GET /v0/commands` needs no workspace. When `applet.open` succeeds, from any origin, the app shows the spec page with that applet's spec. When no spec is open, `spec.get` answers `{"open": false}`, not an error. With a spec open, its result carries `"open": true` with the spec.

```json
{"id":"api-7","ok":true,"result":{"applied":true,"diff":"...","id":"note","warnings":[]}}
{"id":"api-8","ok":false,"error":{"code":"spec.no_open_spec","message":"...","where":""}}
```

The command layer refuses a command in the same envelope:

- `command.unknown` for a name that is not a command. Thus a 404 always means that the app has no command layer.
- `command.invalid_args` for arguments that fail the command's schema, and for a body that is JSON but not an object or nests deeper than 64 levels. A null on an optional argument whose schema does not admit null counts as the argument left out.
- `command.invalid_invocation` for a malformed group or preview.
- Any error of the command itself.

Faults of the request keep their HTTP status ([Errors](#errors)).

**Approval.** A command that deletes or overwrites saved data, or runs a Python loader, needs the person's approval in the app when an agent or a program sends it. For the origins `mcp` and `agent`, the API holds the request for up to 120 seconds while the person decides. It then answers in one of three ways:

- With the command's result, when the person approved.
- With `command.denied`, when the person denied.
- With `command.approval_timeout`, when nobody decided in time. The approval is then withdrawn.

A client must allow at least 130 seconds for such a call, as `caliper-mcp` does. For the origin `api`, your session token's, the request answers at once with `command.needs_approval`. Its error carries the approval id in `approval`, and the program reads the outcome later from `GET /v0/approvals/{id}`:

```json
{"id":"api-9","ok":false,"error":{"approval":"ap-3","code":"command.needs_approval","message":"...","where":""}}
```

A held request blocks nothing else, and the API serves other requests meanwhile. Spec edits from `mcp` and `agent` wait the same way.

The app shows each pending approval to the person as a card above the jobs tray, on every page. The card shows who asks (MCP agent, Side chat, Widget agent or API), the command and its target, and the time left before the request times out. For a spec edit it shows the changed keys with the validator's warnings, and the full diff on demand. For a destructive command it shows what the command would lose. The person chooses Approve or Deny. For a command that can become a rule, a card of the side chat's agent or of the API file's token offers Allow once, Allow always for this session and Always allow `<command>` instead of Approve. On the spec page, the panel that an edit changes shows REVIEW. If what a card showed changes while a request waits, the app replaces the card with a new one. A hidden app test decides through `CALIPER_TEST_APPROVALS` ([Host variables](../reference/folders-and-environment.md#host-variables)).

### `GET /v0/approvals/{id}`

What became of the command that waits, or waited, under approval `{id}`. `state` has one of three values:

- `pending` while the person has not decided.
- `decided` once the person has decided, with the command's result envelope in `result`. The app keeps the last 64 decisions.
- `unknown` for an id the command layer does not know or no longer keeps.

A pending approval also gives what the person is asked about: the command's `name`, the `origin` that sent it, and its `review`. For an edit of the open spec under review, `review` is `{"diff", "warnings"}`: the unified diff of the spec's canonical text and the validator's warnings. It is null for other commands. While a loader runs on the app's worker, the app answers from the pending approvals as they stood before it started, so a lookup never waits for the loader.

```json
{"approval":"ap-2","name":"panel.create","origin":"mcp","result":null,"review":{"diff":"--- run42-dashboard.spec.toml\n+++ ...","warnings":[]},"state":"pending"}
{"approval":"ap-3","name":"applet.save","origin":"api","result":null,"review":null,"state":"pending"}
{"approval":"ap-3","result":{"id":"api-9","ok":true,"result":{"id":"dash","manifest":"...","spec":"..."}},"state":"decided"}
```

An id is 1 to 128 characters of `A-Z`, `a-z`, `0-9`, `.`, `_`, `:` and `-`. The API refuses anything else with 400. Approval ids are not secret, and only the person decides.

### `GET /v0/spec`

The browser viewer of spec applets. It is one HTML page that draws a saved spec applet of the open workspace with the layout and the look of the app's spec page. It takes the token and the applet's workspace id from its address, `/v0/spec?token=<token>&applet=<id>`. Without `applet`, it lists the saved spec applets as links. The page answers with a `Content-Security-Policy` that lets it reach only its own origin, and with `Referrer-Policy: no-referrer`, so the token in its address never goes elsewhere. The policy allows scripts from its own origin and WebAssembly compilation (`'self' 'unsafe-inline' 'wasm-unsafe-eval'`).

The page shows a dashboard as the app does. It uses the layout rules of spec version 1. A layout leaf that names a control takes one control row of 30 px in a vertical split, and the other children share the rest. The split `shares` and the `min` of a leaf work as in the app. The page names each series line and the legend with the name that `panel.data` gives, and it colors the lines with the color key and the palette of the app, in the app's order. The web build generates the palette from the app's C++ constants in `web/src/render/palette.gen.ts`, thus no second list exists.

A control is drawn with its state and the look of a disabled widget. The viewer origin may not run `control.set`, `state.set`, `cursor.*`, `intervene.*` or `run.*`, because they change the app's state or act on the run. A hover shows why: "The browser viewer only reads: use the app to change this control." or "No script is attached." A readout shows its text, and no control is hidden.

The page draws the programs of a scene panel with the WebAssembly views of Caliper, one module instance for each scene panel. The module runs in a mode that the page feeds, with no stream, thus the viewer token still opens no stream. It checks each instance as it checks a relay of the stream: the hash of the pin over the files, the manifest, the imports and the compile of WebGPU (`program.browser_refused`). The app serves the three files of the module beside the page, with the session token or the viewer token in the query:

- `GET /v0/spec/caliper-views-loader.js`
- `GET /v0/spec/caliper-views.js`
- `GET /v0/spec/caliper-views.wasm`

The API serves only these three names, from one folder, and never a path from the request. Each file is at most 64 MiB. The folder is `CALIPER_WEB_DIR` when it is set. Otherwise it is `web/` in the Resources of the app bundle or beside the executable, or else `python/caliper/_web` of the source tree that holds the build folder, where `scripts/build_wasm.py` puts the files. Without the files, the routes answer 404, and the page shows "No program views" with the reason over each scene panel that holds programs.

A pick in a program panel does not work in the browser yet: the scene of the panel gets the click and reports no pick. When the scene gives a pick, it carries the fields of the item, as the pick event does (program, item, step, fields, index), and the page applies the `select` option of the panel and the pick to its own copy of the state and the selection. As a result, a program whose params read that state or selection gets the new values, and the picked item is the linked selection of the programs of the panel. `state.set` and `selection.set` change the session of the app, which the viewer origin may not do. Thus the selection stays in the page and never reaches the app.

Three parts are not in the browser. A live scene that a scene panel binds comes over the stream, which the viewer token does not open, and the panel shows "Live scene" and "<name> shows in the app". The tooltips and the zoom of plots are also not in the browser.

The live data of a dashboard reaches the page only while the app has that dashboard open: its live sources, the values of its state and controls, and the resources of its programs. The viewer cannot open it, because the viewer origin runs no command that changes the app. When the app does not have the dashboard open, a banner under the bar says: "The app does not have this dashboard open. It shows data when the dashboard is open in the app." Each panel whose source is offline, and each scene panel with programs, shows "Dashboard not open in the app". While the dashboard is open, a source that is offline shows "Source offline".

The page reads through three commands, from the origin `viewer`:

- `spec.get` with `applet` gives the saved spec's canonical JSON, read from its files without opening it, and `open_in_app`, which is true when the app has that applet open. The page asks again once a second.
- `panel.data` with `applet` and `panel` gives that panel's data after its sources' transforms, as the app resolves it, within the limits below.
- `applet.list` gives the list of applets.

`panel.data` of a scene panel gives its programs, and `panel.data` with the id of a control of the layout gives the control. Both are described below. `panel.data` limits what it returns. A line of more than 8,192 points becomes a min/max envelope of 2,048 buckets. A grid or image is sampled to at most 2,048 on a side and 100,000 points. A histogram gives its bin counts, not its values. A table gives at most 1,000 rows, with each text cell cut at 256 characters. A text panel gives its one value.

Each series line has a `name`, which is the name that the app draws the line under. A column has the name of its column, and a window or live source has the name of its source, with `[row]` for a row of a matrix. The series data has an `x_name`.

A control answers `primitive: "control"` with `data` that holds `id`, `kind`, `title`, `bind`, `value`, `text`, `enabled` and `why`, and `min`, `max` and `choices` when the control has them. This is the state that the control shows in the app. A toggle of `intervene:<point>` is on while the run has an ablation of the point that has not ended, and its text is "ablate <point> · since step N".

A scene panel also gives `programs`, one entry for each instance of a piece whose draw half targets the panel. An entry holds `id`, `title`, `piece` (`name`, `version`, `from`), `state`, `error`, `instance` and `params`, and `resources`. With the state `ready`, `instance` is the instance form of PROGRAM as the app sends it to a script: the pin with its hash, the manifest, `bind`, the params with their values of now, the panel, the segments, `common`, `derive` and `draw`. `resources` holds the newest kept version of each published resource at the cursor of the session, as `{resource, step, count, stride, width, height, base64}`. The page of a saved applet that is not the open spec gets no resources. A library, workspace or inline piece comes the same way, and the page never names a file. A program of a foreign spec whose hash nobody reviewed has the state `review` and the error `program.needs_review`, with no files, thus it stays in review in the browser. A missing piece, or a hash other than the pin's, gives the state `refused` with its code and no files. One answer holds at most 64 programs and 32 MiB of resources. The `version` of a scene panel changes with its programs and their newest resources, thus `known` still answers `unchanged` while nothing moves.

Arrays of numbers travel as `{"dtype": "f32" | "f64", "shape", "base64"}`, the little-endian bytes in base64. Tables, cells and counts travel as JSON. With the `version` of an earlier answer as `known`, the answer is `unchanged` and carries no data until the panel's sources, the spec or the catalog change. Thus the page asks once a second for every panel, and the app resolves a panel only when its data moved.

```json
{"id":"api-12","ok":true,"result":{"bounds":[],"data":{"lines":[{"binding":"metrics.loss","count":200,"name":"loss","last":0.12,"row":null,"x":{"base64":"...","dtype":"f64","shape":[200]},"y":{"base64":"...","dtype":"f64","shape":[200]}}],"steps":200,"x_index":false,"x_name":"step"},"error":null,"panel":"curves","primitive":"series","state":null,"unchanged":false,"version":"3f2a9c0b5d17e640"}}
```

A panel whose data cannot be read answers with the resolver's `error`, such as `resolve.unavailable` for a live source that is not attached. A panel whose data does not fit, or holds nothing, answers with a `state`, `mismatch` or `empty`.

## Threads and Limits

A request that reads or changes the app's state waits for the app's frame thread, which takes it between two frames. These requests are the list of applets, a launch, a capture, a command and an approval's state. The 5-second limit applies to the wait before the frame thread takes a command. After that, the request waits for the command's result, and a request held for approval waits as [above](#post-v0commandsname).

| Limit | Value | Answer beyond it |
|---|---|---|
| Time a request waits for the frame thread | 5 s | 504. A launch that is slower than that may still complete |
| Requests waiting for the frame thread at once | 8 | 503 with `Retry-After: 1` |
| Size of a request head | 16 KiB | 431 |
| Size of a command's body | 1 MiB | 413, from the head, without reading the body |
| Time for a command's body to arrive after its head | 10 s | 408 |
| Commands held at once (running, waiting for approval, or waiting for the frame thread) | 32 | 503 with `Retry-After: 1` |
| Time a command from `mcp` or `agent` waits for approval | 120 s | 200 with `command.approval_timeout` |
| Time a held command waits for its result | 600 s | 504. The command may still complete |
| TCP connections, the stream included | 512 | The connection closes |

The stream's own bounds are those of the hub ([Run a hub](../python/hub-setup.md)). When the app exits, requests that still wait get 503.

## Errors

The API answers an error with a JSON object `{"error": "<what went wrong>"}`.

| Status | When |
|---|---|
| 400 | An applet id or approval id is malformed. For a command: `X-Caliper-Origin` is not `viewer` with the viewer token, a header that the checks read is repeated, `Content-Length` is not a number, or the body is not JSON. |
| 401 | The token is missing, wrong or revoked. |
| 403 | The `Host` or `Origin` header is refused, or the viewer token asks for more than the spec page, its three files and its three commands. |
| 404 | No such endpoint or applet, or a command endpoint in an app with no command layer. |
| 405 | The endpoint takes another method. `Allow` names it. |
| 408 | A command's body did not arrive within 10 s of its head. |
| 409 | The applet cannot be launched, or failed to launch. |
| 411 | A command came without `Content-Length`. |
| 413 | A request other than a command has a body, or a command's body is over 1 MiB. |
| 415 | A command's body is not `application/json`. |
| 426 | A GET of `/v3/stream` or `/v0/stream` that is not a WebSocket upgrade. |
| 501 | A capture in the app's window on the GL or Vulkan renderer, or a command sent with `Transfer-Encoding`. |
| 503 | Too many requests wait for the app, 32 commands are held, or the app is closing. |
| 504 | The app did not take the request within 5 s, or a held command had no result within 600 s. |

## Worked Examples

These calls use your session token in `CALIPER_API_TOKEN` and run without review:

```sh
export CALIPER_API_TOKEN=<the session token from the start line>
curl -H "Authorization: Bearer $CALIPER_API_TOKEN" http://127.0.0.1:7341/v0/info
curl -H "Authorization: Bearer $CALIPER_API_TOKEN" http://127.0.0.1:7341/v0/applets
curl -X POST -H "Authorization: Bearer $CALIPER_API_TOKEN" http://127.0.0.1:7341/v0/applets/com.example.my-viewer/launch
curl -o frame.png -H "Authorization: Bearer $CALIPER_API_TOKEN" http://127.0.0.1:7341/v0/capture.png
```

The command layer, with your session token:

```sh
curl -H "Authorization: Bearer $CALIPER_API_TOKEN" http://127.0.0.1:7341/v0/commands
curl -X POST -H "Authorization: Bearer $CALIPER_API_TOKEN" -H "Content-Type: application/json" \
     -d '{"id": "dash", "title": "Dashboard"}' http://127.0.0.1:7341/v0/commands/spec.new
curl -X POST -H "Authorization: Bearer $CALIPER_API_TOKEN" -H "Content-Type: application/json" \
     -d '{"overwrite": true}' http://127.0.0.1:7341/v0/commands/applet.save
curl -H "Authorization: Bearer $CALIPER_API_TOKEN" http://127.0.0.1:7341/v0/approvals/ap-1
```

If the applet is saved already, the third call answers `command.needs_approval` with its approval id, `ap-1` here. The last call reads what the person decided.

An outside agent uses the API file's token. The edit runs as `mcp` and waits for your review, so the call answers after you decide:

```sh
AGENT_TOKEN=$(python3 -c 'import json, pathlib; print(json.loads((pathlib.Path.home() / "Library/Application Support/Caliper/api.json").read_text())["token"])')
curl -X POST -H "Authorization: Bearer $AGENT_TOKEN" -H "Content-Type: application/json" -H "X-Caliper-Group: turn-1" \
     -d '{"id": "note", "primitive": "text", "options": {"text": "Loss per step"}}' \
     http://127.0.0.1:7341/v0/commands/panel.create
```

In Python:

```python
import json, os, pathlib, urllib.request

api = json.loads((pathlib.Path.home() / "Library/Application Support/Caliper/api.json").read_text())
token = os.environ["CALIPER_API_TOKEN"]   # yours, from the start line; api["token"] would be reviewed as mcp
req = urllib.request.Request(api["url"] + "/applets", headers={"Authorization": "Bearer " + token})
print(json.load(urllib.request.urlopen(req)))
```
