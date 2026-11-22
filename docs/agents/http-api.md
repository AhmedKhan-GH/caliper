# HTTP API

Caliper API is an HTTP interface to the running app. Version 0 lists the applets, launches one, takes data streams over [wire protocol v3](../reference/wire-protocol.md), returns a capture of the frame the app draws, runs the commands of the command layer, through which specs, loaders and sources are built with undo, a journal and approvals, and serves the browser viewer of spec applets. It listens on 127.0.0.1 only, behind tokens that also say who sends each command, and it is off unless the app starts with `--api`, or the [side chat or Open in browser starts it](#started-by-the-app). The server is in `src/host/api/api_server.h`, and `src/host/api/app_api.h` starts it for the app and for `caliper shot`.

!!! note "Status"
    Version 0, built and tested on macOS. A capture works in the app's window on Metal and in the windowless mode, `caliper shot --api`; on the GL and Vulkan renderers the app in its window answers a capture with 501. The command endpoints serve the command layer of the app's builder session, in the app, `caliper dev` and `caliper shot` built with `CALIPER_WITH_BUILDER` (`src/host/builder/app_session.h`, over `src/host/api/session_gateway.h`); a build without the builder answers 404 on them, and `caliper-mcp` reads that as an app with no command layer. The Windows build refuses `--api` for now, as [Caliper CLI](../reference/cli.md#on-windows) describes.

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

## Starting it

`--api` turns the API on for the app, for `caliper dev` and for `caliper shot`, and so does the environment variable `CALIPER_API=1`. `--api-port <n>` chooses the port, from 0 to 65535, and needs `--api`; without it the API listens on port 7341, and 0 takes any free port. [Caliper CLI](../reference/cli.md#caliper-api) lists the options.

```sh
caliper --api
caliper dev build --api --api-port 0
caliper shot --applets build --launch com.example.my-viewer --frames 216000 --out page.png --api
```

When it listens, the process prints the address and **your session token**, and writes the address and a token for outside agents in the **API file**:

```text
[api] Caliper API on http://127.0.0.1:7341/v0 with the session token <token> (agents read <API file>, whose commands are reviewed)
[api] Spec applets in a browser at http://127.0.0.1:7341/v0/spec?token=<viewer token>&applet=<id>
```

The session token on the first line is yours: commands sent with it run as the origin `api`, without review, and on the stream it may control a run when the app allows control. Caliper writes it to no file, so a program you run that should act without review takes it from this line, for example through an environment variable that you set, such as `CALIPER_API_TOKEN` in the [examples](#examples). The second line is the link of the [spec page](#get-v0spec), with the viewer token, which opens that page and nothing else.

**Commands sent with the API file's token are reviewed.** The API file is what an outside agent reads, through [`caliper-mcp`](mcp-server.md) or otherwise, and its token runs commands as the origin `mcp`: each change to the open spec waits for your review in the app, and each command that deletes or overwrites saved data waits for your approval, whatever the request claims ([Who sends a command](#who-sends-a-command)).

The API file is `api.json` in the app-data folder ([Folders and environment variables](../reference/folders-and-environment.md)). The process writes it with mode 0600, so that only its owner can read it, and removes it when it exits, as `caliper serve` does with its hub file. It holds one JSON object:

| Key | Value |
|---|---|
| `version` | 1 |
| `pid` | The process id |
| `port` | The port the API listens on |
| `token` | The token of outside agents, whose commands run as `mcp` and are reviewed; not your session token. On the stream it watches, and cannot publish a session or control a run |
| `url` | `http://127.0.0.1:<port>/v0` |
| `stream_url` | `ws://127.0.0.1:<port>/v3/stream`, without the token |
| `spec_url` | `http://127.0.0.1:<port>/v0/spec?token=<viewer token>`, the spec page; a saved spec applet opens at this address with `&applet=<id>` added |

Every token is new for each run: your session token, the API file's token, the viewer token and each agent's own. Each has the same form: 32 characters of `A-Z`, `a-z`, `0-9`, `_` and `-` from 24 random bytes. When the port is taken, the app starts without the API and says so on standard error and in the Activity pane, and `caliper shot` exits with 1.

While the API runs, every page's status bar shows its address, such as `api 127.0.0.1:7341`.

### Started by the app

Two things in the app start the API when the app was started without `--api`: the [side chat](side-chat.md), whose agent drives Caliper through `caliper-mcp`, and ⌘O Open in browser on the spec page ([Builder pane](../builder/builder-pane.md#open-in-browser)), which opens a saved dashboard in the browser viewer. Either way the API follows the same rules as with `--api`: it listens on 127.0.0.1 only, at a free port, behind new tokens, and it writes the API file. No setting can make it listen on another address or a chosen port. The start is said in Activity, with the address and never a token, and on standard error, and the status bar shows the address. An API that the person started with `--api` is left alone in both directions: neither starts a second one nor stops it.

**For the side chat.** The side chat starts the API when its agent starts, at the person's first message. The side chat's agent does not read `api.json`. Each agent session gets an API file of its own, `api-agent-<pid>-<n>.json` in the app-data folder, with the same keys and mode 0600, whose token runs commands as the origin `agent`, so the agent's changes to the open spec wait for your review. The app revokes that token and removes the file when the agent's session ends, and a restarted agent gets a new one. The widget agent's sidecar gets its own file in the same way, for as long as its target is open.

The start is visible in three places: a line in the conversation, "Caliper API started on 127.0.0.1:<port> for this chat"; an Activity line; and the status bar. The API is tied to the chat: when the chat's agent ends (it exits, it cannot start, or the person chooses another agent), the side chat stops the API it started, and the app removes the API file, unless Open in browser has given out a link of that API since, which keeps it running until the app quits.

**For Open in browser.** ⌘O starts the API if it does not run, then opens the dashboard's link in the default browser:

```text
http://127.0.0.1:<port>/v0/spec?token=<viewer token>&applet=<id>
```

The link carries the viewer token, never your session token or the API file's token; the app checks that before it opens the link, and refuses otherwise. What can refuse the link is asked before the API starts: a build without the browser viewer, and a dashboard id that is not a workspace's applet id (a lower-case letter, then at most 63 lower-case letters, digits, `_` or `-`). When the link still cannot be handed over, as when the browser does not start, an API started for this request alone stops again, and its API file goes. The browser is started with the standard descriptors only, each on `/dev/null`, so no descriptor of the app reaches it. Anyone who has the link can read, from the origin `viewer`, what the [spec page](#get-v0spec) reads: the list of applets (`applet.list`), the saved spec of every spec applet of the open workspace (`spec.get`), not only the one opened, and the data of their panels as the panels show it (`panel.data`, within the bounds listed there). The link opens nothing else: no other endpoint, no command that changes anything and not the stream. Since the API listens on loopback, the link works only on this machine, but there any program or person that has it can read those, and a browser may keep it in its history.

The viewer token ends when the API that made it stops. An API started for Open in browser runs until the app quits, since a page open in a browser keeps reading; an API started with `--api` also stops when the app quits, and one started for the side chat when the chat's agent ends, unless a link was given out. The next start makes a new viewer token, so an old link is then answered with 401.

A hidden app opens no browser. A test gives it `CALIPER_TEST_OPEN_URL`, a file to which the app appends the link instead ([Host variables](../reference/folders-and-environment.md#host-variables)); without it, the hidden app refuses, says so on standard error and in Activity, and starts no API.

## The windowless mode

`caliper shot` draws the app's applet page into an offscreen canvas and shows nothing on screen ([caliper shot](../reference/cli.md#caliper-shot)). With `--api`, it serves the API while it draws its frames, so it is the app without a window: `--frames` sets how long it runs, at about 60 frames a second, and SIGINT (Ctrl-C) or SIGTERM ends the frames early. Either way it then writes its image of the applet it shows, as usual, stops the API, removes the API file and exits with 0. The API file goes to the app-data folder also when `shot` keeps its stores in a temporary folder.

A capture is read from the offscreen canvas at the end of a frame, with the read-back that `caliper shot` uses for its image.

## Security

Every request gets the checks of the stream ([The upgrade request](../reference/wire-protocol.md#the-upgrade-request)):

- The server listens on 127.0.0.1 only.
- The `Host` header must be `127.0.0.1:<port>` or `localhost:<port>`, with the API's port. Anything else is refused with 403, as a guard against DNS rebinding.
- An `Origin` header, when the request has one, must be one that the protocol accepts: `null`, an `http` or `https` origin on `127.0.0.1`, `localhost` or `[::1]`, or `vscode-webview://`. Any other origin is refused with 403.
- A token must come with the request: your session token, the API file's token, an agent's own token, or for the spec page the viewer token. A request carries it in an `Authorization: Bearer <token>` header. A GET may carry it as the query parameter `token` instead, which is how the WebSocket and a link opened in a browser carry it. A POST takes the header only, so a state-changing request never has the token in its address. The token is compared in constant time with every token the API holds. A missing, wrong or revoked token is refused with 401 and `WWW-Authenticate: Bearer`, and on the stream with 403, as the protocol says; the viewer token does not open the stream.

The API serves nothing beyond its endpoints: no files and no way to run code. The spec page is one page built into the executable, not a file it reads. The commands of the command layer change specs, loaders and sources, which are data; a command that deletes or overwrites saved data, or runs a Python loader, waits for the person's approval in the app, and no endpoint approves or denies. Only `POST /v0/commands/{name}` takes a body, and only after its head passed the checks above; any other request with a body is refused with 413, and launch takes only an applet id in its path that the app's loader lists. A command request is built from its path, its body, the origin of its token and the headers `X-Caliper-Group` and `X-Caliper-Preview` only, so no other header value reaches the command layer or its journal, and the token is never logged. Every response carries `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`, and the connection closes after it.

**The viewer token.** The link of the spec page carries the viewer token, never the session token, because a link is easily copied and kept in a browser's history. A request with the viewer token may only `GET /v0/spec` and `POST /v0/commands/{name}` for the three commands the page reads with, `spec.get`, `applet.list` and `panel.data`; everything else, the other endpoints, the other commands and the stream, is refused with 403. Its commands run from the origin `viewer` (an `X-Caliper-Origin` other than `viewer` is refused with 400), and the command layer runs only commands that change nothing from that origin, refusing any other with `command.not_allowed`.

### Who sends a command

The origin of a command, which decides what is reviewed, comes from the token that the request carries, never from a header:

| Token | Where it is | Origin | Review | The stream |
|---|---|---|---|---|
| Your session token | The start line only | `api` | None for spec edits; a command that deletes or overwrites saved data answers `command.needs_approval` at once | Watches, streams a run in, and controls a run when the app allows control |
| The API file's token | `api.json` | `mcp` | Spec edits wait for your review, and destructive commands for your approval | Watches; cannot publish a session or control a run |
| An agent's own token | `api-agent-<pid>-<n>.json`, while that agent's session runs | `agent` | As `mcp` | Watches; cannot publish a session or control a run |
| The viewer token | The spec page's link | `viewer` | Reads only | Refused |

`X-Caliper-Origin` is ignored for every token but the viewer token, so a request cannot claim `api`, `ui` or any other origin that its token does not have.

**Agent permissions.** The side chat's agent and the clients of the API file's token each have a mode that you set in the app: Ask (the review above), Accept edits (an edit of the open spec applies at once, still journaled and undoable), Read only (reads, dry runs and edits of a preview run; every other change answers `command.denied` with a message that says so), and for the side chat's agent only, Allow all for its session, which is never saved. A card can also allow its command for the session or always. The app tells the side chat's agent from the widget agent's sidecar by the token it issued each of them, never by anything a request carries; the widget agent, your own session token and the viewer token are not governed by a mode. No endpoint and no command reads or sets a mode or a rule: only the app's own interface does. An overwrite of a saved applet and the replacement of a spec with unsaved changes still wait for your card in every mode, and a rule never lets a Python loader run.

## Endpoints

### GET /v0/info

What `caliper info` prints, as JSON. `sdk`, `libtorch` and `ui_stack` are `null` when they are unknown or absent.

```json
{"abi_epoch":2,"host":"0.6.0","libtorch":"2.5.1","sdk":"0.2.0","services":["caliper.artifacts.v1","caliper.data.v1","caliper.log.v1","caliper.ui.v1"],"ui_stack":"03c557182adc5cd9"}
```

The list of services is shortened here.

### GET /v0/applets

What `caliper list` prints for the applets the running app found, in the same order: the search order, then the id, then the highest version first. `status` is the word that `caliper list` prints, such as `READY`, `ACTIVE` for the applet whose page is open, `REFUSED` or `SHADOWED`, `kind` is `library` for a compiled applet or `spec` for a [spec applet](../applets/manifest.md#spec-applets), and `reason` is the text that `caliper list` prints under it, empty for a `READY` applet.

```json
{"applets":[{"folder":"/path/to/build/applets","id":"com.example.my-viewer","kind":"library","reason":"","status":"ACTIVE","version":"0.1.0"}]}
```

### POST /v0/applets/{id}/launch

Opens the page of the applet with that manifest id, as a click on its row on the landing screen does. For a [spec applet](../applets/manifest.md#spec-applets) it is the spec page ([Caliper CLI](../reference/cli.md#the-spec-page)): the answer comes once the spec has been validated, and the page opens with the session's `applet.open` a frame or two later. That `applet.open` is sent from the origin of the request's token: `api` for your own token, `mcp` for the API file's, `agent` for an agent's, as the session's `app.launch` sends it from its own origin. So when the open spec has unsaved changes, a launch with an agent's token waits on an approval card in the app, and the page opens only once the person approves. The answer says so: it comes at once, with `status` `PENDING_APPROVAL` instead of `ACTIVE`, `already_open` false and a `note` that says the open waits for the person's approval and that `spec.get` names the open spec once it has opened (a denial leaves the open spec as it was). The app decides this from the state when it answers, which is the command layer's own condition: review on for the token's origin and an open spec with unsaved changes. When it cannot read the open spec, because a long command runs on its worker, the status is `PENDING` and the note says the open waits for that command and then, if the spec has unsaved changes by then, for the person. The answer carries no approval id: the card's approval belongs to the `applet.open` that the launch sends a frame or two later, so the launch cannot know it when it answers. Every other launch answers `ACTIVE` as before, as a launch with your own token does, which opens at once (⌘Z in the app brings the unsaved spec back).

```json
{"already_open":false,"id":"local.attention-entropy","note":"Accepted, not open yet: opening it would replace the open spec's unsaved changes, so it waits for the person's approval in the app. spec.get names the open spec once it has opened; if the person denies it, the open spec stays as it is.","status":"PENDING_APPROVAL","version":"0.1.0"}
``` The page shown before goes first, as Home takes it: an open applet is torn down, and an attached live session stays attached. When the new applet fails to launch, the app stays on the landing screen with the reason in the Activity pane, and the windowless mode launches the applet it showed before again.

```json
{"already_open":false,"id":"com.example.my-viewer","status":"ACTIVE","version":"0.1.0"}
```

| Status | When |
|---|---|
| 200 | The applet's page is open. `already_open` is true when it was open before the request. |
| 400 | The id is not 1 to 128 characters of `A-Z`, `a-z`, `0-9`, `.`, `_` and `-`. |
| 404 | No folder in the search order holds an applet with that id. |
| 409 | The applet cannot be launched, with the status and reason that `caliper list` shows, or it failed to launch, with the loader's reason. |

### GET /v0/capture.png

The frame the app draws, as an opaque PNG, read back at the end of the next frame without drawing an extra one:

- In the app's window, the PNG is the frame the window presents, at the window's framebuffer size in pixels (twice the window's size in points on a Retina display). This works with the window hidden too ([`--hidden`](../reference/cli.md#the-hidden-window)), which is how tests and agents drive the whole app with nothing on screen. The renderer copies that frame on the GPU as it presents it, and a later frame picks the copy up, so the frame loop never waits for it. This works on Metal. On the GL and Vulkan renderers the window's frame cannot be read back yet, and a capture is answered with 501 and the reason.
- In the windowless mode, the PNG is the offscreen canvas at the end of a frame, of the size `caliper shot --size` gives, 1600 by 1000 by default.

A capture waits for the app's frames like every call that needs the frame thread, so it is answered with 504 when no frame is drawn within 5 seconds. The PNG is encoded on a thread of its own, so the frame loop does not wait for the encoder either.

### GET /v3/stream

The WebSocket of [wire protocol v3](../reference/wire-protocol.md), with the app in the hub role. It is the hub of [`caliper serve`](../python/hub-setup.md) on the API's port: producers connect to it and each becomes a session, and viewers list the sessions and attach to one. Everything the protocol states for a hub applies, with its bounds and its flow control. It is served at the protocol's own path, so every client of the protocol attaches to it unchanged: the app's NetSource, `caliper-viewer --url`, the link field of the app's attach pane and the `<caliper-view>` element. The upgrade is checked for its path, `Host`, `Origin` and the `token` parameter in the protocol's order, so a client gives the token in the address. Your session token, the API file's token and an agent's own token open it, and the viewer token does not. A connection opened with the API file's token or an agent's watches: it gets the data and the controls' state, its WELCOME sets `control` false, and a CONTROL from it is answered `forbidden`, so an agent cannot pause, step, change a parameter of or run an action of a training run. It cannot publish either: a producer's HELLO on it is answered with the error `unsupported` and closed with code 4002, as `caliper.serve()` refuses producers. Only a connection opened with your session token may control, and only as the next paragraph says:

```text
ws://127.0.0.1:7341/v3/stream?token=<token>
```

`GET /v0/stream` is an alias of it, with the same checks, for clients that keep every address under the API's prefix. The stream carries data, viewer events and, from protocol 3.1, controls, never host commands. As for `caliper serve`, the hub forwards no control unless the app was started with `--allow-control` next to `--api`; without it every control is answered `forbidden` ([Controls and selections](../python/hub-setup.md#controls-and-selections)). The app's live page is itself a viewer of protocol 3.1: attached to this stream, or to any other, it shows the session's run bar and selections, and its controls follow the same rule ([Control the run and share a selection](../python/script-attachment.md#control-and-selection-sharing)).

### GET /v0/commands

The command layer's descriptor table: every command with its summary, the JSON Schema of its arguments and of its result, examples, whether it mutates, is destructive, runs on the frame thread or may be sent by an agent (`agents`), and what it does (`effect`: `read` for a read or a proposal, `edit` for an edit of the open spec, `change` for anything else that changes something), by which an agent's mode decides it. It is the table that `caliper-mcp` and `caliper-agent` turn into tools. Requested with the API file's token or an agent's own, it leaves out the commands with `agents` false, `history.undo` and `history.redo`, which are yours: the command layer refuses them to those tokens with `command.not_allowed`, and an agent reverts a change with an edit that you review.

```json
{"commands":[{"agents":true,"args":{"additionalProperties":false,"properties":{},"required":[],"type":"object"},"destructive":false,"effect":"read","examples":[{}],"frame_thread":false,"mutates":false,"name":"applet.list","result":{"...":"..."},"summary":"..."}],"version":0}
```

The list is shortened here.

### POST /v0/commands/{name}

Runs the command `{name}`, such as `spec.new` or `panel.patch`. The body is the command's arguments, a JSON object of at most 1 MiB, with `Content-Type: application/json` and a `Content-Length`. Three headers shape the invocation:

| Header | Meaning |
|---|---|
| `X-Caliper-Origin` | Ignored: the origin is the token's ([Who sends a command](#who-sends-a-command)). With the viewer token it must be `viewer` when it is given. |
| `X-Caliper-Group` | Optional. Commands that share a group undo as one step. |
| `X-Caliper-Preview` | Optional. The id of a stacked preview, from `preview.begin`: an edit of the open spec applies to the preview instead, and `spec.get` reads it. |

The answer is the command layer's result envelope, with 200 whether the command succeeded or not. The server gives each invocation an id, `api-<n>`.

The commands act on the builder session's workspace, the folder of `--workspace` or `CALIPER_WORKSPACE`, else `workspace/` in the app-data folder ([Caliper CLI](../reference/cli.md#the-workspace)). The default workspace is opened by the first command that arrives, so the first call may take a few milliseconds longer; a command that arrives while the workspace is still opening waits for it, and one that arrives when it cannot be opened answers `command.not_available` with the reason. The table of `GET /v0/commands` needs no workspace. When `applet.open` succeeds, from any origin, the app shows the spec page with that applet's spec, as Caliper CLI describes under [The workspace](../reference/cli.md#the-workspace). An agent's first look is a read too when no spec is open: `spec.get` then answers `{"open": false}` and nothing else, not an error, and with a spec open its result carries `"open": true` with the spec.

```json
{"id":"api-7","ok":true,"result":{"applied":true,"diff":"...","id":"note","warnings":[]}}
{"id":"api-8","ok":false,"error":{"code":"spec.no_open_spec","message":"...","where":""}}
```

Everything the command layer refuses comes this way: a name that is not a command (`command.unknown`, so that a 404 always means that the app has no command layer), arguments that fail the command's schema or a body that is JSON but not an object or nests deeper than 64 levels (`command.invalid_args`; a null on an optional argument whose schema does not admit null counts as the argument left out, and the journal and the approval card show the call without it), a malformed group or preview (`command.invalid_invocation`), and every error of the command itself. Faults of the request keep their HTTP status, as the table under [Errors](#errors) lists: a body that is not JSON is 400, for example.

**Approval.** A command that deletes or overwrites saved data, or runs a Python loader, needs the person's approval in the app when an agent or a program sends it. For the origins `mcp` and `agent`, the request is held while the person decides, for up to 120 seconds, and then answers with the command's result when it was approved, with `command.denied` when it was denied, and with `command.approval_timeout` when nobody decided in time, after which the approval is withdrawn. A client allows at least 130 seconds for such a call; `caliper-mcp` does. For the origin `api`, your session token's, the request answers at once with `command.needs_approval`, whose error carries the approval id in `approval`, and the program reads the outcome later from `GET /v0/approvals/{id}`:

```json
{"id":"api-9","ok":false,"error":{"approval":"ap-3","code":"command.needs_approval","message":"...","where":""}}
```

A held request waits without blocking anything else: the stream, the other endpoints and other commands are served meanwhile. Spec edits from `mcp` and `agent` are reviewed by default and therefore wait too.

The app shows each pending approval to the person as a card, in a strip at the bottom of the window above the jobs tray, on every page: who asks (MCP agent, Side chat, Widget agent or API), the command and its target, the change of a spec edit as its changed keys with the validator's warnings and the full diff on demand, what a destructive command would lose, the time left before a held request times out, and Approve and Deny. A card of the side chat's agent or of the API file's token whose command can become a rule offers Allow once, Allow always for this session and Always allow `<command>` instead of Approve. On the spec page the panel an edit changes is marked REVIEW. The app's Activity lists each request when it starts waiting and when it is decided, times out or is withdrawn. When what a card showed changes while a held request waits, such as the content an overwrite would write, an approval no longer holds: the card is replaced by a new one for the new content, and the request keeps waiting on it. Only the person decides there; a test of the hidden app decides through `CALIPER_TEST_APPROVALS` ([Host variables](../reference/folders-and-environment.md#host-variables)).

### GET /v0/approvals/{id}

What became of the command that waits, or waited, under approval `{id}`. `state` is `pending` while the person has not decided, `decided` with the command's result envelope in `result` once they have (the last 64 decisions are kept), and `unknown` for an id the command layer does not know or no longer keeps. A pending approval also gives what the person is asked about: the command's `name`, the `origin` that sent it, and its `review`, which for an edit of the open spec under review is the unified diff of the spec's canonical text and the validator's warnings, `{"diff", "warnings"}`, and is null for other commands. While a loader runs on the app's worker, the app answers from the pending approvals as they stood before it started, so a lookup never waits for the loader.

```json
{"approval":"ap-2","name":"panel.create","origin":"mcp","result":null,"review":{"diff":"--- run42-dashboard.spec.toml\n+++ ...","warnings":[]},"state":"pending"}
{"approval":"ap-3","name":"applet.save","origin":"api","result":null,"review":null,"state":"pending"}
{"approval":"ap-3","result":{"id":"api-9","ok":true,"result":{"id":"dash","manifest":"...","spec":"..."}},"state":"decided"}
```

An id is 1 to 128 characters of `A-Z`, `a-z`, `0-9`, `.`, `_`, `:` and `-`; anything else is refused with 400. Approval ids are not secret: reading one changes nothing, and only the person decides.

### GET /v0/spec

The browser viewer of spec applets: one HTML page, with its script and its two faces inside it, which draws a saved spec applet of the open workspace with the layout and the look of the app's spec page. It takes the token and the applet's workspace id from its own address, `/v0/spec?token=<token>&applet=<id>`, as the printed link and `spec_url` give it; without `applet` it lists the saved spec applets as links. The page is the web package's (`web/src/spec/`), built into the executable from its checked-in copy, `python/caliper/_web/spec.html`, and answers with `Content-Security-Policy` that lets it reach only its own origin, and `Referrer-Policy: no-referrer`, so the token in its address is never sent elsewhere.

The page reads through the command layer, from the origin `viewer`:

- `spec.get` with `applet` gives the saved spec's canonical JSON, read from its files without opening it. The page asks again once a second and lays the panels out again when the text changed.
- `panel.data` with `applet` and `panel` gives that panel's data after its sources' transforms, as the app resolves it, within the bounds of what a panel shows: a line of more than 8,192 points as a min/max envelope of 2,048 buckets, a grid or image sampled at a stride to at most 2,048 on a side, at most 100,000 points, a histogram's bin counts rather than its values, at most 1,000 rows of a table with each text cell cut at 256 characters, and a text panel's one value. Arrays of numbers travel as `{"dtype": "f32" | "f64", "shape", "base64"}`, the little-endian bytes in base64; tables, cells and counts as JSON. With the `version` of an earlier answer as `known`, the answer is `unchanged` and carries no data until the panel's sources, the spec or the catalog change, so the page asks once a second for every panel it shows and the app resolves a panel only when its data moved.
- `applet.list`, for the list of applets.

```json
{"id":"api-12","ok":true,"result":{"bounds":[],"data":{"lines":[{"binding":"metrics.loss","count":200,"last":0.12,"row":null,"x":{"base64":"...","dtype":"f64","shape":[200]},"y":{"base64":"...","dtype":"f64","shape":[200]}}],"steps":200,"x_index":false},"error":null,"panel":"curves","primitive":"series","state":null,"unchanged":false,"version":"3f2a9c0b5d17e640"}}
```

The example is shortened. A panel whose data cannot be read answers with the resolver's `error`, such as `resolve.unavailable` for a live source that is not attached, and one whose data does not fit or holds nothing with a `state`, `mismatch` or `empty`; the page draws the same states as the app.

## Threads and limits

One thread runs the hub's loop and answers every request; it never waits for the app. A request that reads or changes the app's state, which is the list of applets, a launch, a capture, a command or an approval's state, waits for the app's frame thread, which takes it at the start of a frame, between two frames. A launch runs there, as `CALIPER_AUTOLAUNCH` runs before the first frame. A command is handed to the command layer there, and runs when the frame thread drains the command layer's queue; from then on the request is held, without a thread of its own, until the command layer answers it, which the frame thread and the hub's thread both look for without waiting. The 5-second limit therefore applies to the wait before the frame thread takes a command; once taken, the request waits for its result, and one held for approval waits as [above](#post-v0commandsname).

| Limit | Value | Answer beyond it |
|---|---|---|
| Time a request waits for the frame thread | 5 s | 504; a launch that is slower than that may still complete |
| Requests waiting for the frame thread at once | 8 | 503 with `Retry-After: 1` |
| Size of a request head | 16 KiB | 431 |
| Size of a command's body | 1 MiB | 413, from the head, without reading the body |
| Time for a command's body to arrive after its head | 10 s | 408 |
| Commands held at once (running, waiting for approval, or waiting for the frame thread) | 32 | 503 with `Retry-After: 1` |
| Time a command from `mcp` or `agent` waits for approval | 120 s | 200 with `command.approval_timeout` |
| Time a held command waits for its result | 600 s | 504; the command may still complete |
| TCP connections, the stream included | 512 | The connection is closed |

The stream's own bounds are those of the hub; see [Run a hub](../python/hub-setup.md). When the app exits, requests still waiting get 503.

## Errors

An error is answered with a JSON object `{"error": "<what went wrong>"}`.

| Status | When |
|---|---|
| 400 | A launch names something that cannot be an applet id, or an approval lookup something that cannot be an approval id. For a command: with the viewer token, `X-Caliper-Origin` is given and is not `viewer`; a header that the checks read (`Authorization`, `Content-Length`, `Content-Type`, `X-Caliper-Group`, `X-Caliper-Preview`) is repeated; `Content-Length` is not a number, or the body is not JSON. |
| 401 | The token is missing, wrong or revoked. |
| 403 | The `Host` or the `Origin` header is refused, or a request with the viewer token asks for anything but the spec page and its three commands. |
| 404 | No such endpoint, no applet with that id, or a command endpoint in an app that has no command layer. |
| 405 | The endpoint takes another method; `Allow` names it. |
| 408 | A command's body did not arrive within 10 s of its head. |
| 409 | The applet cannot be launched, or failed to launch. |
| 411 | A command came without `Content-Length`. |
| 413 | A request other than a command has a body, or a command's body is over 1 MiB. |
| 415 | A command's body is not `application/json`. |
| 426 | A GET of `/v3/stream` or `/v0/stream` that is not a WebSocket upgrade. |
| 501 | A capture in the app's window on the GL or Vulkan renderer, which cannot read the window's frame back, or a command sent with `Transfer-Encoding`. |
| 503 | Too many requests wait for the app, 32 commands are held, or the app is closing. |
| 504 | The app did not take the request within 5 s, or a held command had no result within 600 s. |

## Examples

With your session token in `CALIPER_API_TOKEN`, copied from the line the app printed, so that these calls run as you, without review:

```sh
export CALIPER_API_TOKEN=<the session token from the start line>
curl -H "Authorization: Bearer $CALIPER_API_TOKEN" http://127.0.0.1:7341/v0/info
curl -H "Authorization: Bearer $CALIPER_API_TOKEN" http://127.0.0.1:7341/v0/applets
curl -X POST -H "Authorization: Bearer $CALIPER_API_TOKEN" http://127.0.0.1:7341/v0/applets/com.example.my-viewer/launch
curl -o frame.png -H "Authorization: Bearer $CALIPER_API_TOKEN" http://127.0.0.1:7341/v0/capture.png
```

The command layer from your own program, with your session token:

```sh
curl -H "Authorization: Bearer $CALIPER_API_TOKEN" http://127.0.0.1:7341/v0/commands
curl -X POST -H "Authorization: Bearer $CALIPER_API_TOKEN" -H "Content-Type: application/json" \
     -d '{"id": "dash", "title": "Dashboard"}' http://127.0.0.1:7341/v0/commands/spec.new
curl -X POST -H "Authorization: Bearer $CALIPER_API_TOKEN" -H "Content-Type: application/json" \
     -d '{"overwrite": true}' http://127.0.0.1:7341/v0/commands/applet.save
curl -H "Authorization: Bearer $CALIPER_API_TOKEN" http://127.0.0.1:7341/v0/approvals/ap-1
```

The third call answers `command.needs_approval` with its approval id, `ap-1` here, when the applet is saved already; the last one reads what the person decided.

As an outside agent calls it, with the API file's token: the edit runs as `mcp` and waits for your review in the app, so the call answers once you have approved or denied it.

```sh
AGENT_TOKEN=$(python3 -c 'import json, pathlib; print(json.loads((pathlib.Path.home() / "Library/Application Support/Caliper/api.json").read_text())["token"])')
curl -X POST -H "Authorization: Bearer $AGENT_TOKEN" -H "Content-Type: application/json" -H "X-Caliper-Group: turn-1" \
     -d '{"id": "note", "primitive": "text", "options": {"text": "Loss per step"}}' \
     http://127.0.0.1:7341/v0/commands/panel.create
```

In Python, the address can be read from the API file, and your session token from the environment:

```python
import json, os, pathlib, urllib.request

api = json.loads((pathlib.Path.home() / "Library/Application Support/Caliper/api.json").read_text())
token = os.environ["CALIPER_API_TOKEN"]   # yours, from the start line; api["token"] would be reviewed as mcp
req = urllib.request.Request(api["url"] + "/applets", headers={"Authorization": "Bearer " + token})
print(json.load(urllib.request.urlopen(req)))
```
