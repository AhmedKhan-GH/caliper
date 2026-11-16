# Builder overview

This page is the map of the builder: the work that lets someone drop data into Caliper, make loaders and build visuals from inside the app. A person builds by hand in a WYSIWYG builder pane, with their own agent in the side chat, with the widget agent on one panel, selection, dropped file or property, or from an agent outside the app over MCP. Everything that is built is data, a spec or a loader, and never code that runs in the app's process. The page says which parts are built, which wait, and where each part is described. The [roadmap](roadmap.md#phase-3-agent-built-panels) gives the order and status.

[Applet builder](applet-builder.md) is the earlier proposal that this work follows. It left a WYSIWYG editor and an agent inside the app for later; both are now part of the plan.

## The parts

```
 Builder pane        Side chat              Widget agent           Outside agent
 (WYSIWYG)           the person's own       caliper-agent          (Claude Code, Codex)
     │               ACP agent              ⌘K on one target           │
     │                   │ MCP                   │                     │ MCP on stdio
     │                   ▼                       │                     ▼
     │               caliper-mcp                 │                 caliper-mcp
     │                   └───────────┬───────────┘─────────────────────┘
     │                               ▼  Caliper API on loopback, session token
     └──────────────────────► Command layer  (typed, journaled, undoable, with approvals and previews)
                                     │
        ┌──────────────┬─────────────┼──────────────┬───────────────┐
        ▼              ▼             ▼              ▼               ▼
     Loaders        Catalog        Spec         Session          The app
     files to       every          sources,     binds them       capture, launch,
     sources        source         panels,      for one          live sources,
                                   layout       workspace        the spec runtime
```

| Part | What it is | Page |
|---|---|---|
| Spec format | A versioned TOML file of sources, panels and a layout, with a canonical JSON form and a JSON Schema, designed to run unchanged in the web viewer | [Spec format](spec-format.md) |
| Loaders | Saved, declarative recipes that turn dropped files into sources: a file or glob, a reader, and SQL or tensor transforms | [Loaders](loaders.md) |
| Catalog | The list of every source with its schema, shape and rate | [Catalog](catalog.md) |
| Command layer | The one surface everything acts through: a registry of typed commands, undo and redo, a journal and approvals | [Command layer](command-layer.md) |
| Session | The headless object that binds the four libraries for one workspace, and the stand-in host for tests | [Builder session](builder-session.md) |
| `caliper-mcp` | A stdio MCP server that an outside agent launches; its tools mirror the commands | [MCP server](mcp-server.md) |
| Side chat | The general chat pane: Caliper runs the person's own ACP agent as a subprocess, hands it `caliper-mcp`, and shows its messages, diffs and permission requests | [ACP integration](acp-integration.md) |
| Widget agent | An inline agent on one target: Caliper gives it a context bundle and a tool scope, and it proposes a change that is drawn in place as a preview | [Widget agent](widget-agent.md) |
| `caliper-agent` | The widget agent's process: a sidecar whose agent loop runs on Pydantic AI, with Claude through the person's API key on its native API and every other model, hosted or local, through an OpenAI-compatible API | [Widget agent sidecar](byok-agent.md) |

The side chat and the widget agent are two roles, not two interchangeable backends, and Agent roles (`docs/explanations/agent-roles.md`) explains when to use each. Both are ACP agents that the app drives with one `AcpClient`, so messages, diffs, approvals and cancellation work the same for both, and "Ask in chat" hands a widget request to the side chat with its context.

## What is built

The parts above are built as libraries and tools under `src/builder/`, each with unit tests, and the session is wired into the app (item 2 below); the others are not yet. They build with the host: the root build adds `src/builder/` when `CALIPER_WITH_BUILDER` is on, which is the default except on Windows, and ctest runs the `builder_*` tests beside the host's, as Testing (`docs/explanations/testing.md`) describes. With the builder on, the host's static DuckDB also links its json extension, which the loaders' JSON and JSONL readers need; DuckDB loads it in every database the process opens, and it adds about 0.7 MB to the `caliper` executable. `src/builder/` still configures on its own against a DuckDB release, without the host build.

- The spec library parses, validates and writes specs, and applies merge patches.
- The loaders library reads CSV, TSV, Parquet, JSON and JSONL through DuckDB, and `.npy`, `.npz`, safetensors, PNG, JPEG and WAV natively. It refuses pickle-based files, and it runs a Python loader out of process for other formats, which loads torch checkpoints weights-only with a warning.
- The command layer has its registry, its schema validator, undo with groups and merging, its journal with replay, its approvals and its queue, follow-up invocations that run as one undo step, and a table of 40 commands, which is also checked in as JSON.
- The session binds every command that needs no window. A headless test drops a Parquet file and a safetensors file, runs their loaders, builds a three-panel dashboard command by command, undoes and redoes, saves it as a spec applet, reopens it and replays the journal.
- The session enforces the review of spec edits for every agent: with review on for an origin, `agent` and `mcp` by default, an edit that would change the open spec waits for the person's approval with its diff and the validator's warnings, whatever the agent does, and `review.status` tells an agent what Caliper asks itself, so that `caliper-agent` does not ask twice.
- The session has stacked previews: edits that name a preview apply to a copy of the open spec and stay out of the journal and the history, the app can resolve a preview's panel to draw it in place, and `preview.commit`, which only the person's origins may send, applies them as one undo step.
- `AcpClient::new_session()` takes an optional `_meta`, passed through to `session/new`, for the widget session's target, tool scope and preview.
- `caliper-mcp` works against Caliper API v0 of the running app for the app's information, its applets, a launch and a capture, and against the command endpoints for everything else, both the stand-in host's and those of the app's API server.
- The app's API server serves the command endpoints, `GET /v0/commands`, `POST /v0/commands/{name}` and `GET /v0/approvals/{id}`, through a `CommandGateway` (`src/host/api/command_gateway.h`), with approvals held for `mcp` and `agent` while the person decides, and the gateway over a `Session` is `caliper_api_session` (`src/host/api/session_gateway.h`). A test runs a real session behind the server and `caliper-mcp` against it. The app sets the gateway once it owns a session (item 2 below); until then the endpoints answer 404.
- The app, `caliper dev` and `caliper shot` own a session for their workspace: the folder of `--workspace`, else of `CALIPER_WORKSPACE`, else `workspace/` under the app-data folder ([Builder session](builder-session.md#the-session-in-the-app)). A chosen workspace opens on a worker after the first frame, and the default one only when something first needs it. The frame thread drains the session's queue between frames within a budget of 4 ms; `data.import`, `data.query` and loader runs go to the host's job system while the frame thread leaves the registry and the store alone; `app.info`, `app.applets`, `app.launch` and `capture.frame` are served by the app, and `panel.render` answers `command.not_available` until the spec runtime implements its seam. Each journaled command is an Activity line with its origin, and the session closes its journal and database before the host's stores. A command script (`CALIPER_COMMAND_SCRIPT`) drives it in tests and demos until the API's command endpoints are connected.
- `caliper-mcp` works against Caliper API v0 of the running app for the app's information, its applets, a launch and a capture, and against the command endpoints of the stand-in host for everything else.
- The ACP client drives a session against a stub agent, and `caliper-acp-harness` runs one session headless.
- The first slice of the spec runtime, in `src/host/spec_runtime/`, draws a spec's panels natively in the house look from the session's resolved data: the layout as rectangles with minimums that grow with the text scale, the seven primitives with the live views' colormaps, palette and point sizes, one colour key, states in place of missing data, bounds on every size, an offscreen canvas, the renderer behind `panel.render`, and the harness `caliper-spec-shot` ([Spec format](spec-format.md#the-spec-runtime)). The spec applet kind in the loader and the page that calls the runtime come next.
- `caliper-agent` runs its loop on Pydantic AI, with Claude on its native API and other models through an OpenAI-compatible API. Its tests use Pydantic AI's test models, a stubbed Claude transport and a fake OpenAI-compatible server, and run it against the real command layer through the stand-in host; no test calls a real model. It runs unscoped today, with every command of the table; the widget role is planned. Server-side refusal fallbacks are off, since Pydantic AI drops the reply's `fallback` block; a refusal ends the turn with a neutral message that offers to rephrase, retry, or switch the model or provider.

On Windows the option is off until a Windows session compiles the builder with MSVC, because the Windows CI lane builds every target and runs every test; host code that uses the builder must therefore also build without it, for example behind `if(TARGET caliper_builder_session)`. The builder installs nothing, so the SDK and embed install components hold none of it, and the wheel's configure sets the option off. The json extension does reach `libcaliper` and the embedding library, since they link the host's DuckDB.

## The integration order after sprint 2

These items need the app's main loop, its window or its API server, or build the two roles on top of them. They come in this order once sprint 2 merges. S is one to two days, M three to five and L six to ten; each figure is rough.

| Order | Item | Size |
|---|---|---|
| 1 | **Built: the command endpoints of Caliper API**, `GET /v0/commands`, `POST /v0/commands/{name}` and `GET /v0/approvals/{id}`, served by the app's API server in place of the stand-in host, with approvals held while the person decides ([Command layer](command-layer.md#caliper-api)). What remains is the one call that sets the session's gateway on the server, with item 2 | None left |
| 2 | **Built: the session in the app**: the app owns a session for its workspace, drains its queue on the frame thread and implements `SessionHost` for the app's information, its applets, a launch and the capture; `panel.render` waits for the spec runtime (5) behind a seam, and the command endpoints (1) connect to it when both land ([Builder session](builder-session.md#the-session-in-the-app)) | None left |
| 3 | **Done: the review of spec edits in the command layer**, an approval for the origins `agent` and `mcp` with the dry-run diff on each pending approval, so that no agent can skip it, and `caliper-agent` relying on it instead of asking a second time ([ACP integration](acp-integration.md#approvals-and-diffs-for-every-agent)). The cards and the setting per origin come with the side chat (8) | None left |
| 4 | **The catalog providers** for live views, feed channels and metrics | M, 4 days |
| 5 | **The spec runtime**, which draws a spec's panels with the live view types, natively and in the web viewer, and the **spec applet kind** in the loader | L, 10 days |
| 6 | **Drag and drop** of files into the window, which calls `data.import` | S, 1 day |
| 7 | **The builder pane**: palettes of sources and primitives, a canvas where panels dock, a property inspector, a live preview, and the spec text kept in sync both ways, with every edit a command | L, 10 days |
| 8 | **The side chat**: its approval cards and diagnostics, the four detected agents with `opencode` added to `known_agents()`, the custom agent command, the setting that marks an agent as running on a local model, and the packaging of `caliper-mcp` with the app | L, 6 days |
| 9 | **Done: stacked previews in the session**, so that edits that build on each other can be previewed together and committed as one step ([Widget agent](widget-agent.md#preview-and-apply-on-the-command-layer)). The app's API server must map `X-Caliper-Preview` to the invocation's `preview`, as the stand-in does | None left |
| 10 | **Done: an optional `_meta` argument on `AcpClient::new_session()`**, for the widget session's target, tool scope and preview | None left |
| 11 | **Done: the evaluation of the ACP Python SDK** for the sidecar's agent side. The hand-written agent side stays, and the SDK's models validate every message in the tests ([Widget agent sidecar](byok-agent.md#the-acp-python-sdk)) | None left |
| 12 | **Widget sessions in the sidecar**: the scoped tools, the widget system prompt, images and embedded context in prompts, the preview header, the turn budget and `session/close` | M, 5 days |
| 13 | **The context bundle**, built in the app from the session | M, 3 days |
| 14 | **The widget agent's inline surface**: ⌘K on a panel, a selection, a dropped file and a property, the preview drawn in place, Accept, Reject and Refine, and Ask in chat; with the packaging of `caliper-agent` and a Python runtime | L, 8 days |
| 15 | **Pydantic AI as the provider layer: adopted**, with the Claude and OpenAI-compatible providers built in the sidecar. What remains is passing the chosen provider, base URL and key from the app when it starts the sidecar; widget sessions (12) and the capability probe (16) build on it | S, 1 day |
| 16 | **The capability probe and its cache**, with the person's override | S, 2 days |
| 17 | **The presets and one key per provider** in the widget agent's settings | M, 3 days |
| 18 | **The provider conformance suite**, with stubs by default and real runs on demand | M, 4 days |
| 19 | **The GPU contention note** while a training run is live | S, 1 day |

### Where the pieces sit

This is a placement sketch for review, not a visual design. Each piece is built in the app and checked with screenshots at two window sizes, as the UI principles ask. The sketch shows a live page in build mode with every pane open.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ page bar    Home ▸ run42                                   [Build]  [Chat]     │
├────────────────────────────────────────────────────────────────────────────────┤
│ run bar     ● running   ❙❙  ▶  ⏭    lr 0.001   …        (a stream only)        │
├────────────┬──────────────────────────────────────────────┬────────────────────┤
│ Sources    │ canvas: the page's views and spec panels      │ Selection │ Props  │
│  table:…   │ ┌────────────────────┐ ┌───────────────────┐  │  (tabs, upper)     │
│  tensor:…  │ │ loss               │ │ attention         │  │                    │
│  live:…    │ │ ┌────────────────┐ │ │                   │  │                    │
│            │ │ │ ⌘K  prompt   ⏎ │ │ │                   │  ├────────────────────┤
│ Primitives │ │ └────────────────┘ │ │                   │  │ Chat               │
│  series    │ │  preview in place  │ │                   │  │  (the side chat,   │
│  heatmap   │ │ Accept Reject Refine│ └───────────────────┘  │   lower)           │
│  points …  │ └────────────────────┘                        │                    │
│            │ ┌ Spec text (toggled, under the canvas) ────┐ │                    │
│            │ └───────────────────────────────────────────┘ │                    │
├────────────┴──────────────────────────────────────────────┴────────────────────┤
│ jobs tray   (while it is open)                                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│ status bar                                                                     │
└────────────────────────────────────────────────────────────────────────────────┘
```

- **The host's strips stay as they are.** The page bar, the run bar of a stream, the jobs tray and the status bar keep their places at the window's edges. The page bar gains two toggles, Build and Chat.
- **The canvas is the page itself.** The live page's views and a spec's panels fill the centre, which stays the largest region. Build mode does not open another page; it adds the columns beside the same views.
- **The right column is where the selection pane already is.** Today the live page gives the selection's details a pane at its right edge. That place becomes a side column with two parts, like the side roles of the applet page. The upper part holds Selection and, in build mode, Properties, as tabs. The lower part holds the side chat. Each part appears only while it has something to show, and the column is a share of the window.
- **The left column exists only in build mode.** It holds the two palettes, Sources (the catalog) and Primitives. Dragging a source or a primitive onto the canvas makes a panel.
- **The spec text** is a strip under the canvas, toggled from the builder, in sync with the panels both ways.
- **The ⌘K prompt is inside the panel it acts on.** It opens at the top of the focused panel, or at a selection, a dropped file's entry in Sources or a property's row. The preview is drawn in that panel in place of its content, with Accept, Reject and Refine under the prompt. Nothing covers another pane.
- **A dropped file** lands anywhere on the page and appears in Sources; outside build mode the page shows the import in the jobs tray.

### The integration branch

The work is rebased onto sprint 2 on the branch `sprint3-integration`, where the integration items above are built. The rebase kept sprint 2's version of the files it owns; the builder's pages joined the design pages that sprint 2 moved here, and their links follow this folder's rules. The standalone build of `src/builder/` stays, because it needs no host build.

## Acceptance

The work is accepted when all of these hold:

- A person drops a Parquet file and a safetensors file and works with them in four ways: builds a three-panel dashboard by hand in the builder pane, through the side chat with their own agent over ACP, and from Claude Code over MCP, and changes it through the widget agent, for example with an edit by ⌘K on a panel that is previewed in place and accepted as one undo step.
- The dashboard is saved as a spec applet and reopened.
- It opens in the browser viewer.
- "Plot attention entropy per layer while it trains" works from the side chat on a live stream.
- Both roles work locally: the side chat with an ACP agent on a local model (opencode on Ollama, the worked example), and the widget agent on a local model server through its OpenAI-compatible provider, with the GPU contention note shown while a run is live.

The checks with a real agent, a real key or a real model server are manual, and each is written as steps on its page; no automated test uses a person's agent login, key or server.

## Decisions

Each page records its own decisions. These hold across the parts:

- **Everything built is data.** A spec and a loader are text files, and a Python loader's script runs in a child process, never in the app.
- **Text is the source of truth.** A workspace is a folder of text files, and its database can be rebuilt from them.
- **One command table.** `caliper-mcp` and `caliper-agent` both derive their tools from the command table, with the same names; the widget agent's tool scope is a subset of it for each kind of target.
- **Two roles.** The side chat is for conversations and bigger jobs with the person's own agent; the widget agent is for one target in seconds, with every change a preview. They share one journal, one undo history and one queue.
- **One session model for agents.** Both roles are ACP agents, so the app has one client for both.
- **Approvals and previews in the command layer.** Destructive commands and, with review on, spec edits are approved in the app, whichever agent sends them; no agent can approve its own request or accept its own preview.
- **Claude on its native API, other models through an OpenAI-compatible API.** Claude keeps its features, such as prompt caching and adaptive thinking, which Anthropic's compatibility layer drops; every other model, hosted or local, is reached through the provider layer, Pydantic AI, adopted after a spike ([Widget agent sidecar](byok-agent.md#providers)). Server-side refusal fallbacks are off until Pydantic AI keeps the reply's `fallback` block.
- **Local models in both roles.** The side chat runs any ACP agent on any model it supports, with no Caliper code for it; the widget agent gains local model servers as presets.
- **Keys.** With the side chat, the agent uses its own login and Caliper never handles a key. The widget agent's keys live only in the OS credential store, one per provider, and reach the sidecar through its environment; a local server needs none.
- **Nothing bundled.** Caliper bundles no model, no model server and no agent; the person installs their own, so there is no bundle size to check.
- **Loopback only.** Agents reach the app through Caliper API on 127.0.0.1 with the session token, and the stream socket never carries commands.
- **JSON values** in these libraries are nlohmann/json; the wire protocol keeps its own strict reader.

## Reserved for later work

The primitive name `scene`, for 3-D scenes, and the catalog kind `recording`, for recorded sessions, are reserved and not built ([Spec format](spec-format.md#reserved-names), [Catalog](catalog.md#reserved-names)).
