# Builder overview

This page is the map of the builder: the work that lets someone drop data into Caliper, make loaders and build visuals from inside the app. A person builds by hand in a WYSIWYG builder pane, with their own agent in the side chat, with the widget agent on one panel, selection, dropped file or property, or from an agent outside the app over MCP. Everything that is built is data, a spec or a loader, and never code that runs in the app's process. The page says which parts are built, which wait, and where each part is described. The [roadmap](../docs/plans/roadmap.md#phase-3-agent-built-panels) gives the order and status.

[Applet builder](../docs/plans/applet-builder.md) is the earlier proposal that this work follows. It left a WYSIWYG editor and an agent inside the app for later; both are now part of the plan.

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
| `caliper-agent` | The widget agent's process: a sidecar that runs an agent loop with Claude through the person's API key today, and other models through the planned provider layer | [Widget agent sidecar](byok-agent.md) |

The side chat and the widget agent are two roles, not two interchangeable backends, and [Agent roles](../docs/explanations/agent-roles.md) explains when to use each. Both are ACP agents that the app drives with one `AcpClient`, so messages, diffs, approvals and cancellation work the same for both, and "Ask in chat" hands a widget request to the side chat with its context.

## What is built

The parts above are built as libraries and tools under `src/builder/`, each with unit tests, and none is wired into the app yet. `src/builder/` configures on its own, without the host build, or from the root build.

- The spec library parses, validates and writes specs, and applies merge patches.
- The loaders library reads CSV, TSV, Parquet, JSON and JSONL through DuckDB, and `.npy`, `.npz`, safetensors, PNG, JPEG and WAV natively. It refuses pickle-based files, and it runs a Python loader out of process for other formats, which loads torch checkpoints weights-only with a warning.
- The command layer has its registry, its schema validator, undo with groups and merging, its journal with replay, its approvals and its queue, and a table of 35 commands, which is also checked in as JSON.
- The session binds every command that needs no window. A headless test drops a Parquet file and a safetensors file, runs their loaders, builds a three-panel dashboard command by command, undoes and redoes, saves it as a spec applet, reopens it and replays the journal.
- `caliper-mcp` works against Caliper API v0 of the running app for the app's information, its applets, a launch and a capture, and against the command endpoints of the stand-in host for everything else.
- The ACP client drives a session against a stub agent, and `caliper-acp-harness` runs one session headless.
- `caliper-agent` runs its loop against a stubbed transport in tests, and against the real command layer through the stand-in host; no test calls the Claude API. It runs unscoped today, with every command of the table; the widget role is planned.

## The integration order after sprint 2

These items need the app's main loop, its window or its API server, or build the two roles on top of them. They come in this order once sprint 2 merges. S is one to two days, M three to five and L six to ten; each figure is rough.

| Order | Item | Size |
|---|---|---|
| 1 | **The command endpoints of Caliper API**, `GET /v0/commands` and `POST /v0/commands/{name}`, served by the app's API server in place of the stand-in host, with approvals held while the person decides ([Command layer](command-layer.md)) | M, 3 days |
| 2 | **The session in the app**: the app owns a session, drains its queue on the frame thread and implements `SessionHost` for capture, launch and panel rendering | L, 6 days |
| 3 | **The command-layer enforcement that remains**: the review of spec edits as an approval for the origins `agent` and `mcp`, with the dry-run diff attached to each pending approval, so that no agent can skip it ([ACP integration](acp-integration.md#approvals-and-diffs-for-every-agent)) | M, 3 days |
| 4 | **The catalog providers** for live views, feed channels and metrics | M, 4 days |
| 5 | **The spec runtime**, which draws a spec's panels with the live view types, natively and in the web viewer, and the **spec applet kind** in the loader | L, 10 days |
| 6 | **Drag and drop** of files into the window, which calls `data.import` | S, 1 day |
| 7 | **The builder pane**: palettes of sources and primitives, a canvas where panels dock, a property inspector, a live preview, and the spec text kept in sync both ways, with every edit a command | L, 10 days |
| 8 | **The side chat**: its approval cards and diagnostics, the four detected agents with `opencode` added to `known_agents()`, the custom agent command, the setting that marks an agent as running on a local model, and the packaging of `caliper-mcp` with the app | L, 6 days |
| 9 | **Stacked previews in the session**, so that edits that build on each other can be previewed together and committed as one step ([Widget agent](widget-agent.md#preview-and-apply-on-the-command-layer)) | M, 4 days |
| 10 | **An optional `_meta` argument on `AcpClient::new_session()`**, for the widget session's target, tool scope and preview | S, 1 day |
| 11 | **The evaluation of the ACP Python SDK** for the sidecar's agent side ([Widget agent sidecar](byok-agent.md#what-is-built-and-what-waits)) | S, 2 days |
| 12 | **Widget sessions in the sidecar**: the scoped tools, the widget system prompt, images and embedded context in prompts, the preview header, the turn budget and `session/close` | M, 5 days |
| 13 | **The context bundle**, built in the app from the session | M, 3 days |
| 14 | **The widget agent's inline surface**: ⌘K on a panel, a selection, a dropped file and a property, the preview drawn in place, Accept, Reject and Refine, and Ask in chat; with the packaging of `caliper-agent` and a Python runtime | L, 8 days |
| 15 | **Adopt Pydantic AI as the provider layer** (spike in progress) | M, about 4 days after the spike, to be sized again by its result |
| 16 | **The capability probe and its cache**, with the person's override | S, 2 days |
| 17 | **The presets and one key per provider** in the widget agent's settings | M, 3 days |
| 18 | **The provider conformance suite**, with stubs by default and real runs on demand | M, 4 days |
| 19 | **The GPU contention note** while a training run is live | S, 1 day |

## Acceptance

The work is accepted when all of these hold:

- A person drops a Parquet file and a safetensors file and works with them in four ways: builds a three-panel dashboard by hand in the builder pane, through the side chat with their own agent over ACP, and from Claude Code over MCP, and changes it through the widget agent, for example with an edit by ⌘K on a panel that is previewed in place and accepted as one undo step.
- The dashboard is saved as a spec applet and reopened.
- It opens in the browser viewer.
- "Plot attention entropy per layer while it trains" works from the side chat on a live stream.
- Both roles work locally: the side chat with an ACP agent on a local model (opencode on Ollama, the worked example), and the widget agent on a local model server once the provider layer has landed, with the GPU contention note shown while a run is live.

The checks with a real agent, a real key or a real model server are manual, and each is written as steps on its page; no automated test uses a person's agent login, key or server.

## Decisions

Each page records its own decisions. These hold across the parts:

- **Everything built is data.** A spec and a loader are text files, and a Python loader's script runs in a child process, never in the app.
- **Text is the source of truth.** A workspace is a folder of text files, and its database can be rebuilt from them.
- **One command table.** `caliper-mcp` and `caliper-agent` both derive their tools from the command table, with the same names; the widget agent's tool scope is a subset of it for each kind of target.
- **Two roles.** The side chat is for conversations and bigger jobs with the person's own agent; the widget agent is for one target in seconds, with every change a preview. They share one journal, one undo history and one queue.
- **One session model for agents.** Both roles are ACP agents, so the app has one client for both.
- **Approvals and previews in the command layer.** Destructive commands and, with review on, spec edits are approved in the app, whichever agent sends them; no agent can approve its own request or accept its own preview.
- **Claude on its native API, other models through an OpenAI-compatible API.** Claude keeps its features, such as prompt caching and adaptive thinking, which Anthropic's compatibility layer drops; every other model, hosted or local, is reached through the planned provider layer, Pydantic AI, once its spike holds up.
- **Local models in both roles.** The side chat runs any ACP agent on any model it supports, with no Caliper code for it; the widget agent gains local model servers as presets.
- **Keys.** With the side chat, the agent uses its own login and Caliper never handles a key. The widget agent's keys live only in the OS credential store, one per provider, and reach the sidecar through its environment; a local server needs none.
- **Nothing bundled.** Caliper bundles no model, no model server and no agent; the person installs their own, so there is no bundle size to check.
- **Loopback only.** Agents reach the app through Caliper API on 127.0.0.1 with the session token, and the stream socket never carries commands.
- **JSON values** in these libraries are nlohmann/json; the wire protocol keeps its own strict reader.

## Reserved for later work

The primitive name `scene`, for 3-D scenes, and the catalog kind `recording`, for recorded sessions, are reserved and not built ([Spec format](spec-format.md#reserved-names), [Catalog](catalog.md#reserved-names)).
