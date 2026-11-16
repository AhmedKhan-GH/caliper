# Builder overview

This page is the map of the builder: the work that lets someone drop data into Caliper, make loaders and build visuals from inside the app. A person builds by hand in a WYSIWYG builder pane, through an agent in the app's agent pane, or from an agent outside the app over MCP. Everything that is built is data, a spec or a loader, and never code that runs in the app's process. The page says which parts are built, which wait, and where each part is described. The [roadmap](roadmap.md#phase-3-agent-built-panels) gives the order and status.

[Applet builder](applet-builder.md) is the earlier proposal that this work follows. It left a WYSIWYG editor and an agent inside the app for later; both are now part of the plan.

## The parts

```
 Builder pane          Agent pane                      Outside agent
 (WYSIWYG)             ACP backend    BYOK backend     (Claude Code, Codex)
     │                 the user's     caliper-agent        │
     │                 own agent      (Claude API)         │ MCP on stdio
     │                     │ MCP           │               ▼
     │                     ▼               │           caliper-mcp
     │                 caliper-mcp         │               │
     │                     └───────┬───────┘───────────────┘
     │                             ▼  Caliper API on loopback, session token
     └──────────────────────► Command layer  (typed, journaled, undoable, with approval)
                                   │
        ┌──────────────┬───────────┼──────────────┬───────────────┐
        ▼              ▼           ▼              ▼               ▼
     Loaders        Catalog      Spec         Session          The app
     files to       every        sources,     binds them       capture, launch,
     sources        source       panels,      for one          live sources,
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
| ACP client | How Caliper runs the user's own agent as a subprocess and shows its messages, diffs and permission requests | [ACP integration](acp-integration.md) |
| `caliper-agent` | The BYOK backend of the agent pane: a sidecar that runs an agent loop against the Claude API with the user's key | [BYOK agent](byok-agent.md) |

The agent pane has two backends behind one chat interface, and [Agent roles](../explanations/agent-roles.md) explains when to use each. Both speak the Agent Client Protocol to the app, so messages, diffs, approvals and cancellation are the same for both.

## What is built

The parts above are built as libraries and tools under `src/builder/`, each with unit tests, and none is wired into the app yet. `src/builder/` configures on its own, without the host build, or from the root build.

- The spec library parses, validates and writes specs, and applies merge patches.
- The loaders library reads CSV, TSV, Parquet, JSON and JSONL through DuckDB, and `.npy`, `.npz`, safetensors, PNG, JPEG and WAV natively. It refuses pickle-based files, and it runs a Python loader out of process for other formats, which loads torch checkpoints weights-only with a warning.
- The command layer has its registry, its schema validator, undo with groups and merging, its journal with replay, its approvals and its queue, and a table of 35 commands, which is also checked in as JSON.
- The session binds every command that needs no window. A headless test drops a Parquet file and a safetensors file, runs their loaders, builds a three-panel dashboard command by command, undoes and redoes, saves it as a spec applet, reopens it and replays the journal.
- `caliper-mcp` works against Caliper API v0 of the running app for the app's information, its applets, a launch and a capture, and against the command endpoints of the stand-in host for everything else.
- The ACP client drives a session against a stub agent, and `caliper-acp-harness` runs one session headless.
- `caliper-agent` runs its loop against a stubbed transport in tests; no test calls the Claude API.

## What waits for the app

These parts need the app's main loop, its window or its API server:

1. **The command endpoints of Caliper API**, `GET /v0/commands` and `POST /v0/commands/{name}`, served by the app's API server in place of the stand-in host ([Command layer](command-layer.md)).
2. **The session in the app**: the app owns a session, drains its queue on the frame thread and implements `SessionHost` for capture, launch and panel rendering.
3. **The catalog providers** for live views, feed channels and metrics.
4. **The spec runtime**, which draws a spec's panels with the live view types, natively and in the web viewer, and the **spec applet kind** in the loader.
5. **Drag and drop** of files into the window, which calls `data.import`.
6. **The builder pane**: palettes of sources and primitives, a canvas where panels dock, a property inspector, a live preview, and the spec text kept in sync both ways, with every edit a command.
7. **The agent pane** with its two backends, its approval cards and its key control, and the packaging of `caliper-mcp` and `caliper-agent` with the app.

## Acceptance

The work is accepted when all of these hold:

- A person drops a Parquet file and a safetensors file and builds a three-panel dashboard in four ways: by hand in the builder pane, through the agent pane with their own agent over ACP, through the agent pane with their own key (BYOK), and from Claude Code over MCP.
- The dashboard is saved as a spec applet and reopened.
- It opens in the browser viewer.
- "Plot attention entropy per layer while it trains" works from the agent pane on a live stream.

The checks with a real agent and with a real key are manual, and each is written as steps on its page; no automated test uses a person's agent login or key.

## Decisions

Each page records its own decisions. These hold across the parts:

- **Everything built is data.** A spec and a loader are text files, and a Python loader's script runs in a child process, never in the app.
- **Text is the source of truth.** A workspace is a folder of text files, and its database can be rebuilt from them.
- **One command table.** `caliper-mcp` and `caliper-agent` both derive their tools from the command table, with the same names.
- **One session model for agents.** The BYOK sidecar is an ACP agent that Caliper built, so the app has one client for both backends.
- **Keys.** With ACP, the agent uses its own login and Caliper never handles a key. With BYOK, the key lives only in the OS credential store and reaches the sidecar through its environment.
- **Loopback only.** Agents reach the app through Caliper API on 127.0.0.1 with the session token, and the stream socket never carries commands.
- **Other model providers** are a later, separate decision, each through its own official SDK.
- **JSON values** in these libraries are nlohmann/json; the wire protocol keeps its own strict reader.

## Reserved for later work

The primitive name `scene`, for 3-D scenes, and the catalog kind `recording`, for recorded sessions, are reserved and not built ([Spec format](spec-format.md#reserved-names), [Catalog](catalog.md#reserved-names)).
