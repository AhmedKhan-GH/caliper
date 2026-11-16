# Applet builder

This page explores how an agent inside Caliper could build applets on the spot: widgets, visuals and connections to
data, made by asking for them rather than by writing C++. It records what the current architecture already provides,
what stands in the way, a proposed design and the changes it needs. Nothing here is built yet, and the open decisions
are listed at the end.

!!! note "Status"
    This proposal is being built. [Builder overview](../design/builder-overview.md) maps the parts, says what is built and what waits, and links to the page of each part. Two things changed since this page was written: a WYSIWYG builder pane and agents inside the app are now part of the plan, and the app has two agent roles, a side chat that runs the user's own agent ([ACP integration](../design/acp-integration.md)) and an inline widget agent ([Widget agent](../design/widget-agent.md)). Where this page and those pages differ, those pages hold.

The proposal in one sentence: the agent builds applets as **data**, a declarative panel spec that one host-side
runtime draws, and it edits them through a **command layer** that an in-app agent and outside agents share. Compiled
applets remain the path for panels that outgrow the spec.

## What already exists

| Piece | Where | Why it matters for a builder |
|---|---|---|
| A service registry | `sdk/include/caliper/abi.h` (`get_service`) | New services, such as a catalog or commands, are added beside the old ones without breaking any applet (see `docs/explanations/compatibility.md`). |
| SQL over the embedded store | `caliper.data.v1` (`docs/references/service-data-v1.md`) | The agent can query and bind tabular data on the spot; results come back as Arrow streams. |
| Live telemetry | `caliper.feed.v1` (`docs/references/service-feed-v1.md`) | Enumerable channels with timestamps, read without blocking. |
| Metrics | `caliper.metrics.v1` (`docs/references/service-metrics-v1.md`) | Runs, tags and steps in TensorBoard's vocabulary. |
| Live views | `python/caliper` (`show`, `points`, `series`, `histogram`), drawn by `src/host/live/live_views_draw.cpp` | Visuals that need no applet code already exist: the host draws them by type and name. |
| A guarded loader | `src/host/applet_loader.h` | Manifest-first checks, guarded calls, quarantine and restart. |
| A frame watchdog | `src/host/frame_watchdog.h` | A bound on panels that take too long. |

## What stands in the way

- **Applets are compiled C++.** The loader keeps every library image mapped and never unloads it, so new code cannot
  be hot-reloaded under the same name.
- **In-process code is trusted code.** The Trust model (`docs/explanations/trust-model.md`) says so plainly, and real
  isolation would need applets that run out of process, which do not exist. Loading native code that an agent wrote into the app's process is therefore risky.
- **No command layer.** UI actions are not typed, scriptable commands, so there is nothing for an agent to call.
- **No catalog.** Each service lists its own data, but nothing gives the agent one view of every source with its
  schema, shape and rate.
- **No capture.** The agent cannot render a panel to an image to check its own work. A host automation and capture
  service is already a [platform candidate](platform-candidates.md).
- **Layout only by role.** Applets dock their windows by the roles they give through
  `caliper.ui.v1_1` (`docs/references/service-ui-v1-1.md`), and nothing places a window by command, so an agent cannot
  place panels.

## Three ways an agent could build an applet

| | Declarative spec | Embedded script (Lua or WASM) | Generated native applet |
|---|---|---|---|
| Iteration speed | Instant | Instant | Seconds to minutes per build |
| Expressiveness | Limited to the primitives | High | Full |
| Can it crash the app? | No; it is data | Sandboxed | Yes; it runs in-process |
| Fit with today's code | Good: reuses the services and live views | Needs a new runtime | Needs unloading, reloading and isolation |

The proposal layers them. Specs come first and should cover most needs. Scripts can follow for custom logic. Native
generation becomes an "eject to C++" step for a panel that needs raw performance, built outside the app and loaded like
any other applet.

For tabular data, DuckDB's SQL can serve as the transform language, so a general scripting language is not needed
early. Tensors get a small fixed set of operations that run on the GPU: slice, reduce and normalise.

## Proposed design

```
 Builder pane (chat · live preview · property inspector)
        │
 Agent loop ──tools──► Command layer ◄── MCP / control server (Claude Code, Codex)
        │                  │  catalog.list/describe/sample, data.query,
        │                  │  panel.create/patch/render, layout.place, applet.save
        ▼                  ▼
   Spec files ──────► Spec runtime (one host-side "composer" applet)
   (TOML in the          primitives with typed input slots:
    workspace)           series · heatmap · scatter · histogram · table · 3-D · controls
                           │
                 data.v1 · feed.v1 · metrics.v1 · live views · tensor_bridge
```

- **The command layer** is the one surface everything acts through: the builder pane, the agent and outside agents.
- **The spec** names the sources, any transforms, the primitives with their slots bound to sources, and the layout.
  It is a text file, so it can be read, diffed and versioned.
- **The spec runtime** is a host-owned applet that draws any spec. It reads data through the existing services, so a
  spec never touches memory directly.
- **A saved applet** becomes a manifest plus a spec, with no library file. The loader learns this second kind of
  applet.
- **The browser.** Spec applets are designed to run in the web viewer unchanged
  ([Streaming and embedding](streaming-and-embedding.md#spec-applets)). Their primitives are the live view types, and
  a spec binds only to sources that have a wire form.

An illustrative spec, not a settled format:

```toml
[applet]
id = "attention-entropy"
title = "Attention entropy"

[[source]]
id = "entropy"
kind = "live_view"              # a tensor a training script publishes with caliper.show
name = "attn.entropy"           # shape [layers, heads]

[[source]]
id = "loss"
kind = "sql"
query = "SELECT step, loss FROM metrics WHERE run = 'current' ORDER BY step"

[[panel]]
id = "heat"
primitive = "heatmap"
values = "entropy"
x_label = "Head"
y_label = "Layer"

[[panel]]
id = "mean"
primitive = "series"
y = { source = "entropy", reduce = "mean" }

[[panel]]
id = "loss"
primitive = "series"
x = "loss.step"
y = "loss.loss"

[layout]
split = "horizontal"
panels = ["heat", { split = "vertical", panels = ["mean", "loss"] }]
```

## What would need to change

1. **A command layer in the host.** Every action becomes a typed, serialisable command with undo. Commands are queued
   and run on the frame thread, because ImGui runs on one thread.
2. **A catalog service.** One list of every source across DuckDB tables and datasets, feed channels, metrics and
   live-session views, with schema, shape and rate, and a way to sample each. This is how the agent learns what exists.
3. **A primitive library.** The drawing in `live_views_draw.cpp` and in the scopes moves into the `caliper::viz` layer
   that [Platform candidates](platform-candidates.md) already proposes, with declared input
   slots.
4. **A data-driven dock layout,** so that panels can be placed by command and not only by the roles applets declare.
5. **A capture service** that renders a panel offscreen to an image and its state, so the agent can see its result.
6. **The agent.** The command layer is exposed over MCP first, so Claude Code or Codex can build panels from outside.
   That proves the tool surface before the app gets its own chat pane. [Agent integration](#agent-integration) describes
   the order: MCP, then an ACP client, then a built-in agent only if needed.
7. **A second applet kind in the loader:** a manifest that points to a spec instead of a library.
8. **Limits.** SQL runs on a read-only connection with row and time limits, and the frame watchdog bounds slow panels.
   Scripts, when they come, run in a sandbox. Native generation stays outside the process until applets can run out of process.

## How a session would feel

1. You ask: "plot the attention entropy per layer while it trains, and its mean".
2. The agent searches the catalog, finds the live view and samples it to learn its shape.
3. It writes a spec with a heatmap and a series, renders a preview through the capture service and checks it.
4. You adjust the result by dragging, by chat or in the inspector. Every change is a command, so it can be undone.
5. Saving the result makes it an applet that you can reopen, version and share.

## A first slice

The smallest version that proves the idea:

- the command layer and the catalog;
- a spec runtime with four primitives: series, heatmap, scatter and table;
- bindings to `data.v1` queries and to live views;
- the capture service;
- all of it exposed over MCP, so an outside agent can drive it.

The in-app chat pane comes after that.

## Agent integration

Three protocols answer three different questions, and the applet builder needs the first in every case.

- **MCP** is how an agent reaches Caliper's tools: the catalog, data, panel, layout and capture commands.
- **ACP**, the [Agent Client Protocol](https://agentclientprotocol.com), lets Caliper host an agent the user already
  has, such as Claude Code, Codex or Gemini CLI, inside its own pane. The agent runs its own loop.
- **BYOK** ("bring your own key") applies when Caliper runs the agent loop itself and calls a model with the user's
  API key.

### The options

| | MCP only | ACP client in Caliper | Built-in agent (BYOK) |
|---|---|---|---|
| Where the chat happens | The user's terminal | Caliper's builder pane | Caliper's builder pane |
| Who runs the loop | The outside agent | The outside agent, as a subprocess | Caliper |
| Keys and billing | The agent's own login or subscription | The agent's own login or subscription | The user's API key in the Keychain, paid per token |
| What Caliper builds | An MCP server | An MCP server, a JSON-RPC client over stdio and the pane | A tool layer, a model client, streaming and the loop |
| What Caliper controls | Little | The UI, the approvals and which tools exist | Everything, including the context and the prompts |
| Main risk | None beyond the MCP server | Adapters vary in maturity | Keeping the loop and the providers working |

### How the ACP client would work

```
Builder pane ◄─ session/update, request_permission ─┐
     │ session/prompt                               │  JSON-RPC over stdio
     ▼                                              │
Caliper ── spawns ──► claude-agent-acp / codex-acp / gemini …
                        │ connects to the MCP servers named in session/new
                        ▼
               caliper mcp ──► Caliper's control port ──► command layer
```

1. Caliper starts the agent that the user chose as a subprocess. It calls `initialize`, then `session/new` with a
   working folder and the list of MCP servers the agent should connect to
   ([session setup](https://agentclientprotocol.com/protocol/session-setup.md)).
2. The MCP server it names is `caliper mcp --connect <port>`, a small subcommand of the `caliper` binary that bridges
   to the running app. This is a common pattern for desktop apps: the app runs a control port, and a command-line tool
   serves MCP over it.
3. The pane shows the agent's messages, its tool calls and its permission requests, which become approve and deny
   buttons. Messages arrive on a background thread and reach the UI through the command queue.
4. Authentication stays with the agent. A user signed in to Claude Code or Codex works on their own plan, and Caliper
   never holds a key.

### Caveats

- Claude Code and Codex speak ACP through adapters rather than natively
  ([codex-acp](https://awesome.ecosyste.ms/projects/github.com%2Fagentclientprotocol%2Fcodex-acp),
  [overview](https://www.morphllm.com/agent-client-protocol)). Vercel advises direct adapters for those two where
  they exist ([Vercel](https://vercel.com/changelog/use-acp-compatible-harnesses-with-the-ai-sdk-harness-layer)).
- Adapters handle the MCP servers passed in `session/new` unevenly. Cursor's ACP mode had a reported bug where they did
  not work ([forum](https://forum.cursor.com/t/mcp-servers-passed-via-session-new-dont-work-in-acp-mode/153823)).
- The protocol is still changing: it has a v2 with migration notes
  ([v2 migration](https://agentclientprotocol.com/protocol/v2/migration.md)).
- Coding agents expect file and terminal access. Caliper can decline both capabilities, or scope the working folder to
  the applet workspace.
- A design proposal would let the client serve its MCP tools over the ACP connection itself, which would remove the
  bridge process ([MCP over ACP](https://agentclientprotocol.com/rfds/mcp-over-acp)).

### Order

1. **MCP first.** The command layer and an MCP server over a localhost control port, as in the first slice above.
   Every later option reuses them, and applets can be built from Claude Code straight away.
2. **ACP second, as the main in-app path.** The pane hosts whichever agent the user already has. Caliper builds only the
   UI and the approvals, the users' own plans pay for the models, and one protocol covers Claude, Codex and Gemini.
3. **A built-in BYOK agent only if needed,** for users with no command-line agent, or for prompts tuned to Caliper. A
   managed mode that routes model calls through a hosted service could follow later, as long as local use never
   depends on it.

### Safety in every option

- The control port listens only on localhost and needs a token for each session.
- SQL runs on a read-only connection.
- Commands that delete or overwrite need an approval in the pane.

## Open decisions

- **Which agents to support first** through ACP. Decided: Caliper detects Claude Code, Codex, Gemini CLI and opencode, and any other ACP agent can be added by its command ([ACP integration](../design/acp-integration.md)); the built-in agent became the widget agent ([Widget agent](../design/widget-agent.md)).
- **Sharing.** Whether spec-built applets can be shared and installed like compiled ones, and through which registry.
- **The first primitives.** Which visuals and controls the spec supports first, and which stay compiled-only.
- **Scripting.** Whether custom logic should eventually come from Lua, WASM or a Python sidecar, or wait until specs
  prove too limited.
