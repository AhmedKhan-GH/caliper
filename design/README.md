# Design

The pages in this folder are proposals for work that is not built yet. They are not part of the docs site: the docs site, built from `docs/`, describes what Caliper is and does today, for the people who build on it and the people who use it.

The [roadmap](roadmap.md) is the one place for order and status. Each other page here is a technical proposal that the roadmap points to. When a proposal is built, its content is folded into the reference under `docs/`, the tutorials, guides, references and explanations, and its page here is deleted. Git history keeps what the page said.

These pages mention pages of the docs site by their paths under `docs/` rather than by links, because the docs site and these pages are built separately.

## Building these pages

Build and serve them from the repository root, naming the config by its relative path. The first command checks them as CI does and writes the pages to `build/design-site`; the second serves them on 127.0.0.1, on a port of your choice that the docs site, on its usual 8000, does not use. `CONTRIBUTING.md` gives the pinned versions of MkDocs and its plugins.

```sh
mkdocs build --strict -f design/mkdocs.yml
mkdocs serve -f design/mkdocs.yml -a 127.0.0.1:8001
```

Both fail with a configuration error when run from inside `design/` or with an absolute path to `design/mkdocs.yml`. MkDocs refuses a docs folder that holds its own config file, and only a relative path to the file avoids that check.

## Pages

| Page | Proposal |
|---|---|
| [Roadmap](roadmap.md) | The phases of work, smaller work, open questions and the ideas still undecided |
| [Applets outside the core](applets-outside-the-core.md) | Build, store and install applets outside this repository, so the host ships with none and names none |
| [Platform candidates](platform-candidates.md) | Pieces that applets built for themselves and that belong in the host or the SDK |
| [Applet builder](applet-builder.md) | Panels that an agent builds from a declarative spec, through a command layer shared with outside agents |
| [Builder overview](builder-overview.md) | The map of the builder: its parts, what is built, the integration order and the acceptance criteria |
| [Spec format](spec-format.md) | The versioned TOML format of a spec applet, with its canonical JSON form and JSON Schema |
| [Loaders](loaders.md) | Declarative recipes that turn dropped files into sources, the readers, the pickle policy and the Python loader |
| [Catalog](catalog.md) | The list of every source with its schema, shape and rate, and its three commands |
| [Command layer](command-layer.md) | The typed commands that everything acts through, with undo, the journal and approvals |
| [Builder session](builder-session.md) | The headless object that binds the builder libraries for one workspace, and the stand-in host for tests |
| [Builder pane](builder-pane.md) | The WYSIWYG builder: palettes of sources and primitives, the canvas, the property inspector and the spec text, with every edit a command |
| [MCP server](mcp-server.md) | `caliper-mcp`, the stdio MCP server whose tools mirror the commands |
| [ACP integration](acp-integration.md) | The side chat, which runs the user's own agent over the Agent Client Protocol |
| [Widget agent](widget-agent.md) | The inline agent that changes one panel, selection, file or property from a ⌘K prompt, with a preview in place |
| [Widget agent sidecar](byok-agent.md) | `caliper-agent`, the process that runs the widget agent's loop with the user's own key or a local model |
| [Applet registry](registry.md) | A registry at caliperapp.ai, a Browse tab and install by id and version |
| [Data and Cloud](data-and-cloud.md) | Run history, run comparison and session recording in the desktop app's local database, and Caliper Cloud and Caliper Web, which offer them online |
| [Windows parity](windows-parity.md) | The steps that bring Windows level with macOS: the applet platform, Caliper LIB, streaming and Caliper API, the web viewer, Caliper Embed, the renderer fallbacks and distribution |
| [Streaming and embedding](streaming-and-embedding.md) | What remains of using Caliper from other programs: the rest of the web viewer, applets in a browser, WebTransport, the security review and the rest of C++ embedding |
| [Renderer Core](renderer-core.md) | The rest of the renderer core's migration: the texture path on the graphics layer, a scene service for applets, one version of each graphics service, the final renderer seam and the Vulkan backend |
| [Scenes](scenes.md) | What remains of scenes: the scene model's additions, the text log and the bar chart, stamps, scenes through shared memory with their data views, and the scene service for applets |
| [Wire Protocol 3.2 Draft](wire-protocol-3-2.md) | What producers, the hub and servers still do with timelines, text logs, bar charts and recordings, and the numbers of 3.2 to confirm |
| [Instruments](instruments.md) | The interfaces of the first instrument slice: spec version 1 with scene panels and controls, the run commands, the event ring of `events.wait` and `caliper.watch(model, optimizer)` |
| [Hosted Runs](hosted-runs.md) | The model runner: the commands that start, fork, stop and list managed runs, the review card, the child processes and the run sources |
| [Web Geometry](web-geometry.md) | What remains of scenes in the browser: pipelines, encoded frames, budgets and lost devices, parity rows, and the fallback without WebGPU |
| [Producer Runtime](producer-runtime.md) | The graphics layer without a window in the training process: the framework adapter, the sync points, the read-only rule, the budget per step and the measurements of its design spike |
| [GPU Programs](gpu-programs.md) | The interfaces of GPU programs: the program declarations of spec version 1 and the program commands, the sandbox, the cadence, the wire path to the script, the program status and events, and the first intervention |
| [Instrument Interface](instrument-interface.md) | The interface layer of an instrument: entities from spec sources, the `surface` and `path` operators, binding references, shared state, the remaining controls, the time transport over kept steps, bound text, annotations, layout shares and the viewport of a scene panel |
