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
| [Renderer core](renderer-core.md) | One graphics layer that each backend implements once, one shader source, a shared scene core, and zero-copy as an import capability |
| [Scenes](scenes.md) | 3-D and 2-D scenes of entities with entity transforms, primitives and cameras, logged from Caliper LIB, with a text log and a bar chart |
| [Wire protocol 3.2 draft](wire-protocol-3-2.md) | An additive minor version of the wire protocol for timelines, scenes, text logs, bar charts and recordings |
| [Web geometry](web-geometry.md) | Scenes in the browser on the WebGPU backend of the renderer core, and the fallback without WebGPU |
