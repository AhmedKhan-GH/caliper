# Design

The pages in this folder are proposals for work that is not built yet. They are not part of the docs site: the docs site, built from `docs/`, describes what Caliper is and does today, for the people who build on it and the people who use it.

The [roadmap](roadmap.md) is the one place for order and status. Each other page here is a technical proposal that the roadmap points to. When a proposal is built, its content is folded into the reference under `docs/`, the tutorials, guides, references and explanations, and its page here is deleted. Git history keeps what the page said.

These pages mention pages of the docs site by their paths under `docs/` rather than by links, because the docs site and these pages are built separately. From the repository root, `mkdocs build --strict -f design/mkdocs.yml` checks these pages, and `mkdocs serve -f design/mkdocs.yml -a 127.0.0.1:8001` shows them locally beside the docs site on its usual port.

| Page | Proposal |
|---|---|
| [Roadmap](roadmap.md) | The phases of work, smaller work, open questions and the ideas still undecided |
| [Applets outside the core](applets-outside-the-core.md) | Build, store and install applets outside this repository, so the host ships with none and names none |
| [Platform candidates](platform-candidates.md) | Pieces that applets built for themselves and that belong in the host or the SDK |
| [Applet builder](applet-builder.md) | Panels that an agent builds from a declarative spec, through a command layer shared with outside agents |
| [Applet registry](registry.md) | A registry at caliperapp.ai, a Browse tab and install by id and version |
| [Data and Cloud](data-and-cloud.md) | Run history, run comparison and session recording in the desktop app's local database, and Caliper Cloud and Caliper Web, which offer them online |
| [Windows parity](windows-parity.md) | The steps that bring Windows level with macOS: the applet platform, Caliper LIB, streaming and Caliper API, the web viewer, Caliper Embed, the renderer fallbacks and distribution |
| [Streaming and embedding](streaming-and-embedding.md) | What remains of using Caliper from other programs: the rest of the web viewer, applets in a browser, WebTransport, the security review and the rest of C++ embedding |
