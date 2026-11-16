# Roadmap

Caliper has one core and many applets. The core is the host, its services and the SDK. Applets are versioned against the SDK and live in their own repositories. A feature moves into the core only when a second applet needs it.

This page is the only place for the order and status of work. Each phase links to the plan that describes it.

## Where things stand

On 15 November 2026, `main` holds the host, the SDK with its 19 service ids, Caliper Embed, the embeddable core, which installs as the `caliper-embed` 1.4.0 package, Caliper LIB, the Python package, and nine applets that live in the tree but that the core build does not compile. The SDK installs as the `caliper-sdk` 0.2.0 package, and every applet builds against it. The loader enforces the whole versioning contract, searches development folders and loads one copy of each applet. The inspector applet is being developed on the `inspector` branch and is not on `main`. The references and explanations under `docs/` describe what is built.

Streaming to notebooks, browsers and the Caliper app, the hub, Caliper API v0 and `caliper build` are built as well. Network streaming (`docs/explanations/network-streaming.md`), Caliper API (`docs/references/api.md`) and Caliper CLI (`docs/references/cli.md`) describe them, and [Streaming and embedding](#streaming-and-embedding) below lists what remains.

## Phase 1: one core with versioned applets

1. Done: the versioning contract and the promotion rule are written in Compatibility and epochs (`docs/explanations/compatibility.md`).
2. Done: the SDK installs as the `caliper-sdk` package, with the headers, the CMake config, the pinned UI stack and `caliper_add_applet()`; see SDK package (`docs/references/sdk-package.md`).
3. Make the host name no applet. Done: windows dock by role, the core build compiles no applets, and CI builds the core without applets and checks that the host names none. Still to do: CI starts the empty core and checks that the landing screen shows its empty state.
4. Load applets from outside the repository. Done: development paths through `--applets <dir>` and `CALIPER_APPLET_PATH`, the user applets folder, and the search order with one copy chosen per id. Still to do: shipped applets signed in the app bundle.
5. Done: the `caliper` subcommands `list`, `check`, `info`, `new`, `install <folder>`, `dev` and `shot`, the `--applets <dir>` flag and `CALIPER_APPLET_PATH`; see Caliper CLI (`docs/references/cli.md`). `caliper shot` draws an applet's page as the app draws it, with nothing on screen, and writes it as a PNG. It is the first piece of the capture service that Phase 2 promotes.
6. Done: the minimal embedding package, which is the `caliper-embed` package with its one shared library, `caliper_embed_api_version()`, and `examples/embed_host` built outside the repository and run offscreen; see Embed Caliper in a program (`docs/guides/embed-caliper.md`).
7. Move the inspector applet's general pieces into the core as services, each in its own commit with tests. Its own panes stay in the applet.
8. Acceptance: the inspector and two existing scopes each build from their own repository against the installed SDK and load into a released Caliper. Two versions of one applet resolve correctly.
9. Move the first-party scopes to a `caliper-applets` repository, and the inspector to its own repository.

OpenGLlama comes back with the inspector work, as an applet beside the inspector. Nothing about it changes before then.

Plan: [Applets outside the core](applets-outside-the-core.md).

## Phase 2: promote proven features

Each promotion is a new service version. In order: capture and automation, which starts from `caliper shot`; the layout grid and text scale; tours; the Python sidecar; the `caliper::viz` components (annotation scene, colour by value, catalog tree); a quiet job kind. Development builds treat ImGui errors as failures.

Phase 2 starts by settling one decision: every applet splits into a computation that runs natively and an interface that compiles to WebAssembly, and the SDK lets interface code use only services that have a wire form. Applets built before the rule stay native-only until they are ported. [Applets in the browser](streaming-and-embedding.md#applets-in-the-browser) describes the split.

Plan: [Platform candidates](platform-candidates.md).

## Phase 3: agent-built panels

Someone drops data into the app, makes loaders and builds visuals from inside the app: by hand in a WYSIWYG builder pane, with their own agent in the side chat, with the widget agent on one panel, selection, file or property, or from an agent outside the app over MCP. Everything built is data, a spec or a loader. The parts are a command layer with undo, a journal, approvals and previews; loaders that turn dropped files into sources; a catalog of every source; a spec format with seven primitives (series, heatmap, points, histogram, table, image, text) and its runtime; a spec applet kind in the loader; the MCP server `caliper-mcp`; the side chat, which runs any ACP agent on any model it supports, local models included; and the widget agent `caliper-agent`, which proposes changes in place with Claude today and later with other models, hosted or local, through an OpenAI-compatible API.

Built as libraries and tools under `src/builder/`, with tests, and not yet wired into the app: the spec format and its validator, the loaders and their readers, the catalog, the command layer, the session that binds them, `caliper-mcp`, the ACP client and the sidecar `caliper-agent`. Still to do, in the app: the command endpoints of Caliper API, the spec runtime and the spec applet kind, drag and drop, the builder pane, the side chat and the widget agent's inline surface.

Plan: [Builder overview](builder-overview.md), which links to the page of each part; [Applet builder](applet-builder.md) is the earlier proposal.

## Phase 4: the registry

A registry at caliperapp.ai of spec applets and signed first-party applets, a Browse tab, versioned install by id, and publishing by pull request to a registry repository.

Plan: [Applet registry](registry.md).

## Developer features

### API

Caliper LIB (`docs/references/python-api.md`), Caliper Embed (`docs/references/embedding.md`) and version 0 of Caliper API (`docs/references/api.md`), the HTTP interface to the running app, are built. The full API comes later, with the command layer of [Phase 3](#phase-3-agent-built-panels), which is what lets a program drive the app's state from outside.

### Caliper CLI

The subcommands that are built, among them `build` and `serve`, are described in Caliper CLI (`docs/references/cli.md`). `caliper install <url>`, `caliper package` and `caliper publish` come later, with the `.caliperapp` bundle format, the signing policy and the [registry](#phase-4-the-registry). Hot reload comes later too, and its entry points are [undecided](#undecided).

### MCP

An MCP server comes in [Phase 3](#phase-3-agent-built-panels), beside the command layer, and uses the capture service that Phase 2 adds, starting from `caliper shot`. Besides building applets in the app, it is for:

- visualising the work of any agent;
- watching training runs;
- running interpretability experiments automatically through the inspector's intervention API;
- answering questions about data through `caliper.data.v1`;
- making figures and screenshots for the documentation;
- driving applets in tests;
- explaining what is on the screen from the tour data.

If agents need to see the screen sooner, the capture service moves earlier, not MCP.

## Streaming and embedding

[Streaming and embedding](streaming-and-embedding.md) gives the plan for what remains of using Caliper from other programs. The surfaces come in this order: Python scripts and notebooks, the web viewer, C++ embedding, Rust bindings and Swift. Data streaming is the only way Caliper reaches a browser.

Built so far: the minimal embedding package; network streaming over wire protocol v3 (`docs/references/wire-protocol.md`), with data views, `caliper.serve()`, the notebook widgets and, from version 3.1, controls that let a viewer pause and step the training loop, change registered parameters and run registered actions; the Caliper app and `caliper-viewer` attached to a stream by its link; the hub, `caliper serve`, with `caliper.connect()`; and the browser views, where Caliper's own view layer, compiled to WebAssembly and drawn through WebGPU, draws every live view kind and the views written for the browser in TypeScript draw where WebGPU is absent. Network streaming (`docs/explanations/network-streaming.md`) describes the design and links the references. Measured on a quiet machine against the commit-matched control, streaming costs `python/bench/bench_net.py`'s training loop 1 to 2.5 % of its throughput for one view with one or two viewers (a 1024 by 1024 heatmap 1.8 %, the loss series 1 %, 100,000 points 2 to 2.5 %), 3 to 4.6 % for all three together, and at most 1 % with no viewer; the table is in Cost of streaming, in that page.

Still to do, in this order:

1. Interventions on a model, such as ablating a head or steering along a direction, after the controls of wire protocol 3.1 (`docs/references/wire-protocol.md`). They come with the inspector work.
2. The rest of the web viewer's order in [The web viewer](streaming-and-embedding.md#the-web-viewer):
    1. For the views that are built: the clipboard and multi-touch in the canvas, a CI job for the module's checks, and testing in Safari, Firefox and on Windows.
    2. Every built-in view: points are built; the geometry path and ImPlot3D remain.
    3. The whole Caliper interface in a browser, with the landing screen, docking and the panes, storage in the browser's file system, DuckDB's WebAssembly build, and cross-origin isolation so that the page can use threads.
    4. Applets in the browser, through the split into computation and interface that [Phase 2](#phase-2-promote-proven-features) settles.
    5. The same WebGPU backend as the native fallback through Dawn, as [Renderers](#renderers) describes.
3. WebTransport beside the WebSocket, and the dedicated security review that comes before any session leaves the machine.
4. Then the rest of C++ embedding, Rust bindings and Swift.

## Data and Cloud

Run history, comparison between runs, and recording and replaying a session are planned for the desktop app after Phase 1, after the streaming work whose format they share. They build on the local DuckDB-backed stores and never need an account or a server. Caliper Cloud and Caliper Web, which offer the same online with sync, sharing, teams, hosted storage and hosted compute, are separate later products outside this repository. [Data and Cloud](data-and-cloud.md) gives the plan.

## Renderers

This work comes after Phase 1. The WebGPU backend behind `HostRenderer` is built for the browser, where it draws the web viewer's views (see `docs/explanations/rendering.md`). Its native role through Dawn is the last step of the web viewer's order and is not part of the current work.

Caliper's own Metal renderer on macOS and Vulkan renderer on Windows stay the normal path, at full speed and with zero-copy from torch. Nothing routes them through another layer.

1. Run the WebGPU backend as the native fallback through Dawn, Google's WebGPU library, which is an optional build component. The host chooses it only when its own Metal or Vulkan renderer is unavailable. Dawn then draws through Direct3D 12 or 11, through OpenGL ES in a limited compatibility mode, or on the CPU through SwiftShader, which covers virtual machines, headless CI and machines with broken graphics drivers. This fallback is slower, but it works and says so, as the [invariants](#invariants) require.
2. Retire the GLEW OpenGL fallback once the Dawn path covers its cases. There is no separate hand-written OpenGL or CPU renderer.
3. Evaluate writing each shader once and translating it for each backend, for example with the Slang shading language. This Slang is not the SystemVerilog library of the same name that `third_party/` vendors. The WebGPU backend's two shaders are written by hand in WGSL for now; a single source becomes worth its toolchain once the backend also serves the Dawn fallback and every shader exists three times.

Caliper then maintains three renderers: Metal, Vulkan and WebGPU. The renderers that exist today are described in Rendering (`docs/explanations/rendering.md`).

## Windows parity

Windows parity follows the macOS sprints; [Windows parity](windows-parity.md) gives its steps.

## Smaller work

- A development build lists the `hello` and `sine_scope` test fixtures as shipped applets, because the tests build them into the build's `applets/` folder, which is the folder beside the executable.
- Code comments still cite sections of the deleted design documents by bare number, such as `(§6c)`. Rewrite each as its file is next touched.
- Fix stale code comments: the `sql_job` comment in `applets/embed_scope/embed_scope.cpp` (the code uses `sql_thread`), and the `CALIPER_VULKAN_SELFTEST` comment in `src/host/cuda_driver.h` (no code reads it).
- Measure EmbedScope's training step time on a Mac without the device drain, then retire the comment in `embed_scope.cpp` that says the drain dominates step time.
- GPTScope's head heatmap releases and re-creates its texture on every probe when the zero-copy pool is not in use. Create it once and update it.
- EmbedScope saves its checkpoint with `artifacts.put` on the frame thread when Save is clicked. Move the save into a job, as the frame-thread rule (`docs/explanations/architecture.md`) requires.
- Add the Vulkan mirror of the `geometry.v1_2` textured POINT row in `tests/gfx`, which exists only for Metal.
- Re-check the G15 instance-offset gate in the Metal and Vulkan backends, not only in the bridge, and isolate the Metal G1 to G12 re-gate in its own test.
- CI: cache the FetchContent downloads outside `cmake-build-release`, and cache or shallow-clone `third_party/`, which is now the longest step of a warm run.

## Not now

Lua or WebAssembly applets, third-party native applets, and accounts. Accounts belong to Caliper Cloud, not the desktop app ([Data and Cloud](data-and-cloud.md)).

Linux is not planned for now.

A second native host is not needed now. `examples/embed_host` stays as the embedding example and test. Revisit this only for a single-applet app or for embedding Caliper in another product.

## Invariants

Every phase keeps these rules. Platform rules (`docs/explanations/architecture.md`) gives the reasons.

- Data flows from tensors to pixels, one way.
- Applets supply no shaders.
- A missing capability falls back to a working slower path and says so.
- Graphics increments are verified byte for byte on both platforms.

## Risks

- Every additive revision grows the ABI surface that must be kept working.
- Once a registry exists, an epoch bump strands every installed applet until it is rebuilt.
- A registry is a supply-chain target; see signing under [Open questions](#open-questions).

## Open questions

- **Audience.** Today the audience is collaborators who build Caliper from source, and the contracts are sized so that they also work for end users who install a prebuilt bundle. Revisit this when the first such user appears.
- **Signing third-party applets on macOS.** A notarized host either carries the `disable-library-validation` entitlement or loads only applets signed by its own team. The choice must be made before the first third-party applet ships; the options are in [Applets outside the core](applets-outside-the-core.md#signing-on-macos).
- **Applet shaders.** Whether applets still supply no shaders once agent-built panels exist.

## Undecided

These ideas are not built. Each needs a decision before it enters a phase.

- **Reviewer bundle.** One download that runs an applet and its demo on a machine that never built Caliper.
- **The em-controller.** An actor-critic policy that steers FieldScope's plasma, as an applet of its own.
- **Digital twins.** A population twin built on InstanceScope's unused per-variant reduction, and a machine thermal twin fed by `caliper.feed.v1`.
- **Research publishing beyond export.** A Python DLPack path into the geometry bridge, and applets written in Python.
- **Artifacts redesign.** A checkpoint browser with previews and resuming training from a checkpoint.
- **Applet dependency packaging.** Whether an applet built outside the core may link its own heavy dependencies, such as DuckDB or llama.cpp, and what happens when two applets each carry a copy. The proposal for large runtimes such as libtorch is runtime packs: the host installs and manages each runtime once instead of every applet bundling its own multi-gigabyte copy, and a process loads one libtorch for the whole session. The stated limit is that two applets that need different torch versions cannot share a session. No pack manager exists; the build downloads torch directly.
- **`.caliperapp` bundles.** A bundle folder with per-platform binaries, and manifest keys for platforms, authors, licence and runtimes.
- **Hot reload.** Which entry points reloading a rebuilt applet needs, such as `save_state`, `load_state`, `on_suspend` and `on_resume`.
- **SDK conformance.** A conformance checker, and a set of reference applets built against each SDK release.
- **Isolation and reach.** Applets in a separate process and scripting bindings.
- **Live streaming.** Several viewers per session, f16, bf16 and RGB zero-copy kernels, a smaller wheel without the stores, and an arm64 wheel job.
- **Embedding.** A canvas-info query.
- **Zero-copy.** Device-backed `alloc_shared` on Metal and a pool of exportable Vulkan images.
- **Geometry.** Thick lines and ribbons, MSAA, a culling flag and volumetric rendering.
- **Telemetry sources.** Feed channels registered by applets or read from serial, UDP or MQTT.
- **GPTScope follow-ups.** Point picking and a learned projection in ThoughtSpace, induction-head metrics and activation patching.
- **Instrument inputs.** Live input slots bound to applet manifests through a `live_input` service.
