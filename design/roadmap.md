# Roadmap

Caliper has one core and many applets. The core is the host, its services and the SDK. Applets are versioned against the SDK and live in their own repositories. A feature moves into the core only when a second applet needs it.

This page is the only place for the order and status of work. Each phase links to the plan that describes it.

## Where things stand

On 15 November 2026, `main` holds the host, the SDK with its 19 service ids, Caliper Embed, the embeddable core, which installs as the `caliper-embed` 1.4.0 package, Caliper LIB, the Python package, and nine applets that live in the tree but that the core build does not compile. The SDK installs as the `caliper-sdk` 0.2.0 package, and every applet builds against it. The loader enforces the whole versioning contract, searches development folders and loads one copy of each applet. The inspector applet is being developed on the `inspector` branch and is not on `main`. The references and explanations under `docs/` describe what is built.

Streaming to notebooks, browsers and the Caliper app, the hub, Caliper API v0 and `caliper build` are built as well. Network streaming (`docs/python/network-streaming.md`), Caliper API (`docs/agents/http-api.md`) and Caliper CLI (`docs/reference/cli.md`) describe them, and [Streaming and embedding](#streaming-and-embedding) below lists what remains.

## Phase 1: one core with versioned applets

1. Done: the versioning contract and the promotion rule are written in Compatibility and epochs (`docs/development/compatibility-and-epochs.md`).
2. Done: the SDK installs as the `caliper-sdk` package, with the headers, the CMake config, the pinned UI stack and `caliper_add_applet()`; see SDK package (`docs/applets/sdk-package.md`).
3. Make the host name no applet. Done: windows dock by role, the core build compiles no applets, and CI builds the core without applets and checks that the host names none. Still to do: CI starts the empty core and checks that the landing screen shows its empty state.
4. Load applets from outside the repository. Done: development paths through `--applets <dir>` and `CALIPER_APPLET_PATH`, the user applets folder, and the search order with one copy chosen per id. Still to do: shipped applets signed in the app bundle.
5. Done: the `caliper` subcommands `list`, `check`, `info`, `new`, `install <folder>`, `dev` and `shot`, the `--applets <dir>` flag and `CALIPER_APPLET_PATH`; see Caliper CLI (`docs/reference/cli.md`). `caliper shot` draws an applet's page as the app draws it, with nothing on screen, and writes it as a PNG. It is the first piece of the capture service that Phase 2 promotes.
6. Done: the minimal embedding package, which is the `caliper-embed` package with its one shared library, `caliper_embed_api_version()`, and `examples/embed_host` built outside the repository and run offscreen; see Embed Caliper in a program (`docs/web/program-embedding.md`).
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

Caliper LIB (`docs/python/caliper-lib-api.md`), Caliper Embed (`docs/web/embedding-api.md`) and version 0 of Caliper API (`docs/agents/http-api.md`), the HTTP interface to the running app, are built. The full API comes later, with the command layer of [Phase 3](#phase-3-agent-built-panels), which is what lets a program drive the app's state from outside.

### Caliper CLI

The subcommands that are built, among them `build` and `serve`, are described in Caliper CLI (`docs/reference/cli.md`). `caliper install <url>`, `caliper package` and `caliper publish` come later, with the `.caliperapp` bundle format, the signing policy and the [registry](#phase-4-the-registry). Hot reload comes later too, and its entry points are [undecided](#undecided).

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

Built so far: the minimal embedding package; network streaming over wire protocol v3 (`docs/reference/wire-protocol.md`), with data views, `caliper.serve()`, the notebook widgets and, from version 3.1, controls that let a viewer pause and step the training loop, change registered parameters and run registered actions; the Caliper app and `caliper-viewer` attached to a stream by its link; the hub, `caliper serve`, with `caliper.connect()`; and the browser views, where Caliper's own view layer, compiled to WebAssembly and drawn through WebGPU, draws every live view kind and the views written for the browser in TypeScript draw where WebGPU is absent. Network streaming (`docs/python/network-streaming.md`) describes the design and links the references. Measured on a quiet machine against the commit-matched control, streaming costs `python/bench/bench_net.py`'s training loop 1 to 2.5 % of its throughput for one view with one or two viewers (a 1024 by 1024 heatmap 1.8 %, the loss series 1 %, 100,000 points 2 to 2.5 %), 3 to 4.6 % for all three together, and at most 1 % with no viewer; the table is in Cost of streaming, in that page.

Still to do, in this order:

1. Interventions on a model, such as ablating a head or steering along a direction, after the controls of wire protocol 3.1 (`docs/reference/wire-protocol.md`). They come with the inspector work.
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

## Scenes and recordings

3-D scenes and recordings are designed and not built. The design waits for review, and nothing of it is part of the current work. It builds on the streaming work of sprint 2, and only its recordings in the workspace and its spec primitive also wait for the agent-built panels of sprint 3. [Sprint 4](sprint-4.md) builds it as increments that a person can see, in this order: the renderer core's minimum cut; then a loss-landscape demo in a notebook, the first visible increment; then the rest of scenes, recordings and the demos, with every remaining migration step of the renderer core interleaved as a measured increment of its own. The parts are these:

1. The renderer core: one graphics layer that each backend implements once, one shader source, a shared scene core, and zero-copy kept as an import capability with a measurement before and after every step. Its minimum cut comes first: the layer with its Metal backend, the shader generator, the scene core and the browser's WebGPU backend, on which scenes start. Every remaining migration step then follows in full: the retirement of the GLEW OpenGL path, early; each existing view moved onto the layer; the removal of the TypeScript WebGPU views; a scene service for applets, which replaces the geometry service, as the user decided, with every 3-D applet of this repository moved onto it; one version of each graphics service, with one ABI epoch bump; the Dawn fallback; and, by the user's decision, the port of one applet that lives outside this repository. The Vulkan backend comes with [Windows parity](windows-parity.md). Each step lands as its own series of commits that replaces what it moves and deletes the old code, since the user decided that the graphics change needs no backwards compatibility, with measurements before and after, byte-exact pixel checks and offscreen captures, and the next step waits until it passes. It takes over the steps of [Renderers](#renderers) below. Plan: [Renderer core](renderer-core.md).
2. Scenes: entities on paths with entity transforms, points, lines, arrows, boxes, meshes, images and pinhole cameras, logged from Caliper LIB or drawn by an applet through the scene service, with a text log and a bar chart. Plan: [Scenes](scenes.md).
3. Timelines and recordings: stamps on several timelines, a recording file, recordings in the workspace database, and a scrubber in the app and in the browser. This is the recording and replay of [Data and Cloud](#data-and-cloud). Plans: [Recordings](recordings.md) and the [wire protocol 3.2 draft](wire-protocol-3-2.md).
4. Scenes in the browser, on the WebGPU backend of the renderer core. Plan: [Web geometry](web-geometry.md).
5. The loss-landscape demo, and four demos on openly licensed data. Plan: [Demos](demos.md).

### Sprint 5

A script sets the camera a 3-D view starts on, and a person switches the view between Y up and Z up; both are part of [Scenes](scenes.md). [Sprint 5](sprint-5.md) follows scenes and recordings with run comparison and linked cameras, slice planes and isosurfaces of 3-D fields, mesh files, a URDF helper, following an entity, trails, ellipsoids and error bands, ROS 2 import, an orthographic projection with preset views, saved viewpoints, more shapes, and events from the app to a script over shared memory. A fly mode is left out.

## Renderers

This work is part of the renderer core of [Sprint 4](sprint-4.md), whose [migration](renderer-core.md#migration) orders the steps. It is designed and not built.

Caliper's own Metal backend on macOS and Vulkan backend on Windows stay the normal path, at full speed and with zero-copy from torch, as backends of Caliper's own thin graphics layer. No third-party graphics library sits between them and the GPU.

1. Retire the GLEW OpenGL fallback now, early in sprint 4 and before Dawn, as the user decided. Until the Dawn fallback lands, a machine where neither Metal nor Vulkan starts gets a clear refusal instead of a slower path. There is no separate hand-written OpenGL or CPU renderer.
2. Add one WebGPU backend of the graphics layer, with two roles. In the browser it runs on the browser's own WebGPU, as part of the web viewer. Later in sprint 4 it becomes the native fallback through Dawn, Google's WebGPU library, which is an optional build component, and the host chooses it only when its own Metal or Vulkan renderer is unavailable. Dawn then draws through Direct3D 12 or 11, through OpenGL ES in a limited compatibility mode, or on the CPU through SwiftShader, which covers virtual machines, headless CI and machines with broken graphics drivers. This fallback is slower, but it works and says so, as the [invariants](#invariants) require.
3. Write each shader once and translate it for each backend. This is decided: the source is WGSL, translated to MSL and SPIR-V by Naga, with the generated sources committed, as [Renderer core](renderer-core.md#one-shader-source) describes.
4. Give applets a scene service. Applets draw in 3-D through `caliper.scene.v1`, which offers the scene model of Caliper LIB to the applet ABI with the shared camera, picking and the overlays, and the geometry service is deleted once its applets have moved, as the user decided ([Scenes](scenes.md#the-scene-service-for-applets)).
5. Keep one version of each graphics service. In sprint 4 the tensor bridge and the export service each collapse to one version, the scene service starts with one, the embedding library's two kinds of tensor views become one, and the ABI epoch is bumped once, from 2 to 3, where the service ids change, with every applet in this repository updated in the same changes, as the user decided. Applets and embedding hosts outside this repository stop loading until they are rebuilt against the new SDK, and those that drew through the geometry service until they are rewritten onto the scene service. One of them is ported in sprint 4 by the user's decision; whether to update the others is the user's decision.

Caliper then maintains three backends: Metal, Vulkan and WebGPU. The Vulkan backend of the layer comes with [Windows parity](windows-parity.md); until then the Windows app draws its ImGui pages and refuses textures, geometry and applet scenes. The user has accepted that gap: Windows and Vulkan are deferred until the user moves to the Windows machine. The renderers that exist today are described in Rendering (`docs/applets/rendering.md`).

## Windows parity

Windows parity follows the macOS sprints; [Windows parity](windows-parity.md) gives its steps.

## Smaller work

- A development build lists the `hello` and `sine_scope` test fixtures as shipped applets, because the tests build them into the build's `applets/` folder, which is the folder beside the executable.
- Code comments still cite sections of the deleted design documents by bare number, such as `(§6c)`. Rewrite each as its file is next touched.
- Fix stale code comments: the `sql_job` comment in `applets/embed_scope/embed_scope.cpp` (the code uses `sql_thread`), and the `CALIPER_VULKAN_SELFTEST` comment in `src/host/cuda_driver.h` (no code reads it).
- Measure EmbedScope's training step time on a Mac without the device drain, then retire the comment in `embed_scope.cpp` that says the drain dominates step time.
- GPTScope's head heatmap releases and re-creates its texture on every probe when the zero-copy pool is not in use. Create it once and update it.
- EmbedScope saves its checkpoint with `artifacts.put` on the frame thread when Save is clicked. Move the save into a job, as the frame-thread rule (`docs/introduction/architecture.md`) requires.
- Add the Vulkan mirror of the `geometry.v1_2` textured POINT row in `tests/gfx`, which exists only for Metal.
- Re-check the G15 instance-offset gate in the Metal and Vulkan backends, not only in the bridge, and isolate the Metal G1 to G12 re-gate in its own test.
- CI: cache the FetchContent downloads outside `cmake-build-release`, and cache or shallow-clone `third_party/`, which is now the longest step of a warm run.

## Not now

Lua or WebAssembly applets, third-party native applets, and accounts. Accounts belong to Caliper Cloud, not the desktop app ([Data and Cloud](data-and-cloud.md)).

Linux is not planned for now.

A second native host is not needed now. `examples/embed_host` stays as the embedding example and test. Revisit this only for a single-applet app or for embedding Caliper in another product.

## Invariants

Every phase keeps these rules. Platform rules (`docs/introduction/architecture.md`) gives the reasons.

- Data flows from tensors to pixels, one way.
- Applets supply no shaders.
- A missing capability falls back to a working slower path where one exists, and says so; where none exists, as for a machine without Metal or Vulkan before the Dawn fallback, Caliper refuses with a message that names what is missing.
- Graphics increments are verified byte for byte on every platform where they draw, and a platform whose path is not yet built refuses it rather than drawing it unverified.

## Risks

- Every additive revision of a service that keeps revisions grows the ABI surface that must be kept working. The graphics services keep one version each from sprint 4, so an incompatible change to one of them costs an epoch bump and a rebuild of every applet instead.
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
