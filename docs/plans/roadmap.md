# Roadmap

Caliper has one core and many applets. The core is the host, its services and the SDK. Applets are versioned against the SDK and live in their own repositories. A feature moves into the core only when a second applet needs it.

This page is the only place for the order and status of work. Each phase links to the plan that describes it.

## Where things stand

On 15 November 2026, `main` holds the host, the SDK with its 19 service ids, the embeddable core, the Python client and nine applets that live in the tree but that the core build does not compile. The SDK installs as the `caliper-sdk` 0.2.0 package, and every applet builds against it. The loader enforces the whole versioning contract, searches development folders and loads one copy of each applet. The inspector applet is being developed on the `inspector` branch and is not on `main`. The [references](../references/index.md) and [explanations](../explanations/index.md) describe what is built.

## Phase 1: one core with versioned applets

1. Done: the versioning contract and the promotion rule are written in [Compatibility and epochs](../explanations/compatibility.md#versioning).
2. Done: the SDK installs as the `caliper-sdk` package, with the headers, the CMake config, the pinned UI stack and `caliper_add_applet()`; see [SDK package](../references/sdk-package.md).
3. Make the host name no applet. Done: windows dock by role, the core build compiles no applets, and CI builds the core without applets and checks that the host names none. Still to do: CI starts the empty core and checks that the landing screen shows its empty state.
4. Load applets from outside the repository. Done: development paths through `--applets <dir>` and `CALIPER_APPLET_PATH`, the user applets folder, and the search order with one copy chosen per id. Still to do: shipped applets signed in the app bundle.
5. Done: the `caliper` subcommands `list`, `check`, `info`, `new`, `install <folder>` and `dev`, the `--applets <dir>` flag and `CALIPER_APPLET_PATH`; see [Command line](../references/cli.md).
6. Ship the minimal embedding package, as [Streaming and embedding](streaming-and-embedding.md#the-embedding-package) describes. It is being built on a branch and is not merged yet.
7. Move the inspector applet's general pieces into the core as services, each in its own commit with tests. Its own panes stay in the applet.
8. Acceptance: the inspector and two existing scopes each build from their own repository against the installed SDK and load into a released Caliper. Two versions of one applet resolve correctly.
9. Move the first-party scopes to a `caliper-applets` repository, and the inspector to its own repository.

OpenGLlama comes back with the inspector work, as an applet beside the inspector. Nothing about it changes before then.

Plan: [Applets outside the core](applets-outside-the-core.md).

## Phase 2: promote proven features

Each promotion is a new service version. In order: capture and automation; the layout grid and text scale; tours; the Python sidecar; the `caliper::viz` components (annotation scene, colour by value, catalog tree); a quiet job kind. Development builds treat ImGui errors as failures.

Plan: [Platform candidates](platform-candidates.md).

## Phase 3: agent-built panels

A command layer with undo, a catalog of every data source, a spec runtime with about six primitives (series, heatmap, points, mesh, table, histogram), a spec applet kind in the loader, and an MCP server so Claude Code or Codex can build and check panels.

Plan: [Applet builder](applet-builder.md).

## Phase 4: the registry

A registry at caliperapp.ai of spec applets and signed first-party applets, a Browse tab, versioned install by id, and publishing by pull request to a registry repository.

Plan: [Applet registry](registry.md).

## Developer features

### API

The [Python API](../references/python-api.md) and the embeddable `libcaliper` ([Embedding Caliper](../references/embedding.md)) exist. Driving the app's state from outside waits for the command layer in [Phase 3](#phase-3-agent-built-panels).

### Command line

The command line is subcommands of the `caliper` binary rather than a separate program, so there is one thing to install. Phase 1 built `list`, `check`, `info`, `new`, `install <folder>` and `dev`, with the `--applets <dir>` flag and `CALIPER_APPLET_PATH`; [Command line](../references/cli.md) describes them.

`caliper install <url>`, `caliper package` and `caliper publish` come later, with the `.caliperapp` bundle format, the signing policy and the [registry](#phase-4-the-registry). Hot reload comes later too, and its entry points are [undecided](#undecided).

### MCP

An MCP server comes in [Phase 3](#phase-3-agent-built-panels), beside the command layer, and uses the capture service that Phase 2 adds. Besides building applets in the app, it is for:

- visualising the work of any agent;
- watching training runs;
- running interpretability experiments automatically through the inspector's intervention API;
- answering questions about data through `caliper.data.v1`;
- making figures and screenshots for the documentation;
- driving applets in tests;
- explaining what is on the screen from the tour data.

If agents need to see the screen sooner, the capture service moves earlier, not MCP.

## Streaming and embedding

Using Caliper from other programs is planned work after Phase 1, apart from the minimal embedding package in Phase 1, and it is not part of the current work. [Streaming and embedding](streaming-and-embedding.md) gives the plan. The surfaces come in this order: Python scripts and notebooks, the web viewer, C++ embedding, Rust bindings and Swift. Video streaming of a running Caliper comes after them.

## Renderers

This work comes after Phase 1, alongside the web viewer of [Streaming and embedding](#streaming-and-embedding). It is not part of the current work.

Caliper's own Metal renderer on macOS and Vulkan renderer on Windows stay the normal path, at full speed and with zero-copy from torch. Nothing routes them through another layer.

1. Add one WebGPU backend behind `HostRenderer`, with two roles. In the browser it runs on the browser's own WebGPU, as part of the web viewer. As the native fallback it runs through Dawn, Google's WebGPU library, which is an optional build component, and the host chooses it only when its own Metal or Vulkan renderer is unavailable. Dawn then draws through Direct3D 12 or 11, through OpenGL ES in a limited compatibility mode, or on the CPU through SwiftShader, which covers virtual machines, headless CI and machines with broken graphics drivers. This fallback is slower, but it works and says so, as the [invariants](#invariants) require.
2. Retire the GLEW OpenGL fallback once the Dawn path covers its cases. There is no separate hand-written OpenGL or CPU renderer.
3. Evaluate writing each shader once and translating it for each backend, for example with the Slang shading language. This Slang is not the SystemVerilog library of the same name that `third_party/` vendors.

Caliper then maintains three renderers: Metal, Vulkan and WebGPU. The renderers that exist today are described in [Rendering](../explanations/rendering.md).

## Smaller work

- A development build lists the `hello` and `sine_scope` test fixtures as shipped applets, because the tests build them into the build's `applets/` folder, which is the folder beside the executable.
- Code comments still cite sections of the deleted design documents by bare number, such as `(§6c)`. Rewrite each as its file is next touched.
- Fix stale code comments: the `sql_job` comment in `applets/embed_scope/embed_scope.cpp` (the code uses `sql_thread`), and the `CALIPER_VULKAN_SELFTEST` comment in `src/host/cuda_driver.h` (no code reads it).
- Measure EmbedScope's training step time on a Mac without the device drain, then retire the comment in `embed_scope.cpp` that says the drain dominates step time.
- GPTScope's head heatmap releases and re-creates its texture on every probe when the zero-copy pool is not in use. Create it once and update it.
- EmbedScope saves its checkpoint with `artifacts.put` on the frame thread when Save is clicked. Move the save into a job, as the [frame-thread rule](../explanations/architecture.md#the-frame-loop) requires.
- Add the Vulkan mirror of the `geometry.v1_2` textured POINT row in `tests/gfx`, which exists only for Metal.
- Re-check the G15 instance-offset gate in the Metal and Vulkan backends, not only in the bridge, and isolate the Metal G1 to G12 re-gate in its own test.
- CI: cache the FetchContent downloads outside `cmake-build-release`, and cache or shallow-clone `third_party/`, which is now the longest step of a warm run.

## Not now

A WYSIWYG editor inside the app, an agent inside the app, Lua or WebAssembly applets, third-party native applets, and accounts.

Linux is not planned for now.

A second native host is not needed now. `examples/embed_host` stays as the embedding example and test. Revisit this only for a single-applet app or for embedding Caliper in another product.

## Invariants

Every phase keeps these rules. [Platform rules](../explanations/architecture.md#platform-rules) gives the reasons.

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

These ideas come from earlier designs and are not built. Each needs a decision before it enters a phase.

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
