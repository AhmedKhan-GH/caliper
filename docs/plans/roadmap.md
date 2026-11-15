# Roadmap

Caliper has one core and many applets. The core is the host, its services and the SDK. Applets are versioned against the SDK and live in their own repositories. A feature moves into the core only when a second applet needs it.

This page is the only place for the order and status of work. Each phase links to the plan that describes it.

## Where things stand

On 15 November 2026, `main` holds the host, the SDK with its 18 service ids, the embeddable core, the Python client and nine applets compiled in the tree. The inspector applet is being developed on the `inspector` branch and is not on `main`. The [references](../references/index.md) and [explanations](../explanations/index.md) describe what is built.

## Phase 1: one core with versioned applets

1. Write the versioning contract into [Compatibility and epochs](../explanations/compatibility.md). It has four layers: the host ABI and service versions, the SDK version, each applet's version and what it requires, and the schema version of each applet's stored data. It also states the promotion rule: when a feature moves into the core, applets that need it declare the new service version, and applets that bundle their own copy keep working.
2. Ship the SDK as an installed package: the headers, the CMake config, the pinned UI stack and the sugar libraries.
3. Make the host name no applet: windows dock by role ([D19](../decisions/d019-host-names-no-applet.md)), the core build compiles no applets, and CI builds and starts an empty core.
4. Load applets from outside the repository: development paths, the user applets folder and the search order ([D20](../decisions/d020-applet-search-order.md)), and shipped applets signed in the app bundle ([D21](../decisions/d021-shipped-applets-in-the-app-bundle.md)).
5. Move the inspector applet's general pieces into the core as services, each in its own commit with tests. Its own panes stay in the applet.
6. Acceptance: the inspector and two existing scopes each build from their own repository against the installed SDK and load into a released Caliper. Two versions of one applet resolve correctly.
7. Move the first-party scopes to a `caliper-applets` repository, and the inspector to its own repository.

Plan: [Applets outside the core](applets-outside-the-core.md).

## Phase 2: promote proven features

Each promotion is a new service version. In order: capture and automation; the layout grid and text scale; tours; the Python sidecar; the `caliper::viz` components (annotation scene, colour by value, catalog tree); a quiet job kind. Development builds treat ImGui errors as failures.

Plan: [Platform candidates](platform-candidates.md).

## Phase 3: agent-built panels

A command layer with undo, a catalog of every data source, a spec runtime with about six primitives (series, heatmap, points, mesh, table, histogram), a spec applet kind in the loader, and an MCP server so Claude Code or Codex can build and check panels.

Plan: [Applet builder](applet-builder.md).

## Phase 4: the registry

A registry at caliperapp.ai of spec applets and signed first-party applets, a Browse tab, versioned install by id, and publishing by pull request to a registry repository ([D41](../decisions/d041-registry-at-caliperapp.md)).

Plan: [Applet registry](registry.md).

## Smaller work

- Code comments still cite sections of the deleted design documents by bare number, such as `(§6c)`. Rewrite each as its file is next touched.
- Fix stale code comments: the `sql_job` comment in `applets/embed_scope/embed_scope.cpp` (the code uses `sql_thread`), and the `CALIPER_VULKAN_SELFTEST` comment in `src/host/cuda_driver.h` (no code reads it).
- Measure EmbedScope's training step time on a Mac without the device drain, then retire the comment in `embed_scope.cpp` that says the drain dominates step time.
- GPTScope's head heatmap releases and re-creates its texture on every probe when the zero-copy pool is not in use. Create it once and update it.
- EmbedScope saves its checkpoint with `artifacts.put` on the frame thread when Save is clicked. Move the save into a job, as [D39](../decisions/d039-no-io-on-the-frame-thread.md) requires.
- Add the Vulkan mirror of the `geometry.v1_2` textured POINT row in `tests/gfx`, which exists only for Metal.
- Re-check the G15 instance-offset gate in the Metal and Vulkan backends, not only in the bridge, and isolate the Metal G1 to G12 re-gate in its own test.
- CI: cache the FetchContent downloads outside `cmake-build-release`, and cache or shallow-clone `third_party/`, which is now the longest step of a warm run.

## Not now

A WYSIWYG editor inside the app, an agent inside the app, Lua or WebAssembly applets, third-party native applets, and accounts.

## Invariants

- Data flows from tensors to pixels, one way ([D32](../decisions/d032-data-flows-one-way.md)).
- Applets supply no shaders ([D32](../decisions/d032-data-flows-one-way.md)).
- A missing capability falls back to a working slower path and says so ([D33](../decisions/d033-honest-fallbacks.md)).
- Graphics increments are verified byte for byte on both platforms ([D31](../decisions/d031-geometry-grows-by-additive-records.md)).

## Risks

- Every additive revision grows the ABI surface that must be kept working.
- Once a registry exists, an epoch bump strands every installed applet until it is rebuilt.
- A registry is a supply-chain target ([D22](../decisions/d022-third-party-applet-signing.md)).

## Open questions

- **Audience.** Today the audience is collaborators who build Caliper from source, and the contracts are sized so that they also work for end users who install a prebuilt bundle. Revisit this when the first such user appears.
- **Signing third-party applets on macOS.** A notarized host either carries the `disable-library-validation` entitlement or loads only applets signed by its own team. The choice must be made before the first third-party applet ships; the options are in [Applets outside the core](applets-outside-the-core.md#signing-on-macos).
- **Applet shaders.** Whether applets still supply no shaders once agent-built panels exist.
- **The OpenGL fallback.** Whether to replace GLEW with a GLAD 3.3-core loader, or to remove the fallback.
- **Older epochs.** Whether the host must accept applets built for epoch N−1 before the first outside applet ships.
- **A second native host.** Whether a second program that embeds `libcaliper` is still needed once applets live outside the core.
- **OpenGLlama.** Whether to revive the archived OpenGLlama applet in its own repository, or delete it.

## Undecided

These ideas come from earlier designs and are not built. Each needs a decision before it enters a phase.

- **Reviewer bundle.** One download that runs an applet and its demo on a machine that never built Caliper.
- **The em-controller.** An actor-critic policy that steers FieldScope's plasma, as an applet of its own.
- **Digital twins.** A population twin built on InstanceScope's unused per-variant reduction, and a machine thermal twin fed by `caliper.feed.v1`.
- **Research publishing beyond export.** A Python DLPack path into the geometry bridge, and applets written in Python.
- **Artifacts redesign.** A checkpoint browser with previews and resuming training from a checkpoint.
- **Applet dependency packaging.** Whether an applet built outside the core may link its own heavy dependencies, such as DuckDB or llama.cpp, and what happens when two applets each carry a copy. The proposal for large runtimes such as libtorch is runtime packs ([D5](../decisions/d005-runtime-packs.md)): the host installs and manages each runtime once instead of every applet bundling its own multi-gigabyte copy, and a process loads one libtorch for the whole session. The stated limit is that two applets that need different torch versions cannot share a session. No pack manager exists; the build downloads torch directly.
- **The `caliper` CLI.** `new`, `dev`, `install`, `package` and `publish` as subcommands of the host binary rather than a separate program, so there is one thing to install, as with the `code` command of Visual Studio Code ([D9](../decisions/d009-cli-as-host-subcommands.md)). The host's `main` takes no arguments yet, and whether `new` is needed depends on the [applet builder](applet-builder.md).
- **`.caliperapp` bundles.** A bundle folder with per-platform binaries, and manifest keys for platforms, authors, licence and runtimes.
- **Hot reload.** Reloading a rebuilt applet, with `save_state`, `load_state`, `on_suspend` and `on_resume` entry points.
- **SDK conformance.** A conformance checker, and a set of reference applets built against each SDK release.
- **Isolation and reach.** Applets in a separate process, scripting bindings, and a Linux build.
- **Live streaming.** Several viewers per session, f16, bf16 and RGB zero-copy kernels, a smaller wheel without the stores, and an arm64 wheel job.
- **Embedding.** Several canvases per core, an `api_version` query and a canvas-info query.
- **Zero-copy.** Device-backed `alloc_shared` on Metal, a pool of exportable Vulkan images, and Linux Vulkan through `OPAQUE_FD` handles.
- **Geometry.** Thick lines and ribbons, MSAA, a culling flag and volumetric rendering.
- **Telemetry sources.** Feed channels registered by applets or read from serial, UDP or MQTT.
- **GPTScope follow-ups.** Point picking and a learned projection in ThoughtSpace, induction-head metrics and activation patching.
- **Instrument inputs.** Live input slots bound to applet manifests through a `live_input` service.
