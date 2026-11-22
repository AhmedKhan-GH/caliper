# Applets

This page lists the applets on the `main` branch. Each entry gives the
applet's manifest id, what it shows, the service revisions it asks for and what
it does when a capability is missing. The service lists come from each
applet's `*.caliper.toml`; every applet also requires `caliper.ui.v1` and
`caliper.log.v1`, which the entries leave out. The manifest format is in
[Manifest](manifest.md), and the services are in the [catalog](services.md).

Nine applets live under `applets/`. Two tutorial applets and an embedding example
live under `examples/`. The core build compiles no applet: each applet folder
builds on its own against the installed SDK, as [SDK package](sdk-package.md)
describes, and `CALIPER_BUILD_APPLETS=ON` also builds the `applets/` folders
inside the core build.

Several applets share one pattern, described in
[Zero-copy](../python/zero-copy-tensors.md). A worker thread computes with torch and
writes into tensors allocated from an
[`ExportablePool`](adapters.md#exportable_poolhpp), and the renderer draws from
that same memory through [`caliper.geometry.v1`](service-geometry-v1.md) without a
copy. Each of these applets states in its window whether the zero-copy path
actually drew the current frame, and says which fallback ran when it did not
([Platform rules](../introduction/architecture.md#platform-rules)).

## EmbedScope

- **Manifest id:** `dev.caliper.embed-scope`, version 0.1.0.
- **Shows:** a small MNIST convolutional network whose last hidden layer is a
  learned 3-D bottleneck, `Linear(64, 3)`. The test-set embeddings are drawn as a
  live ImPlot3D scatter in ten classes. Training runs off the frame thread, and a
  single gray blob splits into ten lobes as the network learns. Hovering a point
  shows its digit. Windows: Training, Cloud, Tensors and Data. The Data window
  shows class centroids and the misclassified count, computed in SQL.
- **Services:** requires `caliper.jobs.v1` and `caliper.device.v1`. Optional:
  `caliper.metrics.v1` (the run's loss and accuracy), `caliper.tensor_bridge.v1`
  (conv kernels and weights drawn on the GPU, and the hover image),
  `caliper.artifacts.v1` (Save and Load; Load restores the visuals without
  training) and `caliper.data.v1` (the SQL panels).
- **How it uses them:** the SQL table is rebuilt as a `TEMP` table on a thread the
  applet owns, never on the frame thread. Display tensors are normalized to a
  fixed range on the worker, and each texture is created once and then updated in
  place. `CALIPER_EMBED_AUTOTRAIN=1` starts training on the first frame.
- **Without a capability:** each missing optional service leaves a visible
  "absent (ok)" line in the panel that needs it, and the rest keeps running. When
  the host does not grant the stream-ordered handoff, the applet stages each
  tensor on the CPU instead of draining the device on the frame thread.

## FieldScope

- **Manifest id:** `dev.caliper.field-scope`, version 0.1.0.
- **Shows:** a charged-particle field simulator in the window "FieldScope: EM
  Field", with 200,000 particles on a GPU and 20,000 otherwise. The default mode
  is a self-consistent electrostatic particle-in-cell plasma: particles deposit
  charge on a 48-cube grid with cloud-in-cell weights, an FFT free-space Poisson
  solve gives their own field, and a Boris pusher advances them. Magnetized
  plasma shows vortices that orbit and merge. Four prescribed-field modes show
  single-particle motion: uniform (gyration), mirror (a magnetic bottle),
  toroidal (confined orbits) and dipole (trapped belts). Four initial conditions
  are available: clumps, ring, blob and two-stream. Left-drag perturbs the
  plasma, right-drag orbits and the wheel zooms.
- **Services:** requires `caliper.jobs.v1` and `caliper.device.v1`. Optional:
  `caliper.tensor_bridge.v1` and `caliper.geometry.v1`.
- **Without a capability:** a subsampled CPU scatter of 10,000 points, labeled as
  a fallback in the window.

## FlowScope

- **Manifest id:** `dev.caliper.flow-scope`, version 0.1.0.
- **Shows:** a flow field of one million particles on a GPU (50,000 on a CPU) in
  the window "FlowScope: Field". The simulation writes into two exportable-pool
  buffers that swap each step, and the renderer's instanced point pass reads the
  same memory in place. Left-drag pushes the field with an impulse along the
  cursor ray, right-drag orbits and the wheel zooms.
- **Services:** requires `caliper.jobs.v1` and `caliper.device.v1`. Optional:
  `caliper.tensor_bridge.v1` and `caliper.geometry.v1`.
- **Without a capability:** a subsampled CPU scatter of 10,000 points. The status
  line reads "CPU fallback" and says whether the cause is a CPU-only torch, a
  missing geometry service or an unavailable pool.

## GPTScope

- **Manifest id:** `dev.caliper.gpt-scope`, version 0.2.0. GPTScope is the
  flagship applet. It
  was written the way an outside author would write an applet, on the full
  service stack from the start, so it tests the platform better than older code
  moved onto the services would.
- **Shows:** mechanistic interpretability of a character-level GPT that trains
  live on TinyShakespeare (4 layers, 4 heads, 128 dimensions, block size 128).
  Seven windows each answer one question about the model:
    - **Logit Lens** asks when the model decides. Rows are depths and columns are
      positions of a fixed probe text, and each cell is that depth's top
      prediction, green when it matches the true next character.
    - **Heads** asks what each head became. A scatter of all 16 heads by mean
      attended distance against attention entropy, colored by layer. Selecting a
      head draws its attention matrix as a texture.
    - **Embeddings** asks what the model has learned about characters. A 3-D PCA
      of the token embedding, with each point drawn as its glyph.
    - **Residual** asks who writes what, and where. Attention against MLP writes
      to the residual stream, with per-group gradient norms.
    - **Sample** asks what the model believes as it speaks. Generated text is
      colored by the model's confidence in each character.
    - **ThoughtSpace** draws the residual stream as a 3-D constellation. Every
      token's residual state at every depth is a point, and interpolated trails
      join the depths so that each token is a thread through the network. With
      the default sizes the window draws 232,800 points. A fixed random
      orthonormal projection means that all on-screen motion is the model
      changing, not the basis. Points are colored by loss, by logit-lens
      confidence or by depth.
    - **Training** holds the controls, the loss and validation perplexity curves,
      and Save and Load.
- **Services:** requires `caliper.jobs.v1` and `caliper.device.v1`. Optional:
  `caliper.metrics.v1` (the run is streamed), `caliper.tensor_bridge.v1` (the
  head heatmap) and `caliper.artifacts.v1` (Save and Load of the model).
  Training runs in a job.
- **Pay per look:** the worker runs the ThoughtSpace probe and projection only
  while its window is open, at about 1 Hz. The lens, heads and residual probes
  run on their own 1 Hz cadence, and the embedding PCA every 5 seconds.
  `CALIPER_GPT_AUTOTRAIN=1` presses Train on the first frame.
- **Without a capability:** ThoughtSpace falls back to an ImPlot3D scatter of the
  residual stations and labels it "CPU fallback", with the reason. The Heads
  window shows a disabled line when `caliper.tensor_bridge.v1` is absent, and its
  texture is CPU-staged when the host does not grant the stream-ordered handoff.
  Save and Load are disabled without `caliper.artifacts.v1`.

## InstanceScope

- **Manifest id:** `dev.caliper.instance-scope`, version 0.1.0.
- **Shows:** instancing in its plainest form. A field of procedural gems (1,000 by
  default, up to 5,000) bobs and spins in a traveling wave, and the whole field is
  drawn with one instanced draw call and no copies of the mesh. Rigid poses, an
  `(N, 16)` tensor, and a tint per gem, an `(N,)` tensor, are recomputed on the
  device each frame and imported zero-copy. The tint uses a fixed `[-1, 1]` MAGMA
  window, so it never saturates.
- **Services:** requires `caliper.jobs.v1` and `caliper.device.v1`. Optional:
  `caliper.tensor_bridge.v1` and `v1_2`, and `caliper.geometry.v1_1`, `v1_2` and
  `v1_3`. The instance tail is the `v1_3` record.
- **Without a capability:** when the backend cannot instance, the window says so
  and draws a single gem through the non-instanced path. With no geometry service,
  pool or GPU it says that instead.

## MeshScope

- **Manifest id:** `dev.caliper.mesh-scope`, version 0.1.0.
- **Shows:** a small MLP (hidden width 64, minibatch 512) learns a 2-D target
  surface live. Every optimizer step, its prediction on a 72 by 72 grid is written
  into imported device tensors and drawn in the same frame as Lambert-lit indexed
  triangles colored by squared error through the MAGMA colormap, with a wireframe
  overlay and the training minibatch as additive points. The window is "MeshScope:
  Surface". The user can paint the target with a Gaussian brush and watch the
  surface chase it. This is the exemplar for `caliper.geometry.v1_1`, and its
  frame uses three `draw_primitives` calls: the surface, the wireframe and the
  sample points.
- **Services:** requires `caliper.jobs.v1` and `caliper.device.v1`. Optional:
  `caliper.tensor_bridge.v1` and `v1_2`, `caliper.geometry.v1_1` and
  `caliper.export.v1`. With export, the window offers "Export figure (4K)" and a
  10-second recording at 720p and 30 fps.
- **Without a capability:** an input-locked CPU heatmap of the error grid. The
  export buttons are disabled, with a tooltip, when the export capability is
  unset.

## PulseScope

- **Manifest id:** `dev.caliper.pulse-scope`, version 0.1.0.
- **Shows:** this machine's own vitals. PulseScope enumerates every
  [`caliper.feed.v1`](service-feed-v1.md) channel the host provides and draws one
  strip chart per channel from a job-thread poller at about 10 Hz. Each chart has
  the current value and units, a "last sample N.N s ago" label that marks a stale
  channel, and a cumulative "lost M samples" counter that makes the feed's
  honest-loss contract visible.
- **Services:** requires `caliper.jobs.v1` and `caliper.feed.v1`. It uses the CPU
  and ImPlot only, with no tensors and no GPU.
- **Without a capability:** a host with no channels, such as Linux, which has no provider, shows the single
  line "telemetry unavailable on this host" and nothing else.

## SculptScope

- **Manifest id:** `dev.caliper.sculpt-scope`, version 0.1.0.
- **Shows:** a small MLP maps latent codes to 3-D points, 150,000 on a GPU and
  20,000 otherwise, and trains live to match a target shape. It uses Adam, real
  backpropagation and an energy-distance loss against a target of sphere, torus,
  helix or two lobes. The network is three `Linear(128)` layers with SiLU and a
  final `Linear(128, 3)`. The final layer's matrix multiply runs as `addmm_out`
  directly into the pool-allocated tensor that the renderer draws, so the buffer
  the last layer writes is the buffer the GPU draws, with zero copies between the
  model and the picture. A formless blob flows into the shape as the network
  learns, and color tracks per-point motion, so a converged cloud dims. The window
  is "SculptScope: Learned Cloud".
- **Services:** requires `caliper.jobs.v1` and `caliper.device.v1`. Optional:
  `caliper.tensor_bridge.v1` and `caliper.geometry.v1`.
- **Without a capability:** a subsampled CPU scatter of 10,000 points in an
  ImPlot3D plot.

## TwinScope

- **Manifest id:** `dev.caliper.twin-scope`, version 0.2.0.
- **Shows:** a synthetic heat twin on a UV-mapped, finned housing. The housing is
  an OBJ asset loaded with [`caliper::obj`](adapters.md#objhpp), subdivided twice
  into a simulation mesh with 16 times the vertices of the render mesh. A batched
  cotangent-Laplacian simulation flows heat over that surface while a small
  network learns the field live. The simulation is a synthetic heat model, not a
  finite-element analysis. The main view splits "physics" and "belief" side by
  side, drawn zero-copy at texture resolution from a 256 by 256 baked texture. A
  toggle switches between the textured draw (`caliper.geometry.v1_2`) and a
  per-vertex draw (`caliper.geometry.v1_1`), which shows the resolution the
  texture adds. The window is "TwinScope: the surface twin", and it uses the
  orbit camera and cursor picking from
  [`orbit_camera.hpp`](adapters.md#orbit_camerahpp).
- **Services:** requires `caliper.jobs.v1` and `caliper.device.v1`. Optional:
  `caliper.tensor_bridge.v1` and `v1_2`, `caliper.geometry.v1_1` and `v1_2`, and
  `caliper.export.v1`, which adds the same figure and recording buttons as
  MeshScope.
- **Without a capability:** a ladder of four rungs, named in the window and in the
  log. Zero-copy textured is the first. Without the stream-ordered handoff the
  texture is staged on the CPU. Without textured geometry the surface draws per
  vertex. Without a geometry service at all it draws a CPU heatmap.

## Tutorial applets

These live in `examples/`. Each builds on its own against the installed SDK, and
the core build also builds them with the tests, because the tests load them:
Hello always, and SineScope in a build with libtorch.

- **Hello** (`dev.caliper.hello`, `examples/hello`). A UI-only applet that shows
  the framebuffer size and DPI scale. It requires only `caliper.ui.v1` and
  `caliper.log.v1`, and it is the fixture the loader tests load.
  `CALIPER_HELLO_CRASH` makes it fault on purpose, to exercise the crash guard.
  The [first applet tutorial](../tutorials/first-applet.md) builds it.
- **SineScope** (`dev.example.sine-scope`, `examples/sine_scope`). A two-layer MLP
  with 32 hidden units learns `sin(x)` live: the loss curve, the prediction
  against the target and the first-layer weights as a heatmap. It requires
  `caliper.jobs.v1` and `caliper.device.v1`, and optionally
  `caliper.tensor_bridge.v1` and `caliper.metrics.v1`. It builds only in a
  configuration that links libtorch. The
  [first ML applet tutorial](../tutorials/first-ml-applet.md) builds it.

## Embedding example

`examples/embed_host` is a second host of the core, not an applet. It is a bare
native window (AppKit on macOS, Win32 on Windows) that makes the five embed calls
and loads one applet by manifest id, which defaults to
`dev.caliper.instance-scope`. It links `caliper::embed` and none of the `caliper`
executable, and it builds outside the repository against the installed
`caliper-embed` package with only `find_package(caliper-embed)`; inside the core
build it links the in-tree target. `CALIPER_EMBED_OFFSCREEN=1` runs it in an
offscreen mode that opens no window, and `CALIPER_EMBED_EXIT_AFTER=<seconds>`
closes it automatically. There is no Linux host. See [Embedding](embedding.md)
and [Embed Caliper in a program](../guides/embed-caliper.md).

## Archived applets

`applets/legacy/` and `applets/legacy-dev/` hold archived applets that the default
build skips. `CALIPER_BUILD_OPENGLLAMA` builds the archived OpenGLlama applet,
which needs `CALIPER_WITH_LLAMA` on.
