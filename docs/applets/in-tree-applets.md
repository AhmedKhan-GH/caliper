# In-Tree Applets

The `main` branch holds nine applets under `applets/`. Two tutorial applets and an embedding example live under `examples/`. Every applet requires `caliper.ui.v1` and `caliper.log.v1`, so the table leaves them out. The manifest format is in [Manifest](manifest.md), and the services are in the [catalog](services/service-catalog.md).

The core build compiles no applet. Each applet folder builds on its own against the installed SDK, as [SDK package](sdk-package.md) describes. `CALIPER_BUILD_APPLETS=ON` also builds the `applets/` folders inside the core build.

| Applet | Manifest id (version) | Requires | Optional | Fallback |
|---|---|---|---|---|
| EmbedScope | `dev.caliper.embed-scope` (0.1.0) | `caliper.jobs.v1`, `caliper.device.v1` | `caliper.metrics.v1`, `caliper.tensor_bridge.v1`, `caliper.artifacts.v1`, `caliper.data.v1` | An "absent (ok)" line in each affected panel |
| FieldScope | `dev.caliper.field-scope` (0.1.0) | `caliper.jobs.v1`, `caliper.device.v1` | `caliper.tensor_bridge.v1`, `caliper.geometry.v1` | CPU scatter of 10,000 points |
| FlowScope | `dev.caliper.flow-scope` (0.1.0) | `caliper.jobs.v1`, `caliper.device.v1` | `caliper.tensor_bridge.v1`, `caliper.geometry.v1` | CPU scatter of 10,000 points |
| GPTScope | `dev.caliper.gpt-scope` (0.2.0) | `caliper.jobs.v1`, `caliper.device.v1` | `caliper.metrics.v1`, `caliper.tensor_bridge.v1`, `caliper.artifacts.v1` | ImPlot3D scatter in ThoughtSpace |
| InstanceScope | `dev.caliper.instance-scope` (0.1.0) | `caliper.jobs.v1`, `caliper.device.v1` | `caliper.tensor_bridge.v1` and `v1_2`, `caliper.geometry.v1_1`, `v1_2` and `v1_3` | One gem on the non-instanced path |
| MeshScope | `dev.caliper.mesh-scope` (0.1.0) | `caliper.jobs.v1`, `caliper.device.v1` | `caliper.tensor_bridge.v1` and `v1_2`, `caliper.geometry.v1_1`, `caliper.export.v1` | CPU heatmap of the error grid |
| PulseScope | `dev.caliper.pulse-scope` (0.1.0) | `caliper.jobs.v1`, `caliper.feed.v1` | None | "telemetry unavailable on this host" |
| SculptScope | `dev.caliper.sculpt-scope` (0.1.0) | `caliper.jobs.v1`, `caliper.device.v1` | `caliper.tensor_bridge.v1`, `caliper.geometry.v1` | ImPlot3D scatter of 10,000 points |
| TwinScope | `dev.caliper.twin-scope` (0.2.0) | `caliper.jobs.v1`, `caliper.device.v1` | `caliper.tensor_bridge.v1` and `v1_2`, `caliper.geometry.v1_1` and `v1_2`, `caliper.export.v1` | A ladder of four rungs |

Several applets share one pattern, described in [Zero-copy](../python/zero-copy-tensors.md). A worker thread computes with torch and writes into tensors allocated from an [`ExportablePool`](adapters.md#exportable_poolhpp). The renderer then draws from that same memory through [`caliper.geometry.v1`](services/caliper-geometry-v1.md) without a copy. Each of these applets states in its window whether the zero-copy path drew the current frame. If it did not, the window says which fallback ran ([Platform rules](../introduction/architecture.md#design-decisions)).

## EmbedScope

EmbedScope trains a small MNIST convolutional network whose last hidden layer is a learned 3-D bottleneck, `Linear(64, 3)`. It draws the test-set embeddings as a live ImPlot3D scatter in ten classes. A gray blob splits into ten lobes as the network learns. Hovering a point shows its digit.

The applet has four windows: Training, Cloud, Tensors and Data. The Data window shows class centroids and the misclassified count, computed in SQL. Training runs off the frame thread.

The optional services add these features:

- `caliper.metrics.v1`: the run's loss and accuracy.
- `caliper.tensor_bridge.v1`: conv kernels and weights drawn on the GPU, and the hover image.
- `caliper.artifacts.v1`: Save and Load. Load restores the visuals without training.
- `caliper.data.v1`: the SQL panels.

The applet rebuilds the SQL table as a `TEMP` table on a thread it owns. `CALIPER_EMBED_AUTOTRAIN=1` starts training on the first frame. When the host does not grant the stream-ordered handoff, the applet stages each tensor on the CPU. The rest of the applet keeps running when an optional service is absent.

## FieldScope

FieldScope simulates charged-particle fields in the window "FieldScope: EM Field". It uses 200,000 particles on a GPU and 20,000 otherwise.

The default mode is a self-consistent electrostatic particle-in-cell plasma. Magnetized plasma shows vortices that orbit and merge. Four prescribed-field modes show single-particle motion: uniform (gyration), mirror (a magnetic bottle), toroidal (confined orbits) and dipole (trapped belts). Four initial conditions are available: clumps, ring, blob and two-stream.

Left-drag perturbs the plasma, right-drag orbits and the wheel zooms. The window labels the CPU scatter as a fallback.

## FlowScope

FlowScope shows a flow field of one million particles on a GPU (50,000 on a CPU) in the window "FlowScope: Field". The renderer reads the simulation buffers in place. Left-drag pushes the field with an impulse along the cursor ray, right-drag orbits and the wheel zooms.

In the fallback, the status line reads "CPU fallback". It names the cause: a CPU-only torch, a missing geometry service or an unavailable pool.

## GPTScope

GPTScope is the flagship applet. It shows the internals of a character-level GPT that trains live on TinyShakespeare (4 layers, 4 heads, 128 dimensions, block size 128). Training runs in a job. Seven windows each answer one question about the model:

- **Logit Lens** asks when the model decides. Rows are depths and columns are positions of a fixed probe text. Each cell is that depth's top prediction, green when it matches the true next character.
- **Heads** asks what each head became. It scatters all 16 heads by mean attended distance against attention entropy, colored by layer. Selecting a head draws its attention matrix as a texture.
- **Embeddings** asks what the model has learned about characters. It shows a 3-D PCA of the token embedding, with each point drawn as its glyph.
- **Residual** asks who writes what, and where. It compares attention and MLP writes to the residual stream, with per-group gradient norms.
- **Sample** asks what the model believes as it speaks. Generated text is colored by the model's confidence in each character.
- **ThoughtSpace** draws the residual stream as a 3-D constellation. Every token's residual state at every depth is a point, and interpolated trails join the depths. A fixed random orthonormal projection means that all on-screen motion is the model changing, not the basis. Points are colored by loss, by logit-lens confidence or by depth.
- **Training** holds the controls, the loss and validation perplexity curves, and Save and Load.

Optional `caliper.metrics.v1` streams the run, `caliper.tensor_bridge.v1` draws the head heatmap and `caliper.artifacts.v1` saves and loads the model. The worker runs the ThoughtSpace probe and projection only while its window is open. `CALIPER_GPT_AUTOTRAIN=1` presses Train on the first frame.

Without a capability, ThoughtSpace falls back to an ImPlot3D scatter of the residual stations. It labels the scatter "CPU fallback" and gives the reason. The Heads window shows a disabled line when `caliper.tensor_bridge.v1` is absent. It stages its texture on the CPU when the host does not grant the stream-ordered handoff. Save and Load are disabled without `caliper.artifacts.v1`.

## InstanceScope

InstanceScope shows instancing in its plainest form. A field of procedural gems (1,000 by default, up to 5,000) bobs and spins in a traveling wave. The renderer draws the whole field with one instanced draw call and no copies of the mesh.

The applet recomputes rigid poses, an `(N, 16)` tensor, and a tint per gem, an `(N,)` tensor, on the device each frame. It imports both zero-copy. The tint uses a fixed `[-1, 1]` MAGMA window, so it never saturates.

When the backend cannot instance, the window says so and draws a single gem through the non-instanced path. With no geometry service, pool or GPU, it says that instead.

## MeshScope

MeshScope is the exemplar for `caliper.geometry.v1_1`. A small MLP (hidden width 64, minibatch 512) learns a 2-D target surface live in the window "MeshScope: Surface". The frame draws the surface as Lambert-lit indexed triangles colored by squared error through the MAGMA colormap. It adds a wireframe overlay and the training minibatch as additive points, so the frame uses three `draw_primitives` calls.

The user can paint the target with a Gaussian brush and watch the surface chase it. With `caliper.export.v1`, the window offers "Export figure (4K)" and a 10-second recording at 720p and 30 fps. Without it, the export buttons are disabled, with a tooltip.

The CPU heatmap fallback locks input.

## PulseScope

PulseScope shows this machine's own vitals. It enumerates every [`caliper.feed.v1`](services/caliper-feed-v1.md) channel the host provides. It draws one strip chart per channel from a job-thread poller at about 10 Hz.

Each chart has the current value and units. A "last sample N.N s ago" label marks a stale channel. A cumulative "lost M samples" counter makes the feed's honest-loss contract visible.

The applet uses the CPU and ImPlot only, with no tensors and no GPU. A host with no channels, such as Linux, shows only the fallback line.

## SculptScope

SculptScope trains a small MLP live to map latent codes to 3-D points that match a target shape. The target is a sphere, torus, helix or two lobes. The cloud has 150,000 points on a GPU and 20,000 otherwise. The window is "SculptScope: Learned Cloud".

A formless blob flows into the shape as the network learns. Color tracks per-point motion, so a converged cloud dims. The final layer writes directly into the pool-allocated tensor that the renderer draws, with zero copies between the model and the picture.

## TwinScope

TwinScope shows a synthetic heat twin on a UV-mapped, finned housing. The housing is an OBJ asset loaded with [`caliper::obj`](adapters.md#objhpp). A simulation flows heat over its surface while a small network learns the field live. The simulation is a synthetic heat model, not a finite-element analysis.

The window "TwinScope: the surface twin" splits "physics" and "belief" side by side, drawn zero-copy from a 256 by 256 baked texture. It uses the orbit camera and cursor picking from [`orbit_camera.hpp`](adapters.md#orbit_camerahpp). A toggle switches between the textured draw (`caliper.geometry.v1_2`) and a per-vertex draw (`caliper.geometry.v1_1`). The toggle shows the resolution the texture adds. With `caliper.export.v1`, the window adds the same figure and recording buttons as MeshScope.

Without a capability, the applet steps down a ladder of four rungs. It names the active rung in the window and in the log:

1. Zero-copy textured draw.
2. Textured draw with the texture staged on the CPU, when the host does not grant the stream-ordered handoff.
3. Per-vertex draw, when textured geometry is absent.
4. CPU heatmap, when no geometry service exists.

## Tutorial Applets

These applets live in `examples/`. Each builds on its own against the installed SDK. The core build also builds them with the tests, because the tests load them: Hello always, and SineScope in a build with libtorch.

- **Hello** (`dev.caliper.hello`, `examples/hello`). A UI-only applet that shows the framebuffer size and DPI scale. It requires only `caliper.ui.v1` and `caliper.log.v1`, and it is the fixture the loader tests load. `CALIPER_HELLO_CRASH` makes it fault on purpose, to exercise the crash guard. The [first applet tutorial](first-applet.md) builds it.
- **SineScope** (`dev.example.sine-scope`, `examples/sine_scope`). A two-layer MLP with 32 hidden units learns `sin(x)` live. It shows the loss curve, the prediction against the target and the first-layer weights as a heatmap. It requires `caliper.jobs.v1` and `caliper.device.v1`, and optionally `caliper.tensor_bridge.v1` and `caliper.metrics.v1`. It builds only in a configuration that links libtorch. The [first ML applet tutorial](first-ml-applet.md) builds it.

## Embedding Example

`examples/embed_host` is a second host of the core, not an applet. It is a bare native window (AppKit on macOS, Win32 on Windows) that makes the five embed calls. It loads one applet by manifest id, which defaults to `dev.caliper.instance-scope`.

It links `caliper::embed` and none of the `caliper` executable. Outside the repository, it builds against the installed `caliper-embed` package with only `find_package(caliper-embed)`. `CALIPER_EMBED_OFFSCREEN=1` runs it in an offscreen mode that opens no window. `CALIPER_EMBED_EXIT_AFTER=<seconds>` closes it automatically. No Linux host exists. See [Embedding](../web/embedding-api.md) and [Embed Caliper in a program](../web/program-embedding.md).

## Archived Applets

`applets/legacy/` and `applets/legacy-dev/` hold archived applets that the default build skips. `CALIPER_BUILD_OPENGLLAMA` builds the archived OpenGLlama applet, which needs `CALIPER_WITH_LLAMA` on.
