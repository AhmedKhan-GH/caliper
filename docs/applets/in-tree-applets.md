# In-Tree Applets

Three applets live under `applets/`. One tutorial applet and an embedding example live under `examples/`. Every applet requires `caliper.ui.v1` and `caliper.log.v1`, so the table leaves them out. The manifest format is in [Manifest](manifest.md), and the services are in the [catalog](services/service-catalog.md).

The core build compiles no applet. Each applet folder builds on its own against the installed SDK, as [SDK package](sdk-package.md) describes. `CALIPER_BUILD_APPLETS=ON` also builds the `applets/` folders inside the core build.

| Applet | Manifest id (version) | Requires | Optional | Fallback |
|---|---|---|---|---|
| EmbedScope | `dev.caliper.embed-scope` (0.1.0) | `caliper.jobs.v1`, `caliper.device.v1` | `caliper.metrics.v1`, `caliper.tensor_bridge.v1`, `caliper.artifacts.v1`, `caliper.data.v1` | An "absent (ok)" line in each affected panel |
| GPTScope | `dev.caliper.gpt-scope` (0.2.0) | `caliper.jobs.v1`, `caliper.device.v1` | `caliper.metrics.v1`, `caliper.tensor_bridge.v1`, `caliper.artifacts.v1` | CPU staging in Heads when stream handoff is absent |
| PulseScope | `dev.caliper.pulse-scope` (0.1.0) | `caliper.jobs.v1`, `caliper.feed.v1` | None | "telemetry unavailable on this host" |

GPTScope uses the zero-copy pattern described in [Zero-copy](../python/zero-copy-tensors.md). A worker thread computes with torch and writes into tensors allocated from an [`ExportablePool`](adapters.md#exportable_poolhpp). The renderer then draws the attention matrix from that same memory through [`caliper.tensor_bridge.v1`](services/caliper-tensor-bridge-v1.md) without a copy. The window states whether the zero-copy path drew the current frame, or which fallback ran ([Platform rules](../introduction/architecture.md#design-decisions)).

## EmbedScope

EmbedScope trains a small MNIST convolutional network whose last hidden layer is a learned 3-D bottleneck, `Linear(64, 3)`. It draws the test-set embeddings as a live ImPlot3D scatter in ten classes. A gray blob splits into ten lobes as the network learns. Hovering a point shows its digit.

The applet has four windows: Training, Cloud, Tensors and Data. The Data window shows class centroids and the misclassified count, computed in SQL. Training runs off the frame thread.

The optional services add these features:

- `caliper.metrics.v1`: the run's loss and accuracy.
- `caliper.tensor_bridge.v1`: conv kernels and weights drawn on the GPU, and the hover image.
- `caliper.artifacts.v1`: Save and Load. Load restores the visuals without training.
- `caliper.data.v1`: the SQL panels.

The applet rebuilds the SQL table as a `TEMP` table on a thread it owns. `CALIPER_EMBED_AUTOTRAIN=1` starts training on the first frame. When the host does not grant the stream-ordered handoff, the applet stages each tensor on the CPU. The rest of the applet keeps running when an optional service is absent.

## GPTScope

GPTScope is the flagship applet. It shows the internals of a character-level GPT that trains live on TinyShakespeare (4 layers, 4 heads, 128 dimensions, block size 128). Training runs in a job. Seven windows each answer one question about the model:

- **Logit Lens** asks when the model decides. Rows are depths and columns are positions of a fixed probe text. Each cell is that depth's top prediction, green when it matches the true next character.
- **Heads** asks what each head became. It scatters all 16 heads by mean attended distance against attention entropy, colored by layer. Selecting a head draws its attention matrix as a texture.
- **Embeddings** asks what the model has learned about characters. It shows a 3-D PCA of the token embedding, with each point drawn as its glyph.
- **Residual** asks who writes what, and where. It compares attention and MLP writes to the residual stream, with per-group gradient norms.
- **Sample** asks what the model believes as it speaks. Generated text is colored by the model's confidence in each character.
- **ThoughtSpace** draws the residual stream as a 3-D constellation in an ImPlot3D scatter of a CPU subsample (about 8,000 stations). Every token's residual state at every depth is a point, and interpolated trails join the depths. A fixed random orthonormal projection means that all on-screen motion is the model changing, not the basis.
- **Training** holds the controls, the loss and validation perplexity curves, and Save and Load.

Optional `caliper.metrics.v1` streams the run, `caliper.tensor_bridge.v1` draws the head heatmap and `caliper.artifacts.v1` saves and loads the model. The worker runs the ThoughtSpace probe and projection only while its window is open. `CALIPER_GPT_AUTOTRAIN=1` presses Train on the first frame.

The Heads window shows a disabled line when `caliper.tensor_bridge.v1` is absent. It stages its texture on the CPU when the host does not grant the stream-ordered handoff. Save and Load are disabled without `caliper.artifacts.v1`.

## PulseScope

PulseScope shows this machine's own vitals. It enumerates every [`caliper.feed.v1`](services/caliper-feed-v1.md) channel the host provides. It draws one strip chart per channel from a job-thread poller at about 10 Hz.

Each chart has the current value and units. A "last sample N.N s ago" label marks a stale channel. A cumulative "lost M samples" counter makes the feed's honest-loss contract visible.

The applet uses the CPU and ImPlot only, with no tensors and no GPU. A host with no channels, such as Linux, shows only the fallback line.

## Tutorial Applets

This applet lives in `examples/`. It builds on its own against the installed SDK. The core build also builds it with the tests, because the tests load it.

- **Hello** (`dev.caliper.hello`, `examples/hello`). A UI-only applet that shows the framebuffer size and DPI scale. It requires only `caliper.ui.v1` and `caliper.log.v1`, and it is the fixture the loader tests load. `CALIPER_HELLO_CRASH` makes it fault on purpose, to exercise the crash guard. The [first applet tutorial](first-applet.md) builds it.

## Embedding Example

`examples/embed_host` is a second host of the core, not an applet. It is a bare native window (AppKit on macOS, Win32 on Windows) that makes the five embed calls. It loads one applet by manifest id.

It links `caliper::embed` and none of the `caliper` executable. Outside the repository, it builds against the installed `caliper-embed` package with only `find_package(caliper-embed)`. `CALIPER_EMBED_OFFSCREEN=1` runs it in an offscreen mode that opens no window. `CALIPER_EMBED_EXIT_AFTER=<seconds>` closes it automatically. No Linux host exists. See [Embedding](../web/embedding-api.md) and [Embed Caliper in a program](../web/program-embedding.md).
