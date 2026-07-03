# Caliper Platform

Caliper is a desktop platform for **ML-native, GPU-resident visualization running in the same frame loop as training**. Applets are shared libraries the host loads at runtime over a small, frozen C ABI; the host owns the window, the rendering backend, and the ImGui/ImPlot contexts, while each applet renders its own UI and drives its own compute.

The platform's reason to exist is the tensor bridge: weights, activations, and saliency maps that live on the GPU (CUDA or MPS) become an `ImGui::Image` *this frame* — no TensorBoard round-trip, no PNG encode, no Python, and on the native backends no CPU staging. That capability is preserved by keeping applets in-process and productized as the `caliper.tensor_bridge.v1` service.

**EmbedScope** (`applets/embed_scope/`) is *the* exemplar — the platform's showcase and the template to copy: a small MNIST net with a **3-D embedding bottleneck** whose test-set embeddings are drawn as a live ImPlot3D scatter (renderer-agnostic, so it works on both Metal and GL) that splits from one blob into ten colored lobes *while training runs*. It exercises **every service** — training off the frame thread via `caliper.jobs.v1` on the host-negotiated device, scalars to the `caliper.metrics.v1` Runs dashboard, hover-a-point digit textures through `caliper.tensor_bridge.v1`, model Save/Load through `caliper.artifacts.v1` (Load skips training), and live SQL over the embedding table through `caliper.data.v1` (class centroids, misclassified counts). No private hooks; every capability it uses is one an out-of-tree applet has.

Earlier exemplars that drove the architecture — **GPTScope** (the Phase-2 flagship mini-GPT on TinyShakespeare), MLScope, and SignalScope — are archived under `applets/legacy-dev/` (not built or loaded); Ahmed's own applets (CircuitNet, OpenGllama, RepNet Demo) are archived under `applets/legacy/` awaiting their own repositories in Phase 3.

This wiki is the docs-as-code companion to `PLATFORM.md` (the governing spec at the repo root). It is organized along the [Diátaxis](https://diataxis.fr/) axes — learning, tasks, information, and understanding.

## What's here

- **[Tutorials](tutorials/first-applet.md)** — learning-oriented, start-to-finish walkthroughs. Begin with [your first applet](tutorials/first-applet.md).
- **How-to guides** — task-oriented recipes: [port a v1 applet](howto/port-v1-applet.md) to the epoch-2 ABI, or [debug an applet](howto/debug-an-applet.md) (attach LLDB, tail the log).
- **Reference** — the frozen contract and its neighbors: the [ABI (epoch 2)](reference/abi.md), the [manifest (`caliper.toml`)](reference/manifest.md), the contractual [refusal messages](reference/refusals.md), the service tables ([`caliper.ui.v1`](reference/services/ui-v1.md), [`caliper.log.v1`](reference/services/log-v1.md)), and the [C++ sugar](reference/sugar.md) layer.
- **Explanation** — the why behind the design: the [architecture](explanation/architecture.md) (layers and the frame loop), [compatibility & epochs](explanation/compatibility.md) (how the contract grows without breaking applets), and the [trust model](explanation/trust-model.md).
- **[Decisions](decisions/index.md)** — the decision log, mirroring `PLATFORM.md` §18.

## How this wiki stays true

Three mechanisms, not effort, keep these pages honest:

1. Doc updates land in the **same commit** as the change they describe.
2. Reference pages **embed the real files** (headers, manifests) via `pymdownx.snippets` with `check_paths: true`, so a moved or renamed file breaks the docs build instead of silently orphaning the page.
3. `mkdocs build --strict` fails the build on any broken link or anchor.

!!! note "Source of truth"
    `PLATFORM.md` at the repo root remains the governing specification. Where this wiki summarizes it, the spec wins on any discrepancy.
