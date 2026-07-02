# Caliper Platform

Caliper is a desktop platform for **ML-native, GPU-resident visualization running in the same frame loop as training**. Applets are shared libraries the host loads at runtime over a small, frozen C ABI; the host owns the window, the rendering backend, and the ImGui/ImPlot contexts, while each applet renders its own UI and drives its own compute.

The platform's reason to exist is the tensor bridge: weights, activations, and saliency maps that live on the GPU (CUDA or MPS) become an `ImGui::Image` *this frame* — no TensorBoard round-trip, no PNG encode, no Python, and on the native backends no CPU staging. That capability is preserved by keeping applets in-process and productized as the `caliper.tensor_bridge.v1` service.

The flagship applet, **GPTScope**, is the proof: a char-level mini-GPT (nanoGPT-style, 4 layers / 4 heads) trained live on TinyShakespeare, built entirely on the public service stack. It trains off the frame thread via `caliper.jobs.v1` on the host-negotiated device, streams train/val loss (and val perplexity) to the `caliper.metrics.v1` Runs dashboard, samples text live as the loss falls (gibberish → words → cadence), and renders per-head attention heatmaps through `caliper.tensor_bridge.v1` — GPU-resident on Metal — with layer switching, hover-highlighting, and a sampling-temperature control. No private hooks; every capability it uses is one an out-of-tree applet has.

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
