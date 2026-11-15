# Streaming and embedding

This page plans how other programs use Caliper: scripts and notebooks that stream data to a viewer, a viewer that runs in a browser, and programs that embed Caliper. The minimal embedding package is built in Phase 1, and the rest is planned for after Phase 1. The [roadmap](roadmap.md#streaming-and-embedding) gives the order and status. What is built today is described in [Live streaming](../explanations/live-streaming.md), the [Python API](../references/python-api.md) and [Embedding Caliper](../references/embedding.md).

## The order of surfaces

The tools that ML engineers use come first, and consumer surfaces come last:

1. Python scripts and notebooks.
2. The web viewer.
3. C++ embedding.
4. Rust bindings.
5. Swift.

## Data streaming

Data streaming is the portable core of the plan. A producer keeps its computation in native torch, and only snapshots of its tensors travel.

- The live protocol gains a network transport: WebSocket first, and WebTransport later.
- Snapshots are downsampled on the device to the size the viewer draws.
- The latest snapshot wins for each view, and a series is sent as deltas.
- Streaming never blocks training.
- The zero-copy claim applies only to viewing on the same machine. A viewer on the network receives copies.

## The web viewer

The web viewer is Caliper's view layer, which is the live views, ImGui, ImPlot and ImPlot3D, compiled to WebAssembly and drawn by the WebGPU renderer. It has no applet loader, no DuckDB and no tensor bridge. It ships as a JavaScript and TypeScript package with a `<caliper-view>` element and a Jupyter widget.

No computation runs in the browser. libtorch does not run there, and the plan adds no second ML stack: the browser shows what a producer sends.

## Serving

Two modes serve the web viewer, both with the same protocol, built in this order:

1. **`caliper.serve()` in Python.** The script itself serves the viewer page and its stream.
2. **`caliper serve`, a subcommand of the `caliper` binary.** It runs a hub. Any number of scripts send to it, any number of browsers or Caliper apps watch it, and it outlives the scripts.

## Video streaming

Streaming video of a running Caliper, for sessions on a remote machine, comes later. It is the only way to show native applets in a browser, because native applets run only in the app.

## Spec applets

The spec applets of Phase 3 ([Applet builder](applet-builder.md)) are designed to run in the browser unchanged. Their primitives are the live view types, and they bind only to sources that have a wire form.

## Renderers

Metal and Vulkan stay the fast paths. One WebGPU renderer serves the browser and, through Dawn, the native fallbacks. The roadmap's [Renderers](roadmap.md#renderers) section gives that plan.

## The embedding package

The minimal embedding package is built in Phase 1. It has three parts:

- `caliper-embed`, a CMake package installed separately from the SDK. It holds `caliper/embed.h`, `caliper/live.h`, `caliper/live_views.h` and the core library, under an include root of its own, so an applet can never include `embed.h`.
- `caliper_embed_api_version()`, an additive call that reports the version of the embedding interface.
- `examples/embed_host`, built outside the repository with only `find_package(caliper-embed)`.

The rest comes when C++ embedding comes up in the order of surfaces: several views in one app, offscreen rendering to an image, HiDPI work and a Qt widget.

## Security

Network streaming stays on the local machine, on the loopback interface and with the session token, until a dedicated security review. That review happens before any session leaves the machine. The streaming socket never carries the command layer.

## Licence

The web package and the protocol are licensed under Apache 2.0, like the core.

## Platforms

The plan covers macOS and Windows. Linux is not planned for now.
