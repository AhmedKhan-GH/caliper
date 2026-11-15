# Streaming and embedding

This page plans how other programs use Caliper: scripts and notebooks that stream data to a viewer, a viewer that runs in a browser, applets that run in a browser over the same data, and programs that embed Caliper. The minimal embedding package is built, and the rest is planned for after Phase 1. The [roadmap](roadmap.md#streaming-and-embedding) gives the order and status. What is built today is described in [Live streaming](../explanations/live-streaming.md), the [Python API](../references/python-api.md) and [Embedding Caliper](../references/embedding.md).

## The order of surfaces

The tools that ML engineers use come first, and consumer surfaces come last:

1. Python scripts and notebooks.
2. The web viewer.
3. C++ embedding.
4. Rust bindings.
5. Swift.

## Data streaming

Data streaming is the portable core of the plan and the only way Caliper reaches a browser. A producer keeps its computation in native torch, and only snapshots of its tensors travel. The browser draws from that data with the WebAssembly view layer, the [web viewer](#the-web-viewer). Data streaming powers notebooks, `caliper.serve()` and the hub.

- The live protocol gains a network transport: WebSocket first, and WebTransport later.
- The latest snapshot wins for each view, and a series is sent as deltas.
- Streaming never blocks training.
- The zero-copy claim applies only to viewing on the same machine. A viewer on the network receives copies.

Large data is handled with sampling and views rather than sent whole:

- Reduction runs on the device, so a snapshot is reduced before it leaves the GPU.
- Level of detail sends only what is visible at the viewer's screen resolution, and refines the region the viewer zooms into, as map tiles do. Bandwidth then follows the viewer's screen pixels, not the size of the data.
- A weak client is served the same way, with a coarser level of detail.

## The web viewer

The web viewer is Caliper's view layer, which is the live views, ImGui, ImPlot and ImPlot3D, compiled to WebAssembly and drawn by the WebGPU renderer. It has no applet loader, no DuckDB and no tensor bridge. It ships as a JavaScript and TypeScript package with a `<caliper-view>` element and a Jupyter widget.

No computation runs in the browser. libtorch does not run there, and the plan adds no second ML stack: the browser shows what a producer sends. An applet reaches a browser through the split described in [Applets in the browser](#applets-in-the-browser): its interface, compiled to WebAssembly, draws there from the data stream. An applet that has not been ported to the split is shown only in the native app.

## Serving

Two modes serve the web viewer, both with the same protocol, built in this order:

1. **`caliper.serve()` in Python.** The script itself serves the viewer page and its stream.
2. **`caliper serve`, a subcommand of the `caliper` binary.** It runs a hub. Any number of scripts send to it, any number of browsers or Caliper apps watch it, and it outlives the scripts.

## Applets in the browser

Every applet splits into a computation and an interface. This is a design decision that the roadmap settles at the start of Phase 2.

- The computation runs natively, with torch, on the machine that has the GPU.
- The interface is ImGui code that compiles to WebAssembly, so the same applet runs in the native app and in a browser, where it draws from the data stream.
- The SDK enforces the rule: interface code may use only services that have a wire form.
- Applets built before the rule stay native-only until they are ported. Until then they are shown only in the native app.

## Spec applets

The spec applets of Phase 3 ([Applet builder](applet-builder.md)) are designed to run in the browser unchanged, and they satisfy the split by construction. Their primitives are the live view types, and they bind only to sources that have a wire form.

## Renderers

Metal and Vulkan stay the fast paths. One WebGPU renderer serves the browser and, through Dawn, the native fallbacks. The roadmap's [Renderers](roadmap.md#renderers) section gives that plan.

## The embedding package

The minimal embedding package is built: the `caliper-embed` package, `caliper_embed_api_version()`, and `examples/embed_host` built outside the repository with only `find_package(caliper-embed)`. [Embedding Caliper](../references/embedding.md) describes it, and [Embed Caliper in a program](../guides/embed-caliper.md) builds an embedder against it.

The rest comes when C++ embedding comes up in the order of surfaces: several views in one app, rendering a frame to an image file, HiDPI work and a Qt widget. Offscreen canvases and `caliper_core_read_pixels()` exist today.

## Security

Network streaming stays on the local machine, on the loopback interface and with the session token, until a dedicated security review. That review happens before any session leaves the machine. The confidentiality of streamed data rests on the transport and on the security of whoever hosts it: the session token, encryption, and that review before anything leaves the machine. The streaming socket never carries the command layer.

## Licence

The web package and the protocol are licensed under Apache 2.0, like the core.

## Platforms

The plan covers macOS and Windows. Linux is not planned for now.
