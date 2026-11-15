# Streaming and embedding

This page plans how other programs use Caliper: scripts and notebooks that stream data to a viewer, a viewer that runs in a browser, applets that run in a browser over the same data, and programs that embed Caliper. The minimal embedding package and the first part of data streaming are built: the WebSocket transport, `caliper.serve()`, the notebook widget, the first browser views, data views of tensors and the `caliper serve` hub. After a successful [spike](web-viewer-spike.md), the web viewer is being built on Caliper's own view layer. The rest is planned. The [roadmap](roadmap.md#streaming-and-embedding) gives the order and status. What is built today is described in [Live streaming](../explanations/live-streaming.md), [Network streaming](../explanations/network-streaming.md), the [Python API](../references/python-api.md), the [Wire protocol](../references/wire-protocol.md) and [Embedding Caliper](../references/embedding.md).

## The order of surfaces

The tools that ML engineers use come first, and consumer surfaces come last:

1. Python scripts and notebooks.
2. The web viewer.
3. C++ embedding.
4. Rust bindings.
5. Swift.

## Data streaming

Data streaming is the portable core of the plan and the only way Caliper reaches a browser. A producer keeps its computation in native torch, and only snapshots of its tensors travel. The browser draws from that data with the [web viewer](#the-web-viewer), which is Caliper's own view layer compiled to WebAssembly, and with small views written for the browser where WebGPU is absent. Data streaming powers notebooks, `caliper.serve()` and the hub, which are built; [Network streaming](../explanations/network-streaming.md) describes them.

- The live protocol has a network transport. The WebSocket transport is built, and WebTransport comes later.
- The latest snapshot wins for each live view, and a series is sent as deltas.
- Streaming never blocks training.
- The zero-copy claim applies only to viewing on the same machine. A viewer on the network receives copies.

Large data is handled by sampling and by data views rather than sent whole:

- **Sampling.** The producer samples a large tensor or series on the device, so that a snapshot carries a sample of the data rather than all of it.
- **Data views.** A data view is a slice, selection, projection or aggregation of a tensor or table, such as one layer, one head, a window of steps, a subset of rows, or a mean over the batch. The viewer requests it, the producer computes it on the device where the data lives, and only the view's result is streamed. This is the same idea as a tensor view or a database view.
- A zoom or a selection is a request for a different data view.

Data views of tensors are built: a producer names the dimensions of a tensor, and a viewer indexes and reduces them within a budget. Data views of tables, such as a subset of rows, are still planned.

A data view decides which data travels, and a live view, such as a series or a histogram, decides how the viewer draws it.

## The web viewer

The web viewer is Caliper's own view layer, which is the live views, ImGui, ImPlot and ImPlot3D, compiled to WebAssembly with Emscripten and drawn by a WebGPU backend behind `HostRenderer`. Its first form holds the views only, with no applet loader and no DuckDB. Whether its texture live views keep the tensor bridge is decided when the view layer is split from the native core. It ships in the `web/` package, behind the `<caliper-view>` element, the notebook widget and the viewer page that `caliper.serve()` serves.

The [web viewer spike](web-viewer-spike.md) showed that this path works. The live views compiled for WebAssembly without changes, and a heatmap and a series drew in headless Chromium on a software and a hardware WebGPU adapter.

The notebook widget and the viewer page are being built on the web viewer. The live views come in this order: heatmap and series, then histogram and image, then points. Small views written for the browser in TypeScript, over the same protocol, are built and remain as the fallback where WebGPU is absent, where they draw with Canvas 2D.

The web viewer then grows in this order:

1. Caliper's real views in notebooks and browsers.
2. Every built-in view, including points, geometry and ImPlot3D.
3. The whole Caliper interface in a browser, with the landing screen, docking and the panes. It needs storage in the browser's file system, DuckDB's WebAssembly build, and cross-origin isolation, without which a browser does not let a page use threads.
4. Applets in the browser, through the split into computation and interface described in [Applets in the browser](#applets-in-the-browser).
5. The same WebGPU backend as the native fallback through Dawn, described in [Renderers](#renderers).

No computation runs in the browser. libtorch does not run there, and the plan adds no second ML stack: the browser shows what a producer sends. An applet reaches a browser through the split described in [Applets in the browser](#applets-in-the-browser): its interface, compiled to WebAssembly, draws there from the data stream. An applet that has not been ported to the split is shown only in the native app.

## Limits in the browser

These limits hold for the web viewer at every step. The facts come from the [web viewer spike](web-viewer-spike.md).

- **Computation stays native.** The browser draws what a producer sends and computes nothing. Data views are computed by the producer on its own device.
- **The heap stops at 4 GB.** WebAssembly in the browser is 32-bit, so the web viewer cannot hold more than 4 GB, and it bounds the size of the snapshots it accepts.
- **Each snapshot is copied twice.** A snapshot crosses the network, and the browser then copies it from WebAssembly memory into its GPU process. In the spike, a 2048 by 2048 heatmap sent 16 MB into the GPU process every frame, and the GPU queue backed up. Sampling and data views keep snapshots small for this reason as well.
- **Colours may differ by one level.** Natively, Caliper's colormap matches its CPU reference byte for byte. WGSL does not promise the same arithmetic, and on Metal the spike saw a few texels of the largest heatmaps differ by one level in one channel. In the browser, the colormap therefore has a tolerance of one level.
- **ImGui is not a web page.** ImGui draws the whole interface into one canvas. Keyboard and text input, the clipboard and touch need work in a browser, and the spike verified none of them. The canvas gives the browser's accessibility tools nothing to read, so a screen reader cannot follow the interface.
- **Most browsers are untested.** The spike ran only in headless Chromium. Chromium-based browsers ship WebGPU, while support in Safari and Firefox depends on the version and the platform, and neither was tested; nor were Windows and HiDPI screens. Where WebGPU is absent, the views written for the browser draw instead.

## Serving

Two modes serve the browser views, both with the same protocol. Both are built:

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

Metal and Vulkan stay the fast paths. One WebGPU renderer serves the browser and, through Dawn, the native fallbacks. The spike's backend is written against the standard `webgpu.h`, which Dawn also offers natively, so the same code can serve both roles. The roadmap's [Renderers](roadmap.md#renderers) section gives that plan.

## The embedding package

The minimal embedding package is built: the `caliper-embed` package, `caliper_embed_api_version()`, and `examples/embed_host` built outside the repository with only `find_package(caliper-embed)`. [Embedding Caliper](../references/embedding.md) describes it, and [Embed Caliper in a program](../guides/embed-caliper.md) builds an embedder against it.

The rest comes when C++ embedding comes up in the order of surfaces: several views in one app, rendering a frame to an image file, HiDPI work and a Qt widget. Offscreen canvases and `caliper_core_read_pixels()` exist today.

## Security

Network streaming stays on the local machine, on the loopback interface and with the session token, until a dedicated security review. That review happens before any session leaves the machine. The confidentiality of streamed data rests on the transport and on the security of whoever hosts it: the session token, encryption, and that review before anything leaves the machine. The streaming socket never carries the command layer.

## Licence

The web package and the protocol are licensed under Apache 2.0, like the core.

## Platforms

The plan covers macOS and Windows. Linux is not planned for now.
