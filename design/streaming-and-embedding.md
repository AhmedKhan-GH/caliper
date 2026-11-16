# Streaming and embedding

This page plans what remains of using Caliper from other programs: the browser views beyond the live views, applets in a browser, the rest of C++ embedding, and the steps before a stream may leave the machine. The [roadmap](roadmap.md#streaming-and-embedding) gives the order and status.

What is built is described in the reference: [Network streaming](../explanations/network-streaming.md) and [Live streaming](../explanations/live-streaming.md) explain the design, [Caliper LIB](../references/python-api.md), [Wire protocol v3](../references/wire-protocol.md), [Browser elements](../references/browser-elements.md), [Caliper CLI](../references/cli.md#caliper-serve) and [Caliper API](../references/api.md) list the interfaces, [Rendering](../explanations/rendering.md#the-webgpu-backend) describes the browser's renderer, and [Watch a training run in a notebook](../tutorials/watch-a-training-run-in-a-notebook.md), [Run a hub](../guides/run-a-hub.md), [Attach to a running script](../guides/attach-to-a-running-script.md), [Control a running script](../guides/control-a-running-script.md) and [Embedding Caliper](../references/embedding.md) show them in use.

## The order of surfaces

The tools that ML engineers use come first, and consumer surfaces come last. Python scripts and notebooks and the browser views are built. The surfaces that remain come in this order:

1. The rest of C++ embedding.
2. Rust bindings.
3. Swift.

## Data streaming

Data streaming is the only way Caliper reaches a browser: a producer keeps its computation in native torch, and only snapshots of its tensors travel. What it still needs:

- **WebTransport** beside the WebSocket.
- **Data views of tables,** such as a subset of rows. Data views of tensors are built.
- **A zoom as a data view.** A zoom or a selection that needs more detail than a snapshot holds becomes a request for another data view, computed by the producer.

## The web viewer

The web viewer is Caliper's own view layer, which is the live views, ImGui, ImPlot and ImPlot3D, compiled to WebAssembly with Emscripten and drawn by a WebGPU backend behind `HostRenderer`. It draws every live view kind today, behind the `<caliper-view>` element, the notebook widget and the viewer page, and views written for the browser in TypeScript draw where WebGPU is absent. It grows in this order:

1. Done: Caliper's real views in notebooks and browsers, for every live view kind.
2. Every built-in view. Points are built; the geometry path and ImPlot3D remain, and need a WebGPU geometry pipeline within WebGPU's limit on the storage buffers of a vertex stage.
3. The whole Caliper interface in a browser, with the landing screen, docking and the panes. It needs storage in the browser's file system, DuckDB's WebAssembly build, and cross-origin isolation, without which a browser does not let a page use threads.
4. Applets in the browser, through the split into computation and interface described in [Applets in the browser](#applets-in-the-browser).
5. The same WebGPU backend as the native fallback through Dawn, described in [Renderers](#renderers).

Smaller steps remain for the views of the first step: the clipboard and multi-touch gestures in the canvas, a CI job that runs the module's checks on SwiftShader (the recipe is in `web/wasm/README.md`), and testing in Safari, in Firefox and on Windows.

No computation runs in the browser. libtorch does not run there, and the plan adds no second ML stack: the browser shows what a producer sends. An applet reaches a browser through the split described in [Applets in the browser](#applets-in-the-browser): its interface, compiled to WebAssembly, draws there from the data stream. An applet that has not been ported to the split is shown only in the native app.

## Limits in the browser

The limits of the browser views as they are, such as the bounded heap, the second copy into the GPU process and the browsers that are untested, are listed under [Known limits](../explanations/network-streaming.md#known-limits) in Network streaming. Every later step works within them, and three more bound the work ahead:

- **The WebGPU binding moves with Emscripten.** The core of `webgpu.h` is stable, but Emscripten's `emdawnwebgpu` port and its Emscripten-specific parts change, and each Emscripten release pins its own Dawn package, so an Emscripten update can need changes in the renderer.
- **Headless tests depend on flags.** Headless Chromium presents a WebGPU canvas on SwiftShader only with `--use-angle=swiftshader`, and a Chromium update can change which flags work.
- **The download grows with the views.** The module is about 495 KB with Brotli, of which Dear ImGui and ImPlot are more than half of the code and the two embedded faces most of the data. ImPlot3D and the geometry path will add to it.

## Applets in the browser

Every applet splits into a computation and an interface. This is a design decision that the roadmap settles at the start of Phase 2.

- The computation runs natively, with torch, on the machine that has the GPU.
- The interface is ImGui code that compiles to WebAssembly, so the same applet runs in the native app and in a browser, where it draws from the data stream.
- The SDK enforces the rule: interface code may use only services that have a wire form.
- Applets built before the rule stay native-only until they are ported. Until then they are shown only in the native app.

## Spec applets

The spec applets of Phase 3 ([Applet builder](applet-builder.md)) are designed to run in the browser unchanged, and they satisfy the split by construction. Their primitives are the live view types, and they bind only to sources that have a wire form.

## Renderers

Metal and Vulkan stay the fast paths. The WebGPU renderer that draws the browser views is written against the standard `webgpu.h`, which Dawn also offers natively, so the same code can serve the native fallbacks through Dawn. The roadmap's [Renderers](roadmap.md#renderers) section gives that plan.

## The embedding package

The minimal embedding package is built ([Embedding Caliper](../references/embedding.md)). The rest comes when C++ embedding comes up in the order of surfaces: several views in one app, rendering a frame to an image file, HiDPI work and a Qt widget.

## Security

Network streaming stays on the local machine, on the loopback interface and with the session token ([Security](../explanations/network-streaming.md#security)), until a dedicated security review, which comes before any session leaves the machine. The confidentiality of streamed data then rests on the transport and on the security of whoever hosts it: the session token, encryption and that review. The streaming socket never carries the command layer.

## Platforms

The plan covers macOS and Windows. Linux is not planned for now.
