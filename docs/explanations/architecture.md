# Architecture

Caliper has four layers, with a single dependency rule: arrows point down only. The repository holds all of them. The applets are separate shared libraries that the host loads at run time.

## Layers

```text
+------------------------------------------------------------------+
| APPLETS  shared libraries, each beside a <stem>.caliper.toml      |
|   embed, field, flow, gpt, instance, mesh, pulse, sculpt and      |
|   twin scopes (the nine applets on main)                          |
+------------------------------------------------------------------+
| SDK  sdk/include/caliper (the contract)                           |
|   abi.h          frozen C entry ABI, versioned by an epoch        |
|   tensor.h       CaliperTensor interchange type                   |
|   arrow_c.h      Arrow C Data Interface                           |
|   services/*.h   named C service tables                           |
|   caliper.hpp    header-only C++ sugar                            |
|   adapters/      header-only torch and allocation helpers         |
|   sdk/testing    the fixture host                                 |
+------------------------------------------------------------------+
| UI STACK  third_party, pinned ImGui, ImPlot and ImPlot3D          |
+------------------------------------------------------------------+
| HOST  the caliper executable and libcaliper                       |
|   window and frame loop, HostRenderer, applet loader,            |
|   service implementations, crash guard, frame watchdog            |
+------------------------------------------------------------------+
```

**Dependency rule.** Applets depend on the SDK and the UI stack. The host implements the SDK. The SDK depends on nothing else in Caliper. Torch and DuckDB types never appear in an SDK header: the interchange types (`CaliperTensor` and Arrow C streams) cross the boundary instead. See [Compatibility and epochs](compatibility.md).

**Only `caliper.ui.v1` may know the toolkit or the renderer.** Every other service describes data, such as tensors, tables, geometry and files, and never an ImGui, Metal, Vulkan or OpenGL type. That keeps the services usable by a host that draws with a different toolkit, and it is why a renderer change needs no applet rebuild. The one service that hands applets the ImGui, ImPlot and ImPlot3D contexts is [`caliper.ui.v1`](../references/service-ui-v1.md).

The repository holds the host, the SDK and the applets together. A plan to move the applets out of the core, with the host and the SDK versioned on their own, is in [Applets outside the core](../plans/applets-outside-the-core.md).

## The frame loop

The host owns the GLFW window, the rendering backend, the ImGui, ImPlot and ImPlot3D contexts, and the frame clock. Per frame:

1. Host **begins** the ImGui frame.
2. The active applet's `on_frame` renders its UI and may consume GPU results produced by its own background jobs.
3. Host **ends** the frame and swaps buffers.

Applets never create windows or touch the graphics API. Long work never runs on the frame thread — that is what the `caliper.jobs.v1` service is for, and the host's watchdog flags applets that stall the loop (see the [Trust model](trust-model.md)).

!!! note "Renderer-agnostic by contract"
    The ABI never mentions a graphics API. Textures cross the boundary as an opaque `CaliperTextureId`, and applets render only through ImGui and ImPlot. Metal on macOS and Vulkan on Windows are the default backends, and OpenGL 3.3 core is a frozen fallback chosen with `CALIPER_RENDERER=gl`. Because the contract hides the backend, a backend change needs no epoch bump and no applet rebuild. See [Rendering](rendering.md) for the `HostRenderer` seam and the backends.

## Embedding

`libcaliper` is the host without the window. It holds the applet loader, the service registry, the services, `HostRenderer`, the tensor bridge and the geometry path behind a small C interface, so another program can run the applet canvas inside a view it owns. The `caliper` executable is the first embedder. The embedder supplies a native view and its input events. `libcaliper` owns the applet canvas end to end: the ImGui context, the renderer, the bridge and the geometry all run inside it, and the embedder's own chrome wraps around the canvas and never paints applet pixels. This is decision [D36](../decisions/d036-libcaliper-owns-the-canvas.md).

A process has one libtorch, and the core owns it. An embedder must not link its own torch, and `caliper_core_create` refuses a second live core in the same process. The interface is described in [Embedding Caliper](../references/embedding.md).
