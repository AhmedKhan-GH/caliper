# Architecture

Caliper has four layers, with a single dependency rule: arrows point down only. The repository holds all of them, but the core build compiles no applet. The applets are separate shared libraries that the host loads at run time.

## Layers

```text
+------------------------------------------------------------------+
| APPLETS  shared libraries, each beside a <stem>.caliper.toml      |
|   built against the installed SDK; the nine scopes in applets/    |
|   and the two tutorial applets in examples/                       |
+------------------------------------------------------------------+
| SDK  the caliper-sdk package (sdk/ in this repository)            |
|   abi.h          frozen C entry ABI, versioned by an epoch        |
|   tensor.h       CaliperTensor interchange type                   |
|   arrow_c.h      Arrow C Data Interface                           |
|   services/*.h   named C service tables                           |
|   caliper.hpp    header-only C++ sugar                            |
|   adapters/      header-only torch and allocation helpers         |
|   sdk/testing    the fixture host                                 |
|   sdk/cmake      the CMake package and caliper_add_applet()       |
+------------------------------------------------------------------+
| UI STACK  pinned ImGui, ImPlot, ImPlot3D and ImGuiFileDialog,     |
|   shipped as sources in the SDK                                   |
+------------------------------------------------------------------+
| HOST  the caliper executable and libcaliper                       |
|   window and frame loop, HostRenderer, applet loader,            |
|   service implementations, crash guard, frame watchdog            |
|   embedders link it as one shared library, caliper_embed          |
+------------------------------------------------------------------+
```

**Dependency rule.** Applets depend on the SDK and the UI stack. The host implements the SDK. The SDK depends on nothing else in Caliper. Torch and DuckDB types never appear in an SDK header: the interchange types (`CaliperTensor` and Arrow C streams) cross the boundary instead. See [Compatibility and epochs](compatibility.md).

**Only `caliper.ui.v1` may know the toolkit or the renderer.** Every other service describes data, such as tensors, tables, geometry and files, and never an ImGui, Metal, Vulkan or OpenGL type. That keeps the services usable by a host that draws with a different toolkit, and it is why a renderer change needs no applet rebuild. The one service that hands applets the ImGui, ImPlot and ImPlot3D contexts is [`caliper.ui.v1`](../references/service-ui-v1.md).

The applets still live in this repository, but the core build compiles none of them, and the `caliper` target depends on none. Each applet builds against the installed SDK package with `find_package(caliper-sdk)` and `caliper_add_applet()`, as an applet outside the repository would ([SDK package](../references/sdk-package.md)). `CALIPER_BUILD_APPLETS=ON` also builds the `applets/` folders inside the core build, and `examples/hello` and `examples/sine_scope` build with the tests, because the tests load them. Moving the applets to their own repositories is planned in [Applets outside the core](../plans/applets-outside-the-core.md).

## Platform rules

These rules hold across the host, the SDK and every applet.

- **Applets run in the host's process and call it through a C ABI.** An applet is a native library that the host loads. It talks to the host through plain C functions, with the optional header-only [C++ sugar](../references/sugar.md) on top, and never through C++ interfaces or a protocol between processes. An applet in the host's frame loop can hand over a tensor without copying it, and a C ABI stays stable across compilers and standard libraries. The cost is that an applet crash can take the host down; the [Trust model](trust-model.md) describes how far the host contains that.
- **The host links no ML framework.** The `caliper` executable and `libcaliper` link neither libtorch nor the CUDA toolkit. The tensor bridge talks to Metal and to the CUDA driver directly, the stores use an embedded DuckDB, and torch is a concern of the applets that use it. The host stays small, and torch arrives only with an applet that needs it.
- **Data flows one way, from tensors to pixels.** Tensors become textures and geometry, ImGui shows them, and nothing flows back. There is no render-to-tensor readback, and applets cannot supply shaders: appearance comes from a fixed menu of shading modes. Showing the state of a model is the purpose of the instrument, shaders from applets would break the boundary between applet and host and weaken its security, and a fixed menu keeps every image reproducible against the CPU reference. [Rendering](rendering.md#data-flows-one-way) lists what this rules out and what it does not.
- **A missing capability falls back to a slower path that works, and says so.** The host never draws a wrong image and never prints a status line that is not true, and it claims zero-copy only when the zero-copy path drew. Applets then run on every machine, and users can trust what the instrument reports. The cost is a second, tested path for every capability.

## The frame loop

The host owns the GLFW window, the rendering backend, the ImGui, ImPlot and ImPlot3D contexts, and the frame clock. Per frame:

1. Host **begins** the ImGui frame.
2. The active applet's `on_frame` renders its UI and may consume GPU results produced by its own background jobs.
3. Host **ends** the frame and swaps buffers.

Applets never create windows or touch the graphics API.

The frame thread does no I/O: no file access, no network call, no SQL query and no store call. The cost of I/O differs by machine and grows with the age of a session, so a call that is fast on one machine freezes the window on another. Work that takes longer than about 5 ms runs on another thread: a `caliper.jobs.v1` job for work that a user watches or cancels, and a thread the applet owns for its own upkeep. A table that is rebuilt several times a second for live display is a DuckDB `TEMP` table, because the persistent store is for data that is saved on purpose and read back. The host's watchdog flags applets that stall the loop (see the [Trust model](trust-model.md)).

!!! note "Renderer-agnostic by contract"
    The ABI never mentions a graphics API. Textures cross the boundary as an opaque `CaliperTextureId`, and applets render only through ImGui and ImPlot. Metal on macOS and Vulkan on Windows are the default backends, and OpenGL 3.3 core is the fallback, chosen with `CALIPER_RENDERER=gl` or when the default backend fails to start. Because the contract hides the backend, a backend change needs no epoch bump and no applet rebuild. See [Rendering](rendering.md) for the `HostRenderer` seam and the backends.

## Embedding

`libcaliper` is the host without the window. It holds the applet loader, the service registry, the services, `HostRenderer`, the tensor bridge and the geometry path behind a small C interface, so another program can run the applet canvas inside a view it owns. The `caliper` executable is the first embedder. The embedder supplies a native view and its input events. `libcaliper` owns the applet canvas end to end: the ImGui context, the renderer, the bridge and the geometry all run inside it, and the embedder's own chrome wraps around the canvas and never paints applet pixels. A toolkit such as wxWidgets paints through its own drawing backends and cannot paint pixels that applets produce on the GPU, and a single owner of the canvas keeps the applet contract the same in every host.

A program outside this repository embeds the core through the `caliper-embed` package. It holds one shared library, `caliper_embed`, linked as the CMake target `caliper::embed`, which carries `libcaliper` and the libraries behind it and exports only the C functions of `caliper/embed.h`, `caliper/live.h` and `caliper/live_views.h`. An embedder therefore links one file and keeps its own copies of ImGui, GLFW or DuckDB apart from the core's. The headers install under an include root of their own, `include/caliper-embed`, so an applet that links `caliper::sdk` cannot include `caliper/embed.h`, even when both packages are installed in one prefix.

A process has one libtorch, and the core owns it. An embedder must not link its own torch, and `caliper_core_create` refuses a second live core in the same process. The interface is described in [Embedding Caliper](../references/embedding.md), and [Embed Caliper in a program](../guides/embed-caliper.md) builds an embedder against the package.
