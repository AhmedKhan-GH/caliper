# Glossary

The terms below are used across these pages with one meaning each.

**Applet.** A shared library that the host loads and runs in its own process. It draws into the host's UI and uses the host's services. Each applet has a [manifest](manifest.md) and exports one descriptor.

**Host.** The program that loads applets, owns the window, the renderer and the frame loop, and implements the services. The `caliper` executable is the host. See [Architecture](../explanations/architecture.md).

**Core.** The host's own code, as opposed to the applets. The core is the loader, the service implementations, the renderer, the crash guard and the frame watchdog. It is built as the `libcaliper` library and the `caliper` executable.

**Caliper Embed.** The host core as a library behind a small C interface, so that another program can run the applet canvas inside a view it owns. It owns the applet canvas, the renderer and the one libtorch of the process. Its code is the static library `libcaliper`, and the `caliper-embed` package installs it as the shared library `caliper_embed`. See [Embedding Caliper](embedding.md).

**Caliper SDK.** The C++ applet SDK, the `caliper-sdk` package that applets build against: the headers in `sdk/include/caliper` (the C ABI, the service tables, the interchange types, the C++ sugar and the adapters), the pinned UI-stack sources, the fixture host and `caliper_add_applet()`. See [SDK package](sdk-package.md) and the [ABI](abi.md).

**Caliper CLI.** The subcommands of the `caliper` executable, such as `list`, `check`, `new`, `install`, `dev` and `shot`. See [Caliper CLI](cli.md).

**Caliper LIB.** The `caliper` Python package, which gives a training script live views of its tensors. See [Caliper LIB](python-api.md).

**Caliper API.** A planned HTTP interface to the running app, so that a program in any language can list and launch applets, stream data to them and fetch what they show. It is not built; the [roadmap](../plans/roadmap.md#api) gives its plan.

**Service.** A named, immutable C function table that the host vends through `get_service("caliper.x.vN")`. The [Service catalog](services.md) lists them all.

**Revision.** An additive version of a service, published as a new id with a suffix such as `.v1_1`. Its table starts with a copy of the previous table and appends entries, and the older ids keep working. See [Compatibility and epochs](../explanations/compatibility.md).

**ABI epoch.** The integer version of the frozen applet contract, which is `abi.h` together with the UI-stack pin. It is 2 today. The host refuses an applet whose epoch differs from its own, before it loads any applet code.

**Manifest.** The `<stem>.caliper.toml` file beside an applet's library. It states the applet's identity, the ABI epoch it was built against, the oldest host it supports and the services it needs. The host checks it before it loads the library. See [Manifest](manifest.md).

**Descriptor.** The `CaliperAppletDescriptor` that an applet's library exports as `caliper_applet_descriptor`. It holds the applet's identity, its required services and its five lifecycle functions. The sugar generates it.

**Sugar.** The header-only C++ layer in `caliper.hpp` over the C ABI. It turns a class and the `CALIPER_APPLET` macro into the descriptor and catches exceptions at the ABI edge. It is optional. See [C++ sugar](sugar.md).

**UI stack.** The pinned ImGui, ImPlot, ImPlot3D and ImGuiFileDialog sources that the host and every applet are built against. The exact versions are part of the ABI epoch, because applets write raw ImGui.

**HostRenderer.** The host's internal rendering interface, with Metal, Vulkan and OpenGL implementations behind it. Applets never see it. See [Rendering](../explanations/rendering.md).

**`CaliperTextureId`.** An opaque 64-bit texture handle returned by the tensor bridge and the geometry service. It converts to an `ImTextureID` and is never a raw graphics handle.

**`CaliperTensor`.** The interchange type that describes a block of memory with a dtype, a shape, strides, a device and an optional stream. A torch or NumPy tensor converts to it without a copy. See [Tensor](tensor.md).

**Tensor bridge.** The service `caliper.tensor_bridge` that turns a `CaliperTensor` into a texture the applet shows this frame, on the GPU where the backend allows it. See [caliper.tensor_bridge.v1](service-tensor-bridge-v1.md).

**Fixture host.** A headless fake `CaliperHost` in `sdk/testing`. It lets an applet's logic and the sugar be tested without the UI. See [Testing](../explanations/testing.md).

**Quarantine.** The state of an applet after a fault in one of its calls. The host does not call a quarantined applet again and keeps its library loaded. See [Trust model](../explanations/trust-model.md).

**Frame watchdog.** The host's timer on each applet's frame call. Three consecutive frames over about 250 ms raise a flag on the applet's page that points long work to background jobs. It flags and does not stop the applet.

**Live session.** A Python process that streams tensors to the Caliper app through shared memory, so a training script keeps running while the app shows its data. See [Live streaming](../explanations/live-streaming.md).

**Viewer.** The Caliper process that attaches to a live session and draws what the script streams. It runs separately from the script. See [Live streaming](../explanations/live-streaming.md).
