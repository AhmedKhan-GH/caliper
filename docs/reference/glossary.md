# Glossary

These pages use each term below with one meaning.

## Terms

| Term | Meaning |
|---|---|
| ABI epoch | The integer version of the frozen applet contract, which is `abi.h` together with the UI-stack pin. It is 2. The host refuses an applet with a different epoch before it loads any applet code. |
| Applet | A shared library that the host loads and runs in its own process. It draws into the host's UI and uses the host's services. Each applet has a [manifest](../applets/manifest.md) and exports one descriptor. |
| Caliper API | The HTTP interface to the running app. A program in any language can use it to list and launch applets, stream data to the app and fetch what the app shows. See [Caliper API](../agents/http-api.md). |
| Caliper CLI | The subcommands of the `caliper` executable, such as `list`, `check`, `new`, `install`, `dev` and `shot`. See [Caliper CLI](cli.md). |
| Caliper Embed | The host core as a library behind a small C interface. Another program uses it to run the applet canvas inside a view that the program owns. It owns the applet canvas, the renderer and the one libtorch of the process. The `caliper-embed` package installs it as the shared library `caliper_embed`. See [Embedding Caliper](../web/embedding-api.md). |
| Caliper LIB | The `caliper` Python package. It gives a training script live views of its tensors. See [Caliper LIB](../python/caliper-lib-api.md). |
| Caliper SDK | The C++ applet SDK. Applets build against the `caliper-sdk` package, which holds the C ABI headers, the service tables, the interchange types, the C++ sugar, the adapters, the pinned UI-stack sources, the fixture host and `caliper_add_applet()`. See [SDK package](../applets/sdk-package.md) and the [ABI](../applets/applet-contract.md). |
| `CaliperTensor` | The interchange type that describes a block of memory with a dtype, a shape, strides, a device and an optional stream. A torch or NumPy tensor converts to it without a copy. See [Tensor](../applets/tensor-and-arrow-headers.md#tensor-header). |
| `CaliperTextureId` | An opaque 64-bit texture handle that the tensor bridge and the geometry service return. It converts to an `ImTextureID` and is never a raw graphics handle. |
| Core | The host's own code, as opposed to the applets. It holds the loader, the service implementations, the renderer, the crash guard and the frame watchdog. It builds as the `libcaliper` library and the `caliper` executable. |
| Descriptor | The `CaliperAppletDescriptor` that an applet's library exports as `caliper_applet_descriptor`. It holds the identity of the applet, its required services and its five lifecycle functions. The sugar generates it. |
| Fixture host | A headless fake `CaliperHost`. It lets you test the logic of an applet and the sugar without the UI. See [Testing](../development/testing.md). |
| Frame watchdog | The host's timer on the frame call of each applet. If three consecutive frames take more than about 250 ms, the host raises a flag on the page of the applet. The flag points long work to background jobs. The watchdog does not stop the applet. |
| Host | The program that loads applets. It owns the window, the renderer and the frame loop, and it implements the services. The `caliper` executable is the host. See [Architecture](../introduction/architecture.md). |
| HostRenderer | The internal rendering interface of the host, with Metal, Vulkan and WebGPU implementations behind it. Applets never see it. See [Rendering](../applets/rendering.md). |
| Live session | A Python process that streams tensors to the Caliper app through shared memory. The training script keeps running while the app shows its data. See [Live streaming](../python/live-streaming.md). |
| Manifest | The `<stem>.caliper.toml` file beside the library of an applet. It states the identity of the applet, the ABI epoch it was built against, the oldest host it supports and the services it needs. The host checks it before it loads the library. See [Manifest](../applets/manifest.md). |
| Quarantine | The state of an applet after a fault in one of its calls. The host does not call a quarantined applet again, and it keeps the library loaded. See [Trust model](../applets/trust-model.md). |
| Revision | An additive version of a service, published as a new id with a suffix such as `.v1_1`. Its table starts as a copy of the previous table and adds entries. The older ids keep working. See [Compatibility and epochs](../development/compatibility-and-epochs.md). |
| Service | A named, immutable C function table that the host vends through `get_service("caliper.x.vN")`. The [Service catalog](../applets/services/service-catalog.md) lists them all. |
| Sugar | The header-only C++ layer in `caliper.hpp` over the C ABI. It turns a class and the `CALIPER_APPLET` macro into the descriptor, and it catches exceptions at the ABI edge. It is optional. See [C++ sugar](../applets/cpp-sugar.md). |
| Tensor bridge | The service `caliper.tensor_bridge`. It turns a `CaliperTensor` into a texture that the applet shows in the current frame, on the GPU where the backend allows it. See [caliper.tensor_bridge.v1](../applets/services/caliper-tensor-bridge-v1.md). |
| UI stack | The pinned ImGui, ImPlot, ImPlot3D and ImGuiFileDialog sources. The host and every applet build against them. The exact versions are part of the ABI epoch because applets write raw ImGui. |
| Viewer | The Caliper process that attaches to a live session and draws what the script streams. It runs separately from the script. See [Live streaming](../python/live-streaming.md). |
