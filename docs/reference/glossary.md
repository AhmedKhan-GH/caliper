# Glossary

These pages use each term below with one meaning.

## Terms

| Term | Meaning |
|---|---|
| ABI epoch | The integer version of the frozen instrument contract, which is `abi.h` together with the UI-stack pin. It is 4. The host refuses a library instrument with a different epoch before it loads any instrument code. |
| Caliper API | The HTTP interface to the running app. A program in any language can use it to list and launch instruments, stream data to the app and fetch what the app shows. See [HTTP API](../agents/http-api.md). |
| Caliper CLI | The subcommands of the `caliper` executable, such as `list`, `check`, `new`, `install`, `dev` and `shot`. See [CLI](cli.md). |
| Caliper Embed | The host core as a library behind a small C interface. Another program uses it to run the instrument canvas inside a view that the program owns. It owns the instrument canvas, the renderer and the one libtorch of the process. The `caliper-embed` package installs it as the shared library `caliper_embed`. See [Embedding API](../web/embedding-api.md). |
| Caliper LIB | The `caliper` Python package. It gives a training script live views of its tensors. See [Caliper LIB API](../python/caliper-lib-api.md). |
| Caliper SDK | The C++ instrument SDK. Library instruments build against the `caliper-sdk` package, which holds the C ABI headers, the service tables, the interchange types, the C++ sugar, the adapters, the pinned UI-stack sources, the fixture host and `caliper_add_instrument()`. See [SDK Package](../instruments/sdk-package.md) and [Instrument Contract](../instruments/instrument-contract.md). |
| `CaliperTensor` | The interchange type that describes a block of memory with a dtype, a shape, strides, a device and an optional stream. A torch or NumPy tensor converts to it without a copy. See [Tensor and Arrow Headers](../instruments/tensor-and-arrow-headers.md#tensor-header). |
| `CaliperTextureId` | An opaque 64-bit texture handle that the tensor bridge returns. It converts to an `ImTextureID` and is never a raw graphics handle. |
| Core | The host's own code, as opposed to the instruments. It holds the loader, the service implementations, the renderer, the crash guard and the frame watchdog. It builds as the `libcaliper` library and the `caliper` executable. |
| Descriptor | The `CaliperInstrumentDescriptor` that an instrument's library exports as `caliper_instrument_descriptor`. It holds the identity of the instrument, its required services and its five lifecycle functions. The sugar generates it. |
| Fixture host | A headless fake `CaliperHost`. It lets you test the logic of a library instrument and the sugar without the UI. See [Testing](../development/testing.md). |
| Frame watchdog | The host's timer on the frame call of each library instrument. If three consecutive frames take more than about 250 ms, the host raises a flag on the page of the instrument. The flag points long work to background jobs. The watchdog does not stop the instrument. |
| Host | The program that opens instruments. It owns the window, the renderer and the frame loop, and it implements the services. The `caliper` executable is the host. See [Architecture](../introduction/architecture.md). |
| HostRenderer | The internal rendering interface of the host, with Metal, Vulkan and WebGPU implementations behind it. Instruments never see it. See [Rendering](../instruments/rendering.md). |
| Instrument | Anything the Caliper app opens, written in C++ (`kind = "library"`, a shared library with a [manifest](../instruments/manifest.md) that exports one descriptor) or written as a spec (`kind = "spec"`, a [manifest](../instruments/manifest.md#spec-based-instruments) paired with a `caliper_spec = 2` TOML file), and optionally connected to a live run. |
| Live session | A Python process that streams tensors to the Caliper app through shared memory. The training script keeps running while the app shows its data. See [Live Streaming](../python/live-streaming.md). |
| Manifest | The `<stem>.caliper.toml` file of an instrument. It states the identity of the instrument, how it is built (`library` or `spec`), the oldest host it supports, and for library instruments the ABI epoch and services it needs. The host checks it before opening the instrument. See [Manifest](../instruments/manifest.md). |
| Quarantine | The state of a library instrument after a fault in one of its calls. The host does not call a quarantined instrument again, and it keeps the library loaded. See [Trust Model](../instruments/trust-model.md). |
| Revision | An additive version of a service, published as a new id with a suffix such as `.v1_1`. Its table starts as a copy of the previous table and adds entries. The older ids keep working. See [Compatibility and Epochs](../development/compatibility-and-epochs.md). |
| Service | A named, immutable C function table that the host vends through `get_service("caliper.x.vN")`. The [Service Catalog](../instruments/services/service-catalog.md) lists them all. |
| Sugar | The header-only C++ layer in `caliper.hpp` over the C ABI. It turns a class and the `CALIPER_INSTRUMENT` macro into the descriptor, and it catches exceptions at the ABI edge. It is optional. See [C++ Sugar](../instruments/cpp-sugar.md). |
| Tensor bridge | The service `caliper.tensor_bridge`. It turns a `CaliperTensor` into a texture that the instrument shows in the current frame, on the GPU where the backend allows it. See [`caliper.tensor_bridge.v1`](../instruments/services/caliper-tensor-bridge-v1.md). |
| UI stack | The pinned ImGui, ImPlot, ImPlot3D and ImGuiFileDialog sources. The host and every library instrument build against them. The exact versions are part of the ABI epoch because library instruments write raw ImGui. |
| Viewer | The Caliper process that attaches to a live session and draws what the script streams. It runs separately from the script. See [Live Streaming](../python/live-streaming.md). |
