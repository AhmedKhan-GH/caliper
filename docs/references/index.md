# References

Reference pages describe the contracts exactly: the ABI, the manifest, each service table, the adapters, the embedding and Python interfaces, the applets, and the folders and environment variables. Where a page shows a header, the header is embedded from the source, so the page cannot drift from the code.

## The applet contract

| Page | Contents |
|---|---|
| [ABI](../applets/applet-contract.md) | `caliper/abi.h`, the epoch 2 descriptor and host structs |
| [Manifest](../applets/manifest.md) | The `caliper.toml` keys and the id rules |
| [SDK package](../applets/sdk-package.md) | The installed `caliper-sdk` package, `caliper_add_applet()` and the UI-stack hash |
| [Caliper CLI](cli.md) | The `caliper` subcommands, the `--applets` flag, the search order and the exit codes |
| [Refusal messages](refusals.md) | Every message the loader and the services show when they refuse an applet |
| [C++ sugar](../applets/cpp-sugar.md) | `caliper.hpp`, the `CALIPER_APPLET` macro and the service wrappers |
| [Tensor](../applets/tensor-and-arrow-headers.md#tensor-header) | `caliper/tensor.h`, the `CaliperTensor` interchange type |
| [Arrow C Data Interface](../applets/tensor-and-arrow-headers.md#arrow-header) | `caliper/arrow_c.h`, the tabular interchange |
| [Adapters](../applets/adapters.md) | The header-only adapters: torch to `CaliperTensor`, the allocation registry, the exportable pool, the OBJ loader and the orbit camera |
| [Torch objects on each platform](../applets/torch-per-platform.md) | Which torch objects reach the renderer on macOS and Windows |

## Services

The [service catalog](../applets/services/service-catalog.md) lists every service id the host serves, with its revisions and threading rule.

| Page | Service |
|---|---|
| [caliper.ui.v1](../applets/services/caliper-ui-v1.md) | The ImGui context and allocators |
| [caliper.ui.v1_1](../applets/services/caliper-ui-v1-1.md) | Adds dock hints, which give each window a role in the docked layout |
| [caliper.log.v1](../applets/services/caliper-log-v1.md) | Log lines |
| [caliper.jobs.v1](../applets/services/caliper-jobs-v1.md) | Background jobs with progress and cancellation |
| [caliper.device.v1](../applets/services/caliper-device-v1.md) | The compute devices the host found |
| [caliper.metrics.v1 and v1_1](../applets/services/caliper-metrics-v1.md) | Scalars, histograms and images per run and step, and SQL queries over them |
| [caliper.tensor_bridge.v1, v1_1 and v1_2](../applets/services/caliper-tensor-bridge-v1.md) | Tensors to textures, stream-ordered handoff and imported allocations |
| [caliper.geometry.v1 to v1_3](../applets/services/caliper-geometry-v1.md) | Points, primitives, textured and instanced geometry in offscreen views |
| [caliper.export.v1 and v1_1](../applets/services/caliper-export-v1.md) | Pixel-exact PNG figures and frame sequences |
| [caliper.artifacts.v1](../applets/services/caliper-artifacts-v1.md) | Content-addressed files such as checkpoints |
| [caliper.data.v1](../applets/services/caliper-data-v1.md) | SQL over the embedded DuckDB store |
| [caliper.feed.v1](../applets/services/caliper-feed-v1.md) | Machine telemetry channels |

## Hosts and clients

| Page | Contents |
|---|---|
| [Embedding Caliper](embedding.md) | `caliper/embed.h`, the C interface for hosting the core in another program |
| [Caliper LIB](../python/caliper-lib-api.md) | The `caliper` Python package |
| [Wire protocol v3](wire-protocol.md) | `caliper/live_net.h`, the WebSocket protocol between producers, viewers and the hub |
| [Browser elements](browser-elements.md) | `<caliper-view>` and `<caliper-controls>`: their attributes, methods and events, and the widgets built on them |
| [Caliper API](api.md) | The HTTP interface to the running app: its endpoints, its session token and its limits |
| [MCP server](mcp-server.md) | `caliper-mcp`: what an outside agent needs running, its command line, the configuration of an MCP client, its tools and resources |
| [Widget agent](widget-agent.md) | `caliper-agent` in its widget role: opening a widget session, its tool scope, its prompts and its answers |
| [Side chat](side-chat.md) | The pane where your own ACP agent works: the agents it detects, what it starts and hands over, and what it shows |
| [Builder pane](../builder/builder-pane.md) | Building a spec by hand on the spec page: the palettes, the canvas, Properties, the spec text and the keys |

## Applets and environment

| Page | Contents |
|---|---|
| [Applets](../applets/in-tree-applets.md) | The applets and examples built in this repository |
| [Folders and environment variables](environment.md) | Where the host keeps its data, and every `CALIPER_*` variable |
| [Glossary](glossary.md) | The terms these pages use |
