# References

Reference pages describe the contracts exactly: the ABI, the manifest, each service table, the adapters and the embedding and Python interfaces. Where a page shows a header, the header is embedded from the source, so the page cannot drift from the code.

## The applet contract

| Page | Contents |
|---|---|
| [ABI](abi.md) | `caliper/abi.h`, the epoch 2 descriptor and host structs |
| [Manifest](manifest.md) | The `caliper.toml` keys and the id rules |
| [Refusal messages](refusals.md) | Every message the loader and the services show when they refuse an applet |
| [C++ sugar](sugar.md) | `caliper.hpp`, the `CALIPER_APPLET` macro and the service wrappers |
| [Tensor](tensor.md) | `caliper/tensor.h`, the `CaliperTensor` interchange type |
| [Arrow C Data Interface](arrow.md) | `caliper/arrow_c.h`, the tabular interchange |
| [Adapters](adapters.md) | The header-only adapters between torch and `CaliperTensor` |
| [Torch objects on each platform](torch-on-each-platform.md) | Which torch objects reach the renderer on macOS and Windows |

## Services

| Page | Service |
|---|---|
| [caliper.ui.v1](service-ui-v1.md) | The ImGui context and allocators |
| [caliper.log.v1](service-log-v1.md) | Log lines |
| [caliper.jobs.v1](service-jobs-v1.md) | Background jobs with progress and cancellation |
| [caliper.device.v1](service-device-v1.md) | The compute devices the host found |
| [caliper.metrics.v1](service-metrics-v1.md) | Scalars, histograms and images per run and step |
| [caliper.tensor_bridge.v1](service-tensor-bridge-v1.md) | Tensors to textures |
| [caliper.geometry.v1 to v1_3](service-geometry-v1.md) | Points, primitives, textured and instanced geometry in offscreen views |
| [caliper.export.v1 and v1_1](service-export-v1.md) | Pixel-exact PNG figures and frame sequences |
| [caliper.artifacts.v1](service-artifacts-v1.md) | Content-addressed files such as checkpoints |
| [caliper.data.v1](service-data-v1.md) | SQL over the embedded DuckDB store |
| [caliper.feed.v1](service-feed-v1.md) | Machine telemetry channels |

## Hosts and clients

| Page | Contents |
|---|---|
| [Embedding Caliper](embedding.md) | `caliper/embed.h`, the C interface for hosting the core in another program |
| [Python API](python-api.md) | The `caliper` Python package |
