# Service Catalog

The host serves 13 service ids, which are 9 services and their additive revisions. An instrument gets a service by id from `CaliperHost.get_service`. The call returns `NULL` for an id the host does not serve. An instrument lists the ids it cannot run without in its manifest. See [Manifest](../manifest.md).

## Revisions

A revision of a service is a new id with a `_N` suffix after the version, for example `caliper.tensor_bridge.v1_2`. A revision is additive. Its table begins with a byte-identical copy of the previous table and appends entries, and the host keeps serving the older ids unchanged. An instrument built against `caliper.tensor_bridge.v1` keeps working when the host also serves `v1_2`. No service has an incompatible `.v2`. For the full rule, see [Compatibility and epochs](../../development/compatibility-and-epochs.md).

## Service IDs

| Id | Purpose | Page | Additive over | Threading |
|---|---|---|---|---|
| `caliper.ui.v1` | The ImGui, ImPlot and ImPlot3D contexts and the host's allocator pair | [ui](caliper-ui-v1.md) | | UI thread only |
| `caliper.ui.v1_1` | Adds `dock_hint`, the role of each window in the host's docked layout | [ui v1_1](caliper-ui-v1-1.md) | `ui.v1` | UI thread only |
| `caliper.log.v1` | Pre-formatted log lines into the host console | [log](caliper-log-v1.md) | | Any thread. Resolve the table in `on_init` and cache it. |
| `caliper.jobs.v1` | Background work with progress and cancellation | [jobs](caliper-jobs-v1.md) | | A job runs on a host worker thread and polls `cancelled()` |
| `caliper.device.v1` | The compute device the host found: CPU, CUDA or Metal | [device](caliper-device-v1.md) | | Not stated in the header |
| `caliper.metrics.v1` | Scalars, histograms and images per run, tag and step | [metrics](caliper-metrics-v1.md) | | Instrument job threads. The host serializes internally. |
| `caliper.metrics.v1_1` | Adds `query`, a read-only SQL query that returns an Arrow stream | [metrics](caliper-metrics-v1.md#calipermetricsv1_1) | `metrics.v1` | Any thread, including a host UI thread. The store serializes internally. |
| `caliper.tensor_bridge.v1` | A `CaliperTensor` becomes a texture | [tensor bridge](caliper-tensor-bridge-v1.md) | | Frame thread only |
| `caliper.tensor_bridge.v1_1` | Adds `caps()` and stream-ordered handoff | [tensor bridge](caliper-tensor-bridge-v1.md) | `tensor_bridge.v1` | Frame thread only |
| `caliper.tensor_bridge.v1_2` | Adds imported device allocations | [tensor bridge](caliper-tensor-bridge-v1.md) | `tensor_bridge.v1_1` | Frame thread only |
| `caliper.artifacts.v1` | Content-addressed files such as checkpoints | [artifacts](caliper-artifacts-v1.md) | | Instrument job threads. The host serializes internally. |
| `caliper.data.v1` | SQL over the embedded DuckDB store, results as Arrow streams | [data](caliper-data-v1.md) | | Instrument job threads. `last_error()` is per thread. |
| `caliper.feed.v1` | Machine telemetry channels read by pull | [feed](caliper-feed-v1.md) | | Any thread, non-blocking, stated in the header |

Every `v1_N` row keeps the threading rule of the service it extends.

## Companion Types

The services pass data in two interchange types instead of framework types. These are [`CaliperTensor`](../tensor-and-arrow-headers.md#tensor-header) for tensors and the [Arrow C Data Interface](../tensor-and-arrow-headers.md#arrow-header) for tables. Neither type depends on a framework, thus the SDK headers contain no torch or DuckDB type.
