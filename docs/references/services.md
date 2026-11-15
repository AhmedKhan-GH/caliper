# Service catalog

The host serves 18 service ids on `main`, which are 11 services and their additive revisions. An applet obtains a service by id from `CaliperHost.get_service`, which returns `NULL` for an id the host does not serve. An applet lists the ids it cannot run without in its manifest; see [Manifest](manifest.md). The ids are defined in `sdk/include/caliper/services/`, and the host serves them from `src/host/host_services.cpp`.

## Revisions

A revision of a service is a new id with a `_N` suffix after the version, for example `caliper.geometry.v1_2`. A revision is additive: its table begins with a byte-identical copy of the previous table and appends entries, and the host keeps serving the older ids unchanged. An applet built against `caliper.geometry.v1` keeps working when the host also serves `v1_3`. The rule is decision [D30](../decisions/d030-additive-service-revisions.md), and [Compatibility and epochs](../explanations/compatibility.md) explains it. No service has an incompatible `.v2`.

## The ids

| Id | Purpose | Page | Additive over | Threading |
|---|---|---|---|---|
| `caliper.ui.v1` | The ImGui, ImPlot and ImPlot3D contexts and the host's allocator pair | [ui](service-ui-v1.md) | | UI thread only |
| `caliper.log.v1` | Pre-formatted log lines into the host console | [log](service-log-v1.md) | | Any thread. Resolve the table in `on_init` and cache it. |
| `caliper.jobs.v1` | Background work with progress and cancellation | [jobs](service-jobs-v1.md) | | A job runs on a host worker thread and polls `cancelled()` |
| `caliper.device.v1` | The compute device the host found: CPU, CUDA or Metal | [device](service-device-v1.md) | | Not stated in the header |
| `caliper.metrics.v1` | Scalars, histograms and images per run, tag and step | [metrics](service-metrics-v1.md) | | Applet job threads. The host serializes internally. |
| `caliper.metrics.v1_1` | Adds `query`, a read-only SQL query that returns an Arrow stream | [metrics](service-metrics-v1.md#calipermetricsv1_1) | `metrics.v1` | Any thread, including a host UI thread. The store serializes internally. |
| `caliper.tensor_bridge.v1` | A `CaliperTensor` becomes a texture | [tensor bridge](service-tensor-bridge-v1.md) | | Frame thread only |
| `caliper.tensor_bridge.v1_1` | Adds `caps()` and stream-ordered handoff | [tensor bridge](service-tensor-bridge-v1.md) | `tensor_bridge.v1` | Frame thread only |
| `caliper.tensor_bridge.v1_2` | Adds imported device allocations | [tensor bridge](service-tensor-bridge-v1.md) | `tensor_bridge.v1_1` | Frame thread only |
| `caliper.geometry.v1` | Instanced points drawn in place from device memory into an offscreen view | [geometry](service-geometry-v1.md) | | Frame thread |
| `caliper.geometry.v1_1` | Adds points, lines and triangles with optional depth | [geometry](service-geometry-v1.md) | `geometry.v1` | Frame thread |
| `caliper.geometry.v1_2` | Adds textured geometry | [geometry](service-geometry-v1.md) | `geometry.v1_1` | Frame thread |
| `caliper.geometry.v1_3` | Adds instanced transforms | [geometry](service-geometry-v1.md) | `geometry.v1_2` | Frame thread |
| `caliper.export.v1` | Pixel-exact PNG figures and frame sequences with a provenance sidecar | [export](service-export-v1.md) | | Frame thread only, stated in the header |
| `caliper.export.v1_1` | Adds `last_error()` | [export](service-export-v1.md) | `export.v1` | Frame thread only, stated in the header |
| `caliper.artifacts.v1` | Content-addressed files such as checkpoints | [artifacts](service-artifacts-v1.md) | | Applet job threads. The host serializes internally. |
| `caliper.data.v1` | SQL over the embedded DuckDB store, results as Arrow streams | [data](service-data-v1.md) | | Applet job threads. `last_error()` is per thread. |
| `caliper.feed.v1` | Machine telemetry channels read by pull | [feed](service-feed-v1.md) | | Any thread, non-blocking, stated in the header |

The threading column follows the headers, except for the bridge and geometry rows, which come from their pages. Every `v1_N` row keeps the threading rule of the service it extends.

## Types that go with the services

The services pass data in two interchange types rather than in framework types: [`CaliperTensor`](tensor.md) for tensors, and the [Arrow C Data Interface](arrow.md) for tables. Neither depends on a framework, which is why the SDK headers contain no torch or DuckDB type.
