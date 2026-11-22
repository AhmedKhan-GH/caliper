# Caliper LIB API

Caliper LIB is the `caliper` Python package. `import caliper` gives a PyTorch script live views of its tensors. The package version is 0.1.0, and `caliper.__version__` is `"0.1.0"`. It needs Python 3.12 and macOS 14 (Sonoma) or later on Apple silicon, and its wheel is tagged `macosx_14_0_arm64`.

The calls of the `caliper` package and of `caliper.live` are on these pages:

- [View Calls](view-calls.md): `show`, `points`, `series`, `histogram` and `watch`, with the rate limit.
- [Routes and Inspection](routes-and-inspection.md): routing, `session`, `window`, `offscreen`, `run`, `spin`, `flush`, `close`, `remove`, the inspection helpers and `caliper.live`.
- [Network Streaming Calls](network-streaming-calls.md): `serve`, `connect` and `widget`.
- [Data View Calls](data-view-calls.md): `dims`, `data_view`, `set_data_view`, `offer` and `update`.
- [Run Control Calls](run-control-calls.md): `step`, `pause`, `param`, `action`, `items` and `controls`.

[Live Streaming](live-streaming.md) explains the design. [Watch a PyTorch training run live](../introduction/python-quickstart.md) walks through a script.

Importing the package does not touch GLFW, AppKit, Metal or torch. The native extension loads on first use. Tensor arguments are torch tensors, and anything else goes through `torch.as_tensor`. The calls detach tensors before use.

## Constants and Types

| Name | Value |
|---|---|
| `CMAPS` | `{"viridis": 0, "magma": 1, "rdbu": 2}` |
| `SERIES_CAPACITY` | 4096 |
| `DEFAULT_RATE_HZ` | 60.0 |
| `ShowStats` | Returned in the offscreen mode, always truthy: `name`, `kind` (`"heatmap"`, ...), `path` (renderer path, such as `"compute-imported"`), `imported`, `total_ms` |
| `HAVE_VIEWS`, `HAVE_LIVE_VIEWS`, `VIEW_FLAG_RAW`, `VIEW_FLAG_WAIT_GPU`, `EMBED_API_VERSION`, `HEATMAP_IMAGE_VIA_LIVE_APPLY` | Native module attributes. Reading one loads the extension |

View names in a session are 1 to 63 bytes, and a session holds at most 64 views. Reusing a name with another kind rebuilds the view. A change of shape, dtype or device reallocates its slots.

## Environment Variables

| Variable | Effect |
|---|---|
| `CALIPER_SESSION` | `0` stops the first view call from opening a session (default `1`) |
| `CALIPER_DATA_DIR` | Root of the session files (`<root>/live`) for the script, the session list of the Caliper app and `caliper-viewer`. Also the data directory of the in-process core when no `data_dir` is given. For the Caliper app it moves the whole app-data folder ([Folders and environment variables](../reference/folders-and-environment.md#app-data-folder)) |
| `CALIPER_VIEWER` | Path of `caliper-viewer` for `auto_launch` and `viewer_executable()` |
| `CALIPER_PY_LOG` | Non-empty and not `0`: echo the log lines of the in-process core to stderr |
| `CALIPER_PY_LIVE_APPLY` | `0`: in-process heatmaps and images use the embed v1.2 view path (`view_tensor` with `WAIT_GPU` on the frame thread) instead of the typed live views |

## Threading

Any thread may call the view calls (`show`, `points`, `series`, `histogram` and the hooks of `watch`), `flush`, `remove` and `stats`. A module lock serializes them. `window()` and `offscreen()` must be called from the main thread. `run()`, `frame()` and `spin()` must also be called from the main thread while a real window is open. They raise `RuntimeError` otherwise.
