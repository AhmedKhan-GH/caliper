---
kind: reference
figure: none
---

# View Calls

The view calls of the `caliper` package turn tensors into live views that every open route receives.

## Call Cost

With nothing attached, a view call is one atomic load and a clock read. It takes no snapshot and no slot, and it does no GPU work.

## Rate Limit

Each view has a rate limit for each route. The `rate` argument of the call sets it.

| `rate` | Meaning |
|---|---|
| `None` (default) | The default of the route: 60 snapshots per second (`DEFAULT_RATE_HZ`) for the session and the window, and every call for `offscreen()`. `window(rate=...)` changes the window default |
| `> 0` | At most that many snapshots per second for this view |
| `0` | A snapshot on every call, when a slot is free |

The interval starts at the last snapshot of the view. A call inside the interval returns `False`. A view other than a series has 3 slots. A slot is busy when the GPU has not finished an earlier snapshot or the viewer still reads it. If all slots are busy, the call takes no snapshot, returns `False` and does not restart the interval. A negative or NaN `rate` raises `ValueError`.

A `series` is the exception. Every call appends its values on every attached route, and the view is republished at most `rate` times per second, with one pending snapshot at a time. `watch(every=N)` drops N-1 of N hook calls before the rate limit applies.

## `show`

```python
caliper.show(t, name="tensor", cmap="viridis", vmin=None, vmax=None, rate=None, *,
             dims=None, data_view=None)
```

Show or update view `name` with tensor `t` as a heatmap or an RGBA image. With `dims` or `data_view`, the view shows a data view of `t` instead of the layouts below. See [Data Views](data-view-calls.md#data-views).

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `t` | tensor | | See the layout table below |
| `name` | `str` | `"tensor"` | View id. The same name updates the view in place. Non-empty, no `##` |
| `cmap` | `str` or `int` | `"viridis"` | `"viridis"`, `"magma"`, `"rdbu"` (case-insensitive) or 0, 1, 2 |
| `vmin`, `vmax` | `float` or `None` | `None` | Fixed ends of the colormap range. `None` takes that end from a GPU `aminmax` of the snapshot. Each end is independent. Both must be finite |
| `rate` | `float` or `None` | `None` | See [Rate Limit](#rate-limit) |

| Input | Shown as |
|---|---|
| 0-D or 1-D | A 1xN heatmap strip |
| 2-D `(H, W)` | Heatmap (cast to float32 in the copy kernel) |
| `(H, W, 3)` or `(H, W, 4)` | RGBA image |
| `(1, H, W)` | Heatmap |
| `(3, H, W)` or `(4, H, W)` | RGBA image |
| `(C, H, W)`, any other C | Channel mosaic heatmap: ceil(sqrt(C)) columns, empty cells NaN, range over all channels |
| `(N, C, H, W)` | Item 0, then as above |
| More than 4 dims | Leading dims stacked as channels, `(-1, H, W)` |

A 3-D tensor whose last dimension is 3 or 4 is read as `(H, W, C)` if it is uint8 or its first dimension is not 1, 3 or 4. Otherwise it is read as `(C, H, W)`.

Image values follow these rules: uint8 stays as it is, floating point is clamped to [0, 1] and scaled to 255, and other integers are clamped to 0..255. A missing alpha channel is 255. Booleans become float32 (uint8 for an image), and complex tensors become their magnitude. With autoscale, a non-finite range keeps the last finite one, and a flat range widens by ±0.5. NaN values draw as colormap index 0.

## `points`

```python
caliper.points(z, labels=None, name="points", classes=None, cmap="viridis",
               vmin=None, vmax=None, rate=None, *, dims=None, data_view=None)
```

A scatter of `z`: `(N, 2)` (pan and zoom) or `(N, 3)` (orbit). Any dtype is cast to float32 on its device, and N must be at least 1.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `labels` | tensor `(N,)` or `None` | `None` | Per-point labels or values, on the device of `z` or on the CPU. CPU labels for MPS points are copied to the device without waiting for its queue, from a private copy. The caller may reuse its tensor at once. Nothing is copied the other way. `None` draws one flat color |
| `classes` | `int` or `None` | `None` | K categorical labels 0..K-1, drawn with a fixed 16-color palette (label L gets color L mod 16). Needs `labels` |
| `cmap`, `vmin`, `vmax` | | | Without `classes`, labels are colormapped over [vmin, vmax], autoscaled like `show` |
| `name`, `rate` | | `"points"`, `None` | As for `show` |

The bounding box for auto-fit is computed on the device. These cases raise `ValueError`: a wrong shape, a label count other than N, labels on a device other than the device of `z` or the CPU, `classes < 1`, and `classes` without `labels`. With `dims` or `data_view`, `z` is a source of data views and `labels` follows its kept dimension. See [Data Views](data-view-calls.md#data-views).

## `series`

```python
caliper.series(value, name="series", rate=None)
```

Append `value` to the series `name`. The value is a Python `int` or `float`, a 0-d tensor, or a tensor of several values. Pass `loss.detach()`, not `loss.item()`, to avoid a GPU sync. A tensor of several values is flattened, and more than 4096 values keep the last 4096.

The series is a ring of the last `SERIES_CAPACITY` (4096) float32 values on the device of the value. A CPU series that later receives device values moves to the device with one host-to-device copy of its history. The x axis is the append index. Values offered while nothing is attached are not recorded. An empty tensor raises `ValueError`.

## `histogram`

```python
caliper.histogram(t, name="histogram", bins=64, rate=None, *, dims=None, data_view=None)
```

Bin counts of every element of `t` over its [min, max]. The tensor is flattened and cast to float32, and it needs at least one element. `bins` is in 1..65536. The range comes from a GPU `aminmax`. Constant data uses [v - 1, v + 1] as `torch.histc` does, and NaN values fall into bin 0.

On MPS the binning runs on the device (`index_add_`) because `torch.histc` on MPS reads its range back to the host and drains the queue. On the CPU it is `torch.histc`. With `dims` or `data_view`, the histogram bins the data view of `t`. See [Data Views](data-view-calls.md#data-views).

## `watch`

```python
handle = caliper.watch(model, activations=True, gradients=False, every=1,
                       include=None, rate=None, groups=True)
```

Hooks that call `show` for the outputs and gradients of a module. The call returns a `WatchHandle`.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `model` | `torch.nn.Module` | | Anything else raises `TypeError` |
| `activations` | `bool` | `True` | A forward hook per selected module shows its output as `<module>.act`. A tuple output gives its first tensor. Only floating-point, non-empty tensors show. `(N, C, H, W)` outputs show item 0 as a channel mosaic, and `(N, F)` outputs show a heatmap |
| `gradients` | `bool` | `False` | Each selected module shows the gradient of its own parameters that have 2 or more dims and `requires_grad`, as `<module>.<param>.grad` with colormap rdbu. Higher-rank weights are reshaped to `(out, -1)` |
| `every` | `int` | 1 | Show 1 of every N hook calls (per hook). Must be at least 1 |
| `include` | `None`, `str`, list of `str`, or callable | `None` | `None` selects leaf modules. Strings are `fnmatch` patterns over `named_modules()` names. A callable is `predicate(name, module) -> bool` |
| `rate` | | `None` | Passed to `show` |
| `groups` | `bool` or `str` | `True` | Also offer the grouped views below. A string prefixes their names with `<string>.`, and `False` offers none |

The root module is labeled with its class name in lower case. `WatchHandle` has these members: `remove()` (takes every hook off and removes the grouped views), `active`, `views` (the view names), `modules` (the hooked module names), `groups` (the grouped view names) and `len()`. It also works as a context manager.

The grouped views are heatmaps with [data views](data-view-calls.md#data-views). Their sources are the selected modules or their parameters, so one widget can switch between layers. The same hooks feed them, so `watch()` adds no hook for them.

| View | Sources | Dims | Default data view |
|---|---|---|---|
| `weights` | `<module>.<param>`, every parameter with 2 or more dims | `out`, `in` (higher-rank weights reshaped to `(out, -1)`) | The first source, whole |
| `activations` | `<module>`, with `activations=True` | From the rank: `batch`, `feature`, or `batch`, `position`, `feature`, or `batch`, `channel`, `height`, `width`, or `batch`, `channel`, `length` for a 1-D convolution | The first module, its leading dims at index 0 |
| `gradients` | `<module>.<param>`, with `gradients=True`, as the `.grad` views | As `weights` | The first source, whole |

The weights and gradients have the same two dims, so `{"source": "fc2.weight"}` alone switches between them. Activations of different ranks have different dims. A request for another module must name its own `index`, and `{}` for a 2-D output.

`weights` takes a snapshot when its module runs forward and on `caliper.update()`. With `activations=False`, it takes the snapshot after its gradient accumulates. A source does tensor work only when a route needs it. The window and the session need the source of the default data view, and a server needs the sources that its subscriptions select. The hook of any other source records the shape and returns.

A view has at most 64 sources, so a group keeps the first 64. Sources with names longer than 63 bytes are left out, each with a `RuntimeWarning`.

## MPS Zero-Copy

A snapshot is a few torch operations queued on the stream of the tensor. They write into a slot that the route allocated as CPU-readable shared memory and handed to torch through DLPack. The session uses a shared-memory segment, and the window uses a Caliper-made shared `MTLBuffer`.

The snapshot code calls no `.item()`, `.cpu()`, `.numpy()`, `.tolist()` and no `synchronize()`. It detects completion by polling a `torch.mps.Event`. `synchronize()` appears only in `flush()` and `close()`.

A CPU tensor is copied once into the slot. Tensors on any other device (CUDA) have no zero-copy path on macOS, and one host copy moves them to the CPU first. Labels for `points` may move host to device, never device to host.
