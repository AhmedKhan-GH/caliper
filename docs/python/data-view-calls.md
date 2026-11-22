---
kind: reference
figure: none
---

# Data View Calls

The data view arguments and calls let a producer and its viewers choose which part of a tensor travels.

## Data Views

A data view decides which data of a tensor travels. It is a slice, selection, projection or aggregation that the producer computes on the device of the tensor. [Data Views](data-views.md) explains the idea, and the [Wire Protocol](../reference/wire-protocol.md#data-views) reference defines the rules exactly. A view offers data views when its tensors have named dimensions:

```python
caliper.show(t, name="tensor", ..., *, dims=None, data_view=None)
caliper.points(z, labels=None, name="points", ..., *, dims=None, data_view=None)
caliper.histogram(t, name="histogram", ..., *, dims=None, data_view=None)
caliper.offer(name, sources, kind="heatmap", dims=None, *, data_view=None, rate=None, **options)
caliper.watch(model, ..., groups=True)       # the grouped views of watch
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `dims` | sequence of `str` or `None` | `None` | One name for each dimension of the tensor, 1 to 63 bytes each, at most 8. An image names every dimension but the color one (the rule of [show](view-calls.md#show) says which one it is). Points name every dimension but the last, which holds the 2 or 3 coordinates, and end with the point dimension. With `data_view` and no `dims`, the names are `dim0`, `dim1`, ... |
| `data_view` | `dict` or `None` | `None` | The default data view of the view, in the form below |

A call with neither argument keeps the layouts of `show`, `points` and `histogram`. With either argument, the view has one source, the tensor of the call, named after the view. A default without an `index` gets one. The leading dimensions that it does not reduce or tile are indexed at 0 until the remaining dimensions fit the live view. Thus a batch shows item 0 as `show` does. A `data_view` that does not fit the tensor raises `DataViewError` at the call.

A data view is a dict in which every field is optional:

| Field | Form | Meaning |
|---|---|---|
| `source` | `str` | The source to read |
| `index` | `{"<dim>": i}`, `{"<dim>": [start, stop]}` or `{"<dim>": [start, stop, step]}` | An integer selects one position and drops the dimension. A range keeps the dimension |
| `reduce` | `{"<dim>": op}`, with `op` one of `"mean"`, `"sum"`, `"max"`, `"min"` and `"norm"` | Reduces over the dimension and removes it. `"norm"` is the square root of the sum of squares |
| `tile` | `"<dim>"` | Heatmaps only: lays out a kept dimension as a mosaic of cells |
| `budget` | `{"max_extent": n, "pool": op, "max_points": n}` | The largest heatmap or image extent (default 512, at most 4096), how a heatmap is pooled to meet it (`"mean"`, `"max"` or `"min"`, and an image always by mean), and the largest number of points (default 100,000, at most 1,000,000) |

Indices count from 0, and a negative index counts from the end, as in Python. A field that a request names replaces the field of the default as a whole. `budget` is the exception, because its keys are taken one by one. Thus `{}` is the default and `{"index": {}}` is the default without its indexing.

The remaining dimensions must fit the live view: two for a heatmap (three with `tile`) or an image, one for points (the point dimension), and any number for a histogram. A budget value under 1 is refused, and a value over the limit is taken as the limit.

The producer applies the index, then the reductions one dimension at a time in the order of the source, then the tile layout and the budget. All of this runs on the device of the tensor. Pooling covers each cell with the window of the adaptive pooling of torch. Points are sampled with a stride, point i being position floor(i × count / size), so the sample stays the same while the size does. A pooled or tiled heatmap reports the range of its data before pooling, with the empty cells left out, for autoscale.

A data view that breaks a rule is refused with the error `bad_data_view` (`DataViewError` in Python), and the view keeps the data view it had. The same happens to a data view whose payload would not fit one frame of the protocol, such as a heatmap or image of 4096 by 4096. Its message names the largest `max_extent` that fits.

**Default and subscriptions.** The default data view of the view is what the window and the live session draw. It is also where the subscription of a viewer starts. Each subscription of a viewer, such as a widget or a browser tab, has its own data view and changes it with a request over the stream. Thus two viewers can show different heads of one view at once.

The producer computes each distinct data view once per snapshot for every subscription that shares it. A source that no route needs costs nothing. A subscription whose data view no longer fits, because the sizes of the source changed, receives no snapshots until it fits again. Nothing raises in the training loop. A default that no longer fits falls back to the default that the view was registered with, then to the one that the rule above gives, with a `RuntimeWarning`.

| Call | Returns | Meaning |
|---|---|---|
| `data_view(name)` | `dict` or `None` | The default data view of the view, resolved: `source`, `index`, `reduce`, `budget`, and `tile` when the heatmap tiles. `None` when the view offers no data views or has no data yet |
| `set_data_view(name, request=None)` | `dict` or `None` | Set the default data view of the view from `request` (a dict or its JSON text). Its fields replace the fields of the current default as above, so `{}` changes nothing, and `None` returns to the default that the view was registered with. Returns the resolved data view and raises `DataViewError` when it does not fit. The next VIEWS message carries the new default, and a subscription with its own data view keeps it. A request for a view that does not exist yet, or has no data yet, is kept and checked at its first snapshot, and `None` is returned. Safe from any thread |
| `offer(name, sources, kind="heatmap", dims=None, *, data_view=None, rate=None, **options)` | `dict` or `None` | Register view `name` with several sources, for tensors that no view call shows, such as attention maps or the output of a layer on a fixed probe batch. Returns its default data view, resolved, or `None` while the default source has no tensor |
| `update(name=None)` | `bool` | Take a snapshot of every offered view that is due and of the `weights` of `watch()`, or of view `name` only. Call it in the training loop. Same return values as `show` |
| `DataViewError` | | A `ValueError` whose message names the broken rule |

`offer()` takes these arguments:

- `sources` maps source names to tensors or to callables that return the current tensor or `None`. A single tensor or callable is one source named `name`.
- `dims` is one tuple for every source or a dict of tuples per source. The default is `dim0`, `dim1`, ...
- `kind` is `"heatmap"`, `"image"`, `"points"` or `"histogram"`. A series has no sources.
- `options` are the arguments of the matching call: `cmap`, `vmin` and `vmax` (heatmap, points), `labels` (a tensor or a callable, along the kept dimension) and `classes` (points), and `bins` (histogram). An unknown option raises `TypeError`.

A tensor is read at each snapshot, so a tensor updated in place, such as a parameter, stays current. A callable runs under `torch.no_grad()`, once at the call to learn the shape and then only when a route needs that source and a snapshot is due. A later `offer()` of the same name replaces the view.

Over a server, a view called without `dims` offers one source named `data` over its payload. A heatmap or image has the dimensions `row` and `col`, points have `point`, and a histogram has none. A viewer can range or pool the source, which is how it zooms in on a plain heatmap. The default is the whole payload within the budget.

```python
caliper.show(attention, "attention", dims=("layer", "sequence", "head", "query", "key"),
             data_view={"index": {"layer": 1, "sequence": 0, "head": 0}})
widget.data_view = {"index": {"layer": 0, "sequence": 0}, "reduce": {"head": "max"}}
caliper.set_data_view("attention", {"index": {"layer": 0, "sequence": 2, "head": 1}})
```

## Related Pages

- [Caliper LIB API](caliper-lib-api.md)
- [View Calls](view-calls.md)
- [Network Streaming Calls](network-streaming-calls.md)
- [Data Views](data-views.md)
- [Wire Protocol](../reference/wire-protocol.md)
