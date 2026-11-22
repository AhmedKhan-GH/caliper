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

<!-- The empty divs keep the anchors of the former hand-written entries, which the roadmap registry links to. -->
<div id="show"></div>

::: caliper.show
    options:
      heading_level: 2

<div id="points"></div>

::: caliper.points
    options:
      heading_level: 2

<div id="series"></div>

::: caliper.series
    options:
      heading_level: 2

<div id="histogram"></div>

::: caliper.histogram
    options:
      heading_level: 2

<div id="watch"></div>

::: caliper.watch
    options:
      heading_level: 2

::: caliper._watch.WatchHandle
    options:
      members: [remove, active, views, modules, groups]

## MPS Zero-Copy

A snapshot is a few torch operations queued on the stream of the tensor. They write into a slot that the route allocated as CPU-readable shared memory and handed to torch through DLPack. The session uses a shared-memory segment, and the window uses a Caliper-made shared `MTLBuffer`.

The snapshot code calls no `.item()`, `.cpu()`, `.numpy()`, `.tolist()` and no `synchronize()`. It detects completion by polling a `torch.mps.Event`. `synchronize()` appears only in `flush()` and `close()`.

A CPU tensor is copied once into the slot. Tensors on any other device (CUDA) have no zero-copy path on macOS, and one host copy moves them to the CPU first. Labels for `points` may move host to device, never device to host.
