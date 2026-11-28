---
kind: reference
figure: none
---

# Routes and Inspection

The `caliper` package sends each view call to the routes that are open, and it offers calls to open and close routes and to inspect the in-process core.

## Routing

A call is offered to every **route** that is open, in this order:

1. The **live session** (attach to app). The first view call of a script opens it when no route is open yet, unless `CALIPER_SESSION=0`. If the build has no session transport, that first call warns once. Then calls do nothing until `window()` or `offscreen()` opens a route.
2. The **in-process core**: the window of `caliper.window()` or the canvas of `caliper.offscreen()`.

Each route that is **attached** takes its own snapshot of the call. The session is attached while the heartbeat of a viewer is younger than 1 s, and the in-process core is attached while it is open. The device-side preparation is built once and shared. With a viewer attached and a window open, both receive every snapshot. With nothing attached, the call returns `False` after the check.

If `caliper.window()` opens before the first view call, no session opens. Call `caliper.session()` as well to have both.

View calls return `True` if at least one route took a snapshot. They return `False` when every route skipped it (rate limit, all slots busy, nothing attached). In the synchronous offscreen mode they return a [`ShowStats`][caliper.ShowStats].

Name, colormap and range arguments are checked on every call. Checks on the tensor itself (shape, label count, bins) run when a snapshot is built. Thus a bad shape raises only when something is attached.

## Routes and Lifetime

::: caliper.session

::: caliper.window

::: caliper.offscreen

::: caliper._inproc.Window
    options:
      members: [run, stats, render, close, closed]

::: caliper.run

::: caliper.spin

::: caliper.flush

::: caliper.close

::: caliper.remove

## Inspection and Test Helpers

These calls read the in-process core. They raise `RuntimeError` without a `window()` or `offscreen()`, except where noted. Arrays are numpy. `view_info(name)` also returns the copy counters `copies`, `copy_bytes`, `last_copies` and `last_copy_bytes` ([Copy Counters and Frame Statistics](../instruments/rendering.md#copy-counters-and-frame-statistics)). `live_view_info(name)` returns the same counters.

::: caliper.stats

::: caliper.frame

::: caliper.render

::: caliper.read_pixels

::: caliper.view_info

::: caliper.view_readback

::: caliper.view_remove

::: caliper.view_count

::: caliper.live_view_info

::: caliper.live_view_values

::: caliper.live_view_readback

::: caliper.live_pause

::: caliper.live_paused

::: caliper.colormap_lut

::: caliper.log_lines

## `caliper.live`

`from caliper import live` imports the module. `import caliper` does not import it.

::: caliper.live.list_sessions

::: caliper.live.data_root

::: caliper.live.viewer_executable

::: caliper.live.SessionTransport
    options:
      members: [session, session_file, closed, launch_viewer, viewer, viewer_alive, wait_attached, wait_viewer, stats, close]
