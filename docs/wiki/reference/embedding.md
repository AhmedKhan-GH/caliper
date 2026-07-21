# Embedding Caliper (libcaliper)

`libcaliper` is the framework core — applet loader, service registry, the
host-neutral services, `HostRenderer` (Metal/Vulkan), the tensor bridge, and
the geometry ladder — behind a small **C ABI** so a host binary can run the
applet canvas inside a view it owns *without linking ImGui, torch, or any
renderer type*. The `caliper` executable is its first embedder; the second
in-tree embedder is `examples/embed_host/` — a ~254-line AppKit host that this
page mirrors.

This is the **embedder-facing** contract (hosts). It is NOT the applet-facing
ABI: applets link `caliper::sdk` and use [`caliper/abi.h`](abi.md) + the
[C++ sugar](sugar.md). `caliper/embed.h` lives on a separate include root so an
applet physically cannot `#include` it — embedding a core from inside an applet
is a category error the include topology forbids.

!!! note "Status (v0)"
    L1+L2 shipped 2026-07-11 (branch `feat/libcaliper`), **run-proven on both
    ecosystems** — Metal on Apple Silicon and Vulkan/HWND on Windows (RTX 500
    Ada, 2026-07-11). Design + phase outcomes:
    `docs/superpowers/specs/2026-07-11-libcaliper-compass-design.md`.

## The five calls

An embedder drives the core with exactly five calls, pumping frames from
**its own** event loop (the core never owns the process loop — that is the
difference between a library and a host):

1. **`caliper_core_create`** — spin up the core (renderer + services + loader).
2. **`caliper_core_attach_canvas`** — hand it the native view (`NSView*` /
   `HWND`) it should paint, or an offscreen target.
3. **`caliper_core_load_applet`** — launch one applet by manifest id (e.g.
   `dev.caliper.instance-scope`). A canvas must be attached **first** — an
   applet's first frame touches the renderer's ImGui backend, so loading before
   attach is an honest refusal, not a crash.
4. **`caliper_core_frame`** / **`caliper_core_event`** — pump exactly ONE frame
   from your loop (wx idle/timer, a `CVDisplayLink`, a Win32 message loop), and
   translate your toolkit's input into `CaliperInputEvent` and forward it.
5. **`caliper_core_shutdown`** — tear it all down (the exact reverse of create)
   on window close; clears the one-core-per-process lock.

```c
CaliperCoreDesc desc = { .struct_size = sizeof desc };
desc.renderer = CALIPER_RENDERER_DEFAULT;      /* Metal on Apple, Vulkan on Windows */
CaliperCore* core = caliper_core_create(&desc);

CaliperCanvasDesc canvas = { .struct_size = sizeof canvas };
canvas.mode = CALIPER_CANVAS_WINDOW;           /* native_view is an NSView* / HWND */
canvas.width = w; canvas.height = h; canvas.content_scale = scale;
caliper_core_attach_canvas(core, native_view, &canvas);

caliper_core_load_applet(core, "dev.caliper.instance-scope");

/* ... from YOUR event loop, once per frame: */
caliper_core_frame(core);
/* ... on each input, translated to a toolkit-neutral event: */
caliper_core_event(core, &event);

caliper_core_shutdown(core);
```

## Ownership (design §4)

The load-bearing decision: **libcaliper owns the applet canvas end-to-end.**
The embedder supplies a native child view; the core runs the ImGui context,
`HostRenderer`, bridge, and geometry inside it. A host's own chrome (Compass's
wx AUI docking, property grids, menus) wraps *around* those canvases and never
paints applet pixels.

| Concern | Owner | Note |
|---|---|---|
| Process event loop | **Embedder** | `caliper_core_frame` does one frame and returns — no polling, no vsync wait |
| Native window / view handle | **Embedder** | `NSView*` / `HWND` passed to `attach_canvas` |
| Input events | **Embedder** translates → core consumes | your GLFW/AppKit/wx event becomes a `CaliperInputEvent`; no toolkit types cross |
| ImGui context (one per canvas) | **libcaliper** | the embedder never touches ImGui state; allocator handoff stays internal |
| `HostRenderer` + tensor bridge + geometry | **libcaliper** | the zero-copy claim travels with the core |
| Applet loader + service registry + services | **libcaliper** | the same registry applets already see |
| libtorch (one per process, D5) | **libcaliper** | the embedder must NOT link its own torch; the core owns device/pack policy |
| Crash containment | **libcaliper** | applet faults are caught by the core's guard and surfaced via `crash_fn`; the embedder is not taken down |

## Honest caveats (v0)

- **One `CaliperCore` per process.** `caliper_core_create` refuses a second
  live core with a `NULL` return (the one-libtorch-per-process policy, D5,
  already binds the process); shut the first down first.
- **`CaliperCoreDesc.data_dir` is IGNORED.** The process app-data path is always
  used; threading a per-core data root is a `host_services` signature change
  deferred past R4.
- **The applet `caliper.log.v1` service bypasses `log_fn`** and writes to process
  stderr in v0. Core diagnostics (renderer pick, refusals, crash text) DO route
  through `log_fn`; applet log lines do not.
- **`CALIPER_CANVAS_WINDOW` has no ctest coverage.** The windowed canvas is
  run-proven live on both OSes — Metal (Apple Silicon) and Vulkan/HWND (Windows,
  RTX 500 Ada) — but there is no automated coverage of window mode on either;
  the live `embed_host` run stays the ritual. Automated byte-exactness rides the
  OFFSCREEN `read_pixels` battery (the §7 host-axis byte-compare), green on both
  backends.
- **GL is not an embed target.** GL's context ownership is GLFW-coupled chrome
  (D13, the frozen fallback); a core whose resolved backend is GL refuses at
  `attach_canvas` ("embed requires Metal or Vulkan").

## The header, verbatim

This page embeds the real header; the docs build fails if the file moves.

```c
--8<-- "include/caliper/embed.h"
```
