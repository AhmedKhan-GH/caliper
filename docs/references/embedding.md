# Embedding Caliper

Caliper Embed is the framework core: the applet loader, the service registry,
the host-neutral services, `HostRenderer` (Metal or Vulkan), the tensor bridge
and the geometry ladder. Its code is the static library `libcaliper`, and the
`caliper-embed` package installs it as one shared library, `libcaliper_embed.dylib`
on macOS and `caliper_embed.dll` on Windows. It sits behind a small C ABI, so a host binary can run
the applet canvas inside a view it owns without linking ImGui, torch or any
renderer type. The `caliper` executable is the first embedder. The second
in-tree embedder is `examples/embed_host/`, an AppKit host of about 260 lines
(`main.mm`) with a Win32 counterpart (`main_win32.cpp`) and an offscreen mode
they share (`embed_host.h`), which this page mirrors. A program outside this
repository builds against the installed [`caliper-embed`
package](#the-caliper-embed-package), as `embed_host` does in the
`caliper_embed_install` test; the guide
[Embed Caliper in a program](../guides/embed-caliper.md) walks through it.

This page describes the embedder-facing contract. It is not the applet-facing
ABI: applets link `caliper::sdk` and use [`caliper/abi.h`](abi.md) and the
[C++ sugar](sugar.md). `caliper/embed.h` lives on a separate include root, so an
applet cannot `#include <caliper/embed.h>`. Embedding a core from inside an
applet would be a category error, and the include topology rules that spelling
out: the installed headers sit under `include/caliper-embed`, which no SDK
target names, and `caliper_sdk_install` checks that an applet source including
`<caliper/embed.h>` fails to compile against every SDK target, even with both
packages in one prefix. With both packages in one prefix, the header is still
reachable as `<caliper-embed/caliper/embed.h>` through the SDK's `include/`
root. Install the two packages into separate prefixes when the separation must
be complete.

The embed ABI has run on both ecosystems: Metal on Apple Silicon and
Vulkan with an `HWND` on Windows. `CALIPER_EMBED_API_VERSION` is 1. The labels
v1.1 to v1.4 in this page and in the headers name additive sections of that one
version, and they are also its minor numbers: the interface is version 1.4.0
([Versions](#versions)). The additions since the first version (service access,
per-core data root, applet log routing, built-in tensor views, typed live views,
the version call) are additive: no struct changed, so a caller built against the
first header still links and runs.

## The core calls

An embedder drives the core with five calls and pumps frames from its own event
loop. The core never owns the process loop, which is the difference between a
library and a host.

1. `caliper_core_create` spins up the core: renderer, services and loader.
2. `caliper_core_attach_canvas` hands it the native view (`NSView*` or `HWND`)
   to paint, or selects an offscreen target.
3. `caliper_core_load_applet` launches one applet by manifest id (for example
   `dev.caliper.instance-scope`). A canvas must be attached first: an applet's
   first frame touches the renderer's ImGui backend, so loading before attach is
   a refusal, not a crash.
4. `caliper_core_frame` and `caliper_core_event` pump exactly one frame from
   your loop (a timer, a `CVDisplayLink`, a Win32 message loop) and forward your
   toolkit's input as `CaliperInputEvent` values.
5. `caliper_core_shutdown` tears everything down in the reverse order of create
   and clears the one-core-per-process lock.

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

### Every function in `embed.h`

| Function | Purpose |
|---|---|
| `caliper_core_create(desc)` | Create the core. Returns `NULL` for a second live core, a renderer this platform does not build, or `CALIPER_RENDERER=gl`. The renderer starts at `attach_canvas`. |
| `caliper_core_shutdown(core)` | Tear down in reverse order of create. Safe on `NULL`. |
| `caliper_core_attach_canvas(core, native_view, desc)` | Attach the one canvas. `native_view` is ignored (pass `NULL`) for `CALIPER_CANVAS_OFFSCREEN`. Returns 1, or 0 with `last_error` set, which names the renderer and why when it could not start. |
| `caliper_core_frame(core)` | Run one frame: clear, draw the loaded applet under the crash guard, composite and present or store. |
| `caliper_core_event(core, event)` | Feed one input event. A no-op before a canvas exists. |
| `caliper_core_load_applet(core, manifest_id)` | Load and launch an applet. Any loaded applet is torn down first, so reloading the same id restarts it and a failed launch leaves no applet loaded. An unknown id is refused without disturbing the running applet. |
| `caliper_core_unload_applet(core)` | Tear down the loaded applet. A no-op if none is loaded. Call between frames. |
| `caliper_core_read_pixels(core, buf, stride)` | Copy the last composited frame as tightly packed RGBA8 with `stride` bytes per row. Offscreen canvases only. |
| `caliper_core_last_error(core)` | The reason for the most recent refusal, or an empty string. Never `NULL` for a live core, and valid until the next core call. |
| `caliper_core_get_service(core, id)` | The service table an applet receives for `id`, or `NULL` ([below](#reaching-services-from-a-host)). |
| `caliper_core_view_tensor`, `_view_remove`, `_view_count`, `_view_info`, `_view_readback` | Built-in tensor views ([below](#built-in-tensor-views)). |
| `caliper_core_colormap_lut(colormap, out256)` | The built-in 256-entry colormap table. Needs no core. |
| `caliper_embed_api_version()` | The version of the interface the loaded library implements, encoded like `CALIPER_EMBED_VERSION` ([below](#versions)). Needs no core. |

The typed live-view calls (`caliper_core_live_*`) are declared in
`caliper/live_views.h` and described under [typed live views](#typed-live-views).

## Ownership

Caliper Embed owns the applet canvas end to end. The embedder supplies a native
child view; the core runs the ImGui context, `HostRenderer`, bridge and geometry
inside it. A host's own chrome (docking, property grids, menus) wraps around
those canvases and never paints applet pixels.

| Concern | Owner | Note |
|---|---|---|
| Process event loop | **Embedder** | `caliper_core_frame` does one frame and returns — no polling, no vsync wait |
| Native window / view handle | **Embedder** | `NSView*` / `HWND` passed to `attach_canvas` |
| Input events | **Embedder** translates → core consumes | your GLFW/AppKit/wx event becomes a `CaliperInputEvent`; no toolkit types cross |
| ImGui context (one per canvas) | **Caliper Embed** | the embedder never touches ImGui state; allocator handoff stays internal |
| `HostRenderer` + tensor bridge + geometry | **Caliper Embed** | the zero-copy claim travels with the core |
| Applet loader + service registry + services | **Caliper Embed** | the same registry applets already see |
| libtorch (one per process) | **Caliper Embed** | the embedder must NOT link its own torch; the core owns device/pack policy |
| Crash containment | **Caliper Embed** | applet faults are caught by the core's guard and surfaced via `crash_fn`; the embedder is not taken down. The guard's signal handlers are installed once per process and pass faults outside applet calls on to the handlers installed before them, so an embedder's own crash reporter still sees its own crashes ([trust model](../explanations/trust-model.md#every-applet-call-is-guarded)) |

## Per-core settings and limits

- **One `CaliperCore` per process.** `caliper_core_create` returns `NULL` for a
  second live core, because the one-libtorch-per-process policy
  already binds the process. Shut the
  first down first.
- **One canvas per core.** `caliper_core_attach_canvas` refuses a second canvas.
  A canvas-info query and several canvases per core are not provided.
- **`CaliperCoreDesc.data_dir` moves the app-data folder.** A non-`NULL`
  directory (created if missing) takes the place of the
  [app-data folder](environment.md#the-app-data-folder) for that core: the
  metrics, data and artifacts stores, the applets' data folders `data/<id>/`,
  and the user applets folder `applets/` that the core searches, unless
  `CALIPER_USER_APPLETS_DIR` names another one. The window layout is kept at
  `<data_dir>/imgui.ini`. The override lasts for the one live core and is
  cleared on shutdown.
- **Without `data_dir`, `CALIPER_DATA_DIR` decides.** With `NULL` the core uses
  the folder that `CALIPER_DATA_DIR` names when it is set and not empty, and
  the operating system's folder otherwise, for the same contents. It keeps no
  window layout then: no `imgui.ini` is read or written anywhere, so every run
  starts with the default layout and nothing lands in the process's working
  directory. `CALIPER_DATA_DIR` is read once, when the process first looks the
  folder up.
- **Live session files follow `CALIPER_DATA_DIR` first.** The session files
  under `live/` (see [Live streaming](../explanations/live-streaming.md#sessions))
  are found in an explicit root when the caller gives one, else in
  `CALIPER_DATA_DIR`, and only then in the core's `data_dir` or the operating
  system's folder (`src/host/live/live_shm.h`).
- **`CaliperCoreDesc.applets_dir` is searched first.** It takes the place of the
  `caliper` executable's `--applets` folders in the
  [search order](cli.md#the-search-order), and like them a folder that holds no
  manifest but has an `applets/` subfolder that does stands for that subfolder,
  so a build folder can be passed as it is.
- **Applet log lines reach `log_fn`.** When the embedder sets `log_fn`, the
  applet `caliper.log.v1` service routes there with the prefix `[applet] `, and
  core diagnostics (renderer choice, refusals, crash text) arrive untagged.
  With `log_fn` unset both go to stderr. The core calls `log_fn` from the frame
  thread for its own lines and from any thread for applet lines, so a sink that
  touches shared UI state must marshal to its UI thread.
- **Window mode has no automated test.** The windowed canvas is checked by
  running `embed_host` on each platform. Automated byte-exactness rides the
  offscreen `read_pixels` tests, which pass on both backends, and
  `caliper_embed_install` runs `embed_host` in its offscreen mode.
- **No fallback renderer.** The core embeds Metal on macOS and Vulkan on
  Windows. When that renderer cannot start, `attach_canvas` returns 0 and
  `last_error` names it and why, as in "attach_canvas: the Metal renderer could
  not start (no Metal device), and there is no fallback renderer until the
  WebGPU fallback through Dawn lands"; the core stays usable. With
  `CALIPER_RENDERER=gl`, which asks for the removed OpenGL renderer,
  `caliper_core_create` returns `NULL` ([Refusal messages](refusals.md#renderer-refusals)).

All calls except the any-thread services below belong to one thread, the one
that calls `caliper_core_frame`.

## Versions

`embed.h` versions the interface in one encoding,
`(major << 16) | (minor << 8) | patch`, which `CALIPER_EMBED_MAKE_VERSION`
builds:

| Part | Macro | Meaning |
|---|---|---|
| Major | `CALIPER_EMBED_VERSION_MAJOR`, equal to `CALIPER_EMBED_API_VERSION` | Bumps when a struct grows, which the `struct_size` gate enforces. It is 1. |
| Minor | `CALIPER_EMBED_VERSION_MINOR` | Counts the additive sections on that major: v1.1 (services, `data_dir`, log routing), v1.2 (tensor views), v1.3 (typed live views) and v1.4 (the version call). It is 4. |
| Patch | `CALIPER_EMBED_VERSION_PATCH` | Counts library fixes that change no declaration. It is 0. |

`CALIPER_EMBED_VERSION` is the version of the header an embedder compiled
against, and `caliper_embed_api_version()` returns the version of the library
the process loaded. The library returns the `CALIPER_EMBED_VERSION` of the
header it was built from. An embedder runs on a library of the same major whose
minor is at least the header's:

```c
uint32_t lib = caliper_embed_api_version();
if ((lib >> 16) != CALIPER_EMBED_VERSION_MAJOR ||
    ((lib >> 8) & 0xff) < CALIPER_EMBED_VERSION_MINOR) {
    /* another major, or a library older than this header: do not embed it */
}
```

The macros are integer constant expressions, so `#if` can test them, and the
call needs no core and may run on any thread. `CALIPER_EMBED_HAVE_VIEWS`
remains defined as 1 for code that tested for the views before the version
call existed. The [`caliper-embed` package](#the-caliper-embed-package) carries
the same version.

## Reaching services from a host

`caliper_core_get_service(core, id)` returns the same service table an applet
receives from `CaliperHost.get_service` for that id, for example
`"caliper.metrics.v1"`, `"caliper.jobs.v1"`, `"caliper.artifacts.v1"` or
`"caliper.data.v1"`. Cast the result to the matching struct from the
`caliper/services` headers and call its function pointers as an applet would.
No renderer, torch or ImGui type crosses the seam;
the interchange
is the C ABI, `CaliperTensor` and Arrow C streams. A C++ host can use the
`caliper.hpp` sugar. The call returns `NULL` for an unknown id or a `NULL`
core. The pointer is a process-static table, valid from create until shutdown.

The header states a thread rule for each service:

| Service | Thread rule |
|---|---|
| `caliper.metrics.v1` | Any thread. The store holds one DuckDB connection under one mutex, so reads from a host thread serialize with applet writes. |
| `caliper.artifacts.v1` | Any thread, under the same one-connection model. `path_of` returns a thread-local buffer, valid until the next artifacts call on that thread. |
| `caliper.data.v1` | Any thread, under one mutex. `last_error()` is thread-local, so each thread sees its own last failure. |
| `caliper.jobs.v1` | Any thread, by design: `submit`, `cancel`, `is_running` and `progress` work over the process job system. |
| `caliper.device.v1` | Any thread. It reads a record negotiated at startup that never changes. |
| `caliper.log.v1` | Any thread. It is reentrant and routes to `log_fn` when one is installed. |
| `caliper.tensor_bridge.v1`, `v1_1`, `v1_2` | Frame thread only. They upload to the GPU. |
| `caliper.geometry.v1` to `v1_3` | Frame thread only. They draw. |
| `caliper.ui.v1` | Frame thread only. It wraps the ImGui and ImPlot contexts, which a host has no use for. |

A host pushes parameters; an applet's worker produces device tensors, because
there is one torch per process and it is never the host's. The host-side use of
the bridge is CPU-staged uploads on the frame thread.

## Built-in tensor views

The view section of `caliper/embed.h` lets a host hand the core a
`CaliperTensor` and have the **core** draw it: a texture made through the
process tensor bridge plus an ImGui window titled with the view's name,
composited in `caliper_core_frame` after the loaded applet's draw. Views work
with no applet loaded and alongside one. No ImGui, renderer or texture type
crosses the seam; the caller only names views and passes tensors. It is the
surface the Python entry point's `caliper.show` was built on.

The section is additive: `CALIPER_EMBED_API_VERSION` stays 1, no existing
struct changed, and `CALIPER_EMBED_HAVE_VIEWS` is defined as 1 so a host can
test for it at compile time.

| Call | Behaviour |
|---|---|
| `caliper_core_view_tensor(core, desc)` | Create or update the view named `desc->name`; the same name updates in place. 1 on success, 0 with `last_error` set. A failed first show leaves no view; a failed update keeps the view's previous pixels. |
| `caliper_core_view_remove(core, name)` | 1 if the view existed and was removed (texture and import released). |
| `caliper_core_view_count(core)` | Number of views. |
| `caliper_core_view_info(core, name, out)` | Fill a `CaliperViewInfo` (below). 1 on success. |
| `caliper_core_view_readback(core, name, buf, buf_bytes)` | Copy the view's **texture** (not the canvas) as tightly packed RGBA8, `width*height*4` bytes. The test and measurement surface. |
| `caliper_core_colormap_lut(colormap, out256)` | The built-in 256-entry RGBA8 LUT, each entry `r | g<<8 | b<<16 | a<<24`. Needs no core; callable from any thread. |

Windowed views are drawn in creation order and tiled left to right; the views
of a live session take the cells of a grid instead (see
[What you see](../tutorials/live-pytorch.md#what-you-see)). A small tensor is
magnified by a whole-number factor and sampled nearest, so cells stay cells.
With `CALIPER_VIEWS_DEBUG=1`, or `views_debug` set on a host's core, each
window carries an info line (`WxH dtype device path updates=N last=ms`). `CALIPER_VIEW_FLAG_RAW` draws the
texture 1:1 at canvas (0,0) with no window chrome, the surface the byte-exact
`read_pixels` tests compare.

### Accepted layouts

| Tensor | Drawn as | Rule |
|---|---|---|
| 2-D f32 `(H,W)` | Colormapped heatmap | `colormap` 0 viridis, 1 magma, 2 rdbu, over `[vmin, vmax]` (both must be finite). `idx = clamp((v-vmin)/(vmax-vmin),0,1)*255+0.5`; NaN maps to index 0, the bridge's frozen rule. |
| 3-D u8 `(H,W,4)` | RGBA image, as is | C must be 4; RGB and gray are refused. |

Both must be row-major contiguous (strides in elements), each dimension in
1..16384, on device CPU or METAL. Anything else is refused with a
`view_tensor('<name>'): ...` reason in `caliper_core_last_error`. The Python
layer converts other dtypes and shapes into one of these two on the tensor's
own device before the call; for an MPS tensor that is a GPU copy, never a host
copy.

### Two device paths, no silent fallback

- **CPU.** A CPU-staged upload of the bytes at `data + byte_offset` (path
  `cpu-staged`). The bytes are copied before the call returns; the caller may
  free or rewrite them at once.
- **METAL.** Zero-copy. `tensor->data` is an in-process `id<MTLBuffer>` (DLPack
  `kDLMetal` `data`, the storage's buffer object pointer) and
  `desc->byte_offset` is DLPack's `byte_offset`, a multiple of 4. The view
  imports the buffer (a strong retain through `tensor_bridge.v1_2`
  `import_allocation`, cached per view and re-imported when the buffer pointer
  changes) and the renderer reads it in place at `byte_offset` on the GPU: the
  colormap compute kernel for f32 (path `compute-imported`), a blit for u8
  (`blit-imported`); `info.imported = 1`.

If the device path refuses (the active renderer is not Metal, the buffer lives
on another GPU, it is shorter than `byte_offset + extent`, or the renderer
cannot import), the call **fails**. The core never quietly downloads a METAL
tensor to the host; a refusal is an error, not a slower success.

`tensor->stream` is passed through: a non-NULL `MTLCommandQueue*` GPU-orders
the read after that queue's committed work (bridge v1_1). NULL is the
normal case for a caller that made sure the producer finished first.

Measured on Apple Silicon:

| Case | CPU-staged | METAL zero-copy |
|---|---|---|
| 1024x1024 f32, mean of 20 updates (`caliper_embed_tests`) | 0.97 ms | 0.34 ms, GPU wait included |
| 64x48 f32 update (`caliper_embed_tests`) | | 0.129 ms with `WAIT_GPU`, 0.009 ms without |
| Native call, median of 50, 256² / 1024² / 2048² (`bench_show.py`, offscreen 1280x800) | 0.047 / 0.680 / 2.991 ms | 0.169 / 0.174 / 0.270 ms with the GPU wait |

The device path barely moves across a 64x range of sizes while the CPU path
scales with the bytes moved. The first METAL f32 view in a process compiles the
colormap shader once (1.6 ms with a warm OS shader cache, about 40 ms cold, in
`bench_show.py`).

### Sync and memory stability

Without `CALIPER_VIEW_FLAG_WAIT_GPU`, the GPU reads
`[byte_offset, byte_offset + extent)` of the buffer **after** the call
returns. The contract follows from that:

- **Before the call** the producer must have finished writing those bytes.
  Device-side ordering through `tensor->stream` only covers producer work
  committed before the call. The first Python `caliper.show` met this with
  `torch.mps.synchronize()`; the [live-streaming](../explanations/live-streaming.md)
  layer instead applies a snapshot only after a polled `torch.mps.Event` reports
  it complete.
- **Without `WAIT_GPU`** the bytes must not be rewritten until the next update
  of the same view (or the next `caliper_core_frame`). A rewrite in that window,
  such as an in-place `opt.step()`, gives a torn or stale frame, never a crash:
  the import retains the `MTLBuffer`, and the command buffer retains every
  resource it references until it retires.
- **With `WAIT_GPU`** the call blocks until the renderer's read has completed,
  so on return the caller may rewrite or free the buffer at once. The Metal
  renderer remembers the last device-update command buffer and waits on it;
  `info.last_update_ms` includes the wait.

The texture is the view's own copy of the pixels; the source buffer is read
only during the update. A view keeps its last `MTLBuffer` retained (the import)
until it is updated from a different buffer or removed.

### Mapping changes and view info

A different `vmin`/`vmax`/`colormap` on an existing f32 view re-pins the
bridge entry's mapping and re-maps the new tensor into the **same** texture
(`creates` does not move). A different shape or dtype recreates the texture
(`creates += 1`); the new texture is built before the old one is released, so a
failed recreate leaves the old view intact.

| `CaliperViewInfo` field | Meaning |
|---|---|
| `path` | Renderer path of the last update: `compute-imported`, `blit-imported`, `compute`, `blit` or `cpu-staged`. Valid until the next core call. |
| `width`, `height` | Texture size in texels. |
| `device` | `CaliperDeviceKind` of the last tensor. |
| `updates` | Successful updates. |
| `last_update_ms` | CPU wall time of the last `view_tensor` call, including the GPU wait when `WAIT_GPU` was set. |
| `imported` | 1 if the last update read an imported `MTLBuffer` in place. |
| `creates` | Texture (re)creations: 1 after the first show, +1 per shape or dtype change. |

### Threading

Every view function except `caliper_core_colormap_lut` is **frame-thread
only**: call it from the thread that calls `caliper_core_frame()`, between
frames, the same class as `caliper.tensor_bridge.*`. Views need an attached
canvas, because the renderer's device exists only after `attach_canvas`; a
`view_tensor` before attach is a refusal. `caliper_core_shutdown` destroys the
views, releasing textures and imports, before the renderer is torn down.

### Typed live views

`include/caliper/live_views.h` builds on this section. `caliper_core_live_apply`
takes a `CaliperLiveSnapshot` of one of five kinds (heatmap, image, points,
series, histogram) and keeps a typed view of that name. Heatmap and image live
views are drawn through the tensor view of the same name, so
`caliper_core_view_info`, `_readback`, `_count` and `_remove` keep working for
them. The header documents the per-kind payload contract, the claim rule that
`caliper_core_live_view_read_done` reports for each kind, and the per-core
pause flag. `include/caliper/live.h` is the shared-memory session protocol and
the client and viewer C API that feed it from another process. A sixth kind,
scene (6), of the wire protocol 3.2 draft, is not applied as a snapshot:
`caliper_core_live_apply` refuses it, and a stream attached with
`caliper_core_live_attach_url` opens and feeds a scene view itself, which
`caliper_core_live_view_info` (path `copied-from-host`) and
`caliper_core_live_view_readback` then answer for. Both are
macOS/Metal only today; the design is in
[Live streaming](../explanations/live-streaming.md).

## The caliper-embed package

`cmake --install <build> --prefix <prefix> --component embed` installs the
`caliper-embed` CMake package on macOS and Windows. It is separate from the SDK
(`--component sdk`) and goes into the SDK's prefix or a prefix of its own.

| Installed path | Contents |
|---|---|
| `lib/libcaliper_embed.dylib` (macOS); `bin/caliper_embed.dll` and `lib/caliper_embed.lib` (Windows) | The embedding library. |
| `include/caliper-embed/caliper/` | `embed.h`, `live.h` and `live_views.h`, with copies of the SDK's C headers they include and of the service tables `caliper_core_get_service` returns: `tensor.h`, `arrow_c.h` and `services/`. |
| `lib/cmake/caliper-embed/` | `caliper-embed-config.cmake`, its version file and the exported target. |
| `share/caliper-embed/` | `LICENSE` and `NOTICE`. |

`find_package(caliper-embed CONFIG)` gives one imported target,
`caliper::embed`, which carries the library and the include root
`include/caliper-embed`. On macOS the library needs macOS 14 (Sonoma) or
later. The package sets `CALIPER_MACOS_MIN` to its minimum, `14.0` by default,
and makes it `CMAKE_OSX_DEPLOYMENT_TARGET` unless the project,
`-DCMAKE_OSX_DEPLOYMENT_TARGET` or `MACOSX_DEPLOYMENT_TARGET` chose one.
The package version is the interface version
([Versions](#versions)), 1.4.0, with CMake's `SameMajorVersion` rule: a request
for 1.2 accepts 1.4, and a request for 2 refuses it. In the core build
`caliper::embed` is an alias of the `caliper_embed` target, so one
`target_link_libraries` line works in the tree and outside it.

### One shared library

`libcaliper` is a static library over a dozen more: `caliper_host_lib`, the
Metal or Vulkan backend, DuckDB, ImGui, ImPlot, ImPlot3D, ImGuiFileDialog and
GLFW. Shipping those archives would make every embedder link all of
them and would put their symbols beside the embedder's own copies of the same
libraries. The package ships one shared library instead, `caliper_embed`,
built from `libcaliper` and everything behind it, which exports only the C
functions that `embed.h`, `live.h` and `live_views.h` declare (56 on macOS).
Everything else stays inside it: an embedder keeps its own ImGui, GLFW or
DuckDB apart from the core's, and the frameworks the core needs (Metal,
QuartzCore, Cocoa, IOKit, CoreVideo) are load commands of the library
rather than link items of the embedder.

`cmake/CaliperEmbed.cmake` reads the export list from the three headers at
configure time. On macOS it becomes an exported-symbols list, and on Windows a
module-definition file that leaves out the `caliper_live_*` client and listing
calls, which exist on macOS only. On macOS `caliper_embed_install` checks that
the library exports exactly the C functions that `libcaliper.a` defines. The
`caliper` executable, the test binaries and the Python extension still link
the static `libcaliper`.

On macOS the install name is `@rpath/libcaliper_embed.dylib`. A program built
in its own build folder finds the library through the build rpath that CMake
adds; an installed program needs an rpath to `<prefix>/lib`. On Windows the
program needs `caliper_embed.dll` beside it or on `PATH`, and `embed_host`
copies the DLL beside itself after each build.

### Building against the package

```sh
cmake --install cmake-build-release --prefix <prefix> --component embed
cp -R examples/embed_host <work>/embed_host
cmake -S <work>/embed_host -B <work>/embed_host-build -DCMAKE_PREFIX_PATH=<prefix>
cmake --build <work>/embed_host-build
```

`examples/embed_host/CMakeLists.txt` calls `cmake_minimum_required`, `project`
and `find_package(caliper-embed 1.4 CONFIG REQUIRED)` only when it is the
top-level project, as `examples/hello` does with the SDK, and links the same
target in both builds. The frameworks serve the host's own window code:

```cmake
target_link_libraries(embed_host PRIVATE
    caliper::embed
    "-framework Cocoa" "-framework QuartzCore" "-framework Metal")
```

### The offscreen mode of embed_host

With `CALIPER_EMBED_OFFSCREEN=1`, `embed_host` runs without a window. `main()`
takes this path before it creates an `NSApplication`, a window class or a
window, so nothing appears on screen. It creates the core, attaches a
`CALIPER_CANVAS_OFFSCREEN` canvas of 1280x800, loads the applet named on the
command line, pumps frames for `CALIPER_EMBED_EXIT_AFTER` seconds (1 by
default), reads the last frame back with `caliper_core_read_pixels`, prints
what it did and exits 0 when the applet drew, otherwise 1. Both modes check
the library's version first and read the applets folder from
`CALIPER_EMBED_APPLETS`. A run of the build above on Apple Silicon:

```text
$ CALIPER_EMBED_OFFSCREEN=1 CALIPER_EMBED_APPLETS=<applets> \
      <work>/embed_host-build/embed_host dev.caliper.hello
[embed-host] offscreen mode, embed 1.4.0
[embed-host] info: renderer: metal
[feed] macOS provider: 5 channels live (cpu=1 mem=1 thermal=1 gpu=1 fan=1 temp=0 power=0) @ 10 Hz
[embed-host] info: canvas attached: 1280x800 @1.000000x (offscreen)
[embed-host] info: [applet] hello.on_init
[embed-host] info: applet loaded: dev.caliper.hello
[embed-host] info: [applet] hello.on_cleanup
[embed-host] offscreen: 43 frames of dev.caliper.hello at 1280x800 in 1.02 s, 13310 drawn pixels in the last frame
```

`caliper_embed_install` (label `embed`) runs that whole path. It installs the
embed component alone into a work folder, checks the installed tree, builds a
copy of `examples/embed_host` there with only the package on
`CMAKE_PREFIX_PATH`, checks that the build names nothing in the source tree,
and runs the copy in its offscreen mode on a copy of the `hello` fixture, with
`HOME` and the user applets folder inside the work folder. When the canvas
cannot attach, as on a machine without a GPU, the test reports Skipped, and
`CALIPER_REQUIRE_GPU=1` makes that a failure. `CALIPER_EMBED_PROBE_DIR` sets
the work folder, which CI puts outside the checkout. The Windows build of the
package and of the offscreen mode has not been run yet.

## The torch-free build

`libcaliper` (core, Metal renderer, embed C ABI) never included a torch header.
The renderers consume `CaliperTensor.data` as a raw `id<MTLBuffer>` (Metal) or
a CUDA device pointer (Vulkan), and `sdk/include/caliper/adapters/torch.hpp` is
applet-side only. The vendored libtorch 2.5.1 reached `libcaliper`, the `caliper`
exe and `embed_host` purely through `${CALIPER_DEPENDENCY_LIBS}`, to which
`cmake/Dependencies.cmake` appended `${TORCH_LIBRARIES}`. The options below make
that boundary buildable, so one process can host Caliper Embed next to a
**different** torch: the torch that the Python script imported, beside the Python extension.

In a torch-free build the core carries no libtorch at all, so the libtorch row
of the [ownership table](#ownership) describes torch builds only; the
embedding process is free to load its own torch.

### Options

| Option | Default | Effect |
|---|---|---|
| `CALIPER_WITH_TORCH` | ON | OFF skips the whole "Category 4: PyTorch" block of `cmake/Dependencies.cmake` (no download, no `find_package(Torch)`, `TORCH_LIBRARIES` empty) and every torch-linking target: the `applets/*` tree (forces `CALIPER_BUILD_APPLETS` OFF), `examples/sine_scope` (the ML tutorial applet), `caliper_torch_tests`, the applet learning-core tests and the MSVC torch-DLL copy. `caliper_embed_tests` is then compiled with `CALIPER_TESTS_NO_TORCH_APPLETS=1`, so its sine-scope swap cases compile out, and no binary carries a libtorch rpath. |
| `CALIPER_BUILD_APPLETS` | OFF | Also builds the `applets/*` folders inside the core build; each also builds on its own against the installed SDK. Forced OFF when torch is OFF, because most applets link libtorch. |
| `CALIPER_BUILD_PYTHON` | OFF | `add_subdirectory(python)`: the nanobind extensions `caliper._native` and `caliper._live`. The wheel build (scikit-build-core) sets it ON. |
| `CALIPER_WITH_SLANG` | ON | Configure the vendored slang (SystemVerilog) library. |
| `CALIPER_WITH_LLAMA` | ON | Configure the vendored llama.cpp and apply its local patches. `CALIPER_BUILD_OPENGLLAMA=ON` stops the configure with an error when this is OFF. |
| `CALIPER_BUILD_VIEWER` | ON | `caliper_viewer`, the live-viewer process (Apple only). The wheel ships it as `caliper/bin/caliper-viewer`. |

slang and llama.cpp are configured `EXCLUDE_FROM_ALL` and linked by nothing
that `libcaliper`, the `caliper` exe or the tests need; llama.cpp only backs the
opt-in archived opengllama applet. They cost about two thirds of a 12.7 s
configure, and slang clones boost_regex and mimalloc over the network at
configure time, which a pip build should not do. The Python build and the
`release-notorch` preset therefore set both OFF: a fresh torch-free configure
takes 6.5 s and clones neither. Two FetchContent clones remain, tomlplusplus
(`caliper_host_lib`) and doctest (tests only).

The wheel's configure (`[tool.scikit-build.cmake.define]` in `pyproject.toml`)
sets `CALIPER_BUILD_PYTHON` ON and `CALIPER_WITH_TORCH`,
`CALIPER_BUILD_APPLETS`, `CALIPER_BUILD_EXAMPLES`, `BUILD_TESTS`, `USE_CUDA`,
`CALIPER_WITH_SLANG` and `CALIPER_WITH_LLAMA` OFF, and builds only the
`_native`, `_live` and `caliper_viewer` targets.

### What an embedder links

`cmake/Dependencies.cmake` produces two link lists:

| List | Contents | Linked by |
|---|---|---|
| `CALIPER_DEPENDENCY_LIBS` | Everything the `caliper` exe and the gfx and export tests link: the UI chrome widgets (imgui-node-editor, ImGuiColorTextEdit, ImTerm), DuckDB and the rest. `TORCH_LIBRARIES` is no longer appended in either branch; applets and torch tests link `"${TORCH_LIBRARIES}"` themselves. | the `caliper` exe |
| `CALIPER_CORE_LIBS` | Only what the sources of `libcaliper` and the Metal and Vulkan backends need: `-framework Cocoa`, `-framework IOKit`, `-framework CoreVideo` (Apple), `glfw` and `imgui_backends`, which holds ImGui's GLFW backend. | libcaliper, `PRIVATE` |

The split matters because `libcaliper` is a static library: its private
dependencies still propagate to every embedder as link-only items, so the full
list used to drag the chrome widgets into `embed_host` and the Python
extension. DuckDB now reaches an embedder through `caliper_host_lib`, and
imgui, implot, implot3d and ImGuiFileDialog through `caliper::ui_stack`.

Inside the core build, an embedder can link `libcaliper caliper::sdk` and
nothing of the exe's list, plus whatever its own window code needs.
`examples/embed_host` links `caliper::embed` instead, the shared library of the
[package](#the-caliper-embed-package), which carries all of that inside it.
`caliper_embed_tests` links `doctest::doctest libcaliper caliper::sdk` only, so
it proves that the link interface of `libcaliper` is self-sufficient. The Python extensions
link the same two targets plus the Apple frameworks (and `glfw` for
`caliper._native`'s window).

### The release-notorch preset

`CMakePresets.json` has configure, build and test presets named
`release-notorch`: binary dir `cmake-build-notorch`, `CMAKE_BUILD_TYPE=Release`,
`BUILD_TESTS=ON`, `CALIPER_WITH_TORCH=OFF`, `CALIPER_WITH_SLANG=OFF`,
`CALIPER_WITH_LLAMA=OFF`, `USE_CUDA=OFF`. It is the configuration the Python
extension builds against.

```sh
cmake --preset release-notorch
cmake --build --preset release-notorch
ctest --preset release-notorch
./cmake-build-notorch/tests/caliper_embed_tests
```

A cold torch-free build of everything (`libcaliper`, exe, `embed_host`, all test
binaries) is 462 steps, 120 s at `-j 8`; `libcaliper.a` itself is 276 KB (its
own six objects; dependencies are separate archives). The default
`cmake --preset release` configures with torch ON (vendored libtorch 2.5.1
found), and a cold `libcaliper` there builds in 123 s
with zero torch symbols.

### Verification

Run in the torch-free tree `cmake-build-notorch`:

```text
$ nm -gU cmake-build-notorch/libcaliper.a | grep -ci "torch\|c10::\|_ZN2at"      -> 0
$ nm -u  cmake-build-notorch/libcaliper.a | grep -ci "torch\|c10\|_ZN2at"        -> 0
  (same 0 for libcaliper_host_lib.a, libcaliper_metal_backend.a, libcaliper_app_paths.a)
$ otool -L <bin> | grep -ciE 'libtorch|libc10|libomp'
  tests/caliper_embed_tests 0 | examples/embed_host/embed_host 0 | caliper 0 | tests/caliper_tests 0
$ nm <bin> | c++filt | grep -cE '[ (<,](at|c10|torch)::|_ZN3c10|_ZN2at|_ZN5torch'  -> 0 for all four
$ grep -rnE '#\s*include\s*[<"](torch|ATen|c10)' src include                      -> 0
```

A case-insensitive `torch` grep over the binaries gives false positives from
DuckDB symbol names (`OperatorChildren`, `VectorChildIndex`), hence the
namespace-anchored patterns. `otool -L` lists only system frameworks, libc++
and libobjc, and the torch-free embed binaries carry no `LC_RPATH` at all.
These lines predate the embedding library: `embed_host` now also loads
`@rpath/libcaliper_embed.dylib` through an `LC_RPATH` to the build folder, and
in the torch build `otool -L` on that library lists only system frameworks,
libc++ and libobjc.

## The header, verbatim

This page embeds the real header; the docs build fails if the file moves.

```c
--8<-- "include/caliper/embed.h"
```
