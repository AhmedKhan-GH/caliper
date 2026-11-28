# Embedding API

Caliper Embed is the framework core. It holds the applet loader, the service registry, the host-neutral services, `HostRenderer` (Metal or Vulkan) and the tensor bridge. Its code is the static library `libcaliper`. The `caliper-embed` package installs it as one shared library, `libcaliper_embed.dylib` on macOS and `caliper_embed.dll` on Windows. A small C ABI sits in front of the library. Thus a host binary can run the applet canvas inside a view it owns without linking ImGui, torch or any renderer type.

The `caliper` executable is the first embedder. The second is `examples/embed_host/`, an AppKit host with a Win32 counterpart and an offscreen mode, and this page mirrors it. A program outside this repository builds against the installed [`caliper-embed` package](#caliper-embed-package). The guide [Embed Caliper in a program](program-embedding.md) shows the steps.

This page describes the embedder-facing contract, not the applet-facing ABI. Applets link `caliper::sdk` and use [`caliper/abi.h`](../applets/applet-contract.md) and the [C++ sugar](../applets/cpp-sugar.md). The header `caliper/embed.h` sits on a separate include root, so an applet cannot `#include <caliper/embed.h>`. If both packages share one prefix, the header is still reachable as `<caliper-embed/caliper/embed.h>` through the `include/` root of the SDK. Install the two packages into separate prefixes when the separation must be complete.

The ABI runs with Metal on Apple Silicon and with Vulkan and an `HWND` on Windows. `CALIPER_EMBED_API_VERSION` is 1. The labels v1.1 to v1.4 in this page and in the headers name additive sections of that version, and they are also its minor numbers. The interface is version 1.4.0 ([Versions](#versions)). No struct changed in these sections, so a caller built against the first header still links and runs.

## Core Calls

An embedder drives the core with five calls and pumps frames from its own event loop. The core never owns the process loop, which is the difference between a library and a host.

1. `caliper_core_create` creates the core: renderer, services and loader.
2. `caliper_core_attach_canvas` gives the core the native view (`NSView*` or `HWND`) to paint, or selects an offscreen target.
3. `caliper_core_load_applet` launches one applet by manifest id, for example `dev.caliper.hello-world`. Attach a canvas first, because the first frame of an applet touches the ImGui backend of the renderer. Loading before attach is a refusal, not a crash.
4. `caliper_core_frame` and `caliper_core_event` pump exactly one frame from your loop (a timer, a `CVDisplayLink` or a Win32 message loop) and forward your toolkit input as `CaliperInputEvent` values.
5. `caliper_core_shutdown` tears everything down in the reverse order of create and clears the one-core-per-process lock.

```c
CaliperCoreDesc desc = { .struct_size = sizeof desc };
desc.renderer = CALIPER_RENDERER_DEFAULT;      /* Metal on Apple, Vulkan on Windows */
CaliperCore* core = caliper_core_create(&desc);

CaliperCanvasDesc canvas = { .struct_size = sizeof canvas };
canvas.mode = CALIPER_CANVAS_WINDOW;           /* native_view is an NSView* / HWND */
canvas.width = w; canvas.height = h; canvas.content_scale = scale;
caliper_core_attach_canvas(core, native_view, &canvas);

caliper_core_load_applet(core, "dev.caliper.hello-world");

/* ... from YOUR event loop, once per frame: */
caliper_core_frame(core);
/* ... on each input, translated to a toolkit-neutral event: */
caliper_core_event(core, &event);

caliper_core_shutdown(core);
```

### `embed.h` Functions

The lifecycle, applet control and readback calls:

<!-- c-api calls include/caliper/embed.h caliper_core_create caliper_core_shutdown caliper_core_attach_canvas caliper_core_frame caliper_core_event caliper_core_load_applet caliper_core_unload_applet caliper_core_read_pixels caliper_core_last_error -->

The service call is in [Host Service Access](#host-service-access), the view calls are in [Built-In Tensor Views](#built-in-tensor-views) and the version call is in [Versions](#versions). The header `caliper/live_views.h` declares the typed live-view calls (`caliper_core_live_*`), described in [Typed Live Views](#typed-live-views).

### `embed.h` Types

<!-- c-api types include/caliper/embed.h CaliperCore CaliperRenderer CaliperLogFn CaliperCrashFn CaliperCoreDesc CaliperCanvasMode CaliperCanvasDesc CaliperEventType CaliperInputEvent level=4 -->

The modifier flags of `CaliperInputEvent.mods` are constants:

<!-- c-api ids include/caliper/embed.h CALIPER_MOD_* -->

## Ownership

Caliper Embed owns the applet canvas end to end. The embedder supplies a native child view, and the core runs the ImGui context, `HostRenderer` and bridge inside it. The chrome of a host (docking, property grids, menus) wraps around those canvases and never paints applet pixels.

| Concern | Owner | Note |
|---|---|---|
| Process event loop | **Embedder** | `caliper_core_frame` does one frame and returns, with no polling and no vsync wait |
| Native window / view handle | **Embedder** | `NSView*` / `HWND` passed to `attach_canvas` |
| Input events | **Embedder** translates, core consumes | Your GLFW, AppKit or wx event becomes a `CaliperInputEvent`. No toolkit types cross. |
| ImGui context (one per canvas) | **Caliper Embed** | The embedder never touches ImGui state |
| `HostRenderer` + tensor bridge | **Caliper Embed** | The zero-copy claim travels with the core |
| Applet loader + service registry + services | **Caliper Embed** | The same registry that applets see |
| libtorch (one per process) | **Caliper Embed** | The embedder must NOT link its own torch, because the core owns the device and pack policy |
| Crash containment | **Caliper Embed** | The core guard catches applet faults and reports them through `crash_fn`. The embedder stays up. |

The signal handlers of the guard are installed once per process. They pass faults outside applet calls on to the handlers installed before them, so the crash reporter of an embedder still sees its own crashes ([trust model](../applets/trust-model.md#guarded-applet-calls)).

## Per-Core Settings and Limits

- **One `CaliperCore` per process.** `caliper_core_create` returns `NULL` for a second live core, because the one-libtorch-per-process policy already binds the process. Shut the first core down first. It also returns `NULL` for a renderer that the platform does not build, and for `CALIPER_RENDERER=gl`. The renderer itself starts at `caliper_core_attach_canvas`.
- **One canvas per core.** `caliper_core_attach_canvas` refuses a second canvas. It returns 0 with `last_error` set when the renderer could not start, and the message names the renderer and the reason. A canvas-info query and several canvases per core are not provided.
- **`CaliperCoreDesc.data_dir` moves the app-data folder.** A non-`NULL` directory (created if missing) replaces the [app-data folder](../reference/folders-and-environment.md#app-data-folder) for that core. This covers the metrics, data and artifacts stores, the applet data folders `data/<id>/`, and the user applets folder `applets/` that the core searches, unless `CALIPER_USER_APPLETS_DIR` names another one. The window layout is kept at `<data_dir>/imgui.ini`. The override lasts for the one live core and ends at shutdown.
- **Without `data_dir`, `CALIPER_DATA_DIR` decides.** With `NULL` the core uses the folder that `CALIPER_DATA_DIR` names when it is set and not empty. Otherwise it uses the folder of the operating system. The core then keeps no window layout: it reads and writes no `imgui.ini`, so every run starts with the default layout and nothing lands in the working directory. The process reads `CALIPER_DATA_DIR` once, at the first lookup.
- **Live session files follow `CALIPER_DATA_DIR` first.** The session files under `live/` (see [Live Session Layout](../development/live-session-layout.md#session-file)) are found in an explicit root when the caller gives one. Otherwise the core looks in `CALIPER_DATA_DIR`, and only then in its `data_dir` or the folder of the operating system.
- **`CaliperCoreDesc.applets_dir` is searched first.** It replaces the `--applets` folders of the `caliper` executable in the [search order](../reference/cli.md#search-order). As with those folders, a folder with no manifest but with an `applets/` subfolder that has one stands for that subfolder, so you can pass a build folder as it is.
- **Applet log lines reach `log_fn`.** When the embedder sets `log_fn`, the applet `caliper.log.v1` service routes there with the prefix `[applet] `. Core diagnostics (renderer choice, refusals, crash text) arrive untagged. With `log_fn` unset, both go to stderr. The core calls `log_fn` from the frame thread for its own lines and from any thread for applet lines. A sink that touches shared UI state must marshal to its UI thread.
- **Window mode has no automated test.** Run `embed_host` on each platform to check a windowed canvas. The offscreen `read_pixels` tests check byte-exactness on both backends.
- **The Windows package build is untested.** The package and its offscreen mode have not run on Windows.
- **No fallback renderer.** The core embeds Metal on macOS and Vulkan on Windows. When that renderer cannot start, `attach_canvas` returns 0 and `last_error` names it and the reason, as in "attach_canvas: the Metal renderer could not start (no Metal device), and there is no fallback renderer until the WebGPU fallback through Dawn lands". The core stays usable. `CALIPER_RENDERER=gl` asks for the removed OpenGL renderer, and `caliper_core_create` returns `NULL` for it ([Renderer Refusals](../reference/refusal-messages.md#renderer-refusals)).
- **METAL views do not copy tensor bytes.** The cost of a view update on the METAL path does not grow with the tensor size ([Device Paths](#device-paths)).

All calls except the any-thread services below belong to one thread, the one that calls `caliper_core_frame`.

## Versions

`embed.h` versions the interface in one encoding, `(major << 16) | (minor << 8) | patch`, which `CALIPER_EMBED_MAKE_VERSION` builds:

<!-- c-api ids include/caliper/embed.h CALIPER_EMBED_API_VERSION CALIPER_EMBED_MAKE_VERSION CALIPER_EMBED_VERSION* -->

The call `caliper_embed_api_version` returns the version of the loaded library:

<!-- c-api calls include/caliper/embed.h caliper_embed_api_version -->

`CALIPER_EMBED_VERSION` is the version of the header that an embedder compiled against. `caliper_embed_api_version()` returns the version of the library that the process loaded, which is the `CALIPER_EMBED_VERSION` of the header the library was built from. An embedder runs on a library of the same major whose minor is at least the minor of the header:

```c
uint32_t lib = caliper_embed_api_version();
if ((lib >> 16) != CALIPER_EMBED_VERSION_MAJOR ||
    ((lib >> 8) & 0xff) < CALIPER_EMBED_VERSION_MINOR) {
    /* another major, or a library older than this header: do not embed it */
}
```

The macros are integer constant expressions, so `#if` can test them. The call needs no core and may run on any thread. `CALIPER_EMBED_HAVE_VIEWS` stays defined as 1 for code that tests for the views. The [`caliper-embed` package](#caliper-embed-package) carries the same version.

## Host Service Access

<!-- c-api calls include/caliper/embed.h caliper_core_get_service -->

`caliper_core_get_service(core, id)` returns the same service table that an applet receives from `CaliperHost.get_service` for that id. Examples are `"caliper.metrics.v1"`, `"caliper.jobs.v1"`, `"caliper.artifacts.v1"` and `"caliper.data.v1"`. Cast the result to the matching struct from the `caliper/services` headers and call its function pointers as an applet would. No renderer, torch or ImGui type crosses the seam. The interchange is the C ABI, `CaliperTensor` and Arrow C streams. A C++ host can use the `caliper.hpp` sugar. The call returns `NULL` for an unknown id or a `NULL` core. The pointer is a process-static table, valid from create until shutdown.

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
| `caliper.ui.v1` | Frame thread only. It wraps the ImGui and ImPlot contexts, which a host has no use for. |

A host pushes parameters, and an applet worker produces device tensors, because there is one torch per process and it is never the host torch. The host-side use of the bridge is CPU-staged uploads on the frame thread.

## Built-In Tensor Views

The view section of `caliper/embed.h` lets a host hand the core a `CaliperTensor` and have the **core** draw it. The core makes a texture through the process tensor bridge and adds an ImGui window titled with the view name. It composites the window in `caliper_core_frame` after the draw of the loaded applet. Views work with no applet loaded and alongside one. No ImGui, renderer or texture type crosses the seam, because the caller only names views and passes tensors. The `caliper.show` call of the Python entry point uses this surface.

The section is additive: `CALIPER_EMBED_API_VERSION` stays 1 and no existing struct changed. `CALIPER_EMBED_HAVE_VIEWS` is defined as 1, so a host can test for views at compile time.

<!-- c-api calls include/caliper/embed.h caliper_core_view_* caliper_core_colormap_lut -->

<!-- c-api types include/caliper/embed.h CaliperViewDesc -->

<!-- c-api ids include/caliper/embed.h CALIPER_VIEW_FLAG_* CALIPER_EMBED_HAVE_VIEWS -->

Windowed views are drawn in creation order and tiled left to right. The views of a live session take the cells of a grid instead (see [What you see](../python/script-attachment.md)). A small tensor is magnified by a whole-number factor and sampled nearest, so cells stay cells. With `CALIPER_VIEWS_DEBUG=1`, or `views_debug` set on the core of a host, each window carries an info line (`WxH dtype device path updates=N last=ms`). `CALIPER_VIEW_FLAG_RAW` draws the texture 1:1 at canvas (0,0) with no window chrome. The byte-exact `read_pixels` tests compare this surface.

### Accepted Layouts

| Tensor | Drawn as | Rule |
|---|---|---|
| 2-D f32 `(H,W)` | Colormapped heatmap | `colormap` 0 viridis, 1 magma, 2 rdbu, over `[vmin, vmax]` (both must be finite). `idx = clamp((v-vmin)/(vmax-vmin),0,1)*255+0.5`. NaN maps to index 0. |
| 3-D u8 `(H,W,4)` | RGBA image, as is | C must be 4. RGB and gray are refused. |

Both layouts must be row-major contiguous (strides in elements), with each dimension in 1..16384, on device CPU or METAL. The core refuses anything else with a `view_tensor('<name>'): ...` reason in `caliper_core_last_error`. The Python layer converts other dtypes and shapes into one of these two layouts on the own device of the tensor before the call. For an MPS tensor this is a GPU copy, never a host copy.

### Device Paths

- **CPU.** A CPU-staged upload of the bytes at `data + byte_offset` (path `cpu-staged`). The core copies the bytes before the call returns, so the caller may free or rewrite them at once.
- **METAL.** Zero-copy. `tensor->data` is an in-process `id<MTLBuffer>` (DLPack `kDLMetal` `data`, the buffer object pointer of the storage). `desc->byte_offset` is the DLPack `byte_offset`, a multiple of 4. The view imports the buffer with a strong retain through `tensor_bridge.v1_2` `import_allocation`. It caches the import per view and imports again when the buffer pointer changes. The renderer reads the buffer in place at `byte_offset` on the GPU. It uses the colormap compute kernel for f32 (path `compute-imported`) and a blit for u8 (`blit-imported`), and sets `info.imported = 1`.

If the device path refuses, the call **fails**. A refusal happens when the active renderer is not Metal, the buffer lives on another GPU, the buffer is shorter than `byte_offset + extent`, or the renderer cannot import. The core never quietly downloads a METAL tensor to the host. A refusal is an error, not a slower success.

`tensor->stream` is passed through. A non-NULL `MTLCommandQueue*` orders the GPU read after the committed work of that queue (bridge v1_1). NULL is the normal case for a caller that made sure the producer finished first.

### Sync and Memory Stability

Without `CALIPER_VIEW_FLAG_WAIT_GPU`, the GPU reads `[byte_offset, byte_offset + extent)` of the buffer **after** the call returns. The contract follows from that:

- **Before the call**, the producer must have finished writing those bytes. Device-side ordering through `tensor->stream` covers only producer work committed before the call. The Python `caliper.show` call uses `torch.mps.synchronize()`. The [live-streaming](../python/live-streaming.md) layer applies a snapshot only after a polled `torch.mps.Event` reports it complete.
- **Without `WAIT_GPU`**, do not rewrite the bytes until the next update of the same view or the next `caliper_core_frame`. A rewrite in that window, such as an in-place `opt.step()`, gives a torn or stale frame, never a crash. The import retains the `MTLBuffer`, and the command buffer retains every resource it references until it retires.
- **With `WAIT_GPU`**, the call blocks until the read of the renderer completes. On return the caller may rewrite or free the buffer at once. The Metal renderer waits on the last device-update command buffer, and `info.last_update_ms` includes the wait.

The texture is the own copy of the pixels in the view, and the core reads the source buffer only during the update. A view keeps its last `MTLBuffer` retained (the import) until the next update from a different buffer or until removal.

### Mapping Changes and View Info

A different `vmin`, `vmax` or `colormap` on an existing f32 view re-pins the mapping of the bridge entry. The core then maps the new tensor into the **same** texture, and `creates` does not change. A different shape or dtype recreates the texture (`creates += 1`). The core builds the new texture before it releases the old one, so a failed recreate leaves the old view intact.

<!-- c-api types include/caliper/embed.h CaliperViewInfo level=0 -->

### Threading

Every view function except `caliper_core_colormap_lut` is **frame-thread only**. Call it from the thread that calls `caliper_core_frame()`, between frames, as with `caliper.tensor_bridge.*`. Views need an attached canvas, because the device of the renderer exists only after `attach_canvas`. A `view_tensor` call before attach is a refusal. `caliper_core_shutdown` destroys the views, and releases textures and imports, before it tears down the renderer.

### Typed Live Views

The header `caliper/live_views.h` builds on this section. `caliper_core_live_apply` takes a `CaliperLiveSnapshot` of one of five kinds (heatmap, image, points, series, histogram) and keeps a typed view of that name. A sixth kind, scene (6), belongs to the wire protocol 3.2 draft and is not applied as a snapshot: `caliper_core_live_apply` refuses it. A stream attached with `caliper_core_live_attach_url` opens and feeds a scene view itself, which `caliper_core_live_view_info` (path `copied-from-host`) and `caliper_core_live_view_readback` then answer for. Heatmap and image live views are drawn through the tensor view of the same name, so `caliper_core_view_info`, `_readback`, `_count` and `_remove` keep working for them. The header documents the payload contract for each kind in `CaliperLiveSnapshot`, the claim rule that `caliper_core_live_view_read_done` reports for each kind, and the pause flag of each core. The header `caliper/live.h` is the shared-memory session protocol, with the client and viewer C API that feed it from another process. Both work on macOS with Metal only. The design is in [Live streaming](../python/live-streaming.md).

All typed live-view calls run on the frame thread, between frames and after `attach_canvas`. `caliper_core_shutdown` destroys the views.

<!-- c-api calls include/caliper/live_views.h -->

<!-- c-api types include/caliper/live_views.h level=4 -->

<!-- c-api ids include/caliper/live_views.h -->

## Live Session Attachment

An embed host attaches to the live session of a training script with the C calls of `caliper/live.h`. The Caliper app uses the same calls for "Attach to process", and `caliper-viewer` uses them too. The app owns its window and renderer, so it attaches through a hosted core that borrows them, and its session list follows the same rules as the viewer's list.

Listing sessions works from any thread. The `caliper_core_live_*` calls need an embed core with an attached canvas, and they run on its frame thread:

```c
#include <caliper/live.h>

/* List sessions (any thread). struct_size must be set on every entry. */
CaliperLiveSessionInfo s[16];
for (int i = 0; i < 16; ++i) s[i].struct_size = sizeof s[i];
int n = caliper_live_list_sessions(s, 16);      /* newest first; -1 + caliper_live_last_error() */

/* Attach: by token, by pid, by a stream link, or show the built-in session list. */
caliper_core_live_attach(core, s[0].session);
caliper_core_live_attach_pid(core, pid, NULL);  /* NULL = the default data root */
caliper_core_live_attach_url(core, "ws://127.0.0.1:8765/v3/stream?token=...");
caliper_core_live_picker(core, 1, NULL);        /* list with Attach buttons; Detach returns to it */

/* Every frame, before caliper_core_frame. */
caliper_core_live_poll(core);                   /* claim newest snapshots, write the heartbeat */
caliper_core_frame(core);

caliper_core_live_detach(core);                 /* release claims, clear the heartbeat */
```

A host whose window can be hidden must keep its frame timer running. macOS slows the frames of a hidden window, and a slow viewer writes its heartbeat late. A stale heartbeat stops the script's snapshots while the user is still attached. For this reason, `caliper-viewer` holds an `NSProcessInfo` activity with `NSActivityUserInitiatedAllowingIdleSystemSleep | NSActivityLatencyCritical`, and the Caliper app holds the same activity while a session is attached.

### Live Session Calls

The listing calls work from any thread. The viewer calls run on the frame thread of a core with an attached canvas.

<!-- c-api calls include/caliper/live.h caliper_live_data_root caliper_live_list_sessions* -->

<!-- c-api calls include/caliper/live.h caliper_core_live_* -->

The embedding library also exports the client and segment calls that a producer uses.

<!-- c-api calls include/caliper/live.h caliper_live_last_error caliper_live_now_ns caliper_live_client_* caliper_live_segment_* -->

### Live Session Types

<!-- c-api types include/caliper/live.h CaliperLiveSessionInfo CaliperLiveStatus CaliperLiveViewStatus CaliperLiveClientDesc CaliperLiveViewDesc CaliperLiveSegmentInfo CaliperLivePublish level=4 -->

The header defines these feature macros and client flags:

<!-- c-api ids include/caliper/live.h CALIPER_EMBED_HAVE_* CALIPER_LIVE_CLIENT_* -->

## `caliper-embed` Package

`cmake --install <build> --prefix <prefix> --component embed` installs the `caliper-embed` CMake package on macOS and Windows. It is separate from the SDK (`--component sdk`) and goes into the prefix of the SDK or a prefix of its own.

| Installed path | Contents |
|---|---|
| `lib/libcaliper_embed.dylib` (macOS), `bin/caliper_embed.dll` and `lib/caliper_embed.lib` (Windows) | The embedding library. |
| `include/caliper-embed/caliper/` | `embed.h`, `live.h` and `live_views.h`, with copies of the SDK C headers they include and of the service tables that `caliper_core_get_service` returns: `tensor.h`, `arrow_c.h` and `services/`. |
| `lib/cmake/caliper-embed/` | `caliper-embed-config.cmake`, its version file and the exported target. |
| `share/caliper-embed/` | `LICENSE` and `NOTICE`. |

`find_package(caliper-embed CONFIG)` gives one imported target, `caliper::embed`, which carries the library and the include root `include/caliper-embed`. On macOS the library needs macOS 14 (Sonoma) or later. The package sets `CALIPER_MACOS_MIN` to its minimum, `14.0` by default. It makes that value `CMAKE_OSX_DEPLOYMENT_TARGET` unless the project, `-DCMAKE_OSX_DEPLOYMENT_TARGET` or `MACOSX_DEPLOYMENT_TARGET` chose one. The package version is the interface version ([Versions](#versions)), 1.4.0, with the CMake `SameMajorVersion` rule: a request for 1.2 accepts 1.4, and a request for 2 refuses it. In the core build, `caliper::embed` is an alias of the `caliper_embed` target, so one `target_link_libraries` line works in the tree and outside it.

### Shared Library

`libcaliper` is a static library over a dozen more, among them the Metal or Vulkan backend, DuckDB, ImGui, ImPlot and GLFW. If the package shipped those archives, every embedder would link all of them, and their symbols would sit beside the embedder copies of the same libraries. The package ships one shared library instead, `caliper_embed`, built from `libcaliper` and everything behind it. It exports only the C functions that `embed.h`, `live.h` and `live_views.h` declare. An embedder keeps its own ImGui, GLFW or DuckDB apart from the core copies. The frameworks that the core needs (Metal, QuartzCore, Cocoa, IOKit, CoreVideo) are load commands of the library, not link items of the embedder.

On Windows the export list leaves out the `caliper_live_*` client and listing calls, which exist on macOS only. The `caliper` executable, the test binaries and the Python extension still link the static `libcaliper`.

On macOS the install name is `@rpath/libcaliper_embed.dylib`. A program built in its own build folder finds the library through the build rpath that CMake adds. An installed program needs an rpath to `<prefix>/lib`. On Windows the program needs `caliper_embed.dll` beside it or on `PATH`, and `embed_host` copies the DLL beside itself after each build.

### Package Linking

```sh
cmake --install cmake-build-release --prefix <prefix> --component embed
cp -R examples/embed_host <work>/embed_host
cmake -S <work>/embed_host -B <work>/embed_host-build -DCMAKE_PREFIX_PATH=<prefix>
cmake --build <work>/embed_host-build
```

The `CMakeLists.txt` of `examples/embed_host` calls `cmake_minimum_required`, `project` and `find_package(caliper-embed 1.4 CONFIG REQUIRED)` only when it is the top-level project, as `examples/hello_world` does with the SDK. It links the same target in both builds. The frameworks serve the window code of the host:

```cmake
target_link_libraries(embed_host PRIVATE
    caliper::embed
    "-framework Cocoa" "-framework QuartzCore" "-framework Metal")
```

### `embed_host` Offscreen Mode

With `CALIPER_EMBED_OFFSCREEN=1`, `embed_host` runs without a window. `main()` takes this path before it creates an `NSApplication`, a window class or a window, so nothing appears on screen. It creates the core and attaches a `CALIPER_CANVAS_OFFSCREEN` canvas of 1280x800. It loads the applet named on the command line and pumps frames for `CALIPER_EMBED_EXIT_AFTER` seconds (1 by default). Then it reads the last frame back with `caliper_core_read_pixels`, prints what it did, and exits 0 when the applet drew, otherwise 1. Both modes check the version of the library first and read the applets folder from `CALIPER_EMBED_APPLETS`.

```sh
CALIPER_EMBED_OFFSCREEN=1 CALIPER_EMBED_APPLETS=<applets> \
    <work>/embed_host-build/embed_host dev.caliper.hello-world
```

The test `caliper_embed_install` (label `embed`) runs this whole path in a work folder, with `HOME` and the user applets folder inside it. When the canvas cannot attach, as on a machine without a GPU, the test reports Skipped, and `CALIPER_REQUIRE_GPU=1` makes that a failure. `CALIPER_EMBED_PROBE_DIR` sets the work folder.

## Torch-Free Build

`libcaliper` (core, Metal renderer, embed C ABI) includes no torch header. The renderers read `CaliperTensor.data` as a raw `id<MTLBuffer>` (Metal) or a CUDA device pointer (Vulkan). The options below make that boundary buildable, so one process can host Caliper Embed next to a **different** torch, such as the torch that a Python script imported beside the Python extension.

In a torch-free build the core carries no libtorch at all. The libtorch row of the [ownership table](#ownership) thus describes torch builds only, and the embedding process is free to load its own torch.

### Options

| Option | Default | Effect |
|---|---|---|
| `CALIPER_WITH_TORCH` | ON | OFF skips the PyTorch dependency block (no download, no `find_package(Torch)`) and every target that links torch: `examples/mnist/applet`, `caliper_torch_tests` and the MSVC torch-DLL copy. No binary carries a libtorch rpath. |
| `CALIPER_BUILD_APPLETS` | OFF | Also builds the example applets (`examples/hello_world` and, when torch is ON, `examples/mnist/applet`) inside the core build. Each also builds on its own against the installed SDK. |
| `CALIPER_BUILD_PYTHON` | OFF | Adds the nanobind extensions `caliper._native` and `caliper._live`. The wheel build sets it ON. |
| `CALIPER_WITH_SLANG` | OFF | Configure the vendored slang (SystemVerilog) library. |
| `CALIPER_BUILD_VIEWER` | ON | Builds `caliper_viewer`, the live-viewer process (Apple only). The wheel ships it as `caliper/bin/caliper-viewer`. |
| `CALIPER_WITH_BUILDER` | ON, but OFF on Windows | Builds the builder, which `libcaliper` does not link and no package installs. It also links the DuckDB json extension into the host DuckDB, so `libcaliper`, the embedding library and the `caliper` exe carry that extension. The wheel sets it OFF. |

slang is configured `EXCLUDE_FROM_ALL`, and nothing that `libcaliper`, the `caliper` exe or the tests need links it. At configure time slang clones boost_regex and mimalloc over the network, which a pip build should not do. `CALIPER_WITH_SLANG` thus defaults to OFF, and the configure presets and the Python build keep it OFF, so a fresh configure clones neither. Two FetchContent clones remain: tomlplusplus (`caliper_host_lib`) and doctest (tests only). With `CALIPER_WITH_BUILDER` on, the configure also downloads the nlohmann/json archive.

The wheel build sets `CALIPER_BUILD_PYTHON` ON. It sets `CALIPER_WITH_TORCH`, `CALIPER_BUILD_APPLETS`, `CALIPER_BUILD_EXAMPLES`, `BUILD_TESTS`, `USE_CUDA`, `CALIPER_WITH_SLANG` and `CALIPER_WITH_BUILDER` OFF. It builds only the `_native`, `_live`, `_netio` and `caliper_viewer` targets.

### Embedder Link Lists

The `caliper` exe links `CALIPER_DEPENDENCY_LIBS`, which holds the UI chrome widgets, DuckDB and the rest, but never libtorch. Applets and torch tests link `"${TORCH_LIBRARIES}"` themselves. `libcaliper` links only `CALIPER_CORE_LIBS` as `PRIVATE`: `glfw`, `imgui_backends`, which holds the GLFW backend of ImGui, and the Apple frameworks Cocoa, IOKit and CoreVideo.

The split matters because `libcaliper` is a static library, and its private dependencies still propagate to every embedder as link-only items. DuckDB reaches an embedder through `caliper_host_lib`, and imgui, implot, implot3d and ImGuiFileDialog reach it through `caliper::ui_stack`. Inside the core build, an embedder can link `libcaliper caliper::sdk` and nothing of the exe list, plus whatever its own window code needs. `examples/embed_host` links `caliper::embed` instead, the shared library of the [package](#caliper-embed-package), which carries all of that inside it. The Python extensions link `libcaliper` and `caliper::sdk` plus the Apple frameworks, and `glfw` for the window of `caliper._native`.

### `release-notorch` Preset

`CMakePresets.json` has configure, build and test presets named `release-notorch`: binary dir `cmake-build-notorch`, `CMAKE_BUILD_TYPE=Release`, `BUILD_TESTS=ON`, `CALIPER_WITH_TORCH=OFF`, `CALIPER_WITH_SLANG=OFF`, `CALIPER_WGSL_FROM_SOURCE=ON` and `USE_CUDA=OFF`. The Python extension builds against this configuration.

```sh
cmake --preset release-notorch
cmake --build --preset release-notorch
ctest --preset release-notorch
./cmake-build-notorch/tests/caliper_embed_tests
```

The default `cmake --preset release` configures with torch ON, using the vendored libtorch 2.5.1. `libcaliper` still builds with zero torch symbols there.

### Verification

Run these checks in the torch-free tree `cmake-build-notorch`. Each count must be 0:

```text
$ nm -gU cmake-build-notorch/libcaliper.a | grep -ci "torch\|c10::\|_ZN2at"
$ nm -u  cmake-build-notorch/libcaliper.a | grep -ci "torch\|c10\|_ZN2at"
$ otool -L <bin> | grep -ciE 'libtorch|libc10|libomp'
$ nm <bin> | c++filt | grep -cE '[ (<,](at|c10|torch)::|_ZN3c10|_ZN2at|_ZN5torch'
$ grep -rnE '#\s*include\s*[<"](torch|ATen|c10)' src include
```

A case-insensitive `torch` grep over the binaries gives false positives from DuckDB symbol names (`OperatorChildren`, `VectorChildIndex`), hence the namespace-anchored patterns. The `otool -L` output lists only system frameworks, libc++ and libobjc. In the torch build, `otool -L` on `libcaliper_embed.dylib` lists the same libraries.

## Header Listing

The tables on this page list every declaration of `caliper/embed.h` and `caliper/live_views.h` with the doc comment of each. The `caliper-embed` package installs both headers, with `live.h`, under `include/caliper-embed/caliper/` ([package contents](#caliper-embed-package)).
