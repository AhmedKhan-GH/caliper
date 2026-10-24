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
| Crash containment | **libcaliper** | applet faults are caught by the core's guard and surfaced via `crash_fn`; the embedder is not taken down. The guard's signal handlers are installed once per process and pass faults outside applet calls on to the handlers installed before them, so an embedder's own crash reporter still sees its own crashes ([trust model](../explanation/trust-model.md#every-applet-call-is-guarded)) |

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

## Built-in tensor views (embed v1.2)

The v1.2 section of `caliper/embed.h` lets a host hand the core a
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

Windowed views are drawn in creation order and tiled left to right. A small
tensor is magnified by a whole-number factor and sampled nearest, so cells
stay cells; each window carries an info line
(`WxH dtype device path updates=N last=ms`). `CALIPER_VIEW_FLAG_RAW` draws the
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
the read after that queue's committed work (bridge v1.1, D24). NULL is the
normal case for a caller that made sure the producer finished first.

Measured on Apple Silicon (SPIKE.md §3 and §6):

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
  `torch.mps.synchronize()`; the [live-streaming](../explanation/live-streaming.md)
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

### Typed live views (embed v1.3)

`include/caliper/live_views.h` builds on this section. `caliper_core_live_apply`
takes a `CaliperLiveSnapshot` of one of five kinds (heatmap, image, points,
series, histogram) and keeps a typed view of that name. Heatmap and image live
views are drawn through the v1.2 view of the same name, so
`caliper_core_view_info`, `_readback`, `_count` and `_remove` keep working for
them. The header documents the per-kind payload contract, the claim rule that
`caliper_core_live_view_read_done` reports for each kind, and the per-core
pause flag. `include/caliper/live.h` is the shared-memory session protocol and
the client and viewer C API that feed it from another process. Both are
macOS/Metal only today; the design is in
[Live streaming](../explanation/live-streaming.md).

## The torch-free build

libcaliper (core, Metal renderer, embed C ABI) never included a torch header.
The renderers consume `CaliperTensor.data` as a raw `id<MTLBuffer>` (Metal) or
a CUDA device pointer (Vulkan), and `sdk/include/caliper/adapters/torch.hpp` is
applet-side only. The vendored libtorch 2.5.1 reached libcaliper, the `caliper`
exe and `embed_host` purely through `${CALIPER_DEPENDENCY_LIBS}`, to which
`cmake/Dependencies.cmake` appended `${TORCH_LIBRARIES}`. The options below make
that boundary buildable, so one process can host libcaliper next to a
**different** torch: Python's own torch 2.14.1 in the Python extension.

In a torch-free build the core carries no libtorch at all, so the libtorch row
of the [ownership table](#ownership-design-4) describes torch builds only; the
embedding process is free to load its own torch.

### Options

| Option | Default | Effect |
|---|---|---|
| `CALIPER_WITH_TORCH` | ON | OFF skips the whole "Category 4: PyTorch" block of `cmake/Dependencies.cmake` (no download, no `find_package(Torch)`, `TORCH_LIBRARIES` empty) and every torch-linking target: the `applets/*` tree (forces `CALIPER_BUILD_APPLETS` OFF), `examples/sine_scope` (the ML tutorial applet), `caliper_torch_tests`, the applet learning-core tests and the MSVC torch-DLL copy. `caliper_embed_tests` is then compiled with `CALIPER_TESTS_NO_TORCH_APPLETS=1`, so its sine-scope swap cases compile out, and no binary carries a libtorch rpath. |
| `CALIPER_BUILD_APPLETS` | ON | The `applets/*` glob. Forced OFF when torch is OFF, because every applet links libtorch today. |
| `CALIPER_BUILD_PYTHON` | OFF | `add_subdirectory(python)`: the nanobind extensions `caliper._native` and `caliper._live`. The wheel build (scikit-build-core) sets it ON. |
| `CALIPER_WITH_SLANG` | ON | Configure the vendored slang (SystemVerilog) library. |
| `CALIPER_WITH_LLAMA` | ON | Configure the vendored llama.cpp and apply its local patches. `CALIPER_BUILD_OPENGLLAMA=ON` stops the configure with an error when this is OFF. |
| `CALIPER_BUILD_VIEWER` | ON | `caliper_viewer`, the live-viewer process (Apple only). The wheel ships it as `caliper/bin/caliper-viewer`. |

slang and llama.cpp are configured `EXCLUDE_FROM_ALL` and linked by nothing
that libcaliper, the `caliper` exe or the tests need; llama.cpp only backs the
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
| `CALIPER_CORE_LIBS` | Only what libcaliper's own sources and the GL/Metal backends need: `OpenGL::GL`, `-framework Cocoa`, `-framework IOKit`, `-framework CoreVideo` (Apple), `libglew_static`, `glfw`. | libcaliper, `PRIVATE` |

The split matters because libcaliper is a static library: its private
dependencies still propagate to every embedder as link-only items, so the full
list used to drag the chrome widgets into `embed_host` and the Python
extension. DuckDB now reaches an embedder through `caliper_host_lib`, and
imgui, implot, implot3d and ImGuiFileDialog through `caliper::ui_stack`.

An embedder links `libcaliper caliper::sdk` and nothing of the exe's list,
plus whatever its own window code needs. `examples/embed_host` on macOS:

```cmake
target_link_libraries(embed_host PRIVATE
    libcaliper caliper::sdk
    "-framework Cocoa" "-framework QuartzCore" "-framework Metal")
```

`caliper_embed_tests` links `doctest::doctest libcaliper caliper::sdk` only, so
it proves libcaliper's link interface is self-sufficient. The Python extensions
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

A cold torch-free build of everything (libcaliper, exe, `embed_host`, all test
binaries) is 462 steps, 120 s at `-j 8`; `libcaliper.a` itself is 276 KB (its
own six objects; dependencies are separate archives). The default
`cmake --preset release` still configures with torch ON (vendored libtorch 2.5.1
found, nine applets discovered), and a cold `libcaliper` there builds in 123 s
with zero torch symbols.

### Verification

Run in the torch-free tree `cmake-build-notorch` (SPIKE.md §2):

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

## The header, verbatim

This page embeds the real header; the docs build fails if the file moves.

```c
--8<-- "include/caliper/embed.h"
```
