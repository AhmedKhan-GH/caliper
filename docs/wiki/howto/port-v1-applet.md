# Port a v1 applet

> **Historical recipe (PLATFORM.md §17).** The v1 ABI this page ports *from* was
> removed at the end of Phase 1 — `abi_v1.h`, the six `extern "C"` functions, and
> the v1 loader no longer exist in the tree. This how-to is preserved as the
> record of the v1→epoch-2 migration (it is exactly how the in-tree applets were
> ported); the "before" snippets below describe the **removed** v1 ABI.

The epoch-2 loader only sees applets that ship a `<stem>.caliper.toml` manifest
and export the epoch-2 descriptor. Porting was therefore a change to the **entry
boundary only** — the file that held the `extern "C"` bridge. The applet class
and all of its internal logic (parsing, rendering, DB access, …) did not change.

This page is the exact recipe used to port **CircuitNet** from v1 to epoch 2.
The whole diff was three files: `plugin.cpp` (rewritten), a new
`circuitnet.caliper.toml`, and one `add_custom_command` appended to
`CMakeLists.txt`. `circuitnet.cpp` / `circuitnet.h` and the four internal
translation units were untouched.

## What changed, at a glance

| v1 (removed) | epoch 2 |
| --- | --- |
| six `extern "C"` functions (`applet_info`, `applet_create`, `applet_destroy`, `applet_initialize`, `applet_draw_ui`, `applet_cleanup`) | one `caliper::Applet` subclass + the `CALIPER_APPLET(...)` macro |
| `#include <caliper/abi_v1.h>` (header removed) | `#include <caliper/caliper.hpp>` (sugar layer + pinned ImGui/ImPlot) |
| manual `ImGui::SetCurrentContext(host->imgui)` (× ImPlot, ImPlot3D) | nothing — the macro's `ui::connect()` does all three plus the shared allocator |
| metadata returned from `applet_info()` | metadata in the macro **and** in `<stem>.caliper.toml` (they must agree) |
| discovered by symbol probing | discovered by manifest; loader refuses on any mismatch |

## Step 1 — delete the six `extern "C"` functions

The port opened the entry file and removed the entire `extern "C" { … }` block.
That block was the whole v1 ABI: the info struct, `create`/`destroy`,
`initialize`, `draw_ui`, `cleanup`. The `#include <caliper/abi_v1.h>` went with
it (that header no longer exists). Everything they forwarded to still lives in
the applet class, so nothing of substance was lost — only the boilerplate bridge.

Before (the v1 CircuitNet entry file, in full — this is the code that was deleted):

```cpp
#include <caliper/abi_v1.h>
#include "circuitnet.h"

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

extern "C" {

APPLET_API CaliperAppletInfo applet_info() {
    return {
        "CircuitNet 3.0", "1.0",
        "Gate-level circuit architecture explorer …",
        "EDA", CALIPER_APPLET_ABI
    };
}

APPLET_API void* applet_create() { return new CircuitNetApplet(); }
APPLET_API void  applet_destroy(void* ctx) { delete static_cast<CircuitNetApplet*>(ctx); }

APPLET_API bool applet_initialize(void* ctx, const CaliperHostContext* host) {
    ImGui::SetCurrentContext(host->imgui);      // <-- manual context wiring,
    ImPlot::SetCurrentContext(host->implot);    //     now handled by the macro
    ImPlot3D::SetCurrentContext(host->implot3d);
    return static_cast<CircuitNetApplet*>(ctx)->initialize();
}

APPLET_API void applet_draw_ui(void* ctx, int w, int h) {
    static_cast<CircuitNetApplet*>(ctx)->draw_ui(w, h);
}

APPLET_API void applet_cleanup(void* ctx) {
    static_cast<CircuitNetApplet*>(ctx)->cleanup();
}

} // extern "C"
```

## Step 2 — wrap the applet class in a thin `caliper::Applet` adapter

Add a small subclass of `caliper::Applet` that *owns* your existing applet as a
member and forwards the three lifecycle hooks. This adapter is the only new code
the port introduces:

- `on_init(Host&)` calls your existing `initialize()`. CircuitNet doesn't need
  the host handle, so it discards it — `(void)host;`. (If you *do* want host
  services or the per-applet data dir, keep the `Host&` and use it.)
- `on_frame(const Frame&)` calls `draw_ui(w, h)`. Map the arguments from the
  frame: `f.fb_width` / `f.fb_height` are **physical pixels** — exactly what v1
  passed to `applet_draw_ui`, so the applet's coordinate assumptions are
  unchanged. (`Frame` also carries `dpi_scale`, `time_sec`, `delta_sec` if you
  want them later.)
- `on_cleanup()` calls your existing `cleanup()`.

## Step 3 — the `CALIPER_APPLET` macro

Below the class, invoke the macro. It generates the descriptor, the five
exception-safe C bridge functions, the single `caliper_applet_descriptor`
export, **and** the `ui::connect()` call that shares the host's ImGui / ImPlot /
ImPlot3D contexts and allocator. Field order is fixed:
`id, version, name, summary, tag, services`.

> The `id` and `version` you pass here **must match the manifest byte-for-byte**.
> The loader compares them and refuses to load the applet on any drift. For
> CircuitNet: `id = "dev.ahmed.circuitnet"`, `version = "1.0.0"`.

`.services` lists the service ids your applet requires; use the provided macros
`CALIPER_UI_V1` / `CALIPER_LOG_V1` (they expand to the `caliper.ui.v1` /
`caliper.log.v1` strings the manifest's `required` array names).

The whole ported entry file — steps 2 and 3 together — is just this:

```cpp
--8<-- "applets/legacy/circuitnet/plugin.cpp"
```

Note there is **no** `ImGui::SetCurrentContext` anywhere in it. If your v1 file
had manual `SetCurrentContext` / `SetAllocatorFunctions` calls, delete them —
`ui::connect()` (invoked by the macro before `on_init`) is now the single place
that wiring happens, and doing it twice is a bug.

## Step 4 — write `<stem>.caliper.toml`

Create the manifest next to the source, named for the library stem
(`circuitnet.caliper.toml` for a `circuitnet` target). This is what makes the
applet visible to the loader:

```toml
[applet]
id      = "dev.ahmed.circuitnet"
name    = "CircuitNet 3.0"
version = "1.0.0"
summary = "Gate-level circuit architecture explorer with DuckDB-powered querying, Verilog netlist parsing, and interactive graph visualization."
tag     = "EDA"

[compat]
abi_epoch = 2
min_host  = "0.6.0"

[services]
required = ["caliper.ui.v1", "caliper.log.v1"]
```

The `[applet].id` / `[applet].version` here are the values the loader checks
against the macro. `[compat].abi_epoch = 2` and `[services].required` complete
the contract.

## Step 5 — CMake: link the SDK, copy the manifest, C++20

Two things the target needs. First, link the SDK and the pinned UI stack (an
in-tree applet uses the monorepo targets; the names are identical out-of-tree):

```cmake
target_link_libraries(circuitnet PRIVATE
    caliper::sdk
    caliper::ui_stack
    # …your applet's own deps stay as they were…
)
```

Second, the manifest is only useful next to the built dylib, so copy it into
`build/applets/` on every build:

```cmake
add_custom_command(TARGET circuitnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/circuitnet.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/circuitnet.caliper.toml)
```

Finally, the sugar header uses designated initializers, so the target must be
**C++20**. CircuitNet already set this; if yours doesn't, add it:

```cmake
set_target_properties(circuitnet PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
```

## What stays untouched

Everything else. The applet class keeps its `initialize()` / `draw_ui(w, h)` /
`cleanup()` signatures; its rendering keeps calling raw ImGui / ImPlot /
ImPlot3D; internal subsystems (for CircuitNet: the Verilog parser, the DuckDB
query layer, the node-editor graph view) are not part of the entry boundary and
were not edited. Any node-editor / ImPlot3D context those internals set up
themselves keeps working — the macro only wires the host's *main* ImGui / ImPlot
/ ImPlot3D contexts, which is what the manual v1 code did too.

## Porting raw GL textures to the bridge

If your applet drew heatmaps or feature maps with **raw OpenGL**
(`glGenTextures` / `glTexImage2D` / `glBindTexture` / `glDeleteTextures`), those
calls are gone under §6c — a raw-GL applet cannot run in the Metal-backed host.
Move them onto [`caliper.tensor_bridge.v1`](../reference/services/tensor-bridge-v1.md).
This is the distilled recipe used to port **opengllama** (attention heatmaps) and
**repnet_demo** (weight/kernel + detail views) — both shipped, bridge-native.

- **The RGBA compose stays.** Whatever loop builds your `RGBA8` pixel buffer
  (colormap, LUT, per-token tint) is unchanged — the bridge takes *pixels*, not
  draw calls. Only the **upload path** swaps.
- **Describe the buffer as a `(H,W,4)` `u8` CPU tensor.** Fill a `CaliperTensor`
  by hand (C-ABI-direct, no torch needed): `dtype = CALIPER_DT_U8`, `ndim = 3`,
  `shape = {H, W, 4}`, row-major `strides = {W*4, 4, 1}`, `device =
  CALIPER_DEV_CPU`, `stream = nullptr`.
- **Create once, then update; recreate on resize.** `texture_from_tensor(&ct, 0)`
  the first time (keep the returned `CaliperTextureId`); `update_texture(id, &ct)`
  thereafter while the shape is stable. When the size changes (reflow, growth),
  `release_texture` the old id and create a new one. `release_texture` on
  teardown — **on the frame thread**, after the job wait. (repnet_demo's viz
  recomposes fresh each dirty and takes a simpler release-then-create path;
  opengllama's context heatmap update-in-place / recreate-on-reflow is the fuller
  pattern.)
- **Draw with `imtex`.** Feed the id to ImGui as
  `ImGui::Image(caliper::Bridge::imtex(id), size)` — the id is opaque, never a raw
  GL/Metal handle.
- **The manifest must require the bridge.** Add `"caliper.tensor_bridge.v1"` to
  `[services].required` (and `CALIPER_TENSOR_BRIDGE_V1` to the macro's
  `.services`). After the port the applet **cannot render without it**, so it is a
  hard requirement — negotiation leaves the card unavailable rather than letting
  it crash. opengllama's manifest moved the bridge from absent to required in
  exactly this step.

The applet code is then identical on Metal and GL; the bridge alone decides
whether the pixels are staged on the device or CPU-staged onto the GL fallback.

## Verify

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
nm -gU build/applets/libcircuitnet.dylib | grep caliper_applet_descriptor  # exported
ls build/applets/circuitnet.caliper.toml                                   # copied
./build/caliper                                                            # card appears, opens as before
```

The exported `caliper_applet_descriptor` and the copied manifest are the two
signals the loader needs; if the card is missing, check that both are present and
that the macro's `id`/`version` match the manifest exactly.
