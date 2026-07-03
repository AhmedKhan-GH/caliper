# Development basics

What building on Caliper actually looks like: what code you write, what code
is already running, where every library comes from, and the edit-build-run
loop. Read this before [Your first applet](first-applet.md); read the
[ML applet cookbook](../howto/ml-applet-cookbook.md) after it.

## The mental model

When your applet runs, **the host is already a living program**: it owns the
window, the renderer (Metal on macOS, GL fallback), the ImGui/ImPlot/ImPlot3D
contexts, the job system, the metrics/artifacts/data stores, and the docked
desktop. You write **one shared library** that the host loads at runtime.
Your code contributes exactly three things:

1. **UI** — `ImGui::`/`ImPlot::`/`ImPlot3D::` calls inside your per-frame
   function, drawn into windows the host docks and composites.
2. **Compute** — work you submit to `jobs.v1`, running on host-owned worker
   threads.
3. **State** — whatever your applet remembers between frames.

Everything else — event loop, GPU device, rendering, persistence, crash
containment — is the host's job. You never initialize a window, a GL/Metal
context, or an ImGui context; if you find yourself wanting to, you're
fighting the model.

## Where every library comes from

The short answer to "is it using the libraries we have installed?": **yes —
they're all in (or referenced by) this repo, and the build wires them for
you.** Nothing needs installing beyond what a clone + CMake configure finds.

| Library | Comes from | You link it via | Notes |
|---|---|---|---|
| SDK headers (ABI, services, sugar, adapters) | `sdk/include/` in this repo | `caliper::sdk` | Header-only INTERFACE target — nothing compiles into the SDK itself |
| ImGui (docking) + ImPlot + ImPlot3D + FileDialog | `third_party/` submodules, **pinned** | `caliper::ui_stack` | Compiled once in-tree. **Never bring your own copy** — the pin is part of the ABI (§9 of PLATFORM.md): your applet and the host must agree on ImGui's memory layout byte-for-byte |
| libtorch | `third_party/libtorch/` (vendored) | `"${TORCH_LIBRARIES}"` + the rpath block | Only applets that do ML link it; the **host never links torch** (D11). Copy the rpath stanza from the exemplar's CMakeLists |
| curl, zlib | macOS system SDK | `CURL::libcurl`, `ZLIB::ZLIB` | Only if you download/decompress data. Data acquisition is applet business — the host has no downloader |
| DuckDB | host-internal | **you can't** | Deliberately unreachable. You get its powers through services: `metrics.v1`, `artifacts.v1`, `data.v1`. No DuckDB type ever crosses the ABI |

The division is deliberate: what must be *shared* (UI stack) is pinned and
provided; what is *yours* (torch, curl) you link privately; what is
*persistent* (DuckDB) hides behind frozen service tables.

## Anatomy of an applet

Four files, one folder. This is the entire footprint:

```
applets/my_scope/
├── my_scope.caliper.toml    the manifest — checked BEFORE your code loads
├── plugin.cpp               ~20 lines: the CALIPER_APPLET macro + descriptor
├── my_scope.cpp             your applet class: initialize / draw_ui / cleanup
└── CMakeLists.txt           ~12 lines (hello) to ~40 (with torch)
```

- **Manifest**: id (reverse-DNS), name, version, summary, required/optional
  service lists. The `id` and `version` must be **byte-identical** between
  the manifest and the descriptor in plugin.cpp, or the loader refuses (with
  a polite card, not a crash). See [manifest reference](../reference/manifest.md).
- **plugin.cpp**: the `CALIPER_APPLET` macro generates the epoch-2 C ABI
  glue — exception walls on every entry point included. You never write
  `extern "C"` yourself.
- **The class**: three lifecycle methods. `initialize(Host&)` — probe your
  services (required ones you assert; optional ones degrade). `draw_ui()` —
  called every frame on the frame thread; submit ImGui windows, read
  worker-published state. `cleanup()` — cancel jobs, bounded-wait, release
  textures, return.
- **CMakeLists**: copy hello's verbatim for a UI-only applet; copy the
  exemplar's for an ML applet. The only parts you edit: target name, source
  files, manifest filename.

### The smallest complete applet, in full

This is everything — all three files. It is **built in this repo** as
`examples/my_scope/` (the MyScope card in your launcher), and the listings
below are the actual files, embedded verbatim — they cannot drift from what
compiles:

```cpp title="examples/my_scope/my_scope.cpp"
--8<-- "examples/my_scope/my_scope.cpp"
```

```toml title="examples/my_scope/my_scope.caliper.toml"
--8<-- "examples/my_scope/my_scope.caliper.toml"
```

```cmake title="examples/my_scope/CMakeLists.txt"
--8<-- "examples/my_scope/CMakeLists.txt"
```

Note what is **absent**: no `main`, no window, no GL/Metal, no ImGui
context creation, no event loop, no `extern "C"`. The macro plus the host
supply all of it. Growing from here toward ML means adding a `jobs.v1`
training job and the bridge — that path is the
[cookbook](../howto/ml-applet-cookbook.md), and its finished form is the
exemplar.

## The development loop

```bash
# 1. create — the applets/* glob auto-discovers it (CONFIGURE_DEPENDS);
#    no root CMake edits, just build:
mkdir applets/my_scope   # + the four files (copy examples/hello/ to start)

# 2. build — your dylib + manifest land in build/applets/
cmake --build build --target my_scope -j

# 3. run — it appears as a card in the launcher
./build/caliper

# 4. iterate fast — skip the launcher click every rebuild:
CALIPER_AUTOLAUNCH=dev.caliper.my-scope ./build/caliper
#    (CALIPER_EXIT_AFTER=<sec> exists too — clean-exit soak for CI)
```

In CLion the same works via the CMake tool window — reload the project once
after creating the folder, then the `my_scope` target exists.

**Testing without a window:** the fixture host
(`caliper::sdk_testing`, `<caliper/fixture_host.h>`) fakes `get_service`
with tables you inject — see `tests/test_sugar_services.cpp` for the
pattern. Applet logic that reads services can be unit-tested headlessly.

**Your data lives at** `~/Library/Application Support/Caliper/data/<your-id>/`
— the host creates it and hands it to you as `host.data_dir()`. Downloads,
caches, scratch files go there and nowhere else.

## The rules (each one guards something real)

- **No raw GL/Metal/Vulkan calls in applets** (§6c). Draw through
  ImGui/ImPlot/ImPlot3D; get tensors on screen through `tensor_bridge.v1`.
  This is what makes your applet run identically on both backends.
- **Bridge and data.v1 calls: frame thread only.** Torch ops: worker thread
  only. The [cookbook](../howto/ml-applet-cookbook.md) §1 has the full spine.
- **Honor cancel within 100 ms** in every job loop (check per batch,
  per download chunk).
- **Don't catch your way across the ABI** — the sugar's exception walls
  already guarantee nothing throws into the host; inside your own code,
  handle errors normally.
- **If a service is optional, degrade visibly** — a disabled panel saying
  "metrics: absent (ok)" beats silent absence. Every optional feature of the
  exemplar shows the idiom.

## Reading order

1. This page — the lay of the land.
2. [Your first applet](first-applet.md) — walk hello line by line.
3. [Your first ML applet](first-ml-applet.md) — the staged climb: compute,
   then torch, then live plots, then the bridge — one capability per stage.
4. `applets/embed_scope/` — the exemplar; every platform capability in one
   annotated file, structured to be copied.
5. [ML applet cookbook](../howto/ml-applet-cookbook.md) — the composition
   idioms (threading, cadences, the device-resident pull, viewport policy).
6. Per-service [reference pages](../reference/services/jobs-v1.md) — the
   contracts, when you need exact semantics.
