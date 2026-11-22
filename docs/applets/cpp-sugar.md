# C++ Sugar

`caliper.hpp` is the header-only, optional C++ layer over the frozen C ABI. A C applet can implement `abi.h` by hand; the sugar exists so C++ authors write a class and a macro instead of five `extern "C"` bridge functions and a hand-built descriptor. It requires **C++20** (the `CALIPER_APPLET` macro uses designated initializers).

## `CALIPER_APPLET` Macro

You write a class deriving from `caliper::Applet` and invoke the macro once:

```cpp
#include <caliper/caliper.hpp>

class MyApplet final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override { host.log_info("hi"); return true; }
    void on_frame(const caliper::Frame& f) override { /* draw */ }
    void on_cleanup() override {}
};

CALIPER_APPLET(MyApplet,
    .id       = "dev.example.myapplet",
    .version  = "0.1.0",
    .name     = "My Applet",
    .summary  = "one-line description",
    .tag      = "Demo",
    .services = {CALIPER_LOG_V1, CALIPER_UI_V1})
```

### Required Field Order

The fields are C++20 designated initializers on `caliper::AppletMeta`, and **the order is fixed** — the aggregate is initialized positionally, so the names are documentation, not a reordering license:

1. `.id` — reverse-DNS identifier, must match `caliper.toml`
2. `.version` — applet semver, must match the manifest
3. `.name` — human-facing title
4. `.summary` — one-line description
5. `.tag` — category label
6. `.services` — brace-list of required service-id macros (e.g. `{CALIPER_LOG_V1}`)

`.services` is a `const char*[15]`; any trailing slots are zero-initialized, so the array the host sees is **NULL-terminated** automatically. Fifteen slots is the fixed capacity — list only the services your applet truly requires (negotiation refuses to load an applet whose required services the host cannot vend).

### Generated Code

- One `static const caliper::AppletMeta kMeta` holding the six fields.
- A `Holder` struct pairing your class instance with a `caliper::Host`.
- Five exception-safe `extern "C"` bridge functions — `create`, `destroy`, `initialize`, `frame`, `cleanup` — that own the `try/catch` so **no C++ exception ever crosses the C boundary** (a throw is caught, logged via `host.log_error("unhandled exception in on_<phase>")`, and swallowed; `create` returns `nullptr` on a failed `new`).
- `caliper::ui::connect(host)` is called inside `initialize`, before `on_init`, so ImGui/ImPlot contexts are live by the time your code runs.
- The single exported symbol `caliper_applet_descriptor()` returning a `static const CaliperAppletDescriptor` wired to `kMeta` and the five bridges.

### One Macro per Library

The macro defines the `caliper_applet_gen` namespace and the exported `caliper_applet_descriptor` symbol, so **exactly one `CALIPER_APPLET` may appear per shared library** (one applet per dylib — the ABI contract). In a test binary that needs to exercise several behaviours, keep a single `CALIPER_APPLET` and toggle behaviour from inside the class (see the fixture-host recipe below).

## `ui::connect` Semantics

`caliper::ui::connect(const CaliperHost*)` performs the ImGui context + allocator handoff described by [`caliper.ui.v1`](services/caliper-ui-v1.md), in the one order authors get wrong when they do it by hand:

1. Fetches `caliper.ui.v1` via `get_service`. **If the host does not vend it (headless), `connect` returns `false` and does nothing** — this is why the fixture host and other headless drivers work without a GL context.
2. Installs the host's allocator pair with `ImGui::SetAllocatorFunctions`, so every allocation in the applet's copy of ImGui lands on the **host** heap (the crux of sound context-sharing across a DLL boundary).
3. Calls `ImGui::SetCurrentContext`, `ImPlot::SetCurrentContext`, and `ImPlot3D::SetCurrentContext` with the host-owned contexts.

Because the macro calls `connect` for you inside `initialize`, most applets never call it directly.

## Service Wrappers

`caliper::Host` wraps the raw `CaliperHost`: `service(id)` calls `get_service`, `data_dir()` returns the applet's data folder, and `log_info` and `log_error` write through `caliper.log.v1`. On top of it the header defines one typed wrapper per service. Each wrapper is built from a `Host`, is falsy when the host does not vend the service, and does nothing (returns 0, `false`, an empty string or `nullptr`) when called on a host without it. There is no wrapper for `caliper.feed.v1`; call that table through `Host::service`.

| Wrapper | Service | What it gives you |
|---|---|---|
| `Jobs` | [`caliper.jobs.v1`](services/caliper-jobs-v1.md) | `submit`, `request_cancel`, `is_running` and `progress_of` for background jobs. |
| `Metrics` | [`caliper.metrics.v1`](services/caliper-metrics-v1.md) | `begin_run`, `end_run`, `scalar`, `histogram`, `image` and `hparams_json`. It does not wrap the `v1_1` query call. |
| `Artifacts` | [`caliper.artifacts.v1`](services/caliper-artifacts-v1.md) | `put` returns the digest as a `std::string` (empty on failure), plus `path_of` and `exists`. |
| `Data` | [`caliper.data.v1`](services/caliper-data-v1.md) | `query`, `register_dataset`, `open_dataset` and `last_error`, plus the static helper `drain_numeric`, which reads an all-numeric Arrow stream into column-major doubles. |
| `Bridge` | [`caliper.tensor_bridge.v1`](services/caliper-tensor-bridge-v1.md) | Texture creation and update, `alloc_shared` and `free_shared`, `caps()` from `v1_1`, and the imported-allocation calls from `v1_2`. `imtex` casts a texture id to an `ImTextureID`. Frame thread only. |
| `Geometry` | [`caliper.geometry.v1`](services/caliper-geometry-v1.md) | View creation, `draw_points` and the `draw_primitives` overloads for each record revision. When the host vends only a later revision, the wrapper widens older records to the layout that revision requires. Frame thread only. |
| `Export` | [`caliper.export.v1`](services/caliper-export-v1.md) | `view_png` and the frame-sequence calls `begin_sequence`, `frame` and `end_sequence`, with the same overloads for each geometry record revision. |
| `Device` | [`caliper.device.v1`](services/caliper-device-v1.md) | A plain struct filled by `Device::query(host)` with `kind`, `index`, `name` and `free_memory_hint`. It reports the CPU when the service is absent. |

## Fixture-Host TDD Recipe

`caliper::testing::FixtureHost` (from `<caliper/fixture_host.h>`, target `caliper::sdk_testing`) is a headless fake `CaliperHost` for test-driving applets and sugar without launching UI. It vends `caliper.log.v1` only; `get_service` returns `NULL` for everything else (so `ui::connect` no-ops). It records every logged line, exposed via `log_lines()` and `log_contains()`.

Because the C ABI carries no user-data pointer, the log/service thunks route through a single static active pointer: **exactly one `FixtureHost` may be live per process at a time** (construct one per `TEST_CASE`; its destructor clears the active slot).

Drive an applet through the generated C table exactly as the host would:

```cpp
#include <doctest/doctest.h>
#include <caliper/caliper.hpp>
#include <caliper/fixture_host.h>

TEST_CASE("applet: initialize logs and frame is exception-safe") {
    caliper::testing::FixtureHost fx;
    const CaliperAppletDescriptor* d = caliper_applet_descriptor();

    void* self = d->api.create();
    REQUIRE(self != nullptr);
    REQUIRE(d->api.initialize(self, fx.host()));   // runs on_init
    CHECK(fx.log_contains("hi"));

    CaliperFrameInfo fi{};
    fi.struct_size = sizeof fi;
    fi.fb_width = 640; fi.fb_height = 480; fi.dpi_scale = 2.0f;
    d->api.frame(self, &fi);                        // runs on_frame

    d->api.cleanup(self);
    d->api.destroy(self);
}
```

## Full Source

```cpp
--8<-- "sdk/include/caliper/caliper.hpp"
```
