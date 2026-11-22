# C++ Sugar

`caliper.hpp` is the optional, header-only C++ layer over the frozen C ABI. A C applet can implement `abi.h` by hand. The C++ layer lets an author write a class and a macro instead of five `extern "C"` bridge functions and a hand-built descriptor. It requires **C++20**, because the `CALIPER_APPLET` macro uses designated initializers.

## `CALIPER_APPLET` Macro

You write a class that derives from `caliper::Applet` and invoke the macro once:

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

The fields are C++20 designated initializers on `caliper::AppletMeta`. **The order is fixed.** The compiler initializes the aggregate by position, so the names document the fields and do not allow reordering. The order is:

1. `.id`: the reverse-DNS identifier, which must match `caliper.toml`.
2. `.version`: the applet semver, which must match the manifest.
3. `.name`: the human-facing title.
4. `.summary`: a one-line description.
5. `.tag`: a category label.
6. `.services`: a brace-list of required service-id macros, for example `{CALIPER_LOG_V1}`.

`.services` is a `const char*[15]`. The compiler zero-initializes unused slots, so the host always sees a **NULL-terminated** array. Fifteen slots is the fixed capacity. List only the services that your applet requires, because negotiation refuses to load an applet when the host cannot vend a required service.

### Generated Code

The macro generates these items:

- One `static const caliper::AppletMeta kMeta` that holds the six fields.
- A `Holder` struct that pairs your class instance with a `caliper::Host`.
- Five exception-safe `extern "C"` bridge functions: `create`, `destroy`, `initialize`, `frame` and `cleanup`.
- The single exported symbol `caliper_applet_descriptor()`, which returns a `static const CaliperAppletDescriptor` wired to `kMeta` and the five bridges.

The bridges own the `try/catch`, so **no C++ exception crosses the C boundary**. A bridge catches a throw, logs it with `host.log_error("unhandled exception in on_<phase>")` and swallows it. `create` returns `nullptr` when `new` fails. The `initialize` bridge calls `caliper::ui::connect(host)` before `on_init`, so the ImGui and ImPlot contexts are live when your code runs.

### One Macro per Library

The macro defines the `caliper_applet_gen` namespace and the exported `caliper_applet_descriptor` symbol. Thus **one shared library can contain exactly one `CALIPER_APPLET`**, because the ABI allows one applet per dylib. If a test binary must exercise several behaviors, keep one `CALIPER_APPLET` and switch the behavior from inside the class.

## `ui::connect` Semantics

`caliper::ui::connect(const CaliperHost*)` hands the ImGui context and allocator to the applet, as [`caliper.ui.v1`](services/caliper-ui-v1.md) describes. It uses the one order that authors get wrong when they do the handoff by hand:

1. It fetches `caliper.ui.v1` with `get_service`. If the host does not vend the service (a headless host), `connect` returns `false` and does nothing. Thus headless drivers work without a GL context.
2. It installs the host allocator pair with `ImGui::SetAllocatorFunctions`. Every allocation in the applet copy of ImGui then lands on the **host** heap, which makes context sharing across a library boundary sound.
3. It calls `ImGui::SetCurrentContext`, `ImPlot::SetCurrentContext` and `ImPlot3D::SetCurrentContext` with the host-owned contexts.

The macro calls `connect` inside `initialize`, so most applets never call it directly.

## Service Wrappers

`caliper::Host` wraps the raw `CaliperHost`. `service(id)` calls `get_service`, `data_dir()` returns the data folder of the applet, and `log_info` and `log_error` write through `caliper.log.v1`. The header also defines one typed wrapper for each service. Each wrapper is built from a `Host` and is falsy when the host does not vend the service. When you call a wrapper on a host without the service, it does nothing and returns 0, `false`, an empty string or `nullptr`. No wrapper exists for `caliper.feed.v1`, so call that table through `Host::service`.

| Wrapper | Service | What it gives you |
|---|---|---|
| `Jobs` | [`caliper.jobs.v1`](services/caliper-jobs-v1.md) | `submit`, `request_cancel`, `is_running` and `progress_of` for background jobs. |
| `Metrics` | [`caliper.metrics.v1`](services/caliper-metrics-v1.md) | `begin_run`, `end_run`, `scalar`, `histogram`, `image` and `hparams_json`. It does not wrap the `v1_1` query call. |
| `Artifacts` | [`caliper.artifacts.v1`](services/caliper-artifacts-v1.md) | `put` returns the digest as a `std::string` (empty on failure), plus `path_of` and `exists`. |
| `Data` | [`caliper.data.v1`](services/caliper-data-v1.md) | `query`, `register_dataset`, `open_dataset` and `last_error`, plus the static helper `drain_numeric`, which reads an all-numeric Arrow stream into column-major doubles. |
| `Bridge` | [`caliper.tensor_bridge.v1`](services/caliper-tensor-bridge-v1.md) | Texture creation and update, `alloc_shared` and `free_shared`, `caps()` from `v1_1`, and the imported-allocation calls from `v1_2`. `imtex` casts a texture id to an `ImTextureID`. Frame thread only. |
| `Geometry` | [`caliper.geometry.v1`](services/caliper-geometry-v1.md) | View creation, `draw_points` and the `draw_primitives` overloads for each record revision. When the host vends only a later revision, the wrapper widens older records to the layout that revision requires. Frame thread only. |
| `Export` | [`caliper.export.v1`](services/caliper-export-v1.md) | `view_png` and the frame-sequence calls `begin_sequence`, `frame` and `end_sequence`, with the same overloads for each geometry record revision. |
| `Device` | [`caliper.device.v1`](services/caliper-device-v1.md) | A plain struct that `Device::query(host)` fills with `kind`, `index`, `name` and `free_memory_hint`. It reports the CPU when the service is absent. |

## Fixture-Host TDD Recipe

`caliper::testing::FixtureHost` (header `<caliper/fixture_host.h>`, target `caliper::sdk_testing`) is a headless fake `CaliperHost` for tests. It vends only `caliper.log.v1`, and `get_service` returns `NULL` for all other services, so `ui::connect` does nothing. It records each logged line, and `log_lines()` and `log_contains()` expose them.

The C ABI has no user-data pointer, so the thunks use one static active pointer. Only one `FixtureHost` may be live in a process. Build one for each `TEST_CASE`, because the destructor clears the slot.

To drive an applet, call `caliper_applet_descriptor()`, then `api.create()`, `api.initialize(self, fx.host())`, `api.frame(self, &fi)` with a `CaliperFrameInfo` (set `struct_size`, `fb_width`, `fb_height`, `dpi_scale`), `api.cleanup(self)` and `api.destroy(self)`. Check the log with `fx.log_contains("hi")`.

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
</Content>
