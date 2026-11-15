# Compatibility and epochs

This page explains how the ABI and the services change over time without breaking applets that are already built. The ABI and the service negotiation described here are implemented. The loader does not download dependencies.

The host is a **service registry, not a struct of fields**. Applets obtain capabilities by name through a single extension point:

```c
struct CaliperHost {
    uint32_t    struct_size;
    uint32_t    abi_epoch;          // epoch this host is speaking
    uint32_t    host_version;       // (major<<16)|(minor<<8)|patch, informational
    const char* applet_data_dir;    // per-applet data folder, UTF-8
    /* THE extension point. Returns a service table or NULL. The pointer is
       valid for the applet's lifetime. Unknown ids return NULL — never UB. */
    const void* (*get_service)(const CaliperHost* host, const char* service_id);
};
```

## Growth rules

These four rules let the contract grow without breaking an applet that is already built:

1. **A published service table is immutable.** A new capability ships as a new service id that carries a revision suffix: `caliper.metrics.v1_1`, `caliper.tensor_bridge.v1_2`, `caliper.geometry.v1_3`. The new table starts with a byte-identical copy of the previous table and appends its new entries, and the host keeps serving every older id unchanged. An applet that asks for `caliper.geometry.v1` still gets the original table. Where a revision adds behaviour that a host may not provide, a `caps()` bit says so, one bit per revision. A change that cannot be made by appending is an incompatible change and gets a new major id, such as `caliper.metrics.v2`, served beside the old one. No service has needed one so far. See decision [D30](../decisions/d030-additive-service-revisions.md).
2. **Hosts may provide any set of services.** Applets declare `required_services` and probe optional ones at run time. A missing required service is refused at the manifest check, and the landing screen lists the applet as `REFUSED` with the reason.
3. **`struct_size` is always the first field.** A reader never touches bytes beyond the size the writer declared.
4. **The ABI epoch changes only for** an entry-point change, a `CaliperHost` layout change, or a UI-stack pin change.

## ABI hygiene

Every type that crosses the boundary follows the same rules:

- C types only. No STL types, and no exceptions across the boundary. The C++ sugar catches every exception at the edge and reports it through `caliper.log.v1`.
- No third-party C++ type in any signature. Tensors and tables use the interchange types in the next table.
- No graphics-API type or handle in the ABI. Applets never issue raw graphics calls. They draw through ImGui and ImPlot and through textures from `caliper.tensor_bridge`, which is what keeps the renderer swappable.
- Memory is freed by the side that allocated it. A string or buffer that the host returns is host-owned, and its header documents how long it stays valid.
- Every string is UTF-8 `const char*`.
- Every struct is a plain struct with `struct_size` first and a stable layout.

### Types that never cross the boundary

| Type | Why not | Interchange used |
|---|---|---|
| `torch::Tensor` | The libtorch C++ ABI differs between versions and builds, so using it would tie every applet to the host's torch. | `CaliperTensor`: a raw pointer with dtype, shape, strides and device. See [Tensor](../references/tensor.md). |
| `duckdb::*` | The same reason. | SQL strings in, Arrow C streams out. See [Arrow](../references/arrow.md). |
| `std::string`, `std::vector` | The libstdc++, libc++ and MSVC layouts differ. | `const char*`, or a pointer and a length. |
| Exceptions | They are undefined across runtimes. | A `bool` or error-code return, plus `caliper.log.v1`. |
| ImGui, Metal, Vulkan, OpenGL handles | They would fix the renderer. | `CaliperTextureId`, an opaque 64-bit handle. |

The UI contexts do cross, because applets write raw ImGui. `caliper.ui.v1` hands over the ImGui, ImPlot and ImPlot3D contexts together with the host's allocator pair, and `caliper::ui::connect(host)` in the sugar installs both. On Windows an applet DLL would otherwise free ImGui memory on its own heap.

## Toolchain contract

A binary applet is compatible with a host only when both are built with the same toolchain family for the platform:

| Platform | Compiler and runtime | Notes |
|---|---|---|
| macOS arm64 | AppleClang with libc++ | CI builds on `macos-15`. |
| Windows x64 | MSVC with the dynamic CRT (`/MD`) | Host and applets share one CRT heap. CI builds on `windows-latest`. |

On Windows the dynamic CRT is the default, so that the CRT state of the host and of every applet agrees: heap, locale, `FILE*` and `errno`. The CMake option `CALIPER_STATIC_CRT=ON` selects the static runtime (`/MT`) as an escape hatch for a build that needs it; see decision [D7](../decisions/d007-dynamic-crt-on-windows.md). Linux is not a supported platform.

## Versioning

Every moving part has its own scheme, so a change in one rarely forces a change in another:

| Thing | Scheme | Changes when |
|---|---|---|
| ABI epoch | An integer, 2 today | The entry points, the `CaliperHost` layout or the UI-stack pin change. A host serves exactly one epoch. |
| SDK | Semver | Minor versions add services, sugar or adapters within an epoch. |
| Services | Id with a revision suffix: `.v1`, `.v1_1`, `.v1_2`, `.v1_3` | A revision appends entries and never changes an older table. A major id such as `.v2` is for an incompatible change. |
| Host application | Semver | The interface or features change. An applet states the oldest host it supports in `min_host`. |
| Applets | Semver, chosen by the applet author | The applet author decides. |

### The epoch check

The epoch check is exact equality. The host compares the manifest's `abi_epoch` with the one epoch it speaks, and any difference is a refusal. The host does not load an applet built for an older epoch or a newer one.

## Negotiation at load

The checks run in order, and the first five happen **before `dlopen`**. The first failure becomes the applet's status on the landing screen, with the reason (see [Refusal messages](../references/refusals.md)), instead of a loader crash:

1. manifest parses, with a valid, path-safe `id`
2. platform binary present
3. the manifest's `abi_epoch` equals the host's epoch
4. `min_host` satisfied
5. required services available (the manifest's list)
6. *then* `dlopen`
7. descriptor sanity — id / version / epoch agree with the manifest
8. required services available (the binary's own list)
9. `create` and `initialize`

!!! note "Why epochs, not a rolling ABI"
    The UI-stack pin, which is the exact ImGui, ImPlot and ImPlot3D commits, is part of the epoch, because applets write raw ImGui against those headers. Changing the pin is therefore an epoch change. The cost is an applet rebuild, not a silent break. See the [Architecture](architecture.md) page.
