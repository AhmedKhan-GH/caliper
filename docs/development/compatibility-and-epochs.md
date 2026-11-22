# Compatibility and Epochs

This page explains how the ABI, the services, the SDK, the applets and their stored data are versioned, and how each changes without breaking applets that are already built. Everything this page describes is implemented. The loader does not download dependencies.

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

New capabilities arrive as new service ids, never as changes to the base ABI, so the base ABI stays small and stable while the host grows. The price is that every applet must cope with a service that is missing, which the fallback rule in [Architecture](../introduction/architecture.md#design-decisions) asks of it anyway.

## ABI Hygiene

Every type that crosses the boundary follows the same rules:

- C types only. No STL types, and no exceptions across the boundary. The C++ sugar catches every exception at the edge and reports it through `caliper.log.v1`.
- No third-party C++ type in any signature. Tensors and tables use the interchange types in the next table.
- No graphics-API type or handle in the ABI. Applets never issue raw graphics calls. They draw through ImGui and ImPlot and through textures from `caliper.tensor_bridge`, which is what keeps the renderer swappable.
- Memory is freed by the side that allocated it. A string or buffer that the host returns is host-owned, and its header documents how long it stays valid.
- Every string is UTF-8 `const char*`.
- Every struct is a plain struct with `struct_size` first and a stable layout.

### Excluded Boundary Types

| Type | Why not | Interchange used |
|---|---|---|
| `torch::Tensor` | The libtorch C++ ABI differs between versions and builds, so using it would tie every applet to the host's torch. | `CaliperTensor`: a raw pointer with dtype, shape, strides and device. See [Tensor](../applets/tensor-and-arrow-headers.md#tensor-header). |
| `duckdb::*` | The same reason. | SQL strings in, Arrow C streams out. See [Arrow](../applets/tensor-and-arrow-headers.md#arrow-header). |
| `std::string`, `std::vector` | The libstdc++, libc++ and MSVC layouts differ. | `const char*`, or a pointer and a length. |
| Exceptions | They are undefined across runtimes. | A `bool` or error-code return, plus `caliper.log.v1`. |
| ImGui, Metal, Vulkan, OpenGL handles | They would fix the renderer. | `CaliperTextureId`, an opaque 64-bit handle. |

The UI contexts do cross, because applets write raw ImGui. `caliper.ui.v1` hands over the ImGui, ImPlot and ImPlot3D contexts together with the host's allocator pair, and `caliper::ui::connect(host)` in the sugar installs both. On Windows an applet DLL would otherwise free ImGui memory on its own heap.

## Toolchain Contract

A binary applet is compatible with a host only when both are built with the same toolchain family for the platform:

| Platform | Compiler and runtime | Notes |
|---|---|---|
| macOS arm64 | AppleClang with libc++ | macOS 14 (Sonoma) or later: every binary is built for macOS 14.0. CI builds on `macos-15`. |
| Windows x64 | MSVC with the dynamic CRT (`/MD`) | Host and applets share one CRT heap. CI builds on `windows-latest`. |

On Windows the dynamic CRT is the default, so that the CRT state of the host and of every applet agrees: heap, locale, `FILE*` and `errno`. The CMake option `CALIPER_STATIC_CRT=ON` selects the static runtime (`/MT`) as an escape hatch for a build that needs it. Linux is not a supported platform.

## Versioning

Four layers are versioned separately, so that a change in one rarely forces a change in another.

| Layer | What it versions | Scheme | Declared in |
|---|---|---|---|
| [The host ABI epoch and service revisions](#host-abi-epoch-and-service-revisions) | The entry points, the `CaliperHost` layout and the UI-stack pin; each service table | An integer epoch, 2 today; a service id with a revision suffix | `[compat].abi_epoch` and `[services]` |
| [The SDK version](#sdk-version) | The installed `caliper-sdk` package | Semver | `[build].sdk`, written by the build |
| [Each applet's version](#applet-versions-and-requirements) | The applet and what it requires | Semver, chosen by the applet's author | `[applet].version`, `[compat]` and `[services]` |
| [The version of stored data](#stored-data-versions) | The applet's data folder | An integer of at least 1, chosen by the applet's author | `[data].schema` |

The host application has a semver version of its own, which an applet compares against in `[compat].min_host`. The keys are listed in [Manifest](../applets/manifest.md#schema).

### Host ABI Epoch and Service Revisions

The ABI epoch is an integer, 2 today, that versions the frozen contract: the entry points in [`abi.h`](../applets/applet-contract.md), the `CaliperHost` layout and the UI-stack pin. A host speaks exactly one epoch, and the epoch check is exact equality. Before it opens an applet's library, the host compares `[compat].abi_epoch` with its own epoch and refuses an applet built for an older or a newer epoch. A host never also accepts the previous epoch, so an epoch change means rebuilding every applet. After opening the library, it checks that the descriptor carries the same epoch.

Within an epoch, each service is versioned by its id. `caliper.geometry.v1` is the first version of a table, `caliper.geometry.v1_1` to `caliper.geometry.v1_3` are revisions that append to it, and a major id such as `caliper.metrics.v2` is reserved for an incompatible change. No service has needed a major id so far.

!!! note "Why epochs, not a rolling ABI"
    Applets write raw ImGui against the host's headers and share its ImGui context, so the UI-stack pin, which is the exact ImGui, ImPlot, ImPlot3D and ImGuiFileDialog versions and the `imconfig`, is part of the epoch. Changing the pin is therefore an epoch change. The cost is an applet rebuild, not a silent break. The UI-stack hash in [The SDK version](#sdk-version) checks the pin itself when an applet loads. See the [Architecture](../introduction/architecture.md) page.

#### Growth rules

These four rules let the contract grow without breaking an applet that is already built:

1. **A published service table is immutable.** A new capability ships as a new service id with a revision suffix, such as `caliper.metrics.v1_1`, `caliper.tensor_bridge.v1_2` or `caliper.geometry.v1_3`. The new table starts with a byte-identical copy of the previous table and appends its new entries, and the host keeps serving every older id unchanged, so an applet that asks for `caliper.geometry.v1` still gets the original table. Where a revision adds behaviour that a host may not provide, a `caps()` bit says so, one bit per revision. A change that cannot be made by appending gets a new major id, served beside the old one.
2. **Hosts may provide any set of services.** An applet lists the services it cannot run without in `[services].required` and probes optional ones at run time. A host that lacks a required service refuses the applet, and the landing screen lists it as `REFUSED` with the reason.
3. **`struct_size` is always the first field.** A reader never touches bytes beyond the size the writer declared.
4. **The ABI epoch changes only for** an entry-point change, a `CaliperHost` layout change or a UI-stack pin change.

Under these rules an applet built years ago keeps running on a new host, and a new applet checks for a newer id and falls back when the host lacks it, with no epoch bump. The cost is that hosts and applets of different ages must be tested against each other, and the ABI tests pin every frozen layout.

### SDK Version

The SDK is the installed `caliper-sdk` package, and it is versioned by semver; it is 0.2.0 today, at epoch 2. From version 1.0 on, a patch version fixes the SDK without changing what it offers, a minor version adds services, sugar or build helpers within the same epoch, and the major version changes when the epoch changes, and only then. Each major version then builds applets for one epoch, and an applet that builds with one minor version builds with every later one. Before 1.0, a minor version may also change the epoch. The package's version file uses CMake's `SameMajorVersion` rule: `find_package(caliper-sdk <version>)` accepts an installed SDK with the same major version, at or above the one requested, and without a version it accepts any installed SDK. While the major version is 0, a request for 0.2 therefore also accepts a later 0.x SDK built for another epoch, and `caliper_add_applet()` then stops the configure, because the manifest's `abi_epoch` must equal the SDK's epoch.

The build records how an applet was built. `caliper_add_applet()` copies the manifest that the author writes to the folder beside the built library and adds a `[build]` table to the copy ([SDK package](../applets/sdk-package.md#stamped-manifest)). Nobody writes the `[build]` table by hand. It holds two keys:

- `[build].sdk` is the SDK version that built the applet. It is a record for diagnosis, for example when reading a bug report, and the host never compares it. Compatibility is decided by the epoch, the UI-stack hash and service negotiation, not by comparing SDK versions. An applet built with a newer minor version therefore runs on any host of its epoch that serves the services it requires, and a host that lacks one of them refuses the applet by naming that service.
- `[build].ui_stack` is a hash of the ImGui version number, the checked-out commits of ImGui, ImPlot, ImPlot3D and ImGuiFileDialog, the `imconfig.h` and the UI-stack definitions that the applet was compiled against ([The UI-stack hash](../applets/sdk-package.md#ui-stack-hash)). Uncommitted edits to those libraries' sources do not change it. An applet shares the host's ImGui context, so a mismatch that the epoch number does not catch, such as a changed `imconfig`, would crash in ways that are hard to trace. When the key is present and differs from the host's own hash, the host refuses the applet before it opens the library. A manifest without the key, or a host built without a hash, is not checked.

### Applet Versions and Requirements

An applet's `[applet].version` is strict semver, chosen by its author, and the descriptor compiled into the library must carry the same string. The manifest also states what the applet requires, and the host checks each requirement before it opens the library:

| Requirement | Key | The host accepts the applet when |
|---|---|---|
| ABI epoch | `[compat].abi_epoch` | The value equals the host's epoch. |
| Oldest host | `[compat].min_host` | The host's version is at least this one, compared numerically. Without the key there is no floor. |
| Required services | `[services].required` | The host serves every id. The refusal names the first missing one, and the binary's own list is checked again after the library is opened. |
| Optional services | `[services].optional` | Always. The applet probes these services at run time and falls back when one is missing. |
| libtorch | `[compat].torch` | The value equals the version of the libtorch that the host provides. Applets that do not link libtorch leave the key out. |

A process loads one libtorch, which the host provides for every applet in it (see [Embedding](../introduction/architecture.md#embedding)), and the libtorch C++ ABI differs between versions. An applet that links libtorch can therefore run only beside the exact version it was built against. A host that provides no libtorch, such as one built with `CALIPER_WITH_TORCH=OFF`, refuses every applet that sets `[compat].torch`. `caliper_add_applet(... TORCH)` writes the key into the built manifest, with the version of the libtorch that the SDK names.

#### Several copies of one applet

Several copies of one applet id can be installed at once, for example a copy shipped with the app, a newer one in the user applets folder and a development build. The host loads one copy per id. It takes the highest version whose library is present and that passes every check made before the library is opened: the requirements above, the [UI-stack hash](#sdk-version) and the [data schema](#stored-data-versions). When the newest copy is refused, the host falls back to the next highest, and so on. The choice uses only those checks, so choosing a copy never runs applet code. The landing screen still lists every copy: a refused copy with its reason, and each other copy that passed as `SHADOWED`, with a text that names the chosen copy.

The host searches in this order:

1. The development folders: the `caliper dev` folder, those given with `--applets <dir>` in order, then those in `CALIPER_APPLET_PATH` (see [Caliper CLI](../reference/cli.md#search-order)).
2. The open workspace's `applets/` folder, which holds [spec applets](../applets/manifest.md#spec-applets) only, when a workspace is open. The app does not open one yet.
3. The user applets folder.
4. The `applets/` folder beside the executable, which holds the applets shipped with the app.

Spec applets and compiled applets share one set of ids, and the copies of one id are compared by the same rule whatever their kind. A spec applet has no library; it passes when its spec file is in place and its `min_host` and spec format version are ones this host meets.

Version decides, and the search order only breaks ties between equal versions, in favour of the copy found first. A development build therefore replaces an installed copy only when its version is at least as high, and a user-installed update replaces a shipped copy without changing the app.

### Stored Data Versions

Each applet owns a data folder, `data/<id>/` under the app-data root, which the host passes to it as `applet_data_dir`. The folder is named by the applet's id, so every version of the applet opens the same folder. `[data].schema` is an integer of at least 1 that versions what the applet stores there.

- The applet migrates its own data upward in `initialize()`. A version that finds data written under a lower schema converts it before using it. The host never migrates the data.
- The host records the highest schema that has opened the folder in a small stamp file inside it, `.caliper-data.toml`. It raises the stamp before it calls `create()` and `initialize()`, because migration begins there, so data that a failed `initialize()` left half-migrated is still marked. The stamp belongs to the host, and the applet does not write or remove it.
- The host refuses an applet whose declared schema is lower than the recorded one, before it opens the library. Together with the fallback in [Several copies of one applet](#several-copies-of-one-applet), this keeps an older version that declares a schema from opening data written under a higher one.
- An applet that declares no schema is never checked and leaves the stamp unchanged. An older version without the key can therefore open data that a newer version has migrated, so an applet that stores data declares a schema from its first version that does.

An author raises the schema when a change to the stored data makes it unreadable for older versions of the applet, and keeps it when older versions can still read the data.

## Promotion Rule

A feature moves into the core only when a second applet needs it. Until then it stays in the applet that built it. A published table cannot be taken back, so a service is extracted from working code rather than designed ahead of use. `caliper.data.v1` and `caliper.artifacts.v1` were frozen when EmbedScope needed them, and each geometry revision was added for an applet that needed it.

A promoted feature arrives as a new service id or as a new revision of an existing service, never as a change to a published table (see [Growth rules](#growth-rules)). From then on:

- An applet that needs the feature declares the new service id. A host that lacks the service refuses the applet when the id is in `[services].required`, and the applet falls back when the id is in `[services].optional`.
- An applet that bundles its own copy of the feature keeps working unchanged, because nothing it depends on was altered. It can move to the service in a later release of its own.

Sugar that is compiled into the applet, such as the `caliper::viz` components, needs nothing from the host at run time. It arrives in a minor version of the SDK, changes no table, and reaches an applet when the applet is rebuilt.

The first candidates for promotion are pieces that applets built for themselves: a Python sidecar, a layout grid and text scale, tours, automation and capture, and the `caliper::viz` components.

## Load Negotiation

The checks run in order, and the first nine happen **before `dlopen`**. The first failure becomes the applet's status on the landing screen, with the reason (see [Refusal messages](../reference/refusal-messages.md)), instead of a loader crash:

1. manifest parses, with a valid, path-safe `id`
2. platform binary present
3. the manifest's `abi_epoch` equals the host's epoch
4. `min_host` satisfied
5. `[build].ui_stack` equals the host's UI-stack hash, when both are present
6. `[compat].torch` equals the host's libtorch version, when the key is present
7. required services available (the manifest's list)
8. `[data].schema` is not lower than the recorded schema, when the key is present
9. of the copies of one id that pass, the highest version is chosen
10. *then* `dlopen`, after the data-schema check runs again
11. descriptor sanity — id / version / epoch agree with the manifest
12. required services available (the binary's own list)
13. the data-schema stamp is raised, then `create` and `initialize` run
