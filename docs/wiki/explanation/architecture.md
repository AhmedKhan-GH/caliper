# Architecture

*Adapted from `PLATFORM.md` §5.1–5.3. The spec is the source of truth.*

!!! warning "Planned topology"
    The layer and repository split below is the target architecture. The current repository still contains the host, SDK headers, examples, and in-tree applets together. Separate SDK releases, runtime packs, an applet registry, and packaged host binaries are not currently published.

Caliper is four layers with a single dependency rule: **arrows point down only.**

## Layers

```text
┌──────────────────────────────────────────────────────────────────┐
│ APPLETS — one repo each, own history, own CI, own releases       │
│   repnet-lab · circuitnet · opengllama · <anyone's idea>         │
│   artifact: <name>.caliperapp  (manifest + dylibs + assets)      │
├──────────────────────────────────────────────────────────────────┤
│ CALIPER-SDK — separate repo, tagged releases (THE CONTRACT)      │
│   include/caliper/abi.h          frozen C entry ABI (epoched)    │
│   include/caliper/tensor.h       CaliperTensor interchange type  │
│   include/caliper/services/*.h   named C service tables          │
│   include/caliper/caliper.hpp    header-only C++ sugar           │
│   ui-stack/                      PINNED imgui/implot/implot3d    │
│   cmake/                         caliper_add_applet, package cfg │
│   conformance/                   ABI lint + fixture host         │
├──────────────────────────────────────────────────────────────────┤
│ CALIPER HOST — this repo, slimmed to a shell                     │
│   window/GL/frame loop · launcher/Browse UI · applet manager     │
│   service implementations · crash guard · dev mode · pack mgr    │
├──────────────────────────────────────────────────────────────────┤
│ RUNTIME PACKS — host-managed heavy deps, shared across applets   │
│   libtorch-2.5.1-{cu121|macos-arm64} · (later: cuda extras, …)   │
└──────────────────────────────────────────────────────────────────┘
```

**Dependency rule.** Applets depend on the SDK; the host depends on the SDK too (as a consumer, via the same pinned releases); the SDK depends on nothing of Caliper's. Torch and DuckDB types never appear in any SDK header — the interchange types (`CaliperTensor`, Arrow C streams) cross the boundary instead. See [Compatibility & epochs](compatibility.md).

## Repo topology

Each layer maps to its own repository and release artifact:

| Repo | Contents | Release artifact |
|---|---|---|
| `caliper` | Host application, service implementations, `examples/hello` fixture applet, the spec | `Caliper-<ver>-<platform>.{dmg,zip}` |
| `caliper-sdk` | Headers, sugar, pinned UI stack (submodules at exact commits), CMake package, conformance harness | `caliper-sdk-<ver>` source tarball + optional prebuilt `ui-stack` libs |
| `caliper-applet-template` | Hello-world applet: 10-line CMakeLists, `caliper.toml`, tests, CI matrix | template — users instantiate |
| `caliper-registry` | `index.json` + per-applet manifests | the repo *is* the artifact |
| `repnet-lab`, `circuitnet`, `opengllama`, … | One applet each, history migrated via `git filter-repo` | `<name>-<ver>.caliperapp` per platform |

## The frame loop

The host owns: the GLFW window, the rendering backend, the ImGui/ImPlot/ImPlot3D contexts, and the frame clock. Per frame:

1. Host **begins** the ImGui frame.
2. The active applet's `frame()` renders its UI — and may consume GPU results produced by its own background jobs.
3. Host **ends** the frame and swaps buffers.

Applets never create windows or touch the graphics API. Long work never runs on the frame thread — that is what the `caliper.jobs.v1` service is for, and the host's watchdog flags applets that stall the loop (see the [Trust model](trust-model.md)).

!!! note "Renderer-agnostic by contract"
    The ABI never mentions a graphics API. Textures cross the boundary as an opaque `CaliperTextureId`, and applets render exclusively through ImGui/ImPlot. Native backends (Metal on macOS, Vulkan on Windows) are the target, with OpenGL 3.3 core as a frozen fallback — but because the contract hides all of this, backend changes require no epoch bump and no applet rebuilds. See [Rendering](rendering.md) for the `HostRenderer` seam and backend status, and `PLATFORM.md` §5.4.
