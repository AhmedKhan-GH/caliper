# Caliper Platform — Architecture & Delivery Plan

| | |
|---|---|
| **Status** | Draft for review |
| **Date** | 2026-06-10 |
| **Owner** | Ahmed Khan |
| **Scope** | Converting Caliper from a monorepo application into a platform: a versioned SDK of ML+visualization primitives, independently developed applets, and a distribution pipeline for hosts, applets, and heavy runtimes. |

> **How to read this document.** §2 audits what exists in the repository **today** — it is the only section describing current reality. Everything from §5 onward is written in the present tense, as specs are, but describes the **proposed target state: none of it exists yet**, except the pieces §2 explicitly marks "keep". The delta table at the end of §2 gives the today→target mapping at a glance, §17 assigns every change to a migration phase, and §18 lists each design decision awaiting your ratification (only D1 reflects code that already exists).

---

## 1. Executive Summary

Caliper today is a working application with a promising seam in the middle: applets are already shared libraries loaded over a C ABI at runtime. What it is **not yet** is a platform. The SDK exists only as include-paths into this repository, so an applet cannot be developed, versioned, tested, or shipped without the entire Caliper tree. The host context is a fixed struct that cannot grow services without breaking every applet. Heavy runtimes (LibTorch, DuckDB) are fused into the master build.

This document specifies the target state and the migration to it:

1. **A contract** — `caliper-sdk`: a tiny frozen C ABI plus named, versioned **services** (the CLAP/Vulkan model), wrapped in a header-only C++ layer so applet authors write modern C++ against ImGui/ImPlot directly.
2. **A set of ML-native primitives** — device negotiation, a DLPack-style tensor interchange type, a zero-copy tensor→texture bridge, TensorBoard-vocabulary metrics, Arrow-based data access, background jobs, and an artifact store. These are extracted from what `repnet_demo` already proves works, not invented.
3. **A packaging story** — applets as `.caliperapp` bundles with manifests, built in their own repositories against tagged SDK releases; heavy dependencies as host-managed **runtime packs**; a git-repo registry (the Homebrew-tap model) for discovery and install.
4. **A migration plan** — six strangler-fig phases, each leaving the repo shippable, ending with `applets/` deleted from this repo and every applet living its own life with its own history.

The unique selling point — **tightly integrated visualization running on bare metal (CUDA/MPS), in the same frame loop as training** — is preserved by keeping applets in-process and is productized as the `tensor_bridge` service, running GPU-resident on native rendering backends (Metal on macOS, Vulkan on Windows) behind a renderer-agnostic contract (§5.4).

---

## 2. Where We Are (Current-State Audit)

### What already works (keep it)

| Asset | Location | Verdict |
|---|---|---|
| C ABI plugin boundary (6 `extern "C"` functions, ABI int check) | `src/applet_api.h` | **Correct foundation.** Most projects get this wrong (C++ vtables across DLLs). Keep the approach, evolve the shape (§6). |
| Runtime discovery via `dlopen` from `<exe>/applets/` + user data dir | `src/applet_host.cpp` | Keep. Becomes bundle-aware (§10). |
| Host owns window/GL/ImGui/ImPlot contexts; applets render into them | `src/main.cpp`, `applet_initialize` | Keep — this *is* the product. Formalize the ImGui version pin (§9). |
| Per-applet CMake targets producing `SHARED` libs | `applets/*/CMakeLists.txt` | Keep concept; replace boilerplate with SDK-provided `caliper_add_applet()` (§10). |
| Pimpl applet classes, opaque `void* ctx` | all applets | Keep. All C++ complexity stays inside the dylib. |

### What blocks the platform (fix it)

| Problem | Evidence | Consequence |
|---|---|---|
| SDK is an in-tree `INTERFACE` target pointing at `${CMAKE_SOURCE_DIR}/src` and `third_party/*` | root `CMakeLists.txt:78-94` | Applets cannot build without a full Caliper checkout. No independent repos, CI, or releases. |
| Host context is a fixed struct (`CaliperHostContext`) | `src/applet_api.h` | Adding any host capability (metrics, jobs, device) is an ABI break for every applet. |
| Heavy deps are monorepo globals | `repnet_demo/CMakeLists.txt` links `duckdb_static`, `${TORCH_LIBRARIES}` from the root tree | Every applet build drags the whole dependency forest; applets can't pin their own versions; the build is the "hot potato". |
| Applets registered by glob into the root build | root `CMakeLists.txt:148-170` | Applet code must physically live in this repo to exist. |
| Static MSVC CRT (`MultiThreaded`) + DLL plugins | root `CMakeLists.txt:23` | Each DLL gets its own heap/CRT state. ImGui contexts cross the boundary today → latent intermittent crashes on Windows the moment an applet triggers an ImGui allocation. (§6d, §14) |
| Applets include monorepo-relative third-party paths | `repnet_demo` includes `third_party/llama.cpp/vendor` for nlohmann/json | Accidental coupling: an applet depends on another applet's vendored tree. |
| No manifest; dlopen first, ask later | `applet_host.cpp` | Version/feature mismatches surface as loader crashes instead of friendly UI. |

### Inventory of latent SDK primitives inside `repnet_demo`

The Training Lab work (commits `7a66d90`…`e33b7f8`) already implements, privately, most of what the platform should offer publicly:

| In repnet_demo today | Becomes platform service |
|---|---|
| Live training thread + progress + cancel (`train/train_engine.cpp`) | `caliper.jobs.v1` |
| Loss/AUROC metric streams plotted live | `caliper.metrics.v1` |
| Kernel/saliency tensors → GL textures (`model_viz.cpp`) | `caliper.tensor_bridge.v1` |
| MPS/CUDA/CPU device pick | `caliper.device.v1` |
| DuckDB/Parquet dataset loading (`dataset.cpp`) | `caliper.data.v1` |
| Per-applet data dir (`app_paths.cpp`, duplicated from host!) | `CaliperHost.applet_data_dir` |

**Strategy: extract, don't invent.** Phase 2 is surgery on proven code.

### Current vs. target at a glance

Every row below is a **proposed change** — the left column is what the repo does today, the right columns are what this document specifies and where in §17 it lands.

| Area | Today (current reality) | Target (this proposal) | Phase |
|---|---|---|---|
| Applet entry ABI | 6 `dlsym`'d C functions, `CALIPER_APPLET_ABI 1` (`src/applet_api.h`) | one `caliper_applet_descriptor()` export, ABI epoch 2 (§6a) | 1 |
| Host → applet capabilities | fixed `CaliperHostContext` struct (3 UI contexts + `data_dir`) | `get_service()` registry + 8 versioned services (§6b, §7) | 1–2 |
| ImGui across the DLL boundary | contexts passed; **no allocator handoff**; static CRT on Windows (latent crash) | `caliper.ui.v1` hands over allocators; Windows moves to `/MD` (§6d) | 1 |
| SDK | in-tree `caliper_applet_sdk` INTERFACE target pointing at `${CMAKE_SOURCE_DIR}` paths | installable `caliper-sdk` CMake package → its own repo with tagged releases (§5.2, §10) | 0, 3 |
| Where applets live & build | `applets/*` glob in the root CMake — must be inside this repo | own repos (history migrated), built against SDK release artifacts (§13.2) | 3–4 |
| Applet discovery | scan for bare dylibs, `dlopen` first | `.caliperapp` bundles, manifest checked **before** `dlopen`, friendly failure cards (§10, §14) | 4 |
| Heavy dependencies | libtorch + DuckDB fused into the monorepo build; applets link from the tree | libtorch as a shared **runtime pack**; DuckDB embedded in the host only (§11) | 4 |
| Metrics / jobs / tensor-viz / device pick | private code inside `repnet_demo` | public services: `metrics.v1`, `jobs.v1`, `tensor_bridge.v1`, `device.v1` (§7) | 2 |
| Renderer | OpenGL + GLEW everywhere; GL texture ids implicit in viz code | renderer-agnostic ABI (`CaliperTextureId`); native Metal (macOS) + Vulkan (Windows); GL frozen fallback (§5.4) | 2, 4 |
| UI stack (imgui/implot/…) | monorepo `third_party/` submodules | pinned inside the SDK; the pin defines the ABI epoch (§9) | 3 |
| Distribution | clone the monorepo and build everything | host binaries via GitHub Releases; applets via registry Browse tab or sideloading; `caliper new/dev/package` CLI (§12–13) | 4–5 |
| Testing | ad-hoc; some applet-level tests | TDD'd loader/negotiation/service contract tests, fixture host in SDK, golden-applet wall in CI (§16) | every phase |

---

## 3. Design Goals & Non-Goals

### Goals

1. **Independent applet lifecycles.** An applet lives in its own repo, with its own history (migrated, not restarted), its own CI, its own release cadence. Building one requires an SDK release artifact — never a Caliper checkout.
2. **A contract that grows without breaking.** Host capabilities are added for years without recompiling existing applets. Breaking changes are rare, explicit (ABI epoch), and detected before `dlopen`.
3. **ML-native primitives.** The SDK's vocabulary is the TensorBoard/W&B data model (runs, tags, steps, artifacts) fused with immediate-mode, on-GPU rendering. Applet authors get dashboards, jobs, and zero-copy visualization for free.
4. **The 10-minute first applet.** Scaffold → build → hot-reload in a released host binary, without compiling the host.
5. **Small host, on-demand heavyweight runtimes.** The host ships in tens of MB; LibTorch arrives once, as a shared runtime pack, only when an applet needs it.
6. **Honest safety.** Manifest checks before load, crash quarantine with visible errors, and a documented trust model (in-process = trusted code, until Phase 6).

### Non-Goals (for this plan)

- **Out-of-process applet isolation.** Kills the zero-copy/same-frame-loop USP for the core path. Reserved for untrusted binaries in Phase 6.
- **A scripting layer (Python/Lua).** Valuable later; the founding audience writes C++/CUDA.
- **A hosted registry service.** A git repo is the registry until real strangers arrive.
- **Cross-compiler binary compatibility on one platform.** We pin toolchains per platform (§14) — the same trade every ImGui-based plugin host makes.

### Audience assumption

Designing for **(b): collaborators building applets from source against published SDK releases**, with contracts strict enough that **(c): strangers installing prebuilt applet binaries** requires only policy additions (signing, review), not redesign.

---

## 4. Prior Art — What We Take From Whom

| System | What it proves | What Caliper takes |
|---|---|---|
| **CLAP** (`clap_host->get_extension()`) | A frozen micro-ABI + named versioned extension tables lets hosts and plugins ship independently for a decade. | The entire service model (§6b, §7). Service structs are immutable once published; new capability = new id (`…v2`) alongside the old. |
| **VST3 / Audio Unit bundles** | Plugins as *bundles* (binary + metadata + resources), validated before load. | `.caliperapp` bundle format + manifest pre-flight (§10). |
| **OBS Studio** | A C plugin API + `find_package(libobs)` SDK yields thousands of third-party plugins in independent repos. | SDK as a standalone CMake package; `caliper_add_applet()` owning build boilerplate (§10). |
| **VS Code** | The dev loop is the product: scaffold → F5 → publish. Extension authors never build the editor. Manifest-declared capabilities, lazy activation, an index-based marketplace. | Template repo + `caliper dev` hot-reload against the *released* host (§13); manifest-declared services; registry-as-index (§12). |
| **Vulkan** | Versioned dispatch tables + `sType/pNext`-style growth; loader↔driver version negotiation. | `struct_size` as first field of every ABI struct; epoch negotiation before load (§6, §14). |
| **TensorBoard / W&B / MLflow** | The industry-standard ML observability vocabulary: experiment → run → tag → step series; hparams; artifacts. | The data model of `caliper.metrics.v1` and `caliper.artifacts.v1` (§7). |
| **DLPack** | A minimal C tensor descriptor is sufficient for zero-copy interchange across every ML framework. | `CaliperTensor` layout (§7.2), deliberately DLPack-aligned so torch↔caliper conversion is trivial and other frameworks can join later. |
| **Apache Arrow C Data Interface** | The purpose-built C ABI for tabular data across library boundaries; DuckDB speaks it natively. | The wire format of `caliper.data.v1` (§7.6). |
| **CUDA Toolkit / conda** | Users accept platform-managed, versioned runtime directories; apps don't bundle gigabyte runtimes. | Runtime packs (§11). |
| **vcpkg registries / Homebrew taps** | A git repo of manifests *is* a registry. Publishing = a PR. | `caliper-registry` (§12). |
| **Dear ImGui's own DLL guidance** | Crossing a DLL boundary requires `SetCurrentContext` **and** `SetAllocatorFunctions`. | `caliper.ui.v1` hands over allocators; sugar applies them in one call (§6d). |
| **wxWidgets `wxGraphicsContext`** | The mature GUI-framework world converged on the same renderer answer: one abstract drawing API over *native* backends (Direct2D / CoreGraphics / Cairo), not one cross-platform GPU API. | Independent validation of the `HostRenderer` pattern (§5.4): standardize the interface, go native underneath. Also the toolkit of Compass, the planned interface-heavy second host (§17 Phase 6). |

---

## 5. Architecture Overview

### 5.1 Layers

```
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

Dependency rule: arrows point **down only**. Applets depend on the SDK; the host depends on the SDK (as a consumer, via the same pinned releases); the SDK depends on nothing of Caliper's. Torch and DuckDB types never appear in any SDK header (§6e).

### 5.2 Repo topology

| Repo | Contents | Release artifact |
|---|---|---|
| `caliper` | Host application, service implementations, `examples/hello` fixture applet, this document | `Caliper-<ver>-<platform>.{dmg,zip}` |
| `caliper-sdk` | Headers, sugar, pinned UI stack (submodules at exact commits), CMake package, conformance harness, applet template's CI snippets | `caliper-sdk-<ver>` source tarball + per-platform prebuilt `ui-stack` static libs (optional convenience) |
| `caliper-applet-template` | Hello-world applet repo: 10-line CMakeLists, `caliper.toml`, tests, CI matrix | (template — users instantiate) |
| `caliper-registry` | `index.json` + per-applet manifests | (the repo *is* the artifact) |
| `repnet-lab`, `circuitnet`, `opengllama`, … | One applet each, **history migrated** from this repo via `git filter-repo` | `<name>-<ver>.caliperapp` per platform |

### 5.3 The frame loop (unchanged, formalized)

The host owns: GLFW window, the rendering backend (§5.4), ImGui/ImPlot/ImPlot3D contexts, the frame clock. Per frame: host begins ImGui frame → active applet's `frame()` renders its UI (and may consume GPU results produced by its own jobs) → host ends frame, swaps. Applets never create windows or touch the graphics API (§6c). Long work never runs on the frame thread — that's what `caliper.jobs.v1` is for, and the host's watchdog flags applets that stall the loop (§15).

### 5.4 Rendering backend strategy — native first, GL demoted to fallback

The USP is GPU-resident visualization, so the renderer must speak the API the tensors actually live in — and OpenGL doesn't: it is deprecated on macOS (capped at 4.1) and has no path to MPS memory, meaning every Mac tensor would take a CPU round-trip on its way to becoming a texture — a copy in the hottest loop the platform owns. Because zero external applets exist yet, the renderer question is settled **now, at the contract level**, and the migration itself becomes a host-internal detail forever:

1. **The ABI never mentions a graphics API.** Textures cross the boundary as opaque `CaliperTextureId` (§7.4); applets render exclusively through ImGui/ImPlot (§6c rule). Backend changes therefore require no epoch bump and no applet rebuilds — ever.
2. **`HostRenderer` — a host-internal interface** (surface/swapchain, ImGui backend binding, frame begin/end, texture create/update/release over `CaliperTensor` descriptors) with three implementations:
   - **Metal (macOS) — primary.** `imgui_impl_metal` + a `GLFW_NO_API` window with a `CAMetalLayer`. MPS torch tensors are `MTLBuffer`s: the bridge aliases them as textures (zero-copy when layout/alignment permit) or blits device-side — either way, no CPU staging.
   - **Vulkan (Windows now, Linux later) — primary.** `imgui_impl_vulkan`; CUDA interop via `VK_KHR_external_memory` + `cudaImportExternalMemory` — the modern successor to CUDA↔GL interop, present on every CUDA-capable machine.
   - **OpenGL 3.3 core profile — frozen fallback** for VMs/remote/odd environments: CPU-staged uploads, kept working, never extended. **Core, forward-compatible context only — the compatibility profile is banned**, so the fallback can never quietly accumulate fixed-function code. (Sibling project Compass shows that drift in the wild: it settled on GL 2.1 `glBegin`/`glEnd` because nothing forced a decision, and is now stranded on a doubly-deprecated path.) When this code is touched in Phase 4, GLEW is replaced by a GLAD-generated 3.3-core-only loader — tiny, and it ends the loader question — or deleted along with the fallback if it proves unneeded.
3. **The tensor bridge is specified against `HostRenderer`** and built once, natively (Phase 2 on Metal, Phase 4 adds Vulkan) — not first on GL and rewritten later.

Sequencing rationale: Metal lands with Phase 2 because that's when the bridge is built and the primary dev machine is Apple Silicon — otherwise the staged-copy ceiling sits in the middle of the reference applet's demo for the platform's entire formative period.

---

## 6. The Contract — ABI Epoch 2

Everything in this section lives in `caliper-sdk/include/caliper/` and is the **only** thing host and applets must agree on. Current `CALIPER_APPLET_ABI 1` applets keep loading through a compatibility shim until Phase 1 completes.

### 6a. Entry point: one descriptor, not six dlsyms

```c
// caliper/abi.h — frozen. Changes here = epoch bump (rare, §14).
#define CALIPER_ABI_EPOCH 2

typedef struct CaliperHost CaliperHost;          // §6b
typedef struct CaliperFrameInfo {
    uint32_t struct_size;        // sizeof(CaliperFrameInfo) — versioning w/o breaks
    int32_t  fb_width, fb_height; /* PHYSICAL framebuffer pixels — never logical units */
    float    dpi_scale;           /* physical = logical × dpi_scale (2.0 on Retina) */
    double   time_sec, delta_sec;
} CaliperFrameInfo;

typedef struct CaliperAppletAPI {
    uint32_t struct_size;
    void* (*create)(void);
    void  (*destroy)(void* self);
    bool  (*initialize)(void* self, const CaliperHost* host);
    void  (*frame)(void* self, const CaliperFrameInfo* info);
    void  (*cleanup)(void* self);
    /* appended later, guarded by struct_size:
       save_state / load_state (hot-reload state preservation),
       on_suspend / on_resume (backgrounded applets) */
} CaliperAppletAPI;

typedef struct CaliperAppletDescriptor {
    uint32_t struct_size;
    uint32_t abi_epoch;                      // must equal a host-supported epoch
    const char* id;                          // reverse-DNS: "dev.ahmed.repnet-lab"
    const char* version;                     // applet semver "0.3.0"
    const char* name;                        // "RepNet Lab"
    const char* summary;
    const char* tag;                         // landing-page category
    const char* const* required_services;    // NULL-terminated ids; host checks pre-load
    CaliperAppletAPI api;
} CaliperAppletDescriptor;

CALIPER_EXPORT const CaliperAppletDescriptor* caliper_applet_descriptor(void);
```

Why a descriptor: one symbol to resolve; metadata readable without instantiating anything; the function table is data, so future entry points are *appended fields*, not new exported symbols. This is the CLAP/VST3 shape.

**The pixel-space contract.** `fb_width`/`fb_height` are **physical framebuffer pixels**; ImGui/ImPlot coordinates are **logical units**; `dpi_scale` converts between them (2.0 on Retina, 1.0 on standard displays); `tensor_bridge` texture dimensions are always physical. Conflating the two spaces is *the* classic cross-platform rendering bug — sibling project Compass documents hitting exactly it (logical `GetSize()` vs. the 2× physical framebuffer on Retina, `compass/README.md` "macOS Retina Display Fix") — so the semantics are part of the ABI, not folklore, and the conformance suite runs the fixture applet at `dpi_scale = 2.0` to catch violations (§16).

### 6b. The host is a service registry, not a struct of fields

```c
struct CaliperHost {
    uint32_t    struct_size;
    uint32_t    abi_epoch;          // epoch this host is speaking
    uint32_t    host_version;       // (major<<16)|(minor<<8)|patch, informational
    const char* applet_data_dir;    // per-applet sandboxed storage, UTF-8
    /* THE extension point. Returns a service table or NULL. The pointer is
       valid for the applet's lifetime. Unknown ids return NULL — never UB. */
    const void* (*get_service)(const CaliperHost* host, const char* service_id);
};
```

**Growth rules (the platform's constitution):**

1. A published service struct is **immutable**. Capability additions ship as a *new id* (`caliper.metrics.v2`) alongside the old one, which keeps working.
2. Hosts may provide any set of services; applets declare `required_services` (refusal happens at manifest check, with a friendly card — §10) and probe optional ones at runtime.
3. `struct_size` is always the first field; a reader never touches bytes beyond the writer's declared size.
4. The ABI **epoch** bumps only for: entry-point changes, `CaliperHost` layout changes, or UI-stack pin changes (§9). Target cadence: ≤1 per year after stabilization.

### 6c. ABI hygiene rules (documented in the SDK, enforced by conformance lint)

- C types only: no STL, no exceptions across the boundary (sugar catches everything at the edge and reports via `caliper.log.v1`).
- No `torch::`, `duckdb::`, or any third-party C++ type in any signature — interchange is `CaliperTensor` (§7.2) and Arrow C streams (§7.6).
- No graphics-API types or handles (GL/Metal/Vulkan) anywhere in the ABI, and applets never issue raw graphics calls — rendering happens exclusively through ImGui/ImPlot and `caliper.tensor_bridge` textures. This single rule is what keeps the host's renderer swappable forever (§5.4); an explicit escape-hatch service could exist later for an applet that truly needs custom GPU drawing, clearly marked as backend-locking.
- Memory allocated by a side is freed by that side. Host-returned strings/buffers are host-owned with documented lifetimes.
- All strings UTF-8 `const char*`.
- All structs are POD, `struct_size`-prefixed, alignment-stable.

### 6d. `caliper.ui.v1` — and the Windows CRT bug this fixes

The defining trade of the platform: applets program **raw ImGui/ImPlot/ImPlot3D** (maximum DX, zero wrapper lag) in exchange for using the SDK's pinned UI stack (§9). The contexts cross the DLL boundary, so the allocators must cross with them — today they don't, which is a latent intermittent crash on Windows where the static CRT gives each DLL its own heap:

```c
// caliper/services/ui_v1.h
#define CALIPER_UI_V1 "caliper.ui.v1"
typedef struct CaliperUiV1 {
    uint32_t struct_size;
    struct ImGuiContext*    (*imgui_context)(void);
    struct ImPlotContext*   (*implot_context)(void);
    struct ImPlot3DContext* (*implot3d_context)(void);
    /* Host's allocator pair — applet side MUST install these into its copy of
       ImGui's globals so every allocation crosses on the host's heap. */
    void (*imgui_allocators)(ImGuiMemAllocFunc* out_alloc,
                             ImGuiMemFreeFunc*  out_free,
                             void** out_user_data);
} CaliperUiV1;
```

The C++ sugar collapses correct usage into one call applet authors cannot get wrong:

```cpp
caliper::ui::connect(host);   // SetAllocatorFunctions + SetCurrentContext ×3
```

Additionally, host and applets move to the **dynamic CRT (`/MD`)** on Windows (§14) so non-ImGui CRT state (locale, FILE*, errno) also agrees. The static-CRT requirement originally came from DuckDB defaults; DuckDB builds fine as `/MD` with `DUCKDB_FORCE_STATIC_CRT=OFF`-style configuration, to be verified in Phase 1.

### 6e. What deliberately never crosses the boundary

| Type | Why not | Interchange instead |
|---|---|---|
| `torch::Tensor` | libtorch C++ ABI varies per version/build; would weld every applet to the host's torch | `CaliperTensor` (raw pointer + dtype/shape/strides/device) |
| `duckdb::*` | same reason | SQL strings in, **Arrow C streams** out |
| `std::string`/`std::vector` | libstdc++/libc++/MSVC STL layouts differ | `const char*`, pointer+length |
| exceptions | undefined across runtimes | `bool`/error-code returns + `caliper.log.v1` |

---

## 7. The Primitives — Service Catalog v1

Each service: a C table in `caliper/services/<name>_v1.h`, a host implementation, a sugar wrapper, and a conformance test. Origin column shows what existing code gets extracted.

**Design rule — the service layer stays host-neutral.** `caliper.ui.v1` is the *only* service allowed to know that a UI toolkit or renderer exists. The other seven must work unchanged in any host: the headless fixture host (§16), and a future second host with entirely different chrome (§17 Phase 6 — Compass). A proposed service that needs to know about ImGui, windows, or the graphics backend is either part of `ui.vN` or doesn't belong in the catalog.

| Id | Purpose | Origin |
|---|---|---|
| `caliper.ui.v1` | ImGui/ImPlot/ImPlot3D contexts + allocators | `applet_initialize` today |
| `caliper.log.v1` | structured logs into host console | `printf` chaos today |
| `caliper.device.v1` | negotiated compute device | per-applet `pick_device()` |
| `caliper.tensor_bridge.v1` | tensor → live texture, GPU-resident (opaque `CaliperTextureId`) | `model_viz.cpp` |
| `caliper.jobs.v1` | background compute w/ progress+cancel | `train_engine.cpp` thread |
| `caliper.metrics.v1` | run/tag/step scalars, histograms, images + free dashboards | Training Lab plots |
| `caliper.data.v1` | dataset catalog + SQL, Arrow out | `dataset.cpp` |
| `caliper.artifacts.v1` | content-addressed weights/checkpoints | ad-hoc file paths |

### 7.1 `caliper.log.v1`

```c
typedef enum { CALIPER_LOG_DEBUG=0, CALIPER_LOG_INFO, CALIPER_LOG_WARN, CALIPER_LOG_ERROR } CaliperLogLevel;
typedef struct CaliperLogV1 {
    uint32_t struct_size;
    void (*log)(CaliperLogLevel level, const char* message_utf8);  // pre-formatted
} CaliperLogV1;
```

Host renders a filterable console panel; dev mode tails it. Sugar: `caliper::log::info("epoch {} done", e)` (fmt-style formatting happens applet-side).

### 7.2 `CaliperTensor` — the lingua franca (a type, not a service)

```c
// caliper/tensor.h — DLPack-aligned on purpose: torch/numpy/jax interop is a cast away.
typedef enum { CALIPER_DT_F32=0, CALIPER_DT_F16, CALIPER_DT_BF16,
               CALIPER_DT_I64, CALIPER_DT_I32, CALIPER_DT_U8 } CaliperDType;
/* Device kinds name the MEMORY/API DOMAIN, not a framework backend: METAL
 * covers torch-MPS, MLX, and ggml-Metal alike (all unified-memory MTLBuffers).
 * Renamed from the earlier MPS sketch before first shipping (Phase 2). */
typedef enum { CALIPER_DEV_CPU=0, CALIPER_DEV_CUDA, CALIPER_DEV_METAL } CaliperDeviceKind;

typedef struct CaliperTensor {
    uint32_t         struct_size;
    void*            data;            // device or host pointer
    CaliperDType     dtype;
    int32_t          ndim;            // ≤ 8
    int64_t          shape[8];
    int64_t          strides[8];      // in elements
    CaliperDeviceKind device;
    int32_t          device_index;
    void*            stream;          // cudaStream_t / MTLCommandQueue* / NULL
} CaliperTensor;
```

Sugar provides free conversions at the edge, inside the applet: `caliper::to_tensor(const torch::Tensor&)` / `caliper::from_tensor(...)` (header-only, compiled against the applet's own libtorch — torch still never enters the ABI).

### 7.3 `caliper.device.v1`

```c
typedef struct CaliperDeviceV1 {
    uint32_t struct_size;
    CaliperDeviceKind (*kind)(void);            // host's negotiated device
    int32_t           (*index)(void);
    const char*       (*name)(void);            // "Apple M3 Max", "RTX 4090"
    uint64_t          (*free_memory_hint)(void);// best-effort, bytes
} CaliperDeviceV1;
```

Applets stop writing device-pick logic; the host decides once (user-overridable in settings) and every applet agrees — which also makes the tensor bridge's interop assumptions valid.

### 7.4 `caliper.tensor_bridge.v1` — the USP, productized

```c
typedef uint64_t CaliperTextureId;   /* opaque; cast to ImTextureID. 0 = invalid.
                                        NEVER a raw GL/Metal/Vulkan handle (§5.4) */

typedef struct CaliperTensorBridgeV1 {
    uint32_t struct_size;
    /* Mirror a 2-D (H,W) or 3-D (H,W,C≤4) tensor as a texture.
       Native backends (§5.4): Metal buffer aliasing on MPS, Vulkan
       external-memory + CUDA import on Windows — GPU-resident, zero-copy
       where layout permits, device-side blit otherwise. GL fallback:
       CPU-staged upload. Returns 0 on failure (reason via caliper.log.v1). */
    CaliperTextureId (*texture_from_tensor)(const CaliperTensor* t, uint32_t flags);
    bool (*update_texture)(CaliperTextureId tex, const CaliperTensor* t);
    void (*release_texture)(CaliperTextureId tex);
    /* Built-in colormaps for 1-channel tensors (viridis, magma, RdBu …) */
    CaliperTextureId (*texture_from_tensor_mapped)(const CaliperTensor* t,
                                                   int32_t colormap,
                                                   float vmin, float vmax,
                                                   uint32_t flags);
    /* Literal zero-copy: allocate tensor memory that IS the texture's backing
       store (shared MTLBuffer / Vulkan external memory imported into CUDA).
       The applet wraps out_tensor->data (torch::from_blob) and writes from
       kernels; the texture sees it after at most a layout transition. */
    bool (*alloc_shared)(CaliperDType dtype, int32_t ndim, const int64_t* shape,
                         CaliperTensor* out_tensor, CaliperTextureId* out_texture);
    void (*free_shared)(CaliperTextureId tex);
} CaliperTensorBridgeV1;
```

This is the one call that turns "weights/activations/saliency living on the GPU" into "an `ImGui::Image` this frame" — `CaliperTextureId` casts straight to `ImTextureID`. No TensorBoard round-trip, no PNG encode, no Python, and on the native backends no CPU staging either. `alloc_shared` completes the story: the training loop writes weights *into the texture's own memory*, so live visualization costs a layout transition, not a copy. The GL fallback stages through the CPU — acceptable for a VM, not for the demo machine. Nothing mainstream offers this in-loop; it is the platform's reason to exist and the heart of the demo story.

### 7.5 `caliper.jobs.v1`

```c
typedef struct CaliperJobControl CaliperJobControl;
struct CaliperJobControl {
    uint32_t struct_size;
    bool (*cancelled)(const CaliperJobControl*);                       // poll in loops
    void (*progress)(const CaliperJobControl*, float frac, const char* msg);
};
typedef void (*CaliperJobFn)(void* user, const CaliperJobControl* ctl);

typedef struct CaliperJobsV1 {
    uint32_t struct_size;
    uint64_t (*submit)(const char* label, CaliperJobFn fn, void* user);
    void     (*request_cancel)(uint64_t job);
    bool     (*is_running)(uint64_t job);
    float    (*progress_of)(uint64_t job);
} CaliperJobsV1;
```

Host owns the worker threads, renders a global jobs tray (label, progress, cancel button), and guarantees `frame()` is never blocked by training. Generalizes `BackgroundProcessor`/`train_engine` — repnet's training loop becomes `jobs->submit("train split-17", …)` with its inner loop polling `cancelled()` and reporting `progress()`.

### 7.6 `caliper.metrics.v1` — TensorBoard vocabulary, ImPlot immediacy

```c
typedef struct CaliperMetricsV1 {
    uint32_t struct_size;
    uint64_t (*begin_run)(const char* experiment, const char* run_name); // 0 = error
    void     (*end_run)(uint64_t run);
    void     (*scalar)(uint64_t run, const char* tag, int64_t step, double value);
    void     (*histogram)(uint64_t run, const char* tag, int64_t step,
                          const float* values, int64_t count);
    void     (*image)(uint64_t run, const char* tag, int64_t step,
                      const CaliperTensor* hwc_u8);
    void     (*hparams_json)(uint64_t run, const char* json_utf8);
} CaliperMetricsV1;
```

Host persists to its embedded DuckDB (`metrics.duckdb` in the data dir) and renders a **Runs dashboard for every applet for free**: run compare, smoothing, step/wall-time axes — the TensorBoard feature set, live at frame rate, plus export to Parquet for offline analysis in Python. Every applet that logs a scalar instantly inherits the platform's observability.

### 7.7 `caliper.data.v1` — Arrow in, Arrow out

```c
struct ArrowSchema; struct ArrowArrayStream;   // Arrow C Data Interface (stable C ABI)

typedef struct CaliperDataV1 {
    uint32_t struct_size;
    /* Run SQL against the host catalog (DuckDB). Caller drains + releases stream. */
    bool (*query)(const char* sql_utf8, struct ArrowArrayStream* out);
    /* Register/open named datasets (parquet/csv/dir-of-files). */
    bool (*register_dataset)(const char* name, const char* uri);
    bool (*open_dataset)(const char* name, struct ArrowArrayStream* out);
    const char* (*last_error)(void);            // host-owned, valid until next call
} CaliperDataV1;
```

Arrow's C interface is *designed* for exactly this boundary; DuckDB exports it natively, and the sugar adapts streams to ranges of typed rows. Applets stop hardcoding file paths; datasets become named, inspectable, shareable across applets.

### 7.8 `caliper.artifacts.v1`

```c
typedef struct CaliperArtifactsV1 {
    uint32_t struct_size;
    /* Store bytes under a content hash, linked to a run (0 = unlinked).
       out_digest: 64 hex chars + NUL. */
    bool (*put)(const char* name, const void* bytes, uint64_t len,
                uint64_t run, char out_digest[65]);
    /* Resolve digest-or-name to a local file path (host-owned string). */
    const char* (*path_of)(const char* digest_or_name);
    bool (*exists)(const char* digest_or_name);
} CaliperArtifactsV1;
```

Checkpoints/exports become content-addressed, deduplicated, and lineage-tracked ("which run produced these weights") — the MLflow artifact idea without the server.

### 7.9 `caliper::viz::` — sugar-level component library (not ABI)

Header-only ImPlot/ImGui composites distilled from repnet_demo, compiled into each applet: `SaliencyOverlay` (the scrubbable signal+saliency view from `e33b7f8`), `KernelGrid`, `ConfusionMatrix`, `RocCurve`, `PrCurve`, `EmbeddingProjector` (later), `SignalScrubber`. Pure functions over `CaliperTensor`/plain arrays + the tensor bridge. Because they're source-level sugar, they iterate fast without ABI ceremony — the platform's "standard library" of ML visualization.

---

## 8. The C++ SDK (Sugar Layer)

Applet authors should never hand-write `extern "C"` boilerplate, service lookups, or allocator wiring. `caliper.hpp` (header-only):

```cpp
#include <caliper/caliper.hpp>
#include <imgui.h>
#include <implot.h>

class RepNetLab final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        metrics_ = host.service<caliper::Metrics>();      // typed, may be nullopt
        jobs_    = host.require<caliper::Jobs>();         // declared required → present
        run_     = metrics_->begin_run("repnet", "split-17");
        return true;
    }
    void on_frame(const caliper::Frame& f) override {
        ImGui::Begin("Training Lab");
        /* … raw ImGui/ImPlot, exactly like today … */
        ImGui::End();
    }
    void on_cleanup() override { metrics_->end_run(run_); }
private:
    std::optional<caliper::Metrics> metrics_;
    caliper::Jobs jobs_;
    uint64_t run_ = 0;
};

CALIPER_APPLET(RepNetLab,
    .id       = "dev.ahmed.repnet-lab",
    .version  = "0.3.0",
    .name     = "RepNet Lab",
    .summary  = "Live ECG training: kernels, saliency, metrics",
    .tag      = "ECG",
    .services = {"caliper.ui.v1", "caliper.jobs.v1", "caliper.metrics.v1"})
```

What the macro generates: the descriptor (with `required_services` array), the five `CaliperAppletAPI` functions bridging to the class, `caliper::ui::connect()` inside `initialize` (contexts **and** allocators), and a top-level `catch (...)` on every bridge function that logs and returns failure instead of unwinding across the boundary.

Wrapper guarantees: zero-overhead (thin inline calls over the tables), exception-safe at the edge, and *optional* — a C applet or another language's FFI can implement the raw ABI directly.

---

## 9. UI-Stack Pinning — the Epoch Policy

The platform's DX superpower is "write raw ImGui/ImPlot." The cost: ImGui is a C++ library whose context crosses the boundary, so **host and applet must compile the same ImGui**. We make that a managed, versioned fact instead of an accident:

- `caliper-sdk` vendors `ui-stack/` — imgui, implot, implot3d, ImGuiFileDialog as submodules at **exact commits**, plus the canonical `imconfig.h` (any `IMGUI_USER_CONFIG` divergence is an ABI break — conformance checks the config hash).
- The pinned set defines the **ABI epoch**. Upgrading ImGui = new epoch = applets rebuild (a `cmake --build` against the bumped SDK — minutes, flagged by CI, surfaced in the registry).
- The host builds from the same pin it implements; `applet_info`-time epoch equality is what makes context-sharing sound.
- Each applet compiles the UI stack from SDK sources (object/static lib via `caliper_add_applet`) but **renders into the host's contexts** — there is exactly one ImGui state at runtime. Symbol duplication across dylibs is fine because no ImGui objects are exchanged beyond the context pointers + allocators.

Toolchain contract per platform (documented in SDK README, checked by conformance where possible): macOS = AppleClang + libc++ (the OS ABI — relaxed in practice); Windows = MSVC, same major toolset as the host release, `/MD`; Linux (future) = gcc/libstdc++ floor pinned per epoch.

---

## 10. Build & Packaging

### 10.1 Consuming the SDK (applet side)

```cmake
cmake_minimum_required(VERSION 3.24)
project(repnet_lab CXX)

include(cmake/CPM.cmake)
CPMAddPackage("gh:ahmed/caliper-sdk@0.4.2")     # or FetchContent / find_package

caliper_add_applet(repnet_lab
    MANIFEST  caliper.toml
    SOURCES   src/repnet_lab.cpp src/training_tab.cpp
)
target_link_libraries(repnet_lab PRIVATE caliper::torch_stub)  # rpath into runtime pack (§11)
```

### 10.2 What `caliper_add_applet()` does (SDK-owned boilerplate)

1. `add_library(<name> MODULE …)` — `MODULE`, not `SHARED`: this is a dlopen-only artifact (also stops accidental link-time coupling).
2. Links `caliper::sdk` (headers) + `caliper::ui_stack` (the pinned ImGui set, built once per build tree).
3. Sets `CALIPER_APPLET_EXPORT`, visibility flags (`-fvisibility=hidden` + explicit export), C++17, platform-correct output name (no `lib` prefix), `@loader_path`/`$ORIGIN`-relative rpaths.
4. Parses + validates `caliper.toml` at configure time (id format, semver, epoch matches the SDK's, declared services exist in this SDK).
5. Generates the bundle layout in `${CMAKE_BINARY_DIR}/bundle/` and a `caliper_package` target producing `<name>-<ver>-<platform>.caliperapp` (zip).
6. Registers a `caliper_conformance` test (ctest) running the SDK's checker against the built dylib: exports present, descriptor sane, epoch/manifest agreement, imconfig hash match.

Ten lines of consumer CMake, impossible to drift — the OBS/`find_package(libobs)` lesson.

### 10.3 The manifest — `caliper.toml`

One file, present in the applet repo root and copied into the bundle:

```toml
[applet]
id         = "dev.ahmed.repnet-lab"     # reverse-DNS, globally unique
name       = "RepNet Lab"
version    = "0.3.0"                    # semver
summary    = "Live ECG training: kernels, saliency, metrics"
tag        = "ECG"
authors    = ["Ahmed Khan <emailahmedebadkhan@gmail.com>"]
license    = "MIT"
repository = "https://github.com/ahmed/repnet-lab"

[compat]
abi_epoch  = 2
min_host   = "0.6.0"                    # first host with jobs.v1
platforms  = ["macos-arm64", "windows-x64-cu121"]

[services]
required = ["caliper.ui.v1", "caliper.device.v1", "caliper.jobs.v1"]
optional = ["caliper.metrics.v1"]       # probed at runtime; degrade gracefully

[runtimes]
libtorch = "~2.5"                       # semver band, resolved by the pack manager
```

Host parses with vendored `toml++` (header-only). **Manifest is checked before any `dlopen`** — every mismatch becomes a friendly landing-page card ("Built for ABI epoch 1; this host speaks 2 — check for an update"), never a loader crash.

### 10.4 The bundle — `<name>.caliperapp/`

```
repnet-lab.caliperapp/
├── caliper.toml
├── bin/
│   ├── macos-arm64/librepnet_lab.dylib
│   └── windows-x64-cu121/repnet_lab.dll
├── assets/                  # models, fonts, sample data (≤ small; big data = datasets)
└── CHANGELOG.md             # surfaced in Browse UI
```

A directory, not an opaque archive — inspectable, diffable, AirDrop-able. Discovery scans `<exe>/applets/` and `<data>/applets/` for `*.caliperapp` (plus bare dylibs in dev mode). Multi-platform bundles are allowed (CI attaches per-platform bundles; a fat bundle is just their union).

### 10.5 Host user-space layout

```
~/Library/Application Support/Caliper/        (Windows: %APPDATA%\Caliper\)
├── applets/        repnet-lab.caliperapp/  circuitnet.caliperapp/
├── runtimes/       libtorch-2.5.1-macos-arm64/   (§11)
├── data/           dev.ahmed.repnet-lab/   …     (per-applet sandbox dirs)
├── metrics.duckdb  artifacts/                    (host services' storage)
└── registry/       index.json (cached)
```

---

## 11. Runtime Packs — solving the 2 GB problem

**Problem:** LibTorch is ~2 GB. Per-applet bundling is absurd; monorepo fusion is the status quo we're escaping.

**Model (CUDA-toolkit/conda-style):** versioned, host-managed, shared directories.

- A pack = `runtimes/<name>-<version>-<platform>/lib/*.dylib|dll` + `pack.toml` (name, version, platform, sha256 set, source URL).
- Known packs are listed in the registry repo (`packs.json`): name → version → platform → {url, sha256}. v1 packs: `libtorch` (cu121 / cpu / macos-arm64). DuckDB stays statically linked inside the host (it *is* a host implementation detail of metrics/data services).
- **Resolution at install/launch:** manifest declares `libtorch = "~2.5"` → pack manager finds or downloads a satisfying pack (checksum-verified, resumable) → records the applet→pack edge for GC.
- **Linking:** applets link `caliper::torch_stub` — SDK-provided import library/stubs matching the pack's soname set, so the build needs headers+stubs only (small, fetched by the SDK), never the full runtime. At load, the host makes the pack resolvable *before* `dlopen`-ing the applet: `AddDllDirectory(pack/lib)` on Windows; on macOS/Linux the loader pre-`dlopen`s the pack's libraries with `RTLD_GLOBAL` so the applet's undefined symbols bind to the already-loaded images.
- **One torch per process (hard policy v1):** the first torch-applet's resolved pack wins for the session; an applet demanding an incompatible band is refused with a clear card ("requires libtorch 2.7; session already holds 2.5 — restart to switch"). Honest and simple; multi-version isolation is exactly the Phase-6 out-of-process case.
- **GC:** packs with zero referencing applets are deletable from settings.

Result: host download stays ~50 MB (no torch in the host — the tensor bridge uses CUDA runtime/Metal APIs on raw pointers; metrics use embedded DuckDB). LibTorch downloads once, on demand, shared by every ML applet thereafter.

---

## 12. Registry & Distribution

### 12.1 `caliper-registry` (a git repo is the registry)

```jsonc
// index.json (schema 1)
{
  "schema": 1,
  "applets": {
    "dev.ahmed.repnet-lab": {
      "name": "RepNet Lab",
      "summary": "Live ECG training: kernels, saliency, metrics",
      "tags": ["ECG", "training"],
      "repo": "https://github.com/ahmed/repnet-lab",
      "releases": [{
        "version": "0.3.0",
        "abi_epoch": 2,
        "min_host": "0.6.0",
        "artifacts": {
          "macos-arm64":      { "url": "https://github.com/ahmed/repnet-lab/releases/download/v0.3.0/repnet-lab-0.3.0-macos-arm64.caliperapp.zip",      "sha256": "…" },
          "windows-x64-cu121":{ "url": "…", "sha256": "…" }
        }
      }]
    }
  },
  "packs": { "libtorch": { "2.5.1": { "macos-arm64": { "url": "…", "sha256": "…" } } } }
}
```

Publishing = a PR adding/updating an entry (CI validates schema, reachability, checksums, epoch sanity). Curation, when wanted, is PR review. Upgrade path to a hosted service exists but is not scheduled.

### 12.2 In-app Browse + sideloading

- **Browse tab** on the landing page: reads the cached index, filters by platform/epoch compatibility, Install/Update/Uninstall buttons; update badges when the index moves.
- **Sideload forever:** drop a `.caliperapp` into `applets/` — today's distribution story, preserved. Registries index; they never gatekeep.

### 12.3 Host distribution

- GitHub Releases per platform: `Caliper-0.6.0-macos-arm64.dmg`, `…-windows-x64-cu121.zip`, `…-windows-x64-cpu.zip` (matrix already sketched in README).
- macOS codesign + notarization, and Sparkle/WinSparkle-style auto-update: **Phase 5 polish**, schema'd for from day one (host knows its own version; releases are discoverable via the GitHub API).
- The `caliper` binary doubles as the CLI (`caliper new|dev|package|publish|install`) — the `code` command model; no second tool to install (Phase 4–5).

---

## 13. Developer Experience — three personas, three loops

### 13.1 You (platform developer)

- Two repos, strict direction: `caliper-sdk` ← `caliper`. The host pins SDK releases via CPM **like any consumer** — the discipline that stops the SDK from rotting back into "whatever the monorepo contains".
- Co-developing both: `cmake -DCPM_caliper-sdk_SOURCE=$HOME/dev/caliper-sdk` overrides the pin with your working copy; when green, tag SDK → bump host pin.
- ABI-touching changes gated by the **golden-applet wall**: host CI keeps `.caliperapp` artifacts built against each supported SDK release and must load all of them headlessly. Break the wall → make the change additive (`…v2`) or queue it for an epoch bump.
- `examples/hello` stays in-tree as the loader-test fixture (TDD substrate for the host).

### 13.2 An applet author (develops FOR the platform — never builds it)

```bash
# Day 0 — first applet in 10 minutes
caliper new ecg-anomaly && cd ecg-anomaly      # scaffold from template
cmake -B build && cmake --build build          # SDK + ui-stack fetched, pinned
caliper dev build/                             # released host, dev mode:
                                               #   loads dylib, watches, hot-reloads
# Day N — ship it
git tag v1.0.0 && git push --tags              # their CI: build matrix → bundles
                                               # → their GitHub Release
# PR to caliper-registry → appears in everyone's Browse tab
```

Scaffold contents: the 10-line CMakeLists (§10.1), `caliper.toml`, an `Applet` subclass rendering a hello plot, unit tests against the SDK's **fixture host** (a headless `CaliperHost` with fake services — applet logic is TDD-able without launching UI), and the CI workflow:

```yaml
# .github/workflows/applet.yml (from template)
jobs:
  build:
    strategy:
      matrix:
        include:
          - { os: macos-14,     triple: macos-arm64 }
          - { os: windows-2022, triple: windows-x64-cu121 }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - run: cmake -B build -DCMAKE_BUILD_TYPE=Release
      - run: cmake --build build --config Release
      - run: ctest --test-dir build --output-on-failure   # unit + conformance
      - run: cmake --build build --target caliper_package
      - uses: softprops/action-gh-release@v2
        if: startsWith(github.ref, 'refs/tags/')
        with: { files: build/*.caliperapp.zip }
```

**No step checks out Caliper.** That pipeline is the independence proof, and the template ships it working.

Debugging: attach LLDB/VS to the running `caliper` process — the dylib's symbols are the author's own build. Logs via `caliper.log.v1` appear in the host console; dev mode tails them.

**Hot-reload mechanics** (host dev mode): watch the build output (FSEvents/`ReadDirectoryChangesW`/inotify) → on change: `cleanup()` → `destroy()` → `dlclose` → **copy dylib to a unique temp path** → `dlopen` the copy → `create()`/`initialize()`. The copy sidesteps Windows file locks and macOS dyld image caching; `dlclose` not truly unloading (thread-locals, ObjC) costs a small leak per reload in dev — accepted. State preservation via `save_state`/`load_state` appended to `CaliperAppletAPI` later (guarded by `struct_size`).

### 13.3 An end user (researcher who just wants the tools)

Download DMG/zip → open Caliper → Browse tab → Install "RepNet Lab" → host downloads the bundle, verifies manifest + checksum, sees `libtorch ~2.5`, asks once ("1.9 GB, shared by all ML applets"), fetches the pack → card appears, click, **live training with saliency at frame rate**. Remove = right-click → Uninstall (or delete the folder). Update = badge on the card. Failure = a card with a reason in plain language, never a crash.

---

## 14. Compatibility & Versioning Policy

| Thing | Scheme | Breaks when | Cadence target |
|---|---|---|---|
| **ABI epoch** | integer | entry/`CaliperHost`/UI-pin changes | ≤ 1/year post-stabilization; host supports N, and N−1 where feasible |
| **SDK** | semver `0.x` → `1.x` | minor = additive (new services, sugar, viz) within an epoch | monthly-ish while building |
| **Services** | id suffix `.v1`, `.v2` | never — old ids keep working alongside new | as needed |
| **Host app** | semver | UI/features; never silently drops epochs/services (deprecation window ≥ 2 releases) | monthly-ish |
| **Applets** | semver, theirs | their business entirely | theirs |
| **Runtime packs** | upstream version + platform | n/a (side-by-side installs; one per process per session §11) | tracks upstream |

**Negotiation at load (in order, all pre-`dlopen`):** platform binary present → epoch supported → `min_host` satisfied → required services available → runtime packs resolvable (download prompt) → *then* `dlopen` → descriptor sanity (id/version/epoch agree with manifest) → `create/initialize`. First failure renders a reasoned card.

**Toolchain matrix (per epoch, documented in SDK):**

| Platform | Compiler | Stdlib/CRT | Notes |
|---|---|---|---|
| macos-arm64 | AppleClang ≥ 15 | libc++ (OS ABI) | relaxed in practice |
| windows-x64 | MSVC v143 (same major as host release) | **`/MD` dynamic CRT** | migration from `/MT` in Phase 1; DuckDB static-lib config re-verified under `/MD` |
| linux-x64 (future) | gcc ≥ 12 | libstdc++, `_GLIBCXX_USE_CXX11_ABI=1` | floor pinned at epoch |

---

## 15. Safety & Trust Model

- **Pre-flight beats post-mortem:** the manifest gate (§14) catches the entire class of "wrong build for this host" failures before any code from the bundle runs.
- **Crash quarantine (best-effort, honest):** every applet call is wrapped — SEH `__try/__except` on Windows; on POSIX a signal trampoline (SIGSEGV/SIGBUS/SIGFPE → `siglongjmp` out of the applet call) — faulting applet is torn down, marked quarantined with the fault summary on its card, host survives. *Documented honestly:* in-process isolation is containment, not a sandbox; memory may be corrupted after a fault, and the host offers a restart. Real isolation = Phase 6 out-of-process mode.
- **Frame watchdog:** `frame()` exceeding budget (e.g. 250 ms repeatedly) flags the applet's card with "blocking the UI thread — long work belongs in caliper.jobs" — making the platform's threading rule observable.
- **Supply chain:** sha256 on every artifact now; bundle signing + registry review policy when audience (c) arrives (the manifest already carries `authors`/`repository` for provenance).
- **Data sandboxing by convention:** per-applet `applet_data_dir`; services namespace storage by applet id. Convention, not enforcement — stated plainly in docs.

---

## 16. Testing Strategy (TDD throughout)

Per the standing rule — no production code without a failing test first — each phase's work is test-led:

| Layer | Tests (written first) | Harness |
|---|---|---|
| Manifest parser | golden + adversarial TOML (missing fields, bad semver, unknown epoch) | host unit tests (ctest) |
| Negotiation | epoch mismatch / missing service / unresolved pack → correct refusal reason | host unit tests with synthetic manifests |
| Loader v2 | descriptor resolution, lifecycle ordering, double-load, unload | `examples/hello` fixture applet built in-tree |
| Crash guard | fixture applet with `crash_on_frame` flag → quarantined, host lives | integration test (headless GL) |
| Hot reload | rebuild fixture mid-run → reload, lifecycle hooks called exactly once each | dev-mode integration test |
| Each service | contract tests against the host implementation (e.g. metrics: write 10k scalars, query back ordered; jobs: cancel honored ≤ 100 ms; bridge: tensor→texture pixel-exact vs CPU reference, run per backend) | service test suite in host CI |
| Pixel-space contract | fixture applet runs at `dpi_scale = 2.0`; fails if viewport/texture/size math conflates logical and physical pixels (§6a) | integration test, macOS CI (native Retina) |
| SDK conformance | the `caliper_conformance` checker itself (exports, descriptor, imconfig hash) | SDK repo CI, and runs in every applet's ctest |
| **Golden-applet wall** | host must load `.caliperapp` artifacts built against every supported SDK release | host CI, artifacts cached from SDK release CI |
| Template | template CI builds against the SDK **release artifact** (not source) and produces a loadable bundle | template repo CI; also smoke-run in host CI |
| Sugar | header-only → compile-time tests + fixture-host unit tests (services mocked) | SDK CI |

The fixture host (headless `CaliperHost` with fake services) ships **in the SDK** so applet authors inherit the same TDD capability for their own logic.

---

## 17. Migration Plan — six strangler-fig phases

Every phase ends with a shippable repo and a demo. No big bang; the v1 loader keeps working until Phase 1 completes.

### Phase 0 — SDK extraction in-tree (mechanics, no behavior change)
- Create `sdk/include/caliper/`, move `applet_api.h` → `abi_v1.h` (compat shim keeps old include path).
- CMake package: `caliper::sdk` target, install rules, `CaliperSDKConfig.cmake`, version file.
- Applets consume via `find_package(caliper-sdk)` against an **installed prefix** (`cmake --install`), even though it's the same repo.
- **Exit:** clean build where every applet compiles against the installed SDK, not `${CMAKE_SOURCE_DIR}` paths. The hot-potato ends here.

### Phase 1 — ABI epoch 2 (the contract)
- `abi.h` (descriptor + `get_service` + frame info), `ui_v1` with allocators, `log_v1`; sugar layer + `CALIPER_APPLET` macro; `caliper.toml` parsing (toml++); loader v2 + negotiation (manifest-first); friendly failure cards; crash guard + watchdog; Windows `/MD` migration (verify DuckDB config).
- Port `examples/hello` + all three applets to epoch 2 via sugar. Delete the v1 loader at phase end.
- **Exit:** all applets on epoch 2; negotiation/loader/crash tests green; Windows build on `/MD`.

### Phase 2 — extract the primitives (surgery on repnet_demo)
- `HostRenderer` abstraction + **Metal backend on macOS** (§5.4); the existing GL path becomes the fallback implementation behind the same interface. `jobs.v1` (from `train_engine` threading) → host jobs tray. `metrics.v1` (DuckDB store + Runs dashboard). `tensor.h` + `tensor_bridge.v1` built **once, natively over `HostRenderer`** (from `model_viz`): Metal aliasing/blit on MPS, GL staging on Windows until Phase 4's Vulkan work. `device.v1`. `data.v1` (Arrow out; from `dataset.cpp`). `artifacts.v1`. First `caliper::viz` components (SaliencyOverlay, KernelGrid).
- repnet_demo consumes services and **shrinks**; it is now the reference applet.
- `examples/ml_scope` — the small, copyable ML exemplar (sibling to Phase 1's `examples/signal_scope`): train a small CNN on **MNIST** (the ubiquitous benchmark — downloaded once into the applet's data dir, cached thereafter; born as a two-moons MLP in step 1, upgraded when metrics land) inside `jobs.v1`, stream loss/accuracy to `metrics.v1` (inheriting the Runs dashboard for free), mirror the first-layer conv kernels live via `tensor_bridge.v1`, negotiate device via `device.v1` — the USP on the benchmark everyone recognizes. Links the in-tree libtorch like any applet until runtime packs (Phase 4) take over; repnet-lab remains the full-scale reference, ml_scope is the "copy this to start an ML applet" answer.
- **Phase-internal sequencing (ratified 2026-07-01):** services are *designed* by extraction from repnet_demo's proven code but **first consumed by `examples/ml_scope`**, which is built incrementally as each service's acceptance vehicle — (1) `device.v1` + `jobs.v1`: the tiny MLP trains off the frame thread with cancel/progress; (2) `metrics.v1`: loss/accuracy stream in and the Runs dashboard comes free; (3) `HostRenderer` + Metal + `tensor.h` + `tensor_bridge.v1` incl. `alloc_shared`: the weight matrix becomes a live texture with zero CPU staging on MPS — the USP demo in miniature; (4) **opengllama sheds its grandfathered raw-GL heatmaps** onto `texture_from_tensor_mapped` — the bridge's first non-torch consumer (ggml/llama.cpp Metal buffers), the test that `CaliperTensor`'s DLPack alignment actually buys framework-agnosticism; note the hard dependency this settles: **the host's default macOS backend flips GL→Metal only after this step** (a raw-GL applet cannot run in a Metal-backed host, so the §6c grandfather clause expires here by necessity); (5) **the flagship applet `gpt_scope`** (ratified 2026-07-02, D16): a mini-GPT trained on TinyShakespeare — the nanoGPT-standard demo — born native on the full service stack (jobs-driven training, metrics streaming, live sampled text, and per-head attention heatmaps via tensor_bridge, which requires manual attention so the weights are exposed), replacing repnet_demo's migration as the generality proof. repnet_demo is **defunct** per the owner: it remains a legacy epoch-2/bridge-native example, never service-migrated, deletable at will. (6) `data.v1`/`artifacts.v1` become **demand-driven** — designed and frozen only when the flagship (or another real applet) actually consumes them; extract-don't-invent applies to freezing headers too. ml_scope is **CUDA-ready by construction** (speaks only `CaliperTensor` + `device.v1`, no raw graphics calls) but CUDA is *verified* only at Phase 4 with the Vulkan/`cudaImportExternalMemory` backend and real hardware — MPS is Phase 2's verified zero-copy path.
- **Exit:** the flagship `gpt_scope` runs entirely on public services with **no CPU staging in its attention/visualization path on macOS**; the Runs dashboard renders metrics from *any* applet; service contract tests green on Metal and GL backends.

### Phase 3 — independence (the milestone that defines the platform)
- Split `caliper-sdk` to its own repo (`git filter-repo` preserves history), tag `v0.1.0`, set up SDK CI (conformance + per-platform ui-stack builds).
- Create `caliper-applet-template` (scaffold, fixture-host tests, CI from §13.2).
- Move **circuitnet** out first (smallest, no torch): `git filter-repo --path applets/circuitnet` → new repo with full history → builds in its own CI against the SDK release artifact.
- Host CI gains the golden-applet wall.
- **Exit:** a `circuitnet.caliperapp` built by CI that never checked out Caliper, drag-dropped into a stock host, runs. **This is the moment Caliper becomes a platform.**

### Phase 4 — runtime packs, bundles, dev mode
- Bundle-aware discovery (`*.caliperapp` + manifest gate). Pack manager (resolve/download/verify/GC; `AddDllDirectory` / pre-`dlopen` wiring; one-torch-per-session policy). `caliper::torch_stub`. `caliper dev <dir>` (file-watch hot reload, log tail) and `caliper new` (template instantiation) as host subcommands. **Vulkan backend on Windows** + CUDA external-memory interop in the bridge; GL demoted to frozen compatibility fallback (GLEW→GLAD swap or outright deletion decided here).
- Move **repnet-lab** and **opengllama** to their own repos (history-preserving), shipping as bundles with `libtorch` pack dependency.
- **Exit:** fresh machine → install 50 MB host → install repnet-lab bundle → host fetches libtorch pack once → Training Lab live. `applets/` in this repo now contains only the `hello` fixture.

### Phase 5 — ecosystem
- `caliper-registry` repo + schema CI; Browse tab (install/update/uninstall, badges); `caliper install/publish`; docs site goes **public** — the wiki itself exists from Phase 1 (D15: MkDocs Material, docs-as-code, in-repo); this phase adds GitHub Pages publishing, per-release versioning (`mike`), the "first applet in 10 minutes" front door, the ABI hygiene guide, and the API reference generated from SDK headers via `mkdocs-cxxdox` (libclang-based MkDocs plugin — adopted once the Phase-2 service catalog makes generated reference worth it); macOS codesign/notarize + simple update check; delete `applets/` from the monorepo entirely — the `hello` fixture moves to `examples/hello`.
- **Exit criterion (the real one):** someone who isn't you ships an applet end-to-end — scaffold to registry PR — without a question that the docs couldn't answer.

### Phase 6 — later, demand-driven
- **Out-of-process applet host** for untrusted binaries (own GL context, composited; or software-isolated with shared-memory tensor transport) — unlocks audience (c) safely.
- **Scripting bindings** (Python first: pybind over the sugar; the fixture host enables notebook-driven applet prototyping).
- **`libcaliper` / second host: Compass** — the platform core (loader, negotiation, host-neutral services, pack manager, registry client) extracted as an embeddable library. First consumer: **Compass**, the interface-heavy sibling (native wxWidgets chrome — AUI docking, property grids, document-style UI; the Adobe-shaped face to Caliper's realtime face). Both hosts share the applet contract, the seven host-neutral services (§7 design rule), runtime packs, and the registry; they differ only in `ui.vN` and rendering — Caliper via `HostRenderer` (Metal/Vulkan), Compass via wx's native backends (Direct2D/CoreGraphics/Cairo). Neither leaks into the contract.
- Linux as a first-class platform triple; bundle signing; hosted registry if PR volume demands it.

### Sequencing rationale

Contract before extraction (Phases 1→2) so services land behind a stable boundary once. Independence (3) before packaging polish (4) because the golden-applet wall and template CI are what keep every later change honest. circuitnet pilots the migration because it's the smallest surface (no torch); repnet-lab graduates last among the three because it's the service extraction donor.

---

## 18. Decision Log

| # | Decision | Status | Rationale / trade accepted |
|---|---|---|---|
| D1 | In-process C ABI + C++ sugar (not C++ interfaces, not IPC) | **Ratified** (existing code + this plan) | Zero-copy same-frame-loop USP; longevity. IPC deferred to Phase 6 for trust, not foundation. |
| D2 | Host context = service registry (`get_service`), CLAP-style | Proposed | Growth without ABI breaks for years. |
| D3 | Torch/DuckDB types never cross the ABI; `CaliperTensor` (DLPack-aligned) + Arrow C streams are the interchange | Proposed | Survives version skew; trivial conversions at the edge. |
| D4 | UI-stack pin defines the ABI epoch; applets write raw ImGui/ImPlot | Proposed | DX superpower kept; cost = applets rebuild on epoch bump (rare, CI-flagged). |
| D5 | Runtime packs host-managed; **one libtorch per process per session** | Proposed | Kills 2 GB-per-applet; honest about the multi-version limit (Phase 6 solves fully). |
| D6 | Registry = git repo (Homebrew-tap model); sideloading always works | Proposed | Zero infrastructure; curation = PR review; upgrade path exists. |
| D7 | Windows moves to `/MD`; `ui.v1` hands over ImGui allocators | Proposed | Fixes the latent static-CRT/DLL-heap crash class. |
| D8 | Applet repos migrate with history (`git filter-repo`), not from scratch | Proposed | "Own lives and histories" includes the history already written. |
| D9 | CLI = subcommands of the host binary (`caliper new/dev/package/install/publish`) | Proposed | One artifact to install; the `code` CLI model. |
| D10 | SDK license: MIT (host may stay separate) | **Decide by Phase 3** | Ecosystem needs a permissive SDK; MIT matches ImGui/ImPlot neighborhood. Apache-2.0 acceptable if patent grant desired. |
| D11 | Host ships without libtorch (bridge uses raw CUDA/Metal; metrics use embedded DuckDB) | Proposed | 50 MB host; packs on demand. |
| D12 | Audience: (b) source-building collaborators now, contracts sized for (c) later | Proposed (assumption) | Stated in §3; revisit when a real (c) user appears. |
| D13 | Renderer-agnostic ABI from epoch 2; native backends as the target — **Metal (macOS) + Vulkan (Windows)** primary, OpenGL 3.3 frozen fallback; textures cross as opaque `CaliperTextureId` | Proposed | The USP demands GPU-resident pixels; GL is deprecated on macOS and cannot touch MPS memory (today every Mac texture takes a CPU round-trip). Decided while zero external applets exist, so the renderer stays host-internal forever — no epoch bump, no applet rebuilds. GLEW dies with the fallback refactor (GLAD 3.3-core loader, §5.4). Live evidence for the GL dead end: sibling project Compass stayed on "cross-platform GL" and is stranded between 2.1 fixed-function and macOS's capped 4.1 core, with per-platform `#ifdef` include paths. |
| D14 | The bridge *allocates* texture-backed shared tensors (`alloc_shared`), not just mirrors existing ones | Proposed | Upgrades "fast device copy" to literal zero-copy for live weights/saliency; applets adopt it with one `torch::from_blob`. |
| D18 | The demand-driven services (D16) `artifacts.v1` + `data.v1` are built in Phase 2F′, their first honest consumer being **EmbedScope** — a data-driven ML exemplar whose 3D embedding projector puts ImPlot3D (renderer-agnostic 3D) front and center. artifacts.v1 is load-bearing there (model reload); data.v1 queries the learned embedding table (centroids/kNN/misclassified). 3D-via-raw-GL was rejected: it violates §6c and is GL-backend-only, contradicting the Metal default (D13). | **Ratified** (2026-07-02) | Demand materialized (checkpoint reload, tabular embeddings), so freezing the two headers no longer violates extract-don't-invent; ImPlot3D is the portable 3D answer. |
| D16 | Flagship pivot: repnet_demo is defunct and will not migrate onto services; the Phase-2 generality proof and long-term reference applet is **`gpt_scope`** — a mini-GPT on TinyShakespeare (the nanoGPT-standard demo), born native on the full stack; `data.v1`/`artifacts.v1` freeze only on real demand | **Ratified** (2026-07-02) | A greenfield applet the owner wants tests the platform as an author experiences it — a better generality proof than retrofitting dead code; unconsumed frozen headers would violate extract-don't-invent. |
| D17 | UI-stack pin switches to ImGui's **docking branch** (same 1.92.x line) for desktop-grade docked/tiled layouts (user directive). Epoch REMAINS 2: no SDK release or external consumer exists, so the pin amends pre-publication; from Phase 3 onward any pin change is a formal epoch bump per §9. Host enables docking + a viewport dockspace with a default tiled layout; applets need no code changes (their windows dock automatically; layouts persist in imgui.ini) | **Ratified** (2026-07-02) | Floating windows read as a demo; docking is the ImGui-native answer to a real desktop layout. Doing it before first release is free; after, it costs an epoch. |
| D15 | Documentation is docs-as-code from Phase 1: MkDocs Material wiki in-repo (`docs/wiki/`, Diátaxis layout), doc pages updated in the same commit as the change, `mkdocs build --strict` gate, reference pages embed the real headers/manifests via snippets (`check_paths` — moved files fail the build). API reference generated from headers via `mkdocs-cxxdox` (libclang) adopted at Phase 2; publishing (Pages) + versioning (`mike`) at Phase 5 | **Ratified** (2026-07-01) | Docs written retroactively rot; same-commit + embed-don't-paste + strict link checks keep the wiki true mechanically. cxxdox is alpha (kfrlib, v0.1.x) but additive to the same MkDocs site and trivially droppable — the snippet-embedded pages remain the fallback. |

---

## 19. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ImGui across DLL w/ static CRT crashes on Windows | High (latent today) | High | D7: `/MD` + allocator handoff (Phase 1); conformance checks imconfig hash. |
| `dlclose` doesn't truly unload (TLS, ObjC) → hot-reload staleness | Medium | Low (dev-only) | Copy-to-unique-temp per reload; accept small dev-mode leak; full teardown semantics. |
| Two applets demand incompatible libtorch | Medium | Medium | One-pack-per-session policy with clear refusal card (D5); Phase 6 isolation as the real fix. |
| GPU interop quirks (Vulkan external-memory alignment/tiling, Metal buffer-aliasing limits, VMs/remote/hybrid GPUs) | Medium | Medium | Bridge degrades per-tensor: alias → device blit → CPU-staged upload (GL fallback); contract tests pin behavior per backend; `device.v1` exposes what was negotiated. |
| Metal/Vulkan backend work displaces platform phases | Medium | Medium | Renderer is host-internal (no ABI coupling, §5.4) — it can slip without blocking any other phase; GL fallback keeps everything shippable; Metal lands with Phase 2 (primary dev machine), Vulkan deferred to Phase 4. |
| `/MD` migration destabilizes DuckDB/static deps on Windows | Medium | Medium | Phase 1 task with its own test gate; fall back = keep `/MT` for host-internal DuckDB while plugins use `/MD` (allocators still fix ImGui). |
| ABI surface creep (services accrete fields/semantics) | Medium | High | Constitution §6b; conformance suite; golden-applet wall; service review checklist in SDK CONTRIBUTING. |
| Epoch bumps strand dormant applets | Low | Medium | N/N−1 host support window; registry surfaces "needs rebuild" state; rebuild = re-tag against bumped SDK. |
| Registry supply-chain abuse (when audience (c) arrives) | Low now | High later | sha256 now; signing + review policy gated to Phase 6 audience change. |
| Solo-maintainer bandwidth: platform work stalls applet work | Medium | Medium | Strangler phases each ship value; repnet_demo keeps working throughout; Phase 2 *reduces* its code. |

---

## 20. Glossary

| Term | Meaning |
|---|---|
| **ABI epoch** | Integer version of the frozen contract (`abi.h` + UI-stack pin). Mismatch = refuse before load. |
| **Service** | A named, versioned, immutable C function table obtained via `get_service("caliper.x.vN")`. |
| **Sugar** | The header-only C++ layer (`caliper.hpp`) over the C ABI. Optional by design. |
| **UI stack** | The SDK-pinned imgui/implot/implot3d/ImGuiFileDialog set whose exact commits define an epoch. |
| **Bundle** | `<name>.caliperapp/` directory: manifest + per-platform binaries + assets. The unit of distribution. |
| **Manifest** | `caliper.toml` — identity, compatibility, required services, runtime packs. Checked pre-`dlopen`. |
| **Runtime pack** | Host-managed shared heavy dependency (e.g. libtorch) resolved from manifest declarations. |
| **Registry** | `caliper-registry` git repo: `index.json` of applets/packs → release artifacts. Publishing = PR. |
| **Golden-applet wall** | Host CI gate: bundles built against every supported SDK release must still load. |
| **Fixture host** | Headless fake `CaliperHost` shipped in the SDK for TDD of applets and sugar. |
| **HostRenderer** | Host-internal rendering interface; Metal/Vulkan/GL implementations live behind the renderer-agnostic ABI (§5.4). |
| **CaliperTextureId** | Opaque 64-bit texture handle from the bridge; castable to `ImTextureID`, never a raw graphics handle. |

---

*Companion documents: `APPLETS.md` (current applet how-to; superseded progressively from Phase 1), `docs/applet-architecture.md` (historical first draft of the plugin split), `docs/applet-dependency-packaging.md` (early sketch of the dependency split — superseded by §11), `../compass/PLATFORM.md` (sibling spec: Compass as the interface-heavy second host — static-binary principle, UI-as-data ABI, phases C0–C4 gated on Phases 3 and 6 here).*
