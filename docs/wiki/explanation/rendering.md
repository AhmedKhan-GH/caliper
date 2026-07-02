# Rendering (the HostRenderer story)

## Why the ABI never names a graphics API

Caliper's USP is GPU-resident visualization, so the renderer must speak the API
the tensors actually live in. OpenGL cannot: it is deprecated on macOS (capped at
4.1) with no path to MPS memory, so every Mac tensor would take a CPU round-trip
on its way to becoming a texture — a copy in the hottest loop the platform owns.
The answer (PLATFORM.md §5.4) is to keep graphics **out of the contract
entirely**: applets render only through ImGui/ImPlot, textures cross the boundary
as an opaque [`CaliperTextureId`](../reference/services/tensor-bridge-v1.md), and
the host keeps an id→backend-handle table so the id is *never* a raw GL/Metal
handle. Because the contract hides the backend, swapping it needs **no epoch bump
and no applet rebuild**.

The seam is `HostRenderer` (`src/host/renderer/host_renderer.h`), a host-internal
interface the ABI never sees. It is validated by prior art: wxWidgets'
`wxGraphicsContext` converged on the same answer — one abstract drawing API over
*native* backends (Direct2D / CoreGraphics / Cairo), not one cross-platform GPU
API (PLATFORM.md §5.4 discussion).

## Backend status

| Backend | Selection | Status | Notes |
|---------|-----------|--------|-------|
| **Metal** | default (macOS) | **Shipping default** | Full app parity; `tensor_bridge.v1` colormaps MPS buffers on-GPU (zero CPU staging), §16 pixel-exact (C5). Apple-only translation unit. |
| **OpenGL 3.3 core** | `CALIPER_RENDERER=gl` | **Frozen fallback** | The retained escape hatch; `tensor_bridge.v1` CPU-stages onto it. The only backend off Apple. |

`main.cpp` does the env-driven selection between two factories
(`make_renderer("gl")` and `make_metal_renderer()`): on macOS the default is
Metal and `CALIPER_RENDERER=gl` selects the frozen GL fallback; if the Metal
backend fails to init it still falls back to GL.

### Why Metal is now the default

Metal became the macOS default once the **opengllama migration cleared the flip
gate**. The ratified Phase-2 sequencing (PLATFORM.md §17, ratified 2026-07-01)
made this a hard dependency: opengllama shed its grandfathered raw-GL heatmaps
onto `texture_from_tensor_mapped` (the bridge's first non-torch consumer —
ggml/llama.cpp Metal buffers), leaving zero raw-GL applets. A raw-GL applet
cannot run in a Metal-backed host, so the §6c grandfather clause expired there by
necessity, and **the host's default macOS backend flipped GL→Metal**. Both
backends stay honest under the same §16 pixel-exact test matrix run per backend.

One honest gap: the landing-page 3D background (`IntroScreen`) is still raw GL,
so it is absent on Metal — you get the plain app shell instead of the animated
backdrop. The cards, launch flow, and dashboards are unaffected and behave
identically on both backends. Restoring the 3D landing on Metal is tracked as the
intro 2D-migration follow-up in the Phase-2 plan notes; until then it renders
only under `CALIPER_RENDERER=gl`.

## The pixel-space contract, recapped

Applets never issue graphics calls (§6c). They describe *what* to draw in ImGui
terms and hand *tensors* to the bridge; the host owns the window, the contexts,
and every actual GPU call. A tensor becomes a texture through
[`caliper.tensor_bridge.v1`](../reference/services/tensor-bridge-v1.md) — the one
place the backend difference (device compute/blit on Metal vs CPU-staged on GL)
lives, and it is invisible to the applet, which runs identical code on both.
