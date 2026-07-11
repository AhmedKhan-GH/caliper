# `libcaliper` — the embeddable platform core (R4), first consumer: Compass

**Date:** 2026-07-11
**Status:** DESIGN — this is the R4 strategic decision doc ROADMAP §7 deferred
("platform call, not geometry's"; intent at PLATFORM.md Phase 6). Opening it is
the owner's call, made 2026-07-11. Nothing here is scheduled for execution;
this doc exists so the decision is made ONCE, with the trade-offs recorded,
before any extraction begins.
**Authority:** PLATFORM.md Phase 6 (`libcaliper` / second host: Compass),
§7 host-neutral service design rule, D13 (native backends; the Compass-on-GL
cautionary tale), D5 (one libtorch per process), D1 (in-process C ABI).
GEOMETRY.md §11 row R4. The graphics ladder R0–R3 is complete and byte-exact
on both ecosystems (2026-07-11) — R4 is the first rung that is NOT a
graphics capability: it changes *where the instrument can live*, not what it
can draw.

---

## 1. One paragraph

Extract the platform core — applet loader + negotiation, the service
registry, the host-neutral services, `HostRenderer` (Metal/Vulkan/GL) with
the tensor bridge and the geometry ladder — out of the `caliper` executable
into **`libcaliper`**, an embeddable library behind a small C ABI, so a
second host can vend the SAME applet contract without owning any of the
machinery. The first consumer is **Compass**, the interface-heavy sibling
(native wxWidgets chrome: AUI docking, property grids, document-style UI —
the Adobe-shaped face to Caliper's realtime face). Both hosts share the
applet contract, the host-neutral services, and (eventually) packs/registry;
they differ only in chrome. Neither difference leaks into the applet
contract.

## 2. The question the owner asked: cross-platform or write-once?

**Answer: write-once API, cross-platform internals — and the cross-platform
part is already built and paid for.** Precisely:

- **The applet contract is already write-once.** One set of C headers; an
  applet written against them runs under any host that embeds libcaliper,
  on any OS (compiled per-platform, authored once). Nothing changes here.
- **libcaliper itself is ONE codebase that is cross-platform by
  construction** — exactly as the host internals are today: the
  platform-specific halves (Metal vs Vulkan renderer, MPS unified-memory vs
  CUDA external-memory import, dylib vs DLL loading) already exist behind
  `HostRenderer` and the bridge, byte-exact-verified on both ecosystems.
  Extraction moves them; it does not multiply them.
- **What "write once" must NOT mean** (the D13 lesson, learned on Compass
  itself): one *rendering backend* everywhere. Compass is today stranded on
  "cross-platform GL" between 2.1 fixed-function and macOS's capped 4.1
  core with per-platform `#ifdef`s — that is the strategy libcaliper
  *rescues* it from. The portable thing is the API surface; the native
  thing is the backend per OS. That split is the whole point.
- **The genuinely NEW cross-platform work is small and singular:** the
  embedding seam (§4) — designed once, with thin per-OS glue where the
  library meets a native window handle (NSView*/HWND, later X11/Wayland).
  Everything else rides code that already ships.

## 3. What moves, what stays, what is deferred

| Into `libcaliper` v0 | Stays in `caliper` (the host) | Deferred out of v0 |
|---|---|---|
| Applet loader + manifest negotiation | ImGui docking shell / chrome | Pack manager + registry client (Compass sideloads in v0 — dragging Phases 4–5 in would gate R4 on packaging work it doesn't need) |
| Service registry (`get_service`) + the host-neutral services: log, jobs, device, metrics, artifacts, data | The `caliper` CLI subcommands | Out-of-process applet isolation (own Phase-6 item) |
| `tensor_bridge` v1–v1.2 + `geometry` v1–v1_3 + `HostRenderer` (Metal/Vulkan/GL fallback) | Host-level window/event loop | Scripting bindings (own Phase-6 item) |
| The applet-canvas ImGui context + frame pump (§4.3) | `ui.vN` *chrome* specifics per host | Linux triple (first-class later; nothing here precludes it) |
| Crash guard + watchdog around applet calls | | |

Rationale for the one contentious inclusion: `HostRenderer` + bridge +
geometry go INTO libcaliper because the zero-copy claim is the product. A
"core" without the renderer would hand Compass the loader and services but
strand its applets without pixels — recreating the D13 dead end one layer
up.

## 4. The embedding seam (the only new API this creates)

### 4.1 Shape: a small C ABI, mirroring the applet-contract discipline

C, not C++: the same longevity/version-skew reasoning as D1. A host binary
built years apart from libcaliper must still embed it. C++ sugar for host
authors ships alongside (as `caliper.hpp` does for applets).

```c
/* sketch — names illustrative, the execution spec pins them */
CaliperCore*  caliper_core_create(const CaliperCoreDesc*);  /* renderer pick,
                                     data dirs, log sink, device policy */
bool          caliper_core_attach_canvas(CaliperCore*, void* native_view,
                                         const CaliperCanvasDesc*);
void          caliper_core_frame(CaliperCore*);   /* pump: applets + render */
void          caliper_core_event(CaliperCore*, const CaliperInputEvent*);
/* loader: enumerate/load/unload applets; vends the same service registry
   applets already see */
void          caliper_core_shutdown(CaliperCore*);
```

### 4.2 The load-bearing design decision: who renders the applet canvas

**Decision (this doc): libcaliper owns the applet canvas end-to-end.** The
embedding host supplies a native child view (NSView*/HWND) per canvas;
libcaliper runs the ImGui context, `HostRenderer`, bridge, and geometry
inside it. Compass's wx chrome (AUI docking, property grids, menus) wraps
AROUND those canvases; wx's own D2D/CoreGraphics/Cairo rendering paints the
chrome only, never applet pixels.

Why not the alternative (Compass renders applet UI via wx): applets write
raw ImGui (D4) and their images are `CaliperTextureId`s in the renderer's
table — re-hosting that on wx graphics would either forfeit zero-copy (CPU
readback per frame: the rejected render-to-tensor sync contract through the
back door) or demand a wx `HostRenderer` backend (a fourth backend, for
chrome-grade graphics — all cost, no capability). "They differ only in
`ui.vN` and rendering" (PLATFORM.md) is honored at the CHROME layer; the
canvas is the contract and travels with the core.

Consequence worth stating plainly: **Compass on Windows renders applet
canvases via Vulkan, on macOS via Metal — because it embeds the same
libcaliper.** Its wx chrome remains native per OS. Two "native" layers, one
seam between them.

### 4.3 Lifecycle & threading (pinned constraints, from shipped behavior)

- **Frame pump:** the embedding host calls `caliper_core_frame` from ITS
  event loop (wx idle/timer for Compass). libcaliper never owns the process
  loop — that is the difference between a library and a host.
- **Frame-thread discipline** carries over verbatim: applet torch work on
  jobs threads, draw from snapshots; the drain-before-publish geometry
  contract (geometry_v1.h) binds regardless of who embeds.
- **One libtorch per process** (D5) now binds the EMBEDDER: Compass must
  not link its own torch. The core owns device/pack policy.
- **One ImGui context per canvas**, owned by libcaliper; the embedder never
  touches ImGui state (allocator handoff stays internal).
- **Crash guard:** applet faults are contained by the core's existing
  guard; the embedder gets a callback, not a crash — a host that dies with
  its applets is not an embeddable library.

## 5. What Compass is FOR (so scope stays honest)

Compass is the demand-driven proof that the platform core is genuinely
host-neutral — the §7 design rule made executable. Its product shape
(per PLATFORM.md): document-style, property-grid-heavy, "Adobe-shaped" —
the slow-thinking face (inspect, arrange, annotate, author) to Caliper's
realtime face (train, watch, steer). Concretely worth building only when a
real workflow demands that chrome; this spec deliberately does NOT invent
one. **Gate (extract-don't-invent, applied to hosts):** Compass work starts
when a named workflow exists that Caliper's docking shell serves badly.
Until then, phases L1–L2 below still pay for themselves inside Caliper.

## 6. Phasing (each phase ships value alone; stop after any)

- **L1 — self-host extraction.** Create the `libcaliper` build target
  in-tree; `caliper` the executable becomes its first embedder. ZERO
  behavior change: all 8 suites green, the gfx byte-exact matrix untouched,
  every applet runs unmodified. This is a pure seam-cutting phase — it
  proves the boundary exists and leaves the tree better even if L2/L3 never
  happen.
- **L2 — the embed ABI + a second in-tree embedder.** Pin the §4.1 C ABI;
  generalize the test fixture host into `examples/embed_host` (a ~200-line
  host: create core, attach canvas to a bare native window, pump frames,
  load one applet). Acceptance: mesh_scope/instance_scope run zero-copy
  under embed_host on both ecosystems — run-proven with the same honest
  provenance lines, byte-exact rows still green.
- **L3 — Compass itself.** Separate repo, wx chrome, embeds the L2 ABI.
  Gated on §5's named-workflow rule. Acceptance: the same applet binary
  (unmodified, same `.caliperapp`) runs under Caliper AND Compass on both
  OSes; zero-copy provenance in both; the golden-applet wall passes against
  both hosts.

## 7. Verification discipline (inherited, restated for hosts)

The byte-exact bar now gains a host axis: the gfx matrix must produce the
SAME bytes under `caliper` and `embed_host` (same backend, same machine) —
a rendering seam that shifts pixels when re-hosted is a broken extraction.
The honest-ladder and pixels-untouched-on-refusal invariants bind in every
embedder. No checkbox without artifacts, per house rule.

## 8. Out of scope for R4 (stated so nobody relitigates)

Render-to-tensor, applet-supplied shaders, PBR (permanent invariants);
out-of-process isolation, Python bindings, Linux triple, packs/registry in
the embed ABI (each its own later item); any change to the applet-facing
ABI — R4 must be invisible to every shipped applet, or the extraction is
wrong.

## Invariants (hold forever)

- Data flows tensors → pixels → ImGui, one way — in every host.
- The applet contract is written once and owes nothing to any host's chrome.
- Portable API, native backends (D13). "Cross-platform rendering backend"
  remains the rejected strategy that stranded Compass v0.
- The core never owns the process event loop; the embedder never touches
  ImGui or the renderer directly.
