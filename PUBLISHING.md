# PUBLISHING.md — the research-publishing track

The authority document for making Caliper serve **research papers**, not
only live instrumentation. Sibling of `GEOMETRY.md` (what the instrument
draws) and `PLATFORM.md` (where the instrument lives): this track is *how
the instrument's output leaves the machine* — as reproducible figures,
supplement videos, Python-driven sessions, and reviewer-runnable demos.
Dated decision record: `docs/superpowers/specs/
2026-07-12-research-publishing-track-design.md` (this file supersedes it as
the living authority; the dated spec stays as the decision provenance).

**The gap this track closes, stated honestly (2026-07-12):** the instrument
is production-grade today for live, in-process, C++/libtorch work on owned
machines — byte-exact on both GPU ecosystems, honest degradation
everywhere. A research paper additionally needs static artifacts, a path
from Python training loops, and a demo a reviewer can run. Three rungs.

---

## 1. The target workflow (the definition of done)

> Train in Python or C++ → watch the run live in Caliper → export the
> paper's figures and supplement video FROM the same instrument,
> deterministic and provenance-stamped → attach a one-download bundle a
> reviewer can run.

Every rung serves a clause of that sentence. Nothing else is on this track.

## 2. Status (checkbox discipline: verified by artifacts only)

| Rung | What | Status | Exemplar / proof |
|---|---|---|---|
| **E — Export** | `caliper.export.v1`: pixel-exact view PNGs at requested resolution, frame sequences, provenance sidecar | DESIGNED — next to execute | mesh_scope/TwinScope export affordance; the landing page's first real demo clip |
| **P — Python** | pybind wheel over the embed ABI + DLPack zero-copy ingestion; torch-free libcaliper variant as prerequisite | DESIGNED — gated on E; verify torch-MPS DLPack in week one | notebook trains PyTorch model, live Caliper view, `export.view_png` figure |
| **B — Bundle** | the reviewer supplement: one signed download, host+applet+assets(+pack) | DESIGNED — demand-driven, builds against the FIRST real submission | a machine that never built Caliper runs the paper demo |

## 3. Rung E — export (figures + video)

**The boundary decision:** Caliper exports **pixel-exact view images and
frame sequences plus a provenance sidecar** — and deliberately does NOT
compose figures. Axes, captions, typography, subfigure layout belong to the
paper toolchain (matplotlib/TikZ), which consumes Caliper's PNGs. The
instrument's claim — *the pixels are the tensors* — survives export
bit-for-bit; a typography engine adds scope with zero claim.

- **Service `caliper.export.v1`** (additive, house pattern):
  `view_png(view, path, w, h)` re-renders the view OFFSCREEN at the
  requested resolution (4K figures from a live session — the geometry
  pipeline is resolution-independent);
  `begin_sequence/frame/end_sequence` writes numbered PNGs (video assembly
  stays outside — one documented ffmpeg line, no in-process codecs).
- **The sidecar** (what makes it research-grade): every export writes JSON —
  caliper version+commit, applet id, timestamp, camera, colormap+range,
  view size, applet-supplied state (step, seed, hparams). A figure without
  its sidecar is a screenshot; Caliper exports figures.
- **Determinism contract:** same tensors + camera + colormap → byte-identical
  PNG on the same backend (the byte-exact discipline extended one step).
- **Already exists:** the offscreen render + tightly-packed RGBA8 readback
  (the embed `CANVAS_OFFSCREEN` path, run-proven on BOTH backends);
  deterministic re-render (the entire byte-exact test matrix is built on it).
  New work is the service veneer + PNG encode (stb_image_write) + sugar +
  the exemplar affordance.
- **Honest split:** ImPlot chart panes (pulse_scope, metrics dashboards) are
  ImGui chrome, NOT exportable views — their paper path is the one that
  already ships: SQL via `metrics.v1_1` → matplotlib.

## 4. Rung P — Python interop (the field-unlock)

**The shape decision (load-bearing):** NOT "libcaliper links torch and
Python calls it" — importing PyTorch would put two libtorches in one
process (D5 violation, symbol chaos). Instead:

- **The `caliper` wheel = pybind11 over the existing embed C ABI**
  (`embed.h` v1.1: create/canvas/frame/get_service — already shipped, both
  platforms) **+ DLPack ingestion**. `CaliperTensor` is DLPack-aligned BY
  DESIGN (D3), so a PyTorch/JAX/CuPy tensor crosses with zero copies from
  Python's own GPU memory. **Python's torch is THE torch** — D5 satisfied
  with the embedder owning it.
- **Prerequisite:** the **torch-free libcaliper link variant** (previously a
  debt item; now demand-justified). The core (bridge/geometry/renderer/
  services) consumes CaliperTensor, not torch; only the applet-side torch
  adapter and torch-linking applets need it, and they keep it as plugins.
- **v0 scope, held hard:** Python creates the core, attaches a canvas,
  feeds tensors via DLPack, drives draws, reads services (metrics/feed/
  export), pumps frames from its own loop. Python does NOT author applets —
  applets stay C-ABI plugins. A notebook drives the instrument; it does not
  become one. (Keeps the seam the size of embed.h, not the size of ImGui.)
- **Pinned risks:** torch-MPS DLPack support (verify week one — it gates
  the zero-copy claim on the primary box; CUDA is mature); GIL vs frame
  pump (frame() is short; never mutate handed tensors from another Python
  thread — the drain-before-publish contract crosses the language);
  wheels via cibuildwheel, macOS first, Windows second (house rhythm).

## 5. Rung B — the reviewer bundle (packaging, scoped to papers)

**The scope decision:** pull forward ONLY the supplement slice of
PLATFORM.md Phases 4–5 — one artifact a reviewer runs — and leave the full
packaging program (SDK split, template CI, packs, registry) where it is.

- `caliper bundle <applet-id>` (host subcommand, D9 pattern) assembles:
  host + applet(s) + assets + the runtime pack if the applet links torch.
- macOS signed+notarized .dmg (pulls exactly the Phase-5 codesign item
  forward, nothing else); Windows .zip with the DLL closure (the embed_host
  DLL-copy step generalized).
- The bundle README carries the honesty registers verbatim — reviewers get
  the same no-overclaim discipline as the docs.
- **Acceptance IS the rung:** a machine that never built Caliper downloads
  one file and runs the paper demo live. Built against the first actual
  submission — extract-don't-invent applied to packaging.

## 6. Sequencing (decided)

**E → P → B.** Export first: smallest (machinery exists), immediate paper
value, and both later rungs consume it (the Python exemplar exports; the
bundle demos it). Python second: the field-unlock, prerequisite now
justified. Bundle last, against a real submission. Each rung: its own
execution spec → the house SDD pipeline (foreground subagents, review
gates, per-platform honesty registers).

## 7. What is NOT on this track (decided, don't relitigate)

Figure typography/composition; in-process video encoding; Python-authored
applets; the registry/browse ecosystem; Linux (its own later item); any new
render path (export re-renders EXISTING views); any ABI break (everything
here is additive services + a wheel over the existing embed seam).

## Invariants (hold forever)

- **Export is a terminal sink.** Pixels leave; nothing returns to compute.
  Render-to-tensor stays rejected — this track does not reopen it.
- **One torch per process, whoever owns it.** The embedder may own it
  (rung P's refinement of D5); there are never two.
- **A figure without its sidecar is a screenshot.** Provenance is attached
  to every export, always.
- **Nothing ships claimed beyond its verified platform.** Wheels and
  bundles inherit the per-platform honesty register verbatim.
- Data flows tensors → pixels → ImGui/PNG, one way. The paper is downstream
  of the instrument, never inside it.
