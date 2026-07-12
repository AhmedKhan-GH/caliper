# The research-publishing track — export, Python, packaging (three rungs)

**Date:** 2026-07-12
**Status:** DESIGN — the strategic doc for making Caliper serve *research
papers*, not just live instrumentation. Opened by the owner's question
("can we create production-grade ML visualization projects for research
papers?") and the honest gap analysis: the instrument is production-grade
for live, in-process, C++/libtorch work on owned machines; a paper
additionally needs (1) static figure/video artifacts, (2) a path from
Python training loops, (3) a way for a reviewer to run a demo. Three rungs,
sequenced §5. Each rung gets its own execution spec before code; this doc
makes the DECISIONS once.
**Authority:** the platform invariants (tensors → pixels → ImGui one-way;
honest degradation; additive frozen ABIs), D3 (CaliperTensor is
DLPack-aligned; Arrow across the ABI), D5 (one libtorch per process), D11
(host ships without libtorch — direction), D13 (native backends),
PLATFORM.md Phases 3–6 (packaging + scripting intent), the debt registry
(the torch-free libcaliper variant — which rung 2 converts from debt into a
prerequisite with a real consumer).

---

## 0. The target workflow (what "done" means)

A researcher: trains in Python or C++ → watches the run live in Caliper →
exports the paper's figures and supplement video FROM the same instrument,
deterministic and provenance-stamped → attaches a one-download demo bundle
a reviewer can run. Every rung below serves a step of that sentence; nothing
else is in scope.

---

## 1. Rung E — the export path (figures + video)

### 1.1 The honest boundary (the load-bearing decision)

Caliper exports **pixel-exact view images and frame sequences, plus a
provenance sidecar** — and deliberately does NOT become a figure-composition
engine. Axes, captions, publication typography, subfigure layout belong to
the paper toolchain (matplotlib/TikZ/Illustrator), which consumes Caliper's
PNGs + sidecars. Rationale: the instrument's claim is *the pixels are the
tensors* — that claim survives export bit-for-bit; a typography engine adds
a million lines of scope with zero claim. (Readback-for-export does NOT
violate the render-to-tensor invariant: nothing feeds back into compute;
it is a terminal sink, exactly like the existing test readbacks and the
embed OFFSCREEN canvas.)

### 1.2 What already exists (most of the machinery)

- Offscreen rendering + tightly-packed RGBA8 readback: the embed
  `CANVAS_OFFSCREEN` path (`canvas_read_pixels`), run-proven on BOTH
  backends (Metal L2, Vulkan Windows embed pass).
- Deterministic re-render: geometry draws are pure functions of
  (tensors, camera, colormap, params) — the byte-exact test matrix is
  literally built on this.
- `debug_readback_rgba8` per-texture (test-only today).

### 1.3 The design

- **New service `caliper.export.v1`** (additive, house pattern):
  - `view_png(view_id, path, w, h, scale)` — re-render the view offscreen
    at the REQUESTED resolution (not the screen's) and write PNG. High-DPI
    figures = large w/h; the geometry pipeline is resolution-independent.
  - `begin_sequence(dir, w, h) / frame() / end_sequence()` — numbered PNG
    frames for video; assembly to mp4/GIF stays OUTSIDE (ffmpeg — document
    the one-liner, do not link an encoder; codecs are a swamp with zero
    claim).
  - Every export writes a **sidecar JSON**: caliper version+commit, applet
    id, timestamp, camera, colormap+range, view size, and an
    applet-supplied `state` blob (step count, seed, hparams) — the
    provenance that makes a figure *reproducible*, which is the actual
    research-grade property.
- **PNG encoding:** stb_image_write (header-only, already-vendored-class
  dependency) — no libpng.
- **Applet sugar:** `caliper::Export` wrapper + an ImGui "📷 Export…"
  affordance pattern documented in the wiki (each applet opts in; the host
  never screenshots chrome — exports are VIEWS, not windows).
- **Determinism contract:** exporting the same tensors + camera + colormap
  twice yields byte-identical PNGs on the same backend (the byte-exact
  discipline extended one step). Cross-backend stays the existing bar
  (exact except the documented Lambert ±2 LSB).
- **UI plots (ImPlot charts, e.g. pulse_scope/metrics):** NOT rendered
  through export.v1 (they are ImGui chrome, not views). The paper-figure
  path for metrics is the one that already ships: SQL via `metrics.v1_1` →
  matplotlib. Document this split honestly in the wiki page.

### 1.4 Acceptance (exemplar-driven, per house rule)

mesh_scope or TwinScope gains the export affordance: a 3840×2160 PNG of the
live view + sidecar, byte-identical on re-export, rendered at export
resolution (not upscaled); a 300-frame sequence assembled by the documented
ffmpeg line into the landing page's first REAL demo clip (closing the GIF
placeholder debt as the proof artifact). Suites green; a new export battery
(golden sidecar, deterministic double-export compare, refusal purity on bad
paths/sizes).

## 2. Rung P — Python interop (the field-unlock)

### 2.1 The load-bearing decision: DLPack in, not torch-in-C++

The naive design (Python calls a libcaliper that links its own libtorch)
violates D5 the moment PyTorch is imported: two libtorches, one process,
symbol chaos. The correct shape drops torch from the seam entirely:

- **`caliper` Python package = pybind11 over the embed C ABI** (embed.h
  v1.1 — create/canvas/frame/get_service already exist) **+ DLPack
  ingestion**: `CaliperTensor` was DLPack-aligned by design (D3), so a
  PyTorch/JAX/CuPy tensor crosses via `__dlpack__` with ZERO copies — the
  same MPS/CUDA memory the bridge already imports, now sourced from
  Python's own runtime. Python's torch is the ONE torch (D5 satisfied
  with the embedder owning it — inverted but legal).
- **Prerequisite (converts a debt item into a real project):** the
  **torch-free libcaliper link variant** — the Python wheel must NOT carry
  a second libtorch. The bridge/geometry/renderer do not need torch (they
  consume CaliperTensor/imported allocations); only the torch ADAPTER
  (`adapters/torch.hpp`, applet-side) and torch-linking applets do. The
  variant compiles libcaliper without `CALIPER_DEPENDENCY_LIBS`' torch
  edge; applets that need torch keep it (they load as plugins with their
  own deps — the packs story, unchanged).
- **What Python gets in v0 (scoped hard):** create core, offscreen or
  windowed canvas, feed tensors to the bridge/geometry via DLPack, drive
  draws, read services (metrics/feed/export), pump frames from Python's
  loop. What it does NOT get: writing applets in Python (applets stay C
  ABI plugins; a notebook drives the INSTRUMENT, it does not become one).
  That keeps the seam the size of embed.h instead of the size of ImGui.

### 2.2 The exemplar (freezes the rung)

A Jupyter/py script: train a small PyTorch model (Python-side, MPS), hand
its weight/field tensors to Caliper via DLPack each step, watch the live
mesh/points view in a Caliper window; call `export.view_png` for the
figure. The whitepaper's local-loop claim then honestly extends to Python:
same GPU memory, no host round-trip, one process.

### 2.3 Risks pinned now

GIL vs frame pump (pump from Python = GIL-held C calls — fine, frame() is
short; document never calling frame() from a thread while another Python
thread mutates handed tensors — the drain-before-publish contract crosses
the language); wheel building/abi3 + per-OS binaries (ride cibuildwheel;
macOS first, Windows second per house rhythm); DLPack device negotiation
(MPS dlpack support in torch — VERIFY EARLY, it gates the zero-copy claim
on Mac; CUDA is mature).

## 3. Rung B — the reviewer bundle (packaging, scoped to papers)

### 3.1 The decision: the supplement slice of Phase 4–5, nothing more

PLATFORM Phases 3–5 (SDK split, template CI, packs, registry) remain the
real packaging program, deliberately not dragged in here. A paper needs ONE
artifact: **a supplement bundle a reviewer can run** — host + applet(s) +
assets + (if torch applets) the runtime pack, one download, no build.

- macOS: signed+notarized .dmg (the codesign machinery is a named Phase-5
  item — this rung pulls exactly that forward, nothing else).
- Windows: .zip with the DLL closure (the embed_host DLL-copy step
  generalized).
- `caliper bundle <applet-id>` host subcommand (D9 pattern) assembles it.
- The bundle README states the honesty registers (platform status, what is
  live vs recorded) — reviewers get the same no-overclaim discipline as
  the docs.

### 3.2 Acceptance

A machine that has never built Caliper (the other laptop, a fresh user
account) downloads one file, opens it, and runs the paper demo applet
live. That test IS the rung; nothing else counts.

## 4. What stays out (all three rungs)

Figure typography/composition; video encoding in-process; Python-authored
applets; the registry/browse ecosystem; Linux (still its own later item);
any new render path (export re-renders EXISTING views); any ABI break
(everything above is additive services + an embed wheel).

## 5. Sequencing (recommendation, decided unless redirected)

1. **Rung E first** — smallest (machinery exists), immediate paper value
   (figures + the landing-page demo clip debt), and rungs P/B both consume
   it (the Python exemplar exports; the bundle demos it).
2. **Rung P second** — the field-unlock; its prerequisite (torch-free
   variant) is now demand-justified. Verify torch-MPS DLPack support in
   week one — it is the rung's only existential risk on the primary box.
3. **Rung B last, demand-driven** — build it against the FIRST actual
   submission (extract-don't-invent applied to packaging); until then the
   repo builds suffice for collaborators.

Each rung: own execution spec → house SDD pipeline (foreground Opus
subagents, review gates, both-platform honesty registers).

## Invariants (hold forever)

- Export is a terminal sink: pixels leave, nothing returns to compute —
  render-to-tensor stays rejected.
- One torch per process, whoever owns it (the embedder may — D5 refined by
  rung P).
- A figure without its sidecar is a screenshot; Caliper exports FIGURES —
  provenance always attached.
- Nothing ships claimed beyond its verified platform — bundles and wheels
  inherit the per-platform honesty register verbatim.
