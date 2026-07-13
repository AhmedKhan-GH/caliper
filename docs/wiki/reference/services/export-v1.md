# caliper.export.v1

Service id `caliper.export.v1` — the **terminal sink** of the publishing track
(`PUBLISHING.md` §3, Rung E). Export re-renders a submitted draw list
**offscreen** at any requested resolution and writes a pixel-exact PNG next to a
JSON **provenance sidecar**: 4K figures and numbered frame sequences from a live
session, deterministic and stamped. It composes pieces that already ship and are
already byte-exact tested — it adds **no new render code**. This page embeds the
header verbatim; the docs build fails if the embedded file moves.

The boundary is deliberate and stated once in `PUBLISHING.md`: Caliper exports
**pixel-exact view images + a sidecar**, and does **not** compose figures. Axes,
captions, typography, and subfigure layout belong to the paper toolchain
(matplotlib / TikZ) that consumes these PNGs. The instrument's claim — *the
pixels are the tensors* — survives export bit-for-bit; a typography engine would
add scope with zero claim.

!!! info "Platform status (honest, stated once)"
    **macOS / Metal:** **run-proven.** The E1 battery is green live on Metal (a
    known FLAT quad exported and decoded back to the CPU reference pixels;
    double-export byte-identity; refusal purity; sidecar golden; a finalized
    sequence). The E2 exemplars produced real artifacts from a live session — a
    3840×2160 TwinScope figure + sidecar (`backend=metal`) and a 300-frame
    twin_scope sequence assembled into a 10 s clip via the documented ffmpeg
    line. **Windows / Vulkan:** **run-proven.** The full battery ran live on
    the Vulkan box (RTX 500 Ada, Windows 11) — 14 cases, 0 skipped: the same
    decoded-quad byte-exactness, double-export byte-identity, and refusal
    purity (sentinel byte-identical after a refused export), plus the
    NTFS-specific claims verified on the box, not from docs:
    `std::filesystem::rename` **atomically replaces** a pre-existing target
    (`MoveFileExW(MOVEFILE_REPLACE_EXISTING)` semantics confirmed); a target
    held open without `FILE_SHARE_DELETE` makes the rename **fail cleanly**
    (refusal, temp removed, original byte-identical); and a sidecar-write
    failure **rolls the PNG back** rather than orphaning it. The E2 exemplars
    reproduced on Vulkan: a 3840×2160 figure + sidecar
    (`backend=vulkan platform=windows`) and a finalized 300-frame sequence
    from each of twin_scope and mesh_scope, plus a deliberate mid-record kill
    whose `sequence.json` finalized with the honest partial count. When the geometry
    primitives cap is absent (headless, or a host without the renderer path) the
    export cap bit is unset and **every entry point is inert** — no file is ever
    written. This is the degradation ladder, not a bug.

```c
--8<-- "sdk/include/caliper/services/export_v1.h"
```

## The composition (why this is small)

Export is a **veneer** over the geometry draw path, not a second renderer. Each
call:

1. `create_view_ex(w, h, CALIPER_GEOM_VIEW_DEPTH)` — a fresh offscreen target at
   the requested resolution (the geometry pipeline is resolution-independent, so
   a 4K figure from a live session costs only pixels);
2. the **existing** `draw_primitives` path over the submitted v1_3 draws — every
   gate, every reason string, every byte-exact behavior reused **verbatim** (it
   *is* `draw_primitives`);
3. `debug_readback_rgba8` — the tightly-packed RGBA8 readback implemented on
   **both** backends since the v1_3 pass, promoted from test-only to the export
   path;
4. PNG encode (`stb_image_write`, vendored and pinned) + the sidecar JSON;
5. destroy the temp view.

No retained draw state. The draw arrays are the **same** immediate-mode arrays
the applet draws with (v1_3 records, stride-widened exactly like
`caliper.geometry.v1_3` — the `caliper::Export` sugar mirrors
`caliper::Geometry`'s v1.1/v1.2/v1.3 overload set and passes `sizeof` for you).
Because export re-renders through the geometry cap, it **degrades in lockstep**
with it: `caps()` bit 0 (`CALIPER_EXPORT_CAP_VIEW_PNG`) is set **iff**
`CALIPER_GEOM_CAP_PRIMITIVES` is live on this host.

## Frame-thread only

Every entry point composes the **frame-thread-owned** renderer, so export
carries the same rule as [`caliper.geometry.v1`](geometry-v1.md): call it from
the `caliper_core_frame()` thread only. Calling from any other thread races the
renderer's texture/geometry maps. The sequence bookkeeping (one sequence live at
a time in v0) is guarded by a mutex, but that mutex serializes **bookkeeping
only** — it does **not** make export safe against a renderer running on another
thread. The E2 exemplars capture **inline on the frame thread** for exactly this
reason: they snapshot the worker-published slot under the usual mutex, then
export from the same arrays they draw, on the same thread they draw on. (A truly
background export would need a serialization mutex around the renderer's shared
maps — a host change beyond this rung.)

## The sidecar — a figure without it is a screenshot

Every export writes `<path>.json` next to the PNG (for a sequence, one
`sequence.json` for the whole run). It carries the provenance that makes a PNG
**research-grade** rather than a screenshot:

```json
{
  "caliper": { "version": "...", "git_commit": "...", "backend": "metal", "platform": "macos" },
  "timestamp_utc": "2026-07-12T17:13:37Z",
  "width": 3840, "height": 2160, "clear_rgba": 4278519050,
  "camera": { "view": [ ...16... ], "proj": [ ...16... ] },
  "draw_count": 3,
  "colormaps": [1],
  "state": { "step": 1200, "seed": 7 }
}
```

`git_commit` is compiled in at configure time (the same mechanism that plumbs
the version string). `state` is the caller's own JSON, copied **verbatim** (step,
seed, hparams — whatever the applet chooses) or `null`. A sequence sidecar adds
`frame_count`. This is a house invariant: **a figure without its sidecar is a
screenshot; Caliper exports figures.**

## Refusal purity extends to the filesystem

A refused export returns `0` and leaves the disk **exactly as it was** — no file
created, no pre-existing file truncated. The PNG is written to a temp name and
atomically renamed; the sidecar follows **only after** the PNG lands, and if the
sidecar write fails the PNG is rolled back rather than orphaned (a PNG without
its sidecar would violate the invariant above). Export refuses for the same
reasons `draw_primitives` does — missing gate/renderer, `cam == NULL`, `w`/`h` of
`0` or `> CALIPER_EXPORT_MAX_DIM` (16384), an unwritable path, a readback
failure, or any draw the geometry gate battery rejects — with the **same reason
strings**. (The atomic-rename-over-existing-file guarantee is verified on POSIX
**and on NTFS** — including the held-open-handle refusal and the
sidecar-failure PNG rollback; see the platform status above.)

## Determinism contract, and its scope

Same draws + camera + clear + `(w, h)` on the **same backend** → a
**byte-identical PNG** across calls. The readback is deterministic (the entire
byte-exact test matrix is built on it) and `stb_image_write` is deterministic, so
this is pinned by a test that exports twice and `memcmp`s the files. **Cross-**
backend is **not** byte-identical — the Lambert ±2-LSB tolerance that the
geometry contract carries survives into the pixels — and the sidecar's `backend`
field makes that honest rather than silent. Determinism is the byte-exact
discipline extended one step to disk, scoped exactly where the instrument can
back it.

## Video assembly stays outside the ABI

Export writes numbered frames (`<dir>/frame_%06u.png`) and one sequence sidecar;
it does **not** encode video (no in-process codecs). Assemble the frames with one
documented ffmpeg line:

```sh
ffmpeg -framerate 30 -i frame_%06d.png -pix_fmt yuv420p out.mp4
```

This is the exact line the E2 twin_scope run used to turn a 300-frame sequence
into its 10 s clip.

## What is not exportable — the ImPlot / metrics split

ImPlot chart panes (pulse_scope, metrics dashboards) are **ImGui chrome, not
exportable views** — they are not tensors re-rendered offscreen, so export does
not touch them. Their paper path is the one that already ships: the numbers go to
**SQL via [`caliper.metrics.v1_1`](metrics-v1.md)** and into matplotlib
downstream. Export is for the 3-D views whose pixels *are* the tensors; charts
are for the metrics surface. (`PUBLISHING.md` §3, the honest split.)

## Worked shape — export a figure and record a clip

Modelled on `applets/mesh_scope` and `applets/twin_scope` (the E2 exemplars).
The applet already draws its 3-D view every frame from worker-published,
mutex-snapshotted slot tensors; export reuses **those same arrays**, on the frame
thread:

```cpp
caliper::Export xport(host);          // falsy when the host vends no export path
if (xport.has_view_png()) {           // caps bit 0, tied to geometry primitives
    // A 4K still: the SAME draws[] the applet just rendered, at a bigger view.
    xport.view_png("figure.png", 3840, 2160, &cam,
                   draws, draw_count, /*clear=*/0xff05050au,
                   /*state_json=*/R"({"step":1200,"seed":7})");   // + figure.png.json

    // A clip: begin → one frame() per captured frame → end (one sequence.json).
    uint64_t seq = xport.begin_sequence("clip_dir", 1280, 720, state_json);
    for (int i = 0; i < 300; ++i)
        xport.frame(seq, &cam_i, draws_i, draw_count_i, clear);   // frame_000000.png …
    xport.end_sequence(seq);                                      // + clip_dir/sequence.json
}
```

Then assemble the clip with the ffmpeg line above. Because capture re-renders and
reads back each frame inline, a long high-resolution record paces the UI while it
runs — the exemplars show a live "Recording N/300" counter; the produced clip is
correct real-time regardless.

---

See also: [`caliper.geometry.v1`](geometry-v1.md) for the draw path and camera
vocabulary export composes, and [`caliper.metrics.v1_1`](metrics-v1.md) for the
chart/metrics paper path export deliberately does not cover.
