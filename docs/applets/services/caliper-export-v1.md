# caliper.export.v1 and v1_1

Service id `caliper.export.v1` has an additive revision, `caliper.export.v1_1`. The revision adds `last_error()` (see [Error Reporting](#caliperexportv1_1-error-reporting)). Export re-renders a submitted draw list **offscreen** at any requested resolution. It writes a pixel-exact PNG next to a JSON **provenance sidecar**. Use it for 4K figures and numbered frame sequences from a live session. The output is deterministic and stamped. Export adds no new render code.

## Reference

### Calls

<!-- c-api calls sdk/include/caliper/services/export_v1.h -->

### Ids

<!-- c-api ids sdk/include/caliper/services/export_v1.h -->

## Terminal Sink

Export is the one place where pixels leave the one-way data flow described in [Platform rules](../../introduction/architecture.md#design-decisions). The readback feeds a file on disk, and nothing returns to compute. No export call hands pixels, a texture or a tensor back to the applet.

Caliper exports **pixel-exact view images plus a sidecar**. It does not compose figures. Axes, captions, typography and subfigure layout belong to the paper toolchain (matplotlib, TikZ) that consumes the PNGs.

## Platform Behaviour

Export runs on macOS with Metal and on Windows with Vulkan. On both platforms, a known flat quad exports and decodes back to the CPU reference pixels. A second export of the same view is byte-identical. A refused export leaves the disk unchanged. If a process is killed during a sequence, the sequence still finalizes with the true partial frame count.

On Windows, a rename atomically replaces an existing target on NTFS. If another process holds the target open without `FILE_SHARE_DELETE`, the rename fails cleanly. Then export is refused and the original stays byte-identical. If the sidecar write fails, export rolls the PNG back.

When the geometry primitives capability is absent (headless, or a host without the renderer path), the export capability bit is unset and **every entry point is inert**. No file is written. This is the [fallback rule](../../introduction/architecture.md#design-decisions) at work and not a bug.

## Draw Path Reuse

Export is a **veneer** over the geometry draw path and not a second renderer. Each call does these steps:

1. `create_view_ex(w, h, CALIPER_GEOM_VIEW_DEPTH)` creates a fresh offscreen target at the requested resolution. The geometry pipeline is resolution-independent, so a 4K figure costs only pixels.
2. The existing `draw_primitives` path runs over the submitted v1_3 draws. Every gate, reason string and byte-exact behavior is the same as in `draw_primitives`.
3. `debug_readback_rgba8` reads back tightly packed RGBA8 pixels on both backends.
4. The PNG encoder and the sidecar writer produce the files.
5. The temporary view is destroyed.

Export keeps no draw state. The draw arrays are the same immediate-mode v1_3 arrays the applet draws with, stride-widened as in `caliper.geometry.v1_3`. The `caliper::Export` sugar mirrors the overload set of `caliper::Geometry` and passes `sizeof` for you.

Export re-renders through the geometry capability, so it degrades with it. `caps()` bit 0 (`CALIPER_EXPORT_CAP_VIEW_PNG`) is set if and only if `CALIPER_GEOM_CAP_PRIMITIVES` is live on this host.

## Frame Thread Only

Every entry point uses the renderer that the frame thread owns. Export carries the same rule as [`caliper.geometry.v1`](caliper-geometry-v1.md): call it from the `caliper_core_frame()` thread only. A call from any other thread races the texture and geometry maps of the renderer.

One sequence can be live at a time in v0. A mutex guards the sequence bookkeeping, but it does not make export safe against a renderer on another thread. For this reason, MeshScope and TwinScope capture inline on the frame thread. They snapshot the worker-published slot under the usual mutex, then export from the same arrays they draw. A background export is not possible, because the host provides no lock around the shared renderer maps.

## Sidecar File

Every export writes `<path>.json` next to the PNG. A sequence writes one `sequence.json` for the whole run. The sidecar carries the provenance that makes a PNG **research-grade** and not a screenshot:

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

The `state` field is the caller's own JSON, copied **verbatim**, or `null`. A sequence sidecar adds `frame_count`. The export rule is this: **a figure without its sidecar is a screenshot, and Caliper exports figures.**

`end_sequence()` writes `sequence.json`. If that write fails, the frames already written are kept. The handle is finalized anyway, so the next sequence can begin. The host logs an ERROR through `caliper.log.v1` that names the file, the frame count and the reason:

```
[export] sequence sidecar write failed: clip_dir/sequence.json (300 frame(s) kept, provenance missing): <reason>
```

In v1, `end_sequence()` returns nothing, so this log line is all a v1 applet gets. The `last_error()` function of [`caliper.export.v1_1`](#caliperexportv1_1-error-reporting) reports the same text to the applet.

## Filesystem Refusal Purity

A refused export returns `0` and leaves the disk **exactly as it was**. It creates no file and truncates no existing file. A **previous export at the same path is kept**. The `view_png` function commits the PNG and its sidecar as a pair:

1. Export writes both files to sibling temp files, `<path>.caliper_tmp` and `<path>.json.caliper_tmp`. A failure here has touched nothing you can see.
2. Export sets aside a previous PNG at `<path>` as `<path>.caliper_bak`.
3. Export renames the new PNG onto `<path>`, then the sidecar onto `<path>.json`.
4. Export removes the backup.

If any step after the first fails, export puts the previous PNG back. If there was no previous PNG, it removes the new one. It also removes the temp files. As a result, a PNG never lands without its sidecar, and a failed new figure never destroys a good earlier one. If the put-back fails too, the previous PNG stays intact at `<path>.caliper_bak`. A `path` that is a directory is refused before anything moves.

Export refuses for the same reasons as `draw_primitives`, with the **same reason strings**. The comment of `view_png` lists them.

## Determinism Contract

The same draws, camera, clear color and `(w, h)` on the **same backend** give a **byte-identical PNG** across calls. Both the readback and the PNG encoder are deterministic.

Output across backends is **not** byte-identical. The Lambert ±2-LSB tolerance of the geometry contract carries into the pixels. The `backend` field of the sidecar records which backend produced the file.

## Video Assembly

Export writes numbered frames (`<dir>/frame_%06u.png`) and one sequence sidecar. It does **not** encode video, because it has no in-process codecs. Assemble the frames with this ffmpeg line:

```sh
ffmpeg -framerate 30 -i frame_%06d.png -pix_fmt yuv420p out.mp4
```

A 300-frame sequence at 30 fps becomes a 10 s clip.

## Non-Exportable Panes

ImPlot chart panes (strip charts, metrics plots) are **ImGui chrome and not exportable views**. They are not tensors re-rendered offscreen, so export does not touch them. Their paper path is **SQL via [`caliper.metrics.v1_1`](caliper-metrics-v1.md)**, and then matplotlib downstream. Export is for the 3-D views whose pixels *are* the tensors.

## Figure and Clip Example

The MeshScope and TwinScope export buttons work this way. The applet already draws its 3-D view every frame from worker-published, mutex-snapshotted slot tensors. Export reuses **those same arrays** on the frame thread:

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

Then assemble the clip with the ffmpeg line above. Capture re-renders and reads back each frame inline. As a result, a long high-resolution record paces the UI while it runs, but the produced clip is correct in real time.

## `caliper.export.v1_1` Error Reporting

This is an additive revision under the same rules as `caliper.metrics.v1_1` and the [bridge revisions](caliper-tensor-bridge-v1.md#additive-revisions). `CaliperExportV1_1` repeats the v1 table byte for byte, with `reserved0` still `NULL`. It appends one entry, `last_error()`. The ABI epoch does not change. The host still vends `caliper.export.v1` unchanged, so a v1 applet keeps the v1 table and the v1 behavior.

The host lists both ids, so a manifest may require either. An applet that wants better diagnostics asks for `caliper.export.v1_1` and falls back to v1.

### Calls

<!-- c-api calls sdk/include/caliper/services/export_v1_1.h -->

### Ids

<!-- c-api ids sdk/include/caliper/services/export_v1_1.h -->

**Reasons.** The table lists the strings that `last_error()` can return. Export is frame-thread only, so in practice there is one status.

| Reason | Cause |
|---|---|
| `no renderer is bound (caps() is 0)` | A headless host, or no renderer bound yet. |
| `the geometry primitives path is not live (caps() is 0)` | The backend has no geometry primitives path. |
| `export is not live on this host (caps() is 0)` | `begin_sequence` on a host where `caps()` is `0`. |
| `cam is NULL`, `path is NULL`, `dir is NULL` | A required argument is missing. |
| `w/h must be 1..16384 (got 0x720)` | A dimension is `0` or above `CALIPER_EXPORT_MAX_DIM`. |
| `a sequence is already active (one at a time)` | `begin_sequence` while another sequence is open. |
| `no active sequence has this handle` | `frame` or `end_sequence` with a stale or unknown handle. |
| `the geometry gates refused the draws (reason logged)` | A draw failed the geometry gates; the gate's own reason is on `caliper.log.v1`. |
| `could not commit <path> and its sidecar: <reason>` | The pair commit failed; the previous export is kept. |
| `cannot create <dir>: <reason>`, `could not write <dir>/frame_000012.png: <reason>` | A sequence directory or frame could not be written. |
| `sequence sidecar write failed: <dir>/sequence.json (N frame(s) kept, provenance missing): <reason>` | `end_sequence` could not write the sequence sidecar. |

The `caliper::Export` sugar wraps v1 and does not expose `last_error()`. Take the v1_1 table from the host directly:

```cpp
auto* ex11 = static_cast<const CaliperExportV1_1*>(
    host.service(CALIPER_EXPORT_V1_1));        // NULL on a host without v1_1
if (ex11) {
    ex11->end_sequence(seq);
    if (*ex11->last_error())                   // "" after a clean finalize
        host.log_error(ex11->last_error());    // e.g. sequence.json not written
}
```

---

See also: [`caliper.geometry.v1`](caliper-geometry-v1.md) for the draw path and camera vocabulary that export composes, and [`caliper.metrics.v1_1`](caliper-metrics-v1.md) for the chart and metrics paper path that export does not cover.
