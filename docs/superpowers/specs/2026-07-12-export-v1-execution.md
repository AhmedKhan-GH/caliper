# `caliper.export.v1` — execution spec (Rung E of PUBLISHING.md)

**Date:** 2026-07-12
**Status:** **EXECUTED E1–E3** on `feat/export-v1` — E1 service+battery
(through `7780138`), E2 exemplars+run-proven artifacts (`f9c36a2`), E3 docs
closeout (`cd03d21` riders + the docs commit). macOS/Metal run-proven;
Windows/Vulkan compiles, battery pending the box. Design authority:
PUBLISHING.md §3 (the boundary decision: pixel-exact views + provenance
sidecar, never figure composition; determinism contract; ImPlot chrome
excluded). This doc pins the ABI, the composition, and the task ladder.

---

## 1. The composition insight (why this is small)

Export re-renders through pieces that already ship and are already tested:
`geom_create_view_ex(w,h,DEPTH)` (any resolution — the pipeline is
resolution-independent) → the EXISTING `draw_primitives` path (every gate,
every byte-exact behavior, reused verbatim) → `debug_readback_rgba8`
(implemented on BOTH backends since the gfx harness / v1_3 pass; promoted
from test-only to the export path) → PNG encode (stb_image_write, vendored
header) + sidecar JSON → destroy the temp view. No new render code. No
retained state.

## 2. The ABI (immediate-mode, mirrors the draw signature)

`sdk/include/caliper/services/export_v1.h`, id `"caliper.export.v1"`:

```c
#define CALIPER_EXPORT_CAP_VIEW_PNG (1u << 0)   /* set iff geometry
                                                   primitives cap is live */

typedef struct CaliperExportV1 {
    uint32_t struct_size;
    uint32_t (*caps)(void);
    /* Render draws (v1_3 records, stride-widened exactly like
       geometry.v1_3) into a fresh offscreen (w,h) target and write:
       <path>          — PNG, tightly-packed RGBA8, top-down
       <path>.json     — the provenance sidecar (§3)
       1 on success; 0 on refusal — NO file is created or truncated on any
       refusal (gate failure, bad w/h (0 or > 16384), unwritable path,
       readback failure). Same gate battery, same reason strings, same
       atomicity as draw_primitives. */
    uint32_t (*view_png)(const char* path, uint32_t w, uint32_t h,
                         const CaliperGeomCamera* cam,
                         const CaliperGeomDrawV1_3* draws,
                         uint32_t draw_count, uint32_t draw_stride,
                         uint32_t clear_rgba,
                         const char* state_json /* nullable, verbatim into
                                                   the sidecar */);
    /* Frame sequences: same contract, frames written as
       <dir>/frame_%06u.png (+ one sidecar for the sequence, updated with
       frame_count at end). begin returns a handle (0 = refusal); frame()
       renders one; end() finalizes. One sequence live at a time (v0). */
    uint64_t (*begin_sequence)(const char* dir, uint32_t w, uint32_t h,
                               const char* state_json);
    uint32_t (*frame)(uint64_t seq, const CaliperGeomCamera* cam,
                      const CaliperGeomDrawV1_3* draws, uint32_t draw_count,
                      uint32_t draw_stride, uint32_t clear_rgba);
    void     (*end_sequence)(uint64_t seq);
    void*    reserved0;
} CaliperExportV1;
```

Widening: the SDK `caliper::Export` sugar mirrors `caliper::Geometry`'s
overload set (v1.1/v1.2/v1.3 records, zero-tail widening — reuse the same
helpers). Video assembly stays outside: the wiki documents the one ffmpeg
line (`ffmpeg -framerate 30 -i frame_%06d.png -pix_fmt yuv420p out.mp4`).

## 3. The sidecar (a figure without it is a screenshot)

`<path>.json`, written atomically after the PNG succeeds:
`{ "caliper": {version, git_commit (compiled-in via configure), backend,
platform}, "timestamp_utc", "width", "height", "clear_rgba", "camera":
{full matrix + params as submitted}, "draw_count", "colormaps": [ids used],
"state": <caller json verbatim or null> }`. The git commit lands via a
configure-time definition (follow how the version string is already
plumbed; add if absent — one CMake line).

## 4. Determinism contract (tested, not asserted)

Same draws + camera + clear + (w,h) on the same backend → **byte-identical
PNG** across two calls (the readback is deterministic — the byte-exact
matrix is built on it; stb_image_write is deterministic). Pinned by a test
that exports twice and memcmps the files. Cross-backend: NOT claimed
byte-identical (Lambert ±2 LSB carries over); the sidecar's `backend` field
makes that honest.

## 5. Task ladder (SDD; foreground Opus implementers, review each)

- **E1 — service + tests.** Header (+abi_c_check, test_abi pins in the
  house register), stb_image_write vendored (`third_party/stb/`), host
  implementation by composition (temp view → draw → readback → PNG +
  sidecar → destroy; sequence bookkeeping), vend `kExport` (caps tied to
  `supports_geometry_primitives()`), SDK sugar with widening. Battery (new
  `test_export.cpp`, folded into a suitable target): ABI pins; a known
  FLAT quad exported and the PNG bytes decoded (vendor stb_image.h
  TEST-ONLY) matching the CPU reference pixels; double-export byte-identity;
  refusal purity — gate-refused draws, w=0, huge w, unwritable path → rc 0
  AND no file/no truncation of a pre-existing file; sidecar golden
  (fixed-timestamp injection for the golden, mirror the C2 report pattern);
  sequence writes N frames + finalized sidecar. Metal-gated live cases +
  platform-neutral logic split per house pattern; MSVC-safe. Windows note:
  the composition is backend-neutral (Vulkan implements every piece) —
  compiled but unproven there; the next Windows pass runs the battery
  (honesty register in header + wiki).
- **E2 — the exemplar affordance + the demo clip.** mesh_scope AND
  twin_scope gain "Export figure (4K)" + "Record 10 s" (button → job-thread
  capture using the SAME spec arrays they draw with; files + log line with
  paths; kLockedPlot untouched). Run-proof: a 3840×2160 TwinScope PNG +
  sidecar from a live session (artifact paths logged); a 300-frame
  instance_scope or twin_scope sequence assembled via the documented ffmpeg
  line into an mp4. **The clip is produced and staged as an artifact
  ONLY — swapping it into the public landing page needs the owner's
  explicit go (standing rule: no unrequested public artifacts).**
- **E3 — docs closeout + branch close.** wiki `export-v1.md` (header
  embed, the ffmpeg line, the ImPlot-excluded/metrics-goes-SQL split,
  per-platform honesty); PUBLISHING.md §2 table: Rung E → SHIPPED
  (macOS-proven, Windows battery pending the box); final review; merge+push.

## 6. Out of scope (from PUBLISHING.md §7, restated)

Figure composition/typography; in-process encoding; exporting ImGui/ImPlot
chrome; SVG/PDF (raster PNG only in v1 — vector is a future rung with its
own demand); retained draw state of any kind; Linux.

## Invariants

Export is a terminal sink (pixels leave, nothing returns); refusal purity
extends to the filesystem (a refused export leaves the disk exactly as it
was); the sidecar is never optional; no ABI breaks — one new additive
service.
