#pragma once
/* caliper.export.v1 — the terminal sink: pixel-exact view PNGs + a provenance
 * sidecar (PUBLISHING.md §3, Rung E). Export re-renders a submitted draw list
 * OFFSCREEN at any requested resolution through pieces that already ship and
 * are already byte-exact tested — geometry.v1_3's create_view_ex + the existing
 * draw_primitives host path + the promoted debug_readback_rgba8 — then encodes
 * the tightly-packed RGBA8 to PNG (stb_image_write, vendored) and writes a JSON
 * sidecar next to it. No new render code, no retained draw state; the pixels
 * that leave are the tensors, bit-for-bit.
 *
 * Immediate-mode, mirrors the geometry.v1_3 draw signature (v1_3 records,
 * stride-widened exactly like geometry.v1_3): the caller hands the SAME arrays
 * it draws with. Video assembly stays OUTSIDE the ABI — the wiki documents the
 * one ffmpeg line over the numbered frames a sequence writes.
 *
 * Refusal purity extends to the FILESYSTEM: a refused export (missing gate, bad
 * w/h, unwritable path, readback failure, a draw the geometry gate battery
 * rejects) returns 0 and leaves the disk EXACTLY as it was — no file created,
 * no pre-existing file truncated (the PNG is written to a temp name and
 * atomically renamed; the sidecar follows only after the PNG lands).
 *
 * FRAME-THREAD ONLY: every entry point composes the frame-thread-owned
 * renderer (offscreen view create → the geometry draw path → readback), so it
 * carries the same rule as caliper.geometry.v1 — call from the caliper_core_
 * frame() thread only. Calling from any other thread races the renderer; the
 * sequence mutex serializes bookkeeping, it does NOT make export any-thread.
 *
 * Determinism: same draws + camera + clear + (w,h) on the SAME backend →
 * BYTE-IDENTICAL PNG across calls (the readback is deterministic, the byte-exact
 * matrix is built on it, stb PNG encode is deterministic). Cross-backend is NOT
 * byte-identical (Lambert ±2 LSB carries over) — the sidecar's `backend` field
 * makes that honest.
 *
 * IMMUTABLE once published; additive growth lands as export.v1_1. */
#include <stdint.h>
#include <caliper/services/geometry_v1_3.h>   /* CaliperGeomCamera, CaliperGeomDrawV1_3 */

#define CALIPER_EXPORT_V1 "caliper.export.v1"

/* caps() bit 0: view_png / sequences are live. Set IFF the geometry primitives
 * cap (CALIPER_GEOM_CAP_PRIMITIVES) is live on this host — export is a veneer
 * over that path, so it degrades in lockstep with it (absent renderer / headless
 * → 0, every entry point inert, no file ever written). */
#define CALIPER_EXPORT_CAP_VIEW_PNG (1u << 0)

/* Largest accepted dimension (inclusive). w or h of 0 or > this is refused. */
#define CALIPER_EXPORT_MAX_DIM 16384u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperExportV1 {
    uint32_t struct_size;
    uint32_t (*caps)(void);
    /* Render `draws` (v1_3 records, stride-widened exactly like geometry.v1_3)
     * into a fresh offscreen (w,h) target and write:
     *   <path>        — PNG, tightly-packed RGBA8, top-down (row 0 = top).
     *   <path>.json   — the provenance sidecar (PUBLISHING.md §3), written
     *                   atomically AFTER the PNG succeeds.
     * Returns 1 on success; 0 on refusal. On ANY refusal — no gate/renderer,
     * cam==0, w/h 0 or > CALIPER_EXPORT_MAX_DIM, unwritable path, readback
     * failure, or a draw the geometry gate battery rejects — NO file is created
     * and NO pre-existing file is truncated. Same gate battery, same reason
     * strings, same atomicity as draw_primitives (it IS draw_primitives).
     * state_json is nullable and copied VERBATIM into the sidecar's "state". */
    uint32_t (*view_png)(const char* path, uint32_t w, uint32_t h,
                         const CaliperGeomCamera* cam,
                         const CaliperGeomDrawV1_3* draws,
                         uint32_t draw_count, uint32_t draw_stride,
                         uint32_t clear_rgba,
                         const char* state_json);
    /* Frame sequences: same per-frame contract as view_png, frames written as
     * <dir>/frame_%06u.png, plus ONE sidecar for the whole sequence written at
     * end (<dir>/sequence.json, carrying frame_count). begin_sequence returns a
     * non-zero handle (0 = refusal); frame() renders one numbered frame (1 ok /
     * 0 refusal); end_sequence() finalizes + drops the handle. ONE sequence live
     * at a time (v0); begin while one is active is refused. */
    uint64_t (*begin_sequence)(const char* dir, uint32_t w, uint32_t h,
                               const char* state_json);
    uint32_t (*frame)(uint64_t seq, const CaliperGeomCamera* cam,
                      const CaliperGeomDrawV1_3* draws, uint32_t draw_count,
                      uint32_t draw_stride, uint32_t clear_rgba);
    void     (*end_sequence)(uint64_t seq);
    void*    reserved0;
} CaliperExportV1;

#ifdef __cplusplus
}
/* --- ABI freeze: sizes + offsets pinned (only C types cross the boundary) --- */
static_assert(offsetof(CaliperExportV1, struct_size) == 0);
static_assert(offsetof(CaliperExportV1, caps) == sizeof(void*));
static_assert(offsetof(CaliperExportV1, view_png) ==
              offsetof(CaliperExportV1, caps) + sizeof(void*));
static_assert(offsetof(CaliperExportV1, begin_sequence) ==
              offsetof(CaliperExportV1, view_png) + sizeof(void*));
static_assert(offsetof(CaliperExportV1, frame) ==
              offsetof(CaliperExportV1, begin_sequence) + sizeof(void*));
static_assert(offsetof(CaliperExportV1, end_sequence) ==
              offsetof(CaliperExportV1, frame) + sizeof(void*));
static_assert(offsetof(CaliperExportV1, reserved0) ==
              offsetof(CaliperExportV1, end_sequence) + sizeof(void*));
static_assert(sizeof(CaliperExportV1) ==
              offsetof(CaliperExportV1, reserved0) + sizeof(void*),
              "CaliperExportV1 vtable layout is frozen");
#endif
