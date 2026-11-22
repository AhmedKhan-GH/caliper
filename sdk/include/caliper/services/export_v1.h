#pragma once
/* caliper.export.v1 — the terminal sink: pixel-exact view PNGs and a provenance sidecar
 * (docs/applets/services/caliper-export-v1.md). Export re-renders a submitted draw list offscreen through the
 * geometry draw path and writes the pixels to disk. Call it from the caliper_core_frame() thread only. A refused
 * export leaves the disk as it was. Immutable once published: additive growth is export.v1_1. */
#include <stdint.h>
#include <caliper/services/geometry_v1_3.h>   /* CaliperGeomCamera, CaliperGeomDrawV1_3 */

/** The id of this service, which `get_service` takes. */
#define CALIPER_EXPORT_V1 "caliper.export.v1"

/**
 * Bit 0 of `caps()`, which means that `view_png` and the sequence entries are live. It is set if and only if the
 * geometry primitives capability `CALIPER_GEOM_CAP_PRIMITIVES` is live on the host. Without it, every entry is inert
 * and writes no file.
 */
#define CALIPER_EXPORT_CAP_VIEW_PNG (1u << 0)

/** The largest width or height that export accepts, inclusive. A dimension of 0 or above this is refused. */
#define CALIPER_EXPORT_MAX_DIM 16384u

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The export service.
 *
 * Every entry uses the renderer that the frame thread owns, so call it from that thread only. The draw arguments
 * have the form of `draw_primitives` in `caliper.geometry.v1_3`, and the caller passes the same arrays that it draws
 * with. A refused call returns 0 and creates no file and truncates no existing file.
 */
typedef struct CaliperExportV1 {
    /** `sizeof(CaliperExportV1)` as the host built it. */
    uint32_t struct_size;
    /** Return the capability bits. Bit 0 is `CALIPER_EXPORT_CAP_VIEW_PNG`. */
    uint32_t (*caps)(void);
    /**
     * Render `draws` into a new offscreen target of `w` by `h` pixels and write the PNG to `path` and the
     * provenance sidecar to `<path>.json`. The PNG holds tightly packed RGBA8 pixels with row 0 at the top, and the
     * sidecar follows only after the PNG lands. Return 1 on success, or 0 on a refusal, which has the same gates and
     * reason strings as `draw_primitives`. A refusal comes from a missing gate or renderer, a `NULL` `cam`, a
     * dimension of 0 or above `CALIPER_EXPORT_MAX_DIM`, an unwritable path, a failed readback or a draw that the
     * geometry gates reject. `state_json` can be `NULL`, and the sidecar copies it verbatim into its "state" field.
     */
    uint32_t (*view_png)(const char* path, uint32_t w, uint32_t h,
                         const CaliperGeomCamera* cam,
                         const CaliperGeomDrawV1_3* draws,
                         uint32_t draw_count, uint32_t draw_stride,
                         uint32_t clear_rgba,
                         const char* state_json);
    /**
     * Start a frame sequence of `w` by `h` pixels in the directory `dir` and return a handle, or 0 on a refusal.
     * The frames are written as `<dir>/frame_%06u.png`, and `end_sequence` writes one sidecar for the whole
     * sequence as `<dir>/sequence.json` with the frame count. Only one sequence can be live, so a call while
     * another is active is refused. `state_json` can be `NULL`.
     */
    uint64_t (*begin_sequence)(const char* dir, uint32_t w, uint32_t h,
                               const char* state_json);
    /** Render one numbered frame of the sequence `seq` as `view_png` does. Return 1 on success, or 0 on a refusal. */
    uint32_t (*frame)(uint64_t seq, const CaliperGeomCamera* cam,
                      const CaliperGeomDrawV1_3* draws, uint32_t draw_count,
                      uint32_t draw_stride, uint32_t clear_rgba);
    /**
     * Finish the sequence `seq`, write its sidecar and drop the handle. The call returns nothing, so a failed
     * sidecar write only reaches the host log. The frames stay on disk, and the handle is dropped in any case.
     */
    void     (*end_sequence)(uint64_t seq);
    /** Reserved. Always `NULL`. */
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
