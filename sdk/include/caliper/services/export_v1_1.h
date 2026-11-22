#pragma once
/* caliper.export.v1_1 — additive status revision of caliper.export.v1.
 *
 * The v1 table answers 1 or 0 and never says why, and end_sequence() returns nothing. This revision appends
 * last_error(). The frozen v1 prefix, reserved0 included, is byte-identical, and a v1 applet keeps the v1 table.
 *
 * Threading is the same as in v1: call it from the frame thread only. Immutable once published: a future entry is
 * export.v1_2. */
#include <stdint.h>
#include <stddef.h>
#include <caliper/services/export_v1.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_EXPORT_V1_1 "caliper.export.v1_1"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The export service with a status entry.
 *
 * The table starts with the entries of `CaliperExportV1`, byte for byte, and appends `last_error`. Call every entry
 * from the frame thread only.
 *
 * A failure is a return of 0 from `view_png`, `begin_sequence` or `frame`, or an `end_sequence` that could not write
 * `<dir>/sequence.json`. When `end_sequence` fails to write the sidecar, it keeps the frames and still finalizes the
 * handle. A call of any of the four that succeeds sets the status to an empty string. `caps` and `last_error` leave
 * the status alone.
 */
typedef struct CaliperExportV1_1 {
    /** `sizeof(CaliperExportV1_1)` as the host built it. */
    uint32_t struct_size;
    /* --- v1-identical prefix (byte-for-byte CaliperExportV1). --- */
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
     * Finish the sequence `seq`, write its sidecar and drop the handle. The call returns nothing, so read
     * `last_error` to learn whether the sidecar write failed. The frames stay on disk, and the handle is dropped in
     * any case.
     */
    void     (*end_sequence)(uint64_t seq);
    /** Reserved. Always `NULL`. */
    void*    reserved0;

    /* --- v1_1 addition: the status surface. --- */
    /**
     * Return why the last `view_png`, `begin_sequence`, `frame` or `end_sequence` on this thread failed, or an empty
     * string if it succeeded. The host owns the string, which is never `NULL` and stays valid until the next export
     * call on the same thread.
     */
    const char* (*last_error)(void);
} CaliperExportV1_1;

#ifdef __cplusplus
}
static_assert(sizeof(CaliperExportV1) == 56,
              "v1 export prefix drift would break the v1_1 status tail offset");
static_assert(sizeof(CaliperExportV1_1) == 64,
              "CaliperExportV1_1 ABI size is frozen");
static_assert(offsetof(CaliperExportV1_1, struct_size) ==
              offsetof(CaliperExportV1, struct_size),
              "v1_1 opens with the frozen v1 struct_size");
static_assert(offsetof(CaliperExportV1_1, caps) ==
              offsetof(CaliperExportV1, caps));
static_assert(offsetof(CaliperExportV1_1, end_sequence) ==
              offsetof(CaliperExportV1, end_sequence),
              "v1_1 prefix must match v1 slot-for-slot");
static_assert(offsetof(CaliperExportV1_1, reserved0) ==
              offsetof(CaliperExportV1, reserved0));
static_assert(offsetof(CaliperExportV1_1, last_error) ==
              offsetof(CaliperExportV1, reserved0) + sizeof(void*),
              "the v1_1 status tail must follow the frozen v1 prefix");
#endif
