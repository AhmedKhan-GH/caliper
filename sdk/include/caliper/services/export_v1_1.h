#pragma once
/* caliper.export.v1_1 — additive STATUS revision of caliper.export.v1.
 *
 * export.v1 answers 1/0 (a 0 handle from begin_sequence) and never says why,
 * and end_sequence() answers nothing at all: it returns void and the v1 vtable
 * is frozen (export_v1.h), so when a sequence's provenance sidecar
 * (<dir>/sequence.json) cannot be written the applet is never told — the host
 * can only log an ERROR line. v1_1 appends ONE entry: last_error(). The frozen
 * v1 prefix, reserved0 included, is byte-identical and unchanged; a v1 applet
 * keeps asking for caliper.export.v1 and gets the v1 table, unchanged.
 *
 * What counts as a failure: view_png / begin_sequence / frame returning 0, and
 * an end_sequence() whose sequence.json could not be written (the frames are
 * kept and the handle is finalized either way, exactly as in v1). After any of
 * those four calls succeeds, last_error() is "". caps() and last_error() leave
 * it alone.
 *
 * Same threading as v1: FRAME-THREAD ONLY. The string is host-owned, never
 * NULL, and valid until the next export call on the same thread (mirrors
 * caliper.data.v1 / caliper.metrics.v1_1 last_error). IMMUTABLE once
 * published; a future entry lands as export.v1_2. */
#include <stdint.h>
#include <stddef.h>
#include <caliper/services/export_v1.h>

#define CALIPER_EXPORT_V1_1 "caliper.export.v1_1"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperExportV1_1 {
    uint32_t struct_size;
    /* --- v1-identical prefix (byte-for-byte CaliperExportV1). --- */
    uint32_t (*caps)(void);
    uint32_t (*view_png)(const char* path, uint32_t w, uint32_t h,
                         const CaliperGeomCamera* cam,
                         const CaliperGeomDrawV1_3* draws,
                         uint32_t draw_count, uint32_t draw_stride,
                         uint32_t clear_rgba,
                         const char* state_json);
    uint64_t (*begin_sequence)(const char* dir, uint32_t w, uint32_t h,
                               const char* state_json);
    uint32_t (*frame)(uint64_t seq, const CaliperGeomCamera* cam,
                      const CaliperGeomDrawV1_3* draws, uint32_t draw_count,
                      uint32_t draw_stride, uint32_t clear_rgba);
    void     (*end_sequence)(uint64_t seq);
    void*    reserved0;                 /* still reserved: NULL */

    /* --- v1_1 addition: the STATUS surface. --- */
    /* Why the last view_png / begin_sequence / frame / end_sequence on THIS
     * thread failed (see header note); "" when it succeeded. Never NULL. */
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
