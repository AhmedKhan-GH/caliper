#pragma once
/* caliper.metrics.v1_1 — additive READ revision of caliper.metrics.v1.
 *
 * metrics.v1 is WRITE-ONLY across the ABI (begin_run/scalar/histogram/image/
 * hparams_json) and IMMUTABLE — so a CONSUMER (the Compass app; embed.h's
 * caliper_core_get_service host) cannot list runs or read scalars through it.
 * v1_1 appends ONE read entry: query(), SQL → Arrow C stream, against the
 * metrics store's OWN live connection — the same DuckDB connection-under-one-
 * mutex the writers use, so a host UI thread may read WHILE an applet worker
 * streams metrics in and sees the freshly-written rows immediately (a separate
 * read-only DuckDB instance would see only CHECKPOINTed state — the reason a
 * host-layer read-only ATTACH was rejected for this surface; see the C0b
 * report). The frozen v1 writer prefix is byte-identical and unchanged.
 *
 * READ-ONLY, ENFORCED. Unlike caliper.data.v1.query (which runs any SQL against
 * a SEPARATE data.duckdb), this query() runs on the LIVE METRICS WRITER
 * connection, so it is parsed (not executed) and REFUSED unless it is exactly
 * one SELECT statement. This blocks statement chaining ("SELECT 1; DROP TABLE
 * runs") and any INSERT/UPDATE/DELETE/DDL that would corrupt the store. A
 * read-only introspection PRAGMA that DuckDB rewrites into a SELECT is allowed;
 * setter pragmas parse as non-SELECT and are refused.
 *
 * Callable from applet job threads AND from a host UI thread that is NOT the
 * caliper_core_frame() thread (metrics is an ANY-THREAD service per embed.h
 * §3.2) — the store serializes internally. IMMUTABLE once published; a future
 * read entry lands as metrics.v1_2.
 *
 * Stream ownership mirrors caliper.data.v1 EXACTLY (D3): on success the host
 * fills *out with a live ArrowArrayStream; the CALLER drains it (get_schema/
 * get_next until a released empty array) and MUST call out->release(out) once.
 * On failure (false) *out is untouched and last_error() explains — a host-owned
 * string valid until the next metrics.v1_1 call on the same thread. */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <caliper/tensor.h>
#include <caliper/arrow_c.h>
#include <caliper/services/metrics_v1.h>

#define CALIPER_METRICS_V1_1 "caliper.metrics.v1_1"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperMetricsV1_1 {
    uint32_t struct_size;
    /* --- v1-identical WRITER prefix (byte-for-byte CaliperMetricsV1). --- */
    uint64_t (*begin_run)(const char* experiment, const char* run_name); /* 0 = error */
    void     (*end_run)(uint64_t run);
    void     (*scalar)(uint64_t run, const char* tag, int64_t step, double value);
    void     (*histogram)(uint64_t run, const char* tag, int64_t step,
                          const float* values, int64_t count);
    void     (*image)(uint64_t run, const char* tag, int64_t step,
                      const CaliperTensor* hwc_u8);
    void     (*hparams_json)(uint64_t run, const char* json_utf8);

    /* --- v1_1 addition: the READ surface. --- */
    /* Run a single read-only SELECT against the metrics store; on success
     * results stream out as Arrow. Non-SELECT / multi-statement SQL is refused
     * (returns false, *out untouched, last_error set) — see header note. */
    bool (*query)(const char* sql_utf8, struct ArrowArrayStream* out);
    /* Why the last query() on THIS thread returned false; never NULL. */
    const char* (*last_error)(void);
} CaliperMetricsV1_1;

#ifdef __cplusplus
}
static_assert(sizeof(CaliperMetricsV1) == 56,
              "v1 metrics prefix drift would break the v1_1 read tail offsets");
static_assert(sizeof(CaliperMetricsV1_1) == 72,
              "CaliperMetricsV1_1 ABI size is frozen");
static_assert(offsetof(CaliperMetricsV1_1, struct_size) ==
              offsetof(CaliperMetricsV1, struct_size),
              "v1_1 opens with the frozen v1 struct_size");
static_assert(offsetof(CaliperMetricsV1_1, begin_run) ==
              offsetof(CaliperMetricsV1, begin_run));
static_assert(offsetof(CaliperMetricsV1_1, hparams_json) ==
              offsetof(CaliperMetricsV1, hparams_json),
              "v1_1 writer prefix must match v1 slot-for-slot");
static_assert(offsetof(CaliperMetricsV1_1, query) ==
              offsetof(CaliperMetricsV1, hparams_json) + sizeof(void*),
              "the v1_1 read tail must follow the frozen v1 writer prefix");
static_assert(offsetof(CaliperMetricsV1_1, last_error) ==
              offsetof(CaliperMetricsV1_1, query) + sizeof(void*));
#endif
