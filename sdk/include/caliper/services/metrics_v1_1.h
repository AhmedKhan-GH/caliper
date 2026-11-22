#pragma once
/* caliper.metrics.v1_1 — additive read revision of caliper.metrics.v1.
 *
 * The v1 table is write-only, so a consumer such as an embedding host cannot read runs through it. This revision
 * appends query(), which runs SQL on the live connection of the metrics store and streams the result as Arrow.
 * The frozen v1 writer prefix is byte-identical and unchanged.
 *
 * Callable from applet job threads and from a host UI thread that is not the caliper_core_frame() thread. The
 * store serializes internally. Immutable once published: a future read entry is metrics.v1_2. */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <caliper/tensor.h>
#include <caliper/arrow_c.h>
#include <caliper/services/metrics_v1.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_METRICS_V1_1 "caliper.metrics.v1_1"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The metrics log with a read entry.
 *
 * The table starts with the six writer entries of `CaliperMetricsV1`, byte for byte, and appends `query` and
 * `last_error`. Any thread except the frame thread can call every entry, and the host serializes the calls.
 *
 * On success, `query` fills `*out` with a live `ArrowArrayStream`. The caller drains it with `get_schema` and
 * `get_next` until an empty array that has been released, and must call `out->release(out)` exactly once.
 */
typedef struct CaliperMetricsV1_1 {
    /** `sizeof(CaliperMetricsV1_1)` as the host built it. */
    uint32_t struct_size;
    /* --- v1-identical writer prefix (byte-for-byte CaliperMetricsV1). --- */
    /**
     * Open a run of `experiment` and return its id, or 0 on error. 0 is never a valid run. The run id scopes every
     * scalar, histogram, image and hyperparameter that the applet logs afterward.
     */
    uint64_t (*begin_run)(const char* experiment, const char* run_name);
    /**
     * Close a run. Call it on every exit path of a job, so that a run never stays open. The points logged before
     * the end are kept.
     */
    void     (*end_run)(uint64_t run);
    /**
     * Append one point to the series `tag` of a run. The tag names the series, for example "train/loss", and a
     * slash groups tags into panes. `step` is the x value of the point and `value` is the y value.
     */
    void     (*scalar)(uint64_t run, const char* tag, int64_t step, double value);
    /** Log a distribution of `count` values at `step` under `tag`. */
    void     (*histogram)(uint64_t run, const char* tag, int64_t step,
                          const float* values, int64_t count);
    /**
     * Log a picture at `step` under `tag`. The tensor must be CPU-resident, contiguous and `u8` in HWC layout. The
     * host drops and logs any other tensor.
     */
    void     (*image)(uint64_t run, const char* tag, int64_t step,
                      const CaliperTensor* hwc_u8);
    /** Attach a flat JSON object of hyperparameters, for example `{"lr":0.001}`, to a run. */
    void     (*hparams_json)(uint64_t run, const char* json_utf8);

    /* --- v1_1 addition: the read surface. --- */
    /**
     * Run one read-only `SELECT` against the metrics store and stream the result as Arrow. The host parses the
     * text without running it and refuses anything that is not exactly one `SELECT`, so it returns `false`, leaves
     * `*out` untouched and sets the error. A read-only introspection `PRAGMA` that DuckDB rewrites into a `SELECT`
     * is allowed. The query sees rows that other threads logged a moment ago.
     */
    bool (*query)(const char* sql_utf8, struct ArrowArrayStream* out);
    /**
     * Return why the last `query` on this thread returned `false`. The host owns the string, which is never `NULL`
     * and stays valid until the next call of this service on the same thread.
     */
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
