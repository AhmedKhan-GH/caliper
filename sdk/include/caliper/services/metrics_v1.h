#pragma once
/* caliper.metrics.v1 — TensorBoard vocabulary (experiment, run, tag, step) with ImPlot immediacy
 * (docs/instruments/services/caliper-metrics-v1.md). Immutable once published. Callable from instrument job threads. */
#include <stdint.h>
#include <caliper/tensor.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_METRICS_V1 "caliper.metrics.v1"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The write-only metrics log.
 *
 * Every entry uses the vocabulary of experiment, run, tag and step. An instrument can call every entry from a job
 * thread, and the host serializes the calls internally.
 */
typedef struct CaliperMetricsV1 {
    /** `sizeof(CaliperMetricsV1)` as the host built it. */
    uint32_t struct_size;
    /**
     * Open a run of `experiment` and return its id, or 0 on error. 0 is never a valid run. The run id scopes every
     * scalar, histogram, image and hyperparameter that the instrument logs afterward.
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
} CaliperMetricsV1;

#ifdef __cplusplus
}
#endif
