#pragma once
/* caliper.metrics.v1 — TensorBoard vocabulary (experiment/run/tag/step),
 * ImPlot immediacy (PLATFORM.md §7.6). IMMUTABLE once published. Callable
 * from applet job threads; the host serializes internally. image() accepts
 * CPU-resident HWC u8 tensors in v1 (GPU-resident paths arrive with the
 * tensor bridge). */
#include <stdint.h>
#include <caliper/tensor.h>

#define CALIPER_METRICS_V1 "caliper.metrics.v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperMetricsV1 {
    uint32_t struct_size;
    uint64_t (*begin_run)(const char* experiment, const char* run_name); /* 0 = error */
    void     (*end_run)(uint64_t run);
    void     (*scalar)(uint64_t run, const char* tag, int64_t step, double value);
    void     (*histogram)(uint64_t run, const char* tag, int64_t step,
                          const float* values, int64_t count);
    void     (*image)(uint64_t run, const char* tag, int64_t step,
                      const CaliperTensor* hwc_u8);
    void     (*hparams_json)(uint64_t run, const char* json_utf8);
} CaliperMetricsV1;

#ifdef __cplusplus
}
#endif
