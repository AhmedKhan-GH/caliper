#pragma once
/* caliper/tensor.h — the tensor interchange TYPE (PLATFORM.md §7.2), not a
 * service. DLPack-aligned on purpose: torch/numpy/mlx interop is a cast away.
 * FROZEN once shipped. Reuses CaliperDeviceKind (memory-domain naming). */
#include <stdint.h>
#include <caliper/services/device_v1.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CaliperDType {
    CALIPER_DT_F32 = 0, CALIPER_DT_F16 = 1, CALIPER_DT_BF16 = 2,
    CALIPER_DT_I64 = 3, CALIPER_DT_I32 = 4, CALIPER_DT_U8 = 5
} CaliperDType;

typedef struct CaliperTensor {
    uint32_t          struct_size;
    void*             data;            /* device or host pointer */
    CaliperDType      dtype;
    int32_t           ndim;            /* <= 8 */
    int64_t           shape[8];
    int64_t           strides[8];      /* in elements */
    CaliperDeviceKind device;
    int32_t           device_index;
    void*             stream;          /* cudaStream_t / MTLCommandQueue* / NULL */
} CaliperTensor;

#ifdef __cplusplus
}
#endif
