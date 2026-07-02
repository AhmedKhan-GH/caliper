#pragma once
/* caliper.device.v1 — the host's negotiated compute device (PLATFORM.md
 * §7.3). IMMUTABLE once published. Kinds name the MEMORY/API DOMAIN, not a
 * framework backend: METAL covers torch-MPS, MLX, and ggml-Metal alike. The
 * host detects without linking any ML framework (D11); applets map the kind
 * to their framework's device (torch: METAL -> torch::kMPS). */
#include <stdint.h>

#define CALIPER_DEVICE_V1 "caliper.device.v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CaliperDeviceKind {
    CALIPER_DEV_CPU   = 0,
    CALIPER_DEV_CUDA  = 1,
    CALIPER_DEV_METAL = 2
} CaliperDeviceKind;

typedef struct CaliperDeviceV1 {
    uint32_t struct_size;
    CaliperDeviceKind (*kind)(void);
    int32_t           (*index)(void);             /* 0 for CPU/METAL */
    const char*       (*name)(void);              /* host-owned, e.g. "Apple M3 Max" */
    uint64_t          (*free_memory_hint)(void);  /* bytes, best-effort; 0 = unknown */
} CaliperDeviceV1;

#ifdef __cplusplus
}
#endif
