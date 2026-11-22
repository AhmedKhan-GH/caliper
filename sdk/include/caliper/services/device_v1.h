#pragma once
/* caliper.device.v1 — the host's negotiated compute device (docs/applets/services/caliper-device-v1.md).
 * Immutable once published. */
#include <stdint.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_DEVICE_V1 "caliper.device.v1"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The kind of a compute device.
 *
 * A kind names a memory and API domain, not a framework backend. `CALIPER_DEV_METAL` covers torch-MPS, MLX and
 * ggml-Metal alike. The host detects the device without linking any ML framework, and an applet maps the kind to
 * the device of its own framework.
 */
typedef enum CaliperDeviceKind {
    /** The CPU, which is the fallback when the host finds no other device. */
    CALIPER_DEV_CPU   = 0,
    /** An NVIDIA GPU that the CUDA driver reports. */
    CALIPER_DEV_CUDA  = 1,
    /** The Metal memory and API domain of Apple GPUs. */
    CALIPER_DEV_METAL = 2
} CaliperDeviceKind;

/**
 * The compute device that the host negotiated.
 *
 * Any function pointer can be `NULL` on an older host, so check each one before you call it.
 */
typedef struct CaliperDeviceV1 {
    /** `sizeof(CaliperDeviceV1)` as the host built it. */
    uint32_t struct_size;
    /** Return the kind of the device. */
    CaliperDeviceKind (*kind)(void);
    /** Return the index of the device. It is 0 for the CPU and for Metal, and the device ordinal for CUDA. */
    int32_t           (*index)(void);
    /** Return the name of the device, for example "Apple M3 Max". The host owns the string. */
    const char*       (*name)(void);
    /** Return a best-effort estimate of the free device memory in bytes. 0 means unknown, not no memory. */
    uint64_t          (*free_memory_hint)(void);
} CaliperDeviceV1;

#ifdef __cplusplus
}
#endif
