#pragma once
/* caliper/tensor.h: the tensor interchange type, not a service. It is aligned with DLPack on purpose.
 * Frozen once shipped. It reuses CaliperDeviceKind for the memory domain. */
#include <stdint.h>
#include <caliper/services/device_v1.h>

#ifdef __cplusplus
extern "C" {
#endif

/** The element type of a tensor. */
typedef enum CaliperDType {
    /** A 32-bit float. */
    CALIPER_DT_F32 = 0,
    /** A 16-bit half float. */
    CALIPER_DT_F16 = 1,
    /** A 16-bit brain float. */
    CALIPER_DT_BF16 = 2,
    /** A 64-bit signed integer. */
    CALIPER_DT_I64 = 3,
    /** A 32-bit signed integer. */
    CALIPER_DT_I32 = 4,
    /** An 8-bit unsigned integer. */
    CALIPER_DT_U8 = 5
} CaliperDType;

/**
 * A block of memory with a dtype, a shape, strides, a device and an optional stream.
 *
 * The layout follows DLPack, so a torch, numpy or mlx tensor is a cast away. An applet passes a tensor into a service.
 * The struct does not own the memory, and no service returns one.
 */
typedef struct CaliperTensor {
    /** The value of `sizeof(CaliperTensor)` as the caller built it. */
    uint32_t          struct_size;
    /** The pointer to the first element, in device memory or in host memory. */
    void*             data;
    /** The element type. */
    CaliperDType      dtype;
    /** The number of dimensions, at most 8. */
    int32_t           ndim;
    /** The size of each dimension in elements. Only the first `ndim` values are used. */
    int64_t           shape[8];
    /** The distance between two neighbors in each dimension, in elements and not in bytes. */
    int64_t           strides[8];
    /** The memory domain that holds the data. */
    CaliperDeviceKind device;
    /** The index of the device within its kind. */
    int32_t           device_index;
    /** The `cudaStream_t` or `MTLCommandQueue*` that orders access to the data, or `NULL` for none. */
    void*             stream;
} CaliperTensor;

#ifdef __cplusplus
}
#endif
