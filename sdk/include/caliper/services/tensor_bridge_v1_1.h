#pragma once
/* caliper.tensor_bridge.v1_1 - additive revision of tensor_bridge.v1 (see
 * docs/development/compatibility-and-epochs.md). It keeps the six operations and the prefix-identical layout
 * and adds one query, caps(). The v1 header, table, and id stay frozen, and the ABI epoch does not change.
 * A host that does not vend this id keeps the v1 contract. */
#include <caliper/services/tensor_bridge_v1.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_TENSOR_BRIDGE_V1_1 "caliper.tensor_bridge.v1_1"

/**
 * Bit 0 of `caps()`, which says that the host honors a non-`NULL` `CaliperTensor.stream`. Ordering on the producer's
 * stream replaces the full device drain of the adapter. An adapter treats a missing bit or a missing v1_1 service
 * as "drain as v1".
 */
#define CALIPER_BRIDGE_CAP_STREAM_ORDERED (1u << 0)

#ifdef __cplusplus
extern "C" {
#endif

/** The function table of `caliper.tensor_bridge.v1_1`. It has the seven members listed here, and the first six are the v1 members. */
typedef struct CaliperTensorBridgeV1_1 {
    /** `sizeof(CaliperTensorBridgeV1_1)` as the host built it. */
    uint32_t struct_size;
    /* v1-identical prefix — same semantics as CaliperTensorBridgeV1. */
    /** Mirror a 3-D `(H,W,C<=4)` u8 tensor as a texture, as in v1. Return 0 on failure. */
    CaliperTextureId (*texture_from_tensor)(const CaliperTensor* t, uint32_t flags);
    /** Upload `t` again into the existing texture `tex`, which must have the same shape and dtype. Return `false` on failure. */
    bool (*update_texture)(CaliperTextureId tex, const CaliperTensor* t);
    /** Free the texture `tex` that `texture_from_tensor` or `texture_from_tensor_mapped` returned. */
    void (*release_texture)(CaliperTextureId tex);
    /** Colormap a 2-D `(H,W)` f32 tensor through the built-in table `colormap` over `[vmin,vmax]`, as in v1. Return 0 on failure. */
    CaliperTextureId (*texture_from_tensor_mapped)(const CaliperTensor* t,
                                                   int32_t colormap,
                                                   float vmin, float vmax,
                                                   uint32_t flags);
    /** Allocate tensor memory that is the backing store of a texture, as in v1. Return `false` on failure. */
    bool (*alloc_shared)(CaliperDType dtype, int32_t ndim, const int64_t* shape,
                         CaliperTensor* out_tensor, CaliperTextureId* out_texture);
    /** Free the memory and the texture that `alloc_shared` returned for `tex`. */
    void (*free_shared)(CaliperTextureId tex);
    /* v1.1 addition: capability bits (CALIPER_BRIDGE_CAP_*). */
    /** Return the capability bits of the host, such as `CALIPER_BRIDGE_CAP_STREAM_ORDERED`. */
    uint32_t (*caps)(void);
} CaliperTensorBridgeV1_1;

#ifdef __cplusplus
}
#endif
