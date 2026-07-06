#pragma once
/* caliper.tensor_bridge.v1_1 — ADDITIVE revision of tensor_bridge.v1 (D24,
 * docs/metal-pipelining.md §4): the SAME six operations, prefix-identical
 * layout, plus one query — caps(). Bit 0 set means the host honors
 * stream-ordered handoff: a non-NULL CaliperTensor.stream orders the device
 * update on the producer's stream/queue (CUstream on CUDA, MTLCommandQueue*
 * on Metal), so the adapter may SKIP its full device drain. Hosts that don't
 * vend this id keep the v1 contract: adapters drain, stream stays NULL. The
 * v1 header, table, and id are untouched (frozen); no ABI epoch bump. */
#include <caliper/services/tensor_bridge_v1.h>

#define CALIPER_TENSOR_BRIDGE_V1_1 "caliper.tensor_bridge.v1_1"

/* caps() bit 0: non-NULL CaliperTensor.stream is honored — producer-stream
 * GPU ordering replaces the adapter's device drain. Adapters must treat a
 * missing bit (or a missing v1_1 service) as "drain as v1". */
#define CALIPER_BRIDGE_CAP_STREAM_ORDERED (1u << 0)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperTensorBridgeV1_1 {
    uint32_t struct_size;
    /* v1-identical prefix — same semantics as CaliperTensorBridgeV1. */
    CaliperTextureId (*texture_from_tensor)(const CaliperTensor* t, uint32_t flags);
    bool (*update_texture)(CaliperTextureId tex, const CaliperTensor* t);
    void (*release_texture)(CaliperTextureId tex);
    CaliperTextureId (*texture_from_tensor_mapped)(const CaliperTensor* t,
                                                   int32_t colormap,
                                                   float vmin, float vmax,
                                                   uint32_t flags);
    bool (*alloc_shared)(CaliperDType dtype, int32_t ndim, const int64_t* shape,
                         CaliperTensor* out_tensor, CaliperTextureId* out_texture);
    void (*free_shared)(CaliperTextureId tex);
    /* v1.1 addition: capability bits (CALIPER_BRIDGE_CAP_*). */
    uint32_t (*caps)(void);
} CaliperTensorBridgeV1_1;

#ifdef __cplusplus
}
#endif
