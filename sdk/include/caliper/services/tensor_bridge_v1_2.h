#pragma once
/* caliper.tensor_bridge.v1_2 — ADDITIVE revision of tensor_bridge.v1_1 (same
 * D24 pattern): the SAME seven members, prefix-identical layout, plus three
 * entry points for imported external allocations. Caps bit 1 set means the
 * host can import an applet-exported device allocation (CUDA VMM shareable
 * handle) and run device texture updates directly FROM it — zero copies of
 * the tensor data. Hosts without the bit: applets keep the v1/v1.1 contract
 * (the D2D-copy interop path). The v1/v1.1 headers, tables, and ids are
 * untouched (frozen); no ABI epoch bump. */
#include <caliper/services/tensor_bridge_v1_1.h>

#define CALIPER_TENSOR_BRIDGE_V1_2 "caliper.tensor_bridge.v1_2"

/* caps() bit 1: import_allocation/update_texture_from_alloc are live. */
#define CALIPER_BRIDGE_CAP_IMPORT_ALLOC (1u << 1)

/* OS handle types accepted by import_allocation. */
#define CALIPER_ALLOC_HANDLE_OPAQUE_WIN32 1u
#define CALIPER_ALLOC_HANDLE_OPAQUE_FD    2u
/* void* is an in-process id<MTLBuffer> (Apple unified memory). No OS handle
 * transfer: the "dup" the host performs is an ObjC strong retain. Additive,
 * same discipline as the two kinds above. */
#define CALIPER_ALLOC_HANDLE_MTLBUFFER    3u

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque imported-allocation id; 0 = invalid. Compare-only, host-internal. */
typedef uint64_t CaliperAllocId;

typedef struct CaliperTensorBridgeV1_2 {
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
    /* v1.1-identical member. */
    uint32_t (*caps)(void);
    /* v1.2 additions. import_allocation: hand the host an OS shareable handle
     * (from cuMemExportToShareableHandle) plus the allocation's byte size;
     * returns 0 when the host cannot import (missing cap, bad handle, no
     * hardware pair) — the applet then stays on the v1 path. The host dups
     * the handle; the applet keeps ownership of its copy. */
    CaliperAllocId (*import_allocation)(void* os_handle, uint64_t size_bytes,
                                        uint32_t handle_type);
    void (*release_allocation)(CaliperAllocId alloc);
    /* Update an existing texture (create it first via texture_from_tensor*)
     * from tensor bytes living INSIDE an imported allocation at offset_bytes.
     * desc describes shape/dtype/strides/stream; desc->data is IGNORED (the
     * imported allocation + offset are the address). Same acceptance gates
     * as update_texture; false = not updated, caller falls back.
     * Memory-stability contract: the pass reads the imported bytes IN PLACE,
     * so [offset_bytes, offset_bytes + extent) must not be rewritten until
     * the next update of the same texture (or the applet's next frame) —
     * device-side ordering covers producer writes BEFORE the call, not
     * writes issued after it. */
    bool (*update_texture_from_alloc)(CaliperTextureId tex, CaliperAllocId alloc,
                                      uint64_t offset_bytes,
                                      const CaliperTensor* desc);
} CaliperTensorBridgeV1_2;

#ifdef __cplusplus
}
#endif
