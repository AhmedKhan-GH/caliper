#pragma once
/* caliper.tensor_bridge.v1_2 - additive revision of tensor_bridge.v1_1. It keeps the seven members and the
 * prefix-identical layout and adds three entry points for imported external allocations. The earlier headers,
 * tables, and ids stay frozen, and the ABI epoch does not change. A host without caps bit 1 keeps the v1 and
 * v1_1 contract, the device-to-device copy path. */
#include <caliper/services/tensor_bridge_v1_1.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_TENSOR_BRIDGE_V1_2 "caliper.tensor_bridge.v1_2"

/**
 * Bit 1 of `caps()`, which says that `import_allocation` and `update_texture_from_alloc` are live. The host can then
 * import an instrument-exported device allocation and update textures from it with no copy of the tensor data.
 */
#define CALIPER_BRIDGE_CAP_IMPORT_ALLOC (1u << 1)

/* OS handle types accepted by import_allocation. */
/** The `handle_type` of a Windows opaque shareable handle. */
#define CALIPER_ALLOC_HANDLE_OPAQUE_WIN32 1u
/** The `handle_type` of a POSIX file descriptor as an opaque shareable handle. */
#define CALIPER_ALLOC_HANDLE_OPAQUE_FD    2u
/**
 * The `handle_type` for which `os_handle` is an in-process `id<MTLBuffer>` on Apple unified memory. No OS handle
 * transfers, and the duplicate that the host makes is an Objective-C strong retain.
 */
#define CALIPER_ALLOC_HANDLE_MTLBUFFER    3u

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The id of an imported allocation. The value 0 is invalid, and an id is compare-only and internal to the host.
 */
typedef uint64_t CaliperAllocId;

/** The function table of `caliper.tensor_bridge.v1_2`. Its first seven members are those of v1_1. */
typedef struct CaliperTensorBridgeV1_2 {
    /** `sizeof(CaliperTensorBridgeV1_2)` as the host built it. */
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
    /* v1.1-identical member. */
    /** Return the capability bits of the host, such as `CALIPER_BRIDGE_CAP_IMPORT_ALLOC`. */
    uint32_t (*caps)(void);
    /* v1.2 additions. */
    /**
     * Import the device allocation that `os_handle` names and return its id. The handle comes from
     * `cuMemExportToShareableHandle`, or it is an `id<MTLBuffer>` for `CALIPER_ALLOC_HANDLE_MTLBUFFER`. The value
     * `size_bytes` is the byte size of the allocation, and `handle_type` is a `CALIPER_ALLOC_HANDLE_*` value. The
     * function returns 0 when the host cannot import, which means a missing cap, a bad handle, or no hardware pair,
     * and the instrument then stays on the v1 path. The host duplicates the handle, and the instrument keeps ownership of
     * its copy.
     */
    CaliperAllocId (*import_allocation)(void* os_handle, uint64_t size_bytes,
                                        uint32_t handle_type);
    /** Free the import `alloc`. */
    void (*release_allocation)(CaliperAllocId alloc);
    /**
     * Update the existing texture `tex` from tensor bytes that live inside the imported allocation `alloc` at
     * `offset_bytes`. Create the texture first with `texture_from_tensor` or `texture_from_tensor_mapped`. The
     * argument `desc` describes shape, dtype, strides, and stream, and the host ignores `desc->data`. The call
     * applies the acceptance gates of `update_texture` and returns `false` when the texture is not updated, and
     * then the caller falls back. The pass reads the imported bytes in place, so the range from `offset_bytes` to
     * `offset_bytes` plus the extent must not change until the next update of the texture or the next frame of the
     * instrument. Device ordering covers producer writes that come before the call and not writes that come after it.
     */
    bool (*update_texture_from_alloc)(CaliperTextureId tex, CaliperAllocId alloc,
                                      uint64_t offset_bytes,
                                      const CaliperTensor* desc);
} CaliperTensorBridgeV1_2;

#ifdef __cplusplus
}
#endif
