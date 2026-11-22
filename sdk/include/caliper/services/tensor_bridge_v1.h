#pragma once
/* caliper.tensor_bridge.v1 - a CaliperTensor becomes a live texture in the current frame. The texture stays
 * GPU-resident on the native backends and is staged through the CPU only on the OpenGL fallback. The ABI never
 * names a graphics API. Textures cross as opaque CaliperTextureId values, and the host keeps a table from each
 * id to a backend handle. The service is frozen once published, and the host never links torch.
 *
 * A violation of the acceptance rules returns 0 or false and writes one caliper.log.v1 line, and it never
 * produces a misinterpreted texture. The acceptance rules are in docs/applets/services/caliper-tensor-bridge-v1.md. */
#include <stdint.h>
#include <stdbool.h>
#include <caliper/tensor.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_TENSOR_BRIDGE_V1 "caliper.tensor_bridge.v1"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The id of a host texture.
 *
 * The value 0 is invalid, and an id is compare-only. Applets never interpret or dereference an id, because its
 * representation is internal to the backend. The value casts directly to `ImTextureID` for `ImGui::Image`, because
 * the host hands out the ImGui-compatible handle of each backend.
 */
typedef uint64_t CaliperTextureId;

/* Built-in 256-entry RGBA8 colormap LUTs for 1-channel tensors; identical
 * numeric output on every backend. */
/**
 * The built-in colormap tables for 1-channel tensors. Each table has 256 RGBA8 entries and gives the same numbers
 * on every backend.
 */
typedef enum CaliperColormap {
    /** The Viridis table, id 0. */
    CALIPER_CMAP_VIRIDIS = 0,
    /** The Magma table, id 1. */
    CALIPER_CMAP_MAGMA   = 1,
    /** The RdBu diverging table, id 2. */
    CALIPER_CMAP_RDBU    = 2
} CaliperColormap;

/** The function table of `caliper.tensor_bridge.v1`. Call every entry from the frame thread. */
typedef struct CaliperTensorBridgeV1 {
    /** `sizeof(CaliperTensorBridgeV1)` as the host built it. */
    uint32_t struct_size;
    /**
     * Mirror a 3-D `(H,W,C<=4)` u8 tensor as a texture. The texture stays GPU-resident on the native backends,
     * with zero copy where the layout permits and a device-side blit otherwise. The OpenGL fallback uploads
     * through the CPU. The function returns 0 on failure and writes the reason to `caliper.log.v1`.
     */
    CaliperTextureId (*texture_from_tensor)(const CaliperTensor* t, uint32_t flags);
    /** Upload `t` again into the existing texture `tex`, which must have the same shape and dtype. Return `false` on failure. */
    bool (*update_texture)(CaliperTextureId tex, const CaliperTensor* t);
    /** Free the texture `tex` that `texture_from_tensor` or `texture_from_tensor_mapped` returned. */
    void (*release_texture)(CaliperTextureId tex);
    /**
     * Colormap a 2-D `(H,W)` f32 tensor through the built-in table `colormap`, which scales `[vmin,vmax]` to
     * `[0,1]`, and mirror the result as a texture. The function returns 0 on failure and writes the reason to
     * `caliper.log.v1`.
     */
    CaliperTextureId (*texture_from_tensor_mapped)(const CaliperTensor* t,
                                                   int32_t colormap,
                                                   float vmin, float vmax,
                                                   uint32_t flags);
    /**
     * Allocate tensor memory that is the backing store of a texture. The applet wraps `out_tensor->data`, for
     * example with `torch::from_blob`, and writes into it. The texture sees the result after at most a layout
     * transition. In v1 the function returns a unified-memory CPU-device tensor, which is zero-copy for CPU
     * writers, and device writers stage through `update_texture`. It returns `false` on failure.
     */
    bool (*alloc_shared)(CaliperDType dtype, int32_t ndim, const int64_t* shape,
                         CaliperTensor* out_tensor, CaliperTextureId* out_texture);
    /** Free the memory and the texture that `alloc_shared` returned for `tex`. */
    void (*free_shared)(CaliperTextureId tex);
} CaliperTensorBridgeV1;

#ifdef __cplusplus
}
#endif
