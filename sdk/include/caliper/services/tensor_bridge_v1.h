#pragma once
/* caliper.tensor_bridge.v1 — the USP, productized (PLATFORM.md §7.4): a
 * CaliperTensor becomes a live texture this frame, GPU-resident on the native
 * backends (Metal buffer aliasing on MPS, Vulkan external-memory + CUDA import
 * on Windows), CPU-staged only on the GL fallback (§5.4). The ABI never names a
 * graphics API: textures cross as opaque CaliperTextureId, and the host keeps
 * an id -> backend-handle table so the renderer stays swappable forever.
 * IMMUTABLE once published; the host never links torch (D11) — the bridge
 * consumes CaliperTensor only.
 *
 * v1 acceptance rules (violations return 0/false and emit a caliper.log.v1 line
 * — never a misinterpreted texture):
 *   - 2-D (H,W) f32           -> texture_from_tensor_mapped (colormapped);
 *   - 3-D (H,W,C<=4) u8       -> texture_from_tensor (direct RGBA/…);
 *   - contiguous (row-major, no gaps);
 *   - device CPU or the active backend's device (e.g. Metal on macOS).
 * §16 contract: a tensor uploaded this way reads back pixel-exact vs a CPU
 * reference, per backend. */
#include <stdint.h>
#include <stdbool.h>
#include <caliper/tensor.h>

#define CALIPER_TENSOR_BRIDGE_V1 "caliper.tensor_bridge.v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t CaliperTextureId; /* Opaque to applets: 0 = invalid; compare-
                                      only — applets must never interpret or
                                      dereference it, its representation is
                                      backend-internal. The value is directly
                                      castable to ImTextureID for ImGui::Image
                                      (the host vends the ImGui-compatible handle
                                      per backend; §5.4). */

/* Built-in 256-entry RGBA8 colormap LUTs for 1-channel tensors; identical
 * numeric output on every backend. */
typedef enum CaliperColormap {
    CALIPER_CMAP_VIRIDIS = 0,
    CALIPER_CMAP_MAGMA   = 1,
    CALIPER_CMAP_RDBU    = 2
} CaliperColormap;

typedef struct CaliperTensorBridgeV1 {
    uint32_t struct_size;
    /* Mirror a 3-D (H,W,C<=4) u8 tensor as a texture. Native backends: Metal
       buffer aliasing on MPS, Vulkan external-memory + CUDA import on Windows —
       GPU-resident, zero-copy where layout permits, device-side blit otherwise.
       GL fallback: CPU-staged upload. Returns 0 on failure (reason via
       caliper.log.v1). */
    CaliperTextureId (*texture_from_tensor)(const CaliperTensor* t, uint32_t flags);
    /* Re-upload into an existing texture (same shape/dtype). false on failure. */
    bool (*update_texture)(CaliperTextureId tex, const CaliperTensor* t);
    void (*release_texture)(CaliperTextureId tex);
    /* Colormap a 2-D (H,W) f32 tensor through a built-in LUT, scaling
       [vmin,vmax] -> [0,1]. Returns 0 on failure (reason via caliper.log.v1). */
    CaliperTextureId (*texture_from_tensor_mapped)(const CaliperTensor* t,
                                                   int32_t colormap,
                                                   float vmin, float vmax,
                                                   uint32_t flags);
    /* Literal zero-copy: allocate tensor memory that IS the texture's backing
       store. The applet wraps out_tensor->data (torch::from_blob) and writes
       from kernels; the texture sees it after at most a layout transition.
       v1 returns a unified-memory CPU-device tensor: zero-copy for CPU writers;
       device writers stage via update_texture. false on failure. */
    bool (*alloc_shared)(CaliperDType dtype, int32_t ndim, const int64_t* shape,
                         CaliperTensor* out_tensor, CaliperTextureId* out_texture);
    void (*free_shared)(CaliperTextureId tex);
} CaliperTensorBridgeV1;

#ifdef __cplusplus
}
#endif
