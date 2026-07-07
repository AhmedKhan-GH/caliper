#pragma once
/* caliper.geometry.v1 — imported 3-D geometry: draw instanced points DIRECTLY
 * from an applet-exported device allocation (tensor_bridge.v1_2's imported
 * blocks) into an offscreen view texture. Zero copies of the point data: the
 * vertex stage reads simulation memory in place, per frame.
 *
 * Deliberately a NEW service, not a tensor_bridge revision: the bridge's
 * frozen identity is "a tensor becomes an image"; cameras and draw calls are
 * a different vocabulary. The two share id spaces on purpose:
 *   - create_view returns a CaliperTextureId in the SAME table the bridge
 *     uses — a view is drawable with ImGui::Image like any other texture;
 *   - point data is addressed as (CaliperAllocId, byte offset) — the v1.2
 *     import machinery, caches, gates, and lifecycle are reused as-is.
 *
 * v1 scope: instanced points only (built for particle clouds — additive
 *   blending, no depth). Meshes/lines are a later additive revision.
 * IMMUTABLE once published; violations return 0/false and emit a
 * caliper.log.v1 line — never a wrong image (the degradation ladder). */
#include <stdint.h>
#include <stdbool.h>
#include <caliper/services/tensor_bridge_v1.h>    /* CaliperTextureId */
#include <caliper/services/tensor_bridge_v1_2.h>  /* CaliperAllocId   */

#define CALIPER_GEOMETRY_V1 "caliper.geometry.v1"

/* caps() bit 0: create_view/draw_points are live (renderer has the imported-
 * geometry path — Vulkan with a UUID-paired CUDA device today). Absent bit:
 * every entry point is inert and the applet keeps its CPU fallback. */
#define CALIPER_GEOM_CAP_IMPORTED_POINTS (1u << 0)

#ifdef __cplusplus
extern "C" {
#endif

/* Column-major 4x4 view and projection, applet-owned math (the service does
 * no camera logic — orbit/zoom/ray-casting are UI and live in the applet). */
typedef struct CaliperGeomCamera {
    float view[16];
    float proj[16];
} CaliperGeomCamera;

typedef struct CaliperGeometryV1 {
    uint32_t struct_size;
    uint32_t (*caps)(void);

    /* Offscreen 3-D render target. The returned id lives in the tensor-bridge
     * texture table: cast it to ImTextureID for ImGui::Image, release it here
     * (not via the bridge). 0 on failure. */
    CaliperTextureId (*create_view)(uint32_t width, uint32_t height);
    void (*release_view)(CaliperTextureId view);

    /* Render ONE frame of `view`, atomically: clear to clear_rgba (packed
     * little-endian r|g<<8|b<<16|a<<24), then draw `count` points whose
     * positions are a contiguous (count,3) f32 array at pos_offset inside the
     * imported allocation pos_alloc. count == 0 is a pure clear.
     *
     * attr_alloc != 0 selects a contiguous (count,) f32 scalar per point at
     * attr_offset, colormapped through the tensor-bridge LUTs over
     * [vmin,vmax] (same index rule as texture_from_tensor_mapped);
     * attr_alloc == 0 draws flat white and ignores attr_offset/colormap.
     *
     * size_px: point size in pixels (clamped to device limits). Points blend
     * ADDITIVELY with no depth test (v1 — built for particle clouds; order-
     * independent, no sort).
     *
     * Memory-stability contract (same as update_texture_from_alloc): the
     * addressed bytes are read IN PLACE and must not be rewritten until this
     * view's next draw. Gates: live view/allocations only, 4-byte-aligned
     * offsets, overflow-safe bounds. false = nothing drawn, the view keeps
     * its prior pixels. */
    bool (*draw_points)(CaliperTextureId view,
                        const CaliperGeomCamera* cam,
                        CaliperAllocId pos_alloc, uint64_t pos_offset,
                        uint64_t count,
                        CaliperAllocId attr_alloc, uint64_t attr_offset,
                        int32_t colormap, float vmin, float vmax,
                        float size_px, uint32_t clear_rgba);
} CaliperGeometryV1;

#ifdef __cplusplus
}
#endif
