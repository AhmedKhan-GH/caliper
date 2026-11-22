#pragma once
/* caliper.geometry.v1 - imported 3-D geometry. The service draws instanced points directly from an
 * applet-exported device allocation into an offscreen view texture, with no copy of the point data.
 * A view lives in the tensor-bridge texture table, and point data is addressed as a
 * (CaliperAllocId, byte offset) pair. The service is frozen once published. A violation returns
 * 0 or false and writes one caliper.log.v1 line, and it never produces a wrong image. */
#include <stdint.h>
#include <stdbool.h>
#include <caliper/services/tensor_bridge_v1.h>    /* CaliperTextureId */
#include <caliper/services/tensor_bridge_v1_2.h>  /* CaliperAllocId   */

/** The id of this service, which `get_service` takes. */
#define CALIPER_GEOMETRY_V1 "caliper.geometry.v1"

/**
 * Bit 0 of `caps()`, which says that `create_view` and `draw_points` are live. The host sets it when the
 * renderer has the imported-geometry path. If the bit is absent, every entry point is inert and the applet keeps
 * its CPU fallback.
 */
#define CALIPER_GEOM_CAP_IMPORTED_POINTS (1u << 0)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The camera of one draw, as two column-major 4x4 matrices.
 *
 * The applet owns the camera math. The service does no camera logic, so orbit, zoom, and ray casting stay in the
 * applet.
 */
typedef struct CaliperGeomCamera {
    /** The column-major 4x4 view matrix, 16 floats. */
    float view[16];
    /** The column-major 4x4 projection matrix, 16 floats. */
    float proj[16];
} CaliperGeomCamera;

/** The function table of `caliper.geometry.v1`. */
typedef struct CaliperGeometryV1 {
    /** `sizeof(CaliperGeometryV1)` as the host built it. */
    uint32_t struct_size;
    /** Return the capability bits of the host, such as `CALIPER_GEOM_CAP_IMPORTED_POINTS`. */
    uint32_t (*caps)(void);

    /**
     * Create an offscreen 3-D render target of `width` by `height` physical pixels and return its id, or 0 on
     * failure. The id lives in the tensor-bridge texture table, so the applet casts it to `ImTextureID` for
     * `ImGui::Image`. The applet releases the view with `release_view` and not through the bridge.
     */
    CaliperTextureId (*create_view)(uint32_t width, uint32_t height);
    /** Release a view that `create_view` returned. Call it on the frame thread. */
    void (*release_view)(CaliperTextureId view);

    /**
     * Render one frame of `view` atomically. The call clears the view to `clear_rgba`, packed little-endian as
     * `r | g<<8 | b<<16 | a<<24`, and then draws `count` points. The positions are a contiguous `(count,3)` f32
     * array at `pos_offset` inside the imported allocation `pos_alloc`, and a `count` of 0 is a pure clear. If
     * `attr_alloc` is not 0, it selects a contiguous `(count,)` f32 scalar per point at `attr_offset`, which the
     * service colormaps through the tensor-bridge tables over `[vmin,vmax]` with the index rule of
     * `texture_from_tensor_mapped`. If `attr_alloc` is 0, the points are flat white and the service ignores
     * `attr_offset` and `colormap`. The value `size_px` is the point size in pixels, clamped to the device limits.
     * Points blend additively with no depth test, so the result does not depend on the order of the points. The
     * addressed bytes are read in place, so the caller must not rewrite them until the next draw of the view and
     * must finish its device writes before the call. The call needs a live view, live allocations, 4-byte-aligned
     * offsets, and in-bounds ranges. The return value is `false` when a gate fails, and then nothing is drawn and
     * the view keeps its earlier pixels.
     */
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
