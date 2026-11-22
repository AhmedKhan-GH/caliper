#pragma once
/* caliper.geometry.v1_2 - textured imported geometry. The v1_1 draw record is a frozen 192-byte prefix. This
 * revision appends UV and bridge-texture sources in a new record that the existing draw_primitives and
 * draw_stride slot carries. */
#include <caliper/services/geometry_v1_1.h>
#include <stddef.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_GEOMETRY_V1_2 "caliper.geometry.v1_2"

/** Bit 2 of `caps()`, which says that `COLOR_TEXTURE` draws are live. */
#define CALIPER_GEOM_CAP_TEXTURED (1u << 2)

/* CaliperGeomDrawV1_2.base.color_mode */
/** Value of `color_mode` that samples the bridge texture `texture` at the per-vertex UV coordinates. */
#define CALIPER_GEOM_COLOR_TEXTURE 3u

#ifdef __cplusplus
extern "C" {
#endif

/**
 * One draw call of `draw_primitives` in v1_2, a record of 216 bytes.
 *
 * The record is the frozen 192-byte `CaliperGeomDraw` followed by a texture tail. The service reads a UV pair per
 * vertex with the same final vertex index as positions, normals, and attributes. The sampler is fixed, with
 * normalized coordinates, linear filtering, clamp-to-edge addressing, and the base level only.
 */
typedef struct CaliperGeomDrawV1_2 {
    /** The frozen v1_1 record, which comes first. */
    CaliperGeomDraw base;
    /** The imported allocation that holds the UV pairs, a contiguous `(vertex_count,2)` f32 array. */
    CaliperAllocId uv_alloc;
    /** The byte offset of the UV pairs inside `uv_alloc`, a multiple of 4. */
    uint64_t uv_offset;
    /** The bridge texture that `COLOR_TEXTURE` samples. The host refuses a geometry view here. */
    CaliperTextureId texture;
} CaliperGeomDrawV1_2;

/** The function table of `caliper.geometry.v1_2`. It has the same slots as v1_1. */
typedef struct CaliperGeometryV1_2 {
    /** `sizeof(CaliperGeometryV1_2)` as the host built it. */
    uint32_t struct_size;
    /* v1-identical prefix. */
    /** Return the capability bits of the host, such as `CALIPER_GEOM_CAP_TEXTURED`. */
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

    /* Same slots as v1.1; only the draw record type and minimum stride grow. */
    /**
     * Create an offscreen 3-D render target like `create_view`, with `flags` that choose its features. The one
     * flag, `CALIPER_GEOM_VIEW_DEPTH`, attaches a D32-float depth buffer that stays internal to the renderer. The
     * function returns 0 on failure, and the applet releases the view with `release_view`. A view without depth
     * makes the host refuse every `depth_flags` bit.
     */
    CaliperTextureId (*create_view_ex)(uint32_t width, uint32_t height,
                                       uint32_t flags);
    /**
     * Render one frame of `view` atomically, as in v1_1, from an array of `CaliperGeomDrawV1_2` records. The value
     * `draw_stride` is the caller's `sizeof(CaliperGeomDrawV1_2)`, and the minimum stride is 216 bytes. A record
     * with a zero texture tail renders like the same v1_1 draw. The return value is `false` when a gate fails,
     * and then the host draws and clears nothing and writes one `caliper.log.v1` line that names the gate.
     */
    bool (*draw_primitives)(CaliperTextureId view,
                            const CaliperGeomCamera* cam,
                            const CaliperGeomDrawV1_2* draws,
                            uint32_t draw_count, uint32_t draw_stride,
                            uint32_t clear_rgba);

    /** A reserved slot that stays `NULL`. */
    void (*reserved0)(void);
} CaliperGeometryV1_2;

#ifdef __cplusplus
}
static_assert(sizeof(CaliperGeomDrawV1_2) == 216,
              "CaliperGeomDrawV1_2 ABI size is frozen");
static_assert(offsetof(CaliperGeomDrawV1_2, uv_alloc) == 192,
              "v1.2 fields must follow the frozen v1.1 prefix");
#endif
