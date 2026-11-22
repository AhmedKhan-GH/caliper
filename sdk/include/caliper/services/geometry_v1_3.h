#pragma once
/* caliper.geometry.v1_3 - instanced imported geometry. The v1_1 and v1_2 draw records are frozen prefixes of
 * 192 and 216 bytes. This revision appends an instance tail in a new record that the existing draw_primitives
 * and draw_stride slot carries. It grows the record in the same way as v1_2. */
#include <caliper/services/geometry_v1_2.h>
#include <stddef.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_GEOMETRY_V1_3 "caliper.geometry.v1_3"

/** Bit 3 of `caps()`, which says that instanced draws are live. */
#define CALIPER_GEOM_CAP_INSTANCED (1u << 3)

/**
 * The rigidity tolerance of gate G14, a relative and dimensionless bound. An instanced `LAMBERT` draw is refused
 * unless the upper 3x3 block of every instance matrix is orthogonal up to a uniform scale within this bound. The
 * value is part of the byte-exact contract and not a tunable. Both backends run the same comparison in the same
 * float order against it.
 */
#define CALIPER_GEOM_RIGID_TOL 1e-4f

#ifdef __cplusplus
extern "C" {
#endif

/**
 * One draw call of `draw_primitives` in v1_3, a record of 256 bytes.
 *
 * The record is the frozen 216-byte `CaliperGeomDrawV1_2` followed by an instance tail. One draw then renders the
 * imported mesh once for each instance. If `instance_count` is 0 or `instance_alloc` is 0, the draw takes the
 * non-instanced path, and a record with a zero tail renders like the same v1_2 record.
 */
typedef struct CaliperGeomDrawV1_3 {
    /** The frozen v1_2 record, which comes first. */
    CaliperGeomDrawV1_2 base;
    /** The imported allocation that holds the instance poses, an `(N,16)` f32 array of column-major model matrices. */
    CaliperAllocId instance_alloc;
    /** The byte offset of the poses inside `instance_alloc`, a multiple of 4. */
    uint64_t       instance_offset;
    /** The number of instances N. A value of 0 selects the non-instanced path. */
    uint64_t       instance_count;
    /** The imported allocation that holds an optional `(N,)` f32 tint per instance, or 0 for no tint. */
    CaliperAllocId instance_attr_alloc;
    /** The byte offset of the tints inside `instance_attr_alloc`, a multiple of 4. */
    uint64_t       instance_attr_offset;
} CaliperGeomDrawV1_3;

/** The function table of `caliper.geometry.v1_3`. It has the same slots as v1_2. */
typedef struct CaliperGeometryV1_3 {
    /** `sizeof(CaliperGeometryV1_3)` as the host built it. */
    uint32_t struct_size;
    /* v1-identical prefix. */
    /** Return the capability bits of the host, such as `CALIPER_GEOM_CAP_INSTANCED`. */
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

    /* Same slots as v1.2; only the draw record type and minimum stride grow. */
    /**
     * Create an offscreen 3-D render target like `create_view`, with `flags` that choose its features. The one
     * flag, `CALIPER_GEOM_VIEW_DEPTH`, attaches a D32-float depth buffer that stays internal to the renderer. The
     * function returns 0 on failure, and the applet releases the view with `release_view`. A view without depth
     * makes the host refuse every `depth_flags` bit.
     */
    CaliperTextureId (*create_view_ex)(uint32_t width, uint32_t height,
                                       uint32_t flags);
    /**
     * Render one frame of `view` atomically, as in v1_2, from an array of `CaliperGeomDrawV1_3` records. The value
     * `draw_stride` is the caller's `sizeof(CaliperGeomDrawV1_3)`, and the minimum stride is 256 bytes. The
     * vertex stage applies the pose of each instance to the world position first. The return value is `false`
     * when a gate fails, and then the host draws and clears nothing and writes one `caliper.log.v1` line that
     * names the gate.
     */
    bool (*draw_primitives)(CaliperTextureId view,
                            const CaliperGeomCamera* cam,
                            const CaliperGeomDrawV1_3* draws,
                            uint32_t draw_count, uint32_t draw_stride,
                            uint32_t clear_rgba);

    /** A reserved slot that stays `NULL`. */
    void (*reserved0)(void);
} CaliperGeometryV1_3;

#ifdef __cplusplus
}
static_assert(sizeof(CaliperGeomDrawV1_2) == 216,
              "v1.2 prefix drift would break the v1.3 tail offsets");
static_assert(sizeof(CaliperGeomDrawV1_3) == 256,
              "CaliperGeomDrawV1_3 ABI size is frozen");
static_assert(offsetof(CaliperGeomDrawV1_3, base) == 0,
              "v1.3 record opens with the frozen v1.2 prefix");
static_assert(offsetof(CaliperGeomDrawV1_3, instance_alloc) == 216,
              "v1.3 instance tail must follow the frozen v1.2 record");
#endif
