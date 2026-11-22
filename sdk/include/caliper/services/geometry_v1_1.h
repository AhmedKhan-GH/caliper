#pragma once
/* caliper.geometry.v1_1 - additive general-primitives revision of caliper.geometry.v1. The v1 prefix is
 * frozen. Views still live in the tensor-bridge texture table, and sources are imported allocations from
 * tensor_bridge.v1_2. The revision appends one atomic multi-draw entry point for points, lines, and
 * triangles, with optional depth and a fixed shading and blending menu. */
#include <caliper/services/geometry_v1.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_GEOMETRY_V1_1 "caliper.geometry.v1_1"

/** Bit 1 of `caps()`, which says that `create_view_ex` and `draw_primitives` are live. It implies nothing about bit 0. */
#define CALIPER_GEOM_CAP_PRIMITIVES (1u << 1)

/* create_view_ex flags */
/** Flag for `create_view_ex` that attaches an internal D32-float depth buffer to the view. */
#define CALIPER_GEOM_VIEW_DEPTH (1u << 0)

/* CaliperGeomDraw.topology */
/** Value of `topology` that draws each vertex as a point. */
#define CALIPER_GEOM_TOPO_POINTS         0u
/** Value of `topology` that draws each pair of vertices as a separate line. */
#define CALIPER_GEOM_TOPO_LINES          1u
/** Value of `topology` that draws one connected line through the vertices. */
#define CALIPER_GEOM_TOPO_LINE_STRIP     2u
/** Value of `topology` that draws each triple of vertices as a separate triangle. */
#define CALIPER_GEOM_TOPO_TRIANGLES      3u
/** Value of `topology` that draws one strip of triangles in which each vertex adds a triangle. */
#define CALIPER_GEOM_TOPO_TRIANGLE_STRIP 4u

/* CaliperGeomDraw.color_mode */
/** Value of `color_mode` that gives every vertex the color `flat_rgba`, unpacked from little-endian bytes and divided by 255. */
#define CALIPER_GEOM_COLOR_FLAT        0u
/** Value of `color_mode` that colormaps one f32 attribute per vertex from `attr` through the LUT `colormap` over `[vmin,vmax]`. */
#define CALIPER_GEOM_COLOR_COLORMAP    1u
/** Value of `color_mode` that reads one packed little-endian `r | g<<8 | b<<16 | a<<24` u32 color per vertex from `attr`, unpacked and divided by 255. */
#define CALIPER_GEOM_COLOR_VERTEX_RGBA 2u

/* CaliperGeomDraw.shade_mode */
/** Value of `shade_mode` that leaves the color as computed. */
#define CALIPER_GEOM_SHADE_UNLIT   0u
/** Value of `shade_mode` that lights the draw per vertex with one headlight in Gouraud style, which requires `normal_alloc` to be set. */
#define CALIPER_GEOM_SHADE_LAMBERT 1u

/* CaliperGeomDraw.blend_mode */
/** Value of `blend_mode` that disables blending. */
#define CALIPER_GEOM_BLEND_OPAQUE   0u
/** Value of `blend_mode` for alpha blending, with the color factors `SRC_ALPHA` and `ONE_MINUS_SRC_ALPHA` and the alpha factors `ONE` and `ONE_MINUS_SRC_ALPHA`. */
#define CALIPER_GEOM_BLEND_ALPHA    1u
/** Value of `blend_mode` for additive blending, with the factors `ONE` and `ONE` on both channels, which matches the look of the v1 points. */
#define CALIPER_GEOM_BLEND_ADDITIVE 2u

/* CaliperGeomDraw.depth_flags */
/** Bit flag of `depth_flags` that turns the depth test on, with a `LESS_OR_EQUAL` compare. */
#define CALIPER_GEOM_DEPTH_TEST  (1u << 0)
/** Bit flag of `depth_flags` that writes depth. */
#define CALIPER_GEOM_DEPTH_WRITE (1u << 1)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * One draw call of `draw_primitives`, a fixed-layout record of 192 bytes.
 *
 * Geometry sources are pairs of an imported allocation id and a byte offset into a `tensor_bridge.v1_2` import.
 * An allocation id of 0 means the source is absent, and all offsets are 4-byte aligned. The clip-space convention
 * for applet-owned camera math is +Y up and Z in `[0,1]`.
 */
typedef struct CaliperGeomDraw {
    /** The imported allocation that holds the positions, a contiguous `(vertex_count,3)` f32 array. */
    CaliperAllocId pos_alloc;
    /** The byte offset of the positions inside `pos_alloc`. */
    uint64_t       pos_offset;
    /** The number of vertices, which must be greater than 0. */
    uint64_t       vertex_count;
    /** The imported allocation that holds the indices, or 0 for an unindexed draw. */
    CaliperAllocId index_alloc;
    /** The byte offset of the indices inside `index_alloc`. */
    uint64_t       index_offset;
    /** The number of indices, which the draw reads as u32 bit patterns when `index_alloc` is not 0. */
    uint64_t       index_count;
    /** The imported allocation that holds the normals, a `(vertex_count,3)` f32 array that `SHADE_LAMBERT` requires. */
    CaliperAllocId normal_alloc;
    /** The byte offset of the normals inside `normal_alloc`. */
    uint64_t       normal_offset;
    /** The imported allocation that holds one attribute per vertex, which `color_mode` interprets. */
    CaliperAllocId attr_alloc;
    /** The byte offset of the attribute inside `attr_alloc`. */
    uint64_t       attr_offset;

    /** The topology that assembles the vertices, one of the `CALIPER_GEOM_TOPO_*` values. */
    uint32_t topology;
    /** The source of the vertex colors, one of the `CALIPER_GEOM_COLOR_*` values. */
    uint32_t color_mode;
    /** The lighting, one of the `CALIPER_GEOM_SHADE_*` values. */
    uint32_t shade_mode;
    /** The blend equation, one of the `CALIPER_GEOM_BLEND_*` values. */
    uint32_t blend_mode;
    /** The depth bits, a set of the `CALIPER_GEOM_DEPTH_*` flags, which are valid only on a view with depth. */
    uint32_t depth_flags;
    /** The color of every vertex under `COLOR_FLAT`, packed little-endian as `r | g<<8 | b<<16 | a<<24`. */
    uint32_t flat_rgba;
    /** The id of the bridge colormap LUT that `COLOR_COLORMAP` uses, such as `CALIPER_CMAP_VIRIDIS`. */
    int32_t  colormap;
    /** The attribute value that maps to the first LUT entry under `COLOR_COLORMAP`. */
    float    vmin;
    /** The attribute value that maps to the last LUT entry under `COLOR_COLORMAP`. */
    float    vmax;
    /** The point size in pixels, which the service ignores for other topologies. */
    float    size_px;

    /**
     * The column-major model transform of the draw. Applets pass an identity matrix for world-space vertices, and
     * the C++ helper `caliper::geom_draw_defaults()` sets it. An all-zero matrix renders nothing.
     */
    float    model[16];

    /** Reserved fields that must be zero. The host refuses a draw with a nonzero value. */
    uint32_t reserved[2];
} CaliperGeomDraw;

/** The function table of `caliper.geometry.v1_1`. Its first five members are identical to `CaliperGeometryV1`. */
typedef struct CaliperGeometryV1_1 {
    /** `sizeof(CaliperGeometryV1_1)` as the host built it. */
    uint32_t struct_size;
    /* v1-identical prefix. */
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

    /* v1_1 additions. */
    /**
     * Create an offscreen 3-D render target like `create_view`, with `flags` that choose its features. The one
     * flag, `CALIPER_GEOM_VIEW_DEPTH`, attaches a D32-float depth buffer that stays internal to the renderer. The
     * function returns 0 on failure, and the applet releases the view with `release_view`. A view without depth
     * makes the host refuse every `depth_flags` bit.
     */
    CaliperTextureId (*create_view_ex)(uint32_t width, uint32_t height,
                                       uint32_t flags);
    /**
     * Render one frame of `view` atomically. The call gates every draw, clears the color to `clear_rgba` and the
     * depth to 1.0 if the view has depth, and then encodes `draws[0..draw_count)` in array order into one render
     * pass. A `draw_count` of 0 is a pure clear. Every source of every draw is read in place, so the caller must
     * not rewrite a drawn slot and must drain the producer before it publishes, as for `draw_points`. The value
     * `draw_stride` is the caller's `sizeof(CaliperGeomDraw)`, and the host refuses a stride below its minimum.
     * The return value is `false` when a gate fails, and then the host draws and clears nothing and writes one
     * `caliper.log.v1` line that names the gate.
     */
    bool (*draw_primitives)(CaliperTextureId view,
                            const CaliperGeomCamera* cam,
                            const CaliperGeomDraw* draws, uint32_t draw_count,
                            uint32_t draw_stride,
                            uint32_t clear_rgba);

    /** A reserved slot that is `NULL` in v1_1. */
    void (*reserved0)(void);
} CaliperGeometryV1_1;

#ifdef __cplusplus
}
static_assert(sizeof(CaliperGeomDraw) == 192,
              "CaliperGeomDraw ABI size is frozen");
#endif
