#pragma once
/* caliper.geometry.v1_1 — additive general-primitives revision of
 * caliper.geometry.v1. The v1 prefix is frozen and unchanged: views still
 * live in the tensor-bridge texture id table, and geometry sources are
 * imported allocations from tensor_bridge.v1_2. v1_1 appends a single atomic
 * multi-draw entry point for points, lines, and triangles, with optional depth
 * and a fixed shading/blending menu. The ABI remains graphics-API-neutral. */
#include <caliper/services/geometry_v1.h>

#define CALIPER_GEOMETRY_V1_1 "caliper.geometry.v1_1"

/* caps() bit 1: create_view_ex / draw_primitives are live. */
#define CALIPER_GEOM_CAP_PRIMITIVES (1u << 1)

/* create_view_ex flags */
#define CALIPER_GEOM_VIEW_DEPTH (1u << 0)

/* CaliperGeomDraw.topology */
#define CALIPER_GEOM_TOPO_POINTS         0u
#define CALIPER_GEOM_TOPO_LINES          1u
#define CALIPER_GEOM_TOPO_LINE_STRIP     2u
#define CALIPER_GEOM_TOPO_TRIANGLES      3u
#define CALIPER_GEOM_TOPO_TRIANGLE_STRIP 4u

/* CaliperGeomDraw.color_mode */
#define CALIPER_GEOM_COLOR_FLAT        0u
#define CALIPER_GEOM_COLOR_COLORMAP    1u
#define CALIPER_GEOM_COLOR_VERTEX_RGBA 2u

/* CaliperGeomDraw.shade_mode */
#define CALIPER_GEOM_SHADE_UNLIT   0u
#define CALIPER_GEOM_SHADE_LAMBERT 1u

/* CaliperGeomDraw.blend_mode */
#define CALIPER_GEOM_BLEND_OPAQUE   0u
#define CALIPER_GEOM_BLEND_ALPHA    1u
#define CALIPER_GEOM_BLEND_ADDITIVE 2u

/* CaliperGeomDraw.depth_flags */
#define CALIPER_GEOM_DEPTH_TEST  (1u << 0)
#define CALIPER_GEOM_DEPTH_WRITE (1u << 1)

#ifdef __cplusplus
extern "C" {
#endif

/* Clip-space convention for applet-owned camera math: +Y up, Z in [0,1]. */
typedef struct CaliperGeomDraw {
    /* Sources are (imported alloc id, byte offset) pairs. Positions and
     * normals are contiguous (vertex_count,3) f32 arrays. Indices are u32
     * bit patterns. Attributes are either f32 scalar values for COLORMAP or
     * packed little-endian RGBA8 u32 values for VERTEX_RGBA. */
    CaliperAllocId pos_alloc;    uint64_t pos_offset;
    uint64_t       vertex_count;
    CaliperAllocId index_alloc;  uint64_t index_offset;
    uint64_t       index_count;
    CaliperAllocId normal_alloc; uint64_t normal_offset;
    CaliperAllocId attr_alloc;   uint64_t attr_offset;

    uint32_t topology;
    uint32_t color_mode;
    uint32_t shade_mode;
    uint32_t blend_mode;
    uint32_t depth_flags;
    uint32_t flat_rgba;
    int32_t  colormap;
    float    vmin;
    float    vmax;
    float    size_px;

    /* Column-major model transform. Applets should use an identity matrix for
     * world-space vertices; the C++ helper caliper::geom_draw_defaults() sets
     * that up. */
    float    model[16];

    uint32_t reserved[2];  /* must be zero */
} CaliperGeomDraw;

typedef struct CaliperGeometryV1_1 {
    uint32_t struct_size;
    /* v1-identical prefix. */
    uint32_t (*caps)(void);
    CaliperTextureId (*create_view)(uint32_t width, uint32_t height);
    void (*release_view)(CaliperTextureId view);
    bool (*draw_points)(CaliperTextureId view,
                        const CaliperGeomCamera* cam,
                        CaliperAllocId pos_alloc, uint64_t pos_offset,
                        uint64_t count,
                        CaliperAllocId attr_alloc, uint64_t attr_offset,
                        int32_t colormap, float vmin, float vmax,
                        float size_px, uint32_t clear_rgba);

    /* v1_1 additions. */
    CaliperTextureId (*create_view_ex)(uint32_t width, uint32_t height,
                                       uint32_t flags);
    /* Render one frame of `view` atomically. Every source (pos/index/normal/
     * attr, incl. the per-vertex COLORMAP attr) obeys the same two-half
     * memory-stability contract as draw_points (see geometry_v1.h): SPATIAL
     * (bytes read in place — don't rewrite a drawn slot) + TEMPORAL (drain the
     * producer BEFORE publishing — this ABI has no producer-stream channel, so
     * it is always the drain rung, never STREAM_ORDERED). Any future added
     * source (e.g. an instanced (N,16) pose stream) inherits both halves.
     * draw_stride = the caller's sizeof(CaliperGeomDraw). */
    bool (*draw_primitives)(CaliperTextureId view,
                            const CaliperGeomCamera* cam,
                            const CaliperGeomDraw* draws, uint32_t draw_count,
                            uint32_t draw_stride,
                            uint32_t clear_rgba);

    void (*reserved0)(void);  /* NULL in v1_1; reserved for a future revision. */
} CaliperGeometryV1_1;

#ifdef __cplusplus
}
static_assert(sizeof(CaliperGeomDraw) == 192,
              "CaliperGeomDraw ABI size is frozen");
#endif
