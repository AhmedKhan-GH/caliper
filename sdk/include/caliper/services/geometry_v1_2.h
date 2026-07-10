#pragma once
/* caliper.geometry.v1_2 - textured imported geometry. The v1.1 draw record is
 * a frozen 192-byte ABI prefix; v1.2 appends UV and bridge-texture sources in
 * a new record carried by the existing draw_primitives + draw_stride slot. */
#include <caliper/services/geometry_v1_1.h>
#include <stddef.h>

#define CALIPER_GEOMETRY_V1_2 "caliper.geometry.v1_2"

/* caps() bit 2: COLOR_TEXTURE draws are live. */
#define CALIPER_GEOM_CAP_TEXTURED (1u << 2)

/* CaliperGeomDrawV1_2.base.color_mode */
#define CALIPER_GEOM_COLOR_TEXTURE 3u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperGeomDrawV1_2 {
    CaliperGeomDraw base;       /* frozen v1.1 prefix */
    CaliperAllocId uv_alloc;    /* contiguous (vertex_count,2) f32 */
    uint64_t uv_offset;
    CaliperTextureId texture;   /* bridge texture id; views are refused */
} CaliperGeomDrawV1_2;

typedef struct CaliperGeometryV1_2 {
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

    /* Same slots as v1.1; only the draw record type and minimum stride grow. */
    CaliperTextureId (*create_view_ex)(uint32_t width, uint32_t height,
                                       uint32_t flags);
    bool (*draw_primitives)(CaliperTextureId view,
                            const CaliperGeomCamera* cam,
                            const CaliperGeomDrawV1_2* draws,
                            uint32_t draw_count, uint32_t draw_stride,
                            uint32_t clear_rgba);

    void (*reserved0)(void);  /* remains NULL */
} CaliperGeometryV1_2;

#ifdef __cplusplus
}
static_assert(sizeof(CaliperGeomDrawV1_2) == 216,
              "CaliperGeomDrawV1_2 ABI size is frozen");
static_assert(offsetof(CaliperGeomDrawV1_2, uv_alloc) == 192,
              "v1.2 fields must follow the frozen v1.1 prefix");
#endif
