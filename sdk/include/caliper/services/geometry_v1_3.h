#pragma once
/* caliper.geometry.v1_3 - instanced imported geometry. The v1.1 draw record is
 * a frozen 192-byte ABI prefix and v1.2 a frozen 216-byte record; v1.3 appends
 * an instance tail ((N,16) f32 poses + optional (N,) f32 tint) in a new record
 * carried by the existing draw_primitives + draw_stride slot. Pure additive
 * struct growth in the exact shape v1.2 used to grow from v1.1. */
#include <caliper/services/geometry_v1_2.h>
#include <stddef.h>

#define CALIPER_GEOMETRY_V1_3 "caliper.geometry.v1_3"

/* caps() bit 3: instanced draws are live. */
#define CALIPER_GEOM_CAP_INSTANCED (1u << 3)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperGeomDrawV1_3 {
    CaliperGeomDrawV1_2 base;           /* frozen 216-byte v1.2 record */
    CaliperAllocId instance_alloc;      /* (N,16) f32 column-major model matrices */
    uint64_t       instance_offset;     /* bytes, 4-byte aligned */
    uint64_t       instance_count;      /* N; 0 or instance_alloc==0 -> non-instanced */
    CaliperAllocId instance_attr_alloc; /* optional (N,) f32; 0 = no per-instance tint */
    uint64_t       instance_attr_offset;/* bytes, 4-byte aligned */
} CaliperGeomDrawV1_3;

typedef struct CaliperGeometryV1_3 {
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

    /* Same slots as v1.2; only the draw record type and minimum stride grow. */
    CaliperTextureId (*create_view_ex)(uint32_t width, uint32_t height,
                                       uint32_t flags);
    bool (*draw_primitives)(CaliperTextureId view,
                            const CaliperGeomCamera* cam,
                            const CaliperGeomDrawV1_3* draws,
                            uint32_t draw_count, uint32_t draw_stride,
                            uint32_t clear_rgba);

    void (*reserved0)(void);  /* remains NULL */
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
