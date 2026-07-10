# `caliper.geometry.v1_2`: textured imported geometry

**Date:** 2026-07-10  
**Status:** implementation contract  
**Requirement source:** `2026-07-10-twinscope-twin-exemplar-design.md`, R2

## Scope

`geometry.v1_2` lets an imported mesh sample an existing tensor-bridge texture
through imported UV coordinates. It adds no service entry point and no texture
creation or synchronization vocabulary. TwinScope is the first consumer.

The revision deliberately does not include materials, configurable samplers,
mipmaps, texture arrays, render-target sampling, or instancing.

## ABI

The v1.1 `CaliperGeomDraw` prefix is frozen at 192 bytes. Already-built v1.1
applets pass that size as `draw_stride`, so its C declaration is not changed.
The v1.2 header defines the appended record:

```c
typedef struct CaliperGeomDrawV1_2 {
    CaliperGeomDraw base;       /* frozen 192-byte v1.1 prefix */
    CaliperAllocId uv_alloc;    /* contiguous (vertex_count, 2) f32 */
    uint64_t uv_offset;         /* bytes, 4-byte aligned */
    CaliperTextureId texture;   /* bridge texture id, not a geometry view */
} CaliperGeomDrawV1_2;
```

Its size is 216 bytes and its alignment is 8. `CaliperGeometryV1_2` has the
same table slots as v1.1. Its `draw_primitives` slot is typed for the extended
record but has the same binary calling convention and still receives
`draw_stride`. `reserved0` remains null.

The host vends both revisions. Calls through v1.1 accept a minimum stride of
192 and cannot request textured color. Calls through v1.2 accept a minimum
stride of 216. This keeps old binaries working while ensuring a v1.2 caller
cannot expose an absent tail.

Additions:

- service id `caliper.geometry.v1_2`
- caps bit `CALIPER_GEOM_CAP_TEXTURED` (`1u << 2`)
- color mode `CALIPER_GEOM_COLOR_TEXTURE` (`3u`)

## Validation

Validation remains atomic: one invalid draw refuses the whole frame and leaves
the target pixels untouched.

For `COLOR_TEXTURE`, in addition to all v1.1 gates:

- `uv_alloc` resolves to a live bridge-v1.2 imported allocation;
- `uv_offset` is 4-byte aligned;
- `uv_offset + vertex_count * 2 * sizeof(float)` is overflow-safe and within
  the imported allocation;
- `texture` resolves to a live tensor-bridge texture entry;
- `texture` must not name a geometry view, including the current target view;
- `attr_alloc` is ignored and need not be present;
- UV values are finite shader inputs; values outside `[0, 1]` are defined by
  clamp-to-edge sampling.

The host resolves both public ids before entering a renderer. Backends repeat
allocation liveness and bounds checks against their native objects before they
encode work. A backend never receives a public texture id.

## Rendering

The vertex stage pulls UVs by the same final vertex index used for positions,
normals, and attributes. The interpolated UV reaches a textured fragment
variant. Sampling is fixed to:

- normalized coordinates;
- bilinear minification and magnification;
- clamp-to-edge on U and V;
- base level only, with no mipmaps.

The sampled RGBA value is the draw color. `SHADE_LAMBERT` multiplies sampled
RGB by the existing `0.30 + 0.70 * max(N dot headlight, 0)` term and leaves
alpha unchanged. Existing blend and depth modes then apply unchanged.

Metal selects a textured fragment function and binds the bridge texture plus
a fixed sampler. Vulkan selects a textured fragment module and binds the
texture image view plus fixed sampler in an added descriptor binding. The
untextured pipelines stay byte-for-byte behaviorally unchanged.

## Synchronization

No new synchronization primitive is added. Texture updates already finish in
shader-readable layout/state before publication. A geometry draw samples the
same native texture object that ImGui would sample, on the renderer's existing
ordered queue. Vulkan includes fragment-shader reads in the existing update /
render dependencies. A geometry view is never sampleable, preventing feedback
loops by construction.

## Verification rows

The same rows run against Metal and Vulkan:

1. ABI: v1.1 prefix offsets and size remain frozen; v1.2 tail offsets and size
   are exact; service ids and caps are exact.
2. Bridge gates: unknown/released UV allocation, UV misalignment, UV overflow,
   unknown/released texture, geometry-view-as-texture, short v1.2 stride, and
   mixed valid/invalid multi-draw all refuse atomically.
3. UV pull: a nonzero UV byte offset selects the intended coordinates.
4. Sampling: exact texel-center samples are byte-exact and out-of-range UVs
   clamp to the edge.
5. Filtering: the center of a 2x2 texture is the bilinear average within one
   RGBA8 LSB.
6. Shading: Lambert times texture matches the existing lighting equation
   within two RGB LSB; alpha is unchanged.
7. Compatibility: an unchanged v1.1 draw and a v1.2 non-textured draw render
   identically.

