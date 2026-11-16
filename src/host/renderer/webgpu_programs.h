// webgpu_programs.h: the programs that the WebGPU renderer draws through the graphics layer, as
// caliper_gfx::ProgramDescs written by hand.
//
// TO BE REPLACED by the shader generator's table (src/host/gfx/shaders/generated/
// programs_webgpu.h) when it holds these two programs. They are the sprint 2 backend's own WGSL,
// moved behind the layer with their arithmetic unchanged, so that the pictures stay byte for byte
// what they were (design/renderer-core.md, Migration, step 4):
//
//  - kColormap, the heatmap's compute colormap by f32 thresholds that the CPU derives from the
//    reference map_f32_to_rgba8. Its parameters moved from a uniform buffer per texture to a
//    dynamic uniform in the frame's arena (group 0); the per-texture resources are group 1.
//    Migration step 8, the texture path, replaces it with the generated colormap_texture, whose
//    integer keys come from src/host/gfx/colormap_keys.h (design/renderer-core.md). Its workgroup is
//    8 x 8 instead of 16 x 16, because a compatibility-mode device allows 128 invocations in a
//    workgroup; every invocation computes one texel on its own, so no texel changes.
//  - kPoints, the points of the geometry path as instanced quads. Its positions and values come
//    from instance-step vertex buffers instead of storage buffers, because a compatibility-mode
//    device allows no storage buffer in a vertex stage, and its colour table moved into the draw's
//    uniform block in the list's arena. The value arrives as raw u32 bits, so NaN reaches the
//    colour rule unchanged; a flat-coloured draw binds a 16-byte constant at stride 0.
//
// Only WGSL is given: these programs run on the WebGPU backend alone, so their MSL and SPIR-V
// slots stay empty.
#pragma once

#include "gfx/gfx.h"

#include <cstdint>

namespace caliper_host::webgpu_programs {

using caliper_gfx::BindingDesc;
using caliper_gfx::BindingType;
using caliper_gfx::EntryDesc;
using caliper_gfx::Format;
using caliper_gfx::ProgramDesc;

// ---- the heatmap colormap ------------------------------------------------------------------------

inline constexpr const char* kColormapWgsl = R"wgsl(
struct CmapParams {
    w: u32, h: u32, sx: u32, sy: u32,
    vmin: f32, vmax: f32, mode: u32, pad1: u32,
};
@group(0) @binding(0) var<uniform> p: CmapParams;            // dynamic: the frame's arena
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read> lut: array<u32>;
@group(1) @binding(2) var<storage, read> th: array<f32>;     // T[1..255]
@group(1) @binding(3) var dst: texture_storage_2d<rgba8unorm, write>;

fn is_nan_bits(v: f32) -> bool {
    let b = bitcast<u32>(v);
    return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

fn index_of(v: f32) -> u32 {
    if (is_nan_bits(v) || p.mode == 0u) { return 0u; }
    // The first k in 1..255 whose threshold v has not passed; the index is k - 1.
    var lo = 1u;
    var hi = 256u;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2u;
        let passed = select(v <= th[mid], th[mid] <= v, p.mode == 1u);
        if (passed) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo - 1u;
}

@compute @workgroup_size(8, 8)
fn cmap_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let c = lut[index_of(src[gid.y * p.sy + gid.x * p.sx])];
    let rgba = vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu),
                         f32((c >> 16u) & 0xffu), f32((c >> 24u) & 0xffu)) / 255.0;
    textureStore(dst, vec2<i32>(gid.xy), rgba);
}
)wgsl";

struct CmapParams {   // the WGSL CmapParams, 32 bytes
    uint32_t w, h, sx, sy;
    float vmin, vmax;
    uint32_t mode, pad1;
};
static_assert(sizeof(CmapParams) == 32, "WGSL uniform layout");

inline constexpr EntryDesc kColormapEntries[] = {{"cmap_f32", caliper_gfx::kStageCompute, {8, 8, 1}}};
inline constexpr uint32_t kColormapGroup = 8;   // texels per workgroup in x and in y
inline constexpr BindingDesc kColormapBindings[] = {
    // group, binding, type, stages, dynamic, size (the minimum binding size), storage format
    {0, 0, BindingType::Uniform, caliper_gfx::kStageCompute, true, sizeof(CmapParams)},
    {1, 0, BindingType::Storage, caliper_gfx::kStageCompute, false, 4},
    {1, 1, BindingType::Storage, caliper_gfx::kStageCompute, false, 4},
    {1, 2, BindingType::Storage, caliper_gfx::kStageCompute, false, 4},
    {1, 3, BindingType::StorageTexture, caliper_gfx::kStageCompute, false, 0, Format::RGBA8Unorm},
};
inline constexpr ProgramDesc kColormap = {
    "caliper.cmap_f32", kColormapEntries, 1, kColormapBindings, 5, kColormapWgsl, nullptr, nullptr, 0};

// ---- the points of the geometry path -------------------------------------------------------------

// WebGPU has only one-pixel points, so each point is two triangles of size_px x size_px pixels
// around its projected centre (the vertex index picks the corner, the instance the point), the
// squares Metal and Vulkan draw with point_size. The colour rule is the native one: a value maps
// through the table with idx = clamp((v - vmin)/(vmax - vmin), 0, 1) * 255 + 0.5, NaN and a
// degenerate range taking index 0.
inline constexpr const char* kPointsWgsl = R"wgsl(
struct Params {
    mvp: mat4x4<f32>,
    color_mode: u32, flat_rgba: u32, vmin: f32, vmax: f32,
    size_px: f32, pad0: u32, viewport: vec2<f32>,
    lut: array<vec4<u32>, 64>,
};
@group(0) @binding(0) var<uniform> p: Params;   // dynamic: the draw's slot in the list's arena

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn unpack(c: u32) -> vec4<f32> {
    return vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu), f32((c >> 16u) & 0xffu),
                     f32((c >> 24u) & 0xffu)) / 255.0;
}

fn is_nan_bits(v: f32) -> bool {
    let b = bitcast<u32>(v);
    return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

@vertex
fn vs(@builtin(vertex_index) vid: u32, @location(0) pos: vec3<f32>, @location(1) attr: u32) -> VOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
                                      vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    var c = p.mvp * vec4<f32>(pos.x, pos.y, pos.z, 1.0);
    // Half the size in NDC is (size_px / 2) / (viewport / 2); times w in clip space.
    let k = corners[vid] * (p.size_px / p.viewport) * c.w;
    c = vec4<f32>(c.x + k.x, c.y + k.y, c.z, c.w);
    var col = unpack(p.flat_rgba);
    if (p.color_mode == 1u) {
        let v = bitcast<f32>(attr);
        var t = 0.0;
        if (!is_nan_bits(v) && p.vmax > p.vmin) { t = clamp((v - p.vmin) / (p.vmax - p.vmin), 0.0, 1.0); }
        let i = u32(t * 255.0 + 0.5);
        col = unpack(p.lut[i >> 2u][i & 3u]);
    }
    return VOut(c, col);
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> { return i.color; }
)wgsl";

struct PointParams {   // the WGSL Params, 1,120 bytes
    float mvp[16];
    uint32_t color_mode, flat_rgba;
    float vmin, vmax;
    float size_px;
    uint32_t pad0;
    float viewport[2];
    uint32_t lut[256];
};
static_assert(sizeof(PointParams) == 1120, "WGSL uniform layout");

inline constexpr EntryDesc kPointsEntries[] = {{"vs", caliper_gfx::kStageVertex, {1, 1, 1}},
                                              {"fs", caliper_gfx::kStageFragment, {1, 1, 1}}};
inline constexpr BindingDesc kPointsBindings[] = {
    {0, 0, BindingType::Uniform, caliper_gfx::kStageVertex, true, sizeof(PointParams)},
};
inline constexpr ProgramDesc kPoints = {
    "caliper.points", kPointsEntries, 2, kPointsBindings, 1, kPointsWgsl, nullptr, nullptr, 0};

}  // namespace caliper_host::webgpu_programs
