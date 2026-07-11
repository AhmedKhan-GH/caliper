#version 450
// caliper.geometry.v1_1 primitive pipeline: vertex-pulled from IMPORTED GPU
// buffers (points / lines / triangles), zero copies. This is the GLSL twin of
// the shipped Metal geom pair (metal_renderer.mm kGeomShaderSrc, geom_compute);
// the body mirrors it byte-for-byte — same order of float ops so both backends
// produce identical pixels. Color is computed HERE (the fragment stage is a
// pass-through of v_color). Colormap indexing uses the SAME index rule as the
// CPU reference map_f32_to_rgba8 and points.vert:
//   idx = floor(clamp((v-vmin)/(vmax-vmin),0,1)*255 + 0.5); NaN -> 0;
//   degenerate range (vmax <= vmin) -> 0.
// Indices are pulled as u32 and clamped against vertex_count so out-of-range
// values are defined (no OOB-read UB — GEOMETRY.md §2.3).

// Bindings (set 0) — matches the renderer's descriptor set layout.
layout(std430, set = 0, binding = 0) readonly buffer Pos  { float pos[];  };
layout(std430, set = 0, binding = 1) readonly buffer Idx  { uint  idx[];  };
layout(std430, set = 0, binding = 2) readonly buffer Nrm  { float nrm[];  };
layout(std430, set = 0, binding = 3) readonly buffer Attr { uint  attr[]; };
layout(std430, set = 0, binding = 4) readonly buffer Lut  { uint  lut[];  };
layout(std430, set = 0, binding = 6) readonly buffer UV   { float uv[];  };
// v1.3 instance streams (§6.2): binding 8 the (N,16) column-major model
// matrices as float[]; binding 9 the (N,) per-instance tint scalars bound as
// uint[] so NaN bit patterns survive to uintBitsToFloat (the binding-3 trick).
layout(std430, set = 0, binding = 8) readonly buffer Inst     { float im[];    };
layout(std430, set = 0, binding = 9) readonly buffer InstAttr { uint  iattr[]; };

// std140 layout is byte-identical to the Metal PrimParams struct, which is
// static_assert'ed to 192 bytes; every member below lands at the SAME offset in
// std140 (mat4 @0, the three vec4 normal-matrix columns @64/80/96, then the
// scalar tail packed 4 bytes each from @112). Offsets are noted per member.
// THREE hand-synced copies of this params layout must move together: this GLSL
// std140 block; PrimParams in vulkan_renderer.cpp (static_assert 192); and
// PrimParams in metal_renderer.mm's MSL string (static_assert 192). grep
// PrimParams when growing. The v1.3 instance tail (use_instance/inst_base/
// use_instance_attr/inst_attr_base) and the instance-PULL body below are the
// GLSL twin of the run-proven Metal geom_compute (metal_renderer.mm) —
// transcribed byte-for-byte in float op order; compiled out on macOS.
layout(std140, set = 0, binding = 5) uniform Params {
    mat4  mvp;          // offset 0
    vec4  nmat0;        // 64  — columns of the 3x3 normal matrix (xyz used)
    vec4  nmat1;        // 80
    vec4  nmat2;        // 96
    uint  pos_base;     // 112 — element bases: byte offsets / 4
    uint  idx_base;     // 116
    uint  nrm_base;     // 120
    uint  attr_base;    // 124
    uint  use_index;    // 128
    uint  vertex_count; // 132
    uint  color_mode;   // 136 — 0 FLAT, 1 COLORMAP, 2 VERTEX_RGBA
    uint  shade_mode;   // 140 — 0 UNLIT, 1 LAMBERT
    uint  flat_rgba;    // 144 — packed LE
    float vmin;         // 148
    float vmax;         // 152
    float size_px;      // 156   (block size 160)
    uint  uv_base;            // 160
    uint  use_instance;       // 164 — 0/1 (v1.3 instance tail)
    uint  inst_base;          // 168 — instance_offset / 4
    uint  use_instance_attr;  // 172 — 0/1
    uint  inst_attr_base;     // 176 — instance_attr_offset / 4
    uint  pad0;               // 180
    uint  pad1;               // 184
    uint  pad2;               // 188   (block size 192)
} p;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;

// bytes LE, /255.0 — same unpack as points.vert / the Metal unpack_rgba.
vec4 unpack_rgba(uint packed) {
    return vec4(float( packed        & 0xffu),
                float((packed >>  8) & 0xffu),
                float((packed >> 16) & 0xffu),
                float((packed >> 24) & 0xffu)) / 255.0;
}

void main() {
    uint i  = uint(gl_VertexIndex);
    uint vi = p.use_index != 0u ? min(idx[p.idx_base + i], p.vertex_count - 1u)
                                : i;

    vec3 wp = vec3(pos[p.pos_base + 3u * vi + 0u],
                   pos[p.pos_base + 3u * vi + 1u],
                   pos[p.pos_base + 3u * vi + 2u]);

    // v1.3 (§4.1): the per-instance model matrix is applied to the world
    // position FIRST, then mvp. use_instance==0 takes the EXACT v1.2 expression
    // (bit-identical — §8 row C). M columns are pulled column-major at
    // inst_base + 16*gl_InstanceIndex. Mirrors the MSL geom_compute grouping
    // p.mvp * (M * vec4(wp,1)) — never fold (mvp*M) (§8 byte-load-bearing note).
    bool inst = (p.use_instance != 0u);
    uint iid  = uint(gl_InstanceIndex);
    mat4 M;
    if (inst) {
        uint b = p.inst_base + 16u * iid;
        M = mat4(vec4(im[b +  0u], im[b +  1u], im[b +  2u], im[b +  3u]),   // column 0
                 vec4(im[b +  4u], im[b +  5u], im[b +  6u], im[b +  7u]),   // column 1
                 vec4(im[b +  8u], im[b +  9u], im[b + 10u], im[b + 11u]),   // column 2
                 vec4(im[b + 12u], im[b + 13u], im[b + 14u], im[b + 15u]));  // column 3
        gl_Position = p.mvp * (M * vec4(wp, 1.0));
    } else {
        gl_Position = p.mvp * vec4(wp, 1.0);                 // byte-identical to v1.2
    }
    v_uv = p.color_mode == 3u
        ? vec2(uv[p.uv_base + 2u * vi + 0u],
               uv[p.uv_base + 2u * vi + 1u])
        : vec2(0.0);

    vec4 c;
    if (inst && p.use_instance_attr != 0u) {
        // §4.3: per-instance tint overrides the color_mode source, looked up
        // once per instance through the same COLORMAP idx math (LUT at binding
        // 4). iattr bound as uint[] so NaN bit patterns survive uintBitsToFloat.
        float v = uintBitsToFloat(iattr[p.inst_attr_base + iid]);
        float t = (v == v && p.vmax > p.vmin)
                ? clamp((v - p.vmin) / (p.vmax - p.vmin), 0.0, 1.0) : 0.0;
        c = unpack_rgba(lut[uint(t * 255.0 + 0.5)]);
    } else if (p.color_mode == 1u) {
        // attr bound as uint[] so NaN bit patterns survive to uintBitsToFloat.
        float v = uintBitsToFloat(attr[p.attr_base + vi]);
        float t = (v == v && p.vmax > p.vmin)
                ? clamp((v - p.vmin) / (p.vmax - p.vmin), 0.0, 1.0) : 0.0;
        c = unpack_rgba(lut[uint(t * 255.0 + 0.5)]);
    } else if (p.color_mode == 2u) {
        c = unpack_rgba(attr[p.attr_base + vi]);
    } else if (p.color_mode == 3u) {
        c = vec4(1.0);
    } else {
        c = unpack_rgba(p.flat_rgba);
    }

    if (p.shade_mode == 1u) {
        vec3 n = normalize(vec3(nrm[p.nrm_base + 3u * vi + 0u],
                                nrm[p.nrm_base + 3u * vi + 1u],
                                nrm[p.nrm_base + 3u * vi + 2u]));
        vec3 nvs;
        if (inst) {
            // §4.4: instance upper-3x3 (M columns 0/1/2, xyz) applied to n
            // FIRST, then the per-draw normal matrix — EXACT MSL float order.
            vec3 ni;
            ni.x = M[0].x * n.x + M[1].x * n.y + M[2].x * n.z;
            ni.y = M[0].y * n.x + M[1].y * n.y + M[2].y * n.z;
            ni.z = M[0].z * n.x + M[1].z * n.y + M[2].z * n.z;
            nvs = normalize(ni.x * p.nmat0.xyz +
                            ni.y * p.nmat1.xyz +
                            ni.z * p.nmat2.xyz);
        } else {
            nvs = normalize(n.x * p.nmat0.xyz +
                            n.y * p.nmat1.xyz +
                            n.z * p.nmat2.xyz);
        }
        float lit = 0.30 + 0.70 * max(dot(nvs, vec3(0.0, 0.0, 1.0)), 0.0);
        c.rgb *= lit;   // alpha untouched
    }

    // gl_PointSize is written unconditionally: Vulkan consumes it only for point
    // topologies, so line/triangle pipelines ignore it and no two-entry-point
    // split is needed (unlike Metal, which rejects [[point_size]] for Line/
    // Triangle classes). NOTE: if validation layers on the target box reject the
    // unconditional write for line/triangle pipelines, the sanctioned fallback
    // is a Metal-style two-entry-point split with an identical body — do NOT
    // implement it preemptively.
    gl_PointSize = p.size_px;
    v_color = c;
}
