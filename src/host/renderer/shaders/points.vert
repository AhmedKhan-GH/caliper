#version 450
// caliper.geometry.v1 instanced points: vertex-pulled from the IMPORTED
// allocation (bridge v1.2) at an element base — zero copies of the point
// data. Color: optional per-point f32 attr mapped through the shared 256-
// entry LUT with the SAME index rule as the CPU reference map_f32_to_rgba8
// (idx = clamp((v-vmin)/(vmax-vmin),0,1)*255+0.5 floored; NaN -> 0;
// degenerate range -> 0), or flat white when use_attr == 0.

layout(std430, set = 0, binding = 0) readonly buffer Pos  { float pos[];  };
layout(std430, set = 0, binding = 1) readonly buffer Attr { float attr[]; };
layout(std430, set = 0, binding = 2) readonly buffer Lut  { uint  lut[];  };

layout(push_constant) uniform PC {
    mat4  mvp;        // proj*view, premultiplied host-side (88 B total fits
    uint  pos_base;   // the 128-byte minimum push budget; two mat4s do not)
    uint  attr_base;  // element (float) bases: byte offsets / 4
    uint  use_attr;
    float vmin;
    float vmax;
    float size_px;
} pc;

layout(location = 0) out vec4 v_color;

void main() {
    uint i = uint(gl_VertexIndex);
    vec3 p = vec3(pos[pc.pos_base + 3u * i + 0u],
                  pos[pc.pos_base + 3u * i + 1u],
                  pos[pc.pos_base + 3u * i + 2u]);
    gl_Position  = pc.mvp * vec4(p, 1.0);
    gl_PointSize = pc.size_px;
    if (pc.use_attr != 0u) {
        float v = attr[pc.attr_base + i];
        float t = (v == v && pc.vmax > pc.vmin)
                      ? clamp((v - pc.vmin) / (pc.vmax - pc.vmin), 0.0, 1.0)
                      : 0.0;
        uint idx = uint(t * 255.0 + 0.5);
        uint c   = lut[idx];
        v_color = vec4(float(c & 0xFFu), float((c >> 8) & 0xFFu),
                       float((c >> 16) & 0xFFu), float((c >> 24) & 0xFFu))
                  / 255.0;
    } else {
        v_color = vec4(1.0);
    }
}
