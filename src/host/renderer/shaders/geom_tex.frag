#version 450
// geometry.v1_2 textured fragment. UVs are interpolated from the imported
// vertex stream; the immutable host sampler supplies linear clamp-to-edge.
layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;
layout(set = 0, binding = 7) uniform sampler2D field_texture;
layout(location = 0) out vec4 o_color;

void main() {
    vec4 sampled = texture(field_texture, v_uv);
    sampled.rgb *= v_color.rgb;
    o_color = sampled;
}
