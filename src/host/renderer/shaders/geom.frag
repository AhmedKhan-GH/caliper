#version 450
// caliper.geometry.v1_1 primitive fragment: pass-through color. Twin of the
// Metal geom_fs (return in.color) — all shading is done in geom.vert, so the FS
// is out = in.color and nothing else.
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 o_color;
void main() { o_color = v_color; }
