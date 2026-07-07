#version 450
// caliper.geometry.v1 point fragment: pass-through color, square points
// (no round-sprite discard — deterministic rasterization is what lets the
// gfx rows assert byte-exact pixels; the additive blend does the glow).
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 o_color;
void main() { o_color = v_color; }
