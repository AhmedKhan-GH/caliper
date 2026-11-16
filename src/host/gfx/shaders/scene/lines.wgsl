// Lines and arrows: a segment is one instance, a quad of the batch's width in pixels with
// each end extended by half the width. Strips bind the positions twice with a stride of one
// vertex and a one-byte break mask; index-pair edges arrive through the edge gather. An arrow
// (line_mode 1) is one instance of nine vertices, a shaft quad and a flat head in screen space.
// fs_color and fs_pick are in fragments.wgsl.

struct LineIn {
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
  @location(0) a: vec3<f32>,     // segment start, or arrow origin
  @location(1) b: vec3<f32>,     // segment end, or arrow vector
  @location(2) color_a: u32,     // colour source at a, or of the whole arrow
  @location(3) color_b: u32,     // colour source at b
  @location(4) keep: u32,        // uint8 break mask: 0 at the last vertex of a strip
};

@vertex
fn vs_lines(in: LineIn) -> VOut {
  var o: VOut;
  o.pick = vec2<u32>(p.pick_id, p.item_base + in.iid);
  let tip = select(in.b, in.a + in.b, p.line_mode == 1u);
  let at_b = quad_corner(in.vid % 6u).x > 0.5 && p.line_mode == 0u;
  o.color = item_color(select(in.color_a, in.color_b, at_b));
  if (in.keep == 0u || any_nan3(in.a) || any_nan3(tip)) { o.pos = hidden(); return o; }
  var a = p.mvp * vec4<f32>(in.a, 1.0);
  var b = p.mvp * vec4<f32>(tip, 1.0);
  if (p.line_mode == 0u) {
    o.pos = segment_corner(a, b, in.vid, p.size_px);
    return o;
  }
  // An arrow: vertices 0 to 5 are the shaft, 6 to 8 the head, a flat triangle in
  // screen space a fifth of the arrow's length, at least three widths wide.
  if (!clip_near(&a, &b)) { o.pos = hidden(); return o; }
  let half_vp = 0.5 * frame.viewport;
  let sa = a.xy / a.w * half_vp;
  let sb = b.xy / b.w * half_vp;
  let len = length(sb - sa);
  let dir = select(vec2<f32>(1.0, 0.0), (sb - sa) / len, len > 1e-6);
  let w = max(p.size_px, 1.0);
  let head = 0.2 * len;
  let base = sb - dir * head;
  if (in.vid < 6u) {
    o.pos = segment_corner(a, vec4<f32>(base / half_vp * b.w, b.z, b.w), in.vid, w);
    return o;
  }
  let side = vec2<f32>(-dir.y, dir.x) * max(1.5 * w, 0.4 * head);
  var s = sb;
  if (in.vid == 7u) { s = base + side; }
  if (in.vid == 8u) { s = base - side; }
  o.pos = to_target(vec4<f32>(s / half_vp * b.w, b.z, b.w));
  return o;
}
