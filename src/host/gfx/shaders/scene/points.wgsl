// Points: each point an instanced quad of two triangles around its projected centre, and a
// depth image under a camera drawn as a point cloud without a point buffer.

struct PointIn {
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
  @location(0) pos: vec3<f32>,   // float32x3, or float32x2 in 2-D (z reads 0)
  @location(1) color_src: u32,   // stride 4, or stride 0 for one colour
  @location(2) radius: f32,      // stride 4, or stride 0 for one radius
};

@vertex
fn vs_points(in: PointIn) -> VOut {
  var o: VOut;
  let q = quad_corner(in.vid);
  let corner = vec2<f32>(q.x * 2.0 - 1.0, q.y);
  o.uv = corner;
  o.pick = vec2<u32>(p.pick_id, p.item_base + in.iid);
  o.color = item_color(in.color_src);
  if (any_nan3(in.pos)) { o.pos = hidden(); return o; }
  let c = p.mvp * vec4<f32>(in.pos, 1.0);
  // The live points view's depth cues (CompileOptions, zero for every other
  // draw): with att_ref a point is nearer larger and farther smaller, clamped
  // to half and twice its size, and past fade_w0 its colour falls linearly to
  // half at fade_w1, its hue and alpha kept.
  var size = p.size_px;
  if (p.att_ref > 0.0) { size = size * clamp(p.att_ref / max(c.w, 1e-6), 0.5, 2.0); }
  if (p.fade_w1 > p.fade_w0) {
    let f = clamp((c.w - p.fade_w0) / (p.fade_w1 - p.fade_w0), 0.0, 1.0);
    o.color = vec4<f32>(o.color.rgb * (1.0 - 0.5 * f), o.color.a);
  }
  // Half the size in clip units: size_px as natively, or a radius in scene
  // units projected with the frame's scale, never below half a pixel.
  var k = (size / frame.viewport) * c.w;
  if (p.radius_mode == 1u) {
    k = max(in.radius * p.radius_scale * frame.proj_scale, c.w / frame.viewport);
  }
  o.pos = to_target(vec4<f32>(c.xy + corner * k, c.z, c.w));
  return o;
}

// A depth image under a camera as a point cloud: instance i is pixel i. The
// scene core folds the inverse intrinsics into mvp, so the pixel (x, y) at
// depth d is (x * d, y * d, d) before it.
@vertex
fn vs_depth_points(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VOut {
  var o: VOut;
  let size = textureDimensions(img_value);
  let px = vec2<u32>(iid % size.x, iid / size.x);
  let bits = textureLoad(img_value, px, 0).r;
  let q = quad_corner(vid);
  let corner = vec2<f32>(q.x * 2.0 - 1.0, q.y);
  o.uv = corner;
  o.pick = vec2<u32>(p.pick_id, p.item_base + iid);
  o.color = select(unpack_rgba(p.flat_rgba), cmap_color(bits), p.color_mode == 1u);
  // Skip zero, negative, infinite and NaN depths, decided on the bits.
  if ((bits >> 31u) != 0u || bits == 0u || (bits & 0x7fffffffu) >= 0x7f800000u) { o.pos = hidden(); return o; }
  let d = bitcast<f32>(bits);
  let xy = (vec2<f32>(px) + 0.5) * d;
  let c = p.mvp * vec4<f32>(xy, d, 1.0);
  o.pos = to_target(vec4<f32>(c.xy + corner * (p.size_px / frame.viewport) * c.w, c.z, c.w));
  return o;
}

@fragment
fn fs_points(in: VOut) -> @location(0) vec4<f32> {
  if (p.point_shape == 1u && dot(in.uv, in.uv) > 1.0) { discard; }
  return in.color;
}

@fragment
fn fs_points_pick(in: VOut) -> @location(0) vec2<u32> {
  if (p.point_shape == 1u && dot(in.uv, in.uv) > 1.0) { discard; }
  return in.pick;
}
