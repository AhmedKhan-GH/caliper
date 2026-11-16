// Boxes: an instance of the unit cube, or of the unit square in 2-D, whose pose the vertex
// stage builds from the center, half sizes and rotation. Edges are segment quads; faces are
// triangles. fs_color and fs_pick are in fragments.wgsl.

struct BoxIn {
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
  @location(0) center: vec3<f32>,     // float32x3, or float32x2 in 2-D (z reads 0)
  @location(1) half_size: vec3<f32>,  // the same
  @location(2) rotation: vec4<f32>,   // a quaternion (x, y, z, w); stride 0 (identity) when absent
  @location(3) color_src: u32,
};

// The column-major 4 x 4 pose that a caliper.geometry.v1_3 instance holds:
// translation, rotation, then the half sizes as scale.
fn box_pose(in: BoxIn) -> mat4x4<f32> {
  var q = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  let n2 = dot(in.rotation, in.rotation);
  if (n2 > 0.0) { q = in.rotation * inverseSqrt(n2); }
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  let c0 = vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y));
  let c1 = vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x));
  let c2 = vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
  let h = in.half_size;
  return mat4x4<f32>(vec4<f32>(c0 * h.x, 0.0), vec4<f32>(c1 * h.y, 0.0),
                     vec4<f32>(c2 * h.z, 0.0), vec4<f32>(in.center, 1.0));
}

fn on_axis(axis: u32, along: f32, s: f32, t: f32) -> vec3<f32> {
  if (axis == 0u) { return vec3<f32>(along, s, t); }
  if (axis == 1u) { return vec3<f32>(s, along, t); }
  return vec3<f32>(s, t, along);
}

// Wire boxes: 12 edges of the unit cube as screen-space quads, 72 vertices; the
// first 24 are the square at z = -1, which is all a 2-D box draws.
@vertex
fn vs_box_wire(in: BoxIn) -> VOut {
  var o: VOut;
  o.pick = vec2<u32>(p.pick_id, p.item_base + in.iid);
  o.color = item_color(in.color_src);
  let e = in.vid / 6u;
  var axis = (e % 4u) / 2u;
  var s = select(-1.0, 1.0, (e & 1u) != 0u);
  var t = select(-1.0, 1.0, e >= 4u);
  if (e >= 8u) { axis = 2u; t = select(-1.0, 1.0, (e & 2u) != 0u); }
  let pose = box_pose(in);
  let a = p.mvp * (pose * vec4<f32>(on_axis(axis, -1.0, s, t), 1.0));
  let b = p.mvp * (pose * vec4<f32>(on_axis(axis, 1.0, s, t), 1.0));
  o.pos = segment_corner(a, b, in.vid % 6u, p.size_px);
  return o;
}

// The faces of a filled box are translucent: a quarter of the item's alpha, so that
// an opaque colour still shows what lies behind them.
const kFaceAlpha: f32 = 0.25;

// Filled boxes: 6 faces of two triangles, 36 vertices; the first 6 are the face
// at z = +1, which is all a 2-D box draws. Each normal is an axis vector, so the
// pose maps it onto a vector parallel to the true normal under any rotation and
// positive scale, and the native normal chain lights a box exactly.
@vertex
fn vs_box_solid(in: BoxIn) -> VOut {
  var o: VOut;
  o.pick = vec2<u32>(p.pick_id, p.item_base + in.iid);
  let face = in.vid / 6u;
  var order = array<u32, 3>(2u, 0u, 1u);
  let axis = order[face / 2u];
  let sgn = select(1.0, -1.0, (face & 1u) != 0u);
  let q = quad_corner(in.vid % 6u);
  let pose = box_pose(in);
  o.pos = to_target(p.mvp * (pose * vec4<f32>(on_axis(axis, sgn, q.x * 2.0 - 1.0, q.y), 1.0)));
  var c = item_color(in.color_src);
  if (p.shade_mode == 1u) { c = lambert(c, inst_normal(pose, on_axis(axis, sgn, 0.0, 0.0))); }
  o.color = vec4<f32>(c.rgb, c.a * kFaceAlpha);
  return o;
}
