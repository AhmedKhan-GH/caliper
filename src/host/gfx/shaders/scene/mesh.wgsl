// Meshes: positions, normals, colour sources and UVs per vertex as vertex buffers, the
// clamped indices as the index buffer, and the v1.3 instance pose and value per instance.
// With normals the per-vertex headlight lights them; without, faces are flat. fs_color and
// fs_pick are in fragments.wgsl.

struct MeshIn {
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,  // stride 0 when the mesh has no normals
  @location(2) color_src: u32,     // RGBA or value bits per vertex; stride 0 when one
  @location(3) uv: vec2<f32>,      // stride 0 when untextured
  @location(4) m0: vec4<f32>,      // the v1.3 instance pose; stride 0 (identity) when absent
  @location(5) m1: vec4<f32>,
  @location(6) m2: vec4<f32>,
  @location(7) m3: vec4<f32>,
  @location(8) inst_src: u32,      // the v1.3 per-instance value bits; stride 0 when absent
};

@vertex
fn vs_mesh(in: MeshIn) -> VOut {
  var o: VOut;
  o.pick = vec2<u32>(p.pick_id, p.item_base + in.vid);
  o.uv = in.uv;
  o.obj = in.pos;
  let inst = p.use_instance != 0u;
  let pose = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
  if (any_nan3(in.pos)) {
    o.pos = hidden();
  } else if (inst) {
    o.pos = to_target(p.mvp * (pose * vec4<f32>(in.pos, 1.0)));   // never fold mvp * pose
    o.obj = (pose * vec4<f32>(in.pos, 1.0)).xyz;
  } else {
    o.pos = to_target(p.mvp * vec4<f32>(in.pos, 1.0));
  }
  var c = item_color(in.color_src);
  if (inst && p.use_instance_attr != 0u) { c = cmap_color(in.inst_src); }
  if (p.shade_mode == 1u) {
    var n = normalize(in.normal);
    if (inst) { n = inst_normal(pose, n); }
    c = lambert(c, n);
  }
  o.color = c;
  return o;
}

// A mesh without normals: flat faces, each lit by the headlight from the normal
// that the derivatives of its entity-space position give, on both of its sides.
@fragment
fn fs_mesh_flat(in: VOut) -> @location(0) vec4<f32> {
  let n = view_normal(cross(dpdx(in.obj), dpdy(in.obj)));
  let lit = 0.30 + 0.70 * abs(n.z);
  return vec4<f32>(in.color.rgb * lit, in.color.a);
}

// The native textured fragment: the sampled colour times the vertex colour's RGB.
@fragment
fn fs_mesh_tex(in: VOut) -> @location(0) vec4<f32> {
  var s = textureSample(img_rgba, img_sampler, in.uv);
  if (p.image_mode == 1u) { s = vec4<f32>(s.rrr, 1.0); }
  return vec4<f32>(s.rgb * in.color.rgb, s.a);
}
