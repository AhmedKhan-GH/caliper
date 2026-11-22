// Images: one quad on a camera's image plane, or at the entity's origin in 2-D. fs_color
// and fs_pick are in fragments.wgsl.

// mvp maps the unit square onto the image (the scene core folds that map into it), so no
// vertex buffer is bound. flat_rgba is a tint whose alpha is the opacity.
@vertex
fn vs_image(@builtin(vertex_index) vid: u32) -> VOut {
  var o: VOut;
  let q = quad_corner(vid);
  o.uv = vec2<f32>(q.x, q.y * 0.5 + 0.5);
  o.color = unpack_rgba(p.flat_rgba);
  o.pick = vec2<u32>(p.pick_id, 0u);
  o.pos = to_target(p.mvp * vec4<f32>(o.uv, 0.0, 1.0));
  return o;
}

@fragment
fn fs_image_rgba(in: VOut) -> @location(0) vec4<f32> {
  var s = textureSample(img_rgba, img_sampler, in.uv);
  if (p.image_mode == 1u) { s = vec4<f32>(s.rrr, 1.0); }
  return s * in.color;
}

fn texel_of(uv: vec2<f32>, size: vec2<u32>) -> vec2<u32> {
  return min(vec2<u32>(uv * vec2<f32>(size)), size - vec2<u32>(1u));
}

// A segmentation image: each pixel's class colour, nearest, at the opacity.
@fragment
fn fs_image_class(in: VOut) -> @location(0) vec4<f32> {
  let c = class_color(textureLoad(img_class, texel_of(in.uv, textureDimensions(img_class)), 0).r);
  return vec4<f32>(c.rgb, c.a * in.color.a);
}

// A depth or value image: each pixel through the colormap, nearest.
@fragment
fn fs_image_value(in: VOut) -> @location(0) vec4<f32> {
  let c = cmap_color(textureLoad(img_value, texel_of(in.uv, textureDimensions(img_value)), 0).r);
  return vec4<f32>(c.rgb, c.a * in.color.a);
}
