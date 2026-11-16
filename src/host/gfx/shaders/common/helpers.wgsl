// The helpers every scene program shares. unpack_rgba and cmap_color are in colormap.wgsl.

fn is_nan_bits(v: f32) -> bool {
  let b = bitcast<u32>(v);
  return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

fn any_nan3(v: vec3<f32>) -> bool {
  return is_nan_bits(v.x) || is_nan_bits(v.y) || is_nan_bits(v.z);
}

// A position the clipper removes (z > w).
fn hidden() -> vec4<f32> { return vec4<f32>(0.0, 0.0, 2.0, 1.0); }

fn class_color(id: u32) -> vec4<f32> {
  if (id >= frame.class_count) { return unpack_rgba(frame.class_default); }
  return textureLoad(class_table, vec2<u32>(id & 255u, id >> 8u), 0);
}

// One item's colour from its 4-byte source, read as bits so that NaN survives.
fn item_color(src: u32) -> vec4<f32> {
  switch (p.color_mode) {
    case 1u: { return cmap_color(src); }
    case 2u: { return unpack_rgba(src); }
    case 3u: { return vec4<f32>(1.0); }
    case 4u: { return class_color(src); }
    default: { return unpack_rgba(p.flat_rgba); }
  }
}

// The native instance normal chain, in the native order of operations.
fn inst_normal(q: mat4x4<f32>, n: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(q[0].x * n.x + q[1].x * n.y + q[2].x * n.z,
                   q[0].y * n.x + q[1].y * n.y + q[2].y * n.z,
                   q[0].z * n.x + q[1].z * n.y + q[2].z * n.z);
}

// The native Lambert headlight: ambient 0.30, alpha untouched.
fn view_normal(n: vec3<f32>) -> vec3<f32> {
  return normalize(n.x * p.nmat0.xyz + n.y * p.nmat1.xyz + n.z * p.nmat2.xyz);
}

fn lambert(c: vec4<f32>, n: vec3<f32>) -> vec4<f32> {
  let lit = 0.30 + 0.70 * max(dot(view_normal(n), vec3<f32>(0.0, 0.0, 1.0)), 0.0);
  return vec4<f32>(c.rgb * lit, c.a);
}

fn to_target(c: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(c.xy * frame.ndc_scale + frame.ndc_offset * c.w, c.z, c.w);
}

// Corner c of a quad of two triangles, as (end 0 or 1, side -1 or 1).
fn quad_corner(c: u32) -> vec2<f32> {
  var t = array<vec2<f32>, 6>(vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
                              vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  return t[c];
}

// Clips a segment to the near plane (clip z >= 0); false when all of it is behind.
fn clip_near(a: ptr<function, vec4<f32>>, b: ptr<function, vec4<f32>>) -> bool {
  let za = (*a).z;
  let zb = (*b).z;
  if (za < 0.0 && zb < 0.0) { return false; }
  if (za < 0.0) { *a = mix(*a, *b, za / (za - zb)); }
  if (zb < 0.0) { *b = mix(*b, *a, zb / (zb - za)); }
  return true;
}

// Corner c of segment (a, b) in clip space, as a quad width_px wide whose ends
// are extended by half the width (square caps).
fn segment_corner(a_in: vec4<f32>, b_in: vec4<f32>, c: u32, width_px: f32) -> vec4<f32> {
  var a = a_in;
  var b = b_in;
  if (!clip_near(&a, &b)) { return hidden(); }
  let half_vp = 0.5 * frame.viewport;
  let sa = a.xy / a.w * half_vp;
  let sb = b.xy / b.w * half_vp;
  let len = length(sb - sa);
  let dir = select(vec2<f32>(1.0, 0.0), (sb - sa) / len, len > 1e-6);
  let q = quad_corner(c);
  let w = max(width_px, 1.0) * 0.5;
  let off = vec2<f32>(-dir.y, dir.x) * (q.y * w) + dir * select(-w, w, q.x > 0.5);
  let e = select(a, b, q.x > 0.5);
  return to_target(vec4<f32>(e.xy + off / half_vp * e.w, e.z, e.w));
}
