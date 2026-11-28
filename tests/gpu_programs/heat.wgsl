// Pixels program: the first 32 x 32 values of fc2.weight (10 x 128) as an image, red for positive and blue for negative.
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> pixels: array<u32>;
@compute @workgroup_size(8, 8) fn paint(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= 32u || id.y >= 32u) { return; }
  let i = id.y * 32u + id.x;
  let v = clamp(w[i] * 6.0, -1.0, 1.0);
  pixels[i] = pack4x8unorm(vec4<f32>(max(v, 0.0), 0.15 + 0.3 * abs(v), max(-v, 0.0), 1.0));
}
