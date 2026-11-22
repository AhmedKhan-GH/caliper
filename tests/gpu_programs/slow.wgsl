// Pixels program whose loop runs far over its time limit. The runtime stops it with program.over_time.
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> pixels: array<u32>;
@compute @workgroup_size(8, 8) fn spin(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= 16u || id.y >= 16u) { return; }
  var acc = 0.0;
  for (var k = 0u; k < 400000u; k++) { acc += w[(k + id.x * 16u + id.y) % 4096u]; }
  pixels[id.y * 16u + id.x] = pack4x8unorm(vec4<f32>(fract(acc), 0.0, 0.0, 1.0));
}
